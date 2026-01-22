
import torch
import torch.nn as nn
from typing import Optional, Tuple


def ensure_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask if mask.dtype == torch.bool else (mask > 0)


def masked_fill_seq(x: torch.Tensor, mask: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    # x: [B,K,*], mask: [B,K]
    mask = ensure_bool_mask(mask)
    return x.masked_fill(~mask.unsqueeze(-1), value)


def make_gate(dim: int, reduction: int, dropout: float) -> nn.Module:
    gdim = max(8, dim // reduction)
    return nn.Sequential(
        nn.Linear(dim, gdim),
        nn.ReLU(),
        nn.Dropout(p=min(0.1, float(dropout))),
        nn.Linear(gdim, dim),
        nn.Sigmoid(),
    )


def make_proj(in_dim: int, out_dim: Optional[int], act: nn.Module, dropout: float) -> Tuple[Optional[nn.Module], int]:
    if out_dim is None or out_dim == in_dim:
        return None, in_dim
    return nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(float(dropout))), out_dim


# -----------------------------
# Regularization: DropPath
# -----------------------------
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(dtype=x.dtype)
        return x * mask / keep


# -----------------------------
# Local Relation Module Blocks
# -----------------------------
class LocalRelationBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilations=(1, 2),
        dropout: float = 0.1,
        drop_path: float = 0.1,
        layer_scale_init: float = 1e-3,
        activation: str = "gelu",
    ):
        super().__init__()
        C = int(channels)
        k = int(kernel_size)
        ds = tuple(int(d) for d in dilations)

        self.ln = nn.LayerNorm(C)

        self.dw_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    C,
                    C,
                    kernel_size=k,
                    padding=(k // 2) * d,
                    dilation=d,
                    groups=C,
                    bias=False,
                )
                for d in ds
            ]
        )

        self.pw = nn.Conv1d(C, 2 * C, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(p=float(dropout))
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(C))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = ensure_bool_mask(mask)
        x = masked_fill_seq(x, mask, 0.0)

        h = self.ln(x).transpose(1, 2)  # [B,C,K]
        h = sum(conv(h) for conv in self.dw_convs)

        h = self.pw(h)                  # [B,2C,K]
        a, b = h.chunk(2, dim=1)
        h = a * torch.sigmoid(b)        # GLU
        h = self.dropout(h)

        h = h.transpose(1, 2)           # [B,K,C]
        h = h * self.gamma              # LayerScale

        out = x + self.drop_path(h)
        return masked_fill_seq(out, mask, 0.0)


class LocalRelationModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        channels: int,
        num_layers: int = 2,
        kernel_size: int = 5,
        dilations=(1, 2),
        dropout: float = 0.1,
        drop_path: float = 0.15,
        token_drop: float = 0.15,
        activation: str = "gelu",
        in_proj_dropout: float = 0.1,
    ):
        super().__init__()
        self.token_drop = float(token_drop)

        self.in_proj = nn.Sequential(
            nn.Conv1d(int(in_dim), int(channels), kernel_size=1, bias=False),
            nn.Dropout(p=float(in_proj_dropout)),
        )

        self.blocks = nn.ModuleList(
            [
                LocalRelationBlock(
                    channels=int(channels),
                    kernel_size=int(kernel_size),
                    dilations=tuple(dilations),
                    dropout=float(dropout),
                    drop_path=float(drop_path),
                    activation=activation,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = ensure_bool_mask(mask)
        tok = masked_fill_seq(tok, mask, 0.0)

        if self.training and self.token_drop > 0:
            keep = (torch.rand(mask.shape, device=mask.device) > self.token_drop) & mask
            tok = masked_fill_seq(tok, keep, 0.0)
            mask = keep

        #
        conv_dtype = self.in_proj[0].weight.dtype
        x = tok.to(dtype=conv_dtype).transpose(1, 2)   # [B,D,K]
        x = self.in_proj(x).transpose(1, 2)            # [B,K,C]

        for blk in self.blocks:
            x = blk(x, mask)

        return x, mask


# -----------------------------
# Main model
# -----------------------------
class DNNPredictor(nn.Module):


    def __init__(
        self,
        vec_dim: int,
        hidden_size: list,
        dropout: float = 0.3,
        activation: str = "gelu",

        # token branch
        use_conv_branch: bool = True,
        conv_channels: int = 64,
        conv_kernel: int = 5,
        conv_dropout: float = 0.1,

        # local relation knobs
        local_num_layers: int = 2,
        local_dilations: tuple = (1, 2),
        local_drop_path: float = 0.15,
        token_drop: float = 0.15,

        # pooling
        pool: str = "mean",        # mean/max/attn

        # attn pooling knobs
        attn_pool_hidden: Optional[int] = None,
        attn_pool_dropout: float = 0.0,
        attn_softmax_fp32: bool = True,

        # fusion
        fuse: str = "concat",      # concat/sum
        vec_proj_dim: Optional[int] = 128,
        tok_proj_dim: Optional[int] = 128,

        # norm / gate
        use_layernorm: bool = True,
        use_gate: bool = True,
        gate_reduction: int = 16,

        # anti-overfit
        tok_branch_drop: float = 0.55,
        tok_pooled_dropout: float = 0.55,
        fusion_dropout: float = 0.30,

        # embedding dim
        feat_dim: int = 32,
        feat_dropout: float = 0.0,
    ):
        super().__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        self.use_conv_branch = bool(use_conv_branch)
        self.pool = pool.lower().strip()
        self.fuse = fuse.lower().strip()
        assert self.pool in ("mean", "max", "attn")
        assert self.fuse in ("concat", "sum")

        self.tok_branch_drop = float(tok_branch_drop)
        self.tok_pooled_dropout = nn.Dropout(p=float(tok_pooled_dropout))
        self.fusion_dropout = nn.Dropout(p=float(fusion_dropout))

        self.attn_softmax_fp32 = bool(attn_softmax_fp32)
        self.attn_pool_dropout = nn.Dropout(p=float(attn_pool_dropout)) if attn_pool_dropout > 0 else nn.Identity()
        self.attn_pool: Optional[nn.Module] = None

        # ---- vec branch ----
        self.vec_norm = nn.LayerNorm(vec_dim) if use_layernorm else nn.Identity()
        self.vec_gate = make_gate(vec_dim, gate_reduction, dropout) if use_gate else None
        self.vec_proj, vec_out_dim = make_proj(vec_dim, vec_proj_dim, act, dropout)

        # ---- tok branch ----
        self.local_rel: Optional[LocalRelationModule] = None
        self.tok_norm: Optional[nn.Module] = None
        self.tok_gate: Optional[nn.Module] = None
        self.tok_proj: Optional[nn.Module] = None
        tok_out_dim = 0

        if self.use_conv_branch:
            self.local_rel = LocalRelationModule(
                in_dim=vec_dim,
                channels=int(conv_channels),
                num_layers=int(local_num_layers),
                kernel_size=int(conv_kernel),
                dilations=tuple(local_dilations),
                dropout=float(conv_dropout),
                drop_path=float(local_drop_path),
                token_drop=float(token_drop),
                activation=activation,
                in_proj_dropout=float(conv_dropout),
            )

            self.tok_norm = nn.LayerNorm(int(conv_channels)) if use_layernorm else nn.Identity()
            self.tok_gate = make_gate(int(conv_channels), gate_reduction, dropout) if use_gate else None

            if self.pool == "attn":
                C = int(conv_channels)
                self.attn_pool = (
                    nn.Linear(C, 1, bias=False)
                    if attn_pool_hidden is None
                    else nn.Sequential(nn.Linear(C, int(attn_pool_hidden)), nn.Tanh(), nn.Linear(int(attn_pool_hidden), 1, bias=False))
                )

            self.tok_proj, tok_out_dim = make_proj(int(conv_channels), tok_proj_dim, act, dropout)

        # ---- fusion dim ----
        if self.use_conv_branch and self.fuse == "concat":
            fused_dim = vec_out_dim + tok_out_dim
        elif self.use_conv_branch and self.fuse == "sum":
            assert tok_out_dim > 0
            assert vec_out_dim == tok_out_dim, f"sum fusion requires vec_out_dim == tok_out_dim, got {vec_out_dim} vs {tok_out_dim}"
            fused_dim = vec_out_dim
        else:
            fused_dim = vec_out_dim

        self.vec_out_dim = int(vec_out_dim)
        self.tok_out_dim = int(tok_out_dim)
        self.fused_dim = int(fused_dim)

        self.fuse_norm = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()

        # ---- head ----
        self.layers = nn.ModuleList()
        for i, out_dim in enumerate(hidden_size):
            in_dim = fused_dim if i == 0 else hidden_size[i - 1]
            self.layers.append(nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(p=float(dropout))))

        last_h = hidden_size[-1]
        self.feat_dim = int(feat_dim)
        self.feat_layer = nn.Identity() if self.feat_dim == last_h else nn.Linear(last_h, self.feat_dim)
        self.feat_dropout = nn.Dropout(p=float(feat_dropout)) if feat_dropout and feat_dropout > 0 else nn.Identity()
        self.output_layer = nn.Linear(self.feat_dim, 2)

    # -----------------------------
    # pooling
    # -----------------------------
    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mask = ensure_bool_mask(mask)
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(eps)

    def _masked_max(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = ensure_bool_mask(mask)
        neg_inf = torch.finfo(x.dtype).min
        return x.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values

    def _masked_attn(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.attn_pool is None:
            raise RuntimeError("pool='attn' but attn_pool is not initialized.")
        mask = ensure_bool_mask(mask)

        scores = self.attn_pool(x).squeeze(-1)           # [B,K]
        scores = scores.masked_fill(~mask, -1e4)

        w = torch.softmax(scores.float(), dim=1).to(dtype=scores.dtype) if self.attn_softmax_fp32 else torch.softmax(scores, dim=1)
        w = self.attn_pool_dropout(w)
        return torch.einsum("bk,bkc->bc", w, x)

    def _pool(self, x_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return self._masked_mean(x_seq, mask)
        if self.pool == "max":
            return self._masked_max(x_seq, mask)
        return self._masked_attn(x_seq, mask)

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, *args, return_feat: bool = False, **kwargs):
        if len(args) == 1:
            vec = args[0]
            tok = kwargs.get("tok", None)
            mask = kwargs.get("mask", None)
        elif len(args) >= 3:
            vec, tok, mask = args[0], args[1], args[2]
        else:
            raise ValueError("Call as model(vec, tok, mask) or model(reprs).")

        # vec branch
        x_vec = self.vec_norm(vec)
        if self.vec_gate is not None:
            x_vec = x_vec * self.vec_gate(x_vec)
        if self.vec_proj is not None:
            x_vec = self.vec_proj(x_vec)

        # tok branch availability
        use_tok = self.use_conv_branch and (tok is not None) and (mask is not None)

        # training-time random drop tok branch
        if use_tok and self.training and self.tok_branch_drop > 0:
            if torch.rand(1, device=x_vec.device).item() < self.tok_branch_drop:
                use_tok = False

        x_tok = None
        if use_tok:
            mask = ensure_bool_mask(mask)
            x_seq, mask = self.local_rel(tok, mask)              # [B,K,C], bool mask
            x_tok = self._pool(x_seq, mask)                      # [B,C]
            x_tok = self.tok_norm(x_tok)

            if self.tok_gate is not None:
                x_tok = x_tok * self.tok_gate(x_tok)
            if self.tok_proj is not None:
                x_tok = self.tok_proj(x_tok)

            x_tok = self.tok_pooled_dropout(x_tok).to(dtype=x_vec.dtype)

        # keep fused dim stable for concat when tok branch is absent
        if self.use_conv_branch and self.fuse == "concat":
            if x_tok is None:
                B = x_vec.size(0)
                x_tok = torch.zeros((B, self.tok_out_dim), device=x_vec.device, dtype=x_vec.dtype)
            x_fused = torch.cat([x_vec, x_tok], dim=-1)
        elif self.use_conv_branch and self.fuse == "sum":
            x_fused = x_vec if x_tok is None else (x_vec + x_tok)
        else:
            x_fused = x_vec

        x = self.fuse_norm(x_fused)
        x = self.fusion_dropout(x)

        for layer in self.layers:
            x = layer(x)

        feat = self.feat_dropout(self.feat_layer(x))
        logits = self.output_layer(feat)

        return (logits, feat) if return_feat else logits
