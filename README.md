# LRCA: Local Residue Cooperation Architecture for Enhanced Prediction of Bile Acid–Related Enzymes with Protein Language Models
LRCA takes residue level embeddings from the pretrained ESM 2 model and applies a LocalRelation module to model dependencies among neighboring residues. This module enriches fine grained features associated with bile acid catalytic specificity while retaining global contextual information.
<p align="center">
  <img src="Fig/LRCA.png" alt="Overview of LRCA" width="600">
  <br>
  <em>Overview of LRCA.</em>
</p>

##  Key Features

LRCA mainly consists of the following two modules:

1.Global semantics branch (Global Semantics): It takes the residue-level embeddings produced by a pretrained protein language model (ESM-2) as the base representation.

2.The LocalRelation module models feature relationships between adjacent residues, producing residue representations enriched with local cooperative information

LRCA was evaluated on the HGBME dataset, achieving an AUPRC improvement from 0.790 to 0.824 under 5-fold cross-validation; when selecting the best fold by AUPRC, the AUPRC increased from 0.804 to 0.845.

##  Usage

### 1. Data Preparation
We use the HGBME dataset introduced by BEAUT, and the original data can be obtained from  [HGBME](https://zenodo.org/records/15388149). Here, we only provide the data relevant to the LRCA experiments. 

#### A. Protein sequence features were obtained using ESM2.
Open LRCA_embed_esm2.py the and update the Configuration section to point to your actual file paths:




