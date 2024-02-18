This repository contains the author's implementation in PyTorch for the paper "Multi-head Attention Empowered Graph Contrastive Learning for Cross-network Node Classification"

Environment Requirement
• python == 3.6.13

• torch==1.10.1

• numpy==1.19.3

• scipy==1.5.4

• scikit_learn==1.1.3

• dgl==0.9.1...

Datasets
input/ contains the 3 datasets used in our paper, i.e., acmv9, dblpv7 and citationv1.

Code
"model.py" is the LHCDA model. "train.py" is an example case of the LHCDA model for Cross-network node classification on one dataset.
