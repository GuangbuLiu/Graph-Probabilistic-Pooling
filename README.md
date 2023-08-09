# BernPool

PyTorch implementation of Graph Bernoulli Pooling.

![architecture](/fig/architecture.png)

## Requirements

* torch = 1.11.0+cu113
* torch-scatter = 2.0.9
* torch-sparse = 0.6.15
* torch-geometric = 2.0.4


## Run
If you want to run the experiments on the PROTEINS, DD, NCI1, ENZYMES, Mutagenicity, IMDB-BINARY, IMDB-MULTI, COLLAB, and REDDIT-MULTI-12K datasets, please type the following command:

`python main.py` 

If you want to run the experiments on the ogbg-ppa dataset, please enter the ogbg-ppa file folder and then type the following command:

`python main_pyg.py`
