# DDA-6060-Rec

# Dataset
- Movielens 
https://grouplens.org/datasets/movielens/100k/

# Requirements
- Python 3.7+
- PyTorch
- dgl https://www.dgl.ai/pages/start.html
- torchtext  https://github.com/pytorch/text  `pip install torchtext
`
# Experimental Results
| Dataset    | Method  | RMSE on validation | RMSE on Test |
|  ----  | ----  |----  |----  |
|  ml-100k   | GNN | |
|   ml-100k  | GNN-Fair | |
|   ml-100k  | GNN-DP | |
|   ml-100k  | GNN-Fair-DP | 0.9569 | 0.9864 |
|  ml-100k | GNN-Fair-DP-GN | 0.9333 | 0.9543 |