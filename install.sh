# Installation tested on RTX3090 with CUDA 11.3 and Python 3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
pip install torchtext 
pip install opacus
pip install pandas
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html