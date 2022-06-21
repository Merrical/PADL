# Modeling Annotator Preference and Stochastic Annotation Error for Medical Image Segmentation (PADL)

This repo contains the official implementation of our paper: Modeling Annotator Preference and Stochastic Annotation Error for Medical Image Segmentation, which highlights the issue of annotator-related biases existed in medical image segmentation tasks.
<p align="center"><img src="https://raw.githubusercontent.com/Merrical/PADL/master/PADL_overview.png" width="90%"></p>

#### [Paper](https://arxiv.org/pdf/2111.13410.pdf)

### Requirements
This repo was tested with Ubuntu 20.04.4 LTS, Python 3.8, PyTorch 1.7.1, and CUDA 10.1.
We suggest using virtual env to configure the experimental environment.

1. Clone this repo:

```bash
git clone https://github.com/Merrical/PADL.git
```

2. Create experimental environment using virtual env:

```bash
virtualenv .env --python=3.8 # create
source .env/bin/activate # activate
pip install -r requirements.txt
```
### Dataset

The dataset details and the download link can be found [here](https://github.com/jiwei0921/MRNet).

### Training 

```bash
python main.py --dataset RIGA --rater_num 6 --phase train --net_arch PADL --loss_func bce --device_id 0 --loop 0
```

### Inference

```bash
python main.py --dataset RIGA --rater_num 6 --phase test --net_arch PADL --loss_func bce --device_id 0 --loop 0
```
### Bibtex
```
@article{Liao2021PADL,
  title   = {Modeling Annotator Preference and Stochastic Annotation Error for Medical Image Segmentation},
  author  = {Liao, Zehui and Hu, Shishuai and Xie, Yutong and Xia, Yong},
  journal = {arXiv preprint arXiv:2111.13410},
  year    = {2021}
}
```

### Contact Us
If you have any questions, please contact us ( merrical@mail.nwpu.edu.cn ).
