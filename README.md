# Semi-URF: Progressive Uncertainty-Aware Region Filtering and Fusion for Semi-Supervised Medical Image Segmentation

This repository offers the official PyTorch implementation of our **Semi-URF** for semi-supervised medical image segmentation.


## Installation
This project is built upon **Python 3.10** and **PyTorch 2.0.1**.

Install PyTorch following the
 **[official PyTorch installation instructions](https://pytorch.org/get-started/locally/)**


```bash

cd semi-URF
pip install -r requirements.txt
```


## Dataset

ACDC: [image and mask](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

## Training
```bash

sh run_train.sh
```
## Acknowledgement
Part of the data preprocessing and comparative experiments are based on the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
