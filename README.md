# Mutual Bootstrapping Deep Convolutional Neural Networks(MB-DCNN)

This is the official pytorch implementation of the MB-DCNN model:<br />

**Paper: A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification.** 
(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8990108) 

## Requirements
Python 3.7<br />
Torch==1.4.0<br />
Torchvision==0.5.0<br />
Apex==0.1<br />
CUDA 10.0<br />

## Usage

### 0. Installation
* Clone this repo
```
git clone https://github.com/YtongXie/MB-DCNN.git
cd MB-DCNN
```
### 1. Data Preparation
* Download [ISIC2017 dataset](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a), [Extra 1320 images](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D)(The ID information of the extra 1320 images is listed in [here](https://drive.google.com/file/d/1XkXPqxT5zJb0OJogsbtb_pCNlwv5qm4m/view?usp=sharing)) <br/>

* Put the data under `./dataset/data/` 

* Run `python ./dataset/extractPatch_cls_train.py` and `python ./dataset/extractPatch_cls_val_test.py` to obtain the cropped training, validation and testing patches for classification task.

* Run `python ./dataset/extractPatch_seg_train.py` to obtain the resized training patches for segmentation task.

* Run `python ./dataset/list_cls.py` and `python ./dataset/list_seg.py` to generate the data lists.

### 2. Training coarse segmentation network (coarse-SN)
* Download pretrained weights from [Deeplabv3+](https://drive.google.com/file/d/11lgslZ4ayeYZTUQ99Ccu5hpgAWzfLPqj/view), [Xception](http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth) and put them under `./models/`  .

* Run `python train_CoarseSN.py` to train the coarse segmentation network for roughly segmenting skin lesions.

* The segmentation network structure is defined in `./net/models.py`, and the hybrid loss is defined in `./net/loss.py`.

### 3. Generating coarse masks
* Run `python generate_Coarse_mask.py` to obtain the coarse masks for mask-CN and enhanced-SN.

### 4. Training mask-guided classification network (mask-CN)
* Run `python train_MaskCN.py` to train the mask-guided classification network for skin lesion classification.

* The classification network structure is defined in `./net/models.py`, and the loss is cross-entropy loss.

### 5. Training enhanced segmentation network (enhanced-SN)
* Run `python train_EnhancedSN.py` to train the enhanced segmentation network for more accurate skin lesion segmentation.

### 6. Evaluation
* Run `python eval_MaskCN.py` and `python eval_EnhancedSN.py` to start the evaluation.

### 7. Citation
If this code is helpful for your study, please cite:

```
@ARTICLE{8990108,
  author={Y. {Xie} and J. {Zhang} and Y. {Xia} and C. {Shen}},
  journal={IEEE Transactions on Medical Imaging}, 
  title={A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification}, 
  year={2020},
  volume={39},
  number={7},
  pages={2482-2493},}
```

### 8. Acknowledgements
The codes for Deeplabv3+ network and Xception network are reused from the [YudeWang](https://github.com/YudeWang/deeplabv3plus-pytorch) and [Cadene](https://github.com/Cadene/pretrained-models.pytorch).<br />
Thanks to [YudeWang](https://github.com/YudeWang/deeplabv3plus-pytorch) and [Cadene](https://github.com/Cadene/pretrained-models.pytorch) for the pretrained weights for Deeplabv3+ network and Xception network.

### Contact
Yutong Xie (xuyongxie@mail.nwpu.edu.cn)
