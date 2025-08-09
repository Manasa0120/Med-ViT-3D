# MedViT-3D: A Robust Vision Transformer for Generalized 3D Medical Image Classification

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2302.09462)
[![Paper](https://img.shields.io/badge/Elsevier-CIBM-blue)](https://doi.org/10.1016/j.compbiomed.2023.106791)

This repo is the official implementation of MedViT-3D: A Robust Vision Transformer for Generalized 3D Medical Image Classification.


  
## Train & Test --- Prepare data
- (beginner friendly🍉) To train/evaluate MedViT on Cifar/Imagenet/CustomDataset follow ["CustomDataset"](https://github.com/Omid-Nejati/MedViT/blob/main/CustomDataset.md). 
- (New version) Updated the code ["Instructions.ipynb"](https://github.com/Omid-Nejati/MedViT/blob/main/Instructions.ipynb), added the installation requirements and adversarial robustness. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omid-Nejati/MedViT/blob/main/Instructions.ipynb)


## Introduction
Medical image classification is a critical step in medical image analysis. Convolutional Neural Networks (CNNs) have advanced existing medical systems for automatic disease diagnosis. The MedViT model presents a hybrid CNN-Transformer architecture optimized for 2D medical image classification [1]. This model does not work well with volumetric data and has a high computational requirement. The objective of this project is to overcome the two limitations by introducing MedViT-3D, a model that can classify volumetric 3D images such as CT scans, MRI, OCT etc, with high accuracy. To enhance generalizability, we propose a cross-modality transfer strategy where the model is trained on X-ray images and evaluated on MRI data. Our work aims to extend the model to support volumetric 3D data, enhance the performance of MedViT. 
<div style="text-align: center">
<img src="images/result.png" title="MedViT-S" height="60%" width="60%">
</div>
Figure 1. Comparison between MedViTs and the baseline ResNets, in terms of average ACC-Parameters and average AUC-Parametrs trade-off over all 2D datasets.</center>


## Overview

<div style="text-align: center">
<img src="images/structure.png" title="MedViT-S" height="75%" width="75%">
</div>
Figure 2. The overall hierarchical architecture of MedViT.</center>

## ImageNet Pre-train
We provide a series of MedViT models pretrained on ILSVRC2012 ImageNet-1K dataset.

| Model      |   Dataset   | Resolution  | Acc@1 | ckpt   |  
|------------|:-----------:|:----------:|:--------:|:--------:|


## Visualization

Visual inspection of MedViT-T and ResNet-18 using Grad-CAM on MedMNIST-2D datasets. The green rectangles is
used to show a specific part of the image that contains information relevant to the diagnosis or analysis of a medical condition,
where the superiority of our proposed method can be clearly seen.
![MedViT-V](images/visualize.png)
<center>Figure 3. The heat maps of the output feature from ResNet and MedViT.</center>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Omid-Nejati/MedViT&type=Date)](https://star-history.com/#Omid-Nejati/MedViT&Date)

## Acknowledgement
We heavily borrow the code from [RVT](https://github.com/vtddggg/Robust-Vision-Transformer) and [LocalViT](https://github.com/ofsoundof/LocalViT).

## Contact Information

