
# MedViT-3D: An enhanced model with CBAM integration





## Introduction

Medical imaging plays an indispensable role in modern diagnostics, facilitating early detection, monitoring, and treatment planning for a variety of conditions. The exponential growth in medical imaging data particularly volumetric data such as CT, MRI, and OCT scans has emphasized the need for automated, accurate, and scalable diagnostic tools. While Convolutional Neural Networks (CNNs) have demonstrated remarkable success in 2D image analysis, their applicability to 3D medical data remains limited due to architectural constraints and high computational overhead.
MedViT is a model that combines two common methods, convolutional layers, which are good at capturing local details, and transformer blocks, which help understand overall patterns. It relies on an attention mechanism that can be demanding in terms of computing power. To address this, our project introduces MedViT-3D, which adjusts the original model to work with 3D scans by using 3D operations and replacing the attention block with CBAM. This allows the model to better process 3D scans like MRIs and CTs while keeping the design simple. We also explored how well the model can adapt by using a cross-modality approach i.e. training it on one type of scan and testing it on another. This helps us understand if the model can generalize across different medical image types. Through this project, we aim to build a model that can work across different scan types and help support medical decision-making.



## Overview
                                                                      Architecture
*ECB: Efficient Convolution Block
*MHCA: Multi Head Convolution Attention
*LFFN: Locally Feed Forward Network
*BN: Batch Normalization
*LTB: Local Transformer Block
*Nx: number of times to repeat this block defined by (4;2;2;2) (example: ECB x N2: N2 times ECB implementation).

<div style="text-align: center">
<img width="686" height="595" alt="image" src="https://github.com/user-attachments/assets/7c1de228-ab2f-4d61-ae58-e1177554d209" />
                                                  Fig. MedViT-3D Architecture
Legend
  
<img width="71" height="22" alt="image" src="https://github.com/user-attachments/assets/a8ef7208-3ec3-4eab-815d-0ddb18b01d83" /> MedViT3D architecture. Breakout of only first block (ECB &LTB) is shown for the explanation purpose. However, blocks across all the stages are modified as indicated by the legend.
  


<img width="79" height="26" alt="image" src="https://github.com/user-attachments/assets/487422f4-ec6d-464e-8e7a-abdfcccd9ef2" />
            
Base architecture



<img width="384" height="528" alt="image" src="https://github.com/user-attachments/assets/80bc56a1-4b03-4d25-805f-65bdf4e7f2fa" />   <img width="398" height="546" alt="image" src="https://github.com/user-attachments/assets/142e5fd4-725c-4a1c-9a11-e7e45e5e21f9" />
       

                                        Fig. Comparison of LTB (Base vs. MedViT-3D)

Legend

<img width="68" height="25" alt="image" src="https://github.com/user-attachments/assets/252e6b87-6f0c-4d73-b123-488a1263936d" />     LTB (MedViT-3D)

<img width="69" height="23" alt="image" src="https://github.com/user-attachments/assets/b71e7825-02d7-4ff5-9387-913be8ac2860" />     LTB (Base Architecture)           


</div>




## Comparative Results for 3D image classification
	

<img width="788" height="287" alt="image" src="https://github.com/user-attachments/assets/f622043b-4b93-4a8b-92f0-330341c1c07f" />


<img width="771" height="236" alt="image" src="https://github.com/user-attachments/assets/949764e9-7bd5-4585-b85e-16a6bbf9052c" />



## Citation
If you find this project useful in your research, please consider cite:
```
@article{Medvit3D with CBAM,
  title={MedViT-3D: An enhanced model with CBAM integration},
  author={Abhijeet Singh, Manasa Lakkakula, Xuan Nguyen,
  journal={Computers in Biology and Medicine},
  
}

## Contact Information


For any inquiries or questions regarding the code, please feel free to contact us directly via email:

- Abhijeet Singh: [abhijeetsingh.abs@gmail.com]
- Manasa Lakkakula: [manasa2001@gmail.com]
-Xuan Nguyen : [xuan.nguyen.intel@gmail.com]
=======

