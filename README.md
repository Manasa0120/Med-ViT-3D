# CNN-tention-for-clinical-images


**Step1: preprocessing data:** 

    ** a. Purpose**: to extract MRI and XRAY data which from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia and from https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset-v2
    
    **b. How to use**: call class data = DATALOAD("mri") or data = DATALOAD("xRay")
    
    **c. Output**: it will auto return train data as data.train_ds and validation data as data.val_ds
    
