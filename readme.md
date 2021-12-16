# CS492i_project_VinCXR_512
This is term project from class CS492i Introduction to Deep learning \
In this project we try to use efficiendet to detect the abnormality in CXR image base on the dataset https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data. Due to computation time and storage space issue, we use the 512*512 png image version from https://www.kaggle.com/xhlulu/vinbigdata instead of original ones \

# Implementation
We use the Efficiendet implementation from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch. \
We add the file Vin_CXR.ipython to call train.py function \
We add the file VIn_CXR_classifier to train the classifier and also efficientdet backbone \
We modify train.py and efficiendet/dataset.py to use custom augmentation \
We modify backbone.py and train.py to use custom backbone and load custom backbone \
 

