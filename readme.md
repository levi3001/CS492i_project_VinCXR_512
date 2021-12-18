# CS492i_project_VinCXR_512
This is term project from class CS492i Introduction to Deep learning \
In this project we try to use efficiendet to detect the abnormality in CXR image base on the dataset https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data. Due to computation time and storage space issue, we use the 512*512 png image version from https://www.kaggle.com/xhlulu/vinbigdata instead of original ones 

# Implementation
We use the Efficiendet implementation from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch. \
We add the file Vin_CXR.ipython to call train.py function \
We add the file VIn_CXR_classifier to train the classifier and also efficientdet backbone \
We modify train.py and efficiendet/dataset.py to use custom augmentation \
We modify backbone.py and train.py to use custom backbone and load custom backbone 
 
# Reproduce the result
To train efficientdet detector you need to excecute the file vin_CXR_512.ipynb\
You need to downloads the datasets from https://drive.google.com/file/d/1L44r7oq4kA7HNFINQgiSb59dcfkFk7qD/view?usp=sharing and put it in cs492i_project folder on your google drive. The path to the dataset when you mount to google drive should be: gdrive/MyDrive/cs492i_project/Vin_CXR_512.zip \
Since efficientnet b3 is to big, we can not upload it to github. To use pretrained efficientnet b3 to train you need to download it from https://drive.google.com/file/d/1IuVHtVrR_q3XoG1dqwuEcD3DJN8qWcZC/view?usp=sharing\
Some useful comma:\
Train efficientdet b0 with custom pretrained model:\
 python train.py -c 0 -cb 0 -p Vin_CXR_512 --batch_size 16 --lr 1e-3 --num_epochs 10 --custom_backbone /content/gdrive/MyDrive/cs492i_project/efficientnet_b0_best.pth --head_only True \
Train efficientdet b0 with efficientnet b3 backbone:\
 python train.py -c 0 -cb 3 -p Vin_CXR_512 --batch_size 16 --lr 1e-3 --num_epochs 10 --custom_backbone /content/gdrive/MyDrive/cs492i_project/efficientnet_b3_best_2.pth --head_only True \
Train efficientdet b0 from pretrained efficientdet b0: \
 python train.py -c 0 -cb 0 -p Vin_CXR_512 --batch_size 16 --lr 1e-3 --num_epochs 50 --load_weights path/to/your/efficientdet/b0/weight

To train efficientnet detector, execute the file vin_CXR_512_classifier.ipynb\
The file utilizes the same dataset as above.\
The overall structure is as following:\
1. Install necessary packages
2. Build Data Loader
3. Define loss function
4. Train models and Evaluation
5. Some statistics for models evaluation
\
Those help us obtain the backbone structure of the model, with several different structures: Resnet50, EfficientNet. Overall, we choose the efficientnet.\
