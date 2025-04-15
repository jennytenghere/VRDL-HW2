# VRDL-HW2
StudentID: 313553027  
Name: 鄧婕妮
## Introduction
This is the second assignment of the VRDL course. The task is to apply Faster RCNN to the SVHN dataset. The model is used to detect the bounding boxes of digits
in the images, as well as classify the digit type within each box. Since training this
model takes a long time, I only experimented with a few simple versions, and their
performances were roughly the same. I tried simple data augmentation (color
adjustment), changing the backbone to ResNeXt, and modifying the prediction head
from a simple linear layer to multiple layers. A bounding box is accepted if its score is greater than 0.7.

## How to install
conda env create -f vrdl.yaml

## How to change the model architecture
1. Use ColorJitter(0.2, 0.2, 0.2, 0.1): activate line 96~98
2. Use ColorJitter(0.3, 0.3, 0.3, 0.3): activate line 100~103
3. ResNet50 with multiple layer prediction head: inactivate line 295, activate line 297
4. ResNeXt50_32*4d: inactivate line 290\~297, activate line 300\~310

## Performance
![image](https://github.com/user-attachments/assets/0b03c3fc-c837-4763-8bff-be4b8f3de5e8)

