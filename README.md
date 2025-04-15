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

## Performance
![image](https://github.com/user-attachments/assets/0b03c3fc-c837-4763-8bff-be4b8f3de5e8)

