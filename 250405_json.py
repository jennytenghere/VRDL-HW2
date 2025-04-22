# openpyxl
import csv
import cv2 as cv
import json
# import math
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
from sklearn.metrics import confusion_matrix
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models, ops
from torch.utils.data import Dataset
# from torchvision.models import ResNet50_Weights
# from torchvision.models import ViT_B_16_Weights, VGG16_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
import gc
gc.collect()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.folder_path = os.path.join(self.root_dir, "test")
        self.img_paths = []
        for img_file in os.listdir(self.folder_path):
            img_path = os.path.join(self.folder_path, img_file)
            self.img_paths.append(img_path)
        # for img_file in os.listdir(self.folder_path):
        #     print("Found file:", img_file)
        #     img_path = os.path.join(self.folder_path, img_file)
        #     try:
        #         img = Image.open(img_path).convert("RGB")
        #     except Exception as e:
        #         print(f"Error loading {img_path}: {e}")
        #     self.img_paths.append(img_path)
        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)
        img_name = img_path.split("\\")[2][:-4]
        return image, img_name


# def collate_fn(batch):
#     images, targets = zip(*batch)
#     return list(images), list(targets)

def a_fn(batch):

    return tuple(zip(*batch))

def test(model, test_loader, json_path):
    results = []
    model.eval()

    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Testing"):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for i in range(len(outputs)):
                # print("-------------------------------------")
                # print(outputs[i])
                image_name = image_names[i]
                box = outputs[i]['boxes'].cpu().numpy()
                score = outputs[i]['scores'].cpu().numpy()
                label = outputs[i]['labels'].cpu().numpy()
                for j in range(len(box)):
                    if score[j] > 0.9:
                        x_min, y_min, x_max, y_max = box[j]
                        
                        one_result = {
                            "image_id": int(image_name),
                            "bbox": [float(x_min), float(y_min), float(x_max-x_min), float(y_max-y_min)],
                            "score": float(score[j]),
                            "category_id": int(label[j])
                        }
                        results.append(one_result)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    dataset = CustomImageDataset("./nycu-hw2-data")
    # print("Dataset length:", len(dataset))
    # img, path = dataset[0]
    # print("First image shape:", img.shape)
    # print("First image path:", path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=a_fn)

    model = models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    state_dict = torch.load("./250327_v1/faster_rcnn_v15.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    
    test(model, dataloader, "./250327_v1/pred_09.json")
    
    # for images, image_names in dataloader:
    #     print(image_names)