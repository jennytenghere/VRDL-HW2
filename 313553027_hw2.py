import csv
import cv2 as cv
import json
import matplotlib.pyplot as plt
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
from torchvision.models import ResNet50_Weights
from torchvision.models import resnext50_32x4d
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
import torchmetrics
from tqdm import tqdm

# clear memory
import gc
gc.collect()
torch.cuda.empty_cache()

# set random seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set gpu device
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mAP calculation
map_metric = torchmetrics.detection.MeanAveragePrecision().to(device)


# transformation for label
def label_transform(label):
    return torch.tensor(label, dtype=torch.long)


# transformation for bounding box
def bbox_transform(bbox):
    return torch.tensor(bbox, dtype=torch.float32)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, type):
        super().__init__()
        # image folder's root
        self.root_dir = root_dir
        # training or validation
        self.type = type
        if type == "train" or type == "valid":
            # image folder
            self.folder_path = os.path.join(self.root_dir, type)
            # json file name
            json_file_name = type+".json"
            with open(os.path.join(root_dir, json_file_name),
                      "r", encoding="utf-8") as file:
                # load json file
                data = json.load(file)
                # get images, annotations, categories
                self.images = data.get("images", [])
                self.annotations = data.get("annotations", [])
                self.categories = {}
                # map category id with category name
                for cat in data.get("categories", []):
                    cat_id = cat["id"]
                    cat_name = cat["name"]
                    self.categories[cat_id] = cat_name
                # image to annotation mapping
                # key initialization
                self.img_to_anns = {}
                for image in self.images:
                    self.img_to_anns[image["id"]] = []
                # value: annotation
                for anno in self.annotations:
                    image_id = anno["image_id"]
                    self.img_to_anns[image_id].append(anno)
                # transform to tensor
                self.img_transform = transforms.Compose([
                    # ColorJitter data augmentation one
                    # transforms.ColorJitter(
                    #     brightness=0.2, contrast=0.2,
                    #     saturation=0.2, hue=0.1),
                    # ColorJitter data augmentation two
                    # transforms.ColorJitter(
                    #     brightness=0.3, contrast=0.3,
                    #     saturation=0.3, hue=0.3),
                    transforms.ToTensor()
                ])
                self.label_transform = label_transform
                self.bbox_transform = bbox_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get image's information
        img_data = self.images[idx]
        # path of the image
        img_path = os.path.join(self.folder_path, img_data["file_name"])
        # load the image
        image = Image.open(img_path).convert("RGB")
        # get the annotation of the image by image id
        anno_data = self.img_to_anns[img_data["id"]]
        bboxes = []
        labels = []
        # append the information of each annotation
        for anno in anno_data:
            if "bbox" in anno:
                # anno["bbox"]:[x, y, w, h] ->  [x, y, x+w, y+h]
                bboxes.append(
                    [anno["bbox"][0],
                     anno["bbox"][1],
                     anno["bbox"][0] + anno["bbox"][2],
                     anno["bbox"][1] + anno["bbox"][3]])
                labels.append(anno["category_id"])
        # do data augmentation and transform into tensor
        image = self.img_transform(image)
        bboxes = self.bbox_transform(bboxes)
        labels = self.label_transform(labels)
        # turn into faster r-cnn input format
        target = {"boxes": bboxes, "labels": labels}

        return image, target


#  [(a1, b1), (a2, b2)] → ([a1, a2], [b1, b2])
def a_fn(batch):
    return tuple(zip(*batch))


# I didn't use this method
# This make all the images of a batch turn into the same size
def pad_collate_fn(batch):
    # max image height and width in this batch
    max_h = max([img.shape[1] for _, img, _ in batch])
    max_w = max([img.shape[2] for _, img, _ in batch])
    # print("max_h", max_h)
    # print("max_w", max_w)

    padded_imgs = []
    targets = []

    for img, target in batch:
        # print("h/img.shape[1]", img.shape[1])
        # print("w/img.shape[2]", img.shape[2])
        pad_w = max_w - img.shape[2]
        pad_h = max_h - img.shape[1]
        # print("pad_h", pad_h)
        # print("pad_w", pad_w)
        # the size of the padding
        padding = (0, 0, pad_w, pad_h)
        # pad zeros to the image
        padded_img = F.pad(img, padding, "constant", 0)
        padded_imgs.append(padded_img)

        # adjust the bounding box
        if "boxes" in target:
            bboxes = target["boxes"]
            if len(bboxes) > 0:
                bboxes = torch.tensor(bboxes)
                bboxes[:, [0, 2]] += padding[0]
                bboxes[:, [1, 3]] += padding[1]
            target["boxes"] = bboxes

        targets.append(target)

    # stack the padded images into batch
    padded_imgs = torch.stack(padded_imgs)

    return padded_imgs, targets


def train(model, train_loader, optimizer, epoch):
    # send the model to gpu
    model = model.to(device)
    # training mode
    model.train()
    # loss initialization
    total_loss = 0
    print("Epoch {epoch+1} Training")
    for images, targets in tqdm(train_loader):
        # send images and targets to gpu
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # loss calculation
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()
        # clear optimizer
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update
        optimizer.step()
    # average loss for each batch
    avg_loss = total_loss / len(train_loader)
    print(f"Traing Loss: {avg_loss:.4f}")

    return avg_loss


def evaluate(model, val_loader):
    # nontraining mode
    model.eval()
    # map calculator reset
    map_metric.reset()

    with torch.no_grad():
        print("Evaluating")
        for images, targets in tqdm(val_loader):
            # send to gpu
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device)
                        for k, v in t.items()} for t in targets]
            # input the data
            outputs = model(images)
            # calculate mAP
            map_metric.update(outputs, targets)

    map_result = map_metric.compute()

    print(f"Validation mAP: {map_result['map']:.4f}")

    return map_result["map"]


# change faster r-cnn predictor into multiple layer structure
class NewBoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)

        self.cls_score = nn.Linear(256, num_classes)
        self.bbox_pred = nn.Linear(256, num_classes * 4)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        cls_score_output = self.cls_score(x)
        bbox_pred_output = self.bbox_pred(x)
        return cls_score_output, bbox_pred_output


if __name__ == "__main__":
    # set the number of epoch
    num_epochs = 15
    # version name (this will be the output folder name)
    version = "250412_v8"
    # make a new folder for saving the results
    if not os.path.exists("./"+version):
        os.mkdir("./"+version)

    # load the dataset
    train_dataset = CustomImageDataset("./nycu-hw2-data", "train")
    val_dataset = CustomImageDataset("./nycu-hw2-data", "valid")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=a_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=4, collate_fn=a_fn)

    # original model version
    # load the model
    model = models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # number of classes for classification + background
    num_classes = len(train_dataset.categories) + 1
    # modify the layer, change the output channels
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # use multiple layer predictor
    # model.roi_heads.box_predictor = NewBoxPredictor(in_features, num_classes)

    # use resnext50_32x4d as backbone
    # resnext_backbone = resnext50_32x4d(pretrained=True,
    #                                    norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # backbone_with_fpn = BackboneWithFPN(
    #     backbone=resnext_backbone,
    #     return_layers={'layer1': '0', 'layer2': '1',
    #                    'layer3': '2', 'layer4': '3'},
    #     in_channels_list=[256, 512, 1024, 2048],
    #     out_channels=256
    # )
    # model = FasterRCNN(backbone=backbone_with_fpn,
    #                    num_classes=len(train_dataset.categories) + 1)

    # show the architecture of the model
    print(model)
    # send to gpu
    model.to(device)
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # initialization for finding highest map
    best_map = 0
    # save the output for drawing the figure
    train_loss_list = []
    val_map_list = []
    # 1 epoch = training+test
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, epoch)
        val_map = evaluate(model, val_dataloader)
        # save the trained model
        torch.save(model.state_dict(),
                   "./" + version + "/" + version + "_" + str(epoch) + ".pth")
        # append the outputs to lists for drawing the figure
        train_loss_list.append(train_loss)
        val_map_list.append(val_map)
        # write the outputs into txt file
        with open("./" + version + "/" + version + ".txt", "a") as f:
            f.write(f"Epoch {epoch+1} {train_loss:.4f} {val_map:.4f}\n")
        # save the best model
        if val_map > best_map:
            best_map = val_map
            torch.save(
                model.state_dict(),
                "./" + version+"/" + version + "_best" + str(epoch) + ".pth")
            print("Best model saved!")

    # set the size of the figure
    plt.figure(figsize=(12, 6))

    # training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_loss_list,
             label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss of Epoch 1~{num_epochs}')

    # validation map
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_map_list,
             label='Validation mAP', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.title(f'mAP of Epoch 1~{num_epochs}')

    plt.tight_layout()
    # save the image
    image_path = f"./{version}/{version}_{num_epochs}.png"
    plt.savefig(image_path)
    plt.close()

    print("Training Finished!")
