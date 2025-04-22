import csv
import json
import numpy as np
import os
import pandas as pd

with open("./250327_v1/pred_09.json", "r") as f:
    json_data = json.load(f)

image_dict = {}

for item in json_data:
    image_id = int(item["image_id"])
    x_min = item["bbox"][0]
    category_id = int(item["category_id"]) - 1
    if category_id == -1:
        category_id = 9

    if image_id not in image_dict:
        image_dict[image_id] = []
    image_dict[image_id].append((x_min, category_id))

output = []

for image_id in range(1, 13069):
    if image_id in image_dict:
        sorted_cats = sorted(image_dict[image_id], key=lambda x: x[0])
        pred_label = ''.join(str(cat_id) for _, cat_id in sorted_cats)
    else:
        pred_label = "-1"
    output.append({"image_id": image_id, "pred_label": pred_label})

df = pd.DataFrame(output)
df.to_csv("./250327_v1/pred_09.csv", index=False)