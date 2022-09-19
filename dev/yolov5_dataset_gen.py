import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import requests
import shutil
import urllib
import PIL.Image as Image
import pandas as pd
from IPython.display import display
import numpy as np
from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from pathlib import Path
# from utils.utils import plot_results

data = {}
mypath = "train/"

files_path = [join(mypath, k) for k in listdir(mypath)]
print(files_path)
anna_s = sorted(listdir(files_path[1]))[:200]
img_s = sorted(listdir(files_path[0]))[:200]

for ann_path, img_path in zip(anna_s, img_s):
    data[ann_path] = img_path

# print(data)

clothing=[]
for ann in data:
#     print(type(ann))
    ann_path = files_path[1]+"/"+ann
    img_path = files_path[0]+"/"+data[ann]
    k=json.load(open(ann_path))
    clothing.append([k,img_path])

# print(clothing[0])

#catogoury
cat = ["short sleeve top","long sleeve top","short sleeve outwear","long sleeve outwear","vest","sling","shorts","trousers","skirt", "short sleeve dress", "long sleeve dress", "vest dress","sling dress"]

train_clothing, val_clothing = train_test_split(clothing, test_size=0.1)
# len(train_clothing), len(val_clothing)


def create_dataset(clothing, categories, dataset_type):
    images_path = Path(f"dress/images/{dataset_type}")
    labels_path = Path(f"dress/labels/{dataset_type}")

    for img_id, rtest in enumerate(tqdm(clothing)):
        image_name = f"{img_id}.jpeg"
        #         img = cv2.imread(rtest[1])
        img = Image.open(rtest[1])
        img = img.convert('RGB')
        img.save(str(images_path / image_name), "JPEG")

        img_width, img_height = img.size

        label_name = f"{img_id}.txt"
        with (labels_path / label_name).open(mode="w") as label_file:
            l = len(rtest[0]) - 2  # removing source and pair id for item count
            # rtest[0]["item1"]
            for i in range(1, l + 1):
                pts = rtest[0][f"item{i}"]["bounding_box"]
                top_l = pts[:2]
                lower_r = pts[2:]

                top_l_n = [pts[0] / img_width, pts[1] / img_height]
                lower_r_n = [pts[2] / img_width, pts[3] / img_height]

                w = lower_r_n[0] - top_l_n[0]
                h = lower_r_n[1] - top_l_n[1]
                #                 tl = [top_l[0]/]
                #                 print("pts     :",pts)
                #                 print("top_l   :",top_l)
                #                 print("lower_r :",lower_r)
                label = rtest[0][f"item{i}"]["category_name"]
                category_idx = categories.index(label)
                label_file.write(
                    f"{category_idx} {top_l_n[0] + w / 2} {top_l_n[1] + h / 2} {w} {h}\n")



create_dataset(train_clothing, cat, 'train')
create_dataset(val_clothing, cat, 'val')

# !tree dress -L 2

# %cd yolov5