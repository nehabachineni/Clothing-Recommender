# !pip3 install pickle5
import pickle5 as pickle
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
import os
import math
import copy

from sklearn.model_selection import train_test_split
from pathlib import Path


# deleting exp folder
location = "runs/detect/"
dirs = listdir(location)
# print(dirs)
for i in range(len(dirs)):
  path = Path(location+dirs[i])
  shutil.rmtree(path)
  # os.remove(path)


# code to run and detect img annotations
os.system("python detect.py --weights weights/best.pt --img 640 --conf 0.4 --source ./inference/images/")

cat = ["short sleeve top","long sleeve top","short sleeve outwear","long sleeve outwear","vest","sling","shorts","trousers","skirt", "short sleeve dress", "long sleeve dress", "vest dress","sling dress"]

def e_dist(row1,row2):
  distance=0
  for i in range (len(row1)):
    distance += (row1[i]-row2[i])**2
  return math.sqrt(distance)

  # with open('database.pickle', 'rb') as handle:
#     db = pickle.load(handle)  

# def create_dataset(dataset):
#   for catoo in tqdm(dataset):
#     img_path = Path(f"data/myntra_r_f/{catoo}")
#     img_path.mkdir(parents=True, exist_ok=True)
  
#     for i in range(len(dataset[catoo])):
#       f_test_img=[[0]*3072]
#       key = list(dataset[catoo][i].keys())
#       img = dataset[catoo][i][key[0]]
#       # img = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
#       r_img = cv2.resize(np.float32(img), (32, 32),interpolation = cv2.INTER_NEAREST)
#       # r_img = img.resize((32, 32),Image.ANTIALIAS)
#       f_img = np.reshape(r_img,[3072,1],order='F')
#       f_test_img = np.hstack((f_test_img, np.atleast_2d(f_img).T))
#     cv2.imwrite(str(img_path / key[0])+".jpeg",f_test_img)
#     # f_test_img.save(str(img_path / key[0]), "JPEG")

# def create_dataset(dataset):
#   for catoo in tqdm(dataset):
#     img_path = Path(f"data/myntra_r/{catoo}")
#     img_path.mkdir(parents=True, exist_ok=True)
  
#     for i in range(len(dataset[catoo])):
#       key = list(dataset[catoo][i].keys())
#       img = dataset[catoo][i][key[0]]
#       r_img = img.resize((32, 32),Image.ANTIALIAS)
#       r_img.save(str(img_path / key[0]), "JPEG")

    
# def create_dataset(dataset):
#   for catoo in tqdm(dataset):
#     img_path = Path(f"data/myntra_rr/{catoo}")
#     img_path.mkdir(parents=True, exist_ok=True)
  
#     for i in range(len(dataset[catoo])):
#       key = list(dataset[catoo][i].keys())
#       img = dataset[catoo][i][key[0]]
#       r_img = img.resize((64, 64),Image.ANTIALIAS)
#       r_img.save(str(img_path / key[0]), "JPEG")

    
# create_dataset(db)



# #-----------------------------------getting input from yolo----------------------------------

# # with open('database.pickle', 'rb') as handle:
# #     db = pickle.load(handle)

# #input for knn
# with open('iteams.pickle', 'rb') as handle:
#     items = pickle.load(handle)
with open('cimg.pickle', 'rb') as handle:
    imgs = pickle.load(handle)



# types = []
# img_set = []
# l_items = len(items[0])

# for cato in items[0]:
#   cato=cat[int(cato)]
#   types.append(cato)

# print(types)

# # img = Image.fromarray(imgs[0], 'RGB')
# img = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
# # type(array(imgs[0]))
# plt.imshow(img)


# int(items[0][1])
# items


#input from yolov5
with open('cimg.pickle', 'rb') as handle:
    img_full = pickle.load(handle)

input_y = {}
o_images = []
yolo_output_path = Path("runs/detect/exp/crops")
o_ty = listdir(yolo_output_path)
# print(o_ty)
for pat in o_ty:
  o_ty_path = Path(yolo_output_path/pat)
  crop_img = Path(o_ty_path/listdir(o_ty_path)[0])
  img = cv2.imread(str(crop_img))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  o_images.append(img)
  input_y[pat]=img
  
# plt.imshow(input_y["long sleeve top"])
# plt.imshow(input_y["shorts"])

input_y_r={}
for typ in input_y:
  test_img = cv2.resize(input_y[typ], (64, 64),interpolation = cv2.INTER_NEAREST)
  f_test_img = np.reshape(test_img,[12288,1],order='F')
  input_y_r[typ] = f_test_img
  # print("f_test_img.shape : ",input_y_r[typ].shape)



# # test_img = img.resize((64, 64),Image.ANTIALIAS)
# test_img = cv2.resize(img, (64, 64),interpolation = cv2.INTER_NEAREST)
# plt.imshow(test_img)

# f_test_img = np.reshape(test_img,[12288,1],order='F')
# print("f_test_img.shape : ",f_test_img.shape)



# img_folders_path = Path("data/myntra_r")    #this is for 32,32 reduced imgs
img_folders_path = Path("data/myntra_rr")    #this is for 64,64 reduced imgs
o_img_folders_path = Path("data/myntra")
types = ['long sleeve top']  #dummie this is only for test
all = []
for ty in input_y_r:
  ty_path=Path(img_folders_path/ty)
  # print(ty_path)
  ty_imgs = list(listdir(ty_path))
  # print(ty_imgs)
  nu_imgs = len(ty_imgs)
  dist = np.zeros(nu_imgs)
  # img = cv2.imread(str(ty_path/ty_imgs[0]))
  # print(img)
  for i in range(nu_imgs):
    sel_img = cv2.imread(str(ty_path/ty_imgs[i]))
    f_img = np.reshape(sel_img,[12288,1],order='F')
    dist[i] = e_dist(input_y_r[ty],f_img)
  


  similar_img = np.argsort(dist)

  xx= []

  o_ty_path=Path(o_img_folders_path/ty)
  o_ty_imgs = list(listdir(o_ty_path))
  for i in range(10):
    img = cv2.imread(str(o_ty_path/o_ty_imgs[similar_img[i]]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xx.append(img)
  all.append(xx)
  # plt.imshow(img)

print("Done")

# plt.imshow(all[0][8])