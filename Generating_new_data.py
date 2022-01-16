#!git clone https://github.com/hoangp/isbi-datasets.git ## run once to get the data set

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import albumentations as A



images_path= "/content/isbi-datasets/data/images" # original images folder
labels_path = "/content/isbi-datasets/data/labels"
images=[] # to store paths of images from the folder
labels=[]
for im in os.listdir(images_path):  # read image names from the folder and append its path into "images" and "labels" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(labels_path):     
    labels.append(os.path.join(labels_path,msk))
aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    A.GridDistortion(p=1)
    ]
)

directory_img = "new_images/new_img100" # path to store new images
  
directory_lbl = "new_images/new_lbl100"

parent_dir = "/content/isbi-datasets/data/"
  
new_img_path = os.path.join(parent_dir, directory_img)
new_lbl_path = os.path.join(parent_dir, directory_lbl)
if not os.path.exists(new_img_path):
        os.mkdir(new_img_path)

if not os.path.exists(new_lbl_path):
        os.mkdir(new_lbl_path)



number_of_new_images=100

for i in range(1, number_of_new_images+1, 1):
    number = random.randint(00, len(images)-1)  #randum number 
    image = images[number]
    label = labels [number]
    print(image, label) 
    original_image = io.imread(image)
    original_label = io.imread(label)
    
    augmented = aug(image=original_image, label=original_label)
    transformed_image = augmented['image']
    transformed_label = augmented['label']

        
    new_image_path= "%s/new_image_%s.png" %(new_img_path, i) #generating new images in the new address
    new_label_path = "%s/new_label_%s.png" %(new_lbl_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
  
