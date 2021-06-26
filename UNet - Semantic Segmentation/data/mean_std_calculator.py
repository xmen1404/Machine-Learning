import numpy as np
from PIL import Image, ImageEnhance
import random
import os
import torchvision.transforms as transforms
import torch
import sys
import tqdm

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


# transform methods

base_data_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\CoE202Spring_finalproject_dataset'
base_image_root = os.path.join(base_data_root, 'Images')
base_label_root = os.path.join(base_data_root, 'Labels')

train_image_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\train\Images'
train_label_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\train\Labels'
val_image_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\validation\Images'
val_label_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\validation\Labels'

filenames = [image_basename(f) for f in os.listdir(base_image_root)]
total_size = len(filenames)

ToTensor = transforms.ToTensor()

tot_sum = torch.tensor([0.0, 0.0, 0.0])
sq_sum = torch.tensor([0.0, 0.0, 0.0])
tot_pixel = 0

for x in (filenames): 
	image_name = x + '.jpg'
	image = Image.open(os.path.join(base_image_root, image_name))
	image = ToTensor(image)

	tot_sum += image.sum(axis=[1, 2])
	sq_sum += (image ** 2).sum(axis=[1, 2])

	tot_pixel += int(image.shape[1] * image.shape[2])

total_mean = tot_sum / tot_pixel
total_var = (sq_sum / tot_pixel) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

print(total_mean, total_std)












