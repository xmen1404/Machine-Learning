import os
import sys
import time
import datetime
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import torch.nn.functional as F
import urllib
import glob
import skimage.io as skio
from torch.utils.data import Dataset
import random
import albumentations.augmentations.transforms as A

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class Relabel:
    def __init__(self, inlabel, outlabel):
        self.inlabel = inlabel
        self.outlabel = outlabel

    def __call__(self, tensor):
        tensor[tensor == self.inlabel] = self.outlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(2):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:
    def __init__(self, n=9):
        self.cmap = colormap(10)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])


    def __call__(self, gray_image): # convert from 2D label into 2D image of colorized label
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class CoE_Dataset(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, _train=False):
        self.images_root = os.path.join(root, 'Images')
        self.labels_root = os.path.join(root, 'Labels')
        self._train = _train

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]

        random.shuffle(self.filenames)

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.A_transform = A.Compose([
            A.Rotate(limit=40, p=0.9), 
            A.HorizontalFlip(p=0.5), 
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            A.OneOf([
                A.Blur(blur_limit=2, p=0.5), 
                A.ColorJitter(p=0.5)
            ], p=1.0)
        ])

    def spm_transform(self, image, label): 
#         image = ImageEnhance.Sharpness(image)
#         image = image.enhance(random.uniform(1, 4))
        
        image = image.resize((256, 256))
        label = label.resize((256, 256))
        
#         if (random.uniform(0, 1) > 0.5) and (self._train == True): 
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             label = label.transpose(Image.FLIP_LEFT_RIGHT)
            
#             deg = random.uniform(-30, 30)
#             image = image.rotate(deg)
#             label = label.rotate(deg)
        if self._train == True: 
            image = np.array(image)
            label = np.array(label)
            augmented = self.A_transform(image=image, mask=label)
            image = Image.fromarray(augmented['image']).convert('RGB')
            label = Image.fromarray(augmented['mask']).convert('P')
        
        return image, label
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        
        image, label = self.spm_transform(image, label)
        
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename

    def __len__(self):
        return len(self.filenames)


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    epsilon = 1e-12
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    
    return area_inter.float(), area_union.float() + epsilon