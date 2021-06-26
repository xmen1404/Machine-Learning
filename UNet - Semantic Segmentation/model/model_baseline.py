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
import albumentations as A

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)
    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        ]
        self.up = nn.Sequential(*layers)

        layers = [
            nn.Conv2d(2*out_channels, out_channels, 3, 1, 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x_old, x_new):
        x_new = self.up(x_new)
        x = torch.cat([x_old, x_new], dim=1)
        return self.conv(x)
    
class Inp(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        self.inp = nn.Sequential(*layers)

    def forward(self, x):
        return self.inp(x)
    
class UNet(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        # input = B * 3 * 128 * 128
        # output = B * 128 * 128
        
        self.inp = Inp(3, 10)
        self.en1 = Down(10, 32)
        self.en2 = Down(32, 64)
        self.en3 = Down(64, 128)
        self.en4 = Down(128, 256)
        self.en5 = Down(256, 512, dropout=False)
        self.en6 = Down(512, 1024)
#         self.en7 = Down(480*2, 480*4)
        
#         self.dec1 = Up(480*4, 480*2)
        self.dec2 = Up(1024, 512)
        self.dec3 = Up(512, 256)
        self.dec4 = Up(256, 128)
        self.dec5 = Up(128, 64)
        self.dec6 = Up(64, 32, dropout=False)
        self.dec7 = Up(32, 10)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x): 
        inp = self.inp(x)
        en1 = self.en1(inp) # 32
        en2 = self.en2(en1) # 64
        en3 = self.en3(en2) # 128
        en4 = self.en4(en3) # 256
        en5 = self.en5(en4) # 512
        en6 = self.en6(en5) # 1024
        # center
        dec2 = self.dec2(en5, en6) # 512
        dec3 = self.dec3(en4, dec2) # 256
        dec4 = self.dec4(en3, dec3) # 128
        dec5 = self.dec5(en2, dec4) # 64
        dec6 = self.dec6(en1, dec5) # 32
        dec7 = self.dec7(inp, dec6) # 10
        
        return dec7