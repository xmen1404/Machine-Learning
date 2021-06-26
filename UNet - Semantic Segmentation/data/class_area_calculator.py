import numpy as np
from PIL import Image, ImageEnhance
import random
import os
import torchvision.transforms as transforms
import torch
import sys

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
target_transform = [ToLabel(),
    Relabel(4, 3),
    Relabel(6, 4),
    Relabel(7, 5),
    Relabel(14, 6),
    Relabel(15, 7),
    Relabel(19, 8),
    Relabel(255, 9),
]
target_transform = transforms.Compose(target_transform)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])
def image_color_aug(img): 
	img = ImageEnhance.Contrast(img)
	img = img.enhance(random.uniform(1, 3))

	img = ImageEnhance.Brightness(img)
	img = img.enhance(random.uniform(1, 1.5))
	
	img = ImageEnhance.Sharpness(img)
	img = img.enhance(random.uniform(1, 3))

	return img


# transform methods

base_data_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\CoE202Spring_finalproject_dataset'
base_image_root = os.path.join(base_data_root, 'Images')
base_label_root = os.path.join(base_data_root, 'Labels')

train_image_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\train\Images'
train_label_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\train\Labels'
val_image_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\validation\Images'
val_label_root = r'C:\Users\Acer\OneDrive - kaist.ac.kr\Documents\KAIST\Academic Courses\2021 Spring\CoE202\final project\provided materials\Custom_dataset\validation\Labels'

filenames = [image_basename(f) for f in os.listdir(base_image_root)]
random.shuffle(filenames)


total_size = len(filenames)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

# find mean of shape of images to resize
total_shape = [0, 0]
total_area = [0 for x in range(10)]
cnt = 0
for x in filenames: 
	# image_name = x + '.jpg'
	# image = Image.open(os.path.join(base_image_root, image_name))
	# image = np.array(image)
	
	# total_shape[0] += image.shape[0]
	# total_shape[1] += image.shape[1]

	label_name = x + '.png'
	label = Image.open(os.path.join(base_label_root, label_name))
	label = target_transform(label)
	for y in range(10): 
		tmp = label[label==y]
		total_area[y] += int(tmp.shape[0])
	cnt += 1
	print('Done file #', cnt)

sum_area = 0.0
for x in range(9): 
	print(x, total_area[x])
	sum_area += 1.0 * total_area[x]
# exclude #9 
for x in range(9): 
	print('weight for class #', x, ' = ', 1.0 / (1.0 * total_area[x] / sum_area))





