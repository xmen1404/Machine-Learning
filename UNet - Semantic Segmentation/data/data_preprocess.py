import numpy as np
from PIL import Image, ImageEnhance
import random
import os
import torchvision.transforms as transforms
import torch
import sys

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

os.makedirs(train_image_root, exist_ok=True)
os.makedirs(train_label_root, exist_ok=True)
os.makedirs(val_image_root, exist_ok=True)
os.makedirs(val_label_root, exist_ok=True)

filenames = [image_basename(f) for f in os.listdir(base_image_root)]
random.shuffle(filenames)


total_size = len(filenames)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

# find mean of shape of images to resize
total_shape = [0, 0]
for x in filenames: 
	image_name = x + '.jpg'
	image = Image.open(os.path.join(base_image_root, image_name))
	image = np.array(image)
	
	total_shape[0] += image.shape[0]
	total_shape[1] += image.shape[1]
total_shape[0] = int(total_shape[0] // len(filenames))
total_shape[1] = int(total_shape[1] // len(filenames))

idx = 0 # index of names

# generate train dataset
for itr in range(train_size): 
	image_name = filenames[itr] + '.jpg'
	label_name = filenames[itr] + '.png'

	image = Image.open(os.path.join(base_image_root, image_name))
	label = Image.open(os.path.join(base_label_root, label_name))

	# resize_image = image.resize((256, 256))
	# resize_label = label.resize((256, 256))

	image = image.resize((total_shape[1], total_shape[0]))
	label = label.resize((total_shape[1], total_shape[0]))

	# crop 4 corners + center
	crop_transform = transforms.FiveCrop(256)
	crop_image = list(crop_transform(image))
	crop_label = list(crop_transform(label))

	# idx += 1
	# resize_image.save(os.path.join(train_image_root, str(idx) + '.jpg'))
	# resize_label.save(os.path.join(train_label_root, str(idx) + '.png'))

	for x in range(5): 
		idx += 1

		img = crop_image[x]
		lab = crop_label[x]

		img.save(os.path.join(train_image_root, str(idx) + '.jpg'))
		lab.save(os.path.join(train_label_root, str(idx) + '.png'))

	# random rotate + center crop
	# deg = random.uniform(-30, 30)
	# rot_image = image.rotate(deg)
	# rot_label = label.rotate(deg)

	# center_crop = transforms.CenterCrop(256)
	# rot_image = center_crop(rot_image)
	# rot_label = center_crop(rot_label)

	# idx += 1
	# rot_image.save(os.path.join(train_image_root, str(idx) + '.jpg'))
	# rot_label.save(os.path.join(train_label_root, str(idx) + '.png'))

# generate validation dataset
idx = 0
for itr in range(train_size, train_size + val_size): 
	image_name = filenames[itr] + '.jpg'
	label_name = filenames[itr] + '.png'

	image = Image.open(os.path.join(base_image_root, image_name))
	label = Image.open(os.path.join(base_label_root, label_name))

	# image = image.resize((total_shape[1], total_shape[0]))
	# label = label.resize((total_shape[1], total_shape[0]))

	# resize but not crop
	image = image.resize((256, 256))
	label = label.resize((256, 256))

	idx += 1
	image.save(os.path.join(val_image_root, str(idx) + '.jpg'))
	label.save(os.path.join(val_label_root, str(idx) + '.png'))







