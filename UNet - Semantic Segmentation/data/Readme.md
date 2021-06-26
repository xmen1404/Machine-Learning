# Processing Data

Originally, there are 1000 pairs of image and label, and this is clearly too small for a convolutional neural network to work well. Original images are cropped 4 corners and 1 center crop with size = 256. Images are then normalized with pixel values mean and standard deviation calculated manually from the dataset. Total 4500 images are completed for main training process.
