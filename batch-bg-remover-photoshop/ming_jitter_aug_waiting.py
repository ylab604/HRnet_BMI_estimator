import imgaug.augmenters as iaa
import numpy as np
import cv2
import os
from tqdm import tqdm

test_image = cv2.imread('Image_train_remove_aug_again/0_F_15_162560_6123497.jpg.jpg')
h,w,c= test_image.shape


aug = iaa.Sequential([
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.WithChannels(0, iaa.Add((0, 20))),
	iaa.WithChannels(1, iaa.Add((0, 40))),
	iaa.WithChannels(2, iaa.Add((0, 256))),
    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
])

image_aug = aug(image=test_image)
cv2.imshow('hi',image_aug)
