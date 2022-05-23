import imgaug.augmenters as iaa
import numpy as np
import cv2
import os
from tqdm import tqdm


image_path = 'Image_train_remove_aug_again'
all_image = os.listdir(image_path)
save_root = 'ming_augmentation_train_data'

for i in tqdm(range(len(all_image))) :
    image_root = os.path.join(image_path, all_image[i])
    test_image=cv2.imread(image_root)
    h, w, c = test_image.shape

    rand_num = np.random.rand(1)
    percent_ran = np.random.randint(80,95)
    percent_ran = percent_ran / 100

    if rand_num <= 0.5 :
        aug= iaa.Sequential([
            iaa.PadToFixedSize(width=w * percent_ran, height=h * percent_ran),
        ])
    else :
        aug = iaa.Sequential([
            iaa.CropToFixedSize(width=w * percent_ran, height=h * percent_ran)
        ])

    image_aug = aug(image=test_image)
    cv2.imwrite(os.path.join(save_root, 'crop_'+all_image[i]), image_aug)



#
#
#
#
#
# test_image = cv2.imread('Image_train_remove_aug_again/0_F_15_162560_6123497.jpg.jpg')
# h,w,c= test_image.shape
# cv2.imshow('hi',test_image)
# cv2.waitKey(0)
#
# cv2.imwrite('hihi.jpg', test_image)
#
#
#
# aug_1 = iaa.Sequential([
#     iaa.PadToFixedSize(width=w*0.8, height=h*0.8),
# ])
#
# aug_2 = iaa.Sequential([
#     iaa.CropToFixedSize(width=w*0.8, height=h*0.8)
# ])
#
# image_aug = aug(image=test_image)
# cv2.imshow('hi',image_aug)
# cv2.imwrite('check_color.jpg', image_aug)
#
# aug_image = aug.show_grid(test_image, cols=1, rows=1)
# cv2.imshow('hi',aug_image, cols=1, rows=1)