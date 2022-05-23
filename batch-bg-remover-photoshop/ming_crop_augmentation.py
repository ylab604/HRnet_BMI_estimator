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


