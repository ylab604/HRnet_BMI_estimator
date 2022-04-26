import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def _get_image_size(img):
    if transforms.functional._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Resize(transforms.Resize):

    def __call__(self, img):
        h, w = _get_image_size(img)
        scale = max(w, h) / float(self.size)
        new_w, new_h = int(w / scale), int(h / scale)
        return transforms.functional.resize(img, (new_w, new_h), self.interpolation)


# dataset_type = get_dataset(config_cls)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_transforms = [
    Resize(512),
    transforms.Pad(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

data_root = 'dataset/BMI/Image_test_remove/0_F_18_157480_4535924.png.jpg'
input_size = 512
output_size = 512

rgb_file_image = Image.open(data_root).convert("RGB")

policy = transforms.AutoAugmentPolicy.IMAGENET
augmentor = transforms.AutoAugment(policy)

imgs = augmentor(rgb_file_image)

aug_array=[augmentor(rgb_file_image) for _ in range(4)]

BMI = []
row_title = [str(policy).split('.')[-1]]
plot(aug_array,with_orig=False,row_title=row_title)
imgs.show()

transformed_rgb = image_transforms(rgb_file_image)


imgs.show()

