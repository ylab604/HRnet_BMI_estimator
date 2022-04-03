# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by ylab604
# ------------------------------------------------------------------------------

import os
import random
import glob
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import torchvision.transforms as transforms
from lib.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

# 아 얘가 문제네

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --------------------------------------------------------------------
# 이미지부르고 노말 부르고 뎁스 부르고
# 텐서화 하고
# get item에 이미지 노말 뎁스 리턴
# --------------------------------------------------------------------


class Thuman(data.Dataset):
    """thuman
    """

    def __init__(
        self,
        cfg,
        is_train=True,
        render_transforms=None,
        normal_transforms=None,
        depth_transforms=None,
    ):
        # specify annotation file for dataset
        # if is_train:
        #    self.csv_file = cfg.DATASET.TRAINSET
        # else:
        #    self.csv_file = cfg.DATASET.TESTSET

        # self.is_train = is_train
        # self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        # self.scale_factor = cfg.DATASET.SCALE_FACTOR
        # self.rot_factor = cfg.DATASET.ROT_FACTOR
        # default = 가우시안
        self.label_type = cfg.MODEL.TARGET_TYPE
        # self.flip = cfg.DATASET.FLIP
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # load annotations
        # self.landmarks_frame = pd.read_csv(self.csv_file)
        # print(type(self.render_transforms))
        self.render_transforms = transforms.Compose(render_transforms)
        self.normal_transforms = transforms.Compose(normal_transforms)
        self.depth_transforms = transforms.Compose(depth_transforms)
        # print(type(self.render_transforms))
        self.render_files = sorted(
            glob.glob(os.path.join(self.data_root, "RENDER\\0000") + "/*.*")
        )
        self.normal_files = sorted(
            glob.glob(os.path.join(self.data_root, "RENDER_NORMAL\\0000") + "/*.*")
        )

        self.depth_files = sorted(
            glob.glob(os.path.join(self.data_root, "RENDER_DEPTH\\0000") + "/*.*")
        )
        self.render_files = sorted(
            glob.glob(os.path.join(self.data_root, "RENDER\\0000") + "/*.*")
        )
        self.normal_files = sorted(
            glob.glob(os.path.join(self.data_root, "RENDER_NORMAL\\0000") + "/*.*")
        )
        self.depth_files = sorted(
            glob.glob(os.path.join(self.data_root, "RENDER_DEPTH\\0000") + "/*.*")
        )

    def __len__(self):
        return len(self.render_files)

    def __getitem__(self, index):
        # we need target and image
        render_img = Image.open(
            self.render_files[index % len(self.render_files)]
        ).convert("RGB")
        # print(render_img)
        normal_img = Image.open(
            self.normal_files[index % len(self.normal_files)]
        ).convert("RGB")
        depth_img = Image.open(self.depth_files[index % len(self.depth_files)]).convert(
            "RGB"
        )
        # print(1)
        # print(type(self.render_transforms()))
        render_img = self.render_transforms(render_img)
        # print(self.render_files)
        normal_img = self.normal_transforms(normal_img)
        depth_img = self.depth_transforms(depth_img)
        img = render_img
        target = normal_img
        # image_path = os.path.join(self.data_root, self.landmarks_frame.iloc[idx, 0])
        # img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        # img = img.astype(np.float32)
        # img = (img / 255.0 - self.mean) / self.std
        # img = img.transpose([2, 0, 1])
        # target = torch.Tensor(target)
        # tpts = torch.Tensor(tpts)
        # center = torch.Tensor(center)

        return {"A": img, "B": target, "C": depth_img}


if __name__ == "__main__":

    pass

