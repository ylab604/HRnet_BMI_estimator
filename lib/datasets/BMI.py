# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Ming! >_<
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
import re
from sklearn.preprocessing import MinMaxScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BMI(data.Dataset):
    """
    BMI
    """

    def __init__(
            self,
            cfg,
            minmaxscaler,
            is_train=True,
            image_transforms=None,
    ):

        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.minmaxscaler = minmaxscaler

        # default = 가우시안
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.image_transforms = transforms.Compose(image_transforms)
        #====================RGB DataLoader==================================
        data_root_path = os.path.join(self.data_root, cfg.DATASET.TRAIN_SET)
        self.data = os.listdir(data_root_path)
        self.data.sort()

        rgb_data_list_root = []
        for i in range(len(self.data)) :
            rgb_data_list_root.append(os.path.join(data_root_path, self.data[i]))

        # print(render_file_list)
        self.rgb_file = rgb_data_list_root
        #====================================================================


    def __len__(self):
        return len(self.rgb_file)

    def __getitem__(self, index):
        # we need target and image
        rgb_file_image = Image.open(
            self.rgb_file[index % len(self.rgb_file)]
        ).convert("RGB")
        transformed_rgb = self.image_transforms(rgb_file_image)

        img = transformed_rgb

        BMI_list = []
        for i in range(len(self.data)) :
            ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", self.data[i])
            BMI_ = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
            BMI_list.append(float(BMI_))
        BMI_list=np.array(BMI_list)
        BMI_list=BMI_list.reshape(len(BMI_list),-1)
        scaled_BMI=self.minmaxscaler.transform(BMI_list)
        scaled_BMI = np.array(scaled_BMI,dtype=np.float)
        scaled_BMI_float = scaled_BMI.astype(np.float)

        target_bmi = torch.tensor(scaled_BMI_float[index % len(scaled_BMI_float)])

        return {"Image": img, 'BMI': target_bmi}


if __name__ == "__main__":
    pass
