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
        
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        
        # default = 가우시안
        self.label_type = cfg.MODEL.TARGET_TYPE
       
        self.render_transforms = transforms.Compose(render_transforms)
        self.normal_transforms = transforms.Compose(normal_transforms)
        self.depth_transforms = transforms.Compose(depth_transforms)
        # print(type(self.render_transforms))
       
        ############################################################################
        data_root_path = self.data_root+"/*"
        data = glob.glob(data_root_path)
        #get renderdata
        render_file_list=[] 
        for n in range(len(data)):
            j = "{:04d}".format(n)
            i = j + "_OBJ"
            path = self.data_root + i + "/RENDER/" + j + "/*"
            # print(path)
            render_file_list_1=glob.glob(path)
            render_file_list+=render_file_list_1
            #print(1)
            #print(render_file_list)
            render_file_list = sorted(render_file_list)
            #

        #print(render_file_list)
        self.render_files=render_file_list
        #get render_normal_data
        render_normal_file_list=[] 
        for n in range(len(data)):
            j = "{:04d}".format(n)
            i = j + "_OBJ"
            path = self.data_root + i + "/RENDER_NORMAL/" + j + "/*"
            # print(path)
            
            render_normal_file_list_1=glob.glob(path)
            render_normal_file_list+=render_normal_file_list_1 
            #print(file_list)
            render_normal_file_list = sorted(render_normal_file_list)
        self.normal_files=render_normal_file_list
        #get render_depth_data
        render_depth_file_list=[]
        for n in range(len(data)):
            j = "{:04d}".format(n)
            i = j + "_OBJ"
            path = self.data_root + i + "/RENDER_DEPTH/" + j + "/*"
            # print(path)
            #render_depth_file_list.append(glob.glob(path))
            render_depth_file_list_1=glob.glob(path)
            render_depth_file_list+=render_depth_file_list_1 
            #print(file_list)
            render_depth_file_list = sorted(render_depth_file_list)
        self.depth_files=render_depth_file_list
        
        
        ############################################################################

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
     

        return {"A": img, "B": target, "C": depth_img}


if __name__ == "__main__":

    pass

