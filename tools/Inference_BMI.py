# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by ylab
# Modified by Ming >_<
# ------------------------------------------------------------------------------
import os
import pprint
import argparse
import matplotlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lib.models as models
from lib.config import config, update_config, config_cls, update_config_cls
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import numpy as np
import re
from pickle import load

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch depth prediction evaluation script"
    )
    parser.add_argument(
        "--path", type=str, default="data", metavar="D", help="image file path"
    )
    parser.add_argument(
        "--cfg", help="experiment configuration filename", required=True, type=str
    )
    parser.add_argument(
        "--model_file", help="model parameters", required=True, type=str
    )
    parser.add_argument(
        "--cfg_cls", help="experiment configuration filename", required=True, type=str
    )
    parser.add_argument(
        "--model_file_cls", help="model parameters", required=True, type=str
    )

    args = parser.parse_args()
    update_config(config, args)
    update_config_cls(config_cls, args)

    return args

def main():
    #====================load model_dn=================================
    args = parse_args()

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model_dn = models.Hrnet2DNnet(config)
    gpus = list(config.GPUS)
    model_dn = nn.DataParallel(model_dn, device_ids=gpus).cuda()

    state_dict = torch.load(args.model_file)

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
        model_dn.load_state_dict(state_dict.state_dict())
    else:
        model_dn.load_state_dict(state_dict)

    #============================load model_cls====================

    config_cls.defrost()
    config_cls.MODEL.INIT_WEIGHTS = False
    config_cls.freeze()

    model_cls = models.get_cls_net(config_cls)
    gpus = list(config_cls.GPUS)
    model_cls = nn.DataParallel(model_cls, device_ids=gpus).cuda()

    state_dict_cls = torch.load(args.model_file_cls)

    if "state_dict" in state_dict_cls.keys():
        state_dict_cls = state_dict_cls["state_dict"]
        model_cls.load_state_dict(state_dict_cls.state_dict())
    else:
        model_cls.load_state_dict(state_dict_cls)

    #========================image transforms & loader ==========================
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
        transforms.Normalize(mean,std)
    ]

#============================================================================
    image_path = args.path
    image_name = image_path.split('/')[-1]
    input_img = Image.open(args.path)
    input_transforms = transforms.Compose(image_transforms)
    input_tensor = input_transforms(input_img)
    input_tensor = input_tensor.view(1, 3, 512, 512)

    result_dn = model_dn(input_tensor)
    cls_input = torch.cat([input_img, result_dn], dim = 1)
    predict_bmi_scaled = model_cls(cls_input)

    ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", image_name)
    GT_BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2

    minmaxscaler = load(open('dataset/BMI/minmaxscaler_bmi.pkl', 'rb'))
    predict_BMI = minmaxscaler.inverse_transform(predict_bmi_scaled)

    print('predicted BMI : '+str(predict_BMI))
    print('GT BMI : ' + str(GT_BMI))



