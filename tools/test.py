# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by ylab
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
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import numpy as np


#parser => --path 이미지경로/ --cfg yaml파일 경로 / --model-file pth파일 경로
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
        "--model-file", help="model parameters", required=True, type=str
    )
    args = parser.parse_args()
    update_config(config, args)
    return args

#tensor 2 image
def reNormalize(img, mean, std):
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = img.clip(0, 1)
    return img

#model inference 메인
def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    args = parse_args()
    # state_dict = torch.load(args.model_file)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.Hrnet2DNnet(config)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    ###############################
    state_dict = torch.load(args.model_file, map_location=torch.device("cpu"))
    ###############################
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict.state_dict())
    else:
        model.load_state_dict(state_dict)
    ############
    input_img = Image.open(args.path)
    input_transforms = [
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=mean, std=std),
    ]
    input_transforms = transforms.Compose(input_transforms)
    input_tensor = input_transforms(input_img)
    input_tensor = input_tensor.view(1, 3, 512, 512)
    output = model(input_tensor)
    output = output.squeeze()
    # print(output[0:3].shape)
    # [0:3] normal map [3:6] depth map
    output_normal = output[0:3]
    output_depth = output[3:6]
    print(output_depth.shape)
    plt.imsave("4_05_3_normal.jpeg", reNormalize(output_normal, mean, std))
    plt.imsave("4_05_4_depth.jpeg", reNormalize(output_depth, mean, std))


if __name__ == "__main__":
    main()
