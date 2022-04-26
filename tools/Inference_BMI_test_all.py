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
import random

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
import pandas as pd
from tqdm import tqdm
from train_dn_cls_2 import new_model_cls


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch depth prediction evaluation script"
    )
    parser.add_argument(
        "--path_all", type=str, default="data", metavar="D", help="image file path"
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
    #
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

    # #============================load model_cls====================

    config_cls.defrost()
    config_cls.MODEL.INIT_WEIGHTS = False
    config_cls.freeze()
    #model_dn = models.Hrnet2DNnet(config)
    model_cls = models.get_cls_net(config_cls,model_dn)
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

    # tensor 2 image
    def reNormalize(img, mean, std):
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = img.clip(0, 1)
        return img

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #============================================================================
    image_path = args.path_all
    all_file_names = os.listdir(image_path)

    # random.shuffle(all_file_names)


    minmaxscaler = load(open('dataset/BMI/minmaxscaler_bmi.pkl', 'rb'))
    predict_list = []
    GT_list=[]
    image_file_name = []
    for i in tqdm(range(len(all_file_names))) :
        one_image=os.path.join(image_path, all_file_names[i])
        one_name = all_file_names[i]
        input_img = Image.open(one_image)
        input_transforms = transforms.Compose(image_transforms)
        input_tensor = input_transforms(input_img)
        input_tensor = input_tensor.view(1, 3, 512, 512)
        input_tensor=input_tensor.cuda()

        #result_dn = model_dn(input_tensor)
        #result_dn=result_dn.cuda()
        #output = result_dn.squeeze()
        #output_normal = output[0:3]
        #output_depth = output[3:6]

    # plt.imsave("test_full_normal_.jpeg", reNormalize(output_normal, mean, std))
    # plt.imsave("test_full_depth_.jpeg", reNormalize(output_depth, mean, std))

        #cls_input = torch.cat([input_tensor, result_dn], dim = 1)
        predict_bmi_scaled = model_cls(input_tensor)
        predict_bmi_scaled_=predict_bmi_scaled.item()
        predict_bmi_scaled_ = np.array(predict_bmi_scaled_)
        predict_bmi_scaled_=predict_bmi_scaled_.reshape(1,1)

        #ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", one_name)
        #GT_BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2

        try:
            name_list = one_name[i].split('_')
            height = name_list[-2]
            weight = name_list[-1][:-8]
            GT_BMI = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
            GT_list.append(float(GT_BMI))
        except:
            one_name[i] = one_name[i].replace(']', '_')
            name_list = one_name[i].split('_')
            height = name_list[-3]
            weight = name_list[-2]
            GT_BMI = (int(weight) / 100000) / ((int(height) / 100000) ** 2)
            GT_list.append(float(GT_BMI))





        predict_BMI = minmaxscaler.inverse_transform(predict_bmi_scaled_)
        predict_list.append(predict_BMI[0][0])
        #GT_list.append(GT_BMI)
        image_file_name.append(one_name)

    predict_df = pd.DataFrame()
    predict_df['image_file'] = image_file_name
    predict_df['GT'] = GT_list
    predict_df['Predict'] = predict_list

    predict_df.loc[predict_df['GT'] < 18.5, 'category'] = 'underweight'
    predict_df.loc[(predict_df['GT'] >= 18.5) & (predict_df['GT'] < 25), 'category'] = 'normal'
    predict_df.loc[(predict_df['GT'] >= 25) & (predict_df['GT'] < 30), 'category'] = 'overweight'
    predict_df.loc[predict_df['GT'] >= 30, 'category'] = 'obesity'

    predict_df['abs_l1'] = abs(predict_df['GT'] - predict_df['Predict'])
    predict_df.to_excel('/home/ylab3/HRnet_BMI_estimator/9_channel/HRnet_48_aug_60_Predict.xlsx')


if __name__ == "__main__":
    main()


