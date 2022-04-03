# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
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


def reNormalize(img,mean,std):
    img = img.detach().cpu().numpy().transpose(1,2,0)
    img = img*std +mean
    img = img.clip(0,1)
    return img






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
    # model.load_state_dict(state_dict)
    # model.eval() , strict=False
    ###############################
    state_dict = torch.load(args.model_file, map_location=torch.device("cpu"))
    #keys = list(check_ge['netG_state_dict'].keys())
    #for i in range(len(keys)):
    #    temp = keys[i].replace('module.', '')
    #    check_ge['netG_state_dict'][temp] = check_ge['netG_state_dict'].pop(keys[i])
    ###############################


    #if "state_dict" in state_dict.keys():
    #    state_dict = state_dict["state_dict"]
    #    model.load_state_dict(state_dict.module.state_dict())
    #else:
    #    model.module.load_state_dict(state_dict)


    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict.state_dict())
    else:
        model.load_state_dict(state_dict)
    ############

    img = Image.open(args.path)
    img = img.resize((512, 512))
    img_np = np.asarray(img)
    img_t = torch.from_numpy(img_np)
    img_t = img_t.view(1, 3, 512, 512)
    img_t = img_t.float()
    output = model(img_t)
    output = output.squeeze()
    plt.imsave('4_2.jpeg',reNormalize(output,mean,std))
    # print(output.size())
    #tf = transforms.ToPILImage()
    #output = tf(output)
    #output = np.array(output)
    # print(output)
    # output.show()
    # output = output.detach().cpu().numpy()
    # print(output.shape)
    #plt.imshow('output.jpeg',output)
    #plt.imsave("output.jpeg", np.transpose(output[0][0], (0, 1)))

    # logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, "test")

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    # config.defrost()
    # config.MODEL.INIT_WEIGHTS = False
    # config.freeze()
    # model = models.get_face_alignment_net(config)

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    # state_dict = torch.load(args.model_file)
    # if "state_dict" in state_dict.keys():
    #    state_dict = state_dict["state_dict"]
    #    model.load_state_dict(state_dict)
    # else:
    #    model.module.load_state_dict(state_dict)

    # dataset_type = get_dataset(config)

    # test_loader = DataLoader(
    #    dataset=dataset_type(config, is_train=False),
    #    batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
    #    shuffle=False,
    #    num_workers=config.WORKERS,
    #    pin_memory=config.PIN_MEMORY,
    # )

    # nme, predictions = function.inference(config, test_loader, model)

    # torch.save(predictions, os.path.join(final_output_dir, "predictions.pth"))


if __name__ == "__main__":
    main()
