from PIL import Image
import os
import pprint
import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
import torchvision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lib.models as models
from lib.config import config, update_config, config_cls, update_config_cls
from lib.datasets import get_dataset
from lib.core import function_cls
from lib.utils import utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pickle import dump, load

class new_model_dn(nn.Module) :
    def __init__(self,model_dn):
        super().__init__()
        self.model_dn = model_dn
        self.head_2 = nn.Sequential(
        nn.BatchNorm2d(6, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(6, 3, kernel_size=(1, 1), stride=(1, 1)),
        )
    def forward(self,x):
        x = self.model_dn(x)
        x = self.head_2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Train Face Alignment")

    parser.add_argument(
        "--cfg", help="experiment configuration filename", required=True, type=str
    )
    # parser.add_argument(
    #     "--cfg_cls", help="experiment configuration filename", required=True, type=str
    # )
    parser.add_argument(
        "--model_file", help="model parameters", required=True, type=str
    )

    args = parser.parse_args()
    update_config(config, args)
    # update_config_cls(config_cls, args)
    return args

def main():
    args = parse_args()
    # state_dict = torch.load(args.model_file)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = list(config.GPUS)

    model_dn = models.Hrnet2DNnet(config)
    model_dn = nn.DataParallel(model_dn, device_ids=gpus).cuda()

    ################ model_dn에 checkpoint_70에 있는 가중치 업로드 ###############
    state_dict = torch.load(args.model_file)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
        model_dn.load_state_dict(state_dict.state_dict())
    else:
        model_dn.load_state_dict(state_dict)
    ##########################################################################

    model_dn_fine = new_model_dn(model_dn)


    for param in model_dn_fine.parameters() :
        param.requires_grad = False
    for param in model_dn_fine.head_2.parameters() :
        param.requires_grad = True

    for param in model_dn_fine.parameters() :
        print(param.requires_grad)

    for param in model_dn_fine.model_dn.parameters():
        print(param)

    # for param in model_dn.parameters():
    #     print(param)
    #
    # print('===================================================')
    # for param in model_dn_fine.head_2.parameters() :
    #     print(param)

if __name__ == "__main__":
    main()