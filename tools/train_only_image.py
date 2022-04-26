# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# modified by MING >_<
# ------------------------------------------------------------------------------
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
from lib.core import function_cls, function_only_image
from lib.utils import utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pickle import dump, load

def parse_args():
    parser = argparse.ArgumentParser(description="Train Face Alignment")

    parser.add_argument(
        "--cfg_cls", help="experiment configuration filename", required=True, type=str
    )


    args = parser.parse_args()
    update_config_cls(config_cls, args)
    return args

def main():

    #=====================Inference========================================


    args = parse_args()

    logger, final_output_dir, bmi_log_dir = utils.create_logger(
        config_cls, args.cfg_cls, "train"
    )

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config_cls))

    cudnn.benchmark = config_cls.CUDNN.BENCHMARK
    cudnn.determinstic = config_cls.CUDNN.DETERMINISTIC
    cudnn.enabled = config_cls.CUDNN.ENABLED

    writer_dict = {
        "writer": SummaryWriter(log_dir=bmi_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    model_cls = models.get_cls_net(config_cls)

    gpus = list(config_cls.GPUS)
    model_cls = nn.DataParallel(model_cls, device_ids=gpus).cuda()

    criterion = torch.nn.MSELoss(size_average=True).cuda()
############################################!!!!!!!!!!!!!!!!!!!!!
    optimizer = utils.get_optimizer(config_cls, model_cls)
    last_epoch = config_cls.TRAIN.BEGIN_EPOCH

    if config_cls.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, "latest.pth")
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint["epoch"]
            best_nme = checkpoint["best_nme"]
            model_cls.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
        else:
            print("=> no checkpoint found")

    if isinstance(config_cls.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config_cls.TRAIN.LR_STEP, config_cls.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config_cls.TRAIN.LR_STEP, config_cls.TRAIN.LR_FACTOR, last_epoch - 1
        )
    print(config_cls.TRAIN.LR_FACTOR)
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

    dataset_type = get_dataset(config_cls)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transforms = [
        Resize(512),
        transforms.Pad(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]

    minmaxscaler = load(open('dataset/BMI/minmaxscaler_bmi.pkl','rb'))

    train_loader = DataLoader(
        dataset=dataset_type(
            config_cls,
            minmaxscaler,
            is_train=True,
            image_transforms=image_transforms,
        ),
        batch_size=config_cls.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config_cls.TRAIN.SHUFFLE,
        num_workers=config_cls.WORKERS,
        pin_memory=config_cls.PIN_MEMORY,
    )

    bindo = 10
    for epoch in range(last_epoch,500) :

        lr_scheduler.step()

        function_only_image.train(config_cls, train_loader, model_cls, criterion ,optimizer, epoch,writer_dict)

        logger.info("=> saving checkpoint to {}".format(final_output_dir))

        if epoch % bindo == 0 :
            utils.save_checkpoint(
                {
                    "state_dict": model_cls,
                    "epoch": epoch + 1,

                    "optimizer": optimizer.state_dict(),
                },

                final_output_dir,
                "checkpoint_{}.pth".format(epoch),
            )


    final_model_state_file = os.path.join(final_output_dir, "final_state.pth")
    logger.info("saving final model state to {}".format(final_model_state_file))
    torch.save(model_cls.module.state_dict(), final_model_state_file)
    writer_dict["writer"].close()



if __name__ == "__main__":
    main()
