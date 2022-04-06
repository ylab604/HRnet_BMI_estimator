# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# modified by ylab
# ------------------------------------------------------------------------------

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
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
from torch.autograd import Variable
import matplotlib.pyplot as plt

def parse_args():

    parser = argparse.ArgumentParser(description="Train Face Alignment")

    parser.add_argument(
        "--cfg", help="experiment configuration filename", required=True, type=str
    )

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = utils.create_logger(
        config, args.cfg, "train"
    )

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.Hrnet2DNnet(config)
    #cuda = True if torch.cuda.is_available() else Fals
    #fig = plt.figure(figsize=(10, 10))
    #rows = 6 
    #cols = 1
    #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #dataset_name='thuman'
    def reNormalize(img,mean,std):
        img = img.numpy().transpose(1,2,0)
        img = img*std +mean
        img = img.clip(0,1)
        return img

    #if you want sampling on training release these code
    #os.makedirs("images/%s/val" % dataset_name, exist_ok=True)
    #def sample_images(epoch, loader, mode):
        #imgs = next(iter(loader))
        #gray = Variable(imgs["A"].type(Tensor))
        #output = model(gray)    
        #output_img = torchvision.utils.make_grid(output.data, nrow=1)
        #rows = 6
        #cols = 1
        #ax1 = fig.add_subplot(rows, cols, 1)
        #ax1.imshow(reNormalize(output_img.cpu(), mean, std))
        #ax1.set_title('output')  
        #fig.savefig("images/%s/%s/epoch_%s.png" % (dataset_name, mode, epoch), pad_inches=0)


    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # loss
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, "latest.pth")
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint["epoch"]
            best_nme = checkpoint["best_nme"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    dataset_type = get_dataset(config)
    print("!!!!")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #render, normal, depth 이미지 resize, tensor, normalize
    render_transforms = [
	transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    normal_transforms = [
	transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    depth_transforms = [
	transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    train_loader = DataLoader(
        dataset=dataset_type(
            config,
            is_train=True,
            render_transforms=render_transforms,
            normal_transforms=normal_transforms,
            depth_transforms=depth_transforms,
        ),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    print("??")
    # need to update val_loader
    # for X in train_loader:
    #    print(X)
    #val_loader = DataLoader(
    #    dataset=dataset_type(config, is_train=False,
    #    render_transforms=render_transforms,
    #        normal_transforms=normal_transforms,
    #        depth_transforms=depth_transforms,
    #        ),
    #    batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
    #    shuffle=False,
    #    num_workers=config.WORKERS,
    #    pin_memory=config.PIN_MEMORY,
    #)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        
        function.train(
            config, train_loader, model, criterion, optimizer, epoch, writer_dict
        )

        # evaluate
        #nme, predictions = function.validate(
        #    config, val_loader, model, criterion, epoch, writer_dict
        #)

        #is_best = nme < best_nme
        #best_nme = min(nme, best_nme)
        #"best_nme": best_nme, predictions,is_best,
        logger.info("=> saving checkpoint to {}".format(final_output_dir))
        #print("best:", is_best)
        utils.save_checkpoint(
            {
                "state_dict": model,
                "epoch": epoch + 1,
                
                "optimizer": optimizer.state_dict(),
            },
            
            final_output_dir,
            "checkpoint_{}.pth".format(epoch),
        )
        #sample_images(epoch, train_loader, 'val')

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth")
    logger.info("saving final model state to {}".format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict["writer"].close()
    print(1111)


if __name__ == "__main__":
    main()
