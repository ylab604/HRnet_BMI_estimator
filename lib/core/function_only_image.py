# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by ylab
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

from .evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config_cls, train_loader, model_cls, critertion, optimizer_cls, epoch, writer_dict, optimizer_dn=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_cls = AverageMeter()
    #losses_dn = AverageMeter()

    model_cls.train()
    nme_count = 1
    nme_batch_sum = 1

    end = time.time()

    for i, batch in enumerate(train_loader):
        # measure data time
        inp_image = batch["Image"]
        inp_image=inp_image.cuda(non_blocking=True)
        # print(inp.shape)
        target = batch['BMI']
        data_time.update(time.time() - end)

        # compute the output
        target = target.cuda(non_blocking=True)

        output=model_cls(inp_image)
        # GT_bmi, predictor_bmi => MSELoss
        output = output.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        loss_cls = critertion(output, target)

        # optimize
        optimizer_cls.zero_grad()
        loss_cls.backward()
        optimizer_cls.step()

        losses_cls.update(loss_cls.item(), inp_image.size(0))

        batch_time.update(time.time() - end)
        if i % config_cls.PRINT_FREQ == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed {speed:.1f} samples/s\t"
                "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    speed=inp_image.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses_cls,
                )
            )
            logger.info(msg)

            # writer_dict = {
            #     "writer": SummaryWriter(log_dir=bmi_log_dir),
            #     "train_global_steps": 0,
            #     "valid_global_steps": 0,
            # # }
            if writer_dict:
                writer = writer_dict["writer"]
                global_steps = writer_dict["train_global_steps"]
                writer.add_scalar("train_loss", losses_cls.val, global_steps)
                writer_dict["train_global_steps"] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = "Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}".format(
        epoch, batch_time.avg, losses_cls.avg, nme
    )
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 1
    nme_batch_sum = 1
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inp = batch["A"]
            target = torch.cat([batch["B"],batch["C"]],dim=1)
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            # preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            # nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            # failure_008 = (nme_temp > 0.08).sum()
            # failure_010 = (nme_temp > 0.10).sum()
            # count_failure_008 += failure_008
            # count_failure_010 += failure_010

            # nme_batch_sum += np.sum(nme_temp)
            # nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #    predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = (
        "Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} "
        "[010]:{:.4f}".format(
            epoch, batch_time.avg, losses.avg, nme, failure_008_rate, failure_010_rate
        )
    )
    logger.info(msg)

    if writer_dict:
        writer = writer_dict["writer"]
        global_steps = writer_dict["valid_global_steps"]
        writer.add_scalar("valid_loss", losses.avg, global_steps)
        writer.add_scalar("valid_nme", nme, global_steps)
        writer_dict["valid_global_steps"] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 1
    nme_batch_sum = 1
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inp = batch["A"]
            target = batch["B"]
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            # preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            # nme_temp = compute_nme(preds, meta)

            # failure_008 = (nme_temp > 0.08).sum()
            # failure_010 = (nme_temp > 0.10).sum()
            # count_failure_008 += failure_008
            # count_failure_010 += failure_010

            # nme_batch_sum += np.sum(nme_temp)
            # nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #   predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = (
        "Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} "
        "[010]:{:.4f}".format(
            batch_time.avg, losses.avg, nme, failure_008_rate, failure_010_rate
        )
    )
    logger.info(msg)

    return nme, predictions