# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------
#
# from .aflw import AFLW
# from .cofw import COFW
# from .face300w import Face300W
# from .wflw import WFLW
from .thuman import Thuman
from .BMI import BMI

__all__ = ["get_dataset", "Thuman", "BMI"]


def get_dataset(config):

    # if config.DATASET.DATASET == "AFLW":
    #     return AFLW
    # elif config.DATASET.DATASET == "COFW":
    #     return COFW
    # elif config.DATASET.DATASET == "300W":
    #     return Face300W
    # elif config.DATASET.DATASET == "WFLW":
    #     return WFLW
    if config.DATASET.DATASET == "Thuman":
        return Thuman
    if config.DATASET.DATASET == 'BMI' :
        return BMI
    else:
        raise NotImplemented()

