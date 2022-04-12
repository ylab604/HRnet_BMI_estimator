# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng (tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import Hrnet2DNnet, HighResolutionNet
from .cls_hrnet import get_cls_net, HighResolutionNet_cls

__all__ = ["HighResolutionNet", "Hrnet2DNnet", "get_cls_net","HighResolutionNet_cls"]
