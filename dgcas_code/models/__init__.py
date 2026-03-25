# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2022- 09- 22
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions:
Input:
Output:
Note:
Link:
*************************************Import***********************************"""
import torch
import torch.nn.functional as F
from torch import nn
from .model import predictor
from .model import build_model
from .backbone_bases import build_backbone
from .model import CFDG
import models.backbone_bases as bkbn_bases
from .asmlp import ASMLP
from .model import build_model


def build_models(args):
    return build_model(args)


