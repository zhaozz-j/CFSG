import torch

import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from models.backbone_bases import  build_backbone
import pandas as pd
# from backbone import build_backbone
import models.backbone_bases as bkbn_bases
import adversarial1 as ad
from models.asmlp import ASMLP
from util.misc import compu_featpart, cate_num_all_dataset, num_coarse_cate_dataset

from models.criterion import SetCriterion
import functools
from torch.nn import init
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_

class CFDG(nn.Module):
    def __init__(self, backbone_f, backbone_c, my_discriminator, my_grl, args):

        super(CFDG, self).__init__()
        self.cate_num_all = cate_num_all_dataset[args.dataset]
        self.num_coarse_cate = args.granu_num - 1
        self.feat_dim = args.feat_dim
        self.feat_num = args.feat_num
        self.feat_len = 2048
        self.device = args.device
        self.batch_size = args.batch_size["train"] #32
        self.max_iter = torch.tensor(args.max_iteration, dtype=torch.float64).requires_grad_(False)
        self.cfdg = args.bn_cfdg
        self.da = args.da
        self.cnfd_type = args.cnfd_type
        self.bn_input_p = args.bn_input_p
        self.b_bkb_poor = args.b_bkb_poor
        self.b_pan = args.b_pan
        self.str_avgpool = args.str_avgpool
        self.conv_type = args.conv_type
        self.b_cond = args.b_cond
        self.b_pass_transition = args.b_pass_transition

        self.model_type = args.model_type
        self.b_bkb_c = args.b_bkb_c
        self.backbone_f = backbone_f
        self.centriods = {}
        self.ones = {}
        self.G = args.granu_num #粒度数量
        feat_ratio, feat_part = compu_featpart(args)
        self.feat_ratio = feat_ratio
        self.feat_part = feat_part
        self.cdistance = args.cdistance #距离度量
        self.pdistance = args.pdistance #距离度量
        self.ndistance = args.ndistance #距离度量
        self.epsilon = 1e-4
        self.Ele_g = list(range(self.G))

        self.use_feature_extractor = args.use_feature_extractor
        if args.b_bkb_c is True:  # 双路
            self.backbone_c = backbone_c

        if 'rn' in args.backbone:
            self.feat_len = 2048
        elif 'vit' in args.backbone:
            self.feat_len = self.backbone_f.model_vit.num_channels
        elif 'mlp' in args.backbone:
            self.feat_len = self.backbone_f.num_channels
        elif 'mamba' in args.backbone:
            self.feat_len = 2048

        feat_ratio, feat_part = compu_featpart(args)
        self.feat_ratio = feat_ratio
        self.feat_part = feat_part

        self.feat_caus = self.feat_part[-1]
        self.nfeat_caus = self.feat_dim
        self.nfeat_cnfd = 0
        btnk_Layer = BottleNeck_Layer_conv1
        self.pred_len = self.nfeat_caus

        self.btnk_layer = {}
        self.dis_lin = {}
        self.clsf_layer = {}
        self.clsf_cnfd_layer = {}
        self.num_g = self.num_coarse_cate + 1
        self.ele_g = list(range(self.num_g))
        if self.b_pass_transition is False:
            if self.b_pan is False:
                if self.cfdg is True:
                    self.btnk_layer[self.ele_g[-1]] = btnk_Layer(self.feat_len, self.feat_dim)
                    if self.num_coarse_cate != 0:
                        for i_e_g in self.ele_g[::-1][1:]:
                            self.btnk_layer[i_e_g] = btnk_Layer(self.feat_len, self.feat_dim)
                        for i_e_g in self.ele_g[::-1]:
                            self.clsf_layer[i_e_g] = predictor(self.pred_len, self.cate_num_all[i_e_g], feat_part)
            if 'conv' in self.model_type and self.str_avgpool == 'avg':
                for i_e_g in self.ele_g[::-1]:
                    self.dis_lin[i_e_g] = nn.AdaptiveAvgPool2d((1, 1))
            self.dis_lin = self.dict2moduledict(self.dis_lin)
            self.b_da_split = False
            self.btnk_layer = self.dict2moduledict(self.btnk_layer)
            self.clsf_layer = self.dict2moduledict(self.clsf_layer)
            if self.nfeat_cnfd != 0:
                self.clsf_cnfd_layer = self.dict2moduledict(self.clsf_cnfd_layer)
    def dict2moduledict(self, module2be_trans):
        if isinstance(module2be_trans, dict):
            pass
        else:
            module2be_init = {module2be_trans}
        newmodu_dict = {}
        for i_name, i_module in module2be_trans.items():
            i_name = str(i_name)
            newmodu_dict[i_name] = i_module
        return nn.ModuleDict(newmodu_dict)
    def initialize_weights(self, module2be_init):
        if isinstance(module2be_init, dict):
            pass
        else:
            module2be_init = {module2be_init}
        for i_module in module2be_init.values():
            for m in i_module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
    def save_bkb_gradients(self, xs_gradients):
        self.xs_gradients = xs_gradients
    def get_bkb_gradients(self):
        return self.xs_gradients
    def get_all_centriods(self):
        return self.centriods
class predictor(nn.Module):
    def __init__(self, feature_len, cate_num, feat_part):
        super(predictor, self).__init__()
        #使用 nn.Linear 创建一个全连接层（线性层）。这个层的输入特征数为 feature_len，输出特征数为 cate_num。它会将输入的特征映射到 cate_num 个类别的预测值。
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

        # 学习未归一化的三部分融合权重 logits
        # self.logits_weight = nn.Parameter(torch.tensor([2.0, 1.5, 1.0]))

        self.feat_part = feat_part
    def forward(self, features, iter_time):
        # if iter_time <10000:
        #     logits_all = self.classifier(features)
        # else:
        #     # activations = self.classifier(features)
        #     weight = self.classifier.weight  # [num_classes, C]
        #     weight_common = weight[:, :self.feat_part[0]]
        #     weight_privacy = weight[:, self.feat_part[0]:self.feat_part[1]]
        #     weight_noise = weight[:, self.feat_part[1]:]
        #
        #     feat_common = features[:, :self.feat_part[0]]
        #     feat_privacy = features[:, self.feat_part[0]:self.feat_part[1]]
        #     feat_noise = features[:, self.feat_part[1]:]
        #
        #     logits_common = torch.matmul(feat_common, weight_common.T)  # [B, num_classes]
        #     logits_privacy = torch.matmul(feat_privacy, weight_privacy.T)
        #     logits_noise = torch.matmul(feat_noise, weight_noise.T)
        #
        #     # 使用 softmax 归一化，确保三个权重和为 1
        #     weights = F.softmax(self.logits_weight, dim=0)  # [3]，值 ∈ (0,1) 且和为1
        #
        #     # 加权融合
        #     logits_all = (logits_common * weights[0] +
        #                 logits_privacy * weights[1] +
        #                 logits_noise * weights[2]
        #                  ) + self.classifier.bias

        # weight = self.classifier.weight  # [num_classes, C]
        # weight_common = weight[:, :self.feat_part[0]]
        # weight_privacy = weight[:, self.feat_part[0]:self.feat_part[1]]
        # weight_noise = weight[:, self.feat_part[1]:]
        #
        # feat_common = features[:, :self.feat_part[0]]
        # feat_privacy = features[:, self.feat_part[0]:self.feat_part[1]]
        # feat_noise = features[:, self.feat_part[1]:]
        #
        # logits_common = torch.matmul(feat_common, weight_common.T)  # [B, num_classes]
        # logits_privacy = torch.matmul(feat_privacy, weight_privacy.T)
        # logits_noise = torch.matmul(feat_noise, weight_noise.T)
        # logits_all = logits_common * 0.45 + logits_privacy * 0.35 + logits_noise * 0.2 + self.classifier.bias

        logits_all = self.classifier(features)
        return logits_all

class BottleNeck_Layer_linear(nn.Module):
    def __init__(self, in_dim, out_dim):

        super(BottleNeck_Layer_linear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.bottleneck = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.initialize_weights()

    def initialize_weights(self):
        # nn.init.xavier_uniform_(self.conv.weight, gain=1)
        # nn.init.constant_(self.conv.bias, 0)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.relu(x)
        x = self.drop_out(x)
        return x


class BottleNeck_Layer_conv1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BottleNeck_Layer_conv1, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.btnk = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.btnk.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        identity = x
        out = self.btnk(x)
        return out
def build_model(args):
    device = args.device
    feature_len = 2048
    if args.b_bkb_c is True:
        backbone_c = build_backbone(args)
    else:
        backbone_c = None
    backbone_f = build_backbone(args)
    model = CFDG(backbone_f, backbone_c, args)
    model.train(True)
    criterion = SetCriterion(args)
    return model, criterion

