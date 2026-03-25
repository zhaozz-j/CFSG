import os.path
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from models.vit import VisionTransformer
from models.asmlp import ASMLP
from util.misc import is_main_process, logger_finish
#from models.mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2, vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from models.models.vmamba import vmamba_tiny_s1l8, Backbone_VSSM
import torch.nn.init as init
class vit(nn.Module):
    def __init__(self, name='vit_small_patch16_384', img_size=(224, 224), patch_size=16,
                 return_intermediate_states=False, in_chans=3, num_classes=0, pretrain=True, pre_path=None):

        super(vit, self).__init__()
        self.model_vit = VisionTransformer(name=name, img_size=img_size, patch_size=patch_size,
                                           return_intermediate_states=return_intermediate_states,
                                           in_chans=3,
                                           num_classes=num_classes, pretrain=pretrain, pre_path=pre_path)
    def forward(self, x):  # [72,3,224,224]

        feat_sp, feat_avgpool = self.model_vit(x)

        return feat_sp, feat_avgpool  # torch.Size([72, 2048])

    def output_num(self):
        return self.__in_features


class vmamba_tiny(nn.Module):
    def __init__(self, pretrained=True):
        super(vmamba_tiny, self).__init__()
        self.model_vmamba = Backbone_VSSM(depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln2d",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224, )

        self.out_proj = nn.Conv2d(768, 2048, kernel_size=1)
        init.xavier_uniform_(self.out_proj.weight, gain=1)
        init.constant_(self.out_proj.bias, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model_vmamba(x)  # (B, C, H, W)

        x = self.out_proj(x)  # (B, 2048, H, W)

        x1 = self.avgpool(x)
        return x, x1

import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import timm
class Resnet50Fc(nn.Module):
    def __init__(self):

        super(Resnet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=is_main_process())
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # torch.Size([batchsize, 2048, 7, 7])
        x1 = self.avgpool(x)  # torch.Size([batchsize, 2048, 1, 1])
        x1 = x1.view(x1.size(0), -1)

        return x, x1  # torch.Size([72, 2048])

    def output_num(self):
        return self.__in_features


class Resnet101Fc(nn.Module):
    def __init__(self):
        super(Resnet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.avgpool(x)
        x1 = x1.view(x1.size(0), -1)
        return x, x1

    def output_num(self):
        return self.__in_features





def build_backbone(args):
    bkb_type = args.backbone #rn50
    if bkb_type == 'rn50':
        model_bck = Resnet50Fc()

    elif bkb_type == 'rn101':
        model_bck = Resnet101Fc()
    elif 'vmamba-tiny' in bkb_type:
        model_bck = vmamba_tiny()
    elif 'vit' in bkb_type:
        model_bck = vit(bkb_type, img_size=(224, 224), pretrain=True)
    elif 'asmlp' in bkb_type:
        if 'tiny' in bkb_type:
            as_depths = [2,2,6,2]
            as_drop_path_rate = 0.2
            as_embed_dim = 96
            args.bkb_pretrain = './data/ckpt/asmlp/asmlp_tiny_patch4_shift5_224.pth'
        elif 'sma' in bkb_type:
            as_depths = [2,2,18,2]
            as_drop_path_rate = 0.3
            as_embed_dim = 96
            args.bkb_pretrain = './data/ckpt/asmlp/asmlp_small_patch4_shift5_224.pth'
        elif 'bas' in bkb_type:
            as_depths = [2,2,18,2]
            as_drop_path_rate = 0.5
            as_embed_dim = 128
            args.bkb_pretrain = './data/ckpt/asmlp/asmlp_base_patch4_shift5_224.pth'

        model_bck = ASMLP(pretrain_img_size=224, patch_size=4, in_chans=3,
                              embed_dim=as_embed_dim, depths=as_depths,
                              shift_size=5, mlp_ratio=4., as_bias=True,
                              drop_rate=0., drop_path_rate=as_drop_path_rate,
                              patch_norm=True,
                              out_indices=[3],
                              # out_indices=(0, 1, 2, 3),
                              frozen_stages=-1,
                              use_checkpoint=False, logger=args.logger,
                              pretrained=args.bkb_pretrain)
    else:
        raise ValueError("Wrong backbone type")
    return model_bck
