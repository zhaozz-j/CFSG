import timm
from torch import nn
import torch
from util.misc import NestedTensor, is_main_process
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

""" Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        nn.init.xavier_uniform_(self.proj.weight, gain=1)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

        nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1)

    def forward(self, x):  # torch.Size([1, 512, 25, 25])
        y = self.conv1(x)  # torch.Size([1, 512, 25, 25])
        y = self.norm1(y)
        y = self.gelu1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = y + x
        y = self.gelu2(y)
        return y

class VisionTransformer(nn.Module):
    def __init__(self, name='vit_small_patch16_224', img_size=(224, 224), patch_size=4,
                 return_intermediate_states=False, in_chans=3, num_classes=10, pretrain=True, pre_path=None):

        super().__init__()

        if pre_path is None:

            self.model = timm.create_model(name, num_classes=num_classes, pretrained=pretrain, img_size=img_size[0])

        else:
            self.model = timm.create_model(name, num_classes=num_classes, pretrained=pretrain, img_size=img_size[0],
                                           pretrained_cfg_overlay=dict(file=pre_path))

        # print(self.model.default_cfg)

        self.model.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = 1024  # new版本
        self.num_channels = 2048  # ijcai版本的vit

        self.return_intermediate_states = return_intermediate_states
        if self.return_intermediate_states:
            # self.conv = nn.Conv2d(in_channels=len(self.model.blocks) * self.model.embed_dim, out_channels=512, kernel_size=1)
            self.conv = nn.Conv2d(in_channels=len(self.model.blocks) * self.model.embed_dim, out_channels=self.num_channels,
                                  kernel_size=3, stride=2, padding=1)

        else:
            # self.conv = nn.Conv2d(in_channels=self.model.embed_dim, out_channels=512, kernel_size=1)
            self.conv = nn.Conv2d(in_channels=self.model.embed_dim, out_channels=self.num_channels, kernel_size=3, stride=2,
                                  padding=1)
        # self.residual_block = Block(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.normout = nn.BatchNorm2d(self.num_channels)
        # self.geluout = nn.GELU()
        self.initialize_weights()

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, num_classes)
        )

    def initialize_weights(self):

        nn.init.xavier_uniform_(self.conv.weight, gain=1)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):

        x = x  # torch.Size([72, 3, 224, 224])
        _, _, h, w = x.shape
        B = x.shape[0]
        x = self.model.patch_embed(x)  # torch.Size([72, 196, 384])

        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # [72,1,384] stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # torch.Size([72, 197, 384])
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)  # torch.Size([72, 197, 384])
        y = []
        for blk in self.model.blocks:
            x = blk(x)
            if self.return_intermediate_states:
                y.append(x)
        # X torch.Size([10, 197, 192])
        if self.return_intermediate_states:
            x = torch.cat(y, -1)
            b, _, c = x.shape
            x = torch.reshape(x[:, 1:, :], (b, h // self.patch_size, w // self.patch_size, c)).permute(0, 3, 1,
                                                                                                       2)  # torch.Size([1, 9216, 25, 25])
        else:
            x = self.model.norm(x)  # torch.Size([72, 197, 192])  -->  [2,1601,384]
            b, _, c = x.shape

            # xx = self.model.fc_norm(x)  # torch.Size([72, 197, 192])
            # xxx = self.model.head(xx)
            # xxxx = x[:, 0]
            # xxxxx = self.to_latent(xxxx)
            # assert torch.equal(xxxx, xxxxx), 'equal'
            # yyy = self.mlp_head(xxxxx)
            # torch.Size([10, 197, 192]) --> torch.Size([10, 14, 14, 192])
            x = torch.reshape(x[:, 1:, :], (b, h // self.patch_size, w // self.patch_size, c))  # torch.Size([72, 192, 14, 14])
            x = x.permute(0, 3, 1, 2)  # torch.Size([72, 192, 14, 14])

        x = self.conv(x)  # torch.Size([10, 192, 14, 14]) --> torch.Size([72, 2048, 7, 7])  # 相当于把通道数扩成2048
        # x = self.normout(x)
        # x = self.geluout(x)
        # x = self.residual_block(x)
        x1 = self.avgpool(x)

        x1 = x1.view(x1.size(0), -1)

        return x, x1

