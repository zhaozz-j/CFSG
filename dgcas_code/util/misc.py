# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F
import math
from scipy.stats import gamma

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

#file_path_all字典
file_path_all = {
            'cp2': {
                    "p": "./dataset_list/cub200_drawing_multi.txt",
                    "c": "./dataset_list/cub200_2011_multi.txt"
                    },
            # 'cp2': {
            #         "p": "/mnt/sdb/zhaojiaojiao/dataset_list/generated_labels/cp2/clip_avg_labels_map/cub200_drawing_multi.txt",
            #         "c": "/mnt/sdb/zhaojiaojiao/dataset_list/generated_labels/cp2/clip_avg_labels_map/cub200_2011_multi.txt"
            #         },
            # 'cp2': {
            #         "p": "/mnt/sdb/zhaojiaojiao/dataset_list/generated_labels/cp2/gpt_labels_map/cub200_drawing_multi.txt",
            #         "c": "/mnt/sdb/zhaojiaojiao/dataset_list/generated_labels/cp2/gpt_labels_map/cub200_2011_multi.txt"
            #         },
            # 'cp2': {
            #         "p": "/mnt/sdb/zhaojiaojiao/dataset_list/generated_labels/cp2/deepcluster_labels_map/cub200_drawing_multi.txt",
            #         "c": "/mnt/sdb/zhaojiaojiao/dataset_list/generated_labels/cp2/deepcluster_labels_map/cub200_2011_multi.txt"
            #         },
            # 'cp2': {
            #         "p": "./dataset_list/test_with_labels.txt",
            #         "c": "./dataset_list/train_with_labels.txt"
            #         },
            'bd': {
                    "c": "./dataset_list/bird31_cub2011_multi.txt",
                    "i": "./dataset_list/bird31_ina_list_2017_multi.txt",
                    "n": "./dataset_list/bird31_nabirds_list_multi.txt"
                    },
            'ci2':{
                     "c": "./dataset_list/bird31_cub2011_multi.txt",
                     "i": "./dataset_list/bird31_ina_list_2017_multi.txt",
                },
            'cn2': {
                    "c": "./dataset_list/bird31_cub2011_multi.txt",
                    "n": "./dataset_list/bird31_nabirds_list_multi.txt",
                },
            'in2': {
                    "i": "./dataset_list/bird31_ina_list_2017_multi.txt",
                    "n": "./dataset_list/bird31_nabirds_list_multi.txt"
                    },
            'cars': {
                    "s": "./dataset_list/ccars_sv_multi.txt",
                    "w": "./dataset_list/ccars_web_multi.txt"
                    },
            'in': {
                "s": "./dataset_list/ccars_sv_multi.txt",
                "w": "./dataset_list/ccars_web_multi.txt"
            },
            '12cp2':{
                "c": "/mnt/sdb/zhaojiaojiao/dataset_list/12cub200_2011_multi.txt",
                "p": "/mnt/sdb/zhaojiaojiao/dataset_list/12cub200_drawing_multi.txt"
            },
            '13cp2':{
                "c": "/mnt/sdb/zhaojiaojiao/dataset_list/13cub200_2011_multi.txt",
                "p": "/mnt/sdb/zhaojiaojiao/dataset_list/13cub200_drawing_multi.txt"
            },
            '14cp2':{
                "c": "/mnt/sdb/zhaojiaojiao/dataset_list/14cub200_2011_multi.txt",
                "p": "/mnt/sdb/zhaojiaojiao/dataset_list/14cub200_drawing_multi.txt"
            },
            '123cp2':{
                "c": "/mnt/sdb/zhaojiaojiao/dataset_list/123cub200_2011_multi.txt",
                "p": "/mnt/sdb/zhaojiaojiao/dataset_list/123cub200_drawing_multi.txt"
            },
            '134cp2':{
                "c": "/mnt/sdb/zhaojiaojiao/dataset_list/134cub200_2011_multi.txt",
                "p": "/mnt/sdb/zhaojiaojiao/dataset_list/134cub200_drawing_multi.txt"
            }
}

file_path_map = {
            'cp2': './dataset_list_yuan/cub_labels_map.txt',
            # 'cp2': '/mnt/sdb/zhaojiaojiao1/GBC/GBC-main/generated_labels/GPT_labels.txt',
            'bd': './dataset_list_yuan/bird31_map.txt',
            # 'cp2': '/mnt/sdb/zhaojiaojiao1/GBC/GBC-main/generated_labels/clip_avg_labels.txt',
            # 'cp2': '/mnt/sdb/zhaojiaojiao1/GBC/GBC-main/generated_labels/deepcluster/cub200_dataset_map.txt',
            'cars': './dataset_list_yuan/CompCars_map.txt',
            'ci2':'./dataset_list_yuan/bird31_map.txt',
            'cn2':'./dataset_list_yuan/bird31_map.txt',
            'in2':'./dataset_list_yuan/bird31_map.txt',
            '12cp2':'/mnt/sdb/zhaojiaojiao/dataset_list_yuan/1200cub_labels_map.txt',
            '13cp2':'/mnt/sdb/zhaojiaojiao/dataset_list_yuan/1300cub_labels_map.txt',
            '14cp2':'/mnt/sdb/zhaojiaojiao/dataset_list_yuan/1400cub_labels_map.txt',
            '123cp2':'/mnt/sdb/zhaojiaojiao/dataset_list_yuan/1230cub_labels_map.txt',
            '134cp2':'/mnt/sdb/zhaojiaojiao/dataset_list_yuan/1340cub_labels_map.txt'
}

cate_num_all_dataset = {
    'cp2': [200, 122, 38, 14, 1, 1, 1],
    # 'cp2': [200, 124, 37, 13],
    'bd': [31, 25, 16, 4, 1, 1, 1],
    'cars': [281, 68, 1, 1],
    'in': [1000],
    'ci2':[31, 25, 16, 4, 1, 1, 1],
    'cn2':[31, 25, 16, 4, 1, 1, 1],
    'in2':[31, 25, 16, 4, 1, 1, 1],
    '12cp2':[200, 122, 1, 1],
    '13cp2':[200, 38, 1, 1],
    '14cp2':[200, 14, 1, 1],
    '123cp2':[200, 122, 38, 1],
    '134cp2':[200, 38, 14, 1],
}

num_coarse_cate_dataset = {
    'cp2': 3,
    'bd': 3,
    'cars': 3,
    'in': 0,
    'ci2':3,
    'cn2':3,
    'in2':3,
}

columns_csv_all = ['value_name',
    'test_acc_ave_max', 'test_acc_0_max', 'test_acc_1_max', 'test_acc_2_max', 'test_acc_3_max', 'test_consist_max',
    'train_acc_ave_max', 'train_acc_0_max', 'train_acc_1_max', 'train_acc_2_max', 'train_acc_3_max', 'train_consist_max']

columns_csv_bd = ['value_name',
    'test_acc_ave_max', 'test_acc_0_max', 'test_acc_3_max', 'test_consist_max',
    'train_acc_ave_max', 'train_acc_0_max', 'train_acc_3_max', 'train_consist_max']


def rbf_dot(pattern1, pattern2, deg):  # 高斯核函数
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = torch.sum(pattern1 ** 2, 1).reshape(size1[0], 1)
    H = torch.sum(pattern2 ** 2, 1).reshape(size2[0], 1)

    Q = torch.repeat_interleave(G, repeats=size2[0], dim=1)

    R = torch.repeat_interleave(H.T, repeats=size1[0], dim=0)

    H = Q + R - 2 * torch.matmul(pattern1, pattern2.T)

    H = torch.exp(-H / 2 / (deg ** 2))

    return H  # [3,3]


def hsic_gam(X, Y, width_x=None, width_y=None):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.shape[0]
    # ----- width of X -----
    if width_x is None:
        Xmed = X  # [3,1]

        G = torch.sum(Xmed ** 2, 1).reshape(n, 1)  # [3,1]
        Q = torch.repeat_interleave(G, n, dim=1)  # 沿着dim=1，重复3次，[3,3]
        R = torch.repeat_interleave(G.T, n, dim=0)
        dists = Q + R - 2 * torch.matmul(Xmed, Xmed.T)  # 是一个对称矩阵

        dists = dists - torch.tril(dists)  # 返回一个矩阵主对角线以下的下三角矩阵
        dists = dists.reshape(n ** 2, 1)
        # del G, Q, R
        # torch.cuda.empty_cache()

        width_x = torch.sqrt(0.5 * torch.median(dists[dists > 0]))  # median返回中位数
        # del dists
        # torch.cuda.empty_cache()
    # ----- -----

    # ----- width of Y -----
    if width_y is None:
        Ymed = Y

        G = torch.sum(Ymed ** 2, 1).reshape(n, 1)
        Q = torch.repeat_interleave(G, n, dim=1)
        R = torch.repeat_interleave(G.T, n, dim=0)

        dists = Q + R - 2 * torch.matmul(Ymed, Ymed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n ** 2, 1)
        # del G, Q, R
        # torch.cuda.empty_cache()
        width_y = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
        # del dists
        # torch.cuda.empty_cache()
    # ----- -----

    H = torch.eye(n).to(X.device) - torch.ones(n, n).to(X.device) / n  # H = I - (1/n) one * oneT 对称矩阵

    K = rbf_dot(X, X, width_x)  # 用高斯核算K
    L = rbf_dot(Y, Y, width_y)

    HSIC_trace = torch.matmul(torch.matmul(torch.matmul(K, H), L), H)
    HSIC_trace = torch.trace(HSIC_trace) /n /n
    # HSIC_trace = torch.trace(HSIC_trace.cpu()) /n /n
    HSIC_trace = HSIC_trace.cuda().item()

    return HSIC_trace


#计算特征比例和特征划分，特征比例和特征划分的结果
def compu_featpart(args, featlenth=None):
    if isinstance(args.feat_ratio, list):
        feat_ratio = args.feat_ratio
    elif isinstance(args.feat_ratio, str):
        if '_' in args.feat_ratio:
            feat_ratio = list(map(int, args.feat_ratio.split('_')))
        else:
            feat_ratio = [int(args.feat_ratio)]
    else:
        feat_ratio = [34, 33, 33]

    if len(feat_ratio) == 1:
        feat_ratio[1] = int((100 - feat_ratio[0]) / 2)
        feat_ratio[2] = 100 - sum(feat_ratio)
    elif len(feat_ratio) == 2:
        feat_ratio[2] = 100 - sum(feat_ratio)
    else:
        pass
    assert sum(feat_ratio) == 100, f'The sum of proportion of the feature ratio is {sum(feat_ratio)}'

    if featlenth is None:
        if args.norm_type == 'T1':
            featlenth = args.feat_dim
        else:
            # featlenth = 49
            featlenth = args.feat_dim
    else:
        pass

    if args.norm_type == 'T1':  # T1：点像素49个256维feat, 把256给分成三部分；
        # feat_part = [math.ceil(i_ratio / 100.0 * self.args.feat_num) for i_ratio in feat_ratio]
        feat_part = [math.ceil(i_ratio / 100.0 * featlenth) for i_ratio in feat_ratio]
        feat_part = [sum(feat_part[:i_ratio]) for i_ratio in range(1, len(feat_ratio) + 1)]
        # feat_part[-1] = self.args.feat_num
        feat_part[-1] = featlenth
    elif args.norm_type == 'T2':  # T2 就是256个49维语义feat"，把49给出三部分
        feat_part = [math.ceil(i_ratio / 100.0 * featlenth) for i_ratio in feat_ratio]
        feat_part = [sum(feat_part[:i_ratio]) for i_ratio in range(1, len(feat_ratio) + 1)]
        feat_part[-1] = featlenth
    else:
        feat_part = None
    if feat_part[1] > feat_part[2]:
        feat_part[1] = feat_part[2]

    return feat_ratio, feat_part



@torch.no_grad()
class find_nan_all(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.logger = args.logger

    @torch.no_grad()
    def forward(self, lable, values=None):
        lable_nan = 0

        for n, p in self.model.named_parameters():
            if p.grad is not None:

                grad = p.grad.detach()
                if not torch.equal(grad, grad):

                    self.logger.info('\n\n\nnan_{}\n'.format(lable))
                    # self.logger.info(n)
                    # self.logger.info(p.data)  # 内存占用太大
                    self.logger.info('judge = 0 ,出现了nan')
                    self.logger.info(n)
                    lable_nan = 1

        if lable_nan == 1:
            if values is not None:
                print(values)
            while 1 < 2:
                for i in range(100):
                    i += 1


@torch.no_grad()
def find_nan(x, lable):

    total_norm = torch.norm(x,2)

    if total_norm.isinf():
        print('\n\n\ninf_{}\n\n\n'.format(lable))
        while 1 < 2:
            for i in range(100):
                i += 1

    if not torch.equal(x, x):

        print('\n\n\nnan_{}\n\n\n'.format(lable))
        while 1 < 2:
            for i in range(100):
                i += 1


def logger_finish(args=None, logger=None, print=True):
    # if print is False:
    #     pass
    # elif args is not None:
    #     log_dir = getattr(args, 'log_dir', 'False')
    #     info = getattr(args, 'info', 'False')
    #     if isinstance(log_dir, str) and isinstance(info, str):
    #         print('Training Finished. Results Path: {}'.format(log_dir))
    #         print('Experiments info: \n {}'.format(info))
    #     print('Have a good results! ')
    while 1 < 2:
        for i in range(100):
            i += 1
    print('May the force be with you! ')
    print('Lucky Lucky Lucky! ')
    print('Paper Paper Paper! ')


#用于动态调整优化器的学习率
#optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, lr_drop=None):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)  # iter_num最大20000的话，这个倍数大概是1~0.1，也就是init_lr随着训练越来越小
    if lr_drop is not None:
        if iter_num > lr_drop:
            lr = init_lr * 0.1
    i_lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i_lr]
        i_lr += 1
    return optimizer


#计算输入张量的熵损失
def entropy_loss_func(input_):  # [36,200]输入的是细粒度的预测分类结果
    mask = input_.ge(0.000001)  # 逐个元素对比，如果大于等于0.000001，这个位置就是1，否则为0 【36，200】
    mask_out = torch.masked_select(input_, mask)  # 返回input中mask为True的元素，组成1维tensor [7200]
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


#该函数根据不同的策略计算平滑参数 smooth 的值
def strategy_progressive(iter_num, initial_smooth, final_smooth, max_iter_num, strategy, erate=-10):
    if strategy == 'e':
        lambda_p = 2 / (1 + math.exp(erate * iter_num / max_iter_num)) - 1  # 从0变到1
    elif strategy == '-e':
        lambda_p = 2 / (1 + math.exp(erate * ((max_iter_num - iter_num) / max_iter_num))) - 1
    elif strategy == 'l':
        lambda_p = iter_num / max_iter_num
    elif strategy == 's':
        lambda_p = iter_num // (max_iter_num // 10) * 0.1
    elif strategy == 'x':
        lambda_p = math.pow(2, (iter_num / max_iter_num)) - 1
    else:
        lambda_p = 2 / (1 + math.exp(erate * iter_num / max_iter_num)) - 1
    smooth = initial_smooth + (final_smooth - initial_smooth) * lambda_p  # 从0.9~0.1 递减
    return smooth


def feat_sim_cos_T1(feat_1, feat_2=None, method='diag'):
    if method == 'diag':
        if len(feat_1.shape) == len(feat_2.shape) == 3:
            feat_1 = F.normalize(feat_1, p=2, dim=1)
            feat_2 = F.normalize(feat_2, p=2, dim=1)
            feat_cos = torch.matmul(feat_1.permute(0, 2, 1), feat_2)
            # for i_cos in range(feat_1.shape[0]):
            #     xxx = torch.norm(torch.diag(feat_cos[i_cos]), p=2)
            feat_cos_sim = sum([torch.norm(torch.diag(feat_cos[i_cos]), p=2) for i_cos in range(feat_1.shape[0])]) / (
                    feat_1.shape[0])  # 要除以样本数

        elif len(feat_1.shape) == len(feat_2.shape) == 2:
            feat_1 = F.normalize(feat_1, p=2, dim=1)
            feat_2 = F.normalize(feat_2, p=2, dim=1)
            feat_cos = torch.matmul(feat_1.permute(1, 0), feat_2)
            feat_cos_sim = torch.norm(torch.diag(feat_cos), p=2)  # 要除以样本数

    elif method == 'hsic':
        hsic_all = 0.0
        if feat_2 is None:
            if len(feat_1.shape) == 2:
                len_feat_1 = feat_1.shape[0]
                len_dim = feat_1.shape[1]
                if len_feat_1 == 1:
                    feat_cos_sim = 0
                else:
                    num_hsic = 0
                    for i in range(len_feat_1):
                        for j in range(i+1, len_feat_1):
                            X = feat_1[i].view(len_dim, 1)  # [1024,1]
                            Y = feat_1[j].view(len_dim, 1)  # [1024,1]
                            # Y = X  # [1024,1]
                            i_hsic = hsic_gam(X, Y)
                            num_hsic += 1
                            hsic_all += i_hsic
                    feat_cos_sim = hsic_all / num_hsic * 10
        else:
            assert 'error in hsic computing!'

    elif method == 'oushi':
        if len(feat_1.shape) == 2:

            feat_cos = torch.cdist(feat_1, feat_1, p=2)

            feat_cos_sim = feat_cos / feat_1.shape[1]  # 除以特征维度
            feat_cos_sim = torch.norm(feat_cos_sim, p=2)  # norm()函数求范数，默认求F范数,p=2表示求F范数

    elif method == 'diag-I' or feat_2 is None:
        if len(feat_1.shape) == 2:
            I_g = torch.eye(feat_1.shape[0]).to(feat_1.device)

            feat_1 = F.normalize(feat_1, p=2, dim=1)
            feat_cos = torch.matmul(feat_1, feat_1.permute(1, 0))

            feat_cos_sim = feat_cos - I_g
            feat_cos_sim = torch.norm(feat_cos_sim, p=2)  # norm()函数求范数，默认求F范数,p=2表示求F范数


    return feat_cos_sim


def feat_sim_cos_T2(feat_1, feat_2=None, method='diag'):
    if method == 'diag':
        if len(feat_1.shape) == len(feat_2.shape) == 3:
            feat_1 = F.normalize(feat_1, p=2, dim=2)
            feat_2 = F.normalize(feat_2, p=2, dim=2)
            feat_cos = torch.matmul(feat_1, feat_2.permute(0, 2, 1))
            feat_cos_sim = sum([torch.norm(torch.diag(feat_cos[i_cos]), p=2) for i_cos in range(feat_1.shape[0])]) / (
                    feat_1.shape[0])  # 要除以样本数，再除以多少个特征

        elif len(feat_1.shape) == len(feat_2.shape) == 2:
            feat_1 = F.normalize(feat_1, p=2, dim=1)
            feat_2 = F.normalize(feat_2, p=2, dim=1)
            feat_cos = torch.matmul(feat_1, feat_2.permute(1, 0))
            feat_cos_sim = torch.norm(torch.diag(feat_cos), p=2)  # 要除以样本数

    elif method == 'diag-I' or feat_2 is None:
        if len(feat_1.shape) == 2:
            I_g = torch.eye(feat_1.shape[0]).to(feat_1.device)

            feat_1 = F.normalize(feat_1, p=2, dim=1)
            feat_cos = torch.matmul(feat_1, feat_1.permute(1, 0))

            feat_cos_sim = feat_cos - I_g
            feat_cos_sim = torch.norm(feat_cos_sim, p=2)  # norm()函数求范数，默认求F范数,p=2表示求F范数

    return feat_cos_sim


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]


    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])

    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        # max_size = [3,640,640]

        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print('rank', args.rank)
        print('world_size', args.world_size)
        print('gpu', args.gpu, '\n\n\n')
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        print('gpu\n\n\n', args.gpu)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def updataargs(args, args_new):

    args_default = args

    args = args_new
    args.h = 0

    # args.backbone = args_new.backbone
    # args.batchsize = args_new.batchsize
    # args.dataset = args_new.dataset
    # args.lr = args_new.lr
    # # args.lr_backbone = args_new.lr_backbone
    # # args.weight_decay = args_new.weight_decay
    # # args.snr = args_new.snr
    # # args.mean = args_new.mean
    # # args.var = args_new.var
    # # args.noise_p = args_new.noise_p
    # # args.train_state = args_new.train_state
    # # args.pretrain = args_new.pretrain
    # # args.eval = args_new.eval
    # # args.viz = args_new.viz
    # # args. = args_new.
    # args.feat_dim = args_new.feat_dim
    # args.feat_num = args_new.feat_num
    # # args.mu = args_new.mu
    # args.bn_cfdg = args_new.bn_cfdg
    # args.b_bkb_c = args_new.b_bkb_c
    # args.b_feat_oth = args_new.b_feat_oth
    # args.b_same_sa_diff_g_com = args_new.b_same_sa_diff_g_com
    # args.b_same_sa_diff_g_pvt = args_new.b_same_sa_diff_g_pvt
    # args.b_diff_sa_same_g_com = args_new.b_diff_sa_same_g_com
    # args.b_diff_sa_same_g_pvt = args_new.b_diff_sa_same_g_pvt
    # args.b_diff_sa_diff_g_com = args_new.b_diff_sa_diff_g_com
    # args.b_diff_sa_diff_g_pvt = args_new.b_diff_sa_diff_g_pvt
    #
    # args.c_feat_oth = args_new.c_feat_oth
    # args.c_same_sa_diff_g_com = args_new.c_same_sa_diff_g_com
    # args.c_same_sa_diff_g_pvt = args_new.c_same_sa_diff_g_pvt
    # args.c_diff_sa_same_g_com = args_new.c_diff_sa_same_g_com
    # args.c_diff_sa_same_g_pvt = args_new.c_diff_sa_same_g_pvt
    # args.c_diff_sa_diff_g_com = args_new.c_diff_sa_diff_g_com
    # args.c_diff_sa_diff_g_pvt = args_new.c_diff_sa_diff_g_pvt
    # args.c_loss_entropy_source = args_new.c_loss_entropy_source
    # args.c_loss_entropy_target = args_new.c_loss_entropy_target
    #
    # # args.oth_num = args_new.oth_num
    # args.feat_ratio = args_new.feat_ratio
    # args.initial_smooth = args_new.initial_smooth
    # args.final_smooth = args_new.final_smooth
    # args.da = args_new.da
    # args.smooth_stratege = args_new.smooth_stratege
    #
    # args.bkb_pretrain_dir = args_new.bkb_pretrain_dir
    # args.h = args_new.h
    # # args.seed = args_new.seed
    # # args.bn_store_ckpt = args_new.bn_store_ckpt
    # args.gpu_location = args_new.gpu_location
    # # args. = args_new.
    # # args. = args_new.
    # # args. = args_new.
    # # args. = args_new.
    # # args. = args_new.

    # if hasattr(args_new, 'b_3_bkb') is False:
    #     args.eval = False
    args.b_3_bkb = getattr(args_new, 'b_3_bkb', False)
    args.model_type = getattr(args_new, 'model_type', 'seri_mlp')
    args.norm_type = getattr(args_new, 'norm_type', 'T1')
    args.b_mgc = getattr(args_new, 'b_mgc', False)
    args.cnfd_type = getattr(args_new, 'cnfd_type', 'no')
    args.c_cnfd = getattr(args_new, 'c_cnfd', '0')

    args.b_pan = getattr(args_new, 'b_pan', False)
    args.b_bkb_poor = getattr(args_new, 'b_bkb_poor', False)
    args.bn_input_p = getattr(args_new, 'bn_input_p', True)
    args.b_bkb_c = getattr(args_new, 'b_bkb_c', False)
    args.str_avgpool = getattr(args_new, 'str_avgpool', False)
    args.bn_cfdg = getattr(args_new, 'bn_cfdg', True)
    args.b_2loss = getattr(args_new, 'b_2loss', False)
    args.conv_type = getattr(args_new, 'conv_type', '2')
    args.bn_align = getattr(args_new, 'bn_align', True)
    args.granu_num = getattr(args_new, 'granu_num', 4)
    args.b_cond = getattr(args_new, 'b_cond', False)
    args.b2_stage = getattr(args_new, 'b2_stage', args_default.b2_stage)
    args.l23_stage = getattr(args_new, 'l23_stage', args_default.l23_stage)
    args.sim_method = getattr(args_new, 'sim_method', args_default.sim_method)
    args.gc = getattr(args_new, 'num_coarse_g', args_default.gc)
    args.b_pass_transition = getattr(args_new, 'b_pass_transition', False)
    args.b_f_ce = getattr(args_new, 'b_f_ce', False)
    args.b_onlytrain_f = getattr(args_new, 'b_onlytrain_f', False)

    return args







