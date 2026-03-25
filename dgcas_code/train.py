import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import random
import time
import datetime
from pathlib import Path
import sys
import scipy.io as io
from sconf import Config
import errno
import numpy as np
import pandas as pd
import csv
import math
from tqdm import tqdm, trange
import yaml
import matplotlib
from data_list import ImageList, ImageList_tr_te
import pre_process as prep

from parser import get_args_parser
from util.misc import inv_lr_scheduler, strategy_progressive, entropy_loss_func, feat_sim_cos_T1, feat_sim_cos_T2, \
    file_path_all, file_path_map, cate_num_all_dataset, num_coarse_cate_dataset, columns_csv_all, columns_csv_bd
import util.misc as misc
from util.loggerset import basicset_logger, log_init_config
from models import build_models
from engine import test_target
from TEST1 import test_target1
from plot_main import main_plot, reprocess_csv

torch.set_num_threads(1)
torch.set_printoptions(threshold=100, profile="short", sci_mode=False, precision=5)
now = int(time.time())
timeArray = time.localtime(now)
Time = time.strftime("%Y%m%d_%H%M", timeArray)

def main(args, i_time, all_times=0):
    logger = basicset_logger(args, i_time, all_times)
    args.logger = logger
    logger.info(f"Command: {' '.join(sys.argv)}")
    log_file_txt = '{}.txt'.format(args.log_file_name)
    log_file_csv = '{}.csv'.format(args.log_file_name)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    cate_all = cate_num_all_dataset[args.dataset]
    if args.granu_num is None:
        args.granu_num = num_coarse_cate_dataset[args.dataset] + 1
    fine_coarse_map = []
    if args.dataset in file_path_map:
        with open(file_path_map[args.dataset], 'r') as file_map:
            line = file_map.readline()
            while line:
                line_list = line.strip().split(' ')
                fine_coarse_map.append([int(line_list[i]) for i in range(len(line_list))]) #转化为整数

                if len(line_list) < 4:
                    for iii in range(len(line_list), 4):
                        fine_coarse_map[-1].append(0)

                line = file_map.readline()
    else:
        fine_coarse_map = None
    args.fine_coarse_map = fine_coarse_map

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        resume_temp = args.resume
        viz_temp = args.viz
        eval_temp = args.eval

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            args = checkpoint['args']
            args.distributed = False
            args.resume = resume_temp
            args.viz = viz_temp
            args.eval = eval_temp
            args.start_epoch = checkpoint['epoch'] + 1
            logger.info('start epoch is {}.\n\n\n\n'.format(args.start_epoch))

    model, criterion = build_models(args)
    model = model.to(device)
    state_dict = model.state_dict()
    model_without_ddp = model
    log_init_config(args, model_without_ddp, all_times)

    file_path = file_path_all[args.dataset]
    dataset_source = file_path[args.source]
    dataset_target = dataset_test = file_path[args.target]
    dataset_loaders = {}
    dataset_list = ImageList(open(dataset_source).readlines(),
                             transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["train"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size['train'],
                                                           shuffle=True, num_workers=2)
    dataset_list = ImageList(open(dataset_target).readlines(),
                             transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["val"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size['val'],
                                                         shuffle=True, num_workers=2)
    dataset_list = ImageList(open(dataset_test).readlines(),
                             transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size['test'],
                                                          shuffle=False, num_workers=2)
    prep_dict_test = prep.image_test_10crop(resize_size=256, crop_size=224)
    for i in range(args.val_times):
        dataset_list = ImageList(open(dataset_test).readlines(), transform=prep_dict_test["val" + str(i)])
        dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dataset_list,
                                                                      batch_size=args.batch_size[
                                                                          "val" + str(i)],
                                                                      shuffle=False, num_workers=2)

    optimizer_dict = []
    # 深度学习模型的不同部分设置不同的优化器参数
    for i_name, i_module in model.named_children():
        if i_name == 'backbone_c':
            optimizer_dict.append(
                {"params": filter(lambda p: p.requires_grad, model.backbone_c.parameters()), "lr": 0.1})
        elif i_name == 'backbone_f':
            optimizer_dict.append(
                {"params": filter(lambda p: p.requires_grad, model.backbone_f.parameters()), "lr": 0.1})

        elif i_name == 'btnk_layer':
            for j_name, j_module in i_module.named_children():
                optimizer_dict.append(
                    {"params": filter(lambda p: p.requires_grad, model.btnk_layer[j_name].parameters()), "lr": 1})
        elif i_name == 'clsf_layer':
            for j_name, j_module in i_module.named_children():
                optimizer_dict.append(
                    {"params": filter(lambda p: p.requires_grad, model.clsf_layer[j_name].parameters()), "lr": 1})
        elif i_name == 'dis_lin':
            for j_name, j_module in i_module.named_children():
                optimizer_dict.append(
                    {"params": filter(lambda p: p.requires_grad, model.dis_lin[j_name].parameters()), "lr": 1})
        elif i_name == 'input_proj':
            for j_name, j_module in i_module.named_children():
                optimizer_dict.append(
                    {"params": filter(lambda p: p.requires_grad, model.input_proj[j_name].parameters()), "lr": 1})
        if args.da == 'da':
            optimizer_dict.append(
                {"params": filter(lambda p: p.requires_grad, model.discriminator.parameters()), "lr": 1})
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    # ************************************************************************************************
    # ****start training
    # ************************************************************************************************
    len_source = len(dataset_loaders["train"]) - 1
    len_target = len(dataset_loaders["val"]) - 1
    iter_source = iter(dataset_loaders["train"])
    iter_target = iter(dataset_loaders["val"])
    acc_best = 0.0
    start_time = time.time()
    if all_times == 1:
        args.max_iteration = int(int(args.max_iteration) / args.world_size)
        args.test_interval = int(int(args.test_interval) / args.world_size)
    else:
        args.max_iteration = int(args.max_iteration)
        args.test_interval = int(args.test_interval)
    start_time_epoch = time.time()

    for iter_num in trange(1, args.max_iteration + 1):
        iter_num_t = torch.tensor(iter_num - 1, dtype=torch.float64).requires_grad_(False)
        model.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()
        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["val"])
        data_source = next(iter_source)
        data_target = next(iter_target)
        inputs_source, labels_source = data_source #labels_source的形状为[32,7]
        inputs_target, labels_target = data_target

        inputs = inputs_source
        inputs = inputs.to(device)
        if args.use_feature_extractor is True:
            logits_fine, logits_coarse, domain_predicted, out_feat_btnk, out_feat, logits_cnfd, cond_w_all, rank_w_all = model(
                inputs,
                iter_num_t)
        else:
            logits_fine, logits_coarse, domain_predicted, out_feat_btnk, logits_cnfd, cond_w_all, rank_w_all = model(
                inputs,
                iter_num_t)
        if args.b_compute_w is True:
            w_weight = [model.clsf_layer[str(0)].classifier.weight.data]  # WT: [L类别，C通道][200, 2048]
        else:
            w_weight = None
        if args.use_feature_extractor is True:
            total_loss_train, loss_dict_train, oth_num_dict_train, oth_num_dict_T1_train, oth_num_dict_T2_train, \
                oth_part_dict_train, w_oth_num_train, w_oth_num_T1_train, w_oth_num_T2_train = criterion(labels_source,
                                                                                                         logits_fine,
                                                                                                         logits_coarse,
                                                                                                         logits_cnfd,
                                                                                                         domain_predicted,
                                                                                                         out_feat_btnk,
                                                                                                         out_feat,
                                                                                                         iter_num_t,
                                                                                                         cond_w_all,
                                                                                                         w_weight=w_weight,
                                                                                                         b_update_centroids=args.b_update_centroids,
                                                                                                         prototype=args.prototype)
        else:
            total_loss_train, loss_dict_train, oth_num_dict_train, oth_num_dict_T1_train, oth_num_dict_T2_train, \
                oth_part_dict_train, w_oth_num_train, w_oth_num_T1_train, w_oth_num_T2_train = criterion(labels_source,
                                                                                                         logits_fine,
                                                                                                         logits_coarse,
                                                                                                         logits_cnfd,
                                                                                                         domain_predicted,
                                                                                                         out_feat_btnk,
                                                                                                         iter_num_t,
                                                                                                         cond_w_all,
                                                                                                         w_weight=w_weight,
                                                                                                         b_update_centroids=args.b_update_centroids,
                                                                                                         prototype=args.prototype)

        centriods_dict1 = criterion.output_all_centriods()
        if args.testclassifier_weights is True:
            classifier_weights = {}
            classifier_biases = {}
            for i in range(args.granu_num):
                weight = state_dict[f'clsf_layer.{i}.classifier.weight']
                bias = state_dict[f'clsf_layer.{i}.classifier.bias']
                classifier_weights[i] = weight
                classifier_biases[i] = bias
        total_loss_train.backward()
        optimizer.step()
        oth_cov_T1 = {}
        oth_cov_T2 = {}

        test_interval = args.test_interval
        ckpt_iter_times = args.test_interval * 1
        if iter_num % test_interval == 0 or iter_num % args.max_iteration == 0:
            model.eval()
            training_time = time.time() - start_time
            epoch_time = time.time() - start_time_epoch
            still_time = epoch_time * (args.max_iteration - iter_num)
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=int(still_time))
            epoch_time_str = str(datetime.timedelta(seconds=float(epoch_time)))
            trainging_time_str = str(datetime.timedelta(seconds=int(training_time)))
            still_time_str = str(datetime.timedelta(seconds=int(still_time)))
            finish_time_str = str(finish_time.strftime('%m%d-%H%M'))
            acc_dict_train, _ = criterion.output_acc()
            csi_dict_train, _ = criterion.output_consistency()
            acc_dict_train = {**{f'train_{k}': v.item() for k, v in acc_dict_train.items()}}
            acc_dict_train['train_acc_ave'] = sum([value_i for value_i in acc_dict_train.values()]) / len(acc_dict_train)
            acc_dict_train['train_consist'] = csi_dict_train.item()
            loss_dict_train = {**{f'train_{k}': v.item() / float(test_interval) for k, v in loss_dict_train.items()}}
            oth_num_dict_train = {**{f'train_{k}': v / float(test_interval) for k, v in oth_num_dict_train.items()}}
            oth_num_dict_T1_train = {**{f'train_{k}': v / float(test_interval) for k, v in oth_num_dict_T1_train.items()}}
            oth_num_dict_T2_train = {**{f'train_{k}': v / float(test_interval) for k, v in oth_num_dict_T2_train.items()}}
            w_oth_num_train = {**{f'train_{k}': v / float(test_interval) for k, v in w_oth_num_train.items()}}
            w_oth_num_T1_train = {**{f'train_{k}': v / float(test_interval) for k, v in w_oth_num_T1_train.items()}}
            w_oth_num_T2_train = {**{f'train_{k}': v / float(test_interval) for k, v in w_oth_num_T2_train.items()}}
            oth_part_dict_train = {**{f'train_{k}': v / float(test_interval) for k, v in oth_part_dict_train.items()}}
            oth_cov_T1_train = {**{f'train_{k}': v / float(test_interval) for k, v in oth_cov_T1.items()}}
            oth_cov_T2_train = {**{f'train_{k}': v / float(test_interval) for k, v in oth_cov_T2.items()}}
            criterion.init_loss_and_oth()

            cond_dict_train = {}
            rank_dict_train = {}
            if args.testcentriods:
                acc_dict_test, loss_dict_test, oth_num_dict_test, oth_num_dict_T1_test, oth_num_dict_T2_test, oth_part_dict_test, w_oth_num_test, w_oth_num_T1_test, w_oth_num_T2_test = test_target1(
                    dataset_loaders, model, criterion, args, centriods_dict1,classifier_weights,classifier_biases)
            else:
                acc_dict_test, loss_dict_test, oth_num_dict_test, oth_num_dict_T1_test, oth_num_dict_T2_test, oth_part_dict_test, w_oth_num_test, w_oth_num_T1_test, w_oth_num_T2_test = test_target(
                    dataset_loaders, model, criterion, args)
            log_stat = {'iter_num': iter_num,
                        **{k: v for k, v in acc_dict_train.items()},
                        **{k: v for k, v in acc_dict_test.items()},
                        **{k: v for k, v in loss_dict_train.items()},
                        **{k: v for k, v in loss_dict_test.items()},
                        **{k: v for k, v in oth_num_dict_train.items()},
                        **{k: v for k, v in oth_num_dict_T1_train.items()},
                        **{k: v for k, v in oth_num_dict_T2_train.items()},
                        **{k: v for k, v in w_oth_num_train.items()},
                        **{k: v for k, v in w_oth_num_T1_train.items()},
                        **{k: v for k, v in w_oth_num_T2_train.items()},
                        **{k: v for k, v in oth_part_dict_train.items()},
                        **{k: v for k, v in oth_cov_T1_train.items()},
                        **{k: v for k, v in oth_cov_T2_train.items()},
                        **{k: v for k, v in oth_num_dict_test.items()},
                        **{k: v for k, v in oth_num_dict_T1_test.items()},
                        **{k: v for k, v in oth_num_dict_T2_test.items()},
                        **{k: v for k, v in w_oth_num_test.items()},
                        **{k: v for k, v in w_oth_num_T1_test.items()},
                        **{k: v for k, v in w_oth_num_T2_test.items()},
                        **{k: v for k, v in oth_part_dict_test.items()},
                        **{k: v for k, v in cond_dict_train.items()},
                        **{k: v for k, v in rank_dict_train.items()},
                        }
            tb_dict = {'iter_num': iter_num,
                       'train_acc_ave': log_stat['train_acc_ave'], 'test_acc_ave': log_stat['test_acc_ave'],
                       'train_consistency': log_stat['train_consist'], 'test_consistency': log_stat['test_consist'],
                       'train_acc_0': log_stat['train_acc_0'], 'test_acc_0': log_stat['test_acc_0'],
                       'train_total_loss': log_stat['train_total_loss'], 'test_total_loss': log_stat['test_total_loss']}
            args.logger.info('iter{:05d}: train_ave_acc {:.4f}, test_ave_acc {:.4f}, '
                             'train_consistency {:.4f}, test_consistency {:.4f}, '
                             'train_acc_0 {:.4f}, test_acc_0 {:.4f}, '
                             'train_total_loss{:.6f}, test_total_loss{:.6f}, '
                             'single_epoch_time {}, total_time {}, '
                             'still_need {}, would finish in {}.'.format(iter_num,
                                                                         log_stat['train_acc_ave'],
                                                                         log_stat['test_acc_ave'],
                                                                         log_stat['train_consist'],
                                                                         log_stat['test_consist'],
                                                                         log_stat['train_acc_0'],
                                                                         log_stat['test_acc_0'],
                                                                         log_stat['train_total_loss'],
                                                                         log_stat['test_total_loss'],
                                                                         epoch_time_str, trainging_time_str,
                                                                         still_time_str,
                                                                         finish_time_str))

            log_stats = {**{f'{k}': v for k, v in log_stat.items()}}
            with Path(os.path.join(args.log_dir, log_file_txt)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if i_time == 0:
                acc_now = log_stat['test_acc_ave']
                checkpoint_paths = []
                if acc_now > acc_best:
                    checkpoint_paths.append(Path(args.log_dir) / f'ckpt_best.pth')
                    best_iter_num = iter_num
                    acc_best = acc_now
            else:
                checkpoint_paths = []
            if args.bn_store_ckpt is True and iter_num % ckpt_iter_times == 0 and i_time == 0:
                checkpoint_paths.append(Path(args.log_dir) / f'ckpt_{iter_num:03}.pth')

            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'args': args,
                    'logs': log_stat,
                    'centroids': centriods_dict1,
                }, checkpoint_path)
    time.sleep(10)
    all_domain_csv_dir = '{}/{}.csv'.format(args.output_path, args.log_file_name_1)
    args.data_path = args.output_dir

    result_value = main_plot(args, b_resume=False, b_from_code=True, other_plot_type=2, logger=args.logger)

    index_i = [f'{args.source}2{args.target}_{i_time}', f'{args.source}2{args.target}_{i_time}']

    pd1 = pd.DataFrame(result_value, index=index_i).reindex(columns=columns_csv_all)

    args.logger.info('Training Over\n')
    args.logger.info('changing txt to csv and portraying figures')
    centriods_dict = criterion.output_all_centriods()
    centroids_to_save = {}
    for i_e_g in list(range(args.g))[::-1]:
        centroids_to_save[f'feat_centroid_common_{i_e_g}'] = centriods_dict[
            f'feat_centroid_common_{i_e_g}'].detach().cpu().numpy()
        centroids_to_save[f'feat_centroid_privac_{i_e_g}'] = centriods_dict[
            f'feat_centroid_privac_{i_e_g}'].detach().cpu().numpy()
        centroids_to_save[f'feat_centroid_noise_{i_e_g}'] = centriods_dict[
            f'feat_centroid_noise_{i_e_g}'].detach().cpu().numpy()
        centroids_to_save[f'feat_centroid_all_{i_e_g}'] = centriods_dict[
            f'feat_centroid_all_{i_e_g}'].detach().cpu().numpy()
    io.savemat(args.output_dir + '/' + 'features_centroids.mat', centroids_to_save)
    if os.path.exists(all_domain_csv_dir) is False:
        pd1.to_csv(all_domain_csv_dir, mode='w', encoding='utf-8')
    else:
        pd1.to_csv(all_domain_csv_dir, mode='a+', encoding='utf-8', header=False)

    args.logger.info('log_file_name = {}'.format(args.log_file_name))
    args.logger.info('log_dir = {}'.format(args.log_dir))
    args.logger.info(f"Command: {' '.join(sys.argv)}")
    args.logger.info('########################## Good Luck ##########################')
    args.logger.info('########################## Good Luck ##########################\n\n')
    time.sleep(10)
    del args.logger, logger
    return pd1, all_domain_csv_dir

if __name__ == '__main__':
    all_times = 0
    start_time_all_all = time.time()
    parser = argparse.ArgumentParser('Causal inspired Fine grained Domain Generalization', parents=[get_args_parser()])
    document_root = os.getcwd()
    config_file = os.path.join(document_root, 'config.yaml')
    with open(config_file, 'r', encoding="utf8") as f:
        default_args = yaml.safe_load(f)
    parser.set_defaults(**default_args)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.abs_times = 0
    if args.bs_test is None:
        args.batch_size = {"train": args.batchsize, "val": args.batchsize, "test": args.batchsize}
        for i in range(args.val_times):
            args.batch_size["val" + str(i)] = args.batchsize
    else:
        args.batch_size = {"train": args.batchsize, "val": args.batchsize, "test": args.bs_test}
        for i in range(args.val_times):
            args.batch_size["val" + str(i)] = args.bs_test

    # 开始在一个大domain测试
    if args.dataset == 'cp2':
        domain_all = ["c", "p"]
    elif args.dataset == 'bd':
        domain_all = ["c", "i", "n"]
    elif args.dataset == 'cars':
        domain_all = ["w", "s"]
    elif args.dataset == 'in':
        domain_all = ['single']
        assert args.train_type == 'tr', 'only train on imagenet dataset, set args.train_type == tr'
    domain_total_num = len(domain_all)
    if args.b_onlytrain_f is True:
        args.granu_num = 1

    domain_s = []
    domain_t = []
    for i_domain in range(domain_total_num):
        if i_domain < domain_total_num - 1:
            domain_s.append(i_domain)
            domain_t.append(i_domain + 1)
            domain_s.append(i_domain + 1)
            domain_t.append(i_domain)
        elif i_domain == domain_total_num - 1 and domain_total_num > 2:
            domain_s.append(0)
            domain_t.append(i_domain)
            domain_s.append(i_domain)
            domain_t.append(0)
        else:
            pass
    print('domain_s: ', domain_s)
    print('domain_t: ', domain_t)

    num_domain_conb = len(domain_s) #2
    args.time_all_start = time.strftime('%m%d%H%M')
    if args.train_type == 'tral':
        for i_domain in range(args.startdomain, num_domain_conb): #源域的数量长度 CP2为2，大循环循环两次

            result_pd_domain = pd.DataFrame()

            domain_all_d = domain_all.copy()
            args.target = domain_all_d[domain_t[i_domain]]
            args.source = domain_all_d[domain_s[i_domain]]

            print('source name is: {}'.format(args.source))
            print('to')
            print('domain name is: {}'.format(args.target))

            for code_time in range(args.times):
                all_times += 1
                args.abs_times = all_times
                result_pd, all_domain_csv_dir = main(args, code_time, all_times)
        if misc.is_main_process():
            reprocess_csv(all_domain_csv_dir)
        print('finish all {}\n'.format(domain_all))
    training_time_all_all = time.time() - start_time_all_all
    trainging_time_all_all_str = str(datetime.timedelta(seconds=int(training_time_all_all)))
    print('All time {}.'.format(trainging_time_all_all_str))
    misc.logger_finish(args)