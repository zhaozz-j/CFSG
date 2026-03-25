# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2022- 10- 07
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions: 
Input: 
Output: 
Note:

Link:
*************************************Import***********************************"""
import torch
import torch.nn as nn
from tqdm import tqdm, trange

"""**********************************Import***********************************"""
'''***************************************************************************'''
print('************************************************************************')


# 把计算acc放到criterion里
#该函数用于评估模型在验证集上的性能
def test_target(loader, model, criterion, args):
    acc_dict = {}

    with torch.no_grad():

        start_test = True
        #为每个验证集创建一个迭代器，args.val_times 表示验证集的数量
        iter_val = [iter(loader['val' + str(i)]) for i in range(args.val_times)]

        acc_dict_test_all = []
        csi_dict_test_all = []
        loss_dict_test_all = []
        oth_num_dict_test_all = []
        oth_num_dict_T1_test_all = []
        oth_num_dict_T2_test_all = []

        w_oth_num_dict_test_all = []
        w_oth_num_dict_T1_test_all = []
        w_oth_num_dict_T2_test_all = []

        oth_part_dict_test_all = []

        #验证次数10次，循环10次
        for j in range(args.val_times):
            iter_time = 0
            criterion.init_loss_and_oth()
            for i in range(len(loader['val' + str(j)])):
                iter_time += 1

                # data = iter_val[j].next()
                data = next(iter_val[j])
                inputs = data[0].to(args.device)
                labels = data[1]

                # model输入 ############################################################
                logits_fine, logits_coarse, domain_predicted, out_feat_btnk, logits_cnfd, cond_w_all, _ = model(inputs,
                                                                                                                1)
                if args.b_compute_w is True:
                    w_weight = [model.clsf_layer[str(0)].classifier.weight.data]  # WT: [L类别，C通道][200, 2048]
                else:
                    w_weight = None

                total_loss_test_val, loss_dict_test_val, oth_num_dict_test_val, oth_num_dict_T1_test_val, oth_num_dict_T2_test_val, oth_part_dict_test_val, w_oth_num_test_val, w_oth_num_T1_test_val, w_oth_num_T2_test_val = criterion(labels, logits_fine,logits_coarse, logits_cnfd,domain_predicted, out_feat_btnk,1,cond_w_all,w_weight=w_weight,b_update_centroids=False,b_eva=True)

                if args.other == 'debug':
                    if iter_time == 20:
                        break
            loss_dict_test_all.append(loss_dict_test_val.copy())
            oth_num_dict_test_all.append(oth_num_dict_test_val.copy())
            oth_num_dict_T1_test_all.append(oth_num_dict_T1_test_val.copy())
            oth_num_dict_T2_test_all.append(oth_num_dict_T2_test_val.copy())

            w_oth_num_dict_test_all.append(w_oth_num_test_val.copy())
            w_oth_num_dict_T1_test_all.append(w_oth_num_T1_test_val.copy())
            w_oth_num_dict_T2_test_all.append(w_oth_num_T2_test_val.copy())

            oth_part_dict_test_all.append(oth_part_dict_test_val.copy())

            # #################################################################################################
            # 非常重要，初始化loss和oth_num_x 字典
            criterion.init_loss_and_oth()
            # #################################################################################################
            acc_dict_test_all.append(criterion.output_acc()[0])
            csi_dict_test_all.append(criterion.output_consistency()[0])

        acc_dict_test = {**{f'{k}': 0.0 for k in acc_dict_test_all[0].keys()}}
        loss_dict_test = {**{f'{k}': 0.0 for k in loss_dict_test_val.keys()}}
        oth_num_dict_test = {**{f'{k}': 0.0 for k in oth_num_dict_test_val.keys()}}
        oth_num_dict_T1_test = {**{f'{k}': 0.0 for k in oth_num_dict_T1_test_val.keys()}}
        oth_num_dict_T2_test = {**{f'{k}': 0.0 for k in oth_num_dict_T2_test_val.keys()}}

        w_oth_num_dict_test = {**{f'{k}': 0.0 for k in w_oth_num_test_val.keys()}}
        w_oth_num_dict_T1_test = {**{f'{k}': 0.0 for k in w_oth_num_T1_test_val.keys()}}
        w_oth_num_dict_T2_test = {**{f'{k}': 0.0 for k in w_oth_num_T2_test_val.keys()}}

        oth_part_dict_test = {**{f'{k}': 0.0 for k in oth_part_dict_test_val.keys()}}

        for ii in range(args.val_times):
            for jj in acc_dict_test_all[ii].keys():
                acc_dict_test[jj] += acc_dict_test_all[ii][jj].item()
            for jj in loss_dict_test_all[ii].keys():
                loss_dict_test[jj] += loss_dict_test_all[ii][jj].item() / iter_time

            for jj in oth_num_dict_test_all[ii].keys():
                oth_num_dict_test[jj] += oth_num_dict_test_all[ii][jj] / iter_time
            for jj in oth_num_dict_T1_test_all[ii].keys():
                oth_num_dict_T1_test[jj] += oth_num_dict_T1_test_all[ii][jj] / iter_time
            for jj in oth_num_dict_T2_test_all[ii].keys():
                oth_num_dict_T2_test[jj] += oth_num_dict_T2_test_all[ii][jj] / iter_time

            for jj in w_oth_num_dict_test_all[ii].keys():
                w_oth_num_dict_test[jj] += w_oth_num_dict_test_all[ii][jj] / iter_time
            for jj in w_oth_num_dict_T1_test_all[ii].keys():
                w_oth_num_dict_T1_test[jj] += w_oth_num_dict_T1_test_all[ii][jj] / iter_time
            for jj in w_oth_num_dict_T2_test_all[ii].keys():
                w_oth_num_dict_T2_test[jj] += w_oth_num_dict_T2_test_all[ii][jj] / iter_time

            for jj in oth_part_dict_test_all[ii].keys():
                oth_part_dict_test[jj] += oth_part_dict_test_all[ii][jj] / iter_time

        acc_dict_test = {**{f'test_{k}': v / args.val_times for k, v in acc_dict_test.items()}}
        acc_dict_test['test_acc_ave'] = sum([value_i for value_i in acc_dict_test.values()]) / len(acc_dict_test)
        acc_dict_test['test_consist'] = sum(csi_dict_test_all).item() / args.val_times

        loss_dict_test = {**{f'test_{k}': v / args.val_times for k, v in loss_dict_test.items()}}
        oth_num_dict_test = {**{f'test_{k}': v / args.val_times for k, v in oth_num_dict_test.items()}}
        oth_num_dict_T1_test = {**{f'test_{k}': v / args.val_times for k, v in oth_num_dict_T1_test.items()}}
        oth_num_dict_T2_test = {**{f'test_{k}': v / args.val_times for k, v in oth_num_dict_T2_test.items()}}

        w_oth_num_dict_test = {**{f'test_{k}': v / args.val_times for k, v in w_oth_num_dict_test.items()}}
        w_oth_num_dict_T1_test = {**{f'test_{k}': v / args.val_times for k, v in w_oth_num_dict_T1_test.items()}}
        w_oth_num_dict_T2_test = {**{f'test_{k}': v / args.val_times for k, v in w_oth_num_dict_T2_test.items()}}

        oth_part_dict_test = {**{f'test_{k}': v / args.val_times for k, v in oth_part_dict_test.items()}}

    return acc_dict_test, loss_dict_test, oth_num_dict_test, oth_num_dict_T1_test, oth_num_dict_T2_test, oth_part_dict_test, \
           w_oth_num_dict_test, w_oth_num_dict_T1_test, w_oth_num_dict_T2_test
'''*'''
