import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.io import loadmat
from util.misc import inv_lr_scheduler, strategy_progressive, entropy_loss_func, \
    feat_sim_cos_T1, feat_sim_cos_T2, compu_featpart, file_path_all, file_path_map, cate_num_all_dataset, \
    num_coarse_cate_dataset
import numpy as np
import scipy.linalg as linalg
import copy
import torch.multiprocessing as mp
from models import build_models
def test_target1(loader, model, criterion, args,centriods_dict1,classifier_weights,classifier_biases):
    acc_dict = {}
    with torch.no_grad():
        start_test = True
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
        feat_ratio, feat_part = compu_featpart(args)
        for j in range(args.val_times):
            iter_time = 0
            criterion.init_loss_and_oth()
            for i in range(args.val_times):
                iter_time += 1

                data = next(iter_val[j])
                inputs = data[0].to(args.device)
                labels = data[1]


                if args.use_feature_extractor is True:
                    logits_fine, logits_coarse, domain_predicted, out_feat_btnk, out_feat, logits_cnfd, cond_w_all, _ = model(
                        inputs,
                        1)
                else:
                    logits_fine, logits_coarse, domain_predicted, out_feat_btnk, logits_cnfd, cond_w_all, _ = model(
                        inputs,
                        1)
                out_feat_btnk_copy = out_feat_btnk
                if args.use_feature_extractor is True:
                    out_feat_copy = out_feat
                if len(out_feat_btnk) > 1:
                    out_feat_btnk = torch.stack(out_feat_btnk[1:])
                distances_list = []
                logits_list = []
                if args.testclassifier_weights is True:
                    out_feat_btnk = out_feat_btnk.flip(0)
                for i_g in range(args.g):
                    if args.norm_type == 'T1':
                        if args.use_feature_extractor is True:
                            g_feat_common1 = out_feat[i_g, :, :feat_part[0], :]
                            g_feat_privacy1 = out_feat[i_g, :, feat_part[0]:feat_part[1], :]
                            g_feat_noise1 = out_feat[i_g, :, feat_part[1]:, :]
                            g_feat_all1 = out_feat[i_g, :, :, :]
                        else:
                            g_feat_common1 = out_feat_btnk[i_g, :, :feat_part[0], :]
                            g_feat_privacy1 = out_feat_btnk[i_g, :, feat_part[0]:feat_part[1], :]
                            g_feat_noise1 = out_feat_btnk[i_g, :, feat_part[1]:, :]
                            g_feat_all1 = out_feat_btnk[i_g, :, :, :]
                    elif args.norm_type == 'T2':
                        g_feat_common1 = out_feat_btnk[i_g, :, :, :feat_part[0]]
                        g_feat_privacy1 = out_feat_btnk[i_g, :, :, feat_part[0]:feat_part[1]]
                        g_feat_noise1 = out_feat_btnk[i_g, :, :, feat_part[1]:]
                        g_feat_all1 = out_feat_btnk[i_g, :, :, :]

                    if args.testclassifier_weights is True:
                        weight_i_g = classifier_weights[i_g]
                        bias_i_g = classifier_biases[i_g]
                        weight_i_g_common = weight_i_g[:, :feat_part[0]]
                        weight_i_g_privacy = weight_i_g[:, feat_part[0]:feat_part[1]]
                        weight_i_g_noise = weight_i_g[:, feat_part[1]:]

                        g_feat_common1_mean = g_feat_common1.mean(dim=-1)
                        g_feat_privacy1_mean = g_feat_privacy1.mean(dim=-1)
                        g_feat_noise1_mean = g_feat_noise1.mean(dim=-1)

                        logits_common = torch.matmul(g_feat_common1_mean, weight_i_g_common.T)
                        logits_privacy = torch.matmul(g_feat_privacy1_mean, weight_i_g_privacy.T)
                        logits_noise = torch.matmul(g_feat_noise1_mean, weight_i_g_noise.T)
                        logits_all = logits_common * args.cdistance + logits_privacy * args.pdistance + logits_noise * args.ndistance
                        logits_list.append(logits_all)
                    else:
                        g_feat_centroid_common = centriods_dict1[f'feat_centroid_common_{i_g}']
                        g_feat_centroid_privac = centriods_dict1[f'feat_centroid_privac_{i_g}']
                        g_feat_centroid_noise = centriods_dict1[f'feat_centroid_noise_{i_g}']


                        g_feat_common_avg = torch.mean(g_feat_common1, dim=2)
                        g_feat_privacy_avg = torch.mean(g_feat_privacy1, dim=2)
                        g_feat_noise_avg = torch.mean(g_feat_noise1, dim=2)
                        g_feat_all_avg = torch.mean(g_feat_all1, dim=2)

                        g_feat_common_avg_expanded = g_feat_common_avg.unsqueeze(1)
                        g_feat_privacy_avg_expanded = g_feat_privacy_avg.unsqueeze(1)
                        g_feat_noise_avg_expanded = g_feat_noise_avg.unsqueeze(1)
                        g_feat_all_avg_expanded = g_feat_all_avg.unsqueeze(1)

                        g_feat_centroid_common = torch.tensor(g_feat_centroid_common)
                        g_feat_centroid_privac = torch.tensor(g_feat_centroid_privac)
                        g_feat_centroid_noise = torch.tensor(g_feat_centroid_noise)

                        g_feat_centroid_common_expanded = g_feat_centroid_common.unsqueeze(0)
                        g_feat_centroid_privac_expanded = g_feat_centroid_privac.unsqueeze(0)
                        g_feat_centroid_noise_expanded = g_feat_centroid_noise.unsqueeze(0)


                        g_feat_centroid_common_expanded = g_feat_centroid_common_expanded.to(args.device)
                        g_feat_centroid_privac_expanded = g_feat_centroid_privac_expanded.to(args.device)
                        g_feat_centroid_noise_expanded = g_feat_centroid_noise_expanded.to(args.device)
                        distances1 = torch.norm(g_feat_common_avg_expanded - g_feat_centroid_common_expanded,
                                                dim=2)
                        distances2 = torch.norm(g_feat_privacy_avg_expanded - g_feat_centroid_privac_expanded,
                                                dim=2)
                        distances3 = torch.norm(g_feat_noise_avg_expanded - g_feat_centroid_noise_expanded,
                                                dim=2)

                        distances = distances1 * args.cdistance + distances2 * args.pdistance + distances3 * args.ndistance
                        distances = F.softmax(-distances, dim=1)
                        distances_list.append(distances)


                if args.testclassifier_weights is True:
                    logits_fine1 = logits_list[0]
                    logits_coarse1 = logits_list[1:]
                else:
                    all_distances = torch.stack(distances_list, dim=0)
                    logits_coarse = all_distances[:3, :, :]
                    logits_coarse = torch.tensor(logits_coarse)
                    logits_coarse1 = torch.flip(logits_coarse, dims=[0])
                    logits_fine1 = all_distances[3, :, :]
                if args.b_compute_w is True:
                    w_weight = [model.clsf_layer[str(0)].classifier.weight.data]  # WT: [L类别，C通道][200, 2048]
                else:
                    w_weight = None

                if args.use_feature_extractor is True:
                    total_loss_test_val, loss_dict_test_val, oth_num_dict_test_val, oth_num_dict_T1_test_val, oth_num_dict_T2_test_val, oth_part_dict_test_val, w_oth_num_test_val, w_oth_num_T1_test_val, w_oth_num_T2_test_val = criterion(
                        labels, logits_fine1, logits_coarse1, logits_cnfd, domain_predicted, out_feat_btnk_copy, out_feat_copy,
                        1, cond_w_all, w_weight=w_weight, b_update_centroids=False, b_eva=True)
                else:
                    total_loss_test_val, loss_dict_test_val, oth_num_dict_test_val, oth_num_dict_T1_test_val, oth_num_dict_T2_test_val, oth_part_dict_test_val, w_oth_num_test_val, w_oth_num_T1_test_val, w_oth_num_T2_test_val = criterion(
                        labels, logits_fine1, logits_coarse1, logits_cnfd, domain_predicted, out_feat_btnk_copy,
                        1, cond_w_all, w_weight=w_weight, b_update_centroids=False, b_eva=True)

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

            criterion.init_loss_and_oth()
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
