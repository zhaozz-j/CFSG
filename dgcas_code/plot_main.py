# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2021- 10- 19
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions:
Input:
Output:
Note:

Link:

*********************************************************************************
********************************Fighting! GO GO GO*******************************
*************************************Import***********************************"""

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path, PurePath
import numpy as np
import os
import logging
import time
import errno
import argparse
import datetime
import json
import numpy as np
import csv
import sys
from util.misc import columns_csv_all, columns_csv_bd

from util.box_ops import box_cxcywh_to_xyxy
from util.plot_utils import plot_logs, plot_log
from util.plot_utils import plot_precision_recall, plot_APs

"""*********************************Import************************************"""
"""********************************Variable***********************************"""
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

"""********************************Variable***********************************"""
"""***************************************************************************"""
'''***************************************************************************'''

torch.set_printoptions(profile="full", sci_mode=False)
now = int(time.time())
timeArray = time.localtime(now)
Time = time.strftime("%Y%m%d_%H%M", timeArray)


def get_args_parser():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--other', default='')
    parser.add_argument('--force_re_process_all', default=True, type=bool)
    parser.add_argument('--force_re_process_exl', default=False, type=bool)
    parser.add_argument('--b_force_del_jpg', default=False, type=bool)
    parser.add_argument('--b_force_del_pth', default=False, type=bool)

    parser.add_argument('--only_force_re_process_exl', '-onex', type=str2bool, default=False)
    parser.add_argument('--max_iteration', type=float, default=20000, help="0 for 不处理，其他数都截断df")

    parser.add_argument('--data_path', default='')

    parser.add_argument('--main_resume_root', '-mres', help='优先级第一，需要先清空才能用别的',
        default='',
        # default='',
                        )

    parser.add_argument('--resume_single', '-res', help='优先级第二，需要先清空才能用别的',
        # default='',
        default='/mnt/home/yuwenlong/works/causal/dgcas/dgcas_code/output/dg-2048-new/cp2/rn50/dgtral_b32lr0.03g4_para_convT1fn128fd2048_ot0.05_ssdgc1.0_ssdgp1.0_dssgc0.5_dssgp1.0_dsdgc0.05_dsdgpFs_cfnoFs_otpFs_db-L234561.1_02271515/dgtral_b32lr0.03g4_para_convT1fn128fd2048_ot0.05_ssdgc1.0_ssdgp1.0_dssgc0.5_dssgp1.0_dsdgc0.05_dsdgpFs_cfnoFs_otpFs_db-L234561.1_02271515_p2c_2/',
                        )

    parser.add_argument('--resumelist', help='优先级第三，这是个list，需要先清空才能用别的', default=[
        '',
    ])
    parser.add_argument('--resume', help='用来读取checkpoint信息，通常是属于调试模式的',
        default='',
                        )

    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default='')

    parser.add_argument('--fields', default='voc')
    parser.add_argument('--b_use_batch_process', default=False)

    parser.add_argument('--labels', default=('class_error', 'loss_bbox_unscaled', 'mAP'))
    parser.add_argument('--plot_type', default=2, type=int)
    return parser


#该函数 main_plot 主要用于处理和绘制训练日志数据
def main_plot(args, txt_input='', b_resume=False, b_from_code=False, other_plot_type=1, eval=False, prefix='', logger=None):
    if b_resume is True:
        if args.resume:  # 通常是属于调试模式的
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')

            args = checkpoint['args']
            print(args)
            print('over')
            return

    if txt_input == '':
        if args.data_path:
            txt_dir = args.data_path
        else:
            txt_dir = os.path.join(args.document_root, 'data2process')
    else:
        txt_dir = txt_input

    if txt_dir.find('.txt') == -1:  # 不是以.txt直接输入，而是以文件形式输入
        output_dir = txt_dir
        txt_base_name = os.path.basename(txt_dir.rstrip('/'))

        b_txt_base_name = os.path.exists(Path(os.path.join(output_dir, f'{txt_base_name}.txt')))
        if b_txt_base_name is not True:
            douc_txt_list = os.listdir(output_dir)
            for i_douc_txt in douc_txt_list:
                if '.txt' in i_douc_txt:
                    txt_base_name = i_douc_txt[:-4]
                    b_txt_base_name = True
                else:
                    pass
        assert b_txt_base_name is True, f'txt not in this file{txt_dir}, please input the right file path.'

        if prefix != '':
            txt_base_name = f'[{prefix}]_{txt_base_name}'

        txt_file_name = f'{txt_base_name}.txt'
        csv_file_name = f'{txt_base_name}.csv'
        txt_file_path = Path(os.path.join(output_dir, txt_file_name))
        csv_file_path = Path(os.path.join(output_dir, csv_file_name))

    else:  # 输入文件直接以.txt文件输入
        output_dir = os.path.dirname(txt_dir)

        txt_base_name_ori = os.path.basename(txt_dir.rstrip('/'))[:-4]
        if prefix != '':
            txt_base_name = f'[{prefix}]_{txt_base_name_ori}'

        csv_file_name = '{}.csv'.format(txt_base_name)
        txt_file_name = '{}.txt'.format(txt_base_name_ori)
        txt_file_path = txt_dir
        csv_file_path = Path(os.path.join(Path(output_dir), csv_file_name))

    if b_from_code is False:  # 是直接单独执行这个函数
        plot_type = args.plot_type

    else:  # 从别处调用该函数
        plot_type = other_plot_type

    figure_title = txt_base_name

    # ----------------------------------------------------------------------------------------------------------- #
    #################################################  figure fields  #############################################
    # ----------------------------------------------------------------------------------------------------------- #

    figure_fields_Main = ['acc_ave', 'acc_0', 'acc_1', 'acc_2', 'acc_3', 'consist', 'total_loss']
    figure_fields_oth = ['oth_fp', 'oth_0', 'oth_1', 'oth_2', 'oth_3', 'cnfd_loss']
    figure_fields_oth_T1 = ['oth_fp_T1', 'oth_T1_0', 'oth_T1_1', 'oth_T1_2', 'oth_T1_3']
    figure_fields_oth_Tall = ['total_loss', 'oth_fp_T1', 'oth_T1_0', 'oth_T1_1', 'oth_T1_2', 'oth_T1_3',
                              'oth_fp_T2', 'oth_T2_0', 'oth_T2_1', 'oth_T2_2', 'oth_T2_3',
                              'oth_grad_fp', 'oth_grad_fp_T1', 'oth_grad_fp_T2',
                              'oth_grad_0', 'oth_grad_T1_0', 'oth_grad_T2_0',
                              'oth_grad_1', 'oth_grad_T1_1', 'oth_grad_T2_1']
    figure_fields_oth_T2 = ['oth_fp_T2', 'oth_T2_0', 'oth_T2_1', 'oth_T2_2', 'oth_T2_3']
    figure_fields_cond_rank = ['cond_g0', 'cond_g1', 'cond_g2', 'cond_g3', 'rank_g0', 'rank_g1', 'rank_g2', 'rank_g3']
    figure_fields_oth_part = ['oth_fp_part', 'oth_3_part', 'oth_2_part', 'oth_1_part', 'oth_0_part', 'cnfd_loss_part']
    figure_fields_w_oth = ['oth_w', 'oth_w_T1', 'oth_w_T2', 'oth_w_1', 'oth_w_T1_1', 'oth_w_T2_1', ]

    figure_fields_loss = ['total_loss', 'fine_classifier_loss', 'coarse_classifier_loss', 'classifier_loss',
                          'entropy_loss_source', 'entropy_loss_target', 'transfer_loss', 'cndg_loss']
    figure_fields_loss_granu = ['loss_feat_oth', 'loss_same_sa_diff_g_com', 'loss_same_sa_diff_g_pvt',
                                'loss_diff_sa_same_g_com', 'loss_diff_sa_same_g_pvt',
                                'loss_diff_sa_diff_g_com', 'loss_diff_sa_diff_g_pvt', 'loss_singular']

    figure_fields_cos_segloss = ['cossim_bkb', 'cossim_joint']

    # ----------------------------------------------------------------------------------------------------------- #
    #################################################  figure fields  #############################################
    # ----------------------------------------------------------------------------------------------------------- #

    df = pd.read_json(txt_file_path, lines=True)
    pd.read_json(txt_file_path, lines=True).to_csv(csv_file_path)

    index_all = df.axes[1].tolist()  # 获取txt里面存储的值

    ACC_RESULTS, ACC_RESULTS_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['Acc'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_Main,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS, LOSS_RESULTS_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['Loss'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_loss,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_g, LOSS_RESULTS_g_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['Loss_granu'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_loss_granu,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_oth, LOSS_RESULTS_oth_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['oth'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_oth,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_w_oth, LOSS_RESULTS_w_oth_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['w_oth'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_w_oth,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_oth_T1, LOSS_RESULTS_oth_T1_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['oth_T1'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_oth_T1,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_oth_T2, LOSS_RESULTS_oth_T2_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['oth_T2'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_oth_T2,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_oth_T12, LOSS_RESULTS_oth_T12_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['oth_T12'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_oth_Tall,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_oth_part, LOSS_RESULTS_oth_mAPemax_part = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['oth_part'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_oth,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)
    LOSS_RESULTS_condrank, LOSS_RESULTS_condrank_mAPemax = plot_log(log_file=Path(output_dir),  # txt文件所在文件夹
                                                        log_name_txt=txt_file_name,  # txt文件名字（注意不是路径，而是直接名字）
                                                        label=['cond_rank'],  # 要输出的图片命名为 labels_title
                                                        title=figure_title,
                                                        fields=figure_fields_cond_rank,
                                                        log_output_path=output_dir,
                                                        modes={},
                                                        print_in_sigle=True,
                                                        ewm_col=0, max_iter=args.max_iteration)


    # ACC_RESULTS = plot_logs([Path(log_dir)], labels=['acc'], print_in_sigle=True, title=other_info,
    #                         fields=figure_fields_Main, ewm_col=0,
    #                         log_name=os.path.basename(log_dir.rstrip('/')),
    #                         log_output_path=output_dir, modes=tuple(['train', 'test']))
    #
    # LOSS_RESULTS = plot_logs([Path(log_dir)], labels=['Loss'], print_in_sigle=True, title=other_info,
    #                          fields=figure_fields_loss, ewm_col=0,
    #                          log_name=os.path.basename(log_dir.rstrip('/')),
    #                          log_output_path=output_dir, modes=tuple(['train', 'test']))
    # LOSS_RESULTS_g = plot_logs([Path(log_dir)], labels=['Loss_granu'], print_in_sigle=True, title=other_info,
    #                            fields=figure_fields_loss_granu, ewm_col=0,
    #                            log_name=os.path.basename(log_dir.rstrip('/')),
    #                            log_output_path=output_dir, modes=tuple(['train', 'test']))
    #
    # LOSS_RESULTS_oth = plot_logs([Path(log_dir)], labels=['oth'], print_in_sigle=True, title=other_info,
    #                              fields=figure_fields_oth, ewm_col=0,
    #                              log_name=os.path.basename(log_dir.rstrip('/')),
    #                              log_output_path=output_dir, modes=tuple(['train', 'test']))

    # Results = {**{k: v for k, v in ACC_RESULTS.items()},
    #            **{k: v for k, v in LOSS_RESULTS.items()},
    #            **{k: v for k, v in LOSS_RESULTS_g.items()},
    #            }
    Results = {**{k: v for k, v in ACC_RESULTS.items()},
               **{k: v for k, v in LOSS_RESULTS.items()},
               **{k: v for k, v in LOSS_RESULTS_g.items()},
               **{k: v for k, v in LOSS_RESULTS_oth.items()},
               **{k: v for k, v in LOSS_RESULTS_oth_T1.items()},
               **{k: v for k, v in LOSS_RESULTS_oth_T2.items()},
               **{k: v for k, v in LOSS_RESULTS_oth_part.items()},
               }
    Results_accavemax = {**{k: v for k, v in ACC_RESULTS_mAPemax.items()},
                       **{k: v for k, v in LOSS_RESULTS_mAPemax.items()},
                       **{k: v for k, v in LOSS_RESULTS_g_mAPemax.items()},
                       **{k: v for k, v in LOSS_RESULTS_oth_mAPemax.items()},
                       **{k: v for k, v in LOSS_RESULTS_oth_T1_mAPemax.items()},
                       **{k: v for k, v in LOSS_RESULTS_oth_T2_mAPemax.items()},
                       **{k: v for k, v in LOSS_RESULTS_oth_mAPemax_part.items()},
                       }

    results_name = [[k for k, v in ACC_RESULTS.items()]]
    # results_value = [[v * 100 for k, v in ACC_RESULTS.items()]]

    results_value = [[v * 100 if 'acc' in k else v for k, v in ACC_RESULTS.items()]]
    results_value_accavemax = [[v * 100 if k.count('acc') > 1 else v for k, v in ACC_RESULTS_mAPemax.items()]]

    csv_results = {'value_name': ['every_max', 'accave_max']}
    for i_key, i_value in ACC_RESULTS.items():
        if i_key != 'iter_num':
            csv_results[i_key] = [i_value * 100]
            csv_results[i_key].append(ACC_RESULTS_mAPemax[f'{i_key}_accavemax'] * 100)
        else:
            csv_results[i_key] = [i_value, i_value]

    b = pd.DataFrame(csv_results, index=['every_max', 'accave_max']).reindex(columns=columns_csv_all)
    b.to_csv(csv_file_path, mode='a+', encoding='utf-8')

    log_file_log = '{}.log'.format(txt_base_name)
    with Path(os.path.join(output_dir, log_file_log)).open("a") as f:
        f.write('\n\nTraining Finished. Results:\n')
        for k, v in Results.items():
            f.write('{}:{}\n'.format(k, v))
        for k, v in Results_accavemax.items():
            f.write('{}:{}\n'.format(k, v))

    if logger is not None:
        logger.info('\n\nbest acc: {}\n'.format(results_value))

    print('plotting finished')
    return csv_results


def reprocess_csv(all_domain_csv_dir):

    store_num = 6
    # all_domain_csv_dir = '/home/yuwenlong/works/causal/dgcas/dgcas_code/output/para_conv1_adjlr/cp2/rn50/' \
    #                      'dgtral_b36lr0.01_seri_conv_T1_fn128fd256_othFs_ssdgcFs_ssdgpFs_dssgcFs_dssgpFs_dsdgcFs_dsdgpFs_L0_conv1_adj_lr_10122223/' \
    #                      'dgtral_b36lr0.01_seri_conv_T1_fn128fd256_othFs_ssdgcFs_ssdgpFs_dssgcFs_dssgpFs_dsdgcFs_dsdgpFs_L0_conv1_adj_lr_10122223.csv'
    # #
    all_domain_csv_dir_sta = all_domain_csv_dir.rstrip('.csv') + '_statis.csv'

    pd_max = pd.DataFrame()
    pd_avemax = pd.DataFrame()

    all_domain_name = []
    try:
        all_results_pd = pd.read_csv(all_domain_csv_dir)
    except:
        print('error @ reprocess_csv.{pd.read_csv(all_domain_csv_dir)}')
        return
    all_results_pd.rename(columns={'Unnamed: 0': 'domain_name'}, inplace=True)
    all_results_dict = all_results_pd.to_dict('records')

    for i_e, i_dict in enumerate(all_results_dict):
        for j_e, (j_key, j_value) in enumerate(i_dict.items()):
            if j_key == 'domain_name':
                all_domain_name.append(j_value.split('_')[0])

    set_domain_name = set(all_domain_name)
    num_domain = len(set_domain_name)

    domain_all_max = {i_domain_name: [] for i_domain_name in set(all_domain_name)}
    domain_all_avemax = {i_domain_name: [] for i_domain_name in set(all_domain_name)}

    # domain_all_all_max = []
    all_all_max_pd = pd.DataFrame()
    # domain_all_all_avemax = []
    all_all_avemax_pd = pd.DataFrame()

    for i_e, i_dict in enumerate(all_results_dict):
        if i_dict['value_name'] == 'every_max':
            domain_all_max[i_dict['domain_name'].split('_')[0]].append(i_dict)
        elif i_dict['value_name'] == 'accave_max':
            domain_all_avemax[i_dict['domain_name'].split('_')[0]].append(i_dict)
    all_all_name = []

    if len(set_domain_name) > 2:
        store_num = 3
        # columns_csv_domain = columns_csv
        # columns_csv_all = columns_csv_bd
        columns_csv = columns_csv_bd
    else:
        columns_csv = columns_csv_all

    iii = 0
    for i_domain_name in sorted(set_domain_name):
        iii += 1
        pd_domain_max = pd.DataFrame()
        pd_domain_avemax = pd.DataFrame()
        i_domain_dictlist_max = domain_all_max[i_domain_name]

        for i_domain_dict_max in i_domain_dictlist_max:
            i_domain_pd_max = pd.DataFrame(i_domain_dict_max, index=[i_domain_name]).reindex(columns=columns_csv)
            pd_domain_max = pd.concat([pd_domain_max, i_domain_pd_max], axis=0)
            pd_max = pd.concat([pd_max, i_domain_pd_max], axis=0)

        i_domain_dictlist_avemax = domain_all_avemax[i_domain_name]
        for i_domain_dict_avemax in i_domain_dictlist_avemax:
            i_domain_pd_avemax = pd.DataFrame(i_domain_dict_avemax, index=[i_domain_name]).reindex(
                columns=columns_csv)
            pd_domain_avemax = pd.concat([pd_domain_avemax, i_domain_pd_avemax], axis=0)
            pd_avemax = pd.concat([pd_avemax, i_domain_pd_avemax], axis=0)

        index_all = pd_domain_max.axes[1].tolist()  # 获取txt里面存储的值
        indexi_all = pd_domain_max.axes[0].tolist()  # 获取txt里面存储的值
        for index in index_all:
            # xxx = [[]]
            # xxx[0].append(pd_domain_max.loc[indexi_all[0], index])
            # element_ii = pd.DataFrame(xxx[0].append(pd_domain_max.loc[indexi_all[0], index]))
            # if isinstance(element_ii.iloc[0, 0], str):
            # if isinstance(pd_domain_max.loc[indexi_all[0], index], str):
            if index == 'value_name':
                del pd_domain_max[index]
                del pd_domain_avemax[index]
                del pd_max[index]
                del pd_avemax[index]

        domain_mean_max_dict = pd_domain_max.mean().to_dict()
        domain_mean_avemax_dict = pd_domain_avemax.mean().to_dict()
        domain_mean_max_dict['value_name'] = 'every_max'
        domain_mean_avemax_dict['value_name'] = 'accave_max'

        # domain_all_all_max.append(domain_mean_max_dict)
        # domain_all_all_avemax.append(domain_mean_avemax_dict)

        domain_mean_max_pd = pd.DataFrame(domain_mean_max_dict, index=[f'{i_domain_name}_domain_ave']).reindex(
            columns=columns_csv)
        domain_mean_avemax_pd = pd.DataFrame(domain_mean_avemax_dict,
                                             index=[f'{i_domain_name}_domain_ave']).reindex(columns=columns_csv)

        all_all_name.extend([f'{i_domain_name}_{columns_csv[i_name]}' for i_name in range(1, store_num + 1)])
        all_all_max_pd = pd.concat([all_all_max_pd, pd.DataFrame(
            domain_mean_max_pd.drop('value_name', axis=1).iloc[:, :store_num].values)], axis=1)
        all_all_avemax_pd = pd.concat([all_all_avemax_pd, pd.DataFrame(
            domain_mean_avemax_pd.drop('value_name', axis=1).iloc[:, :store_num].values)], axis=1,
                                      ignore_index=True)

        if iii == 1:
            domain_mean_max_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=True)
        else:
            domain_mean_max_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=False)
        domain_mean_avemax_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=False)

    all_ave_max = pd_max.mean().to_dict()
    all_ave_avemax = pd_avemax.mean().to_dict()
    all_ave_max['value_name'] = 'every_max'
    all_ave_avemax['value_name'] = 'accave_max'
    all_mean_max_pd = pd.DataFrame(all_ave_max, index=['all_ave']).reindex(columns=columns_csv)
    all_mean_avemax_pd = pd.DataFrame(all_ave_avemax, index=['all_ave']).reindex(columns=columns_csv)
    all_mean_max_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=False)
    all_mean_avemax_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=False)

    all_all_name.extend([f'all_{columns_csv[i_name]}' for i_name in range(1, store_num + 1)])
    all_all_max_pd = pd.concat(
        [all_all_max_pd, pd.DataFrame(all_mean_max_pd.drop('value_name', axis=1).iloc[:, :store_num].values)],
        axis=1, ignore_index=True)
    all_all_max_pd.columns = all_all_name
    all_all_max_pd.insert(0, 'value_name', ['every_max'])

    all_all_avemax_pd = pd.concat(
        [all_all_avemax_pd, pd.DataFrame(all_mean_avemax_pd.drop('value_name', axis=1).iloc[:, :store_num].values)],
        axis=1, ignore_index=True)
    all_all_avemax_pd.columns = all_all_name
    all_all_avemax_pd.insert(0, 'value_name', ['accave_max'])

    all_all_max_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=True)
    all_all_avemax_pd.to_csv(all_domain_csv_dir_sta, mode='a+', encoding='utf-8', header=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data processing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main_resume_root = args.main_resume_root

    if args.resume != '' and args.resume is not None:
        args.data_path = args.resume
        main_plot(args, b_resume=True)
        print('finished \n{}\n\n'.format(args.resume))
        sys.exit()

    resume_list = []
    if main_resume_root != '' and main_resume_root is not None:
        resume_list = os.listdir(main_resume_root)
        resume_list = [os.path.join(main_resume_root, i_resume_list) for i_resume_list in resume_list]

    elif args.resume_single != '' and args.resume_single is not None:
        resume_list.append(args.resume_single)

    elif len(args.resume_list) != 0:
        resume_list = args.resume_list

    elif args.resume != '':
        args.data_path = args.resume
        main_plot(args)
        print('finished {}\n\n\n'.format(args.resume))

    else:
        raise ValueError('no resume type set')

    for i_file in resume_list:

        if i_file[-1] == '/':
            i_file = i_file[:-1]
        xxx = i_file.split('/')[-1]
        all_domain_csv_dir = '{}/{}.csv'.format(i_file, i_file.split('/')[-1])
        if args.only_force_re_process_exl is True:
            try:
                print(xxx)
                reprocess_csv(all_domain_csv_dir)
                continue
            except:
                print('error')
                continue

        if '.txt' in i_file:
            main_plot(args, txt_input=i_file, b_from_code=True, prefix='test_clipart')
        elif 'pth' in i_file:
            pass
        elif os.path.isdir(i_file):

            file_list = os.listdir(i_file)
            i_iii = 0
            for i_resume in file_list:
                if os.path.isdir(os.path.join(i_file, i_resume)):
                    douc_path = os.path.join(i_file, i_resume)
                    douc_list = os.listdir(douc_path)
                    has_been_processed = False
                    has_csv_result = False
                    has_txt = False

                    for douc in douc_list:
                        if args.force_re_process_all is True:
                            has_been_processed = False
                            has_no_result = True  # 出现csv意味着原始代码已经完成，可以运行这个画图操作了
                            has_csv_result = True

                        elif '.jpg' in douc and args.force_re_process_all is False:
                            has_been_processed = True
                            break

                        elif '.csv' in douc and args.force_re_process_all is False or args.force_re_process_exl is True:
                            has_csv_result = True  # 出现csv意味着原始代码已经完成，可以运行这个画图操作了

                        if '.txt' in douc:
                            has_txt = True

                        if args.b_force_del_jpg is True:
                            if '.jpg' in douc:
                                os.remove(os.path.join(douc_path, douc))
                        if args.b_force_del_pth is True:
                            if '.pth' in douc:
                                os.remove(os.path.join(douc_path, douc))

                    if douc_path != '' and has_been_processed is False and has_txt is True and has_csv_result is True:

                        args.data_path = douc_path
                        name_split = i_resume.split('_')
                        i_time = name_split[-1]
                        args.source = list(name_split[-2])[0]
                        args.target = list(name_split[-2])[-1]
                        xxx = i_file.split('/')[-1]
                        all_domain_csv_dir = '{}/{}{}.csv'.format(i_file, i_file.split('/')[-1], int(args.max_iteration))
                        print('\nnow {}'.format(douc_path))
                        result_value_i = main_plot(args)

                        index_i = [f'{args.source}2{args.target}_{i_time}', f'{args.source}2{args.target}_{i_time}']
                        pd1 = pd.DataFrame(result_value_i, index=index_i).reindex(columns=columns_csv_all)
                        if i_iii == 0:
                            pd1.to_csv(all_domain_csv_dir, mode='w', encoding='utf-8')
                        else:
                            pd1.to_csv(all_domain_csv_dir, mode='a+', encoding='utf-8', header=False)
                        i_iii += 1
                        print('finished\n\n')

                    elif has_been_processed == True:
                        print('has been processed : \n{}\n\n'.format(douc_path))

                    elif has_csv_result == True:
                        print('has no result or the main code has not finish, please wait : \n{}\n\n'.format(douc_path))

                    elif has_txt == False:
                        print('There has no txt file : \n{}\n\n'.format(douc_path))

                    elif has_no_result == False:
                        print('There has no csv file : \n{}\n\n'.format(douc_path))
                else:
                    pass
            # try:
            reprocess_csv(all_domain_csv_dir)
            #     continue
            # except:
            #     print('error')
            #     continue

