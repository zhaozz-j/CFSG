import os
import errno
import torch
import time
import logging
import shutil
import sys
from util.misc import compu_featpart, cate_num_all_dataset, num_coarse_cate_dataset
def basicset_logger(args, i_time, all_time, loggerpredix=''):
    args.document_root = os.getcwd()
    if args.b_part_oth is False:
        args.c_part_oth = 'Fs'
    if args.b_feat_oth is False:
        args.c_feat_oth = 'Fs'
    if args.b_w_oth is False:
        args.c_w_oth = 'Fs'
    if args.b_singular_dist is False:
        args.c_singular_dist = 'Fs'
    if args.b_same_sa_diff_g_com is False:
        args.c_same_sa_diff_g_com = 'Fs'
    if args.b_same_sa_diff_g_pvt is False:
        args.c_same_sa_diff_g_pvt = 'Fs'
    if args.b_diff_sa_same_g_com is False:
        args.c_diff_sa_same_g_com = 'Fs'
    if args.b_diff_sa_same_g_pvt is False:
        args.c_diff_sa_same_g_pvt = 'Fs'
    if args.b_diff_sa_diff_g_com is False:
        args.c_diff_sa_diff_g_com = 'Fs'
    if args.b_diff_sa_diff_g_pvt is False:
        args.c_diff_sa_diff_g_pvt = 'Fs'
    if args.cnfd_type == 'no':
        args.c_cnfd = 'Fs'

    if args.granu_num is None:
        args.granu_num = num_coarse_cate_dataset[args.dataset] + 1
    args.g = args.granu_num
    args.gc = args.granu_num - 1
    if args.resume is not None:
        args.train_type = 'viz'
        args.output_root = args.resume
        args.output_dir = args.resume
        args.output_path = args.output_dir
        args.log_dir = args.output_dir
        base_docu_name = args.log_file_name = os.path.basename(args.resume.rstrip('/'))
    else:
        args.resume = None
        base_docu_name = '{}{}_b{}lr{}g{}_{}{}fn{}fd{}_ot{}_wot{}_sd{}_ssdgc{}_ssdgp{}_dssgc{}_dssgp{}_dsdgc{}_dsdgp{}_cf{}{}_otp{}_{}' \
                .format(args.da, args.train_type, args.batchsize, args.lr, args.g,
                        args.model_type, args.norm_type,
                        args.feat_num, args.feat_dim,
                        args.c_feat_oth, args.c_w_oth, args.c_singular_dist,
                        args.c_same_sa_diff_g_com, args.c_same_sa_diff_g_pvt,
                        args.c_diff_sa_same_g_com, args.c_diff_sa_same_g_pvt,
                        args.c_diff_sa_diff_g_com, args.c_diff_sa_diff_g_pvt,
                        args.cnfd_type, args.c_cnfd, args.c_part_oth,
                        args.other)

        if args.train_type == 'tral':
            print('In to all train configuration')
            args.log_file_name_1 = '{}_{}'.format(base_docu_name, args.time_all_start)
        else:
            args.time_all_start = time.strftime('%m%d%H%M')
            print('into single domain train configuration')
            args.log_file_name_1 = '{}_{}2{}_{}'.format(base_docu_name, args.source, args.target, args.time_all_start)

        if args.gpu_location == 'liu90':
            args.output_root = os.path.join('/data/wzj/result/dg/', 'output')
        elif args.gpu_location == 'autodl':
            args.output_root = os.path.join('/root/autodl-tmp/', 'output')
        elif args.gpu_location == 'hy':
            args.output_root = os.path.join('/hy-tmp/', 'output')
        else:
            args.output_root = os.path.join(args.document_root, 'output')

        args.domain_name = args.source + '2' + args.target

        if args.out_forder_type == 'o':
            if len(args.out_forder_name) == 0 or args.out_forder_name is None:
                pass
            else:
                args.output_root = os.path.join(args.output_root, args.out_forder_name)

        # 判断数据集
        if args.dataset == 'cp2' or args.dataset == 'cars':
            if args.other == 'debug':
                args.output_path = os.path.join(args.output_root, 'debug')
                args.max_iteration = 4
                args.test_interval = 2
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
            else:
                args.output_path = os.path.join(args.output_root, args.dataset)
                args.writer_name = '{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)


        elif args.dataset == 'ci2':
            if args.other == 'debug':
                args.output_path = os.path.join(args.output_root, 'debug')
                args.max_iteration = 4
                args.test_interval = 2
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
            else:
                args.output_path = os.path.join(args.output_root, args.dataset)
                args.writer_name = '{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
        elif args.dataset == 'cn2':
            if args.other == 'debug':
                args.output_path = os.path.join(args.output_root, 'debug')
                args.max_iteration = 4
                args.test_interval = 2
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
            else:
                args.output_path = os.path.join(args.output_root, args.dataset)
                args.writer_name = '{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
        elif args.dataset == 'in2':
            if args.other == 'debug':
                args.output_path = os.path.join(args.output_root, 'debug')
                args.max_iteration = 4
                args.test_interval = 2
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
            else:
                args.output_path = os.path.join(args.output_root, args.dataset)
                args.writer_name = '{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)



        elif args.dataset == 'bd':
            if args.other == 'debug':
                args.output_path = os.path.join(args.output_root, 'debug')
                args.max_iteration = 4
                args.test_interval = 2
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
            else:
                args.output_path = os.path.join(args.output_root, args.dataset)
                args.writer_name = '{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
        else:
            if args.other == 'debug':
                args.output_path = os.path.join(args.output_root, 'debug')
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)
            else:
                args.output_path = os.path.join(args.output_root, 'coco')
                args.writer_name = 'debug_{}_{}_{}'.format(args.dataset, args.backbone, args.time_all_start)

        args.output_path = os.path.join(args.output_path, args.backbone)

        if args.out_forder_type == 'n':
            if len(args.out_forder_name) == 0 or args.out_forder_name is None:
                args.output_dir = os.path.join(args.output_dir, 'allothers')
            else:
                args.output_dir = os.path.join(args.output_path, args.out_forder_name)
        else:
            args.output_dir = args.output_path

        args.output_path = os.path.join(args.output_dir, args.log_file_name_1)

        if args.train_type == 'tral':
            args.log_file_name = args.log_file_name_1 + '_' + args.domain_name + '_' + str(i_time)
            args.output_dir = os.path.join(args.output_path, args.log_file_name)
        else:
            args.log_file_name = args.log_file_name_1 + '_' + str(i_time)
            args.output_dir = args.output_path

        #函数用于创建目录，如果目录已存在则不会抛出异常
        os.makedirs(args.output_dir, exist_ok=True)

        args.log_dir = args.output_dir
        os.makedirs(args.log_dir, exist_ok=True)

        if all_time == 1:
            config_dir = os.path.join(args.document_root, 'config.yaml')
            train_doc_dir = os.path.join(args.document_root, 'train.py')
            model_doc_dir = os.path.join(args.document_root, 'models/model.py')
            crit_doc_dir = os.path.join(args.document_root, 'models/criterion.py')
            shutil.copy(config_dir, args.output_path + '/copy_config.py')
            shutil.copy(train_doc_dir, args.output_path + '/copy_train.py')
            shutil.copy(model_doc_dir, args.output_path + '/copy_model.py')
            shutil.copy(crit_doc_dir, args.output_path + '/copy_criterion.py')

            if 'vit' in args.backbone:
                vit_doc_dir = os.path.join(args.document_root, 'models/vit.py')
                shutil.copy(vit_doc_dir, args.output_path + '/copy_vit.py')
            if 'mlp' in args.backbone:
                mlp_doc_dir = os.path.join(args.document_root, 'models/asmlp.py')
                shutil.copy(mlp_doc_dir, args.output_path + '/copy_mlp.py')


    if loggerpredix != '':
        log_file_log = f'{args.log_file_name}_{loggerpredix}.log'
    else:
        log_file_log = f'{args.log_file_name}.log'
    logger = logging.getLogger(__name__)

    logger.handlers = []
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.log_dir, log_file_log))
    handler.setLevel(logging.DEBUG)
    han_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(han_formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    con_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(con_formatter)
    logger.addHandler(console)
    args.log_file_log = os.path.join(args.log_dir, log_file_log)
    return logger
def log_init_config(args, model, all_times):

    args.logger.info('------------------------------------------------------------------------------------------------')
    args.logger.info('This experiment is to {}. \n'.format(args.goal))
    args.logger.info('************************************************************************************************')
    # args.logger.info("git:{}\n".format(utils.get_sha()))
    args.logger.info('output_file_folder = {}'.format(args.output_dir))
    args.logger.info('log_file_name = {}\n'.format(args.log_file_name))
    if args.granu_num is None:
        args.granu_num = num_coarse_cate_dataset[args.dataset] + 1

    args.num_g = args.granu_num
    args.ele_g = list(range(args.num_g))
    args.logger.info('tortal number of granu. = {}\n'.format(args.num_g))
    args.logger.info('number of coarse granu. = {}\n'.format(args.granu_num - 1))
    args.logger.info('list of granu. = {}\n'.format(args.ele_g))

    args.logger.info('**********************************************************************************************')

    n_parameters = {}
    if args.da == 'da':
        n_parameters['discrimi'] = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)

    grad_layers = []
    for i_name, i_module in model.named_children():
        if i_name == 'btnk_layer':
            for j_name, j_module in i_module.named_children():
                n_parameters[f'btnk_{j_name}'] = sum(p.numel() for p in j_module.parameters() if p.requires_grad)
        elif i_name == 'clsf_layer':
            for j_name, j_module in i_module.named_children():
                n_parameters[f'clsf_{j_name}'] = sum(p.numel() for p in j_module.parameters() if p.requires_grad)
        elif i_name == 'dis_lin':
            for j_name, j_module in i_module.named_children():
                n_parameters[f'dist_lin_{j_name}'] = sum(p.numel() for p in j_module.parameters() if p.requires_grad)
        elif i_name == 'input_proj':
            for j_name, j_module in i_module.named_children():
                n_parameters[f'input_proj_{j_name}'] = sum(p.numel() for p in j_module.parameters() if p.requires_grad)
        elif i_name == 'backbone_c':
            n_parameters[f'bkb_c'] = sum(p.numel() for p in i_module.parameters() if p.requires_grad)
        elif i_name == 'backbone_f':
            n_parameters[f'bkb_f'] = sum(p.numel() for p in i_module.parameters() if p.requires_grad)
    n_parameters['all'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for item in n_parameters:
        args.logger.info('{} number of params: {}M'.format(item, n_parameters[item] / 1000000))
    args.logger.info('all number of params: {}M\n'.format(n_parameters['all'] / 1000000))

    args.logger.info('**********************************************************************************************')
    args.info = '\n other = {}\n' \
                'backbone = {}\n' \
                'number of params = {}\n' \
                'dataset = {}\n' \
                'bn_use_oth_loss = {}\n' \
                'feature number     = {}\n' \
                'feature dimension  = {}\n' \
                'feature ratio      = {}\n' \
                'c_loss_feature_cos = {}\n\n' \
                'model_type = {}   {}\n' \
                'cnfd_type = {}  {}\n' \
                'norm_type = {}\n\n' \
                'b_da or dg = {}\n' \
                'b_pan = {}\n' \
                'b_cfdg = {}\n' \
                'b_bkb_poor = {}\n' \
                'bn_input_p = {}\n' \
                'str_avgpool = {}\n' \
                'b_bkb_c = {}\n' \
                'bn_align = {}\n' \
                'b_2loss = {}\n' \
                'lr = {}\n' \
                'feat_ratio = {}\n' \
                'b2_stage = {}\n' \
                'l23_stage = {}\n' \
                'sim_method = {}\n' \
                'c_singular_dist = {}\n' \
        .format(args.other, args.backbone, n_parameters['all']/1000000, args.dataset, args.b_feat_oth,
                args.feat_num, args.feat_dim, args.feat_ratio, args.c_feat_oth, args.model_type, args.conv_type,
                args.cnfd_type, args.c_cnfd, args.norm_type, args.da,
                args.b_pan, args.bn_cfdg, args.b_bkb_poor, args.bn_input_p, args.str_avgpool,
                args.b_bkb_c, args.bn_align,
                args.b_2loss, args.lr, args.feat_ratio, args.b2_stage, args.l23_stage,
                args.sim_method, args.c_singular_dist,)
    args.logger.info('Experiments info: \n {}'.format(args.info))

    args.logger.info('************************************************************************************************')
    args.logger.info('************************************************************************************************')

    args_all = '*****************************************Training args:**********************************************\n'
    for k, v in sorted(vars(args).items()):
        args_all += str(k) + '=' + str(v) + '\n'
    args.logger.info(args_all)

    args.logger.info('************************************************************************************************')
    device = torch.device(args.device)
    if all_times == 1:
        acc = torch.zeros((4, args.h, args.h, args.h)).to(device)
        print('ok')
    args.logger.info('************************************************************************************************')
    args.logger.info('model:\n')
    args.logger.info(model)
    args.logger.info('************************************************************************************************')
    args.logger.info('all number of params: {}M\n'.format(n_parameters['all'] / 1000000))
    args.logger.info(f"Command: {' '.join(sys.argv)}")
    args.logger.info('************************************* Training Start *******************************************')
    args.logger.info('********************************************************************************************\n\n')
