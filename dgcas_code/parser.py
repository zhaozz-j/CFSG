import argparse
from util.misc import compu_featpart
def get_args_parser():
    parser = argparse.ArgumentParser(description='CFSG', add_help=False)
    str2bool = lambda x: x.lower() == "true"
    parser.add_argument('--goal', default='xai')
    parser.add_argument('--other', '-ot', default='train')
    parser.add_argument('--out_forder_name', '-ofn', default='xxx', help='输出文件夹名字, output_adjloss_3')
    parser.add_argument('--out_forder_type', '-oft', default='o', help='o: 所有dataset和bkb汇总到一个output下的一个文件夹；'
                                                                       'n：每个dataset和bkb下的汇总')
    parser.add_argument('--da', type=str, default='dg', choices=('da', 'dg', 'trte'))

    parser.add_argument('--model_type', default='para_conv', choices=('para_conv', 'para_mlp', 'seri_conv', 'seri_mlp'))
    parser.add_argument('--conv_type', default='1', choices=('1', '2', '3', '1i', '2i', '3i'))
    parser.add_argument('--granu_num', '-g', default=4, type=int)

    parser.add_argument('--b_pan', type=str2bool, default=False, help="默认是true")
    parser.add_argument('--bn_cfdg', type=str2bool, default=True, help="默认是true")
    parser.add_argument('--b_bkb_poor', type=str2bool, default=False, help="默认是true")
    parser.add_argument('--bn_input_p', type=str2bool, default=False, help="默认是true")
    parser.add_argument('--str_avgpool', type=str, default='avg', choices=('avg', 'none', 'para'))
    parser.add_argument('--b_bkb_c', type=str2bool, default=True, help="默认是false")
    parser.add_argument('--b_cond', type=str2bool, default=False, help="默认是false")
    parser.add_argument('--b_compute_grad', type=str2bool, default=False, help="默认是true")
    parser.add_argument('--b_compute_w', type=str2bool, default=False, help="默认是true")
    parser.add_argument('--b_lamda', type=str2bool, default=True, help="默认是true")
    parser.add_argument('--bn_align', type=str2bool, default=True, help="默认是True")
    parser.add_argument('--b_mgc', action='store_true', help="默认是False")
    parser.add_argument('--b_2loss', type=str2bool, default=False, help="默认是false")
    parser.add_argument('--b_L23jo', type=str2bool, default=False, help="L23loss联合一起优化，如果是false，ijcai版本")
    parser.add_argument('--b_pass_transition', type=str2bool, default=False, help="跳过g transition layer")
    parser.add_argument('--b_onlytrain_f', type=str2bool, default=False, help="只训练f层")
    parser.add_argument('--b_f_ce', type=str2bool, default=False, help="f层训练只用celoss，而不用别的")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--feat_dim', type=int, default=2048, help="")
    parser.add_argument('--feat_num', type=int, default=128, help="")
    parser.add_argument('--feat_ratio', type=str, default='50_30_20')
    parser.add_argument('--sim_method', type=str, default='no', choices=('no', 'hsic', 'oushi', 'ot'))
    parser.add_argument('--b_feat_oth', '-b_fo', default=True, help="默认是False")
    parser.add_argument('--b_w_oth', '-b_wo', action='store_true', help="默认是False")
    parser.add_argument('--b_part_oth', '-b_po', type=str2bool,default='True', help="默认是False")
    parser.add_argument('--b_singular_dist', '-b_sd', default='False', help="默认是False")
    parser.add_argument('--b_same_sa_diff_g_com', '-b_ssdgc', action='store_true', help="默认是False")
    parser.add_argument('--b_same_sa_diff_g_pvt', '-b_ssdgp', action='store_true', help="默认是False")
    parser.add_argument('--b_diff_sa_same_g_com', '-b_dssgc', action='store_true', help="默认是False")
    parser.add_argument('--b_diff_sa_same_g_pvt', '-b_dssgp', action='store_true', help="默认是False")
    parser.add_argument('--b_diff_sa_diff_g_com', '-b_dsdgc', action='store_true', help="默认是False")
    parser.add_argument('--b_diff_sa_diff_g_pvt', '-b_dsdgp', action='store_true', help="默认是False")
    parser.add_argument('--cnfd_type', default='no', choices=('no', 'iso', 'all'),
                        help="isolate: 90%作为causal输出，all：100%作为causal输出。这两种都有confounding loss")
    parser.add_argument('--b_cndg', action='store_true', help="默认是False")
    parser.add_argument('--oth_num', type=int, default=-1, help="oth的数量") #感觉应该是4
    parser.add_argument('--b2_stage', type=str, default='no')
    parser.add_argument('--l23_stage', type=str, default='no')
    parser.add_argument('--b_update_centroids', type=str2bool, default=True)
    parser.add_argument('--c_feat_oth', '-c_fo', type=float, default=0.01,help="同一样本同一粒度所有特征正交化loss 0.001")
    parser.add_argument('--c_w_oth', '-c_wo', type=float, default=0.1, help="同一样本同一粒度所有特征正交化loss 0.001")
    parser.add_argument('--c_part_oth', '-c_po', type=float, default=0.1, help="同一样本同一粒度所有特征正交化loss 0.001")
    parser.add_argument('--c_singular_dist', '-c_sd', type=float, default=0.1, help="同一样本同一粒度所有特征正交化loss 0.001")
    parser.add_argument('--c_same_sa_diff_g_com', '-c_ssdgc', type=float, default=0.05,help='同一样本不同粒度前50%相似性loss')
    parser.add_argument('--c_same_sa_diff_g_pvt', '-c_ssdgp', type=float, default=0.01, help='不同样本同一粒度相似性loss')
    parser.add_argument('--c_diff_sa_same_g_com', '-c_dssgc', type=float, default=0.0005, help='不同样本同一粒度相似性loss 1')
    parser.add_argument('--c_diff_sa_same_g_pvt', '-c_dssgp', type=float, default=1, help='不同样本同一粒度相似性loss 0.01')
    parser.add_argument('--c_diff_sa_diff_g_com', '-c_dsdgc', type=float, default=0.5, help='不同样本同一粒度相似性loss 0.001')
    parser.add_argument('--c_diff_sa_diff_g_pvt', '-c_dsdgp', type=float, default=1, help='不同样本同一粒度相似性loss')

    parser.add_argument('--c_cnfd', type=float, default=0.5, help='最后的confounding部分')
    parser.add_argument('--c_cndg', type=float, default=0.001, help='最后的confounding部分')
    parser.add_argument('--second_order_power', '-second', default=0)
    parser.add_argument('--c_loss_entropy_source', type=float, default=0, help="target dataset")
    parser.add_argument('--c_loss_entropy_target', type=float, default=0.01, help="target dataset")
    parser.add_argument('--config', default='configfile/config.yaml')
    parser.add_argument('--train_state', default='tral', choices=('tr', 'te', 'val', 'tral', 'plot'))
    parser.add_argument('--train_type', '-tt', default='tral', choices=('tr', 'te','val', 'tral', 'plot','eval', 'viz'))
    parser.add_argument('--mu', type=float, default=0.9, help="")

    parser.add_argument('--norm_type', default='T1', choices=('T1', 'T2', '', ''),
                        help="配置模型后期的norm，feat[256*7*7]。 T1：点像素49个256维feat； T2 就是256个49维语义属性feat") #
    parser.add_argument('--dataset', type=str, default='cp2', choices=('cp2', 'bd', 'cars', 'in', 'ci2', 'cn2', 'in2', '12cp2', '13cp2', '14cp2', '123cp2', '134cp2'))
    parser.add_argument('--source', type=str, default='c', help="source dataset")
    parser.add_argument('--target', type=str, default='p', help="target dataset")
    parser.add_argument('--batchsize', '-bs', type=int, default=32)
    parser.add_argument('--bs_test', type=int, default=None, help='None 代表train和test都用一样的，否则给出具体的int数值')
    parser.add_argument('--backbone', '-bkb', default='rn50', type=str, choices=(
        'generator',
        'mambaout_small',
        'mambaout_tiny',
        'mamba_small+',
        'mamba_tiny',
        'vmamba-tiny',
        'rn50', 'rn101', 'rn152',
        'vit_tiny_patch16_384', 'vit_tiny_patch16_224',
        'vit_small_patch16_384', 'vit_small_patch16_224', 'vit_small_patch32_384',
        'vit_base_patch16_384', 'vit_base_patch16_224', 'vit_base_patch32_224',
        'vit_large_patch16_384', 'vit_large_patch16_224', 'vit_large_patch32_224',
        'dino_vits16', 'dino_vitb16', 'dino_resnet50',
        'mixer_b16_224_in21k',
        'asmlp-tiny', 'asmlp-small', 'asmlp-base',
        'swin_tiny_patch4_window7_224.ms_in1k'))

    parser.add_argument('--b_3_bkb', action='store_true', default=False,help="默认是false")
    parser.add_argument('--initial_smooth', type=float, nargs='?', default=0.9, help="target dataset")
    parser.add_argument('--final_smooth', type=float, nargs='?', default=0.1, help="target dataset")

    parser.add_argument('--max_iteration', '-iter', type=float, default=20000, help="20000")
    parser.add_argument('--test_interval', type=float, default=1000, help="1000")
    parser.add_argument('--time_all_start', type=str, help="训练开始时间")

    parser.add_argument('--smooth_stratege', type=str, nargs='?', default='e', help="smooth stratege")
    parser.add_argument('--bkb_pretrain_dir',default='/mnt/sdb/zhaojiaojiao/data/pytorch_model.bin')


    parser.add_argument("--h", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', help='resume from checkpoint',
        default=None
                        )
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', '-b_dis', type=str2bool, default=False)

    parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")
    parser.add_argument("--startdomain", default=0, type=int)

    parser.add_argument("--val_times", default=10, type=int, help="val Repeat times")
    parser.add_argument("--output_dir", default='', type=str, help="Repeat times")
    parser.add_argument("--b_use_tensorboard", type=bool, default=False)
    parser.add_argument("--bn_store_ckpt", type=str2bool, default=False)
    parser.add_argument("--gpu_location", '-gpul', default='normal', choices=('liu90', 'siton', 'autodl', 'hy', 'normal', 'other'))

    parser.add_argument('--viz', default=False)
    parser.add_argument('--eval', default=False)

    parser.add_argument('--cdistance',  type=float, default=0.75, help="共性特征距离的比例")
    parser.add_argument('--pdistance', type=float, default=0.2, help="特异性特征距离的比例")
    parser.add_argument('--ndistance', type=float, default=0.05, help="混淆特征距离的比例")
    parser.add_argument('--testcentriods', type=str2bool, default = True, help="默认是true")
    parser.add_argument('--prototype', type=str2bool, default = False , help="默认是true")
    parser.add_argument('--prototyperaio', type=float, default=0.01, help="子质心损失的系数")
    parser.add_argument('--num_centriods', type=float, default=500, help="子质心初始化迭代次数")
    parser.add_argument('--use_same_class_distance', type=str2bool, default=False, help="默认是true,子质心损失只使用同类损失")
    parser.add_argument('--use_feature_extractor', type=str2bool, default=False,
                        help="默认是false,消融实验，使用特征提取器提取特征构造子质心")
    parser.add_argument('--testclassifier_weights', type=str2bool, default=True, help="默认是true")
    return parser
if __name__ == '__main__':

    parser = argparse.ArgumentParser('CFSG', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)




