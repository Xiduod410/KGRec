import numpy as np
import torch
from utils.parser import parse_args_kgsr
from utils.data_loader import load_data
from modules.causalKG import KGRec
from train_causal_kgrec import train_causal_kgrec
from utils.helper import init_logger
from utils.sampler import UniformSampler

if __name__ == '__main__':
    args = parse_args_kgsr()
    device = torch.device(f"cuda:{args.gpu_id}" if args.cuda else "cpu")
    logger = init_logger(args)

    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat = mat_list[2][0]  # mean-normalized adjacency
    model = KGRec(n_params, args, graph, adj_mat).to(device)

    # 使用 Python 版负采样器
    sampler = UniformSampler(seed=2020)
    train_cf_neg = sampler.sample_negative(train_cf[:, 0], n_params['n_items'], user_dict['train_user_set'], 1)
    train_cf_with_neg = np.concatenate([train_cf, np.array(train_cf_neg)], axis=1)

    train_causal_kgrec(model, train_cf_with_neg, user_dict, n_params, args, device, logger)
