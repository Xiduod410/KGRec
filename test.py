# 文件名：run_ablation_study.py

import torch
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from utils.data_loader import load_data
from utils.parser import parse_args_kgsr
from utils.sampler import UniformSampler
from utils.evaluate_kgsr import test
from modules.KGRec import KGRec
from modules.KGRecwith import KGRecWithConceptAndCausal
from utils.helper import init_logger
import random

# 设置随机种子
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def run_model_variant(model_class, model_name, n_params, args, graph, mean_mat, train_cf, user_dict):
    sampling = UniformSampler(seed)
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    model = model_class(n_params, args, graph, mean_mat).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n🚀 开始训练模型: {model_name}")
    for epoch in range(5):  # 训练 5 轮
        train_cf_with_neg = sampling.sample_negative(train_cf[:, 0], n_params['n_items'], user_dict['train_user_set'], 1)
        train_cf_with_neg = np.asarray(train_cf_with_neg)
        train_cf_triples = np.concatenate([train_cf, train_cf_with_neg], axis=1)
        np.random.shuffle(train_cf_triples)

        s = 0
        batch_size = args.batch_size
        n_batches = len(train_cf_triples) // batch_size

        with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/5 - {model_name}") as pbar:
            while s + batch_size <= len(train_cf_triples):
                entity_pairs = torch.from_numpy(train_cf_triples[s:s + batch_size]).long()
                try:
                    entity_pairs = entity_pairs.to(device)
                except RuntimeError as oom:
                    print("⚠️ RuntimeError: Device OOM during batch move, skipping batch.")
                    s += batch_size
                    pbar.update(1)
                    continue

                batch = {
                    'users': entity_pairs[:, 0],
                    'pos_items': entity_pairs[:, 1],
                    'neg_items': entity_pairs[:, 2],
                    'batch_start': s
                }
                try:
                    loss, loss_dict = model(batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except RuntimeError as oom:
                    if 'out of memory' in str(oom):
                        print("⚠️ CUDA OOM during forward/backward, skipping batch.")
                        torch.cuda.empty_cache()
                    else:
                        raise oom
                s += batch_size
                pbar.update(1)

    model.eval()
    with torch.no_grad():
        print(f"\n🔍 测试模型: {model_name}")
        ret = test(model, user_dict, n_params, show_progress=True)

    print(f"\n✅ 模型: {model_name}")
    print(f"Recall@20: {ret['recall'][0]:.4f}  NDCG@20: {ret['ndcg'][0]:.4f}  Hit@20: {ret['hit_ratio'][0]:.4f}\n")
    return ret

if __name__ == '__main__':
    args = parse_args_kgsr()
    init_logger(args)

    # 加载数据集
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    # 数据传参准备
    results = {}

    print("========== Ablation Study (with 5-epoch training) ==========")

    # baseline KGRec
    args.ablation_mode = 'KGRec'
    results['KGRec'] = run_model_variant(KGRec, 'KGRec', n_params, args, graph, mean_mat_list[0], train_cf, user_dict)

    # KGRec + ConceptCL only
    args.ablation_mode = 'only_concept'
    results['KGRec + ConceptCL'] = run_model_variant(KGRecWithConceptAndCausal, 'ConceptCL only', n_params, args, graph, mean_mat_list[0], train_cf, user_dict)

    # KGRec + Causal γCF only
    args.ablation_mode = 'only_causal'
    results['KGRec + Causal'] = run_model_variant(KGRecWithConceptAndCausal, 'Causal only', n_params, args, graph, mean_mat_list[0], train_cf, user_dict)

    # KGRec + ConceptCL + Causal（全模块）
    args.ablation_mode = 'full'
    results['Full Model'] = run_model_variant(KGRecWithConceptAndCausal, 'Full Model', n_params, args, graph, mean_mat_list[0], train_cf, user_dict)

    # 输出总结
    print("\n===== Summary: Recall@20 =====")
    for name, r in results.items():
        print(f"{name:25s}: {r['recall'][0]:.4f}  |  NDCG@20: {r['ndcg'][0]:.4f}  |  Hit@20: {r['hit_ratio'][0]:.4f}")
