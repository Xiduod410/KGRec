import torch
import numpy as np
from modules import testKGRec
from modules.KGRec import KGRec
from modules.testKGRec import TestKGRec
from utils.data_loader_kgcl import load_data
from utils.evaluate_kgsr import evaluate_model
import matplotlib.pyplot as plt
from utils.parser import parse_args_kgsr
from utils.data_loader import load_data
from utils.evaluate_kgsr import test
from utils.helper import early_stopping, init_logger
from logging import getLogger
import os
def evaluate_all_modes(model, dataloader, topk=[5, 10, 20, 50]):
    model.eval()
    modes = ['default', 'causal', 'counterfactual']
    recalls, aucs = {}, {}

    u_embed, i_embed = model.generate()

    for mode in modes:
        scores = model.rating(u_embed, i_embed, mode=mode)
        recall_k, auc_k = evaluate_model(scores, dataloader, topk=topk)
        recalls[mode] = recall_k
        aucs[mode] = auc_k
        print(f"[{mode}] Recall@K: {recall_k}")
        print(f"[{mode}] AUC@K: {auc_k}")

    return recalls, aucs


def plot_metrics(recalls, aucs, topk):
    plt.figure(figsize=(12, 5))

    # Recall
    plt.subplot(1, 2, 1)
    for mode, values in recalls.items():
        plt.plot(topk, values, label=mode, marker='o')
    plt.xlabel("Top-K")
    plt.ylabel("Recall")
    plt.title("Recall@K Comparison")
    plt.grid(True)
    plt.legend()

    # AUC
    plt.subplot(1, 2, 2)
    for mode, values in aucs.items():
        plt.plot(topk, values, label=mode, marker='o')
    plt.xlabel("Top-K")
    plt.ylabel("AUC")
    plt.title("AUC@K Comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args_kgsr()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    log_fn = init_logger(args)
    logger = getLogger()

    logger.info('PID: %d', os.getpid())
    logger.info(f"DESC: {args.desc}\n")

    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    dataloader = {
        "test_cf": test_cf,
        "user_dict": user_dict,
    }

    """define model"""
    model_dict = {
        'KGSR': TestKGRec,
    }
    model = model_dict[args.model]
    print(f"Data type of mat_list[2]: {type(mat_list[2])}")
    # Convert mat_list[2] to a sparse tensor
    adj_matrix_data_list = mat_list[2]
    adj_matrix_data = adj_matrix_data_list[0]
    print(f"Data type of mat_list[2]: {type(adj_matrix_data)}")
    #print(f"Data type of adj_matrix_data: {type(adj_matrix_data)}")
    model = model(n_params, args, graph,adj_matrix_data).to(device)
    model.load_state_dict(torch.load("weight/last-fm-kgrec-01.log_last-fm.ckpt"))
    model.eval()

    topk = [5, 10, 20, 30, 50]
    recalls, aucs = evaluate_all_modes(model, dataloader, topk=topk)
    plot_metrics(recalls, aucs, topk)
