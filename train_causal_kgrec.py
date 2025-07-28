# 新版训练脚本（适配因果整合后的 KGRec 模型）

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.evaluate_kgsr import test
from utils.helper import early_stopping, init_logger
from logging import getLogger

def train_causal_kgrec(model, train_cf, user_dict, n_params, args, device, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    test_interval = 5 if args.dataset == 'last-fm' else 1
    early_stop_step = 5 if args.dataset == 'last-fm' else 10

    cur_best_pre_0 = 0
    cur_stopping_step = 0
    should_stop = False

    log_fn = init_logger(args)
    logger = getLogger()
    for epoch in range(args.epoch):
        model.train()
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        loss_log, s = defaultdict(float), 0
        with tqdm(total=len(train_cf) // args.batch_size, desc=f"Epoch {epoch}") as pbar:
            while s + args.batch_size <= len(train_cf):
                batch_pairs = torch.from_numpy(train_cf[s:s+args.batch_size]).to(device)
                batch = {
                    'users': batch_pairs[:, 0],
                    'pos_items': batch_pairs[:, 1],
                    'neg_items': batch_pairs[:, 2],
                    'batch_start': s
                }

                batch_loss, batch_dict = model(batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for k, v in batch_dict.items():
                    loss_log[k] += v

                s += args.batch_size
                pbar.update(1)

        # Logging
        logger.info(f"Epoch {epoch} loss: {dict(loss_log)}")

        # Eval
        if epoch % test_interval == 0 and epoch >= 1:
            model.eval()
            with torch.no_grad():
                ret = test(model, user_dict, n_params)

            logger.info(f"Epoch {epoch} Test Metrics: Recall@20: {ret['recall'][0]:.4f} | NDCG@20: {ret['ndcg'][0]:.4f}")

            # Early stop
            cur_best_pre_0, cur_stopping_step, should_stop = early_stopping(
                ret['recall'][0], cur_best_pre_0, cur_stopping_step, expected_order='acc', flag_step=early_stop_step)

            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), f"{args.out_dir}/best_model.ckpt")
                logger.info(f"Saved best model at epoch {epoch}")

