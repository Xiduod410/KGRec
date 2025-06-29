from .metrics import *

import torch
import numpy as np
import heapq
import multiprocessing

cores = multiprocessing.cpu_count() // 2

class Evaluator:
    def __init__(self, args) -> None:
        self.args = args
        self.Ks = eval(args.Ks)
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        self.BATCH_SIZE = args.test_batch_size
        self.batch_test_flag = args.batch_test_flag

    def ranklist_by_heapq(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc

    def get_auc(self, item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = AUC(ground_truth=r, prediction=posterior)
        return auc

    def ranklist_by_sorted(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    def get_performance(self, user_pos_test, r, auc, Ks):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in Ks:
            precision.append(precision_at_k(r, K))
            recall.append(recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(ndcg_at_k(r, K, user_pos_test))
            hit_ratio.append(hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

    def test_one_user(self, x):
        # user u's ratings for user u
        rating = x[0]
        # uid
        u = x[1]
        # user u's items in the training set
        try:
            training_items = self.train_user_set[u]
        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = self.test_user_set[u]

        all_items = set(range(0, self.n_items))

        test_items = list(all_items - set(training_items))

        if self.args.test_flag == 'part':
            r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)
        else:
            r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating, self.Ks)

        return self.get_performance(user_pos_test, r, auc, self.Ks)

    def test(self, model, user_dict, n_params):
        result = {'precision': np.zeros(len(self.Ks)),
                'recall': np.zeros(len(self.Ks)),
                'ndcg': np.zeros(len(self.Ks)),
                'hit_ratio': np.zeros(len(self.Ks)),
                'auc': 0.}

        n_items = n_params['n_items']
        self.n_items = n_items
        n_users = n_params['n_users']
        self.n_users = n_users

        train_user_set = user_dict['train_user_set']
        self.train_user_set = train_user_set
        test_user_set = user_dict['test_user_set']
        self.test_user_set = test_user_set

        pool = multiprocessing.Pool(cores)

        u_batch_size = self.BATCH_SIZE
        i_batch_size = self.BATCH_SIZE

        test_users = list(test_user_set.keys())
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        with torch.no_grad():
            entity_gcn_emb, user_gcn_emb = model.generate()

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if self.batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(self.device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(self.device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(self.test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision']/n_test_users
                result['recall'] += re['recall']/n_test_users
                result['ndcg'] += re['ndcg']/n_test_users
                result['hit_ratio'] += re['hit_ratio']/n_test_users
                result['auc'] += re['auc']/n_test_users

        assert count == n_test_users
        pool.close()
        return result
