# 文件名：kgrec_with_concept_and_causal.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from modules.KGRec import KGRec, _relation_aware_edge_sampling, _sparse_dropout, _mae_edge_mask_adapt_mixed, \
    _adaptive_kg_drop_cl, _adaptive_ui_drop_cl
from modules.concept_contrastive import ConceptContrastiveLearning

class KGRecWithConceptAndCausal(KGRec):
    def __init__(self, *args, **kwargs):
        super(KGRecWithConceptAndCausal, self).__init__(*args, **kwargs)

        self.concept_cl_module = ConceptContrastiveLearning(
            entity_emb=self.all_embed[self.n_users:],
            relation_emb=self.gcn.relation_emb,
            temperature=self.tau
        )

    def compute_edgewise_gamma_cf(self, user_emb, item_emb, edge_index, edge_type,
                                  topk_edge_ids, user_batch, pos_item_batch):
        gamma_cf_scores = torch.zeros_like(topk_edge_ids, dtype=torch.float, device=item_emb.device)
        for i, edge_id in enumerate(topk_edge_ids):
            mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
            mask[edge_id] = False
            edge_index_cf = edge_index[:, mask]
            edge_type_cf = edge_type[mask]

            entity_emb_cf, _ = self.gcn(user_emb, item_emb, edge_index_cf, edge_type_cf,
                                        self.inter_edge, self.inter_edge_w, mess_dropout=self.mess_dropout)
            item_emb_cf = entity_emb_cf[:self.n_items]

            score_origin = (user_emb[user_batch] * item_emb[pos_item_batch]).sum(dim=-1)
            score_cf = (user_emb[user_batch] * item_emb_cf[pos_item_batch]).sum(dim=-1)
            gamma_cf_scores[i] = torch.abs(score_origin - score_cf).mean()

        return gamma_cf_scores

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        epoch_start = batch['batch_start'] == 0

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        edge_index, edge_type = _relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)

        edge_attn_score, _ = self.gcn.norm_attn_computer(
            item_emb, edge_index, edge_type, print=epoch_start, return_logits=True)

        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]

        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score_noisy = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(edge_attn_score_noisy, self.mae_msize, sorted=False)

        ablation_mode = getattr(self.args_config, 'ablation_mode', 'full')

        if ablation_mode != 'only_concept' and ablation_mode != 'KGRec':
            gamma_cf_scores = self.compute_edgewise_gamma_cf(user_emb, item_emb, edge_index, edge_type,
                                                             topk_attn_edge_id, user, pos_item)
        else:
            gamma_cf_scores = None

        if ablation_mode != 'only_causal' and ablation_mode != 'KGRec':
            important_edges, important_types = self.concept_cl_module.identify_important_concepts(
                edge_index, edge_type, omega_scores=edge_attn_score, threshold=0.5)

            pos_pairs, neg_pairs = self.concept_cl_module.construct_samples(
                important_edges, important_types, num_samples=512)

            concept_cl_loss = self.concept_cl_module.contrastive_loss(pos_pairs, neg_pairs)
        else:
            concept_cl_loss = torch.tensor(0.0, device=self.device)

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, _ = _mae_edge_mask_adapt_mixed(
            edge_index, edge_type, topk_attn_edge_id)

        inter_edge, inter_edge_w = _sparse_dropout(self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb, item_emb, enc_edge_index, enc_edge_type,
                                                inter_edge, inter_edge_w, mess_dropout=self.mess_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type - 1]
        mae_loss = self.mae_coef * self.create_mae_loss(node_pair_emb, masked_edge_emb)

        cl_kg_edge, cl_kg_type = _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate=1 - self.cl_drop)
        cl_ui_edge, cl_ui_w = _adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, 1 - self.cl_drop,
                                                   samp_func=self.samp_func)

        item_agg_ui = self.gcn.forward_ui(user_emb, item_emb[:self.n_items], cl_ui_edge, cl_ui_w)
        item_agg_kg = self.gcn.forward_kg(item_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        cl_loss = self.cl_coef * self.contrast_fn(item_agg_ui, item_agg_kg)

        # 正确控制总损失组合逻辑
        if ablation_mode == 'KGRec':
            total_loss = loss
        elif ablation_mode == 'only_concept':
            total_loss = loss + mae_loss + self.cl_coef * concept_cl_loss
        elif ablation_mode == 'only_causal':
            total_loss = loss + mae_loss + cl_loss
        else:  # full
            total_loss = loss + mae_loss + cl_loss + self.cl_coef * concept_cl_loss

        loss_dict = {
            "rec_loss": loss.item(),
            "mae_loss": mae_loss.item(),
            "cl_loss": cl_loss.item(),
            "concept_cl_loss": concept_cl_loss.item(),
        }
        return total_loss, loss_dict
