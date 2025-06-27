from utils.timer import TimeCounter
import random
import numpy as np
import torch
import torch.nn as nn
from .AttnHGCN import AttnHGCN
from .concept_contrastive import ConceptContrastiveLearning
from .contrast import Contrast
from logging import getLogger
import math
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from scipy import sparse as sp

from .path_reasoning import PathReasoningReconstruction


def _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
    _, least_attn_edge_id = torch.topk(-edge_attn_score,
                                       int((1 - keep_rate) * edge_attn_score.shape[0]), sorted=False)
    cl_kg_mask = torch.ones_like(edge_attn_score).bool()
    cl_kg_mask[least_attn_edge_id] = False
    cl_kg_edge = edge_index[:, cl_kg_mask]
    cl_kg_type = edge_type[cl_kg_mask]
    return cl_kg_edge, cl_kg_type


def _adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func="torch"):
    inter_attn_prob = item_attn_mean[inter_edge[1]]
    # add gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
    """ prob based drop """
    inter_attn_prob = inter_attn_prob + noise
    inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

    if samp_func == "np":
        # we observed abnormal behavior of torch.multinomial on mind
        sampled_edge_idx = np.random.choice(np.arange(inter_edge_w.shape[0]),
                                            size=int(keep_rate * inter_edge_w.shape[0]), replace=False,
                                            p=inter_attn_prob.cpu().numpy())
    else:
        sampled_edge_idx = torch.multinomial(inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False)

    return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx] / keep_rate


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v


class KGRec(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat, hp_dict=None):
        super(KGRec, self).__init__()
        self.args_config = args_config
        self.logger = getLogger()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        self.gcn = AttnHGCN(channel=self.emb_size,
                            n_hops=self.context_hops,
                            n_users=self.n_users,
                            n_relations=self.n_relations,
                            node_dropout_rate=self.node_dropout_rate,
                            mess_dropout_rate=self.mess_dropout_rate)
        self.ablation = args_config.ab

        self.mae_coef = args_config.mae_coef
        self.mae_msize = args_config.mae_msize
        self.cl_coef = args_config.cl_coef
        self.tau = args_config.cl_tau
        self.cl_drop = args_config.cl_drop_ratio
        self.cl_sample_size = args_config.cl_sample_size
        self.cl_sample_size = 4096
        self.samp_func = "torch"

        if args_config.dataset == 'last-fm':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.01
            self.tau = 1.0
            self.cl_drop = 0.5
        elif args_config.dataset == 'mind-f':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.1
            self.cl_drop = 0.6
            self.samp_func = "np"
        elif args_config.dataset == 'alibaba-fashion':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.001
            self.tau = 0.2
            self.cl_drop = 0.5
        elif args_config.dataset == 'movielens':
            self.mae_coef = 0.1
            self.mae_msize = 256
            self.cl_coef = 0.01
            self.tau = 0.1
            self.cl_drop = 0.5
            self.samp_func = "torch"

        # update hps
        if hp_dict is not None:
            for k, v in hp_dict.items():
                setattr(self, k, v)

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            adj_mat)

        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)


        self.contrast_fn = Contrast(self.emb_size, tau=self.tau)
        self.path_reconstruction = PathReasoningReconstruction(self.gcn.relation_emb, self.all_embed)
        self.concept_contrastive = ConceptContrastiveLearning(self.all_embed[self.n_users:, :], self.gcn.relation_emb, temperature=0.1)
        # self.print_shapes()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def compute_gamma_cf_batch(self, user_emb, item_emb, edge_index, edge_type,
                               edge_ids_to_remove, user_batch, pos_item_batch):
        """
        Efficient approximation of γ_CF: delete all K edges, and run GCN once.
        Returns one scalar γ_CF value broadcast to all selected edges.
        """
        with torch.no_grad():
            # 构造反事实图（删边）
            mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
            mask[edge_ids_to_remove] = False
            edge_index_cf = edge_index[:, mask]
            edge_type_cf = edge_type[mask]

            # 在反事实图上跑一次 GCN（只 forward，不做梯度）
            entity_emb_cf, _ = self.gcn(
                user_emb.detach(), item_emb.detach(),
                edge_index_cf, edge_type_cf,
                self.inter_edge, self.inter_edge_w,
                mess_dropout=self.mess_dropout
            )
            item_emb_cf = entity_emb_cf[:self.n_items]

            # 计算推荐分数变化
            s_origin = (user_emb[user_batch] * item_emb[pos_item_batch]).sum(dim=-1)
            s_cf = (user_emb[user_batch] * item_emb_cf[pos_item_batch]).sum(dim=-1)
            gamma_scalar = torch.abs(s_origin - s_cf).mean().detach()

            # 每条边分配相同的 γ_CF 分数
            gamma_cf_scores = torch.ones_like(edge_ids_to_remove, dtype=torch.float,
                                              device=edge_index.device) * gamma_scalar

            return gamma_cf_scores

    def compute_edgewise_gamma_cf(self, user_emb, item_emb, edge_index, edge_type,
                                  topk_edge_ids, user_batch, pos_item_batch):
        """
        Compute γ_CF per edge: how much deleting this edge affects recommendation score.
        """
        gamma_cf_scores = torch.zeros_like(topk_edge_ids, dtype=torch.float, device=item_emb.device)

        for i, edge_id in enumerate(topk_edge_ids):
            # 构造反事实图：仅删除当前 edge_id
            mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
            mask[edge_id] = False
            edge_index_cf = edge_index[:, mask]
            edge_type_cf = edge_type[mask]

            # 重新跑一次 GCN 得到反事实图下的表示
            entity_emb_cf, _ = self.gcn(user_emb, item_emb, edge_index_cf, edge_type_cf,
                                        self.inter_edge, self.inter_edge_w, mess_dropout=self.mess_dropout)
            item_emb_cf = entity_emb_cf[:self.n_items]

            # 比较推荐分数差异
            score_origin = (user_emb[user_batch] * item_emb[pos_item_batch]).sum(dim=-1)
            score_cf = (user_emb[user_batch] * item_emb_cf[pos_item_batch]).sum(dim=-1)
            gamma_cf_scores[i] = torch.abs(score_origin - score_cf).mean()

        return gamma_cf_scores

    def forward(self, batch=None):
        # 获取当前批次的用户和物品信息
        user = batch['users']  # 当前批次的用户索引
        pos_item = batch['pos_items']  # 当前批次的正样本物品索引
        neg_item = batch['neg_items']  # 当前批次的负样本物品索引
        epoch_start = batch['batch_start'] == 0  # 判断是否是当前 epoch 的开始

        # 提取用户和物品的嵌入向量
        user_emb = self.all_embed[:self.n_users, :]  # 用户嵌入向量
        item_emb = self.all_embed[self.n_users:, :]  # 物品嵌入向量

        # 节点丢弃操作
        # 1. 图结构稀疏化：对边进行采样以减少计算量
        edge_index, edge_type = _relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)

        # 2. 计算边的注意力分数（rationale scores）
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(
            item_emb, edge_index, edge_type, print=epoch_start, return_logits=True)

        # 为自适应用户-物品掩码自动编码器（UI MAE）计算物品的注意力均值
        # 通过边的注意力分数计算物品的注意力均值
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.  # 防止注意力均值为零
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.  # 防止注意力均值为零

        # 综合两个方向的注意力均值，得到最终的物品注意力均值
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]
        # 为自适应 MAE（掩码自动编码器）训练添加噪声
        # 1. 计算边的注意力分数的标准差，用于衡量注意力分数的分布范围
        std = torch.std(edge_attn_score).detach()

        # 2. 生成 Gumbel 噪声，用于增强注意力分数的随机性
        # Gumbel 噪声的公式为 -log(-log(U))，其中 U 是一个均匀分布的随机数
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))

        # 3. 将生成的噪声添加到边的注意力分数中，以引入随机性
        # 这样可以帮助模型更好地探索不同的边权重组合，提高鲁棒性
        edge_attn_score = edge_attn_score + noise

        # 1. 获取结构打分 top-K
        topk_v, topk_attn_edge_id = torch.topk(edge_attn_score, self.mae_msize, sorted=False)

        # 使用轻量版 γ_CF（仅跑一次 GCN，no_grad）
        gamma_cf = self.compute_gamma_cf_batch(user_emb, item_emb,
                                               edge_index, edge_type, topk_attn_edge_id,
                                               user, pos_item)

        # Normalize γ_CF
        gamma_cf = (gamma_cf - gamma_cf.min()) / (gamma_cf.max() - gamma_cf.min() + 1e-8)

        # 融合结构打分和 γ_CF
        alpha = 0.7
        tilde_score = alpha * topk_v + (1 - alpha) * gamma_cf
        edge_attn_score[topk_attn_edge_id] = tilde_score.detach()
        #top_attn_edge_type = edge_type[topk_attn_edge_id]
        # 此刻，edge_attn_score 就是融合后的分数 ω̃
        fused_omega_scores = edge_attn_score

        # 使用自适应掩码方法对边进行掩码处理，生成编码图的边索引和类型，以及掩码后的边索引和类型
        # _mae_edge_mask_adapt_mixed 函数会根据 topk_attn_edge_id 提供的边选择掩码边，并返回相关信息
        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(
            edge_index, edge_type, topk_attn_edge_id)

        # 对交互边进行稀疏化处理，使用 _sparse_dropout 函数对边和权重进行随机丢弃操作
        # 通过设置 self.node_dropout_rate 来控制丢弃比例，以减少计算量并增强模型的鲁棒性
        inter_edge, inter_edge_w = _sparse_dropout(
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)


        # 推荐任务（Recommendation Task）
        # 使用 GCN（图卷积网络）对用户和物品嵌入进行更新，生成新的嵌入向量
        entity_gcn_emb, user_gcn_emb = self.gcn(
            user_emb,  # 用户嵌入向量
            item_emb,  # 物品嵌入向量
            enc_edge_index,  # 编码图的边索引
            enc_edge_type,  # 编码图的边类型
            inter_edge,  # 用户-物品交互边索引
            inter_edge_w,  # 用户-物品交互边权重
            mess_dropout=self.mess_dropout,  # 消息传递中的丢弃率
        )

        # 提取当前批次的用户嵌入向量
        u_e = user_gcn_emb[user]
        # 提取当前批次的正样本物品和负样本物品嵌入向量
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        # 计算推荐任务的损失，包括 BPR 损失和正则化损失
        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # 掩码自动编码器任务（MAE Task）使用点积解码器
        # 从编码图中提取掩码边的节点对嵌入向量
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]  # 掩码边的节点对嵌入
        # 从编码图中提取掩码边的关系嵌入向量
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type - 1]  # 掩码边的关系嵌入

        # 计算掩码自动编码器任务的损失，并乘以系数进行加权
        mae_loss = self.mae_coef * self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # # CL task
        # # 自适应采样过程
        # # 1. 对知识图谱边进行自适应丢弃，保留注意力分数较高的边。
        # #    使用 `_adaptive_kg_drop_cl` 函数，根据边的注意力分数 `edge_attn_score` 和丢弃率 `1 - self.cl_drop` 进行采样。
        # cl_kg_edge, cl_kg_type = _adaptive_kg_drop_cl(
        #     edge_index, edge_type, edge_attn_score, keep_rate=1 - self.cl_drop)
        #
        # # 2. 对用户-物品交互边进行自适应丢弃，保留注意力分数较高的边。
        # #    使用 `_adaptive_ui_drop_cl` 函数，根据物品的注意力均值 `item_attn_mean` 和丢弃率 `1 - self.cl_drop` 进行采样。
        # #    `samp_func` 参数指定采样方法（如 "torch" 或 "np"）。
        # cl_ui_edge, cl_ui_w = _adaptive_ui_drop_cl(
        #     item_attn_mean, inter_edge, inter_edge_w, 1 - self.cl_drop, samp_func=self.samp_func)
        #
        # # 3. 使用 GCN（图卷积网络）对用户-物品交互边进行聚合，生成物品的嵌入表示。
        # #    `forward_ui` 函数接收用户嵌入 `user_emb`、物品嵌入 `item_emb`、采样后的交互边 `cl_ui_edge` 和权重 `cl_ui_w`。
        # item_agg_ui = self.gcn.forward_ui(
        #     user_emb, item_emb[:self.n_items], cl_ui_edge, cl_ui_w)
        #
        # # 4. 使用 GCN 对知识图谱边进行聚合，生成物品的嵌入表示。
        # #    `forward_kg` 函数接收物品嵌入 `item_emb`、采样后的知识图谱边 `cl_kg_edge` 和类型 `cl_kg_type`。
        # item_agg_kg = self.gcn.forward_kg(
        #     item_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        #
        # # 5. 计算对比学习损失（Contrastive Learning Loss）。
        # #    使用 `contrast_fn` 函数对用户-物品交互边的聚合结果 `item_agg_ui` 和知识图谱边的聚合结果 `item_agg_kg` 进行对比。
        # #    损失值乘以系数 `self.cl_coef` 进行加权。
        # cl_loss = self.cl_coef * self.contrast_fn(item_agg_ui, item_agg_kg)

        # # 再次用原始图删除 TopK 边以构建反事实图（Counterfactual Graph）
        # # 1. 克隆当前的边索引和边类型，以便进行修改
        # edge_index_cf = edge_index.clone()
        # edge_type_cf = edge_type.clone()
        #
        # # 2. 创建一个布尔掩码，用于标记需要删除的边
        # #    初始掩码设置为全 True，表示所有边都保留
        # cf_mask = torch.ones(edge_index_cf.shape[1], dtype=torch.bool, device=edge_index_cf.device)
        #
        # # 3. 根据 TopK 边的索引，将对应的掩码设置为 False，表示这些边将被删除
        # cf_mask[topk_attn_edge_id] = False
        #
        # # 4. 使用掩码过滤边索引和边类型，生成反事实图的边索引和边类型
        # edge_index_cf = edge_index_cf[:, cf_mask]
        # edge_type_cf = edge_type_cf[cf_mask]
        #
        # # 对比物品的嵌入表示
        # # 1. 在反事实图上运行知识图谱聚合（forward_kg），生成物品的嵌入表示
        # item_agg_kg_cf = self.gcn.forward_kg(item_emb, edge_index_cf, edge_type_cf)[:self.n_items]
        #
        # # 2. 计算对比学习损失（Contrastive Learning Loss）
        # #    使用原始图的物品嵌入表示（item_agg_kg）和反事实图的物品嵌入表示（item_agg_kg_cf）进行对比
        # #    损失值乘以系数 self.cl_coef 进行加权
        # cf_cl_loss = self.cl_coef * self.contrast_fn(item_agg_kg, item_agg_kg_cf)

        # # Path Reasoning Reconstruction
        # # 使用在当前batch中采样的边(edge_index, edge_type)和计算出的融合分数
        # sampled_edges, sampled_types = self.path_reconstruction.sample_high_quality_paths(
        #     edge_index,
        #     edge_type,
        #     omega_scores=fused_omega_scores)
        # if sampled_edges.shape[1] > 0:  # 确保有路径被采样
        #     masked_edges, masked_types, mask_idx = self.path_reconstruction.mask_path(sampled_edges, sampled_types)
        #     path_loss = self.path_reconstruction.reconstruct_loss(masked_edges, masked_types, sampled_edges,
        #                                                           sampled_types)
        # else:
        #     path_loss = torch.tensor(0.0, device=loss.device)

        # Concept-level Contrastive Learning
        # 使用在当前batch中采样的边(edge_index, edge_type)和计算出的融合分数
        important_edges, important_types = self.concept_contrastive.identify_important_concepts(
            edge_index,
            edge_type,
            omega_scores=fused_omega_scores,
            threshold=0.5)  # 可以将threshold也设为超参数

        # 传入采样数量
        positive_pairs, negative_pairs = self.concept_contrastive.construct_samples(
            important_edges,
            important_types,
            num_samples=self.cl_sample_size)  # 使用超参数

        concept_loss = self.concept_contrastive.contrastive_loss(positive_pairs, negative_pairs)


        loss_dict = {
            "rec_loss": loss.item(),
            "mae_loss": mae_loss.item(),
            # "cl_loss": cl_loss.item(),
            # "cf_cl_loss": cf_cl_loss.item(),
            # "path_loss": path_loss.item(),
            "concept_loss": concept_loss.item()
        }
        return loss + mae_loss + concept_loss, loss_dict

    def calc_topk_attn_edge(self, entity_emb, edge_index, edge_type, k):
        edge_attn_score = self.gcn.norm_attn_computer(
            entity_emb, edge_index, edge_type, return_logits=True)
        positive_mask = edge_attn_score > 0
        edge_attn_score = edge_attn_score[positive_mask]
        edge_index = edge_index[:, positive_mask]
        edge_type = edge_type[positive_mask]
        topk_values, topk_indices = torch.topk(
            edge_attn_score, k, sorted=False)
        return edge_index[:, topk_indices], edge_type[topk_indices]

    # def generate(self):
    #     user_emb = self.all_embed[:self.n_users, :]
    #     item_emb = self.all_embed[self.n_users:, :]
    #     return self.gcn(user_emb,
    #                     item_emb,
    #                     self.edge_index,
    #                     self.edge_type,
    #                     self.inter_edge,
    #                     self.inter_edge_w,
    #                     mess_dropout=False)[:2]
    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        with torch.no_grad():  # 防止构建 autograd graph
            entity_emb, user_emb = self.gcn(
                user_emb,
                item_emb,
                self.edge_index,
                self.edge_type,
                self.inter_edge,
                self.inter_edge_w,
                mess_dropout=False
            )
        return entity_emb.detach(), user_emb.detach()  # 强制脱离计算图

    def rating(self, u_g_embeddings, i_g_embeddings, mode='default'):
        """
        用于评估阶段计算评分矩阵：
        mode:
          - 'default': dot(u, i)
          - 'causal': 使用 Causal Score: g(u,i) * σ(h(i,a))
          - 'counterfactual': 使用反事实评分（如需扩展）
        """
        if mode == 'default':
            scores = torch.matmul(u_g_embeddings, i_g_embeddings.t())  # [n_users, n_items]
        elif mode == 'causal':
            # 推荐阶段使用因果融合得分（若已训练 g(u,i) 结构）
            scores = torch.matmul(u_g_embeddings, i_g_embeddings.t())
            # 若有结构注意力、γ_CF 融合逻辑，可扩展此处
        else:
            raise ValueError(f"Unknown rating mode: {mode}")

        return scores

    def compute_gamma_cf_batch(self, user_emb, item_emb, edge_index, edge_type,
                               edge_ids_to_remove, user_batch, pos_item_batch):
        """
        Efficient approximation of γ_CF: delete all K edges, and run GCN once.
        """
        # 构造 counterfactual 图
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        mask[edge_ids_to_remove] = False
        edge_index_cf = edge_index[:, mask]
        edge_type_cf = edge_type[mask]

        # 一次性跑 GCN
        entity_emb_cf, _ = self.gcn(user_emb, item_emb, edge_index_cf, edge_type_cf,
                                    self.inter_edge, self.inter_edge_w, mess_dropout=self.mess_dropout)
        item_emb_cf = entity_emb_cf[:self.n_items]

        # 每个 user-item 对的打分差
        score_origin = (user_emb[user_batch] * item_emb[pos_item_batch]).sum(dim=-1)
        score_cf = (user_emb[user_batch] * item_emb_cf[pos_item_batch]).sum(dim=-1)
        gamma_cf = torch.abs(score_origin - score_cf).detach()  # shape: [batch_size]

        # 统一分配一个值：mean γ_CF 用于 Top-K 所有边（统一替代方案）
        gamma_scalar = gamma_cf.mean()
        gamma_cf_per_edge = torch.ones_like(edge_ids_to_remove, dtype=torch.float,
                                            device=edge_index.device) * gamma_scalar

        return gamma_cf_per_edge



    # @TimeCounter.count_time(warmup_interval=4)
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        if torch.isnan(mf_loss):
            raise ValueError("nan mf_loss")

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

    def print_shapes(self):
        self.logger.info("########## Ablation ##########")
        self.logger.info("ablation: {}".format(self.ablation))
        self.logger.info("########## Model HPs ##########")
        self.logger.info("tau: {}".format(self.contrast_fn.tau))
        self.logger.info("cL_drop: {}".format(self.cl_drop))
        self.logger.info("cl_coef: {}".format(self.cl_coef))
        self.logger.info("mae_coef: {}".format(self.mae_coef))
        self.logger.info("mae_msize: {}".format(self.mae_msize))
        self.logger.info("########## Model Parameters ##########")
        self.logger.info("context_hops: %d", self.context_hops)
        self.logger.info("node_dropout: %d", self.node_dropout)
        self.logger.info("node_dropout_rate: %.1f", self.node_dropout_rate)
        self.logger.info("mess_dropout: %d", self.mess_dropout)
        self.logger.info("mess_dropout_rate: %.1f", self.mess_dropout_rate)
        self.logger.info('all_embed: {}'.format(self.all_embed.shape))
        self.logger.info('interact_mat: {}'.format(self.inter_edge.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))

    def generate_kg_drop(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        edge_index, edge_type = _edge_sampling(
            self.edge_index, self.edge_type, self.kg_drop_test_keep_rate)
        return self.gcn(user_emb,
                        item_emb,
                        edge_index,
                        edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
                        mess_dropout=False)[:2]

    def generate_global_attn_score(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        edge_attn_score = self.gcn.norm_attn_computer(
            item_emb, self.edge_index, self.edge_type)

        return edge_attn_score, self.edge_index, self.edge_type
