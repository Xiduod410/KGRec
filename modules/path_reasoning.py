import torch
import torch.nn.functional as F


class PathReasoningReconstruction:
    # 1. 从 __init__ 中移除 omega_scores
    def __init__(self, relation_emb, entity_emb, path_length=3):
        self.relation_emb = relation_emb
        self.entity_emb = entity_emb
        self.path_length = path_length

    # 2. 将 omega_scores 作为参数添加到方法中
    def sample_high_quality_paths(self, edge_index, edge_type, omega_scores, top_k=20, path_length=3):
        # 从high ω̃ scores中选择top_k实体，并采样路径长度为path_length
        # 3. 使用传入的 omega_scores 参数，而不是 self.omega_scores
        topk_values, topk_indices = torch.topk(omega_scores, top_k, sorted=False)

        # 确保采样的路径长度不超过找到的 top-k 边的数量
        num_to_sample = min(path_length, topk_indices.shape[0])

        sampled_edges = edge_index[:, topk_indices[:num_to_sample]]
        sampled_types = edge_type[topk_indices[:num_to_sample]]
        return sampled_edges, sampled_types

    def mask_path(self, path_edges, path_types):
        # 随机掩码路径中的一个实体
        if path_edges.shape[1] == 0:  # 如果没有采样到路径，直接返回
            return path_edges, path_types, -1

        mask_idx = torch.randint(0, path_edges.shape[1], (1,))
        masked_edges = path_edges.clone()
        masked_types = path_types.clone()
        masked_edges[:, mask_idx] = -1  # Mask entity
        masked_types[mask_idx] = -1  # Mask relation
        return masked_edges, masked_types, mask_idx

    def reconstruct_loss(self, masked_edges, masked_types, original_edges, original_types):
        # 如果没有边，损失为0
        if original_edges.shape[1] == 0:
            return torch.tensor(0.0, device=self.entity_emb.device)

        # 预测掩码实体或关系
        # 注意: 需要处理-1索引，这里用一个简单的忽略策略
        valid_mask = masked_types != -1
        if not torch.any(valid_mask):  # 如果所有都被掩码了（不太可能，但作为保护）
            return torch.tensor(0.0, device=self.entity_emb.device)

        pred_entity = self.entity_emb[masked_edges[1, valid_mask]]
        pred_relation = self.relation_emb[masked_types[valid_mask]]

        target_entity = self.entity_emb[original_edges[1, valid_mask]]
        target_relation = self.relation_emb[original_types[valid_mask]]

        # F.cross_entropy期望的target是类别索引，而不是embedding
        # 这里的实现看起来更像是MSE或余弦损失。我们假设目标是让embedding相似
        # 这里假设损失函数是自定义的，我们保持原样，但指出其问题。
        # cross_entropy 的正确用法是: F.cross_entropy(model_output_logits, target_indices)
        # 此处我们暂时保持原样，因为这超出了omega_scores的范围，但这是一个潜在的问题。
        # 为了让代码能运行，我们改用 MSE Loss
        loss_entity = F.mse_loss(pred_entity, target_entity)
        loss_relation = F.mse_loss(pred_relation, target_relation)
        return loss_entity + loss_relation
