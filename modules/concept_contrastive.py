import torch
import torch.nn.functional as F
import random


class ConceptContrastiveLearning:
    def __init__(self, entity_emb, relation_emb, temperature=0.1):
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.temperature = temperature

    def identify_important_concepts(self, edge_index, edge_type, omega_scores, threshold=0.5):
        important_indices = omega_scores > threshold
        important_edges = edge_index[:, important_indices]
        important_types = edge_type[important_indices]
        return important_edges, important_types

    # 在 class ConceptContrastiveLearning 中
    def construct_samples(self, important_edges, important_types, num_samples=4096):
        """
        构造正负样本对，对每个正样本对都匹配一个负样本，保证数量一致。
        """
        n_important = important_edges.shape[1]
        if n_important < 2:
            return [], []

        positive_pairs = []
        negative_pairs = []

        # 1. 关系类型分组
        type_to_indices = {}
        for idx, r_type in enumerate(important_types.tolist()):
            type_to_indices.setdefault(r_type, []).append(idx)

        # 过滤掉无法形成正样本对的关系类型
        valid_types_for_pos = {t: idxs for t, idxs in type_to_indices.items() if len(idxs) > 1}
        unique_types = list(type_to_indices.keys())

        if not valid_types_for_pos or len(unique_types) < 2:
            return [], []

        # 2. 采样样本对
        # 尝试采样 num_samples 次，直到收集到足够的样本或尝试次数用尽
        attempts = 0
        max_attempts = num_samples * 2  # 防止无限循环

        while len(positive_pairs) < num_samples and attempts < max_attempts:
            attempts += 1
            try:
                # Step A: 采样一个正样本对
                # 随机选择一个可以形成正样本对的关系类型
                anchor_type = random.choice(list(valid_types_for_pos.keys()))
                # 从中随机选择两个不同的边作为 anchor 和 positive
                anchor_idx, positive_idx = random.sample(valid_types_for_pos[anchor_type], 2)
                anchor_edge = important_edges[:, anchor_idx]
                positive_edge = important_edges[:, positive_idx]

                # Step B: 为这个 anchor 采样一个负样本
                # 随机选择一个不同的关系类型
                while True:
                    negative_type = random.choice(unique_types)
                    if negative_type != anchor_type:
                        break

                negative_idx = random.choice(type_to_indices[negative_type])
                negative_edge = important_edges[:, negative_idx]

                # Step C: 添加到列表中
                positive_pairs.append((anchor_edge, positive_edge))
                # 负样本对的 anchor 与正样本对的 anchor 保持一致
                negative_pairs.append((anchor_edge, negative_edge))

            except (ValueError, IndexError):
                continue

        return positive_pairs, negative_pairs

    def contrastive_loss(self, pos_pairs, neg_pairs):
        """
        批量化计算对比损失：InfoNCE

        参数:
            pos_pairs (list): 正样本对列表，每个样本对包含两个边的索引。
            neg_pairs (list): 负样本对列表，每个样本对包含两个边的索引。

        返回:
            torch.Tensor: 对比损失值。

        逻辑:
        1. 如果正样本对或负样本对为空，直接返回损失值为 0。
        2. 提取正样本对和负样本对的实体嵌入：
           - 锚点实体嵌入（anchors）：从正样本对中提取第一个边的尾部实体嵌入。
           - 正样本实体嵌入（positives）：从正样本对中提取第二个边的尾部实体嵌入。
           - 负样本实体嵌入（negatives）：从负样本对中提取第二个边的尾部实体嵌入。
        3. 计算锚点与正样本、负样本之间的余弦相似度：
           - sim_pos：锚点与正样本的余弦相似度。
           - sim_neg：锚点与负样本的余弦相似度。
        4. 将正样本和负样本的相似度拼接为 logits，并除以温度参数（temperature）。
        5. 创建标签（labels），所有标签均为
        6. 使用交叉熵损失函数计算最终的对比损失。
        """
        device_to_use = self.entity_emb.device

        if len(pos_pairs) == 0 or len(neg_pairs) == 0:
            return torch.tensor(0.0, device=device_to_use)

        # 确保两边样本数量一致（使用最小长度）
        n_samples = min(len(pos_pairs), len(neg_pairs))
        positive_pairs = pos_pairs[:n_samples]
        negative_pairs = neg_pairs[:n_samples]

        anchors = torch.stack([self.entity_emb[p[0][1]].detach() for p in pos_pairs])  # [B, d]
        positives = torch.stack([self.entity_emb[p[1][1]].detach() for p in pos_pairs])
        negatives = torch.stack([self.entity_emb[n[1][1]].detach() for n in neg_pairs])

        sim_pos = F.cosine_similarity(anchors, positives, dim=1)
        sim_neg = F.cosine_similarity(anchors, negatives, dim=1)

        logits = torch.stack([sim_pos, sim_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
