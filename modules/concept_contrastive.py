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

    def construct_samples(self, important_edges, important_types, num_samples=4096, max_pairs=5000):
        """
        构造正负样本对，每个正对对应相同关系下不同实体，负对来自不同关系。
        限制最大正负样本对数量为 max_pairs。
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

        unique_types = list(type_to_indices.keys())
        if len(unique_types) < 2:
            return [], []

        # 2. 采样样本对
        for _ in range(num_samples):
            anchor_idx = random.randint(0, n_important - 1)
            anchor_edge = important_edges[:, anchor_idx]
            anchor_type = important_types[anchor_idx].item()

            # 正样本
            positive_pool = type_to_indices[anchor_type]
            if len(positive_pool) > 1:
                while True:
                    positive_idx = random.choice(positive_pool)
                    if positive_idx != anchor_idx:
                        break

                positive_edge = important_edges[:, positive_idx]
                positive_pairs.append((anchor_edge, positive_edge))

            # 负样本
            other_types = [t for t in unique_types if t != anchor_type]
            if other_types:
                negative_type = random.choice(other_types)
                negative_idx = random.choice(type_to_indices[negative_type])
                negative_edge = important_edges[:, negative_idx]
                negative_pairs.append((anchor_edge, negative_edge))

        # 3. 限制样本数
        positive_pairs = random.sample(positive_pairs, min(len(positive_pairs), max_pairs))
        negative_pairs = random.sample(negative_pairs, min(len(negative_pairs), max_pairs))

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
        if not pos_pairs or not neg_pairs:
            return torch.tensor(0.0, device=self.entity_emb.device)

        anchors = torch.stack([self.entity_emb[p[0][1]].detach() for p in pos_pairs])  # [B, d]
        positives = torch.stack([self.entity_emb[p[1][1]].detach() for p in pos_pairs])
        negatives = torch.stack([self.entity_emb[n[1][1]].detach() for n in neg_pairs])

        sim_pos = F.cosine_similarity(anchors, positives, dim=1)
        sim_neg = F.cosine_similarity(anchors, negatives, dim=1)

        logits = torch.stack([sim_pos, sim_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
