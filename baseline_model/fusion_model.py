import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class EmbeddingGatedFusion(nn.Module):
    """
    输入两个 embedding 生成一个融合 embedding
    fused = w * emb1 + (1 - w) * emb2
    w 由一个小 MLP 学习得到/ 图·1
    自动处理全0输入，生成 mask
    """
    def __init__(self, embedding_dim, hidden_dim=None, eps=1e-6):
        super().__init__()
        self.eps = eps  # 用于判断全0
        if hidden_dim is None:
            hidden_dim = embedding_dim

        # MLP 生成权重
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # 输出一个标量
            nn.Sigmoid()                # 将 w 压到 [0,1]
        )

    def forward(self, emb1, emb2):
        """
        emb1, emb2: (batch_size, embedding_dim)
        返回: fused embedding (batch_size, embedding_dim), weight, mask
        mask = 1 表示有效 embedding, mask = 0 表示全0
        """
        # 生成 mask，判断全0向量
        mask1 = (emb1.abs().sum(dim=-1, keepdim=True) > self.eps).float()  # (batch_size,1)
        mask2 = (emb2.abs().sum(dim=-1, keepdim=True) > self.eps).float()

        # 对全0 embedding 用0代替，避免 MLP 给出奇异输出
        safe_emb1 = emb1 * mask1
        safe_emb2 = emb2 * mask2

        # 拼接 embedding
        x = torch.cat([safe_emb1, safe_emb2], dim=-1)  # (batch_size, 2*embedding_dim)

        # 生成权重
        w = self.mlp(x)  # (batch_size,1)

        # 如果某个 embedding 全0，则权重固定指向另一个
        w = w * mask1 + (1 - mask2) * mask2  # mask 修正

        # 融合 embedding
        fused = w * safe_emb1 + (1 - w) * safe_emb2

        # 最终 mask：两者都为0时，输出也为0
        final_mask = ((mask1 + mask2) > 0).float()

        return fused, w, final_mask


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embs, pool_dim, eps: float = 1e-6, keepdim: bool = False):
        """
        对张量 `embs` 按给定维度 `pool_dim` 做平均池化，忽略全0的 embedding。

        约定：最后一维为特征维(embedding 维度)。判断“全0的 emb”基于最后一维。

        参数：
        - embs: 任意形状张量，最后一维为特征维 D。
        - pool_dim: 需要做平均池化的维度索引（可为负）。不得等于最后一维。
        - eps: 判断全0的阈值，默认 1e-6。
        - keepdim: 是否保留被归约的维度。

        返回：
        - pooled: 在 `pool_dim` 上求平均后的张量。全0向量不参与平均；若该维上全为0，则输出为0。
        """
        if pool_dim < 0:
            pool_dim = embs.dim() + pool_dim

        feature_dim = embs.dim() - 1
        if pool_dim == feature_dim:
            raise ValueError("pool_dim 不能为最后一维(特征维)，请在样本/时间等维度上做池化")

        # 基于特征维判断全0 emb：mask 形状与 embs 相同，但特征维为 1，以便广播
        mask = (embs.abs().sum(dim=feature_dim, keepdim=True) > eps).float()

        # 仅累计非零 emb
        safe_embs = embs * mask
        num = safe_embs.sum(dim=pool_dim, keepdim=keepdim)
        den = mask.sum(dim=pool_dim, keepdim=keepdim).clamp(min=1.0)

        pooled = num / den
        return pooled