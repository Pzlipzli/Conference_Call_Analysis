import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_moe import *  
from fusion_model import *


class MaskedPairProjector(nn.Module):
    """
    将两路向量 x1, x2 （形状 [..., D]）融合到同一维度 D，自动掩蔽全0分支。
    gate_mode:
      - "norm": 用 ||x|| 作为权重（推荐，稳定简单）
      - "learned": 小门控网络学习权重（两路都存在时才生效）
    """
    def __init__(self, d: int, gate_mode: str = "norm", eps: float = 1e-8, dropout: float = 0.1):
        super().__init__()
        assert gate_mode in ("norm", "learned")
        self.eps = eps
        self.gate_mode = gate_mode

        # 两路各自投影到 D（用 FP32 计算更稳）
        self.lin1 = nn.Linear(d, d).to(torch.float32)
        self.lin2 = nn.Linear(d, d).to(torch.float32)
        self.act  = nn.ReLU()
        self.ln   = nn.LayerNorm(d)
        self.dp   = nn.Dropout(dropout)

        if gate_mode == "learned":
            # 学习型门控，仅在两路都“非零”时启用
            self.gate = nn.Sequential(
                nn.Linear(2 * d, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # 输出两路权重的logits
            )
        else:
            self.gate = None

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        # 清理 NaN/Inf，并限制极端值，防止半精度溢出
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x.clamp_(-1e4, 1e4)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1, x2: [..., D]（例如 [B, D] 或 [B, #fac, D]）
        返回:   [..., D]
        """
        dtype = torch.float32  # 强制在 FP32 里做融合更稳
        x1 = self._sanitize(x1).to(dtype)
        x2 = self._sanitize(x2).to(dtype)

        # 路径存在性：按最后一维范数判断是否“全0/近0”
        n1 = x1.norm(p=2, dim=-1, keepdim=True)
        n2 = x2.norm(p=2, dim=-1, keepdim=True)
        m1 = (n1 > self.eps).float()   # [..., 1]
        m2 = (n2 > self.eps).float()   # [..., 1]
        both = (m1 > 0) & (m2 > 0)     # [..., 1], bool

        # 各自投影
        y1 = self.act(self.lin1(x1))
        y2 = self.act(self.lin2(x2))

        # 计算权重
        if self.gate is None:
            # 基于范数的简单权重（仅当两路都有效）
            s = (n1 + n2).clamp_min(self.eps)
            w1_both = n1 / s
            w2_both = n2 / s
        else:
            # 学习门控（仅当两路都有效）
            g_in = torch.cat([x1, x2], dim=-1)
            g_logits = self.gate(g_in)
            w = F.softmax(g_logits, dim=-1)  # [..., 2]
            w1_both = w[..., 0:1]
            w2_both = w[..., 1:2]

        # 组合掩码逻辑：
        # - 两路都有效：用 w*_both
        # - 仅一路有效：该路权重=1，另一条=0
        # - 两路都无效：权重全0（输出为0）
        zeros = torch.zeros_like(n1)
        ones  = torch.ones_like(n1)

        w1 = torch.where(both, w1_both, torch.where(m1 > 0, ones, zeros))
        w2 = torch.where(both, w2_both, torch.where(m2 > 0, ones, zeros))

        # 归一化（以防仅一路时权重不是 {1,0} 的边界数值）
        w_sum = (w1 + w2).clamp_min(self.eps)
        w1 = w1 / w_sum
        w2 = w2 / w_sum

        fused = w1 * y1 + w2 * y2  # [..., D]
        fused = self.dp(self.ln(fused))
        return fused


class Structure(nn.Module):
    """
    精简版：
    - 仅使用 pres_emb, qnda_emb, pres_history_emb, qnda_history_emb 四个输入
    - 将四个 [B, D] 的向量交给 __init__ 传入的 moe 模块
    - moe 的 forward 可以接受 List[Tensor] 或者单个堆叠 Tensor；本类做了自适配
    """
    def __init__(self, 
                 moe: nn.Module,                   # 你的 MoE 框架模块（外部提供）
                 history_aggr: nn.Module = MaskedDecayAggregator(),
                 moe_accepts: str = "list"         # "list" 或 "stack"（B, 4, D）
                 ):
        super().__init__()
        self.moe = moe
        self.history_aggr = history_aggr
        assert moe_accepts in ("list", "stack")
        self.moe_accepts = moe_accepts

        self.proj_p = MaskedPairProjector(d=768, gate_mode="norm", dropout=0.1)
        self.proj_q = MaskedPairProjector(d=768, gate_mode="norm", dropout=0.1)
        # 如想让模型自己学权重，把 gate_mode 改成 "learned"

    @staticmethod
    def _reduce_to_vector(x: torch.Tensor) -> torch.Tensor:
        """
        将任意形状的 embedding 压到 [B, D]
        - [B, D]      -> 原样返回
        - [B, T, D]   -> 对 T 做平均（或你可以改成加权/cls位等）
        - [D]         -> 扩一个 batch 维度：[1, D]
        其他维度情况按需要再扩展
        """
        if x is None:
            return None
        if x.dim() == 2:
            return x
        elif x.dim() == 3:
            return x.mean(dim=1)
        elif x.dim() == 1:
            return x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected embedding shape: {x.shape}")

    def forward(self, batch):
        # 取当前会议的两个向量
        pres_emb = batch.store['pres_emb']    # 期望 [B, D]
        qnda_emb = batch.store['qnda_emb']

        # 取历史因子对应的 embedding 序列（一般是 [B, #fac, #his, D]）
        pres_history = batch.store['pres_fac_hist']
        qnda_history = batch.store['qnda_fac_hist']

        pres_hist_count = batch.store.get('pres_fac_hist_len', None)  # list of int, len=B
        qnda_hist_count = batch.store.get('qnda_fac_hist_len', None)

        # # 若提供了历史收益率序列（仅 ret_close），则用其作为加权依据（可选）
        # pres_fac_hist_label = batch.store.get('pres_fac_hist_label', None).squeeze(-1)  # [B, #fac, #his]
        # qnda_fac_hist_label = batch.store.get('qnda_fac_hist_label', None).squeeze(-1)  # [B, #fac, #his]

        # # 得到历史聚合后的嵌入（尽量复用你原先的调用习惯）
        # try:
        #     pres_history_emb = self.history_aggr(pres_history, pres_fac_hist_label, pool_dim=2)
        #     qnda_history_emb = self.history_aggr(qnda_history, qnda_fac_hist_label, pool_dim=2)
        # except:
        pres_history_emb = self.history_aggr(pres_history, valid_lens=pres_hist_count)
        qnda_history_emb = self.history_aggr(qnda_history, valid_lens=qnda_hist_count)

        pres_factors = batch.store['pres_factors']    # [B, #fac, D]
        qnda_factors = batch.store['qnda_factors']    # [B, #fac, D]

        pres_factors_hist = self.proj_p(pres_factors, pres_history_emb)  # [..., D]
        qnda_factors_hist = self.proj_q(qnda_factors, qnda_history_emb)  # [..., D]

        # 将四路都规约为 [B, D]，实际上是多个factor的emb平均
        own_pres = self._reduce_to_vector(pres_emb)
        own_qnda = self._reduce_to_vector(qnda_emb)
        peer_pres = self._reduce_to_vector(pres_factors_hist)
        peer_qnda = self._reduce_to_vector(qnda_factors_hist)

        # 组装给 MoE 的输入
        inputs = [own_pres, own_qnda, peer_pres, peer_qnda]

        # 适配不同 MoE 的 forward 签名
        if self.moe_accepts == "list":
            preds = self.moe(inputs)                 # 你的 MoE 接受 List[Tensor] -> [B, 1] or [B, n_out]
        else:
            stacked = torch.stack(inputs, dim=1)    # [B, 4, D]
            preds = self.moe(stacked)

        return preds
