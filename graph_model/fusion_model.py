import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDecayAggregator(nn.Module):
    """
    输入:
      H: [B, F, T, D] 历史嵌入
      valid_lens: [B, F] 每个 (B,F) 的有效历史条数(<=T)，可选；若无则用非零范数作为mask
      mode:
        - "mean": 掩码平均
        - "decay": 指数时间衰减平均 (最近权重大)
        - "robust": 截尾+衰减的鲁棒平均
    选项:
      decay_alpha: 指数衰减参数，越大越快衰减
      trim_q: 截尾比例 (如0.1 表示两端各去掉10%)
      use_ln: 聚合后是否做 LayerNorm（仅在存在有效样本时）
    输出:
      E: [B, F, D]
    """
    def __init__(self, mode="mean", decay_alpha=0.1, trim_q=0.1, use_ln=True, eps=1e-8):
        super().__init__()
        assert mode in ("mean", "decay", "robust")
        self.mode = mode
        self.decay_alpha = decay_alpha
        self.trim_q = trim_q
        self.use_ln = use_ln
        self.eps = eps
        self.ln = nn.LayerNorm(768)  # 若 D 非 768，可在 forward 时动态实例化或传入

    def forward(self, H: torch.Tensor, valid_lens: torch.Tensor | None = None) -> torch.Tensor:
        # H: [B,F,T,D]
        B, F, T, D = H.shape
        device = H.device
        # 构造 mask: [B,F,T,1]
        if valid_lens is not None:
            arange_t = torch.arange(T, device=device).view(1, 1, T, 1)
            valid = arange_t < valid_lens.view(B, F, 1, 1)  # True where valid
            mask = valid.float()
        else:
            # 非零范数作为有效（允许全0 padding）
            mask = (H.abs().sum(dim=-1, keepdim=True) > self.eps).float()

        Hm = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0) * mask  # 清理极值+mask
        # 默认时间索引 0..T-1, 最近= T-1
        t_idx = torch.arange(T, device=device).view(1,1,T,1).float()

        if self.mode == "mean":
            w = mask
        elif self.mode == "decay":
            # 指数衰减权重，最近样本权重大
            # 让 t=0 最远，t=T-1 最近
            dist = (T - 1 - t_idx)  # 越大越远
            w = torch.exp(- self.decay_alpha * dist) * mask
        else:  # "robust": 截尾 + 衰减
            dist = (T - 1 - t_idx)
            base_w = torch.exp(- self.decay_alpha * dist) * mask  # [B,F,T,1]
            # 基于范数做简单的异常检测：对每个 (B,F) 的 ||Hi|| 排序，截掉两端
            norms = (Hm**2).sum(dim=-1, keepdim=True).sqrt()  # [B,F,T,1]
            # 把无效位置设为 +inf，确保不影响排序
            norms_valid = torch.where(mask > 0, norms, torch.full_like(norms, float('inf')))
            # 取得分位阈值（近似实现：用 torch.quantile 需要 1.10+，或拉到 numpy）
            # 简化版：按 batch 展平排序后获取 indices（效率一般但稳）
            w = base_w.clone()
            if self.trim_q > 0:
                k_low = int(self.trim_q * T)
                k_high = max(0, T - k_low)
                # 按时间维排序（从小到大），对于无效点是 +inf，会排到最后
                sort_idx = torch.argsort(norms_valid.squeeze(-1), dim=2)  # [B,F,T]
                keep = torch.zeros_like(sort_idx, dtype=torch.bool)
                # 仅在有效计数 >= 2*k_low 时保留中间段
                eff_lens = mask.squeeze(-1).sum(dim=2).long()  # [B,F]
                for b in range(B):
                    for f in range(F):
                        L = int(eff_lens[b,f].item())
                        if L == 0:
                            continue
                        lo = min(k_low, L//2)
                        hi = max(lo, L - lo)
                        sel = sort_idx[b,f,:L]
                        keep_idx = sel[lo:hi]
                        keep[b,f,keep_idx] = True
                keep = keep.unsqueeze(-1)  # [B,F,T,1]
                w = torch.where(keep, w, torch.zeros_like(w))

        # 归一化并求加权均值
        w_sum = w.sum(dim=2, keepdim=True).clamp_min(self.eps)
        E = (Hm * w).sum(dim=2) / w_sum.squeeze(2)  # [B,F,D]

        # 对于完全无历史的 (B,F)，强制输出 0，且不做 LN
        has_any = (w_sum.squeeze(2) > self.eps)     # [B,F,1]→[B,F,1]
        if self.use_ln:
            # 只对有历史的单元做 LayerNorm
            E_ln = self.ln(E.reshape(-1, D)).reshape(B, F, D)
            E = torch.where(has_any, E_ln, E)

        return E
    

class SelfAttnMeanPooling(nn.Module):
    """
    稳健版：
    - 3D: [B, T, D]
    - 4D: [B, G, T, D]  (对每个 G 维块独立沿 T 做 self-attn，再 mean pool)
    - 自动屏蔽全 0 token；全屏蔽序列绕过 MHA 直接输出 0
    - 预 LayerNorm + 强制 FP32 的 MHA 计算（避免混合精度溢出）
    - 当有效 token 占比很低时可退化到纯 mean（可选）
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_dropout: float = 0.0,
        pre_norm: bool = True,
        force_fp32_mha: bool = True,
        fallback_mean_when_ratio_lt: float = 0.0,  # e.g. 0.05 -> 有效token比例<5%时退化为均值
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.pre_norm = pre_norm
        self.ln = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
        self.force_fp32_mha = force_fp32_mha
        self.fallback_ratio = float(fallback_mean_when_ratio_lt)
        self.eps = float(eps)

    @staticmethod
    def _kpm_from_embs(x, eps: float):
        # x: [..., T, D] -> [..., T]; True 表示该 token 全 0（屏蔽）
        return (x.abs().sum(dim=-1) <= eps)

    def forward(self, embs: torch.Tensor, pool_dim: int = 1, keepdim: bool = False) -> torch.Tensor:
        if embs.dim() == 3:
            B, T, D = embs.shape
            G = 1
            x4 = embs.unsqueeze(1)                   # [B, 1, T, D]
            squeeze_3d = True
            assert pool_dim == 1, "pool_dim = 1 for 3D input"
        elif embs.dim() == 4:
            B, G, T, D = embs.shape
            x4 = embs                                  # [B, G, T, D]
            squeeze_3d = False
            assert pool_dim == 2, "pool_dim = 2 for 4D input"
        else:
            raise ValueError("只支持 3D [B,T,D] 或 4D [B,G,T,D]")

        # 预归一化（更稳的 Q/K/V 范围）
        x4 = self.ln(x4)

        # key padding mask：True=忽略（全 0）
        kpm4 = self._kpm_from_embs(x4, eps=self.eps)    # [B, G, T]

        # 合并 B 与 G
        NG = B * G
        x = x4.reshape(NG, T, D).contiguous()           # [NG, T, D]
        kpm = kpm4.reshape(NG, T).contiguous()          # [NG, T]

        # 全屏蔽序列索引
        all_masked = kpm.all(dim=1)                     # [NG]
        any_valid = (~all_masked).any()

        # 预分配输出
        attn_out = x.new_zeros(NG, T, D)

        # 当有效 token 比例极低时（可选）直接退化为均值
        if self.fallback_ratio > 0.0:
            valid_counts = (~kpm).sum(dim=1)            # [NG]
            low_ratio = (valid_counts <= (self.fallback_ratio * T)) & (~all_masked)

            # 先处理“退化批次”：直接均值（只对有效 token）
            if low_ratio.any():
                lr_idx = low_ratio.nonzero(as_tuple=False).squeeze(-1)
                x_lr = x[lr_idx]                        # [n, T, D]
                kpm_lr = kpm[lr_idx]                    # [n, T]
                mask_lr = (~kpm_lr).unsqueeze(-1).float()
                num_lr = (x_lr * mask_lr).sum(dim=1)
                den_lr = mask_lr.sum(dim=1).clamp_min(1.0)
                mean_lr = num_lr / den_lr               # [n, D]
                # 写回到 attn_out 的每个 token（供后续统一的 pooling 使用）
                attn_out[lr_idx] = mean_lr.unsqueeze(1).expand(-1, T, -1)

            # 非退化且非全屏蔽的，走 MHA
            run_idx = ((~low_ratio) & (~all_masked)).nonzero(as_tuple=False).squeeze(-1)
        else:
            run_idx = (~all_masked).nonzero(as_tuple=False).squeeze(-1)

        # 对需要跑 MHA 的子批次做正规计算
        if run_idx.numel() > 0:
            x_run = x[run_idx]                          # [n, T, D]
            kpm_run = kpm[run_idx]                      # [n, T]
            # 强制用 FP32 计算（更稳）
            if self.force_fp32_mha:
                x_cast = x_run.float()
                attn_run, _ = self.mha(x_cast, x_cast, x_cast, key_padding_mask=kpm_run)
                attn_run = attn_run.to(x.dtype)
            else:
                attn_run, _ = self.mha(x_run, x_run, x_run, key_padding_mask=kpm_run)

            # 清理潜在 NaN/Inf（极端情况下仍可能出现）
            attn_run = torch.nan_to_num(attn_run, nan=0.0, posinf=0.0, neginf=0.0)

            attn_out[run_idx] = attn_run

        # 全屏蔽的子批次，输出保持为 0（已预分配）

        # token 级有效 mask（有效且数值有限）
        token_all_finite = torch.isfinite(attn_out).all(dim=-1)        # [NG, T]
        token_valid = (~kpm) & token_all_finite                        # [NG, T]
        mask = token_valid.unsqueeze(-1).float()                       # [NG, T, 1]

        # mean pooling
        safe = attn_out * mask
        num = safe.sum(dim=1, keepdim=keepdim)                         # [NG, D] / [NG, 1, D]
        den = mask.sum(dim=1, keepdim=keepdim).clamp_min(1.0)
        pooled = num / den

        # 还原形状
        if keepdim:
            pooled = pooled.reshape(B, G, 1, D)
            if squeeze_3d:
                pooled = pooled[:, 0:1, :, :].squeeze(1)               # [B,1,D]
        else:
            pooled = pooled.reshape(B, G, D)
            if squeeze_3d:
                pooled = pooled[:, 0, :]                               # [B, D]

        return pooled


class LabelSelfAttnMeanPooling(nn.Module):
    """
    稳健版：
    - 3D: [B, T, D]
    - 4D: [B, G, T, D]  (对每个 G 维块独立沿 T 做 self-attn，再 mean pool)
    - 自动屏蔽全 0 token；全屏蔽序列绕过 MHA 直接输出 0
    - 预 LayerNorm + 强制 FP32 的 MHA 计算（避免混合精度溢出）
    - 当有效 token 占比很低时可退化到纯 mean（可选）
    - 新增：forward(embs, labels, ...) —— labels 必填，用于对 QK logits 做加性调节
            logits = QK^T/sqrt(d) + alpha * labels + beta
            （labels 会按样本与 head 广播到 [n*h, T, T]）
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_dropout: float = 0.0,
        pre_norm: bool = True,
        force_fp32_mha: bool = True,
        fallback_mean_when_ratio_lt: float = 0.0,  # e.g. 0.05 -> 有效token比例<5%时退化为均值
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.pre_norm = pre_norm
        self.ln = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
        self.force_fp32_mha = force_fp32_mha
        self.fallback_ratio = float(fallback_mean_when_ratio_lt)
        self.eps = float(eps)

        # 训练“系数 + 偏置”，用于加到 logits 上
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _kpm_from_embs(x, eps: float):
        # x: [..., T, D] -> [..., T]; True 表示该 token 全 0（屏蔽）
        return (x.abs().sum(dim=-1) <= eps)

    def forward(
        self,
        embs: torch.Tensor,
        labels: torch.Tensor,         # <<< 必填： [B,T] 或 [B,G,T]，与时间维 T 对齐
        pool_dim: int = 1,
        keepdim: bool = False
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError("labels 是必填参数，形状需为 [B,T] 或 [B,G,T]，与 embs 的时间维 T 对齐。")

        # ------- 形状解析 -------
        if embs.dim() == 3:
            B, T, D = embs.shape
            G = 1
            x4 = embs.unsqueeze(1)                   # [B, 1, T, D]
            squeeze_3d = True
            assert pool_dim == 1, "pool_dim = 1 for 3D input"
        elif embs.dim() == 4:
            B, G, T, D = embs.shape
            x4 = embs                                  # [B, G, T, D]
            squeeze_3d = False
            assert pool_dim == 2, "pool_dim = 2 for 4D input"
        else:
            raise ValueError("只支持 3D [B,T,D] 或 4D [B,G,T,D]")

        # 预归一化（更稳的 Q/K/V 范围）
        x4 = self.ln(x4)

        # key padding mask：True=忽略（全 0）
        kpm4 = self._kpm_from_embs(x4, eps=self.eps)    # [B, G, T]

        # ------- labels 对齐到 [B,G,T] 并屏蔽 padding -------
        if labels.dim() == 2:              # [B, T]
            if G != 1:
                labels = labels.unsqueeze(1).expand(B, G, T)  # -> [B,G,T]
        elif labels.dim() == 3:            # [B, G, T]
            pass
        else:
            raise ValueError("labels 仅支持 [B,T] 或 [B,G,T]")

        # 对被 padding 的 token，将 label 置 0（不影响 logits）
        labels = torch.where(kpm4, torch.zeros_like(labels), labels)    # [B,G,T]

        # ------- 合并 B 与 G -------
        NG = B * G
        x = x4.reshape(NG, T, D).contiguous()           # [NG, T, D]
        kpm = kpm4.reshape(NG, T).contiguous()          # [NG, T]
        w  = labels.reshape(NG, T).contiguous()         # [NG, T]

        # 全屏蔽序列索引
        all_masked = kpm.all(dim=1)                     # [NG]

        # 预分配输出
        attn_out = x.new_zeros(NG, T, D)

        # （可选）低有效比例退化为均值
        if self.fallback_ratio > 0.0:
            valid_counts = (~kpm).sum(dim=1)            # [NG]
            low_ratio = (valid_counts <= (self.fallback_ratio * T)) & (~all_masked)

            if low_ratio.any():
                lr_idx = low_ratio.nonzero(as_tuple=False).squeeze(-1)
                x_lr   = x[lr_idx]                      # [n, T, D]
                kpm_lr = kpm[lr_idx]                    # [n, T]
                mask_lr = (~kpm_lr).unsqueeze(-1).float()
                num_lr = (x_lr * mask_lr).sum(dim=1)
                den_lr = mask_lr.sum(dim=1).clamp_min(1.0)
                mean_lr = num_lr / den_lr               # [n, D]
                attn_out[lr_idx] = mean_lr.unsqueeze(1).expand(-1, T, -1)

            run_idx = ((~low_ratio) & (~all_masked)).nonzero(as_tuple=False).squeeze(-1)
        else:
            run_idx = (~all_masked).nonzero(as_tuple=False).squeeze(-1)

        # ------- 需要跑 MHA 的子批 -------
        if run_idx.numel() > 0:
            x_run   = x[run_idx]        # [n, T, D]
            kpm_run = kpm[run_idx]      # [n, T]
            w_run   = w[run_idx]        # [n, T]   (key 侧偏置)

            # 构造可加到 logits 的偏置：对每个 query 位置都加同样的 key 偏置
            # 形状 [n, T(query), T(key)]，再复制到每个 head
            bias = (self.alpha * w_run.unsqueeze(1).expand(-1, T, -1) + self.beta)  # [n, T, T]
            h = self.mha.num_heads
            attn_mask = bias.repeat_interleave(h, dim=0)  # [n*h, T, T]

            if self.force_fp32_mha:
                x_cast = x_run.float()
                attn_run, _ = self.mha(
                    x_cast, x_cast, x_cast,
                    key_padding_mask=kpm_run,
                    attn_mask=attn_mask.float()
                )
                attn_run = attn_run.to(x.dtype)
            else:
                attn_run, _ = self.mha(
                    x_run, x_run, x_run,
                    key_padding_mask=kpm_run,
                    attn_mask=attn_mask.to(x_run.dtype)
                )

            attn_run = torch.nan_to_num(attn_run, nan=0.0, posinf=0.0, neginf=0.0)
            attn_out[run_idx] = attn_run

        # 全屏蔽子批保持 0

        # ------- 有效 token 的 mean pooling -------
        token_all_finite = torch.isfinite(attn_out).all(dim=-1)        # [NG, T]
        token_valid = (~kpm) & token_all_finite                        # [NG, T]
        mask = token_valid.unsqueeze(-1).float()                       # [NG, T, 1]

        safe = attn_out * mask
        num = safe.sum(dim=1, keepdim=keepdim)                         # [NG, D] / [NG, 1, D]
        den = mask.sum(dim=1, keepdim=keepdim).clamp_min(1.0)
        pooled = num / den

        # 还原形状
        if keepdim:
            pooled = pooled.reshape(B, G, 1, D)
            if squeeze_3d:
                pooled = pooled[:, 0:1, :, :].squeeze(1)               # [B,1,D]
        else:
            pooled = pooled.reshape(B, G, D)
            if squeeze_3d:
                pooled = pooled[:, 0, :]                               # [B, D]

        return pooled
