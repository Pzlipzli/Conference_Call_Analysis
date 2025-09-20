import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SimpleMLP(nn.Module):
    """
    普通 MLP，自动对全0输入加 mask，并带 LayerNorm
    """
    def __init__(self, hidden_dim, output_dim, eps=1e-6, dropout=0.1):
        super().__init__()
        self.eps = eps
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),   # 加入 LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_list):
        """
        x_list: [tensor1, tensor2, ...] 各 [B, d_i]
        返回: out:[B,output_dim], mask:[B,1]
        mask = 1 表示有效输入，mask = 0 表示全0
        """
        x = torch.cat(x_list, dim=-1)   # 拼接输入
        mask = (x.abs().sum(dim=-1, keepdim=True) > self.eps).float()
        safe_x = x * mask               # 对全0输入置 0
        out = self.model(safe_x)
        return out, mask
    

class CrossABThenMLP(nn.Module):
    """
    输入: [t, A, B]
      - t: [B, h]
      - A: [B, T_a, h]
      - B: [B, T_b, h]
    流程:
      1) A←B, B←A 互相 cross-attention
      2) 对 A' 与 B' 做平均池化 (可自动mask全零token)
      3) 拼接 [t, a_pool, b_pool] -> LayerNorm -> MLP -> 预测
    """
    def __init__(self,
                 h: int,
                 nhead: int = 4,
                 hidden: int = 512,
                 out_dim: int = 1,
                 dropout: float = 0.1,
                 use_auto_mask: bool = True,
                 zero_eps: float = 1e-8):
        super().__init__()
        self.use_auto_mask = use_auto_mask
        self.zero_eps = zero_eps

        self.attn_ab = nn.MultiheadAttention(embed_dim=h * 2, num_heads=nhead,
                                             dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(embed_dim=h * 2, num_heads=nhead,
                                             dropout=dropout, batch_first=True)

        # 拼接后维度为 3h，先做 LayerNorm 再过 MLP
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    @staticmethod
    def _auto_padding_mask(x: torch.Tensor, eps: float) -> torch.Tensor:
        """
        x: [B, T, h] -> 返回 key_padding_mask [B, T] (True=mask该token)
        依据: token 全零/近似全零则视作 padding
        """
        return (x.abs().sum(dim=-1) <= eps)

    def forward(self, inputs: List[torch.Tensor]):
        assert len(inputs) == 3, "inputs 应为 [t, A, B]"
        t, A, B = inputs

        # --- 1) 自动mask（可选） ---
        kpm_A = self._auto_padding_mask(A, self.zero_eps) if self.use_auto_mask else None  # [B, T_a]
        kpm_B = self._auto_padding_mask(B, self.zero_eps) if self.use_auto_mask else None  # [B, T_b]

        # --- 2) 互相 cross-attention ---
        # A' = Attn(query=A, key=B, value=B)
        A_out, _ = self.attn_ab(query=A, key=B, value=B, key_padding_mask=kpm_B)
        # B' = Attn(query=B, key=A, value=A)
        B_out, _ = self.attn_ba(query=B, key=A, value=A, key_padding_mask=kpm_A)

        # --- 3) 拼接 -> LayerNorm -> MLP ---
        fused = torch.cat([t, A_out, B_out], dim=-1)          
        fused_dim = fused.size(-1)
        self.pre_ln = nn.LayerNorm(fused_dim).to(device=fused.device, dtype=fused.dtype)        
        pred = self.mlp(fused)                                   # [B, out_dim]

        return pred, fused
    

class LabelBiasedSelfAttnMeanPooling(nn.Module):
    """
    Label-Biased Self-Attention + Mean Pooling
    - 3D 输入: [B, T, D]   (pool_dim=1)
    - 4D 输入: [B, G, T, D](pool_dim=2，对每个 G 独立沿 T 做 self-attn + pooling)
    - 通过 label 生成上下文向量，偏置 query:  q_bias = f(label)  →  mha(query = x + q_bias, key=x, value=x)
    - 忽略全0 token（kpm）并在 pooling 时仅对有效 token 求平均
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 attn_dropout: float = 0.0,
                 label_hidden: int = 64,
                 bias_scale_init: float = 1.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        # 自注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # [N, T, D]
        )

        # 用 label 生成 query 偏置向量 q_bias: [NG, D]
        # 这里把标量/小向量的 label 投影为与 embed_dim 对齐的向量
        self.label_to_qbias = nn.Sequential(
            nn.LayerNorm(1),                  # label 归一化（标量也可稳定一些）
            nn.Linear(1, label_hidden),
            nn.ReLU(),
            nn.Linear(label_hidden, embed_dim)
        )
        # 可学习缩放，控制 label 偏置强度
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale_init, dtype=torch.float32))

    @staticmethod
    def _kpm_from_embs(x: torch.Tensor, eps: float):
        # x: [..., T, D] -> [ ..., T ]  True=需要屏蔽（全0）
        return (x.abs().sum(dim=-1) <= eps)

    def _prep_labels(self, labels: torch.Tensor, B: int, G: int, device, dtype,
                    reduce: str = "mean"):
        """
        目标：返回 [B, G, 1]
        允许输入：
        [B], [B,1], [B,G], [B,G,1], [B,1,1],
        [B,T], [B,G,T],
        [B,G,T,1]  <-- 新增支持
        reduce: 当带 T 维时的降维方式：'mean' | 'last' | 'sum'
        """
        L = labels.to(device=device, dtype=dtype)

        # 先把末尾多余的 size=1 维去掉一层（避免 [B,G,T,1] 阻碍判断）
        if L.dim() >= 2 and L.size(-1) == 1:
            L = L[..., 0]  # e.g. [B,G,T,1]->[B,G,T], [B,G,1]->[B,G], [B,1]->[B]

        if L.dim() == 1:
            # [B] -> [B,G,1]
            assert L.size(0) == B, f"labels batch={L.size(0)} 与 B={B} 不符"
            L = L.view(B, 1, 1).expand(B, G, 1)

        elif L.dim() == 2:
            # [B,1] 或 [B,G] 或 [B,T]
            assert L.size(0) == B, f"labels batch={L.size(0)} 与 B={B} 不符"
            if L.size(1) == 1:
                # [B,1] -> [B,G,1]
                L = L.view(B, 1, 1).expand(B, G, 1)
            elif L.size(1) == G:
                # [B,G] -> [B,G,1]
                L = L.unsqueeze(-1)
            else:
                # 视为 [B,T]：沿 T 降维 -> [B,1] -> [B,G,1]
                if reduce == "mean":
                    L = L.mean(dim=1, keepdim=True)
                elif reduce == "last":
                    L = L[:, -1:].contiguous()
                elif reduce == "sum":
                    L = L.sum(dim=1, keepdim=True)
                else:
                    raise ValueError(f"未知 reduce: {reduce}")
                L = L.view(B, 1, 1).expand(B, G, 1)

        elif L.dim() == 3:
            # [B,G] / [B,G,T] / [B,1,1]
            assert L.size(0) == B, f"labels batch={L.size(0)} 与 B={B} 不符"
            if L.size(1) == G and L.size(2) == 1:
                # [B,G,1] -> OK
                pass
            elif L.size(1) == 1 and L.size(2) == 1:
                # [B,1,1] -> [B,G,1]
                L = L.expand(B, G, 1)
            elif L.size(1) == G:
                # [B,G,T]：沿 T 降维 -> [B,G,1]
                if reduce == "mean":
                    L = L.mean(dim=2, keepdim=True)
                elif reduce == "last":
                    L = L[:, :, -1:].contiguous()
                elif reduce == "sum":
                    L = L.sum(dim=2, keepdim=True)
                else:
                    raise ValueError(f"未知 reduce: {reduce}")
            else:
                raise ValueError(f"无法解释的 labels 形状: {tuple(L.shape)} (期望第二维=G)")

        elif L.dim() == 4:
            # 你当前报错的情况：[B,G,T,1] -> squeeze last -> [B,G,T]，再按 3D 逻辑处理
            assert L.size(0) == B and L.size(1) == G, \
                f"labels 形状 {tuple(L.shape)} 与 B={B}, G={G} 不符"
            # 已经在最开始 squeeze 掉了最后一维，这里应该变成 [B,G,T]
            if reduce == "mean":
                L = L.mean(dim=2, keepdim=True)   # [B,G,1]
            elif reduce == "last":
                L = L[:, :, -1:].contiguous()     # [B,G,1]
            elif reduce == "sum":
                L = L.sum(dim=2, keepdim=True)    # [B,G,1]
            else:
                raise ValueError(f"未知 reduce: {reduce}")

        else:
            raise ValueError(f"labels 维度 {L.dim()} 不支持: 形状 {tuple(L.shape)}")

        return L  # [B,G,1]

    def forward(self,
                embs: torch.Tensor,
                labels: torch.Tensor,
                pool_dim: int = 1,
                eps: float = 1e-6,
                keepdim: bool = False) -> torch.Tensor:
        """
        embs: 3D [B,T,D] 或 4D [B,G,T,D]
        labels: 与 batch 对齐的收益率标签（见 _prep_labels 支持的形状）
        返回：
          3D: [B, D] 或 [B, 1, D] (keepdim=True)
          4D: [B, G, D] 或 [B, G, 1, D] (keepdim=True)
        """
        if embs.dim() == 3:
            B, T, D = embs.shape
            G = 1
            x4 = embs.unsqueeze(1)               # [B, 1, T, D]
            squeeze_3d = True
            assert pool_dim == 1, "pool_dim = 1 for 3D input"
        elif embs.dim() == 4:
            B, G, T, D = embs.shape
            x4 = embs                              # [B, G, T, D]
            squeeze_3d = False
            assert pool_dim == 2, "pool_dim = 2 for 4D input"
        else:
            raise ValueError("只支持 3D [B,T,D] 或 4D [B,G,T,D]")

        device, dtype = embs.device, embs.dtype

        # 1) key padding mask: True 表示该 token 被忽略（全零）
        kpm4 = self._kpm_from_embs(x4, eps=eps)        # [B, G, T]

        # 2) 展平 B 与 G 方便送入 MHA
        NG = B * G
        x = x4.reshape(NG, T, D).contiguous()          # [NG, T, D]
        kpm = kpm4.reshape(NG, T).contiguous()         # [NG, T]

        # 3) 处理 labels -> 生成 query 偏置向量 q_bias: [NG, D]
        L = self._prep_labels(labels, B, G, device, dtype)   # [B,G,1]
        L_flat = L.reshape(NG, 1)                            # [NG, 1]
        q_bias = self.label_to_qbias(L_flat)                 # [NG, D]
        q_bias = (self.bias_scale * q_bias).unsqueeze(1)     # [NG, 1, D] 对整条序列共享

        # 4) Label-biased self-attention
        #    在 query 侧加偏置：query = x + q_bias
        attn_out, _ = self.mha(query=x + q_bias, key=x, value=x, key_padding_mask=kpm)  # [NG, T, D]

        # 5) 数值清洗
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)

        # 6) token 级有效性：该 token 非 padding 且输出 finite
        token_all_finite = torch.isfinite(attn_out).all(dim=-1)        # [NG, T]
        token_valid = (~kpm) & token_all_finite                        # [NG, T]
        mask = token_valid.unsqueeze(-1).float()                       # [NG, T, 1]

        # 7) 仅对有效 token 做 mean pooling；若无有效 token，则输出 0
        safe = attn_out * mask                                         # [NG, T, D]
        num = safe.sum(dim=1, keepdim=keepdim)                         # [NG, D] 或 [NG, 1, D]
        den = mask.sum(dim=1, keepdim=keepdim).clamp(min=1.0)
        pooled = num / den                                             # [NG, D] 或 [NG, 1, D]

        # 8) 还原形状
        if keepdim:
            pooled = pooled.reshape(B, G, 1, D)                        # [B, G, 1, D]
            if squeeze_3d:
                pooled = pooled[:, 0:1, :, :].squeeze(1)               # [B, 1, D]
        else:
            pooled = pooled.reshape(B, G, D)                           # [B, G, D]
            if squeeze_3d:
                pooled = pooled[:, 0, :]                               # [B, D]

        return pooled


class LabelTempBiasSelfAttn(nn.Module):
    """
    Label-biased Self-Attention
    - 两个可学习映射:
        1) tau(label): 标量温度，控制分布尖/平
        2) b_key(label): 列偏置向量（长度 T），对每一行 logits 加同一列向量
    - 输入:
        x: [B,T,D] 或 [B,G,T,D]
        labels: [B]/[B,1]/[B,G]/[B,G,1]/[B,T]/[B,G,T]/[B,G,T,1]
        key_padding_mask(可选): [B,T] 或 [B,G,T]，True=屏蔽
    - 输出:
        若 return_seq=True: 与 x 同形状（[B,T,D] 或 [B,G,T,D]）
        若 return_seq=False: mean pooling 后的 [B,D] 或 [B,G,D]
        可选返回 attn_weights（平均到 head 后的 [B,T,T] 或 [B,G,T,T]）
    """
    def __init__(self, d_model: int, nhead: int = 4,
                 temp_hidden: int = 32,
                 bias_hidden: int = 64,
                 dropout: float = 0.1,
                 max_len: int = 1024,
                 reduce: str = 'mean',
                 return_seq: bool = False,
                 return_attn: bool = False):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        assert reduce in ('mean', 'last')
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.do = nn.Dropout(dropout)
        self.reduce = reduce
        self.return_seq = return_seq
        self.return_attn = return_attn
        self.tau_eps = 1e-3

        # QKV
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # tau(label) 标量
        self.tau_mlp = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, temp_hidden), nn.ReLU(),
            nn.Linear(temp_hidden, 1)
        )

        # b_key(label) 标量 → 乘以可学习的位置嵌入得到列偏置
        self.bias_mlp = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, bias_hidden), nn.ReLU(),
            nn.Linear(bias_hidden, 1)
        )
        # 位置嵌入（会按需自动扩容）
        self.register_parameter('pos_embed', nn.Parameter(torch.zeros(1, max_len)))
        nn.init.normal_(self.pos_embed, std=0.02)

    # ---------- utils ----------
    @staticmethod
    def _merge_BG(x: torch.Tensor) -> Tuple[torch.Tensor,int,int,int,int,bool]:
        if x.dim() == 3:
            B, T, D = x.shape
            G = 1
            return x, B, G, T, D, True
        elif x.dim() == 4:
            B, G, T, D = x.shape
            X = x.reshape(B*G, T, D).contiguous()
            return X, B, G, T, D, False
        else:
            raise ValueError("x must be [B,T,D] or [B,G,T,D]")

    @staticmethod
    def _auto_kpm(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # 全0 token 作为 padding
        return (X.abs().sum(dim=-1) <= eps)  # [NG, T], True=mask

    @staticmethod
    def _expand_kpm(kpm: Optional[torch.Tensor], B: int, G: int, T: int, device, dtype):
        if kpm is None:
            return None
        M = kpm.to(device=device, dtype=torch.bool)
        if M.dim() == 2:         # [B,T] -> [B,G,T] -> [NG,T]
            M = M.view(B, 1, T).expand(B, G, T).reshape(B*G, T)
        elif M.dim() == 3:       # [B,G,T] -> [NG,T]
            assert M.size(0) == B and M.size(1) == G and M.size(2) == T
            M = M.reshape(B*G, T)
        else:
            raise ValueError("key_padding_mask 必须是 [B,T] 或 [B,G,T]")
        return M

    @staticmethod
    def _safesqueeze_last(L: torch.Tensor):
        return L[..., 0] if (L.dim() >= 2 and L.size(-1) == 1) else L

    @staticmethod
    def _reduce_time(L: torch.Tensor, dim: int, how: str):
        if how == 'mean':
            return L.mean(dim=dim)
        elif how == 'last':
            return L.select(dim, L.size(dim)-1)
        else:
            raise ValueError("reduce must be mean/last")

    @staticmethod
    def _grow_pos_embed(param: nn.Parameter, need_T: int) -> nn.Parameter:
        cur_T = param.size(1)
        if need_T <= cur_T:
            return param
        # 扩容：新位置初始化为正态噪声
        new_pe = torch.zeros(param.size(0), need_T, device=param.device, dtype=param.dtype)
        nn.init.normal_(new_pe, std=0.02)
        new_pe[:, :cur_T] = param.data
        param.data = new_pe
        return param

    def _prep_labels(self, labels: torch.Tensor, B: int, G: int, T: int, device, dtype):
        """
        统一到序列级标量: [NG,1]
        支持 [B]/[B,1]/[B,G]/[B,G,1]/[B,T]/[B,G,T]/[B,G,T,1]
        """
        L = labels.to(device=device, dtype=dtype)
        L = self._safesqueeze_last(L)

        if L.dim() == 1:                         # [B]
            assert L.size(0) == B
            L = L.view(B, 1).expand(B, G)        # [B,G]
        elif L.dim() == 2:                       # [B,1]/[B,G]/[B,T]
            assert L.size(0) == B
            if L.size(1) == 1:
                L = L.expand(B, G)               # [B,G]
            elif L.size(1) == G:
                pass                             
            else:                                 # [B,T]
                L = self._reduce_time(L, dim=1, how=self.reduce).view(B, 1).expand(B, G)
        elif L.dim() == 3:                       # [B,G,T]
            assert L.size(0) == B and L.size(1) == G
            L = self._reduce_time(L, dim=2, how=self.reduce)   # [B,G]
        else:
            raise ValueError(f"unsupported label shape {tuple(L.shape)}")

        return L.reshape(B*G, 1)  # [NG,1]

    # ---------- forward ----------
    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor,
                pool_dim: int = 1,
                key_padding_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-6):
        """
        返回:
        y_seq 或 y_pool, 以及可选 attn_weights（平均到 head）
        约束:
        - 3D 输入 [B,T,D] 仅允许 pool_dim=1 并在 dim=1 (T) 上池化
        - 4D 输入 [B,G,T,D] 仅允许 pool_dim=2 并在 dim=2 (T) 上池化
        """
        # ---- 形状展开 ----
        X, B, G, T, D, squeeze_3d = self._merge_BG(x)   # X:[NG,T,D]
        NG = B * G
        device, dtype = X.device, X.dtype

        # ---- pool_dim 校验 ----
        if squeeze_3d:
            assert pool_dim == 1, f"3D 输入 [B,T,D] 仅支持 pool_dim=1，收到 {pool_dim}"
        else:
            assert pool_dim == 2, f"4D 输入 [B,G,T,D] 仅支持 pool_dim=2，收到 {pool_dim}"

        # ---- KPM：外部 kpm 与“全零自动 mask”合并 ----
        kpm_ext = self._expand_kpm(key_padding_mask, B, G, T, device, dtype)
        kpm_auto = self._auto_kpm(X, eps=eps)
        kpm = kpm_auto if kpm_ext is None else (kpm_auto | kpm_ext)      # [NG,T], True=mask

        # ---- Q K V ----
        Q = self.Wq(X).reshape(NG, T, self.nhead, self.head_dim).transpose(1, 2)  # [NG,H,T,dh]
        K = self.Wk(X).reshape(NG, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.Wv(X).reshape(NG, T, self.nhead, self.head_dim).transpose(1, 2)

        # ---- (1) tau(label): logits / tau ----
        L_seq = self._prep_labels(labels, B, G, T, device, dtype)    # [NG,1]
        tau_raw = self.tau_mlp(L_seq)                                # [NG,1]
        tau = F.softplus(tau_raw) + self.tau_eps                     # >0
        Q = Q / tau.view(NG, 1, 1, 1).sqrt()

        # ---- (2) b_key(label): 列偏置 ----
        b_scalar = self.bias_mlp(L_seq).view(NG, 1)                  # [NG,1]
        self._grow_pos_embed(self.pos_embed, T)                      # 确保 pos_embed 足够长
        b_key = b_scalar * self.pos_embed[:, :T]                     # [1,T] -> [NG,T]
        b_key = b_key.expand(NG, T).contiguous()                     # [NG,T]

        # ---- SDPA ----
        q = Q.reshape(NG*self.nhead, T, self.head_dim)
        k = K.reshape(NG*self.nhead, T, self.head_dim)
        v = V.reshape(NG*self.nhead, T, self.head_dim)

        # 构造加性 mask：列偏置 + key padding（被屏蔽的 key → -inf）
        logits_bias = b_key.unsqueeze(1).expand(NG, T, T)            # [NG,T,T]
        neg_inf = torch.finfo(dtype).min
        key_mask = kpm.unsqueeze(1).expand(NG, T, T)                 # True 的列置 -inf
        logits_bias = logits_bias.masked_fill(key_mask, neg_inf)     # [NG,T,T]

        # 🔧 修正：沿 head 维复制，再 reshape
        attn_mask = logits_bias.unsqueeze(1).expand(NG, self.nhead, T, T) \
                                    .reshape(NG * self.nhead, T, T)  # [NG*H, T, T]

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
            )                                                         # [NG*H,T,dh]

        y = attn.reshape(NG, self.nhead, T, self.head_dim).transpose(1, 2).contiguous()
        y = y.reshape(NG, T, self.d_model)
        y = self.Wo(self.do(y))                                      # [NG,T,D]

        # ---- 注意力权重（可选，平均到 head）----
        attn_weights = None
        if self.return_attn:
            qh = q / (self.head_dim ** 0.5)
            logits = torch.matmul(qh, k.transpose(-2, -1))           # [NG*H,T,T]
            logits = logits + logits_bias.reshape(NG*self.nhead, T, T)
            weights = torch.softmax(logits, dim=-1)                  # [NG*H,T,T]
            weights = weights.reshape(NG, self.nhead, T, T).mean(dim=1)  # [NG,T,T]
            attn_weights = weights if squeeze_3d else weights.reshape(B, G, T, T)

        # ---- 还原形状 / 池化 ----
        if self.return_seq:
            if squeeze_3d:
                return (y, attn_weights) if self.return_attn else y   # [B,T,D]
            else:
                y = y.reshape(B, G, T, self.d_model)                  # [B,G,T,D]
                return (y, attn_weights) if self.return_attn else y
        else:
            # mean pooling：对 3D/4D 都是在时间维 T 上池化
            # 这里显式用 pool_dim 判断，保持和你的接口语义一致
            # 3D: pool_dim==1 ⇒ 沿 dim=1 池化；4D: pool_dim==2 ⇒ 每个 group 内沿时间池化
            mask = (~kpm).unsqueeze(-1).float()                       # [NG,T,1]
            safe = y * mask
            num = safe.sum(dim=1)                                     # [NG,D] (沿 T)
            den = mask.sum(dim=1).clamp(min=1.0)
            pooled = num / den                                        # [NG,D]
            if squeeze_3d:
                return (pooled, attn_weights) if self.return_attn else pooled  # [B,D]
            else:
                pooled = pooled.view(B, G, self.d_model)              # [B,G,D]
                return (pooled, attn_weights) if self.return_attn else pooled

