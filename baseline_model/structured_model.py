import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 基础模块 ============

class PreNormMLPResidual(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout_hidden: float = 0.1, dropout_residual: float = 0.1,
                 activation: str = "gelu",
                 add_final_linear: bool = False):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop1 = nn.Dropout(dropout_hidden)
        self.drop2 = nn.Dropout(dropout_residual)
        self.add_final_linear = add_final_linear

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError("Unsupported activation")

        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

        if add_final_linear:
            self.final_linear = nn.Linear(out_dim, out_dim)
        else:
            self.final_linear = nn.Identity()

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        out = self.proj(x) + self.drop2(h)
        out = self.final_linear(out)   # 多一个线性层平衡输出分布
        return out


class SimpleMLP(nn.Module):
    """
    改进版：Pre-LN 残差 MLP（两层），外部再可叠加一层 LN。
    维持你的 mask 逻辑：全0行直接输出0。
    """
    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int,
                 eps: float = 1e-6, dropout_hidden: float = 0.1, dropout_residual: float = 0.1):
        super().__init__()
        self.eps = eps
        self.block = PreNormMLPResidual(in_dim, hidden_dim, output_dim,
                                        dropout_hidden=dropout_hidden,
                                        dropout_residual=dropout_residual)

    def forward(self, x):
        mask = (x.abs().sum(dim=-1, keepdim=True) > self.eps).float()
        safe_x = x * mask
        out = self.block(safe_x)
        # 对全0样本保持为0（可选，根据任务选择是否保留）
        out = out * mask
        return out, mask


class EmbeddingGatedFusion(nn.Module):
    """
    两路 embedding 的门控融合。
    - 使用 Softmax-2 或 Sigmoid(./tau) 生成权重；
    - Gate MLP 内含 LayerNorm + Dropout；
    - 掩码传递：两者皆0则输出0。
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = None,
                 use_softmax2: bool = False, gate_tau: float = 0.8, eps: float = 1e-6,
                 gate_dropout: float = 0.1):
        super().__init__()
        self.eps = eps
        self.use_softmax2 = use_softmax2
        self.tau = gate_tau
        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden_dim, 2 if use_softmax2 else 1),
        )

    def forward(self, emb1, emb2):
        b, d = emb1.shape
        mask1 = (emb1.abs().sum(dim=-1, keepdim=True) > self.eps).float()
        mask2 = (emb2.abs().sum(dim=-1, keepdim=True) > self.eps).float()
        safe1 = emb1 * mask1
        safe2 = emb2 * mask2

        x = torch.cat([safe1, safe2], dim=-1)  # [B, 2D]
        gate_logits = self.gate_mlp(x)         # [B, 1] 或 [B, 2]

        if self.use_softmax2:
            w12 = F.softmax(gate_logits, dim=-1)      # [B, 2]
            w1, w2 = w12[:, :1], w12[:, 1:]
            fused = w1 * safe1 + w2 * safe2
            w = w1  # 返回一个权重以便监控（可选）
        else:
            # Sigmoid with temperature
            w = torch.sigmoid(gate_logits / self.tau) # [B,1]
            fused = w * safe1 + (1 - w) * safe2

        final_mask = ((mask1 + mask2) > 0).float()
        fused = fused * final_mask
        return fused, w, final_mask


# ============ 你的结构封装 ============

class Structure(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 512,
                 predict_out_dim: int = 1,
                 dropout_hidden: float = 0.1,
                 dropout_residual: float = 0.1,
                 gate_dropout: float = 0.1,
                 gate_tau: float = 0.8,
                 use_softmax2_gate: bool = False):
        super().__init__()
        self.text_gate = EmbeddingGatedFusion(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,     # 也可以设成更小
            use_softmax2=use_softmax2_gate,
            gate_tau=gate_tau,
            gate_dropout=gate_dropout
        )
        self.predict = SimpleMLP(
            in_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=predict_out_dim,
            dropout_hidden=dropout_hidden,
            dropout_residual=dropout_residual
        )

    def forward(self, batch):
        pres_emb, qnda_emb = batch.store['pres_emb'], batch.store['qnda_emb']  # [B,D]
        fused, w, mask = self.text_gate(pres_emb, qnda_emb)                    # [B,D]
        pred, _ = self.predict(fused)                                          # [B,1]
        return pred.squeeze(-1)
