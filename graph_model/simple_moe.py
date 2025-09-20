# simple_moe_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# --------- Building Blocks ---------
class PreNormMLPResidual(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout_hidden: float = 0.1, dropout_residual: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop1 = nn.Dropout(dropout_hidden)
        self.drop2 = nn.Dropout(dropout_residual)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError("Unsupported activation")
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        out = self.proj(x) + self.drop2(h)
        return out


class ExpertMLP(nn.Module):
    """Expert: [B,D] -> [B,E] with PreNorm residual + dropout"""
    def __init__(self, in_dim: int, hidden_dim: int = 256, expert_out_dim: int = 128,
                 dropout_hidden: float = 0.1, dropout_residual: float = 0.1):
        super().__init__()
        self.block = PreNormMLPResidual(
            in_dim, hidden_dim, expert_out_dim,
            dropout_hidden=dropout_hidden,
            dropout_residual=dropout_residual,
            activation="gelu"
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return self.block(x)  # [B,E]


class GateNet(nn.Module):
    """Task-specific gate: LN -> MLP -> logits (with temperature)"""
    def __init__(self, in_dim: int, n_experts: int, hidden: int = 256, temperature: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.tau = temperature
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_experts)
        )
    def forward(self, x):
        logits = self.net(x) / max(1e-6, self.tau)   # [B, n_experts]
        return logits


class TaskTower(nn.Module):
    """Return logits for binary; raw value for regression"""
    def __init__(self, in_dim: int, hidden_dim: int = 128, use_ln: bool = False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*(([nn.LayerNorm(in_dim)] if use_ln else []) + layers))
    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


# --------- MMoE with regularizations and residual mixing ---------
class TaskSpecificMMoE(nn.Module):
    """
    Improvements:
      - PreNorm residual Experts
      - GateNet with LN/MLP/temperature
      - Optional top-k soft routing
      - Post-mix LayerNorm + residual from fused input (projected)
      - Entropy regularization & load-balancing loss (optional)
      - Expert dropout
    """
    def __init__(
        self,
        input_dim: int = 768,
        n_experts: int = 4,
        n_tasks: int = 5,
        expert_hidden: int = 256,
        expert_out: int = 128,
        tower_hidden: int = 128,
        task_names: List[str] = ("ret_close", "ret_sign", "ret_close_20d", "ret_sign_20d", "ret_vol_20d"),
        task_type: Optional[Dict[str, str]] = None,
        gate_hidden: int = 256,
        gate_temperature: float = 1.0,
        gate_dropout: float = 0.0,
        expert_dropout: float = 0.0,   # randomly drop expert outputs during training
        use_topk: Optional[int] = None, # e.g., 2 -> only top-2 experts per task
        postmix_ln: bool = True,
        add_residual_from_input: bool = True,
        entropy_reg_coef: float = 0.0,   # encourage high-entropy gates
        balance_reg_coef: float = 0.0    # encourage balanced expert usage
    ):
        super().__init__()
        assert n_tasks == len(task_names)
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.task_names = list(task_names)
        self.use_topk = use_topk
        self.postmix_ln = postmix_ln
        self.add_residual_from_input = add_residual_from_input
        self.entropy_reg_coef = entropy_reg_coef
        self.balance_reg_coef = balance_reg_coef
        self.expert_dropout = expert_dropout

        # experts
        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, expert_hidden, expert_out) for _ in range(n_experts)
        ])

        # gates per task
        self.gates = nn.ModuleDict({
            name: GateNet(input_dim * n_experts, n_experts, hidden=gate_hidden,
                          temperature=gate_temperature, dropout=gate_dropout)
            for name in self.task_names
        })

        # post-mix ln + residual proj
        if postmix_ln:
            self.mix_ln = nn.LayerNorm(expert_out)
        else:
            self.mix_ln = nn.Identity()
        self.res_proj = nn.Linear(input_dim * n_experts, expert_out) if add_residual_from_input else nn.Identity()

        # towers
        self.towers = nn.ModuleDict({
            name: TaskTower(expert_out, tower_hidden, use_ln=False) for name in self.task_names
        })

        if task_type is None:
            task_type = {
                "ret_close": "reg", "ret_sign": "bin",
                "ret_close_20d": "reg", "ret_sign_20d": "bin",
                "ret_vol_20d": "reg"
            }
        self.task_type = task_type

    def _mix(self, expert_stack: torch.Tensor, gate_logits: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        expert_stack: [B, n_experts, E]
        gate_logits:  [B, n_experts]
        return: mixed [B,E], probs [B,n_experts]
        """
        # optional top-k soft routing
        if self.use_topk and self.use_topk < self.n_experts:
            topk = torch.topk(gate_logits, k=self.use_topk, dim=-1)   # values, indices
            mask = torch.full_like(gate_logits, float('-inf'))
            mask.scatter_(1, topk.indices, topk.values)
            probs = F.softmax(mask, dim=-1)  # masked softmax over top-k
        else:
            probs = F.softmax(gate_logits, dim=-1)

        # optional expert dropout（均匀drop，重新归一化）
        if training and self.expert_dropout > 0.0:
            drop_mask = (torch.rand_like(probs) > self.expert_dropout).float()
            probs = probs * drop_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True).clamp_min(1e-6))

        mixed = torch.sum(expert_stack * probs.unsqueeze(-1), dim=1)  # [B,E]
        return mixed, probs

    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        inputs: list of [B,D], len = n_experts
        return: dict of logits/raw predictions
        """
        assert len(inputs) == self.n_experts
        # expert pass
        expert_feats = [exp(x) for exp, x in zip(self.experts, inputs)]  # list [B,E]
        expert_stack = torch.stack(expert_feats, dim=1)                  # [B, n_experts, E]
        fused_for_gate = torch.cat(inputs, dim=-1)                       # [B, D*n_experts]

        outputs = {}
        # collect stats for regularization
        all_probs = []

        for name in self.task_names:
            gate_logits = self.gates[name](fused_for_gate)               # [B, n_experts]
            mixed, probs = self._mix(expert_stack, gate_logits, self.training)
            if self.postmix_ln:
                mixed = self.mix_ln(mixed)
            if self.add_residual_from_input:
                mixed = mixed + self.res_proj(fused_for_gate)

            logit = self.towers[name](mixed)                             # [B] (logit or reg)
            # 不在这里 sigmoid，交给外部 loss
            outputs[name] = logit
            all_probs.append(probs.detach())

        # 可选：把 gate 正则项挂到模块属性上，训练 loop 里加到 loss
        # 1) 熵正则（希望每个样本的 gate 更“分散”）
        if self.entropy_reg_coef > 0:
            # H = -sum p log p
            H = 0.0
            for p in all_probs:
                H = H + (- (p * (p.clamp_min(1e-8)).log()).sum(dim=-1).mean())
            self.gate_entropy_reg = self.entropy_reg_coef * (H / len(all_probs))
        else:
            self.gate_entropy_reg = torch.tensor(0.0, device=expert_stack.device)

        # 2) 负载均衡正则（希望平均使用率接近均匀）
        if self.balance_reg_coef > 0:
            # 对 batch 求平均使用率，再与均匀分布 L2
            usage = torch.stack(all_probs, dim=0).mean(dim=(0,1))  # [n_experts]
            uniform = torch.full_like(usage, 1.0/self.n_experts)
            self.gate_balance_reg = self.balance_reg_coef * F.mse_loss(usage, uniform)
        else:
            self.gate_balance_reg = torch.tensor(0.0, device=expert_stack.device)

        return outputs
