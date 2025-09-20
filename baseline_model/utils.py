import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np

# ---- 任务损失（加入 pos_weight 支持）----
def loss_function(y_true, y_pred, name='ret_close', pos_weight=None, mu=None, sd=None):
    if name in ('ret_close', 'ret_close_20d'):
        y_true = torch.log1p(y_true.clamp(min=-0.999999))
        if mu is not None and sd is not None:
            y_true = (y_true - mu) / sd
        return F.mse_loss(y_pred.squeeze(-1), y_true.squeeze(-1))

    elif name in ('ret_sign', 'ret_sign_20d'):
        y = y_true.float()
        # 去除 NaN
        valid = torch.isfinite(y)
        y = y[valid]; z = y_pred[valid]
        # {-1,1} -> {0,1}
        if y.min() < 0:
            y = (y > 0).float()
        return F.binary_cross_entropy_with_logits(z.squeeze(-1), y.squeeze(-1), pos_weight=pos_weight)

    elif name == 'ret_vol_20d':
        # 这里建议 y_true 传“方差”（若是标准差请先平方）
        y = y_true.squeeze(-1)
        z = y_pred.squeeze(-1)                  # 让模型输出 log-variance
        if mu is not None and sd is not None:
            z = z * sd + mu
        # QLIKE = z + y * exp(-z)
        return (z + y * torch.exp(-z)).mean()

    else:
        raise ValueError(f"Unknown task name: {name}")


# ---- 评估指标（返回 score 以及分类任务的 best_thr）----
def _best_threshold_by_mcc(y_true01_np, prob_np):
    ths = np.linspace(0.05, 0.95, 19)
    best = (0.5, -1.0, 0.0)  # (thr, mcc, f1)
    for t in ths:
        pred = (prob_np >= t).astype(int)
        if pred.max() == pred.min():
            mcc, f1v = 0.0, 0.0
        else:
            mcc = matthews_corrcoef(y_true01_np, pred)
            f1v = f1_score(y_true01_np, pred, zero_division=0)
        if mcc > best[1]:
            best = (t, mcc, f1v)
    return best  # thr, mcc, f1

def evaluate(y_true, y_pred, name='ret_close', mu=None, sd=None):
    """
    返回:
      score: 用于早停的标量（越大越好）
      best_thr: 仅分类任务返回最佳阈值，其它任务返回 None
    """
    if name in ('ret_close', 'ret_close_20d'):
        yt = torch.log1p(y_true.clamp(min=-0.999999)).squeeze(-1)
        yp = y_pred.squeeze(-1)
        if mu is not None and sd is not None:
            yp = yp * sd + mu
        # R^2 越大越好
        ss_res = torch.sum((yt - yp) ** 2)
        ss_tot = torch.sum((yt - mu) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        return float(r2.item()), None

    elif name in ('ret_sign', 'ret_sign_20d'):
        y = y_true.float().squeeze(-1)
        # 去 NaN
        valid = torch.isfinite(y)
        y = y[valid]
        z = y_pred[valid].squeeze(-1)
        # {-1,1}->{0,1}
        if y.min() < 0:
            y = (y > 0).float()
        prob = torch.sigmoid(z).detach().cpu().numpy()
        y01  = y.detach().cpu().numpy().astype(int)
        thr, mcc, f1v = _best_threshold_by_mcc(y01, prob)
        # 用 MCC 作为早停指标（越大越好）
        return mcc, thr

    elif name == 'ret_vol_20d':
        # 评分用 -QLIKE（越大越好）
        y = y_true.squeeze(-1).clamp(min=1e-12)
        z = y_pred.squeeze(-1)   # logvar
        if mu is not None and sd is not None:
            z = z * sd + mu

        qlike = (z + y * torch.exp(-z)).mean().item()
        return -qlike, None

    else:
        raise ValueError(f"Unknown task name: {name}")