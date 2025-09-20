import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


# ---- 示例：嵌入你的多任务损失 ----
def compute_multitask_loss(
    # 5 个任务的预测
    pred_ret: torch.Tensor,             # [B,1] or [B]    （log-return 标准化空间 or 直出 log-return）
    pred_sign: torch.Tensor,            # [B,1] or [B]    （logits）
    pred_ret_20: torch.Tensor,          # [B,1] or [B]    （同上）
    pred_sign_20: torch.Tensor,         # [B,1] or [B]    （logits）
    pred_var: torch.Tensor,             # [B,1] or [B]    （log-variance，若标准化过需反标准化）

    # 5 个任务的真实值
    ret_close: torch.Tensor,            # [B,1] or [B]    （原始收益率）
    ret_sign: torch.Tensor,             # [B,1] or [B]    （{-1,1} 或 {0,1}）
    ret_close_20d: torch.Tensor,        # [B,1] or [B]
    ret_sign_20d: torch.Tensor,         # [B,1] or [B]
    ret_var_20d: torch.Tensor,          # [B,1] or [B]    （20日**方差**，>0）

    # 反标准化所需训练集统计量（均在 log 空间上）
    pos_weight_1=None, pos_weight_2=None,                                                           # 分类任务的正样本权重
    mu: float | torch.Tensor | None = None, sd: float | torch.Tensor | None = None,                 # 对 log(1+ret_close) 的训练集 μ/σ
    mu_20: float | torch.Tensor | None = None, sd_20: float | torch.Tensor | None = None,           # 对 log(1+ret_close_20d) 的训练集 μ/σ
    mu_var_20: float | torch.Tensor | None = None, sd_var_20: float | torch.Tensor | None = None,   # 对 log(var_20d) 的训练集 μ/σ

    w_close=1.0, w_sign=1.0, w_close_20d=1.0, w_sign_20d=1.0, w_vol = 1.0,                          # 各任务权重
):
    # 1) ret_close：建议你已在外部转 log-return；若没有，这里也可 log1p 再 MSE
    y_close_log = torch.log1p(ret_close.clamp(min=-0.999999))
    if mu is not None and sd is not None:
        y_close_log = (y_close_log - mu) / sd
    loss_close = F.mse_loss(pred_ret.squeeze(-1), y_close_log.squeeze(-1))

    # 2) ret_sign：BCEWithLogits（若 pred_sign 是概率会自动转 logits）
    y_sign = ret_sign.float()
    if y_sign.min() < 0:
        y_sign = (y_sign + 1.0) * 0.5

    valid = torch.isfinite(y_sign)
    y = y_sign[valid]; z = pred_sign[valid]
    loss_sign = F.binary_cross_entropy_with_logits(z.squeeze(-1), y.squeeze(-1), pos_weight=pos_weight_1)

    # 3) ret_close_20d： log1p 再 MSE
    y_close_20d_log = torch.log1p(ret_close_20d.clamp(min=-0.999999))
    if mu_20 is not None and sd_20 is not None:
        y_close_20d_log = (y_close_20d_log - mu_20) / sd_20
    loss_close_20d = F.mse_loss(pred_ret_20.squeeze(-1), y_close_20d_log.squeeze(-1))

    # 4) ret_sign_20d：BCEWithLogits
    y_sign_20d = ret_sign_20d.float()
    if y_sign_20d.min() < 0:
        y_sign_20d = (y_sign_20d + 1.0) * 0.5
    
    valid_20 = torch.isfinite(y_sign_20d)
    y_20 = y_sign_20d[valid_20]; z_20 = pred_sign_20[valid_20]
    loss_sign_20d = F.binary_cross_entropy_with_logits(z_20.squeeze(-1), y_20.squeeze(-1), pos_weight=pos_weight_2)

    # 5) vol_20d：qlike
    y_var_20d = ret_var_20d.squeeze(-1)
    y_var_pred = pred_var.squeeze(-1)
    if mu_var_20 is not None and sd_var_20 is not None:
        y_var_pred = y_var_pred * sd_var_20 + mu_var_20
    loss_vol = (y_var_pred + y_var_20d * torch.exp(-y_var_pred)).mean()

    total = w_close*loss_close + w_sign*loss_sign + w_close_20d*loss_close_20d + w_sign_20d*loss_sign_20d + w_vol*loss_vol
    return total, {
        "ret_close_mse":    loss_close.detach(),
        "ret_sign_bce":     loss_sign.detach(),
        "ret_close_20d_mse":loss_close_20d.detach(),
        "ret_sign_20d_bce": loss_sign_20d.detach(),
        "vol_qlike":        loss_vol.detach(),
        "total":            total.detach(),
    }


def _best_threshold_by_mcc(y_true01: np.ndarray, prob: np.ndarray):
    """在 [0,1] 上网格搜索阈值，最大化 MCC，同时回报 F1。"""
    from sklearn.metrics import matthews_corrcoef, f1_score

    # 过滤 NaN
    m = np.isfinite(y_true01) & np.isfinite(prob)
    y = y_true01[m].astype(int)
    p = prob[m].astype(float)

    if y.size == 0:
        return 0.5, 0.0, 0.0

    # 阈值网格（可按需加密）
    thr_grid = np.linspace(0.0, 1.0, 41)
    best_mcc, best_f1, best_thr = -1.0, 0.0, 0.5
    for thr in thr_grid:
        pred = (p >= thr).astype(int)
        # 若全为同一类，F1/MCC 可能为 0/负，仍然允许比较
        try:
            mcc = matthews_corrcoef(y, pred)
        except Exception:
            mcc = 0.0
        f1 = f1_score(y, pred, zero_division=0)
        if mcc > best_mcc:
            best_mcc, best_f1, best_thr = mcc, f1, thr
    return float(best_thr), float(best_mcc), float(best_f1)


def evaluate_multitask(
    # 5 个任务的预测
    pred_ret: torch.Tensor,             # [B,1] or [B]    （log-return 标准化空间 or 直出 log-return）
    pred_sign: torch.Tensor,            # [B,1] or [B]    （logits）
    pred_ret_20: torch.Tensor,          # [B,1] or [B]    （同上）
    pred_sign_20: torch.Tensor,         # [B,1] or [B]    （logits）
    pred_vol: torch.Tensor,             # [B,1] or [B]    （log-variance，若标准化过需反标准化）

    # 5 个任务的真实值
    ret_close: torch.Tensor,            # [B,1] or [B]    （原始收益率）
    ret_sign: torch.Tensor,             # [B,1] or [B]    （{-1,1} 或 {0,1}）
    ret_close_20d: torch.Tensor,        # [B,1] or [B]
    ret_sign_20d: torch.Tensor,         # [B,1] or [B]
    ret_var_20d: torch.Tensor,          # [B,1] or [B]    （20日**方差**，>0）

    # 反标准化所需训练集统计量（均在 log 空间上）
    mu: float | torch.Tensor | None = None, sd: float | torch.Tensor | None = None,                 # 对 log(1+ret_close) 的训练集 μ/σ
    mu_20: float | torch.Tensor | None = None, sd_20: float | torch.Tensor | None = None,           # 对 log(1+ret_close_20d) 的训练集 μ/σ
    mu_var_20: float | torch.Tensor | None = None, sd_var_20: float | torch.Tensor | None = None,   # 对 log(var_20d) 的训练集 μ/σ

    # 其它
    search_threshold: bool = True,   # 是否在验证集上网格找最佳阈值
):
    """
    返回:
      metrics: dict
        {
          'ret_close':     {'R2_OOS': ...},
          'ret_sign':      {'MCC': ..., 'F1': ..., 'best_thr': ...},
          'ret_close_20d': {'R2_OOS': ...},
          'ret_sign_20d':  {'MCC': ..., 'F1': ..., 'best_thr': ...},
          'ret_vol_20d':   {'QLIKE': ...}
        }
      best_thresholds: dict  # 仅分类任务
        {'ret_sign': thr1, 'ret_sign_20d': thr2}
    """
    eps = 1e-12
    device = pred_ret.device

    # ------- 1) ret_close: OOS R^2 -------
    yt = torch.log1p(ret_close.clamp(min=-0.999999)).view(-1).to(device)
    yp = pred_ret.view(-1).to(device)
    if (mu is not None) and (sd is not None):
        # 若 yp 在标准化空间（基于训练集的 log-space 标准化），则反标准化
        yp = yp * float(sd) + float(mu)
        baseline = float(mu)   # OOS：以训练集均值为基准
    else:
        # 若 yp 已是 log-space 原尺度（未标准化），OOS 基准仍需训练均值；若无则退化为当前集均值
        baseline = float(mu) if (mu is not None) else float(yt.mean().item())

    ss_res = torch.sum((yt - yp) ** 2)
    ss_tot = torch.sum((yt - baseline) ** 2)
    r2_close = float((1.0 - ss_res / (ss_tot + eps)).item())

    # ------- 2) ret_sign: MCC+F1，阈值搜索 -------
    y = ret_sign.view(-1).to(device).float()
    if torch.isfinite(y).any():
        if y.min() < 0:  # {-1,1} -> {0,1}
            y = (y > 0).float()
        z = pred_sign.view(-1).to(device)   # logits
        prob = torch.sigmoid(z).detach().cpu().numpy()
        y01  = y.detach().cpu().numpy().astype(int)

        if search_threshold:
            thr1, mcc1, f1_1 = _best_threshold_by_mcc(y01, prob)
        else:
            thr1 = 0.5
            from sklearn.metrics import matthews_corrcoef, f1_score
            yhat = (prob >= thr1).astype(int)
            mcc1 = float(matthews_corrcoef(y01, yhat))
            f1_1 = float(f1_score(y01, yhat, zero_division=0))
    else:
        thr1, mcc1, f1_1 = 0.5, 0.0, 0.0

    # ------- 3) ret_close_20d: OOS R^2 -------
    yt20 = torch.log1p(ret_close_20d.clamp(min=-0.999999)).view(-1).to(device)
    yp20 = pred_ret_20.view(-1).to(device)
    if (mu_20 is not None) and (sd_20 is not None):
        yp20 = yp20 * float(sd_20) + float(mu_20)
        baseline20 = float(mu_20)
    else:
        baseline20 = float(mu_20) if (mu_20 is not None) else float(yt20.mean().item())

    ss_res20 = torch.sum((yt20 - yp20) ** 2)
    ss_tot20 = torch.sum((yt20 - baseline20) ** 2)
    r2_close_20 = float((1.0 - ss_res20 / (ss_tot20 + eps)).item())

    # ------- 4) ret_sign_20d: MCC+F1，阈值搜索 -------
    y20 = ret_sign_20d.view(-1).to(device).float()
    if torch.isfinite(y20).any():
        if y20.min() < 0:
            y20 = (y20 > 0).float()
        z20 = pred_sign_20.view(-1).to(device)
        prob20 = torch.sigmoid(z20).detach().cpu().numpy()
        y01_20 = y20.detach().cpu().numpy().astype(int)

        if search_threshold:
            thr2, mcc2, f1_2 = _best_threshold_by_mcc(y01_20, prob20)
        else:
            thr2 = 0.5
            from sklearn.metrics import matthews_corrcoef, f1_score
            yhat20 = (prob20 >= thr2).astype(int)
            mcc2 = float(matthews_corrcoef(y01_20, yhat20))
            f1_2 = float(f1_score(y01_20, yhat20, zero_division=0))
    else:
        thr2, mcc2, f1_2 = 0.5, 0.0, 0.0

    # ------- 5) vol_20d: QLIKE（更小更好；报告正向指标时可用 -QLIKE） -------
    # 这里 pred_vol 表示 log-variance；若在标准化空间，需要反标准化
    z = pred_vol.view(-1).to(device)  # log(var)
    if (mu_var_20 is not None) and (sd_var_20 is not None):
        z = z * float(sd_var_20) + float(mu_var_20)
    yvar = ret_var_20d.view(-1).to(device).clamp(min=eps)  # realized variance
    qlike = float((z + yvar * torch.exp(-z)).mean().item())

    metrics = {
        'ret_close':      {'R2_OOS': r2_close},
        'ret_sign':       {'MCC': mcc1, 'F1': f1_1, 'best_thr': thr1},
        'ret_close_20d':  {'R2_OOS': r2_close_20},
        'ret_sign_20d':   {'MCC': mcc2, 'F1': f1_2, 'best_thr': thr2},
        'ret_vol_20d':    {'QLIKE': qlike}
    }
    best_thresholds = {'ret_sign': thr1, 'ret_sign_20d': thr2}
    return metrics, best_thresholds
