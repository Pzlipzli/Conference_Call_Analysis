import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, r2_score,
    f1_score, matthews_corrcoef
)

def _qlike_from_logvar(y_true_var, z_pred_logvar, eps=1e-12):
    y = np.asarray(y_true_var, float)
    z = np.asarray(z_pred_logvar, float)
    y = np.clip(y, eps, None)
    return float(np.mean(z + y * np.exp(-z)))  # 越小越好

import numpy as np
from sklearn.metrics import r2_score

def _r2_score_oos(y_true_log, y_pred, mu_train_log):
    """
    计算 Out-of-Sample R² (Campbell–Thompson 风格)
    
    y_true_log: 测试集/验证集的 log1p(y)，shape [N]
    y_pred: 模型预测的 log1p(y)，shape [N]
    mu_train_log: 训练集 log1p(y) 的均值 (float)
    """
    y_true_log = np.asarray(y_true_log).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # SSR: 残差平方和
    ss_res = np.sum((y_true_log - y_pred) ** 2)

    # SST: 相对于训练集均值的平方和
    ss_tot = np.sum((y_true_log - mu_train_log) ** 2)

    r2_oos = 1 - ss_res / (ss_tot + 1e-12)
    return r2_oos


def generate_report(true_values, pred_values, name='ret_close', threshold=None,
                    mu=None, sd=None):
    """
    true_values: np.array
    pred_values: np.array (回归直接是预测；分类为 logits)
    threshold:   分类任务用到的阈值（来自验证集最优）。若 None 则用 0.5。
    返回 dict 指标；同时可按需落盘。
    """
    metrics = {}
    y_true = np.asarray(true_values).reshape(-1)
    y_pred = np.asarray(pred_values).reshape(-1)

    if name in ('ret_close', 'ret_close_20d'):
        y_true_log = np.log1p(np.clip(y_true, -0.999999, None))
        if mu is not None and sd is not None:
            y_pred = y_pred * sd + mu

        mse = mean_squared_error(y_true_log, y_pred)
        r2  = _r2_score_oos(y_true_log, y_pred, mu_train_log=mu)
        metrics.update({"MSE": float(mse), "R2": float(r2)})

    elif name in ('ret_sign', 'ret_sign_20d'):
        # logits -> prob
        prob = 1.0 / (1.0 + np.exp(-y_pred))
        y = y_true.copy()
        # {-1,1}->{0,1}
        if np.nanmin(y) < 0:
            y = (y > 0).astype(int)
        else:
            y = np.clip(y, 0, 1).astype(int)

        thr = 0.5 if threshold is None else float(threshold)
        yhat = (prob >= thr).astype(int)

        # 处理全同类的极端情形
        if yhat.max() == yhat.min():
            mcc = 0.0
            f1  = 0.0
        else:
            mcc = matthews_corrcoef(y, yhat)
            f1  = f1_score(y, yhat, zero_division=0)

        metrics.update({"MCC": float(mcc), "F1": float(f1), "Threshold": thr})

    elif name == 'ret_vol_20d':
        # 这里假设 pred_values 是 z=log-variance
        if mu is not None and sd is not None:
            y_pred = y_pred * sd + mu
        qlike = _qlike_from_logvar(y_true, y_pred)
        metrics.update({"QLIKE": float(qlike)})

    else:
        raise ValueError(f"Unknown task name: {name}")
    
    return metrics
