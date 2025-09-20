import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, r2_score,
    f1_score, matthews_corrcoef
)

# ---------- 通用工具 ----------
def _to_class(x):
    return (x >= 0.5).astype(int) 

def qlike_from_pred(
    y_true: np.array,
    pred: np.array,
    eps: float = 1e-6,
):
    """
    QLIKE:  L = log(f) + y/f
      - y_true: 真实 hl_ratio (>0)
      - pred:   模型输出
    """
    y = np.clip(y_true, eps, None)
    f = np.clip(pred, eps, None)

    return np.mean(np.log(f) + y / f)

# ---------- 多任务：四个目标的综合报告 ----------
def generate_report_multitask(
    y_true_dict,
    y_pred_dict,
    version=100,
    name='multitask',
):
    """
    y_true_dict / y_pred_dict:
        {
          'ret_close': np.array([...])         # 若是log-return请自行在外部处理一致
          'ret_sign':  np.array([...])         # 0/1 或 {-1,1}
          'hl_ratio':  np.array([...])         # >0
          'low_crash': np.array([...])         # 0/1
        }
    分类头的 y_pred 若是 logits 将自动 sigmoid；若是概率[0,1]将直接使用。
    """

    os.makedirs("./graph_model/model_reports/", exist_ok=True)
    os.makedirs("./graph_model/predictions/", exist_ok=True)

    # --- 回归1：ret_close（保持与传入空间一致的MSE/R2） ---
    rc_true = np.log1p(np.asarray(y_true_dict['ret_close'], dtype=float))
    rc_pred = np.asarray(y_pred_dict['ret_close'], dtype=float)

    rc_mse = mean_squared_error(rc_true, rc_pred)
    rc_r2  = r2_score(rc_true, rc_pred)

    # --- 分类1：ret_sign ---
    rs_true = np.asarray(y_true_dict['ret_sign'])
    # 兼容{-1,1}
    if rs_true.min() < 0:
        rs_true = (rs_true > 0).astype(int)
    rs_pred = _to_class(y_pred_dict['ret_sign'])
    rs_mcc = matthews_corrcoef(rs_true, rs_pred)
    rs_f1  = f1_score(rs_true, rs_pred, zero_division=0)

    # --- 回归2：ret_close_20d ---
    rc_20d_true = np.log1p(np.asarray(y_true_dict['ret_close_20d'], dtype=float))
    rc_20d_pred = np.asarray(y_pred_dict['ret_close_20d'], dtype=float)

    rc20_mse = mean_squared_error(rc_20d_true, rc_20d_pred)
    rc20_r2  = r2_score(rc_20d_true, rc_20d_pred)

    # --- 分类2：ret_sign_20d ---
    rs_20d_true = np.asarray(y_true_dict['ret_sign_20d']).astype(int)
    if rs_20d_true.min() < 0:
        rs_20d_true = (rs_20d_true > 0).astype(int)
    rs_20d_pred = _to_class(y_pred_dict['ret_sign_20d'])
    rs20_mcc  = matthews_corrcoef(rs_20d_true, rs_20d_pred)
    rs20_f1   = f1_score(rs_20d_true, rs_20d_pred, zero_division=0)

    # --- 回归3：ret_vol_20d ---
    rv_true = np.asarray(y_true_dict['ret_vol_20d'], dtype=float)
    rv_pred = np.asarray(y_pred_dict['ret_vol_20d'], dtype=float)

    rv_qlike = qlike_from_pred(torch.tensor(rv_true), torch.tensor(rv_pred)).item()

    # --- 汇总保存 ---
    metrics = {
        # ret_close
        'ret_close/MSE': rc_mse,
        'ret_close/R2': rc_r2,
        # ret_sign
        'ret_sign/MCC': rs_mcc,
        'ret_sign/F1': rs_f1,
        # ret_close_20d
        'ret_close_20d/MSE': rc20_mse,
        'ret_close_20d/R2': rc20_r2,
        # ret_sign_20d
        'ret_sign_20d/MCC': rs20_mcc,
        'ret_sign_20d/F1': rs20_f1,
        # ret_vol_20d
        'ret_vol_20d/QLIKE': rv_qlike
    }
    # metrics_df = pd.DataFrame([metrics])

    # metrics_file = f"{name}_multitask_metrics_v{version}.csv"
    # metrics_df.to_csv(os.path.join("./graph_model/model_reports/", metrics_file), index=False)

    print("Multitask evaluation report (CSV) generated successfully.")
    return metrics
