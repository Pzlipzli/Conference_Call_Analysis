import os
import argparse
from fusion_model import *


# 1) 在导入 torch 之前设置环境变量
def setup_env(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"

setup_env(0)  # 提前给个默认，之后再精确设置

import random
import numpy as np
import pandas as pd
import torch
from utils import *


def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


# 定义全局参数
EMBEDDING_DIM = 768
D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 512
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 20


def log_return(ret):
    return torch.log1p(ret)

def log_return_inverse(ret):
    return torch.expm1(ret)


def dataset_to_batch(dataset):
    valid_start = 0
    test_start = 0

    for ec in dataset.store['ec_name']:
        year = "20" + ec.split("_")[1]
        month = ec.split("_")[2]
        day = ec.split("_")[3]
        date = f"{year}-{month}-{day}"

        if date > "2023-12-31" and valid_start == 0:
            valid_start = dataset.get_idx(ec)
        if date > "2024-06-30":
            test_start = dataset.get_idx(ec)
            break

    train_dataset = dataset.copy_dataset(0, valid_start)
    valid_dataset = dataset.copy_dataset(valid_start, test_start)
    test_dataset = dataset.copy_dataset(test_start, len(dataset.store['ec_name']) + 1)

    return train_dataset, valid_dataset, test_dataset


# train.py
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import json


def main():
    import copy
    from torch.optim import Adam 
    from data_structure import MeetingStore 
    from structured_model import Structure 
    from generate_report import generate_report_multitask
    from simple_moe import TaskSpecificMMoE

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hl_pred_space", type=str, default="log", choices=["log", "pos"],
                        help="hl_ratio 头的输出空间：log 或 pos（原空间，内部 softplus 保正）")
    parser.add_argument("--name", type=str, default="simple_moe")
    args = parser.parse_args()

    # === 环境 ===
    seed = args.seed if args.seed is not None else args.version
    set_seed(seed, deterministic=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === 数据 ===
    dataset = MeetingStore()
    dataset.load_pickle("./meeting_data_store_15_20.pkl")
    train_dataset, valid_dataset, test_dataset = dataset_to_batch(dataset)

    len_train = len(train_dataset.store['ec_name'])
    len_valid = len(valid_dataset.store['ec_name'])
    len_test  = len(test_dataset.store['ec_name'])

    # ---- 预计算统计数据 ----
    pos_weight_1 = None
    pos_weight_2 = None
    mu = None
    sd = None
    mu_20 = None
    sd_20 = None
    mu_var_20 = None
    sd_var_20 = None
    with torch.no_grad():
        ytr = train_dataset.store['ret_sign'].float().view(-1)
        ytr = ytr[torch.isfinite(ytr)]
        if ytr.numel() > 0 and ytr.min() < 0:
            ytr = (ytr > 0).float()
        pos = float((ytr > 0.5).sum().item())
        neg = float(ytr.numel() - pos)
        w = (neg / max(1.0, pos)) if pos > 0 else 1.0
        pos_weight_1 = torch.tensor([w], device=device)

        ytr = train_dataset.store['ret_sign_20d'].float().view(-1)
        ytr = ytr[torch.isfinite(ytr)]
        if ytr.numel() > 0 and ytr.min() < 0:
            ytr = (ytr > 0).float()
        pos = float((ytr > 0.5).sum().item())
        neg = float(ytr.numel() - pos)
        w = (neg / max(1.0, pos)) if pos > 0 else 1.0
        pos_weight_2 = torch.tensor([w], device=device)

        print(f"Train ret_sign     pos_rate={pos/max(1.0,pos+neg):.4f}, pos_weight≈{float(pos_weight_1.item()):.2f}")
        print(f"Train ret_sign_20d pos_rate={pos/max(1.0,pos+neg):.4f}, pos_weight≈{float(pos_weight_2.item()):.2f}")

        y_tr = torch.log1p(train_dataset.store['ret_close'].clamp(min=-0.999999)).float().view(-1)
        mu = y_tr.mean().item()
        sd = y_tr.std(unbiased=False).item() or 1.0

        y_tr_20 = torch.log1p(train_dataset.store['ret_close_20d'].clamp(min=-0.999999)).float().view(-1)
        mu_20 = y_tr_20.mean().item()
        sd_20 = y_tr_20.std(unbiased=False).item() or 1.0

        print(f"Train ret_close     log1p(y) mu={mu:.4f}, sd={sd:.4f}")
        print(f"Train ret_close_20d log1p(y) mu={mu_20:.4f}, sd={sd_20:.4f}")

        y_tr_log = torch.log(train_dataset.store['ret_vol_20d']).float().view(-1)
        mu_var_20 = y_tr_log.mean().item()
        sd_var_20 = y_tr_log.std(unbiased=False).item() or 1.0
        print(f"Train ret_vol_20d log y mu={mu_var_20:.4f}, sd_var_20={sd_var_20:.4f}")

    # === 模型 ===
    # 注意：这里的 Structure 需要是“吃四路 embedding -> 调用你的 task-specific MoE -> 返回 dict”
    model = Structure(
        moe=TaskSpecificMMoE(),                      # 如果你传的是 TaskSpecificMMoE 实例，这里写 moe=your_moe
        # history_aggr=LabelSelfAttnMeanPooling(embed_dim=EMBEDDING_DIM, num_heads=NHEAD),
        moe_accepts="list"
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    # ============== 训练 ==============
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len_train, BATCH_SIZE):
            batch = train_dataset.copy_dataset(i, min(i + BATCH_SIZE, len_train))
            batch.update_device(device)

            pred_dict = model(batch)   # {'ret_close': [B], ...}
            pred_ret = pred_dict['ret_close']
            pred_sign = pred_dict['ret_sign']
            pred_ret_20 = pred_dict['ret_close_20d']
            pred_sign_20 = pred_dict['ret_sign_20d']
            pred_vol = pred_dict['ret_vol_20d']

            ret_close = batch.store['ret_close'].float()
            ret_sign  = batch.store['ret_sign'].float()
            ret_close_20d  = batch.store['ret_close_20d'].float()
            ret_sign_20d = batch.store['ret_sign_20d'].float()
            ret_vol_20d = batch.store['ret_vol_20d'].float()

            loss, _ = compute_multitask_loss(
                pred_ret, pred_sign, pred_ret_20, pred_sign_20, pred_vol,
                ret_close, ret_sign, ret_close_20d, ret_sign_20d, ret_vol_20d,
                pos_weight_1=pos_weight_1, pos_weight_2=pos_weight_2,
                mu=mu, sd=sd, mu_20=mu_20, sd_20=sd_20, mu_var_20=mu_var_20, sd_var_20=sd_var_20,
                w_close=1.0, w_sign=1.0, w_close_20d=1.0, w_sign_20d=1.0, w_vol = 1.0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  |  Train Total Loss: {avg_train_loss:.6f}")

        # ------------- 验证 -------------
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for i in range(0, len_valid, BATCH_SIZE):
                batch = valid_dataset.copy_dataset(i, min(i + BATCH_SIZE, len_valid))
                batch.update_device(device)

                pred_dict = model(batch)
                pred_ret = pred_dict['ret_close']
                pred_sign = pred_dict['ret_sign']
                pred_ret_20 = pred_dict['ret_close_20d']
                pred_sign_20 = pred_dict['ret_sign_20d']
                pred_vol = pred_dict['ret_vol_20d']

                ret_close = batch.store['ret_close'].float()
                ret_sign  = batch.store['ret_sign'].float()
                ret_close_20d  = batch.store['ret_close_20d'].float()
                ret_sign_20d = batch.store['ret_sign_20d'].float()
                ret_vol_20d = batch.store['ret_vol_20d'].float()

                loss, _ = evaluate_multitask(
                    pred_ret, pred_sign, pred_ret_20, pred_sign_20, pred_vol,
                    ret_close, ret_sign, ret_close_20d, ret_sign_20d, ret_vol_20d,
                    pos_weight_1=pos_weight_1, pos_weight_2=pos_weight_2,
                    mu=mu, sd=sd, mu_20=mu_20, sd_20=sd_20, mu_var_20=mu_var_20, sd_var_20=sd_var_20,
                    w_close=1.0, w_sign=1.0, w_close_20d=1.0, w_sign_20d=1.0, w_vol = 1.0,
                )
                val_loss_sum += float(loss.item())
                val_batches += 1

        avg_valid_loss = val_loss_sum / max(1, val_batches)
        print(f"                 Valid Total Loss: {avg_valid_loss:.6f}\n")

        # Early Stopping
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch
            }
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 恢复最佳
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        optimizer.load_state_dict(best_state["optimizer"])
        print(f"Restored best model at epoch {best_state['epoch']+1} (Valid Total Loss={best_loss:.6f})")

    # ============== 测试（多任务评估） ==============
    model.eval()
    preds = {k: [] for k in ['ret_close', 'ret_sign', 'ret_close_20d', 'ret_sign_20d', 'ret_vol_20d']}
    trues = {k: [] for k in ['ret_close', 'ret_sign', 'ret_close_20d', 'ret_sign_20d', 'ret_vol_20d']}

    with torch.no_grad():
        for i in range(0, len_test, BATCH_SIZE):
            batch = test_dataset.copy_dataset(i, min(i + BATCH_SIZE, len_test))
            batch.update_device(device)

            pdict = model(batch)
            # 收集预测
            for k in preds:
                preds[k].append(pdict[k].detach().squeeze(-1).cpu().numpy())
            # 收集真值（与模型输出空间一致：ret_close按你训练时选择的空间）
            trues['ret_close'].append(batch.store['ret_close'].detach().cpu().numpy())
            trues['ret_sign' ].append(batch.store['ret_sign' ].detach().cpu().numpy())
            trues['ret_close_20d' ].append(batch.store['ret_close_20d' ].detach().cpu().numpy())
            trues['ret_sign_20d'].append(batch.store['ret_sign_20d'].detach().cpu().numpy())
            trues['ret_vol_20d'].append(batch.store['ret_vol_20d'].detach().cpu().numpy())

    for k in preds:
        preds[k] = np.concatenate(preds[k], axis=0)
        trues[k] = np.concatenate(trues[k], axis=0)

    metrics = generate_report_multitask(
        y_true_dict=trues,
        y_pred_dict=preds,
        version=seed, name=args.name,
    )

    # —— 关键：打印一行 JSON，给 sweep.py 解析 ——
    print("METRICS_JSON:", json.dumps(metrics, ensure_ascii=False))

    return metrics


if __name__ == '__main__':
    metrics = main()
    # 控制台末行已打印 METRICS_JSON，sweep 会去解析这行
