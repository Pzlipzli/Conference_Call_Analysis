import os, random
import numpy as np
import torch

def set_seed_all(seed: int, deterministic: bool = True):
    # 环境变量：必须在进程早期设置
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUBLAS 严格确定性（需要 PyTorch 2.x + CUDA 10.2+）
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或 ":4096:8"

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN 确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # （可选）进一步严格化：有些算子会报错不支持
        # torch.use_deterministic_algorithms(True)

import json
import argparse
import torch.nn.functional as F
from utils import *

# 定义全局参数
EMBEDDING_DIM = 768
D_MODEL = 256
NHEAD = 2
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 512
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
PATIENCE = 20


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    seed = args.seed if args.seed is not None else args.version
    set_seed_all(seed, deterministic=True)
    name = args.name

    # === 你的原始训练代码（删掉 for 循环） ===
    from data_structure import MeetingStore
    from torch.optim import Adam, AdamW
    import copy
    from generate_report import generate_report
    from structured_model import Structure
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MeetingStore()
    dataset.load_pickle("./meeting_data_store_15_20.pkl")

    train_dataset, valid_dataset, test_dataset = dataset_to_batch(dataset)

    # ---- 预计算统计数据 ----
    pos_weight = None
    mu = None
    sd = None
    if name in ('ret_sign', 'ret_sign_20d'):
        with torch.no_grad():
            ytr = train_dataset.store[name].float().view(-1)
            ytr = ytr[torch.isfinite(ytr)]
            if ytr.numel() > 0 and ytr.min() < 0:
                ytr = (ytr > 0).float()
            pos = float((ytr > 0.5).sum().item())
            neg = float(ytr.numel() - pos)
            w = (neg / max(1.0, pos)) if pos > 0 else 1.0
            pos_weight = torch.tensor([w], device=device)
        print(f"[{name}] train pos_rate={pos/max(1.0,pos+neg):.4f}, pos_weight≈{float(pos_weight.item()):.2f}")
    elif name in ('ret_close', 'ret_close_20d'):
        with torch.no_grad():
            y_tr = torch.log1p(train_dataset.store[name].clamp(min=-0.999999)).float().view(-1)
            mu = y_tr.mean().item()
            sd = y_tr.std(unbiased=False).item() or 1.0
        print(f"[{name}] train log1p(y) mu={mu:.4f}, sd={sd:.4f}")
    elif name in ('ret_vol_20d',):
        with torch.no_grad():
            y_tr_log = torch.log(train_dataset.store[name]).float().view(-1)
            mu = y_tr_log.mean().item()
            sd = y_tr_log.std(unbiased=False).item() or 1.0
        print(f"[{name}] train log y mu={mu:.4f}, sd={sd:.4f}")

    model = Structure().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    best_score = float('-inf')
    best_state = None
    best_thr = None  # 仅分类任务有效
    patience_counter = 0

    len_train = len(train_dataset.store['ec_name'])
    len_valid = len(valid_dataset.store['ec_name'])
    len_test  = len(test_dataset.store['ec_name'])

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        for i in range(0, len_train, BATCH_SIZE):
            batch = train_dataset.copy_dataset(i, min(i + BATCH_SIZE, len_train))
            batch.update_device(device)

            y_pred = model(batch)              # logits or regression
            y_true = batch.store[name].float()

            loss = loss_function(y_true, y_pred, name=name, pos_weight=pos_weight, mu=mu, sd=sd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()); n_batches += 1

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:\nTrain Loss: {total_loss/max(1,n_batches):.6f}")

        # --- Valid ---
        model.eval()
        scores, thrs = [], []
        with torch.no_grad():
            for i in range(0, len_valid, BATCH_SIZE):
                batch = valid_dataset.copy_dataset(i, min(i + BATCH_SIZE, len_valid))
                batch.update_device(device)
                y_pred = model(batch)
                y_true = batch.store[name].float()
                s, t = evaluate(y_true, y_pred, name=name, mu=mu, sd=sd)
                scores.append(s)
                if t is not None: thrs.append(t)
        avg_valid_score = float(np.mean(scores))
        print(f"Valid score: {avg_valid_score:.6f}\n")

        # --- Early Stopping（最大化 score）---
        if avg_valid_score > best_score:
            best_score = avg_valid_score
            patience_counter = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch
            }
            # 记录本轮验证的最佳阈值均值（也可用加权/中位数）
            best_thr = (float(np.mean(thrs)) if thrs else None)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 恢复最佳
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        optimizer.load_state_dict(best_state["optimizer"])
        print(f"Restored best model at epoch {best_state['epoch']+1} (Valid Score={best_score:.6f})")
        if best_thr is not None:
            print(f"[INFO] Using validation best threshold: {best_thr:.4f}")

    # --- Test ---
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len_test, BATCH_SIZE):
            batch = test_dataset.copy_dataset(i, min(i + BATCH_SIZE, len_test))
            batch.update_device(device)
            y_pred = model(batch).detach().cpu().numpy()
            preds_list.append(y_pred)
    y_pred_test = np.concatenate(preds_list, axis=0)  # logits/regression
    y_true_test = test_dataset.store[name].cpu().numpy()

    # 把 best_thr 传给 generate_report（分类任务沿用验证阈值）
    from generate_report import generate_report
    metrics = generate_report(y_true_test, y_pred_test, name=name, threshold=best_thr, mu=mu, sd=sd)

    return metrics

if __name__ == '__main__':
    metrics = main()

    # 人类可读的打印（可选）
    print("METRICS_JSON:", json.dumps(metrics, ensure_ascii=False))


