# sweep.py
import subprocess
import re
import pandas as pd
import os
import json
import numpy as np

# ===== 配置 =====
exp_name = 'ret_vol_20d'   # 实验名（用于文件夹命名）
versions = range(1, 11)       # 1..10

log_path   = f"./baseline_model/log/{exp_name}/"       # 按任务名分类日志更直观
report_dir = "./baseline_model/model_reports/"
os.makedirs(log_path, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

env = os.environ.copy()
env["MKL_THREADING_LAYER"] = "GNU"

# ===== 工具函数 =====
json_line_re = re.compile(r"METRICS_JSON:\s*(\{.*\})\s*$")

rows = []

# ===== 主循环 =====
for v in versions:
    cmd = ["python", "./baseline_model/train.py", "--version", str(v), "--name", exp_name]
    log_file = os.path.join(log_path, f"version_{v}.txt")
    err_file = os.path.join(log_path, f"version_{v}.err.txt")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,     # 训练失败时抛 CalledProcessError
            env=env
        )

        # 记录 stdout/stderr 到文件（原样）
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== CMD ===\n{' '.join(cmd)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout or "")
            f.write("\n")
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(result.stderr or "")

        # 解析 METRICS_JSON
        metrics_json = None
        for line in (result.stdout or "").splitlines():
            m = json_line_re.search(line.strip())
            if m:
                try:
                    metrics_json = json.loads(m.group(1))
                except Exception:
                    metrics_json = None
                break

        if metrics_json is None:
            print(f"Version {v}: No METRICS_JSON found (see logs: {log_file})")
            continue

        row = {"version": v}
        row.update(metrics_json)  # 展开所有指标列

        rows.append(row)
        print(f"Version {v}: Completed")

    except subprocess.CalledProcessError as e:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== CMD ===\n{' '.join(cmd)}\n\n")
            f.write(f"=== RETURN CODE ===\n{e.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(e.stdout or "")
            f.write("\n")
        with open(err_file, "w", encoding="utf-8") as f:
            f.write("=== STDERR ===\n")
            f.write(e.stderr or "")
        print(f"Version {v}: Subprocess failed (code {e.returncode}). See logs: {log_file}, {err_file}")

    except Exception as e:
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(f"[UNCAUGHT EXCEPTION] {type(e).__name__}: {e}\n")
        print(f"Version {v}: Unexpected error: {e}. See {err_file}")

# 汇总输出
if rows:
    df = pd.DataFrame(rows)
    # 你可以在这里挑一些主要指标做 mean/std
    # 例如：ret_close_MSE, low_crash_MCC
    summary_cols = [c for c in df.columns if c != "version"]
    summary = df[summary_cols].agg(["mean", "std"])

    out_file = os.path.join(report_dir, f"{exp_name}_results.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n=== Results ===\n")
        f.write(df.to_string(index=False))
        f.write("\n\n=== Summary ===\n")
        f.write(summary.to_string())
    print(f"Saved report to: {out_file}")
else:
    print("No successful runs with parsed METRICS_JSON. Check logs for details.")
