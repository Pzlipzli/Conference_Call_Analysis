# sweep.py
import subprocess
import re
import pandas as pd
import os
import json

rows = []

name = 'simple'
log_path = f"./graph_model/log/{name}/"
report_dir = "./graph_model/model_reports/"
os.makedirs(log_path, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

env = os.environ.copy()
env["MKL_THREADING_LAYER"] = "GNU"

# 匹配一行形如：METRICS_JSON: {...}
json_line_re = re.compile(r"METRICS_JSON:\s*(\{.*\})\s*$")

for v in range(1, 11):
    cmd = ["python", "./graph_model/train.py", "--version", str(v), "--name", name]
    log_file = os.path.join(log_path, f"version_{v}.txt")
    err_file = os.path.join(log_path, f"version_{v}.err.txt")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, env=env
        )

        # 写日志
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

        # 挑选关心的指标列（可自行增删）
        row = {
            "version": v,
            "ret_close/MSE": metrics_json.get("ret_close/MSE"),
            "ret_close/R2": metrics_json.get("ret_close/R2"),
            "ret_sign/MCC": metrics_json.get("ret_sign/MCC"),
            "ret_sign/F1": metrics_json.get("ret_sign/F1"),
            "hl_ratio/QLIKE": metrics_json.get("hl_ratio/QLIKE"),
            "low_crash/MCC": metrics_json.get("low_crash/MCC"),
            "low_crash/F1": metrics_json.get("low_crash/F1"),
        }
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

    out_file = os.path.join(report_dir, f"{name}_results.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n=== Results ===\n")
        f.write(df.to_string(index=False))
        f.write("\n\n=== Summary ===\n")
        f.write(summary.to_string())
    print(f"Saved report to: {out_file}")
else:
    print("No successful runs with parsed METRICS_JSON. Check logs for details.")
