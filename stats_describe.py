import pandas as pd
import json

# 读取 JSON 文件
with open("conference_info.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 连续型变量
continuous_cols = ["ret_close", "ret_close_20d", 'ret_vol_20d']
continuous_stats = df[continuous_cols].describe(percentiles=[0.25, 0.5, 0.75]).T

# 二值变量
binary_cols = ["ret_sign", "ret_sign_20d"]
binary_stats = {}
for col in binary_cols:
    counts = df[col].value_counts(normalize=True) * 100  # 百分比
    binary_stats[col] = counts.to_dict()
binary_stats_df = pd.DataFrame(binary_stats)

# 保存结果
# continuous_stats.to_csv("continuous_statistics.csv")
# binary_stats_df.to_csv("binary_statistics.csv")

print("=== 连续变量分布 ===")
print(continuous_stats)
print("\n=== 二值变量占比 (%) ===")
print(binary_stats_df)
