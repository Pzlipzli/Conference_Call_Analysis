import pandas as pd
import os
import json
import numpy as np


def get_one_info(ticker, filename):
    info_dict = {}

    def replace_nan_inf(obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        return obj

    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    if len(parts) < 5:
        raise ValueError(f"文件名不符合规则: {filename}")
    ticker = parts[0]
    year  = str(2000 + int(parts[1]))
    month = parts[2]
    day   = parts[3]
    hhmm  = parts[4]
    event_date = f"{year}-{month}-{day}"

    info_dict['date'] = event_date
    info_dict['time'] = hhmm[:2] + ":" + hhmm[2:]
    info_dict['ec_name'] = name
    info_dict["ebds_addr"] =  f"./embeddings/{ticker}/{name}.pt"

    label_df = pd.read_csv(f"./stooq_label_data/{ticker}_labels.csv")

    # 确保 Date 列是 datetime 类型
    label_df['Date'] = pd.to_datetime(label_df['Date'])

    # 将 event_date 转换为 datetime 类型（如果还不是）
    event_date = pd.to_datetime(event_date)

    # 筛选 Date 大于等于 event_date 的行
    filtered_df = label_df[label_df['Date'] >= event_date]

    current_row = filtered_df.iloc[0]
    info_dict['ret_close'] = replace_nan_inf(current_row['ret_close'].astype(float))
    info_dict['ret_sign'] = replace_nan_inf(current_row['ret_sign'].astype(float))
    info_dict['ret_close_20d'] = replace_nan_inf(current_row['ret_close_20d'].astype(float))
    info_dict['ret_sign_20d'] = replace_nan_inf(current_row['ret_sign_20d'].astype(float))
    info_dict['ret_vol_20d'] = replace_nan_inf(current_row['ret_vol_20d'].astype(float))

    return info_dict


def get_all_info():
    conference_info = []
    dataset_path = "./bloomberg_dataset/"

    for folder in os.listdir(dataset_path):
        if folder.startswith('.'):
            continue
        fpath = os.path.join(dataset_path, folder)
        if not os.path.isdir(fpath):
            continue
        for file in os.listdir(fpath):
            if not (file.endswith(".txt") and not file.endswith("_test.txt") and not file.endswith("timestamp.txt")):
                continue
            conference_info.append(get_one_info(folder, file))

    conference_info.sort(key=lambda x: (list(x.values())[0], list(x.values())[1]))

    with open("./conference_info.json", "w", encoding="utf-8") as f:
        json.dump(conference_info, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(conference_info)} records to conference_info.json")


if __name__ == "__main__":
    get_all_info()
    
