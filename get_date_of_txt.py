import os
import json
import pandas as pd


def get_date_of_txt():
    folder_path = "/Users/lipeizhe/fintech/bloomberg_dataset/"
    label_path = "/Users/lipeizhe/fintech/return_labels/"

    date_list = []

    for folder in os.listdir(folder_path):
        if not folder.startswith('.'):            
            print(f"Read name for {folder}")

            file_list = os.listdir(os.path.join(folder_path, folder))
            file_list.sort()

            label_file = os.path.join(label_path, f'{folder}.csv')
            df_label = pd.read_csv(label_file)

            for file in file_list:
                if file.endswith('.txt') and not file.endswith('test.txt') and not file.endswith('timestamp.txt'):
                    date_str = "20" + file.split('_')[1] + "-" + file.split('_')[2] + "-" + file.split('_')[3]
                    time_str = file.split('_')[4].split('.')[0]
                    time_str = time_str[:2] + ":" + time_str[2:]
                    ec_name = file.split('.')[0]
                    ebds_addr = f"/Users/lipeizhe/fintech/embeddings/{folder}/{ec_name}.pt"

                    day_label = df_label[df_label['date'].astype(str) == date_str]['1_day_return'].values[0]
                    week_label = df_label[df_label['date'].astype(str) == date_str]['1_week_return'].values[0]
                    month_label = df_label[df_label['date'].astype(str) == date_str]['1_month_return'].values[0]

                    date_list.append({"date": date_str, "time": time_str, "ec_name": ec_name, "ebds_addr": ebds_addr,
                                      "1_day_return": day_label, "1_week_return": week_label, "1_month_return": month_label})

    # 按 date 和 time 排序
    date_list.sort(key=lambda x: (x["date"], x["time"]))

    with open('date_list.json', 'w') as f:
        json.dump(date_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    get_date_of_txt()