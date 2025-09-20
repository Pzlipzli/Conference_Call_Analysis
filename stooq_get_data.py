# pip install pandas pandas-datareader
import pandas as pd
from pandas_datareader import data as pdr
import os

def get_data_stooq(ticker: str, start="2020-12-01", end="2024-12-31"):
    """
    从 Stooq 获取日频 OHLC（无需 API Key）
    注意：Stooq 返回是逆序日期，需要排序；ticker 写法通常与交易所一致（如 AAPL）。
    """
    df = pdr.DataReader(ticker, "stooq")  # 返回到最新，倒序
    df = df.sort_index()                  # 升序
    # 只留所需区间
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    # 统一列名
    # df = df.rename(columns={
    #     "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"
    # })
    # 重置索引
    df = df.reset_index().rename(columns={"Date":"Date"})
    return df


def get_all_stocks():
    folder_path = "./bloomberg_dataset/"
    save_path = "./stooq_daily_data/"
    for folder in os.listdir(folder_path):
        if folder.startswith("."):
            continue
        df_data = get_data_stooq(folder)
        df_data.to_csv(save_path + folder + "_daily.csv", index=False)

        print(f"Saved {folder} data")


if __name__ == "__main__":
    get_all_stocks()

