import pandas as pd

# 股票列表
tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN', 'META']

# 创建完整日期范围
all_dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
trading_days_df = pd.DataFrame({'Date': all_dates})

# 用于存储所有交易日期
all_trading_dates = set()

# 遍历每只股票，读取股价数据，提取交易日期
for ticker in tickers:
    df = pd.read_csv(f"./daily_price_data/{ticker}_daily.csv", parse_dates=['日期'])
    all_trading_dates.update(df['日期'].dt.date)

# 标记交易日
trading_days_df['Is_Trading_Day'] = trading_days_df['Date'].dt.date.isin(all_trading_dates)
trading_days_df['Is_Trading_Day'] = trading_days_df['Is_Trading_Day'].map({True:'Yes', False:'No'})

# 保存 CSV
trading_days_df.to_csv("US_Trading_Days_2021_2024.csv", index=False)

# 打印统计信息
print("交易日 CSV 已生成")
print(trading_days_df['Is_Trading_Day'].value_counts())
