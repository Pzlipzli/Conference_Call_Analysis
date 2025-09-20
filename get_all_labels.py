import os
from datetime import timedelta
import pandas as pd

# --------- 工具：从文件名解析事件日期与时间（不涉及时区） ----------
def parse_event(file: str):
    """文件名: TICKER_YY_MM_DD_HHMM.txt -> (ticker, event_date(pd.Timestamp), hhmm(int))"""
    name = os.path.splitext(file)[0]
    parts = name.split("_")
    if len(parts) < 5:
        raise ValueError(f"文件名不符合规则: {file}")
    ticker = parts[0]
    year  = 2000 + int(parts[1])
    month = int(parts[2])
    day   = int(parts[3])
    hhmm  = int(parts[4])
    event_date = pd.Timestamp(year=year, month=month, day=day)  # 仅日期
    return ticker, event_date, hhmm

# --------- 步骤1：收集每个ticker需要打标的“目标交易日”（按你的简化规则） ----------
def collect_needed_dates_simple(dataset_path: str, kline_dir: str, file_pattern="{ticker}.csv"):
    """
    返回: {ticker: set[pd.Timestamp(date)]}
    规则：
      - 如果 HHMM < 1600 → 目标=当日或之后第一个有数据的交易日 (>= event_date)
      - 如果 HHMM >= 1600 → 目标=事件日之后第一个有数据的交易日 (>  event_date)
    """
    # 预读每个ticker的可交易日期（来自其K线文件）
    def load_dates_for_ticker(tkr):
        path = os.path.join(kline_dir, file_pattern.format(ticker=tkr))
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=["Date"])
        dates = pd.to_datetime(df["Date"].dropna().unique())
        return sorted(pd.Series(dates).dt.normalize().unique())

    # 为了避免反复读K线，先收集涉及到的ticker
    by_ticker_events = {}  # {ticker: [(event_date, hhmm), ...]}
    for folder in os.listdir(dataset_path):
        if folder.startswith('.'):
            continue
        fpath = os.path.join(dataset_path, folder)
        if not os.path.isdir(fpath):
            continue
        for file in os.listdir(fpath):
            if not (file.endswith(".txt") and not file.endswith("_test.txt") and not file.endswith("timestamp.txt")):
                continue
            try:
                tkr, event_date, hhmm = parse_event(file)
            except Exception as e:
                print(f"[WARN] 跳过无法解析的文件 {file}: {e}")
                continue
            by_ticker_events.setdefault(tkr, []).append((event_date, hhmm))

    # 映射到“目标交易日”
    need = {}  # {ticker: set of dates}
    for tkr, events in by_ticker_events.items():
        dates_available = load_dates_for_ticker(tkr)
        if dates_available is None:
            print(f"[WARN] 找不到 {tkr} 的K线文件，跳过")
            continue

        need_set = set()
        for event_date, hhmm in events:
            if hhmm >= 1600:
                # 严格大于事件日
                target = next((d for d in dates_available if d > event_date), None)
            else:
                # 大于等于事件日
                target = next((d for d in dates_available if d >= event_date), None)

            if target is None:
                print(f"[INFO] {tkr}: 事件日 {event_date.date()} 之后未找到K线日期（可能样本截止）")
                continue
            need_set.add(target)

        if need_set:
            need[tkr] = need_set

    return need

# --------- 指标计算（与你原先一致） ----------
def compute_labels(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    df = df_ohlc.copy()
    if "Date" not in df.columns:
        raise ValueError("DataFrame 必须包含 'Date'")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    # 1) 收盘相对昨收收益率
    df["ret_close"] = df["Close"].pct_change()

    # 2) 收益正负（1/-1；首行NaN保持缺失）
    df["ret_sign"] = df["ret_close"].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) else pd.NA)
    )

    # 3) 20日收益率
    df["ret_close_20d"] = df["Close"].pct_change(periods=20)

    # 4) 20日收益率正负
    df["ret_sign_20d"] = df["ret_close_20d"].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) else pd.NA)
    )

    # 5) 20日每日收益率的方差
    df["ret_vol_20d"] = df["ret_close"].rolling(window=20, min_periods=20).var()

    return df[["Date", "ret_close", "ret_sign", "ret_close_20d", "ret_sign_20d", "ret_vol_20d"]]


# --------- 主流程：生成“电话会后的目标交易日”的四项label并输出 ----------
def generate_labels_for_needed_dates_simple(
    dataset_path: str,
    kline_dir: str,
    out_dir: str,
    file_pattern: str = "{ticker}.csv",
    date_col: str = "Date"
):
    os.makedirs(out_dir, exist_ok=True)

    need = collect_needed_dates_simple(dataset_path, kline_dir, file_pattern=file_pattern)
    if not need:
        print("[INFO] 未解析到需要打标的日期")
        return

    for tkr, need_dates in need.items():
        kline_path = os.path.join(kline_dir, file_pattern.format(ticker=tkr))
        if not os.path.exists(kline_path):
            print(f"[WARN] 缺少K线文件: {kline_path}")
            continue

        try:
            ohlc = pd.read_csv(kline_path, parse_dates=[date_col])
        except Exception as e:
            print(f"[WARN] 读取 {kline_path} 失败: {e}")
            continue

        # 统一列名
        ohlc = ohlc.rename(columns={date_col: "Date"})
        required = {"Date", "Open", "High", "Low", "Close"}
        if not required.issubset(ohlc.columns):
            print(f"[WARN] {kline_path} 缺少必要列：{required}")
            continue

        labels_all = compute_labels(ohlc)

        # 只保留目标交易日
        need_df = pd.DataFrame({"Date": sorted(list(need_dates))})
        out = need_df.merge(labels_all, on="Date", how="left")

        out_file = os.path.join(out_dir, f"{tkr}_labels.csv")
        out.to_csv(out_file, index=False)
        print(f"[OK] {tkr}: 输出 {len(out)} 条 → {out_file}")


# ============== 示例调用 ==============
if __name__ == "__main__":
    dataset_path = "./bloomberg_dataset/"          # 包含电话会议文本的根目录
    kline_dir = "./stooq_daily_data/"              # K 线数据目录（每只股票一个文件）
    out_dir = "./stooq_label_data/"    # 输出目录


    generate_labels_for_needed_dates_simple(
        dataset_path=dataset_path,
        kline_dir=kline_dir,
        out_dir=out_dir,
        file_pattern="{ticker}_daily.csv",   # 如需改名: "{ticker}_daily.csv"
        date_col="Date"
    )
