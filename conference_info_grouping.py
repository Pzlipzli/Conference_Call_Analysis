import json
import pandas as pd
from datetime import datetime, timedelta

# 1. 读取交易日信息
trading_days_df = pd.read_csv("US_Trading_Days_2021_2024.csv")
trading_days = set(pd.to_datetime(trading_days_df[trading_days_df['Is_Trading_Day'] == 'Yes']['Date']).dt.strftime('%Y-%m-%d'))

# 2. 读取会议信息
with open('conference_info.json', 'r') as f:
    date_list = json.load(f)

# 3. 会议按时间排序
def get_datetime(record):
    return datetime.strptime(record['date'] + ' ' + record['time'], "%Y-%m-%d %H:%M")

date_list_sorted = sorted(date_list, key=get_datetime)

# 4. 分组
grouped_meetings = []
current_group = []

for i, record in enumerate(date_list_sorted):
    if not current_group:
        current_group.append(record)
        continue

    prev_record = current_group[-1]
    prev_dt = get_datetime(prev_record)
    curr_dt = get_datetime(record)

    # 检查 prev_dt 到 curr_dt 之间是否有完整的交易日9:30-16:00区间
    if prev_dt.time() < datetime.strptime("09:30", "%H:%M").time():
        check_date = prev_dt.date() - timedelta(days=1)
    else:
        check_date = prev_dt.date()
    
    has_full_trading_session = False
    while True:
        check_date += timedelta(days=1)
        if check_date > curr_dt.date():
            break
        date_str = check_date.strftime("%Y-%m-%d")
        if date_str in trading_days:
            trading_start = datetime.strptime(date_str + " 09:30", "%Y-%m-%d %H:%M")
            trading_end = datetime.strptime(date_str + " 16:00", "%Y-%m-%d %H:%M")
            # 只有当这一天的9:30-16:00区间完全在prev_dt和curr_dt之间，才算有完整交易时段
            if prev_dt < trading_start and curr_dt > trading_end:
                has_full_trading_session = True
                break

    if has_full_trading_session:
        grouped_meetings.append(current_group)
        current_group = [record]
    else:
        current_group.append(record)

if current_group:
    grouped_meetings.append(current_group)

# 5. 保存为 json
with open('conference_info_grouped.json', 'w') as f:
    json.dump(grouped_meetings, f, indent=2, ensure_ascii=False)

print(f"分组完毕，共 {len(grouped_meetings)} 组，已保存为 conference_info_grouped.json")