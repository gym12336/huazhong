import pandas as pd
import numpy as np
import os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

def _read_csv(name):
    """按常见编码顺序探测，避免 GBK/UTF-8 中文文件名+编码报错"""
    for enc in ('utf-8', 'utf-8-sig', 'gb18030', 'gbk'):
        try:
            return pd.read_csv(os.path.join(_HERE, name), encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f'无法识别 {name} 编码')

# 1. 客户坐标信息
coords = _read_csv('客户坐标信息_Sheet1.csv')
print("=== 客户坐标信息 ===")
print(f"形状: {coords.shape}")
print(f"列名: {list(coords.columns)}")
print(coords.to_string())

coord_cols = coords.columns
id_col = coord_cols[1]
x_col = coord_cols[2]
y_col = coord_cols[3]

# 2. 时间窗
tw = _read_csv('时间窗_Sheet1.csv')
print("\n=== 时间窗 ===")
print(f"形状: {tw.shape}")
print(f"列名: {list(tw.columns)}")
print(tw.to_string())

# 3. 订单信息
orders = _read_csv('订单信息_Sheet1.csv')
print("\n=== 订单信息 ===")
print(f"形状: {orders.shape}")
print(f"列名: {list(orders.columns)}")
print(orders.head(20).to_string())
print(f"\n... 共 {len(orders)} 行")

# 4. 距离矩阵
dist = _read_csv('距离矩阵_Sheet1.csv')
print("\n=== 距离矩阵 ===")
print(f"形状: {dist.shape}")
print(f"列名: {list(dist.columns[:10])}")
print(f"前10x10:\n{dist.iloc[:10, :10].to_string()}")

# 5. 按客户汇总需求
print("\n=== 按客户汇总需求 ===")
col_order = orders.columns[0]
col_weight = orders.columns[1]
col_volume = orders.columns[2]
col_cust = orders.columns[3]

cust_demand = orders.groupby(col_cust).agg(
    总重量=(col_weight, 'sum'),
    总体积=(col_volume, 'sum'),
    订单数=(col_order, 'count')
).sort_index()
print(cust_demand.to_string())
print(f"\n客户数: {len(cust_demand)}")
print(f"总重量: {cust_demand['总重量'].sum():.2f} kg")
print(f"总体积: {cust_demand['总体积'].sum():.4f} m3")
cust_demand.to_csv(os.path.join(_HERE, '客户需求汇总.csv'), encoding='utf-8-sig')

# 6. 绿色配送区分析
print("\n=== 绿色配送区分析 ===")
coords['dist_to_center'] = np.sqrt(coords[x_col]**2 + coords[y_col]**2)
green_zone = coords[coords['dist_to_center'] <= 10]
non_green = coords[(coords['dist_to_center'] > 10) & (coords[id_col] != 0)]
depot = coords[coords[id_col] == 0]
print(f"配送中心坐标: ({depot[x_col].values[0]}, {depot[y_col].values[0]})")
print(f"配送中心到市中心距离: {depot['dist_to_center'].values[0]:.2f} km")
print(f"绿色配送区内客户数(<=10km): {len(green_zone[green_zone[id_col] != 0])}")
print(f"绿色配送区外客户数(>10km): {len(non_green)}")
green_ids = sorted(green_zone[green_zone[id_col] != 0][id_col].tolist())
print(f"绿色配送区内客户编号: {green_ids}")

# 7. 需求统计
print("\n=== 需求统计 ===")
print(f"重量 - 最小: {cust_demand['总重量'].min():.2f}, 最大: {cust_demand['总重量'].max():.2f}, 平均: {cust_demand['总重量'].mean():.2f}")
print(f"体积 - 最小: {cust_demand['总体积'].min():.4f}, 最大: {cust_demand['总体积'].max():.4f}, 平均: {cust_demand['总体积'].mean():.4f}")

# 8. 时间窗分析
print("\n=== 时间窗分析 ===")
tw_col_id = tw.columns[0]
tw_col_start = tw.columns[1]
tw_col_end = tw.columns[2]

def time_to_hours(t):
    if isinstance(t, str):
        parts = t.split(':')
        return int(parts[0]) + int(parts[1])/60
    return t

tw['start_h'] = tw[tw_col_start].apply(time_to_hours)
tw['end_h'] = tw[tw_col_end].apply(time_to_hours)
tw['window_len'] = tw['end_h'] - tw['start_h']

print(f"最早开始: {tw['start_h'].min():.2f}h, 最晚开始: {tw['start_h'].max():.2f}h")
print(f"最早结束: {tw['end_h'].min():.2f}h, 最晚结束: {tw['end_h'].max():.2f}h")
print(f"时间窗长度 - 最小: {tw['window_len'].min():.2f}h, 最大: {tw['window_len'].max():.2f}h, 平均: {tw['window_len'].mean():.2f}h")

# 按时段分布
print("\n上午(8-12)开始的客户数:", len(tw[(tw['start_h'] >= 8) & (tw['start_h'] < 12)]))
print("中午(12-14)开始的客户数:", len(tw[(tw['start_h'] >= 12) & (tw['start_h'] < 14)]))
print("下午(14-18)开始的客户数:", len(tw[(tw['start_h'] >= 14) & (tw['start_h'] < 18)]))
print("晚间(18-21)开始的客户数:", len(tw[(tw['start_h'] >= 18) & (tw['start_h'] < 21)]))

# 9. 车辆容量vs需求分析
print("\n=== 车辆容量 vs 需求 ===")
print(f"总需求重量: {cust_demand['总重量'].sum():.2f} kg")
print(f"总需求体积: {cust_demand['总体积'].sum():.4f} m3")
v_cap_w = 3000*60 + 1500*50 + 1250*50 + 3000*10 + 1250*15
v_cap_v = 13.5*60 + 10.8*50 + 6.5*50 + 15*10 + 8.5*15
print(f"总车辆载重能力: {v_cap_w} kg")
print(f"总车辆容积能力: {v_cap_v} m3")

# 单客户最大需求
print(f"\n单客户最大重量: {cust_demand['总重量'].max():.2f} kg (客户{cust_demand['总重量'].idxmax()})")
print(f"单客户最大体积: {cust_demand['总体积'].max():.4f} m3 (客户{cust_demand['总体积'].idxmax()})")

# 超过最小车辆容量的客户
over_1250 = cust_demand[cust_demand['总重量'] > 1250]
over_1500 = cust_demand[cust_demand['总重量'] > 1500]
over_3000 = cust_demand[cust_demand['总重量'] > 3000]
print(f"\n重量>1250kg的客户: {len(over_1250)}个")
print(f"重量>1500kg的客户: {len(over_1500)}个")
print(f"重量>3000kg的客户: {len(over_3000)}个")

print("\n=== 分析完成 ===")
