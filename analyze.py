import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)
import pandas as pd
from solve_p1 import load_data

coords,dm,dw,dv,tw_s,tw_e,green,custs,n2o=load_data()
green_orig={c for c in green if c<99}
ev_forced_orig={c for c in green_orig if tw_s.get(c,0)<16 and tw_e.get(c,24)>8}
print(f'强制EV客户: {sorted(ev_forced_orig)}')

# 读取 p2_detail.csv（新格式：逐站清单 + 限行违规列）
detail = pd.read_csv('p2/p2_detail.csv', encoding='utf-8-sig')

# 出发行（停靠序号==0）记录了每趟的车型/出发时间/限行违规
trips = detail[detail['停靠序号'] == 0].copy()
# 客户行（停靠序号>0 且 类型不含"配送中心"）记录访问的客户原始ID
cust_rows = detail[(detail['停靠序号'] > 0) &
                   (~detail['类型'].astype(str).str.contains('配送中心'))]

# 把同一趟访问的客户聚合为列表
cust_per_trip = (cust_rows.groupby(['车辆ID', '第几趟'])['客户原始ID']
                 .apply(lambda s: [int(x) for x in s.tolist()])
                 .reset_index().rename(columns={'客户原始ID': '客户列表'}))
trips = trips.merge(cust_per_trip, on=['车辆ID', '第几趟'], how='left')

# 返回行（最后一行）记录返回时刻
ret_rows = (detail[detail['类型'].astype(str).str.contains('返回')]
            [['车辆ID', '第几趟', '到达时刻']]
            .rename(columns={'到达时刻': '返回时刻'}))
trips = trips.merge(ret_rows, on=['车辆ID', '第几趟'], how='left')

def to_h(s):
    h, m = s.split(':')
    return int(h) + int(m)/60

# 仅看燃油 + 经过绿色区客户 的 trip
def has_green(custs_list):
    if not isinstance(custs_list, list): return False
    return any(c in green_orig for c in custs_list)

mask = trips['车型'].str.contains('燃油') & trips['客户列表'].apply(has_green)
fuel_green = trips[mask].copy()

print('\n燃油+绿色区 trip（按车-趟次去重）:')
all_pass = True
for _, r in fuel_green.iterrows():
    dep = to_h(r['离开时刻']); ret = to_h(r['返回时刻'])
    overlap = dep < 16 and ret > 8                # 时间窗与限行段[8,16]相交
    flag = (r['限行违规'] == '是')                # detail 自带的违规标记
    violate = overlap or flag                     # 任一为真即违规
    if violate: all_pass = False
    custs_str = str(r['客户列表'])[:60]
    print(f'  车{r["车辆ID"]} 趟{r["第几趟"]} {r["离开时刻"]}-{r["返回时刻"]} '
          f'载重{r["累计载重kg"]} 违规={violate}  客户:{custs_str}')

print('\n✅ 全部合规' if all_pass else '\n❌ 存在违规 trip')
