# -*- coding: utf-8 -*-
"""
问题1结果导出器（不修改 solve_p1.py）
- p1_detail.csv : 每辆车每个停靠点的详细货物/到达时间清单
- p1_run.txt    : solve_p1 完整执行报告（含日志）
"""
import sys, io, os
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE); sys.path.insert(0, _HERE)

import pandas as pd

class _Tee:
    """同时把 stdout 写到屏幕与文件"""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s); st.flush()
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass

def _fmt_h(h):
    return f'{int(h)}:{int(round((h%1)*60)):02d}'

def _load_orders():
    """读订单明细，返回 {客户原始ID: [订单号,...]}"""
    df = pd.read_csv(f'{_HERE}/订单信息_Sheet1.csv', encoding='gbk')
    df.columns = ['order_id', 'weight', 'volume', 'cust_id']
    cust2orders = {}
    for _, r in df.iterrows():
        cust2orders.setdefault(int(r['cust_id']), []).append(int(r['order_id']))
    return cust2orders

def _build_detail_rows(best_flex, dm, dw, dv, tw_s, tw_e, n2o, cust2orders):
    """
    遍历每辆车的每个 trip，逐站模拟到达/离开时刻并展开为表格行。
    """
    from solve_p1 import tt, SVC_H
    rows = []
    for vid, s in enumerate(best_flex, 1):
        vt = s['vt']
        for tid, t in enumerate(s['trips'], 1):
            r = t['route']
            # 1) 路线汇总行
            custs_orig = [n2o.get(x, x) for x in r[1:-1]]
            tot_w = sum(dw.get(x, 0) for x in r[1:-1])
            tot_v = sum(dv.get(x, 0) for x in r[1:-1])
            tot_d = sum(dm[r[i]][r[i+1]] for i in range(len(r)-1))
            rows.append({
                '车辆ID': vid, '车型': vt['name'], '第几趟': tid,
                '停靠序号': 0, '类型': '配送中心(出发)',
                '客户原始ID': '', '到达时刻': '', '离开时刻': _fmt_h(t['start']),
                '该点重量kg': '', '该点体积m3': '', '该点订单编号': '',
                '累计载重kg': round(tot_w, 2), '累计体积m3': round(tot_v, 3),
                '路线总距离km': round(tot_d, 2),
                '行驶能耗元': round(t['travel'], 2),
                '惩罚元': round(t['penalty'], 2), 'CO2kg': round(t['carbon'], 2),
            })
            # 2) 每个客户停靠点
            cur_t = t['start']
            for i in range(1, len(r)-1):
                a, b = r[i-1], r[i]
                d = dm[a][b]
                cur_t += tt(d, cur_t)
                arrive = cur_t
                # 早到等待至 tw_s
                ws = tw_s.get(b, 0)
                if arrive < ws: cur_t = ws
                serve_start = cur_t
                cur_t += SVC_H
                leave = cur_t
                orig = n2o.get(b, b)
                w_here = dw.get(b, 0); v_here = dv.get(b, 0)
                orders_here = cust2orders.get(orig, [])
                rows.append({
                    '车辆ID': vid, '车型': vt['name'], '第几趟': tid,
                    '停靠序号': i, '类型': f'客户{orig}',
                    '客户原始ID': orig,
                    '到达时刻': _fmt_h(arrive),
                    '离开时刻': _fmt_h(leave),
                    '该点重量kg': round(w_here, 2),
                    '该点体积m3': round(v_here, 3),
                    '该点订单编号': str(orders_here),
                    '累计载重kg': '', '累计体积m3': '',
                    '路线总距离km': '',
                    '行驶能耗元': '', '惩罚元': '', 'CO2kg': '',
                })
            # 3) 返回 depot 行
            a, b = r[-2], r[-1]
            cur_t += tt(dm[a][b], cur_t)
            rows.append({
                '车辆ID': vid, '车型': vt['name'], '第几趟': tid,
                '停靠序号': len(r)-1, '类型': '配送中心(返回)',
                '客户原始ID': '', '到达时刻': _fmt_h(cur_t), '离开时刻': '',
                '该点重量kg': '', '该点体积m3': '', '该点订单编号': '',
                '累计载重kg': '', '累计体积m3': '',
                '路线总距离km': '',
                '行驶能耗元': '', '惩罚元': '', 'CO2kg': '',
            })
    return rows

if __name__ == '__main__':
    # 先导入（其内部会重设 sys.stdout）
    from solve_p1 import solve, print_report

    out_txt = open(f'{_HERE}/p1_run.txt', 'w', encoding='utf-8')
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, out_txt)
    try:
        rvt, coords, dm, dw, dv, tw_s, tw_e, green, n2o, schedule, best_flex = solve()
        print_report(rvt, dm, dw, dv, tw_s, tw_e, n2o, best_flex)

        cust2orders = _load_orders()
        rows = _build_detail_rows(best_flex, dm, dw, dv, tw_s, tw_e, n2o, cust2orders)
        pd.DataFrame(rows).to_csv(f'{_HERE}/p1_detail.csv',
                                  index=False, encoding='utf-8-sig')
        print(f"\n[导出] 详细停靠表: p1_detail.csv ({len(rows)}行)")
        print(f"[导出] 执行报告: p1_run.txt")
    finally:
        sys.stdout = orig_stdout
        out_txt.close()
