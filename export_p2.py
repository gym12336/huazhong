# -*- coding: utf-8 -*-
"""
问题2结果导出器（不修改 solve_p2.py）
- p2_detail.csv : 每辆车每个停靠点的详细货物/到达时间清单（含是否进绿色区标记）
- p2_run.txt    : solve_p2 完整执行报告
"""
import sys, io, os
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE); sys.path.insert(0, _HERE)

import pandas as pd
from export_p1 import _Tee, _fmt_h, _load_orders

def _build_detail_rows_p2(best_sched, dm, dw, dv, tw_s, tw_e,
                          n2o, green_orig, cust2orders):
    """与 P1 相同结构，但增加'是否绿色区客户'列与限行违规标记"""
    from solve_p1 import tt, SVC_H
    RESTRICT_S, RESTRICT_E = 8.0, 16.0
    rows = []
    for vid, s in enumerate(best_sched, 1):
        vt = s['vt']
        for tid, t in enumerate(s['trips'], 1):
            r = t['route']
            in_green_pts = [n2o.get(x, x) for x in r[1:-1]
                            if n2o.get(x, x) in green_orig]
            trip_overlaps_restrict = (t['start'] < RESTRICT_E and t['end'] > RESTRICT_S)
            violate = (vt['type'] == 'fuel' and in_green_pts and trip_overlaps_restrict)
            tot_w = sum(dw.get(x, 0) for x in r[1:-1])
            tot_v = sum(dv.get(x, 0) for x in r[1:-1])
            tot_d = sum(dm[r[i]][r[i+1]] for i in range(len(r)-1))
            rows.append({
                '车辆ID': vid, '车型': vt['name'], '第几趟': tid,
                '停靠序号': 0, '类型': '配送中心(出发)',
                '客户原始ID': '', '是否绿色区客户': '',
                '到达时刻': '', '离开时刻': _fmt_h(t['start']),
                '该点重量kg': '', '该点体积m3': '', '该点订单编号': '',
                '累计载重kg': round(tot_w, 2), '累计体积m3': round(tot_v, 3),
                '路线总距离km': round(tot_d, 2),
                '行驶能耗元': round(t['travel'], 2),
                '惩罚元': round(t['penalty'], 2), 'CO2kg': round(t['carbon'], 2),
                '限行违规': '是' if violate else '否',
            })
            cur_t = t['start']
            for i in range(1, len(r)-1):
                a, b = r[i-1], r[i]
                cur_t += tt(dm[a][b], cur_t)
                arrive = cur_t
                ws = tw_s.get(b, 0)
                if arrive < ws: cur_t = ws
                cur_t += SVC_H
                leave = cur_t
                orig = n2o.get(b, b)
                in_g = orig in green_orig
                rows.append({
                    '车辆ID': vid, '车型': vt['name'], '第几趟': tid,
                    '停靠序号': i, '类型': f'客户{orig}',
                    '客户原始ID': orig,
                    '是否绿色区客户': '是' if in_g else '否',
                    '到达时刻': _fmt_h(arrive),
                    '离开时刻': _fmt_h(leave),
                    '该点重量kg': round(dw.get(b, 0), 2),
                    '该点体积m3': round(dv.get(b, 0), 3),
                    '该点订单编号': str(cust2orders.get(orig, [])),
                    '累计载重kg': '', '累计体积m3': '',
                    '路线总距离km': '',
                    '行驶能耗元': '', '惩罚元': '', 'CO2kg': '',
                    '限行违规': '',
                })
            cur_t += tt(dm[r[-2]][r[-1]], cur_t)
            rows.append({
                '车辆ID': vid, '车型': vt['name'], '第几趟': tid,
                '停靠序号': len(r)-1, '类型': '配送中心(返回)',
                '客户原始ID': '', '是否绿色区客户': '',
                '到达时刻': _fmt_h(cur_t), '离开时刻': '',
                '该点重量kg': '', '该点体积m3': '', '该点订单编号': '',
                '累计载重kg': '', '累计体积m3': '',
                '路线总距离km': '',
                '行驶能耗元': '', '惩罚元': '', 'CO2kg': '',
                '限行违规': '',
            })
    return rows

if __name__ == '__main__':
    from solve_p2 import solve_p2
    out_txt = open(f'{_HERE}/p2_run.txt', 'w', encoding='utf-8')
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, out_txt)
    try:
        rvt, coords, dm, dw, dv, tw_s, tw_e, green_orig, n2o, best_sched = solve_p2()

        cust2orders = _load_orders()
        rows = _build_detail_rows_p2(best_sched, dm, dw, dv, tw_s, tw_e,
                                     n2o, green_orig, cust2orders)
        pd.DataFrame(rows).to_csv(f'{_HERE}/p2_detail.csv',
                                  index=False, encoding='utf-8-sig')
        print(f"\n[导出] 详细停靠表: p2_detail.csv ({len(rows)}行)")
        print(f"[导出] 执行报告: p2_run.txt")
    finally:
        sys.stdout = orig_stdout
        out_txt.close()
