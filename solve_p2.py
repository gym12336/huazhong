# -*- coding: utf-8 -*-
"""华中杯A题 问题2：绿色配送区限行政策下的车辆调度"""
import sys, io, os, time, math, random
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.chdir(r'c:\Users\LENOVO\Desktop\华中杯1.1')
sys.path.insert(0, r'c:\Users\LENOVO\Desktop\华中杯1.1')

from solve_p1 import (
    load_data, eval_r, opt_start,
    construct_savings, reassign, best_vt, intra_opt_all,
    merge_routes, relocate, swap, seg_rel, two_opt_star, two_opt,
    shift_assign, sched_cost, upgrade_ev,
    try_eliminate_vehicle, sched_trip_swap,
    VEHICLE_TYPES, STARTUP, GREEN_R, BASE, log
)
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings; warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
random.seed(42); np.random.seed(42)

RESTRICT_S, RESTRICT_E = 8.0, 16.0

# ── 约束判断 ─────────────────────────────────────
def route_in_green(route, green_orig, n2o):
    return any(n2o.get(c, c) in green_orig for c in route[1:-1])

def overlap_restrict(t_start, t_end):
    return t_start < RESTRICT_E and t_end > RESTRICT_S

def count_violations(sched, green_orig, n2o):
    v = 0
    for veh in sched:
        if veh['vt']['type'] != 'fuel': continue
        for t in veh['trips']:
            if route_in_green(t['route'], green_orig, n2o) and overlap_restrict(t['start'], t['end']):
                v += 1
    return v

# ── 问题2求解 ────────────────────────────────────
def fix_violations(sched, dm, dw, dv, tw_s, tw_e, green_orig, n2o):
    """
    修复违规：
    1) 尝试整车换EV（EV3000或EV1250，按容量和额度）
    2) EV额度不足时，把违规trip的出发时间推迟到RESTRICT_E(16:00)之后
       这样限行解除，燃油车可合法进入（代价是迟到罚款）
    """
    ev3000 = next(vt for vt in VEHICLE_TYPES if vt['name'] == '新能源3000')
    ev1250 = next(vt for vt in VEHICLE_TYPES if vt['name'] == '新能源1250')
    ev3000_used = sum(1 for s in sched if s['vt']['name'] == '新能源3000')
    ev1250_used = sum(1 for s in sched if s['vt']['name'] == '新能源1250')
    fixed = 0

    vi = 0
    while vi < len(sched):
        veh = sched[vi]
        if veh['vt']['type'] != 'fuel':
            vi += 1; continue

        viol_tis = [ti for ti, t in enumerate(veh['trips'])
                    if route_in_green(t['route'], green_orig, n2o)
                    and overlap_restrict(t['start'], t['end'])]
        if not viol_tis:
            vi += 1; continue

        max_w = max(sum(dw.get(c, 0) for c in t['route'][1:-1]) for t in veh['trips'])
        max_v = max(sum(dv.get(c, 0) for c in t['route'][1:-1]) for t in veh['trips'])

        # 尝试整车换EV
        swapped = False
        for ev_vt, ev_used_now, ev_limit in [
            (ev3000, ev3000_used, 10), (ev1250, ev1250_used, 15)]:
            if max_w <= ev_vt['max_w'] and max_v <= ev_vt['max_v'] and ev_used_now < ev_limit:
                new_trips = []
                for t in veh['trips']:
                    c2, tr2, pen2, co2, ok, et2 = eval_r(
                        t['route'], dm, dw, dv, tw_s, tw_e, ev_vt, t['start'])
                    new_trips.append({'route': t['route'], 'vt': ev_vt,
                                      'start': t['start'], 'end': et2,
                                      'cost': c2, 'travel': tr2,
                                      'penalty': pen2, 'carbon': co2})
                veh['vt'] = ev_vt; veh['trips'] = new_trips
                if ev_vt == ev3000: ev3000_used += 1
                else: ev1250_used += 1
                fixed += 1; swapped = True; break

        if not swapped:
            # EV不够，把违规trip的出发时间推迟到 RESTRICT_E（限行结束时间）
            for ti in viol_tis:
                t = veh['trips'][ti]
                new_start = max(t['start'], RESTRICT_E + 0.1)  # 16:06
                c2, tr2, pen2, co2, ok, et2 = eval_r(
                    t['route'], dm, dw, dv, tw_s, tw_e, veh['vt'], new_start)
                if ok:
                    veh['trips'][ti] = {'route': t['route'], 'vt': veh['vt'],
                                        'start': new_start, 'end': et2,
                                        'cost': c2, 'travel': tr2,
                                        'penalty': pen2, 'carbon': co2}
                    fixed += 1
                    log(f"    推迟燃油trip到{new_start:.2f}h(16:00后), 惩罚+{pen2-t['penalty']:.2f}")
        vi += 1

    sched = [s for s in sched if s['trips']]
    viol_remain = count_violations(sched, green_orig, n2o)
    log(f"  fix_violations: 修复{fixed}次 剩余违规:{viol_remain} EV3000:{ev3000_used}/10 EV1250:{ev1250_used}/15")
    return sched

def solve_p2():
    T = time.time()
    coords, dm, dw, dv, tw_s, tw_e, green, custs, n2o = load_data()

    green_orig = {c for c in green if c < 99}
    log(f"\n绿色区原始客户({len(green_orig)}个): {sorted(green_orig)}")

    # 强制EV：绿色区内 且 时间窗与[8,16]重叠
    ev_forced_orig = {c for c in green_orig
                      if tw_s.get(c, 0) < RESTRICT_E and tw_e.get(c, 24) > RESTRICT_S}
    log(f"强制EV客户: {sorted(ev_forced_orig)}")
    ev_forced_nodes = {nd for nd in dw if n2o.get(nd, nd) in ev_forced_orig}
    log(f"强制EV节点数(含虚拟): {len(ev_forced_nodes)}")

    def is_forced_ev(route):
        return any(c in ev_forced_nodes for c in route[1:-1])

    def reassign_p2(rvt):
        """强制EV路线用EV，普通路线正常分配。EV分配：小载重用EV1250（为EV3000留额度）"""
        used = {vt['id']: 0 for vt in VEHICLE_TYPES}
        ev3000_vt = next(vt for vt in VEHICLE_TYPES if vt['name'] == '新能源3000')
        ev1250_vt = next(vt for vt in VEHICLE_TYPES if vt['name'] == '新能源1250')
        out_forced = []
        # 强制EV路线：按载重降序，大的先抢EV3000
        forced = sorted([(r, vt) for r, vt in rvt if is_forced_ev(r)],
                        key=lambda x: sum(dw.get(c, 0) for c in x[0][1:-1]), reverse=True)
        for r, _ in forced:
            TW = sum(dw.get(c, 0) for c in r[1:-1])
            TV = sum(dv.get(c, 0) for c in r[1:-1])
            chosen = None
            # 大载重优先EV3000，小载重优先EV1250
            order = [ev3000_vt, ev1250_vt] if TW > ev1250_vt['max_w'] else [ev1250_vt, ev3000_vt]
            for ev_vt in order:
                if ev_vt['max_w'] < TW or ev_vt['max_v'] < TV: continue
                if used.get(ev_vt['id'], 0) >= ev_vt['total']: continue
                chosen = ev_vt; break
            if chosen is None:
                chosen = ev3000_vt if TW > ev1250_vt['max_w'] else ev1250_vt
            used[chosen['id']] = used.get(chosen['id'], 0) + 1
            out_forced.append((r, chosen))
        out_normal = []
        for r, _ in rvt:
            if is_forced_ev(r): continue
            vt = best_vt(r, dm, dw, dv, tw_s, tw_e, used) or VEHICLE_TYPES[0]
            used[vt['id']] = used.get(vt['id'], 0) + 1
            out_normal.append((r, vt))
        return out_forced + out_normal

    log("\n[1] Savings 构造")
    routes = construct_savings(custs, dm, dw, dv, tw_s, tw_e)
    rvt = reassign_p2([(r, VEHICLE_TYPES[0]) for r in routes])
    ev_c = lambda: sum(eval_r(r, dm, dw, dv, tw_s, tw_e, vt)[0] for r, vt in rvt)
    log(f"  {len(rvt)}条, EV:{sum(1 for _,v in rvt if v['type']=='ev')}辆, 初始:{ev_c():.2f}")

    log("\n[2] 路线内优化")
    rvt = intra_opt_all(rvt, dm, dw, dv, tw_s, tw_e)
    log(f"  {ev_c():.2f}")

    log("\n[3] 多随机种子迭代")
    orig = [(r[:], vt) for r, vt in rvt]
    gbest = ev_c(); grvt = list(rvt)
    for seed in [42, 7, 123, 2024, 99]:
        random.seed(seed)
        rvt = [(r[:], vt) for r, vt in orig]
        lcost = ev_c(); lrvt = list(rvt); stall = 0
        for rd in range(1, 10):
            rvt = reassign_p2(rvt)
            thr = 1500 if rd <= 2 else (1000 if rd <= 4 else (600 if rd <= 7 else 400))
            rvt = merge_routes(rvt, dm, dw, dv, tw_s, tw_e, thr, tlim=4)
            rvt = relocate(rvt, dm, dw, dv, tw_s, tw_e, tlim=4)
            rvt = swap(rvt, dm, dw, dv, tw_s, tw_e, tlim=3)
            rvt = seg_rel(rvt, dm, dw, dv, tw_s, tw_e, sl=2, tlim=3)
            rvt = two_opt_star(rvt, dm, dw, dv, tw_s, tw_e, tlim=4)
            rvt = intra_opt_all(rvt, dm, dw, dv, tw_s, tw_e)
            rvt = reassign_p2(rvt)
            c = ev_c()
            if c < lcost - 1: lcost = c; lrvt = list(rvt); stall = 0
            else:
                stall += 1
                if stall >= 2: break
        log(f"  seed={seed}: {lcost:.2f} ({len(lrvt)}条)")
        if lcost < gbest: gbest = lcost; grvt = lrvt; log("    ★")

    rvt = reassign_p2(grvt)
    log(f"\n*** 子路线 {gbest:.2f}, {len(rvt)}条, 耗时 {time.time()-T:.1f}s ***")

    log("\n[4] Shift Multi-Trip + 违规检查")
    best_sched = None; best_total = 1e18
    for delay in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
        for rev in [False, True]:
            s = shift_assign(rvt, dm, dw, dv, tw_s, tw_e, max_delay=delay, reverse=rev)
            viol = count_violations(s, green_orig, n2o)
            ft, fv, ftr, fpen, _ = sched_cost(s)
            log(f"  d={delay}{'R' if rev else 'F'}: {fv}辆 {ft:.2f} 违规:{viol}")
            if viol == 0 and ft < best_total:
                best_total = ft; best_sched = s; log("    ★")

    if best_sched is None:
        log("  ⚠ 无零违规方案，取违规最少+成本最低")
        cands = [shift_assign(rvt, dm, dw, dv, tw_s, tw_e, max_delay=d)
                 for d in [2.0, 4.0, 8.0]]
        best_sched = min(cands, key=lambda s: count_violations(s, green_orig, n2o)*1e6 + sched_cost(s)[0])

    # 强制修复残留违规（延迟装箱可能把EV路线错配到燃油车）
    log("  强制修复违规...")
    best_sched = fix_violations(best_sched, dm, dw, dv, tw_s, tw_e, green_orig, n2o)
    viol_after = count_violations(best_sched, green_orig, n2o)
    bt_now, bv_now, *_ = sched_cost(best_sched)
    log(f"  修复后: {bv_now}辆 {bt_now:.2f} 违规:{viol_after}")

    log("\n[5] EV升级")
    best_sched = upgrade_ev(best_sched, dm, dw, dv, tw_s, tw_e)

    log("\n[6] 穷举消除多余车辆")
    for _ in range(4):
        before = len(best_sched)
        best_sched = try_eliminate_vehicle(best_sched, dm, dw, dv, tw_s, tw_e,
                                           max_delay=3.0, tlim=30)
        if len(best_sched) == before: break

    log("\n[7] trip交换")
    best_sched = sched_trip_swap(best_sched, dm, dw, dv, tw_s, tw_e, tlim=15)
    best_sched = upgrade_ev(best_sched, dm, dw, dv, tw_s, tw_e)

    # 最终兜底修复：确保所有违规都被消除
    log("  最终违规修复...")
    best_sched = fix_violations(best_sched, dm, dw, dv, tw_s, tw_e, green_orig, n2o)

    # 对被推迟到16:00后的trip做路线内二次优化（减少晚到惩罚）
    log("  优化推迟trip的路线顺序...")
    n_reimproved = 0
    for veh in best_sched:
        for ti, t in enumerate(veh['trips']):
            if t['start'] < RESTRICT_E + 0.05: continue  # 非推迟trip跳过
            if not route_in_green(t['route'], green_orig, n2o): continue
            r_new = two_opt(t['route'], dm, dw, dv, tw_s, tw_e, veh['vt'], passes=3)
            c2, tr2, pen2, co2, ok, et2 = eval_r(r_new, dm, dw, dv, tw_s, tw_e, veh['vt'], t['start'])
            if ok and tr2 + pen2 < t['travel'] + t['penalty'] - 1:
                veh['trips'][ti] = {'route': r_new, 'vt': veh['vt'], 'start': t['start'],
                                    'end': et2, 'cost': c2, 'travel': tr2,
                                    'penalty': pen2, 'carbon': co2}
                n_reimproved += 1
    log(f"  重优化 {n_reimproved} 条推迟trip")

    bt, bv, btr, bpen, bco2 = sched_cost(best_sched)
    viol = count_violations(best_sched, green_orig, n2o)
    log(f"\n最终: {bv}辆 {bt:.2f} (启动{bv*STARTUP:.0f} 行驶{btr:.2f} 惩罚{bpen:.2f} CO2={bco2:.2f})")
    log(f"绿色区违规: {viol} {'✅' if viol==0 else '❌'}")

    return rvt, coords, dm, dw, dv, tw_s, tw_e, green_orig, n2o, best_sched

# ── 对比报告 ─────────────────────────────────────
def compare_report(bt1, bco2_1, bv1, sched1, bt2, bco2_2, bv2, sched2):
    log("\n" + "="*60)
    log("          问题1 vs 问题2 对比分析")
    log("="*60)
    for name, v1, v2 in [("总成本(元)", bt1, bt2), ("车辆数", bv1, bv2), ("碳排放(kg)", bco2_1, bco2_2)]:
        d = v2 - v1
        log(f"  {name:<12} Q1:{v1:>10.2f}  Q2:{v2:>10.2f}  Δ:{'+' if d>0 else ''}{d:.2f}")
    for sched, lbl in [(sched1, "问题1"), (sched2, "问题2")]:
        vc = {}
        for s in sched: vc[s['vt']['name']] = vc.get(s['vt']['name'], 0) + 1
        ev_n = sum(c for k, c in vc.items() if '新能源' in k)
        log(f"  {lbl}: {dict(sorted(vc.items()))}  新能源占比:{ev_n/sum(vc.values())*100:.1f}%")
    log(f"\n政策影响:")
    log(f"  总成本增加: {bt2-bt1:.2f}元 (+{(bt2-bt1)/bt1*100:.1f}%)")
    log(f"  碳排放减少: {bco2_1-bco2_2:.2f}kg (-{(bco2_1-bco2_2)/bco2_1*100:.1f}%)")

# ── 保存 & 可视化 ─────────────────────────────────
def save_schedule(best_flex, dm, dw, dv, tw_s, tw_e, n2o, green_orig, fname):
    bt, bv, btr, bpen, bco2 = sched_cost(best_flex)
    vc = {}; tc = []
    for s in best_flex:
        vc[s['vt']['name']] = vc.get(s['vt']['name'], 0) + 1; tc.append(len(s['trips']))
    log(f"\n{fname}: {bv}辆 趟/均:{sum(tc)/bv:.2f} 总计:{bt:.2f} CO2:{bco2:.2f}")
    for n, c in sorted(vc.items()): log(f"  {n}:{c}辆")
    rows = []
    for vid, s in enumerate(best_flex, 1):
        vt = s['vt']
        for tid, t in enumerate(s['trips'], 1):
            r = t['route']; c = r[1:-1]; orig = [n2o.get(x, x) for x in c]
            ig = any(n2o.get(x, x) in green_orig for x in c)
            w = sum(dw.get(x, 0) for x in c); v = sum(dv.get(x, 0) for x in c)
            st = t['start']
            rows.append({'车辆': vid, '车型': vt['name'], '第几趟': tid,
                         '出发': f'{int(st)}:{int((st%1)*60):02d}',
                         '返回': f'{int(t["end"])}:{int((t["end"]%1)*60):02d}',
                         '经绿色区': '是' if ig else '否',
                         '客户(原始ID)': str(orig),
                         '载重kg': round(w, 2), '体积m3': round(v, 3),
                         '行驶元': round(t['travel'], 2), '惩罚元': round(t['penalty'], 2),
                         'CO2kg': round(t['carbon'], 2)})
    pd.DataFrame(rows).to_csv(f'{BASE}/{fname}', index=False, encoding='utf-8-sig')
    log(f"已保存: {fname}")

def visualize_p2(rvt, coords, green_orig, n2o, fname):
    fig, ax = plt.subplots(figsize=(13, 11))
    th = np.linspace(0, 2*np.pi, 300)
    ax.fill(GREEN_R*np.cos(th), GREEN_R*np.sin(th), color='lightgreen', alpha=0.25)
    ax.plot(GREEN_R*np.cos(th), GREEN_R*np.sin(th), 'g--', lw=1.5, label='绿色区(10km)')
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(rvt), 1)))
    def co(n): return coords.get(n2o.get(n, n), (0, 0))
    for idx, (r, vt) in enumerate(rvt):
        xs = [co(n)[0] for n in r]; ys = [co(n)[1] for n in r]
        ax.plot(xs, ys, '--' if vt['type'] == 'ev' else '-',
                color=colors[idx % len(colors)], lw=1.6 if vt['type'] == 'ev' else 0.9, alpha=0.75)
    for c in range(1, 99):
        x, y = coords[c]; col = 'red' if c in green_orig else 'steelblue'
        ax.scatter(x, y, c=col, s=55 if c in green_orig else 28, zorder=5)
        ax.text(x+.3, y+.3, str(c), fontsize=5, alpha=0.8)
    ax.scatter(*coords[0], c='black', s=250, marker='*', zorder=10)
    ax.scatter(0, 0, c='orange', s=100, marker='^', zorder=10)
    handles = [Line2D([0],[0],ls='-',c='gray',lw=1.5,label='燃油车'),
               Line2D([0],[0],ls='--',c='blue',lw=2,label='新能源车'),
               Line2D([0],[0],ls='none',marker='o',c='red',ms=8,label='绿色区客户'),
               Line2D([0],[0],ls='none',marker='*',c='black',ms=12,label='配送中心')]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.set_xlabel('X(km)'); ax.set_ylabel('Y(km)')
    ax.set_title(f'问题2：绿色区限行调度（{len(rvt)}条子路线）')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal'); plt.tight_layout()
    plt.savefig(f'{BASE}/{fname}', dpi=140, bbox_inches='tight'); plt.close()
    log(f"路线图: {fname}")

def visualize_costs(sched1, sched2, bt1, bv1, bco2_1, bt2, bv2, bco2_2):
    """生成6张对比可视化图，保存为 p2_cost_analysis.png"""
    df1 = _sched_df(sched1)
    df2 = _sched_df(sched2)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('问题1 vs 问题2 成本对比分析', fontsize=15, fontweight='bold')

    # ── 图1: 总成本构成对比（分组柱状图）
    ax = axes[0, 0]
    cats = ['启动成本', '行驶能耗', '时间窗惩罚']
    v1 = [bv1*STARTUP, df1['行驶元'].sum(), df1['惩罚元'].sum()]
    v2 = [bv2*STARTUP, df2['行驶元'].sum(), df2['惩罚元'].sum()]
    x = np.arange(3); w = 0.35
    b1 = ax.bar(x - w/2, v1, w, label=f'问题1 ({bt1:.0f}元)', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x + w/2, v2, w, label=f'问题2 ({bt2:.0f}元)', color='#DD8452', alpha=0.85)
    ax.bar_label(b1, fmt='%.0f', fontsize=8)
    ax.bar_label(b2, fmt='%.0f', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel('元'); ax.set_title('① 总成本构成对比')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # ── 图2: 成本构成饼图并排
    ax = axes[0, 1]
    ax.axis('off')
    # 左右两个小饼图用 inset
    ax_p1 = fig.add_axes([0.38, 0.58, 0.12, 0.28])
    ax_p2 = fig.add_axes([0.52, 0.58, 0.12, 0.28])
    colors_pie = ['#4C72B0', '#55A868', '#C44E52']
    for axi, vals, lbl in [(ax_p1, v1, '问题1'), (ax_p2, v2, '问题2')]:
        wedges, _, autotexts = axi.pie(vals, colors=colors_pie, autopct='%1.0f%%',
                                       startangle=90, textprops={'fontsize': 7})
        axi.set_title(lbl, fontsize=9)
    ax_p1.legend(['启动','行驶','惩罚'], loc='lower center', fontsize=6,
                 bbox_to_anchor=(0.5, -0.25), ncol=1)
    ax.set_title('② 成本比例', x=0.5, y=0.95)

    # ── 图3: 车辆类型与数量对比
    ax = axes[0, 2]
    def vt_breakdown(sched):
        vc = {}
        for s in sched: vc[s['vt']['name']] = vc.get(s['vt']['name'], 0) + 1
        return vc
    vc1, vc2 = vt_breakdown(sched1), vt_breakdown(sched2)
    all_types = sorted(set(list(vc1.keys()) + list(vc2.keys())))
    x = np.arange(len(all_types)); w = 0.35
    ax.bar(x - w/2, [vc1.get(t, 0) for t in all_types], w, color='#4C72B0', alpha=0.85, label='问题1')
    ax.bar(x + w/2, [vc2.get(t, 0) for t in all_types], w, color='#DD8452', alpha=0.85, label='问题2')
    ax.set_xticks(x); ax.set_xticklabels(all_types, rotation=15, fontsize=8)
    ax.set_ylabel('辆数'); ax.set_title(f'③ 车辆类型对比 (Q1:{bv1}辆 Q2:{bv2}辆)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # ── 图4: 每趟子路线行驶能耗分布（箱线图）
    ax = axes[1, 0]
    data1 = df1['行驶元'].tolist()
    data2 = df2['行驶元'].tolist()
    bp = ax.boxplot([data1, data2], labels=['问题1', '问题2'],
                    patch_artist=True, notch=False)
    bp['boxes'][0].set_facecolor('#4C72B0'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#DD8452'); bp['boxes'][1].set_alpha(0.7)
    ax.set_ylabel('元/趟')
    ax.set_title(f'④ 单趟行驶能耗分布\n(Q1均值:{np.mean(data1):.0f} Q2均值:{np.mean(data2):.0f})')
    ax.grid(axis='y', alpha=0.3)

    # ── 图5: 碳排放对比
    ax = axes[1, 1]
    # 按车型分类碳排放
    def co2_by_type(sched):
        res = {}
        for s in sched:
            n = s['vt']['name']
            res[n] = res.get(n, 0) + sum(t['carbon'] for t in s['trips'])
        return res
    co1, co2_d = co2_by_type(sched1), co2_by_type(sched2)
    all_t = sorted(set(list(co1.keys()) + list(co2_d.keys())))
    x = np.arange(len(all_t)); w = 0.35
    ax.bar(x - w/2, [co1.get(t, 0) for t in all_t], w, color='#4C72B0', alpha=0.85, label='问题1')
    ax.bar(x + w/2, [co2_d.get(t, 0) for t in all_t], w, color='#DD8452', alpha=0.85, label='问题2')
    ax.set_xticks(x); ax.set_xticklabels(all_t, rotation=15, fontsize=8)
    ax.set_ylabel('kg CO₂'); ax.set_title(f'⑤ 各车型碳排放 (Q1:{bco2_1:.0f} Q2:{bco2_2:.0f} kg)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # ── 图6: 政策影响汇总文字表
    ax = axes[1, 2]; ax.axis('off')
    rows_data = [
        ['指标', '问题1', '问题2', '变化'],
        ['总成本(元)', f'{bt1:.0f}', f'{bt2:.0f}',
         f'{bt2-bt1:+.0f} ({(bt2-bt1)/bt1*100:+.1f}%)'],
        ['车辆数', str(bv1), str(bv2), f'{bv2-bv1:+d}'],
        ['启动成本', f'{bv1*STARTUP:.0f}', f'{bv2*STARTUP:.0f}',
         f'{(bv2-bv1)*STARTUP:+.0f}'],
        ['行驶能耗', f'{df1["行驶元"].sum():.0f}', f'{df2["行驶元"].sum():.0f}',
         f'{df2["行驶元"].sum()-df1["行驶元"].sum():+.0f}'],
        ['时间窗惩罚', f'{df1["惩罚元"].sum():.0f}', f'{df2["惩罚元"].sum():.0f}',
         f'{df2["惩罚元"].sum()-df1["惩罚元"].sum():+.0f}'],
        ['碳排放(kg)', f'{bco2_1:.0f}', f'{bco2_2:.0f}',
         f'{bco2_2-bco2_1:+.0f} ({(bco2_2-bco2_1)/bco2_1*100:+.1f}%)'],
        ['新能源占比', f'{sum(1 for s in sched1 if s["vt"]["type"]=="ev")/bv1*100:.1f}%',
         f'{sum(1 for s in sched2 if s["vt"]["type"]=="ev")/bv2*100:.1f}%', ''],
    ]
    table = ax.table(cellText=rows_data[1:], colLabels=rows_data[0],
                     cellLoc='center', loc='center',
                     colColours=['#D5E8F7']*4)
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.2, 1.6)
    # 变化列：正数红色，负数绿色
    for i, row in enumerate(rows_data[1:], 1):
        cell = table[i, 3]
        val = row[3]
        if val.startswith('+') and val != '+0': cell.set_facecolor('#FFD6D6')
        elif val.startswith('-'): cell.set_facecolor('#D6FFD6')
    ax.set_title('⑥ 政策影响汇总', fontsize=10, pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = f'{BASE}/p2_cost_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"费用分析图: {save_path}")

def _sched_df(sched):
    """把 flex schedule 转成 DataFrame（含行驶元、惩罚元列）"""
    rows = []
    for s in sched:
        for t in s['trips']:
            rows.append({'车型': s['vt']['name'], '行驶元': t['travel'],
                         '惩罚元': t['penalty'], 'CO2': t['carbon']})
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────
if __name__ == '__main__':
    log(f"Python {sys.version.split()[0]}")
    log("="*60 + "\n问题2：8-16时禁止燃油车进入半径10km绿色区\n" + "="*60)

    # 问题1基准
    log("\n>>> 问题1基准 <<<")
    from solve_p1 import solve as solve_p1
    rvt1, _, dm1, dw1, dv1, tw_s1, tw_e1, green1, n2o1, _, sched1 = solve_p1()
    bt1, bv1, _, _, bco2_1 = sched_cost(sched1)
    log(f"问题1: {bv1}辆 {bt1:.2f}元 CO2={bco2_1:.2f}kg")

    # 问题2
    log("\n>>> 问题2 <<<")
    rvt2, coords, dm, dw, dv, tw_s, tw_e, green_orig, n2o, sched2 = solve_p2()
    bt2, bv2, _, _, bco2_2 = sched_cost(sched2)

    save_schedule(sched2, dm, dw, dv, tw_s, tw_e, n2o, green_orig, 'p2_schedule.csv')
    compare_report(bt1, bco2_1, bv1, sched1, bt2, bco2_2, bv2, sched2)
    visualize_p2(rvt2, coords, green_orig, n2o, 'p2_routes.png')
    visualize_costs(sched1, sched2, bt1, bv1, bco2_1, bt2, bv2, bco2_2)
    log("\n===== 问题2 完成 =====")
