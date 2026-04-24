# -*- coding: utf-8 -*-
"""华中杯A题 问题3：动态事件下的实时车辆调度策略"""
import sys, io, os, time, math, random, copy
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.chdir(r'c:\Users\LENOVO\Desktop\华中杯1.1')
sys.path.insert(0, r'c:\Users\LENOVO\Desktop\华中杯1.1')

from solve_p1 import (
    load_data, eval_r, opt_start, two_opt, or_opt,
    VEHICLE_TYPES, STARTUP, WAIT_H, LATE_H, SVC_H,
    GREEN_R, BASE, log
)
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings; warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
random.seed(42)

def _sched_cost_flex(sched):
    """计算 flex schedule 总成本"""
    n = len(sched); tr = pen = co2 = 0.
    for s in sched:
        for t in s['trips']:
            tr += t['travel']; pen += t['penalty']; co2 += t['carbon']
    return n*STARTUP+tr+pen, n, tr, pen, co2

# ═══════════════════════════════════════════════════
# 动态调度核心状态
# ═══════════════════════════════════════════════════

class DynamicSchedule:
    """
    维护当前调度状态，支持实时事件响应
    每辆车的状态：
      - trips: 未来还未开始的子路线列表（已出发的不可更改）
      - current_trip: 正在执行的子路线（None表示在depot）
      - current_time: 当前时刻
    """
    def __init__(self, flex_sched, dm, dw, dv, tw_s, tw_e, n2o, coords):
        self.dm = dm
        self.dw = dw
        self.dv = dv
        self.tw_s = tw_s
        self.tw_e = tw_e
        self.n2o = n2o
        self.coords = coords
        # deep copy 调度方案
        self.vehicles = [
            {'vt': s['vt'],
             'trips': [dict(t) for t in s['trips']],
             'done_trips': []}
            for s in flex_sched
        ]
        self.event_log = []  # 事件记录
        self._recompute_all_costs()

    def _recompute_all_costs(self):
        """重新计算所有路线成本"""
        for veh in self.vehicles:
            for t in veh['trips']:
                c, tr, pen, co2, ok, et = eval_r(
                    t['route'], self.dm, self.dw, self.dv,
                    self.tw_s, self.tw_e, veh['vt'], t['start'])
                t.update({'cost': c, 'travel': tr, 'penalty': pen,
                          'carbon': co2, 'end': et})

    def total_cost(self):
        n = len(self.vehicles)
        tr = sum(t['travel'] for v in self.vehicles for t in v['trips'] + v['done_trips'])
        pen = sum(t['penalty'] for v in self.vehicles for t in v['trips'] + v['done_trips'])
        co2 = sum(t['carbon'] for v in self.vehicles for t in v['trips'] + v['done_trips'])
        return n * STARTUP + tr + pen, n, tr, pen, co2

    def get_all_pending_customers(self, current_time):
        """获取当前时刻还未开始服务的客户节点"""
        pending = []
        for vi, veh in enumerate(self.vehicles):
            for ti, t in enumerate(veh['trips']):
                if t['start'] > current_time:
                    pending.append({'vi': vi, 'ti': ti, 'route': t['route'],
                                    'start': t['start'], 'vt': veh['vt']})
        return pending

    # ── 事件1：订单取消 ──────────────────────────
    def event_cancel_order(self, target_cust_orig, current_time):
        """
        取消某原始客户ID的所有未送达订单
        current_time: 事件发生时刻
        """
        log(f"\n  [事件] 订单取消: 客户{target_cust_orig}, 当前时刻 {_fmt_h(current_time)}")
        removed = 0
        cost_before, *_ = self.total_cost()

        for veh in self.vehicles:
            new_trips = []
            for t in veh['trips']:
                if t['start'] <= current_time:
                    # 已在途中，无法取消
                    new_trips.append(t)
                    continue
                # 从路线中删除该客户节点
                r = t['route']
                nodes_to_remove = [nd for nd in r[1:-1]
                                   if self.n2o.get(nd, nd) == target_cust_orig]
                if not nodes_to_remove:
                    new_trips.append(t)
                    continue
                new_r = [nd for nd in r if nd not in nodes_to_remove or nd == 0]
                if len(new_r) < 3:
                    # 路线为空，删除这趟
                    removed += len(nodes_to_remove)
                    log(f"    删除空路线 (原:{_fmt_route(r, self.n2o)})")
                    continue
                # 重新评估
                c, tr, pen, co2, ok, et = eval_r(
                    new_r, self.dm, self.dw, self.dv,
                    self.tw_s, self.tw_e, veh['vt'], t['start'])
                t_new = dict(t)
                t_new.update({'route': new_r, 'cost': c, 'travel': tr,
                              'penalty': pen, 'carbon': co2, 'end': et})
                new_trips.append(t_new)
                removed += len(nodes_to_remove)
                log(f"    路线更新: {_fmt_route(r, self.n2o)} → {_fmt_route(new_r, self.n2o)}")
            veh['trips'] = new_trips

        # 清除空车
        self.vehicles = [v for v in self.vehicles if v['trips'] or v['done_trips']]
        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '订单取消', 'customer': target_cust_orig,
            'time': current_time, 'removed_nodes': removed,
            'cost_before': cost_before, 'cost_after': cost_after,
            'saving': cost_before - cost_after
        })
        log(f"    取消{removed}个节点, 成本 {cost_before:.2f}→{cost_after:.2f} (节省{cost_before-cost_after:.2f})")
        return cost_after

    # ── 事件2：新增订单 ──────────────────────────
    def event_add_order(self, new_cust_id, new_w, new_v, new_tw_s, new_tw_e,
                        new_x, new_y, current_time):
        """
        新增一个客户订单，用最小插入成本法找最优插入位置
        new_cust_id: 新客户ID（唯一）
        """
        log(f"\n  [事件] 新增订单: 客户{new_cust_id}, w={new_w:.0f}kg, "
            f"v={new_v:.3f}m³, TW=[{_fmt_h(new_tw_s)},{_fmt_h(new_tw_e)}], "
            f"当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()

        # 扩展距离矩阵（新节点与depot及所有已有节点计算欧氏距离）
        orig_x, orig_y = self.coords.get(0, (20, 20))
        def dist_to_new(nd):
            ox = self.coords.get(self.n2o.get(nd, nd), (0, 0))
            return math.sqrt((ox[0] - new_x)**2 + (ox[1] - new_y)**2)

        # 更新数据
        self.dw[new_cust_id] = new_w
        self.dv[new_cust_id] = new_v
        self.tw_s[new_cust_id] = new_tw_s
        self.tw_e[new_cust_id] = new_tw_e
        self.coords[new_cust_id] = (new_x, new_y)
        self.n2o[new_cust_id] = new_cust_id
        # 更新距离矩阵（仅对新节点行列）
        n_old = self.dm.shape[0]
        if new_cust_id >= n_old:
            new_size = new_cust_id + 1
            new_dm = np.zeros((new_size, new_size))
            new_dm[:n_old, :n_old] = self.dm
            # 填充新节点距离
            for nd in range(new_size):
                if nd == new_cust_id: continue
                ox = self.coords.get(self.n2o.get(nd, nd), (0, 0))
                d = math.sqrt((ox[0]-new_x)**2 + (ox[1]-new_y)**2)
                new_dm[nd][new_cust_id] = d
                new_dm[new_cust_id][nd] = d
            self.dm = new_dm

        # 寻找最佳插入位置
        best_cost_inc = 1e18
        best_vi = -1; best_ti = -1; best_ins = -1; best_route = None

        for vi, veh in enumerate(self.vehicles):
            for ti, t in enumerate(veh['trips']):
                if t['start'] <= current_time: continue  # 已出发
                r = t['route']
                w_cur = sum(self.dw.get(c, 0) for c in r[1:-1])
                v_cur = sum(self.dv.get(c, 0) for c in r[1:-1])
                if w_cur + new_w > veh['vt']['max_w']: continue
                if v_cur + new_v > veh['vt']['max_v']: continue
                # 尝试每个插入位置
                for ins in range(1, len(r)):
                    new_r = r[:ins] + [new_cust_id] + r[ins:]
                    c_new, tr_new, pen_new, co2_new, ok, et_new = eval_r(
                        new_r, self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    if not ok: continue
                    inc = (tr_new + pen_new) - (t['travel'] + t['penalty'])
                    if inc < best_cost_inc:
                        best_cost_inc = inc
                        best_vi = vi; best_ti = ti; best_ins = ins
                        best_route = (new_r, t['start'], et_new, c_new, tr_new, pen_new, co2_new)

        if best_vi >= 0:
            # 插入
            new_r, st, et, c, tr, pen, co2 = best_route
            old_r = self.vehicles[best_vi]['trips'][best_ti]['route']
            self.vehicles[best_vi]['trips'][best_ti].update({
                'route': new_r, 'end': et, 'cost': c,
                'travel': tr, 'penalty': pen, 'carbon': co2})
            log(f"    插入到 车辆{best_vi+1}趟{best_ti+1}: "
                f"{_fmt_route(old_r, self.n2o)} → {_fmt_route(new_r, self.n2o)}")
            log(f"    成本增量: {best_cost_inc:+.2f}")
        else:
            # 无法插入已有路线，新开一辆车
            vt = next((v for v in VEHICLE_TYPES
                       if v['max_w'] >= new_w and v['max_v'] >= new_v), VEHICLE_TYPES[0])
            st = opt_start([0, new_cust_id, 0], self.dm, self.tw_s)
            st = max(st, current_time + 0.5)
            c, tr, pen, co2, ok, et = eval_r(
                [0, new_cust_id, 0], self.dm, self.dw, self.dv,
                self.tw_s, self.tw_e, vt, st)
            new_trip = {'route': [0, new_cust_id, 0], 'vt': vt, 'start': st,
                        'end': et, 'cost': c, 'travel': tr, 'penalty': pen, 'carbon': co2}
            self.vehicles.append({'vt': vt, 'trips': [new_trip], 'done_trips': []})
            log(f"    新开一辆 {vt['name']} 服务新客户{new_cust_id}")
            best_cost_inc = STARTUP + tr + pen

        cost_after, *_ = self.total_cost()
        actual_inc = cost_after - cost_before
        self.event_log.append({
            'type': '新增订单', 'customer': new_cust_id,
            'time': current_time, 'cost_before': cost_before,
            'cost_after': cost_after, 'cost_inc': actual_inc
        })
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} (增加{actual_inc:.2f})")
        return cost_after

    # ── 事件3：配送地址变更 ──────────────────────
    def event_address_change(self, cust_orig, new_x, new_y, current_time):
        """
        某客户配送地址变更，更新坐标+距离矩阵，对涉及路线重做2-opt
        """
        log(f"\n  [事件] 地址变更: 客户{cust_orig} → ({new_x:.2f},{new_y:.2f}), "
            f"当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()
        old_x, old_y = self.coords.get(cust_orig, (0, 0))

        # 更新坐标
        self.coords[cust_orig] = (new_x, new_y)
        # 更新虚拟节点坐标映射（n2o中所有映射到cust_orig的节点）
        affected_nodes = [nd for nd, orig in self.n2o.items() if orig == cust_orig]
        # 重新计算距离矩阵（影响行列）
        n = self.dm.shape[0]
        for nd in affected_nodes:
            if nd >= n: continue
            for other in range(n):
                ox = self.coords.get(self.n2o.get(other, other), (0, 0))
                d = math.sqrt((ox[0]-new_x)**2 + (ox[1]-new_y)**2)
                self.dm[nd][other] = d
                self.dm[other][nd] = d

        # 对含该客户的未出发路线做局部重优化
        for veh in self.vehicles:
            for ti, t in enumerate(veh['trips']):
                if t['start'] <= current_time: continue
                if not any(self.n2o.get(nd, nd) == cust_orig for nd in t['route'][1:-1]):
                    continue
                # 重新评估成本
                r_new = two_opt(t['route'], self.dm, self.dw, self.dv,
                                self.tw_s, self.tw_e, veh['vt'], passes=2)
                c, tr, pen, co2, ok, et = eval_r(
                    r_new, self.dm, self.dw, self.dv,
                    self.tw_s, self.tw_e, veh['vt'], t['start'])
                old_r = t['route']
                t.update({'route': r_new, 'end': et, 'cost': c,
                          'travel': tr, 'penalty': pen, 'carbon': co2})
                log(f"    路线重优化: {_fmt_route(old_r, self.n2o)} → {_fmt_route(r_new, self.n2o)}")

        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '地址变更', 'customer': cust_orig,
            'time': current_time, 'old_pos': (old_x, old_y),
            'new_pos': (new_x, new_y),
            'cost_before': cost_before, 'cost_after': cost_after
        })
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} (变化{cost_after-cost_before:+.2f})")
        return cost_after

    # ── 事件4：时间窗调整 ─────────────────────────
    def event_tw_adjust(self, cust_orig, new_tw_s, new_tw_e, current_time):
        """
        某客户时间窗调整，重新评估受影响路线，必要时重排序
        """
        log(f"\n  [事件] 时间窗调整: 客户{cust_orig} → "
            f"[{_fmt_h(new_tw_s)},{_fmt_h(new_tw_e)}], "
            f"当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()
        affected_nodes = [nd for nd, orig in self.n2o.items() if orig == cust_orig]
        for nd in affected_nodes:
            self.tw_s[nd] = new_tw_s
            self.tw_e[nd] = new_tw_e
        # 直接更新原始ID的时间窗
        self.tw_s[cust_orig] = new_tw_s
        self.tw_e[cust_orig] = new_tw_e

        # 对含该客户的未出发路线重评估
        for veh in self.vehicles:
            for ti, t in enumerate(veh['trips']):
                if t['start'] <= current_time: continue
                if not any(self.n2o.get(nd, nd) == cust_orig for nd in t['route'][1:-1]):
                    continue
                # 尝试2-opt改善惩罚
                r_new = two_opt(t['route'], self.dm, self.dw, self.dv,
                                self.tw_s, self.tw_e, veh['vt'], passes=3)
                c, tr, pen, co2, ok, et = eval_r(
                    r_new, self.dm, self.dw, self.dv,
                    self.tw_s, self.tw_e, veh['vt'], t['start'])
                old_pen = t['penalty']
                t.update({'route': r_new, 'end': et, 'cost': c,
                          'travel': tr, 'penalty': pen, 'carbon': co2})
                log(f"    路线重评估: {_fmt_route(r_new, self.n2o)} "
                    f"惩罚 {old_pen:.2f}→{pen:.2f}")

        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '时间窗调整', 'customer': cust_orig,
            'time': current_time, 'new_tw': (new_tw_s, new_tw_e),
            'cost_before': cost_before, 'cost_after': cost_after
        })
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} (变化{cost_after-cost_before:+.2f})")
        return cost_after


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════
def _fmt_h(h):
    return f'{int(h)}:{int((h%1)*60):02d}'

def _fmt_route(r, n2o):
    return str([n2o.get(nd, nd) for nd in r])


# ═══════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════
def visualize_dynamic(ds_initial, ds_after_events, coords, green_orig, n2o, events, save_path):
    """对比初始方案 vs 事件响应后方案"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('问题3：动态事件响应前后路线对比', fontsize=14, fontweight='bold')

    def draw_schedule(ax, ds, title):
        th = np.linspace(0, 2*np.pi, 300)
        ax.fill(GREEN_R*np.cos(th), GREEN_R*np.sin(th),
                color='lightgreen', alpha=0.2, label='绿色区')
        ax.plot(GREEN_R*np.cos(th), GREEN_R*np.sin(th), 'g--', lw=1.2)
        colors = plt.cm.tab20(np.linspace(0, 1, len(ds.vehicles)))
        def co(nd):
            orig = n2o.get(nd, nd)
            return coords.get(orig, (0, 0))
        for vi, veh in enumerate(ds.vehicles):
            c = colors[vi % len(colors)]
            for t in veh['trips']:
                xs = [co(nd)[0] for nd in t['route']]
                ys = [co(nd)[1] for nd in t['route']]
                ls = '--' if veh['vt']['type'] == 'ev' else '-'
                ax.plot(xs, ys, ls, color=c, lw=1.2, alpha=0.7)
        for cid in range(1, 99):
            x, y = coords.get(cid, (0, 0))
            col = 'red' if cid in green_orig else 'steelblue'
            ax.scatter(x, y, c=col, s=40 if cid in green_orig else 22, zorder=5)
            ax.text(x+.3, y+.3, str(cid), fontsize=4.5, alpha=0.7)
        dx, dy = coords.get(0, (20, 20))
        ax.scatter(dx, dy, c='black', s=200, marker='*', zorder=10)
        ax.scatter(0, 0, c='orange', s=80, marker='^', zorder=10)
        # 标注事件影响的客户
        for ev in events:
            cid = ev.get('customer', None)
            if cid and cid in coords:
                ex, ey = coords[cid]
                ax.scatter(ex, ey, c='purple', s=120, marker='D', zorder=12)
                ax.annotate(ev['type'][:2], (ex, ey), fontsize=7,
                            color='purple', fontweight='bold',
                            xytext=(ex+1, ey+1))
        ct, cn, ctr, cpen, cco2 = ds.total_cost()
        ax.set_title(f'{title}\n总成本:{ct:.0f}元 车辆:{cn}辆 惩罚:{cpen:.0f}元', fontsize=10)
        ax.set_xlabel('X(km)'); ax.set_ylabel('Y(km)')
        ax.grid(True, alpha=0.25); ax.set_aspect('equal')
        handles = [Line2D([0],[0],ls='-',c='gray',lw=1.5,label='燃油车'),
                   Line2D([0],[0],ls='--',c='blue',lw=2,label='新能源车'),
                   Line2D([0],[0],ls='none',marker='o',c='red',ms=7,label='绿色区客户'),
                   Line2D([0],[0],ls='none',marker='D',c='purple',ms=8,label='事件客户')]
        ax.legend(handles=handles, fontsize=7, loc='upper right')

    draw_schedule(axes[0], ds_initial, '初始方案（问题1最优解）')
    draw_schedule(axes[1], ds_after_events, '事件响应后调整方案')
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    log(f"动态路线图: {save_path}")


def visualize_event_impact(event_log, save_path):
    """事件影响分析图"""
    if not event_log: return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('问题3：突发事件影响分析', fontsize=13, fontweight='bold')

    # ── 图1: 各事件成本变化
    ax = axes[0]
    types = [e['type'] for e in event_log]
    changes = [e['cost_after'] - e['cost_before'] for e in event_log]
    colors_ev = ['green' if c < 0 else 'red' for c in changes]
    bars = ax.barh(range(len(event_log)), changes, color=colors_ev, alpha=0.8)
    ax.set_yticks(range(len(event_log)))
    ax.set_yticklabels([f"{e['type']}\n客户{e['customer']}" for e in event_log], fontsize=8)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('成本变化(元)')
    ax.set_title('① 各事件成本变化')
    for bar, val in zip(bars, changes):
        ax.text(val + (50 if val >= 0 else -50), bar.get_y() + bar.get_height()/2,
                f'{val:+.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # ── 图2: 成本累计变化折线
    ax = axes[1]
    costs = [event_log[0]['cost_before']] + [e['cost_after'] for e in event_log]
    labels = ['初始'] + [f"事件{i+1}\n{e['type'][:3]}" for i, e in enumerate(event_log)]
    ax.plot(range(len(costs)), costs, 'bo-', lw=2, ms=7)
    for i, (x, y) in enumerate(zip(range(len(costs)), costs)):
        ax.annotate(f'{y:.0f}', (x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('总成本(元)')
    ax.set_title('② 成本动态变化轨迹')
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(costs)), costs, alpha=0.15, color='blue')
    # 让 Y 轴范围聚焦变化区间
    margin = (max(costs) - min(costs)) * 2 or 500
    ax.set_ylim(min(costs) - margin, max(costs) + margin)

    # ── 图3: 事件类型与响应效果
    ax = axes[2]; ax.axis('off')
    rows = [['事件', '客户', '发生时刻', '成本变化', '说明']]
    for e in event_log:
        change = e['cost_after'] - e['cost_before']
        sign = '↓' if change < 0 else '↑'
        rows.append([
            e['type'], str(e['customer']),
            _fmt_h(e['time']),
            f"{sign}{abs(change):.0f}元",
            '节省' if change < 0 else '增加'
        ])
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc='center', loc='center', colColours=['#D5E8F7']*5)
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for i, e in enumerate(event_log, 1):
        change = e['cost_after'] - e['cost_before']
        col = '#D6FFD6' if change < 0 else '#FFD6D6'
        table[i, 3].set_facecolor(col)
    ax.set_title('③ 事件响应汇总', fontsize=10, pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    log(f"事件影响图: {save_path}")


# ═══════════════════════════════════════════════════
# 主程序：设计并演示突发场景
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    log(f"Python {sys.version.split()[0]}")
    log("="*60)
    log("问题3：动态事件下的实时车辆调度策略")
    log("="*60)

    # 1. 加载问题1最优解作为初始方案
    log("\n>>> 加载问题1最优方案 <<<")
    from solve_p1 import solve as solve_p1
    rvt, coords, dm, dw, dv, tw_s, tw_e, green, n2o, schedule, flex_sched = solve_p1()
    bt0, bv0, btr0, bpen0, bco2_0 = _sched_cost_flex(flex_sched)
    green_orig = {c for c in green if c < 99}
    log(f"初始方案: {bv0}辆 总成本:{bt0:.2f}元 CO2:{bco2_0:.2f}kg")

    # 2. 初始化动态调度系统
    ds = DynamicSchedule(flex_sched, dm, dw, dv, tw_s, tw_e, n2o, coords)
    ds_snapshot = copy.deepcopy(ds)  # 保存初始状态用于对比可视化

    log("\n" + "="*60)
    log("演示场景一：订单取消 + 新增订单")
    log("="*60)

    # ── 场景一 Step1: 10:30 客户9取消订单（11:19出发还来得及）
    log("\n── Step1: 10:30 客户9取消订单（其路线11:19出发，可处理）")
    ds.event_cancel_order(target_cust_orig=9, current_time=10.5)

    # ── 场景一 Step2: 11:00 新增紧急订单客户999（位于市区，小批量）
    log("\n── Step2: 11:00 新增紧急订单（客户999, 靠近客户坐标中心）")
    ds.event_add_order(
        new_cust_id=999,
        new_w=500.0, new_v=1.5,
        new_tw_s=13.0, new_tw_e=15.0,
        new_x=3.0, new_y=5.0,
        current_time=11.0
    )

    log("\n" + "="*60)
    log("演示场景二：地址变更 + 时间窗调整")
    log("="*60)

    # ── 场景二 Step3: 13:30 客户52地址变更（14:18出发，还未出发）
    log("\n── Step3: 13:30 客户52地址变更（向市中心靠近）")
    old_x52, old_y52 = coords.get(52, (0, 0))
    ds.event_address_change(
        cust_orig=52,
        new_x=old_x52 * 0.5,
        new_y=old_y52 * 0.5,
        current_time=13.5
    )

    # ── 场景二 Step4: 14:00 客户55时间窗推迟2小时（16:03出发还来得及）
    log("\n── Step4: 14:00 客户55时间窗推迟2小时（16:03出发）")
    orig_s55 = tw_s.get(55, 0)
    orig_e55 = tw_e.get(55, 24)
    ds.event_tw_adjust(
        cust_orig=55,
        new_tw_s=orig_s55 + 2.0,
        new_tw_e=orig_e55 + 2.0,
        current_time=14.0
    )

    # 3. 汇总报告
    log("\n" + "="*60)
    log("动态调度最终方案汇总")
    log("="*60)
    bt_f, bv_f, btr_f, bpen_f, bco2_f = ds.total_cost()
    log(f"\n{'指标':<15} {'初始方案':>12} {'调整后':>12} {'变化':>12}")
    log("-"*51)
    for name, v0, vf in [
        ("总成本(元)", bt0, bt_f),
        ("车辆数",     bv0, bv_f),
        ("行驶能耗",   btr0, btr_f),
        ("时间窗惩罚", bpen0, bpen_f),
        ("碳排放(kg)", bco2_0, bco2_f)
    ]:
        d = vf - v0
        log(f"  {name:<13} {v0:>12.2f} {vf:>12.2f} {'+' if d>=0 else ''}{d:.2f}")

    log(f"\n事件记录({len(ds.event_log)}次):")
    for i, e in enumerate(ds.event_log, 1):
        change = e['cost_after'] - e['cost_before']
        log(f"  {i}. {e['type']} 客户{e['customer']} @{_fmt_h(e['time'])} → "
            f"成本变化{change:+.2f}元")

    # 保存调度表
    rows = []
    for vid, veh in enumerate(ds.vehicles, 1):
        for tid, t in enumerate(veh['trips'], 1):
            r = t['route']; c = r[1:-1]
            orig = [n2o.get(x, x) for x in c]
            w = sum(dw.get(x, 0) for x in c); v = sum(dv.get(x, 0) for x in c)
            st = t['start']
            rows.append({'车辆': vid, '车型': veh['vt']['name'], '第几趟': tid,
                         '出发': _fmt_h(st), '返回': _fmt_h(t['end']),
                         '客户(原始ID)': str(orig), '载重kg': round(w, 2),
                         '行驶元': round(t['travel'], 2), '惩罚元': round(t['penalty'], 2)})
    pd.DataFrame(rows).to_csv(f'{BASE}/p3_schedule.csv', index=False, encoding='utf-8-sig')
    log("\n调度表: p3_schedule.csv")

    # 4. 可视化
    visualize_dynamic(ds_snapshot, ds, coords, green_orig, n2o,
                      ds.event_log, f'{BASE}/p3_routes.png')
    visualize_event_impact(ds.event_log, f'{BASE}/p3_event_impact.png')

    log("\n===== 问题3 完成 =====")
