# -*- coding: utf-8 -*-
"""华中杯A题 问题3：动态事件下的实时车辆调度策略"""
import sys, io, os, time, math, random, copy, json
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, _HERE)

from solve_p1 import (
    load_data, eval_r, opt_start, two_opt, or_opt,
    relocate, swap, two_opt_star, intra_opt_all,
    mk_trip, sched_cost, construct_savings, best_vt,
    tt, SVC_H,
    VEHICLE_TYPES, STARTUP, WAIT_H, LATE_H, SVC_H as _SVC,
    GREEN_R, BASE, log
)
# 限行时段（与问题2一致）
RESTRICT_S, RESTRICT_E = 8.0, 16.0
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

# ── 绿色区 / 限行约束工具 ───────────────────────
def _in_green(x, y):
    """坐标是否落入绿色配送区（市中心半径10km圆）"""
    return (x*x + y*y) <= GREEN_R*GREEN_R

def _trip_overlaps_restrict(start, end):
    """该子路线时间区间是否与限行时段[8,16]重叠"""
    return start < RESTRICT_E and end > RESTRICT_S

def _is_ev(vt):
    return vt['type'] == 'ev'

def _pick_ev_vt(w, v, used_count):
    """挑选可用且能装下的最小新能源车型"""
    evs = sorted([vt for vt in VEHICLE_TYPES if vt['type'] == 'ev'
                  and vt['max_w'] >= w and vt['max_v'] >= v],
                 key=lambda vt: vt['max_w'])
    for vt in evs:
        if used_count.get(vt['id'], 0) < vt['total']:
            return vt
    return evs[-1] if evs else None


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
        # 订单级数据：{客户原始ID: {订单号: (重量, 体积)}}
        self.orders_per_cust = self._load_orders()
        # 反查：{订单号: 客户原始ID}
        self.order2cust = {oid: c for c, ods in self.orders_per_cust.items()
                           for oid in ods}
        self._recompute_all_costs()

    @staticmethod
    def _load_orders():
        """读取 订单信息_Sheet1.csv，返回 {客户ID: {订单号: (w,v)}}"""
        try:
            df = pd.read_csv(f'{BASE}/订单信息_Sheet1.csv', encoding='gbk')
        except Exception:
            return {}
        df.columns = ['order_id', 'weight', 'volume', 'cust_id']
        out = {}
        for _, r in df.iterrows():
            cid = int(r['cust_id'])
            out.setdefault(cid, {})[int(r['order_id'])] = (
                float(r['weight']), float(r['volume']))
        return out

    # ── 客户/订单状态判别 ────────────────────────
    def _trip_state_for_cust(self, cust_orig, current_time):
        """
        返回 (state, vi, ti, node)：
          state ∈ {'delivered','in_transit','pending','absent'}
        当客户分散在多个 trip 时，优先级：delivered > in_transit > pending > absent
        delivered: trip 已发车且经过该客户的离开时刻 < current_time
        in_transit: trip 已发车但客户尚未服务完
        pending: trip 未发车
        absent: 不在任何路线
        """
        hits_pending = []; hits_in_transit = []; hits_delivered = []
        for vi, veh in enumerate(self.vehicles):
            for ti, t in enumerate(veh['trips']):
                hs = [nd for nd in t['route'][1:-1]
                      if self.n2o.get(nd, nd) == cust_orig]
                if not hs: continue
                if t['start'] > current_time:
                    hits_pending.append((vi, ti, hs[0])); continue
                leave_t = self._sim_leave_time(t, hs[0])
                if leave_t is not None and leave_t < current_time:
                    hits_delivered.append((vi, ti, hs[0]))
                else:
                    hits_in_transit.append((vi, ti, hs[0]))
        if hits_delivered:
            vi, ti, nd = hits_delivered[0]
            return ('delivered', vi, ti, nd)
        if hits_in_transit:
            vi, ti, nd = hits_in_transit[0]
            return ('in_transit', vi, ti, nd)
        if hits_pending:
            vi, ti, nd = hits_pending[0]
            return ('pending', vi, ti, nd)
        return ('absent', None, None, None)

    def _sim_leave_time(self, trip, target_nd):
        """模拟 trip 中 target_nd 的服务完成（离开）时刻"""
        cur_t = trip['start']
        r = trip['route']
        for i in range(1, len(r)):
            a, b = r[i-1], r[i]
            cur_t += tt(self.dm[a][b], cur_t)
            if b == 0: return cur_t
            ws = self.tw_s.get(b, 0)
            if cur_t < ws: cur_t = ws
            cur_t += SVC_H
            if b == target_nd: return cur_t
        return None

    # ── 重做问题1：对未发车 trip 整体重新优化 ────
    def _redo_p1_on_pending(self, current_time, label=''):
        """
        对所有未发车 trip 应用问题1的"路线优化"算子链：
          intra-opt → relocate → swap → two_opt_star → or-opt
        保留现有 pending 结构（避免重建造成车辆数爆炸），但充分搜索
        路线间客户迁移与交换的可能。已发车 trip 锁定不变。
        """
        idx_map = []
        rvt = []
        for vi, veh in enumerate(self.vehicles):
            for ti, t in enumerate(veh['trips']):
                if t['start'] > current_time:
                    idx_map.append((vi, ti))
                    rvt.append((list(t['route']), veh['vt']))
        if len(rvt) < 1:
            return
        cost_before, *_ = self.total_cost()
        try:
            rvt = intra_opt_all(rvt, self.dm, self.dw, self.dv,
                                self.tw_s, self.tw_e)
            rvt = relocate(rvt, self.dm, self.dw, self.dv,
                           self.tw_s, self.tw_e, tlim=2)
            rvt = swap(rvt, self.dm, self.dw, self.dv,
                       self.tw_s, self.tw_e, tlim=2)
            rvt = two_opt_star(rvt, self.dm, self.dw, self.dv,
                               self.tw_s, self.tw_e, tlim=2)
            rvt = intra_opt_all(rvt, self.dm, self.dw, self.dv,
                                self.tw_s, self.tw_e)
        except Exception as e:
            log(f"    [重做问题1{label}] 算子异常: {e}")
        # 写回
        new_pending = {}
        for (vi, ti), (r, vt) in zip(idx_map, rvt):
            new_pending.setdefault(vi, {})[ti] = r
        for vi, veh in enumerate(self.vehicles):
            if vi not in new_pending: continue
            kept = []
            for ti, t in enumerate(veh['trips']):
                if ti in new_pending[vi]:
                    r = new_pending[vi][ti]
                    if len(r) < 3: continue
                    st = max(opt_start(r, self.dm, self.tw_s),
                             current_time + 0.05)
                    c, tr, pen, co2, ok, et = eval_r(
                        r, self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], st)
                    t.update({'route': r, 'start': st, 'end': et,
                              'cost': c, 'travel': tr,
                              'penalty': pen, 'carbon': co2})
                    kept.append(t)
                else:
                    kept.append(t)
            veh['trips'] = kept
        self.vehicles = [v for v in self.vehicles if v['trips'] or v['done_trips']]
        cost_after, *_ = self.total_cost()
        log(f"    [重做问题1{label}] 总成本 {cost_before:.2f}→{cost_after:.2f} "
            f"({cost_after-cost_before:+.2f})")

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

    def _reoptimize_pending(self, current_time, label=''):
        """
        事件后对所有"未出发"路线做跨路线重优化：
          1) 路线内 2-opt / or-opt（提升单趟）
          2) 跨路线 relocate（把单点移到更省的路线）
        已出发(start<=current_time)的路线视为已锁定，不参与。
        """
        idx_map = []   # (vi, ti) for pending trips
        rvt = []
        for vi, veh in enumerate(self.vehicles):
            for ti, t in enumerate(veh['trips']):
                if t['start'] > current_time:
                    idx_map.append((vi, ti))
                    rvt.append((list(t['route']), veh['vt']))
        if len(rvt) < 2:
            return
        cost_before = sum(eval_r(r, self.dm, self.dw, self.dv,
                                 self.tw_s, self.tw_e, vt)[0] for r, vt in rvt)
        # 跨路线 relocate（限制时间，避免主流程过慢）
        try:
            rvt = relocate(rvt, self.dm, self.dw, self.dv,
                           self.tw_s, self.tw_e, tlim=2)
        except Exception:
            pass
        # 再做一次路线内打磨
        rvt = intra_opt_all(rvt, self.dm, self.dw, self.dv, self.tw_s, self.tw_e)
        # 写回
        new_pending = {}
        for (vi, ti), (r, vt) in zip(idx_map, rvt):
            new_pending.setdefault(vi, {})[ti] = r
        for vi, veh in enumerate(self.vehicles):
            if vi not in new_pending: continue
            for ti, r in new_pending[vi].items():
                t = veh['trips'][ti]
                # 路线为空（被 relocate 转空）→ 删除整趟
                if len(r) < 3:
                    veh['trips'][ti] = None
                    continue
                st = max(opt_start(r, self.dm, self.tw_s), current_time + 0.05)
                c, tr, pen, co2, ok, et = eval_r(
                    r, self.dm, self.dw, self.dv,
                    self.tw_s, self.tw_e, veh['vt'], st)
                t.update({'route': r, 'start': st, 'end': et,
                          'cost': c, 'travel': tr, 'penalty': pen, 'carbon': co2})
            veh['trips'] = [t for t in veh['trips'] if t is not None]
        self.vehicles = [v for v in self.vehicles if v['trips'] or v['done_trips']]
        cost_after = sum(eval_r(t['route'], self.dm, self.dw, self.dv,
                                self.tw_s, self.tw_e, v['vt'], t['start'])[0]
                         for v in self.vehicles for t in v['trips']
                         if t['start'] > current_time)
        if cost_after < cost_before - 1e-3:
            log(f"    [跨路线重优化{label}] 待发路段 {cost_before:.2f}→{cost_after:.2f} "
                f"(节省{cost_before-cost_after:.2f})")

    def _split_route_by_time(self, trip, current_time):
        """
        把已发车 trip 的 route 切分成 (locked_prefix, remaining_suffix)：
        - locked_prefix: 已访问过的节点序列（含 depot 起点 + 已服务客户）
                         车辆当前所处位置 = locked_prefix[-1]
        - remaining_suffix: 尚未访问的节点序列（不含车辆当前位置，
                            含末尾返回 depot 节点）
        判定依据：模拟离开时刻 < current_time → 已访问
        若 trip 还未发车，整条路线都属于 remaining。
        """
        r = trip['route']
        if trip['start'] > current_time:
            return [r[0]], r[1:]
        cur_t = trip['start']
        last_served_idx = 0
        for i in range(1, len(r)):
            a, b = r[i-1], r[i]
            cur_t += tt(self.dm[a][b], cur_t)
            if b == 0:
                if cur_t < current_time:
                    last_served_idx = i
                break
            ws = self.tw_s.get(b, 0)
            if cur_t < ws: cur_t = ws
            cur_t += SVC_H
            leave_t = cur_t
            if leave_t < current_time:
                last_served_idx = i
            else:
                break
        locked = r[:last_served_idx + 1]
        remaining = r[last_served_idx + 1:]
        return locked, remaining

    # ── 事件1：订单取消（订单级） ──────────────────
    def event_cancel_order(self, order_id, current_time, kind='auto'):
        """
        kind='order'    : 视为订单号
        kind='customer' : 视为客户原始ID（取消该客户全部订单）
        kind='auto'     : 优先订单号；不存在则视为客户ID
        """
        if kind == 'customer' or (kind == 'auto'
                                   and order_id not in self.order2cust
                                   and order_id in self.orders_per_cust):
            cust_orig = order_id
            cancel_orders = list(self.orders_per_cust.get(cust_orig, {}).keys())
        elif kind == 'order' or order_id in self.order2cust:
            cust_orig = self.order2cust[order_id]
            cancel_orders = [order_id]
        else:
            log(f"\n  [事件] 订单取消: 订单/客户{order_id} 未找到")
            return self.total_cost()[0]

        log(f"\n  [事件] 订单取消: 订单{cancel_orders}({len(cancel_orders)}单) "
            f"客户{cust_orig}, 当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()
        state, vi, ti, node = self._trip_state_for_cust(cust_orig, current_time)

        # 1) 已送达：记录损失即可
        if state == 'delivered':
            for oid in cancel_orders:
                self.orders_per_cust[cust_orig].pop(oid, None)
                self.order2cust.pop(oid, None)
            log(f"    客户{cust_orig}已送达 → 无法撤销，仅记录损失")
            self.event_log.append({
                'type': '订单取消', 'customer': cust_orig, 'time': current_time,
                'state': 'delivered', 'cost_before': cost_before,
                'cost_after': cost_before, 'note': '已送达不调整'})
            return cost_before

        # 2) 不在任何路线
        if state == 'absent':
            for oid in cancel_orders:
                self.orders_per_cust[cust_orig].pop(oid, None)
                self.order2cust.pop(oid, None)
            log(f"    客户{cust_orig}不在当前调度路线中 → 无需调整")
            return cost_before

        # 3) 更新订单表 + 从被取消订单的承运节点扣减需求
        cancel_w_total = 0.0; cancel_v_total = 0.0
        for oid in cancel_orders:
            wv = self.orders_per_cust.get(cust_orig, {}).get(oid)
            if wv:
                cancel_w_total += wv[0]; cancel_v_total += wv[1]
            self.orders_per_cust[cust_orig].pop(oid, None)
            self.order2cust.pop(oid, None)
        remaining = self.orders_per_cust.get(cust_orig, {})
        # 找该客户在调度中所占的虚拟节点（按当前 dw 排序，从最大者扣减）
        cust_nodes = [nd for nd, orig in self.n2o.items() if orig == cust_orig]
        cust_nodes.sort(key=lambda nd: self.dw.get(nd, 0), reverse=True)
        rem_w = cancel_w_total; rem_v = cancel_v_total
        for nd in cust_nodes:
            if rem_w <= 0 and rem_v <= 0: break
            dw_here = self.dw.get(nd, 0); dv_here = self.dv.get(nd, 0)
            dec_w = min(rem_w, dw_here); dec_v = min(rem_v, dv_here)
            self.dw[nd] = dw_here - dec_w
            self.dv[nd] = dv_here - dec_v
            rem_w -= dec_w; rem_v -= dec_v

        # 4) 待发车：重做问题1
        if state == 'pending':
            log(f"    客户{cust_orig}的 trip 未发车 → 触发重做问题1")
            self._redo_p1_on_pending(current_time, label='-cancel/pending')
            cost_after, *_ = self.total_cost()
            self.event_log.append({
                'type': '订单取消', 'customer': cust_orig, 'time': current_time,
                'state': 'pending→redoP1',
                'cost_before': cost_before, 'cost_after': cost_after})
            log(f"    成本 {cost_before:.2f}→{cost_after:.2f} "
                f"(节省{cost_before-cost_after:.2f})")
            return cost_after

        # 5) 在途：分支处理
        # 5a) 仍有其它订单 → 路径不变，仅更新载重再评估
        if remaining:
            t = self.vehicles[vi]['trips'][ti]
            c, tr, pen, co2, ok, et = eval_r(
                t['route'], self.dm, self.dw, self.dv,
                self.tw_s, self.tw_e, self.vehicles[vi]['vt'], t['start'])
            t.update({'cost': c, 'travel': tr, 'penalty': pen,
                      'carbon': co2, 'end': et})
            log(f"    客户{cust_orig}仍有{len(remaining)}单未取消 → 路径不变，仅减需求")
            cost_after, *_ = self.total_cost()
            self.event_log.append({
                'type': '订单取消', 'customer': cust_orig, 'time': current_time,
                'state': 'in_transit/partial',
                'cost_before': cost_before, 'cost_after': cost_after})
            log(f"    成本 {cost_before:.2f}→{cost_after:.2f}")
            return cost_after

        # 5b) 客户全部订单都被取消 → 仅在"剩余路线"上删节点 + 局部重优化
        veh = self.vehicles[vi]; t = veh['trips'][ti]
        locked, remaining = self._split_route_by_time(t, current_time)
        # 在剩余路线中删除该客户节点
        remaining_new = [nd for nd in remaining
                         if not (self.n2o.get(nd, nd) == cust_orig)]
        # 锁定前缀的最后一个节点是车辆当前位置（depot 或 已服务客户点）
        cur_pos = locked[-1]
        # 子路线起点用 cur_pos 接续，便于 eval_r 评估剩余成本
        sub = [cur_pos] + remaining_new
        if len(sub) < 3:
            # 剩余仅余 depot/单点，整条 trip 视为已基本完成
            full_route = locked + remaining_new
            log(f"    在途 trip 剩余路线为空 → 直接收尾 "
                f"{_fmt_route(t['route'], self.n2o)} → {_fmt_route(full_route, self.n2o)}")
            if len(full_route) >= 2 and full_route[-1] != 0:
                full_route = full_route + [0]
            c, tr, pen, co2, ok, et = eval_r(
                full_route, self.dm, self.dw, self.dv,
                self.tw_s, self.tw_e, veh['vt'], t['start'])
            t.update({'route': full_route, 'cost': c, 'travel': tr,
                      'penalty': pen, 'carbon': co2, 'end': et})
        else:
            # 局部重优化：仅对"剩余路线"做 or-opt + 2-opt（已访问部分锁定）
            sub_opt = or_opt(sub, self.dm, self.dw, self.dv,
                             self.tw_s, self.tw_e, veh['vt'])
            sub_opt = two_opt(sub_opt, self.dm, self.dw, self.dv,
                              self.tw_s, self.tw_e, veh['vt'], passes=2)
            # 确保子路线起点保持为 cur_pos（two_opt/or_opt 不会动头尾，但保险检查）
            if sub_opt[0] != cur_pos:
                sub_opt = [cur_pos] + [nd for nd in sub_opt if nd != cur_pos]
            full_route = locked[:-1] + sub_opt
            c, tr, pen, co2, ok, et = eval_r(
                full_route, self.dm, self.dw, self.dv,
                self.tw_s, self.tw_e, veh['vt'], t['start'])
            t.update({'route': full_route, 'cost': c, 'travel': tr,
                      'penalty': pen, 'carbon': co2, 'end': et})
            log(f"    在途 trip 剩余路线删节点+局部重优化:")
            log(f"      锁定前缀: {_fmt_route(locked, self.n2o)}")
            log(f"      剩余优化: {_fmt_route(sub, self.n2o)} → {_fmt_route(sub_opt, self.n2o)}")
        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '订单取消', 'customer': cust_orig, 'time': current_time,
            'state': 'in_transit/full',
            'cost_before': cost_before, 'cost_after': cost_after})
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} "
            f"(节省{cost_before-cost_after:.2f})")
        return cost_after

    # ── 事件2：新增订单 ──────────────────────────
    def event_add_order(self, new_order_id, cust_orig, w, v,
                        tw_s_new, tw_e_new, x, y, current_time):
        """
        新增一个订单。规则：
        1) 优先插入"原计划已带该客户订单的未发车 trip"（同客户合并，无需改路径）
        2) 否则按最小插入成本法插入可容纳的未发车 trip + 局部 2-opt
        3) 都失败 → 对未发车车辆和订单做"重做问题1"
           （若有已返回 depot 的车辆优先复用 → _redo_p1_on_pending 已实现）
        参数 cust_orig: 该订单关联的目标客户（可为现有客户ID或新客户ID）
        """
        log(f"\n  [事件] 新增订单: 订单{new_order_id} 客户{cust_orig} "
            f"w={w:.0f}kg v={v:.3f}m³ TW=[{_fmt_h(tw_s_new)},{_fmt_h(tw_e_new)}] "
            f"位置({x:.2f},{y:.2f}) 当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()
        is_new_cust = cust_orig not in self.orders_per_cust

        # 登记订单与客户
        self.orders_per_cust.setdefault(cust_orig, {})[new_order_id] = (w, v)
        self.order2cust[new_order_id] = cust_orig

        if is_new_cust:
            # 新客户：注册基础信息 + 距离矩阵
            self.coords[cust_orig] = (x, y)
            self.n2o[cust_orig] = cust_orig
            self.dw[cust_orig] = w; self.dv[cust_orig] = v
            self.tw_s[cust_orig] = tw_s_new; self.tw_e[cust_orig] = tw_e_new
            n_old = self.dm.shape[0]
            if cust_orig >= n_old:
                new_size = cust_orig + 1
                new_dm = np.zeros((new_size, new_size))
                new_dm[:n_old, :n_old] = self.dm
                for nd in range(new_size):
                    if nd == cust_orig: continue
                    ox = self.coords.get(self.n2o.get(nd, nd), (0, 0))
                    d = math.sqrt((ox[0]-x)**2 + (ox[1]-y)**2)
                    new_dm[nd][cust_orig] = d
                    new_dm[cust_orig][nd] = d
                self.dm = new_dm
        else:
            # 老客户：累加聚合需求到所有虚拟节点
            cust_nodes = [nd for nd, orig in self.n2o.items() if orig == cust_orig]
            for nd in cust_nodes or [cust_orig]:
                self.dw[nd] = self.dw.get(nd, 0) + w / max(len(cust_nodes), 1)
                self.dv[nd] = self.dv.get(nd, 0) + v / max(len(cust_nodes), 1)

        # ── 策略1：优先合并到"原计划已带该客户订单的未发车 trip" ──
        if not is_new_cust:
            for vi, veh in enumerate(self.vehicles):
                for ti, t in enumerate(veh['trips']):
                    if t['start'] <= current_time: continue
                    if not any(self.n2o.get(nd, nd) == cust_orig
                               for nd in t['route'][1:-1]): continue
                    # 容量检查（含新增）
                    w_cur = sum(self.dw.get(c, 0) for c in t['route'][1:-1])
                    v_cur = sum(self.dv.get(c, 0) for c in t['route'][1:-1])
                    if w_cur > veh['vt']['max_w'] or v_cur > veh['vt']['max_v']:
                        log(f"    ✗ 同客户车辆{vi+1}容量超出，转策略2")
                        break
                    c, tr, pen, co2, ok, et = eval_r(
                        t['route'], self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    t.update({'cost': c, 'travel': tr, 'penalty': pen,
                              'carbon': co2, 'end': et})
                    log(f"    ✓ 合并到 车辆{vi+1}趟{ti+1}（原已含该客户）路径不变")
                    cost_after, *_ = self.total_cost()
                    self.event_log.append({
                        'type': '新增订单', 'customer': cust_orig,
                        'time': current_time, 'state': 'merged',
                        'cost_before': cost_before, 'cost_after': cost_after})
                    log(f"    成本 {cost_before:.2f}→{cost_after:.2f}")
                    return cost_after

        # ── 策略2：按最小插入成本法插入可容纳的未发车 trip ──
        # 绿色区+限行 → 仅允许 EV
        in_green = _in_green(x, y)
        ev_required = in_green and (tw_s_new < RESTRICT_E and tw_e_new > RESTRICT_S)
        if ev_required:
            log(f"    新订单位于绿色区且时窗与限行重叠 → 仅允许新能源车承运")
        target_node = cust_orig
        best = (1e18, -1, -1, None, None)
        for vi, veh in enumerate(self.vehicles):
            if ev_required and not _is_ev(veh['vt']): continue
            for ti, t in enumerate(veh['trips']):
                if t['start'] <= current_time: continue
                if ev_required and not _is_ev(veh['vt']) \
                   and _trip_overlaps_restrict(t['start'], t['end']): continue
                w_cur = sum(self.dw.get(c, 0) for c in t['route'][1:-1])
                v_cur = sum(self.dv.get(c, 0) for c in t['route'][1:-1])
                if w_cur + w > veh['vt']['max_w']: continue
                if v_cur + v > veh['vt']['max_v']: continue
                for ins in range(1, len(t['route'])):
                    new_r = t['route'][:ins] + [target_node] + t['route'][ins:]
                    c2, tr2, pen2, co2_2, ok, et2 = eval_r(
                        new_r, self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    if not ok: continue
                    inc = (tr2 + pen2) - (t['travel'] + t['penalty'])
                    if inc < best[0]:
                        best = (inc, vi, ti, new_r,
                                (c2, tr2, pen2, co2_2, et2))
        if best[1] >= 0:
            inc, vi, ti, new_r, (c2, tr2, pen2, co2_2, et2) = best
            old_r = self.vehicles[vi]['trips'][ti]['route']
            self.vehicles[vi]['trips'][ti].update({
                'route': new_r, 'end': et2, 'cost': c2,
                'travel': tr2, 'penalty': pen2, 'carbon': co2_2})
            # 局部 2-opt 优化插入后的路径
            r_opt = two_opt(new_r, self.dm, self.dw, self.dv,
                            self.tw_s, self.tw_e, self.vehicles[vi]['vt'], passes=2)
            c2, tr2, pen2, co2_2, ok, et2 = eval_r(
                r_opt, self.dm, self.dw, self.dv, self.tw_s, self.tw_e,
                self.vehicles[vi]['vt'],
                self.vehicles[vi]['trips'][ti]['start'])
            self.vehicles[vi]['trips'][ti].update({
                'route': r_opt, 'end': et2, 'cost': c2,
                'travel': tr2, 'penalty': pen2, 'carbon': co2_2})
            log(f"    ✓ 最小成本插入 车辆{vi+1}趟{ti+1}: "
                f"{_fmt_route(old_r, self.n2o)} → {_fmt_route(r_opt, self.n2o)} "
                f"(增量{inc:+.2f})")
            cost_after, *_ = self.total_cost()
            self.event_log.append({
                'type': '新增订单', 'customer': cust_orig,
                'time': current_time, 'state': 'inserted',
                'cost_before': cost_before, 'cost_after': cost_after})
            log(f"    成本 {cost_before:.2f}→{cost_after:.2f} "
                f"({cost_after-cost_before:+.2f})")
            return cost_after

        # ── 策略3：都失败 → 对未发车 + 该订单一起重做问题1 ──
        log(f"    ✗ 无可插入位置 → 触发重做问题1（含已返depot车辆复用）")
        # 把新客户加入一个"占位"trip，使其纳入 redo
        used_count = {}
        for vh in self.vehicles:
            used_count[vh['vt']['id']] = used_count.get(vh['vt']['id'], 0) + 1
        if ev_required:
            vt_pick = _pick_ev_vt(w, v, used_count) or VEHICLE_TYPES[0]
        else:
            vt_pick = next((vh for vh in VEHICLE_TYPES
                            if vh['max_w'] >= w and vh['max_v'] >= v),
                           VEHICLE_TYPES[0])
        st0 = max(opt_start([0, target_node, 0], self.dm, self.tw_s),
                  current_time + 0.05)
        c0, tr0, pen0, co20, ok0, et0 = eval_r(
            [0, target_node, 0], self.dm, self.dw, self.dv,
            self.tw_s, self.tw_e, vt_pick, st0)
        self.vehicles.append({'vt': vt_pick, 'trips': [{
            'route': [0, target_node, 0], 'vt': vt_pick, 'start': st0, 'end': et0,
            'cost': c0, 'travel': tr0, 'penalty': pen0, 'carbon': co20}],
            'done_trips': []})
        self._redo_p1_on_pending(current_time, label='-add/redoP1')
        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '新增订单', 'customer': cust_orig,
            'time': current_time, 'state': 'redoP1',
            'cost_before': cost_before, 'cost_after': cost_after})
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} "
            f"({cost_after-cost_before:+.2f})")
        return cost_after

    # ── 事件3：配送地址变更 ──────────────────────
    def event_address_change(self, cust_orig, new_x, new_y, current_time):
        """
        某客户配送地址变更：
        - 该客户不在调度中 / 已送达 → 仅更新坐标
        - trip 未发车 → 重做问题1
        - trip 在途 → 对该车剩余路线做优化（2-opt / or-opt）
        """
        log(f"\n  [事件] 地址变更: 客户{cust_orig} → ({new_x:.2f},{new_y:.2f}), "
            f"当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()
        old_x, old_y = self.coords.get(cust_orig, (0, 0))
        was_in_green = _in_green(old_x, old_y)
        now_in_green = _in_green(new_x, new_y)

        # 更新坐标 + 距离矩阵
        self.coords[cust_orig] = (new_x, new_y)
        affected_nodes = [nd for nd, orig in self.n2o.items() if orig == cust_orig]
        n = self.dm.shape[0]
        for nd in affected_nodes:
            if nd >= n: continue
            for other in range(n):
                ox = self.coords.get(self.n2o.get(other, other), (0, 0))
                d = math.sqrt((ox[0]-new_x)**2 + (ox[1]-new_y)**2)
                self.dm[nd][other] = d
                self.dm[other][nd] = d

        state, vi, ti, node = self._trip_state_for_cust(cust_orig, current_time)
        if state in ('absent', 'delivered'):
            log(f"    客户{cust_orig}状态={state} → 仅更新坐标，不调路线")
            cost_after, *_ = self.total_cost()
            self.event_log.append({
                'type': '地址变更', 'customer': cust_orig, 'time': current_time,
                'state': state, 'old_pos': (old_x, old_y), 'new_pos': (new_x, new_y),
                'cost_before': cost_before, 'cost_after': cost_after})
            log(f"    成本 {cost_before:.2f}→{cost_after:.2f}")
            return cost_after

        if state == 'pending':
            log(f"    trip 未发车 → 触发重做问题1")
            self._redo_p1_on_pending(current_time, label='-addr/pending')
        else:  # in_transit → 对该车剩余路线做优化
            veh = self.vehicles[vi]; t = veh['trips'][ti]
            old_r = t['route']
            r_opt = two_opt(old_r, self.dm, self.dw, self.dv,
                            self.tw_s, self.tw_e, veh['vt'], passes=3)
            r_opt = or_opt(r_opt, self.dm, self.dw, self.dv,
                           self.tw_s, self.tw_e, veh['vt'])
            c, tr, pen, co2, ok, et = eval_r(
                r_opt, self.dm, self.dw, self.dv,
                self.tw_s, self.tw_e, veh['vt'], t['start'])
            t.update({'route': r_opt, 'cost': c, 'travel': tr,
                      'penalty': pen, 'carbon': co2, 'end': et})
            log(f"    在途 trip 优化: {_fmt_route(old_r, self.n2o)} → "
                f"{_fmt_route(r_opt, self.n2o)}")

        # 跨绿色区检查（限行约束）
        if (not was_in_green) and now_in_green:
            for veh in self.vehicles:
                if _is_ev(veh['vt']): continue
                for t in veh['trips']:
                    if t['start'] <= current_time: continue
                    if not _trip_overlaps_restrict(t['start'], t['end']): continue
                    if any(self.n2o.get(nd, nd) == cust_orig
                           for nd in t['route'][1:-1]):
                        if self._migrate_to_ev(cust_orig, current_time):
                            log(f"    地址跨入绿色区限行段 → 已迁移到EV路线")
                        break
                else: continue
                break

        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '地址变更', 'customer': cust_orig, 'time': current_time,
            'state': state, 'old_pos': (old_x, old_y), 'new_pos': (new_x, new_y),
            'cost_before': cost_before, 'cost_after': cost_after})
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} "
            f"({cost_after-cost_before:+.2f})")
        return cost_after

    def _migrate_to_ev(self, cust_orig, current_time):
        """
        把某 orig 客户从燃油车未发路线中弹出，重插到 EV 路线（最小插入成本）。
        若无可用 EV 路线则新开 EV 车。返回是否完成迁移。
        """
        # 1) 弹出
        popped_node = None; popped_w = popped_v = 0
        for veh in self.vehicles:
            if _is_ev(veh['vt']): continue
            for t in veh['trips']:
                if t['start'] <= current_time: continue
                if not _trip_overlaps_restrict(t['start'], t['end']): continue
                hit = [nd for nd in t['route'][1:-1] if self.n2o.get(nd, nd) == cust_orig]
                if not hit: continue
                nd = hit[0]
                popped_node = nd
                popped_w = self.dw.get(nd, 0); popped_v = self.dv.get(nd, 0)
                new_r = [x for x in t['route'] if x != nd or x == 0]
                if len(new_r) < 3:
                    t['route'] = []  # 标记删除
                else:
                    c, tr, pen, co2, ok, et = eval_r(
                        new_r, self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    t.update({'route': new_r, 'cost': c, 'travel': tr,
                              'penalty': pen, 'carbon': co2, 'end': et})
                break
            if popped_node is not None: break
        if popped_node is None:
            return False
        # 清理空trip / 空车
        for veh in self.vehicles:
            veh['trips'] = [t for t in veh['trips'] if t.get('route')]
        self.vehicles = [v for v in self.vehicles if v['trips'] or v['done_trips']]
        # 2) 找最佳 EV 插入
        best = (1e18, None, None, None, None)
        for vi, veh in enumerate(self.vehicles):
            if not _is_ev(veh['vt']): continue
            for ti, t in enumerate(veh['trips']):
                if t['start'] <= current_time: continue
                w_cur = sum(self.dw.get(c, 0) for c in t['route'][1:-1])
                v_cur = sum(self.dv.get(c, 0) for c in t['route'][1:-1])
                if w_cur + popped_w > veh['vt']['max_w']: continue
                if v_cur + popped_v > veh['vt']['max_v']: continue
                for ins in range(1, len(t['route'])):
                    new_r = t['route'][:ins] + [popped_node] + t['route'][ins:]
                    c, tr, pen, co2, ok, et = eval_r(
                        new_r, self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    if not ok: continue
                    inc = (tr + pen) - (t['travel'] + t['penalty'])
                    if inc < best[0]:
                        best = (inc, vi, ti, new_r, (c, tr, pen, co2, et))
        if best[1] is not None:
            _, vi, ti, new_r, (c, tr, pen, co2, et) = best
            self.vehicles[vi]['trips'][ti].update({
                'route': new_r, 'cost': c, 'travel': tr,
                'penalty': pen, 'carbon': co2, 'end': et})
            return True
        # 3) 新开 EV
        used_count = {}
        for v in self.vehicles:
            used_count[v['vt']['id']] = used_count.get(v['vt']['id'], 0) + 1
        vt = _pick_ev_vt(popped_w, popped_v, used_count)
        if vt is None: return False
        st = max(opt_start([0, popped_node, 0], self.dm, self.tw_s),
                 current_time + 0.5)
        c, tr, pen, co2, ok, et = eval_r(
            [0, popped_node, 0], self.dm, self.dw, self.dv,
            self.tw_s, self.tw_e, vt, st)
        self.vehicles.append({'vt': vt, 'trips': [{
            'route': [0, popped_node, 0], 'vt': vt, 'start': st, 'end': et,
            'cost': c, 'travel': tr, 'penalty': pen, 'carbon': co2}],
            'done_trips': []})
        return True

    # ── 事件4：时间窗调整 ─────────────────────────
    def event_tw_adjust(self, cust_orig, new_tw_s, new_tw_e, current_time):
        """
        某客户时间窗调整：
        - 不在调度 / 已送达 → 仅更新数据
        - 未发车 → 重做问题1
        - 在途：
            · 时间窗变宽 → 仅重新评估惩罚（通常下降），路线不变
            · 时间窗变窄/前移/后移 → 看是否赶得上：
                · 赶得上 → 重新评估即可
                · 赶不上 → 对该车剩余路线做 2-opt
        """
        log(f"\n  [事件] 时间窗调整: 客户{cust_orig} → "
            f"[{_fmt_h(new_tw_s)},{_fmt_h(new_tw_e)}], "
            f"当前时刻 {_fmt_h(current_time)}")
        cost_before, *_ = self.total_cost()
        old_s = self.tw_s.get(cust_orig, 0)
        old_e = self.tw_e.get(cust_orig, 24)
        wider = (new_tw_s <= old_s and new_tw_e >= old_e)

        affected_nodes = [nd for nd, orig in self.n2o.items() if orig == cust_orig]
        for nd in affected_nodes + [cust_orig]:
            self.tw_s[nd] = new_tw_s; self.tw_e[nd] = new_tw_e

        state, vi, ti, node = self._trip_state_for_cust(cust_orig, current_time)
        if state in ('absent', 'delivered'):
            log(f"    客户{cust_orig}状态={state} → 仅更新数据")
            cost_after, *_ = self.total_cost()
            self.event_log.append({
                'type': '时间窗调整', 'customer': cust_orig, 'time': current_time,
                'state': state, 'new_tw': (new_tw_s, new_tw_e),
                'cost_before': cost_before, 'cost_after': cost_after})
            log(f"    成本 {cost_before:.2f}→{cost_after:.2f}")
            return cost_after

        if state == 'pending':
            log(f"    trip 未发车 → 触发重做问题1")
            self._redo_p1_on_pending(current_time, label='-tw/pending')
        else:  # in_transit
            veh = self.vehicles[vi]; t = veh['trips'][ti]
            if wider:
                # 时窗变宽 → 仅重新评估
                c, tr, pen, co2, ok, et = eval_r(
                    t['route'], self.dm, self.dw, self.dv,
                    self.tw_s, self.tw_e, veh['vt'], t['start'])
                old_pen = t['penalty']
                t.update({'cost': c, 'travel': tr, 'penalty': pen,
                          'carbon': co2, 'end': et})
                log(f"    时窗变宽 → 仅重评估 惩罚 {old_pen:.2f}→{pen:.2f}")
            else:
                # 收窄/前移/后移：先看赶得上吗
                arrive = self._sim_leave_time(t, node)
                if arrive is None: arrive = 1e9
                arrive_only = arrive - SVC_H
                can_meet = (new_tw_s <= arrive_only <= new_tw_e)
                if can_meet:
                    c, tr, pen, co2, ok, et = eval_r(
                        t['route'], self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    old_pen = t['penalty']
                    t.update({'cost': c, 'travel': tr, 'penalty': pen,
                              'carbon': co2, 'end': et})
                    log(f"    时窗收窄但赶得上({_fmt_h(arrive_only)} ∈"
                        f" [{_fmt_h(new_tw_s)},{_fmt_h(new_tw_e)}]) → 仅重评估")
                else:
                    log(f"    时窗收窄且赶不上({_fmt_h(arrive_only)} ∉"
                        f" [{_fmt_h(new_tw_s)},{_fmt_h(new_tw_e)}]) → 该车 2-opt")
                    old_r = t['route']
                    r_new = two_opt(old_r, self.dm, self.dw, self.dv,
                                    self.tw_s, self.tw_e, veh['vt'], passes=3)
                    c, tr, pen, co2, ok, et = eval_r(
                        r_new, self.dm, self.dw, self.dv,
                        self.tw_s, self.tw_e, veh['vt'], t['start'])
                    t.update({'route': r_new, 'cost': c, 'travel': tr,
                              'penalty': pen, 'carbon': co2, 'end': et})
                    log(f"    路线重排: {_fmt_route(old_r, self.n2o)} → "
                        f"{_fmt_route(r_new, self.n2o)}")

        cost_after, *_ = self.total_cost()
        self.event_log.append({
            'type': '时间窗调整', 'customer': cust_orig, 'time': current_time,
            'state': state, 'new_tw': (new_tw_s, new_tw_e),
            'cost_before': cost_before, 'cost_after': cost_after})
        log(f"    成本 {cost_before:.2f}→{cost_after:.2f} "
            f"({cost_after-cost_before:+.2f})")
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

    # 1. 加载问题2方案（含绿色配送区限行约束）作为初始
    log("\n>>> 加载问题2最优方案（含绿色区8-16限行）作为动态调度起点 <<<")
    from solve_p2 import solve_p2
    rvt, coords, dm, dw, dv, tw_s, tw_e, green_orig, n2o, flex_sched = solve_p2()
    bt0, bv0, btr0, bpen0, bco2_0 = _sched_cost_flex(flex_sched)
    # green 集合（含虚拟节点）以便兼容
    green = {nd for nd in dw if n2o.get(nd, nd) in green_orig}
    log(f"初始方案(P2): {bv0}辆 总成本:{bt0:.2f}元 CO2:{bco2_0:.2f}kg")

    # 2. 初始化动态调度系统
    ds = DynamicSchedule(flex_sched, dm, dw, dv, tw_s, tw_e, n2o, coords)
    ds_snapshot = copy.deepcopy(ds)  # 保存初始状态用于对比可视化

    # ── 从配置文件读取异常事件并按序触发 ─────────
    cfg_path = os.environ.get('P3_EVENTS', f'{BASE}/p3/p3_events.json')
    if not os.path.exists(cfg_path):
        log(f"\n⚠ 未找到事件配置 {cfg_path}，使用空事件列表")
        events_cfg = []
    else:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        events_cfg = cfg.get('events', [])
        log(f"\n事件配置: {cfg_path} ({len(events_cfg)}个事件)")

    log("\n" + "="*60)
    log("按配置文件依次触发异常事件")
    log("="*60)

    for idx, ev in enumerate(events_cfg, 1):
        etype = ev.get('type'); etime = float(ev.get('time', 0))
        note = ev.get('note', '')
        log(f"\n── 事件{idx}/{len(events_cfg)} type={etype} @{_fmt_h(etime)} {note}")

        if etype == 'cancel_order':
            if 'order_id' in ev:
                ds.event_cancel_order(order_id=int(ev['order_id']),
                                      current_time=etime, kind='order')
            elif 'customer_id' in ev:
                ds.event_cancel_order(order_id=int(ev['customer_id']),
                                      current_time=etime, kind='customer')
            else:
                log(f"    ⚠ cancel_order 缺少 order_id 或 customer_id，跳过")

        elif etype == 'add_order':
            cust = int(ev['customer_id'])
            x = float(ev.get('x', coords.get(cust, (0, 0))[0]))
            y = float(ev.get('y', coords.get(cust, (0, 0))[1]))
            ds.event_add_order(
                new_order_id=int(ev['order_id']),
                cust_orig=cust,
                w=float(ev['weight']), v=float(ev['volume']),
                tw_s_new=float(ev['tw_start']),
                tw_e_new=float(ev['tw_end']),
                x=x, y=y, current_time=etime
            )

        elif etype == 'tw_adjust':
            ds.event_tw_adjust(
                cust_orig=int(ev['customer_id']),
                new_tw_s=float(ev['tw_start']),
                new_tw_e=float(ev['tw_end']),
                current_time=etime
            )

        elif etype == 'address_change':
            ds.event_address_change(
                cust_orig=int(ev['customer_id']),
                new_x=float(ev['x']), new_y=float(ev['y']),
                current_time=etime
            )

        else:
            log(f"    ⚠ 未知事件类型 {etype}，跳过")

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
    pd.DataFrame(rows).to_csv(f'{BASE}/p3/p3_schedule.csv', index=False, encoding='utf-8-sig')
    log("\n调度表: p3/p3_schedule.csv")

    # 4. 可视化
    visualize_dynamic(ds_snapshot, ds, coords, green_orig, n2o,
                      ds.event_log, f'{BASE}/p3/p3_routes.png')
    visualize_event_impact(ds.event_log, f'{BASE}/p3/p3_event_impact.png')

    log("\n===== 问题3 完成 =====")
