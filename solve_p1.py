# -*- coding: utf-8 -*-
"""华中杯A题 问题1：静态车辆调度（精简版）"""
import sys, io, time, math, random
# 幂等包装：避免被多次 import 时重复包装导致 I/O 关闭
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
random.seed(42); np.random.seed(42)

BASE = r'c:\Users\LENOVO\Desktop\华中杯1.1'
def log(msg): print(msg, flush=True)

# ── 常量 ──────────────────────────────────────────
VEHICLE_TYPES = [
    {'id':0,'name':'燃油3000', 'type':'fuel','max_w':3000,'max_v':13.5,'total':60},
    {'id':1,'name':'燃油1500', 'type':'fuel','max_w':1500,'max_v':10.8,'total':50},
    {'id':2,'name':'燃油1250', 'type':'fuel','max_w':1250,'max_v': 6.5,'total':50},
    {'id':3,'name':'新能源3000','type':'ev',  'max_w':3000,'max_v':15.0,'total':10},
    {'id':4,'name':'新能源1250','type':'ev',  'max_w':1250,'max_v': 8.5,'total':15},
]
STARTUP=400.; WAIT_H=20.; LATE_H=50.; SVC_H=20/60.
FUEL_P=7.61; ELEC_P=1.64; CO2_P=0.65; ETA=2.547; GAMMA=0.501
DEPOT_T=8.; GREEN_R=10.
SPEED_SEGS=[(8,9,9.8),(9,10,55.3),(10,11.5,35.4),(11.5,13,9.8),(13,15,55.3),(15,17,35.4)]
DEF_SPD=35.4

# ── 时速 / 行驶时间 ───────────────────────────────
def spd(t):
    for a,b,v in SPEED_SEGS:
        if a<=t<b: return v
    return DEF_SPD

_TTC={}
def tt(d,t0):
    if d<1e-9: return 0.
    k=(round(d,3),round(t0*4)/4)
    if k in _TTC: return _TTC[k]
    rem,t,el=d,t0,0.
    for _ in range(50):
        v=spd(t); se=next((b for a,b,_ in SPEED_SEGS if a<=t<b),None)
        if se is None: el+=rem/v; break
        dur=se-t; di=v*dur
        if di>=rem: el+=rem/v; break
        rem-=di; el+=dur; t+=dur
    _TTC[k]=el; return el

def energy(d,t0,vtype,lr):
    if d<1e-9: return 0.,0.
    v=spd(t0)
    if vtype=='fuel':
        e=(0.0025*v*v-0.2554*v+31.75)*(1+0.4*lr)*d/100
        return e*FUEL_P+e*ETA*CO2_P, e*ETA
    e=(0.0014*v*v-0.12*v+36.19)*(1+0.35*lr)*d/100
    return e*ELEC_P+e*GAMMA*CO2_P, e*GAMMA

# ── 数据加载 ──────────────────────────────────────
def load_data():
    log("=== 加载数据 ===")
    cd=pd.read_csv(f'{BASE}/客户坐标信息_Sheet1.csv',encoding='gbk')
    cd.columns=['type','id','x','y']
    coords={int(r['id']):(float(r['x']),float(r['y'])) for _,r in cd.iterrows()}

    dm=pd.read_csv(f'{BASE}/距离矩阵_Sheet1.csv',encoding='gbk',index_col=0).values.astype(float)
    log(f"  距离矩阵: {dm.shape}")

    od=pd.read_csv(f'{BASE}/订单信息_Sheet1.csv',encoding='gbk')
    od.columns=['oid','w','v','cid']
    od['w']=pd.to_numeric(od['w'],errors='coerce').fillna(0)
    od['v']=pd.to_numeric(od['v'],errors='coerce').fillna(0)
    od['cid']=od['cid'].astype(int)
    agg=od.groupby('cid').agg(w=('w','sum'),v=('v','sum')).reset_index()
    dw_r={int(r['cid']):float(r['w']) for _,r in agg.iterrows()}
    dv_r={int(r['cid']):float(r['v']) for _,r in agg.iterrows()}

    # 大客户拆分（满装+余量）
    MW,MV=3000.,13.5
    split_map={}; vid=1000; dw={}; dv={}; n2o={0:0}
    for cid in sorted(dw_r):
        w,v=dw_r[cid],dv_r[cid]
        ns=max(math.ceil(w/MW) if w>MW else 1, math.ceil(v/MV) if v>MV else 1)
        if ns==1:
            dw[cid]=w; dv[cid]=v; split_map[cid]=[(cid,w,v)]; n2o[cid]=cid
        else:
            parts=[]
            if v/MV>=w/MW:
                for _ in range(ns-1):
                    vi2=MV; wi2=w*(vi2/v)
                    dw[vid]=wi2; dv[vid]=vi2; n2o[vid]=cid; parts.append((vid,wi2,vi2)); vid+=1
                rv=v-MV*(ns-1); rw=w-sum(p[1] for p in parts)
            else:
                for _ in range(ns-1):
                    wi2=MW; vi2=v*(wi2/w)
                    dw[vid]=wi2; dv[vid]=vi2; n2o[vid]=cid; parts.append((vid,wi2,vi2)); vid+=1
                rw=w-MW*(ns-1); rv=v-sum(p[2] for p in parts)
            dw[vid]=rw; dv[vid]=rv; n2o[vid]=cid; parts.append((vid,rw,rv)); vid+=1
            split_map[cid]=parts
    log(f"  拆分后节点: {sum(len(p) for p in split_map.values())} (理论≥{math.ceil(sum(dw_r.values())/MW)}辆)")

    tw=pd.read_csv(f'{BASE}/时间窗_Sheet1.csv',encoding='gbk'); tw.columns=['cid','s','e']
    def h(s): p=str(s).strip().split(':'); return int(p[0])+int(p[1])/60.
    ts_r={int(r['cid']):h(r['s']) for _,r in tw.iterrows()}
    te_r={int(r['cid']):h(r['e']) for _,r in tw.iterrows()}
    tw_s={}; tw_e={}
    for cid,pts in split_map.items():
        for v2,_,_ in pts: tw_s[v2]=ts_r.get(cid,0); tw_e[v2]=te_r.get(cid,24)

    green={cid for cid,(x,y) in coords.items() if cid>0 and math.sqrt(x*x+y*y)<=GREEN_R}
    for cid in list(green):
        for v2,_,_ in split_map.get(cid,[]): green.add(v2)
    log(f"  绿色区原始客户: {sorted(c for c in green if c<99)}")

    custs=[v2 for pts in split_map.values() for v2,_,_ in pts]
    mid=max(custs+[0])+1; n0=dm.shape[0]
    ext=np.zeros((mid,mid))
    for i in range(mid):
        oi=n2o.get(i,i if i<n0 else 0)
        for j in range(mid):
            ext[i][j]=dm[oi][n2o.get(j,j if j<n0 else 0)]
    return coords,ext,dw,dv,tw_s,tw_e,green,custs,n2o

# ── 路线评估 ──────────────────────────────────────
def opt_start(route,dm,tw_s):
    if len(route)<3: return DEPOT_T
    c1=route[1]; d=dm[0][c1]; e=tw_s.get(c1,0)
    lo,hi=DEPOT_T,20.
    for _ in range(30):
        mid=(lo+hi)/2
        if mid+tt(d,mid)<e: lo=mid
        else: hi=mid
    return max(DEPOT_T,lo)

def eval_r(route,dm,dw,dv,tw_s,tw_e,vt,start=None):
    """返回 (cost,travel,pen,co2,ok,end_t)"""
    custs=route[1:-1]
    if not custs: return 0.,0.,0.,0.,True,start or DEPOT_T
    TW=sum(dw.get(c,0) for c in custs); TV=sum(dv.get(c,0) for c in custs)
    if TW>vt['max_w']+1e-6 or TV>vt['max_v']+1e-6: return 1e9,0,0,0,False,0
    t=start if start is not None else opt_start(route,dm,tw_s)
    cw=TW; tr=pen=co2=0.
    for i in range(len(route)-1):
        u,v=route[i],route[i+1]; d=dm[u][v]; lr=cw/vt['max_w']
        ec,c2=energy(d,t,vt['type'],lr); tr+=ec; co2+=c2; t+=tt(d,t)
        if v!=0:
            e,l=tw_s.get(v,0),tw_e.get(v,24)
            if t<e: pen+=(e-t)*WAIT_H; t=e
            elif t>l: pen+=(t-l)*LATE_H
            t+=SVC_H; cw-=dw.get(v,0)
    return STARTUP+tr+pen, tr, pen, co2, True, t

# ── Clarke-Wright Savings 构造 ────────────────────
def construct_savings(custs,dm,dw,dv,tw_s,tw_e):
    MW,MV=3000.,15.
    routes={c:[0,c,0] for c in custs}
    rw={c:dw.get(c,0) for c in custs}; rv={c:dv.get(c,0) for c in custs}
    ep={c:c for c in custs}
    svgs=sorted(((dm[0][i]+dm[0][j]-dm[i][j],i,j)
                 for i in custs for j in custs if i<j), reverse=True)

    def tw_ok(r):
        t=DEPOT_T; ml=0.
        for k in range(len(r)-1):
            u,v=r[k],r[k+1]; t+=tt(dm[u][v],t)
            if v!=0:
                e,l=tw_s.get(v,0),tw_e.get(v,24)
                if t<e: t=e
                elif t>l: ml=max(ml,t-l)
                t+=SVC_H
        return ml<=4.

    for s,i,j in svgs:
        if s<=0: break
        ri,rj=ep[i],ep[j]
        if ri==rj: continue
        Ri,Rj=routes[ri],routes[rj]
        ip,jp=Ri.index(i),Rj.index(j)
        if ip not in (1,len(Ri)-2) or jp not in (1,len(Rj)-2): continue
        nw=rw[ri]+rw[rj]; nv=rv[ri]+rv[rj]
        if nw>MW or nv>MV: continue
        Ra=Ri[::-1] if ip==1 else Ri[:]
        Rb=Rj[::-1] if jp==len(Rj)-2 else Rj[:]
        nr=Ra[:-1]+Rb[1:]
        if not tw_ok(nr): continue
        routes[ri]=nr; rw[ri]=nw; rv[ri]=nv
        for c in nr[1:-1]: ep[c]=ri
        if rj!=ri: del routes[rj],rw[rj],rv[rj]
    return list(routes.values())

# ── 车型分配 ──────────────────────────────────────
def best_vt(route,dm,dw,dv,tw_s,tw_e,used):
    TW=sum(dw.get(c,0) for c in route[1:-1]); TV=sum(dv.get(c,0) for c in route[1:-1])
    bv,bc=None,1e18
    for vt in VEHICLE_TYPES:
        if vt['max_w']<TW or vt['max_v']<TV: continue
        if used.get(vt['id'],0)>=vt['total']: continue
        c,*_=eval_r(route,dm,dw,dv,tw_s,tw_e,vt)
        if c<bc: bc=c; bv=vt
    return bv

def reassign(routes_vt,dm,dw,dv,tw_s,tw_e):
    used={vt['id']:0 for vt in VEHICLE_TYPES}; out=[]
    for r,_ in routes_vt:
        vt=best_vt(r,dm,dw,dv,tw_s,tw_e,used) or VEHICLE_TYPES[0]
        used[vt['id']]+=1; out.append((r,vt))
    return out

# ── 路线内优化 ────────────────────────────────────
def two_opt(route,dm,dw,dv,tw_s,tw_e,vt,passes=2):
    best=route[:]; bc,*_=eval_r(best,dm,dw,dv,tw_s,tw_e,vt)
    for _ in range(passes):
        imp=False; n=len(best)
        for i in range(1,n-2):
            for j in range(i+1,n-1):
                nr=best[:i]+best[i:j+1][::-1]+best[j+1:]
                c,*_,ok,_2=eval_r(nr,dm,dw,dv,tw_s,tw_e,vt)
                if ok and c<bc-1e-6: best=nr; bc=c; imp=True
        if not imp: break
    return best

def or_opt(route,dm,dw,dv,tw_s,tw_e,vt,sl=1):
    best=route[:]; bc,*_=eval_r(best,dm,dw,dv,tw_s,tw_e,vt)
    imp=True
    while imp:
        imp=False; n=len(best)
        for i in range(1,n-sl):
            seg=best[i:i+sl]; rest=best[:i]+best[i+sl:]
            for j in range(1,len(rest)):
                for s in [seg,seg[::-1]]:
                    nr=rest[:j]+s+rest[j:]
                    c,*_,ok,_2=eval_r(nr,dm,dw,dv,tw_s,tw_e,vt)
                    if ok and c<bc-1e-6: best=nr; bc=c; imp=True; break
                if imp: break
            if imp: break
    return best

def intra_opt_all(rvt,dm,dw,dv,tw_s,tw_e):
    out=[]
    for r,vt in rvt:
        if len(r)>4:
            r=two_opt(r,dm,dw,dv,tw_s,tw_e,vt)
            r=or_opt(r,dm,dw,dv,tw_s,tw_e,vt,1)
            r=or_opt(r,dm,dw,dv,tw_s,tw_e,vt,2)
        out.append((r,vt))
    return out

# ── 路线间局部搜索 ────────────────────────────────
def relocate(rvt,dm,dw,dv,tw_s,tw_e,tlim=4):
    t0=time.time(); rvt=[(r[:],vt) for r,vt in rvt]
    while time.time()-t0<tlim:
        imp=False; order=list(range(len(rvt))); random.shuffle(order)
        for ri in order:
            if time.time()-t0>tlim: break
            r_i,vt_i=rvt[ri]
            if len(r_i)<=3: continue
            ci_b,*_=eval_r(r_i,dm,dw,dv,tw_s,tw_e,vt_i)
            for pos in range(1,len(r_i)-1):
                c=r_i[pos]; dwc,dvc=dw.get(c,0),dv.get(c,0)
                nr_i=r_i[:pos]+r_i[pos+1:]
                ci_a,*_=eval_r(nr_i,dm,dw,dv,tw_s,tw_e,vt_i); sav=ci_b-ci_a
                bg=1e-3; bj=None; bnr=None
                for rj in order:
                    if rj==ri: continue
                    r_j,vt_j=rvt[rj]
                    if sum(dw.get(x,0) for x in r_j[1:-1])+dwc>vt_j['max_w']: continue
                    if sum(dv.get(x,0) for x in r_j[1:-1])+dvc>vt_j['max_v']: continue
                    cj_b,*_=eval_r(r_j,dm,dw,dv,tw_s,tw_e,vt_j)
                    cands=sorted(range(1,len(r_j)),key=lambda k:dm[r_j[k-1]][c])[:3]
                    for ins in cands:
                        nr_j=r_j[:ins]+[c]+r_j[ins:]
                        cj_a,*_,ok,_2=eval_r(nr_j,dm,dw,dv,tw_s,tw_e,vt_j)
                        if not ok: continue
                        g=sav-(cj_a-cj_b)
                        if g>bg: bg=g; bj=rj; bnr=nr_j
                if bj is not None:
                    rvt[ri]=(nr_i,vt_i); rvt[bj]=(bnr,rvt[bj][1]); imp=True; break
        if not imp: break
    return [(r,vt) for r,vt in rvt if len(r)>2]

def swap(rvt,dm,dw,dv,tw_s,tw_e,tlim=3):
    t0=time.time(); rvt=[(r[:],vt) for r,vt in rvt]; tot=0
    while time.time()-t0<tlim:
        imp=False; order=list(range(len(rvt))); random.shuffle(order)
        for ri in order:
            if time.time()-t0>tlim: break
            r_i,vt_i=rvt[ri]
            if len(r_i)<=3: continue
            ci_b,*_=eval_r(r_i,dm,dw,dv,tw_s,tw_e,vt_i)
            for pi in range(1,len(r_i)-1):
                ci=r_i[pi]
                for rj in order:
                    if rj<=ri: continue
                    r_j,vt_j=rvt[rj]
                    if len(r_j)<=3: continue
                    cj_b,*_=eval_r(r_j,dm,dw,dv,tw_s,tw_e,vt_j)
                    for pj in range(1,len(r_j)-1):
                        cj=r_j[pj]
                        nri=r_i[:pi]+[cj]+r_i[pi+1:]; nrj=r_j[:pj]+[ci]+r_j[pj+1:]
                        if sum(dw.get(x,0) for x in nri[1:-1])>vt_i['max_w']: continue
                        if sum(dv.get(x,0) for x in nri[1:-1])>vt_i['max_v']: continue
                        if sum(dw.get(x,0) for x in nrj[1:-1])>vt_j['max_w']: continue
                        if sum(dv.get(x,0) for x in nrj[1:-1])>vt_j['max_v']: continue
                        ci_a,*_=eval_r(nri,dm,dw,dv,tw_s,tw_e,vt_i)
                        cj_a,*_=eval_r(nrj,dm,dw,dv,tw_s,tw_e,vt_j)
                        if ci_a+cj_a<ci_b+cj_b-1e-3:
                            rvt[ri]=(nri,vt_i); rvt[rj]=(nrj,vt_j)
                            tot+=ci_b+cj_b-ci_a-cj_a; imp=True; break
                    if imp: break
                if imp: break
            if imp: break
        if not imp: break
    log(f"  swap: +{tot:.2f}"); return rvt

def seg_rel(rvt,dm,dw,dv,tw_s,tw_e,sl=2,tlim=3):
    t0=time.time(); rvt=[(r[:],vt) for r,vt in rvt]; tot=0
    while time.time()-t0<tlim:
        imp=False; order=list(range(len(rvt))); random.shuffle(order)
        for ri in order:
            if time.time()-t0>tlim: break
            r_i,vt_i=rvt[ri]
            if len(r_i)<sl+3: continue
            ci_b,*_=eval_r(r_i,dm,dw,dv,tw_s,tw_e,vt_i)
            for p in range(1,len(r_i)-sl):
                seg=r_i[p:p+sl]; sw=sum(dw.get(c,0) for c in seg); sv=sum(dv.get(c,0) for c in seg)
                nri=r_i[:p]+r_i[p+sl:]
                if len(nri)<3: continue
                ci_a,*_=eval_r(nri,dm,dw,dv,tw_s,tw_e,vt_i); sav=ci_b-ci_a
                for rj in order:
                    if rj==ri: continue
                    r_j,vt_j=rvt[rj]
                    if sum(dw.get(x,0) for x in r_j[1:-1])+sw>vt_j['max_w']: continue
                    if sum(dv.get(x,0) for x in r_j[1:-1])+sv>vt_j['max_v']: continue
                    cj_b,*_=eval_r(r_j,dm,dw,dv,tw_s,tw_e,vt_j)
                    cands=sorted(range(1,len(r_j)),key=lambda k:dm[r_j[k-1]][seg[0]])[:3]
                    for ins in cands:
                        for s in [seg,seg[::-1]]:
                            nrj=r_j[:ins]+s+r_j[ins:]
                            cj_a,*_,ok,_2=eval_r(nrj,dm,dw,dv,tw_s,tw_e,vt_j)
                            if not ok: continue
                            if sav-(cj_a-cj_b)>1e-3:
                                rvt[ri]=(nri,vt_i); rvt[rj]=(nrj,vt_j)
                                tot+=sav-(cj_a-cj_b); imp=True; break
                        if imp: break
                    if imp: break
                if imp: break
            if imp: break
        if not imp: break
    routes_vt2=[(r,vt) for r,vt in rvt if len(r)>2]
    log(f"  seg_rel({sl}): +{tot:.2f}"); return routes_vt2

def two_opt_star(rvt,dm,dw,dv,tw_s,tw_e,tlim=4):
    t0=time.time(); rvt=[(r[:],vt) for r,vt in rvt]; tot=0
    while time.time()-t0<tlim:
        imp=False; order=list(range(len(rvt))); random.shuffle(order)
        for ri in order:
            if time.time()-t0>tlim: break
            r_i,vt_i=rvt[ri]
            if len(r_i)<=3: continue
            ci_b,*_=eval_r(r_i,dm,dw,dv,tw_s,tw_e,vt_i)
            for i in range(1,len(r_i)-1):
                for rj in order:
                    if rj==ri: continue
                    r_j,vt_j=rvt[rj]
                    if len(r_j)<=3: continue
                    cj_b,*_=eval_r(r_j,dm,dw,dv,tw_s,tw_e,vt_j)
                    for j in range(1,len(r_j)-1):
                        nri=r_i[:i]+r_j[j:]; nrj=r_j[:j]+r_i[i:]
                        if len(nri)<3 or len(nrj)<3: continue
                        if sum(dw.get(x,0) for x in nri[1:-1])>vt_i['max_w']: continue
                        if sum(dw.get(x,0) for x in nrj[1:-1])>vt_j['max_w']: continue
                        ci_a,*_,ok1,_2=eval_r(nri,dm,dw,dv,tw_s,tw_e,vt_i)
                        cj_a,*_,ok2,_2=eval_r(nrj,dm,dw,dv,tw_s,tw_e,vt_j)
                        if ok1 and ok2 and ci_a+cj_a<ci_b+cj_b-1e-3:
                            rvt[ri]=(nri,vt_i); rvt[rj]=(nrj,vt_j)
                            tot+=ci_b+cj_b-ci_a-cj_a; imp=True; break
                    if imp: break
                if imp: break
            if imp: break
        if not imp: break
    log(f"  2opt*: +{tot:.2f}"); return rvt

def merge_routes(rvt,dm,dw,dv,tw_s,tw_e,thr,tlim=6):
    t0=time.time(); rvt=[(r[:],vt) for r,vt in rvt]; saved=0
    while time.time()-t0<tlim:
        imp=False
        idxs=sorted(range(len(rvt)),key=lambda k:sum(dw.get(c,0) for c in rvt[k][0][1:-1]))
        for ai in idxs[:len(idxs)//2+5]:
            if ai>=len(rvt) or time.time()-t0>tlim: break
            Ra,vt_a=rvt[ai]; ca,*_=eval_r(Ra,dm,dw,dv,tw_s,tw_e,vt_a)
            wa=sum(dw.get(c,0) for c in Ra[1:-1]); va=sum(dv.get(c,0) for c in Ra[1:-1])
            c_ref=Ra[1] if len(Ra)>2 else 0
            cands=sorted([k for k in range(len(rvt)) if k!=ai],
                         key=lambda k:min(dm[c_ref][x] for x in rvt[k][0][1:-1]) if len(rvt[k][0])>2 else 1e9)
            best=None
            for bi in cands[:20]:
                Rb,vt_b=rvt[bi]; cb,*_=eval_r(Rb,dm,dw,dv,tw_s,tw_e,vt_b)
                wb=sum(dw.get(c,0) for c in Rb[1:-1]); vb=sum(dv.get(c,0) for c in Rb[1:-1])
                for vt_n in VEHICLE_TYPES:
                    if wa+wb>vt_n['max_w'] or va+vb>vt_n['max_v']: continue
                    for ra in [Ra,Ra[::-1]]:
                        for rb in [Rb,Rb[::-1]]:
                            nr=ra[:-1]+rb[1:]
                            cn,*_,ok,_2=eval_r(nr,dm,dw,dv,tw_s,tw_e,vt_n)
                            if not ok: continue
                            delta=cn-(ca+cb)
                            if delta<thr-1 and (best is None or thr-delta>best[0]):
                                best=(thr-delta,bi,nr,vt_n)
                    # 插入法
                    smaller,larger=(Ra,Rb) if len(Ra)<len(Rb) else (Rb,Ra)
                    sc=smaller[1:-1]
                    if 1<=len(sc)<=4:
                        base=list(larger); fail=False
                        for c in sc:
                            bp=None; bcc=1e18; bt=None
                            for k in range(1,len(base)):
                                tr=base[:k]+[c]+base[k:]
                                cc,*_,ok2,_2=eval_r(tr,dm,dw,dv,tw_s,tw_e,vt_n)
                                if ok2 and cc<bcc: bcc=cc; bp=k; bt=tr
                            if bp is None: fail=True; break
                            base=bt
                        if not fail:
                            cn,*_,ok3,_2=eval_r(base,dm,dw,dv,tw_s,tw_e,vt_n)
                            if ok3:
                                delta=cn-(ca+cb)
                                if delta<thr-1 and (best is None or thr-delta>best[0]):
                                    best=(thr-delta,bi,base,vt_n)
            if best:
                _,bi,nr,vt_n=best; rvt[ai]=(nr,vt_n); rvt.pop(bi); saved+=1; imp=True; break
        if not imp: break
    log(f"  merge(thr={thr:.0f}): -{saved}辆"); return rvt

# ── Multi-Trip: Shift 装箱 + EV 升级 ──────────────
def mk_trip(r,vt,dm,dw,dv,tw_s,tw_e,start=None):
    st=start if start is not None else opt_start(r,dm,tw_s)
    c,tr,pen,co2,ok,et=eval_r(r,dm,dw,dv,tw_s,tw_e,vt,st)
    return {'route':r,'vt':vt,'start':st,'end':et,'cost':c,'travel':tr,'penalty':pen,'carbon':co2}

def shift_assign(rvt,dm,dw,dv,tw_s,tw_e,buf=0.05,max_delay=2.0,reverse=False):
    infos=[]
    for r,vt in rvt:
        st=opt_start(r,dm,tw_s)
        c,tr,pen,co2,_,et=eval_r(r,dm,dw,dv,tw_s,tw_e,vt,st)
        infos.append({'route':r,'vt':vt,'st':st,'et':et,'tr':tr,'pen':pen,'co2':co2})
    infos.sort(key=lambda x:x['st'],reverse=reverse)
    vehs=[]
    for info in infos:
        vt=info['vt']; best_vi=-1; best_ex=1e18; best_t=None
        for vi,veh in enumerate(vehs):
            if veh['vt']['id']!=vt['id']: continue
            tsorted=sorted(veh['trips'],key=lambda t:t['start'])
            prev=0
            for i in range(len(tsorted)+1):
                ns=tsorted[i]['start'] if i<len(tsorted) else 1e18
                for ast in [info['st'],max(info['st'],prev+buf)]:
                    if ast<prev+buf-1e-6 or ast-info['st']>max_delay+1e-6: continue
                    c,tr,pen,co2,ok,et=eval_r(info['route'],dm,dw,dv,tw_s,tw_e,vt,ast)
                    if not ok or et+buf>ns: continue
                    ex=(tr+pen)-(info['tr']+info['pen'])
                    if ex<STARTUP-30 and ex<best_ex:
                        best_ex=ex; best_vi=vi; best_t=(ast,et,c,tr,pen,co2)
                if i<len(tsorted): prev=tsorted[i]['end']
        if best_vi>=0:
            ast,et,c,tr,pen,co2=best_t
            vehs[best_vi]['trips'].append({'route':info['route'],'vt':vt,'start':ast,'end':et,
                                           'cost':c,'travel':tr,'penalty':pen,'carbon':co2})
        else:
            vehs.append({'vt':vt,'trips':[mk_trip(info['route'],vt,dm,dw,dv,tw_s,tw_e)]})
    return vehs

def sched_trip_swap(sched,dm,dw,dv,tw_s,tw_e,buf=0.05,tlim=20):
    """两辆车互换一条 trip，若总成本降低则接受（不改变车辆数，降行驶成本）"""
    import time as _t; t0=_t.time()
    sched=[{'vt':s['vt'],'trips':list(s['trips'])} for s in sched]
    tot_gain=0
    while _t.time()-t0<tlim:
        imp=False
        for vi in range(len(sched)):
            if _t.time()-t0>tlim: break
            for ti,ta in enumerate(sched[vi]['trips']):
                for vj in range(len(sched)):
                    if vj==vi: continue
                    for tj,tb in enumerate(sched[vj]['trips']):
                        # 检查容量互换可行性
                        wa=sum(dw.get(c,0) for c in ta['route'][1:-1])
                        wb=sum(dw.get(c,0) for c in tb['route'][1:-1])
                        va2=sum(dv.get(c,0) for c in ta['route'][1:-1])
                        vb2=sum(dv.get(c,0) for c in tb['route'][1:-1])
                        vt_i=sched[vi]['vt']; vt_j=sched[vj]['vt']
                        # 新的最大载重检查
                        wi_new_trips=[t for k,t in enumerate(sched[vi]['trips']) if k!=ti]+[tb]
                        wj_new_trips=[t for k,t in enumerate(sched[vj]['trips']) if k!=tj]+[ta]
                        if max(sum(dw.get(c,0) for c in t['route'][1:-1]) for t in wi_new_trips)>vt_i['max_w']: continue
                        if max(sum(dv.get(c,0) for c in t['route'][1:-1]) for t in wi_new_trips)>vt_i['max_v']: continue
                        if max(sum(dw.get(c,0) for c in t['route'][1:-1]) for t in wj_new_trips)>vt_j['max_w']: continue
                        if max(sum(dv.get(c,0) for c in t['route'][1:-1]) for t in wj_new_trips)>vt_j['max_v']: continue
                        # 时间冲突检查
                        def fits(trips,buf):
                            ts=sorted(trips,key=lambda x:x['start'])
                            prev=0
                            for t in ts:
                                if t['start']<prev+buf-1e-6: return False
                                prev=t['end']
                            return True
                        if not fits(wi_new_trips,buf) or not fits(wj_new_trips,buf): continue
                        # 重新计算 ta 在 vi 里的新起始时间（插入 vj）
                        # 直接比较 travel+penalty 变化
                        cost_before=(ta['travel']+ta['penalty'])+(tb['travel']+tb['penalty'])
                        # 用 ta 的起始时间留在 vj（近似），tb 留在 vi
                        c_ta_new,_,pen_ta,_,ok1,_=eval_r(ta['route'],dm,dw,dv,tw_s,tw_e,vt_j,ta['start'])
                        c_tb_new,_,pen_tb,_,ok2,_=eval_r(tb['route'],dm,dw,dv,tw_s,tw_e,vt_i,tb['start'])
                        if not ok1 or not ok2: continue
                        cost_after=(c_ta_new-STARTUP)+(c_tb_new-STARTUP)
                        if cost_after<cost_before-1:
                            # 执行交换
                            gain=cost_before-cost_after
                            sched[vi]['trips'][ti]={'route':tb['route'],'vt':vt_i,'start':tb['start'],
                                'end':eval_r(tb['route'],dm,dw,dv,tw_s,tw_e,vt_i,tb['start'])[5],
                                'cost':c_tb_new,'travel':eval_r(tb['route'],dm,dw,dv,tw_s,tw_e,vt_i,tb['start'])[1],
                                'penalty':pen_tb,'carbon':eval_r(tb['route'],dm,dw,dv,tw_s,tw_e,vt_i,tb['start'])[3]}
                            sched[vj]['trips'][tj]={'route':ta['route'],'vt':vt_j,'start':ta['start'],
                                'end':eval_r(ta['route'],dm,dw,dv,tw_s,tw_e,vt_j,ta['start'])[5],
                                'cost':c_ta_new,'travel':eval_r(ta['route'],dm,dw,dv,tw_s,tw_e,vt_j,ta['start'])[1],
                                'penalty':pen_ta,'carbon':eval_r(ta['route'],dm,dw,dv,tw_s,tw_e,vt_j,ta['start'])[3]}
                            tot_gain+=gain; imp=True; break
                    if imp: break
                if imp: break
            if imp: break
        if not imp: break
    log(f"  trip_swap: +{tot_gain:.2f}"); return sched

def try_eliminate_vehicle(sched,dm,dw,dv,tw_s,tw_e,buf=0.05,max_delay=3.0,tlim=30):
    """穷举：把单趟或少趟车的所有trip插入已有车辆，若总成本降低则消除该车"""
    import time as _t; t0=_t.time()
    sched=[{'vt':s['vt'],'trips':list(s['trips'])} for s in sched]
    saved=0
    while _t.time()-t0<tlim:
        imp=False
        # 按趟数升序（优先消除趟数少的车）
        order=sorted(range(len(sched)),key=lambda k:len(sched[k]['trips']))
        for vi in order:
            if _t.time()-t0>tlim or vi>=len(sched): break
            Va=sched[vi]; trips_a=Va['trips']
            cur_cost=STARTUP+sum(t['travel']+t['penalty'] for t in trips_a)
            # 尝试把 trips_a 的每个 trip 分别插入其他车
            best_plan=None; best_gain=-1e18
            for vj in range(len(sched)):
                if vj==vi: continue
                Vb=sched[vj]; vt_b=Vb['vt']
                # 尝试把 trips_a 全部插入 vj
                base_trips=list(Vb['trips'])
                cost_before=sum(t['travel']+t['penalty'] for t in base_trips)
                all_ok=True; new_trips=list(base_trips); added_cost=0
                for ta in sorted(trips_a,key=lambda x:x['start']):
                    mw=max(sum(dw.get(c,0) for c in t['route'][1:-1]) for t in new_trips+[ta])
                    mv=max(sum(dv.get(c,0) for c in t['route'][1:-1]) for t in new_trips+[ta])
                    if mw>vt_b['max_w'] or mv>vt_b['max_v']: all_ok=False; break
                    # 找最佳插入时间槽
                    ts=sorted(new_trips,key=lambda x:x['start'])
                    best_slot=None; best_ex=STARTUP-10
                    prev=0
                    for i in range(len(ts)+1):
                        ns=ts[i]['start'] if i<len(ts) else 1e18
                        for ast in [ta['start'],max(ta['start'],prev+buf)]:
                            if ast<prev+buf-1e-6 or ast-ta['start']>max_delay+1e-6: continue
                            c,tr,pen,co2,ok,et=eval_r(ta['route'],dm,dw,dv,tw_s,tw_e,vt_b,ast)
                            if not ok or et+buf>ns: continue
                            ex=(tr+pen)-(ta['travel']+ta['penalty'])
                            if ex<best_ex:
                                best_ex=ex; best_slot=(ast,et,c,tr,pen,co2)
                        if i<len(ts): prev=ts[i]['end']
                    if best_slot is None: all_ok=False; break
                    ast,et,c,tr,pen,co2=best_slot
                    new_trips.append({'route':ta['route'],'vt':vt_b,'start':ast,'end':et,
                                      'cost':c,'travel':tr,'penalty':pen,'carbon':co2})
                    added_cost+=best_ex
                if not all_ok: continue
                gain=cur_cost-added_cost  # 节省的净成本（包含启动400）
                if gain>best_gain: best_gain=gain; best_plan=(vj,new_trips)
            if best_plan and best_gain>50:
                vj,new_trips=best_plan
                sched[vj]['trips']=new_trips
                sched.pop(vi); saved+=1; imp=True
                log(f"    消除车辆 vi={vi} → 节省 {best_gain:.2f}")
                break
        if not imp: break
    log(f"  eliminate: 消除{saved}辆"); return sched

def sched_cost(sched):
    n=len(sched); s=n*STARTUP
    tr=sum(t['travel'] for v in sched for t in v['trips'])
    pen=sum(t['penalty'] for v in sched for t in v['trips'])
    co2=sum(t['carbon'] for v in sched for t in v['trips'])
    return s+tr+pen, n, tr, pen, co2

def upgrade_ev(sched,dm,dw,dv,tw_s,tw_e):
    eu={vt['name']:{'used':sum(1 for s in sched if s['vt']['name']==vt['name']),
                    'lim':vt['total'],'vt':vt}
        for vt in VEHICLE_TYPES if vt['type']=='ev'}
    cands=[]
    for vi,veh in enumerate(sched):
        if veh['vt']['type']=='ev': continue
        mw=max(sum(dw.get(c,0) for c in t['route'][1:-1]) for t in veh['trips'])
        mv=max(sum(dv.get(c,0) for c in t['route'][1:-1]) for t in veh['trips'])
        ct=sum(t['travel']+t['penalty'] for t in veh['trips'])
        bsv=0; bvt=None; btrips=None
        for en,ei in eu.items():
            ev=ei['vt']
            if mw>ev['max_w'] or mv>ev['max_v']: continue
            ntr=[]; nt=0; ok=True
            for t in veh['trips']:
                c,tr,pen,co2,o,et=eval_r(t['route'],dm,dw,dv,tw_s,tw_e,ev,t['start'])
                if not o: ok=False; break
                ntr.append({'route':t['route'],'vt':ev,'start':t['start'],'end':et,
                             'cost':c,'travel':tr,'penalty':pen,'carbon':co2})
                nt+=tr+pen
            if not ok: continue
            sv=ct-nt
            if sv>bsv: bsv=sv; bvt=ev; btrips=ntr
        if bvt: cands.append((bsv,vi,bvt,btrips))
    cands.sort(reverse=True); saved=0
    for sv,vi,ev,ntrips in cands:
        n=ev['name']
        if eu[n]['used']>=eu[n]['lim']: continue
        sched[vi]['vt']=ev; sched[vi]['trips']=ntrips
        eu[n]['used']+=1; saved+=sv
    log(f"  upgrade_ev: +{saved:.2f}元"); return sched

# ── 主求解 ────────────────────────────────────────
def solve():
    T=time.time()
    coords,dm,dw,dv,tw_s,tw_e,green,custs,n2o=load_data()

    log("\n[1] Savings 构造")
    routes=construct_savings(custs,dm,dw,dv,tw_s,tw_e)
    rvt=reassign([(r,VEHICLE_TYPES[0]) for r in routes],dm,dw,dv,tw_s,tw_e)
    ev=lambda: sum(eval_r(r,dm,dw,dv,tw_s,tw_e,vt)[0] for r,vt in rvt)
    log(f"  {len(rvt)}条路线, 初始成本 {ev():.2f}")

    log("\n[2] 路线内优化")
    rvt=intra_opt_all(rvt,dm,dw,dv,tw_s,tw_e)
    log(f"  {ev():.2f}")

    log("\n[3] 多随机种子迭代")
    orig=[(r[:],vt) for r,vt in rvt]; gbest=ev(); grvt=[(r[:],vt) for r,vt in rvt]
    for seed in [42,7,123,2024,99]:
        random.seed(seed); rvt=[(r[:],vt) for r,vt in orig]
        lcost=ev(); lrvt=[(r[:],vt) for r,vt in rvt]; stall=0
        for rd in range(1,10):
            rvt=reassign(rvt,dm,dw,dv,tw_s,tw_e)
            thr=1500 if rd<=2 else(1000 if rd<=4 else(600 if rd<=7 else 400))
            rvt=merge_routes(rvt,dm,dw,dv,tw_s,tw_e,thr,tlim=4)
            rvt=relocate(rvt,dm,dw,dv,tw_s,tw_e,tlim=4)
            rvt=swap(rvt,dm,dw,dv,tw_s,tw_e,tlim=3)
            rvt=seg_rel(rvt,dm,dw,dv,tw_s,tw_e,sl=2,tlim=3)
            rvt=two_opt_star(rvt,dm,dw,dv,tw_s,tw_e,tlim=4)
            rvt=intra_opt_all(rvt,dm,dw,dv,tw_s,tw_e)
            rvt=reassign(rvt,dm,dw,dv,tw_s,tw_e)
            c=ev()
            if c<lcost-1: lcost=c; lrvt=[(r[:],vt) for r,vt in rvt]; stall=0
            else:
                stall+=1
                if stall>=2: break
        log(f"  seed={seed}: {lcost:.2f} ({len(lrvt)}条)")
        if lcost<gbest: gbest=lcost; grvt=lrvt; log("    ★ 全局最优")

    rvt=reassign(grvt,dm,dw,dv,tw_s,tw_e)
    log(f"\n*** 子路线成本 {gbest:.2f}, {len(rvt)}条, 耗时 {time.time()-T:.1f}s ***")

    log("\n[4] Shift Multi-Trip 择优")
    best_sched=None; best_total=1e18
    for delay in [0.5,1.0,1.5,2.0,2.5,3.0,4.0,6.0,8.0]:
        for rev in [False,True]:
            s=shift_assign(rvt,dm,dw,dv,tw_s,tw_e,max_delay=delay,reverse=rev)
            ft,fv,ftr,fpen,_=sched_cost(s)
            log(f"  delay={delay}{'R' if rev else 'F'}: {fv}辆 {ft:.2f} (行驶{ftr:.2f} 惩罚{fpen:.2f})")
            if ft<best_total: best_total=ft; best_sched=s; log("    ★")

    log("\n[5] EV升级")
    best_sched=upgrade_ev(best_sched,dm,dw,dv,tw_s,tw_e)
    bt,bv,btr,bpen,bco2=sched_cost(best_sched)
    log(f"  EV后: {bv}辆 {bt:.2f}")

    log("\n[6] 穷举消除多余车辆")
    for _ in range(4):
        before=len(best_sched)
        best_sched=try_eliminate_vehicle(best_sched,dm,dw,dv,tw_s,tw_e,max_delay=3.0,tlim=30)
        bt,bv,btr,bpen,bco2=sched_cost(best_sched)
        log(f"  消除后: {bv}辆 {bt:.2f}")
        if len(best_sched)==before: break

    log("\n[7] 车辆间 trip 交换（降行驶成本）")
    best_sched=sched_trip_swap(best_sched,dm,dw,dv,tw_s,tw_e,tlim=15)
    best_sched=upgrade_ev(best_sched,dm,dw,dv,tw_s,tw_e)
    bt,bv,btr,bpen,bco2=sched_cost(best_sched)
    log(f"  最终: {bv}辆 {bt:.2f} (启动{bv*STARTUP:.0f} 行驶{btr:.2f} 惩罚{bpen:.2f} CO2={bco2:.2f}kg)")

    schedule=[[(t['route'],t['vt']) for t in s['trips']] for s in best_sched]
    return rvt,coords,dm,dw,dv,tw_s,tw_e,green,n2o,schedule,best_sched

# ── 输出报告 ──────────────────────────────────────
def print_report(rvt,dm,dw,dv,tw_s,tw_e,n2o,best_flex):
    bt,bv,btr,bpen,bco2=sched_cost(best_flex)
    log("\n"+"="*60)
    log("          问题1 最终调度方案（Multi-Trip）")
    log("="*60)
    vc={} ; tc=[]
    for s in best_flex:
        vc[s['vt']['name']]=vc.get(s['vt']['name'],0)+1; tc.append(len(s['trips']))
    log(f"\n车辆: {bv}辆  子路线/车均: {sum(tc)/bv:.2f}  最多: {max(tc)}趟")
    for n,c in sorted(vc.items()): log(f"  {n}: {c}辆")
    log(f"\n启动: {bv*STARTUP:.2f}  行驶: {btr:.2f}  惩罚: {bpen:.2f}  总计: {bt:.2f}")
    log(f"碳排放: {bco2:.2f} kg CO2")

    rows=[]
    for vid,s in enumerate(best_flex,1):
        vt=s['vt']
        for tid,t in enumerate(s['trips'],1):
            r=t['route']; c=r[1:-1]; orig=[n2o.get(x,x) for x in c]
            w=sum(dw.get(x,0) for x in c); v=sum(dv.get(x,0) for x in c)
            st=t['start']
            rows.append({'车辆':vid,'车型':vt['name'],'第几趟':tid,
                         '出发时刻':f'{int(st)}:{int((st%1)*60):02d}',
                         '返回时刻':f'{int(t["end"])}:{int((t["end"]%1)*60):02d}',
                         '客户序列(原始ID)':str(orig),'载重(kg)':round(w,2),
                         '体积(m3)':round(v,3),'行驶能耗(元)':round(t['travel'],2),
                         '时间窗惩罚(元)':round(t['penalty'],2),'CO2(kg)':round(t['carbon'],2)})
    pd.DataFrame(rows).to_csv(f'{BASE}/p1_schedule.csv',index=False,encoding='utf-8-sig')
    log("调度表已保存: p1_schedule.csv")

    # 子路线明细
    det=[]
    for idx,(r,vt) in enumerate(rvt):
        c2,tr,pen,co2,_,_2=eval_r(r,dm,dw,dv,tw_s,tw_e,vt)
        c=r[1:-1]; orig=[n2o.get(x,x) for x in c]
        det.append({'路线':idx+1,'车型':vt['name'],'客户序列(原始ID)':str(orig),
                    '载重(kg)':round(sum(dw.get(x,0) for x in c),2),
                    '体积(m3)':round(sum(dv.get(x,0) for x in c),3),
                    '成本(元)':round(c2,2),'CO2(kg)':round(co2,2)})
    pd.DataFrame(det).to_csv(f'{BASE}/p1_result.csv',index=False,encoding='utf-8-sig')

def visualize(rvt,coords,green,n2o,save_path):
    fig,ax=plt.subplots(figsize=(13,11))
    th=np.linspace(0,2*np.pi,300)
    ax.fill(GREEN_R*np.cos(th),GREEN_R*np.sin(th),color='lightgreen',alpha=0.25,label=f'绿色区(r={GREEN_R}km)')
    ax.plot(GREEN_R*np.cos(th),GREEN_R*np.sin(th),'g--',lw=1.5)
    colors=plt.cm.tab20(np.linspace(0,1,max(len(rvt),1)))
    def co(n): orig=n2o.get(n,n); return coords.get(orig,(0,0))
    for idx,(r,vt) in enumerate(rvt):
        xs=[co(n)[0] for n in r]; ys=[co(n)[1] for n in r]
        ax.plot(xs,ys,'-' if vt['type']=='fuel' else '--',color=colors[idx%len(colors)],lw=1.,alpha=0.7)
    for c in range(1,99):
        x,y=coords[c]; ax.scatter(x,y,c='green' if c in green else 'steelblue',s=35,zorder=5)
        ax.text(x+.3,y+.3,str(c),fontsize=5,alpha=0.7)
    dx,dy=coords[0]; ax.scatter(dx,dy,c='red',s=250,marker='*',zorder=10,label='配送中心')
    ax.scatter(0,0,c='orange',s=100,marker='^',zorder=10,label='市中心')
    ax.set_xlabel('X(km)'); ax.set_ylabel('Y(km)')
    ax.set_title(f'问题1路线图({len(rvt)}条子路线)'); ax.legend(loc='upper right')
    ax.grid(True,alpha=0.3); ax.set_aspect('equal'); plt.tight_layout()
    plt.savefig(save_path,dpi=140,bbox_inches='tight'); plt.close()
    log(f"路线图: {save_path}")

# ── main ──────────────────────────────────────────
if __name__=='__main__':
    log(f"Python {sys.version.split()[0]}")
    rvt,coords,dm,dw,dv,tw_s,tw_e,green,n2o,schedule,best_flex=solve()
    print_report(rvt,dm,dw,dv,tw_s,tw_e,n2o,best_flex)
    visualize(rvt,coords,green,n2o,f'{BASE}/p1_routes.png')
    bt,_,btr,bpen,bco2=sched_cost(best_flex)
    fig,ax=plt.subplots(figsize=(6,5))
    bv=len(best_flex); st=bv*STARTUP
    ax.pie([st,btr,bpen],labels=['启动','行驶能耗','时间窗惩罚'],
           colors=['#FF6B6B','#4ECDC4','#45B7D1'],autopct='%1.1f%%',
           startangle=120,explode=(.04,.04,.04))
    ax.set_title(f'成本构成(总{bt:.0f}元)')
    plt.savefig(f'{BASE}/p1_cost_pie.png',dpi=140,bbox_inches='tight'); plt.close()
    log("===== 问题1 完成 =====")
