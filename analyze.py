import sys, os
sys.path.insert(0, r'C:/Users/LENOVO/Desktop/华中杯1.1')
os.chdir(r'C:/Users/LENOVO/Desktop/华中杯1.1')
import pandas as pd
from solve_p1 import load_data

coords,dm,dw,dv,tw_s,tw_e,green,custs,n2o=load_data()
green_orig={c for c in green if c<99}
ev_forced_orig={c for c in green_orig if tw_s.get(c,0)<16 and tw_e.get(c,24)>8}
print(f'强制EV客户: {sorted(ev_forced_orig)}')

sched=pd.read_csv('p2_schedule.csv',encoding='utf-8-sig')
fuel_green=sched[(sched['车型'].str.contains('燃油'))&(sched['经绿色区']=='是')]
print('\n燃油+绿色区 trip:')
for _,r in fuel_green.iterrows():
    dep=float(r['出发'].split(':')[0])+float(r['出发'].split(':')[1])/60
    ret=float(r['返回'].split(':')[0])+float(r['返回'].split(':')[1])/60
    overlap = dep<16 and ret>8
    col=r['客户(原始ID)']
    print(f'  车{r.车辆} 趟{r.第几趟} {r.出发}-{r.返回} 载重{r["载重kg"]} 违规={overlap}  客户:{col[:60]}')
