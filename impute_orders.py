# -*- coding: utf-8 -*-
"""
订单缺失重量/体积插补
规则：
  若该客户有完整订单 且 完整订单数 >= MIN_COUNT
              且 完整订单总重量 >= MIN_TOTAL_W
              且 完整订单总体积 >= MIN_TOTAL_V
  → 用同客户重量/体积比率（密度）插补
  else
  → 用全局平均密度插补
输出：订单信息_Sheet1_imputed.csv
"""
import os
import pandas as pd
import numpy as np

# ── 阈值参数（可调整）──────────────────────────────
MIN_COUNT   = 5      # 完整订单数量阈值
MIN_TOTAL_W = 50.0   # 完整订单总重量阈值（kg）
MIN_TOTAL_V = 0.1    # 完整订单总体积阈值（m³）
# ──────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH  = os.path.join(HERE, '订单信息_Sheet1.csv')
OUT_PATH = os.path.join(HERE, '订单信息_Sheet1_imputed.csv')


def load(path):
    for enc in ['gbk', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f'读取成功（编码={enc}），共 {len(df)} 行')
            return df, enc
        except Exception:
            continue
    raise RuntimeError(f'无法读取文件: {path}')


def compute_global_density(df):
    """全局平均密度 = 所有完整订单的总重量 / 总体积"""
    complete = df.dropna(subset=[df.columns[1], df.columns[2]])
    total_w  = complete.iloc[:, 1].sum()
    total_v  = complete.iloc[:, 2].sum()
    density  = total_w / total_v   # kg/m³
    print(f'\n全局平均密度: {density:.4f} kg/m3'
          f'  完整订单 {len(complete)} 条，总重 {total_w:.2f} kg，总体积 {total_v:.4f} m3')
    return density


def compute_customer_density(df, cid, col_w, col_v):
    """
    计算某客户的重量/体积密度，以及是否满足阈值。
    返回 (density, count, total_w, total_v, meets_threshold)
    """
    cust_df  = df[df.iloc[:, 3] == cid]
    complete = cust_df.dropna(subset=[col_w, col_v])
    count    = len(complete)
    total_w  = complete[col_w].sum()
    total_v  = complete[col_v].sum()

    meets = (count >= MIN_COUNT and
             total_w >= MIN_TOTAL_W and
             total_v >= MIN_TOTAL_V)

    density = total_w / total_v if total_v > 0 else None
    return density, count, total_w, total_v, meets


def impute(df, global_density):
    col_oid = df.columns[0]
    col_w   = df.columns[1]
    col_v   = df.columns[2]
    col_cid = df.columns[3]

    result = df.copy()
    log_rows = []

    # 缺重量的行
    miss_w = df[df[col_w].isna()].index.tolist()
    # 缺体积的行
    miss_v = df[df[col_v].isna()].index.tolist()
    # 同时缺两者的行（先处理，避免插补时互相干扰）
    miss_both = df[df[col_w].isna() & df[col_v].isna()].index.tolist()

    all_missing = sorted(set(miss_w + miss_v))

    for idx in all_missing:
        row   = df.loc[idx]
        cid   = int(row[col_cid])
        oid   = row[col_oid]
        w_nan = pd.isna(row[col_w])
        v_nan = pd.isna(row[col_v])

        density, count, t_w, t_v, meets = compute_customer_density(
            df, cid, col_w, col_v)

        if meets and density is not None:
            method = f'同客户密度（{count}条完整订单，密度={density:.4f}）'
        else:
            density = global_density
            method  = f'全局密度（{density:.4f}）'

        if w_nan and not v_nan:
            v_known = row[col_v]
            imputed_w = density * v_known
            result.at[idx, col_w] = round(imputed_w, 4)
            log_rows.append({
                '订单编号': oid, '客户': cid,
                '缺失字段': '重量', '已知值': f'体积={v_known:.5f}',
                '插补值': round(imputed_w, 4), '方法': method
            })

        elif v_nan and not w_nan:
            w_known = row[col_w]
            imputed_v = w_known / density
            result.at[idx, col_v] = round(imputed_v, 5)
            log_rows.append({
                '订单编号': oid, '客户': cid,
                '缺失字段': '体积', '已知值': f'重量={w_known:.4f}',
                '插补值': round(imputed_v, 5), '方法': method
            })

        else:
            # 重量和体积都缺失：无法用比率，直接用客户或全局均值
            cust_df  = df[df[col_cid] == cid]
            complete = cust_df.dropna(subset=[col_w, col_v])
            if len(complete) >= MIN_COUNT:
                imp_w = complete[col_w].mean()
                imp_v = complete[col_v].mean()
                m     = f'同客户均值（{len(complete)}条完整订单）'
            else:
                imp_w = df.dropna(subset=[col_w])[col_w].mean()
                imp_v = df.dropna(subset=[col_v])[col_v].mean()
                m     = '全局均值'
            result.at[idx, col_w] = round(imp_w, 4)
            result.at[idx, col_v] = round(imp_v, 5)
            log_rows.append({
                '订单编号': oid, '客户': cid,
                '缺失字段': '重量+体积', '已知值': '—',
                '插补值': f'w={round(imp_w,4)}, v={round(imp_v,5)}',
                '方法': m
            })

    return result, pd.DataFrame(log_rows)


def main():
    df, enc = load(IN_PATH)

    col_w = df.columns[1]
    col_v = df.columns[2]

    print(f'\n缺失统计：重量缺失 {df[col_w].isna().sum()} 条，'
          f'体积缺失 {df[col_v].isna().sum()} 条')

    global_density = compute_global_density(df)

    print(f'\n阈值设定：完整订单数 >= {MIN_COUNT}，'
          f'总重量 >= {MIN_TOTAL_W} kg，总体积 >= {MIN_TOTAL_V} m3')

    result, log = impute(df, global_density)

    # 验证
    remaining_w = result[col_w].isna().sum()
    remaining_v = result[col_v].isna().sum()
    print(f'\n插补后缺失：重量 {remaining_w} 条，体积 {remaining_v} 条')

    # 保存结果
    result.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'已保存：{OUT_PATH}')

    # 打印插补日志
    print('\n── 插补明细 ──')
    print(log.to_string(index=False))

    # 保存插补日志
    log_path = os.path.join(HERE, '订单插补日志.csv')
    log.to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f'\n插补日志：{log_path}')


if __name__ == '__main__':
    main()
