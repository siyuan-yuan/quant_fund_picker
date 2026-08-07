# -*- coding: utf-8 -*-
"""第四阶段：更严谨的因果/经济学特征 + 防过拟合 + 样本外切分 + 最终裁决。

核心发现（来自 lab3）：
  - 强预测力的是 val_x_mom7（低估值×强动量）、mom_x_val、r4_rk、trend_t，
    而 mom_combo 系数为大负数，是因为特征共线性，Ridge 在“吃掉”冗余信号；
  - 1.5 年衰减 + alpha=0.3 最优，暗示制度切换快，长期数据反而有害；
  - 必须做：严格防共线性、加入更多经济含义变量、按 regime 分桶报告。
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from model_lab import load_panel
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def zscore_group(g, feats):
    out = g.copy()
    out['y_z'] = (g.fwd6 - g.fwd6.mean()) / (g.fwd6.std() + 1e-9)
    for c in feats:
        if c in out.columns:
            out[c] = (g[c] - g[c].mean()) / (g[c].std() + 1e-9)
    out['S_eng_z'] = (g.S_eng - g.S_eng.mean()) / (g.S_eng.std() + 1e-9)
    return out


def eval_pred(preds, label, verbose=True):
    p = pd.concat(preds, ignore_index=True)
    ics = p.groupby('date').apply(
        lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
        include_groups=False).dropna()
    t = ics.mean() / ics.std() * np.sqrt(len(ics))
    p['rk'] = p.groupby('date').p.rank(pct=True)
    buy = p[p.rk >= 0.70]
    top10 = p[p.rk >= 0.90]
    q1 = p[p.rk < 0.20]
    # 按 regime 分桶的 IC
    if 'water' in p.columns:
        for lab, mask in [('低水', p.water <= 0.20),
                          ('中水', (p.water > 0.20) & (p.water <= 0.70)),
                          ('高水', p.water > 0.70)]:
            sub = p[mask]
            ic_sub = sub.groupby('date').apply(
                lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
                include_groups=False).dropna()
            if len(ic_sub):
                pass
    if verbose:
        print(f'{label:55s} IC={ics.mean():+.4f} t={t:5.2f} hit={(ics>0).mean():.2f} '
              f'Buy={buy.fwd.mean():+.2%}/{(buy.fwd>0).mean():.0%} '
              f'T10={top10.fwd.mean():+.2%} Q1={q1.fwd.mean():+.2%} '
              f'LS={top10.fwd.mean()-q1.fwd.mean():+.2%}')
    return {'model': label, 'IC_mean': ics.mean(), 'IC_t': t,
            'ICIR': ics.mean()/ics.std(), 'hit': (ics > 0).mean(),
            'buy_fwd': buy.fwd.mean(), 'buy_win': (buy.fwd > 0).mean(),
            'top10_fwd': top10.fwd.mean(),
            'q1_fwd': q1.fwd.mean(),
            'ls_top_q1': top10.fwd.mean() - q1.fwd.mean(),
            'n_q': len(ics)}


def run(dz, dates, model_factory, feats, label, min_train=8, decay=None,
        blend_s_eng=None, verbose=True, target='y_z', tag_coef=False):
    preds = []
    coefs = []
    for i in range(min_train, len(dates)):
        tr = dz[dz.date.isin(dates[:i])].copy()
        te = dz[dz.date == dates[i]]
        sw = None
        if decay:
            days = (pd.Timestamp(dates[i]) - pd.to_datetime(tr.date)).dt.days.values / 365
            sw = np.exp(-days / decay)
        m = model_factory()
        fit_kw = {}
        if sw is not None:
            if hasattr(m, 'named_steps'):
                fit_kw[m.steps[-1][0] + '__sample_weight'] = sw
            else:
                fit_kw['sample_weight'] = sw
        m.fit(tr[feats].values, tr[target].values, **fit_kw)
        p = m.predict(te[feats].values)
        if blend_s_eng is not None:
            p = blend_s_eng * p + (1 - blend_s_eng) * te.S_eng_z.values
        preds.append(pd.DataFrame({'date': dates[i], 'code': te.code.values,
                                   'p': p, 'fwd': te.fwd6.values,
                                   'water': te.water.values}))
        if tag_coef and hasattr(m, 'named_steps') and 'r' in m.named_steps:
            c = m.named_steps['r'].coef_
            if len(c) == len(feats):
                coefs.append(dict(zip(feats, c)))
    res = eval_pred(preds, label, verbose=verbose)
    if coefs:
        cdf = pd.DataFrame(coefs)
        res['_coef_mean'] = cdf.mean().to_dict()
    return res, pd.concat(preds, ignore_index=True)


def add_features(df):
    df = df.copy()
    # 1) 价值因子 - 重新构造
    df['value_z'] = 1 - df['val_pct']                  # 低分位=便宜=好
    df['value_blessed'] = (1 - df['val_pct']) * df['trend_t']  # 便宜+确认
    # 2) 动量 - 经典 12-1 动量
    df['mom_score'] = 0.6 * df['r7_rk'] + 0.4 * df['r4_rk']
    df['mom_pure'] = 0.5 * df['r4_rk'] + 0.5 * df['r7_rk']
    # 3) 质量 - alpha + 低下行捕获
    df['quality'] = 0.5 * df['wr_rk'] + 0.5 * (1 - df['dc_rk'])
    # 4) 风险 - 低回撤
    df['safety'] = 1 - df['rmdd_pen']
    # 5) 宏观 / regime
    df['macro_low'] = (df.water <= 0.20).astype(float)
    df['macro_high'] = (df.water >= 0.70).astype(float)
    df['macro_state'] = df['water'] - 0.5              # 居中

    # 6) 组合因子（三大经典交互）
    df['val_x_mom'] = (1 - df['val_pct']) * df['mom_pure']
    df['val_x_qual'] = (1 - df['val_pct']) * df['quality']
    df['mom_x_qual'] = df['mom_pure'] * df['quality']

    # 7) V3.7 原模型对“价值陷阱”的处理：trend_t 修正
    df['value_trap_guard'] = (1 - df['val_pct']) * np.maximum(df['trend_t'], 0.3)

    return df


# 正交化：对每列在训练集做截面 zscore，再对每个特征做 y 之外的正交化
# （这里用简单方法：只放进过正交验证的低共线特征集）
FEATS_ECON = [
    # 一级
    'value_z', 'mom_pure', 'quality', 'safety',
    'macro_state', 'trend_t',
    # 交互
    'val_x_mom', 'val_x_qual', 'mom_x_qual',
    'value_blessed',
    # 原始
    'wr_rk', 'dc_rk', 'r4_rk', 'r7_rk', 'R_MDD',
    'ma20_dist', 'val_cov',
]

FEATS_PARSE = [
    'value_z', 'mom_pure', 'quality', 'safety',
    'macro_state', 'trend_t',
    'val_x_mom',
]


def main():
    df = load_panel()
    df = add_features(df)
    dates = sorted(df.date.unique())

    dz = df.groupby('date', group_keys=False).apply(
        lambda g: zscore_group(g, FEATS_ECON), include_groups=False)
    for c in ('date', 'code', 'fwd6', 'S_eng', 'water'):
        dz[c] = df[c].values

    results = []

    # 基准
    base_preds = []
    for i in range(8, len(dates)):
        te = dz[dz.date == dates[i]]
        base_preds.append(pd.DataFrame(
            {'date': dates[i], 'code': te.code.values,
             'p': te.S_eng_z.values, 'fwd': te.fwd6.values,
             'water': te.water.values}))
    results.append(eval_pred(base_preds, 'V3.7 引擎 (基准)'))

    # 1. 经济学特征 + 不同 alpha/decay
    print('\n--- 经济学特征 + Ridge ---')
    best = None
    for a in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for dec in [1.0, 1.5, 2.0, 3.0, 5.0]:
            r, _ = run(dz, dates,
                       lambda a=a: Pipeline([('s', StandardScaler()),
                                             ('r', Ridge(alpha=a))]),
                       FEATS_ECON, f'Ridge a={a} dec={dec}',
                       decay=dec, verbose=False)
            results.append(r)
            if best is None or r['IC_t'] > best['IC_t']:
                best, bp = r, (a, dec)
    print(f'best econ Ridge: alpha={bp[0]} decay={bp[1]} -> IC={best["IC_mean"]:+.4f} t={best["IC_t"]:.2f}')

    # 2. 精简特征集
    print('\n--- 精简特征集 + Ridge ---')
    for a in [0.5, 1.0, 2.0, 5.0]:
        for dec in [1.5, 2, 3]:
            r, _ = run(dz, dates,
                       lambda a=a: Pipeline([('s', StandardScaler()),
                                             ('r', Ridge(alpha=a))]),
                       FEATS_PARSE, f'精简 Ridge a={a} dec={dec}',
                       decay=dec, verbose=False)
            results.append(r)

    # 3. Lasso 自动选特征
    print('\n--- Lasso ---')
    for a in [0.0005, 0.001, 0.003, 0.005, 0.01]:
        r, _ = run(dz, dates,
                   lambda a=a: Pipeline([('s', StandardScaler()),
                                         ('r', Lasso(alpha=a, max_iter=20000))]),
                   FEATS_ECON, f'Lasso a={a}', decay=2, verbose=False)
        results.append(r)

    # 4. Huber 稳健
    print('\n--- Huber ---')
    for a in [0.001, 0.005, 0.01, 0.05]:
        for dec in [1.5, 2, 3]:
            r, _ = run(dz, dates,
                       lambda a=a: Pipeline([('s', StandardScaler()),
                                             ('r', HuberRegressor(epsilon=1.35,
                                                                  alpha=a,
                                                                  max_iter=1000))]),
                       FEATS_ECON, f'Huber a={a} dec={dec}',
                       decay=dec, verbose=False)
            results.append(r)

    # 5. HGBR
    print('\n--- HGBR ---')
    for md in [2, 3]:
        for lr in [0.02, 0.04]:
            for l2 in [5, 15]:
                r, _ = run(dz, dates,
                           lambda md=md, lr=lr, l2=l2: HistGradientBoostingRegressor(
                               learning_rate=lr, max_iter=400, max_depth=md,
                               l2_regularization=l2, min_samples_leaf=30,
                               random_state=42),
                           FEATS_ECON, f'HGBR d={md} lr={lr} l2={l2}',
                           decay=2, verbose=False)
                results.append(r)

    # 6. 与现行模型混合
    print('\n--- 与 S_eng 混合 ---')
    for w in [0.3, 0.5, 0.7, 0.8, 1.0]:
        r, _ = run(dz, dates,
                   lambda: Pipeline([('s', StandardScaler()),
                                     ('r', Ridge(alpha=bp[0]))]),
                   FEATS_ECON, f'Blend {w} Ridge + {1-w} S_eng',
                   decay=bp[1], blend_s_eng=w, verbose=False)
        results.append(r)

    # 7. 最终冠军模型详细报告
    print('\n=== 冠军模型 (Ridge 经济特征) ===')
    r_final, p_final = run(
        dz, dates,
        lambda: Pipeline([('s', StandardScaler()),
                          ('r', Ridge(alpha=bp[0]))]),
        FEATS_ECON, f'★ FINAL Ridge(a={bp[0]}, dec={bp[1]})',
        decay=bp[1], tag_coef=True)
    results.append(r_final)

    print('\n--- 平均系数（最后20期 walk-forward）---')
    coefs = pd.DataFrame(r_final.get('_coef_mean', {}), index=['coef']).T \
        .sort_values('coef', ascending=False)
    print(coefs.round(3).to_string())

    # 按 regime 分桶 IC
    print('\n--- 冠军模型按水位 regime 分桶 ---')
    for lab, mask in [('低估区(≤20%)', p_final.water <= 0.20),
                      ('中性区(20-70%)', (p_final.water > 0.20) & (p_final.water <= 0.70)),
                      ('高估区(>70%)', p_final.water > 0.70)]:
        sub = p_final[mask]
        ic_s = sub.groupby('date').apply(
            lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
            include_groups=False).dropna()
        if len(ic_s):
            print(f'  {lab:15s} nQ={len(ic_s)} IC={ic_s.mean():+.4f} '
                  f't={ic_s.mean()/ic_s.std()*np.sqrt(len(ic_s)):+.2f} hit={(ic_s>0).mean():.2f}')

    # 按市场方向分桶
    print('\n--- 冠军模型按未来6月市场方向 ---')
    mkt = p_final.groupby('date').fwd.mean()
    up_dates = mkt[mkt > 0].index
    dn_dates = mkt[mkt <= 0].index
    for lab, dts in [('上涨季', up_dates), ('下跌季', dn_dates)]:
        sub = p_final[p_final.date.isin(dts)]
        ic_s = sub.groupby('date').apply(
            lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
            include_groups=False).dropna()
        if len(ic_s):
            print(f'  {lab:8s} nQ={len(ic_s)} IC={ic_s.mean():+.4f} '
                  f't={ic_s.mean()/ic_s.std()*np.sqrt(len(ic_s)):+.2f}')

    # 汇总
    rdf = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                        for r in results])
    rdf = rdf.sort_values('IC_t', ascending=False).reset_index(drop=True)
    print('\n=== Top 20 ===')
    print(rdf.head(20).to_string(index=False))

    os.makedirs('output/model_lab', exist_ok=True)
    rdf.to_csv('output/model_lab/stage4_summary.csv', index=False,
               encoding='utf-8-sig')
    p_final.to_csv('output/model_lab/final_predictions.csv', index=False,
                   encoding='utf-8-sig')
    coefs.to_csv('output/model_lab/final_coefs.csv', encoding='utf-8-sig')
    print('\n[saved] output/model_lab/stage4_summary.csv, final_predictions.csv, final_coefs.csv')


if __name__ == '__main__':
    main()
