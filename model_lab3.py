# -*- coding: utf-8 -*-
"""第三阶段：细致网格 + 特征工程 + 纯样本外评估 + 经济解释。"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from model_lab import load_panel
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATS_BASE = ['val_pct', 'trend_t', 'wr_rk', 'dc_rk', 'r4_rk', 'r7_rk',
              'rmdd_pen', 'R_MDD', 'water', 'val_x_trend', 'mom_x_val',
              'val_cov', 'ma20_dist']


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
    if verbose:
        print(f'{label:50s} IC={ics.mean():+.4f} t={t:5.2f} '
              f'Buy={buy.fwd.mean():+.2%}/{(buy.fwd>0).mean():.0%} '
              f'Top10={top10.fwd.mean():+.2%} Q1={q1.fwd.mean():+.2%} '
              f'LS={top10.fwd.mean()-q1.fwd.mean():+.2%}')
    return {'model': label, 'IC_mean': ics.mean(), 'IC_t': t,
            'ICIR': ics.mean()/ics.std(),
            'hit': (ics > 0).mean(),
            'buy_fwd': buy.fwd.mean(), 'buy_win': (buy.fwd > 0).mean(),
            'top10_fwd': top10.fwd.mean(),
            'q1_fwd': q1.fwd.mean(),
            'ls_top_q1': top10.fwd.mean() - q1.fwd.mean(),
            'n_q': len(ics)}


def run(dz, dates, make_model, feats, label, min_train=8, decay=None,
        blend_s_eng=None, verbose=True, target='y_z'):
    preds = []
    coefs = []
    for i in range(min_train, len(dates)):
        tr = dz[dz.date.isin(dates[:i])].copy()
        te = dz[dz.date == dates[i]]
        sw = None
        if decay:
            days = (pd.Timestamp(dates[i]) - pd.to_datetime(tr.date)).dt.days.values / 365
            sw = np.exp(-days / decay)
        m = make_model()
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
                                   'p': p, 'fwd': te.fwd6.values}))
        if hasattr(m, 'named_steps') and 'r' in m.named_steps:
            c = m.named_steps['r'].coef_
            if len(c) == len(feats):
                coefs.append(dict(zip(feats, c)))
    res = eval_pred(preds, label, verbose=verbose)
    if coefs:
        cdf = pd.DataFrame(coefs)
        res['_coef_mean'] = cdf.mean().to_dict()
    return res, pd.concat(preds, ignore_index=True)


def main():
    df = load_panel()
    dates = sorted(df.date.unique())

    # 加入更多特征工程
    df = df.copy()
    df['mom_combo'] = 0.5 * df['r4_rk'] + 0.5 * df['r7_rk']
    df['mom_accel'] = df['r4_rk'] - df['r7_rk']           # 短期动量加速
    df['alpha_combo'] = 0.5 * df['wr_rk'] + 0.5 * df['dc_rk']
    df['val_value_proxy'] = 1 - df['val_pct']             # 估值反向 (低=好)
    df['dd_score'] = 1 - df['rmdd_pen']                   # 低回撤好
    df['val_x_mom7'] = (1 - df['val_pct']) * df['r7_rk']
    df['wr_x_dc'] = df['wr_rk'] * df['dc_rk']
    df['trend_x_ma'] = df['trend_t'] * np.sign(df['ma20_dist'])

    FEATS = FEATS_BASE + ['mom_combo', 'mom_accel', 'alpha_combo',
                          'val_value_proxy', 'dd_score', 'val_x_mom7',
                          'wr_x_dc']
    dz = df.groupby('date', group_keys=False).apply(
        lambda g: zscore_group(g, FEATS), include_groups=False)
    for c in ('date', 'code', 'fwd6', 'S_eng'):
        dz[c] = df[c].values

    results = []

    # 基准
    base_preds = []
    for i in range(8, len(dates)):
        te = dz[dz.date == dates[i]]
        base_preds.append(pd.DataFrame(
            {'date': dates[i], 'code': te.code.values,
             'p': te.S_eng_z.values, 'fwd': te.fwd6.values}))
    results.append(eval_pred(base_preds, 'V3.7 引擎(基准)'))

    # 网格扫描 alpha + decay
    best = None
    for a in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for dec in [1.5, 2, 3, 4, 5, None]:
            tag = f'Ridge a={a} decay={dec}'
            r, _ = run(dz, dates,
                       lambda a=a: Pipeline([('s', StandardScaler()),
                                             ('r', Ridge(alpha=a))]),
                       FEATS, tag, decay=dec, verbose=False)
            results.append(r)
            if best is None or r['IC_t'] > best['IC_t']:
                best = r
                best_params = (a, dec)
    print(f'\n[best simple Ridge] alpha={best_params[0]} decay={best_params[1]} '
          f'IC={best["IC_mean"]:+.4f} t={best["IC_t"]:.2f}')

    # 选 top 特征(用全样本)
    # 用最佳模型看系数
    r_best, preds_best = run(
        dz, dates,
        lambda: Pipeline([('s', StandardScaler()),
                          ('r', Ridge(alpha=best_params[0]))]),
        FEATS, f'★ BEST Ridge(a={best_params[0]}, decay={best_params[1]})',
        decay=best_params[1])
    results.append(r_best)

    print('\n=== 平均系数 (最后20期walk-forward) ===')
    coefs = pd.DataFrame(r_best.get('_coef_mean', {}), index=['coef']).T.sort_values('coef', ascending=False)
    print(coefs.round(3).to_string())

    # 简化模型: 只保留高贡献特征
    top_feats = coefs[coefs.coef.abs() > 0.02].index.tolist()
    print(f'\n[parsimonious feats] {top_feats}')

    # 仅用主要因子的简单 Ridge
    for a in [0.3, 0.5, 1.0, 2.0]:
        for dec in [2, 3, 4]:
            r, _ = run(dz, dates,
                       lambda a=a: Pipeline([('s', StandardScaler()),
                                             ('r', Ridge(alpha=a))]),
                       top_feats, f'精简 Ridge a={a} decay={dec}',
                       decay=dec, verbose=False)
            results.append(r)

    # Huber 稳健模型
    for a in [0.001, 0.01, 0.05]:
        for dec in [2, 3, None]:
            r, _ = run(dz, dates,
                       lambda a=a: Pipeline([('s', StandardScaler()),
                                             ('r', HuberRegressor(epsilon=1.35,
                                                                  alpha=a,
                                                                  max_iter=1000))]),
                       FEATS, f'Huber a={a} decay={dec}', decay=dec, verbose=False)
            results.append(r)

    # 与 S_eng 混合
    for w in [0.3, 0.5, 0.7, 0.8]:
        r, _ = run(dz, dates,
                   lambda: Pipeline([('s', StandardScaler()),
                                     ('r', Ridge(alpha=0.5))]),
                   FEATS, f'Blend Ridge*({w}) + S_eng*({1-w})',
                   decay=3, blend_s_eng=w, verbose=False)
        results.append(r)

    # HGBR 调参
    for md in [2, 3]:
        for lr in [0.02, 0.05]:
            for l2 in [3, 10]:
                r, _ = run(dz, dates,
                           lambda md=md, lr=lr, l2=l2: HistGradientBoostingRegressor(
                               learning_rate=lr, max_iter=400, max_depth=md,
                               l2_regularization=l2, min_samples_leaf=30,
                               random_state=42),
                           FEATS, f'HGBR d={md} lr={lr} l2={l2}',
                           decay=3, verbose=False)
                results.append(r)

    # 排序汇总
    rdf = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                        for r in results])
    rdf = rdf.sort_values('IC_t', ascending=False).reset_index(drop=True)
    print('\n=== Top 15 模型 ===')
    print(rdf.head(15).to_string(index=False))
    rdf.to_csv('output/model_lab/stage3_summary.csv', index=False,
               encoding='utf-8-sig')
    preds_best.to_csv('output/model_lab/best_predictions.csv', index=False,
                      encoding='utf-8-sig')
    coefs.to_csv('output/model_lab/best_coefs.csv', encoding='utf-8-sig')
    print('\n[saved] output/model_lab/stage3_summary.csv, best_predictions.csv, best_coefs.csv')


if __name__ == '__main__':
    main()
