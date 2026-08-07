# -*- coding: utf-8 -*-
"""第五阶段：冠军混合模型 + 严格样本外（前半训练，后半测试）+ 最终产出。"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from model_lab import load_panel
from model_lab4 import add_features, zscore_group, FEATS_ECON, FEATS_PARSE
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    df = add_features(load_panel())
    dates = sorted(df.date.unique())
    dz = df.groupby('date', group_keys=False).apply(
        lambda g: zscore_group(g, FEATS_ECON), include_groups=False)
    for c in ('date', 'code', 'fwd6', 'S_eng', 'water'):
        dz[c] = df[c].values

    # ---- 严格 hold-out：前 14 季训，后 14 季测（纯样本外，非 walk-forward） ----
    split = dates[14]
    print(f"[strict hold-out] train<={dates[13].date()} test>={dates[14].date()}")
    tr = dz[dz.date < split]
    te = dz[dz.date >= split]
    print(f"  train n={len(tr)}  test n={len(te)}")

    def eval_oos(p, y, dates_oos, label, water=None):
        d = pd.DataFrame({'p': p, 'y': y, 'date': dates_oos})
        if water is not None:
            d['water'] = water.values
        ics = d.groupby('date').apply(
            lambda g: spearmanr(g.p, g.y)[0] if g.p.nunique() > 5 else np.nan,
            include_groups=False).dropna()
        t = ics.mean() / ics.std() * np.sqrt(len(ics))
        # Buy/Top10 fwd
        d2 = d.copy()
        d2['fwd'] = te.fwd6.values
        d2['rk'] = d2.groupby('date').p.rank(pct=True)
        buy = d2[d2.rk >= 0.70]
        top10 = d2[d2.rk >= 0.90]
        q1 = d2[d2.rk < 0.20]
        print(f'{label:55s} IC={ics.mean():+.4f} t={t:5.2f} hit={(ics>0).mean():.2f} '
              f'Buy={buy.fwd.mean():+.2%}/{(buy.fwd>0).mean():.0%} '
              f'T10={top10.fwd.mean():+.2%} Q1={q1.fwd.mean():+.2%} LS={top10.fwd.mean()-q1.fwd.mean():+.2%}')
        return ics

    # 基准 V3.7
    eval_oos(te.S_eng_z.values, te.fwd6.values, te.date.values, 'V3.7 引擎 (strict OOS)')

    # Ridge 全特征
    for a, dec in [(0.3, 1.0), (1.0, 1.0), (3.0, 1.0), (5.0, 1.0),
                   (0.5, 1.5), (1.0, 1.5), (3.0, 1.5)]:
        days = (pd.Timestamp(split) - pd.to_datetime(tr.date)).dt.days.values / 365
        sw = np.exp(-days / dec)
        m = Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=a))])
        m.fit(tr[FEATS_ECON].values, tr.y_z.values, r__sample_weight=sw)
        p = m.predict(te[FEATS_ECON].values)
        eval_oos(p, te.fwd6.values, te.date.values,
                 f'Ridge a={a} dec={dec} (strict OOS)')

    # Huber
    for dec in [1.0, 1.5]:
        days = (pd.Timestamp(split) - pd.to_datetime(tr.date)).dt.days.values / 365
        sw = np.exp(-days / dec)
        m = Pipeline([('s', StandardScaler()),
                      ('r', HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000))])
        m.fit(tr[FEATS_ECON].values, tr.y_z.values, r__sample_weight=sw)
        p = m.predict(te[FEATS_ECON].values)
        eval_oos(p, te.fwd6.values, te.date.values, f'Huber dec={dec} (strict OOS)')

    # HGBR
    for md in [2, 3]:
        for l2 in [5, 15]:
            days = (pd.Timestamp(split) - pd.to_datetime(tr.date)).dt.days.values / 365
            sw = np.exp(-days / 1.5)
            m = HistGradientBoostingRegressor(
                learning_rate=0.03, max_iter=300, max_depth=md,
                l2_regularization=l2, min_samples_leaf=30, random_state=42)
            m.fit(tr[FEATS_ECON].values, tr.y_z.values, sample_weight=sw)
            p = m.predict(te[FEATS_ECON].values)
            eval_oos(p, te.fwd6.values, te.date.values,
                     f'HGBR d={md} l2={l2} (strict OOS)')

    # 混合: Ridge 0.8 + S_eng 0.2
    print('\n--- 混合模型 strict OOS ---')
    for w in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        days = (pd.Timestamp(split) - pd.to_datetime(tr.date)).dt.days.values / 365
        sw = np.exp(-days / 1.0)
        m = Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=1.0))])
        m.fit(tr[FEATS_ECON].values, tr.y_z.values, r__sample_weight=sw)
        p = w * m.predict(te[FEATS_ECON].values) + (1-w) * te.S_eng_z.values
        eval_oos(p, te.fwd6.values, te.date.values,
                 f'Blend {w}Ridge + {1-w}S_eng (strict OOS)')

    # 精简特征
    print('\n--- 精简特征集 strict OOS ---')
    for a in [0.5, 1.0, 2.0]:
        for dec in [1.0, 1.5, 2.0]:
            days = (pd.Timestamp(split) - pd.to_datetime(tr.date)).dt.days.values / 365
            sw = np.exp(-days / dec)
            m = Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=a))])
            m.fit(tr[FEATS_PARSE].values, tr.y_z.values, r__sample_weight=sw)
            p = m.predict(te[FEATS_PARSE].values)
            eval_oos(p, te.fwd6.values, te.date.values,
                     f'精简 Ridge a={a} dec={dec}')

    # 用全部 28 季的 walk-forward 模型，按真实时点预测 fwd12（12个月前瞻）
    print('\n--- fwd12 12月前瞻 ---')
    df12 = df.dropna(subset=['fwd12']).copy()
    dz12 = df12.groupby('date', group_keys=False).apply(
        lambda g: zscore_group(g, FEATS_ECON), include_groups=False)
    for c in ('date', 'code', 'fwd12', 'S_eng', 'water'):
        dz12[c] = df12[c].values
    # 重新做 y_z with fwd12
    def yz12(g):
        g['y_z'] = (g.fwd12 - g.fwd12.mean()) / (g.fwd12.std() + 1e-9)
        return g
    dz12 = dz12.groupby('date', group_keys=False).apply(yz12, include_groups=False)
    dates12 = sorted(dz12.date.unique())

    # V3.7 baseline
    base_preds = []
    for i in range(8, len(dates12)):
        te1 = dz12[dz12.date == dates12[i]]
        base_preds.append(pd.DataFrame(
            {'date': dates12[i], 'p': te1.S_eng_z.values, 'fwd': te1.fwd12.values}))
    bp = pd.concat(base_preds)
    ics = bp.groupby('date').apply(
        lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
        include_groups=False).dropna()
    bp['rk'] = bp.groupby('date').p.rank(pct=True)
    buy = bp[bp.rk >= 0.70]
    print(f'V3.7 fwd12: IC={ics.mean():+.4f} t={ics.mean()/ics.std()*np.sqrt(len(ics)):.2f} '
          f'Buy fwd={buy.fwd.mean():+.2%} win={(buy.fwd>0).mean():.0%}')

    # Ridge fwd12
    preds = []
    for i in range(8, len(dates12)):
        tr1 = dz12[dz12.date.isin(dates12[:i])]
        te1 = dz12[dz12.date == dates12[i]]
        days = (pd.Timestamp(dates12[i]) - pd.to_datetime(tr1.date)).dt.days.values / 365
        sw = np.exp(-days / 1.0)
        m = Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=1.0))])
        m.fit(tr1[FEATS_ECON].values, tr1.y_z.values, r__sample_weight=sw)
        p = m.predict(te1[FEATS_ECON].values)
        preds.append(pd.DataFrame({'date': dates12[i], 'p': p, 'fwd': te1.fwd12.values}))
    pp = pd.concat(preds)
    ics = pp.groupby('date').apply(
        lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
        include_groups=False).dropna()
    pp['rk'] = pp.groupby('date').p.rank(pct=True)
    buy = pp[pp.rk >= 0.70]
    print(f'Ridge fwd12: IC={ics.mean():+.4f} t={ics.mean()/ics.std()*np.sqrt(len(ics)):.2f} '
          f'Buy fwd={buy.fwd.mean():+.2%} win={(buy.fwd>0).mean():.0%}')


if __name__ == '__main__':
    main()
