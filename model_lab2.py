# -*- coding: utf-8 -*-
"""第二阶段实验：截面 z-score + 样本权重 + 混合模型。"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from model_lab import load_panel
from sklearn.linear_model import Ridge, RidgeCV, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATS = ['val_pct', 'trend_t', 'wr_rk', 'dc_rk', 'r4_rk', 'r7_rk',
         'rmdd_pen', 'R_MDD', 'water', 'val_x_trend', 'mom_x_val',
         'val_cov', 'ma20_dist']


def zscore_group(g):
    out = g.copy()
    out['y_z'] = (g.fwd6 - g.fwd6.mean()) / (g.fwd6.std() + 1e-9)
    for c in FEATS:
        out[c] = (g[c] - g[c].mean()) / (g[c].std() + 1e-9)
    out['S_eng_z'] = (g.S_eng - g.S_eng.mean()) / (g.S_eng.std() + 1e-9)
    return out


def eval_pred(preds, label):
    p = pd.concat(preds, ignore_index=True)
    ics = p.groupby('date').apply(
        lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
        include_groups=False).dropna()
    t = ics.mean() / ics.std() * np.sqrt(len(ics))
    p['rk'] = p.groupby('date').p.rank(pct=True)
    buy = p[p.rk >= 0.70]
    q10 = p[p.rk >= 0.90]
    print(f'{label:42s} IC={ics.mean():+.4f} t={t:5.2f} hit={(ics>0).mean():.2f} '
          f'Buy fwd={buy.fwd.mean():+.2%} win={(buy.fwd>0).mean():.2f} '
          f'Top10 fwd={q10.fwd.mean():+.2%}')
    return ics, p


def main():
    df = load_panel()
    dates = sorted(df.date.unique())
    dz = df.groupby('date', group_keys=False).apply(zscore_group, include_groups=False)
    for c in ('date', 'code', 'fwd6'):
        dz[c] = df[c].values

    min_train = 8

    def run(make_model, label, feats=None, use_decay=None, add_s_eng=False,
            blend_s_eng=None, target='y_z'):
        feats = feats or FEATS
        if add_s_eng:
            feats = feats + ['S_eng_z']
        preds = []
        for i in range(min_train, len(dates)):
            tr = dz[dz.date.isin(dates[:i])].copy()
            te = dz[dz.date == dates[i]]
            sw = None
            if use_decay:
                days = (pd.Timestamp(dates[i]) - pd.to_datetime(tr.date)).dt.days.values / 365
                sw = np.exp(-days / use_decay)
            m = make_model()
            fit_kw = {}
            if sw is not None and hasattr(m, 'named_steps'):
                fit_kw[m.steps[-1][0] + '__sample_weight'] = sw
            elif sw is not None:
                fit_kw['sample_weight'] = sw
            m.fit(tr[feats].values, tr[target].values, **fit_kw)
            p = m.predict(te[feats].values)
            if blend_s_eng is not None:
                p = blend_s_eng * p + (1 - blend_s_eng) * te.S_eng_z.values
            preds.append(pd.DataFrame({'date': dates[i], 'p': p, 'fwd': te.fwd6.values}))
        return eval_pred(preds, label)

    # 基准
    base_preds = []
    for i in range(min_train, len(dates)):
        te = dz[dz.date == dates[i]]
        base_preds.append(pd.DataFrame(
            {'date': dates[i], 'p': te.S_eng_z.values, 'fwd': te.fwd6.values}))
    eval_pred(base_preds, 'V3.7 S_eng (cs-z) baseline')

    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        'cs-z + Ridge(a=5)')
    run(lambda: Pipeline([('s', StandardScaler()),
                          ('r', RidgeCV(alphas=[0.5, 1, 5, 10, 50]))]),
        'cs-z + RidgeCV')
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        'cs-z + Ridge + decay(3y)', use_decay=3)
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        'cs-z + Ridge + decay(5y)', use_decay=5)
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=2.0))]),
        'cs-z + Ridge(a=2) + decay(3y)', use_decay=3)
    run(lambda: Pipeline([('s', StandardScaler()), ('r', HuberRegressor(
        epsilon=1.35, alpha=0.01, max_iter=500))]),
        'cs-z + Huber')
    run(lambda: HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=400, max_depth=2,
        l2_regularization=5.0, min_samples_leaf=30, random_state=42),
        'cs-z + HGBR(d2, lr03)')
    run(lambda: HistGradientBoostingRegressor(
        learning_rate=0.03, max_iter=400, max_depth=2,
        l2_regularization=5.0, min_samples_leaf=30, random_state=42),
        'cs-z + HGBR + decay(3y)', use_decay=3)
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        'cs-z Ridge + S_eng as feature', add_s_eng=True)
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        '0.7 Ridge + 0.3 S_eng', blend_s_eng=0.7)
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        '0.5 Ridge + 0.5 S_eng', blend_s_eng=0.5)

    # 目标用 rank 而非 z
    run(lambda: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=5.0))]),
        'cs-z + Ridge target=rank', target='fwd6')

    # 更多 alpha 网格
    for a in [0.5, 1, 2, 5, 10, 20, 50]:
        run(lambda a=a: Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=a))]),
            f'cs-z Ridge a={a} + decay(3y)', use_decay=3)


if __name__ == '__main__':
    main()
