# -*- coding: utf-8 -*-
"""第六阶段：在 strict hold-out 上挑选最终模型，并在 walk-forward 上交叉验证。
重点模型：Huber dec=1.5, HGBR d2, Blend 0.5。"""
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


def eval_ic(pred_df, label=''):
    p = pred_df.copy()
    ics = p.groupby('date').apply(
        lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
        include_groups=False).dropna()
    t = ics.mean() / ics.std() * np.sqrt(len(ics))
    p['rk'] = p.groupby('date').p.rank(pct=True)
    buy = p[p.rk >= 0.70]
    t10 = p[p.rk >= 0.90]
    q1 = p[p.rk < 0.20]
    print(f'{label:55s} IC={ics.mean():+.4f} t={t:5.2f} hit={(ics>0).mean():.2f} '
          f'Buy={buy.fwd.mean():+.2%}/{(buy.fwd>0).mean():.0%} '
          f'T10={t10.fwd.mean():+.2%} Q1={q1.fwd.mean():+.2%} LS={t10.fwd.mean()-q1.fwd.mean():+.2%}')
    return ics, p


def main():
    df = add_features(load_panel())
    dates = sorted(df.date.unique())
    dz = df.groupby('date', group_keys=False).apply(
        lambda g: zscore_group(g, FEATS_ECON), include_groups=False)
    for c in ('date', 'code', 'fwd6', 'S_eng', 'water'):
        dz[c] = df[c].values

    # 模型候选
    def ridge_model(a=0.5, dec=1.5):
        def f():
            return Pipeline([('s', StandardScaler()), ('r', Ridge(alpha=a))])
        return f, dec

    def huber_model(alpha=0.01, eps=1.35, dec=1.5):
        def f():
            return Pipeline([('s', StandardScaler()),
                             ('r', HuberRegressor(epsilon=eps, alpha=alpha, max_iter=1000))])
        return f, dec

    def hgbr_model(md=2, lr=0.03, l2=10, dec=1.5):
        def f():
            return HistGradientBoostingRegressor(
                learning_rate=lr, max_iter=400, max_depth=md,
                l2_regularization=l2, min_samples_leaf=30, random_state=42)
        return f, dec

    # ---- 第一关: strict OOS (train <=2022-06, test >=2022-09) ----
    split = dates[14]
    tr = dz[dz.date < split]
    te = dz[dz.date >= split]
    days = (pd.Timestamp(split) - pd.to_datetime(tr.date)).dt.days.values / 365

    candidates = [
        ('V3.7 baseline', None, None, None),
        ('Ridge a=0.5 d=1.5', *ridge_model(0.5, 1.5), FEATS_ECON),
        ('Ridge a=1.0 d=1.5', *ridge_model(1.0, 1.5), FEATS_ECON),
        ('Huber a=0.005 d=1.5', *huber_model(0.005, 1.35, 1.5), FEATS_ECON),
        ('Huber a=0.01 d=1.5', *huber_model(0.01, 1.35, 1.5), FEATS_ECON),
        ('Huber a=0.02 d=1.5', *huber_model(0.02, 1.35, 1.5), FEATS_ECON),
        ('Huber a=0.01 d=2.0', *huber_model(0.01, 1.35, 2.0), FEATS_ECON),
        ('HGBR d2 l2=10', *hgbr_model(2, 0.03, 10, 1.5), FEATS_ECON),
        ('HGBR d2 l2=20', *hgbr_model(2, 0.03, 20, 1.5), FEATS_ECON),
        ('精简 Ridge a=1 d=2', *ridge_model(1.0, 2.0), FEATS_PARSE),
        ('精简 Huber a=0.01 d=2', *huber_model(0.01, 1.35, 2.0), FEATS_PARSE),
    ]
    print('=== 严格样本外 (train <=2022-06-30, test >=2022-09-30) ===\n')
    oos_results = {}
    for name, factory, dec, feats in candidates:
        if factory is None:
            pred = te[['date', 'fwd6']].copy()
            pred['p'] = te.S_eng_z.values
            pred = pred.rename(columns={'fwd6': 'fwd'})
        else:
            m = factory()
            sw = np.exp(-days / dec)
            fit_kw = {}
            if hasattr(m, 'named_steps'):
                fit_kw[m.steps[-1][0] + '__sample_weight'] = sw
            else:
                fit_kw['sample_weight'] = sw
            m.fit(tr[feats].values, tr.y_z.values, **fit_kw)
            p = m.predict(te[feats].values)
            pred = pd.DataFrame({'date': te.date.values, 'fwd': te.fwd6.values, 'p': p})
        ic_s, _ = eval_ic(pred, name)
        oos_results[name] = float(ic_s.mean())

    # ---- 第二关: walk-forward 全样本 (确认模型稳定) ----
    print('\n=== Walk-forward (expanding, min_train=8) ===\n')

    def wf_eval(factory, dec, feats, label):
        preds = []
        for i in range(8, len(dates)):
            tr_i = dz[dz.date.isin(dates[:i])]
            te_i = dz[dz.date == dates[i]]
            days_i = (pd.Timestamp(dates[i]) - pd.to_datetime(tr_i.date)).dt.days.values / 365
            sw = np.exp(-days_i / dec) if dec else None
            m = factory()
            fit_kw = {}
            if sw is not None:
                if hasattr(m, 'named_steps'):
                    fit_kw[m.steps[-1][0] + '__sample_weight'] = sw
                else:
                    fit_kw['sample_weight'] = sw
            m.fit(tr_i[feats].values, tr_i.y_z.values, **fit_kw)
            p = m.predict(te_i[feats].values)
            preds.append(pd.DataFrame(
                {'date': dates[i], 'code': te_i.code.values,
                 'p': p, 'fwd': te_i.fwd6.values, 'water': te_i.water.values}))
        return eval_ic(pd.concat(preds, ignore_index=True), label), pd.concat(preds, ignore_index=True)

    wf_results = {}
    wf_preds = {}
    for name, factory, dec, feats in candidates:
        if factory is None:
            preds = []
            for i in range(8, len(dates)):
                te_i = dz[dz.date == dates[i]]
                preds.append(pd.DataFrame(
                    {'date': dates[i], 'code': te_i.code.values,
                     'p': te_i.S_eng_z.values, 'fwd': te_i.fwd6.values,
                     'water': te_i.water.values}))
            pp = pd.concat(preds, ignore_index=True)
        else:
            (_, pp) = wf_eval(factory, dec, feats, name + ' (WF)') if False else (None, None)
            # 重新跑
            preds = []
            for i in range(8, len(dates)):
                tr_i = dz[dz.date.isin(dates[:i])]
                te_i = dz[dz.date == dates[i]]
                days_i = (pd.Timestamp(dates[i]) - pd.to_datetime(tr_i.date)).dt.days.values / 365
                sw = np.exp(-days_i / dec) if dec else None
                m = factory()
                fit_kw = {}
                if sw is not None:
                    if hasattr(m, 'named_steps'):
                        fit_kw[m.steps[-1][0] + '__sample_weight'] = sw
                    else:
                        fit_kw['sample_weight'] = sw
                m.fit(tr_i[feats].values, tr_i.y_z.values, **fit_kw)
                p = m.predict(te_i[feats].values)
                preds.append(pd.DataFrame(
                    {'date': dates[i], 'code': te_i.code.values,
                     'p': p, 'fwd': te_i.fwd6.values, 'water': te_i.water.values}))
            pp = pd.concat(preds, ignore_index=True)
        ic_s, _ = eval_ic(pp, name + ' (WF)')
        wf_results[name] = float(ic_s.mean())
        wf_preds[name] = pp

    # 综合分: 0.5 strict OOS + 0.5 WF
    print('\n=== 综合评分 (0.5*OOS + 0.5*WF) ===\n')
    rows = []
    for name in oos_results:
        rows.append({'model': name, 'IC_OOS': oos_results[name],
                     'IC_WF': wf_results.get(name, np.nan),
                     'score': 0.5 * oos_results[name] + 0.5 * wf_results.get(name, 0)})
    rows = sorted(rows, key=lambda r: -r['score'])
    for r in rows:
        print(f"{r['model']:35s} OOS={r['IC_OOS']:+.4f}  WF={r['IC_WF']:+.4f}  score={r['score']:+.4f}")
    pd.DataFrame(rows).to_csv('output/model_lab/finalist_comparison.csv',
                              index=False, encoding='utf-8-sig')

    # 选冠军（基于综合分）
    winner = rows[0]['model']
    print(f"\n[champion] {winner}")
    champ_pp = wf_preds[winner]
    champ_pp.to_csv('output/model_lab/champion_wf_predictions.csv',
                    index=False, encoding='utf-8-sig')

    # ---- 冠军模型分 regime 表现 ----
    print('\n=== 冠军模型分 regime 表现 (WF) ===')
    for lab, mask in [('低估区(≤20%)', champ_pp.water <= 0.20),
                      ('中性区(20-70%)', (champ_pp.water > 0.20) & (champ_pp.water <= 0.70)),
                      ('高估区(>70%)', champ_pp.water > 0.70)]:
        sub = champ_pp[mask]
        ic_s = sub.groupby('date').apply(
            lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
            include_groups=False).dropna()
        if len(ic_s):
            print(f'  {lab:15s} nQ={len(ic_s)} IC={ic_s.mean():+.4f} '
                  f't={ic_s.mean()/ic_s.std()*np.sqrt(len(ic_s)):+.2f} hit={(ic_s>0).mean():.2f}')

    # ---- 按市场方向 ----
    mkt = champ_pp.groupby('date').fwd.mean()
    for lab, dts in [('上涨季', mkt[mkt > 0].index), ('下跌季', mkt[mkt <= 0].index)]:
        sub = champ_pp[champ_pp.date.isin(dts)]
        ic_s = sub.groupby('date').apply(
            lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
            include_groups=False).dropna()
        if len(ic_s):
            print(f'  {lab:8s} nQ={len(ic_s)} IC={ic_s.mean():+.4f} '
                  f't={ic_s.mean()/ic_s.std()*np.sqrt(len(ic_s)):+.2f}')

    # ---- 逐年 IC ----
    print('\n=== 冠军模型逐年 IC (WF) ===')
    champ_pp['year'] = pd.to_datetime(champ_pp.date).dt.year
    for y, sub in champ_pp.groupby('year'):
        ic_s = sub.groupby('date').apply(
            lambda g: spearmanr(g.p, g.fwd)[0] if g.p.nunique() > 5 else np.nan,
            include_groups=False).dropna()
        if len(ic_s):
            print(f'  {y}: IC={ic_s.mean():+.4f} ({len(ic_s)}季, hit={(ic_s>0).mean():.0%})')


if __name__ == '__main__':
    main()
