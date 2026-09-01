#!/usr/bin/env python3
"""
基于实验结果的打分系统改进实现
1. 用earn_momentum替代F_value
2. 测试不同权重组合
3. Walk-forward验证改进效果
4. 分regime验证
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

CACHE = Path('cache')

SECTOR_PE_FILES = {
    '全指材料': 'csi_000987', '全指工业': 'csi_000988',
    '全指消费': 'csi_000990', '全指医药': 'csi_000991',
    '全指金融': 'csi_000992', '全指信息': 'csi_000993',
}
SECTOR_PRICE_FILES = {
    '上证50': 'idx_sh000016', '沪深300': 'idx_sh000300',
    '中证500': 'idx_sh000905', '中证1000': 'idx_sh000852',
    '创业板50': 'idx_sz399673', '上证红利': 'idx_sh000015',
    '恒生指数': 'idx_hk_HSI', '恒生科技': 'idx_hk_HSTECH',
    '纳斯达克100': 'idx_us__NDX', '标普500': 'idx_us__INX',
}

def load_all_data():
    """加载所有数据"""
    # 指数
    indices = {}
    for sector, code in SECTOR_PE_FILES.items():
        fp = CACHE / f'{code}.csv'
        if not fp.exists(): continue
        df = pd.read_csv(fp, parse_dates=['date']).set_index('date').sort_index()
        indices[sector] = df
    for sector, code in SECTOR_PRICE_FILES.items():
        fp = CACHE / f'{code}.csv'
        if not fp.exists(): continue
        df = pd.read_csv(fp, parse_dates=['date'])
        df.columns = [c.lower() for c in df.columns]
        df = df.set_index('date').sort_index()
        col = 'close' if 'close' in df.columns else df.columns[1]
        indices[sector] = pd.DataFrame({'close': df[col]})
    
    # 基金
    fund_df = pd.read_csv('output/scan_20260810.csv')
    def parse_rbsa(s):
        try:
            if pd.isna(s) or s == '': return {}
            return json.loads(s.replace('""', '"').strip('"'))
        except: return {}
    fund_df['rbsa_dict'] = fund_df['rbsa'].apply(parse_rbsa)
    
    # 净值
    navs = {}
    for code in fund_df['code'].astype(str).unique():
        fp = CACHE / f'nav_{code}.csv'
        if not fp.exists(): continue
        try:
            df = pd.read_csv(fp, parse_dates=['date']).set_index('date').sort_index()
            if len(df) >= 600:
                navs[str(code)] = df['nav']
        except: continue
    
    print(f"加载: {len(indices)}指数, {len(fund_df)}基金, {len(navs)}净值")
    return indices, fund_df, navs

def build_earn_momentum(indices):
    """只构建earn_momentum信号（实验中验证有效的前视性信号）"""
    signals = {}
    for sector, df in indices.items():
        if 'pe' not in df.columns: continue
        close = df['close']
        pe = df['pe']
        earnings = close / pe
        
        sig = pd.DataFrame(index=df.index)
        sig['earnings_growth_3m'] = earnings.pct_change(63)
        sig['earnings_momentum'] = sig['earnings_growth_3m'].diff(63)
        sig['price_mom_6m'] = close.pct_change(126)
        sig['pe_percentile'] = pe.rolling(756, min_periods=252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)
        signals[sector] = sig
    
    return signals

def build_eval_data(fund_df, navs, industry_signals, indices):
    """构建评估数据集"""
    SECTOR_INDICES = list(industry_signals.keys())
    rows = []
    
    # 沪深300用于regime判断
    hs300 = indices.get('沪深300')
    if hs300 is not None:
        hs300_ret_12m = hs300['close'].pct_change(252)
    else:
        hs300_ret_12m = None
    
    for _, fund in fund_df.iterrows():
        code = str(fund['code'])
        if code not in navs: continue
        
        nav = navs[code]
        rbsa = fund['rbsa_dict']
        
        fund_sectors = {}
        total_weight = 0
        for sector in SECTOR_INDICES:
            w = rbsa.get(sector, 0)
            if w > 0 and sector in industry_signals:
                fund_sectors[sector] = w
                total_weight += w
        
        if total_weight < 0.1: continue
        for s in fund_sectors: fund_sectors[s] /= total_weight
        
        dates = nav.index
        for idx in range(252, len(dates) - 126, 21):
            obs_date = dates[idx]
            
            # 行业加权信号
            weighted = {}
            for sector, weight in fund_sectors.items():
                asof = industry_signals[sector].asof(obs_date)
                for col in asof.index:
                    weighted[col] = weighted.get(col, 0) + weight * asof[col]
            
            # 未来收益
            nav_now = nav.iloc[idx]
            nav_6m = nav.iloc[min(idx+126, len(dates)-1)]
            nav_12m = nav.iloc[min(idx+252, len(dates)-1)]
            if pd.isna(nav_now) or nav_now <= 0: continue
            
            # Regime
            regime = '未知'
            if hs300_ret_12m is not None:
                r12 = hs300_ret_12m.asof(obs_date)
                if pd.notna(r12):
                    if r12 > 0.20: regime = '牛市'
                    elif r12 < -0.20: regime = '熊市'
                    else: regime = '震荡'
            
            row = {
                'code': code, 'obs_date': obs_date, 'regime': regime,
                'F_value': fund['F_value'], 'F_alpha': fund['F_alpha'],
                'F_momentum': fund['F_momentum'], 'val_pct': fund['val_pct'],
                'ir_winrate': fund['ir_winrate'], 'S_total': fund['S_total'],
                'earn_momentum': weighted.get('earnings_momentum', np.nan),
                'earnings_g3m': weighted.get('earnings_growth_3m', np.nan),
                'price_mom_6m': weighted.get('price_mom_6m', np.nan),
                'pe_percentile': weighted.get('pe_percentile', np.nan),
                'fwd_6m': nav_6m/nav_now - 1,
                'fwd_12m': nav_12m/nav_now - 1,
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def norm(s):
    """标准化到0-100"""
    s2 = s.fillna(s.median())
    mn, mx = s2.min(), s2.max()
    if mx - mn < 1e-10: return pd.Series(50, index=s.index)
    return (s2 - mn) / (mx - mn) * 100

def walk_forward_test(df_eval, scoring_funcs, train_months=12, test_months=6):
    """
    Walk-forward测试多个打分函数
    """
    df_eval = df_eval.copy()
    df_eval['obs_date'] = pd.to_datetime(df_eval['obs_date'])
    df_eval = df_eval.sort_values('obs_date')
    
    dates = sorted(df_eval['obs_date'].unique())
    splits = pd.date_range(
        dates[0] + pd.DateOffset(months=train_months),
        dates[-1] - pd.DateOffset(months=test_months),
        freq=f'{test_months}MS')
    
    results = {name: [] for name in scoring_funcs}
    
    for split_date in splits:
        train_start = split_date - pd.DateOffset(months=train_months)
        train = df_eval[(df_eval['obs_date'] >= train_start) & (df_eval['obs_date'] < split_date)]
        test = df_eval[(df_eval['obs_date'] >= split_date) & 
                      (df_eval['obs_date'] < split_date + pd.DateOffset(months=test_months))]
        
        if len(train) < 200 or len(test) < 50: continue
        
        for name, func in scoring_funcs.items():
            # 在测试集上计算IC
            scores = func(test)
            valid = pd.DataFrame({'s': scores, 'r': test['fwd_12m']}).dropna()
            if len(valid) > 30:
                ic = valid['s'].corr(valid['r'])
                results[name].append({'date': split_date, 'ic': ic, 'n': len(valid)})
    
    return results

def main():
    print("="*80)
    print("打分系统改进实现")
    print("="*80)
    
    indices, fund_df, navs = load_all_data()
    industry_signals = build_earn_momentum(indices)
    df_eval = build_eval_data(fund_df, navs, industry_signals, indices)
    
    print(f"评估数据: {len(df_eval)}行, {df_eval['code'].nunique()}只基金")
    print(f"时间: {df_eval['obs_date'].min()} ~ {df_eval['obs_date'].max()}")
    print(f"Regime: {df_eval['regime'].value_counts().to_dict()}")
    
    # ============================================================
    # 定义不同的打分函数
    # ============================================================
    
    def score_current(df):
        """当前系统"""
        return df['S_total']
    
    def score_no_value(df):
        """去掉F_value: Mom55 + IR45"""
        return 0.55 * norm(df['F_momentum']) + 0.45 * norm(df['ir_winrate'])
    
    def score_earn_mom_10(df):
        """加入10% earn_momentum"""
        return 0.45 * norm(df['F_momentum']) + 0.40 * norm(df['ir_winrate']) + 0.10 * norm(df['earn_momentum']) + 0.05 * norm(df['F_alpha'])
    
    def score_earn_mom_15(df):
        """加入15% earn_momentum"""
        return 0.40 * norm(df['F_momentum']) + 0.35 * norm(df['ir_winrate']) + 0.15 * norm(df['earn_momentum']) + 0.10 * norm(df['F_alpha'])
    
    def score_earn_mom_20(df):
        """加入20% earn_momentum"""
        return 0.35 * norm(df['F_momentum']) + 0.30 * norm(df['ir_winrate']) + 0.20 * norm(df['earn_momentum']) + 0.15 * norm(df['F_alpha'])
    
    def score_pure_earn_mom(df):
        """纯earn_momentum"""
        return norm(df['earn_momentum'])
    
    def score_mom_earn(df):
        """Momentum + earn_momentum (无IR, 无F_value)"""
        return 0.60 * norm(df['F_momentum']) + 0.40 * norm(df['earn_momentum'])
    
    def score_val_add(df):
        """后视基线 + val_pct (反向使用)"""
        return 0.45 * norm(df['F_momentum']) + 0.40 * norm(df['ir_winrate']) + 0.15 * norm(df['val_pct'] * 100)
    
    def score_best_combined(df):
        """综合最优：Mom + IR + earn_mom + val_pct"""
        return (0.35 * norm(df['F_momentum']) + 0.30 * norm(df['ir_winrate']) + 
                0.20 * norm(df['earn_momentum']) + 0.15 * norm(df['val_pct'] * 100))
    
    scoring_funcs = {
        '当前系统': score_current,
        '去掉F_value': score_no_value,
        'Mom+IR+earn10%': score_earn_mom_10,
        'Mom+IR+earn15%': score_earn_mom_15,
        'Mom+IR+earn20%': score_earn_mom_20,
        '纯earn_momentum': score_pure_earn_mom,
        'Mom60+earn40': score_mom_earn,
        '后视+val_pct': score_val_add,
        '综合最优': score_best_combined,
    }
    
    # ============================================================
    # 全样本评估
    # ============================================================
    print(f"\n{'='*80}")
    print("全样本评估")
    print(f"{'='*80}")
    
    print(f"\n{'策略':<25s} {'IC_12m':>8s} {'RankIC':>8s} {'选中>60%':>8s} {'均收益':>8s} {'胜率':>6s}")
    print("-"*70)
    
    full_results = []
    for name, func in scoring_funcs.items():
        scores = func(df_eval)
        valid = pd.DataFrame({'s': scores, 'r': df_eval['fwd_12m']}).dropna()
        if len(valid) < 100: continue
        
        ic = valid['s'].corr(valid['r'])
        rank_ic = stats.spearmanr(valid['s'], valid['r'])[0]
        
        # 选top 30%
        threshold = valid['s'].quantile(0.7)
        selected = valid[valid['s'] >= threshold]
        sel_ret = selected['r'].mean() * 100
        sel_wr = (selected['r'] > 0).mean() * 100
        
        print(f"{name:<25s} {ic:>+8.4f} {rank_ic:>+8.4f} {len(selected):>8d} {sel_ret:>7.2f}% {sel_wr:>5.1f}%")
        full_results.append((name, ic, rank_ic))
    
    # ============================================================
    # Walk-forward 测试
    # ============================================================
    print(f"\n{'='*80}")
    print("Walk-forward 测试（训练12个月 → 测试6个月）")
    print(f"{'='*80}")
    
    wf_results = walk_forward_test(df_eval, scoring_funcs)
    
    print(f"\n{'策略':<25s} {'测试IC':>8s} {'IC_std':>8s} {'正IC率':>8s} {'平均N':>8s}")
    print("-"*60)
    
    wf_summary = {}
    for name, results_list in wf_results.items():
        if not results_list: continue
        ics = [r['ic'] for r in results_list]
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        pos_rate = np.mean([ic > 0 for ic in ics])
        avg_n = np.mean([r['n'] for r in results_list])
        
        marker = "✓" if pos_rate > 0.6 else ("△" if pos_rate > 0.4 else "✗")
        print(f"{marker}{name:<24s} {mean_ic:>+8.4f} {std_ic:>8.4f} {pos_rate:>7.0%} {avg_n:>8.0f}")
        wf_summary[name] = {'ic': mean_ic, 'std': std_ic, 'pos_rate': pos_rate}
    
    # ============================================================
    # 分Regime评估
    # ============================================================
    print(f"\n{'='*80}")
    print("分Regime评估")
    print(f"{'='*80}")
    
    for regime in ['牛市', '震荡', '熊市']:
        sub = df_eval[df_eval['regime'] == regime]
        if len(sub) < 100: continue
        
        print(f"\n--- {regime} (样本={len(sub)}) ---")
        print(f"{'策略':<25s} {'IC_12m':>8s} {'IC_6m':>8s}")
        print("-"*45)
        
        for name, func in scoring_funcs.items():
            scores = func(sub)
            v12 = pd.DataFrame({'s': scores, 'r': sub['fwd_12m']}).dropna()
            v6 = pd.DataFrame({'s': scores, 'r': sub['fwd_6m']}).dropna()
            ic12 = v12['s'].corr(v12['r']) if len(v12) > 30 else np.nan
            ic6 = v6['s'].corr(v6['r']) if len(v6) > 30 else np.nan
            print(f"{name:<25s} {ic12:>+8.4f} {ic6:>+8.4f}")
    
    # ============================================================
    # 最终排名
    # ============================================================
    print(f"\n{'='*80}")
    print("最终排名（综合全样本IC + Walk-forward IC）")
    print(f"{'='*80}")
    
    combined_scores = []
    for name in scoring_funcs:
        full_ic = next((ic for n, ic, _ in full_results if n == name), np.nan)
        wf_ic = wf_summary.get(name, {}).get('ic', np.nan)
        wf_pos = wf_summary.get(name, {}).get('pos_rate', np.nan)
        
        if not np.isnan(full_ic) and not np.isnan(wf_ic):
            combined = 0.4 * full_ic + 0.6 * wf_ic  # Walk-forward权重更高
            combined_scores.append((name, full_ic, wf_ic, wf_pos, combined))
    
    combined_scores.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n{'排名':<5s} {'策略':<25s} {'全IC':>8s} {'WF_IC':>8s} {'WF正率':>8s} {'综合':>8s}")
    print("-"*65)
    for i, (name, full_ic, wf_ic, wf_pos, combined) in enumerate(combined_scores):
        marker = "★" if i == 0 else " "
        print(f"{marker}{i+1:<4d} {name:<25s} {full_ic:>+8.4f} {wf_ic:>+8.4f} {wf_pos:>7.0%} {combined:>+8.4f}")
    
    # ============================================================
    # 改进方案总结
    # ============================================================
    print(f"\n{'='*80}")
    print("改进方案总结")
    print(f"{'='*80}")
    
    best_name = combined_scores[0][0]
    baseline_name = '当前系统'
    baseline_full = next((ic for n, ic, _ in full_results if n == baseline_name), 0)
    baseline_wf = wf_summary.get(baseline_name, {}).get('ic', 0)
    best_full = combined_scores[0][1]
    best_wf = combined_scores[0][2]
    
    print(f"\n基线（当前系统）: 全IC={baseline_full:+.4f}, WF_IC={baseline_wf:+.4f}")
    print(f"最优方案（{best_name}）: 全IC={best_full:+.4f}, WF_IC={best_wf:+.4f}")
    print(f"全IC变化: {best_full - baseline_full:+.4f}")
    print(f"WF_IC变化: {best_wf - baseline_wf:+.4f}")

if __name__ == '__main__':
    main()
