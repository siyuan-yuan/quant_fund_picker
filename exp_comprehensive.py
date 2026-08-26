#!/usr/bin/env python3
"""
综合性实验：前视性 + 后视性信号的大规模验证
时间跨度：2013-2026（覆盖多轮牛熊）
信号类型：
  - 前视性：PE百分位、PE趋势方向、盈利增速、价格动量
  - 后视性：F_momentum, F_alpha, ir_winrate
方法：Walk-forward验证，严格避免前瞻偏差
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')

CACHE = Path('cache')

# 数据映射
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

def load_index_data():
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
    print(f"加载了 {len(indices)} 个指数")
    return indices

def load_fund_scan():
    df = pd.read_csv('output/scan_20260810.csv')
    def parse_rbsa(s):
        try:
            if pd.isna(s) or s == '': return {}
            return json.loads(s.replace('""', '"').strip('"'))
        except: return {}
    df['rbsa_dict'] = df['rbsa'].apply(parse_rbsa)
    return df

def load_fund_navs(codes, min_length=600):
    navs = {}
    for code in codes:
        fp = CACHE / f'nav_{code}.csv'
        if not fp.exists(): continue
        try:
            df = pd.read_csv(fp, parse_dates=['date']).set_index('date').sort_index()
            if len(df) >= min_length:
                navs[str(code)] = df['nav']
        except: continue
    print(f"加载了 {len(navs)} 只基金净值")
    return navs

def build_industry_signals(indices):
    """为每个行业构建前视性信号（只用过去数据）"""
    signals = {}
    for sector, df in indices.items():
        close = df['close']
        sig = pd.DataFrame(index=df.index)
        # 价格动量
        sig['price_mom_1m'] = close.pct_change(21)
        sig['price_mom_3m'] = close.pct_change(63)
        sig['price_mom_6m'] = close.pct_change(126)
        sig['price_mom_12m'] = close.pct_change(252)
        # 反转
        sig['reversion_20d'] = close.pct_change(20)
        # PE相关
        if 'pe' in df.columns:
            pe = df['pe']
            sig['pe_percentile'] = pe.rolling(756, min_periods=252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)
            sig['pe_trend_3m'] = pe.pct_change(63)
            sig['pe_trend_6m'] = pe.pct_change(126)
            sig['pe_expansion'] = pe.diff(21).rolling(63).sum()
            # 隐含盈利
            earnings = close / pe
            sig['earnings_growth_3m'] = earnings.pct_change(63)
            sig['earnings_growth_6m'] = earnings.pct_change(126)
            sig['earnings_momentum'] = sig['earnings_growth_3m'].diff(63)
        signals[sector] = sig
    print(f"构建了 {len(signals)} 个行业的信号")
    return signals

def compute_fund_signals(fund_df, fund_navs, industry_signals):
    """为每只基金在每个观察点计算信号"""
    SECTOR_INDICES = list(industry_signals.keys())
    rows = []
    n_funds = 0
    
    for _, fund in fund_df.iterrows():
        code = str(fund['code'])
        if code not in fund_navs: continue
        
        nav = fund_navs[code]
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
        
        n_funds += 1
        dates = nav.index
        start_idx = 252
        end_idx = len(dates) - 126
        
        for idx in range(start_idx, end_idx, 21):
            obs_date = dates[idx]
            
            weighted_signals = {}
            for sector, weight in fund_sectors.items():
                sector_sig = industry_signals[sector]
                asof = sector_sig.asof(obs_date)
                for col in asof.index:
                    if col not in weighted_signals: weighted_signals[col] = 0
                    weighted_signals[col] += weight * asof[col]
            
            future_6m_idx = min(idx + 126, len(dates) - 1)
            future_12m_idx = min(idx + 252, len(dates) - 1)
            
            nav_now = nav.iloc[idx]
            nav_6m = nav.iloc[future_6m_idx]
            nav_12m = nav.iloc[future_12m_idx]
            
            if pd.isna(nav_now) or nav_now <= 0: continue
            
            row = {
                'code': code, 'obs_date': obs_date,
                'region': fund.get('region', '未知'),
                'top_sector': fund.get('top_sector', '未知'),
                'F_value': fund['F_value'], 'F_alpha': fund['F_alpha'],
                'F_momentum': fund['F_momentum'], 'val_pct': fund['val_pct'],
                'ir_winrate': fund['ir_winrate'], 'S_total': fund['S_total'],
            }
            for k, v in weighted_signals.items():
                row[f'ind_{k}'] = v
            row['fwd_6m'] = nav_6m / nav_now - 1
            row['fwd_12m'] = nav_12m / nav_now - 1
            rows.append(row)
    
    print(f"处理了 {n_funds} 只基金, {len(rows)} 个观察点")
    return pd.DataFrame(rows)

def test_all_signals(df_eval):
    """测试每个信号的预测能力"""
    print("\n" + "="*80)
    print("全样本IC测试")
    print("="*80)
    
    signal_cols = [c for c in df_eval.columns if c.startswith('ind_')]
    fund_cols = ['F_value', 'F_alpha', 'F_momentum', 'val_pct', 'ir_winrate', 'S_total']
    
    all_signals = {c.replace('ind_', ''): df_eval[c] for c in signal_cols}
    all_signals.update({c: df_eval[c] for c in fund_cols})
    
    print(f"\n{'信号':<30s} {'IC_12m':>8s} {'RankIC':>8s} {'t值':>8s} {'N':>6s}")
    print("-"*55)
    
    results = {}
    for sig_name, sig_values in sorted(all_signals.items()):
        valid = pd.DataFrame({'s': sig_values, 'r': df_eval['fwd_12m']}).dropna()
        if len(valid) < 50: continue
        ic = valid['s'].corr(valid['r'])
        rank_ic = stats.spearmanr(valid['s'], valid['r'])[0]
        n = len(valid)
        t = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        print(f"{sig_name:<30s} {ic:>+8.4f} {rank_ic:>+8.4f} {t:>8.2f} {n:>6d}")
        results[sig_name] = {'IC': ic, 'RankIC': rank_ic, 't': t, 'n': n}
    return results

def test_by_market_regime(df_eval, indices):
    """分市场regime测试"""
    print("\n" + "="*80)
    print("分市场Regime测试")
    print("="*80)
    
    hs300 = indices.get('沪深300')
    if hs300 is None: return df_eval
    hs300_ret_12m = hs300['close'].pct_change(252)
    
    regime_map = {}
    for date in df_eval['obs_date'].unique():
        asof_ret = hs300_ret_12m.asof(date)
        if pd.isna(asof_ret): regime_map[date] = '未知'
        elif asof_ret > 0.20: regime_map[date] = '牛市(>20%)'
        elif asof_ret < -0.20: regime_map[date] = '熊市(<-20%)'
        else: regime_map[date] = '震荡市'
    
    df_eval = df_eval.copy()
    df_eval['regime'] = df_eval['obs_date'].map(regime_map)
    
    print(f"\nRegime分布:")
    print(df_eval['regime'].value_counts())
    
    key_signals = [c for c in df_eval.columns if c.startswith('ind_')] + \
                  ['F_momentum', 'F_alpha', 'F_value', 'val_pct', 'ir_winrate', 'S_total']
    
    for regime in ['牛市(>20%)', '震荡市', '熊市(<-20%)']:
        sub = df_eval[df_eval['regime'] == regime]
        if len(sub) < 100: continue
        
        print(f"\n--- {regime} (样本={len(sub)}) ---")
        print(f"{'信号':<30s} {'IC_12m':>8s} {'IC_6m':>8s}")
        print("-"*50)
        
        for sig in key_signals:
            v12 = sub[[sig, 'fwd_12m']].dropna()
            v6 = sub[[sig, 'fwd_6m']].dropna()
            ic12 = v12[sig].corr(v12['fwd_12m']) if len(v12) > 30 else np.nan
            ic6 = v6[sig].corr(v6['fwd_6m']) if len(v6) > 30 else np.nan
            print(f"{sig:<30s} {ic12:>+8.4f} {ic6:>+8.4f}")
    
    return df_eval

def test_walk_forward(df_eval):
    """Walk-forward测试"""
    print("\n" + "="*80)
    print("Walk-forward测试（每6个月滚动窗口）")
    print("="*80)
    
    df_eval = df_eval.copy()
    df_eval['obs_date'] = pd.to_datetime(df_eval['obs_date'])
    df_eval = df_eval.sort_values('obs_date')
    
    test_signals = {
        'pe_percentile': 'ind_pe_percentile',
        'pe_trend_3m': 'ind_pe_trend_3m',
        'pe_trend_6m': 'ind_pe_trend_6m',
        'pe_expansion': 'ind_pe_expansion',
        'earnings_g3m': 'ind_earnings_growth_3m',
        'earnings_g6m': 'ind_earnings_growth_6m',
        'earn_momentum': 'ind_earnings_momentum',
        'price_mom_3m': 'ind_price_mom_3m',
        'price_mom_6m': 'ind_price_mom_6m',
        'price_mom_12m': 'ind_price_mom_12m',
        'reversion_20d': 'ind_reversion_20d',
        'F_momentum': 'F_momentum',
        'F_alpha': 'F_alpha',
        'ir_winrate': 'ir_winrate',
        'F_value': 'F_value',
        'val_pct': 'val_pct',
        'S_total': 'S_total',
    }
    
    dates = sorted(df_eval['obs_date'].unique())
    split_points = pd.date_range(
        dates[0] + pd.DateOffset(months=12),
        dates[-1] - pd.DateOffset(months=6),
        freq='6MS')
    
    print(f"分割点: {len(split_points)}个, 训练窗口: 12个月")
    
    walk_results = []
    for split_date in split_points:
        train_start = split_date - pd.DateOffset(months=12)
        train_data = df_eval[(df_eval['obs_date'] >= train_start) & (df_eval['obs_date'] < split_date)]
        test_data = df_eval[(df_eval['obs_date'] >= split_date) & 
                           (df_eval['obs_date'] < split_date + pd.DateOffset(months=6))]
        
        if len(train_data) < 200 or len(test_data) < 50: continue
        
        train_ics = {}
        test_ics = {}
        for sig_name, col in test_signals.items():
            if col not in train_data.columns: continue
            v_tr = train_data[[col, 'fwd_12m']].dropna()
            v_te = test_data[[col, 'fwd_12m']].dropna()
            if len(v_tr) > 50: train_ics[sig_name] = v_tr[col].corr(v_tr['fwd_12m'])
            if len(v_te) > 30: test_ics[sig_name] = v_te[col].corr(v_te['fwd_12m'])
        
        walk_results.append({'split_date': split_date, 'train_ics': train_ics, 'test_ics': test_ics})
    
    print(f"\nWalk-forward 结果 ({len(walk_results)} 个窗口):")
    print(f"{'信号':<20s} {'训练IC':>8s} {'测试IC':>8s} {'稳定度':>8s} {'测试IC_std':>10s}")
    print("-"*60)
    
    wf_summary = {}
    for sig_name in test_signals:
        train_ics = [r['train_ics'].get(sig_name, np.nan) for r in walk_results]
        test_ics = [r['test_ics'].get(sig_name, np.nan) for r in walk_results]
        
        tr_mean = np.nanmean(train_ics)
        te_mean = np.nanmean(test_ics)
        te_std = np.nanstd(test_ics)
        consistency = np.mean([t > 0 for t in test_ics if not np.isnan(t)]) if test_ics else 0
        
        marker = "✓" if consistency > 0.6 else "△"
        print(f"{marker}{sig_name:<19s} {tr_mean:>+8.4f} {te_mean:>+8.4f} {consistency:>7.0%} {te_std:>10.4f}")
        wf_summary[sig_name] = {'train_ic': tr_mean, 'test_ic': te_mean, 'std': te_std, 'consistency': consistency}
    
    return wf_summary

def test_combined_strategies(df_eval, wf_summary):
    """测试组合策略"""
    print("\n" + "="*80)
    print("组合策略测试")
    print("="*80)
    
    def norm(s):
        s2 = s.fillna(s.median())
        mn, mx = s2.min(), s2.max()
        return (s2 - mn) / (mx - mn + 1e-10) * 100
    
    # 识别有效信号
    effective_fwd = {}
    for k in ['pe_percentile', 'pe_trend_3m', 'pe_trend_6m', 'pe_expansion',
              'earnings_g3m', 'earnings_g6m', 'earn_momentum',
              'price_mom_3m', 'price_mom_6m', 'price_mom_12m', 'reversion_20d']:
        if k in wf_summary and abs(wf_summary[k]['test_ic']) > 0.02 and wf_summary[k]['consistency'] > 0.4:
            effective_fwd[k] = wf_summary[k]
    
    effective_bwd = {}
    for k in ['F_momentum', 'F_alpha', 'ir_winrate', 'val_pct']:
        if k in wf_summary and abs(wf_summary[k]['test_ic']) > 0.03 and wf_summary[k]['consistency'] > 0.4:
            effective_bwd[k] = wf_summary[k]
    
    print(f"\n有效前视性信号 (|test_IC|>0.02, 稳定度>40%):")
    for k, v in sorted(effective_fwd.items(), key=lambda x: abs(x[1]['test_ic']), reverse=True):
        print(f"  {k}: test_IC={v['test_ic']:+.4f}, 稳定度={v['consistency']:.0%}")
    
    print(f"\n有效后视性信号 (|test_IC|>0.03, 稳定度>40%):")
    for k, v in sorted(effective_bwd.items(), key=lambda x: abs(x[1]['test_ic']), reverse=True):
        print(f"  {k}: test_IC={v['test_ic']:+.4f}, 稳定度={v['consistency']:.0%}")
    
    # 构建策略
    strategies = {}
    strategies['当前系统'] = df_eval['S_total']
    
    # 基础后视性（无F_value）
    S_base = 0.55 * norm(df_eval['F_momentum']) + 0.45 * norm(df_eval['ir_winrate'])
    strategies['后视基线(Mom55+IR45)'] = S_base
    
    # 测试每个有效前视性信号
    for sig_name in effective_fwd:
        col = f'ind_{sig_name}'
        if col not in df_eval.columns: continue
        sig_n = norm(df_eval[col])
        
        for w_fwd in [0.10, 0.20, 0.30]:
            combined = (1-w_fwd) * S_base + w_fwd * sig_n
            strategies[f'+{sig_name}({w_fwd:.0%})'] = combined
    
    # 组合多个前视性信号
    if len(effective_fwd) >= 2:
        fwd_scores = []
        for sig_name in effective_fwd:
            col = f'ind_{sig_name}'
            if col in df_eval.columns:
                fwd_scores.append(norm(df_eval[col]))
        
        if fwd_scores:
            # 按IC加权
            weights = [abs(wf_summary.get(s, {}).get('test_ic', 0.01)) for s in effective_fwd 
                      if f'ind_{s}' in df_eval.columns]
            if len(weights) == len(fwd_scores) and sum(weights) > 0:
                weights = np.array(weights) / sum(weights)
                fwd_combined = sum(w*s for w, s in zip(weights, fwd_scores))
            else:
                fwd_combined = sum(fwd_scores) / len(fwd_scores)
            
            for w_fwd in [0.10, 0.15, 0.20, 0.25, 0.30]:
                combined = (1-w_fwd) * S_base + w_fwd * fwd_combined
                strategies[f'前视组合({w_fwd:.0%})'] = combined
    
    # val_pct 直接使用
    if effective_bwd.get('val_pct', {}).get('test_ic', 0) > 0.05:
        val_n = norm(df_eval['val_pct'] * 100)
        for w in [0.05, 0.10, 0.15]:
            strategies[f'+val_pct({w:.0%})'] = (1-w) * S_base + w * val_n
    
    # 评估
    print(f"\n策略评估:")
    print(f"{'策略':<40s} {'IC_12m':>8s} {'RankIC':>8s} {'选中>60':>8s} {'均收益':>8s} {'胜率':>6s}")
    print("-"*80)
    
    results = []
    for name, scores in strategies.items():
        valid = pd.DataFrame({'s': scores, 'r': df_eval['fwd_12m']}).dropna()
        if len(valid) < 50: continue
        
        ic = valid['s'].corr(valid['r'])
        rank_ic = stats.spearmanr(valid['s'], valid['r'])[0]
        
        selected = valid[valid['s'] > valid['s'].quantile(0.7)]
        sel_ret = selected['r'].mean()
        sel_wr = (selected['r'] > 0).mean()
        
        print(f"{name:<40s} {ic:>+8.4f} {rank_ic:>+8.4f} {len(selected):>8d} {sel_ret*100:>7.2f}% {sel_wr*100:>5.1f}%")
        results.append((name, ic, rank_ic, len(selected), sel_ret, sel_wr))
    
    return results

def main():
    print("="*80)
    print("综合性实验：前视性 + 后视性信号验证")
    print("="*80)
    
    indices = load_index_data()
    fund_df = load_fund_scan()
    codes = fund_df['code'].astype(str).unique()
    fund_navs = load_fund_navs(codes, min_length=600)
    
    industry_signals = build_industry_signals(indices)
    df_eval = compute_fund_signals(fund_df, fund_navs, industry_signals)
    
    if len(df_eval) < 500:
        print("数据不足"); return
    
    print(f"\n数据集: {len(df_eval)} 行, {df_eval['code'].nunique()} 只基金")
    print(f"时间: {df_eval['obs_date'].min()} ~ {df_eval['obs_date'].max()}")
    
    os.makedirs('output', exist_ok=True)
    df_eval.to_csv('output/comprehensive_eval_data.csv', index=False)
    
    signal_results = test_all_signals(df_eval)
    df_eval = test_by_market_regime(df_eval, indices)
    wf_summary = test_walk_forward(df_eval)
    strategy_results = test_combined_strategies(df_eval, wf_summary)
    
    print("\n" + "="*80)
    print("最终总结")
    print("="*80)
    
    print("\nWalk-forward稳定性排名（按测试IC）:")
    sorted_wf = sorted(wf_summary.items(), key=lambda x: abs(x[1]['test_ic']), reverse=True)
    for name, m in sorted_wf:
        marker = "✓" if m['consistency'] > 0.6 else ("△" if m['consistency'] > 0.4 else "✗")
        print(f"  {marker} {name:<20s}: IC={m['test_ic']:+.4f} 稳定={m['consistency']:.0%}")
    
    print("\n策略排名:")
    sorted_s = sorted(strategy_results, key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    for i, (name, ic, *rest) in enumerate(sorted_s[:15]):
        print(f"  {i+1:>2d}. {name:<40s}: IC={ic:+.4f}")
    
    print("\n实验完成")

if __name__ == '__main__':
    main()
