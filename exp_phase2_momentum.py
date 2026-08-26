#!/usr/bin/env python3
"""
阶段2：动量信号 + 组合信号测试
测试不同时间窗口的动量效应，以及估值+动量的组合
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

CACHE_DIR = Path('cache')

# 所有有PE数据的行业
SECTOR_FILES = {
    'csi_000987': 'A股新材料',
    'csi_000988': 'A股工业',
    'csi_000990': 'A股消费',
    'csi_000991': 'A股医药',
    'csi_000992': 'A股金融',
    'csi_000993': 'A股信息',
    'csi_H30090': '港股消费',
    'csi_H30533': '港股科技',
}

# 只有价格数据的行业指数
IDX_FILES = {
    'idx_sh000300': '沪深300',
    'idx_sh000905': '中证500',
    'idx_sh000852': '中证1000',
    'idx_sh000688': '科创50',
    'idx_sh000300': '沪深300',
}

def load_all_sector_data():
    """加载所有有PE数据的行业指数"""
    sectors = {}
    for code, name in SECTOR_FILES.items():
        filepath = CACHE_DIR / f'{code}.csv'
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df.loc['2021-08-01':]  # 数据从2021-08开始
        if len(df) > 200:
            sectors[name] = df
    return sectors

def load_all_index_data():
    """加载所有价格数据（包括没有PE的指数）"""
    indices = {}
    for f in Path('cache').glob('idx_*.csv'):
        name = f.stem
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        col = 'close' if 'close' in df.columns else df.columns[1]
        indices[name] = df[col].sort_index()
    return indices

def test_momentum_signals(sectors):
    """
    测试多种时间窗口的动量信号
    动量 = 过去N个月收益
    未来收益 = 未来M个月收益
    """
    print("="*80)
    print("阶段2-A：动量信号多窗口测试")
    print("="*80)
    
    # 测试不同窗口组合
    past_windows = {
        '1M': 21,
        '3M': 63,
        '6M': 126,
        '9M': 189,
        '12M': 252,
    }
    
    future_windows = {
        '3M': 63,
        '6M': 126,
        '12M': 252,
    }
    
    all_results = []
    
    for name, df in sectors.items():
        print(f"\n--- {name} ---")
        
        for pw_name, pw_days in past_windows.items():
            # 计算过去收益
            past_ret = df['close'].pct_change(pw_days)
            
            for fw_name, fw_days in future_windows.items():
                # 计算未来收益
                future_ret = df['close'].pct_change(fw_days).shift(-fw_days)
                
                # 删除NaN
                valid = pd.DataFrame({'past': past_ret, 'future': future_ret}).dropna()
                
                if len(valid) < 50:
                    continue
                
                # 计算相关系数
                corr = valid['past'].corr(valid['future'])
                
                # 分位数分析
                past_q = valid['past'].quantile([0.2, 0.4, 0.6, 0.8])
                
                bottom = valid[valid['past'] <= past_q[0.2]]['future'].mean()
                top = valid[valid['past'] >= past_q[0.8]]['future'].mean()
                
                all_results.append({
                    'sector': name,
                    'past_window': pw_name,
                    'future_window': fw_name,
                    'corr': corr,
                    'bottom_quintile_return': bottom,
                    'top_quintile_return': top,
                    'spread': top - bottom,
                    'n_samples': len(valid),
                })
    
    df_results = pd.DataFrame(all_results)
    
    # 汇总分析
    print("\n" + "="*80)
    print("动量信号汇总（按过去窗口分组的平均相关性）")
    print("="*80)
    
    for pw in past_windows.keys():
        subset = df_results[df_results['past_window'] == pw]
        for fw in future_windows.keys():
            sub2 = subset[subset['future_window'] == fw]
            if len(sub2) > 0:
                avg_corr = sub2['corr'].mean()
                avg_spread = sub2['spread'].mean()
                positive = (sub2['corr'] > 0).sum()
                print(f"过去{pw:>3s} → 未来{fw:>3s}: "
                      f"平均相关性={avg_corr:+.4f}, "
                      f"平均收益差={avg_spread*100:+.2f}%, "
                      f"正相关行业数={positive}/{len(sub2)}")
    
    # 找到最优窗口
    print("\n" + "="*80)
    print("最优动量窗口分析")
    print("="*80)
    
    # 按所有行业汇总
    for fw in future_windows.keys():
        subset = df_results[df_results['future_window'] == fw]
        best = subset.loc[subset['corr'].abs().idxmax()]
        print(f"\n预测未来{fw}的最优窗口:")
        print(f"  过去{best['past_window']}: 平均相关性={best['corr']:+.4f}")
    
    return df_results

def test_lagged_momentum(sectors):
    """
    测试滞后动量（类似factors.py中的4M-1M, 7M-1M）
    跳过最近1个月，避免短期反转
    """
    print("\n" + "="*80)
    print("阶段2-B：滞后动量测试（跳过最近1个月）")
    print("="*80)
    
    all_results = []
    
    # 测试不同的滞后动量定义
    lagged_defs = {
        '4M-1M': (84, 21),     # 过去第4个月到第1个月
        '7M-1M': (189, 21),    # 过去第7个月到第1个月
        '6M-2M': (126, 42),    # 过去第6个月到第2个月
        '12M-1M': (252, 21),   # 过去第12个月到第1个月
        '12M-3M': (252, 63),   # 过去第12个月到第3个月
        '3M直接': (63, 0),     # 直接3个月收益
        '6M直接': (126, 0),    # 直接6个月收益
    }
    
    for name, df in sectors.items():
        for def_name, (long_window, skip_window) in lagged_defs.items():
            # 计算滞后动量
            close = df['close']
            if skip_window > 0:
                momentum = close.shift(skip_window) / close.shift(long_window) - 1
            else:
                momentum = close / close.shift(long_window) - 1
            
            # 未来6个月和12个月收益
            future_6m = close.pct_change(126).shift(-126)
            future_12m = close.pct_change(252).shift(-252)
            
            for fw_name, fw_ret in [('6M', future_6m), ('12M', future_12m)]:
                valid = pd.DataFrame({'mom': momentum, 'future': fw_ret}).dropna()
                
                if len(valid) < 50:
                    continue
                
                corr = valid['mom'].corr(valid['future'])
                
                all_results.append({
                    'sector': name,
                    'momentum_def': def_name,
                    'future_window': fw_name,
                    'corr': corr,
                    'n_samples': len(valid),
                })
    
    df_results = pd.DataFrame(all_results)
    
    # 按动量定义汇总
    print("\n按动量定义汇总：")
    for def_name in lagged_defs.keys():
        for fw in ['6M', '12M']:
            subset = df_results[(df_results['momentum_def'] == def_name) & 
                               (df_results['future_window'] == fw)]
            if len(subset) > 0:
                avg_corr = subset['corr'].mean()
                std_corr = subset['corr'].std()
                positive = (subset['corr'] > 0).sum()
                print(f"  {def_name:>10s} → 未来{fw}: "
                      f"corr={avg_corr:+.4f} (std={std_corr:.4f}), "
                      f"正相关={positive}/{len(subset)}")
    
    return df_results

def test_combined_signals(sectors):
    """
    测试组合信号：PE + 动量
    """
    print("\n" + "="*80)
    print("阶段2-C：组合信号测试（PE估值 + 动量）")
    print("="*80)
    
    all_data = []
    
    for name, df in sectors.items():
        # PE百分位
        pe_pct = df['pe'].rolling(window=504, min_periods=252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
        # 6个月动量
        momentum_6m = df['close'].pct_change(126)
        
        # 未来1年收益
        future_1y = df['close'].pct_change(252).shift(-252)
        
        valid = pd.DataFrame({
            'pe_pct': pe_pct,
            'mom': momentum_6m,
            'future': future_1y
        }).dropna()
        
        if len(valid) < 50:
            continue
        
        # 添加市场标记
        valid['market'] = 'A股' if 'A股' in name else '港股'
        valid['sector'] = name
        
        all_data.append(valid)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 测试不同的组合策略
    strategies = {
        '纯PE低估值': lambda r: r['pe_pct'] < 0.3,
        '纯PE高估值': lambda r: r['pe_pct'] > 0.7,
        '纯动量（正）': lambda r: r['mom'] > 0,
        '纯动量（负）': lambda r: r['mom'] < 0,
        '低PE+高动量': lambda r: (r['pe_pct'] < 0.3) & (r['mom'] > 0),
        '低PE+低动量': lambda r: (r['pe_pct'] < 0.3) & (r['mom'] < 0),
        '高PE+高动量': lambda r: (r['pe_pct'] > 0.7) & (r['mom'] > 0),
        '高PE+低动量': lambda r: (r['pe_pct'] > 0.7) & (r['mom'] < 0),
        '中PE+高动量': lambda r: (0.4 <= r['pe_pct']) & (r['pe_pct'] <= 0.6) & (r['mom'] > 0),
        '中PE+低动量': lambda r: (0.4 <= r['pe_pct']) & (r['pe_pct'] <= 0.6) & (r['mom'] < 0),
    }
    
    print("\n--- 全市场（所有行业）---")
    print(f"{'策略':<20s} {'样本数':>8s} {'平均收益':>10s} {'胜率':>8s} {'中位数':>10s}")
    print("-"*60)
    
    for strat_name, strat_func in strategies.items():
        mask = strat_func(df_all)
        subset = df_all[mask]
        if len(subset) >= 10:
            mean_ret = subset['future'].mean()
            winrate = (subset['future'] > 0).mean()
            median_ret = subset['future'].median()
            print(f"{strat_name:<20s} {len(subset):>8d} {mean_ret*100:>9.2f}% {winrate*100:>7.1f}% {median_ret*100:>9.2f}%")
    
    # 分A股和港股
    for market in ['A股', '港股']:
        print(f"\n--- {market} ---")
        print(f"{'策略':<20s} {'样本数':>8s} {'平均收益':>10s} {'胜率':>8s} {'中位数':>10s}")
        print("-"*60)
        
        market_data = df_all[df_all['market'] == market]
        
        for strat_name, strat_func in strategies.items():
            mask = strat_func(market_data)
            subset = market_data[mask]
            if len(subset) >= 10:
                mean_ret = subset['future'].mean()
                winrate = (subset['future'] > 0).mean()
                median_ret = subset['future'].median()
                print(f"{strat_name:<20s} {len(subset):>8d} {mean_ret*100:>9.2f}% {winrate*100:>7.1f}% {median_ret*100:>9.2f}%")
    
    # 相关性分析
    print("\n--- 相关性矩阵 ---")
    corr_pe_future = df_all['pe_pct'].corr(df_all['future'])
    corr_mom_future = df_all['mom'].corr(df_all['future'])
    
    a_share = df_all[df_all['market'] == 'A股']
    hk = df_all[df_all['market'] == '港股']
    
    print(f"全市场 PE与未来收益: {corr_pe_future:+.4f}")
    print(f"全市场 动量与未来收益: {corr_mom_future:+.4f}")
    print(f"A股   PE与未来收益: {a_share['pe_pct'].corr(a_share['future']):+.4f}")
    print(f"A股   动量与未来收益: {a_share['mom'].corr(a_share['future']):+.4f}")
    print(f"港股   PE与未来收益: {hk['pe_pct'].corr(hk['future']):+.4f}")
    print(f"港股   动量与未来收益: {hk['mom'].corr(hk['future']):+.4f}")
    
    return df_all

def test_pe_change_signal(sectors):
    """
    测试PE变化（而非PE水平）的预测能力
    假设：PE改善（盈利增长或价格下跌导致PE下降后回升）= 好信号
    """
    print("\n" + "="*80)
    print("阶段2-D：PE变化趋势信号测试")
    print("="*80)
    
    all_data = []
    
    for name, df in sectors.items():
        # PE变化（6个月）
        pe_change_6m = df['pe'].pct_change(126)
        
        # PE变化（3个月）
        pe_change_3m = df['pe'].pct_change(63)
        
        # PE百分位变化
        pe_pct = df['pe'].rolling(window=504, min_periods=252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        pe_pct_change_6m = pe_pct - pe_pct.shift(126)
        
        # 未来收益
        future_6m = df['close'].pct_change(126).shift(-126)
        future_1y = df['close'].pct_change(252).shift(-252)
        
        valid = pd.DataFrame({
            'pe_change_3m': pe_change_3m,
            'pe_change_6m': pe_change_6m,
            'pe_pct_change_6m': pe_pct_change_6m,
            'future_6m': future_6m,
            'future_1y': future_1y,
        }).dropna()
        
        if len(valid) < 50:
            continue
        
        valid['sector'] = name
        valid['market'] = 'A股' if 'A股' in name else '港股'
        all_data.append(valid)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    print("\n--- PE变化与未来收益相关性 ---")
    for signal_name in ['pe_change_3m', 'pe_change_6m', 'pe_pct_change_6m']:
        for future_name in ['future_6m', 'future_1y']:
            corr = df_all[signal_name].corr(df_all[future_name])
            
            a_mask = df_all['market'] == 'A股'
            hk_mask = df_all['market'] == '港股'
            
            a_corr = df_all.loc[a_mask, signal_name].corr(df_all.loc[a_mask, future_name])
            hk_corr = df_all.loc[hk_mask, signal_name].corr(df_all.loc[hk_mask, future_name])
            
            print(f"{signal_name:>20s} → {future_name:>10s}: "
                  f"全市场={corr:+.4f}, A股={a_corr:+.4f}, 港股={hk_corr:+.4f}")
    
    # PE扩张/收缩分析
    print("\n--- PE扩张（估值提升）vs PE收缩（估值下降）---")
    
    df_all['pe_expanding'] = df_all['pe_change_6m'] > 0
    
    for label, mask in [('PE扩张', df_all['pe_expanding']), 
                         ('PE收缩', ~df_all['pe_expanding'])]:
        subset = df_all[mask]
        if len(subset) > 10:
            mean_ret = subset['future_1y'].mean()
            winrate = (subset['future_1y'] > 0).mean()
            print(f"{label}: 样本={len(subset)}, "
                  f"未来1年收益={mean_ret*100:.2f}%, "
                  f"胜率={winrate*100:.1f}%")
    
    return df_all

def test_earnings_signal(sectors):
    """
    从PE和价格推导出盈利信号
    PE = Price / Earnings → Earnings = Price / PE
    盈利增速 = 盈利变化率
    """
    print("\n" + "="*80)
    print("阶段2-E：盈利增速信号测试")
    print("="*80)
    
    all_data = []
    
    for name, df in sectors.items():
        # 推导盈利
        earnings = df['close'] / df['pe']
        
        # 盈利增速（不同窗口）
        earnings_growth_6m = earnings.pct_change(126)  # 6个月
        earnings_growth_1y = earnings.pct_change(252)  # 1年
        
        # 盈利动量（加速度）
        earnings_momentum = earnings_growth_6m - earnings_growth_6m.shift(126)
        
        # 未来收益
        future_6m = df['close'].pct_change(126).shift(-126)
        future_1y = df['close'].pct_change(252).shift(-252)
        
        valid = pd.DataFrame({
            'earnings_growth_6m': earnings_growth_6m,
            'earnings_growth_1y': earnings_growth_1y,
            'earnings_momentum': earnings_momentum,
            'future_6m': future_6m,
            'future_1y': future_1y,
        }).dropna()
        
        if len(valid) < 50:
            continue
        
        valid['sector'] = name
        valid['market'] = 'A股' if 'A股' in name else '港股'
        all_data.append(valid)
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    print("\n--- 盈利增速与未来收益相关性 ---")
    for signal_name in ['earnings_growth_6m', 'earnings_growth_1y', 'earnings_momentum']:
        for future_name in ['future_6m', 'future_1y']:
            corr = df_all[signal_name].corr(df_all[future_name])
            
            a_mask = df_all['market'] == 'A股'
            hk_mask = df_all['market'] == '港股'
            
            a_corr = df_all.loc[a_mask, signal_name].corr(df_all.loc[a_mask, future_name])
            hk_corr = df_all.loc[hk_mask, signal_name].corr(df_all.loc[hk_mask, future_name])
            
            print(f"{signal_name:>25s} → {future_name:>10s}: "
                  f"全市场={corr:+.4f}, A股={a_corr:+.4f}, 港股={hk_corr:+.4f}")
    
    # 盈利增速分组
    print("\n--- 盈利增速分组 vs 未来1年收益 ---")
    
    for signal_name in ['earnings_growth_6m', 'earnings_growth_1y']:
        print(f"\n{signal_name}:")
        
        # 分组
        q = df_all[signal_name].quantile([0.2, 0.4, 0.6, 0.8])
        
        groups = {
            '最低20%': df_all[df_all[signal_name] <= q[0.2]],
            '20-40%': df_all[(df_all[signal_name] > q[0.2]) & (df_all[signal_name] <= q[0.4])],
            '40-60%': df_all[(df_all[signal_name] > q[0.4]) & (df_all[signal_name] <= q[0.6])],
            '60-80%': df_all[(df_all[signal_name] > q[0.6]) & (df_all[signal_name] <= q[0.8])],
            '最高20%': df_all[df_all[signal_name] > q[0.8]],
        }
        
        print(f"  {'分组':<10s} {'样本数':>8s} {'平均收益':>10s} {'胜率':>8s}")
        for g_name, g_data in groups.items():
            if len(g_data) > 5:
                mean_ret = g_data['future_1y'].mean()
                winrate = (g_data['future_1y'] > 0).mean()
                print(f"  {g_name:<10s} {len(g_data):>8d} {mean_ret*100:>9.2f}% {winrate*100:>7.1f}%")
    
    return df_all

def main():
    print("="*80)
    print("阶段2：多信号综合测试")
    print("="*80)
    
    sectors = load_all_sector_data()
    print(f"加载了 {len(sectors)} 个行业")
    
    # 2-A: 动量信号
    mom_results = test_momentum_signals(sectors)
    
    # 2-B: 滞后动量
    lagged_results = test_lagged_momentum(sectors)
    
    # 2-C: 组合信号
    combined_results = test_combined_signals(sectors)
    
    # 2-D: PE变化信号
    pe_change_results = test_pe_change_signal(sectors)
    
    # 2-E: 盈利增速信号
    earnings_results = test_earnings_signal(sectors)
    
    print("\n" + "="*80)
    print("阶段2完成")
    print("="*80)
    
    # 保存结果
    os.makedirs('output', exist_ok=True)
    mom_results.to_csv('output/phase2a_momentum.csv', index=False)
    lagged_results.to_csv('output/phase2b_lagged_momentum.csv', index=False)
    combined_results.to_csv('output/phase2c_combined.csv', index=False)
    pe_change_results.to_csv('output/phase2d_pe_change.csv', index=False)
    earnings_results.to_csv('output/phase2e_earnings.csv', index=False)
    print("结果已保存到 output/ 目录")

if __name__ == '__main__':
    main()
