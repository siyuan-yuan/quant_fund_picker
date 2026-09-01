#!/usr/bin/env python3
"""
Regime检测 + 双系统切换实验

目标：
1. 找到最可靠的regime检测方法
2. 检测误差对系统的影响
3. 双系统 vs 单系统的IC对比
4. Walk-forward验证（严格避免前瞻偏差）
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

CACHE = 'cache'

# ============================================================
# 1. 加载评估数据
# ============================================================
print("="*80)
print("Regime检测 + 双系统切换实验")
print("="*80)

df = pd.read_csv('output/comprehensive_eval_data.csv', parse_dates=['obs_date'])
df['earn_momentum'] = df['ind_earnings_momentum']
print(f"评估数据: {len(df)}行, {df['obs_date'].min().date()} ~ {df['obs_date'].max().date()}")

# ============================================================
# 2. 定义真实regime（用未来12个月收益回测）
# ============================================================
# 先定义"真实"regime，作为参照基准

# 方法A: 沪深300过去12月涨幅
hs300 = pd.read_csv(f'{CACHE}/idx_sh000300.csv', parse_dates=['date']).set_index('date').sort_index()
hs300_ret_12m = hs300['close'].pct_change(252)

# 方法B: 大盘水位（PE百分位）—— 已有系统的方法
# 方法C: 波动率（20日滚动标准差）
hs300_vol = hs300['close'].pct_change().rolling(20).std() * np.sqrt(252)

# 为每个观察点计算regime指标
def get_regime_signals(obs_date):
    """获取观察日期的所有regime指标"""
    ts = pd.Timestamp(obs_date)
    
    r12m = hs300_ret_12m.asof(ts)
    r6m = hs300['close'].pct_change(126).asof(ts)
    r3m = hs300['close'].pct_change(63).asof(ts)
    r1m = hs300['close'].pct_change(21).asof(ts)
    vol = hs300_vol.asof(ts)
    
    # 距离高点/低点的距离
    recent_high = hs300['close'].loc[:ts].tail(252).max()
    recent_low = hs300['close'].loc[:ts].tail(252).min()
    dist_from_high = hs300['close'].asof(ts) / recent_high - 1
    dist_from_low = hs300['close'].asof(ts) / recent_low - 1
    
    # 均线位置
    ma120 = hs300['close'].loc[:ts].tail(120).mean()
    ma250 = hs300['close'].loc[:ts].tail(250).mean()
    above_ma120 = hs300['close'].asof(ts) / ma120 - 1
    above_ma250 = hs300['close'].asof(ts) / ma250 - 1
    
    return {
        'hs300_12m': r12m, 'hs300_6m': r6m, 'hs300_3m': r3m, 'hs300_1m': r1m,
        'vol_20d': vol, 'dist_from_high': dist_from_high, 'dist_from_low': dist_from_low,
        'above_ma120': above_ma120, 'above_ma250': above_ma250,
    }

print("\n计算每个观察点的regime指标...")
regime_data = []
for d in df['obs_date'].unique():
    signals = get_regime_signals(d)
    signals['date'] = d
    regime_data.append(signals)

regime_df = pd.DataFrame(regime_data).set_index('date')
print(f"Regime指标: {len(regime_df)} 个日期")

# 合并到主数据
df = df.merge(regime_df, left_on='obs_date', right_index=True, how='left')

# ============================================================
# 3. 测试不同regime检测方法的有效性
# ============================================================
print("\n" + "="*80)
print("3. Regime检测方法有效性测试")
print("="*80)

# 用每个regime指标将数据分成"牛市"和"熊市"两组
# 然后测试每组中 earn_momentum 的IC

regime_methods = {
    'HS300_12M>20%': lambda r: r['hs300_12m'] > 0.20,
    'HS300_12M<-20%': lambda r: r['hs300_12m'] < -0.20,
    'HS300_6M>10%': lambda r: r['hs300_6m'] > 0.10,
    'HS300_6M<-10%': lambda r: r['hs300_6m'] < -0.10,
    'HS300_3M>5%': lambda r: r['hs300_3m'] > 0.05,
    'HS300_3M<-5%': lambda r: r['hs300_3m'] < -0.05,
    '高于MA250': lambda r: r['above_ma250'] > 0,
    '低于MA250': lambda r: r['above_ma250'] < 0,
    '高于MA120': lambda r: r['above_ma120'] > 0,
    '低于MA120': lambda r: r['above_ma120'] < 0,
    '距高点>-10%': lambda r: r['dist_from_high'] > -0.10,
    '距高点<-20%': lambda r: r['dist_from_high'] < -0.20,
    '低波动(<15%)': lambda r: r['vol_20d'] < 0.15,
    '高波动(>25%)': lambda r: r['vol_20d'] > 0.25,
}

print(f"\n各regime检测方法下的 earn_momentum IC:")
print(f"{'方法':<20s} {'牛市IC':>8s} {'熊市IC':>8s} {'差异':>8s} {'N_bull':>7s} {'N_bear':>7s}")
print("-"*65)

for method_name, condition_func in regime_methods.items():
    bull_mask = condition_func(regime_df)
    bear_mask = ~bull_mask
    
    bull_dates = regime_df[bull_mask].index
    bear_dates = regime_df[bear_mask].index
    
    bull_data = df[df['obs_date'].isin(bull_dates)]
    bear_data = df[df['obs_date'].isin(bear_dates)]
    
    if len(bull_data) < 50 or len(bear_data) < 50:
        continue
    
    bull_ic = bull_data['earn_momentum'].corr(bull_data['fwd_12m']) if len(bull_data) > 50 else np.nan
    bear_ic = bear_data['earn_momentum'].corr(bear_data['fwd_12m']) if len(bear_data) > 50 else np.nan
    
    diff = bear_ic - bull_ic if not (np.isnan(bear_ic) or np.isnan(bull_ic)) else np.nan
    
    print(f"{method_name:<20s} {bull_ic:>+8.4f} {bear_ic:>+8.4f} {diff:>+8.4f} {len(bull_data):>7d} {len(bear_data):>7d}")

# ============================================================
# 4. 最佳regime检测方法的详细分析
# ============================================================
print("\n" + "="*80)
print("4. 最佳regime检测方法详细分析")
print("="*80)

# 测试不同阈值的HS300_12M
print("\n--- HS300 12M涨幅的不同阈值 ---")
print(f"{'阈值':<20s} {'上方IC':>8s} {'下方IC':>8s} {'差异':>8s} {'N_above':>8s} {'N_below':>8s}")
print("-"*70)

for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    above = df[df['hs300_12m'] > threshold]
    below = df[df['hs300_12m'] <= threshold]
    
    ic_above = above['earn_momentum'].corr(above['fwd_12m']) if len(above) > 50 else np.nan
    ic_below = below['earn_momentum'].corr(below['fwd_12m']) if len(below) > 50 else np.nan
    diff = ic_below - ic_above if not (np.isnan(ic_below) or np.isnan(ic_above)) else np.nan
    
    print(f"HS300_12M>{threshold:.0%}    {ic_above:>+8.4f} {ic_below:>+8.4f} {diff:>+8.4f} {len(above):>8d} {len(below):>8d}")

# 测试MA方法
print("\n--- 均线方法 ---")
for ma_col in ['above_ma120', 'above_ma250']:
    above = df[df[ma_col] > 0]
    below = df[df[ma_col] <= 0]
    
    ic_above = above['earn_momentum'].corr(above['fwd_12m']) if len(above) > 50 else np.nan
    ic_below = below['earn_momentum'].corr(below['fwd_12m']) if len(below) > 50 else np.nan
    
    # 同时测试所有因子
    print(f"\n{ma_col}>0 (上方 N={len(above)}) / <=0 (下方 N={len(below)}):")
    print(f"  {'因子':<20s} {'上方IC':>8s} {'下方IC':>8s}")
    print(f"  {'-'*40}")
    
    for factor in ['earn_momentum', 'F_momentum', 'ir_winrate', 'F_alpha', 'F_value', 'val_pct']:
        if factor not in df.columns: continue
        ic_a = above[factor].corr(above['fwd_12m']) if len(above) > 50 else np.nan
        ic_b = below[factor].corr(below['fwd_12m']) if len(below) > 50 else np.nan
        print(f"  {factor:<20s} {ic_a:>+8.4f} {ic_b:>+8.4f}")

# ============================================================
# 5. 双系统切换的Walk-forward测试
# ============================================================
print("\n" + "="*80)
print("5. 双系统切换 Walk-forward测试")
print("="*80)

def norm_train_test(train_s, test_s):
    """用训练集的min/max归一化测试集"""
    mn, mx = train_s.min(), train_s.max()
    if mx - mn < 1e-10: return pd.Series(50, index=test_s.index)
    return ((test_s.fillna(train_s.median()) - mn) / (mx - mn)).clip(0, 1) * 100

df = df.sort_values('obs_date')
dates = sorted(df['obs_date'].unique())
splits = pd.date_range(
    dates[0] + pd.DateOffset(months=12),
    dates[-1] - pd.DateOffset(months=6),
    freq='6MS')

print(f"Walk-forward: {len(splits)}个窗口, 训练12月→测试6月")

# 定义策略
strategy_names = [
    '单系统(当前)',
    '单系统(最优)',
    '双系统(HS300_12M>15%)',
    '双系统(MA250)',
    '双系统(HS300_6M>0%)',
    ' oracle (完美预知)',
]

results = {n: [] for n in strategy_names}

for split_date in splits:
    train_start = split_date - pd.DateOffset(months=12)
    train = df[(df['obs_date'] >= train_start) & (df['obs_date'] < split_date)]
    test = df[(df['obs_date'] >= split_date) & (df['obs_date'] < split_date + pd.DateOffset(months=6))]
    
    if len(train) < 200 or len(test) < 50:
        continue
    
    # 归一化因子
    mom = norm_train_test(train['F_momentum'], test['F_momentum'])
    ir = norm_train_test(train['ir_winrate'], test['ir_winrate'])
    earn = norm_train_test(train['earn_momentum'], test['earn_momentum'])
    alpha = norm_train_test(train['F_alpha'], test['F_alpha'])
    val = norm_train_test(train['val_pct'] * 100, test['val_pct'] * 100)
    
    # 策略1: 当前系统 (S_total, 用训练集权重重新模拟)
    s_current = 0.40 * norm_train_test(train['F_value'].clip(0,100), test['F_value'].clip(0,100)) + \
                0.35 * ir + 0.25 * mom
    
    # 策略2: 单系统最优 (固定权重)
    s_single = 0.35 * mom + 0.30 * ir + 0.20 * earn + 0.15 * alpha
    
    # 策略3-5: 双系统切换
    # 牛市: 不用earn_momentum (因为IC=-0.10)
    # 熊市/震荡: 用earn_momentum
    
    # 方法3: HS300 12M > 15% 作为牛市阈值
    test_hs300_12m = test['hs300_12m']
    bull_3 = test_hs300_12m > 0.15
    
    s_bull = 0.40 * mom + 0.35 * ir + 0.10 * val + 0.15 * alpha  # 牛市：重动量+IR，加val_pct
    s_bear = 0.35 * mom + 0.30 * ir + 0.20 * earn + 0.15 * alpha  # 熊/震荡：加earn_momentum
    
    s_dual_3 = pd.Series(np.nan, index=test.index)
    s_dual_3[bull_3] = s_bull[bull_3]
    s_dual_3[~bull_3] = s_bear[~bull_3]
    
    # 方法4: MA250
    bull_4 = test['above_ma250'] > 0
    s_dual_4 = pd.Series(np.nan, index=test.index)
    s_dual_4[bull_4] = s_bull[bull_4]
    s_dual_4[~bull_4] = s_bear[~bull_4]
    
    # 方法5: HS300 6M > 0%
    bull_5 = test['hs300_6m'] > 0
    s_dual_5 = pd.Series(np.nan, index=test.index)
    s_dual_5[bull_5] = s_bull[bull_5]
    s_dual_5[~bull_5] = s_bear[~bull_5]
    
    # 策略6: Oracle (完美预知regime)
    # 用实际IC符号决定：如果earn_momentum在测试期的IC>0则用bear系统，否则用bull
    earn_ic = test['earn_momentum'].corr(test['fwd_12m'])
    if earn_ic > 0:
        s_oracle = s_bear  # earn有效，用含earn的系统
    else:
        s_oracle = s_bull  # earn无效，用不含earn的系统
    
    # 计算IC
    scores_dict = {
        '单系统(当前)': s_current,
        '单系统(最优)': s_single,
        '双系统(HS300_12M>15%)': s_dual_3,
        '双系统(MA250)': s_dual_4,
        '双系统(HS300_6M>0%)': s_dual_5,
        'oracle (完美预知)': s_oracle,
    }
    
    for name, scores in scores_dict.items():
        v = pd.DataFrame({'s': scores, 'r': test['fwd_12m']}).dropna()
        if len(v) > 30:
            ic = v['s'].corr(v['r'])
            results[name].append(ic)

# 汇总
print(f"\n{'策略':<30s} {'WF_IC':>8s} {'IC_std':>8s} {'正IC率':>8s} {'N窗口':>6s}")
print("-"*65)

for name, ics in results.items():
    if not ics: continue
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    pos_rate = np.mean([ic > 0 for ic in ics])
    print(f"{name:<30s} {mean_ic:>+8.4f} {std_ic:>8.4f} {pos_rate:>7.0%} {len(ics):>6d}")

# ============================================================
# 6. Regime检测误差分析
# ============================================================
print("\n" + "="*80)
print("6. Regime检测误差分析")
print("="*80)

# 关键问题：如果错误判断regime，代价有多大？
# 牛市误判为熊市：加了earn_momentum，但earn在牛市中IC<0 → 损失
# 熊市误判为牛市：没加earn_momentum，但earn在熊市中有效 → 损失

df_sub = df[df['earn_momentum'].notna()].copy()

# 用HS300_12M>15%定义牛市
df_sub['is_bull'] = df_sub['hs300_12m'] > 0.15

bull_data = df_sub[df_sub['is_bull']]
bear_data = df_sub[~df_sub['is_bull']]

print(f"\n牛市样本: {len(bull_data)}, 熊市/震荡样本: {len(bear_data)}")

# 在牛市中：
# - 用牛市系统（不加earn）: IC_bull_noearn
# - 用熊市系统（加earn）: IC_bull_withearn
# → 误判代价 = IC_bull_noearn - IC_bull_withearn

# 在熊市中：
# - 用熊市系统（加earn）: IC_bear_withearn
# - 用牛市系统（不加earn）: IC_bear_noearn
# → 误判代价 = IC_bear_withearn - IC_bear_noearn

def norm_col(s):
    s2 = s.fillna(s.median())
    mn, mx = s2.min(), s2.max()
    return (s2 - mn) / (mx - mn + 1e-10) * 100

mom_n = norm_col(df_sub['F_momentum'])
ir_n = norm_col(df_sub['ir_winrate'])
earn_n = norm_col(df_sub['earn_momentum'])
alpha_n = norm_col(df_sub['F_alpha'])
val_n = norm_col(df_sub['val_pct'] * 100)

# 牛市系统（不加earn）
s_bull_full = 0.40 * mom_n + 0.35 * ir_n + 0.10 * val_n + 0.15 * alpha_n
# 熊市系统（加earn）
s_bear_full = 0.35 * mom_n + 0.30 * ir_n + 0.20 * earn_n + 0.15 * alpha_n

for label, data, idx in [('牛市', bull_data, bull_data.index), 
                          ('熊市/震荡', bear_data, bear_data.index)]:
    ic_bull = s_bull_full[idx].corr(data['fwd_12m'])
    ic_bear = s_bear_full[idx].corr(data['fwd_12m'])
    
    print(f"\n{label}:")
    print(f"  用牛市系统(不加earn): IC={ic_bull:+.4f}")
    print(f"  用熊市系统(加earn):   IC={ic_bear:+.4f}")
    
    if label == '牛市':
        cost_wrong = ic_bull - ic_bear  # 牛市误判为熊市的代价
        print(f"  误判代价(牛市误为熊市): {cost_wrong:+.4f}")
    else:
        cost_wrong = ic_bear - ic_bull  # 熊市误判为牛市的代价
        print(f"  误判代价(熊市误为牛市): {cost_wrong:+.4f}")

# ============================================================
# 7. 最终推荐
# ============================================================
print("\n" + "="*80)
print("7. 最终推荐")
print("="*80)

# 找最佳regime检测方法和阈值
print("\n--- 不同regime检测方法的Walk-forward IC对比 ---")
# 已经在上面做了，现在总结

# 测试更多检测方法的组合
print("\n--- 细粒度阈值搜索 ---")
best_configs = []

for method in ['hs300_12m', 'hs300_6m', 'hs300_3m', 'above_ma120', 'above_ma250']:
    if method in ['above_ma120', 'above_ma250']:
        thresholds = [0]
    elif method == 'hs300_3m':
        thresholds = [-0.05, -0.03, 0, 0.03, 0.05]
    elif method == 'hs300_6m':
        thresholds = [-0.15, -0.10, -0.05, 0, 0.05, 0.10]
    else:
        thresholds = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    for threshold in thresholds:
        if method in ['above_ma120', 'above_ma250']:
            bull_mask = df[method] > threshold
        else:
            bull_mask = df[method] > threshold
        
        n_bull = bull_mask.sum()
        n_bear = (~bull_mask).sum()
        
        if n_bull < 500 or n_bear < 500:
            continue
        
        # 牛市系统中earn的IC
        bull_earn_ic = df.loc[bull_mask, 'earn_momentum'].corr(df.loc[bull_mask, 'fwd_12m'])
        bear_earn_ic = df.loc[~bull_mask, 'earn_momentum'].corr(df.loc[~bull_mask, 'fwd_12m'])
        
        # 切换是否有益的判断：bull中earn IC < 0 且 bear中earn IC > 0
        switch_benefit = bear_earn_ic - bull_earn_ic
        
        best_configs.append({
            'method': method,
            'threshold': threshold,
            'n_bull': n_bull, 'n_bear': n_bear,
            'bull_earn_ic': bull_earn_ic,
            'bear_earn_ic': bear_earn_ic,
            'switch_benefit': switch_benefit,
        })

best_configs_df = pd.DataFrame(best_configs).sort_values('switch_benefit', ascending=False)

print(f"\n{'方法':<15s} {'阈值':>8s} {'N_bull':>7s} {'N_bear':>7s} {'bull_earn':>10s} {'bear_earn':>10s} {'切换收益':>10s}")
print("-"*80)

for _, row in best_configs_df.head(15).iterrows():
    print(f"{row['method']:<15s} {row['threshold']:>+8.2f} {row['n_bull']:>7d} {row['n_bear']:>7d} "
          f"{row['bull_earn_ic']:>+10.4f} {row['bear_earn_ic']:>+10.4f} {row['switch_benefit']:>+10.4f}")

print("\n" + "="*80)
print("实验完成")
