#!/usr/bin/env python3
"""
DCA行业轮动分析：前瞻性信号 vs 后向性信号
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 加载所有行业指数
sectors = {
    '000987': '材料', '000988': '工业', '000990': '消费',
    '000991': '医药', '000992': '金融', '000993': '信息',
    'sh000015': '红利', 'sh000016': '50大', 'sh000300': '300',
    'sh000852': '1000', 'sh000905': '500', 'sz399673': '创业50',
    'hk_HSI': '恒生', 'hk_HSTECH': '恒科', 'us_INX': '标普', 'us_NDX': '纳指'
}

print("=" * 100)
print("第一部分：行业PE估值分位数 vs 未来收益（前瞻性信号验证）")
print("=" * 100)

# 只有6个行业有PE数据
pe_sectors = {
    '000987': '材料', '000988': '工业', '000990': '消费',
    '000991': '医药', '000992': '金融', '000993': '信息'
}

pe_data = {}
for code, name in pe_sectors.items():
    df = pd.read_csv(f'cache/csi_{code}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    pe_data[code] = df

# 测试多个时间点
test_dates = [
    '2022-01-01', '2022-04-01', '2022-07-01', '2022-10-01',
    '2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01',
    '2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'
]

results = []

for test_date_str in test_dates:
    test_date = pd.to_datetime(test_date_str)
    
    for code, name in pe_sectors.items():
        df = pe_data[code]
        
        # 找到测试日期的索引
        mask = df['date'] <= test_date
        if mask.sum() == 0:
            continue
        idx = df[mask].index[-1]
        
        # 需要历史数据计算PE分位
        if idx < 250:
            continue
        
        # 需要未来数据计算收益
        if idx + 250 >= len(df):
            continue
        
        # 计算PE分位数（过去5年）
        pe_history = df.iloc[max(0, idx-1250):idx+1]['pe'].dropna()
        current_pe = df.iloc[idx]['pe']
        
        if pd.isna(current_pe) or len(pe_history) < 100:
            continue
        
        pe_percentile = (pe_history < current_pe).mean()
        
        # 计算未来1年收益
        future_return = df.iloc[idx + 250]['close'] / df.iloc[idx]['close'] - 1
        
        results.append({
            'date': test_date_str,
            'sector': name,
            'code': code,
            'pe_percentile': pe_percentile,
            'current_pe': current_pe,
            'future_return': future_return
        })

df_results = pd.DataFrame(results)

print(f"\n样本数: {len(df_results)}")
print(f"时间范围: {df_results['date'].min()} 到 {df_results['date'].max()}")
print(f"行业数: {df_results['sector'].nunique()}")

# 按PE分位分组
df_results['pe_bucket'] = pd.cut(
    df_results['pe_percentile'], 
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['<20%极低', '20-40%低', '40-60%中', '60-80%高', '>80%极高']
)

print("\n" + "=" * 100)
print("PE估值分位 vs 未来1年平均收益")
print("=" * 100)
print(f"{'PE分位':<12} {'样本数':<8} {'平均收益':<12} {'胜率':<10} {'中位数':<10} {'信号强度'}")
print("-" * 100)

for bucket in ['<20%极低', '20-40%低', '40-60%中', '60-80%高', '>80%极高']:
    subset = df_results[df_results['pe_bucket'] == bucket]
    if len(subset) > 0:
        avg_return = subset['future_return'].mean()
        win_rate = (subset['future_return'] > 0).mean()
        median_return = subset['future_return'].median()
        
        # 信号强度评估
        if bucket == '<20%极低' and avg_return > 0.10:
            strength = "★★★ 强"
        elif bucket == '<20%极低' and avg_return > 0:
            strength = "★★ 中"
        elif bucket == '>80%极高' and avg_return < 0:
            strength = "★★★ 强"
        elif bucket == '>80%极高' and avg_return < 0.05:
            strength = "★★ 中"
        else:
            strength = "★ 弱"
        
        print(f"{bucket:<12} {len(subset):<8} {avg_return:>+10.1%} {win_rate:>8.1%} {median_return:>+8.1%} {strength}")

# 相关性分析
corr = df_results['pe_percentile'].corr(df_results['future_return'])
print(f"\nPE分位与未来收益相关性: {corr:.3f}")
if corr < -0.2:
    print("✓ 显著负相关：低估值→高收益，前瞻性信号有效！")
elif corr < 0:
    print("△ 弱负相关：有一定预测能力")
else:
    print("✗ 无相关性或正相关：PE分位预测能力弱")

# 策略模拟
print("\n" + "=" * 100)
print("策略模拟：基于PE分位的行业轮动")
print("=" * 100)

# 策略：买入PE<30%的行业，卖出PE>70%的行业
low_pe_threshold = 0.30
high_pe_threshold = 0.70

low_pe_signals = df_results[df_results['pe_percentile'] < low_pe_threshold]
high_pe_signals = df_results[df_results['pe_percentile'] > high_pe_threshold]

print(f"\n买入信号 (PE < {low_pe_threshold:.0%}):")
print(f"  信号数: {len(low_pe_signals)}")
print(f"  平均未来收益: {low_pe_signals['future_return'].mean():+.1%}")
print(f"  胜率: {(low_pe_signals['future_return'] > 0).mean():.1%}")
print(f"  最佳案例: {low_pe_signals.loc[low_pe_signals['future_return'].idxmax(), 'sector']} "
      f"+{low_pe_signals['future_return'].max():.1%}")
print(f"  最差案例: {low_pe_signals.loc[low_pe_signals['future_return'].idxmin(), 'sector']} "
      f"{low_pe_signals['future_return'].min():.1%}")

print(f"\n卖出信号 (PE > {high_pe_threshold:.0%}):")
print(f"  信号数: {len(high_pe_signals)}")
print(f"  平均未来收益: {high_pe_signals['future_return'].mean():+.1%}")
print(f"  胜率: {(high_pe_signals['future_return'] > 0).mean():.1%}")

print("\n" + "=" * 100)
print("第二部分：动量信号验证（过去6个月涨幅 vs 未来6个月收益）")
print("=" * 100)

# 加载所有行业指数（不需要PE数据）
all_sector_data = {}
for code, name in sectors.items():
    try:
        df = pd.read_csv(f'cache/idx_{code}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        all_sector_data[code] = {'name': name, 'data': df}
    except:
        pass

momentum_results = []

for test_date_str in test_dates:
    test_date = pd.to_datetime(test_date_str)
    
    for code, info in all_sector_data.items():
        name = info['name']
        df = info['data']
        
        # 找到测试日期的索引
        mask = df['date'] <= test_date
        if mask.sum() == 0:
            continue
        idx = df[mask].index[-1]
        
        # 需要过去180天和未来180天
        if idx < 180 or idx + 180 >= len(df):
            continue
        
        # 计算过去6个月动量
        momentum_6m = df.iloc[idx]['close'] / df.iloc[idx - 180]['close'] - 1
        
        # 计算未来6个月收益
        future_6m = df.iloc[idx + 180]['close'] / df.iloc[idx]['close'] - 1
        
        momentum_results.append({
            'date': test_date_str,
            'sector': name,
            'code': code,
            'momentum_6m': momentum_6m,
            'future_6m': future_6m
        })

df_momentum = pd.DataFrame(momentum_results)

# 按动量分组
df_momentum['mom_bucket'] = pd.cut(
    df_momentum['momentum_6m'],
    bins=[-np.inf, -0.2, -0.1, 0, 0.1, 0.2, np.inf],
    labels=['<-20%', '-20~-10%', '-10~0%', '0~10%', '10~20%', '>20%']
)

print(f"\n样本数: {len(df_momentum)}")
print(f"\n{'动量区间':<12} {'样本数':<8} {'未来收益':<12} {'胜率':<10} {'信号'}")
print("-" * 70)

for bucket in ['<-20%', '-20~-10%', '-10~0%', '0~10%', '10~20%', '>20%']:
    subset = df_momentum[df_momentum['mom_bucket'] == bucket]
    if len(subset) > 0:
        avg_future = subset['future_6m'].mean()
        win_rate = (subset['future_6m'] > 0).mean()
        
        # 动量信号评估
        if bucket in ['10~20%', '>20%'] and avg_future > 0.05:
            signal = "✓ 动量延续"
        elif bucket in ['<-20%', '-20~-10%'] and avg_future > 0:
            signal = "✓ 反转效应"
        else:
            signal = "△ 信号弱"
        
        print(f"{bucket:<12} {len(subset):<8} {avg_future:>+10.1%} {win_rate:>8.1%} {signal}")

mom_corr = df_momentum['momentum_6m'].corr(df_momentum['future_6m'])
print(f"\n动量与未来收益相关性: {mom_corr:.3f}")
if mom_corr > 0.2:
    print("✓ 正相关：动量延续效应，强者恒强")
elif mom_corr < -0.2:
    print("✓ 负相关：反转效应，跌多了会涨")
else:
    print("△ 相关性弱：动量信号不稳定")

print("\n" + "=" * 100)
print("第三部分：均值回归效应（低估值+高动量组合）")
print("=" * 100)

# 合并PE和动量数据
df_combined = df_results.merge(
    df_momentum[['sector', 'date', 'momentum_6m']],
    on=['sector', 'date'],
    how='inner'
)

if len(df_combined) > 0:
    print(f"\n组合信号样本数: {len(df_combined)}")
    
    # 策略：低估值(PE<40%) + 高动量(涨幅>0%)
    best_combo = df_combined[
        (df_combined['pe_percentile'] < 0.4) & 
        (df_combined['momentum_6m'] > 0)
    ]
    
    # 对照：高估值(PE>60%) + 低动量(涨幅<0%)
    worst_combo = df_combined[
        (df_combined['pe_percentile'] > 0.6) & 
        (df_combined['momentum_6m'] < 0)
    ]
    
    print(f"\n最佳组合（低估值+高动量）:")
    print(f"  样本数: {len(best_combo)}")
    if len(best_combo) > 0:
        print(f"  平均未来收益: {best_combo['future_return'].mean():+.1%}")
        print(f"  胜率: {(best_combo['future_return'] > 0).mean():.1%}")
    
    print(f"\n最差组合（高估值+低动量）:")
    print(f"  样本数: {len(worst_combo)}")
    if len(worst_combo) > 0:
        print(f"  平均未来收益: {worst_combo['future_return'].mean():+.1%}")
        print(f"  胜率: {(worst_combo['future_return'] > 0).mean():.1%}")
    
    if len(best_combo) > 0 and len(worst_combo) > 0:
        spread = best_combo['future_return'].mean() - worst_combo['future_return'].mean()
        print(f"\n收益差: {spread:+.1%}")
        if spread > 0.15:
            print("✓ 组合信号强：低估值+高动量显著优于高估值+低动量")
        elif spread > 0.05:
            print("△ 组合信号中等：有一定区分度")
        else:
            print("✗ 组合信号弱：区分度不够")

print("\n" + "=" * 100)
print("结论")
print("=" * 100)
print("""
1. PE估值分位数：
   - 如果低估值行业未来收益显著高于高估值行业 → 前瞻性信号有效
   - 适合用于行业选择（买低估行业，避开高估行业）

2. 动量信号：
   - 如果正相关 → 动量延续，追涨有效
   - 如果负相关 → 反转效应，抄底有效
   - 如果无相关 → 动量信号不可靠

3. 组合信号（低估值+高动量）：
   - 如果显著优于其他组合 → 这是最优的DCA策略
   - 本质是：买"便宜且在涨"的行业

4. 对DCA策略的启示：
   - 传统方法（看基金历史）是自下而上
   - 你的思路（看行业前景）是自上而下
   - 最佳策略可能是：自上而下选行业 + 自下而上选基金
""")
