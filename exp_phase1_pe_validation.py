#!/usr/bin/env python3
"""
阶段1：PE估值信号的大规模验证
用所有可用数据（8个行业，2020-2024）验证PE与未来收益的关系
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('cache')

def load_all_sector_data():
    """加载所有有PE数据的行业指数"""
    sectors = {}
    
    # A股6个行业
    a_share_sectors = {
        'csi_000987': 'A股新材料',
        'csi_000988': 'A股工业',
        'csi_000990': 'A股消费',
        'csi_000991': 'A股医药',
        'csi_000992': 'A股金融',
        'csi_000993': 'A股信息'
    }
    
    # 港股2个行业
    hk_sectors = {
        'csi_H30090': '港股消费',
        'csi_H30533': '港股科技'
    }
    
    all_sectors = {**a_share_sectors, **hk_sectors}
    
    for code, name in all_sectors.items():
        filepath = CACHE_DIR / f'{code}.csv'
        if not filepath.exists():
            print(f"警告: {filepath} 不存在，跳过")
            continue
            
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # 从2020年开始（数据较完整）
        df = df.loc['2020-01-01':]
        
        if len(df) > 0:
            sectors[name] = df
            print(f"加载 {name}: {len(df)} 条数据, {df.index[0].date()} ~ {df.index[-1].date()}")
    
    return sectors

def calculate_pe_percentile(pe_series, window_days=756):
    """计算PE的历史百分位（过去3年窗口）"""
    return pe_series.rolling(window=window_days, min_periods=252).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
        raw=False
    )

def test_pe_signal_comprehensive(sectors):
    """
    综合测试PE估值信号
    1. 按PE分位分组，计算未来收益
    2. 计算PE与未来收益的相关性
    3. 分年度分析（检查稳定性）
    """
    print("="*80)
    print("阶段1：PE估值信号综合验证")
    print("="*80)
    
    all_results = []
    
    for name, df in sectors.items():
        print(f"\n{'='*80}")
        print(f"分析行业: {name}")
        print(f"{'='*80}")
        
        # 计算PE百分位（过去3年窗口）
        df['pe_percentile'] = calculate_pe_percentile(df['pe'], window_days=756)
        
        # 计算未来收益（多个时间窗口）
        df['future_6m'] = df['close'].pct_change(126).shift(-126)
        df['future_1y'] = df['close'].pct_change(252).shift(-252)
        df['future_2y'] = df['close'].pct_change(504).shift(-504)
        
        # 添加年份列
        df['year'] = df.index.year
        
        # 删除NaN
        df_clean = df.dropna(subset=['pe_percentile', 'future_1y'])
        
        if len(df_clean) < 100:
            print(f"数据不足，跳过")
            continue
        
        print(f"有效数据点: {len(df_clean)}")
        print(f"时间范围: {df_clean.index[0].date()} ~ {df_clean.index[-1].date()}")
        
        # 1. 按PE分位分组
        df_clean['pe_bucket'] = pd.cut(
            df_clean['pe_percentile'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['<20%', '20-40%', '40-60%', '60-80%', '>80%']
        )
        
        print(f"\n--- 按PE分位分组的未来1年收益 ---")
        bucket_stats = df_clean.groupby('pe_bucket', observed=True)['future_1y'].agg([
            'mean', 'std', 'count', 
            lambda x: (x > 0).mean()  # 胜率
        ]).rename(columns={'mean': '平均收益', 'std': '标准差', 'count': '样本数', 'lambda_0': '胜率'})
        
        print(bucket_stats)
        
        # 2. 计算相关性
        corr_6m = df_clean['pe_percentile'].corr(df_clean['future_6m'])
        corr_1y = df_clean['pe_percentile'].corr(df_clean['future_1y'])
        corr_2y = df_clean.dropna(subset=['pe_percentile', 'future_2y'])['pe_percentile'].corr(
            df_clean.dropna(subset=['pe_percentile', 'future_2y'])['future_2y']
        )
        
        print(f"\n--- PE分位与未来收益相关性 ---")
        print(f"未来6个月: {corr_6m:+.4f}")
        print(f"未来1年:   {corr_1y:+.4f}")
        print(f"未来2年:   {corr_2y:+.4f}")
        
        # 3. 分年度分析
        print(f"\n--- 分年度分析（未来1年收益）---")
        yearly_stats = df_clean.groupby('year').agg({
            'pe_percentile': 'count',
            'future_1y': lambda x: df_clean.loc[x.index, 'pe_percentile'].corr(x)
        }).rename(columns={'pe_percentile': '样本数', 'future_1y': '相关性'})
        
        print(yearly_stats)
        
        # 4. 低PE vs 高PE对比
        low_pe = df_clean[df_clean['pe_percentile'] < 0.3]
        high_pe = df_clean[df_clean['pe_percentile'] > 0.7]
        
        print(f"\n--- 低PE vs 高PE对比 ---")
        print(f"低PE (<30%分位):")
        print(f"  样本数: {len(low_pe)}")
        print(f"  平均PE: {low_pe['pe'].mean():.2f}")
        print(f"  未来1年收益: {low_pe['future_1y'].mean()*100:.2f}%")
        print(f"  胜率: {(low_pe['future_1y'] > 0).mean()*100:.1f}%")
        
        print(f"高PE (>70%分位):")
        print(f"  样本数: {len(high_pe)}")
        print(f"  平均PE: {high_pe['pe'].mean():.2f}")
        print(f"  未来1年收益: {high_pe['future_1y'].mean()*100:.2f}%")
        print(f"  胜率: {(high_pe['future_1y'] > 0).mean()*100:.1f}%")
        
        # 保存结果
        all_results.append({
            'sector': name,
            'data_points': len(df_clean),
            'corr_6m': corr_6m,
            'corr_1y': corr_1y,
            'corr_2y': corr_2y,
            'low_pe_return': low_pe['future_1y'].mean(),
            'high_pe_return': high_pe['future_1y'].mean(),
            'low_pe_winrate': (low_pe['future_1y'] > 0).mean(),
            'high_pe_winrate': (high_pe['future_1y'] > 0).mean(),
        })
    
    # 汇总分析
    print("\n" + "="*80)
    print("汇总分析：所有行业的PE信号")
    print("="*80)
    
    df_summary = pd.DataFrame(all_results)
    
    print(f"\n行业数量: {len(df_summary)}")
    print(f"总数据点: {df_summary['data_points'].sum()}")
    
    print(f"\n--- 平均相关性 ---")
    print(f"未来6个月: {df_summary['corr_6m'].mean():+.4f} (std={df_summary['corr_6m'].std():.4f})")
    print(f"未来1年:   {df_summary['corr_1y'].mean():+.4f} (std={df_summary['corr_1y'].std():.4f})")
    print(f"未来2年:   {df_summary['corr_2y'].mean():+.4f} (std={df_summary['corr_2y'].std():.4f})")
    
    print(f"\n--- 低PE vs 高PE ---")
    avg_low_return = df_summary['low_pe_return'].mean()
    avg_high_return = df_summary['high_pe_return'].mean()
    spread = avg_high_return - avg_low_return
    
    print(f"低PE平均收益: {avg_low_return*100:.2f}%")
    print(f"高PE平均收益: {avg_high_return*100:.2f}%")
    print(f"收益差距: {spread*100:.2f}%")
    
    print(f"\n--- 结论 ---")
    if spread > 0.10:
        print("✓ 强证据：高PE > 低PE（成长 > 价值）")
        print("  → F_value应该反转方向")
    elif spread > 0.05:
        print("△ 中等证据：高PE略优于低PE")
        print("  → F_value可能需要调整")
    elif spread < -0.10:
        print("✓ 强证据：低PE > 高PE（价值 > 成长）")
        print("  → F_value当前设计正确")
    else:
        print("✗ 无显著差异：PE无法预测未来收益")
        print("  → F_value可能无效")
    
    # 分市场类型分析
    print(f"\n--- 分市场类型分析 ---")
    a_share = df_summary[df_summary['sector'].str.contains('A股')]
    hk = df_summary[df_summary['sector'].str.contains('港股')]
    
    print(f"A股（{len(a_share)}个行业）:")
    print(f"  平均相关性(1年): {a_share['corr_1y'].mean():+.4f}")
    print(f"  高PE-低PE收益差: {(a_share['high_pe_return'].mean() - a_share['low_pe_return'].mean())*100:.2f}%")
    
    print(f"港股（{len(hk)}个行业）:")
    print(f"  平均相关性(1年): {hk['corr_1y'].mean():+.4f}")
    print(f"  高PE-低PE收益差: {(hk['high_pe_return'].mean() - hk['low_pe_return'].mean())*100:.2f}%")
    
    return df_summary

def main():
    print("="*80)
    print("PE估值信号大规模验证实验")
    print("目标：验证F_value因子的设计是否正确")
    print("="*80)
    
    sectors = load_all_sector_data()
    print(f"\n成功加载 {len(sectors)} 个行业")
    
    df_summary = test_pe_signal_comprehensive(sectors)
    
    print("\n" + "="*80)
    print("阶段1完成")
    print("="*80)
    
    # 保存结果
    import os
    os.makedirs('output', exist_ok=True)
    df_summary.to_csv('output/phase1_pe_validation.csv', index=False)
    print(f"结果已保存到 output/phase1_pe_validation.csv")

if __name__ == '__main__':
    main()
