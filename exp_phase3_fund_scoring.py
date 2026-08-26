#!/usr/bin/env python3
"""
阶段3：用真实基金数据验证打分系统改进
1. 分析当前F_value的有效性
2. 测试不同的F_value设计
3. 测试加入行业因子后的改进
4. 用基金未来收益验证
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings
import os
warnings.filterwarnings('ignore')

def load_scan_data():
    """加载最新评分数据"""
    df = pd.read_csv('output/scan_20260810.csv')
    
    # 解析RBSA数据
    def parse_rbsa(rbsa_str):
        try:
            if pd.isna(rbsa_str) or rbsa_str == '':
                return {}
            return json.loads(rbsa_str.replace('""', '"').strip('"'))
        except:
            return {}
    
    df['rbsa_dict'] = df['rbsa'].apply(parse_rbsa)
    
    # 提取主要行业配置
    def get_top_sector(rbsa_dict):
        # 排除大盘指数，找行业配置
        sector_indices = ['全指材料', '全指工业', '全指消费', '全指医药', '全指金融', '全指信息']
        sector_weights = {k: v for k, v in rbsa_dict.items() if k in sector_indices}
        if not sector_weights:
            return '未知'
        return max(sector_weights, key=sector_weights.get)
    
    def get_sector_weight(rbsa_dict, sector):
        return rbsa_dict.get(sector, 0)
    
    df['top_sector'] = df['rbsa_dict'].apply(get_top_sector)
    df['region'] = df['region'].fillna('未知')
    
    print(f"加载基金数据: {len(df)} 只基金")
    print(f"市场分布: {df['region'].value_counts().to_dict()}")
    print(f"行业分布: {df['top_sector'].value_counts().to_dict()}")
    
    return df

def load_nav_data(codes):
    """加载基金净值数据"""
    navs = {}
    for code in codes:
        filepath = f'cache/nav_{code}.csv'
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            df = df.set_index('date').sort_index()
            if len(df) > 200:
                navs[str(code)] = df
        except:
            continue
    return navs

def calculate_forward_returns(navs, forward_days=252):
    """计算基金的前瞻收益"""
    forward_returns = {}
    
    for code, nav_df in navs.items():
        if len(nav_df) < forward_days + 60:
            continue
        
        # 计算多个时间点的未来收益
        dates = nav_df.index
        nav = nav_df['nav']
        
        # 从最近日期往前推
        for months_back in [0, 3, 6, 12, 18, 24]:
            if months_back * 21 >= len(nav_df):
                continue
            
            start_idx = len(nav_df) - 1 - months_back * 21
            
            if start_idx < forward_days:
                continue
            
            start_date = dates[start_idx]
            future_date = dates[min(start_idx + forward_days, len(dates) - 1)]
            
            start_nav = nav.iloc[start_idx]
            end_nav = nav.iloc[start_idx + forward_days] if start_idx + forward_days < len(nav) else nav.iloc[-1]
            
            ret = end_nav / start_nav - 1
            
            key = (code, start_date.strftime('%Y-%m-%d'))
            forward_returns[key] = ret
    
    return forward_returns

def analyze_f_value_effectiveness(df):
    """
    分析当前F_value的有效性
    F_value基于val_pct（PE百分位）
    """
    print("\n" + "="*80)
    print("阶段3-A：当前F_value有效性分析")
    print("="*80)
    
    # 只看有F_value数据的基金
    df_valid = df[df['F_value'].notna() & (df['F_value'] > 0)].copy()
    
    print(f"\n有效基金数: {len(df_valid)}")
    
    # 按F_value分组
    df_valid['f_value_bucket'] = pd.cut(
        df_valid['F_value'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['0-20', '20-40', '40-60', '60-80', '80-100']
    )
    
    print(f"\n--- 按F_value分组的基金统计 ---")
    bucket_stats = df_valid.groupby('f_value_bucket', observed=True).agg({
        'code': 'count',
        'S_total': 'mean',
        'val_pct': 'mean',
    }).rename(columns={'code': '基金数', 'S_total': '平均总分', 'val_pct': '平均PE分位'})
    
    print(bucket_stats)
    
    # 按val_pct分组
    df_valid['val_pct_bucket'] = pd.cut(
        df_valid['val_pct'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['<20%', '20-40%', '40-60%', '60-80%', '>80%']
    )
    
    print(f"\n--- 按PE分位分组的基金统计 ---")
    pct_stats = df_valid.groupby('val_pct_bucket', observed=True).agg({
        'code': 'count',
        'F_value': 'mean',
        'S_total': 'mean',
    }).rename(columns={'code': '基金数', 'F_value': '平均估值分', 'S_total': '平均总分'})
    
    print(pct_stats)
    
    # 分市场分析
    print(f"\n--- 分市场的PE分位分布 ---")
    for region in ['A股', '港股', '海外']:
        region_data = df_valid[df_valid['region'] == region]
        if len(region_data) > 0:
            print(f"\n{region} ({len(region_data)}只基金):")
            print(f"  平均PE分位: {region_data['val_pct'].mean():.3f}")
            print(f"  平均F_value: {region_data['F_value'].mean():.1f}")
            print(f"  PE分位中位数: {region_data['val_pct'].median():.3f}")
    
    # 分行业分析
    print(f"\n--- 分行业的PE分位分布 ---")
    sector_stats = df_valid.groupby('top_sector').agg({
        'code': 'count',
        'val_pct': ['mean', 'median'],
        'F_value': 'mean',
    }).round(3)
    
    print(sector_stats)
    
    return df_valid

def test_alternative_f_value(df, navs):
    """
    测试不同的F_value设计
    用实际基金收益来验证哪个设计更好
    """
    print("\n" + "="*80)
    print("阶段3-B：不同F_value设计对比")
    print("="*80)
    
    # 计算基金的未来收益
    print("计算基金未来收益...")
    codes = df['code'].astype(str).tolist()
    fund_navs = load_nav_data(codes[:2000])  # 限制数量以节省时间
    print(f"加载了 {len(fund_navs)} 只基金的净值数据")
    
    # 计算未来1年收益（从最近日期）
    forward_returns = {}
    for code, nav_df in fund_navs.items():
        if len(nav_df) > 252 + 60:
            nav = nav_df['nav']
            # 用6个月前的数据，这样有1年的未来收益
            idx_6m_ago = len(nav_df) - 126  # 6个月前
            if idx_6m_ago > 252:
                start = nav.iloc[idx_6m_ago - 252]  # 1年前（从6个月前看）
                end = nav.iloc[idx_6m_ago]
                ret_1y = end / start - 1
                forward_returns[code] = ret_1y
    
    print(f"计算了 {len(forward_returns)} 只基金的未来1年收益")
    
    # 合并数据
    df_analysis = df.copy()
    df_analysis['code'] = df_analysis['code'].astype(str)
    df_analysis['forward_1y'] = df_analysis['code'].map(forward_returns)
    df_valid = df_analysis[df_analysis['forward_1y'].notna()].copy()
    
    print(f"有未来收益数据的基金: {len(df_valid)}")
    
    if len(df_valid) < 100:
        print("数据不足，跳过此分析")
        return None
    
    # 测试不同的F_value设计
    designs = {}
    
    # 设计1：当前设计（低PE=高分）
    designs['当前（低PE=高分）'] = df_valid['F_value']
    
    # 设计2：反转设计（高PE=高分）
    def reverse_f_value(val_pct):
        if pd.isna(val_pct):
            return np.nan
        if val_pct >= 0.7:
            return 99.0
        elif val_pct >= 0.5:
            return 69.0
        elif val_pct >= 0.3:
            return 40.0
        else:
            return 20.0
    
    designs['反转（高PE=高分）'] = df_valid['val_pct'].apply(reverse_f_value)
    
    # 设计3：中性设计（中间PE=高分）
    def neutral_f_value(val_pct):
        if pd.isna(val_pct):
            return np.nan
        # 50%附近最高分
        distance = abs(val_pct - 0.5)
        return max(0, 100 - distance * 200)
    
    designs['中性（中间PE=高分）'] = df_valid['val_pct'].apply(neutral_f_value)
    
    # 设计4：分市场设计（A股高PE=高分，港股低PE=高分）
    def market_adaptive_f_value(row):
        val_pct = row['val_pct']
        region = row['region']
        if pd.isna(val_pct):
            return np.nan
        
        if region == 'A股':
            # A股：高PE给高分
            if val_pct >= 0.7:
                return 95.0
            elif val_pct >= 0.5:
                return 70.0
            elif val_pct >= 0.3:
                return 40.0
            else:
                return 20.0
        elif region == '港股':
            # 港股：低PE给高分
            if val_pct <= 0.3:
                return 95.0
            elif val_pct <= 0.5:
                return 70.0
            elif val_pct <= 0.7:
                return 40.0
            else:
                return 20.0
        else:
            # 海外：中性
            distance = abs(val_pct - 0.5)
            return max(0, 100 - distance * 200)
    
    designs['分市场自适应'] = df_valid.apply(market_adaptive_f_value, axis=1)
    
    # 设计5：无F_value（只用F_alpha和F_momentum）
    designs['无F_value'] = pd.Series(50, index=df_valid.index)  # 中性值
    
    # 评估每个设计
    print(f"\n--- 不同F_value设计与未来收益的关系 ---")
    print(f"{'设计':<25s} {'IC':>8s} {'RankIC':>8s} {'t统计量':>8s} {'显著性':>8s}")
    print("-" * 70)
    
    ic_results = {}
    
    for design_name, design_scores in designs.items():
        valid_mask = design_scores.notna() & df_valid['forward_1y'].notna()
        
        if valid_mask.sum() < 50:
            print(f"{design_name:<25s} {'数据不足':>8s}")
            continue
        
        scores = design_scores[valid_mask]
        returns = df_valid.loc[valid_mask, 'forward_1y']
        
        # IC (Pearson correlation)
        ic = scores.corr(returns)
        
        # Rank IC (Spearman correlation)
        rank_ic = stats.spearmanr(scores, returns)[0]
        
        # t-statistic
        n = len(scores)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        
        # Significance
        significant = '***' if abs(t_stat) > 2.576 else ('**' if abs(t_stat) > 1.96 else ('*' if abs(t_stat) > 1.645 else ''))
        
        print(f"{design_name:<25s} {ic:>+8.4f} {rank_ic:>+8.4f} {t_stat:>8.2f} {significant:>8s}")
        
        ic_results[design_name] = {
            'IC': ic,
            'RankIC': rank_ic,
            't_stat': t_stat,
            'n_samples': n,
        }
    
    # 按IC排名
    print(f"\n--- 设计排名（按IC）---")
    ranked = sorted(ic_results.items(), key=lambda x: x[1]['IC'], reverse=True)
    for i, (name, metrics) in enumerate(ranked):
        print(f"  {i+1}. {name}: IC={metrics['IC']:+.4f}, RankIC={metrics['RankIC']:+.4f}")
    
    return ic_results

def test_industry_factor(df, navs):
    """
    测试加入行业因子的效果
    """
    print("\n" + "="*80)
    print("阶段3-C：行业因子测试")
    print("="*80)
    
    # 计算基金未来收益
    codes = df['code'].astype(str).tolist()
    fund_navs = load_nav_data(codes[:2000])
    
    forward_returns = {}
    for code, nav_df in fund_navs.items():
        if len(nav_df) > 252 + 60:
            nav = nav_df['nav']
            idx_6m_ago = len(nav_df) - 126
            if idx_6m_ago > 252:
                start = nav.iloc[idx_6m_ago - 252]
                end = nav.iloc[idx_6m_ago]
                ret_1y = end / start - 1
                forward_returns[code] = ret_1y
    
    df_analysis = df.copy()
    df_analysis['code'] = df_analysis['code'].astype(str)
    df_analysis['forward_1y'] = df_analysis['code'].map(forward_returns)
    df_valid = df_analysis[df_analysis['forward_1y'].notna()].copy()
    
    print(f"有未来收益数据的基金: {len(df_valid)}")
    
    if len(df_valid) < 100:
        print("数据不足")
        return None
    
    # 计算行业因子：同行业基金的平均未来收益
    sector_future = df_valid.groupby('top_sector')['forward_1y'].mean()
    df_valid['sector_avg_return'] = df_valid['top_sector'].map(sector_future)
    
    # 行业动量因子：用RBSA中提取的行业配置加权
    # 简化版：只看top_sector
    
    print(f"\n--- 行业平均收益排名 ---")
    print(sector_future.sort_values(ascending=False))
    
    # 测试：如果按照行业平均收益调整分数
    # 高收益行业的基金加分，低收益行业的基金减分
    
    # 基线：当前总分
    baseline_scores = df_valid['S_total']
    baseline_ic = baseline_scores.corr(df_valid['forward_1y'])
    baseline_rank_ic = stats.spearmanr(baseline_scores, df_valid['forward_1y'])[0]
    
    print(f"\n--- 基线打分系统 ---")
    print(f"IC: {baseline_ic:+.4f}")
    print(f"RankIC: {baseline_rank_ic:+.4f}")
    
    # 变体1：加入行业调整
    sector_adjustment = df_valid['top_sector'].map(sector_future)
    # 标准化到0-100
    sector_adj_norm = (sector_adjustment - sector_adjustment.min()) / (sector_adjustment.max() - sector_adjustment.min() + 1e-10) * 100
    
    # 混合：80%原始分 + 20%行业调整
    adjusted_scores = 0.8 * baseline_scores + 0.2 * sector_adj_norm
    adj_ic = adjusted_scores.corr(df_valid['forward_1y'])
    adj_rank_ic = stats.spearmanr(adjusted_scores, df_valid['forward_1y'])[0]
    
    print(f"\n--- 加入行业调整（权重20%）---")
    print(f"IC: {adj_ic:+.4f} (变化: {adj_ic - baseline_ic:+.4f})")
    print(f"RankIC: {adj_rank_ic:+.4f} (变化: {adj_rank_ic - baseline_rank_ic:+.4f})")
    
    # 变体2：只用行业因子
    sector_only_ic = sector_adj_norm.corr(df_valid['forward_1y'])
    sector_only_rank_ic = stats.spearmanr(sector_adj_norm, df_valid['forward_1y'])[0]
    
    print(f"\n--- 纯行业因子 ---")
    print(f"IC: {sector_only_ic:+.4f}")
    print(f"RankIC: {sector_only_rank_ic:+.4f}")
    
    # 变体3：不同权重的行业调整
    print(f"\n--- 不同行业权重的效果 ---")
    print(f"{'行业权重':>10s} {'IC':>8s} {'RankIC':>8s} {'IC变化':>8s}")
    print("-" * 40)
    
    for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        mixed = (1 - w) * baseline_scores + w * sector_adj_norm
        ic = mixed.corr(df_valid['forward_1y'])
        rank_ic = stats.spearmanr(mixed, df_valid['forward_1y'])[0]
        print(f"{w*100:>9.0f}% {ic:>+8.4f} {rank_ic:>+8.4f} {ic - baseline_ic:>+8.4f}")
    
    return {
        'baseline_ic': baseline_ic,
        'baseline_rank_ic': baseline_rank_ic,
    }

def test_combined_improvements(df, navs):
    """
    测试综合改进方案
    """
    print("\n" + "="*80)
    print("阶段3-D：综合改进方案测试")
    print("="*80)
    
    # 计算基金未来收益
    codes = df['code'].astype(str).tolist()
    fund_navs = load_nav_data(codes[:2000])
    
    forward_returns = {}
    forward_returns_6m = {}
    
    for code, nav_df in fund_navs.items():
        if len(nav_df) > 252:
            nav = nav_df['nav']
            
            # 未来1年收益（从6个月前开始）
            idx_6m = len(nav_df) - 126
            if idx_6m > 252:
                start_1y = nav.iloc[idx_6m - 252]
                end_1y = nav.iloc[idx_6m]
                forward_returns[code] = end_1y / start_1y - 1
            
            # 未来6个月收益（从当前开始）
            idx_now = len(nav_df) - 1
            if idx_now > 126:
                start_6m = nav.iloc[idx_now - 126]
                end_6m = nav.iloc[idx_now]
                forward_returns_6m[code] = end_6m / start_6m - 1
    
    df_analysis = df.copy()
    df_analysis['code'] = df_analysis['code'].astype(str)
    df_analysis['fwd_1y'] = df_analysis['code'].map(forward_returns)
    df_analysis['fwd_6m'] = df_analysis['code'].map(forward_returns_6m)
    
    # 用未来1年收益评估
    df_valid = df_analysis[df_analysis['fwd_1y'].notna()].copy()
    
    print(f"评估样本: {len(df_valid)} 只基金")
    
    if len(df_valid) < 100:
        print("数据不足")
        return
    
    # 策略对比
    strategies = {}
    
    # 策略1：当前系统（买入S_total > 70的基金）
    strategies['当前系统 (S>70买入)'] = df_valid[df_valid['S_total'] > 70]['fwd_1y']
    
    # 策略2：调整阈值（S>60买入）
    strategies['调整阈值 (S>60买入)'] = df_valid[df_valid['S_total'] > 60]['fwd_1y']
    
    # 策略3：只买A股高PE基金
    a_share_high_pe = df_valid[(df_valid['region'] == 'A股') & (df_valid['val_pct'] > 0.6)]
    strategies['A股高PE (val_pct>0.6)'] = a_share_high_pe['fwd_1y']
    
    # 策略4：只买F_alpha高的基金（>80）
    high_alpha = df_valid[df_valid['F_alpha'] > 80]
    strategies['高Alpha (F_alpha>80)'] = high_alpha['fwd_1y']
    
    # 策略5：只买F_momentum高的基金（>80）
    high_momentum = df_valid[df_valid['F_momentum'] > 80]
    strategies['高动量 (F_momentum>80)'] = high_momentum['fwd_1y']
    
    # 策略6：组合（A股 + 高PE + 高Alpha）
    combo1 = df_valid[(df_valid['region'] == 'A股') & 
                      (df_valid['val_pct'] > 0.6) & 
                      (df_valid['F_alpha'] > 70)]
    strategies['A股+高PE+高Alpha'] = combo1['fwd_1y']
    
    # 策略7：组合（高Alpha + 高动量）
    combo2 = df_valid[(df_valid['F_alpha'] > 70) & (df_valid['F_momentum'] > 60)]
    strategies['高Alpha+高动量'] = combo2['fwd_1y']
    
    # 策略8：降低F_value权重（只用F_alpha和F_momentum）
    # 模拟：S_new = 0.5*F_alpha + 0.5*F_momentum
    df_valid['S_no_value'] = np.where(
        df_valid['F_alpha'].notna() & df_valid['F_momentum'].notna(),
        0.5 * df_valid['F_alpha'].fillna(0) + 0.5 * df_valid['F_momentum'].fillna(0),
        df_valid['S_total']
    )
    strategies['无F_value (S>60)'] = df_valid[df_valid['S_no_value'] > 60]['fwd_1y']
    
    print(f"\n--- 不同策略的收益对比（未来1年）---")
    print(f"{'策略':<30s} {'样本数':>8s} {'平均收益':>10s} {'胜率':>8s} {'中位数':>10s} {'Sharpe':>8s}")
    print("-" * 80)
    
    for strat_name, returns in strategies.items():
        if len(returns) >= 10:
            mean_ret = returns.mean()
            winrate = (returns > 0).mean()
            median_ret = returns.median()
            sharpe = mean_ret / (returns.std() + 1e-10) * np.sqrt(1)  # annualized
            
            print(f"{strat_name:<30s} {len(returns):>8d} {mean_ret*100:>9.2f}% {winrate*100:>7.1f}% "
                  f"{median_ret*100:>9.2f}% {sharpe:>8.3f}")
        else:
            print(f"{strat_name:<30s} {'样本不足':>8s}")
    
    return strategies

def main():
    print("="*80)
    print("阶段3：真实基金数据验证")
    print("="*80)
    
    df = load_scan_data()
    
    # 加载净值数据
    print("\n加载基金净值数据...")
    codes = df['code'].astype(str).tolist()
    navs = load_nav_data(codes[:3000])
    print(f"加载了 {len(navs)} 只基金的净值数据")
    
    # 3-A: F_value有效性
    df_valid = analyze_f_value_effectiveness(df)
    
    # 3-B: 不同F_value设计
    ic_results = test_alternative_f_value(df, navs)
    
    # 3-C: 行业因子
    industry_results = test_industry_factor(df, navs)
    
    # 3-D: 综合改进
    strategy_results = test_combined_improvements(df, navs)
    
    print("\n" + "="*80)
    print("阶段3完成")
    print("="*80)

if __name__ == '__main__':
    main()
