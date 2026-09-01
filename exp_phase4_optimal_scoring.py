#!/usr/bin/env python3
"""
阶段4：设计最优打分系统
基于阶段1-3的发现，设计并验证改进后的打分系统
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
    """加载评分数据"""
    df = pd.read_csv('output/scan_20260810.csv')
    
    def parse_rbsa(rbsa_str):
        try:
            if pd.isna(rbsa_str) or rbsa_str == '':
                return {}
            return json.loads(rbsa_str.replace('""', '"').strip('"'))
        except:
            return {}
    
    df['rbsa_dict'] = df['rbsa'].apply(parse_rbsa)
    
    # 行业配置
    sector_indices = ['全指材料', '全指工业', '全指消费', '全指医药', '全指金融', '全指信息']
    
    def get_top_sector(rbsa_dict):
        sector_weights = {k: v for k, v in rbsa_dict.items() if k in sector_indices}
        if not sector_weights:
            return '未知'
        return max(sector_weights, key=sector_weights.get)
    
    def get_max_sector_weight(rbsa_dict):
        sector_weights = {k: v for k, v in rbsa_dict.items() if k in sector_indices}
        if not sector_weights:
            return 0
        return max(sector_weights.values())
    
    df['top_sector'] = df['rbsa_dict'].apply(get_top_sector)
    df['max_sector_weight'] = df['rbsa_dict'].apply(get_max_sector_weight)
    df['region'] = df['region'].fillna('未知')
    
    return df

def load_nav_data(codes):
    """加载基金净值"""
    navs = {}
    for code in codes:
        filepath = f'cache/nav_{code}.csv'
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            df = df.set_index('date').sort_index()
            if len(df) > 500:
                navs[str(code)] = df
        except:
            continue
    return navs

def calculate_multi_period_returns(navs):
    """计算多个时间段的收益"""
    results = {}
    
    for code, nav_df in navs.items():
        nav = nav_df['nav']
        dates = nav_df.index
        
        if len(nav_df) < 600:
            continue
        
        # 计算多个观察点的收益
        for months_back in range(0, 24, 3):  # 从0到24个月前，每3个月一个点
            for forward_months in [6, 12]:  # 6个月和12个月收益
                start_idx = len(nav_df) - 1 - months_back * 21
                end_idx = min(start_idx + forward_months * 21, len(nav_df) - 1)
                
                if start_idx < 0 or end_idx >= len(nav_df):
                    continue
                
                start_nav = nav.iloc[start_idx]
                end_nav = nav.iloc[end_idx]
                
                if pd.isna(start_nav) or pd.isna(end_nav) or start_nav <= 0:
                    continue
                
                ret = end_nav / start_nav - 1
                obs_date = dates[start_idx].strftime('%Y-%m-%d')
                
                key = (code, obs_date)
                results[key] = {
                    'forward_return': ret,
                    'forward_months': forward_months,
                    'obs_date': obs_date,
                }
    
    return results

def design_optimal_scoring(df):
    """
    设计最优打分系统
    基于前几个阶段的发现：
    1. F_value应该反转（或按市场自适应）
    2. 加入行业因子（权重20-30%）
    3. F_alpha和F_momentum是核心
    """
    print("="*80)
    print("阶段4：设计最优打分系统")
    print("="*80)
    
    # 加载净值和收益数据
    codes = df['code'].astype(str).tolist()
    navs = load_nav_data(codes[:5000])
    print(f"加载了 {len(navs)} 只基金的净值数据")
    
    returns_data = calculate_multi_period_returns(navs)
    print(f"计算了 {len(returns_data)} 个观察点的收益")
    
    # 构建评估数据集
    eval_rows = []
    for (code, obs_date), ret_info in returns_data.items():
        if ret_info['forward_months'] != 12:  # 只看1年收益
            continue
        
        fund_data = df[df['code'].astype(str) == code]
        if len(fund_data) == 0:
            continue
        
        row = fund_data.iloc[0].to_dict()
        row['obs_date'] = obs_date
        row['forward_1y'] = ret_info['forward_return']
        eval_rows.append(row)
    
    df_eval = pd.DataFrame(eval_rows)
    print(f"评估数据集: {len(df_eval)} 条记录")
    
    if len(df_eval) < 100:
        print("数据不足，需要更多基金数据")
        return
    
    # ============== 测试不同的打分公式 ==============
    
    print(f"\n{'='*80}")
    print("测试不同的打分公式")
    print(f"{'='*80}")
    
    formulas = {}
    
    # 公式1：当前系统
    formulas['当前系统'] = df_eval['S_total'].copy()
    
    # 公式2：反转F_value
    # S = (wv * F_value_reversed + wa * F_alpha + wm * F_momentum) / (wv + wa + wm)
    def reverse_value(val_pct):
        if pd.isna(val_pct):
            return np.nan
        # 高PE = 高分
        if val_pct >= 0.8:
            return 100.0
        elif val_pct >= 0.6:
            return 80.0
        elif val_pct >= 0.4:
            return 50.0
        elif val_pct >= 0.2:
            return 25.0
        else:
            return 10.0
    
    f_value_rev = df_eval['val_pct'].apply(reverse_value)
    
    # 获取权重
    wv = 0.3  # 假设估值权重30%（从config推测）
    wa = 0.5  # alpha权重50%
    wm = 0.2  # 动量权重20%
    
    # 如果有weights_mode信息，用更精确的权重
    # 暂时用假设的权重
    
    S_rev = (wv * f_value_rev + wa * df_eval['F_alpha'].fillna(0) + wm * df_eval['F_momentum'].fillna(0)) / (wv + wa + wm)
    S_rev = S_rev.clip(0, 100)
    formulas['反转F_value'] = S_rev
    
    # 公式3：分市场自适应F_value
    def adaptive_value(row):
        val_pct = row['val_pct']
        region = row['region']
        if pd.isna(val_pct):
            return np.nan
        
        if region == 'A股':
            # A股：高PE = 高分（成长溢价）
            if val_pct >= 0.8:
                return 100.0
            elif val_pct >= 0.6:
                return 80.0
            elif val_pct >= 0.4:
                return 50.0
            else:
                return 20.0
        else:
            # 其他：中性偏正向
            if val_pct >= 0.7:
                return 85.0
            elif val_pct >= 0.5:
                return 60.0
            elif val_pct >= 0.3:
                return 40.0
            else:
                return 25.0
    
    f_value_adaptive = df_eval.apply(adaptive_value, axis=1)
    S_adaptive = (wv * f_value_adaptive + wa * df_eval['F_alpha'].fillna(0) + wm * df_eval['F_momentum'].fillna(0)) / (wv + wa + wm)
    S_adaptive = S_adaptive.clip(0, 100)
    formulas['分市场自适应'] = S_adaptive
    
    # 公式4：去掉F_value（只用alpha和momentum）
    S_no_value = (wa * df_eval['F_alpha'].fillna(0) + wm * df_eval['F_momentum'].fillna(0)) / (wa + wm)
    S_no_value = S_no_value.clip(0, 100)
    formulas['无F_value'] = S_no_value
    
    # 公式5：加入行业因子
    # 计算行业得分（基于该行业基金的历史表现）
    sector_scores = {}
    for sector in df_eval['top_sector'].unique():
        sector_data = df_eval[df_eval['top_sector'] == sector]
        if len(sector_data) > 5:
            # 用行业平均Alpha作为行业得分
            sector_scores[sector] = sector_data['F_alpha'].mean()
    
    # 标准化行业得分到0-100
    if sector_scores:
        sector_df = pd.DataFrame(list(sector_scores.items()), columns=['sector', 'raw_score'])
        sector_df['industry_score'] = (sector_df['raw_score'] - sector_df['raw_score'].min()) / \
                                      (sector_df['raw_score'].max() - sector_df['raw_score'].min() + 1e-10) * 100
        sector_score_map = dict(zip(sector_df['sector'], sector_df['industry_score']))
    else:
        sector_score_map = {}
    
    industry_scores = df_eval['top_sector'].map(sector_score_map).fillna(50)
    
    # 混合：70%原分 + 30%行业分
    S_industry = 0.7 * S_no_value + 0.3 * industry_scores
    formulas['Alpha+Momentum+行业'] = S_industry
    
    # 公式6：最优组合（反转F_value + 行业因子）
    S_best = (0.15 * f_value_rev + 0.40 * df_eval['F_alpha'].fillna(0) + 
              0.20 * df_eval['F_momentum'].fillna(0) + 0.25 * industry_scores) / (0.15 + 0.40 + 0.20 + 0.25)
    S_best = S_best.clip(0, 100)
    formulas['最优组合'] = S_best
    
    # 公式7：纯Alpha+动量+行业（无F_value，行业权重更高）
    S_best2 = 0.35 * df_eval['F_alpha'].fillna(0) + 0.30 * df_eval['F_momentum'].fillna(0) + 0.35 * industry_scores
    formulas['Alpha35+Mom30+行业35'] = S_best2
    
    # ============== 评估每个公式 ==============
    
    print(f"\n--- 公式评估（IC与未来1年收益）---")
    print(f"{'公式':<30s} {'IC':>8s} {'RankIC':>8s} {'t值':>8s} {'显著':>6s}")
    print("-" * 65)
    
    ic_results = {}
    
    for formula_name, scores in formulas.items():
        valid_mask = scores.notna() & df_eval['forward_1y'].notna()
        
        if valid_mask.sum() < 50:
            continue
        
        s = scores[valid_mask]
        r = df_eval.loc[valid_mask, 'forward_1y']
        
        ic = s.corr(r)
        rank_ic = stats.spearmanr(s, r)[0]
        n = len(s)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        sig = '***' if abs(t_stat) > 2.576 else ('**' if abs(t_stat) > 1.96 else '*')
        
        print(f"{formula_name:<30s} {ic:>+8.4f} {rank_ic:>+8.4f} {t_stat:>8.2f} {sig:>6s}")
        
        ic_results[formula_name] = {
            'IC': ic,
            'RankIC': rank_ic,
            't_stat': t_stat,
        }
    
    # ============== 回测模拟 ==============
    
    print(f"\n{'='*80}")
    print("回测模拟：用不同公式选基金")
    print(f"{'='*80}")
    
    # 模拟：每个观察点，用公式选出top N基金，计算平均收益
    for threshold in [60, 70, 80]:
        print(f"\n--- 选基阈值: S > {threshold} ---")
        print(f"{'公式':<30s} {'选中数':>8s} {'平均收益':>10s} {'胜率':>8s} {'Sharpe':>8s}")
        print("-" * 70)
        
        for formula_name, scores in formulas.items():
            valid_mask = scores.notna() & df_eval['forward_1y'].notna() & (scores > threshold)
            
            if valid_mask.sum() < 10:
                print(f"{formula_name:<30s} {'样本不足':>8s}")
                continue
            
            returns = df_eval.loc[valid_mask, 'forward_1y']
            mean_ret = returns.mean()
            winrate = (returns > 0).mean()
            sharpe = mean_ret / (returns.std() + 1e-10)
            
            print(f"{formula_name:<30s} {len(returns):>8d} {mean_ret*100:>9.2f}% {winrate*100:>7.1f}% {sharpe:>8.3f}")
    
    # ============== 行业分析 ==============
    
    print(f"\n{'='*80}")
    print("行业因子详细分析")
    print(f"{'='*80}")
    
    print(f"\n--- 各行业平均收益和基金数量 ---")
    sector_analysis = df_eval.groupby('top_sector').agg({
        'forward_1y': ['mean', 'std', 'count'],
        'F_alpha': 'mean',
        'F_momentum': 'mean',
        'val_pct': 'mean',
    }).round(4)
    
    print(sector_analysis)
    
    # ============== 分年度稳定性测试 ==============
    
    print(f"\n--- 分年度IC稳定性 ---")
    df_eval['year'] = pd.to_datetime(df_eval['obs_date']).dt.year
    
    for formula_name, scores in formulas.items():
        df_eval[f'score_{formula_name}'] = scores
    
    yearly_ics = {}
    for year in sorted(df_eval['year'].unique()):
        year_data = df_eval[df_eval['year'] == year]
        if len(year_data) < 20:
            continue
        
        year_ics = {}
        for formula_name in formulas.keys():
            score_col = f'score_{formula_name}'
            valid = year_data[[score_col, 'forward_1y']].dropna()
            if len(valid) > 10:
                ic = valid[score_col].corr(valid['forward_1y'])
                year_ics[formula_name] = ic
        
        yearly_ics[year] = year_ics
    
    df_yearly = pd.DataFrame(yearly_ics).T
    print(f"\n分年度IC:")
    print(df_yearly.round(4))
    
    if len(df_yearly) > 1:
        print(f"\nIC均值和标准差:")
        for col in df_yearly.columns:
            mean_ic = df_yearly[col].mean()
            std_ic = df_yearly[col].std()
            print(f"  {col:<30s}: IC={mean_ic:+.4f} (std={std_ic:.4f})")
    
    # ============== 最终推荐 ==============
    
    print(f"\n{'='*80}")
    print("最终推荐")
    print(f"{'='*80}")
    
    # 综合评估
    print(f"\n综合IC排名（所有数据）:")
    ranked = sorted(ic_results.items(), key=lambda x: x[1]['IC'], reverse=True)
    for i, (name, metrics) in enumerate(ranked):
        print(f"  {i+1}. {name}: IC={metrics['IC']:+.4f}, RankIC={metrics['RankIC']:+.4f}")
    
    # 推荐改进方向
    print(f"\n推荐改进方向:")
    best_name = ranked[0][0]
    best_ic = ranked[0][1]['IC']
    baseline_ic = ic_results.get('当前系统', {}).get('IC', 0)
    
    print(f"  当前系统IC: {baseline_ic:+.4f}")
    print(f"  最优公式: {best_name} (IC={best_ic:+.4f})")
    print(f"  IC提升: {best_ic - baseline_ic:+.4f}")

def main():
    print("="*80)
    print("阶段4：最优打分系统设计")
    print("="*80)
    
    df = load_scan_data()
    print(f"加载了 {len(df)} 只基金")
    
    design_optimal_scoring(df)
    
    print("\n" + "="*80)
    print("阶段4完成")
    print("="*80)

if __name__ == '__main__':
    main()
