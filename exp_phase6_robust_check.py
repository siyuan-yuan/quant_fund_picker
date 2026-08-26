#!/usr/bin/env python3
"""
阶段6：稳健性检验
1. 确保行业因子没有前瞻偏差
2. 用滚动窗口构建行业因子
3. 最终验证改进效果
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

def load_scan_data():
    df = pd.read_csv('output/scan_20260810.csv')
    
    def parse_rbsa(rbsa_str):
        try:
            if pd.isna(rbsa_str) or rbsa_str == '':
                return {}
            return json.loads(rbsa_str.replace('""', '"').strip('"'))
        except:
            return {}
    
    df['rbsa_dict'] = df['rbsa'].apply(parse_rbsa)
    sector_indices = ['全指材料', '全指工业', '全指消费', '全指医药', '全指金融', '全指信息']
    
    def get_sector_weights(rbsa_dict):
        return {k: v for k, v in rbsa_dict.items() if k in sector_indices}
    
    def get_top_sector(rbsa_dict):
        sw = get_sector_weights(rbsa_dict)
        if not sw:
            return '未知'
        return max(sw, key=sw.get)
    
    df['top_sector'] = df['rbsa_dict'].apply(get_top_sector)
    df['region'] = df['region'].fillna('未知')
    return df

def load_all_navs():
    """加载所有净值"""
    navs = {}
    for f in os.listdir('cache'):
        if f.startswith('nav_') and f.endswith('.csv'):
            code = f[4:-4]
            try:
                df = pd.read_csv(f'cache/{f}', parse_dates=['date'])
                df = df.set_index('date').sort_index()
                if len(df) > 500:
                    navs[code] = df['nav']
            except:
                continue
    return navs

def load_sector_indices():
    """加载行业指数数据"""
    sectors = {}
    sector_files = {
        'csi_000987': '全指材料',
        'csi_000988': '全指工业',
        'csi_000990': '全指消费',
        'csi_000991': '全指医药',
        'csi_000992': '全指金融',
        'csi_000993': '全指信息',
    }
    
    for code, name in sector_files.items():
        filepath = f'cache/{code}.csv'
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        sectors[name] = df
    
    return sectors

def main():
    print("="*80)
    print("阶段6：稳健性检验")
    print("="*80)
    
    # 加载数据
    df = load_scan_data()
    navs = load_all_navs()
    sectors = load_sector_indices()
    
    print(f"基金: {len(df)}, 净值: {len(navs)}, 行业指数: {len(sectors)}")
    
    # ========================================
    # 构建行业动量因子（用行业指数数据，不是基金收益）
    # ========================================
    print("\n--- 构建行业动量因子（基于行业指数，避免前瞻偏差）---")
    
    # 计算每个行业在每个日期的动量（过去6个月收益）
    sector_momentum = {}
    for name, sdf in sectors.items():
        mom_6m = sdf['close'].pct_change(126)  # 6个月动量
        sector_momentum[name] = mom_6m
    
    # 构建行业动量面板
    sector_mom_df = pd.DataFrame(sector_momentum)
    print(f"行业动量面板: {sector_mom_df.shape}")
    
    # ========================================
    # 构建基金级别的评估数据
    # ========================================
    
    # 为每只基金，在每个观察点：
    # 1. 获取当时的行业配置（RBSA）
    # 2. 获取当时行业的动量
    # 3. 计算行业加权动量得分
    # 4. 计算未来收益
    
    eval_rows = []
    
    for _, fund in df.iterrows():
        code = str(fund['code'])
        if code not in navs:
            continue
        
        nav = navs[code]
        rbsa_dict = fund['rbsa_dict']
        sector_weights = {k: v for k, v in rbsa_dict.items() if k in sector_momentum}
        
        if not sector_weights:
            continue
        
        # 总权重归一化
        total_w = sum(sector_weights.values())
        if total_w < 0.1:
            continue
        
        # 对每个可能的观察点
        dates = nav.index
        if len(dates) < 500:
            continue
        
        # 取最近24个月的观察点（每3个月一个）
        for months_back in range(0, 24, 3):
            idx = len(dates) - 1 - months_back * 21
            if idx < 252:
                continue
            
            obs_date = dates[idx]
            
            # 获取观察日期的行业动量
            # 找到最接近的行业数据日期
            industry_mom_scores = {}
            for sector_name, mom_series in sector_momentum.items():
                mom_asof = mom_series.asof(obs_date)
                if pd.notna(mom_asof):
                    industry_mom_scores[sector_name] = mom_asof
            
            if not industry_mom_scores:
                continue
            
            # 计算加权行业动量
            weighted_mom = 0
            for sector_name, weight in sector_weights.items():
                if sector_name in industry_mom_scores:
                    weighted_mom += (weight / total_w) * industry_mom_scores[sector_name]
            
            # 计算未来1年收益
            future_idx = min(idx + 252, len(dates) - 1)
            start_nav = nav.iloc[idx]
            end_nav = nav.iloc[future_idx]
            
            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav <= 0:
                continue
            
            fwd_ret = end_nav / start_nav - 1
            
            eval_rows.append({
                'code': code,
                'obs_date': obs_date,
                'top_sector': fund['top_sector'],
                'region': fund['region'],
                'industry_momentum': weighted_mom,
                'F_value': fund['F_value'],
                'F_alpha': fund['F_alpha'],
                'F_momentum': fund['F_momentum'],
                'val_pct': fund['val_pct'],
                'ir_winrate': fund['ir_winrate'],
                'S_total': fund['S_total'],
                'forward_1y': fwd_ret,
            })
    
    df_eval = pd.DataFrame(eval_rows)
    print(f"\n评估数据集: {len(df_eval)} 条记录")
    print(f"唯一基金数: {df_eval['code'].nunique()}")
    print(f"观察日期范围: {df_eval['obs_date'].min()} ~ {df_eval['obs_date'].max()}")
    
    if len(df_eval) < 100:
        print("数据不足")
        return
    
    # ========================================
    # 测试行业动量因子（无前瞻偏差）
    # ========================================
    print(f"\n{'='*80}")
    print("测试行业动量因子（滚动窗口，无前瞻偏差）")
    print(f"{'='*80}")
    
    # 单个因子的IC
    factors = {
        'industry_momentum': df_eval['industry_momentum'],
        'F_value': df_eval['F_value'],
        'F_alpha': df_eval['F_alpha'],
        'F_momentum': df_eval['F_momentum'],
        'val_pct': df_eval['val_pct'] * 100,
        'ir_winrate': df_eval['ir_winrate'],
        'S_total': df_eval['S_total'],
    }
    
    print(f"\n{'因子':<25s} {'IC':>8s} {'RankIC':>8s} {'t值':>8s}")
    print("-" * 55)
    
    for fname, fvalues in factors.items():
        valid = pd.DataFrame({'factor': fvalues, 'return': df_eval['forward_1y']}).dropna()
        if len(valid) < 50:
            continue
        ic = valid['factor'].corr(valid['return'])
        rank_ic = stats.spearmanr(valid['factor'], valid['return'])[0]
        n = len(valid)
        t = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        print(f"{fname:<25s} {ic:>+8.4f} {rank_ic:>+8.4f} {t:>8.2f}")
    
    # ========================================
    # 测试组合策略
    # ========================================
    print(f"\n{'='*80}")
    print("组合策略测试（无前瞻偏差）")
    print(f"{'='*80}")
    
    # 标准化行业动量到0-100
    ind_mom = df_eval['industry_momentum']
    ind_mom_norm = (ind_mom - ind_mom.min()) / (ind_mom.max() - ind_mom.min() + 1e-10) * 100
    
    # 标准化其他因子
    def norm(s):
        s_clean = s.fillna(s.median())
        return (s_clean - s_clean.min()) / (s_clean.max() - s_clean.min() + 1e-10) * 100
    
    alpha_norm = norm(df_eval['F_alpha'])
    mom_norm = norm(df_eval['F_momentum'])
    ir_norm = norm(df_eval['ir_winrate'])
    val_norm = norm(df_eval['val_pct'] * 100)
    
    strategies = {}
    
    # 基线：当前系统
    strategies['当前系统'] = df_eval['S_total']
    
    # 无F_value
    S_no_value = (0.6 * alpha_norm + 0.4 * mom_norm)
    strategies['无F_value (Alpha60+Mom40)'] = S_no_value
    
    # 加入行业动量
    S_with_industry_10 = 0.9 * S_no_value + 0.1 * ind_mom_norm
    S_with_industry_20 = 0.8 * S_no_value + 0.2 * ind_mom_norm
    S_with_industry_30 = 0.7 * S_no_value + 0.3 * ind_mom_norm
    
    strategies['行业10%+Alpha54%+Mom36%'] = S_with_industry_10
    strategies['行业20%+Alpha48%+Mom32%'] = S_with_industry_20
    strategies['行业30%+Alpha42%+Mom28%'] = S_with_industry_30
    
    # 加入行业动量（用ir_winrate替代F_alpha）
    S_ir_mom = 0.5 * ir_norm + 0.5 * mom_norm
    S_ir_mom_ind = 0.7 * S_ir_mom + 0.3 * ind_mom_norm
    strategies['IR50+Mom50 → 行业30%'] = S_ir_mom_ind
    
    # 纯动量+行业
    S_mom_ind = 0.6 * mom_norm + 0.4 * ind_mom_norm
    strategies['Mom60+行业40'] = S_mom_ind
    
    # 反转val_pct + 行业
    val_rev = 100 - val_norm
    S_rev_ind = 0.2 * val_rev + 0.4 * alpha_norm + 0.2 * mom_norm + 0.2 * ind_mom_norm
    strategies['RevVal20%+Alpha40%+Mom20%+Ind20%'] = S_rev_ind
    
    print(f"\n{'策略':<40s} {'IC':>8s} {'RankIC':>8s} {'选中>60':>8s} {'平均收益':>10s}")
    print("-" * 80)
    
    for strat_name, scores in strategies.items():
        valid = pd.DataFrame({'score': scores, 'return': df_eval['forward_1y']}).dropna()
        if len(valid) < 50:
            continue
        
        ic = valid['score'].corr(valid['return'])
        rank_ic = stats.spearmanr(valid['score'], valid['return'])[0]
        
        # 选基结果
        selected = valid[valid['score'] > 60]
        if len(selected) > 10:
            avg_ret = selected['return'].mean()
            n_selected = len(selected)
        else:
            avg_ret = np.nan
            n_selected = 0
        
        print(f"{strat_name:<40s} {ic:>+8.4f} {rank_ic:>+8.4f} {n_selected:>8d} {avg_ret*100:>9.2f}%")
    
    # ========================================
    # 分年度稳定性
    # ========================================
    print(f"\n{'='*80}")
    print("分年度稳定性（行业动量因子）")
    print(f"{'='*80}")
    
    df_eval['year'] = pd.to_datetime(df_eval['obs_date']).dt.year
    
    print(f"\n{'年份':<8s} {'行业动量IC':>12s} {'F_mom IC':>10s} {'F_value IC':>10s} {'S_total IC':>10s}")
    print("-" * 55)
    
    for year in sorted(df_eval['year'].unique()):
        y_data = df_eval[df_eval['year'] == year]
        if len(y_data) < 30:
            continue
        
        ics = {}
        for fname in ['industry_momentum', 'F_momentum', 'F_value', 'S_total']:
            valid = y_data[[fname, 'forward_1y']].dropna()
            if len(valid) > 10:
                ics[fname] = valid[fname].corr(valid['forward_1y'])
            else:
                ics[fname] = np.nan
        
        print(f"{year:<8d} {ics['industry_momentum']:>+12.4f} {ics['F_momentum']:>+10.4f} "
              f"{ics['F_value']:>+10.4f} {ics['S_total']:>+10.4f}")
    
    # ========================================
    # 行业动量的边际贡献
    # ========================================
    print(f"\n{'='*80}")
    print("行业动量的边际贡献分析")
    print(f"{'='*80}")
    
    # 在不同基金子集中测试
    for sector_name in ['全指信息', '全指医药', '全指消费', '全指材料']:
        sub = df_eval[df_eval['top_sector'] == sector_name]
        if len(sub) < 50:
            continue
        
        ind_corr = sub['industry_momentum'].corr(sub['forward_1y'])
        mom_corr = sub['F_momentum'].corr(sub['forward_1y'])
        
        print(f"\n{sector_name} ({len(sub)}条):")
        print(f"  行业动量IC: {ind_corr:+.4f}")
        print(f"  F_momentum IC: {mom_corr:+.4f}")
        print(f"  平均未来收益: {sub['forward_1y'].mean()*100:.2f}%")
    
    # ========================================
    # 最终结论
    # ========================================
    print(f"\n{'='*80}")
    print("最终结论")
    print(f"{'='*80}")
    
    # 关键数字
    baseline_ic = df_eval['S_total'].corr(df_eval['forward_1y'])
    
    # 最优无偏差策略
    best_ic = max(
        S_with_industry_20.corr(df_eval['forward_1y']),
        S_with_industry_30.corr(df_eval['forward_1y']),
    )
    
    print(f"\n基线系统IC: {baseline_ic:+.4f}")
    print(f"最优改进IC: {best_ic:+.4f}")
    print(f"IC提升: {best_ic - baseline_ic:+.4f}")
    print(f"相对提升: {(best_ic - baseline_ic) / baseline_ic * 100:.1f}%")

if __name__ == '__main__':
    main()
