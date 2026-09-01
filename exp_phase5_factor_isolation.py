#!/usr/bin/env python3
"""
阶段5：因子隔离测试
精确分析每个因子对打分系统的贡献
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
    
    def get_top_sector(rbsa_dict):
        sector_weights = {k: v for k, v in rbsa_dict.items() if k in sector_indices}
        if not sector_weights:
            return '未知'
        return max(sector_weights, key=sector_weights.get)
    
    df['top_sector'] = df['rbsa_dict'].apply(get_top_sector)
    df['region'] = df['region'].fillna('未知')
    return df

def load_nav_data(codes):
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

def main():
    print("="*80)
    print("阶段5：因子隔离测试")
    print("="*80)
    
    df = load_scan_data()
    codes = df['code'].astype(str).tolist()
    navs = load_nav_data(codes[:5000])
    print(f"加载了 {len(navs)} 只基金")
    
    # 计算多期收益
    returns_by_obs = {}
    
    for code, nav_df in navs.items():
        nav = nav_df['nav']
        dates = nav_df.index
        
        if len(nav_df) < 600:
            continue
        
        for months_back in range(0, 24, 3):
            start_idx = len(nav_df) - 1 - months_back * 21
            end_idx = min(start_idx + 252, len(nav_df) - 1)
            
            if start_idx < 0 or start_idx >= len(nav_df):
                continue
            
            start_nav = nav.iloc[start_idx]
            end_nav = nav.iloc[end_idx]
            
            if pd.isna(start_nav) or pd.isna(end_nav) or start_nav <= 0:
                continue
            
            ret = end_nav / start_nav - 1
            obs_date = dates[start_idx].strftime('%Y-%m-%d')
            
            key = (code, obs_date)
            returns_by_obs[key] = ret
    
    print(f"计算了 {len(returns_by_obs)} 个观察点")
    
    # 合并基金数据和收益
    eval_rows = []
    for (code, obs_date), fwd_ret in returns_by_obs.items():
        fund_data = df[df['code'].astype(str) == code]
        if len(fund_data) == 0:
            continue
        row = fund_data.iloc[0].to_dict()
        row['obs_date'] = obs_date
        row['forward_1y'] = fwd_ret
        eval_rows.append(row)
    
    df_eval = pd.DataFrame(eval_rows)
    print(f"评估数据集: {len(df_eval)} 条记录")
    
    # ========================================
    # 核心分析1：单个因子的预测能力
    # ========================================
    print(f"\n{'='*80}")
    print("核心分析1：单个因子的预测能力")
    print(f"{'='*80}")
    
    factors = {
        'F_value': df_eval['F_value'],
        'F_alpha': df_eval['F_alpha'],
        'F_momentum': df_eval['F_momentum'],
        'val_pct': df_eval['val_pct'] * 100,  # 转换为百分比方便看
        'mom_4m1m': df_eval['mom_4m1m'],
        'mom_7m1m': df_eval['mom_7m1m'],
        'ir_winrate': df_eval['ir_winrate'],
        'down_capture': df_eval['down_capture'],
        'rank4': df_eval['rank4'],
        'rank7': df_eval['rank7'],
        'S_total': df_eval['S_total'],
        'water': df_eval['water'],
    }
    
    print(f"\n{'因子':<15s} {'有效样本':>8s} {'IC':>8s} {'RankIC':>8s} {'t值':>8s} {'显著':>5s}")
    print("-" * 60)
    
    factor_ics = {}
    for fname, fvalues in factors.items():
        valid = pd.DataFrame({'factor': fvalues, 'return': df_eval['forward_1y']}).dropna()
        if len(valid) < 50:
            continue
        
        ic = valid['factor'].corr(valid['return'])
        rank_ic = stats.spearmanr(valid['factor'], valid['return'])[0]
        n = len(valid)
        t = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        sig = '***' if abs(t) > 2.576 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.645 else ''))
        
        print(f"{fname:<15s} {n:>8d} {ic:>+8.4f} {rank_ic:>+8.4f} {t:>8.2f} {sig:>5s}")
        factor_ics[fname] = ic
    
    # ========================================
    # 核心分析2：因子相关性
    # ========================================
    print(f"\n{'='*80}")
    print("核心分析2：因子间相关性")
    print(f"{'='*80}")
    
    factor_df = pd.DataFrame({k: v for k, v in factors.items()})
    corr_matrix = factor_df.corr()
    
    print(f"\n{'':>12s}", end='')
    for col in ['F_value', 'F_alpha', 'F_momentum', 'S_total']:
        print(f"{col:>12s}", end='')
    print()
    
    for row in ['F_value', 'F_alpha', 'F_momentum', 'S_total']:
        print(f"{row:>12s}", end='')
        for col in ['F_value', 'F_alpha', 'F_momentum', 'S_total']:
            print(f"{corr_matrix.loc[row, col]:>12.3f}", end='')
        print()
    
    # ========================================
    # 核心分析3：系统化的因子组合测试
    # ========================================
    print(f"\n{'='*80}")
    print("核心分析3：系统化的因子组合测试")
    print(f"{'='*80}")
    
    # 标准化因子到0-100
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-10) * 100
    
    f_value_norm = normalize(df_eval['F_value'].fillna(50))
    f_alpha_norm = normalize(df_eval['F_alpha'].fillna(0))
    f_mom_norm = normalize(df_eval['F_momentum'].fillna(0))
    val_pct_norm = normalize(df_eval['val_pct'].fillna(0.5) * 100)
    
    # 反转val_pct
    val_pct_rev = 100 - val_pct_norm
    
    # 行业因子
    sector_ret = df_eval.groupby('top_sector')['forward_1y'].mean()
    sector_norm = normalize(df_eval['top_sector'].map(sector_ret).fillna(0))
    
    # 测试大量组合
    print(f"\n--- 全组合搜索（F_value方向 × 权重组合）---")
    print(f"{'组合':<40s} {'IC':>8s} {'RankIC':>8s}")
    print("-" * 60)
    
    best_combos = []
    
    # F_value的几种处理方式
    value_options = {
        '原始F_value': f_value_norm,
        '反转val_pct': val_pct_rev,
        '去掉F_value': pd.Series(50, index=df_eval.index),  # 中性值
    }
    
    # 权重组合
    weight_combos = [
        (0.3, 0.5, 0.2, '标准权重(30/50/20)'),
        (0.2, 0.5, 0.3, '轻value(20/50/30)'),
        (0.1, 0.5, 0.4, '极轻value(10/50/40)'),
        (0.0, 0.6, 0.4, '无value(0/60/40)'),
        (0.0, 0.5, 0.5, '无value(0/50/50)'),
        (0.2, 0.4, 0.4, '轻value(20/40/40)'),
        (0.1, 0.6, 0.3, '轻value(10/60/30)'),
    ]
    
    for value_name, value_scores in value_options.items():
        for wv, wa, wm, weight_desc in weight_combos:
            total_w = wv + wa + wm
            combined = (wv * value_scores + wa * f_alpha_norm + wm * f_mom_norm) / total_w
            
            valid = pd.DataFrame({'score': combined, 'return': df_eval['forward_1y']}).dropna()
            if len(valid) < 100:
                continue
            
            ic = valid['score'].corr(valid['return'])
            rank_ic = stats.spearmanr(valid['score'], valid['return'])[0]
            
            combo_name = f"{value_name} + {weight_desc}"
            best_combos.append((combo_name, ic, rank_ic))
    
    # 加上行业因子
    for value_name, value_scores in value_options.items():
        for wi in [0.1, 0.2, 0.3]:
            remaining = 1.0 - wi
            for wa_ratio in [0.5, 0.6, 0.7]:
                wm_ratio = 1.0 - wa_ratio
                wa = remaining * wa_ratio
                wm = remaining * wm_ratio
                
                combined = (wa * f_alpha_norm + wm * f_mom_norm + wi * sector_norm)
                
                valid = pd.DataFrame({'score': combined, 'return': df_eval['forward_1y']}).dropna()
                if len(valid) < 100:
                    continue
                
                ic = valid['score'].corr(valid['return'])
                rank_ic = stats.spearmanr(valid['score'], valid['return'])[0]
                
                combo_name = f"行业{wi*100:.0f}% + Alpha{wa*100:.0f}% + Mom{wm*100:.0f}%"
                best_combos.append((combo_name, ic, rank_ic))
    
    # 排序
    best_combos.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20组合（按IC排名）:")
    for i, (name, ic, rank_ic) in enumerate(best_combos[:20]):
        print(f"  {i+1:>2d}. {name:<40s} IC={ic:>+8.4f}  RankIC={rank_ic:>+8.4f}")
    
    print(f"\nBottom 10组合（最差IC）:")
    for i, (name, ic, rank_ic) in enumerate(best_combos[-10:]):
        print(f"  {len(best_combos)-9+i:>2d}. {name:<40s} IC={ic:>+8.4f}  RankIC={rank_ic:>+8.4f}")
    
    # ========================================
    # 核心分析4：F_value的影响隔离
    # ========================================
    print(f"\n{'='*80}")
    print("核心分析4：F_value对系统的边际贡献")
    print(f"{'='*80}")
    
    # 有F_value vs 无F_value
    S_with_value = df_eval['S_total'].copy()
    
    # 无F_value的分数（只用alpha和momentum）
    valid_mask = df_eval['F_alpha'].notna() & df_eval['F_momentum'].notna()
    S_without_value = pd.Series(np.nan, index=df_eval.index)
    S_without_value[valid_mask] = (
        0.6 * normalize(df_eval.loc[valid_mask, 'F_alpha']) + 
        0.4 * normalize(df_eval.loc[valid_mask, 'F_momentum'])
    )
    
    # 比较
    for label, scores in [('有F_value', S_with_value), ('无F_value', S_without_value)]:
        valid = pd.DataFrame({'score': scores, 'return': df_eval['forward_1y']}).dropna()
        if len(valid) < 100:
            continue
        ic = valid['score'].corr(valid['return'])
        rank_ic = stats.spearmanr(valid['score'], valid['return'])[0]
        
        # 分位数分析
        q = valid['score'].quantile([0.2, 0.5, 0.8])
        top_quintile = valid[valid['score'] >= q[0.8]]['return'].mean()
        bottom_quintile = valid[valid['score'] <= q[0.2]]['return'].mean()
        spread = top_quintile - bottom_quintile
        
        print(f"\n{label}:")
        print(f"  IC: {ic:+.4f}")
        print(f"  RankIC: {rank_ic:+.4f}")
        print(f"  Top quintile收益: {top_quintile*100:.2f}%")
        print(f"  Bottom quintile收益: {bottom_quintile*100:.2f}%")
        print(f"  收益差距: {spread*100:.2f}%")
    
    # ========================================
    # 核心分析5：分市场测试
    # ========================================
    print(f"\n{'='*80}")
    print("核心分析5：分市场因子有效性")
    print(f"{'='*80}")
    
    for market in ['A股', '海外']:
        market_data = df_eval[df_eval['region'] == market]
        if len(market_data) < 100:
            print(f"\n{market}: 数据不足 ({len(market_data)}条)")
            continue
        
        print(f"\n{market} ({len(market_data)}条):")
        
        for fname in ['F_value', 'F_alpha', 'F_momentum', 'val_pct']:
            fvalues = market_data[fname]
            if fname == 'val_pct':
                fvalues = fvalues * 100
            
            valid = pd.DataFrame({'factor': fvalues, 'return': market_data['forward_1y']}).dropna()
            if len(valid) < 50:
                continue
            
            ic = valid['factor'].corr(valid['return'])
            print(f"  {fname:<15s}: IC={ic:+.4f}")
    
    # ========================================
    # 核心分析6：时间稳定性
    # ========================================
    print(f"\n{'='*80}")
    print("核心分析6：分时间段因子稳定性")
    print(f"{'='*80}")
    
    df_eval['obs_date_dt'] = pd.to_datetime(df_eval['obs_date'])
    df_eval['year'] = df_eval['obs_date_dt'].dt.year
    df_eval['quarter'] = df_eval['obs_date_dt'].dt.to_period('Q')
    
    # 按季度分析
    print(f"\n分季度IC（F_value, F_alpha, F_momentum）:")
    print(f"{'季度':<12s} {'F_value':>10s} {'F_alpha':>10s} {'F_momentum':>10s} {'S_total':>10s}")
    print("-" * 55)
    
    for quarter in sorted(df_eval['quarter'].unique()):
        q_data = df_eval[df_eval['quarter'] == quarter]
        if len(q_data) < 30:
            continue
        
        ics = {}
        for fname in ['F_value', 'F_alpha', 'F_momentum', 'S_total']:
            valid = q_data[[fname, 'forward_1y']].dropna()
            if len(valid) > 10:
                ics[fname] = valid[fname].corr(valid['forward_1y'])
            else:
                ics[fname] = np.nan
        
        print(f"{str(quarter):<12s} {ics.get('F_value', np.nan):>+10.4f} "
              f"{ics.get('F_alpha', np.nan):>+10.4f} {ics.get('F_momentum', np.nan):>+10.4f} "
              f"{ics.get('S_total', np.nan):>+10.4f}")
    
    # ========================================
    # 总结
    # ========================================
    print(f"\n{'='*80}")
    print("总结与推荐")
    print(f"{'='*80}")
    
    # 计算因子重要性
    print(f"\n因子重要性排名（按IC）:")
    sorted_factors = sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (fname, ic) in enumerate(sorted_factors):
        direction = "正向" if ic > 0 else "反向"
        print(f"  {i+1}. {fname:<15s}: IC={ic:+.4f} ({direction})")

if __name__ == '__main__':
    main()
