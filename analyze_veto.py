# -*- coding: utf-8 -*-
"""分析: R_MDD 否决规则的证据 + 三个变体的反事实对比"""
import glob, re
import numpy as np
import pandas as pd

rows = pd.concat([pd.read_csv(fp, dtype={"code": str}) for fp in glob.glob("output/veto_rows/*.csv")])
rows = rows.dropna(subset=["S"])
# 日期截面收益分位 (剔除行情beta)
rows["pct6"] = rows.groupby("date")["fwd6"].rank(pct=True)
rows["pct12"] = rows.groupby("date")["fwd12"].rank(pct=True)

# ---- 重建 base 分 & 惩罚分解 ----
def parse_pens(s):
    out = []
    if isinstance(s, str) and s:
        for item in s.split("; "):
            m = re.search(r"\(-(\d+)%\)", item)
            out.append((item[: item.find("(-")], float(m.group(1)) / 100 if m else 0.0))
    return out


def base_score(r):
    num, den = 0.0, 0.0
    if pd.notna(r.F_value):
        num += r.wv * min(r.F_value, 100); den += r.wv
    if pd.notna(r.F_alpha):
        num += r.wa * r.F_alpha; den += r.wa
    if pd.notna(r.F_mom):
        num += r.wm * r.F_mom; den += r.wm
    return num / den if den > 1e-9 else np.nan


def rnow(rm):
    if pd.isna(rm): return 1.0
    if rm > 1.5: return 0.0
    if rm > 1.2: return 0.5
    return 1.0

def rsoft(rm):                       # 变体S1 阶梯惩罚
    if pd.isna(rm): return 1.0
    if rm > 2.0: return 0.35
    if rm > 1.5: return 0.60
    if rm > 1.2: return 0.85
    return 1.0


rows["pens_l"] = rows.pens.apply(parse_pens)
rows["other_pen"] = rows.pens_l.apply(
    lambda l: float(np.prod([1 - p for n, p in l if "回撤比值" not in n])) if l else 1.0)
rows["base"] = rows.apply(base_score, axis=1)
rows["S_now"] = (rows.base * rows.other_pen * rows.R_MDD.apply(rnow)).clip(0, 100).round(1)
rows["S_soft"] = (rows.base * rows.other_pen * rows.R_MDD.apply(rsoft)).clip(0, 100).round(1)
rows["S_clean"] = (rows.base * rows.other_pen).clip(0, 100).round(1)   # 变体S2: 完全取消
chk = (rows.S_now - rows.S).abs()
print(f"[校验] 重建S与引擎S差值: max={chk.max():.2f} 相关系数={rows.S_now.corr(rows.S):.4f} n={len(rows)}")

# ================= 1. R_MDD 分桶 vs 前瞻收益 =================
bins = [-0.01, 1.0, 1.2, 1.5, 2.0, 999]
labs = ["≤1.0", "(1.0,1.2]", "(1.2,1.5]", "(1.5,2.0]", ">2.0"]
rows["rb"] = pd.cut(rows.R_MDD, bins=bins, labels=labs)
g = rows.groupby("rb", observed=True).agg(
    n=("S", "size"),
    fwd6_mean=("fwd6", "mean"), fwd6_med=("fwd6", "median"),
    win6=("fwd6", lambda s: (s > 0).mean()),
    pct6=("pct6", "mean"),
    big6=("fwd6", lambda s: (s > 0.15).mean()),
    fwd12_mean=("fwd12", "mean"), pct12=("pct12", "mean"),
    avg_S=("S", "mean"))
print("\n===== R_MDD 分桶 × 前瞻收益 (28季度, n=%d) =====" % len(rows))
print(g.round(4).to_string())

# 高水位期 (>0.5) vs 低水位期 的否决区表现
veto = rows[rows.R_MDD > 1.5]
print(f"\n[否决区 R_MDD>1.5] n={len(veto)}  占全部{len(veto)/len(rows):.1%}")
for lab, d in [("高/中水位(water>0.35)", veto[veto.water > 0.35]),
               ("低水位(water≤0.35)", veto[veto.water <= 0.35])]:
    d = d.dropna(subset=["fwd6"])
    if len(d):
        print(f"  {lab} n={len(d)}: fwd6均值{d.fwd6.mean():+.1%} 胜率{(d.fwd6>0).mean():.0%} "
              f"截面分位{d.pct6.mean():.3f} fwd6>+15%占比{(d.fwd6>0.15).mean():.0%} fwd12均值{d.fwd12.mean():+.1%}")

# ================= 2. 被一票否决的样本里有多少大赢家 =================
v = veto.dropna(subset=["fwd6"])
print("\n===== 否决区错杀画像 (R_MDD>1.5, n=%d) =====" % len(v))
print(f"  fwd6>0:      {(v.fwd6>0).mean():.1%}")
print(f"  fwd6>+15%:   {(v.fwd6>0.15).mean():.1%}   ← 一票否决错杀大赢家的比例")
print(f"  fwd6>+30%:   {(v.fwd6>0.30).mean():.1%}")
print(f"  fwd6<-15%:   {(v.fwd6<-0.15).mean():.1%}   ← 否决确实避雷的比例")
print(f"  fwd6中位数:  {v.fwd6.median():+.1%} | 均值 {v.fwd6.mean():+.1%} | 截面分位 {v.pct6.mean():.3f}")
halve = rows[(rows.R_MDD > 1.2) & (rows.R_MDD <= 1.5)].dropna(subset=["fwd6"])
print(f"\n[腰斩区 (1.2,1.5]] n={len(halve)}: fwd6均值{halve.fwd6.mean():+.1%} 胜率{(halve.fwd6>0).mean():.0%} "
      f"分位{halve.pct6.mean():.3f} >+15%占比{(halve.fwd6>0.15).mean():.0%}")

# ================= 3. 反事实: 三个规则下的信号表现 =================
def sig_stats(mask, lab):
    d = rows[mask & rows.fwd6.notna()]
    if not len(d):
        print(f"  {lab}: 无样本"); return
    print(f"  {lab}: n={len(d)} fwd6均值{d.fwd6.mean():+.1%} 胜率{(d.fwd6>0).mean():.0%} "
          f"截面分位{d.pct6.mean():.3f} fwd12均值{d.fwd12.mean():+.1%}")

print("\n===== Buy线(S≥70) 信号对比 =====")
sig_stats(rows.S_now >= 70, "现行规则 Buy")
sig_stats(rows.S_soft >= 70, "变体S1阶梯 Buy")
sig_stats((rows.S_soft >= 70) & (rows.S_now < 70), "  其中新增(被R_MDD压下去的)")
sig_stats(rows.S_clean >= 70, "变体S2取消 Buy")
sig_stats((rows.S_clean >= 70) & (rows.S_now < 70), "  其中新增")

print("\n===== ⚔️动量战术信号 (S≥50 & 池分位≥0.95 & F_mom≥100) =====")
m100 = (rows.F_mom >= 100) & (rows.rank_pct >= 0.95)
sig_stats(m100 & (rows.S_now >= 50), "现行规则 ⚔️")
sig_stats(m100 & (rows.S_soft >= 50), "变体S1阶梯 ⚔️")
sig_stats(m100 & (rows.S_soft >= 50) & (rows.S_now < 50), "  其中新增")
sig_stats(m100 & (rows.S_clean >= 50), "变体S2取消 ⚔️")
sig_stats(m100 & (rows.S_clean >= 50) & (rows.S_now < 50), "  其中新增")

# 新增信号的水位分布
new_sig = rows[m100 & (rows.S_soft >= 50) & (rows.S_now < 50) & rows.fwd6.notna()]
if len(new_sig):
    print(f"  新增⚔️水位分布: 均值water={new_sig.water.mean():.2f} | "
          f"fwd6胜率(高水位>{0.5}区)={((new_sig[new_sig.water>0.5].fwd6>0).mean() if (new_sig.water>0.5).any() else float('nan')):.0%}")

# ================= 4. 否决区大赢家案例 =================
print("\n===== R_MDD>1.5 却被证真(其后6月>+25%)的案例Top8 =====")
cs = v[v.fwd6 > 0.25].nlargest(8, "fwd6")[["date", "code", "name", "R_MDD", "S", "F_mom", "fwd6", "fwd12"]]
print(cs.to_string(index=False))

rows.to_csv("output/veto_analysis_rows.csv", index=False, encoding="utf-8-sig")
print("\n[saved] output/veto_analysis_rows.csv")
