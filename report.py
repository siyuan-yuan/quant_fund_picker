# -*- coding: utf-8 -*-
"""从 output/*.json 与指数分位生成中文投研报告 report.md"""
import json, glob, datetime as dt
import pandas as pd
import rbsa
from config import RBSA_INDICES

j = sorted(glob.glob("output/scores_*.json"))[-1]
rows = json.load(open(j))
df = pd.DataFrame(rows)
stamp = j.split("_")[-1].split(".")[0]

# 指数分位地形
pct = rbsa.index_pe_percentile()
lines = []
lines.append(f"# 量化选基系统 V3.1 — 实盘扫描报告\n")
lines.append(f"**数据截至**: {df['last_date'].dropna().max()} | **报告日期**: {dt.date.today()} | "
             f"**数据源**: 天天基金/新浪(aq)/乐咕乐股/中证指数公司\n")
lines.append("> ⚠️ 本报告为量化模型研究输出，不构成投资建议。基金有风险，投资须谨慎。\n")

lines.append("\n## 一、底层资产估值地形图（12因子面板，5年PE分位）\n")
lines.append("| 指数 | 类型 | 当前PE-TTM | 5年分位 | 估值信号 |")
lines.append("|---|---|---|---|---|")
for src, code, name, pe_key, tag in RBSA_INDICES:
    if pe_key == "none":
        t = "境外"
        lines.append(f"| {name} | {t} | — | 盲区 | ⬜ 无PE源(不计入) |")
        continue
    pe = __import__('provider').get_pe_by_key(pe_key).dropna()
    cur = pe.iloc[-1]; p = pct.get(name, float('nan'))
    sig = "🟢 极低估" if p <= 0.10 else ("🟩 低估" if p <= 0.30 else ("🟡 中性" if p <= 0.70 else "🔴 高估"))
    t = "风格" if src == "sina" else ("行业" if src == "csindex" else "境外")
    lines.append(f"| {name} | {t} | {cur:.2f} | {p:.1%} | {sig} |")
lines.append("\n**当前市场状态**: 宽基风格指数全面处于 5 年 80%~99% 分位（2025-26 牛市后段），"
             "行业端仅消费(31%)、医药(28%)尚存估值洼地。模型整体处于**防守姿态**，"
             "这是估值因子 40% 权重的设计本意：贵的时候，模型的职责就是拦住你。\n")

lines.append("\n## 二、演示池评分榜单（13只代表基金）\n")
lines.append("| 排名 | 代码 | 名称 | **总分** | 评级 | F_value | F_alpha | F_mom | PE分位 | 趋势确认 | IR胜率 | 触发风控 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for rank, (_, r) in enumerate(df.dropna(subset=['S_total']).iterrows(), 1):
    pen = r['penalty_str'] if isinstance(r.get('penalty_str'), str) and r['penalty_str'] else '—'
    val = f"{r['val_pct']:.1%}" if pd.notna(r.get('val_pct')) else '—'
    lines.append(f"| {rank} | {r['code']} | {r['name']} | **{r['S_total']}** | {r['rating'].split()[0]} | "
                 f"{r['F_value']} | {r['F_alpha']} | {r['F_momentum']} | {val} | "
                 f"{'✅' if r['trend_ok'] else '❌'} | {r['ir_winrate']:.0%} | {pen} |")

lines.append("\n\n### 榜单解读\n")
lines.append("- **无基金达到 Buy(≥70)**：PE 分位 >70% → F_value=0，上限被锁死在 60 分。模型拒绝在高位推荐买入。\n"
             "- **榜首 004744（创业板ETF联接，38.3）**：动量双前30%（F_mom=100）+ 无风处罚项，但估值 75.7% 分位 → 模型说\"趋势强，但别追\"。\n"
             "- **消费/医药左侧军团（003096/000083/161725/260108）**：F_value 35~44 分（全场唯四非零），"
             "验证了 V3.1 行业面板成功识别估值洼地；但 glass ceiling 在于 R_MDD 1.3~2.3 的超额回撤罚则与底部动量排名——"
             "模型判定为\"便宜但下降通道未走完\"，即**价值陷阱观察名单**而非买入名单。\n"
             "- **161725 白酒指数、260108 景顺长城被一票否决（总分0）**：R_MDD 2.28/1.51 触发 1.5 红线——"
             "熊市里跌得比自己的动态基准深 1.5 倍以上，模型定义为不可修复的尾部风险。\n")


def waterfall(code):
    r = df[df['code'] == code].iloc[0]
    w = r['rbsa']
    top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
    s = [f"#### {r['code']} {r['name']} → 总分 {r['S_total']}（{r['rating']}）\n",
         f"- RBSA隐形仓位: " + " / ".join(f"{k} {v:.0%}" for k, v in top),
         f"- F_value = {r['F_value']}（基准分{r['f_value_base']} + 缩水加分{r['bonus']}，PE分位 {r['val_pct']:.1%}，趋势{'确认' if r['trend_ok'] else '破位'}）",
         f"- F_alpha = {r['F_alpha']}（IR胜率 {r['ir_winrate']:.0%}→{r['s_ir']}分，下行捕获 {r['down_capture']:.2f}→{r['s_dc']}分）",
         f"- F_momentum = {r['F_momentum']}（4M-1M {r['mom_4m1m']:.1%} / 7M-1M {r['mom_7m1m']:.1%}）",
         f"- 风控乘数: {(r['penalty_str'] or '未触发')}；R_MDD={r['penalty_detail'].get('R_MDD')}，经理任职 {r['tenure_days']//365} 年，规模 {r['scale']}亿"]
    return "\n".join(s)

lines.append("\n\n## 三、单基打分瀑布解剖\n")
lines.append(waterfall(df.iloc[0]['code']))
lines.append("")
lines.append(waterfall("161725"))
lines.append("")
lines.append(waterfall("003096"))

lines.append("""
\n## 四、与 PDF 模型的实现映射 & 必要近似

| 模型条款 | 实现方式 | 近似说明 |
|---|---|---|
| Ridge-RBSA（60日/L2/非负/6板块） | ✅ 12因子面板（6风格+6全指行业），3个错位窗口取平均 | 升级为行业面板，修正"消费被误认为红利"的归因失真 |
| 当前P/E过去5年百分位 | ✅ RBSA权重 × 各指数PE-TTM五年分位加权 | 乐咕+中证官方双源，纯PiT时序 |
| 趋势确认（1M MACD>0） | ✅ 净值MACD(12,26)DIF>0 | — |
| 缩量加分（AUM缩水>50%、5-20亿、经理未变、换手正常） | ⚠️ 近5个季度规模史+经理任期+**股票仓位变动<10pp作流动性代理** | 缺少重仓股换手率数据，为保守代理 |
| 动态基准滚动IR胜率（6M窗口/IR>0.3） | ✅ 当期RBSA权重合成动态基准，3年月度步进 | — |
| 动态下行捕获率 | ✅ 基准下跌月份的基金/基准月均跌幅比 | — |
| 滞后截面动量（4M-1M、7M-1M）排名 | ✅ 候选池内分位排名 | 演示池仅13只，全市场跑批时为同类池 |
| 经理任职<3年一票否决 | ✅ （被动指数基金豁免） | 对指数/ETF联接看赛道不看经理 |
| R_MDD>1.5否决 / 1.2~1.5腰斩 | ✅ vs动态基准3年最大回撤 | — |
| 中小盘风格+规模>150亿罚0.3 | ✅ RBSA中小盘/成长暴露>50%判定风格 | — |
| 前三大板块集中>70%且IR胜率<50%罚0.4 | ✅ RBSA前三大权重和 | — |
| F_alpha两子项合成权重 | ⚠️ 原文未给权重，采用 IR胜率50% + 下行捕获50% | 可调参 |
| Point-in-Time 数据库 | ⚠️ PE/净值/指数全PiT；规模史仅近5季，存在轻微幸存者视角 | 回测需接专业PiT库 |

## 五、工程架构

```
quant_fund_picker/
├── config.py      # 全部模型参数（权重/阈值/12因子面板定义）
├── provider.py    # 数据层: 净值/档案/指数/PE + 本地缓存
├── rbsa.py        # Ridge-RBSA穿透 + 估值分位
├── factors.py     # F_value / F_alpha / F_momentum
├── risk.py        # 风控乘数与一票否决
├── engine.py      # 单基流水线 + 截面排名 + 总分合成
├── run_demo.py    # 跑批入口: python run_demo.py [基金代码...]
├── report.py      # 本报告生成器
├── cache/         # 当日数据缓存
└── output/        # scores_YYYYMMDD.csv/json
```

**运行任意基金**: `python run_demo.py 110011 161725 ...`（支持任意代码列表）

## 六、路线图（下一步讨论）

1. **Step 2A — 全市场扫描器**: 用基金排行大数据表做漏斗初筛（类型/规模/任期/动量预排名），
   缩小到 300-500 只再深算，全库一夜跑完。
2. **Step 2B — Web 仪表盘**: 排行榜 + 单基瀑布图 + 估值地形图，本地浏览器打开即用。
3. **Step 2C — 持仓导入与盯市**: 导入你的支付宝基金列表，每日收盘后自动评分，
   评分跨阈值时推送提醒（邮件/Server酱微信）。
4. **Step 2D — 历史回测**: 用 PiT 切片在 2018、2021、2024 三个拐点验证模型信号的胜率和赔率，
   用数据决定参数（如 IR 阈值 0.3、R_MDD 1.5 是否最优）。
5. **微信小程序版**: 需要云开发后端 + 合规考量（个人使用没问题，对外提供"荐基"属持牌业务）。
""")
open("output/report.md", "w", encoding="utf-8").write("\n".join(lines))
print("saved output/report.md")
