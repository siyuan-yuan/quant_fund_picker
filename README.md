# 量化选基系统 V3.1

基于《量化选基系统核心打分模型_V3》的可执行实现，附 Web 控制台。

> ⚠️ 仅供量化研究，不构成投资建议。

## 快速开始

```bash
pip install akshare pandas numpy scikit-learn flask
cd quant_fund_picker
python webapp.py
```

浏览器打开 **http://127.0.0.1:8000**。

## 严格 Point-in-Time 动态可投池 / 严格复盘回测

`backtest_local.py` 现已改为**严格模式**，不会再把今天的 `fund_meta`、`rank_all.csv`、旧研究池或自选静态代码池倒灌到历史。每一个信号月会：

1. 只读取当时已经可得的历史基金名录快照；每月独立建池，历史清盘基金也必须保留在快照中；
2. 按该快照中的历史名称、基金类型判定策略范围/被动属性，阻断“今天的分类”穿越；
3. 仅用 `as_of` 当日及以前的净值、指数、估值计算因子；
4. 月末信号不允许同日净值成交。默认延迟 **2 个真实交易日**（T+1 披露净值，下一交易日按未知净值成交）；
5. 输出 `output/bt_pit_audit.json`，逐月记录使用的快照、快照时点、可得日期、成员数、评分数和快照指纹，以便复核。

### 历史快照是严格回测的必要输入

将不可变、UTF-8 的 CSV 快照放入 `data/pit_universe/`，文件名带日期，例如：

```text
data/pit_universe/2018-01-31.csv
data/pit_universe/2018-02-28.csv
```

每份快照至少须有 `code`（或 `基金代码`）和 `fund_type`（或 `基金类型`）。推荐完整字段：

```csv
code,name,fund_type,status,inception_date,as_of,known_at
000001,XX成长混合A,混合型-偏股,正常,2010-01-01,2018-01-31,2018-01-31
```

- `as_of`：名录实际截面日；`known_at`：该信息在市场上可获得的日期，二者都不得晚于信号日。
- 快照必须包括当时存在、后来清盘的基金；否则仍有幸存者偏差，程序不会替你用当前名录“补齐”。
- `status` 若明确写有清盘、终止、暂停申购等，自动排除。C/E 重复份额自动剔除。
- 没有快照目录、快照早于首个信号日不可得、或没有历史基金类型时，程序会**拒绝运行**，而不是退回非 PiT 口径。

运行：

```bash
python backtest_local.py --universe-dir data/pit_universe \
  --start 2018-01-31 --end 2025-12-31 --rebuild
# 调整披露/成交假设；不得小于 1，默认 2
python backtest_local.py --universe-dir data/pit_universe --execution-lag 2
```

`--codes` 已在严格模式禁用，因为固定的、今天才挑出的自选池会破坏动态可投性。输出包括交易账、日净值、汇总报告、净值图，以及 PiT 审计文件。

### 仍需披露的边界

历史基金经理任期、AUM 与申赎状态若没有逐日档案，不能通过当前网页档案回填；回测中这些非 PiT 惩罚已豁免。严格的“无幸存者偏差”取决于输入快照是否完整保留清盘基金，系统会审计来源但无法凭空恢复缺失历史数据。

## 模型公式

```
S_total = (0.40×min(F_value,100) + 0.35×F_alpha + 0.25×F_momentum) × Π(1−惩罚率)
评级: 85+ StrongBuy | 70+ Buy | 50+ Hold | <50 Sell/Avoid
```

## 结构

```
config.py          模型参数/12因子面板定义
provider.py        数据层（天天基金/乐咕/中证/雪球，本地缓存）
pit_universe.py    不可向未来偷看的历史动态可投池快照读取/审计
engine.py          单基流水线 + 截面排名 + 总分合成
backtest_local.py  严格 PiT 动态建池、延迟成交、日频复盘
scan_market.py     实时全市场扫描器（不可用于历史严格回测）
webapp.py          Web 控制台
output/            回测/扫描结果
```

## 实时扫描与页面

实时扫描仍使用当前市场名单，适用于当日研究；它与严格历史回测是两条数据路径，不能混用。页面可启动全市场扫描、单基金透视和持仓批算；缓存以交易日刷新。
