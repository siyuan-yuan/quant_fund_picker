# 严格 PIT 基金池：构建、验证与接入手册

> 目标问题：**在每一个历史决策日，当时真实存在、属于目标类型、且投资者当时能够申购的基金有哪些？**
>
> 本仓库已实现 B + C 路线管线（`pit/` 包 + `python -m pit`），并把严格读取接入
> `PITUniverseStore` 与 `backtest_local.py --pit-store`。
> 若采购 Wind/Choice/CSMAR，用同一 schema 导入即可（路线 A）。

---

## 1. 为什么"下载历史净值"≠ 严格 PIT

| 常见做法 | 问题 |
|---|---|
| 用今天的基金主表 + 成立日反推 | 幸存者偏差：清盘基金消失；且今天的 `type/status` 未必是历史值 |
| 用历史并集池 `top100_history_pool.txt` | 提前暴露未来入榜基金（未来函数） |
| 用今天"基金名称/类型"批量抓历史净值 | 名称、类型、申购状态都不是 PIT |
| 只判断"成立 ≤ t" | 缺少类型变更、暂停申购、限购、封闭/到期区间 |

严格 PIT 需要三种**独立**过滤，分别保存、分别审计：

1. **生命周期**（基金池过滤）：`成立 ≤ t < 清盘/到期` —— 可由全历史主表（含已清盘）用
   `found_date / delist_date / due_date` 边界重建（这是事件日期，不是状态快照，不引入未来信息）。
2. **类型**（基金池过滤）：当时的基金类型 —— 只能来自**按期历史截面**（证监会指数）
   或**类型变更事件**；拿今天的 `fund_type` 当历史 = 假定（PIT-lite）。
3. **申购状态**（基金池过滤）：当时能否申购 —— 只能来自**逐日/逐期申赎状态表**或
   **暂停/恢复/大额限购事件**；"日常申购起始日"（Tushare `purc_startdate`）只是首次开放日，
   不是暂停区间，不能当作 PIT。
4. **净值历史长度 / 数据完整度**（**模型可计算性**）：与上面三者分开，
   只标记 `history_ok`，不混入基金池过滤。

## 2. 路线选择（直接抄给决策者）

- **路线 A（最推荐，如果有预算）**：向 Wind / Choice / CSMAR 确认能否
  **按历史日期查询当时存续的全体基金，并包含已经清盘的产品、历史基金分类及历史申购状态**。
  必须书面确认以下四张表可导出，且含"数据发布日/可得日"：
  1. 全历史公募基金主表（含已清盘）：code / 名称 / 类型 / 成立 / 到期 / 清盘 / 份额类别 / 主基金关系
  2. 基金成立、转型、合并、清盘事件（生效日 + 公告日）
  3. 历史基金类型变更（生效日 + 公告日）
  4. 历史申购暂停 / 恢复 / 大额限购（生效日 + 公告日 + 限额值）

  关键问法（不是"有没有基金历史数据"）：
  > 能否按历史日期查询当时存续的全体基金，并包含已经清盘的产品、历史基金分类及历史申购状态？
  > 数据发布日期或数据库可得日期是什么？

  拿到后按第 4 节 schema 导入即可（主表走 `master --import`，类型走证监会索引格式或
  事件表，申赎状态走 `purchase_status` 状态表）。

- **路线 B（低成本，工程量大）**：本仓库已实现
  `Tushare 主表（生命周期） + 证监会历期产品索引（类型/名单截面） + 公告事件（申赎/转型） + 状态表`。
  Tushare `fund_basic` 需要 ≥2000 积分；`status/fund_type` 是当前口径，
  `purc_startdate` 不是暂停区间 —— 这些边界写入 meta 与文档，**不会**冒充严格 PIT。

- **路线 C（保 2026 年以后）**：`python -m pit daily` 每天归档当日真实截面
  （AKShare：`fund_open_fund_daily_em` 含申购/赎回状态 + `fund_name_em` 全集），
  原始响应 + 哈希 + 抓取时间不可变保存。建议 cron 每日执行；
  从今天起的快照是真正 PIT，但**不能修复 2010–2025**。

**默认落地方案：B + C；A 若采购后接同一 schema。** 没有历史申赎状态和类型截面时，
产物自动标记 `PIT-lite`，严格读取默认拒绝。

## 3. 目录与数据模型

```
data/
├── pit_raw/                         # 不可变原始归档（gitignore；绝不覆盖）
│   ├── master/<抓取时间戳>/         # payload.json(原始API响应) + fund_basic.csv + meta.json
│   ├── vendor/<文件名>              # Wind/Choice/CSMAR 导出的原始 CSV
│   ├── csrc/<as_of>/                # 证监会《产品索引》原始 xlsx + meta.json
│   ├── daily/<日期>/                # raw_response.json + universe.csv + meta.json
│   └── purchase_status/<as_of>/     # R0 供应商逐期申赎状态表 + meta.json
├── pit_events/                      # 人工/公告事件表（CSV，见下）
├── pit_manifest/csrc_sources.csv    # 证监会历期索引清单（as_of, published_at, page_url, file_url）
└── pit_universe/                    # 月末不可变快照 + _manifest.json + _qa_report.md
```

事件表（`data/pit_events/*.csv`，UTF-8）：

| 列 | 含义 |
|---|---|
| `event_id` | 唯一ID |
| `code` | 六位基金代码 |
| `event_type` | `inception/liquidation/transform/merge/name_change/type_change/suspend_all/restore_all/suspend_limit:<元>/restore_limit/share_class_add` |
| `effective_date` | 事件**生效日**（回测按 `effective_date <= signal` 应用边界） |
| `known_at` | 公告**可见日**（状态断言必须 `known_at <= signal`） |
| `value / value_prev` | 事件值（如新类型名、限购金额）与旧值 |
| `source / source_file / source_sha256` | 来源与原始件哈希（缺一不可） |
| `confidence / note` | 置信度与备注 |

月末快照列（`data/pit_universe/YYYY-MM-DD.csv`）：

```
code, name, fund_type, fund_type_raw, type_taxonomy, status, inception_date,
share_class, share_class_group, as_of, known_at, source, source_file, source_sha256,
lifecycle_ok, type_ok, purchase_ok, target_ok,
purchase_status, purchase_status_source, purchase_status_basis,
history_ok, history_rows, history_reason,
pit_level, lite_reason, master_fetched_at, build_time
```

- `type_taxonomy`: `fine`（如 混合型-偏股）/ `coarse`（证监会口径 混合型/股票型：会放宽到平衡混合型）
- `purchase_status`: `open | suspend_all | suspend_limit:<单日限额元> | unknown`
- `pit_level`: 整行 `strict` 当且仅当 类型来自按期截面 且 申购状态来自状态表/事件；
  否则 `lite`（`lite_reason` 写明缺什么）。
- 快照整体级别（`_manifest.json`）：全部行 strict 才写 strict。

## 4. 操作手册

```bash
# 0) 初始化目录与证监会清单模板
python -m pit init

# 1) 全历史主表（二选一）
TUSHARE_TOKEN=xxx python -m pit master          # Tushare fund_basic (≥2000积分)
python -m pit master --import wind_master.csv --source wind   # 供应商导出（任意列名）

# 2) 证监会历期产品索引（逐期：list page→manifest）
python -m pit csrc --manifest data/pit_manifest/csrc_sources.csv
# 找到历史期页面：把 {as_of, published_at, page_url, file_url} 逐行补进 manifest。
# published_at 尽量填页面“发文日期”，**绝不**拿下载日期冒充（known_at_approx=True 会标记）。

# 3) 每日真实截面（路线 C；建议 cron）：
python -m pit daily

# 4) 事件表：人工/公告抽取 → data/pit_events/*.csv；校验
python -m pit events

# 5) 物化月末快照（默认严格：无法证明就剔除，绝不回填）
python -m pit build --enforce-qa
# 显式降级为 PIT-lite（只有你能接受时）：
python -m pit build --policy lite --type-policy fallback --purchase-policy unknown-keep

# 6) 质量门禁 & 数据覆盖
python -m pit qa
python -m pit report
```

常用 build 参数：`--markets O/E`、`--slot-capital 10000`（单槽资金，低于限额的"暂停大额申购"视为可投）、
`--min-nav-rows 756 --require-history`（可计算性过滤，默认只标记）、`--start/--end`。

## 5. 质量门禁（G1–G8，`pit/quality.py`）

| 门禁 | 规则 | 级别 |
|---|---|---|
| G1 | 月度基金数跳变 >±10% | 警告 |
| G2 | 快照含成立日在未来的基金 | 致命 |
| G3 | 状态断言 `known_at > signal` | 致命 |
| G4 | 已清盘基金仍出现 / 快照基金不在全历史主表 | 致命 |
| G5 | 同组 A/C/E 重复进入候选池 | 致命 |
| G6 | 基金类型缺失或无法归一 | 致命 |
| G7 | strict 行缺 `source` / `source_sha256` | 致命 |
| G8 | strict 行来自今日截面/当前表回填 | 致命 |

`--enforce-qa` 在致命问题存在时拒绝落盘；每次 build 后自动生成 `_qa_report.md`。

## 6. 严格读取与回测接入

```python
from pit_universe import PITUniverseStore
store = PITUniverseStore("data/pit_universe")          # 默认拒绝 PIT-lite
df, audit = store.universe("2024-12-31", require_history=False)
# df 已含：fund_type/share_class/share_class_group/purchase_status/history_ok/pit_level/source
```

回测：

```bash
python backtest_local.py --pool-mode pit-top --pit-top-n 100 \
  --pit-store data/pit_universe --pit-require-history \
  --score-suffix auto
```

- 候选 = 当日快照成分 ∩ 当日 S 分 TopN；打分宇宙 = 各日快照并集（避免逐日重打分）。
- 快照文件哈希与 `_manifest.json` 不一致 → 拒绝（防篡改/手工改写）。
- `--pit-allow-lite` 是显式"我接受降级"开关，报告会标注，**不算严格 PIT**。

## 7. 常见反模式（=本管线拒绝的行为）

1. 用今天名单回填 2015 年快照 → G8 致命。
2. 把"净值不足最低窗口"和"基金池过滤"混为一列 → 本管线 `history_ok` 独立。
3. 把 Tushare 当前 `fund_type` 当历史类型而不标记 → `type_pit=current-assumed → lite`。
4. 把 `purc_startdate` 当暂停区间 → schema 明确不用于状态。
5. 覆盖旧的证监会 Excel / 抓取文件 → `archive_immutable` 抛错。
6. 只有成立日/清盘日就宣称"严格 PIT" → 快照级别必然是 `lite`，严格读取拒绝。

## 8. 已知边界与下一步

- 证监会历史期索引需人工发现（网站栏目页不提供机器可读列表）；当前脚手架已支持
  `manifest` 清单驱动 + 多期自动解析，`report` 会显示已覆盖期数。
- 公告事件（暂停/恢复/转型）目前是 schema + 校验 + 事件推断，抽取依赖人工或后续
  公告爬虫；没有事件的行在严格模式下被剔除而非猜测。
- 定开/封闭期内的"不能申购"由 `due_date`（封闭到期）+ 申购状态表共同覆盖；
  若主表缺 `due_date`，会把定开基金当作普通开放基金 —— 需要在主表补齐（路线 A 必问字段）。
- 若采购到专业数据库，优先接入 `purchase_status` 逐日状态表与"类型变更事件表"，
  可把 2010–2025 升级为严格 PIT。

## 9. 快速自检

```bash
python -m pytest test_pit.py -v        # 8 项端到端 + 门禁测试
python -m pit report                   # 数据覆盖（主表版本/索引期数/每日截面天数）
python -m pit build --policy lite      # 只有显式选择时才允许 PIT-lite 产物
```
