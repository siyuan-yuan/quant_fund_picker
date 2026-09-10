# Quant Fund Picker V6 Dynamic Alpha and FULL-PIT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 2006-01 至 2026-03 的 FULL-PIT 全基金池上，用历史动态风格暴露重建 alpha，加入基金经营与资金行为信息，并以 raw return、style return、skill alpha 三类目标验证 V3.7 世界观，而不是继续微调 V3.7 参数。

**Architecture:** V6 是一条有硬门控的研究流水线：先完成 FULL-PIT 数据资格与覆盖验收，再产生逐日动态 RBSA 暴露和异常收益路径，随后构造新信息集与多任务目标，最后只用 ElasticNet、LightGBM、分类和排序四类极简基线做 walk-forward OOS 检验。生产 V3.7 在全部研究门通过前保持不变；mixture-of-experts 与 TFT 仅作为条件解锁的后续阶段。

**Tech Stack:** Python 3、pandas 3.0.5、NumPy 2.4.6、scikit-learn 1.9.0、SciPy 1.17.1、statsmodels 0.15.0、pytest 9.1.1；LightGBM 仅在基线阶段通过独立依赖门后引入。

**Spec:** 用户提供的 V6 评审建议 `C:/Users/10941/.codex/attachments/54547666-ad9f-4ad3-88c3-3c960f1f334e/pasted-text.txt`；统计和执行协议继承 `docs/优化计划_V5_重规范_2026-09.md`。

## Global Constraints

- 唯一裁决宇宙为 2006-01 至 2026-03、含已清盘/合并/转型基金的 FULL-PIT；SURV-ADJ 只作迁移对照，不能产生采纳结论。
- 决策日只能使用当日收盘前已知信息；宇宙、基金类型、经理、AUM、费率、换手率和基金公司字段必须带 `known_at <= as_of` 证明。
- 前瞻 h 月标签只用于评估；训练集截止日必须满足 `feature_date <= prediction_date - h months`。
- 执行继续使用复权净值、T+1 成交、QDII T+2 假设、沪深300全收益基准和 V5 阶梯成本。
- 单因子门：月度 Spearman `|IC| > 0.05` 且 Newey-West HAC `t > 2`；fwd1/fwd3/fwd6 的 lag 分别为 1/3/5。
- 因子增量门：相对既有 V3 信息集的截面残差 IC 必须 HAC `t > 2`。
- 模型比较必须报告配对差 HAC t；模型级采纳要求全期与近3年两个窗口内 Calmar 和 MaxDD 双改善、Holm 显著，并报告 DSR 相对变化。
- 所有尝试过的变体，包括失败项，必须进入统一试验台账和 CSCV-PBO 计算；选择指标与阈值必须在运行前冻结。
- 所有结果必须携带口径、样本起止、截面数、缺失率、冷启动起点、代码版本、参数签名和输入/输出 manifest。
- V6 研究期间不得修改 `engine.resolve_weights()`、V3.7 的 `0.40/0.35/0.25` 权重、动量窗口或执行参数。
- `ntuw252` 和 MDD penalty 仅在 FULL-PIT 上按已登记门复核，不得在此之前改入生产模型。
- 当前工作副本没有 `.git`；执行提交步骤应在正式 Git checkout 中运行。若仍无 Git，保留逐任务变更清单与 manifest，禁止伪造 commit id。

---

## File Structure

### New modules

- `v6/contracts.py`：V6 数据列、枚举、目标名称和口径常量的唯一来源。
- `v6/pit_fund_master.py`：历史基金主数据规范化、份额归并和月度 PIT 快照生成。
- `v6/full_pit_validate.py`：死亡基金覆盖、快照因果性、净值连续性和缺口验收。
- `v6/dynamic_rbsa.py`：扩展 Kalman filter、约束投影、暴露路径和动态基准收益。
- `v6/alpha_features.py`：由异常收益路径生成 IR persistence、downside alpha、hit rate、decay、skill volatility 和状态条件 alpha。
- `v6/fund_features.py`：flow、AUM change、flow shock、fee、turnover、manager、family、fund age 的 PIT 特征。
- `v6/targets.py`：raw/style/skill 三类连续、二分类和排序目标。
- `v6/build_panel.py`：合并 FULL-PIT 宇宙、动态 alpha、新信息和目标，输出统一 V6 面板。
- `v6/walk_forward.py`：带标签成熟期和历史填充约束的统一 OOS 训练器。
- `v6/baselines.py`：V3.7、ElasticNet、LightGBM、分类与排序基线定义。
- `v6/evaluate.py`：IC/HAC、top-k、端到端、Holm、DSR、CSCV-PBO 和消融评估。
- `v6/run_stage.py`：按预登记阶段执行、停线和落盘 manifest 的 CLI。
- `tests/test_v6_*.py`：V6 的因果性、数值性质、schema、训练截止和集成回归测试。
- `docs/V6_实验协议与执行台账.md`：不可回写规则的预登记、状态和结论台账。

### Existing files intentionally reused or minimally modified

- `pit_universe.py`：保留 `PITUniverseStore` 作为读取接口，只增加 V6 快照 schema 校验。
- `rbsa.py`：保留当前静态 RBSA 供 A/B 对照；不得直接替换 `rbsa_weights()`。
- `engine.py`：研究期只增加显式 `alpha_mode` 注入点；默认仍为 `static_current_weights`。
- `p1_panel_build.py`：仅改为调用统一 FULL-PIT store/面板入口，移除重复宇宙逻辑。
- `sim_core.py`：继续作为唯一端到端执行器，不复制成交或成本逻辑。
- `stats_hac.py`、`p4_analysis.py`、`v5_manifest.py`：继续作为统计和产物实现；必要时只补可复用 API。
- `_build_ml_panel.py`、`_model_zoo.py`：冻结为 V4/V5 历史复现脚本，V6 不在其上继续堆条件分支。

---

## Stage gates

| Gate | 解锁条件 | 未通过时动作 |
|---|---|---|
| G0 FULL-PIT | 死亡基金净值覆盖率 ≥90%；每月快照因果校验 100%；2006-01 至 2026-03 月份齐全 | 停止全部模型研究，只发布缺口报告 |
| G1 Dynamic RBSA | 合成真值暴露 MAE ≤0.08；权重非负且和为1；未来数据扰动不改变历史路径；静态基准 parity 用例通过 | 保留静态模型，不构建动态 alpha 裁决面板 |
| G2 New information | 新特征 PIT 可得率按月披露；任何待进入模型的特征残差 IC HAC t>2，或作为预登记控制变量明确保留 | 淘汰无增量特征，不调参救援 |
| G3 Simple baselines | 至少一个新目标/信息基线相对 V3.7 的 OOS 配对差及端到端双门通过 | 停止，不做 MoE/TFT |
| G4 Architecture | 简单模型已证明增量且子群误差存在稳定异质性 | 才允许 MoE；序列样本与增量门再通过后才允许 TFT |

---

### Task 1: Freeze the V6 protocol and contracts

**Files:**
- Create: `docs/V6_实验协议与执行台账.md`
- Create: `v6/__init__.py`
- Create: `v6/contracts.py`
- Create: `tests/test_v6_contracts.py`

**Interfaces:**
- Produces: `UniverseLabel`, `AlphaMode`, `TargetKind`, `HORIZON_MONTHS`, `TARGET_LAGS`, `MIN_CROSS_SECTION`, `THIN_CROSS_SECTION`, `DECISION_START`, `DECISION_END`.
- Consumes: V5 §0 的数据、执行、统计和 manifest 协议。

- [ ] **Step 1: 写失败测试，冻结名称和数值**

```python
from v6.contracts import (
    UniverseLabel, AlphaMode, TargetKind, HORIZON_MONTHS,
    TARGET_LAGS, MIN_CROSS_SECTION, THIN_CROSS_SECTION,
)

def test_v6_contract_values_are_frozen():
    assert UniverseLabel.FULL_PIT.value == "FULL-PIT"
    assert AlphaMode.DYNAMIC_PATH.value == "dynamic_path"
    assert TargetKind.SKILL.value == "skill"
    assert HORIZON_MONTHS == (1, 3, 6, 12)
    assert TARGET_LAGS == {1: 1, 3: 3, 6: 5, 12: 11}
    assert MIN_CROSS_SECTION == 30
    assert THIN_CROSS_SECTION == 100
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest tests/test_v6_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'v6'`.

- [ ] **Step 3: 实现最小常量模块，并在实验台账写死假设、门、变体编号和停止规则**

```python
from enum import Enum

class UniverseLabel(str, Enum):
    FULL_PIT = "FULL-PIT"
    SURV_ADJ = "SURV-ADJ"

class AlphaMode(str, Enum):
    STATIC_CURRENT_WEIGHTS = "static_current_weights"
    DYNAMIC_PATH = "dynamic_path"

class TargetKind(str, Enum):
    RAW = "raw"
    STYLE = "style"
    SKILL = "skill"

HORIZON_MONTHS = (1, 3, 6, 12)
TARGET_LAGS = {1: 1, 3: 3, 6: 5, 12: 11}
MIN_CROSS_SECTION = 30
THIN_CROSS_SECTION = 100
DECISION_START = "2006-01-01"
DECISION_END = "2026-03-31"
```

- [ ] **Step 4: 在台账登记且只登记以下首轮配置**

固定动态 RBSA 候选为 `static_60d`、`rolling_ridge_60d`、`kalman_rw`；过程噪声只比较预登记的 `q_scale ∈ {1e-5, 1e-4, 1e-3}`，由 2014 前合成/真实拟合诊断选择一次后锁定。模型候选仅为 V3.7、ElasticNet regression、LightGBM regression、LightGBM binary、LightGBM ranker；不得附加模型动物园。

- [ ] **Step 5: 运行契约测试**

Run: `python -m pytest tests/test_v6_contracts.py -q`

Expected: PASS.

- [ ] **Step 6: 提交预登记，且必须早于任何结果运行**

```bash
git add docs/V6_实验协议与执行台账.md v6/__init__.py v6/contracts.py tests/test_v6_contracts.py
git commit -m "docs: preregister V6 dynamic alpha study"
```

### Task 2: Build and validate the FULL-PIT fund master

**Files:**
- Create: `v6/pit_fund_master.py`
- Create: `v6/full_pit_validate.py`
- Create: `tests/test_v6_fund_master.py`
- Modify: `pit_universe.py`
- Output: `data/v6/fund_master.csv`, `data/pit_universe/*.csv`, `output/v6/g0_full_pit/coverage.csv`, `output/v6/g0_full_pit/validation.md`

**Interfaces:**
- Produces: `normalize_fund_master(raw: DataFrame) -> DataFrame`, `build_monthly_snapshots(master: DataFrame, end_dates: DatetimeIndex) -> dict[Timestamp, DataFrame]`, `validate_full_pit(...) -> ValidationReport`.
- Required columns: `share_code`, `canonical_fund_id`, `name`, `fund_type`, `status`, `inception_date`, `end_date`, `known_at`, `source`.

- [ ] **Step 1: 写份额归并和历史资格失败测试**

```python
def test_share_class_merge_uses_earliest_eligible_share():
    raw = make_master([("000001", "某基金A", "2010-01-01"),
                       ("000002", "某基金C", "2015-01-01")])
    out = normalize_fund_master(raw)
    assert out.canonical_fund_id.nunique() == 1
    assert out.loc[out.share_code.eq("000001"), "is_representative"].item()

def test_delisted_fund_is_present_before_end_and_absent_after():
    snaps = build_monthly_snapshots(delisted_master(),
                                    pd.to_datetime(["2018-05-31", "2018-06-30"]))
    assert "D00001" in set(snaps[pd.Timestamp("2018-05-31")].share_code)
    assert "D00001" not in set(snaps[pd.Timestamp("2018-06-30")].share_code)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_fund_master.py -q`

Expected: FAIL because builder functions do not exist.

- [ ] **Step 3: 实现规范化与月末快照生成**

资格表达式必须等价于：

```python
eligible = (
    (master.inception_date <= as_of)
    & (master.known_at <= as_of)
    & (master.end_date.isna() | (master.end_date >= as_of))
    & master.fund_type.isin(TARGET_TYPES)
    & master.is_representative
)
```

同一基金 A/C/E 份额以规范化名称、基金公司、成立关系和来源映射归并；冲突不得猜测，写入 `merge_conflicts.csv` 并排除出裁决池。

- [ ] **Step 4: 给 `PITUniverseStore` 增加 schema 校验**

读取时强制 `known_at <= snapshot.as_of`、`inception_date <= snapshot.as_of`，并拒绝重复 `canonical_fund_id`。保持 `as_of()` 现有选择“最近且不晚于决策日”的语义。

- [ ] **Step 5: 运行离线验收**

Run: `python -m v6.full_pit_validate --master data/v6/fund_master.csv --nav-root cache --snapshot-root data/pit_universe --out output/v6/g0_full_pit`

Expected: 退出码仅在月份齐全、因果违规数为 0、死亡基金净值覆盖率 ≥90% 时为 0；否则退出码 2，并生成缺口清单。

- [ ] **Step 6: 运行旧 PIT 回归与新测试**

Run: `python -m pytest tests/test_v6_fund_master.py tests/test_research_invariants.py::TestPiTInvariance -q`

Expected: PASS.

- [ ] **Step 7: 写 manifest 并提交**

```bash
python v5_manifest.py output/v6/g0_full_pit --inputs data/v6/fund_master.csv --params '{"stage":"G0","universe":"FULL-PIT"}'
git add v6/pit_fund_master.py v6/full_pit_validate.py pit_universe.py tests/test_v6_fund_master.py docs/V6_实验协议与执行台账.md
git commit -m "feat: build validated FULL-PIT fund universe"
```

### Task 3: Rebuild the FULL-PIT scoring panel before alpha redesign

**Files:**
- Create: `tests/test_v6_full_pit_panel.py`
- Create: `v6/build_panel.py`
- Modify: `p1_panel_build.py`
- Output: `output/v6/panel_base/<date>.parquet`, `output/v6/g0_full_pit/panel_summary.csv`

**Interfaces:**
- Consumes: `PITUniverseStore.as_of(date)` and `engine.score_fund(..., bt=True, pit_meta=...)`.
- Produces: `build_base_panel(decision_dates, store, max_funds=1000, seed_rule="YYYYMMDD") -> DataFrame`.

- [ ] **Step 1: 写未来成员和未来净值篡改不变量测试**

```python
def test_panel_membership_ignores_future_snapshot_and_nav(tmp_path):
    base = build_fixture_panel(tmp_path, tamper_future=False)
    changed = build_fixture_panel(tmp_path, tamper_future=True)
    pd.testing.assert_frame_equal(base, changed)

def test_dead_fund_remains_until_last_disclosure_month(tmp_path):
    panel = build_dead_fund_fixture(tmp_path)
    assert panel.query("date == '2018-05-31'").code.eq("D00001").any()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_full_pit_panel.py -q`

Expected: FAIL because `build_base_panel` is undefined.

- [ ] **Step 3: 实现统一面板入口**

每月使用最后交易日；资格同时满足 PIT 成员、至少 800 条当时可见净值、7 天内披露。截面大于 1000 时用 `np.random.default_rng(int(as_of.strftime('%Y%m%d')))` 固定抽样；保留 `n_eligible`、`n_sampled`、`is_thin`、`universe_label`。

- [ ] **Step 4: 生成静态 V3.7 FULL-PIT 基准面板**

Run: `python -m v6.build_panel base --start 2006-01-01 --end 2026-03-31 --universe FULL-PIT --alpha-mode static_current_weights --max-funds 1000`

Expected: 243 个自然月均有文件；`n<30` 的月保留数据但 `ic_eligible=False`，`30<=n<100` 标记 `is_thin=True`。

- [ ] **Step 5: 对 SURV-ADJ 与 FULL-PIT 做同协议差异分解**

输出逐年基金数、死亡在池数、静态 V3.7 IC/CAGR 差；阈值沿用 V5：`|ΔIC| < 0.01`、`|ΔCAGR| < 1pp` 仅是偏差大小描述，不改变 FULL-PIT 的唯一裁决地位。

- [ ] **Step 6: 测试并提交**

Run: `python -m pytest tests/test_v6_full_pit_panel.py tests/test_research_invariants.py -q`

Expected: 全部 PASS。

```bash
git add v6/build_panel.py p1_panel_build.py tests/test_v6_full_pit_panel.py docs/V6_实验协议与执行台账.md
git commit -m "feat: rebuild base panel on FULL-PIT universe"
```

### Task 4: Implement causal dynamic RBSA exposure paths

**Files:**
- Create: `v6/dynamic_rbsa.py`
- Create: `tests/test_v6_dynamic_rbsa.py`
- Modify: `requirements-v5-lock.txt` only if the chosen implementation requires no already-locked primitive
- Output: `output/v6/g1_dynamic_rbsa/diagnostics.csv`

**Interfaces:**
- Produces: `DynamicRBSAConfig`, `fit_exposure_path(fund_ret, factor_ret, config) -> DataFrame`, `dynamic_benchmark_return(exposures, factor_ret) -> Series`.
- Exposure frame columns equal factor names plus `intercept`, `resid`, `r2_rolling`, `n_obs`; index equals fund return dates.

- [ ] **Step 1: 写常量暴露与风格突变的合成真值测试**

```python
def test_dynamic_rbsa_tracks_known_style_break():
    y, x, truth = synthetic_style_break(seed=20260903, n=800, break_at=400)
    beta = fit_exposure_path(y, x, DynamicRBSAConfig(q_scale=1e-4))
    mae = np.abs(beta[x.columns].iloc[120:] - truth.iloc[120:]).to_numpy().mean()
    assert mae <= 0.08
    assert beta.loc[:truth.index[350], "消费"].mean() > beta.loc[truth.index[500]:, "消费"].mean()
    assert beta.loc[:truth.index[350], "科技"].mean() < beta.loc[truth.index[500]:, "科技"].mean()
```

- [ ] **Step 2: 写约束与因果性失败测试**

```python
def test_exposure_path_is_long_only_and_sums_to_one():
    beta = fit_fixture()
    w = beta[FACTOR_NAMES].dropna()
    assert (w >= -1e-12).all().all()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-8)

def test_future_return_tamper_cannot_change_past_exposure():
    y, x = fixture_returns()
    b0 = fit_exposure_path(y, x, CFG)
    y.loc[y.index > "2020-12-31"] *= -7
    b1 = fit_exposure_path(y, x, CFG)
    pd.testing.assert_frame_equal(b0.loc[:"2020-12-31"], b1.loc[:"2020-12-31"])
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_dynamic_rbsa.py -q`

Expected: FAIL because dynamic RBSA APIs do not exist.

- [ ] **Step 4: 实现 filter-only 状态空间估计**

状态方程为 `beta_t = beta_{t-1} + eta_t`，观测方程为 `fund_ret_t = intercept_t + factors_t @ beta_t + epsilon_t`。每个时点更新后把风格权重投影到概率单纯形；不得使用 Rauch–Tung–Striebel smoother，因为它会让未来收益影响历史 beta。缺失因子日跳过观测更新但保留预测步；少于 120 个有效日返回 warm-up NaN。

- [ ] **Step 5: 冻结 q_scale**

只在预登记合成数据和 2006-2013 诊断段比较 `{1e-5,1e-4,1e-3}`，选择规则为先满足 MAE≤0.08，再最小化一步预测误差；平局取更小 q_scale。选择后写入台账，2014-2026 OOS 不得改值。

- [ ] **Step 6: 增加静态对照 parity 测试**

`dynamic_benchmark_return()` 在传入逐日重复的 `rbsa.rbsa_weights()` 时，必须与当前 `engine.py:122` 的 `(idx_ret[w.index] * w).sum(axis=1)` 在 `1e-12` 内一致。

- [ ] **Step 7: 运行 G1 测试并提交**

Run: `python -m pytest tests/test_v6_dynamic_rbsa.py tests/test_research_invariants.py::TestPiTInvariance -q`

Expected: PASS，且 diagnostics 明确列出 MAE、单纯形违规数、未来篡改差异最大值和静态 parity 差异。

```bash
git add v6/dynamic_rbsa.py tests/test_v6_dynamic_rbsa.py docs/V6_实验协议与执行台账.md
git commit -m "feat: add causal dynamic RBSA exposure paths"
```

### Task 5: Reconstruct abnormal returns and alpha features

**Files:**
- Create: `v6/alpha_features.py`
- Create: `tests/test_v6_alpha_features.py`
- Modify: `engine.py`
- Output: `output/v6/alpha_paths/<code>.parquet`, `output/v6/g1_dynamic_rbsa/static_vs_dynamic.csv`

**Interfaces:**
- Produces: `build_abnormal_returns(fund_ret, factor_ret, exposures) -> DataFrame` and `alpha_feature_snapshot(abnormal_ret, fund_ret, benchmark_ret, as_of) -> dict[str, float]`.
- `engine.score_fund(..., alpha_mode=AlphaMode.STATIC_CURRENT_WEIGHTS)` retains current default and accepts precomputed dynamic alpha features only when explicitly requested.

- [ ] **Step 1: 写异常收益恒等式和 as-of 不变量测试**

```python
def test_abnormal_return_identity():
    out = build_abnormal_returns(FUND_RET, FACTOR_RET, EXPOSURES)
    assert np.allclose(out.fund_ret, out.benchmark_ret + out.abnormal_ret, atol=1e-12)

def test_alpha_snapshot_uses_only_history_through_as_of():
    a0 = alpha_feature_snapshot(AR, FUND, BENCH, "2020-12-31")
    a1 = alpha_feature_snapshot(tamper_after(AR, "2020-12-31"), FUND, BENCH, "2020-12-31")
    assert a0 == a1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_alpha_features.py -q`

Expected: FAIL because alpha feature functions are undefined.

- [ ] **Step 3: 实现六类动态 alpha 特征**

固定定义如下：`ir_persistence_3y` 复用 `rolling_ir_winrate`；`downside_alpha_3y` 为基准下跌月异常收益均值年化；`alpha_hit_rate_3y` 为月异常收益大于 0 的比例；`alpha_decay_12v36` 为近12月异常收益均值减过去36月均值；`skill_vol_3y` 为月异常收益标准差年化；`state_alpha_low/high` 按当时 market water ≤0.35 与 >0.35 分组计算。全部窗口在 `as_of` 截断。

- [ ] **Step 4: 给引擎增加显式模式，不替换默认行为**

```python
def score_fund(code, as_of=None, bt=False, pit_meta=None, indices=None,
               alpha_mode="static_current_weights", alpha_snapshot=None):
    ...
```

`dynamic_path` 模式若没有 `alpha_snapshot` 必须抛出 `ValueError`，禁止静默回退静态 alpha；默认模式输出与修改前逐位相同。

- [ ] **Step 5: 比较静态和动态测量模型**

在同一 FULL-PIT 面板报告：暴露换手、rolling R²、残差自相关、F_alpha 年度/子群 IC、2024/2025 符号、downside component 差异。此步骤回答“体制敏感”与“测量错设”哪种解释更符合证据，但不得直接改生产权重。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/test_v6_alpha_features.py tests/test_research_invariants.py::TestPiTInvariance -q`

Expected: PASS，静态默认 parity 为 0 差异。

```bash
git add v6/alpha_features.py engine.py tests/test_v6_alpha_features.py docs/V6_实验协议与执行台账.md
git commit -m "feat: derive alpha from dynamic benchmark paths"
```

### Task 6: Add point-in-time fund flow and operating features

**Files:**
- Create: `v6/fund_features.py`
- Create: `tests/test_v6_fund_features.py`
- Output: `data/v6/fund_observations.parquet`, `output/v6/g2_features/coverage.csv`

**Interfaces:**
- Produces: `build_fund_features(observations, nav, as_of) -> DataFrame`.
- Input observation schema: `canonical_fund_id`, `report_date`, `known_at`, `aum`, `shares`, `expense_ratio`, `turnover`, `manager_id`, `manager_start`, `family_id`.
- Output includes `flow_1q`, `aum_change_1q`, `flow_shock_z`, `flow_vol_1y`, `expense_ratio`, `turnover_1y`, `manager_change_1y`, `manager_tenure_days`, `family_aum`, `fund_age_days`, and per-field `*_known_at`.

- [ ] **Step 1: 写 flow 恒等式和报告日/可得日测试**

```python
def test_flow_removes_investment_return_from_aum_change():
    out = build_fund_features(obs(aum0=100, aum1=120), nav_return=0.10, as_of="2020-05-01")
    assert out.flow_1q == pytest.approx(120 - 100 * 1.10)

def test_feature_is_unavailable_before_known_at():
    assert build_fund_features(OBS, NAV, as_of="2020-04-29").turnover_1y.isna().all()
    assert build_fund_features(OBS, NAV, as_of="2020-04-30").turnover_1y.notna().all()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_fund_features.py -q`

Expected: FAIL because feature builder is undefined.

- [ ] **Step 3: 实现 PIT as-of join 和特征公式**

`known_at` 是 join 的唯一可得时间；`report_date` 只描述经济期间，不可替代发布日期。flow 采用 `AUM_t - AUM_{t-1} * (1 + fund_return)`；flow shock 用同类基金当期截面 median/MAD 标准化；经理变更只在变更公告 known_at 后置 1。

- [ ] **Step 4: 输出覆盖和缺失机制报告**

按年、基金类型、存续/清盘状态报告每列可得率。缺失指示器可进入模型；数值填充仍只允许当期截面中位数后接历史扩展窗中位数。

- [ ] **Step 5: 做未来公告篡改测试**

将 `known_at > as_of` 的 AUM、经理和费率放大 100 倍，所有 as-of 特征必须逐位不变。

- [ ] **Step 6: 运行测试、写 manifest 并提交**

Run: `python -m pytest tests/test_v6_fund_features.py -q`

Expected: PASS.

```bash
git add v6/fund_features.py tests/test_v6_fund_features.py docs/V6_实验协议与执行台账.md
git commit -m "feat: add PIT fund flow and operating features"
```

### Task 7: Build raw, style and skill targets

**Files:**
- Create: `v6/targets.py`
- Create: `tests/test_v6_targets.py`
- Output: target columns in `output/v6/panel/v6_panel.parquet`

**Interfaces:**
- Produces: `build_targets(fund_adj_nav, factor_total_return, future_exposure_policy, decision_dates, horizons) -> DataFrame`.
- Columns per horizon h: `y_raw_h`, `y_style_h`, `y_skill_h`, `y_outperform_h`, `y_skill_rank_h`, `target_end_h`.

- [ ] **Step 1: 写三目标分解和标签成熟测试**

```python
def test_target_decomposition():
    t = build_fixture_targets(horizon=6)
    assert np.allclose(t.y_raw_6, t.y_style_6 + t.y_skill_6, atol=1e-12)
    assert t.y_outperform_6.eq(t.y_skill_6 > 0).all()

def test_target_end_is_strictly_after_decision_date():
    t = build_fixture_targets(horizon=6)
    assert (t.target_end_6 > t.date).all()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_targets.py -q`

Expected: FAIL because target builder is undefined.

- [ ] **Step 3: 实现目标定义并冻结未来暴露政策**

`y_raw_h = fund total return`；`y_style_h` 为预测起点时已知 beta 冻结后作用于未来指数总收益的组合收益；`y_skill_h = y_raw_h - y_style_h`。同时额外生成一组仅作归因诊断的 realized-path skill（用未来逐日 filter beta），明确禁止作为可交易标签，以避免把未来路径知识混入训练定义。

- [ ] **Step 4: 生成分类和截面排序标签**

`y_outperform_h = 1[y_skill_h > 0]`；`y_skill_rank_h` 为同月、同 horizon 的百分位秩。薄截面月仍保留 raw columns，但 rank/IC eligibility 遵循 30 只门槛。

- [ ] **Step 5: 对终止基金使用最后可得净值而非删除整行**

若目标期内清盘，使用清盘/合并条款可验证的终值；无法验证时将该标签标记 `target_status="terminal_value_missing"`，不得以前向存续过滤删除特征行。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/test_v6_targets.py tests/test_research_invariants.py::TestMLPanelRules -q`

Expected: PASS.

```bash
git add v6/targets.py tests/test_v6_targets.py docs/V6_实验协议与执行台账.md
git commit -m "feat: add raw style and skill prediction targets"
```

### Task 8: Assemble and audit the V6 research panel

**Files:**
- Modify: `v6/build_panel.py`
- Create: `tests/test_v6_panel_audit.py`
- Output: `output/v6/panel/v6_panel.parquet`, `output/v6/panel/schema.json`, `output/v6/panel/audit.csv`

**Interfaces:**
- Produces: `assemble_v6_panel(base, alpha, fund_features, targets) -> DataFrame`, `audit_panel(panel) -> PanelAudit`.
- Primary key: `(date, canonical_fund_id)`.

- [ ] **Step 1: 写主键、时间、宇宙和恒等式失败测试**

```python
def test_v6_panel_has_unique_primary_key(panel):
    assert not panel.duplicated(["date", "canonical_fund_id"]).any()

def test_all_feature_timestamps_are_known_by_decision(panel):
    known_cols = [c for c in panel if c.endswith("_known_at")]
    assert all((panel[c].isna() | (panel[c] <= panel.date)).all() for c in known_cols)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_panel_audit.py -q`

Expected: FAIL before assembler exists.

- [ ] **Step 3: 实现确定性拼接与 schema**

禁止 many-to-many merge；每次 join 使用 `validate="one_to_one"`。schema.json 记录 dtype、经济含义、来源、known-at 字段、允许缺失原因和是否可用于训练。

- [ ] **Step 4: 运行面板审计**

Run: `python -m v6.build_panel assemble --base output/v6/panel_base --alpha output/v6/alpha_paths --fund-features data/v6/fund_observations.parquet --out output/v6/panel`

Expected: 主键重复 0、未来可得违规 0、raw=style+skill 误差最大值 ≤1e-12、FULL-PIT 标签占比 100%。

- [ ] **Step 5: 复核 ntuw252 与 MDD penalty**

只在此面板执行既有预登记：ntuw252 报 fwd1/3/6 自身 IC 和相对 V3 因子残差 IC；MDD 报保留/移除两臂的配对 IC-HAC 与 `sim_core` 双窗端到端。结果不与模型搜索混在同一 family，全部计入台账。

- [ ] **Step 6: 测试、manifest、提交**

Run: `python -m pytest tests/test_v6_panel_audit.py tests/test_v6_*.py -q`

Expected: PASS.

```bash
git add v6/build_panel.py tests/test_v6_panel_audit.py docs/V6_实验协议与执行台账.md
git commit -m "feat: assemble audited V6 research panel"
```

### Task 9: Implement leakage-safe walk-forward training

**Files:**
- Create: `v6/walk_forward.py`
- Create: `tests/test_v6_walk_forward.py`

**Interfaces:**
- Produces: `WalkForwardSpec`, `walk_forward_predict(panel, estimator_factory, feature_cols, target_col, spec) -> DataFrame`.
- Prediction output: `date`, `canonical_fund_id`, `pred`, `train_end`, `n_train_months`, `model_id`, `target_col`.

- [ ] **Step 1: 写逐 horizon 训练截止测试**

```python
@pytest.mark.parametrize("h,target", [(1,"y_skill_1"),(3,"y_skill_3"),(6,"y_skill_6"),(12,"y_skill_12")])
def test_train_end_respects_target_maturity(h, target):
    pred = run_recorder(target)
    assert (pred.train_end <= pred.date - pd.DateOffset(months=h)).all()
```

- [ ] **Step 2: 写填充和标准化只拟合训练集的测试**

把预测月特征乘以 1000，训练期 imputer/scaler 的统计量必须不变；首个无历史中位数的缺失值必须保留并使该行不可预测，而不是读取全样本中位数。

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_walk_forward.py -q`

Expected: FAIL because trainer is undefined.

- [ ] **Step 4: 实现 expanding-window trainer**

月频预测、每季度重训；最少 36 个已成熟训练月。imputation 顺序为预测月截面中位数、训练历史扩展窗中位数；模型 pipeline 内的 scaler 只 fit 训练行。所有随机模型固定 seed `20260903`。

- [ ] **Step 5: 输出可审计训练元数据**

每个预测月保存 feature list、target、train_start/end、训练行数、缺失率、estimator params；同一输入和 seed 重跑 prediction hash 必须相同。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/test_v6_walk_forward.py tests/test_research_invariants.py::TestMLPanelRules -q`

Expected: PASS.

```bash
git add v6/walk_forward.py tests/test_v6_walk_forward.py
git commit -m "feat: add leakage-safe V6 walk-forward trainer"
```

### Task 10: Run minimal models, targets and ablations

**Files:**
- Create: `v6/baselines.py`
- Create: `tests/test_v6_baselines.py`
- Modify: `requirements-v5-lock.txt`
- Output: `output/v6/g3_baselines/predictions/*.parquet`, `output/v6/g3_baselines/metrics.csv`, `output/v6/g3_baselines/ablations.csv`

**Interfaces:**
- Produces: `make_baseline(model_id, task, seed=20260903)` and registered feature groups `V3`, `DYNAMIC_ALPHA`, `FUND_OPERATING`, `ALL_V6`.
- Consumes: `walk_forward_predict()` and V6 target names.

- [ ] **Step 1: 写模型注册表和固定参数测试**

```python
def test_only_preregistered_baselines_exist():
    assert set(MODEL_REGISTRY) == {
        "v37", "elasticnet_reg", "lgbm_reg", "lgbm_binary", "lgbm_rank"
    }

def test_all_stochastic_models_use_fixed_seed():
    for key in ("lgbm_reg", "lgbm_binary", "lgbm_rank"):
        assert make_baseline(key, task_for(key)).get_params()["random_state"] == 20260903
```

- [ ] **Step 2: 先检测 LightGBM 可安装性**

Run: `python -c "import lightgbm; print(lightgbm.__version__)"`

Expected: 输出锁定版本。若模块不存在，仅在正式环境执行 `python -m pip install lightgbm==4.6.0`，然后把精确版本加入 `requirements-v5-lock.txt`；安装失败则记录 `dependency_blocked`，ElasticNet 仍可完成 G3 的线性基线，但不得拿缺失的 LightGBM 作负面结论。

- [ ] **Step 3: 实现固定基线**

ElasticNet 固定 `alpha=0.002, l1_ratio=0.5, max_iter=5000`；LightGBM 固定 `n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=20260903`。ranker 按决策月提供 group sizes；不做 Bayesian search。

- [ ] **Step 4: 运行预登记矩阵**

主 horizon 为 6 个月，目标为 `raw/style/skill` 回归、`skill>0` 分类、`skill_rank` 排序；辅助 horizon 1/3/12 只作方向稳定性报告。每个任务依次比较 V3、V3+dynamic alpha、V3+fund operating、ALL_V6，形成信息集消融，不额外改变模型参数。

- [ ] **Step 5: 运行单因子与残差 IC 门**

每个新特征先相对 V3 feature ranks 做当月截面 OLS 残差，再计算月度 Spearman IC 与 HAC t。未过门特征从候选组移除，但基础控制变量的保留理由必须在预登记中已注明。

- [ ] **Step 6: 运行测试和 baseline CLI**

Run: `python -m pytest tests/test_v6_baselines.py tests/test_v6_walk_forward.py -q`

Run: `python -m v6.run_stage baselines --panel output/v6/panel/v6_panel.parquet --horizon 6 --out output/v6/g3_baselines`

Expected: 每个预登记单元都有预测或明确的 blocked 状态，无未登记模型 id。

- [ ] **Step 7: manifest 并提交**

```bash
git add v6/baselines.py tests/test_v6_baselines.py requirements-v5-lock.txt docs/V6_实验协议与执行台账.md
git commit -m "feat: evaluate minimal V6 prediction baselines"
```

### Task 11: Evaluate prediction and end-to-end portfolio value

**Files:**
- Create: `v6/evaluate.py`
- Create: `tests/test_v6_evaluate.py`
- Modify: `sim_core.py` only if a generic precomputed-score adapter is absent
- Output: `output/v6/g3_baselines/evaluation.md`, `output/v6/g3_baselines/trial_registry.csv`, `output/v6/g3_baselines/pbo.csv`

**Interfaces:**
- Produces: `evaluate_predictions(actual, predictions, spec) -> EvaluationBundle`, `run_score_backtest(scores, sim_config) -> BacktestResult`, `gate_decision(bundle, baseline) -> GateDecision`.

- [ ] **Step 1: 写配对比较和严格门测试**

```python
def test_gate_rejects_cagr_only_improvement():
    candidate = metrics(cagr=0.12, calmar=0.40, maxdd=-0.30, holm_p=0.01)
    baseline = metrics(cagr=0.10, calmar=0.45, maxdd=-0.20, holm_p=0.01)
    assert gate_decision(candidate, baseline).accepted is False

def test_failed_trials_are_counted_in_registry():
    registry = build_trial_registry([passed_trial(), failed_trial()])
    assert len(registry) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_evaluate.py -q`

Expected: FAIL because evaluator is undefined.

- [ ] **Step 3: 实现预测层指标**

报告月度 Spearman IC/HAC t、配对差/HAC t、top-10 hit rate、top decile-minus-bottom decile、binary AUC/Brier、ranking NDCG@10；全部按全期、2019+、近3年、主动/被动、A股/海外分组，子群只作预登记异质性诊断。

- [ ] **Step 4: 通过 `sim_core` 做唯一端到端模拟**

所有模型分数逐月分位映射到当月 V3.7 分数分布后进入同一执行器，保持 slots、买卖线、风控、成本和成交日完全一致。不得为某模型单独调买卖阈值。

- [ ] **Step 5: 实现多重检验**

更新包含 V5 历史失败项和 V6 全部尝试的 N；报告 DSR、Holm/BH、CSCV-PBO，CSCV 同时使用 `g={8,12,16}` 和 mean/Calmar 两个预登记选择规则。不得删掉负结果降低 N。

- [ ] **Step 6: 执行 G3 裁决**

只有候选在全期与近3年同时 Calmar 上升、MaxDD 改善，相关配对检验 Holm 显著，且 DSR 相对提升时标记 `accepted_for_architecture_research=True`。否则结论是维持 V3.7、保留研究产物、停止 Task 12。

- [ ] **Step 7: 运行测试、生成报告并提交**

Run: `python -m pytest tests/test_v6_evaluate.py tests/test_research_invariants.py -q`

Run: `python -m v6.run_stage evaluate --pred-root output/v6/g3_baselines/predictions --panel output/v6/panel/v6_panel.parquet --out output/v6/g3_baselines`

Expected: 测试 PASS，报告中每个结论可追溯到 trial id、输入 hash 和复现命令。

```bash
git add v6/evaluate.py tests/test_v6_evaluate.py sim_core.py docs/V6_实验协议与执行台账.md
git commit -m "feat: gate V6 models with OOS and portfolio evidence"
```

### Task 12: Conditionally test hierarchical experts; defer TFT

**Files:**
- Create only after G3 passes: `v6/experts.py`
- Create only after G3 passes: `tests/test_v6_experts.py`
- Output: `output/v6/g4_experts/*`

**Interfaces:**
- Produces: `SoftHierarchicalModel(global_estimator, group_col, shrinkage)`, with `fit(X, y, groups)` and `predict(X, groups)`.
- Consumes: exact winning information set and target from G3; no new features or hyperparameter search.

- [ ] **Step 1: 检查硬解锁条件**

Run: `python -m v6.run_stage gate --stage G3 --report output/v6/g3_baselines/evaluation.md`

Expected: 仅 `accepted_for_architecture_research=True` 时继续；否则退出码 3，本任务结束且不创建代码文件。

- [ ] **Step 2: 写 partial-pooling 行为测试**

```python
def test_sparse_group_prediction_shrinks_toward_global():
    model = fitted_sparse_group_model()
    pred_group = model.predict(X_ONE, groups=["overseas_active"])[0]
    pred_global = model.global_estimator.predict(X_ONE)[0]
    assert abs(pred_group - pred_global) < abs(unpooled_prediction() - pred_global)
```

- [ ] **Step 3: 实现四组 soft hierarchy**

固定 group 为 `active_ashare`、`passive_ashare`、`active_overseas`、`passive_overseas`；每组系数等于 global coefficients 加 group delta，delta 使用固定 ridge shrinkage。不得为每组独立搜索模型。

- [ ] **Step 4: 用 Task 11 完全相同的门评估**

若未超过简单 G3 胜者，停止架构研究。只有 soft hierarchy 再过门，且每月有效训练样本中位数 ≥1000、有效序列长度 ≥120 月，才另写 TFT 预登记计划；本计划不实现 TFT。

- [ ] **Step 5: 测试并提交**

Run: `python -m pytest tests/test_v6_experts.py -q`

Expected: PASS.

```bash
git add v6/experts.py tests/test_v6_experts.py docs/V6_实验协议与执行台账.md
git commit -m "experiment: test partial-pooling fund experts"
```

### Task 13: Publish the V6 decision package and start true forward OOS

**Files:**
- Create: `docs/V6_终版裁决报告.md`
- Create: `v6/paper_log.py`
- Create: `tests/test_v6_paper_log.py`
- Modify: `README.md`
- Output: `output/v6/final/*`, `output/v6/paper/<decision_date>.json`

**Interfaces:**
- Produces: immutable `PaperDecision` with `decision_date`, `universe_hash`, `model_hash`, ordered `top_n`, `scores`, `created_at`.

- [ ] **Step 1: 写 append-only 和模型变更重置测试**

```python
def test_paper_decision_cannot_be_overwritten(tmp_path):
    write_paper_decision(DECISION, tmp_path)
    with pytest.raises(FileExistsError):
        write_paper_decision(DECISION, tmp_path)

def test_model_hash_change_starts_new_vintage():
    assert vintage_id("model-a", "2026-10-31") != vintage_id("model-b", "2026-10-31")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v6_paper_log.py -q`

Expected: FAIL because paper log APIs are undefined.

- [ ] **Step 3: 实现不可覆盖纸面记录**

每个自然决策月只写一次，包含当时 Top-N 与分数、模型/宇宙 hash；记录生成后 12 个月内不得回写结果字段，成熟后另写 evaluation 文件。任何生产参数变化都开新 vintage，旧观察期不拼接。

- [ ] **Step 4: 编写终版裁决报告**

逐项回答：FULL-PIT 与 SURV 差多少；动态 RBSA 是否改变 2024/2025 alpha 结论；raw/style/skill 哪个可预测；收益来自赛道还是同赛道选基；flow/AUM/manager 等谁有独立增量；简单模型是否过 G3；ntuw252 与 MDD 是否改判；统计证据属于成立、相对最优还是弱-中等。

- [ ] **Step 5: 更新 README，仅发布获准口径**

任何收益数字带 `{FULL-PIT/SURV-ADJ, 窗口, 复现命令, 已知缺口}`；若 G3 未过，生产章节继续写 V3.7，V6 只作为研究否决/无区分结论。

- [ ] **Step 6: 运行全套验证**

Run: `python -m pytest -q`

Expected: 全部 PASS，无 xfail 用于隐藏 V6 核心门。

Run: `python -m v6.run_stage final --root output/v6 --out output/v6/final`

Expected: 验证所有阶段 manifest、试验台账计数和报告引用；缺任一证据时退出码 2。

- [ ] **Step 7: 提交终裁与前瞻日志工具**

```bash
git add docs/V6_终版裁决报告.md v6/paper_log.py tests/test_v6_paper_log.py README.md docs/V6_实验协议与执行台账.md
git commit -m "docs: publish V6 decision and forward OOS protocol"
```

---

## Execution order and stopping rules

1. Task 1 必须单独提交，形成可外验的预登记时间点。
2. Tasks 2-3 完成 G0；G0 不过，Tasks 4-13 全部冻结。
3. Tasks 4-5 完成 G1；G1 不过，只发布动态 RBSA 失败诊断，不改 `engine.py` 默认行为。
4. Tasks 6-8 完成新信息与新目标底座；任何数据列如果无法证明 `known_at <= as_of`，不得进入训练。
5. Tasks 9-11 是唯一第一轮模型实验；不得追加模型、窗口、权重或执行参数来“救”结果。
6. Task 12 只有 G3 通过才执行；TFT 明确不属于本计划的实现范围。
7. Task 13 无论正负都执行，发布完整证据链并启动真正的前瞻 OOS。

## Definition of done

- FULL-PIT G0 四项证据齐全：历史主表、月度快照、死亡基金覆盖、因果验收。
- 动态 RBSA G1 四项证据齐全：合成 MAE、单纯形约束、未来篡改不变量、静态 parity。
- V6 面板的主键、known-at、标签分解和训练成熟期测试全部通过。
- 三种经济目标与四种学习任务均有 OOS 结果；每个结果都能追溯到已登记 trial。
- 信息集消融能区分 V3、动态 alpha、fund operating 三类增量来源。
- ntuw252 与 MDD penalty 在 FULL-PIT 上得到明确的按门维持/改判结论。
- 所有失败试验进入 DSR/CSCV/Holm/BH 台账，不因负结果删除。
- 生产 V3.7 只在模型级双门和多重检验门全部通过后才允许另立部署计划；本计划本身不授权部署。
- `python -m pytest -q` 通过，终版报告、manifest 和复现命令一致。

## Deliberately excluded

- 继续搜索 V3.7 因子权重、动量窗口、买卖线、CPPI 档位或更多 sklearn 模型。
- 在 FULL-PIT、动态 benchmark 和新信息集完成前运行 TFT、MLP 或大规模超参优化。
- 用 realized future exposure path 作为可交易 skill target。
- 用 SURV-ADJ 的正结果替代 FULL-PIT 裁决。
- 在研究通过前直接替换生产 `F_alpha` 或删除 MDD penalty。
