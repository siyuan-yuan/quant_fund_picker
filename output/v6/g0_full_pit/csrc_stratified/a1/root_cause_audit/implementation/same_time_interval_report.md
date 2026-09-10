# A1 同时点区间修复（round 1）

## 结论

`build_checkpoint_timeline` 现在按 family 收集全部可用状态的 `effective_from`，对排序后的唯一时点建立“下一严格时点”映射，再将同一边界赋给该时点的全部记录。边界不再依赖 `records` 的位置或 `known_at` 排序。

所有 checkpoint 均保留；`UNRESOLVED` 和 `NaT effective_from` 不参与边界映射。同一时点的兼容区间收紧、类型或区间冲突、`VERIFIED_NOOP` 的显式前序继承及 causal flag 语义均未改变。

## 根因与审查修正

旧实现按 `(known_at, effective_date, checkpoint_id)` 排序后，只在当前记录之后寻找后继。上一轮增加 `successor.effective_from > record.effective_from` 只能跳过同时点记录，不能保证找到全 family 中最近的严格后继：当 `effective_from` 顺序与 `known_at` 顺序不同，较早时点可能位于较晚时点之后，产生空终点或越过最近边界。

两条测试在生产修改前真实 RED：

- `known_at` 先出现但 `effective_from` 较晚时，较早时点得到 `NaT`，预期为下一严格时点。
- `effective_from` 呈 5 月、4 月、6 月顺序时，4 月错误映射到 6 月，预期为 5 月。

因此，旧报告中的“一行条件足以修复”结论及“全部 V6 共 239 项”证据不足，已由本轮真实测试替代。

## TDD 与回归证据

- RED：2 failed，72 deselected；原始输出 `same_time_interval_round1_red.log`，JUnit `same_time_interval_round1_red.xml`。
- focused GREEN：9 passed，65 deselected；原始输出 `same_time_interval_round1_focused.log`，JUnit `same_time_interval_round1_focused.xml`。
- 完整 `tests/test_v6_classification_checkpoints.py`：74 passed；原始输出 `same_time_interval_round1_file.log`，JUnit `same_time_interval_round1_file.xml`。
- 完整 `tests/test_v6_*.py`：241 passed；原始输出 `same_time_interval_round1_v6.log`，JUnit `same_time_interval_round1_v6.xml`。

## 冻结 v2 只读内存探针

可复现探针的完整 Python 源码和真实输出保存在 `same_time_interval_round1_probe.log`。它只读取 frozen `a1/v2` manifest 与 dispositions，在内存重建 timeline 并运行现有 validator；未写入任何正式裁决工件。

- checkpoint：1,371 行、1,371 个唯一 ID。
- 重建 timeline：1,371 行、1,371 个唯一 ID，ID 集合完全相同。
- `nonpositive_intervals = 0`。
- `causal_violations = 6`。
- frozen denominator SHA-256：`477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd`。
- 六项 `a1/v2` 正式工件 SHA-256 与首次裁决记录逐项一致。

本轮未修改 causal 定义、日期、其他模块或正式裁决；版本控制操作为 `NO_GIT`。
