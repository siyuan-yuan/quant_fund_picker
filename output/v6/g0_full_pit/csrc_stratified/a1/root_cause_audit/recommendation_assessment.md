# V6-G0-A1 后续建议客观评估

日期：2026-09-08。范围：首次 A1 裁决后的诊断；不构成第二次裁决，不修改 `a1/v2`。

## 总体裁决

建议的核心方向正确：永久保存首次 FAIL，先建立 checkpoint-level 根因证据，再修实现，最后用同一冻结分母产生新版本裁决。本次已经执行建议中最优先的 `A1 unresolved checkpoint root-cause audit`，并增加了时间异常全量审计及 SECTION_UNCOMPARABLE 六层样本审计。

但三项表述需要修正：

1. `1250 SECTION_EXTRACTION_FAILED` 不能按现有数据直接细分成题述的所有互斥“根因”。现有 pipeline 把章节定位和仓位语义解析合并，缓存没有保存完整章节级中间状态。审计因此采用“最早可证明的主要阻断点 + 次级标志”，将证据不足保留为 `UNDETERMINED`。
2. 6 个 causal violations 不是 6 次实际 PIT 回填。全部节点的 `effective_from == known_at`；其中至少一项是确定的日期抽取错误，四项有历史法律生效日文本支持，另一项涉及拟议条款阶段。需要分别修日期抽取、阶段识别和指标语义。
3. 1110 个 SECTION_UNCOMPARABLE 不能整体归因于通用章节抽取器。其中文档代码 FC900090 有 772 个；18 个分层样本中 10 个是销售、费率、分红、经理、会议等事项公告。8 个 FA 样本均有可读条款，其中 4 个是直接股票区间漏识别，4 个是股票与可转债等联合约束，不能强拆成股票区间。

## 对八项建议的逐项判断

1. **采纳并已执行。** `a1/v2` 六项首次正式裁决工件永久保存，首次 `G0 FAIL` 不覆盖。后续结果必须写新版本并引用相同 1371 个 checkpoint 及 SHA-256 `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd`。
2. **完善后采纳并已执行。** 已输出 1251 行 checkpoint-level census、60 行 INCEPTION trace、49 行 TRANSFORMATION trace和候选文档长表。分类方法保留不确定性，未把本地缺失冒充官方证据不存在。
3. **采纳“第二次裁决前清零真实时间轴错误”，不接受把 21 项预设成同一种 P0 bug。** 15 个非正区间均为同日节点顺次关闭造成的零长区间；应在状态层合并同一时点的来源承载，而非删 checkpoint。6 个 causal flag 需按上述三类分别 TDD。
4. **采纳，且系统性 join 缺口已坐实。** 60 个 INCEPTION 使用合成 upload ID，而裁决器只按官方 upload ID 精确连接，因此 60 个全部必然无证据匹配。15/60 在本地索引中有当时或更早的招募书候选，11/60 已有解析条款；这证明 resolver 缺失，但其余 45 个不能据此断言官方无证据。
5. **采纳架构方向，调整实施顺序。** 应拆为 `Document applicability -> Section -> Clause -> State`，并保留联合资产集合语义。文档适用性必须先于通用章节抽取，否则 772 个 FC900090 会继续被错误当作应含投资章节的文档。禁止逐基金补丁。
6. **完全采纳。** 0 个 NOOP 是调查信号。当前样本存在重复核心比例候选，但尚未证明完整章节、主体、前驱和时序等价，不能放宽 A1 或直接升级 NOOP。若未来需要改变测量定义，只能形成 A2 并保留 A1 FAIL。
7. **完全采纳。** 49 个 transformation 已全量进入 trace；47 个未决均有精确本地文档匹配，但统一错误不足以区分章节、语义和阶段原因。后续应进行定点法律证据审计。
8. **条件采纳，当前不执行第二次裁决。** 只有 lifecycle resolver、文档适用性、section/clause 分层、日期/阶段及同日时间轴处理经过 TDD 与独立复核后，才能用同一 frozen denominator 写新版本。报告必须按根因修复归因三态增量。

## 当前证据支持的实施顺序

1. TDD 修 lifecycle checkpoint 到官方文档的因果 resolver，先覆盖 INCEPTION 60 和 TERMINAL 30；证据不足仍为 UNRESOLVED。
2. TDD 修确定的日期跨段误抽、拟议/历史条款阶段识别，并重定义实现层的 causal invariant，使其检测实际 PIT 使用时间而不是单纯的历史法律生效日。
3. TDD 修同一 `effective_from` 多证据节点的状态区间表示，checkpoint 继续完整保留。
4. 增加文档适用性层，再把完整 Section 与 Clause/State parser 解耦；先覆盖已取证的四个直接区间漏识别与四个联合约束案例。
5. 对 49 个 transformation 做证据核签；再扩展 SECTION_UNCOMPARABLE 的概率/风险分层样本。
6. 独立复核后建立新 adjudication 版本。若系统取证证明 A1 所需状态在官方制度下大规模不可观测，再预注册 A2；不能倒改 A1。

## 工件

- `unresolved_checkpoint_census.csv`：1251 行未决检查点。
- `inception_trace.csv`：60 个成立节点。
- `transformation_trace.csv`：49 个转型节点，含 2 个基线成功。
- `lifecycle_candidate_documents.csv`：生命周期候选官方文档。
- `timeline_anomalies.csv`：6 个 causal flag 与 15 个非正区间逐项证据。
- `section_uncomparable_sample.csv` 与 `section_uncomparable_sample_raw.json`：六层确定性样本和页码证据。

审计前后 `a1/v2` 六项工件 SHA-256 完全一致。
