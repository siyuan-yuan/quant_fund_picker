# ⚠️ 未推送备份（GH_TOKEN 失效期间的抗回滚冗余）

本目录是**纯冗余副本**，用于对抗沙箱回滚（本会话已发生 3 次；观测到的规律：
`.git` 被换成新 clone → 已提交的跟踪文件回退到远端最新 commit，
而**新增的未跟踪文件在工作区快照层存活**）。

- 对应本地 commit：`291255c`（R3.5 结论）与其前序 `aa59464`（已成功推送）。
- 一旦 GitHub 通道恢复：`git push origin HEAD:arena/01a05d9b-quant-fund-picker`，
  推送成功后**本目录即可整体删除**（它不是研究资产，只是保险）。
- 目录内含：三份治理文档副本、M15 修复后的 `_build_ml_panel.py`、
  新增的 `r35_zoo_cache_shard.py`、以及 R3.5 全部产物 CSV + summary.md。
