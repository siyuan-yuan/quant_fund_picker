# -*- coding: utf-8 -*-
"""
严格 Point-in-Time（PIT）基金池构建管线。

回答的问题：
    在每一个历史决策日，当时真实存在、属于目标类型、且投资者当时能够申购的
    基金有哪些？

管线（B + C 方案落地；A 方案由同一 schema 导入）::

    Tushare 全历史主表(生命周期)
      + 证监会历期产品索引(类型/名单 as-of 快照)
      + 申赎状态明细/公告事件(暂停/恢复/大额限购)
      + 每日真实截面(从今天起)
      ↓
    事件化历史数据库 (data/pit_raw/)
      ↓
    月末不可变 CSV 快照 (data/pit_universe/YYYY-MM-DD.csv + _manifest.json)
      ↓
    PITUniverseStore 严格读取 (拒绝 PIT-lite / 拒绝回填)

命令行入口: ``python -m pit <subcommand>``，详见 docs/PIT_UNIVERSE.md。
"""

__version__ = "1.0.0"
