# -*- coding: utf-8 -*-
"""
V8 / py_mini_racer 进程级护栏。

Windows 上运行 webapp 出现：

    [FATAL:partition_address_space.cc(243)]
    Check failed: !IsConfigurablePoolInitialized().

不是 Flask 业务异常，是 Chromium PartitionAlloc 的 C++ 断言。
新版 mini-racer（V8 11+/12+）每次构造 MiniRacer() 都会走
PartitionAddressSpace::Init()；该池进程内只能 Init 一次。
第二次构造（含：析构后再建、waitress 多线程同时建、
超时弃线程后重试）直接把 python.exe 打死，Python 层无法 try/except。

本模块必须在 `import akshare` 之前 install()：
  1) MiniRacer 钉成永不销毁的单例
  2) eval / call / execute 加锁（isolate 非完全线程安全）
  3) 提供 call_ak() 把东财 JS 解密路径串行化
"""
from __future__ import annotations

import threading

_LOCK = threading.RLock()
_STATE = {
    "installed": False,
    "available": False,
    "singleton": False,
    "module": None,
    "error": None,
}


def status() -> dict:
    return dict(_STATE)


def _reset_for_tests():
    """仅单测使用：允许重新 install。"""
    with _LOCK:
        _STATE.update(installed=False, available=False, singleton=False,
                      module=None, error=None)


def call_ak(fn, *args, **kwargs):
    """串行执行可能触发 MiniRacer 的 akshare 调用。"""
    with _LOCK:
        return fn(*args, **kwargs)


def install() -> dict:
    """幂等。必须在 import akshare 之前调用。"""
    with _LOCK:
        if _STATE["installed"]:
            return status()
        _STATE["installed"] = True
        try:
            import py_mini_racer as pmr
        except Exception as e:
            _STATE["error"] = f"py_mini_racer 不可用: {e}"[:160]
            return status()

        _STATE["module"] = "py_mini_racer"
        _STATE["available"] = True
        orig = getattr(pmr, "MiniRacer", None)
        if orig is None:
            _STATE["error"] = "py_mini_racer.MiniRacer 不存在"
            return status()

        holder = {"ctx": None}

        def _locked(method):
            def _wrap(self, *a, **k):
                with _LOCK:
                    return method(self, *a, **k)
            return _wrap

        if not getattr(orig, "_qfp_eval_locked", False):
            for name in ("eval", "call", "execute", "eval_raw"):
                fn = getattr(orig, name, None)
                if callable(fn):
                    setattr(orig, name, _locked(fn))
            orig._qfp_eval_locked = True

        class _SingletonMiniRacer:
            """所有构造返回同一 isolate，且永不释放，避免二次 Init。"""

            def __new__(cls, *args, **kwargs):
                with _LOCK:
                    if holder["ctx"] is None:
                        holder["ctx"] = orig(*args, **kwargs)
                    return holder["ctx"]

        pmr.MiniRacer = _SingletonMiniRacer
        # 个别版本从 py_mini_racer.py_mini_racer 再导出一次
        inner = getattr(pmr, "py_mini_racer", None)
        if inner is not None and hasattr(inner, "MiniRacer"):
            inner.MiniRacer = _SingletonMiniRacer
        _STATE["singleton"] = True
        return status()


def prewarm() -> dict:
    """主线程预先构造 isolate，避免首个 waitress 工作线程抢 Init。"""
    st = install()
    if not st.get("available"):
        return st
    try:
        import py_mini_racer
        with _LOCK:
            py_mini_racer.MiniRacer()
        _STATE["error"] = None
    except Exception as e:
        _STATE["error"] = f"prewarm 失败: {e}"[:160]
    return status()
