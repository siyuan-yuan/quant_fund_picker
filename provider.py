# -*- coding: utf-8 -*-
"""
数据层: 全免费数据源, 带本地缓存
  - 基金净值/日增长率: 天天基金(EM)
  - 基金档案(规模变动/仓位配置/经理任期): 天天基金 pingzhongdata
  - 指数日行情: 新浪
  - 指数PE历史: 乐咕乐股 (Point-in-Time时序)
"""
import os, re, json, time, datetime as dt, threading, tempfile
import requests
import pandas as pd
import akshare as ak

from config import CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)
TODAY = dt.date.today().isoformat()
_memo = {}
_memo_day = TODAY
_FETCHED_TODAY = set()      # 每文件每日至多同步一次(防假期空转抓取)
_BG_REFRESHING = set()      # 后台刷新中的路径，避免并发重复打源
STALE_SERVED = []           # (文件, 旧数据截止日, 原因) — 降级不沉默
STALE_OK = False   # True=使用过期缓存(walk-forward回测用, 历史数据不变)

# 外网（东财/乐咕/中证）不通或无超时挂起时，页面会永远转圈。
# 给 requests/akshare 补默认超时，并用线程墙钟兜底；有本地缓存则先回缓存。
_HTTP_TIMEOUT = float(os.environ.get("QFP_FETCH_TIMEOUT", "8"))


def _patch_requests_timeout(timeout=_HTTP_TIMEOUT):
    """akshare 内部 requests 默认 timeout=None，sandbox/预览环境会永久挂起。"""
    try:
        if getattr(requests.Session.request, "_qfp_patched", False):
            return
        _orig = requests.Session.request

        def _wrapped(self, method, url, **kwargs):
            kwargs.setdefault("timeout", timeout)
            return _orig(self, method, url, **kwargs)

        _wrapped._qfp_patched = True
        requests.Session.request = _wrapped
    except Exception:
        pass


_patch_requests_timeout()


def _run_with_timeout(fn, timeout=None):
    """硬超时执行抓取，避免单次 akshare 调用卡死 Flask 线程。"""
    timeout = _HTTP_TIMEOUT + 4 if timeout is None else timeout
    box, err = {}, {}

    def _target():
        try:
            box["v"] = fn()
        except Exception as e:
            err["e"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"数据源超时({timeout:.0f}s)")
    if err:
        raise err["e"]
    return box.get("v")

# ---- 缓存并发安全：原子写 + 每路径锁 ----
# 根因修复：旧代码 `df.to_csv(path)` / `json.dump(open(path,'w'))` 直接覆盖写，非原子。
# 当日缓存过期时，持仓诊断的多线程评分 + 全市场扫描会并发读到过期文件并同时重抓，
# 写盘瞬间并发读会读到半写/空文件 → pandas `EmptyDataError: No columns to parse from file`，
# 表现为「生成方案」首次点击总失败、重试才正常。这里改为「临时文件 + os.replace 原子替换」
# 并用每路径锁串行化同一文件的重抓，读方永远只见完整文件（旧或新），不再见空文件。
_fetch_locks = {}
_fetch_locks_guard = threading.Lock()


def _lock_for(path):
    with _fetch_locks_guard:
        lock = _fetch_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _fetch_locks[path] = lock
        return lock


def _atomic_write_csv(df, path):
    """原子写 CSV：同目录临时文件 + os.replace，杜绝并发读读到空/半写文件"""
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                               dir=os.path.dirname(os.path.abspath(path)) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(obj, path):
    """原子写 JSON：同目录临时文件 + os.replace，杜绝并发读读到空/半写文件"""
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                               dir=os.path.dirname(os.path.abspath(path)) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _roll_day():
    """跨日自动清进程缓存 — 修复 webapp 长驻隔夜后全程服务昨日数据的暗伤"""
    global _memo_day
    t = dt.date.today().isoformat()
    if _memo_day != t:
        _memo.clear(); _FETCHED_TODAY.clear(); _memo_day = t


def expected_last_td(lag=0):
    """期望的最新数据日: 18点前/周末 → 回退到上一交易日(周一至五近似); lag 用于境外源"""
    now = dt.datetime.now()
    d = now.date()
    if now.hour < 18 or d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    for _ in range(lag):
        d -= dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= dt.timedelta(days=1)
    return d.isoformat()


def _content_fresh(path, lag=0):
    """内容保鲜: 文件内最后数据日 ≥ 期望交易日 (mtime会被拷贝刷新, 内容日期不会撒谎)"""
    try:
        last = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"].max()
        return pd.notna(last) and last.date().isoformat() >= expected_last_td(lag)
    except Exception:
        return False


def _cached_or_fetch(path, fetch_df, lag=0):
    """统一装载: 有缓存先回；内容过期则后台刷新。禁止同步打源卡住首屏。"""
    _roll_day()

    def _read():
        return pd.read_csv(path, parse_dates=["date"])

    def _stale():
        # 回测 STALE_OK：历史缓存就是事实，绝不再打今日源
        if STALE_OK:
            return False
        return path not in _FETCHED_TODAY and not (_fresh(path) and _content_fresh(path, lag))

    have = os.path.exists(path) and os.path.getsize(path) > 32

    def _refresh_sync():
        df = _run_with_timeout(fetch_df)
        _atomic_write_csv(df, path)
        _FETCHED_TODAY.add(path)
        return df

    def _refresh_bg():
        try:
            _refresh_sync()
        except Exception as e:
            if have:
                try:
                    old = _read()
                    asof = str(old["date"].max().date()) if old is not None and len(old) else "?"
                except Exception:
                    asof = "?"
                STALE_SERVED.append((os.path.basename(path), asof, str(e)[:60]))
            # 失败不记 FETCHED_TODAY：持仓页「刷新净值」还能再试
        finally:
            _BG_REFRESHING.discard(path)

    if have and not _stale():
        return _read()

    if have and _stale():
        # 首屏关键路径：立刻返回本地缓存，刷新放到后台
        with _lock_for(path):
            if _stale() and path not in _BG_REFRESHING:
                _BG_REFRESHING.add(path)
                threading.Thread(target=_refresh_bg, daemon=True).start()
        return _read()

    # 无缓存才同步抓；仍带硬超时，避免永久挂起
    with _lock_for(path):
        if os.path.exists(path) and os.path.getsize(path) > 32:
            return _read()
        df = _refresh_sync()
        return df if df is not None else _read()


def stale_warnings():
    seen, out = set(), []
    for f, d, why in STALE_SERVED:
        if f not in seen:
            seen.add(f); out.append({"file": f, "asof": d, "reason": why})
    return out


def _fresh(path):
    return os.path.exists(path) and (STALE_OK or dt.date.fromtimestamp(
        os.path.getmtime(path)).isoformat() == TODAY)


def _retry(fn, n=3, sleep=1.5):
    last = None
    for _ in range(n):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise last


# ---------------- 基金净值 ----------------
def get_fund_nav(code: str) -> pd.DataFrame:
    """返回 [date, nav, ret(日增长率小数)] —— ret为东财官方复权日收益"""
    key = f"nav_{code}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/nav_{code}.csv"

    def _build():
        raw = _retry(lambda: ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势"))
        return pd.DataFrame({
            "date": pd.to_datetime(raw["净值日期"]),
            "nav": raw["单位净值"].astype(float).values,
            "ret": raw["日增长率"].astype(float).div(100).values,
        }).dropna().sort_values("date").reset_index(drop=True)

    df = _cached_or_fetch(path, _build)
    _memo[key] = df
    return df


def refresh_fund_nav(code: str, timeout=15):
    """同步刷新一只基金净值。失败不记入 FETCHED_TODAY，允许稍后重试。"""
    code = str(code or "").zfill(6)
    key = f"nav_{code}"
    path = f"{CACHE_DIR}/nav_{code}.csv"
    _roll_day()

    def _build():
        raw = _retry(lambda: ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势"),
                     n=2, sleep=1.0)
        return pd.DataFrame({
            "date": pd.to_datetime(raw["净值日期"]),
            "nav": raw["单位净值"].astype(float).values,
            "ret": raw["日增长率"].astype(float).div(100).values,
        }).dropna().sort_values("date").reset_index(drop=True)

    try:
        df = _run_with_timeout(_build, timeout=timeout)
        if df is None or len(df) == 0:
            raise RuntimeError("净值空表")
        _atomic_write_csv(df, path)
        _memo[key] = df
        _FETCHED_TODAY.add(path)
        return {"ok": True, "code": code,
                "asof": str(pd.Timestamp(df["date"].max()).date()), "n": int(len(df))}
    except Exception as e:
        _memo.pop(key, None)
        asof = None
        try:
            if os.path.exists(path):
                old = pd.read_csv(path, parse_dates=["date"])
                asof = str(old["date"].max().date())
        except Exception:
            pass
        return {"ok": False, "code": code, "asof": asof, "error": str(e)[:120]}


# ---------------- 基金档案 (pingzhongdata) ----------------
def get_fund_dossier(code: str) -> dict:
    """规模变动史 / 股债现金仓位 / 现任经理任期 / 基金名称 — 增加重试与降级容错，档案拉取失败不阻断评分"""
    key = f"dossier_{code}"
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/dossier_{code}.json"
    d = None
    if _fresh(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            d = None      # 缓存损坏或旧GBK编码(Windows) → 扔掉自动重爬
    if d is None:
        url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js"
        # 带重试的档案拉取：东财 pingzhongdata 偶发 RemoteDisconnected，退避重试3次
        def _fetch():
            r = requests.get(url, headers={"Referer": "http://fund.eastmoney.com/"}, timeout=20)
            r.encoding = "utf-8"
            if r.status_code != 200 or not r.text or "fS_name" not in r.text:
                raise RuntimeError(f"档案空响应 {r.status_code}")
            return r
        try:
            r = _retry(_fetch, n=3, sleep=1.2)
        except Exception as e:
            # 降级：若本地有旧缓存（即使过期）则复用，避免单只档案失败拖垮整批
            if os.path.exists(path):
                try:
                    d = json.load(open(path, encoding="utf-8"))
                    _memo[key] = d
                    STALE_SERVED.append((os.path.basename(path), d.get("name",""), f"档案回退旧缓存: {str(e)[:50]}"))
                    return d
                except Exception:
                    pass
            # 彻底失败则返回空档案，评分以降级模式继续（不抛异常）
            d = {"code": code, "name": code, "scale_hist": [], "asset_alloc": {}, "managers": [], "_stale": True, "_err": str(e)[:80]}
            _memo[key] = d
            return d
        t = r.text

        def grab(var):
            m = re.search(r'var\s+' + var + r'\s*=\s*(.*?);\s*(?:/\*|var|$)', t, re.S)
            return m.group(1).strip() if m else None

        d = {"code": code}
        m = re.search(r'var\s+fS_name\s*=\s*"([^"]*)"', t)
        d["name"] = m.group(1) if m else code
        # 规模变动史 (净资产, 亿)
        try:
            js = grab("Data_fluctuationScale")
            obj = json.loads(js)
            d["scale_hist"] = [{"date": c, "scale": s["y"]}
                               for c, s in zip(obj["categories"], obj["series"])]
        except Exception:
            d["scale_hist"] = []
        # 股债现金仓位 (%)
        try:
            obj = json.loads(grab("Data_assetAllocation"))
            alloc = {}
            for ser in obj["series"]:
                alloc[ser["name"]] = {"dates": obj["categories"], "values": ser["data"]}
            d["asset_alloc"] = alloc
        except Exception:
            d["asset_alloc"] = {}
        # 现任经理
        try:
            mgrs = json.loads(grab("Data_currentFundManager") or "[]")
            d["managers"] = [{"name": m.get("name"), "workTime": m.get("workTime", "")}
                             for m in mgrs]
        except Exception:
            d["managers"] = []
        _atomic_write_json(d, path)
    _memo[key] = d
    return d


def parse_worktime_days(worktime: str) -> int:
    """'13年又311天' -> 天数"""
    y = re.search(r'(\d+)\s*年', worktime or "")
    d = re.search(r'(\d+)\s*天', worktime or "")
    return (int(y.group(1)) * 365 if y else 0) + (int(d.group(1)) if d else 0)


# ---------------- 指数日行情 (新浪) ----------------
def get_index_close(sina_code: str) -> pd.Series:
    key = f"idx_{sina_code}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idx_{sina_code}.csv"

    def _build():
        raw = _retry(lambda: ak.stock_zh_index_daily(symbol=sina_code))
        df = raw[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    df = _cached_or_fetch(path, _build)
    s = df.set_index("date")["close"].sort_index()
    _memo[key] = s
    return s


# ---------------- 指数OHLCV (含成交量, 右侧反转信号用) ----------------
def get_index_ohlcv(sina_code: str) -> pd.DataFrame:
    key = f"idxfull_{sina_code}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idxfull_{sina_code}.csv"

    def _build():
        raw = _retry(lambda: ak.stock_zh_index_daily(symbol=sina_code))
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    df = _cached_or_fetch(path, _build)
    _memo[key] = df
    return df


def market_reversal_signal(sina_code="sh000300") -> dict:
    """右侧反转信号: 收盘>MA20 且 当日量>20日均量×1.5"""
    try:
        df = get_index_ohlcv(sina_code)
        c, v = df["close"], df["volume"]
        ma20 = c.rolling(20).mean().iloc[-1]
        vr = float(v.iloc[-1] / v.tail(20).mean()) if v.tail(20).mean() > 0 else None
        return {"above_ma20": bool(c.iloc[-1] >= ma20),
                "vol_ratio": None if vr is None else round(vr, 2),
                "reversal": bool(c.iloc[-1] >= ma20 and vr and vr >= 1.5)}
    except Exception as e:
        return {"error": str(e)[:80]}


# ---------------- 中证全指行业指数 (csindex: 收盘+滚动PE同源) ----------------
def _csindex_df(csicode: str) -> pd.DataFrame:
    key = f"csi_{csicode}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/csi_{csicode}.csv"

    def _build():
        raw = _retry(lambda: ak.stock_zh_index_hist_csindex(
            symbol=csicode, start_date="20210801",
            end_date=dt.date.today().strftime("%Y%m%d")))   # V3.7.3: 修硬编码20260801化石
        return pd.DataFrame({"date": pd.to_datetime(raw["日期"]),
                             "close": raw["收盘"].astype(float).values,
                             "pe": raw["滚动市盈率"].astype(float).values})

    df = _cached_or_fetch(path, _build)
    _memo[key] = df
    return df


def get_index_close_csindex(csicode: str) -> pd.Series:
    df = _csindex_df(csicode)
    return df.set_index("date")["close"].sort_index()


def get_index_pe_csindex(csicode: str) -> pd.Series:
    df = _csindex_df(csicode)
    return df.set_index("date")["pe"].dropna().sort_index()


# ---------------- 境外指数 (V3.3) ----------------
def get_us_index_close(us_code: str) -> pd.Series:
    """新浪美股指数: .NDX / .INX"""
    key = f"idxus_{us_code}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idx_us_{us_code.replace('.','_')}.csv"

    def _build():
        raw = _retry(lambda: ak.stock_us_daily(symbol=us_code, adjust=""))
        df = raw[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    df = _cached_or_fetch(path, _build, lag=1)   # 境外源晚一个交易日
    s = df.set_index("date")["close"].sort_index()
    _memo[key] = s
    return s


def get_hk_index_close(hk_code: str) -> pd.Series:
    """新浪港股指数: HSI / HSTECH / HSCEI"""
    key = f"idxhk_{hk_code}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idx_hk_{hk_code}.csv"

    def _build():
        raw = _retry(lambda: ak.stock_hk_index_daily_sina(symbol=hk_code))
        df = raw[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    df = _cached_or_fetch(path, _build, lag=1)
    s = df.set_index("date")["close"].sort_index()
    _memo[key] = s
    return s


def get_close_by_src(src: str, code: str) -> pd.Series:
    if src == "sina":
        return get_index_close(code)
    if src == "us_sina":
        return get_us_index_close(code)
    if src == "hk_sina":
        return get_hk_index_close(code)
    return get_index_close_csindex(code)


def get_pe_by_key(pe_key: str) -> pd.Series:
    """PE调度: 'lg:xxx'=乐咕 | 'csi:xxxxxx'=中证官方(含H系港股) | 'none'=无"""
    src, _, code = pe_key.partition(":")
    if src == "lg":
        return get_index_pe(code)
    if src == "csi":
        return _csindex_df(code).set_index("date")["pe"].dropna().sort_index()
    return pd.Series(dtype=float)


def get_pe_by_src(src: str, pe_key: str) -> pd.Series:
    """旧接口兼容(报告/网页): sina→乐咕 csindex→中证"""
    return get_index_pe(pe_key) if src == "sina" else get_index_pe_csindex(pe_key)


# ---------------- 指数PE历史 (乐咕) ----------------
def get_index_pe(lg_symbol: str) -> pd.Series:
    key = f"pe_{lg_symbol}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/pe_{lg_symbol}.csv"

    def _build():
        raw = _retry(lambda: ak.stock_index_pe_lg(symbol=lg_symbol))
        return pd.DataFrame({"date": pd.to_datetime(raw["日期"]),
                             "pe": raw["滚动市盈率"].astype(float).values})

    df = _cached_or_fetch(path, _build)
    s = df.set_index("date")["pe"].sort_index()
    _memo[key] = s
    return s


# ---------------- 基金全市场名录 ----------------
def get_fund_meta() -> pd.DataFrame:
    if "meta" in _memo:
        return _memo["meta"]
    path = f"{CACHE_DIR}/fund_meta.csv"
    if _fresh(path):
        df = pd.read_csv(path, dtype={"基金代码": str})
    else:
        # 名录是全场共享文件：多线程评分会同时触发重抓，加锁串行化 + 原子写防并发读空文件
        with _lock_for(path):
            if not _fresh(path):
                df = _retry(lambda: ak.fund_name_em())
                df["基金代码"] = df["基金代码"].astype(str).str.zfill(6)
                _atomic_write_csv(df, path)
            else:
                df = pd.read_csv(path, dtype={"基金代码": str})
    df = df.set_index("基金代码")
    _memo["meta"] = df
    return df


def fund_type(code: str) -> str:
    try:
        return str(get_fund_meta().loc[code, "基金类型"])
    except Exception:
        return ""


def is_passive_fund(code: str, name: str = "") -> bool:
    ftype = fund_type(code)
    name = name or ""
    return ("指数" in ftype) or ("指数" in name) or ("ETF" in name) or ("联接" in name)


# ---------------- 基金申购费率 (天天基金 fundf10) ----------------
FEE_FALLBACK_DEFAULT = 0.0015   # 网络/解析失败时的兜底默认（折后 0.15%）


def _parse_fee_pct(v):
    """把 '0.12%' / '1.20%' / 0.12 解析为小数费率；'每笔1000元' 等固定额返回 None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if f == 0:
            return 0.0
        return None if not (0 < f <= 100) else f / 100.0
    s = str(v).strip().replace("%", "").replace("％", "").replace(",", "").replace(" ", "")
    if not s or "每笔" in s or "笔" in s:
        return None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if f == 0:
        return 0.0
    return None if not (0 < f <= 100) else f / 100.0


def get_fund_buy_fee(code: str) -> dict:
    """自动查某基金的**申购费率**（折后实扣费率）。

    来源: 天天基金基金档案-购买信息 `ak.fund_fee_em(symbol, "申购费率")`，
    返回各金额档的 `原费率` 与 `天天基金优惠费率`（即平台折后实扣）。取**最小金额档**
    （散户实际买入档）的优惠费率，无优惠则回退原费率。结果缓存到 cache/fee_<code>.json。

    返回: {"rate": 小数费率, "source": "nominal"|"discounted"|"default"|"override",
           "original": 名义费率, "discounted": 优惠费率, "bracket": "适用金额档"}
    任何异常都不会抛错——回退到 FEE_FALLBACK_DEFAULT，保证不阻断打分/台账。
    """
    code = str(code or "").zfill(6)
    key = f"fee_{code}"
    _roll_day()
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/fee_{code}.json"
    # 读缓存（含旧缓存复用；费率不随交易日变，当日新鲜即可）
    d = None
    if _fresh(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            d = None
    if d is None:
        try:
            df = _retry(lambda: ak.fund_fee_em(symbol=code, indicator="申购费率"),
                        n=2, sleep=1.0)
            if df is not None and len(df):
                df = df.reset_index(drop=True)
                # 列名防御
                cols = {str(c): c for c in df.columns}
                ocol = cols.get("原费率")
                dcol = cols.get("天天基金优惠费率")
                bcol = cols.get("适用金额")
                orig = None if ocol is None else _parse_fee_pct(df.loc[0, ocol])
                disc = None if dcol is None else _parse_fee_pct(df.loc[0, dcol])
                bracket = None if bcol is None else str(df.loc[0, bcol])
                # 优惠费率优先（=平台折后实扣），否则名义费率
                rate = disc if disc is not None else orig
                d = {
                    "rate": rate if rate is not None else FEE_FALLBACK_DEFAULT,
                    "source": "discounted" if (disc is not None) else
                              ("nominal" if orig is not None else "default"),
                    "original": orig, "discounted": disc, "bracket": bracket,
                    "fetched": dt.date.today().isoformat(),
                }
                _atomic_write_json(d, path)
            else:
                raise RuntimeError("fund_fee_em 空表")
        except Exception as e:
            # 查不到时不要臆造 0.15% 把所有基金市值削一刀；按 0 计并标记，UI 可提示
            d = {"rate": 0.0, "source": "default",
                 "original": None, "discounted": None, "bracket": None,
                 "fetched": dt.date.today().isoformat(), "_err": str(e)[:80]}
    _memo[key] = d
    return d
