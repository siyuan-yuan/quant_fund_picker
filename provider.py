# -*- coding: utf-8 -*-
"""
数据层: 全免费数据源, 带本地缓存
  - 基金净值/日增长率: 天天基金(EM)
  - 基金档案(规模变动/仓位配置/经理任期): 天天基金 pingzhongdata
  - 指数日行情: 新浪
  - 指数PE历史: 乐咕乐股 (Point-in-Time时序)
"""
import os, re, json, time, datetime as dt
import requests
import pandas as pd
import akshare as ak

from config import CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)
TODAY = dt.date.today().isoformat()
_memo = {}
STALE_OK = False   # True=使用过期缓存(walk-forward回测用, 历史数据不变)


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
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/nav_{code}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势"))
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["净值日期"]),
            "nav": raw["单位净值"].astype(float).values,
            "ret": raw["日增长率"].astype(float).div(100).values,
        }).dropna().sort_values("date").reset_index(drop=True)
        df.to_csv(path, index=False)
    _memo[key] = df
    return df


# ---------------- 基金档案 (pingzhongdata) ----------------
def get_fund_dossier(code: str) -> dict:
    """规模变动史 / 股债现金仓位 / 现任经理任期 / 基金名称"""
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
        r = requests.get(url, headers={"Referer": "http://fund.eastmoney.com/"}, timeout=20)
        r.encoding = "utf-8"
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
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
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
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idx_{sina_code}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.stock_zh_index_daily(symbol=sina_code))
        df = raw[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(path, index=False)
    s = df.set_index("date")["close"].sort_index()
    _memo[key] = s
    return s


# ---------------- 指数OHLCV (含成交量, 右侧反转信号用) ----------------
def get_index_ohlcv(sina_code: str) -> pd.DataFrame:
    key = f"idxfull_{sina_code}"
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idxfull_{sina_code}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.stock_zh_index_daily(symbol=sina_code))
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(path, index=False)
    df = df.sort_values("date").reset_index(drop=True)
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
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/csi_{csicode}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.stock_zh_index_hist_csindex(
            symbol=csicode, start_date="20210801", end_date="20260801"))
        df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]),
                           "close": raw["收盘"].astype(float).values,
                           "pe": raw["滚动市盈率"].astype(float).values})
        df.to_csv(path, index=False)
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
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idx_us_{us_code.replace('.','_')}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.stock_us_daily(symbol=us_code, adjust=""))
        df = raw[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(path, index=False)
    s = df.set_index("date")["close"].sort_index()
    _memo[key] = s
    return s


def get_hk_index_close(hk_code: str) -> pd.Series:
    """新浪港股指数: HSI / HSTECH / HSCEI"""
    key = f"idxhk_{hk_code}"
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/idx_hk_{hk_code}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.stock_hk_index_daily_sina(symbol=hk_code))
        df = raw[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(path, index=False)
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
    if key in _memo:
        return _memo[key]
    path = f"{CACHE_DIR}/pe_{lg_symbol}.csv"
    if _fresh(path):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = _retry(lambda: ak.stock_index_pe_lg(symbol=lg_symbol))
        df = pd.DataFrame({"date": pd.to_datetime(raw["日期"]),
                           "pe": raw["滚动市盈率"].astype(float).values})
        df.to_csv(path, index=False)
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
        df = _retry(lambda: ak.fund_name_em())
        df["基金代码"] = df["基金代码"].astype(str).str.zfill(6)
        df.to_csv(path, index=False)
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
