# -*- coding: utf-8 -*-
"""
天天基金 / 东财 HTTP 直连（不走 MiniRacer / V8）。

akshare 的 fund_*_em 在 Windows 上会构造 py_mini_racer，新版 V8
第二次 Init 直接 FATAL。净值与名录是热路径，这里用公开 HTTP 接口替代。
"""
from __future__ import annotations

import json
import re
import time

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": _UA, "Accept": "*/*"}

_JSONP_RE = re.compile(r"\{.*\}", re.S)
_ARRAY_RE = re.compile(r"\[.*\]", re.S)


def parse_jsonp(text: str) -> dict:
    """jQuery({...}) / callback({...}) → dict。"""
    m = _JSONP_RE.search(text or "")
    if not m:
        raise RuntimeError("东财接口返回非 JSONP")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise RuntimeError("东财 JSONP 根节点不是对象")
    return obj


def lsjz_records(payload: dict) -> tuple[list[tuple], int]:
    """解析 f10/lsjz 载荷 → ([(date, nav, ret_pct), ...], total_count)。"""
    data = payload.get("Data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        data = {}
    lsjz = data.get("LSJZList") or []
    try:
        total = int(data.get("TotalCount") or 0)
    except (TypeError, ValueError):
        total = 0
    rows = []
    for it in lsjz:
        if not isinstance(it, dict):
            continue
        rows.append((it.get("FSRQ"), it.get("DWJZ"), it.get("JZZZL")))
    return rows, total


def fetch_nav_lsjz(code: str, timeout: float = 8.0, page_size: int = 500,
                   max_pages: int = 50) -> list[tuple]:
    """单位净值走势。返回 [(date_str, nav, ret_pct), ...]，ret 为百分数。"""
    import requests
    code = str(code or "").zfill(6)
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"基金代码无效: {code}")
    headers = dict(HEADERS)
    headers["Referer"] = f"https://fundf10.eastmoney.com/jjjz_{code}.html"
    rows, total, page = [], None, 1
    while page <= max_pages:
        r = requests.get(
            "https://api.fund.eastmoney.com/f10/lsjz",
            params={
                "callback": "jQuery",
                "fundCode": code,
                "pageIndex": page,
                "pageSize": page_size,
                "startDate": "",
                "endDate": "",
                "_": int(time.time() * 1000),
            },
            headers=headers,
            timeout=timeout,
        )
        r.raise_for_status()
        chunk, tot = lsjz_records(parse_jsonp(r.text))
        if total is None:
            total = tot
        if not chunk:
            break
        rows.extend(chunk)
        if total and page * page_size >= total:
            break
        if len(chunk) < page_size:
            break
        page += 1
    if not rows:
        raise RuntimeError(f"{code} 净值空表")
    return rows


def parse_fundcode_search(text: str) -> list[dict]:
    """解析 fund.eastmoney.com/js/fundcode_search.js。"""
    m = _ARRAY_RE.search(text or "")
    if not m:
        raise RuntimeError("基金名录接口无数组")
    data = json.loads(m.group(0))
    out = []
    for row in data:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        code = str(row[0]).zfill(6)
        if not re.fullmatch(r"\d{6}", code):
            continue
        out.append({
            "基金代码": code,
            "拼音缩写": "" if len(row) < 2 else str(row[1]),
            "基金简称": "" if len(row) < 3 else str(row[2]),
            "基金类型": "" if len(row) < 4 else str(row[3]),
            "拼音全称": "" if len(row) < 5 else str(row[4]),
        })
    if not out:
        raise RuntimeError("基金名录空表")
    return out


def fetch_fund_meta(timeout: float = 20.0) -> list[dict]:
    import requests
    r = requests.get(
        "https://fund.eastmoney.com/js/fundcode_search.js",
        headers=HEADERS,
        timeout=timeout,
    )
    r.encoding = "utf-8"
    r.raise_for_status()
    return parse_fundcode_search(r.text)
