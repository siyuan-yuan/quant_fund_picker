# -*- coding: utf-8 -*-
"""
量化选基系统 V3.1 — Web 控制台
手动触发爬取→计算→出榜, 本地运行
启动: python webapp.py  (默认运行于 0.0.0.0:8000，生产级 WSGI 服务)
"""
import os, glob, json, time, re, random, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

import provider, rbsa, factors, risk
from engine import score_fund, finalize, market_water, resolve_weights
from scan_market import build_universe
import holding_diag
from config import (
    RBSA_INDICES, OUTPUT_DIR,
    STRAT_VERSION, STRAT_BUY_TH, STRAT_SELL_TH, STRAT_SLOTS,
    STRAT_CASH_YIELD, STRAT_REBALANCE, STRAT_TRAIL_STOP,
    STRAT_CRISIS_MA, STRAT_CRISIS_VOL_WINDOW, STRAT_CRISIS_VOL_Q,
    STRAT_CPPI, STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1,
    STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2,
    STRAT_CPPI_DD3, STRAT_CPPI_SLOTS3,
    STRAT_CPPI_HYSTERESIS, STRAT_STYLE_CAP,
    STRAT_OVERLAP_SKIP, STRAT_OVERSEAS_SLOT_CAP,
    STRAT_INDEX_TOP1, STRAT_INDEX_PORT, STRAT_CLUSTER_MAX, STRAT_CLONE_L1,
    OVERSEAS_FUND_TYPES,
)

app = Flask(__name__)

# ---------------- 操作台账（买卖记录持久化，output/ledger.json） ----------------
LEDGER_FILE = os.path.join(OUTPUT_DIR, "ledger.json")


def _load_ledger() -> list:
    """读取台账记录；文件缺失/损坏返回 []"""
    try:
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("txns", [])
        if not isinstance(data, list):
            return []
        return [t for t in data if isinstance(t, dict)]
    except Exception:
        return []


def _save_ledger(txns: list) -> bool:
    """原子写台账（tmp+rename 防半写损坏）"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tmp = LEDGER_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "txns": txns}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, LEDGER_FILE)
        return True
    except Exception:
        return False


def _norm_txn(t) -> dict:
    """规范化单条台账记录 → 合法记录或 None"""
    if not isinstance(t, dict):
        return None
    code = str(t.get("code", "")).strip()
    if not re.fullmatch(r"\d{6}", code):
        return None
    d = _parse_date(t.get("date"))
    if not d:
        return None
    side = str(t.get("side", "")).strip().lower()
    if side in ("买", "买入", "b", "buy"):
        side = "buy"
    elif side in ("卖", "卖出", "s", "sell"):
        side = "sell"
    else:
        return None
    amt = _parse_yuan(t.get("amount"))
    if not amt or amt <= 0:
        return None
    tid = str(t.get("id") or f"l{int(time.time()*1000)}{random.randint(0,9999)}")
    isDca = bool(t.get("isDca", False))
    return {"id": tid, "code": code, "date": d, "side": side,
            "amount": round(float(amt), 2), "note": str(t.get("note", "") or "")[:120], "isDca": isDca}


def _ledger_by_code(txns: list) -> dict:
    """code -> [(date, side, amount), ...]（按日期排序，供持仓诊断）"""
    by = {}
    for t in txns:
        by.setdefault(t["code"], []).append((t["date"], t["side"], t["amount"]))
    return by


@app.get("/api/ledger")
def ledger_get():
    """返回台账全部记录 + 每只基金实时状态（份额/成本/市值/回撤/止损），+组合CPPI(cash可选)"""
    txns = _load_ledger()
    cash = _parse_yuan(request.args.get("cash")) if request.args.get("cash") else None
    by = _ledger_by_code(txns)
    states = []
    names = {}
    try:
        meta = provider.get_fund_meta()
        names = dict(zip(meta.index.astype(str), meta["基金简称"])) if len(meta) else {}
    except Exception:
        pass
    curves = {}
    for code in sorted(by.keys()):
        lots = by[code]
        st = None
        try:
            nav_df = provider.get_fund_nav(code)
            st = holding_diag.fund_lots_diag(lots, nav_df)
            if st.get("curve") is not None:
                curves[code] = st["curve"]
        except Exception as e:
            st = {"computable": False, "status": "no_data", "reason": str(e)[:100]}
        states.append({
            "code": code,
            "name": names.get(code, code),
            "lots": [{"date": d, "side": s, "amount": a} for d, s, a in lots],
            "state": {k: v for k, v in st.items() if k != "curve"},
        })
    # 组合级 CPPI（可选现金；无现金时仅基金侧状态）
    cppi = None
    if cash is not None and curves:
        cppi = holding_diag.portfolio_cppi([(c,) for c in curves.values()], cash=cash,
                                           rules=[(STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1),
                                                  (STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2),
                                                  (STRAT_CPPI_DD3, STRAT_CPPI_SLOTS3)],
                                           full_slots=STRAT_SLOTS, hysteresis=STRAT_CPPI_HYSTERESIS)
    return jsonify(clean(dict(ok=True, txns=txns, funds=states, cppi=cppi)))


@app.post("/api/ledger")
def ledger_post():
    """整体替换台账记录（前端持有全量，新增/删除后整存）；返回规范化后的记录"""
    body = request.get_json(silent=True) or {}
    raw = body.get("txns")
    if not isinstance(raw, list):
        return jsonify({"ok": False, "message": "txns 需要是数组"}), 400
    txns = [t for t in (_norm_txn(t) for t in raw) if t]
    if not _save_ledger(txns):
        return jsonify({"ok": False, "message": "台账写入失败（output 目录不可写？）"}), 500
    return jsonify(clean(dict(ok=True, txns=txns, message=f"已保存 {len(txns)} 条记录")))


@app.post("/api/ledger/import")
def ledger_import():
    """把持仓输入（代码 市值 [买入日期|成本|收益率]）转换为买入记录并入台账。
    只导入含买入日期或成本的持仓；仅市值+日期时成本按净值折算（锚定市值口径）。"""
    body = request.get_json(silent=True) or {}
    holdings_in = body.get("holdings")
    if not isinstance(holdings_in, list):
        return jsonify({"ok": False, "message": "holdings 需要是数组"}), 400
    txns = _load_ledger()
    existing_ids = {t["id"] for t in txns}
    existing_keys = {(t["code"], t["date"], t["side"], round(float(t["amount"]), 2)) for t in txns}
    added = 0
    for h in holdings_in:
        if not isinstance(h, dict):
            continue
        code = str(h.get("code", "")).zfill(6)
        if not re.fullmatch(r"\d{6}", code):
            continue
        amt = _parse_yuan(h.get("amount"))
        buy_date = _parse_date(h.get("buy_date"))
        cost = _parse_yuan(h.get("cost"))
        ret_pct = _parse_pct(h.get("ret_pct"))
        if not amt or amt <= 0:
            continue
        if buy_date is None and cost is None and ret_pct is None:
            continue          # 无入场信息，无法转成买入记录
        if buy_date is None:
            # 用净值反推入场日（收益率或 成本→收益率）
            try:
                adj = holding_diag.adj_series(provider.get_fund_nav(code))
                r_infer = ret_pct
                if r_infer is None and cost and cost > 0:
                    r_infer = (amt / cost - 1.0) * 100.0
                if r_infer is None:
                    continue
                d0, _ = holding_diag.infer_entry_date(adj, r_infer)
                if d0 is None:
                    continue
                buy_date = str(d0.date())
            except Exception:
                continue
        if cost is None or cost <= 0:
            # 只有市值+日期：成本 = 市值 × adj(买入日)/adj(now)（锚定口径）
            try:
                adj = holding_diag.adj_series(provider.get_fund_nav(code))
                d = pd.Timestamp(buy_date)
                pos = int(adj.index.searchsorted(d))
                pos = min(max(pos, 0), len(adj) - 1)
                cost = amt * float(adj.iloc[pos]) / float(adj.iloc[-1])
            except Exception:
                cost = amt
        t = _norm_txn({"code": code, "date": buy_date, "side": "buy", "amount": cost, "note": "导入"})
        key = (t["code"], t["date"], t["side"], round(t["amount"], 2)) if t else None
        if t and t["id"] not in existing_ids and key not in existing_keys:
            txns.append(t)
            existing_ids.add(t["id"])
            existing_keys.add(key)
            added += 1
    _save_ledger(txns)
    return jsonify(clean(dict(ok=True, txns=txns, added=added)))


@app.delete("/api/ledger")
def ledger_delete():
    _save_ledger([])
    return jsonify({"ok": True, "txns": []})


STATE = {"phase": "idle", "done": 0, "total": 0, "started": None,
         "elapsed": 0, "message": "空闲", "stamp": None, "scan_mode": "default"}
LOCK = threading.Lock()


def clean(o):
    """NaN/NaT/np类型 → JSON可序列化"""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        # V3.7.1: NaN 和 ±Inf 一并拦截(Inf曾致整批JSON非法)
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o



def _safe_list(v):
    return v if isinstance(v, list) else []


def _safe_dict(v):
    return v if isinstance(v, dict) else {}


def _parse_int(v, default, low=None, high=None):
    try:
        out = int(float(v))
    except (TypeError, ValueError):
        out = default
    if low is not None:
        out = max(low, out)
    if high is not None:
        out = min(high, out)
    return out


# ---------------- 页面 ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- 估值地形图 + 大盘水位计(V3.2) ----------------
def regime_label(w):
    if w is None or w != w:
        return "—"
    if w <= 0.20:
        return "🟢 极度低估 · 左侧权重已激活 (估值0.55/动量0.10)"
    if w >= 0.90:
        return "🔴 极度高估 · 全面防守"
    return "🟡 中性区 · 标准权重"


@app.get("/api/terrain")
def terrain():
    pct = rbsa.index_pe_percentile()
    kind_map = {"sina": "风格", "csindex": "行业", "us_sina": "境外", "hk_sina": "境外"}
    out = []
    for src, code, name, pe_key, tag in RBSA_INDICES:
        if pe_key == "none":
            out.append({"name": name, "error": "估值盲区(无PE源)"})
            continue
        try:
            pe = provider.get_pe_by_key(pe_key).dropna()
            v = pct.get(name)
            out.append({"name": name, "kind": kind_map.get(src, src),
                        "pe": round(float(pe.iloc[-1]), 2),
                        "pct": None if v is None else round(v * 100, 1),
                        "date": str(pe.index[-1].date())})
        except Exception as e:
            out.append({"name": name, "error": str(e)[:60]})
    w = market_water(None)
    _dates = [o["date"] for o in out if o.get("date")]
    return jsonify({"items": out, "water": None if w != w else round(w * 100, 1),
                    "water_style": "6风格等权PE分位", "regime": regime_label(w),
                    "asof": max(_dates) if _dates else None,
                    "asof_expected": provider.expected_last_td(),
                    "stale": provider.stale_warnings(),
                    "reversal": provider.market_reversal_signal("sh000300")})


# ---------------- V3.8 执行层策略状态 —— 抽出危机计算供复用 ----------------
def _compute_crisis():
    crisis = dict(
        enabled=True,
        index="沪深300",
        ma_window=STRAT_CRISIS_MA,
        vol_window=STRAT_CRISIS_VOL_WINDOW,
        vol_quantile=STRAT_CRISIS_VOL_Q,
        hs300_close=None,
        ma=None,
        vol20=None,
        vol_threshold=None,
        active=False,
        reason="",
        data_insufficient=False,
    )
    try:
        bench = provider.get_close_by_src("sina", "sh000300").dropna().sort_index()
        if len(bench) < STRAT_CRISIS_MA + STRAT_CRISIS_VOL_WINDOW + 1:
            crisis["data_insufficient"] = True
            crisis["reason"] = f"沪深300数据不足({len(bench)}行)，无法计算MA{STRAT_CRISIS_MA}或Vol{STRAT_CRISIS_VOL_WINDOW}"
        else:
            ma_series = bench.rolling(STRAT_CRISIS_MA).mean()
            ret = bench.pct_change()
            vol20_series = ret.rolling(STRAT_CRISIS_VOL_WINDOW).std() * sqrt(252)
            vol_threshold_series = vol20_series.expanding().quantile(STRAT_CRISIS_VOL_Q)
            last_close = float(bench.iloc[-1])
            last_ma = float(ma_series.iloc[-1])
            last_vol20 = float(vol20_series.iloc[-1])
            last_vol_th = float(vol_threshold_series.iloc[-1])
            crisis["hs300_close"] = round(last_close, 2)
            crisis["ma"] = round(last_ma, 2)
            crisis["vol20"] = round(last_vol20, 4)
            crisis["vol_threshold"] = round(last_vol_th, 4)
            below_ma = last_close < last_ma
            vol_extreme = last_vol20 > last_vol_th
            crisis_active = below_ma and vol_extreme
            crisis["active"] = crisis_active
            parts = []
            if below_ma:
                parts.append(f"沪深300({last_close:.0f}) < MA{STRAT_CRISIS_MA}({last_ma:.0f})")
            if vol_extreme:
                parts.append(f"Vol20({last_vol20:.2%}) > 历史{int(STRAT_CRISIS_VOL_Q*100)}%分位({last_vol_th:.2%})")
            crisis["reason"] = " 且 ".join(parts) if crisis_active else "不满足危机条件"
    except Exception as e:
        crisis["data_insufficient"] = True
        crisis["reason"] = f"沪深300数据获取失败: {str(e)[:80]}"
    return crisis


@app.get("/api/strategy_state")
def strategy_state():
    """返回 V3.8 组合执行层实时状态: 信号阈值 / 危机过滤 / CPPI / 买入决策"""
    signal = dict(
        buy_th=STRAT_BUY_TH,
        sell_th=STRAT_SELL_TH,
        hold_band=[STRAT_SELL_TH, STRAT_BUY_TH],
        slots=STRAT_SLOTS,
    )
    cash = dict(annual_yield=STRAT_CASH_YIELD)
    micro_risk = dict(trail_stop=STRAT_TRAIL_STOP)
    cppi_rules = [
        dict(drawdown_lte=STRAT_CPPI_DD1, max_slots=STRAT_CPPI_SLOTS1),
        dict(drawdown_lte=STRAT_CPPI_DD2, max_slots=STRAT_CPPI_SLOTS2),
        dict(drawdown_lte=STRAT_CPPI_DD3, max_slots=STRAT_CPPI_SLOTS3),
    ]
    cppi = dict(
        enabled=STRAT_CPPI,
        rules=cppi_rules,
        requires_portfolio_equity=True,
        note="网页版如果没有用户组合净值曲线，则只能展示规则，不能自动判断用户是否触发CPPI。",
    )
    crisis = _compute_crisis()
    crisis_active = crisis["active"] or crisis["data_insufficient"]
    new_buy_allowed = not crisis_active
    max_slots_by_macro = 0 if crisis_active else STRAT_SLOTS
    if crisis_active:
        msg = "危机模式：禁止新开权益仓，仅允许持仓按S卖出或止损退出"
    else:
        msg = "当前非危机，可按S信号买入"
    decision = dict(
        new_buy_allowed=new_buy_allowed,
        max_slots_by_macro=max_slots_by_macro,
        message=msg,
    )
    return jsonify(clean(dict(
        version=STRAT_VERSION,
        signal=signal,
        cash=cash,
        micro_risk=micro_risk,
        crisis=crisis,
        cppi=cppi,
        decision=decision,
    )))


# ---------------- 榜单读取 ----------------
def _latest_scan():
    files = sorted(glob.glob(f"{OUTPUT_DIR}/scan_*.csv"))
    return files[-1] if files else None


@app.get("/api/results")
def results():
    f = _latest_scan()
    if not f:
        return jsonify({"rows": [], "stamp": None})
    df = pd.read_csv(f, dtype={"code": str})
    df = df[df["error"].isna()] if "error" in df else df
    stamp = os.path.basename(f).split("_")[-1].split(".")[0]
    # V3.7.3: 榜单墙钟 vs 数据内容截至 — 两个时钟必须同框展示
    asof = str(df["last_date"].max())[:10] if "last_date" in df else None
    exp = provider.expected_last_td()
    # V3.9: 分市场榜单计数（A股/海外）
    n_a = n_ov = None
    if "region" in df:
        n_a = int((df["region"] == "A股").sum())
        n_ov = int((df["region"] == "海外").sum())
    return jsonify({"rows": clean(df.where(pd.notna(df), None).to_dict("records")),
                    "stamp": stamp, "n": len(df), "n_a": n_a, "n_ov": n_ov,
                    "asof": asof, "asof_expected": exp,
                    "asof_stale": bool(asof and asof < exp),
                    "stale": provider.stale_warnings()})


# ---------------- 定投导入 (DCA) ----------------
@app.post("/api/dca/preview")
def dca_preview():
    """生成定投买入序列（预览，不落库）。
    body: {code, start_date, amount, freq, end_date?}
    返回: lots + 序列当前市值估算（按真实复权净值折算）。"""
    body = request.get_json(silent=True) or {}
    code = str(body.get("code", "")).zfill(6)
    start = _parse_date(body.get("start_date"))
    amt = _parse_yuan(body.get("amount"))
    freq = str(body.get("freq") or "monthly").lower()
    if freq not in ("daily", "monthly", "biweekly", "weekly"):
        return jsonify({"ok": False, "message": "频率仅支持 daily / monthly / biweekly / weekly"}), 400
    if not re.fullmatch(r"\d{6}", code) or not start or not amt or amt <= 0:
        return jsonify({"ok": False, "message": "需要 基金代码 + 开始日期 + 每期金额"}), 400
    try:
        nav_df = provider.get_fund_nav(code)
    except Exception as e:
        return jsonify({"ok": False, "message": f"净值获取失败: {str(e)[:120]}"}), 400
    adj = holding_diag.adj_series(nav_df)
    end = _parse_date(body.get("end_date"))
    lots = holding_diag.dca_lots(start, amt, freq, end, adj=adj)
    if not lots:
        return jsonify({"ok": False, "message": "生成 0 期（开始日期晚于净值最新日？）"}), 400
    # 序列当前市值估算（每期按当日净值折份额 × 最新净值）
    now = float(adj.iloc[-1])
    n = len(adj)
    mv = 0.0
    for d, _, a in lots:
        pos = int(min(max(int(adj.index.searchsorted(pd.Timestamp(d))), 0), n - 1))
        mv += a * now / float(adj.iloc[pos])
    total = sum(a for _, _, a in lots)
    return jsonify(clean(dict(
        ok=True, code=code, freq=freq, freq_label=holding_diag.DCA_FREQ_LABELS.get(freq, freq),
        lots=[{"date": d, "amount": a} for d, _, a in lots],
        n=len(lots), first=lots[0][0], last=lots[-1][0],
        total=round(total, 2), implied_mv=round(mv, 2))))


@app.post("/api/dca/infer")
def dca_infer():
    """混合持仓反推定投参数。
    body: {code, total_mv, manual_lots?: [{date, amount, side?}], freqs?}
    manual_lots 缺省 → 自动读取台账中该基金的非定投记录（note 以"定投"开头的除外）。"""
    body = request.get_json(silent=True) or {}
    code = str(body.get("code", "")).zfill(6)
    total_mv = _parse_yuan(body.get("total_mv"))
    if not re.fullmatch(r"\d{6}", code):
        return jsonify({"ok": False, "message": "基金代码无效"}), 400
    if not total_mv or total_mv <= 0:
        return jsonify({"ok": False, "message": "需要当前总市值"}), 400
    freqs = body.get("freqs") or ("monthly", "biweekly", "weekly")
    manual = body.get("manual_lots")
    source = "用户输入"
    if isinstance(manual, list) and manual:
        lots = []
        for m in manual:
            if not isinstance(m, dict):
                continue
            d = _parse_date(m.get("date"))
            a = _parse_yuan(m.get("amount"))
            side = str(m.get("side", "buy")).lower()
            side = "sell" if side in ("sell", "卖", "s") else "buy"
            if d and a and a > 0:
                lots.append((d, side, a))
        if not lots:
            return jsonify({"ok": False, "message": "主动买入记录格式无效（每行: 日期 金额）"}), 400
    else:
        txns = [t for t in _load_ledger()
                if t.get("code") == code and not str(t.get("note", "")).startswith("定投")]
        lots = [(t["date"], t["side"], t["amount"]) for t in txns]
        source = f"台账（{len(lots)} 条主动记录）" if lots else "台账（无记录）"
    try:
        nav_df = provider.get_fund_nav(code)
    except Exception as e:
        return jsonify({"ok": False, "message": f"净值获取失败: {str(e)[:120]}"}), 400
    adj = holding_diag.adj_series(nav_df)
    r = holding_diag.infer_dca(lots, total_mv, adj, freqs=freqs)
    r.update(code=code, source=source)
    if r.get("candidates"):
        c = r["candidates"][0]
        r["message"] = (f"推荐：{holding_diag.DCA_FREQ_LABELS.get(c['freq'], c['freq'])}定投 "
                        f"{c['amount']:,.0f} 元/期 · 自 {c['start_date']} 起 · 共 {c['periods']} 期"
                        f"（点击候选填入上方生成序列）")
    return jsonify(clean(r))


# ---------------- 手动触发: 全市场扫描 ----------------
def _run_scan(right_n, left_n, workers=6, scan_mode="default"):
    t0 = time.time()
    try:
        mode_text = {"all_target": "全部目标类型", "all_main": "全部主池", "default": "全市场漏斗"}.get(scan_mode, "全市场漏斗")
        with LOCK:
            STATE.update(phase="universe", message=f"正在构建{mode_text}(拉取排行总库)...",
                         scan_mode=scan_mode)
        pool = build_universe(right_n, left_n, mode=scan_mode)
        chan = dict(zip(pool["基金代码"], pool["channel"]))
        codes = pool["基金代码"].tolist()
        with LOCK:
            STATE.update(phase="scoring", total=len(codes), done=0,
                         message="爬取净值/档案 → RBSA穿透 → 风控乘数 ...")
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(score_fund, c): c for c in codes}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"code": c, "name": c, "error": str(e)[:100]}
                r["channel"] = chan.get(c)
                rows.append(r)
                with LOCK:
                    STATE["done"] += 1
                    STATE["elapsed"] = round(time.time() - t0)
        df = finalize(rows)
        stamp = dt.date.today().strftime("%Y%m%d")
        # 动量腿参照列(mom_4m1m/mom_7m1m)落盘：单基透视/批量评分/持仓诊断的
        # 全市场参照快照(engine.get_global_ref_universe 读取)以它为 ECDF 标尺
        keep = ["code", "name", "ftype", "channel", "S_total", "rating", "F_value",
                "val_pct", "trend_ok", "trend_ma20", "bonus", "F_alpha", "ir_winrate",
                "down_capture", "F_momentum", "mom_4m1m", "mom_7m1m", "rank4", "rank7",
                "scale", "tenure_days", "is_passive", "penalty_str", "water",
                "weights_mode", "last_date", "error"]
        df[[k for k in keep if k in df]].to_csv(
            f"{OUTPUT_DIR}/scan_{stamp}.csv", index=False, encoding="utf-8-sig")
        with LOCK:
            STATE.update(phase="done", message=f"完成: 深算 {len(df)} 只", stamp=stamp,
                         elapsed=round(time.time() - t0))
    except Exception as e:
        with LOCK:
            STATE.update(phase="error", message=f"扫描失败: {str(e)[:200]}")


@app.post("/api/scan")
def scan():
    if STATE["phase"] in ("universe", "scoring"):
        return jsonify({"ok": False, "message": "扫描进行中"}), 409
    body = request.get_json(silent=True) or {}
    req_mode = str(body.get("scan_mode") or "").strip().lower()
    scan_all_main = bool(body.get("all_main") or body.get("scan_all_main"))
    scan_all_target = bool(body.get("all_target") or body.get("scan_all_target"))
    right = _parse_int(body.get("right", 400), 400, low=1, high=2000)
    left = _parse_int(body.get("left", 150), 150, low=0, high=1000)
    if req_mode in ("all_main", "all_target", "default"):
        scan_mode = req_mode
    elif scan_all_target:
        scan_mode = "all_target"
    elif scan_all_main:
        scan_mode = "all_main"
    else:
        scan_mode = "default"
    workers = 4 if scan_mode in ("all_target", "all_main") else 6
    boot_msg = {
        "all_main": "启动中（全部主池，耗时较长）...",
        "all_target": "启动中（全部目标类型，耗时较长）...",
        "default": "启动中...",
    }.get(scan_mode, "启动中...")
    STATE.update(phase="universe", done=0, total=0, started=time.time(), elapsed=0,
                 message=boot_msg, scan_mode=scan_mode)
    threading.Thread(target=_run_scan, args=(right, left, workers, scan_mode), daemon=True).start()
    return jsonify({"ok": True, "scan_mode": scan_mode})


@app.get("/api/scan/status")
def status():
    return jsonify(STATE)


# ---------------- 手动触发: 单基透视(以全市场动量作为参照系) ----------------
def _final_single(r: dict) -> dict:
    """给单基金补上截面动量排名 → F_momentum → S_total → 评级 (统一采用 engine.finalize 全域标尺)"""
    df = finalize([r], use_global_ref=True)
    return df.iloc[0].to_dict()


@app.post("/api/fund/<code>")
def fund(code):
    code = str(code).zfill(6)
    try:
        r = score_fund(code)
        if "error" in r and r.get("error"):
            return jsonify({"ok": False, "message": r["error"]}), 400
        return jsonify({"ok": True, "fund": clean(_final_single(r))})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)[:200]}), 400


# ---------------- 批量自选(支付宝持仓) ----------------
@app.post("/api/watchlist")
def watchlist():
    body = request.get_json(silent=True) or {}
    codes = [str(c).zfill(6) for c in body.get("codes", [])][:50]
    rows = []
    # V3.7: 并行打分(4线程), 单只异常不拖垮整批
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(c):
        try:
            return score_fund(c)
        except Exception as e:
            return {"code": c, "name": c, "error": str(e)[:100]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_one, c) for c in codes]
        for fut in as_completed(futs):
            rows.append(fut.result())
    df = finalize(rows, use_global_ref=True)
    # V3.6+: 与单基金透视同源的全量字段(摘要表+展开透视复用)
    cols = ["code", "name", "ftype", "S_total", "rating", "F_value", "val_pct",
            "val_coverage", "valuation_blind", "trend_ok", "bonus",
            "F_alpha", "ir_winrate", "s_ir", "down_capture", "s_dc",
            "F_momentum", "mom_4m1m", "mom_7m1m", "rbsa", "panel_mode",
            "tenure_days", "is_passive", "penalties", "penalty_detail",
            "penalty_str", "scale", "n_days", "last_date", "error",
            "model_version", "ref_stamp", "data_incomplete"]
    # V3.7.2 批判清单⑧: 组合级 RBSA 穿透 — 等权聚合整批隐形仓位, 单一板块≥35% 告警
    portfolio = None
    rb = [r["rbsa"] for r in rows if isinstance(r.get("rbsa"), dict) and r["rbsa"]]
    if rb:
        agg = pd.DataFrame(rb).fillna(0).mean().sort_values(ascending=False)
        flags = [dict(style=k, pct=round(float(v), 3)) for k, v in agg.items() if v >= STRAT_STYLE_CAP]
        portfolio = dict(n=len(rb), cap=STRAT_STYLE_CAP,
                         exposure={k: round(float(v), 3) for k, v in agg.items() if v > 0.001},
                         top=[dict(style=k, pct=round(float(v), 3))
                              for k, v in agg.head(6).items() if v > 0.01],
                         flags=flags)
    # V3.8: 组合交易纪律字段 — 展示规则但不假装判断CPPI触发
    portfolio_discipline = clean(dict(
        max_slots=STRAT_SLOTS,
        buy_threshold=STRAT_BUY_TH,
        sell_threshold=STRAT_SELL_TH,
        crisis_active=None,       # 需要实时计算，前端自行从 /api/strategy_state 获取
        new_buy_allowed=None,
        cppi_rules=[
            dict(drawdown_lte=STRAT_CPPI_DD1, max_slots=STRAT_CPPI_SLOTS1),
            dict(drawdown_lte=STRAT_CPPI_DD2, max_slots=STRAT_CPPI_SLOTS2),
            dict(drawdown_lte=STRAT_CPPI_DD3, max_slots=STRAT_CPPI_SLOTS3),
        ],
        note="CPPI需要用户组合净值/HWM，当前网页仅展示规则，不自动判断个人账户是否触发。",
    ))

    # 参照戳与模型版本：三入口一致性自检与前端披露用
    ref_stamp = str(df["ref_stamp"].iloc[0]) if "ref_stamp" in df and len(df) else None
    model_ver = str(df["model_version"].iloc[0]) if "model_version" in df and len(df) else None
    return jsonify(dict(ok=True, ref_stamp=ref_stamp, model_version=model_ver,
                        portfolio=clean(portfolio),
                        portfolio_discipline=portfolio_discipline,
                        rows=clean(df[[c for c in cols if c in df]].where(pd.notna(df), None).to_dict("records"))))


# ============================================================
# 新增：智能调仓引擎 /api/rebalance
#  用户输入：总可支配金额、可用现金、持仓列表(代码+金额)
#  策略自动：V3.8 信号阈值 + 危机过滤 + CPPI + 等权槽位 + RBSA集中度
# ============================================================
def _parse_yuan(v):
    """解析金额：支持 10000 / '1.5万' / '15,000' / '2w' 等；失败返回 None"""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except:
            return None
    s = str(v).strip().replace(",", "").replace("，", "").replace(" ", "")
    if not s:
        return None
    s_low = s.lower()
    mult = 1
    # 中文/英文万
    if s_low.endswith("万元") or s_low.endswith("万"):
        # 去掉后缀
        if s_low.endswith("万元"):
            s = s[:-2]
        else:
            s = s[:-1]
        mult = 10000
    elif s_low.endswith("w") or s_low.endswith("k"):
        # w=万, k=千
        suffix = s_low[-1]
        s = s[:-1]
        mult = 10000 if suffix == "w" else 1000
    elif s.endswith("元"):
        s = s[:-1]
    # 去掉可能的 '¥' '￥'
    s = s.replace("¥", "").replace("￥", "")
    try:
        return float(s) * mult
    except:
        return None


def _parse_pct(v):
    """解析收益率：支持 12.3 / '12.3%' / '+8.5%' / '-8.5%'；失败返回 None"""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except:
            return None
    s = str(v).strip().replace("%", "").replace("％", "").replace(",", "").replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except:
        return None


def _parse_date(v):
    """解析日期：YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD / YYYY.MM.DD；失败返回 None"""
    if v is None:
        return None
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except:
            continue
    return None


def _is_veto_row(row):
    """判断是否否决池：penalties含 -100% 或 S_total==0且有回撤惩罚"""
    ps = str(row.get("penalty_str") or "")
    if "-100%" in ps:
        return True
    pens = _safe_list(row.get("penalties"))
    for _, p in pens:
        try:
            if float(p) >= 0.999:
                return True
        except:
            pass
    return False


@app.post("/api/rebalance")
def rebalance():
    """
    请求 JSON:
    {
      "total_capital": 100000,  // 总可支配金额（可选，0则自动=持仓+现金）
      "cash": 20000,             // 可用现金
      "holdings": [ {"code":"110011","amount":25000}, ... ]  // amount 支持数字/字符串含万
      // 兼容：也支持 holdings_text: "110011 2.5万\n161725 30000"
    }
    返回：持仓诊断 + 买卖指令 + 目标配置
    """
    body = request.get_json(silent=True) or {}
    # ---- 解析总资本 & 现金 ----
    total_capital_raw = body.get("total_capital", body.get("totalCapital", 0))
    cash_raw = body.get("cash", body.get("available_cash", 0))
    total_capital = _parse_yuan(total_capital_raw)
    cash = _parse_yuan(cash_raw)
    if cash is None:
        cash = 0.0
    if total_capital is None:
        total_capital = 0.0
    # ---- 解析持仓 ----
    holdings_in = body.get("holdings", None)
    # 兼容 holdings_text：每行 "代码 市值 [买入日期|成本|收益率%]"
    if (not holdings_in) and body.get("holdings_text"):
        txt = str(body.get("holdings_text"))
        holdings_in = []
        for line in re.split(r"[\n;]+", txt):
            line=line.strip()
            if not line:
                continue
            # 按空白/逗号/冒号切
            parts = re.split(r"[\s,，、:：]+", line)
            if not parts or not parts[0].strip().isdigit():
                continue
            code = parts[0].strip().zfill(6)
            amt = _parse_yuan(parts[1]) if len(parts)>1 else None
            h = {"code": code, "amount": amt}
            if len(parts) > 2:
                p3 = parts[2].strip()
                d = _parse_date(p3)
                if d:
                    h["buy_date"] = d
                elif p3.endswith("%") or p3.endswith("％"):
                    h["ret_pct"] = _parse_pct(p3)
                else:
                    c = _parse_yuan(p3)
                    if c is not None:
                        h["cost"] = c
            holdings_in.append(h)
    if not isinstance(holdings_in, list):
        holdings_in = []
    # 标准化
    norm_holdings = []
    for h in holdings_in:
        buy_date = cost = ret_pct = None
        if isinstance(h, (list, tuple)) and len(h)>=1:
            code = str(h[0]).zfill(6)
            amt = _parse_yuan(h[1]) if len(h)>1 else None
            if len(h) > 2:
                p3 = str(h[2]).strip()
                d = _parse_date(p3)
                if d:
                    buy_date = d
                elif p3.endswith("%") or p3.endswith("％"):
                    ret_pct = _parse_pct(p3)
                else:
                    cost = _parse_yuan(p3)
        elif isinstance(h, dict):
            code = str(h.get("code", h.get("symbol",""))).zfill(6)
            # amount 字段多种别名（市值）
            amt_raw = h.get("amount", h.get("value", h.get("market_value", h.get("amt", h.get("holding", None)))))
            amt = _parse_yuan(amt_raw)
            buy_date = _parse_date(h.get("buy_date", h.get("date", h.get("entry_date", None))))
            cost = _parse_yuan(h.get("cost", None))
            ret_pct = _parse_pct(h.get("ret_pct", h.get("return_pct", None)))
        else:
            code = str(h).zfill(6)
            amt = None
        if not re.fullmatch(r"\d{6}", code):
            continue
        norm_holdings.append({"code": code, "amount": amt, "buy_date": buy_date, "cost": cost, "ret_pct": ret_pct})
        if len(norm_holdings) >= 50:
            break
    # ---- 操作台账并入：台账基金自动加入诊断（无需每次重输持仓，新增操作只记台账）----
    use_ledger = bool(body.get("use_ledger", True))
    ledger_by = _ledger_by_code(_load_ledger()) if use_ledger else {}
    if ledger_by:
        seen = {h["code"] for h in norm_holdings}
        for code in ledger_by:
            if code not in seen:
                norm_holdings.append({"code": code, "amount": None, "buy_date": None,
                                      "cost": None, "ret_pct": None, "_ledger": True})
                seen.add(code)
    # V3.9: 允许空组合（无持仓但有可用现金）生成纯买入方案；
    # 只有"既无持仓也无现金/总资金"才拒绝
    if not norm_holdings and not cash and not total_capital:
        return jsonify({"ok": False, "message": "请至少输入 1 只持仓基金代码，或在操作台账中添加买入记录，或填写可用现金"}), 400

    # ---- 危机 & 策略快照 ----
    crisis = _compute_crisis()
    crisis_active = bool(crisis.get("active") or crisis.get("data_insufficient"))
    # 注: max_slots_by_macro 在 CPPI 计算后定义（crisis → 0, 否则 10）；slots_eff 为最终生效槽位
    # ---- 评分持仓 ----
    codes = [h["code"] for h in norm_holdings]
    code_to_amount = {h["code"]: (h["amount"] if h["amount"] is not None else 0.0) for h in norm_holdings}
    # 去重 codes
    uniq_codes = list(dict.fromkeys(codes))
    rows = []

    def _friendly_err(e: str) -> str:
        s = str(e)
        if "RemoteDisconnected" in s or "Connection aborted" in s or "ConnectionAborted" in s:
            return "数据源繁忙（天天基金限流/远端断开），请稍后重试或分批重试（建议≤5只/次）"
        if "净值历史不足" in s:
            return s
        if "timeout" in s.lower() or "timed out" in s.lower():
            return "数据源超时，请稍后重试"
        return s[:120]

    def _one(c):
        # 轻量重试：首次并发易触发限流，失败后退避 1.5s 再试一次；仍失败则返回友好错误
        last_e = None
        for attempt in range(2):
            try:
                r = score_fund(c)
                # score_fund 内已返回 error 字段的视为业务错误，不重试
                if r.get("error"):
                    # 净值不足等直接返回，但把限流类错误转友好
                    if "RemoteDisconnected" in str(r.get("error")) or "Connection aborted" in str(r.get("error")):
                        r["error"] = _friendly_err(r.get("error"))
                    return r
                return r
            except Exception as e:
                last_e = e
                msg = str(e)
                is_retryable = ("RemoteDisconnected" in msg or "Connection aborted" in msg or "timeout" in msg.lower())
                if is_retryable and attempt == 0:
                    time.sleep(1.8)
                    continue
                return {"code": c, "name": c, "error": _friendly_err(msg)}
        return {"code": c, "name": c, "error": _friendly_err(last_e) if last_e else "未知错误"}

    # 降低并发以避开东财限流：2 线程最稳，5只以内几乎不触发限流
    workers = 2 if len(uniq_codes) > 4 else 3
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, c): c for c in uniq_codes}
        for fut in as_completed(futs):
            rows.append(fut.result())
    # 统一全局评分口径：采用 use_global_ref=True 确保与单基透视、自选池完全统一
    # 保留 error 行也进入 finalize，但单独处理
    try:
        df = finalize(rows, use_global_ref=True)
    except Exception as e:
        # finalize 异常时回退为原 rows
        df = pd.DataFrame(rows)
        if "S_total" not in df:
            df["S_total"] = np.nan
    # 参照戳与模型版本（finalize 回退路径可能没有这些列 → 防御读取）
    ref_stamp = str(df["ref_stamp"].iloc[0]) if "ref_stamp" in df and len(df) else None
    model_ver = str(df["model_version"].iloc[0]) if "model_version" in df and len(df) else None
    # 映射回金额
    holdings_detail = []
    # 为便于查找，建立 code->row 映射（df 已按 S_total 排序，但映射不依赖顺序）
    row_by_code = {}
    for _, r in df.iterrows():
        row_by_code[str(r.get("code")).zfill(6)] = r.to_dict()
    # 也包含完全失败的 rows（可能不在 df）
    for r in rows:
        c = str(r.get("code")).zfill(6)
        if c not in row_by_code:
            row_by_code[c] = r

    # 计算组合级 RBSA
    rb_list = [r["rbsa"] for r in rows if isinstance(r.get("rbsa"), dict) and r["rbsa"]]
    portfolio_rbsa = None
    if rb_list:
        try:
            agg = pd.DataFrame(rb_list).fillna(0).mean().sort_values(ascending=False)
            flags = [dict(style=k, pct=round(float(v),3)) for k,v in agg.items() if v >= STRAT_STYLE_CAP]
            portfolio_rbsa = dict(
                n=len(rb_list), cap=STRAT_STYLE_CAP,
                exposure={k: round(float(v),3) for k,v in agg.items() if v>0.001},
                top=[dict(style=k, pct=round(float(v),3)) for k,v in agg.head(6).items() if v>0.01],
                flags=flags
            )
        except Exception:
            portfolio_rbsa = None

    scored_holdings = []
    for h in norm_holdings:
        code = h["code"]
        amt = code_to_amount.get(code, 0.0) or 0.0
        rec = row_by_code.get(code, {"code": code, "name": code})
        # ---- 入场高点回撤止损：用真实净值历史自动计算 ----
        # 路径A（操作台账）：多笔买入/卖出 → FIFO 份额与成本 → 持仓曲线 → 回撤
        # 路径B（输入行）：代码+市值+买入日期/成本/收益率 → 单笔（内部同样走多笔引擎）
        stop = None
        stop_curve = None
        try:
            nav_df = provider.get_fund_nav(code)
            if ledger_by.get(code):
                stop = holding_diag.fund_lots_diag(
                    ledger_by[code], nav_df,
                    anchor_amount=amt if amt and amt > 0 else None,
                    stop=STRAT_TRAIL_STOP)
            else:
                # 用户输入(买入日期/成本/收益率)合并进评分行，供净值曲线反推入场
                rec_in = dict(rec) if isinstance(rec, dict) else {"code": code, "name": code}
                for _k in ("amount", "buy_date", "cost", "ret_pct"):
                    if h.get(_k) is not None:
                        rec_in[_k] = h[_k]
                stop = holding_diag.fund_stop_diag(rec_in, nav_df, stop=STRAT_TRAIL_STOP)
            if isinstance(stop, dict):
                stop_curve = stop.pop("curve", None)
        except Exception:
            stop = None
        # ---- 关键修复：pandas 的 NaN 误判为真错误 ----
        raw_err = rec.get("error")
        # pandas 的 NaN、None、空串、字符串 "nan" 均视为无错
        if raw_err is None or (isinstance(raw_err, float) and pd.isna(raw_err)) or str(raw_err).strip().lower() in ("", "nan", "none"):
            is_err = False
            err_msg = None
            is_retryable = False
        else:
            is_err = True
            err_msg = str(raw_err)[:180]
            # 二次友好化（兜底）
            if "RemoteDisconnected" in err_msg or "Connection aborted" in err_msg:
                err_msg = "数据源繁忙（天天基金限流/远端断开），请稍后重试或分批重试（建议≤5只/次）"
            # 新基历史不足不提供重试（重试也不会成功）
            is_retryable = not any(k in err_msg for k in ["成立仅", "历史不足", "200天", "观察仓"])
        # 统一全局字段：直接采用 use_global_ref 统一打分的 finalize 结果，无需冗余重算
        s_total = rec.get("S_total")
        try:
            if s_total is None or (isinstance(s_total, float) and pd.isna(s_total)):
                s_val = None
            else:
                fv = float(s_total)
                s_val = None if not np.isfinite(fv) else fv
        except:
            s_val = None
        veto = _is_veto_row(rec) if not is_err else False
        rating = rec.get("rating") or ""
        ftype = rec.get("ftype") or ""
        penalties = _safe_list(rec.get("penalties"))
        penalty_detail = _safe_dict(rec.get("penalty_detail"))
        rbsa_detail = _safe_dict(rec.get("rbsa"))
        # 台账路径下当前市值由份额×净值自动算出（锚定用户填写值）；用户没填市值时直接用计算值
        if isinstance(stop, dict) and stop.get("mv_now"):
            amt = float(stop["mv_now"])
        holdings_detail.append({
            "code": code,
            "name": rec.get("name") or code,
            "ftype": ftype,
            "amount": round(float(amt),2),
            "from_ledger": bool(ledger_by.get(code)),
            "mv_computed": round(float(stop["mv_now"]), 2) if isinstance(stop, dict) and stop.get("mv_now") else None,
            "basis": round(float(stop["basis"]), 2) if isinstance(stop, dict) and stop.get("basis") else None,
            "lots_n": int(stop["lots_n"]) if isinstance(stop, dict) and stop.get("lots_n") else None,
            "over_sell": bool(stop.get("over_sell")) if isinstance(stop, dict) else False,
            "stop_curve": stop_curve,
            "S_total": None if s_val is None or not np.isfinite(s_val) else round(float(s_val),1),
            "rating": rating,
            "F_value": rec.get("F_value"),
            "F_alpha": rec.get("F_alpha"),
            "F_momentum": rec.get("F_momentum"),
            "val_pct": rec.get("val_pct"),
            "ir_winrate": rec.get("ir_winrate"),
            "penalty_str": rec.get("penalty_str") or "",
            "penalties": penalties,
            "penalty_detail": penalty_detail,
            "rbsa": rbsa_detail,
            "is_passive": rec.get("is_passive"),
            "scale": rec.get("scale"),
            "tenure_days": rec.get("tenure_days"),
            "last_date": rec.get("last_date"),
            "error": err_msg,
            "is_veto": veto,
            "is_error": is_err,
            "retryable": is_retryable if is_err else False,
            "data_incomplete": bool(rec.get("data_incomplete")),
            "stop": stop,
        })
        scored_holdings.append({
            "code": code, "s": s_val, "veto": veto, "err": is_err, "amt": amt, "rec": rec
        })

    # ---- 台账中已全部卖出的基金剔除出组合（flat），提示但不参与诊断 ----
    flat_warns = []
    flat_codes = [h["code"] for h in holdings_detail
                  if isinstance(h.get("stop"), dict) and h["stop"].get("flat")]
    if flat_codes:
        holdings_detail = [h for h in holdings_detail if h["code"] not in flat_codes]
        for fc in flat_codes:
            flat_warns.append(f"{fc} 在操作台账中已全部卖出，已从本次诊断中剔除")

    # ---- 组合级 CPPI：真实净值重建组合曲线 → HWM → 动态槽位 ----
    # 仅需"买入日期(或收益率) + 成本(或市值)"即可自动计算，无需用户维护净值曲线
    cppi_rules = [
        (STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1),
        (STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2),
        (STRAT_CPPI_DD3, STRAT_CPPI_SLOTS3),
    ]
    fund_series = []
    for h in holdings_detail:
        st = h.get("stop") or {}
        # 优先用台账/单笔引擎算出的持仓市值曲线（已锚定当前市值，含多笔买卖）
        if st.get("computable") and h.get("stop_curve") is not None and len(h["stop_curve"]) >= 5:
            fund_series.append((h["stop_curve"],))
            continue
        # 兜底：单笔输入重建 市值×复权净值比 曲线
        scale = (st.get("amount") or 0) if (st.get("amount") or 0) > 0 else (st.get("cost") or 0)
        if st.get("computable") and st.get("entry_date") and scale > 0:
            try:
                nav_df = provider.get_fund_nav(h["code"])
                fund_series.append((st["entry_date"], scale, holding_diag.adj_series(nav_df)))
            except Exception:
                continue
    # Series 曲线仅用于计算，不进入 JSON 响应
    for h in holdings_detail:
        h.pop("stop_curve", None)
    cppi = holding_diag.portfolio_cppi(
        fund_series, cash=cash if cash and cash > 0 else 0.0,
        rules=cppi_rules, full_slots=STRAT_SLOTS, hysteresis=STRAT_CPPI_HYSTERESIS,
    )
    cppi_ok = bool(cppi.get("computable"))
    cppi_slots = int(cppi.get("slots", STRAT_SLOTS)) if cppi_ok else STRAT_SLOTS
    # 生效槽位 = min(宏观槽位(危机=0), CPPI档位槽位)
    max_slots_by_macro = 0 if crisis_active else STRAT_SLOTS
    slots_eff = min(cppi_slots, max_slots_by_macro)

    # ---- 资产汇总 ----
    holdings_value = round(float(sum(h["amount"] for h in holdings_detail)),2)
    # 若 total_capital 未给或为0，则自动 = 持仓 + 现金
    if total_capital <= 0:
        total_capital = holdings_value + cash
    # 防止 total_capital 小于已有资产：仍以输入为准，但标记警告
    per_slot = round(total_capital / slots_eff, 2) if total_capital>0 and slots_eff>0 else 0.0
    # ---- 持仓诊断 & 操作分类 ----
    # V3.8 规则：S<45 卖出；S>=70 候选买入/持有；45-70持有
    sells = []
    holds = []
    keeps = []  # 最终保留的持仓（holds中的 非卖出）
    for h in holdings_detail:
        if h["is_error"]:
            # 无法评分的，建议人工复核，默认持有不动
            h["action"] = "hold"
            h["action_label"] = "待复核"
            h["action_reason"] = f"评分失败：{h['error']}"
            h["target_amount"] = h["amount"]
            holds.append(h)
            keeps.append(h)
            continue
        # V3.8 移动止损（自动计算）：自入场日起高点回撤 > 20% → 清仓
        st = h.get("stop") or {}
        if st.get("status") == "triggered":
            h["action"] = "sell"
            h["action_label"] = "纪律卖出·入场高点回撤20%"
            dd_txt = f"{st['dd']*100:.1f}%" if st.get("dd") is not None else "—"
            h["action_reason"] = (f"自入场日({st.get('entry_date')})高点 {st.get('peak')} 回撤 {dd_txt}，"
                                  f"已击穿20%移动止损线（触发价 {st.get('trigger_nav')}）→ 清仓"
                                  + ("（买入日由收益率推断）" if st.get("inferred") else "")
                                  + ("；反推疑似多笔/定投买入，回撤为近似值，建议先在操作台账核对每笔买卖再执行"
                                     if st.get("infer_ambiguous") else ""))
            h["target_amount"] = 0.0
            h["sell_amount"] = h["amount"]
            sells.append(h)
            continue
        # CPPI 清仓档（回撤≤-25%）：组合整体清仓，等待右侧信号重启（HWM重置纪律）
        if cppi_ok and cppi_slots == 0:
            rs = cppi.get("restore") or {}
            h["action"] = "sell"
            h["action_label"] = "纪律卖出·CPPI清仓档"
            h["action_reason"] = (f"组合自高点 {cppi['hwm']:,.0f} 元回撤 {cppi['dd']*100:.1f}%，跌破-25%清仓线，"
                                  + (f"待回升至 {rs.get('value',0):,.0f} 元（-{abs(rs.get('dd',0))*100:.0f}%）恢复 {rs.get('slots',3)} 槽或出现右侧信号后再重启" if rs else "待右侧信号重启"))
            h["target_amount"] = 0.0
            h["sell_amount"] = h["amount"]
            sells.append(h)
            continue
        s = h["S_total"]
        veto = h["is_veto"]
        if veto or (s is not None and s < STRAT_SELL_TH):
            h["action"] = "sell"
            if veto:
                h["action_label"] = "纪律卖出·否决池"
                h["action_reason"] = h["penalty_str"] or "否决池"
            else:
                h["action_label"] = f"纪律卖出·S<{STRAT_SELL_TH:.0f}"
                h["action_reason"] = f"S={s} 低于卖出线 {STRAT_SELL_TH:.0f}"
            h["target_amount"] = 0.0
            h["sell_amount"] = h["amount"]
            sells.append(h)
        else:
            # 持有
            h["action"] = "hold"
            if s is not None and s >= STRAT_BUY_TH:
                h["action_label"] = "持有·强"
                h["action_reason"] = f"S={s}≥{STRAT_BUY_TH:.0f}，符合买入线，保留至目标权重"
            elif s is not None and s >= 50:
                h["action_label"] = "持有·观望"
                h["action_reason"] = f"S={s} 在持有带 [{STRAT_SELL_TH:.0f},{STRAT_BUY_TH:.0f}]，保留"
            else:
                h["action_label"] = "持有"
                h["action_reason"] = f"S={s} 未触发卖出，保留"
            # 目标金额：持有仓位建议向每槽目标对齐（不做强制再平衡，仅提示）
            # 若已持有且为强信号且金额显著低于目标，提示可补至目标；否则维持现额但不超过目标1.2倍提示可减
            if h["amount"] < per_slot * 0.9 and s is not None and s >= 70:
                h["target_amount"] = per_slot
                h["rebalance_hint"] = f"可补仓至 {per_slot:,.0f} 元/槽"
            elif h["amount"] > per_slot * 1.25:
                h["target_amount"] = per_slot
                h["rebalance_hint"] = f"仓位偏重，可考虑减至 {per_slot:,.0f} 元/槽（季度再平衡）"
            else:
                h["target_amount"] = h["amount"]
                h["rebalance_hint"] = "权重适中，持有不动"
            holds.append(h)
            keeps.append(h)

    # 处理超槽位：若保留持仓数 > 允许槽位（常见于从未调仓的老组合持有20只），则把最弱的持仓加入卖出
    # 仅在非危机且未触发 CPPI 极端时生效；危机时 max_slots=0 不做强制清退（按纪律仅禁新买）
    # 槽位上限 = 生效槽位 slots_eff（宏观×CPPI 动态档位）
    num_keep = len(keeps)
    extra_sells = []
    if not crisis_active and slots_eff > 0 and num_keep > slots_eff:
        # 按 S 升序把多余的转为卖出
        keeps_sorted = sorted(keeps, key=lambda x: (x["S_total"] if x["S_total"] is not None else -1))
        overflow = num_keep - slots_eff
        for h in keeps_sorted[:overflow]:
            # 已是卖出的不重复
            if h["action"] == "sell":
                continue
            h["action"] = "sell"
            h["action_label"] = "纪律卖出·超槽位"
            h["action_reason"] = f"持仓数{num_keep}超过生效槽位上限{slots_eff}（CPPI档位），按 S 最弱优先退出（S={h['S_total']}）"
            h["target_amount"] = 0.0
            h["sell_amount"] = h["amount"]
            extra_sells.append(h)
        # 重算 keeps/sells
        sells = [h for h in holdings_detail if h["action"]=="sell"]
        keeps = [h for h in holdings_detail if h["action"]=="hold"]

    sell_proceeds = round(float(sum(h.get("sell_amount", h["amount"]) for h in sells)),2)
    cash_after_sells = round(cash + sell_proceeds,2)
    num_keep_after = len(keeps)
    free_slots = max(0, slots_eff - num_keep_after) if not crisis_active else 0
    # 危机时即使有 free_slots 也不允许新买，故强制 0
    if crisis_active:
        free_slots = 0

    # ---- 候选买入池（V3.9）：分市场配额 + 组合重复度过滤 + 顺位推荐 ----
    # 原则：① 不跨市场混比——A股/海外各自按 S 降序；② 海外(美股QDII)最多占
    #       STRAT_OVERSEAS_SLOT_CAP 槽（防单边堆满）；③ 重复度三档规则（详见 config）：
    #         Tier0 复制盘：候选与任一持仓/已选候选的 RBSA 暴露 L1 ≤ STRAT_CLONE_L1
    #               → 同策略孪生产品（如 001801 达欣 vs 001417 医疗服务），必排除；
    #         Tier1 指数级：候选 top1≥0.40 且该风格组合已实质持有 → 必排除（同指数）；
    #         Tier2 簇上限：同 top1 风格最多 STRAT_CLUSTER_MAX 只 → 第2只起顺位换风格；
    #       ④ 全部满足后仍按 S 高者优先（市场内）。
    candidates = []
    dup_skips = []
    scan_msg = ""
    buy_note = None
    cand_stats = None     # 建议买入统计: gt70=榜单S>70总数, total=未持有候选数, dup=已排除重复数
    scan_file = _latest_scan()
    if scan_file and free_slots>0:
        try:
            df_scan = pd.read_csv(scan_file, dtype={"code": str})
            # 过滤错误行
            if "error" in df_scan:
                df_scan = df_scan[df_scan["error"].isna()]
            # 旧榜单无 region → 按基金类型归类；无 rbsa → 重复度过滤自动降级
            if "region" not in df_scan:
                df_scan["region"] = np.where(df_scan["ftype"].isin(OVERSEAS_FUND_TYPES), "海外", "A股")
            has_rbsa = "rbsa" in df_scan.columns

            def _parse_rbsa(v):
                if isinstance(v, dict):
                    return v
                if isinstance(v, str):
                    v = v.strip()
                    if v.startswith("{"):
                        try:
                            return json.loads(v)
                        except Exception:
                            return {}
                return {}

            held_codes = set(h["code"] for h in holdings_detail)
            filt = df_scan[(pd.to_numeric(df_scan["S_total"], errors="coerce") > STRAT_BUY_TH)
                           & (~df_scan["code"].isin(list(held_codes)))].copy()
            cand_stats = dict(
                gt70=int((pd.to_numeric(df_scan["S_total"], errors="coerce") > STRAT_BUY_TH).sum()),
                total=int(len(filt)),
                dup=0,
            )
            if has_rbsa:
                filt["rbsa"] = filt["rbsa"].map(_parse_rbsa)
            else:
                filt["rbsa"] = {}
            rows_cand = []
            for _, r in filt.iterrows():
                rows_cand.append({
                    "code": str(r["code"]).zfill(6),
                    "name": r.get("name",""),
                    "S_total": float(r["S_total"]) if pd.notna(r["S_total"]) else None,
                    "rating": r.get("rating",""),
                    "F_value": r.get("F_value") if pd.notna(r.get("F_value")) else None,
                    "F_alpha": r.get("F_alpha") if pd.notna(r.get("F_alpha")) else None,
                    "F_momentum": r.get("F_momentum") if pd.notna(r.get("F_momentum")) else None,
                    "val_pct": float(r["val_pct"]) if pd.notna(r.get("val_pct")) else None,
                    "ir_winrate": float(r["ir_winrate"]) if "ir_winrate" in r and pd.notna(r["ir_winrate"]) else None,
                    "channel": r.get("channel",""),
                    "penalty_str": r.get("penalty_str","") or "",
                    "last_date": r.get("last_date",""),
                    "region": r.get("region", "A股"),
                    "rbsa": _parse_rbsa(r.get("rbsa")) if has_rbsa else {},
                })
            rows_cand.sort(key=lambda c: -(c["S_total"] or 0))
            a_rows = [c for c in rows_cand if c["region"] == "A股"]
            ov_rows = [c for c in rows_cand if c["region"] == "海外"]
            # 组合已有暴露 = 保留持仓 RBSA 暴露加总（被卖出的不占）
            port_expo = {}
            for h in keeps:
                for k, v in (_safe_dict(h.get("rbsa")) or {}).items():
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if fv > 0:
                        port_expo[k] = port_expo.get(k, 0.0) + fv
            # 持仓的 top1 风格簇：凡实质暴露（top1≥STRAT_INDEX_PORT=0.15）即占用
            # 该簇 1 个名额，与候选侧"同风格最多1只"口径对称。
            # 【2026-08-11 修复】此前仅 top1≥0.40 的指数级持仓才占簇，导致弥散型
            #   主动持仓(如 519770/013107 top1=全指信息0.225~0.229, 相互收益相关
            #   0.91)不占簇 → 持有其一仍推荐另一只, 同风格实际持有两只。
            cluster_count = {}
            cluster_owner = {}
            for h in keeps:
                rb = _safe_dict(h.get("rbsa"))
                if not rb:
                    continue
                t1 = max(rb.items(), key=lambda kv: kv[1])
                try:
                    t1w = float(t1[1])
                except (TypeError, ValueError):
                    continue
                if t1w >= STRAT_INDEX_PORT:
                    cluster_count[t1[0]] = cluster_count.get(t1[0], 0) + 1
                    cluster_owner.setdefault(t1[0], f"{h['code']} {h.get('name') or ''}".strip())
            # 复制盘参照池：保留持仓的 RBSA 暴露。Tier2 按 top1 风格簇判重，
            # 仍检不出"暴露逐格几乎一致、但 top1 相同风格被表述为不同簇"的极端
            # 孪生（以及 RBSA 键集不同的跨市场孪生）；由 Tier0 逐对 L1 距离兜底。
            # 每选中一只候选也并入参照池，候选之间同样互查
            clone_refs = []
            for h in keeps:
                rb_h = _safe_dict(h.get("rbsa"))
                if rb_h:
                    clone_refs.append((h["code"], h.get("name") or h["code"], rb_h))
            # 贪心选取：市场内保持 S 降序；跨市场按 S 高者优先（海外受配额限制）；
            # 重复度三档规则（详见 config）：
            #   Tier0 复制盘：候选与任一持仓/已选候选 RBSA 暴露 L1≤STRAT_CLONE_L1
            #         → 同一策略孪生产品（复制盘），必排除，只留一只；
            #   Tier1 指数级重复：候选 top1≥0.40 且该风格组合已实质持有 → 必排除（同指数）
            #   Tier2 簇上限：同 top1 风格最多 STRAT_CLUSTER_MAX 只 → 第2只起顺位换风格
            # 被排除的记入 dup_skips（前端展示原因）
            ov_cap = min(STRAT_OVERSEAS_SLOT_CAP, free_slots)
            ov_used = 0
            ia = ib = 0
            while len(candidates) < free_slots:
                ca = a_rows[ia] if ia < len(a_rows) else None
                co = ov_rows[ib] if (ib < len(ov_rows) and ov_used < ov_cap) else None
                if ca is None and co is None:
                    break
                if co is not None and (ca is None or co["S_total"] >= (ca["S_total"] or 0)):
                    cand = co; ib += 1; cand_overseas = True
                else:
                    cand = ca; ia += 1; cand_overseas = False
                cand["overlap"] = None
                cand["dup_reason"] = ""
                if has_rbsa and cand.get("rbsa"):
                    rb = cand["rbsa"]
                    top1_style, top1_w = max(rb.items(), key=lambda kv: kv[1])
                    cand["overlap"] = round(top1_w, 3)
                    # Tier0 复制盘：与任一持仓/已选候选暴露逐格几乎一致 → 只留一只
                    clone_ref, clone_l1 = holding_diag.find_clone_exposure(rb, clone_refs, STRAT_CLONE_L1)
                    if clone_ref is not None:
                        cand["dup_reason"] = (f"复制盘重复（与 {clone_ref['code']} {clone_ref['name']} "
                                              f"暴露几乎一致，L1={clone_l1:.3f}）")
                        dup_skips.append(cand)
                        continue
                    if top1_w >= STRAT_INDEX_TOP1 and port_expo.get(top1_style, 0.0) >= STRAT_INDEX_PORT:
                        cand["dup_reason"] = f"同指数重复（{top1_style} 已持有）"
                        dup_skips.append(cand)
                        continue
                    if cluster_count.get(top1_style, 0) >= STRAT_CLUSTER_MAX:
                        owner = cluster_owner.get(top1_style)
                        owner_txt = f"已由 {owner} 占用" if owner else f"已选{STRAT_CLUSTER_MAX}只"
                        cand["dup_reason"] = f"同风格重复（{top1_style} {owner_txt}）"
                        dup_skips.append(cand)
                        continue
                    cluster_count[top1_style] = cluster_count.get(top1_style, 0) + 1
                    cluster_owner.setdefault(top1_style, f"{cand['code']} {cand.get('name') or ''}".strip())
                    clone_refs.append((cand["code"], cand.get("name") or cand["code"], rb))
                    # 已选候选也并入暴露池（供①的"已实质持有"判断）
                    for k, v in rb.items():
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        if fv > 0:
                            port_expo[k] = port_expo.get(k, 0.0) + fv
                # 海外配额只计"真正入选"的候选——被去重跳过的海外候选不占配额，
                # 让位给顺位下的下一只海外候选
                if cand_overseas:
                    ov_used += 1
                candidates.append(cand)
            if cand_stats is not None:
                cand_stats["dup"] = len(dup_skips)
            if not candidates and not dup_skips:
                scan_msg = "扫描榜单中暂无 S>70 的候选（或均已持有），建议等待新扫描或放宽槽位"
            else:
                n_a = sum(1 for c in candidates if c["region"] == "A股")
                n_ov = len(candidates) - n_a
                scan_msg = (f"候选 {len(candidates)} 只（A股 {n_a} / 海外 {n_ov}，海外≤{STRAT_OVERSEAS_SLOT_CAP}槽）："
                            f"市场内按 S 排序 + 重复度过滤（复制盘/同指数/同风格自动顺位）")
                if not has_rbsa:
                    scan_msg += "；旧榜单无暴露数据，重复度过滤未生效（重新扫描后自动开启）"
        except Exception as e:
            scan_msg = f"读取扫描榜单失败：{str(e)[:80]}"
    elif free_slots==0 and not crisis_active:
        if num_keep_after >= STRAT_SLOTS:
            scan_msg = f"当前已满 {num_keep_after}/{STRAT_SLOTS} 槽，无空槽可建仓"
        else:
            scan_msg = "无空槽"
    elif crisis_active:
        scan_msg = "危机模式：禁止新开权益仓，不提供买入候选"
    elif not scan_file:
        scan_msg = "暂无扫描榜单数据，请先执行“全市场扫描”以生成候选池"

    # ---- 计算买入金额分配 ----
    buys = []
    cash_remaining = cash_after_sells
    orders = []
    # 先把卖出订单加入
    for h in sells:
        orders.append({
            "side": "SELL",
            "code": h["code"],
            "name": h["name"],
            "amount": round(float(h.get("sell_amount", h["amount"])),2),
            "reason": h["action_reason"],
            "S": h["S_total"],
        })
    if candidates and free_slots>0:
        # V3.9 等权槽位买入：先把可用现金按生效槽位平分成"份"，
        # 每只候选买 1 份（不超过每槽目标 per_slot），剩余现金保留现金池。
        #   份 = min(每槽目标, 可用现金 ÷ 槽位数)   （现金不足时每份按槽缩水）
        #   份 ≥ 最低买入额 10 元；可买只数 = min(候选数, 现金能买起的份数)
        # 例: 现金100 + 10槽 → 每份10元, 5个候选各买10元, 花50剩50（不花光）
        MIN_BUY = 10.0
        slots_now = slots_eff if slots_eff > 0 else STRAT_SLOTS
        share = 0.0
        if cash_after_sells > 0:
            share = cash_after_sells / slots_now
        if per_slot > 0 and per_slot < share:
            share = per_slot
        share = round(max(share, MIN_BUY), 2) if share > 0 else 0.0
        n_buy = len(candidates)
        if share > 0:
            n_buy = min(n_buy, int(cash_after_sells // share))
        per_buy = share
        for c in candidates:
            if len(buys) >= n_buy:
                break
            if cash_remaining < share:
                break
            buy_amt = share
            # 最低 10 元门槛（基金申购起点）
            if buy_amt < MIN_BUY:
                continue
            c["suggested_amount"] = round(buy_amt,2)
            # 估算目标占比
            c["target_pct"] = round(buy_amt/total_capital*100,2) if total_capital>0 else 0
            buys.append(c)
            cash_remaining = round(cash_remaining - buy_amt,2)
            ov_txt = "海外候选" if c.get("region") == "海外" else "A股候选"
            dup_txt = f" · 与组合重叠 {c['overlap']*100:.0f}%" if c.get("overlap") is not None else ""
            orders.append({
                "side": "BUY",
                "code": c["code"],
                "name": c["name"],
                "amount": round(buy_amt,2),
                "reason": f"S={c['S_total']:.1f} 候选买入 · {c.get('rating','')} · {ov_txt}{dup_txt}",
                "S": c["S_total"],
                "region": c.get("region", "A股"),
            })
        # 若现金有剩余，说明槽位未填满，提示
        if buys and cash_remaining > 1000:
            # 可选：把剩余现金留在现金池吃 2.5% 收益，不强制用完
            pass
        # V3.9: 有候选但一只都没买成 → 明确原因（避免"推荐0只"无解释）
        buy_note = None
        if candidates and not buys:
            if cash_after_sells < MIN_BUY:
                buy_note = (f"可用现金仅 {cash_after_sells:,.0f} 元（<{MIN_BUY:.0f} 元），"
                            f"暂不买入；请补充可用现金或减少持仓")
            elif share and cash_after_sells < share:
                buy_note = (f"可用现金 {cash_after_sells:,.0f} 元低于每份 {share:,.0f} 元"
                            f"（现金÷{slots_now}槽），暂不买入")
            else:
                buy_note = "候选均未达到买入条件，建议检查现金与门槛"

    # ---- 目标配置 & 风险提示 ----
    total_invested_after = sum(h["target_amount"] for h in keeps) + sum(b.get("suggested_amount",0) for b in buys)
    # 若有持仓未约定金额（amount=0），则 total_invested_after 可能偏小；用 total_capital 归一
    # 生成仓位分布明细
    allocation = []
    for h in keeps:
        pkt = round(h["target_amount"]/total_capital*100,1) if total_capital>0 else 0
        allocation.append({"code": h["code"], "name": h["name"], "amount": h["target_amount"], "pct": pkt, "S": h["S_total"], "type": "hold"})
    for b in buys:
        pkt = b.get("target_pct", round(b["suggested_amount"]/total_capital*100,1) if total_capital>0 else 0)
        allocation.append({"code": b["code"], "name": b["name"], "amount": b["suggested_amount"], "pct": pkt, "S": b["S_total"], "type": "buy"})
    # 现金占比
    cash_pct_after = round(cash_remaining/total_capital*100,1) if total_capital>0 else 0

    warnings = []
    warnings.extend(flat_warns)
    # 集中度
    if portfolio_rbsa and portfolio_rbsa.get("flags"):
        for f in portfolio_rbsa["flags"]:
            warnings.append(f"赛道集中告警：{f['style']} 占比 {f['pct']*100:.0f}%≥35% 上限，建议分散")
    # 持仓中个别高风险
    for h in holdings_detail:
        if h["is_error"]:
            warnings.append(f"{h['code']} {h['name']} 评分失败，暂不纳入纪律，需人工复核")
        else:
            if h.get("data_incomplete"):
                warnings.append(f"{h['code']} {h['name']} 档案数据缺失（数据源限流），任期类风控已豁免、评分仅供参考，建议稍后重试")
            if h.get("penalty_detail", {}).get("R_MDD") and h["penalty_detail"]["R_MDD"] and h["penalty_detail"]["R_MDD"]>2.0:
                warnings.append(f"{h['code']} 超额回撤比 {h['penalty_detail']['R_MDD']} 偏高（>2.0 毒性区），即使暂未触发卖出也建议控制仓位")
        # 单基 20% 移动止损接近线（真实净值自动计算）
        st = h.get("stop") or {}
        if st.get("status") == "near":
            warnings.append(f"{h['code']} {h['name']} 自入场日({st.get('entry_date')})高点 {st.get('peak')} 已回撤 {st['dd']*100:.1f}%，接近20%移动止损线（触发价 {st.get('trigger_nav')}），建议收紧仓位")
        elif st.get("status") == "triggered":
            warnings.append(f"{h['code']} {h['name']} 已触发20%移动止损（自入场高点回撤 {st['dd']*100:.1f}%）→ 清仓")
        elif st.get("status") in ("need_entry", "no_data") and st.get("reason"):
            warnings.append(f"{h['code']} {h['name']}：{st['reason']}")
    # 危机
    if crisis_active:
        warnings.append(f"危机模式已激活（{crisis.get('reason','')}），已禁止新开仓，现有持仓仅按 S<{STRAT_SELL_TH:.0f} 或 20%移动止损退出")
    # CPPI 提示（真实净值自动计算 → 动态槽位）
    if total_capital>0 and holdings_value + cash >0 and STRAT_CPPI:
        if cppi_ok:
            w = (f"CPPI 动态槽位：组合净值自高点 {cppi['hwm']:,.0f} 元回撤 {cppi['dd']*100:.1f}% → {cppi['tier_name']}（{cppi['slots']} 槽）")
            if cppi_slots == 0:
                w += "；已进入清仓档，全部持仓转卖出，等待右侧信号重启"
            if cppi.get("next_trigger"):
                w += f"；再跌至 {cppi['next_trigger']['value']:,.0f} 元（-{abs(cppi['next_trigger']['dd'])*100:.0f}%）降为 {cppi['next_trigger']['slots']} 槽"
            if cppi.get("restore"):
                w += f"；回升至 {cppi['restore']['value']:,.0f} 元（回撤 {cppi['restore']['dd']*100:.0f}% 内）恢复 {cppi['restore']['slots']} 槽"
            warnings.append(w)
        else:
            warnings.append(f"CPPI 风险预算：回撤≤-15%限6槽 / ≤-20%限3槽 / ≤-25%清仓（{cppi.get('reason','提供买入日期/成本后自动计算真实触发状态')}）")
    # 现金不足提示（V3.9 等权槽位：每份=share，剩余现金保留）
    if buys and per_slot > 0 and per_slot * len(candidates) > cash_after_sells:
        warnings.append(f"可用现金 {cash_after_sells:,.0f} 元不足以按每槽 {per_slot:,.0f} 元买满候选，已按每份 {share:,.0f} 元买入 {len(buys)} 只，剩余现金保留（吃 {STRAT_CASH_YIELD*100:.1f}% 现金收益）")
    # 空槽
    if free_slots==0 and not crisis_active and num_keep_after < STRAT_SLOTS:
        pass

    summary = dict(
        total_capital=round(float(total_capital),2),
        cash_before=round(float(cash),2),
        holdings_value_before=holdings_value,
        total_before=round(holdings_value+cash,2),
        per_slot=per_slot,
        sell_proceeds=sell_proceeds,
        cash_after_sells=cash_after_sells,
        num_holdings_before=len(holdings_detail),
        num_sells=len(sells),
        num_keeps=num_keep_after,
        free_slots_before=free_slots,
        num_buys=len(buys),
        total_invested_after=round(float(total_invested_after),2),
        cash_after=round(float(cash_remaining),2),
        cash_pct_after=cash_pct_after,
        max_slots=STRAT_SLOTS,
        max_slots_by_macro=max_slots_by_macro,
        cppi_slots=cppi_slots if cppi_ok else STRAT_SLOTS,
        slots_eff=slots_eff,
        cppi_tier=cppi.get("tier_name") if cppi_ok else None,
        portfolio_dd=round(cppi["dd"], 4) if cppi_ok else None,
        crisis_active=crisis_active,
    )

    resp = dict(
        ok=True,
        strategy=dict(
            version=STRAT_VERSION,
            buy_th=STRAT_BUY_TH, sell_th=STRAT_SELL_TH, slots=STRAT_SLOTS,
            per_slot=per_slot,
            cash_yield=STRAT_CASH_YIELD,
            trail_stop=STRAT_TRAIL_STOP,
            rebalance=STRAT_REBALANCE,
            overseas_slot_cap=STRAT_OVERSEAS_SLOT_CAP,
            overlap_skip=STRAT_OVERLAP_SKIP,
            cluster_max=STRAT_CLUSTER_MAX,
            index_top1=STRAT_INDEX_TOP1,
            clone_l1=STRAT_CLONE_L1,
            cppi_rules=[
                dict(dd=STRAT_CPPI_DD1, slots=STRAT_CPPI_SLOTS1),
                dict(dd=STRAT_CPPI_DD2, slots=STRAT_CPPI_SLOTS2),
                dict(dd=STRAT_CPPI_DD3, slots=STRAT_CPPI_SLOTS3),
            ],
            crisis=crisis,
            max_slots_by_macro=max_slots_by_macro,
            slots_eff=slots_eff,
        ),
        cppi=clean(cppi),
        portfolio=portfolio_rbsa,
        summary=summary,
        holdings=clean(holdings_detail),
        sells=clean(sells),
        keeps=clean(keeps),
        buys=clean(buys),
        candidates=clean(candidates),
        dup_skips=clean(dup_skips),
        cand_stats=cand_stats,
        buy_note=buy_note,
        allocation=clean(allocation),
        orders=clean(orders),
        warnings=warnings,
        scan_msg=scan_msg,
        ref_stamp=ref_stamp,
        model_version=model_ver,
        asof=dict(expected=provider.expected_last_td(), ref=ref_stamp),
    )
    return jsonify(clean(resp))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("============================================================", flush=True)
    print(" 量化选基系统 V3.8 — 生产级 WSGI 引擎启动 (Production WSGI Server)", flush=True)
    print("============================================================", flush=True)
    print(" * Running on all addresses (0.0.0.0:8000)", flush=True)
    print("============================================================", flush=True)
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8000, threads=8)
    except ImportError:
        from wsgiref.simple_server import make_server
        server = make_server("0.0.0.0", 8000, app)
        server.serve_forever()