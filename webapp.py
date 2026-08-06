# -*- coding: utf-8 -*-
"""
量化选基系统 V3.1 — Web 控制台
手动触发爬取→计算→出榜, 本地运行
启动: python webapp.py  然后浏览器打开 http://127.0.0.1:8000
"""
import os, glob, json, time, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

import provider, rbsa, factors, risk
from engine import score_fund, finalize, market_water, resolve_weights
from scan_market import build_universe
from config import (
    RBSA_INDICES, OUTPUT_DIR,
    STRAT_VERSION, STRAT_BUY_TH, STRAT_SELL_TH, STRAT_SLOTS,
    STRAT_CASH_YIELD, STRAT_REBALANCE, STRAT_TRAIL_STOP,
    STRAT_CRISIS_MA, STRAT_CRISIS_VOL_WINDOW, STRAT_CRISIS_VOL_Q,
    STRAT_CPPI, STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1,
    STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2,
    STRAT_CPPI_DD3, STRAT_CPPI_SLOTS3,
)

app = Flask(__name__)
STATE = {"phase": "idle", "done": 0, "total": 0, "started": None,
         "elapsed": 0, "message": "空闲", "stamp": None}
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


# ---------------- V3.8 执行层策略状态 ----------------
@app.get("/api/strategy_state")
def strategy_state():
    """返回 V3.8 组合执行层实时状态: 信号阈值 / 危机过滤 / CPPI / 买入决策"""
    from math import sqrt

    # ---- 信号 & 现金 & 微观风控 ----
    signal = dict(
        buy_th=STRAT_BUY_TH,
        sell_th=STRAT_SELL_TH,
        hold_band=[STRAT_SELL_TH, STRAT_BUY_TH],
        slots=STRAT_SLOTS,
    )
    cash = dict(annual_yield=STRAT_CASH_YIELD)
    micro_risk = dict(trail_stop=STRAT_TRAIL_STOP)

    # ---- CPPI 规则 ----
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

    # ---- 危机过滤计算 (沪深300) ----
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
            # 用截至当日的全部历史(非未来函数)计算分位数阈值
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

    # ---- 综合买入决策 ----
    crisis_active = crisis["active"] or crisis["data_insufficient"]
    new_buy_allowed = not crisis_active
    # 危机模式: 0槽; 非危机: 满槽
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
    return jsonify({"rows": clean(df.where(pd.notna(df), None).to_dict("records")),
                    "stamp": stamp, "n": len(df),
                    "asof": asof, "asof_expected": exp,
                    "asof_stale": bool(asof and asof < exp),
                    "stale": provider.stale_warnings()})


# ---------------- 手动触发: 全市场扫描 ----------------
def _run_scan(right_n, left_n, workers=6):
    t0 = time.time()
    try:
        with LOCK:
            STATE.update(phase="universe", message="正在构建全市场漏斗(拉取排行总库)...")
        pool = build_universe(right_n, left_n)
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
    right = int(body.get("right", 400))
    left = int(body.get("left", 150))
    STATE.update(phase="universe", done=0, total=0, started=time.time(), elapsed=0,
                 message="启动中...")
    threading.Thread(target=_run_scan, args=(right, left), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/scan/status")
def status():
    return jsonify(STATE)


# ---------------- 手动触发: 单基透视(以最近扫描池为动量参照系) ----------------
def _final_single(r: dict) -> dict:
    """给单基金补上截面动量排名 → F_momentum → S_total → 评级"""
    f = _latest_scan()
    rk4 = rk7 = None
    if f and r.get("mom_4m1m") is not None:
        df = pd.read_csv(f, dtype={"code": str})
        p4 = pd.to_numeric(df.get("mom_4m1m"), errors="coerce").dropna()
        p7 = pd.to_numeric(df.get("mom_7m1m"), errors="coerce").dropna()
        if len(p4) > 10:
            rk4 = float((p4 <= r["mom_4m1m"]).mean())
            rk7 = float((p7 <= r["mom_7m1m"]).mean())
    fm = factors.momentum_score_smooth_m1(rk4, rk7) if rk4 is not None else None
    water = market_water(None)
    (wv, wa, wm), mode = resolve_weights(water)
    num, den = 0.0, 0.0
    if r.get("F_value") is not None:
        num += wv * min(r["F_value"], 100); den += wv
    if r.get("F_alpha") is not None:
        num += wa * r["F_alpha"]; den += wa
    if fm is not None:
        num += wm * fm; den += wm
    st = risk.apply_penalties(num / den if den > 1e-9 else 0.0, r.get("penalties") or [])
    r["water"] = None if water != water else round(water, 4)
    r["weights_mode"] = mode
    band = next((lab for th, lab in
                 [(85, "Strong Buy 绿灯"), (70, "Buy 浅绿"),
                  (50, "Hold 黄灯"), (0, "Sell/Avoid 红灯")] if st >= th),
                "Sell/Avoid 红灯")
    r.update(rank4=rk4, rank7=rk7,
             F_momentum=None if fm is None else round(fm, 1),
             S_total=round(st, 1), rating=band)
    return r


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
    df = finalize(rows)
    # V3.6+: 与单基金透视同源的全量字段(摘要表+展开透视复用)
    cols = ["code", "name", "ftype", "S_total", "rating", "F_value", "val_pct",
            "val_coverage", "valuation_blind", "trend_ok", "bonus",
            "F_alpha", "ir_winrate", "s_ir", "down_capture", "s_dc",
            "F_momentum", "mom_4m1m", "mom_7m1m", "rbsa", "panel_mode",
            "tenure_days", "is_passive", "penalties", "penalty_detail",
            "penalty_str", "scale", "n_days", "last_date", "error"]
    # V3.7.2 批判清单⑧: 组合级 RBSA 穿透 — 等权聚合整批隐形仓位, 单一板块≥35% 告警
    portfolio = None
    rb = [r["rbsa"] for r in rows if isinstance(r.get("rbsa"), dict) and r["rbsa"]]
    if rb:
        agg = pd.DataFrame(rb).fillna(0).mean().sort_values(ascending=False)
        from config import STRAT_STYLE_CAP
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

    return jsonify(dict(ok=True, portfolio=clean(portfolio),
                        portfolio_discipline=portfolio_discipline,
                        rows=clean(df[[c for c in cols if c in df]].where(pd.notna(df), None).to_dict("records"))))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
