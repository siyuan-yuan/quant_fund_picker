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
from config import RBSA_INDICES, OUTPUT_DIR

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
    return jsonify({"items": out, "water": None if w != w else round(w * 100, 1),
                    "water_style": "6风格等权PE分位", "regime": regime_label(w),
                    "reversal": provider.market_reversal_signal("sh000300")})


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
    return jsonify({"rows": clean(df.where(pd.notna(df), None).to_dict("records")),
                    "stamp": stamp, "n": len(df)})


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
    return jsonify({"ok": True, "rows": clean(
        df[[c for c in cols if c in df]].where(pd.notna(df), None).to_dict("records"))})


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
