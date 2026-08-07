# -*- coding: utf-8 -*-
"""
量化选基系统 V3.1 — Web 控制台
手动触发爬取→计算→出榜, 本地运行
启动: python webapp.py  (默认运行于 0.0.0.0:8000，生产级 WSGI 服务)
"""
import os, glob, json, time, re, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt

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
    STRAT_STYLE_CAP,
)

app = Flask(__name__)
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
    return jsonify({"rows": clean(df.where(pd.notna(df), None).to_dict("records")),
                    "stamp": stamp, "n": len(df),
                    "asof": asof, "asof_expected": exp,
                    "asof_stale": bool(asof and asof < exp),
                    "stale": provider.stale_warnings()})


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
        # S_v4_raw(V4原始z值)必须落盘：它作为单基透视/批量评分/持仓诊断的
        # 全市场V4参照快照(engine.get_global_ref_universe 读取)，缺列则三入口V4闸门降级
        keep = ["code", "name", "ftype", "channel", "S_total", "S_v4_raw", "rating", "F_value",
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
    # 兼容 holdings_text
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
            holdings_in.append({"code": code, "amount": amt})
    if not isinstance(holdings_in, list):
        holdings_in = []
    # 标准化
    norm_holdings = []
    for h in holdings_in:
        if isinstance(h, (list, tuple)) and len(h)>=1:
            code = str(h[0]).zfill(6)
            amt = _parse_yuan(h[1]) if len(h)>1 else None
        elif isinstance(h, dict):
            code = str(h.get("code", h.get("symbol",""))).zfill(6)
            # amount 字段多种别名
            amt_raw = h.get("amount", h.get("value", h.get("market_value", h.get("amt", h.get("holding", None)))))
            amt = _parse_yuan(amt_raw)
        else:
            code = str(h).zfill(6)
            amt = None
        if not re.fullmatch(r"\d{6}", code):
            continue
        norm_holdings.append({"code": code, "amount": amt})
        if len(norm_holdings) >= 50:
            break
    if not norm_holdings:
        return jsonify({"ok": False, "message": "请至少输入 1 只持仓基金代码（6位数字）"}), 400

    # ---- 危机 & 策略快照 ----
    crisis = _compute_crisis()
    crisis_active = bool(crisis.get("active") or crisis.get("data_insufficient"))
    max_slots_by_macro = 0 if crisis_active else STRAT_SLOTS
    # CPPI 默认不自动触发（需用户净值曲线），仅展示规则
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
        holdings_detail.append({
            "code": code,
            "name": rec.get("name") or code,
            "ftype": ftype,
            "amount": round(float(amt),2),
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
        })
        scored_holdings.append({
            "code": code, "s": s_val, "veto": veto, "err": is_err, "amt": amt, "rec": rec
        })

    # ---- 资产汇总 ----
    holdings_value = round(float(sum(h["amount"] for h in holdings_detail)),2)
    # 若 total_capital 未给或为0，则自动 = 持仓 + 现金
    if total_capital <= 0:
        total_capital = holdings_value + cash
    # 防止 total_capital 小于已有资产：仍以输入为准，但标记警告
    per_slot = round(total_capital / STRAT_SLOTS, 2) if total_capital>0 else 0.0
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
    num_keep = len(keeps)
    extra_sells = []
    if not crisis_active and num_keep > STRAT_SLOTS:
        # 按 S 升序把多余的转为卖出
        keeps_sorted = sorted(keeps, key=lambda x: (x["S_total"] if x["S_total"] is not None else -1))
        overflow = num_keep - STRAT_SLOTS
        for h in keeps_sorted[:overflow]:
            # 已是卖出的不重复
            if h["action"] == "sell":
                continue
            h["action"] = "sell"
            h["action_label"] = "纪律卖出·超槽位"
            h["action_reason"] = f"持仓数{num_keep}超过槽位上限{STRAT_SLOTS}，按 S 最弱优先退出（S={h['S_total']}）"
            h["target_amount"] = 0.0
            h["sell_amount"] = h["amount"]
            extra_sells.append(h)
        # 重算 keeps/sells
        sells = [h for h in holdings_detail if h["action"]=="sell"]
        keeps = [h for h in holdings_detail if h["action"]=="hold"]

    sell_proceeds = round(float(sum(h.get("sell_amount", h["amount"]) for h in sells)),2)
    cash_after_sells = round(cash + sell_proceeds,2)
    num_keep_after = len(keeps)
    free_slots = max(0, max_slots_by_macro - num_keep_after) if not crisis_active else 0
    # 危机时即使有 free_slots 也不允许新买，故强制 0
    if crisis_active:
        free_slots = 0

    # ---- 候选买入池：从最新扫描榜单取 S>70 且不在持仓的 Top ----
    candidates = []
    scan_msg = ""
    scan_file = _latest_scan()
    if scan_file and free_slots>0:
        try:
            df_scan = pd.read_csv(scan_file, dtype={"code": str})
            # 过滤错误行
            if "error" in df_scan:
                df_scan = df_scan[df_scan["error"].isna()]
            held_codes = set(h["code"] for h in holdings_detail)
            # S>70 且不在持仓
            filt = df_scan[(pd.to_numeric(df_scan["S_total"], errors="coerce") > STRAT_BUY_TH) & (~df_scan["code"].isin(list(held_codes)))]
            filt = filt.sort_values("S_total", ascending=False).head(free_slots*3)  # 多取 3 倍，现金不够时再筛
            # 按 S 降序取前 free_slots
            for _, r in filt.head(free_slots).iterrows():
                candidates.append({
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
                })
            if not candidates:
                scan_msg = "扫描榜单中暂无 S>70 的候选（或均已持有），建议等待新扫描或放宽槽位"
            else:
                scan_msg = f"已从扫描榜单挑选 Top{len(candidates)} 候选"
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
        # 均分可用现金，但每只不超过 per_slot，且不低于 1000 元才执行（避免碎单）
        per_buy = per_slot if per_slot>0 else 0
        # 若现金不足以按 per_slot 填满，则按现金均分
        total_need = per_buy * len(candidates)
        if total_need > cash_after_sells and cash_after_sells>0:
            per_buy = cash_after_sells / len(candidates)
        per_buy = round(per_buy,2)
        for c in candidates:
            if cash_remaining < 1000:
                break
            buy_amt = min(per_buy, cash_remaining)
            # 最低 100 元门槛（基金申购起点）
            if buy_amt < 100:
                continue
            c["suggested_amount"] = round(buy_amt,2)
            # 估算目标占比
            c["target_pct"] = round(buy_amt/total_capital*100,2) if total_capital>0 else 0
            buys.append(c)
            cash_remaining = round(cash_remaining - buy_amt,2)
            orders.append({
                "side": "BUY",
                "code": c["code"],
                "name": c["name"],
                "amount": round(buy_amt,2),
                "reason": f"S={c['S_total']:.1f} 候选买入 · {c.get('rating','')}",
                "S": c["S_total"],
            })
        # 若现金有剩余，说明槽位未填满，提示
        if buys and cash_remaining > 1000:
            # 可选：把剩余现金留在现金池吃 2.5% 收益，不强制用完
            pass

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
    # 危机
    if crisis_active:
        warnings.append(f"危机模式已激活（{crisis.get('reason','')}），已禁止新开仓，现有持仓仅按 S<{STRAT_SELL_TH:.0f} 或 20%移动止损退出")
    # CPPI 提示
    if total_capital>0 and holdings_value + cash >0:
        # 无法计算用户回撤，仅提示规则
        warnings.append(f"CPPI 风险预算：回撤≤-15%限6槽 / ≤-20%限3槽 / ≤-25%清仓（需跟踪你组合自高点回撤，本页仅提示规则）")
    # 现金不足
    if buys and total_need > cash_after_sells:
        warnings.append(f"可用现金 {cash_after_sells:,.0f} 元不足以按每槽 {per_slot:,.0f} 元填满 {len(candidates)} 个候选，已按均分 {per_buy:,.0f} 元/只 调整；可追加现金或减少持仓")
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
            cppi_rules=[
                dict(dd=STRAT_CPPI_DD1, slots=STRAT_CPPI_SLOTS1),
                dict(dd=STRAT_CPPI_DD2, slots=STRAT_CPPI_SLOTS2),
                dict(dd=STRAT_CPPI_DD3, slots=STRAT_CPPI_SLOTS3),
            ],
            crisis=crisis,
            max_slots_by_macro=max_slots_by_macro,
        ),
        portfolio=portfolio_rbsa,
        summary=summary,
        holdings=clean(holdings_detail),
        sells=clean(sells),
        keeps=clean(keeps),
        buys=clean(buys),
        candidates=clean(candidates),
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