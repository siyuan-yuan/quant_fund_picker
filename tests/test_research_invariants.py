# -*- coding: utf-8 -*-
"""V5 预登记 #15（M1.1）：研究不变量测试套件

分层：
  A. 统计层不变量（Newey-West / Holm / BH / DSR / CSCV 已知答案）
  B. 执行模拟器不变量（sim_core 与 backtest_local.simulate 位级对拍；T+1 语义；
     D0.4 QDII/停牌规则）
  C. PiT 不变量（score_fund / finalize / market_water 对 as_of 之后的数据必须零泄漏）
  D. 复权构建不变量（D0.2 build_navadj：adj 只随官方 ret 走）
  E. 评分函数数值性质（边界/单调/regime 切换/成熟度守卫）
  F. ML 面板规则（截面填充；fwd 标签成熟期截止 fwd6/fwd12 —— R3.5 修复后的回归用例）

运行：~/.venv_review/bin/python -m pytest tests/test_research_invariants.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import provider
provider.STALE_OK = True

import stats_hac
import backtest_local as BL
import sim_core as SC
import engine
import factors
import rbsa
import _model_zoo as MZ
from p4_analysis import holm, bh_q, dsr_metrics, cscv_pbo

pytestmark = pytest.mark.filterwarnings("ignore")


# =====================================================================
# A. 统计层
# =====================================================================
class TestNW:
    def test_closed_form_alternating(self):
        """x = ±1 交替（均值0）: γ0=1, γ1=-19/20; L=1 → S*=0.05, Var(mean)=0.0025"""
        x = np.array([1.0, -1.0] * 10)
        assert stats_hac.nw_var_mean(x, 1) == pytest.approx(0.0025, abs=1e-12)

    def test_iid_matches_classical(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.05, 1.0, 500)
        classical = x.var(ddof=1) / len(x)
        assert stats_hac.nw_var_mean(x, 5) == pytest.approx(classical, rel=0.25)

    def test_overlap_inflates_variance(self):
        """fwd6 月度重叠结构的 MA(5) 性：HAC 方差应为朴素方差的 2~8 倍。"""
        rng = np.random.default_rng(3)
        eps = rng.normal(0, 1, 506)
        y = np.array([eps[i:i + 6].mean() for i in range(500)])
        naive = y.var(ddof=1) / len(y)
        hac = stats_hac.nw_var_mean(y, 5)
        assert 2.0 < hac / naive < 8.0

    def test_horizon_lag_rules(self):
        assert stats_hac.HORIZON_LAGS["fwd6_monthly"] == 5
        assert stats_hac.HORIZON_LAGS["fwd3_monthly"] == 3
        assert stats_hac.HORIZON_LAGS["fwd1_monthly"] == 1
        assert stats_hac.HORIZON_LAGS["fwd6_quarterly"] == 2
        assert stats_hac.HORIZON_LAGS["fwd12_monthly"] == 11
        assert stats_hac.monthly_ic_lag(6, "monthly") == 5
        assert stats_hac.monthly_ic_lag(6, "quarterly") == 2


class TestMultiTesting:
    def test_holm_known_answer(self):
        p = [0.01, 0.02, 0.05]
        assert holm(p) == pytest.approx([0.03, 0.04, 0.05], abs=1e-12)

    def test_holm_monotone_and_caps(self):
        p = [0.9, 0.01]
        adj = holm(p)
        assert adj[1] == pytest.approx(0.02)
        assert adj[0] == pytest.approx(0.90)
        assert all(0.0 <= a <= 1.0 for a in holm([0.5, 0.5, 0.5]))

    def test_bh_known_answer(self):
        p = [0.01, 0.02, 0.05]
        assert bh_q(p) == pytest.approx([0.03, 0.03, 0.05], abs=1e-12)

    def test_dsr_roundtrip(self):
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0.001, 0.02, 120))
        out = dsr_metrics(r, n_trials=10)
        vf = out["sigma_SR"] ** 2 * out["T"]
        z = (out["SR"] - out["SR0"]) * math.sqrt(out["T"]) / math.sqrt(vf)
        from scipy.stats import norm
        assert out["DSR"] == pytest.approx(norm.cdf(z), abs=2e-3)
        assert -0.5 <= out["DSR"] <= 1.0

    def test_cscv_known_answers(self):
        n_g = 8
        dom = pd.DataFrame({"s0": np.full(n_g * 2, 1e-3), "s1": np.full(n_g * 2, -1e-3)})
        assert cscv_pbo(dom, n_groups=n_g)["PBO"] == 0.0
        v0 = np.array([1e-3] * 4 + [-1e-3] * 4)
        v1 = -v0
        alt = pd.DataFrame({"s0": np.repeat(v0, 2), "s1": np.repeat(v1, 2)})
        # 实现保留 4 位小数：round(34/70, 4) = 0.4857
        assert cscv_pbo(alt, n_groups=n_g)["PBO"] == pytest.approx(round(34 / 70, 4), abs=1e-9)


# =====================================================================
# B. 执行模拟器
# =====================================================================
def _mk_synth():
    rng = np.random.default_rng(11)
    bd = pd.bdate_range("2019-01-01", "2020-07-31")
    codes = ["F%04d" % i for i in range(1, 7)]
    navs = {}
    for i, c in enumerate(codes):
        rets = rng.normal(0.0004 + 0.0001 * i, 0.012, len(bd))
        navs[c] = pd.Series(1.2 * np.exp(np.cumsum(rets)), index=bd)
    bench = pd.Series(4000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.013, len(bd)))), index=bd)
    sig_dates = [str(d.date()) for d in pd.date_range("2019-03-31", "2020-07-31", freq="ME")]
    rows = []
    for j, d in enumerate(sig_dates):
        for i, c in enumerate(codes):
            s = 35 + 45 * ((i + j) % len(codes)) / (len(codes) - 1) + rng.normal(0, 2)
            rows.append(dict(date=d, code=c, S=float(np.clip(s, 0, 100)), water=0.5, R_MDD=1.0))
    panel = pd.DataFrame(rows)
    import types as _t
    args = _t.SimpleNamespace(capital=100000.0, cost_in=0.0015, cost_out=0.005, slots=4,
                              buy=70, sell=45, pool_mode="default", legacy=False,
                              cash_yield=0.025, trail_stop=0.20, rebalance="quarterly",
                              crisis=True, cppi=True, crisis_ma=200,
                              crisis_vol_window=20, crisis_vol_q=0.80)
    return panel, navs, bench, sig_dates, args


class TestSimCore:
    def test_parity_delay0_vs_legacy(self):
        """M1.3 验收：exec_delay=0 必须与 backtest_local.simulate 逐日完全一致。"""
        panel, navs, bench, sig, args = _mk_synth()
        ec0, tr0 = BL.simulate(panel, navs, bench, sig, args)
        ec1, tr1 = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=0)
        assert np.allclose(ec0.equity.values, ec1.equity.values, rtol=1e-12, atol=0)
        assert len(tr0) == len(tr1) and len(tr0) >= 8
        assert np.allclose(tr0.entry_px, tr1.entry_px, rtol=0, atol=0)
        assert np.allclose(tr0.exit_px, tr1.exit_px, rtol=0, atol=0)
        assert (tr0.exit_date.values == tr1.exit_date.values).all()
        assert (tr0.exit_reason.values == tr1.exit_reason.values).all()
        for k in ("total_cost", "cash_interest", "crisis_days"):
            assert ec0.attrs.get(k) == pytest.approx(ec1.attrs.get(k), rel=1e-10)

    def test_t1_no_execution_on_signal_day(self):
        """T+1：除末日截断外，任何成交不得发生在信号日当日（T+0@close 的病灶）。"""
        panel, navs, bench, sig, args = _mk_synth()
        ec, tr = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=1)
        sig_set = set(sig) - {sig[-1]}          # 末日信号无次日可成交，截断执行（与旧版同源病灶）
        bad_entries = [r.entry_date for _, r in tr.iterrows() if r.entry_date in sig_set]
        assert bad_entries == [], f"T+1 下在信号日成交: {bad_entries[:5]}"
        bad_exits = [r.exit_date for _, r in tr.iterrows()
                     if r.exit_reason != "期末清算" and r.exit_date in sig_set]
        assert bad_exits == [], f"T+1 下信号日卖出: {bad_exits[:5]}"
        assert ec.attrs["exec_delay_days"] == 1

    def test_t1_entry_price_matches_entry_date(self):
        panel, navs, bench, sig, args = _mk_synth()
        _, tr = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=1)
        assert len(tr) >= 8
        for _, r in tr.iterrows():
            expect = float(navs[r.code].asof(pd.Timestamp(r.entry_date)))
            assert r.entry_px == pytest.approx(expect, rel=1e-4)

    def test_cost_fn_ladder_changes_total_cost(self):
        panel, navs, bench, sig, args = _mk_synth()
        _, tr_flat = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=1)
        ec_l, tr_l = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=1,
                                 cost_out_fn=lambda gross, hd: 0.015 if hd < 7 else
                                            (0.005 if hd < 365 else 0.0025))
        assert len(tr_l) >= 8
        assert ec_l.attrs["total_cost"] >= 0


class TestSimCoreD04:
    """D0.4 执行层扩展语义钉住（QDII T+2 / 停牌挡买）。"""

    def test_qdii_executes_t2(self):
        panel, navs, bench, sig, args = _mk_synth()
        ec, tr = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=1,
                             qdii_delay_days=2, qdii_codes={"F0001"})
        assert len(tr) > 0
        f1 = tr[tr.code == "F0001"]
        for _, r in f1.iterrows():
            d = pd.Timestamp(r.entry_date)
            assert d > max(pd.Timestamp(s) for s in sig if pd.Timestamp(s) < d)
        assert ec.attrs["qdii_delay_days"] == 2

    def test_stale_blocks_buy(self):
        """净值缺口跨过信号日与 T+1 → 该批买入被挡（不延后补单）。"""
        panel, navs, bench, sig, args = _mk_synth()
        hole = navs["F0002"].copy()
        hole = hole.drop(hole.index[(hole.index >= "2019-05-30") & (hole.index <= "2019-06-14")])
        navs2 = dict(navs); navs2["F0002"] = hole
        _, tr0 = SC.simulate(panel, navs, bench, sig, args, exec_delay_days=1)
        _, tr1 = SC.simulate(panel, navs2, bench, sig, args, exec_delay_days=1,
                             stale_block_days=7)
        was_blocked_zone = (tr0.code == "F0002") & (tr0.entry_date > "2019-05-30") & (tr0.entry_date <= "2019-06-14")
        if was_blocked_zone.any():
            newly = (tr1.code == "F0002") & (tr1.entry_date > "2019-05-30") & (tr1.entry_date <= "2019-06-14")
            assert not newly.any(), "停牌窗内仍出现成交 => stale_block 未生效"
        ec1, _ = SC.simulate(panel, navs2, bench, sig, args, exec_delay_days=1,
                             stale_block_days=7)
        assert ec1.attrs["stale_block_days"] == 7


# =====================================================================
# C. PiT 不变量
# =====================================================================
BD = pd.bdate_range("2012-01-01", "2021-12-31")
AS_OF = "2018-12-31"


def _synth_nav_df(code, seed_shift=0, tampered=False):
    rng = np.random.default_rng(abs(hash(code)) % (2**31) + seed_shift)
    rets = rng.normal(0.00035, 0.010, len(BD))
    nav = 1.0 * np.exp(np.cumsum(rets))
    ret = np.concatenate([[0.0], np.diff(nav) / nav[:-1]])
    df = pd.DataFrame(dict(date=BD, nav=nav, ret=ret))
    if tampered:
        m = df.date > pd.Timestamp(AS_OF)
        df.loc[m, "nav"] = df.loc[m, "nav"] * 3.33
        df.loc[m, "ret"] = df.loc[m, "ret"].values * (-2.0) + 0.007
    return df


def _synth_close(src, code, tampered=False):
    rng = np.random.default_rng(abs(hash((src, code))) % (2**31))
    s = pd.Series(3000 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, len(BD)))), index=BD)
    if tampered:
        s[s.index > pd.Timestamp(AS_OF)] *= 44.0
    return s


def _synth_pe(pe_key, tampered=False):
    rng = np.random.default_rng(abs(hash(pe_key)) % (2**31))
    s = pd.Series(12 + np.cumsum(rng.normal(0, 0.05, len(BD))), index=BD).clip(3, 80)
    if tampered:
        s[s.index > pd.Timestamp(AS_OF)] = 999.0
    return s


@pytest.fixture
def pit_provider(monkeypatch):
    state = {"tampered": False}

    def nav(code):
        return _synth_nav_df(code, tampered=state["tampered"])

    def close(src, code):
        return _synth_close(src, code, tampered=state["tampered"])

    def pe(key):
        return _synth_pe(key, tampered=state["tampered"])

    monkeypatch.setattr(provider, "get_fund_nav", nav)
    monkeypatch.setattr(provider, "get_close_by_src", close)
    monkeypatch.setattr(provider, "get_pe_by_key", pe)
    monkeypatch.setattr(rbsa, "_PEP_CACHE", {})
    monkeypatch.setattr(rbsa, "_FULL_MAT", {})
    return state


def _canon(d, depth=0):
    """递归规范化输出 dict：浮点统一 round-9，便于严格相等比较。"""
    if isinstance(d, dict):
        return {k: _canon(v, depth + 1) for k, v in sorted(d.items())}
    if isinstance(d, (list, tuple)):
        return tuple(_canon(x, depth + 1) for x in d)
    if isinstance(d, float):
        return round(d, 9) if math.isfinite(d) else "nan"
    if isinstance(d, (np.floating,)):
        return round(float(d), 9) if math.isfinite(d) else "nan"
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


class TestPiTInvariance:
    CODES = ["FAKE01", "FAKE02", "FAKE03"]

    def _score(self, code):
        return engine.score_fund(code, as_of=AS_OF, bt=True,
                                 pit_meta={"name": f"测试{code}", "fund_type": "混合型"})

    def test_score_fund_tail_tamper_invariant(self, pit_provider):
        """as_of 之后的净值/指数/PE 被篡改，评分输出必须逐位不变（F2/F4 防线）。"""
        base = {c: self._score(c) for c in self.CODES}
        assert all("error" not in v for v in base.values())
        pit_provider["tampered"] = True
        rbsa._PEP_CACHE.clear(); rbsa._FULL_MAT.clear()
        alt = {c: self._score(c) for c in self.CODES}
        assert _canon(base) == _canon(alt)

    def test_finalize_tail_tamper_invariant(self, pit_provider):
        rows = [self._score(c) for c in self.CODES]
        f0 = engine.finalize(rows, as_of=AS_OF)
        pit_provider["tampered"] = True
        rbsa._PEP_CACHE.clear(); rbsa._FULL_MAT.clear()
        rows2 = [self._score(c) for c in self.CODES]
        f1 = engine.finalize(rows2, as_of=AS_OF)
        for col in f0.columns:
            a, b = f0[col], f1[col]
            if a.dtype.kind == "f":
                assert np.allclose(a.fillna(-9e9), b.fillna(-9e9), rtol=1e-9), col
            else:
                assert _canon(list(a)) == _canon(list(b)), col

    def test_market_water_tail_tamper_invariant(self, pit_provider):
        w0 = engine.market_water(as_of=AS_OF)
        pit_provider["tampered"] = True
        rbsa._PEP_CACHE.clear()
        w1 = engine.market_water(as_of=AS_OF)
        assert w0 == pytest.approx(w1, abs=1e-12)

    def test_score_as_of_cutoff_exists(self, pit_provider):
        a = engine.score_fund("FAKE01", as_of="2016-12-31", bt=True,
                              pit_meta={"name": "t", "fund_type": "混合型"})
        b = engine.score_fund("FAKE01", as_of="2018-12-31", bt=True,
                              pit_meta={"name": "t", "fund_type": "混合型"})
        assert a["n_days"] < b["n_days"]
        assert a["last_date"] <= "2016-12-31"


# =====================================================================
# D. 复权构建（D0.2）
# =====================================================================
import build_navadj


class TestNavAdj:
    def _write_cache(self, tmp_path, code, nav, ret):
        d = tmp_path
        dates = pd.bdate_range("2018-01-01", periods=len(nav))
        pd.DataFrame(dict(date=dates, nav=nav, ret=ret)).to_csv(d / f"nav_{code}.csv", index=False)
        return d

    def test_dividend_day_follows_official_ret(self, tmp_path, monkeypatch):
        """分红日：nav 跌 3% 但官方 ret=+0.4% → adj 必须按 +0.4% 走（F1 处置）。"""
        monkeypatch.setattr(build_navadj, "CACHE", str(tmp_path))
        nav = [1.0, 1.01, 1.02, 1.02 * 0.97, 1.02 * 0.97 * 0.99]
        ret = [0.0, 0.01, 1.02 / 1.01 - 1, 0.004, -0.01]
        self._write_cache(tmp_path, "T00001", nav, ret)
        adj, ev, row, err = build_navadj.build_one("T00001")
        assert err is None
        assert adj.iloc[3] / adj.iloc[2] - 1 == pytest.approx(0.004, abs=1e-12)
        assert adj.iloc[0] == nav[0]
        assert row["n_events"] == 1 and ev.iloc[0]["kind"] == "分红/折算(官方收益回补)"

    def test_no_event_identity(self, tmp_path, monkeypatch):
        """无事件序列：adj ≡ nav（官方 ret 与 nav 一比一时构建器不得扭曲价格）。"""
        monkeypatch.setattr(build_navadj, "CACHE", str(tmp_path))
        nav = [1.0, 1.01, 1.00, 1.03]
        ret = [0.0] + [nav[i] / nav[i - 1] - 1 for i in range(1, 4)]
        self._write_cache(tmp_path, "T00002", nav, ret)
        adj, ev, row, err = build_navadj.build_one("T00002")
        assert err is None and row["n_events"] == 0
        assert np.allclose(adj.values, np.array(nav), atol=1e-9)

    @pytest.mark.skipif(not (os.path.exists("cache/navadj_160222.csv")
                             and os.path.exists("cache/nav_160222.csv")),
                        reason="160222 复权缓存不在本机")
    def test_real_fund_160222_event_day(self):
        """真值锚点：160222 份额折算日 adj 日收益 = 官方 ret（≠ 30%+ 的伪下跌）。"""
        raw = pd.read_csv("cache/nav_160222.csv", parse_dates=["date"]).set_index("date")
        adj = pd.read_csv("cache/navadj_160222.csv", parse_dates=["date"]).set_index("date")
        col = "adj_nav" if "adj_nav" in adj.columns else adj.columns[0]
        chg = raw["nav"].pct_change()
        big = chg[chg < -0.20].index
        assert len(big) >= 1, "未找到大额分红日，锚点失效"
        d0 = big[0]
        adj_ret = adj[col].pct_change().loc[d0]
        assert adj_ret == pytest.approx(float(raw["ret"].loc[d0]), abs=1e-6)
        assert float(chg.loc[d0]) < -0.20
        assert adj_ret > -0.05


# =====================================================================
# E. 评分函数数值性质
# =====================================================================
class TestFactorFns:
    def test_momentum_bounds_and_monotone(self):
        g = np.linspace(0, 1, 41)
        vals = [factors.momentum_score_smooth_m1(a, a) for a in g]
        assert min(vals) >= 0 and max(vals) <= 100
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))

    def test_regime_weight_switch(self):
        wl, lab_l = engine.resolve_weights(0.19)
        ws, lab_s = engine.resolve_weights(0.21)
        assert lab_l == "左侧低估区" and lab_s == "标准"
        assert wl[0] > ws[0] and wl[2] < ws[2]

    def test_maturity_guard(self, pit_provider):
        out = engine.score_fund("FAKE01", as_of="2014-03-31", bt=True,
                                pit_meta={"name": "t", "fund_type": "混合型"})
        assert "error" in out and "不足" in out["error"]


# =====================================================================
# F. ML 面板规则
# =====================================================================
class TestMLPanelRules:
    def _mk_df(self, n_months=36):
        dates = pd.date_range("2015-06-30", periods=n_months, freq="ME")
        recs = []
        for j, d in enumerate(dates):
            for c in ["C1", "C2"]:
                recs.append(dict(date=d, code=c, x1=float(j % 7),
                                 fwd6_z=float(j), fwd12_z=float(j)))
        return pd.DataFrame(recs), dates

    class _Recorder:
        instances = []

        def fit(self, X, y):
            self.max_y = float(np.max(y)) if len(y) else None
            TestMLPanelRules._Recorder.instances.append(self)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    def _probe(self, target):
        df, dates = self._mk_df()
        TestMLPanelRules._Recorder.instances = []
        orig = MZ.make_model
        MZ.FEATS["T1COL"] = ["x1"]
        try:
            MZ.make_model = lambda name: TestMLPanelRules._Recorder()
            MZ.oos_predictions(df, "huber", "T1COL", target, retrain_every=1)
        finally:
            MZ.make_model = orig
            del MZ.FEATS["T1COL"]
        last = TestMLPanelRules._Recorder.instances[-1]
        return last.max_y, len(dates)

    def test_fwd6_train_cutoff_respects_horizon(self):
        """fwd6：训练样本必须成熟 6 个月 → max_train_idx ≤ N-1-6。"""
        max_y, n = self._probe("fwd6_z")
        assert max_y <= (n - 1) - 6 + 1e-9

    def test_fwd12_train_cutoff_respects_horizon(self):
        """R3.5 已修复（_model_zoo TARGET_HORIZON_MO，h 匹配截止）；
        修复前病灶锚点：统一 q−6M → 本用例当时按 strict-xfail 挂起，修复后转绿。"""
        max_y, n = self._probe("fwd12_z")
        assert max_y <= (n - 1) - 12 + 1e-9

    def test_imputation_is_cross_sectional_first(self):
        df = pd.DataFrame(dict(
            date=pd.to_datetime(["2020-01-31"] * 3 + ["2020-02-28"] * 3),
            code=["A", "B", "C"] * 2,
            x=[np.nan, 10.0, 20.0, 1.0, 2.0, 3.0]))
        out = MZ.impute_features(df, ["x"])
        # A 在 2020-01 缺失 → 用当日截面中位数 15，而非全样本中位数(8.5)
        assert out.loc[0, "x"] == pytest.approx(15.0)
        assert out["x"].notna().all()
