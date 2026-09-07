# -*- coding: utf-8 -*-
"""V6 / T2 不变量测试 —— v6_panel（三套 target + 前视护栏）

预登记依据：计划书 §3 T2。全部合成数据，零外部依赖，不产出裁决数字。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v6_panel as vp


def _series(n=800, mu=0.0005, sd=0.01, seed=0, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series((1 + rng.normal(mu, sd, n)).cumprod(), index=idx)


def _idx_close(n=800, k=3, seed=1, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {f"idx{i}": (1 + rng.normal(0.0004, 0.011, n)).cumprod() for i in range(k)},
        index=idx)


# ---------------------------------------------------------------- 恒等式
def test_skill_identity_holds():
    """raw = style + skill 必须逐行成立（这是三套 target 的定义基础）。"""
    ic = _idx_close()
    dates = [pd.Timestamp("2019-01-02"), pd.Timestamp("2019-06-03")]
    adj = {"F1": _series(seed=2)}
    b = pd.Series([0.5, 0.3, 0.2], index=["idx0", "idx1", "idx2"])
    beta_at = {(d, "F1"): b for d in dates}

    t = vp.build_targets(dates, ["F1"], adj, beta_at, ic, horizons=(6,))
    ok = t[["raw6", "style6", "skill6"]].dropna()
    assert len(ok) > 0
    assert np.allclose(ok["raw6"], ok["style6"] + ok["skill6"], atol=1e-12)


def test_style_equals_weighted_index_return():
    ic = _idx_close()
    d = pd.Timestamp("2019-01-02")
    b = pd.Series([0.6, 0.4, 0.0], index=["idx0", "idx1", "idx2"])
    t = vp.build_targets([d], ["F1"], {"F1": _series(seed=3)},
                         {(d, "F1"): b}, ic, horizons=(6,))
    fi = vp.index_forward_return(ic, d, 6)
    assert abs(float(t["style6"].iloc[0]) - float((b * fi).sum())) < 1e-12


def test_pure_style_fund_has_zero_skill():
    """一只完全复制风格组合的基金，skill 应约等于 0。

    这是 T2 的核心动机检验：现行 raw 标签会把它误判为"赛道涨=好基金"，
    skill 标签必须能识别出它其实没有超额。
    """
    ic = _idx_close(n=900, seed=4)
    rets = ic.pct_change().fillna(0.0)
    b = pd.Series([0.5, 0.5, 0.0], index=ic.columns)
    fund_adj = (1 + (rets * b).sum(axis=1)).cumprod()

    d = pd.Timestamp("2019-01-02")
    t = vp.build_targets([d], ["F1"], {"F1": fund_adj},
                         {(d, "F1"): b}, ic, horizons=(6,))
    # 日频复利与区间加权的差异属二阶项，量级应远小于 raw 本身
    assert abs(float(t["skill6"].iloc[0])) < 0.02


# ---------------------------------------------------------------- β 缺失
def test_missing_beta_yields_nan_not_fallback():
    """β 缺失时 style/skill 必须为 NaN —— 回填等于编造暴露。"""
    ic = _idx_close()
    d = pd.Timestamp("2019-01-02")
    t = vp.build_targets([d], ["F1"], {"F1": _series(seed=5)}, {}, ic,
                         horizons=(6,))
    assert pd.isna(t["style6"].iloc[0]) and pd.isna(t["skill6"].iloc[0])
    assert not pd.isna(t["raw6"].iloc[0])          # raw 不受影响


def test_low_index_coverage_yields_nan():
    """指数覆盖不足 50% 时不出数（与 F_value 估值盲区同精神）。"""
    ic = _idx_close()
    ic.loc[:, "idx1"] = np.nan
    ic.loc[:, "idx2"] = np.nan
    d = pd.Timestamp("2019-01-02")
    b = pd.Series([0.2, 0.4, 0.4], index=ic.columns)      # 有效权重仅 0.2
    t = vp.build_targets([d], ["F1"], {"F1": _series(seed=6)},
                         {(d, "F1"): b}, ic, horizons=(6,))
    assert pd.isna(t["style6"].iloc[0])


# ---------------------------------------------------------------- 截面变换
def test_cross_sectional_transforms():
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 5 + ["2020-02-29"] * 5,
        "skill6": [0.1, -0.2, 0.3, -0.4, 0.05, 1.0, -1.0, 0.5, -0.5, 0.0],
    })
    out = vp.cross_sectional_transforms(df, ["skill6"])
    g = out.groupby("date")["skill6_z"]
    assert np.allclose(g.mean().to_numpy(), 0.0, atol=1e-8)
    assert out["skill6_rk"].between(0, 1).all()
    assert set(out["skill6_pos"].dropna().unique()) <= {0.0, 1.0}
    # pos 标签语义：跑赢自己的动态基准
    assert out.loc[0, "skill6_pos"] == 1.0
    assert out.loc[1, "skill6_pos"] == 0.0


def test_transforms_are_per_cross_section():
    """截面变换不得混用跨期信息：两期同值应各自标准化。"""
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 3 + ["2020-02-29"] * 3,
        "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    out = vp.cross_sectional_transforms(df, ["x"])
    a = out[out.date == "2020-01-31"]["x_z"].to_numpy()
    b = out[out.date == "2020-02-29"]["x_z"].to_numpy()
    assert np.allclose(a, b, atol=1e-8)


# ---------------------------------------------------------------- 成熟期
@pytest.mark.parametrize("h", [3, 6, 12])
def test_mature_mask_per_horizon(h):
    """每个 horizon 各自计算成熟期 —— 修复 fwd12 硬编码 6M 的历史错误。"""
    dates = pd.Series(pd.date_range("2020-01-31", periods=24, freq="ME"))
    as_of = pd.Timestamp("2021-12-31")
    m = vp.mature_mask(dates, as_of, h)
    cutoff = as_of - pd.DateOffset(months=h)
    assert (dates[m] <= cutoff).all()
    assert (dates[~m] > cutoff).all()


def test_mature_mask_h12_stricter_than_h6():
    dates = pd.Series(pd.date_range("2020-01-31", periods=36, freq="ME"))
    as_of = pd.Timestamp("2022-12-31")
    assert vp.mature_mask(dates, as_of, 12).sum() < \
           vp.mature_mask(dates, as_of, 6).sum()


# ---------------------------------------------------------------- 前视断言
def test_assert_no_lookahead_passes_on_clean_panel():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"date": dates, "y": np.nan})
    df.loc[df.date <= dates[-1] - pd.DateOffset(months=6), "y"] = 1.0
    info = vp.assert_no_lookahead(df, 6, label_col="y")
    assert info["matures_at"] <= info["last_decision"] + pd.Timedelta(days=1)


def test_assert_no_lookahead_catches_violation():
    """标签成熟日超出面板末日 → 必须抛错（C8 护栏的新标签族版本）。"""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    df = pd.DataFrame({"date": dates, "y": 1.0})       # 末月也有标签 = 前视
    with pytest.raises(AssertionError, match="标签前视"):
        vp.assert_no_lookahead(df, 6, label_col="y")


def test_build_targets_labels_are_naturally_immature_at_tail():
    """数据尾部不足 h 个月时，标签应自然为 NaN（而不是被截断/外推）。"""
    ic = _idx_close(n=600, start="2019-01-01")
    adj = {"F1": _series(n=600, seed=7, start="2019-01-01")}
    last = ic.index[-1]
    d_tail = last - pd.DateOffset(months=2)            # 距末日仅 2 个月
    b = pd.Series([0.4, 0.3, 0.3], index=ic.columns)
    t = vp.build_targets([d_tail], ["F1"], adj, {(d_tail, "F1"): b}, ic,
                         horizons=(6,))
    assert pd.isna(t["raw6"].iloc[0])


# ---------------------------------------------------------------- 冻结 β 诊断
def test_style_drift_diagnostic_reports_drift():
    """冻结 β 的失真必须可量化：漂移基金的 L1 应显著大于稳定基金。"""
    idx = pd.bdate_range("2019-01-01", periods=400)
    cols = ["idx0", "idx1", "idx2"]
    stable = pd.DataFrame(np.tile([0.5, 0.3, 0.2], (400, 1)), index=idx,
                          columns=cols)
    drift = stable.copy()
    drift.iloc[200:] = [0.0, 0.0, 1.0]

    d = idx[100]
    b0 = pd.Series([0.5, 0.3, 0.2], index=cols)
    out = vp.style_drift_diagnostic({(d, "S"): b0, (d, "D"): b0},
                                    {"S": stable, "D": drift}, 12)
    l1 = out.set_index("code")["l1_drift"]
    assert l1["S"] < 1e-8
    assert l1["D"] > 0.3


# ---------------------------------------------------------------- 组合
def test_blend_scores_endpoints_and_midpoint():
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([0.0, 0.0, 0.0])
    assert np.allclose(vp.blend_scores(a, b, 1.0), a)
    assert np.allclose(vp.blend_scores(a, b, 0.0), b)
    assert np.allclose(vp.blend_scores(a, b, 0.5), a * 0.5)


def test_blend_scores_accepts_row_wise_lambda():
    a = pd.Series([1.0, 1.0])
    b = pd.Series([0.0, 0.0])
    assert np.allclose(vp.blend_scores(a, b, pd.Series([0.2, 0.8])), [0.2, 0.8])


def test_blend_lambda_is_clipped():
    a, b = pd.Series([1.0]), pd.Series([0.0])
    assert float(vp.blend_scores(a, b, 5.0).iloc[0]) == 1.0
    assert float(vp.blend_scores(a, b, -3.0).iloc[0]) == 0.0


# ---------------------------------------------------------------- 与旧口径对账
def test_forward_return_matches_legacy_impl():
    """与 `_build_ml_panel.fwd_ret` 同口径（asof 语义），保证可逐位对账。"""
    adj = _series(seed=8)
    d = pd.Timestamp("2019-03-01")

    def legacy(adj, d, months):
        q = pd.Timestamp(d)
        t1 = q + pd.DateOffset(months=months)
        if t1 > adj.index[-1] or q < adj.index[0]:
            return np.nan
        v0, v1 = adj.asof(q), adj.asof(t1)
        if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
            return np.nan
        return float(v1 / v0 - 1)

    for h in (3, 6, 12):
        x, y = vp.forward_return(adj, d, h), legacy(adj, d, h)
        assert (pd.isna(x) and pd.isna(y)) or abs(x - y) < 1e-15


# ---------------------------------------------------------------- T1×T2 集成
def test_skill_label_separates_sector_beta_from_true_alpha():
    """V6 的核心命题，端到端最小复现（T1 动态 β + T2 skill 标签）。

    场景：后半段科技赛道大涨。
      基金 A —— 平庸，但 80% 压在 tech（纯 beta，无 alpha）
      基金 B —— 有真实 +10%/yr alpha，但完全不碰 tech

    现行 raw 标签会把 A 判为正样本（它涨得多），模型于是学"哪种风格会涨"；
    skill 标签必须反过来选出 B —— 这正是计划书 §3 T2 要拆开的两种能力。
    """
    import dyn_rbsa as dr

    rng = np.random.default_rng(42)
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = ["big", "growth", "tech"]
    fr = pd.DataFrame(rng.normal(0.0003, 0.011, (n, 3)), index=idx, columns=cols)
    fr.iloc[750:, fr.columns.get_loc("tech")] += 0.0012      # 赛道行情
    ic = (1 + fr).cumprod()

    bA = np.tile([0.1, 0.1, 0.8], (n, 1))
    bB = np.tile([0.5, 0.5, 0.0], (n, 1))
    rA = pd.Series((fr.values * bA).sum(1) + rng.normal(0, 0.002, n), index=idx)
    rB = pd.Series((fr.values * bB).sum(1) + 0.0004 + rng.normal(0, 0.002, n),
                   index=idx)
    adj = {"A": (1 + rA).cumprod(), "B": (1 + rB).cumprod()}

    beta = {c: dr.kalman_beta_path(r, fr, q=0.03) for c, r in [("A", rA), ("B", rB)]}
    d = idx[900]
    t = vp.build_targets([d], ["A", "B"], adj,
                         {(d, c): beta[c].loc[d] for c in ("A", "B")},
                         ic, horizons=(6,)).set_index("code")

    assert t["raw6"].idxmax() == "A", "场景构造失败：raw 标签本应偏向赛道基金"
    assert t["skill6"].idxmax() == "B", "skill 标签未能识别真实 alpha"
    assert t.loc["A", "skill6"] < 0 < t.loc["B", "skill6"]
