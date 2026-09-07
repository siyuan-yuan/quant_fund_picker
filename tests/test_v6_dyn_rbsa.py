# -*- coding: utf-8 -*-
"""V6 / T1.1 不变量测试 —— dyn_rbsa

预登记依据：计划书 §3 T1（PIT 硬约束、三臂定义）与 §5-R1（复制性错误对策：
"KF 在 σ_η→0 时应收敛到静态解"是一条可执行自检）。

这些测试在**没有任何外部数据源**的条件下即可运行（全部用合成数据），
因此不受沙箱网络限制影响，也不产出任何裁决数字。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dyn_rbsa as dr


# ---------------------------------------------------------------- 合成数据
def _panel(n=1000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    f = pd.DataFrame(rng.normal(0, 0.012, (n, k)), index=idx,
                     columns=[f"idx{i}" for i in range(k)])
    return idx, f


def _fund_from_beta(f, beta_path, noise=0.002, seed=1):
    rng = np.random.default_rng(seed)
    r = (f.values * beta_path).sum(axis=1) + rng.normal(0, noise, len(f))
    return pd.Series(r, index=f.index)


# ---------------------------------------------------------------- 单纯形投影
def test_project_simplex_basic():
    v = np.array([0.5, 0.2, -0.4, 1.1])
    p = dr.project_simplex(v)
    assert p.min() >= -1e-12
    assert abs(p.sum() - 1.0) < 1e-10


def test_project_simplex_idempotent():
    p = dr.project_simplex(np.array([0.2, 0.3, 0.5]))
    assert np.allclose(p, dr.project_simplex(p), atol=1e-12)
    # 已在单纯形内的点应保持不动
    assert np.allclose(p, [0.2, 0.3, 0.5], atol=1e-10)


# ---------------------------------------------------------------- §5-R1 自检
def test_kf_converges_to_static_when_q_zero():
    """q → 0 ⇒ KF 退化为常数 β，且应逼近全样本约束 LS 解。

    这是计划书 §5-R1 登记的双实现对账：KF（递归）与 constrained_ls（批量投影
    梯度）是两套独立实现，二者在 Q=0 极限下必须给出同一个答案。
    """
    idx, f = _panel(n=1200, k=3, seed=2)
    true_b = np.array([0.6, 0.3, 0.1])
    y = _fund_from_beta(f, np.tile(true_b, (len(f), 1)), noise=0.001, seed=3)

    B = dr.kalman_beta_path(y, f, q=0.0, p0=1e-4)
    static = dr.constrained_ls(y.to_numpy(), f.to_numpy())

    assert np.allclose(B.iloc[-1].to_numpy(), static, atol=0.05), \
        f"KF(Q=0) 终端 β={B.iloc[-1].to_numpy()} 与约束 LS={static} 不一致"
    # 且路径本身应几乎不动
    assert float(B.iloc[200:].std().max()) < 0.02


def test_kf_recovers_constant_beta():
    idx, f = _panel(n=1500, k=3, seed=4)
    true_b = np.array([0.5, 0.5, 0.0])
    y = _fund_from_beta(f, np.tile(true_b, (len(f), 1)), noise=0.0015, seed=5)
    B = dr.kalman_beta_path(y, f, q=0.003)
    assert np.allclose(B.iloc[-1].to_numpy(), true_b, atol=0.12)


def test_kf_tracks_style_drift_better_than_static():
    """核心动机检验：基金风格从 idx0 切到 idx2 时，KF 应跟上，静态解不应。

    这正是 engine.py 现行口径的问题在合成数据上的最小复现。
    """
    idx, f = _panel(n=1600, k=3, seed=6)
    n = len(f)
    bp = np.zeros((n, 3))
    bp[: n // 2] = [1.0, 0.0, 0.0]
    bp[n // 2:] = [0.0, 0.0, 1.0]
    y = _fund_from_beta(f, bp, noise=0.001, seed=7)

    B = dr.kalman_beta_path(y, f, q=0.1)
    # 后半段应认出 idx2 占主导
    assert B.iloc[-1]["idx2"] > 0.7
    # 前半段（burn-in 之后）应认出 idx0
    assert B.iloc[n // 2 - 100]["idx0"] > 0.7

    # 静态全样本解无法同时满足两端 → 明显偏离
    static = dr.constrained_ls(y.to_numpy(), f.to_numpy())
    assert static[2] < B.iloc[-1]["idx2"]


def test_kf_reduces_residual_style_r2_under_drift():
    """H1-1 检验量在合成漂移数据上必须站得住：
    动态 β 的 AR 残余风格 R² 应显著低于静态 β 的 active return。"""
    idx, f = _panel(n=1600, k=3, seed=8)
    n = len(f)
    bp = np.zeros((n, 3))
    bp[: n // 2] = [1.0, 0.0, 0.0]
    bp[n // 2:] = [0.0, 0.0, 1.0]
    y = _fund_from_beta(f, bp, noise=0.001, seed=9)

    B_kf = dr.kalman_beta_path(y, f, q=0.1)
    ar_kf = dr.abnormal_return(y, B_kf, f)

    static = dr.constrained_ls(y.to_numpy(), f.to_numpy())
    B_st = pd.DataFrame(np.tile(static, (n, 1)), index=f.index, columns=f.columns)
    ar_st = dr.abnormal_return(y, B_st, f)

    r2_kf = dr.residual_style_r2(ar_kf, f)
    r2_st = dr.residual_style_r2(ar_st, f)
    assert r2_kf < r2_st, f"KF 残余风格 R²={r2_kf:.3f} 未低于静态 {r2_st:.3f}"


# ---------------------------------------------------------------- PIT 约束
def test_beta_path_is_pit_safe_flagged():
    idx, f = _panel(n=400, k=3, seed=10)
    y = _fund_from_beta(f, np.tile([0.4, 0.4, 0.2], (len(f), 1)), seed=11)
    assert dr.kalman_beta_path(y, f).attrs["pit_safe"] is True
    assert dr.rolling_beta_path(y, f).attrs["pit_safe"] is True


def test_rts_smoother_is_blocked():
    with pytest.raises(NotImplementedError):
        dr.rts_smooth()


def test_kf_is_causal():
    """因果性硬检验：篡改 t 之后的数据，不得改变 t 时刻及之前的 β。

    这是 PIT 硬约束的可执行版本 —— 任何未来信息泄漏都会让本测试失败。
    """
    idx, f = _panel(n=800, k=3, seed=12)
    y = _fund_from_beta(f, np.tile([0.5, 0.3, 0.2], (len(f), 1)), seed=13)
    cut = 500

    B1 = dr.kalman_beta_path(y, f, q=0.03)
    y2 = y.copy()
    y2.iloc[cut:] = y2.iloc[cut:] * 5.0 + 0.03      # 大幅篡改未来
    B2 = dr.kalman_beta_path(y2, f, q=0.03)

    pd.testing.assert_frame_equal(B1.iloc[:cut], B2.iloc[:cut],
                                  check_exact=False, atol=1e-12)


def test_rolling_path_is_causal():
    idx, f = _panel(n=800, k=3, seed=14)
    y = _fund_from_beta(f, np.tile([0.5, 0.3, 0.2], (len(f), 1)), seed=15)
    cut = 500
    B1 = dr.rolling_beta_path(y, f)
    y2 = y.copy()
    y2.iloc[cut:] = y2.iloc[cut:] * 5.0 + 0.03
    B2 = dr.rolling_beta_path(y2, f)
    pd.testing.assert_frame_equal(B1.iloc[:cut], B2.iloc[:cut],
                                  check_exact=False, atol=1e-12)


# ---------------------------------------------------------------- 约束/退化
def test_beta_path_respects_simplex():
    idx, f = _panel(n=600, k=4, seed=16)
    y = _fund_from_beta(f, np.tile([0.25] * 4, (len(f), 1)), seed=17)
    B = dr.kalman_beta_path(y, f, q=0.1).dropna()
    assert (B.to_numpy() >= -1e-9).all()
    assert np.allclose(B.sum(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_empty_and_short_inputs_do_not_crash():
    idx, f = _panel(n=30, k=3, seed=18)
    y = pd.Series(np.zeros(30), index=idx)
    assert len(dr.kalman_beta_path(y, f)) == 30
    assert dr.rolling_beta_path(y, f).isna().all().all()
    e = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert dr.kalman_beta_path(e, f).attrs["n_obs"] == 0


# ---------------------------------------------------------------- A3 断点
def test_break_test_detects_true_break():
    idx, f = _panel(n=1200, k=3, seed=19)
    n = len(f)
    bp = np.zeros((n, 3))
    bp[: n // 2] = [1.0, 0.0, 0.0]
    bp[n // 2:] = [0.0, 0.0, 1.0]
    y = _fund_from_beta(f, bp, noise=0.001, seed=20)
    res = dr.break_test(y, f)
    assert res["sup_F"] > 20
    assert abs(res["break_idx"] - n // 2) < 80


def test_break_test_null_is_quiet():
    """无漂移时 sup-F 应远小于有漂移的情形（自助 p 不应显著）。"""
    idx, f = _panel(n=1000, k=3, seed=21)
    y = _fund_from_beta(f, np.tile([0.4, 0.4, 0.2], (len(f), 1)),
                        noise=0.002, seed=22)
    res = dr.break_test(y, f, n_boot=60, seed=0)
    assert res["p_boot"] > 0.05


def test_abnormal_return_matches_manual():
    idx, f = _panel(n=300, k=3, seed=23)
    y = _fund_from_beta(f, np.tile([0.5, 0.5, 0.0], (len(f), 1)), seed=24)
    B = pd.DataFrame(np.tile([0.5, 0.5, 0.0], (len(f), 1)),
                     index=f.index, columns=f.columns)
    ar = dr.abnormal_return(y, B, f)
    manual = y - (f * [0.5, 0.5, 0.0]).sum(axis=1)
    pd.testing.assert_series_equal(ar, manual.dropna(), check_names=False)


# ---------------------------------------------------------------- q 标定
@pytest.mark.parametrize("q,lo,hi", [(0.01, 60, 200), (0.03, 20, 70), (0.1, 5, 30)])
def test_q_tracking_speed_calibration(q, lo, hi):
    """预登记三档 q 的物理含义：断点后 β 收敛到 0.7 所需交易日数应落在标定区间。

    这条测试把 §3 T1 的 "σ_η 三档" 锚定成**可读、可复核的跟踪速度**，
    防止后人把 q 当成一个可以随便搜的超参（那会直接落入 §7 禁区）。
    """
    idx, f = _panel(n=1600, k=3, seed=6)
    n = len(f)
    bp = np.zeros((n, 3))
    bp[: n // 2] = [1.0, 0.0, 0.0]
    bp[n // 2:] = [0.0, 0.0, 1.0]
    y = _fund_from_beta(f, bp, noise=0.001, seed=7)

    B = dr.kalman_beta_path(y, f, q=q)
    post = B["idx2"].to_numpy()[n // 2:]
    hit = np.argmax(post > 0.7) if (post > 0.7).any() else -1
    assert hit > 0, f"q={q} 未在样本内收敛到 0.7"
    assert lo <= hit <= hi, f"q={q} 收敛用了 {hit} 日，超出标定区间 [{lo},{hi}]"


def test_q_is_scale_invariant():
    """信噪比参数化的核心收益：同一 q 在不同残差波动率下跟踪速度应基本一致。

    这正是初版绝对尺度 sigma_eta 参数化做不到的（修订理由见 dyn_rbsa 文档串）。
    """
    idx, f = _panel(n=1600, k=3, seed=6)
    n = len(f)
    bp = np.zeros((n, 3))
    bp[: n // 2] = [1.0, 0.0, 0.0]
    bp[n // 2:] = [0.0, 0.0, 1.0]

    hits = []
    for noise in (0.001, 0.004):
        y = _fund_from_beta(f, bp, noise=noise, seed=7)
        B = dr.kalman_beta_path(y, f, q=0.03)
        post = B["idx2"].to_numpy()[n // 2:]
        hits.append(int(np.argmax(post > 0.7)))
    assert abs(hits[0] - hits[1]) <= 10, f"跟踪速度随噪声漂移过大: {hits}"


def test_high_noise_does_not_break_tracking():
    """负反馈缺陷的回归测试：真实 β 已切换时，终端暴露必须认出新风格。

    初版（R 做 EWMA 自适应）在本场景给出 idx2=0.275 —— 断点处大新息抬高 R、
    压制卡尔曼增益，漂移越大跟得越慢。此测试锁死该缺陷不复现。
    """
    idx, f = _panel(n=1600, k=3, seed=6)
    n = len(f)
    bp = np.zeros((n, 3))
    bp[: n // 2] = [1.0, 0.0, 0.0]
    bp[n // 2:] = [0.0, 0.0, 1.0]
    y = _fund_from_beta(f, bp, noise=0.001, seed=7)
    assert dr.kalman_beta_path(y, f, q=0.03).iloc[-1]["idx2"] > 0.9
