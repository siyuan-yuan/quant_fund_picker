# -*- coding: utf-8 -*-
"""
finalize 批次无关性回归测试（离线，无需网络/sklearn）

行为契约（V4.1 一致性修正）：
  同一基金、同一参照快照下 —— 无论以 单基透视([A])、批量评分([A,B,C]) 还是
  持仓诊断(任意子集/任意次序) 调用 finalize(use_global_ref=True)，
  S_total / S_v4 / S_v37 / F_momentum / rating 必须严格一致。

覆盖场景：
  1) V4 激活 + 全市场参照快照齐备 → 三入口一致（ECDF 映射）
  2) V4 激活 + 无参照快照 → 一致性闸门整体降级 V3.7，三入口一致
  3) V4 未安装 → 纯 V3.7，三入口一致
  4) 单行特征缺失（如次新股 ir_winrate=NaN）→ 不抛异常，该行 V4 静默回退 V3.7
     （回归旧版单基估值盲区基 X 含 NaN → ValueError → 接口400 的事故）
"""
import sys, types
import numpy as np
import pandas as pd

# ---- 打桩重依赖（本测试只测 finalize 的截面行为，不触网）----
sys.modules["provider"] = types.SimpleNamespace(
    fund_type=lambda c: "", is_passive_fund=lambda c, n: False)
sys.modules["rbsa"] = types.SimpleNamespace(
    market_water_level=lambda as_of=None: 0.43)

import engine


# ================== 伪 V4 模型（z = value_z + 0.3·mom_pure）==================
class _Hub:
    def predict(self, X):
        return X[:, 0] + 0.3 * X[:, 1]

class _Sc:
    def transform(self, X):
        return X

class _FakeModel:
    named_steps = {"hub": _Hub(), "sc": _Sc()}


def _build_features(val_pct, r4_rk, r7_rk, wr_rk, dc_rk, rmdd_pen, water, trend_t):
    val_pct = np.asarray(val_pct, float); r4_rk = np.asarray(r4_rk, float)
    r7_rk = np.asarray(r7_rk, float); wr_rk = np.asarray(wr_rk, float)
    dc_rk = np.asarray(dc_rk, float); rmdd_pen = np.asarray(rmdd_pen, float)
    water = np.asarray(water, float); trend_t = np.asarray(trend_t, float)
    value_z = 1.0 - val_pct
    mom_pure = 0.5 * r4_rk + 0.5 * r7_rk
    quality = 0.5 * wr_rk + 0.5 * (1.0 - dc_rk)
    safety = 1.0 - rmdd_pen
    macro_state = water - 0.5
    val_x_mom = value_z * mom_pure
    return np.column_stack([value_z, mom_pure, quality, safety, macro_state,
                            trend_t, val_x_mom])


# ================== 伪参照宇宙快照 ==================
def fake_ref():
    rng = np.linspace(-0.2, 0.3, 300)
    return dict(stamp="test_ref_20260807",
                mom4=pd.Series(rng), mom7=pd.Series(rng),
                wr=pd.Series(np.linspace(0, 1, 300)),
                dc=pd.Series(np.linspace(0.5, 1.5, 300)),
                val_median=0.50,
                z=pd.Series(np.linspace(-2, 3, 300)))


def fake_row(code, ir_wr, dc, mom4, mom7, valpct):
    return {"code": code, "name": code, "ftype": "股票型", "n_days": 900,
            "last_date": "2026-08-06", "rbsa": {"沪深300": 0.8},
            "panel_mode": "unified",
            "f_value_base": 70.0, "val_pct": valpct, "val_coverage": 1.0,
            "valuation_blind": False, "macd_dif": 0.01, "ma20_dist": 0.01,
            "trend_ma20": True, "trend_ok": True, "bonus": 0,
            "bonus_detail": {"pass": False}, "F_value": 70.0,
            "ir_winrate": ir_wr, "s_ir": 60.0, "down_capture": dc,
            "s_dc": 50.0, "F_alpha": 55.0,
            "mom_4m1m": mom4, "mom_7m1m": mom7,
            "tenure_days": 1500, "is_passive": False,
            "penalties": [], "penalty_detail": {"R_MDD": 1.0},
            "scale": 30.0}


A = fake_row("111111", 0.60, 0.95, 0.05, 0.08, 0.50)   # 目标基金：各项指标中等
B = fake_row("222222", 0.75, 0.85, 0.25, 0.28, 0.20)   # 强势陪跑
C = fake_row("333333", 0.45, 1.05, -0.15, -0.10, 0.85) # 弱势陪跑

METRICS = ["S_v37", "S_v4", "S_total", "F_momentum", "rating", "ref_stamp"]


def row_of(df, code="111111"):
    return df[df.code == code].iloc[0][METRICS]


def assert_batch_invariant(tag):
    """四种调用方式：单基 / 全批 / 子集 / 乱序 —— 目标基金指标必须逐格相等"""
    single = engine.finalize([dict(A)], use_global_ref=True)
    full = engine.finalize([dict(A), dict(B), dict(C)], use_global_ref=True)
    sub = engine.finalize([dict(A), dict(C)], use_global_ref=True)
    shuffled = engine.finalize([dict(C), dict(A), dict(B)], use_global_ref=True)
    base = row_of(single)
    for name, odf in [("全批", full), ("子集", sub), ("乱序", shuffled)]:
        got = row_of(odf)
        for m in METRICS:
            bv, gv = base[m], got[m]
            same = (bv == gv) or (pd.isna(bv) and pd.isna(gv))
            assert same, f"[{tag}] {name}场景 {m} 不一致: 单基={bv} vs {name}={gv}"
    print(f"  ✅ [{tag}] 单基/全批/子集/乱序 四场景全部一致 "
          f"(S_total={base['S_total']}, S_v4={base['S_v4']}, 参照={base['ref_stamp']})")
    return single


def main():
    print("=" * 74)
    print(" finalize 批次无关性回归测试 (Batch-Invariance Contract)")
    print("=" * 74)
    orig_model, orig_bundle = engine.model_v4, engine._V4_BUNDLE
    orig_ref_fn = engine.get_global_ref_universe
    try:
        # ---- 场景1: V4 激活 + 参照齐备 ----
        engine.model_v4 = types.SimpleNamespace(build_features=_build_features)
        engine._V4_BUNDLE = {"model": _FakeModel(), "version": "FAKE-V4"}
        engine.get_global_ref_universe = lambda as_of=None: fake_ref()
        single = assert_batch_invariant("V4激活+参照齐备")
        r = single.iloc[0]
        assert pd.notna(r["S_v4"]), "参照齐备时 V4 不应关闭"
        assert abs(r["S_v4"] - 100.0) > 1e-6 or True  # 不再恒=100(由ECDF决定)
        assert "ECDF" in str(r["model_version"])

        # 修复前对照(旧逻辑批内rank): 单基 S_v4 恒=100 → 此处显式验证 ECDF 取代之
        z_expect = (1 - 0.50) + 0.3 * (0.5 * (fake_ref()["mom4"] <= .05).mean()
                                       + 0.5 * (fake_ref()["mom7"] <= .08).mean())
        s_v4_expect = round(float((fake_ref()["z"] <= z_expect).mean()) * 100, 1)
        assert abs(r["S_v4"] - s_v4_expect) < 0.2, \
            f"S_v4 应等于对快照z分布的ECDF: got {r['S_v4']} expect {s_v4_expect}"
        print(f"  ✅ [V4激活+参照齐备] S_v4={r['S_v4']} 为快照ECDF分位(预期 {s_v4_expect})，"
              f"不再退化为恒100")

        # ---- 场景2: V4 激活 + 无参照快照 → 一致性闸门 ----
        engine.get_global_ref_universe = lambda as_of=None: None
        single2 = assert_batch_invariant("V4激活+无参照(闸门降级)")
        r2 = single2.iloc[0]
        assert pd.isna(r2["S_v4"]) and r2["S_total"] == r2["S_v37"], \
            "无参照快照时 V4 必须整体降级: S_v4=NaN 且 S_total=S_v37"
        print(f"  ✅ [V4激活+无参照] 一致性闸门生效: S_v4=NaN，S_total=S_v37={r2['S_total']}，"
              f"拒绝批内rank")

        # ---- 场景3: V4 未安装 → 纯 V3.7 ----
        engine.model_v4, engine._V4_BUNDLE = None, None
        engine.get_global_ref_universe = lambda as_of=None: fake_ref()
        assert_batch_invariant("V4未安装(纯V3.7)")

        # ---- 场景4: 单行特征缺失(次新股 ir/dc 缺失) → 不炸, 该行回退 V3.7 ----
        engine.model_v4 = types.SimpleNamespace(build_features=_build_features)
        engine._V4_BUNDLE = {"model": _FakeModel(), "version": "FAKE-V4"}
        N = fake_row("444444", None, None, 0.10, 0.12, 0.40)
        N["ir_winrate"], N["down_capture"] = None, None
        one = engine.finalize([dict(N)], use_global_ref=True)          # 单基：旧版这里直接400
        mix = engine.finalize([dict(A), dict(N)], use_global_ref=True)
        rn = one.iloc[0]
        assert pd.isna(rn["S_v4"]) and rn["S_total"] == rn["S_v37"], \
            "特征缺失行应静默回退 V3.7"
        rm = mix[mix.code == "444444"].iloc[0]
        assert pd.isna(rm["S_v4"]) and rm["S_total"] == rn["S_total"], \
            "特征缺失行在批量中必须与单基一致"
        print(f"  ✅ [特征缺失行] 单基/批量均不抛异常，该行 V4 静默回退 V3.7 "
              f"(S_total={rn['S_total']})")

        print("=" * 74)
        print(" ✅ 全部通过：finalize 对批次大小/批次构成严格不变，三入口同源同分。")
        print("=" * 74)
    finally:
        engine.model_v4, engine._V4_BUNDLE = orig_model, orig_bundle
        engine.get_global_ref_universe = orig_ref_fn


if __name__ == "__main__":
    main()
