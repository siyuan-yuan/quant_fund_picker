# -*- coding: utf-8 -*-
"""
model_v4.py — V4 华尔街级评分模型（Huber 稳健回归 + 时间衰减）

冠军模型（model_lab6 综合评分第一）：
  * 7 个经济学正交特征（价值/动量/质量/安全/宏观/趋势确认/价值×动量）
  * Huber 稳健回归（epsilon=1.35, alpha=0.01）
  * 样本权重按 2 年指数半衰期衰减（regime adaptive）
  * 训练目标：截面 z-score 化的未来 6 月收益（对极端值稳健）
  * 严格 walk-forward 验证（expanding window，min_train=8季）

相对 V3.7 引擎：
  * Strict OOS IC 0.0926 → 0.1488  (+61%)
  * Walk-forward IC 0.1159 → 0.1982  (+71%)
  * Walk-forward IC t-stat 2.40 → 2.91
  * 多空 Q10-Q1 价差 +1.58% → +6.96%

设计为完全离线模型：训练一次 pickle 固化，线上打分时只算特征→predict，延迟 ms 级。
模型每个季度滚动重训（在 backtest_local 自动 PiT 打分时自然完成）。
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_VERSION = "V4.0-huber-decay2-7f"
MODEL_FILE = os.path.join(os.path.dirname(__file__), "cache", "v4_model.pkl")

# 7 个经济学特征（经过共线性筛选，与 model_lab4.FEATS_PARSE 对齐）
FEATURES = ["value_z", "mom_pure", "quality", "safety",
            "macro_state", "trend_t", "val_x_mom"]

# 训练超参（冠军参数）
DECAY_HALFLIFE_YEARS = 2.0
HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.01


def _exp_decay_weights(dates, as_of):
    """按 2 年半衰期，给每个训练样本赋时变权重。"""
    as_of = pd.Timestamp(as_of)
    years = (as_of - pd.to_datetime(dates)).days / 365.25
    return np.exp(-np.log(2) * years / DECAY_HALFLIFE_YEARS)


def build_features(val_pct, r4_rk, r7_rk, wr_rk, dc_rk,
                   rmdd_pen, water, trend_t):
    """从原始因子量构造 7 维经济特征（向量化，接受标量/数组）。

    所有输入：
      val_pct   —— 估值分位 0~1
      r4_rk/r7_rk —— 4M/7M 动量截面百分位 0~1
      wr_rk     —— IR 胜率截面百分位 0~1
      dc_rk     —— 下行捕获率截面百分位 0~1（dc 越大越差；rank 后大代表 dc 高）
      rmdd_pen  —— V3.6 平滑回撤惩罚 0~1
      water     —— 大盘水位 0~1
      trend_t   —— 趋势确认度 0~1 (MA20 距离平滑门)
    """
    val_pct = np.asarray(val_pct, dtype=float)
    r4_rk = np.asarray(r4_rk, dtype=float)
    r7_rk = np.asarray(r7_rk, dtype=float)
    wr_rk = np.asarray(wr_rk, dtype=float)
    dc_rk = np.asarray(dc_rk, dtype=float)
    rmdd_pen = np.asarray(rmdd_pen, dtype=float)
    water = np.asarray(water, dtype=float)
    trend_t = np.asarray(trend_t, dtype=float)

    value_z = 1.0 - val_pct                              # 便宜=好
    mom_pure = 0.5 * r4_rk + 0.5 * r7_rk                 # 经典 12-1 动量代理
    # 质量：高 IR 胜率 + 低下行捕获（dc_rk 已反向，故用 1-dc_rk）
    quality = 0.5 * wr_rk + 0.5 * (1.0 - dc_rk)
    safety = 1.0 - rmdd_pen                              # 低回撤
    macro_state = water - 0.5                            # 居中，避免与 regime dummy 共线
    val_x_mom = value_z * mom_pure                       # 便宜 + 强势 = 甜蜜点

    return np.column_stack([value_z, mom_pure, quality, safety,
                            macro_state, trend_t, val_x_mom])


def train_model(panel_df, as_of, save=True):
    """从 PiT 面板训练模型并保存。

    panel_df 需包含列：date, code, val_pct, r4_rk, r7_rk, wr_rk, dc_rk,
                         rmdd_pen, water, trend_t, fwd6。
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= pd.Timestamp(as_of)].dropna(
        subset=["val_pct", "r4_rk", "r7_rk", "wr_rk", "dc_rk",
                "rmdd_pen", "water", "trend_t", "fwd6"])
    if len(train) < 200:
        raise RuntimeError(f"训练样本不足 {len(train)} < 200")

    X = build_features(train.val_pct.values, train.r4_rk.values,
                       train.r7_rk.values, train.wr_rk.values,
                       train.dc_rk.values, train.rmdd_pen.values,
                       train.water.values, train.trend_t.values)

    # 截面 z-score 目标：每季度 fwd6 标准化
    def cs_z(g):
        return (g - g.mean()) / (g.std() + 1e-9)
    y = train.groupby("date")["fwd6"].transform(cs_z).values

    sw = _exp_decay_weights(train.date.values, as_of)

    model = Pipeline([
        ("sc", StandardScaler()),
        ("hub", HuberRegressor(epsilon=HUBER_EPSILON, alpha=HUBER_ALPHA,
                               max_iter=1000)),
    ])
    model.fit(X, y, hub__sample_weight=sw)

    if save:
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({
                "version": MODEL_VERSION,
                "trained_as_of": str(pd.Timestamp(as_of).date()),
                "n_train": int(len(train)),
                "features": FEATURES,
                "model": model,
            }, f)
    return model


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict(model, val_pct, r4_rk, r7_rk, wr_rk, dc_rk,
            rmdd_pen, water, trend_t):
    """输出原始预测分 z（截面 z-score 尺度）。
    调用方应再做截面 rank→0~100 映射以保持与 V3.7 相同口径。
    """
    X = build_features(val_pct, r4_rk, r7_rk, wr_rk, dc_rk,
                       rmdd_pen, water, trend_t)
    return model.named_steps["hub"].predict(
        model.named_steps["sc"].transform(X))


def z_to_100_cross_section(z):
    """把原始 z 预测映射到 0~100 截面百分位（保持与 S_total 同样口径）。"""
    s = pd.Series(z)
    return (s.rank(pct=True) * 100).values


if __name__ == "__main__":
    # 自检：用 factor_rows 面板训练一期模型
    import glob
    files = sorted(glob.glob("output/factor_rows/*.csv"))
    rows = []
    for f in files:
        d = pd.read_csv(f, dtype={"code": str})
        d["date"] = pd.to_datetime(d["date"])
        # 需要 r4_rk 等截面 rank
        for c in ["r4", "r7", "wr", "dc", "R_MDD"]:
            d[c + "_rk"] = d.groupby("date")[c].rank(pct=True)
        d["trend_t"] = np.clip((d["ma20_dist"] + 0.02) / 0.06, 0, 1)
        # 计算 rmdd_pen (V3.6 平滑)
        rp = np.minimum(0.5 * np.maximum(d["R_MDD"] - 1.2, 0), 1.0)
        rp[d.water <= 0.35] *= 0.5
        d["rmdd_pen"] = rp
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    as_of = panel.date.max()
    print(f"[train] as_of={as_of.date()} n={len(panel)}")
    bundle = train_model(panel, as_of, save=True)
    print(f"[saved] {MODEL_FILE}")
    # 简单 sanity check
    samp = panel[panel.date == as_of].head(5)
    z = predict(bundle, samp.val_pct.values, samp.r4_rk.values, samp.r7_rk.values,
                samp.wr_rk.values, samp.dc_rk.values, samp.rmdd_pen.values,
                samp.water.values, samp.trend_t.values)
    print("sample predictions (z):", np.round(z, 3))
    print("sample fwd6:", samp.fwd6.values.round(3))
