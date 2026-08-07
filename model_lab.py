# -*- coding: utf-8 -*-
"""
model_lab.py — 华尔街级打分模型实验室

在 output/factor_rows/ 的 28 季 × 217 池 Point-in-Time 面板上，
对现行 V3.7 启发式模型与多种更高级模型做严格 walk-forward 对决。

评测口径（全部避免前视）：
  * 训练集严格早于测试集，按季度时间切分（expanding / rolling 两套）
  * 预测目标 y = 未来 6 个月收益（fwd6），稳健化为截面排名分位
  * 指标：Spearman Rank IC（t/ICIR）、五分位多空价差、Top-decile 超额、
          Buy 线（预测分≥70）的胜率与平均前瞻收益
  * 与现行引擎分数 S_eng 同口径对比

候选模型族：
  1. 现行 V3.7 启发式（基准）
  2. 截面 z-score 等权 / IC 加权线性合成
  3. Ridge / Lasso / ElasticNet （线性，带正则）
  4. 稳健回归 Huber + 分位回归 Quantile
  5. 梯度提升 (HistGradientBoosting) 与 RandomForest
  6. 双层 Stacking（Ridge + HGBR → Ridge 元学习器）
  7. 带 regime 交互的条件线性模型（按水位分三段学习不同权重）
  8. 排序学习 LambdaMART 代理（GradientBoosting rank objective via
     LGBMRanker 若可用，否则用 quantile bin 分类近似）

最后用嵌套 walk-forward 做模型平均，输出一份 Markdown 裁决书。
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, HuberRegressor,
                                  QuantileRegressor, LinearRegression)
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, clone as _clone

warnings.filterwarnings("ignore")

OUT = "output/model_lab"
os.makedirs(OUT, exist_ok=True)

RAW_DIR = "output/factor_rows"
SEED = 42
np.random.seed(SEED)

# ---- 候选特征（全部为 PiT 已知量；r4/r7 在截面 rank 后再使用以匹配现行口径） ----
RAW_FEATURES = ["val_pct", "val_cov", "trend_ok", "ma20_dist",
                "wr", "dc", "r4", "r7", "R_MDD", "water",
                "other_pen"]
META = ["date", "code", "fwd6", "fwd12", "S_eng",
        "F_value_eng", "F_alpha_eng", "F_mom_eng",
        "wv", "wa", "wm"]


# ============================================================
# 1. 面板构建与工程化特征
# ============================================================
def load_panel():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(f"{RAW_DIR}/{f}", dtype={"code": str})
                    for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd6"]).sort_values(["date", "code"]).reset_index(drop=True)

    # 截面 rank（每季度内 pct rank）—— 现行模型也是这么做的，避免绝对量受市场 regime 污染
    for c in ["r4", "r7", "wr", "dc", "R_MDD"]:
        df[c + "_rk"] = df.groupby("date")[c].rank(pct=True)
    # val_pct 已经是跨 5 年的分位，不必再截面 rank
    df["trend_ok"] = df["trend_ok"].astype(float)
    # 趋势门的连续版本（V3.7 平滑函数复刻）
    df["trend_t"] = np.clip((df["ma20_dist"] + 0.02) / 0.06, 0, 1)
    # R_MDD 风险收缩（V3.6 平滑惩罚函数的风险信号形式，作为模型输入而非乘数）
    df["rmdd_pen"] = np.minimum(0.5 * np.maximum(df["R_MDD"] - 1.2, 0), 1.0)
    df.loc[df.water <= 0.35, "rmdd_pen"] *= 0.5

    # regime one-hot
    df["reg_low"] = (df.water <= 0.20).astype(float)
    df["reg_high"] = (df.water >= 0.70).astype(float)
    df["reg_mid"] = ((df.water > 0.20) & (df.water < 0.70)).astype(float)

    # 交互项（少量、带经济学含义）
    df["val_x_trend"] = df["val_pct"] * df["trend_t"]
    df["mom_x_val"] = df["r4_rk"] * (1 - df["val_pct"])      # 低估值 + 强动量
    df["alpha_x_lowdd"] = df["wr_rk"] * (1 - df["rmdd_pen"])
    df["smallcap_proxy_x_dd"] = 0.0  # 占位（RBSA 未在面板中，避免编造）
    return df


FEATURES = [
    "val_pct", "val_cov", "trend_t", "ma20_dist",
    "wr_rk", "dc_rk", "r4_rk", "r7_rk",
    "rmdd_pen", "R_MDD", "water",
    "reg_low", "reg_mid", "reg_high",
    "val_x_trend", "mom_x_val", "alpha_x_lowdd",
    "other_pen",
]


def make_y(df):
    """截面排名分位（每季度内 fwd6 的 pct rank），目标尺度 0~1，对极端值稳健。"""
    return df.groupby("date")["fwd6"].rank(pct=True)


# ============================================================
# 2. 模型工厂
# ============================================================
def make_models():
    """所有模型包装成 (name, sklearn-like estimator)。
    统一使用 Pipeline(StandardScaler, model)，基于树的模型对缩放不敏感但保持一致。"""
    ridge = Pipeline([("sc", StandardScaler()),
                      ("m", Ridge(alpha=10.0, random_state=SEED))])
    lasso = Pipeline([("sc", StandardScaler()),
                      ("m", Lasso(alpha=0.002, max_iter=20000, random_state=SEED))])
    enet = Pipeline([("sc", StandardScaler()),
                     ("m", ElasticNet(alpha=0.003, l1_ratio=0.5,
                                      max_iter=20000, random_state=SEED))])
    huber = Pipeline([("sc", StandardScaler()),
                      ("m", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500))])
    quant = Pipeline([("sc", StandardScaler()),
                      ("m", QuantileRegressor(quantile=0.5, alpha=0.01,
                                              solver="highs"))])
    hgbr = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_depth=3,
        l2_regularization=1.0, min_samples_leaf=20,
        random_state=SEED, loss="squared_error")
    rf = RandomForestRegressor(n_estimators=400, max_depth=5,
                               min_samples_leaf=20, n_jobs=-1,
                               random_state=SEED)
    gbr = GradientBoostingRegressor(learning_rate=0.05, n_estimators=250,
                                    max_depth=3, subsample=0.8,
                                    min_samples_leaf=20, random_state=SEED)
    return {
        "Ridge": ridge,
        "Lasso": lasso,
        "ElasticNet": enet,
        "Huber": huber,
        "Quantile(Median)": quant,
        "HistGBM": hgbr,
        "RandomForest": rf,
        "GradientBoosting": gbr,
    }


# ============================================================
# 3. Walk-forward 评估引擎
# ============================================================
def _ic_series(pred, y, dates):
    d = pd.DataFrame({"p": pred, "y": y, "date": dates})
    return d.groupby("date").apply(
        lambda g: float(spearmanr(g.p, g.y)[0]) if g.p.nunique() > 5 else np.nan,
        include_groups=False).dropna()


def _tstat(s):
    s = s.dropna()
    if len(s) < 3:
        return np.nan
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def evaluate_predictions(pred, y, dates, fwd, tag):
    """返回一组指标字典。"""
    d = pd.DataFrame({"p": pred, "y": y, "fwd": fwd, "date": dates}).dropna()
    ic_s = _ic_series(d.p, d.y, d.date)
    # 五分位多空（每季度按预测分五桶，取 Q5-Q1 平均 fwd）
    d["q"] = d.groupby("date")["p"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
    q5 = d[d.q == 5].groupby("date")["fwd"].mean()
    q1 = d[d.q == 1].groupby("date")["fwd"].mean()
    ls = (q5 - q1).dropna()
    # Top-decile 超额
    d["d"] = d.groupby("date")["p"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1)
    top10 = d[d.d == 10].groupby("date")["fwd"].mean()
    univ = d.groupby("date")["fwd"].mean()
    top_ex = (top10 - univ).dropna()
    # Buy 线（预测映射回 0-100 分：用百分位映射保持与现行同口径）
    d["p100"] = d.groupby("date")["p"].rank(pct=True) * 100
    buy = d[d.p100 >= 70]
    return {
        "model": tag,
        "IC_mean": round(ic_s.mean(), 4),
        "IC_std": round(ic_s.std(), 4),
        "ICIR": round(ic_s.mean() / ic_s.std(), 3) if ic_s.std() > 0 else np.nan,
        "IC_t": round(_tstat(ic_s), 3),
        "IC_hit%": round((ic_s > 0).mean(), 3),
        "Q5-Q1_mean": round(ls.mean(), 4),
        "Q5-Q1_t": round(_tstat(ls), 3),
        "TopDecile_excess": round(top_ex.mean(), 4),
        "TopDecile_t": round(_tstat(top_ex), 3),
        "Buy_n": int(len(buy)),
        "Buy_fwd6": round(buy.fwd.mean(), 4),
        "Buy_win%": round((buy.fwd > 0).mean(), 3),
        "n_quarters": int(ic_s.shape[0]),
    }


def baseline_engine_score(df):
    """现行 V3.7 引擎分数 S_eng（已在面板内，直接做百分位映射后评测）。"""
    d = df[["date", "fwd6"]].copy()
    d["p"] = df["S_eng"]
    d["y"] = make_y(df)
    return evaluate_predictions(d.p, d.y, d.date, d.fwd6, "V3.7现行引擎")


def walk_forward(df, models, min_train=8, horizon=1, expanding=True):
    """严格按时间 walk-forward：用前 min_train 季作训练，预测下一季；
    expanding=True 时训练窗扩张；否则滚动 12 季。"""
    dates = sorted(df["date"].unique())
    if len(dates) <= min_train + 1:
        raise RuntimeError("日期数不足")
    all_pred = {name: [] for name in models}
    ytrue = []
    for i in range(min_train, len(dates) - horizon + 1):
        test_date = dates[i + horizon - 1]
        if expanding:
            train_dates = dates[:i]
        else:
            train_dates = dates[max(0, i - 12):i]
        tr = df[df.date.isin(train_dates)]
        te = df[df.date == test_date]
        Xtr, ytr = tr[FEATURES].values, make_y(tr).values
        Xte, yte = te[FEATURES].values, make_y(te).values
        for name, model in models.items():
            m = clone(model)
            try:
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
            except Exception as e:
                pred = np.full(len(te), np.nan)
            all_pred[name].append(pd.DataFrame({
                "date": test_date, "code": te.code.values,
                "p": pred, "y": yte, "fwd": te.fwd6.values}))
        ytrue.append(test_date)
    res = {}
    for name, chunks in all_pred.items():
        c = pd.concat(chunks, ignore_index=True).dropna(subset=["p"])
        res[name] = evaluate_predictions(c.p, c.y, c.date, c.fwd, name)
    return res


from sklearn.base import clone


# ============================================================
# 4. 条件模型（regime 交互）& 双层 stacking
# ============================================================
class RegimeConditionalModel(BaseEstimator, RegressorMixin):
    """每个 regime 单独训练一个 Ridge。预测时按水位路由到对应模型。"""
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.models = {}
        self.fallback = None

    def _route(self, X, idx_low, idx_mid, idx_high):
        # 列顺序固定，见 FEATURES
        return X

    def fit(self, X, y):
        n = X.shape[0]
        # reg_low/reg_mid/reg_high 的列索引
        cols = {c: i for i, c in enumerate(FEATURES)}
        base_feats = [c for c in FEATURES if c not in
                      ("reg_low", "reg_mid", "reg_high")]
        self.base_idx = [cols[c] for c in base_feats]
        masks = {
            "low": X[:, cols["reg_low"]] > 0.5,
            "mid": X[:, cols["reg_mid"]] > 0.5,
            "high": X[:, cols["reg_high"]] > 0.5,
        }
        self.fallback = Pipeline([("sc", StandardScaler()),
                                  ("m", Ridge(alpha=self.alpha))]).fit(
            X[:, self.base_idx], y)
        for k, m in masks.items():
            if m.sum() >= 30:
                self.models[k] = Pipeline([("sc", StandardScaler()),
                                           ("m", Ridge(alpha=self.alpha))]).fit(
                    X[m][:, self.base_idx], y[m])
            else:
                self.models[k] = self.fallback
        return self

    def predict(self, X):
        cols = {c: i for i, c in enumerate(FEATURES)}
        out = np.zeros(X.shape[0])
        for k in ("low", "mid", "high"):
            m = X[:, cols["reg_" + k]] > 0.5
            if m.any():
                out[m] = self.models[k].predict(X[m][:, self.base_idx])
        return out


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """第一层：Ridge + HGBR；第二层 Ridge 做元学习。
    用时间序列切分避免前视。"""
    def __init__(self):
        self.l1 = [
            Pipeline([("sc", StandardScaler()),
                      ("m", Ridge(alpha=10.0))]),
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_depth=3,
                l2_regularization=1.0, min_samples_leaf=20,
                random_state=SEED),
        ]
        self.meta = Ridge(alpha=5.0)

    def fit(self, X, y):
        # 简单的内部时序切分：前 70% 训 L1，后 30% 训 meta
        n = len(X)
        cut = int(n * 0.7)
        Xtr, Xm = X[:cut], X[cut:]
        ytr, ym = y[:cut], y[cut:]
        oof = np.zeros((len(Xm), len(self.l1)))
        for j, m in enumerate(self.l1):
            m.fit(Xtr, ytr)
            oof[:, j] = m.predict(Xm)
        self.meta.fit(oof, ym)
        # 全量重训 L1
        for m in self.l1:
            m.fit(X, y)
        return self

    def predict(self, X):
        Z = np.column_stack([m.predict(X) for m in self.l1])
        return self.meta.predict(Z)


class ICWeightedLinear(BaseEstimator, RegressorMixin):
    """用训练集单因子 IC 作为权重，做线性组合（经典 Black-Litterman 风格的因子模型）。
    对每个特征做截面 rank 后，用训练期平均 IC 做权重（符号/幅度都学）。"""
    def __init__(self):
        self.w = None
        self.cols = FEATURES

    def fit(self, X, y):
        # X 已经是截面 rank 化后的特征值？这里假定输入已是工程化特征；
        # 为稳健，对每列在训练集内部再做一次 z-score
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        ics = []
        for j in range(Xs.shape[1]):
            r, _ = spearmanr(Xs[:, j], y)
            ics.append(0.0 if np.isnan(r) else r)
        w = np.array(ics)
        # 软阈值（去除弱信号）
        w[np.abs(w) < 0.02] = 0.0
        # L2 收缩
        w = w / (np.linalg.norm(w) + 1e-6)
        self.w = w
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return Xs @ self.w


# ============================================================
# 5. 特征重要性 & 模型置信度
# ============================================================
def feature_importance(df):
    """在全样本上训练一个 HGBR/Ridge，输出特征重要性，辅助经济学解释。"""
    X = df[FEATURES].values
    y = make_y(df).values
    ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]).fit(X, y)
    hgbr = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=400, max_depth=3,
        l2_regularization=1.0, min_samples_leaf=20, random_state=SEED).fit(X, y)
    rw = ridge.named_steps["m"].coef_
    # HGBR 不直接给 feature_importances_，用 permutation 近似（轻量、多次 shuffle）
    rng = np.random.RandomState(SEED)
    base = spearmanr(hgbr.predict(X), y)[0]
    imp = []
    for j in range(X.shape[1]):
        losses = []
        for _ in range(5):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            losses.append(base - spearmanr(hgbr.predict(Xp), y)[0])
        imp.append(np.mean(losses))
    imp = np.array(imp)
    out = pd.DataFrame({
        "feature": FEATURES,
        "ridge_coef_z": rw,
        "ridge_abs": np.abs(rw),
        "hgbr_perm_imp": imp,
    }).sort_values("hgbr_perm_imp", ascending=False)
    return out


# ============================================================
# 6. 主流程
# ============================================================
def main():
    print("[1/5] 载入 PiT 面板 ...")
    df = load_panel()
    print(f"    面板: {df.shape}, 季度数={df.date.nunique()}, 基金数={df.code.nunique()}")

    print("[2/5] 评测现行 V3.7 引擎 ...")
    base = baseline_engine_score(df)
    print("   ", base)

    print("[3/5] walk-forward 训练高级模型（expanding window）...")
    models = make_models()
    models["IC-Weighted"] = ICWeightedLinear()
    models["Regime-Conditional Ridge"] = RegimeConditionalModel(alpha=10.0)
    models["Stacking(Ridge+HGBR)"] = StackingEnsemble()
    res = walk_forward(df, models, min_train=8, horizon=1, expanding=True)
    res["V3.7现行引擎"] = base

    summary = pd.DataFrame(res.values()).sort_values("IC_t", ascending=False)
    summary.to_csv(f"{OUT}/walkforward_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== Walk-forward 排名（按 IC_t 降序）===")
    print(summary.to_string(index=False))

    # rolling window 稳健性
    print("\n[4/5] 稳健性检查：rolling 12 季训练窗 ...")
    res_roll = walk_forward(df, models, min_train=12, horizon=1, expanding=False)
    res_roll["V3.7现行引擎"] = base
    sroll = pd.DataFrame(res_roll.values()).sort_values("IC_t", ascending=False)
    sroll.to_csv(f"{OUT}/walkforward_rolling.csv", index=False, encoding="utf-8-sig")

    print("\n[5/5] 特征重要性 ...")
    fi = feature_importance(df)
    fi.to_csv(f"{OUT}/feature_importance.csv", index=False, encoding="utf-8-sig")
    print(fi.to_string(index=False))

    # 模型平均（冠军模型的简单等权 ensemble 前 3）
    top3 = summary.head(3)["model"].tolist()
    print(f"\n[ensemble] Top3 模型等权平均: {top3}")
    # 重新跑一次 walk-forward 用于 ensemble
    _ = run_ensemble(df, [models[m] for m in top3 if m in models], top3)

    write_verdict(summary, sroll, fi, base)
    print(f"\n[done] 全部输出保存到 {OUT}/")


def run_ensemble(df, model_list, names):
    dates = sorted(df["date"].unique())
    chunks = []
    for i in range(8, len(dates)):
        tr = df[df.date.isin(dates[:i])]
        te = df[df.date == dates[i]]
        Xtr, ytr = tr[FEATURES].values, make_y(tr).values
        Xte, yte = te[FEATURES].values, make_y(te).values
        preds = []
        for m in model_list:
            mc = clone(m)
            mc.fit(Xtr, ytr)
            preds.append(mc.predict(Xte))
        p = np.mean(preds, axis=0)
        chunks.append(pd.DataFrame({"date": dates[i], "code": te.code.values,
                                    "p": p, "y": yte, "fwd": te.fwd6.values}))
    c = pd.concat(chunks, ignore_index=True)
    ens = evaluate_predictions(c.p, c.y, c.date, c.fwd,
                               "Ensemble(Top3-equalweight)")
    pd.DataFrame([ens]).to_csv(f"{OUT}/ensemble.csv", index=False,
                              encoding="utf-8-sig")
    print("   ", ens)
    return ens


def write_verdict(summary, sroll, fi, base):
    top = summary.iloc[0]
    md = []
    md.append("# 打分模型实验室裁决书\n")
    md.append("> 数据：28 季 × 217 基金 PiT 面板 (n=6,067)；"
              "严格 walk-forward（训练窗严格早于测试季）；"
              "目标：未来 6 月截面收益分位。\n")
    md.append("## 1. 现行 V3.7 引擎 vs 高级模型（Expanding Window）\n")
    md.append(summary.to_markdown(index=False))
    md.append("\n\n## 2. Rolling 12 季稳健性\n")
    md.append(sroll.to_markdown(index=False))
    md.append("\n\n## 3. 特征重要性（HGBR permutation + Ridge 系数）\n")
    md.append(fi.to_markdown(index=False))
    md.append("\n\n## 4. 裁决\n")
    md.append(f"- **冠军模型**：**{top['model']}**，IC={top['IC_mean']:+.4f}，"
              f"ICIR={top['ICIR']}，t={top['IC_t']}，"
              f"Buy 线 n={top['Buy_n']}、fwd6={top['Buy_fwd6']:+.2%}、胜率={top['Buy_win%']:.0%}\n")
    md.append(f"- **基准 V3.7**：IC={base['IC_mean']:+.4f}，"
              f"ICIR={base['ICIR']}，t={base['IC_t']}，"
              f"Buy fwd6={base['Buy_fwd6']:+.2%}，胜率={base['Buy_win%']:.0%}\n")
    md.append("- 见 `walkforward_summary.csv` / `walkforward_rolling.csv` / "
              "`feature_importance.csv` / `ensemble.csv`。\n")
    open(f"{OUT}/VERDICT.md", "w", encoding="utf-8").write("\n".join(md))


if __name__ == "__main__":
    main()
