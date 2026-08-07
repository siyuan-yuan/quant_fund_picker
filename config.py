# -*- coding: utf-8 -*-
"""
量化选基系统 V4 —— 全局配置
默认打分模型: V4 Huber 稳健回归 (7 经济学特征 + 2 年时间衰减)
  - 训练: 28 季 PiT 面板 n=6,067; Huber(ε=1.35, α=0.01)
  - 验证: WF IC 0.198 vs V3.7 0.116 (+71%); strict OOS IC 0.171 vs 0.093 (+85%)
  - 模型文件: cache/v4_model.pkl (用 model_v4.py 训练/刷新)
  - 若 sklearn 缺失或模型加载失败，自动回退到 V3.7 线性合成
旧公式 (作为 S_v37 保留，便于 A/B):
  S_V37 = (0.40*min(F_value,100) + 0.35*F_alpha + 0.25*F_momentum)
          × Π(1 - PenaltyRate_j)
"""

# V4 模型默认开关（False 时 engine.finalize 强制回退 V3.7，便于回滚）
# V4.1 注: 小样本入口(单基/批量/诊断)的 V4 必须配合全市场参照快照使用(ECDF)，
#          快照缺失时一致性闸门自动整体降级 V3.7，绝不使用批内 rank —— 见 engine.finalize
USE_V4_MODEL = True
# V4 与 V3.7 混合权重: S_final = W_V4*S_V4 + (1-W_V4)*S_v37
# V4 pure IC 最高但在高水位有追涨倾向；混合 0.5/0.5 保留 V3.7 的估值安全垫
W_V4 = 0.5

# ============ V3.7 线性合成的因子权重 (保留做 S_v37 与回退) ============
W_VALUE, W_ALPHA, W_MOMENTUM = 0.40, 0.35, 0.25

# ============ V3.2 Regime 自适应 (回测证据驱动) ============
# 底部水位 7.6%~18.3%, 非底部最低47% → 20%阈值最大间隔分离
REGIME_LOW_WATER = 0.20                       # 大盘水位≤20% → 左侧低估区
W_VALUE_LOW, W_ALPHA_LOW, W_MOM_LOW = 0.55, 0.35, 0.10
REGIME_HIGH_WATER = 0.90                      # ≥90% → 防御姿态(展示)
WATER_PANEL = "style6"                        # 水位计仅用6风格等权(宽基水位的代表)

# ============ RBSA 十六因子面板 (V3.3: 12个A股 + 4个境外腿) ============
# (收益源, 收益代码, 名称, PE代码, 风格标签)
#   PE代码前缀: "lg:"=乐咕 | "csi:"=中证官方 | "none"=无PE(估值盲区权重)
RBSA_INDICES = [
    ("sina",    "sh000016", "上证50",   "lg:上证50",   "mega"),      # 超大盘
    ("sina",    "sh000300", "沪深300",  "lg:沪深300",  "large"),     # 大盘
    ("sina",    "sh000905", "中证500",  "lg:中证500",  "mid"),       # 中盘
    ("sina",    "sh000852", "中证1000", "lg:中证1000", "small"),     # 小盘
    ("sina",    "sz399673", "创业板50", "lg:创业板50", "growth"),    # 高成长/高换手
    ("sina",    "sh000015", "上证红利", "lg:上证红利", "dividend"),  # 深度价值
    ("csindex", "000987",   "全指材料", "csi:000987",  "material"),
    ("csindex", "000988",   "全指工业", "csi:000988",  "industrial"),
    ("csindex", "000990",   "全指消费", "csi:000990",  "consumer"),
    ("csindex", "000991",   "全指医药", "csi:000991",  "health"),
    ("csindex", "000992",   "全指金融", "csi:000992",  "finance"),
    ("csindex", "000993",   "全指信息", "csi:000993",  "tech"),
    # ---- V3.3 境外腿: QDII母市场暴露 ----
    ("us_sina", ".NDX",     "纳斯达克100", "none",        "us_growth"),
    ("us_sina", ".INX",     "标普500",     "none",        "us_broad"),
    ("hk_sina", "HSI",      "恒生指数",    "csi:H30090",  "hk_broad"),   # PE代理: 中证香港300
    ("hk_sina", "HSTECH",   "恒生科技",    "csi:H30533",  "hk_tech"),    # PE代理: 中概互联网50
]
# 中小盘/高换手风格集合(规模反噬阀用)
SMALLCAP_STYLE = {"mid", "small", "growth"}
# 境外腿 (V3.3)
OVERSEAS_SRCS = {"us_sina", "hk_sina"}
OVERSEAS_NAMES = {x[2] for x in RBSA_INDICES if x[0] in OVERSEAS_SRCS}
# 海外类型基金的"借壳"判定: 统一面板预筛境外权重≥此值 → 切换纯境外面板
OVERSEAS_SWITCH_THRESHOLD = 0.25

# ============ RBSA 参数 ============
RBSA_WINDOW = 60                 # 主窗口: 近60个交易日
RBSA_SMOOTH_OFFSETS = [0, 5, 10] # 平滑: 终点向前偏移0/5/10天分别回归后取平均
RIDGE_ALPHA = 1.0                # L2 岭回归惩罚强度

# ============ 估值分位 (F_value) ============
PE_PERCENTILE_WINDOW = 1260      # 5年交易日

# ============ Alpha (F_alpha) ============
ALPHA_LOOKBACK_DAYS = 756        # 3年交易日
IR_WINDOW = 126                  # 6个月滚动窗口
IR_STEP = 21                     # 窗口步长(月度)
IR_THRESHOLD = 0.3               # IR>0.3 记为胜

# ============ 动量 (F_momentum) ============
# 滞后动量: 过去4个月剔除最近1个月 / 过去7个月剔除最近1个月
MOM_4M_START, MOM_4M_END = 100, 21   # 约 t-100 → t-21
MOM_7M_START, MOM_7M_END = 147, 21   # 约 t-147 → t-21

# ============ 风控阈值 ============
TENURE_MIN_DAYS = 3 * 365        # 完整归因期(语义保留)
# ---- V3.6 任期归因平滑折价 (仅主动型基金; 被动指数豁免) ----
# 证据: 剔除时点后任期无预测力(p=0.318) → 折价而非否决; 用户定调"平滑、温和"
TENURE_SMOOTH_MAX = 0.30         # 平滑曲线: 任期0 → -30%, 线性衰减至 3年 → 0
TENURE_CAP_DAYS = 730            # 任期<2年: 评级封顶 Buy(不得StrongBuy)
# ---- V3.6 R_MDD 平滑惩罚 ----
# 证据锚点: 毒性悬崖在 R≈2.0(>2: -14.7%/7%胜率), ≤1.2与≤1.0桶无差异(+9.3%/+7.4%)
MDD_SMOOTH_FREE = 1.2            # ≤1.2 免罚 (与≤1.0桶统计无差异)
MDD_SMOOTH_K = 0.5               # 斜率: p=min(0.5*(R-1.2),1) → 1.5→-15% 2.0→-40% 2.5→-65% 3.2→-100%
MDD_VETO_LOW_WATER = 0.35        # 水位≤35%底部区
MDD_LOWWATER_DAMP = 0.5          # 底部区惩罚减半
# ---- V3.5 条款4: 新星观察池 ----
YOUNG_MAX_DAYS = 756             # 净值<3年 → 观察池(评级封顶🌱观察仓)
YOUNG_TOP_N = 100                # 近1年收益Top100入观察池
AUM_STYLE_LIMIT_YI = 150         # 中小盘风格规模>150亿 扣0.3
CONCENTRATION_LIMIT = 0.70       # 前三大板块集中度>70% 且胜率<50% 扣0.4

# ============ 缩量加分信号 ============
AUM_SHRINK_TRIGGER = 0.50        # 较高峰缩水>50%
AUM_SHRINK_RANGE = (5, 20)       # 当前规模须在5-20亿
BONUS_POINTS = 20

# ============ 评级 ============
RATING_BANDS = [(85, "Strong Buy 绿灯"), (70, "Buy 浅绿"),
                (50, "Hold 黄灯"), (0, "Sell/Avoid 红灯")]

# ============ V3.8 战略交易纪律层（Calmar 优化版） ============
# 与打分模型分层的执行规则: 信号=S, 交易纪律=本组常量。
# 最新主候选来自 `strategy_experiment.py` 全样本实验：
# fixedslot_qreb_trail20_y25_crisis_cppi_15_20_25
#   固定10槽 + 季度再平衡 + 单基20%移动止损 + 现金2.5%
#   + 沪深300 MA200&Vol80危机禁买 + CPPI(-15/-20/-25)重置HWM。
STRAT_VERSION = "V3.8 crisis_cppi_15_20_25"
STRAT_BUY_TH = 70.0          # 战略仓建仓线: S>70
STRAT_SELL_TH = 45.0         # 战略仓平仓线: S<45 (迟滞带 45~70 持有)
                             #  裁决: <50→<45 总收益+76.1%→+100.4%, Calmar 0.44→0.55
                             #  <40 次之(+92.7%); <60 更差(+70.2%) —— 迟滞带宜宽不宜紧
STRAT_SLOTS = 10             # 等权槽位数

# 收益增强层：空置现金进入货基/短债代理，日度单利计提
STRAT_CASH_YIELD = 0.025     # 闲置现金年化收益假设 2.5%
STRAT_REBALANCE = "quarterly"# 季度等权再平衡（只在季末决策日执行）
STRAT_TRAIL_STOP = 0.20      # 单基金自入场后高点回撤20%止损

# 宏观危机过滤：沪深300跌破MA200 且 20日实现波动率 > 历史80%分位 → 禁止新开权益仓
STRAT_CRISIS_MA = 200
STRAT_CRISIS_VOL_WINDOW = 20
STRAT_CRISIS_VOL_Q = 0.80

# 组合级动态风险预算 / CPPI（以策略自身净值曲线为锚）
STRAT_CPPI = True
STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1 = -0.15, 6   # 回撤≤-15% → 最多6槽
STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2 = -0.20, 3   # 回撤≤-20% → 最多3槽
STRAT_CPPI_DD3, STRAT_CPPI_SLOTS3 = -0.25, 0   # 回撤≤-25% → 清仓，等待右侧信号重启
STRAT_CPPI_HWM_MODE = "reset"                  # 熔断后遇右侧信号且非危机：HWM重置为当前权益

# 保留的辅助安全阀/展示项
STRAT_HI_WATER = 0.90        # 大盘水位≥90% → 持仓侧强制瘦身（历史变体中多为休眠）
STRAT_HI_WATER_SLOTS = 5
STRAT_STYLE_CAP = 0.35       # 组合级单一RBSA板块暴露上限 (批判清单⑧, 批算页警示)
# 已数据否决、归档封死的执行层方案:
#   MA20破位离场 / 15%移动止损 / 提前<60卖 / 单纯截面前10%强制买 / 单纯水位高估降仓

CACHE_DIR = "cache"
OUTPUT_DIR = "output"
