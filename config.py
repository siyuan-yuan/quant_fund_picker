# -*- coding: utf-8 -*-
"""
量化选基系统 V3 —— 全局配置
模型公式: S_total = (0.40*min(F_value,100) + 0.35*F_alpha + 0.25*F_momentum)
                   × Π(1 - PenaltyRate_j)
"""

# ============ 因子权重 ============
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

CACHE_DIR = "cache"
OUTPUT_DIR = "output"
