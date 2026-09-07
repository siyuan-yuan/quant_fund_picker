#!/usr/bin/env bash
# V6 环境引导：沙箱重置后一键重建 venv 并跑不变量测试。
#
# 背景：venv 位于仓库外（/home/user/.venv_v6），不进快照，跨会话会丢失。
# 这已经是第三次沙箱重置（见 docs/V5_复跑事故与M11发现_2026-09-01.md）。
# 与其每次手工重来，不如把它固化成一条命令。
#
#   bash v6_bootstrap.sh
set -euo pipefail

VENV="${VENV:-/home/user/.venv_v6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[1/3] 创建 venv: $VENV"
  python3 -m venv "$VENV"
else
  echo "[1/3] venv 已存在: $VENV"
fi

echo "[2/3] 安装钉版依赖 (requirements-v5-lock.txt)"
"$VENV/bin/pip" -q install -r "$REPO/requirements-v5-lock.txt" requests

"$VENV/bin/python" - <<'PY'
import pandas, numpy, sklearn, scipy, statsmodels
exp = {"pandas": "3.0.5", "numpy": "2.4.6", "scikit-learn": "1.9.0",
       "scipy": "1.17.1", "statsmodels": "0.15.0"}
got = {"pandas": pandas.__version__, "numpy": numpy.__version__,
       "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
       "statsmodels": statsmodels.__version__}
bad = {k: (v, exp[k]) for k, v in got.items() if v != exp[k]}
print("     钉版核对:", got)
if bad:
    raise SystemExit(f"!! 版本不符（实测, 期望）: {bad}")
PY

echo "[3/3] 不变量测试"
cd "$REPO" && "$VENV/bin/python" -m pytest tests/ -q

cat <<'EOF'

就绪。常用命令：
  /home/user/.venv_v6/bin/python -m pytest tests/ -q
  /home/user/.venv_v6/bin/python -m pytest tests/test_v6_dyn_rbsa.py tests/test_v6_panel.py -q

注意：T0 数据轨仍受阻（沙箱网络层阻断全部行情源，取证见
output/v6/t0_source_probe.csv）。T1.2 起需要用户本机回填 cache/ 与 data/pit_universe/。
EOF
