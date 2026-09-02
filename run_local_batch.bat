@echo off
REM ============================================================
REM  V5 本机批跑入口（Windows CMD 版）
REM
REM  用法（在本仓库目录下打开 cmd）:
REM     run_local_batch.bat setup          安装虚拟环境
REM     run_local_batch.bat parity         跨机一致性闸门（D0.2 复权对账，约 1 分钟）
REM     run_local_batch.bat base  2        构建规范面板（canonical，159 月，约 40 分钟）
REM     run_local_batch.bat s61   6        S6.1：11 组面板，6 路并行
REM     run_local_batch.bat r35   8        R3.5：动物园缓存 8 路并行 + ML 面板 + zoo 重跑
REM     run_local_batch.bat all   6        依次：parity -> base -> r35(缓存) -> s61
REM     run_local_batch.bat status         查看完成度
REM
REM  说明:
REM   1. 所有长任务均断点续跑：中断后重跑同一条命令即可，已完成部分不重算。
REM   2. 并行度建议 <= 物理核数/2；单面板进程约 257MB 内存，瓶颈是 CPU 不是内存。
REM   3. output/ 与 cache/ 不入库（.gitignore）。跑完只把小汇总回传仓库。
REM   4. 口径标签：本批产物一律 SURV-ADJ，禁止当作 FULL-PIT 证据（U4/A5 未解锁）。
REM ============================================================
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if "%PY%"=="" set "PY=.venv_review\Scripts\python.exe"
set "LANE=%1"
if "%LANE%"=="" set "LANE=status"
set "N=%2"
if "%N%"=="" set "N=4"

if /i "%LANE%"=="setup"  goto setup
if /i "%LANE%"=="parity" goto parity
if /i "%LANE%"=="base"   goto base
if /i "%LANE%"=="s61"    goto s61
if /i "%LANE%"=="r35"    goto r35
if /i "%LANE%"=="all"    goto all
if /i "%LANE%"=="status" goto status
echo 用法: run_local_batch.bat {setup^|parity^|base^|s61^|r35^|all^|status} [并行度N]
exit /b 1

:setup
echo [setup] 创建虚拟环境并安装钉版依赖
python -m venv .venv_review || exit /b 1
"%PY%" -m pip install --upgrade pip || exit /b 1
"%PY%" -m pip install -r requirements-v5-lock.txt || exit /b 1
"%PY%" -m pip install requests py_mini_racer || exit /b 1
"%PY%" -c "import pandas,numpy,sklearn,statsmodels;print('OK',pandas.__version__,numpy.__version__,sklearn.__version__,statsmodels.__version__)"
echo [setup] 完成。若上面出现版本，即可继续。
goto end

:parity
echo [parity] 1/2 构建复权序列（约 1 分钟）
"%PY%" build_navadj.py --workers %N% || exit /b 1
echo [parity] 2/2 与权威值对账
"%PY%" status_local.py parity || exit /b 1
goto end

:base
echo [base] 构建规范面板 canonical（159 月，约 40 分钟，断点续跑）
"%PY%" p1_panel_build.py --workers %N% || exit /b 1
"%PY%" status_local.py status
goto end

:s61
echo [S6.1] 11 组面板（parallel=%N%）
"%PY%" s61_runner.py --parallel %N% --workers 2 || exit /b 1
echo [S6.1] 汇总（仅报告不平判）
"%PY%" s61_summarize.py || exit /b 1
echo [S6.1] done -^> output\v5\s61_robustness.csv / s61_summary.md
goto end

:r35
echo [R3.5] 动物园评分缓存（145 月，workers=%N%）
"%PY%" r35_rebuild_zoo_cache.py --workers %N% || exit /b 1
echo [R3.5] ML 面板 + 修复版动物园重跑
"%PY%" _build_ml_panel.py || exit /b 1
"%PY%" r35_zoo_redo.py || exit /b 1
echo [R3.5] done -^> output\v5\r35_zoo_redo\ 与 output\model_zoo_report.md 追加修订段
goto end

:all
call :parity || exit /b 1
call :base || exit /b 1
call :r35 || exit /b 1
call :s61 || exit /b 1
goto end

:status
"%PY%" status_local.py status
goto end

:end
endlocal
