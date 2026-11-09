#!/usr/bin/env bash
# APB 一键训练 —— Linux / macOS
# 用法: bash train.sh [--horizons 1,5] [--trees 250] ...
set -uo pipefail

cd "$(dirname "$0")" || exit 1

echo
echo "============================================================"
echo "  APB 一键训练"
echo "============================================================"
echo

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "[错误] 找不到 $PY，请先安装 Python 3.10+"; exit 1; }

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "[错误] Python 版本太旧，需要 3.10+"; "$PY" --version; exit 1; }

if ! "$PY" -c 'import numpy, pandas' >/dev/null 2>&1; then
  echo "[提示] 缺少依赖，正在安装 numpy / pandas ..."
  "$PY" -m pip install --quiet numpy pandas || {
    echo "[错误] 依赖安装失败。请手动执行: pip install numpy pandas"; exit 1; }
fi

echo "启动训练（首次运行需要下载约 545MB 数据，请耐心等待）..."
echo
"$PY" tools/train_one_click.py "$@"
rc=$?

echo
echo "============================================================"
echo "  训练结束。把 dist/models/ 里的 zip 拷到 NAS 即可。"
echo "============================================================"
echo
exit $rc
