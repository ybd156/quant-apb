#!/usr/bin/env bash
# APB 训练机 · 一键入口（Linux / macOS）
#
#   bash apb.sh              菜单
#   bash apb.sh auto         一键：体检 → 配置 → 连通 → 起守护
#   bash apb.sh status|start|stop
set -u
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && PY="$c" && break; done
if [ -z "$PY" ]; then
  echo "  [X] 没找到 python3，请先安装 Python 3.10+"
  exit 1
fi
case "${1:-}" in
  auto)  exec "$PY" tools/train_machine_auto.py ;;
  "")    exec "$PY" tools/kit_menu.py ;;
  *)     exec "$PY" tools/kit_menu.py "$@" ;;
esac
