#!/bin/sh
# APB 容器入口：后台跑行情采集，前台跑 Web 服务
#
# 为什么不把采集放进 webui：
#   INV-8 规定 UI 只做 IO、不做计算。采集是独立进程，只负责写 data/state/market.json，
#   Web 服务只读 JSON。两者解耦，采集挂了也不影响页面。
set -e

HOST="${APB_HOST:-0.0.0.0}"
PORT="${APB_PORT:-8765}"
MARKET_INTERVAL="${APB_MARKET_INTERVAL:-60}"
ENABLE_MARKET="${APB_MARKET:-1}"

echo "[entrypoint] APB starting, version=$(cat /app/VERSION 2>/dev/null || echo unknown)"
echo "[entrypoint] web  : ${HOST}:${PORT}"
echo "[entrypoint] market collector: ${ENABLE_MARKET} (interval ${MARKET_INTERVAL}s)"

# 后台：实时行情与市场情绪采集
if [ "$ENABLE_MARKET" = "1" ]; then
    (
        while true; do
            python -m qcore.market_live --interval="${MARKET_INTERVAL}" || true
            sleep 5
        done
    ) &
    MARKET_PID=$!
    echo "[entrypoint] market collector pid=${MARKET_PID}"
fi

# 前台：Web 服务（收到 TERM 时一并清理子进程）
trap 'kill ${MARKET_PID:-0} 2>/dev/null || true' INT TERM
exec python -m webui.server --host "${HOST}" --port "${PORT}"
