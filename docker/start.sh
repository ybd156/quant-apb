#!/bin/sh
# APB 容器入口：后台跑行情采集与版本检查，前台跑 Web 服务
#
# 为什么不把采集放进 webui：
#   INV-8 规定 UI 只做 IO、不做计算。采集是独立进程，只负责写 data/state/*.json，
#   Web 服务只读 JSON。两者解耦，采集挂了也不影响页面。
set -e

HOST="${APB_HOST:-0.0.0.0}"
PORT="${APB_PORT:-8765}"
MARKET_INTERVAL="${APB_MARKET_INTERVAL:-60}"
ENABLE_MARKET="${APB_MARKET:-1}"
UPDATE_INTERVAL="${APB_UPDATE_INTERVAL:-21600}"   # 6 小时检查一次新版本
ENABLE_UPDATE_CHECK="${APB_UPDATE_CHECK:-1}"

echo "[entrypoint] APB starting, version=$(cat /app/VERSION 2>/dev/null || echo unknown)"
echo "[entrypoint] web  : ${HOST}:${PORT}"
echo "[entrypoint] market collector: ${ENABLE_MARKET} (interval ${MARKET_INTERVAL}s)"
echo "[entrypoint] update checker  : ${ENABLE_UPDATE_CHECK} (interval ${UPDATE_INTERVAL}s)"

MARKET_PID=""
UPDATE_PID=""

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

# 后台：版本检查（写 data/state/update.json，供 UI 显示"有新版本"）
# 启动后先跑一次，之后每 UPDATE_INTERVAL 秒一次
if [ "$ENABLE_UPDATE_CHECK" = "1" ]; then
    (
        sleep 15                       # 等网络就绪
        while true; do
            python -m qcore.update_check 2>&1 | tail -3 || true
            sleep "${UPDATE_INTERVAL}"
        done
    ) &
    UPDATE_PID=$!
    echo "[entrypoint] update checker pid=${UPDATE_PID}"
fi

cleanup() {
    [ -n "$MARKET_PID" ] && kill "$MARKET_PID" 2>/dev/null || true
    [ -n "$UPDATE_PID" ] && kill "$UPDATE_PID" 2>/dev/null || true
}
trap cleanup INT TERM

exec python -m webui.server --host "${HOST}" --port "${PORT}"
