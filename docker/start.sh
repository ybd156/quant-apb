#!/bin/sh
# APB 容器入口：后台跑四类采集/演算，前台跑 Web 服务
#
# 为什么不把采集放进 webui：
#   INV-8 规定 UI 只做 IO、不做计算。采集是独立进程，只负责写 data/state/*.json，
#   Web 服务只读 JSON。两者解耦，采集挂了也不影响页面。
#
# 四类后台任务（按频率从高到低）：
#   1. 秒级快照   qcore.tick        —— 交易时段轮询，落盘 data/tick/
#   2. 行情情绪   qcore.market_live —— 60 秒，写 market.json
#   3. 版本检查   qcore.update_check—— 6 小时，写 update.json
#   4. 收盘演算   qcore.live_model  —— 每日 15:10 打分+盯市；每周重训
#   5. 数据源探活 qcore.health      —— 6 小时，写 source_health.json
#   6. 告警检查   qcore.alert       —— 30 分钟，写 alerts.jsonl
set -e

HOST="${APB_HOST:-0.0.0.0}"
PORT="${APB_PORT:-8765}"
MARKET_INTERVAL="${APB_MARKET_INTERVAL:-60}"
ENABLE_MARKET="${APB_MARKET:-1}"
UPDATE_INTERVAL="${APB_UPDATE_INTERVAL:-21600}"
ENABLE_UPDATE_CHECK="${APB_UPDATE_CHECK:-1}"
ENABLE_TICK="${APB_TICK:-1}"
TICK_INTERVAL="${APB_TICK_INTERVAL:-3}"
ENABLE_LIVE="${APB_LIVE:-1}"
LIVE_MAX_AGE="${APB_LIVE_MAX_AGE_DAYS:-7}"
ENABLE_HEALTH="${APB_HEALTH:-1}"
HEALTH_INTERVAL="${APB_HEALTH_INTERVAL:-21600}"
ENABLE_ALERT="${APB_ALERT:-1}"
ALERT_INTERVAL="${APB_ALERT_INTERVAL:-1800}"

echo "[entrypoint] APB $(cat /app/VERSION 2>/dev/null || echo unknown) on ${HOST}:${PORT}"
echo "[entrypoint] market=${ENABLE_MARKET} tick=${ENABLE_TICK} live=${ENABLE_LIVE} health=${ENABLE_HEALTH} alert=${ENABLE_ALERT}"

PIDS=""

# 1) 秒级快照：只盯「模型推荐 + 自选」的票，不扫全市场（N100 扛不住）
if [ "$ENABLE_TICK" = "1" ]; then
    (
        sleep 20
        while true; do
            python -c "
import json,sys
codes=[]
try:
    d=json.load(open('/app/data/state/picks.json'))
    codes=[r['code'] for r in (d.get('rows') or [])][:20]
except Exception: pass
try:
    w=json.load(open('/app/data/state/watchlist.json'))
    codes+= [it['code'] for it in (w.get('items') or [])][:10]
except Exception: pass
print(','.join(dict.fromkeys(codes)) or '600519')
" > /tmp/tick_codes.txt 2>/dev/null
            python -m qcore.tick --loop --interval="${TICK_INTERVAL}" \
                --codes="$(cat /tmp/tick_codes.txt 2>/dev/null || echo 600519)" || true
            sleep 10
        done
    ) &
    PIDS="$PIDS $!"
fi

# 2) 行情与市场情绪
if [ "$ENABLE_MARKET" = "1" ]; then
    (
        while true; do
            python -m qcore.market_live --interval="${MARKET_INTERVAL}" || true
            sleep 5
        done
    ) &
    PIDS="$PIDS $!"
fi

# 3) 版本检查
if [ "$ENABLE_UPDATE_CHECK" = "1" ]; then
    (
        sleep 15
        while true; do
            python -m qcore.update_check 2>&1 | tail -2 || true
            sleep "${UPDATE_INTERVAL}"
        done
    ) &
    PIDS="$PIDS $!"
fi

# 4) 三段式演算：盘前预测 / 盘中偏离 / 收盘复盘
#    - 08:30-09:30  盘前打分（用 T-1 数据，与训练口径一致）
#    - 09:30-15:00  盘中偏离监控（每 5 分钟刷新，不做择时）
#    - 15:05-17:00  收盘复盘 + 生成明日建议
#    - 每周（模型超龄）重训一次
if [ "$ENABLE_LIVE" = "1" ]; then
    (
        sleep 30
        while true; do
            python -c "
from qcore import session as S
r = S.run('auto', verbose=True)
print('[session]', r.get('session'), r.get('reason') or 'ok')
" 2>&1 | tail -3 || true
            # 盘中 5 分钟一刷，其它时段 30 分钟一查
            sleep 300
        done
    ) &
    PIDS="$PIDS $!"

    # 每周重训（模型超龄才跑，避免每天烧 10 分钟）
    (
        sleep 120
        while true; do
            python -c "
from qcore import live_model as LM
if LM.should_retrain(${LIVE_MAX_AGE}):
    print('[retrain] 模型超龄，开始重训 ...')
    print(LM.train(verbose=True))
else:
    print('[retrain] 模型仍新鲜，跳过')
" 2>&1 | tail -3 || true
            sleep 21600
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5) 数据源探活
if [ "$ENABLE_HEALTH" = "1" ]; then
    (
        sleep 45
        while true; do
            python -m qcore.health 2>&1 | tail -6 || true
            sleep "${HEALTH_INTERVAL}"
        done
    ) &
    PIDS="$PIDS $!"
fi

# 6) 告警检查
if [ "$ENABLE_ALERT" = "1" ]; then
    (
        sleep 60
        while true; do
            python -c "from qcore import alert as A; print(A.run_once('file'))" 2>&1 | tail -2 || true
            sleep "${ALERT_INTERVAL}"
        done
    ) &
    PIDS="$PIDS $!"
fi

cleanup() {
    for p in $PIDS; do kill "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM

exec python -m webui.server --host "${HOST}" --port "${PORT}"
