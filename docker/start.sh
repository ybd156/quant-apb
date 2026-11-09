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

    # 模型超龄只告警，**不在 NAS 上训练**。
    # 训练是训练机的活（可以是有 GPU 的机器、或任意能跑 Python 的电脑），
    # NAS 只做推理 —— 秒级完成，不给存储/CPU 增加负担。
    # 训练机跑完会自动上传，NAS 收到即生效。
    # 模型超龄自动重训。
    #
    # 训练机是主力（能上 GPU、不影响服务），但**不是一直开着**。
    # NAS 兜底：训练机不在时自己保持模型新鲜。两边靠模型文件年龄协调，
    # 训练机刚传完 NAS 就看到"新鲜"，不会重复劳动。
    #
    # 三条保护，避免抢服务资源：
    #   1. 低优先级（nice）：API 需要 CPU 时训练让路
    #   2. 只在非交易时段（22:00-08:00），盘中绝不训练
    #   3. 超龄才跑（LIVE_MAX_AGE 天），不是每次都训
    if [ "$ENABLE_RETRAIN" = "1" ]; then
        (
            sleep 180
            while true; do
                nice -n 19 python -c "
from datetime import datetime
from qcore import live_model as LM
now = datetime.now()
hm = now.hour * 60 + now.minute
in_window = (hm >= 22 * 60) or (hm < 8 * 60)
a = LM.model_age_days()
if a is None:
    print('[retrain] 还没有模型')
elif not LM.should_retrain(${LIVE_MAX_AGE}):
    print(f'[retrain] 模型新鲜（{a:.1f} 天），跳过')
elif not in_window:
    print(f'[retrain] 模型已 {a:.1f} 天，等 22:00 后再训（盘中不训）')
else:
    print(f'[retrain] 模型已 {a:.1f} 天，开始重训（低优先级）')
    r = LM.train(verbose=True)
    print('[retrain] 完成' if r.get('ok') else '[retrain] 失败')
" 2>&1 | tail -4 || true
                sleep 3600
            done
        ) &
        PIDS="$PIDS $!"
    fi
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

# 5.5) 实时政策/事件快讯（10 分钟一刷，交易时段）
if [ "$ENABLE_NEWS" != "0" ]; then
    (
        sleep 50
        while true; do
            python -c "
from qcore import news_live as NL
r = NL.refresh(verbose=True)
if r.get('ok'):
    NL.merge_into_events(verbose=False)
" 2>&1 | tail -2 || true
            sleep 600
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.6) 事件学习（每小时：登记新事件 + 到期对账 + 重算命中率）
if [ "$ENABLE_EVENT_LEARN" != "0" ]; then
    (
        sleep 70
        while true; do
            python -c "
from qcore import event_learning as EL
r = EL.run(verbose=True)
" 2>&1 | tail -4 || true
            sleep 3600
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.7) 当日风口快照（交易时段每 30 分钟存一次，收盘后也存一份）
if [ "$ENABLE_HOTSPOT" != "0" ]; then
    (
        sleep 90
        while true; do
            python -c "
from qcore import hotspot as HS
HS.snapshot(verbose=True)
" 2>&1 | tail -3 || true
            sleep 1800
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.8) 模拟盘自动调仓（每 2 小时一次，按最新 picks）
if [ "$ENABLE_PAPER_AUTO" != "0" ]; then
    (
        sleep 120
        while true; do
            python -c "
from qcore import paper_auto as PA
PA.rebalance(verbose=True)
" 2>&1 | tail -3 || true
            sleep 7200
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.9) 实时特征提取（收盘后把当天 tick 压成日频特征）
if [ "$ENABLE_LIVE_FEAT" != "0" ]; then
    (
        sleep 150
        while true; do
            python -c "
from qcore import live_features as LF
LF.extract_all(verbose=True)
" 2>&1 | tail -3 || true
            sleep 3600
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.10) 周报（每天算一次，展示最近 7 天；周一看上周）
if [ "$ENABLE_WEEKLY" != "0" ]; then
    (
        sleep 200
        while true; do
            python -c "
from qcore import weekly as WK
WK.build(days=7, verbose=True)
" 2>&1 | tail -12 || true
            sleep 21600
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
