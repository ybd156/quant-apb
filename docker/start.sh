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
#   3. 版本检查   qcore.update_check—— 1 小时，写 update.json（用户点检查时另有 force 实时查）
#   4. 收盘演算   qcore.live_model  —— 每日 15:10 打分+盯市；每周重训
#   5. 数据源探活 qcore.health      —— 6 小时，写 source_health.json
#   6. 告警检查   qcore.alert       —— 30 分钟，写 alerts.jsonl
set -e

# 0) 从快照恢复缺失的环境变量。
#    fnOS 应用管理器重建容器时会覆盖手工设的 -e 参数 —— 实测 APB_PASSWORD
#    消失后服务变成完全公开。data/ 是挂载卷，快照在那里，启动时读回来即可自愈。
if [ -f /app/data/state/env_snapshot.json ] && [ -f /app/scripts/restore_env.py ]; then
    # 恢复结果打到 stderr（docker logs 可见），不再 2>/dev/null 藏起来
    eval "$(python /app/scripts/restore_env.py)" || true
    # 启动成功后把当前生效环境回写快照：否则用户改了密码、快照还是旧的，
    # 下次重建丢变量就会恢复出**已退役的旧密码**。
    ( sleep 45; python /app/scripts/restore_env.py --refresh || true ) &
    PIDS="$PIDS $!"
fi

HOST="${APB_HOST:-0.0.0.0}"
PORT="${APB_PORT:-8765}"
MARKET_INTERVAL="${APB_MARKET_INTERVAL:-60}"
ENABLE_MARKET="${APB_MARKET:-1}"
UPDATE_INTERVAL="${APB_UPDATE_INTERVAL:-3600}"
ENABLE_UPDATE_CHECK="${APB_UPDATE_CHECK:-1}"
ENABLE_TICK="${APB_TICK:-1}"
TICK_INTERVAL="${APB_TICK_INTERVAL:-3}"
ENABLE_LIVE="${APB_LIVE:-1}"
LIVE_MAX_AGE="${APB_LIVE_MAX_AGE_DAYS:-7}"
# NAS 上训练默认关闭（训练机负责）；需要兜底时设 APB_RETRAIN=1
ENABLE_RETRAIN="${APB_RETRAIN:-0}"
ENABLE_DATA_SYNC="${APB_DATA_SYNC:-1}"
ENABLE_DAILY_BAR="${APB_DAILY_BAR:-1}"
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

# ---- 主流 ETF ----
# ETF 免印花税、免过户费，往返成本约 6.08bps vs 股票 11.28bps。
# 分钟级交易能不能成立，成本是关键变量 —— 所以 ETF 必须纳入监控。
# 选的都是日成交额十亿以上的，流动性足够。
codes += ['510300','510500','588000','159915','512880',
          '512480','515030','512690','513100','518880',
          '159949','512170','515790','588080','159920']

# ---- 港美股（腾讯接口支持 hk/us 前缀）----
codes += ['00700.HK','09988.HK','03690.HK',
          'AAPL','MSFT','NVDA','TSLA','SPY','QQQ']

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
                # 自举：遍历所有周期，缺模型/超龄的都补上。
            # 注意：默认只在 APB_RETRAIN=1 时才启用（见上方说明）。
            # 常规做法是在训练机跑 tools/train_one_click.py（一次训全部周期后上传），
            # NAS 只在训练机长期不在线时才兜底。
            nice -n 19 python -c "
from datetime import datetime
from qcore import model_bootstrap as MB
now = datetime.now()
hm = now.hour * 60 + now.minute
in_window = (hm >= 22 * 60) or (hm < 8 * 60)
todo = [x for x in MB.status() if (not x['has_model']) or x['stale']]
if not todo:
    print('[bootstrap] 所有周期模型都齐且新鲜')
elif not in_window:
    miss = [x['horizon'] for x in todo]
    print(f'[bootstrap] 待补 {miss}，等 22:00 后再训（盘中不训）')
else:
    print(f'[bootstrap] 开始补齐 {[x[chr(104)+chr(111)+chr(114)+chr(105)+chr(122)+chr(111)+chr(110)] for x in todo]}')
    r = MB.bootstrap(only_missing=False)
    print(f'[bootstrap] 完成 {r.get(chr(111)+chr(107)+chr(95)+chr(99)+chr(111)+chr(117)+chr(110)+chr(116))}/{len(todo)}')
" 2>&1 | tail -6 || true
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

# 5.11) 盘中在线训练（每分钟级：T 预测 T+3，3 分钟后对账）
# 只在交易时段跑，每 5 分钟一轮。用逻辑回归在线 SGD（8 毫秒级），
# 不是 GBDT —— 分钟级信号极弱，复杂模型必然过拟合。
if [ "$ENABLE_INTRADAY_TRAIN" != "0" ]; then
    (
        sleep 240
        while true; do
            python -c "
from datetime import datetime
now = datetime.now()
hm = now.hour * 60 + now.minute
trading = (9*60+35 <= hm <= 11*60+30) or (13*60+5 <= hm <= 14*60+55)
if now.weekday() >= 5:
    print('[intraday] 周末，跳过')
elif not trading:
    print('[intraday] 非交易时段，跳过')
else:
    from qcore import intraday_online as IO
    IO.run_all_markets(verbose=True)
" 2>&1 | tail -6 || true
            sleep 300
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.12) 不变量检查（每 6 小时，结果供「风险体检」页展示）
# 之前只靠手动跑，导致页面上 8 项全是「未提供」。
if [ "$ENABLE_INVARIANTS" != "0" ]; then
    (
        sleep 300
        while true; do
            nice -n 19 python -m qcore.invariants > /app/logs/invariants.log 2>&1 || true
            tail -3 /app/logs/invariants.log 2>/dev/null || true
            sleep 21600
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.13) 在线学习闭环（每交易日收盘后一次）
#   1) 对账：把到期的预测算实际收益
#   2) 校准：用已到期样本更新 a/b（预测->实际->调整）
# 这是「自学习」真正在跑的地方 —— 之前只有定期重训，没有在线调整。
if [ "$ENABLE_ONLINE_LEARN" != "0" ]; then
    (
        sleep 300
        while true; do
            python -c "
from datetime import datetime
now = datetime.now()
hm = now.hour * 60 + now.minute
# 收盘后 15:10-17:00 之间跑一次；周末不跑
if now.weekday() >= 5:
    print('[online] 周末跳过')
elif not (15*60+10 <= hm <= 17*60):
    print('[online] 非收盘时段，跳过')
else:
    from qcore import horizon_ledger as HL
    from qcore import online_calib as OC
    r = HL.reconcile(verbose=False)
    print('[online] 对账', r.get('done', 0), '条')
    OC.update_all(verbose=True)
" 2>&1 | tail -8 || true
            sleep 3600
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.14) 启动时快照 HostConfig（组权限/网络/挂载），供重建时对照
(
    sleep 20
    python -c "
try:
    from qcore import docker_api as D, env_guard as EG
    info = D.inspect('apb')
    if info:
        hc = EG.capture_hostconfig(info)
        print('[hostconfig] 快照:', {k: (v if not isinstance(v, list) else len(v)) for k, v in hc.items()})
except Exception as e:
    print('[hostconfig] 跳过:', type(e).__name__)
" 2>&1 | tail -2 || true
) &
PIDS="$PIDS $!"

# 5.15) 每日行情同步（否则数据永远停在装机那天）
#   之前完全没有这个任务 —— 实测 NAS 数据停在 2026-11-06，
#   而远端已经到 11-08，预测/台账/模拟盘全在用旧数据。
#   APB_DATA_NO_SELF=1：不从自己的 /api/data/qlib 下（那是本地旧数据的打包），
#   直接走 GitHub。
if [ "$ENABLE_DATA_SYNC" != "0" ]; then
    (
        sleep 420
        while true; do
            python -c "
from datetime import datetime
now = datetime.now()
hm = now.hour * 60 + now.minute
# 每天 08:00-09:00 之间同步一次（开盘前拿到前一日数据）
if now.weekday() >= 5:
    print('[data] 周末跳过')
elif not (8*60 <= hm <= 9*60):
    print('[data] 非同步时段')
else:
    import os as _os
    _os.environ["APB_DATA_NO_SELF"] = "1"
    from qcore import data_sync as DS
    r = DS.sync(verbose=True)
    print('[data] ok=', r.get('ok'), 'local=', r.get('new_local_date') or r.get('local_date'),
          'remote=', r.get('remote_tag'), r.get('reason') or '')
" 2>&1 | tail -6 || true
            sleep 1800
        done
    ) &
    PIDS="$PIDS $!"
fi

# 5.16) 收盘后自行补齐当日 K 线
#   上游数据包每天只发布一次（通常凌晨），实测晚上 19:47 本地还停在上一交易日。
#   这里收盘后（15:05-17:00）自己从东财抓当日数据，以「相对前收的比值」
#   写进 data/qlib/overlay/，qlib_data 读取时追加，保证收益序列连续。
if [ "$ENABLE_DAILY_BAR" != "0" ]; then
    (
        sleep 480
        while true; do
            python -c "
from datetime import datetime
now = datetime.now()
hm = now.hour * 60 + now.minute
if now.weekday() >= 5:
    print('[daily] 周末跳过')
elif not (15*60+5 <= hm <= 17*60):
    print('[daily] 非收盘时段，跳过')
else:
    from qcore import daily_append as DA
    r = DA.append_day(verbose=True)
    print('[daily] ok=', r.get('ok'), 'n=', r.get('n'), r.get('reason') or '')
" 2>&1 | tail -5 || true
            sleep 1800
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
