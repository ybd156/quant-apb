"""
持续运行器 —— 为 N100 小主机 / 飞牛 NAS 设计的"每天跑一次"入口。

两条路径：
  ① `python -m qcore.live --rebuild`  首次/每月全量重建（慢，几十秒）
  ② `python -m qcore.live --once`     每天增量（快，秒级），断电/停机后自动补跑

增量为什么能和批量对上：
    权重状态存在 `data/state/weighter.npz`，每天只处理"上次处理日之后"的新交易日；
    信号在**完整历史**上计算（不是截断后算），所以同一天的信号值在增量和批量里逐位相同。

运行顺序（严格对应 SPEC §0.0）：
    读状态 → 拉新行情 → 算当天分数（先封存）→ 用已到期样本对账 → 更新权重 → 落盘
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import get_pool, load_index
from .dataset import build_dataset
from .engine import RunConfig, run_walkforward
from .learner import WeightConfig
from . import store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(LOG_DIR, "live.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _family_map() -> Dict[str, str]:
    from .signals import signal_zoo
    return {s.sid: s.family for s in signal_zoo()}


def _latest_closed_trading_day() -> pd.Timestamp:
    """最近一个**已收盘**的交易日 —— 由墙钟 + 交易日历决定。

    旧实现取 `load_index("000300")` 的最后一根，而这份缓存有 5 天 mtime 容忍度。
    实测事故：缓存最后一根是 2026-11-06，于是 `as_of=11-06`，
    而 `last_processed` 也是 11-06 → `last >= as_of` 成立 → 返回
    `{"ok": True, "skipped": True}`。**系统因此看不见自己落后一整个交易日**：
    判定基准与被检对象同源，永远自洽。
    """
    import datetime as _dt
    now = _dt.datetime.now()
    try:
        from . import calendar_cn as CAL
        today = now.date()
        after_close = (now.hour, now.minute) >= (15, 5)
        d = today if (after_close and CAL.is_trading_day_bool(today)) else CAL.prev_trading_day(today)
        return pd.Timestamp(d)
    except Exception:  # noqa: BLE001
        return pd.Timestamp(load_index("000300")["date"].max())


def _latest_index_date() -> pd.Timestamp:
    """兼容旧名（内部已改为按交易日历取）。"""
    return _latest_closed_trading_day()


# ------------------------------------------------------------------ 全量重建
def rebuild(pool_n: int = 300, start: str = "2013-01-01", horizon: int = 5,
            market: str = "CN") -> Dict:
    t0 = time.time()
    pool = get_pool(top_n=pool_n)
    codes = pool["code"].tolist()
    _log(f"rebuild start: {len(codes)} stocks, start={start}, h={horizon}")

    p3d = build_dataset(codes, start=start, horizon=horizon,
                        extra_horizons=[1, 20], verbose=False)
    cfg = RunConfig(horizon=horizon, warmup=120, record_ledger=True)
    res = run_walkforward(p3d, cfg)

    store.save_weighter(res["weighter"], market=market, horizon=horizon)
    fam = _family_map()
    store.export_ui_bundle(res, p3d, family_of=fam)

    led = res.get("ledger")
    if led is not None and len(led):
        recent = led[led["as_of"] >= p3d.dates[-1] - np.timedelta64(90, "D")]
        store.append_ledger(recent.to_dict("records"), horizon=horizon, market=market)

    st = store.read_json("status", default={}) or {}
    st["last_processed"] = str(pd.Timestamp(p3d.dates[-1]).date())
    st["rebuilt_at"] = datetime.now().isoformat(timespec="seconds")
    store.write_json("status", st)

    dt = time.time() - t0
    _log(f"rebuild done in {dt:.1f}s: T={p3d.shape[0]} N={p3d.shape[1]} S={p3d.shape[2]}")
    return {"ok": True, "seconds": round(dt, 1), "shape": p3d.shape,
            "last_processed": st["last_processed"]}


# ------------------------------------------------------------------ 每日增量
def step_once(as_of: Optional[str] = None, horizon: int = 5, pool_n: int = 300,
              window_days: int = 900, max_catchup: int = 40) -> Dict:
    t0 = time.time()
    st = store.read_json("status", default={}) or {}
    last = st.get("last_processed")
    if last is None:
        _log("no state found -> run --rebuild first")
        return {"ok": False, "reason": "not_initialized", "hint": "python -m qcore.live --rebuild"}

    as_of_ts = pd.Timestamp(as_of) if as_of else _latest_closed_trading_day()
    if pd.Timestamp(last) >= as_of_ts:
        # 「不用再往前跑」≠「已经是最新」。状态日落后于最近已收盘交易日时必须报错，
        # 否则 NAS 停一天、一周，都只会得到一句 ok:True/skipped:True。
        ref = _latest_closed_trading_day()
        if pd.Timestamp(last) < ref:
            gap = int(np.busday_count(pd.Timestamp(last).date(), ref.date()))
            _log(f"stale state: last_processed={last} < latest closed {ref.date()} "
                 f"(~{gap} trading days behind)")
            return {"ok": False, "reason": "stale_state", "last_processed": last,
                    "expected": str(ref.date()), "missing_trading_days": gap,
                    "hint": "数据/状态落后于最近已收盘交易日；先补齐行情再跑 --once"}
        _log(f"already up to date ({last})")
        return {"ok": True, "skipped": True, "last_processed": last}

    start = (as_of_ts - timedelta(days=window_days)).strftime("%Y-%m-%d")
    pool = get_pool(top_n=pool_n)
    codes = pool["code"].tolist()
    _log(f"incremental: {last} -> {as_of_ts.date()}, {len(codes)} stocks")

    # 盘中（含午休）：取一次实时快照，把「今天这根未收盘 bar」补成当前价。
    # 这样信号反映的是**现在**，而不是昨天收盘。
    # 非交易时段不取（收盘价已定，用 K 线即可），也避免无谓的网络请求。
    _live = None
    _live_meta = None
    try:
        from . import session as _S
        _phase = _S.in_window()
        if _phase in ("morning", "lunch", "afternoon"):
            from .paper_auto import _live_quotes
            _live = _live_quotes(codes)
            if _live:
                _live_meta = {"phase": _phase, "n": len(_live),
                              "as_of": next(iter(_live.values())).get("as_of")}
                _log(f"live quotes: {len(_live)}/{len(codes)} @ {_live_meta['as_of']}")
    except Exception as e:  # noqa: BLE001
        _log(f"live quotes unavailable: {type(e).__name__}")

    p3d = build_dataset(codes, start=start, horizon=horizon, cutoff=str(as_of_ts.date()),
                        verbose=False, live_quotes=_live)
    dates = pd.DatetimeIndex(p3d.dates)

    # 本地索引：从"上次处理日的下一交易日"开始
    t_from = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(last)), side="right"))
    t_from = max(t_from, 0)
    if len(dates) - t_from > max_catchup:
        # 旧实现：只打一行日志，仍返回 ok:True，并把 last_processed 推到最新日
        # → 中间那些交易日**永不入账**，且事后看起来是连续的。宁可报错也不静默丢样本。
        _log(f"catch-up capped: {len(dates) - t_from} days > {max_catchup} "
             f"-> refuse (run --rebuild)")
        return {"ok": False, "reason": "catchup_exceeded",
                "pending_days": int(len(dates) - t_from), "max_catchup": int(max_catchup),
                "last_processed": last,
                "hint": f"积压 {len(dates) - t_from} 个交易日 > 上限 {max_catchup}，"
                        f"请跑 python -m qcore.live --rebuild（否则中间样本会被静默丢弃）"}
    if t_from >= len(dates):
        return {"ok": True, "skipped": True, "last_processed": last}

    w = store.load_weighter(horizon=horizon)
    if w is None:
        _log("weighter state missing -> rebuild required")
        return {"ok": False, "reason": "no_weighter"}

    cfg = RunConfig(horizon=horizon, warmup=0, start_t=t_from, from_t=t_from,
                    record_ledger=True)
    res = run_walkforward(p3d, cfg, weighter=w)
    store.save_weighter(w, horizon=horizon)

    # 台账：只追加新产生的预测
    led = res.get("ledger")
    new_rows = []
    if led is not None and len(led):
        new_rows = led[led["as_of"] > pd.Timestamp(last)].to_dict("records")
        if new_rows:
            store.append_ledger(new_rows, horizon=horizon)

    # UI 产物（用最近 60 天台账 + 当前权重）
    fam = _family_map()
    store.export_ui_bundle(res, p3d, family_of=fam)

    st = store.read_json("status", default={}) or {}
    # `last_processed` 必须是「**已封存**到的最后一天」，不是「数据到哪天」。
    # 台账按 INV-2 只封存能对账的决策（尾部 lag+h 天不能入账），
    # 旧代码却把游标推到 dates[-1] → 台账最后一根 2026-10-29 而状态写 11-06，
    # 缺口 6 天 = lag+h，且下一次增量从 11-07 起跑，那 6 天**永久丢失**。
    _sealed = max(len(dates) - 1 - int(cfg.exec_lag) - int(horizon), -1)
    _sealed_day = str(dates[_sealed].date()) if _sealed >= 0 else None
    st["last_processed"] = _sealed_day or str(dates[-1].date())
    st["data_through"] = str(dates[-1].date())
    st["sealed_through"] = _sealed_day
    st["pending_tail_days"] = int(int(cfg.exec_lag) + int(horizon))
    # 如实记录「这次打分用了实时价」—— 盘中分数因此不可复现，
    # 必须让人能看出来，而不是事后猜。
    if _live_meta:
        st["used_live_quotes"] = True
        st["live_quotes"] = _live_meta
        st["score_basis"] = "live_partial_bar"
    else:
        st["used_live_quotes"] = False
        st["score_basis"] = "closed_bar"
    st["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    st["new_predictions"] = len(new_rows)
    store.write_json("status", st)

    dt = time.time() - t0
    _log(f"incremental done in {dt:.1f}s: +{len(dates) - t_from} days, +{len(new_rows)} predictions")
    return {"ok": True, "seconds": round(dt, 1), "new_days": int(len(dates) - t_from),
            "new_predictions": len(new_rows), "last_processed": st["last_processed"]}


def loop(interval_sec: int = 3600) -> None:
    """常驻模式：每小时检查一次是否收盘/有新数据。"""
    _log(f"loop started, interval={interval_sec}s")
    while True:
        try:
            step_once()
        except Exception as e:  # noqa: BLE001
            _log(f"ERROR: {type(e).__name__}: {e}")
        time.sleep(interval_sec)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="APB 持续运行器")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rebuild", action="store_true", help="全量重建（首次/月度）")
    g.add_argument("--once", action="store_true", help="增量跑一次")
    g.add_argument("--loop", action="store_true", help="常驻循环")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--interval", type=int, default=3600)
    a = ap.parse_args(argv)

    if a.rebuild:
        print(json.dumps(rebuild(a.pool, a.start, a.horizon), ensure_ascii=False))
    elif a.once:
        print(json.dumps(step_once(a.as_of, a.horizon, a.pool), ensure_ascii=False))
    else:
        loop(a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
