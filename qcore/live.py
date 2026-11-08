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


def _latest_index_date() -> pd.Timestamp:
    ix = load_index("000300")
    return pd.Timestamp(ix["date"].max())


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

    store.save_weighter(res["weighter"])
    fam = _family_map()
    store.export_ui_bundle(res, p3d, family_of=fam)

    led = res.get("ledger")
    if led is not None and len(led):
        recent = led[led["as_of"] >= p3d.dates[-1] - np.timedelta64(90, "D")]
        store.append_ledger(recent.to_dict("records"))

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

    as_of_ts = pd.Timestamp(as_of) if as_of else _latest_index_date()
    if pd.Timestamp(last) >= as_of_ts:
        _log(f"already up to date ({last})")
        return {"ok": True, "skipped": True, "last_processed": last}

    start = (as_of_ts - timedelta(days=window_days)).strftime("%Y-%m-%d")
    pool = get_pool(top_n=pool_n)
    codes = pool["code"].tolist()
    _log(f"incremental: {last} -> {as_of_ts.date()}, {len(codes)} stocks")

    p3d = build_dataset(codes, start=start, horizon=horizon, cutoff=str(as_of_ts.date()),
                        verbose=False)
    dates = pd.DatetimeIndex(p3d.dates)

    # 本地索引：从"上次处理日的下一交易日"开始
    t_from = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(last)), side="right"))
    t_from = max(t_from, 0)
    if len(dates) - t_from > max_catchup:
        _log(f"catch-up capped: {len(dates) - t_from} days -> {max_catchup} (consider --rebuild)")
        t_from = len(dates) - max_catchup
    if t_from >= len(dates):
        return {"ok": True, "skipped": True, "last_processed": last}

    w = store.load_weighter()
    if w is None:
        _log("weighter state missing -> rebuild required")
        return {"ok": False, "reason": "no_weighter"}

    cfg = RunConfig(horizon=horizon, warmup=0, start_t=t_from, from_t=t_from,
                    record_ledger=True)
    res = run_walkforward(p3d, cfg, weighter=w)
    store.save_weighter(w)

    # 台账：只追加新产生的预测
    led = res.get("ledger")
    new_rows = []
    if led is not None and len(led):
        new_rows = led[led["as_of"] > pd.Timestamp(last)].to_dict("records")
        if new_rows:
            store.append_ledger(new_rows)

    # UI 产物（用最近 60 天台账 + 当前权重）
    fam = _family_map()
    store.export_ui_bundle(res, p3d, family_of=fam)

    st = store.read_json("status", default={}) or {}
    st["last_processed"] = str(dates[-1].date())
    st["updated_at"] = datetime.now().isoformat(timespec="seconds")
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
