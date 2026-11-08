"""
模拟盘 vs 实盘 —— "模型说的"和"实际做的"差多少。

为什么必须做这件事：
    回测里所有成交都假设能按 T+1 开盘价全额成交。真实世界有滑点、有犹豫、
    有"看到信号没下手"、有"手痒多买一只"。如果模拟盘和实盘的净值曲线越走越开，
    说明 alpha 在"执行"这一层就漏掉了 —— 这比回测数字本身更重要。

口径（严格无前视）：
    - 决策在 t 日收盘后，成交价只能是 OPEN[t+1]；
    - 非重叠调仓（每 hold 天一次），避免重复计成本；
    - 实盘按用户录入的真实成交价与数量计算，不做任何"美化"。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_TRADES = os.path.join(ROOT, "data", "state", "live_trades.jsonl")


@dataclass
class PaperConfig:
    top_k: int = 10
    hold: int = 5
    cost_bps: float = 10.0
    initial: float = 1_000_000.0
    long_only: bool = True


# ------------------------------------------------------------------ 模拟盘
def simulate(scores: np.ndarray, OPEN: np.ndarray, CLOSE: np.ndarray,
             VALID: np.ndarray, dates: np.ndarray, cfg: PaperConfig = PaperConfig(),
             start_t: int = 120) -> Dict:
    """
    scores (T,N)；OPEN/CLOSE (T,N)；VALID (T,N)。
    每 hold 天调仓一次：t 日收盘选分最高的 top_k → OPEN[t+1] 建仓 → 持有 hold 天 → OPEN[t+1+hold] 平仓。
    """
    T, N = scores.shape
    h = cfg.hold
    cost = cfg.cost_bps / 1e4
    nav = np.full(T, np.nan, dtype=np.float64)
    trades: List[dict] = []
    equity = cfg.initial
    nav[start_t] = equity

    t = start_t
    while t + 1 + h < T:
        sc = scores[t]
        ok = np.isfinite(sc) & VALID[t]
        if ok.sum() < cfg.top_k:
            t += h
            continue
        idx = np.argsort(-np.where(ok, sc, -np.inf))[:cfg.top_k]
        entry, exit_ = t + 1, t + 1 + h
        px_in = OPEN[entry, idx]
        px_out = OPEN[exit_, idx]
        good = np.isfinite(px_in) & np.isfinite(px_out)
        if good.sum() == 0:
            t += h
            continue
        idx, px_in, px_out = idx[good], px_in[good], px_out[good]
        alloc = equity * (1 - cost) / len(idx)
        shares = alloc / px_in
        gross = float(np.sum(shares * px_out))
        equity_new = gross * (1 - cost)

        for d in range(entry, exit_):
            nav[d] = float(np.sum(shares * CLOSE[d, idx]))
        nav[exit_] = equity_new

        for j, n in enumerate(idx):
            trades.append({
                "entry_date": str(pd.Timestamp(dates[entry]).date()),
                "exit_date": str(pd.Timestamp(dates[exit_]).date()),
                "code": str(n), "entry_price": float(px_in[j]),
                "exit_price": float(px_out[j]),
                "ret": float(px_out[j] / px_in[j] - 1.0),
            })
        equity = equity_new
        t = exit_

    nav = pd.Series(nav).ffill().to_numpy()
    ret = pd.Series(nav).pct_change()
    valid = nav > 0
    nav_c = nav[valid]
    mdd = float((nav_c / np.maximum.accumulate(nav_c) - 1).min()) if len(nav_c) else 0.0
    ann = float((nav_c[-1] / nav_c[0]) ** (252 / max(len(nav_c), 1)) - 1) if len(nav_c) > 1 else 0.0
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() and ret.std() > 0 else 0.0
    hit = float(np.mean([tr["ret"] > 0 for tr in trades])) if trades else float("nan")

    return {
        "nav": nav, "ret": ret.to_numpy(), "annual": ann, "sharpe": sharpe, "mdd": mdd,
        "hit_rate": hit, "n_trades": len(trades), "trades": trades,
        "final": float(nav_c[-1]) if len(nav_c) else cfg.initial,
    }


# ------------------------------------------------------------------ 实盘
def append_live_trade(date: str, code: str, side: str, price: float, qty: float,
                      note: str = "", path: str = LIVE_TRADES) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"date": str(date), "code": str(code).zfill(6), "side": side.lower(),
           "price": float(price), "qty": float(qty), "note": note}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_live_trades(path: str = LIVE_TRADES) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "code", "side", "price", "qty", "note"])
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def live_nav(trades: pd.DataFrame, CLOSE: np.ndarray, dates: np.ndarray,
             codes: List[str], initial: float = 1_000_000.0) -> Dict:
    """按真实成交记录做市值重估（收盘价 MTM）。"""
    T, N = CLOSE.shape
    cidx = {c: i for i, c in enumerate(codes)}
    nav = np.full(T, np.nan, dtype=np.float64)
    if trades is None or len(trades) == 0:
        return {"nav": nav, "ret": np.full(T, np.nan), "final": initial,
                "annual": float("nan"), "mdd": float("nan"), "n_trades": 0}

    cash = initial
    pos = np.zeros(N)
    didx = pd.DatetimeIndex(dates)
    t = 0
    trades = trades.copy()
    trades["_ti"] = didx.searchsorted(pd.DatetimeIndex(trades["date"]), side="left")
    for ti, grp in trades.groupby("_ti"):
        ti = int(min(ti, T - 1))
        while t < ti:
            nav[t] = cash + float(np.sum(pos * np.nan_to_num(CLOSE[t])))
            t += 1
        for r in grp.itertuples(index=False):
            n = cidx.get(str(r.code))
            if n is None:
                continue
            if r.side == "buy":
                cash -= r.price * r.qty
                pos[n] += r.qty
            else:
                cash += r.price * r.qty
                pos[n] -= r.qty
        nav[ti] = cash + float(np.sum(pos * np.nan_to_num(CLOSE[ti])))
        t = ti + 1
    while t < T:
        nav[t] = cash + float(np.sum(pos * np.nan_to_num(CLOSE[t])))
        t += 1

    # 只向前填充（起始未建仓段用初始资金），绝不 bfill —— bfill 会把未来净值搬到当下
    nav = pd.Series(nav).ffill().to_numpy()
    first = np.argmax(np.isfinite(nav))
    nav[:first] = initial
    ret = pd.Series(nav).pct_change()
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float((nav[-1] / nav[0]) ** (252 / max(len(nav), 1)) - 1) if nav[0] > 0 else float("nan")
    return {"nav": nav, "ret": ret.to_numpy(), "final": float(nav[-1]),
            "annual": ann, "mdd": mdd, "n_trades": int(len(trades))}


# ------------------------------------------------------------------ 对比
def compare(sim: Dict, live: Dict, live_trades: Optional[pd.DataFrame] = None,
            sim_trades: Optional[List[dict]] = None) -> Dict:
    """模拟盘 vs 实盘：净值差、命中率差、执行滑点。"""
    a, b = sim["ret"], live["ret"]
    m = np.isfinite(a) & np.isfinite(b)
    nan = float("nan")
    if m.sum() < 2:
        # 没有实盘成交（或区间不重叠）时，只报模拟盘，差异字段留空
        return {
            "tracking_error_ann": nan, "pnl_gap": nan,
            "sim_final": sim["final"], "live_final": live["final"],
            "sim_annual": sim["annual"], "live_annual": live["annual"],
            "sim_hit_rate": sim["hit_rate"],
            "live_trades": int(live.get("n_trades", 0)),
            "exec_slippage_bps": nan, "n_days_compared": int(m.sum()),
            "note": "no_live_trades",
        }
    diff = a[m] - b[m]
    te = float(np.nanstd(diff) * np.sqrt(252)) if m.sum() > 2 else nan

    sim_nav = sim["nav"][m] / (sim["nav"][m][0] or 1)
    live_nav_ = live["nav"][m] / (live["nav"][m][0] or 1)
    gap = float(sim_nav[-1] - live_nav_[-1]) if len(sim_nav) else nan

    slip = float("nan")
    if live_trades is not None and len(live_trades) and sim_trades:
        sim_px = {(t["entry_date"], t["code"]): t["entry_price"] for t in sim_trades}
        bps = []
        for r in live_trades.itertuples(index=False):
            if r.side != "buy":
                continue
            key = (str(pd.Timestamp(r.date).date()), str(r.code))
            ref = sim_px.get(key)
            if ref and ref > 0:
                bps.append((r.price / ref - 1.0) * 1e4)
        if bps:
            slip = float(np.mean(bps))

    return {
        "tracking_error_ann": te,
        "pnl_gap": gap,
        "sim_final": sim["final"], "live_final": live["final"],
        "sim_annual": sim["annual"], "live_annual": live["annual"],
        "sim_hit_rate": sim["hit_rate"],
        "live_trades": int(live["n_trades"]),
        "exec_slippage_bps": slip,
        "n_days_compared": int(m.sum()),
    }


def export_paper(sim: Dict, live: Dict, cmp: Dict, dates: np.ndarray,
                 tail: int = 250) -> Dict:
    """给 UI 的 paper.json（只保留最近 tail 天，避免 JSON 过大）。"""
    k = min(tail, len(dates))
    d = [str(pd.Timestamp(x).date()) for x in dates[-k:]]
    sn = sim["nav"][-k:]; ln = live["nav"][-k:]
    base_s = next((x for x in sn if np.isfinite(x) and x > 0), 1.0)
    base_l = next((x for x in ln if np.isfinite(x) and x > 0), 1.0)
    return {
        "dates": d,
        "sim_nav": [None if not np.isfinite(x) else round(float(x / base_s), 4) for x in sn],
        "live_nav": [None if not np.isfinite(x) else round(float(x / base_l), 4) for x in ln],
        "compare": cmp,
    }
