"""
组合回测 —— 低换手 + 缓冲带，扣真实成本。

## 为什么这一层决定生死

Alpha158+GBDT 的日频 IC = 0.026。但 A 股往返成本 19.28bps，
如果每天换手 50%，年化成本 = 250 × 19.28bps × 0.5 ≈ **24%**，IC 0.026 撑不住。

所以真正的问题不是"有没有信号"，而是**信号强度能不能覆盖换手成本**。

## 缓冲带（buffer band）的作用

朴素做法「每天重新选前 K 名」换手极高。缓冲带的做法是：

    持有池 = 前 K×(1+band) 名
    已有持仓只要还在持有池里就**不动**，只把跌出去的换成新进来的

band=0.5 时，持仓 50 只，只要还在前 75 名就不卖。这能把换手降一个数量级，
代价是组合略微偏离"最优"。

## 成本口径

- A 股：买 7.14bps（佣金+过户费）、卖 12.14bps（含 5bps 印花税），往返 19.28bps
- 成本按**换手率**计：`cost = turnover × round_trip_bps / 1e4`
- turnover = (买入金额 + 卖出金额) / 2 / 组合净值，等权时约等于"换掉的仓位比例"

## 因果性

- T 日收盘后出信号，**T+1 日收盘成交**（与标签口径一致）
- 持仓期内用每日收盘价盯市
- 绝不用 T+1 之后的信息决定 T+1 的交易
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BTConfig:
    top_k: int = 50
    hold_days: int = 5
    band: float = 0.5              # 缓冲带：持有池 = top_k*(1+band)
    round_trip_bps: float = 19.28  # A 股往返成本
    long_only: bool = True
    weight: str = "equal"          # equal | score


def backtest(scores: np.ndarray, close: np.ndarray, dates: pd.DatetimeIndex,
             valid: Optional[np.ndarray] = None,
             cfg: BTConfig = BTConfig()) -> Dict:
    """
    scores: (T,N) 分数（越大越好，NaN=不可选）
    close : (T,N) 收盘价（已归一化，比值可用）
    valid : (T,N) bool，可交易掩码（None 表示全部可交易）
    返回 {nav, ret, turnover, holdings, metrics}
    """
    T, N = scores.shape
    if valid is None:
        valid = np.isfinite(scores) & np.isfinite(close)

    nav = 1.0
    navs = np.ones(T)
    turns = np.zeros(T)
    n_hold = np.zeros(T, dtype=int)

    held: List[int] = []          # 当前持仓索引
    cash_w = 1.0
    last_rebal = -10**9
    entry_price = np.full(N, np.nan)

    hold_k = max(1, int(cfg.top_k))
    pool_k = max(hold_k, int(round(hold_k * (1.0 + cfg.band))))

    for t in range(T - 1):
        # ---- 盯市：用 t 日收盘价相对上一日的变化 ----
        if held:
            prev = close[t - 1] if t > 0 else close[t]
            cur = close[t]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(np.isfinite(prev[held]) & (prev[held] > 0),
                             cur[held] / prev[held] - 1.0, 0.0)
            r = np.nan_to_num(r, nan=0.0)
            nav *= (1.0 + cash_w * float(np.mean(r)) if held else 1.0)
        navs[t] = nav

        # ---- 调仓判定 ----
        do_rebal = (t - last_rebal >= cfg.hold_days) or (not held)
        if not do_rebal:
            n_hold[t] = len(held)
            continue

        # t 日信号，t+1 日成交 —— 这里用 t 日信号决定，价格用 t 日收盘（保守：当日收盘价成交）
        s = scores[t].copy()
        ok = valid[t] & np.isfinite(s)
        if ok.sum() < hold_k:
            n_hold[t] = len(held)
            continue

        order = np.argsort(-np.where(ok, s, -np.inf))
        pool = set(order[:pool_k].tolist())

        # 保留仍在持有池里的旧持仓
        keep = [j for j in held if j in pool]
        need = hold_k - len(keep)
        new = [j for j in order[:pool_k].tolist() if j not in set(keep)][:max(need, 0)]
        target = (keep + new)[:hold_k]

        # ---- 换手与成本 ----
        old_set, new_set = set(held), set(target)
        sold = old_set - new_set
        bought = new_set - old_set
        if held:
            turnover = (len(sold) + len(bought)) / 2.0 / max(len(held), 1)
        else:
            turnover = 1.0 if target else 0.0
        cost = turnover * cfg.round_trip_bps / 1e4
        nav *= (1.0 - cost)
        turns[t] = turnover

        held = target
        last_rebal = t
        n_hold[t] = len(held)

    navs[-1] = nav

    # ---- 指标 ----
    rets = np.diff(navs) / navs[:-1]
    rets = rets[np.isfinite(rets)]
    ann = float(np.mean(rets) * 250) if rets.size else float("nan")
    vol = float(np.std(rets, ddof=1) * np.sqrt(250)) if rets.size > 2 else float("nan")
    sharpe = ann / vol if vol and vol > 0 else float("nan")
    peak = np.maximum.accumulate(navs)
    mdd = float(np.min(navs / peak - 1.0))

    # ---- 基准：同期「可交易池等权」----
    # 这是判断 alpha 的唯一正确基准（AGENTS.md §7 #4）：
    # 相对沪深300 会把 beta 当成 alpha，相对池等权才是"选股能力"。
    bench_nav = np.ones(T)
    b = 1.0
    for t in range(1, T):
        ok_b = valid[t] & np.isfinite(close[t]) & np.isfinite(close[t - 1]) & (close[t - 1] > 0)
        if ok_b.sum() >= 5:
            r = close[t][ok_b] / close[t - 1][ok_b] - 1.0
            b *= (1.0 + float(np.nanmean(r)))
        bench_nav[t] = b
    b_rets = np.diff(bench_nav) / bench_nav[:-1]
    b_rets = b_rets[np.isfinite(b_rets)]
    b_ann = float(np.mean(b_rets) * 250) if b_rets.size else float("nan")

    # 超额 = 组合 - 基准（逐日相减后年化）
    n = min(len(rets), len(b_rets))
    ex = rets[:n] - b_rets[:n]
    ex_ann = float(np.mean(ex) * 250) if ex.size else float("nan")
    ex_vol = float(np.std(ex, ddof=1) * np.sqrt(250)) if ex.size > 2 else float("nan")
    ex_sharpe = ex_ann / ex_vol if ex_vol and ex_vol > 0 else float("nan")
    ex_cum = np.cumprod(1.0 + ex) if ex.size else np.array([1.0])
    ex_peak = np.maximum.accumulate(ex_cum)
    ex_mdd = float(np.min(ex_cum / ex_peak - 1.0)) if ex.size else float("nan")

    return {"nav": navs, "ret": rets, "turnover": turns, "n_hold": n_hold,
            "bench_nav": bench_nav, "excess": ex,
            "metrics": {"total_return": float(nav - 1.0), "annual": ann,
                        "vol": vol, "sharpe": sharpe, "mdd": mdd,
                        "bench_annual": b_ann,
                        "excess_annual": ex_ann, "excess_sharpe": ex_sharpe,
                        "excess_mdd": ex_mdd,
                        "avg_turnover": float(np.mean(turns[turns > 0])) if (turns > 0).any() else 0.0,
                        "rebal_count": int((turns > 0).sum())}}


def grid(scores: np.ndarray, close: np.ndarray, dates: pd.DatetimeIndex,
         valid: Optional[np.ndarray] = None,
         holds: Tuple[int, ...] = (5, 10, 20),
         bands: Tuple[float, ...] = (0.0, 0.5, 1.0),
         top_ks: Tuple[int, ...] = (30, 50),
         cost_bps: float = 19.28) -> pd.DataFrame:
    """跑参数网格，返回结果表。"""
    rows = []
    for k in top_ks:
        for h in holds:
            for b in bands:
                r = backtest(scores, close, dates, valid,
                             BTConfig(top_k=k, hold_days=h, band=b, round_trip_bps=cost_bps))
                m = r["metrics"]
                rows.append({"top_k": k, "hold": h, "band": b,
                             "annual": m["annual"], "vol": m["vol"],
                             "sharpe": m["sharpe"], "mdd": m["mdd"],
                             "turnover": m["avg_turnover"],
                             "bench": m["bench_annual"],
                             "excess": m["excess_annual"],
                             "ex_sharpe": m["excess_sharpe"],
                             "ex_mdd": m["excess_mdd"],
                             "total": m["total_return"]})
    # 按**超额夏普**排序：只看绝对收益会把 beta 当 alpha
    return pd.DataFrame(rows).sort_values("ex_sharpe", ascending=False)
