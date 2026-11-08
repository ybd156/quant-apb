"""
带完整交易规则的回测 —— 进场、出场、止盈、止损、加仓、减仓。

## 为什么需要这一层

`qcore/bt.py` 只做「定期按排名换仓」。真实交易还有一层**价格规则**：
亏了要不要砍？赚了要不要落袋？信号变强要不要加？

这些规则对**横截面因子策略**未必有益——甚至可能有害，因为因子策略赚的是
相对强弱，而止损砍掉的往往是「暂时落后但即将反弹」的票。
但**必须实测**，不能凭直觉。

## 规则清单

### 进场（entry）
- `min_score_pct`：分数必须在当日横截面的前 X% 才买（默认 top_k 内即可）
- `entry_confirm`：可选确认——要求分数比上一期上升（`score_rising`）

### 出场（exit）
- `out_of_pool`：跌出 `top_k×(1+band)` 名 → 卖（缓冲带）
- `max_hold`：持有超过 N 天强制卖（0 = 不限制）
- `stop_loss`：相对**买入价**亏 X% → 卖
- `take_profit`：相对**买入价**赚 X% → 卖（可只卖一半，见 `tp_partial`）
- `trailing_stop`：相对**持有期最高价**回撤 X% → 卖

### 加仓 / 减仓（scale）
- `scale_step`：每次调整的比例（默认 0.25）
- `scale_max` / `scale_min`：单票权重相对基准的上下限（默认 1.5 / 0.5）
- `add_on_rise`：排名上升且浮盈 → 加仓
- `reduce_on_fall`：排名下降或浮亏超过 `reduce_dd` → 减仓

## 成本

每次买卖都按**成交金额**计成本（19.28bps 往返 → 单边 9.64bps），
加仓/减仓的增量部分也要计成本，这一点很容易漏。

## 因果性

T 日收盘后评估规则，**T+1 日按 T 日收盘价成交**（保守：不给日内优势）。
止损/止盈的触发判定用 T 日收盘价，不用 T+1 的盘中价（避免"事后知道最低点"）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RuleConfig:
    # --- 基础 ---
    top_k: int = 30
    band: float = 1.0
    round_trip_bps: float = 19.28
    # 调仓节奏：排名类规则（进出池）只在调仓日生效；
    # 价格类规则（止损/止盈/移动止损）每日生效——这才是它们的意义。
    # 不设这个，排名规则会天天换仓，换手 100%+，成本直接吃光收益。
    rebalance_days: int = 20

    # --- 出场 ---
    max_hold: int = 0            # 0 = 不限制（靠排名出场）
    stop_loss: float = 0.0       # 0 = 不用，如 0.08 表示亏 8% 砍
    take_profit: float = 0.0     # 0 = 不用，如 0.20 表示赚 20% 卖
    tp_partial: float = 0.5      # 止盈时卖掉的比例（1.0 = 全卖）
    trailing_stop: float = 0.0   # 0 = 不用，如 0.10 表示从高点回撤 10% 卖

    # --- 加仓/减仓 ---
    scale_step: float = 0.0      # 0 = 不做仓位调整
    scale_max: float = 1.5
    scale_min: float = 0.5
    add_on_rise: bool = True     # 排名上升 + 浮盈 → 加
    reduce_dd: float = 0.10      # 浮亏超过 10% → 减

    # --- 进场 ---
    entry_confirm: bool = False  # 要求分数较上期上升才买


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """横截面百分位（1.0 = 最高）。NaN 保持 NaN。"""
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out
    out[m] = pd.Series(x[m]).rank(pct=True).to_numpy()
    return out


def backtest_rules(scores: np.ndarray, close: np.ndarray, dates: pd.DatetimeIndex,
                   valid: Optional[np.ndarray] = None,
                   cfg: RuleConfig = RuleConfig()) -> Dict:
    """
    逐日状态机。返回净值、交易记录、指标。
    scores: (T,N)；close: (T,N)。
    """
    T, N = scores.shape
    if valid is None:
        valid = np.isfinite(scores) & np.isfinite(close)

    side_cost = cfg.round_trip_bps / 2.0 / 1e4      # 单边成本
    hold_k = max(1, int(cfg.top_k))
    pool_k = max(hold_k, int(round(hold_k * (1.0 + cfg.band))))
    base_w = 1.0 / hold_k

    # 持仓状态
    pos_w = np.zeros(N)          # 当前权重
    entry_px = np.full(N, np.nan)
    peak_px = np.full(N, np.nan)
    held_days = np.zeros(N, dtype=int)
    entry_rank = np.full(N, np.nan)

    nav = 1.0
    navs = np.ones(T)
    trades: List[Dict] = []
    turnovers = np.zeros(T)
    last_rebal = -10**9

    for t in range(1, T):
        px = close[t]
        px_prev = close[t - 1]
        # ---------- 1. 盯市 ----------
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where((pos_w > 0) & np.isfinite(px) & np.isfinite(px_prev) & (px_prev > 0),
                         px / px_prev - 1.0, 0.0)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        port_r = float(np.sum(pos_w * r))
        nav *= (1.0 + port_r)

        # 更新峰值与持有天数
        holding = pos_w > 0
        peak_px[holding] = np.fmax(peak_px[holding], px[holding])
        held_days[holding] += 1

        # ---------- 2. 规则评估（用 t 日收盘价）----------
        s = scores[t]
        ok = valid[t] & np.isfinite(s)
        if ok.sum() < hold_k:
            navs[t] = nav
            continue
        rank_pct = _pct_rank(np.where(ok, s, np.nan))
        order = np.argsort(-np.where(ok, s, -np.inf))
        pool = set(order[:pool_k].tolist())
        is_rebal = (t - last_rebal >= max(1, int(cfg.rebalance_days)))

        exit_now: List[Tuple[int, str, float]] = []   # (idx, reason, sell_frac)

        for j in np.where(holding)[0]:
            p = px[j]
            if not np.isfinite(p) or p <= 0:
                continue
            ep = entry_px[j]
            # 止损
            if cfg.stop_loss > 0 and np.isfinite(ep) and p / ep - 1.0 <= -cfg.stop_loss:
                exit_now.append((int(j), "stop_loss", 1.0)); continue
            # 止盈
            if cfg.take_profit > 0 and np.isfinite(ep) and p / ep - 1.0 >= cfg.take_profit:
                exit_now.append((int(j), "take_profit", float(np.clip(cfg.tp_partial, 0.01, 1.0)))); continue
            # 移动止损
            if cfg.trailing_stop > 0 and np.isfinite(peak_px[j]) and peak_px[j] > 0 \
                    and p / peak_px[j] - 1.0 <= -cfg.trailing_stop:
                exit_now.append((int(j), "trailing", 1.0)); continue
            # 超期
            if cfg.max_hold > 0 and held_days[j] >= cfg.max_hold:
                exit_now.append((int(j), "max_hold", 1.0)); continue
            # 跌出持有池（**只在调仓日判定**，否则天天换仓）
            if is_rebal and j not in pool:
                exit_now.append((int(j), "out_of_pool", 1.0)); continue

        # ---------- 3. 执行卖出 ----------
        turn = 0.0
        for j, reason, frac in exit_now:
            sold = pos_w[j] * frac
            turn += sold
            pos_w[j] -= sold
            trades.append({"t": t, "date": str(dates[t].date()), "side": "sell",
                           "idx": j, "weight": round(float(sold), 6), "reason": reason,
                           "ret": round(float(px[j] / entry_px[j] - 1.0), 6)
                           if np.isfinite(entry_px[j]) and entry_px[j] > 0 else None})
            if pos_w[j] <= 1e-9:
                pos_w[j] = 0.0
                entry_px[j] = np.nan
                peak_px[j] = np.nan
                held_days[j] = 0
                entry_rank[j] = np.nan

        # ---------- 4. 加仓 / 减仓 ----------
        if cfg.scale_step > 0 and is_rebal:
            for j in np.where(pos_w > 0)[0]:
                rp = rank_pct[j]
                if not np.isfinite(rp) or not np.isfinite(entry_rank[j]):
                    continue
                pnl = px[j] / entry_px[j] - 1.0 if np.isfinite(entry_px[j]) else 0.0
                delta = 0.0
                if cfg.add_on_rise and rp > entry_rank[j] and pnl > 0:
                    delta = +cfg.scale_step * base_w
                elif rp < entry_rank[j] - 0.2 or pnl <= -cfg.reduce_dd:
                    delta = -cfg.scale_step * base_w
                if delta != 0.0:
                    new_w = float(np.clip(pos_w[j] + delta,
                                          base_w * cfg.scale_min, base_w * cfg.scale_max))
                    delta = new_w - pos_w[j]
                    if abs(delta) > 1e-9:
                        turn += abs(delta)
                        pos_w[j] = new_w
                        trades.append({"t": t, "date": str(dates[t].date()),
                                       "side": "add" if delta > 0 else "reduce",
                                       "idx": j, "weight": round(float(abs(delta)), 6),
                                       "reason": "scale", "ret": round(float(pnl), 6)})

        # ---------- 5. 补仓（进场）----------
        need = hold_k - int((pos_w > 1e-9).sum()) if is_rebal else 0
        if need > 0:
            cand = [j for j in order[:pool_k].tolist() if pos_w[j] <= 1e-9]
            if cfg.entry_confirm and t > 0:
                sp = scores[t - 1]
                cand = [j for j in cand
                        if np.isfinite(sp[j]) and np.isfinite(s[j]) and s[j] > sp[j]]
            for j in cand[:need]:
                pos_w[j] = base_w
                entry_px[j] = px[j]
                peak_px[j] = px[j]
                held_days[j] = 0
                entry_rank[j] = rank_pct[j]
                turn += base_w
                trades.append({"t": t, "date": str(dates[t].date()), "side": "buy",
                               "idx": int(j), "weight": round(float(base_w), 6),
                               "reason": "entry", "ret": None})

        if is_rebal:
            last_rebal = t

        # ---------- 6. 成本 ----------
        if turn > 0:
            cost = turn * side_cost
            nav *= (1.0 - cost)
            turnovers[t] = turn

        navs[t] = nav

    # ---------- 指标 ----------
    rets = np.diff(navs) / navs[:-1]
    rets = rets[np.isfinite(rets)]
    ann = float(np.mean(rets) * 250) if rets.size else float("nan")
    vol = float(np.std(rets, ddof=1) * np.sqrt(250)) if rets.size > 2 else float("nan")
    sharpe = ann / vol if vol and vol > 0 else float("nan")
    peak = np.maximum.accumulate(navs)
    mdd = float(np.min(navs / peak - 1.0))

    # 基准：池等权
    bench = np.ones(T); b = 1.0
    for t in range(1, T):
        okb = valid[t] & np.isfinite(close[t]) & np.isfinite(close[t - 1]) & (close[t - 1] > 0)
        if okb.sum() >= 5:
            b *= (1.0 + float(np.nanmean(close[t][okb] / close[t - 1][okb] - 1.0)))
        bench[t] = b
    brets = np.diff(bench) / bench[:-1]
    brets = brets[np.isfinite(brets)]
    b_ann = float(np.mean(brets) * 250) if brets.size else float("nan")

    n = min(len(rets), len(brets))
    ex = rets[:n] - brets[:n]
    ex_ann = float(np.mean(ex) * 250) if ex.size else float("nan")
    ex_vol = float(np.std(ex, ddof=1) * np.sqrt(250)) if ex.size > 2 else float("nan")
    ex_sharpe = ex_ann / ex_vol if ex_vol and ex_vol > 0 else float("nan")

    return {"nav": navs, "trades": trades, "turnover": turnovers,
            "metrics": {"annual": ann, "vol": vol, "sharpe": sharpe, "mdd": mdd,
                        "bench_annual": b_ann, "excess_annual": ex_ann,
                        "excess_sharpe": ex_sharpe,
                        "total_return": float(nav - 1.0),
                        "n_trades": len(trades),
                        "avg_turnover": float(np.mean(turnovers[turnovers > 0]))
                        if (turnovers > 0).any() else 0.0}}


def grid_rules(scores: np.ndarray, close: np.ndarray, dates: pd.DatetimeIndex,
               valid: Optional[np.ndarray] = None,
               variants: Optional[List[Tuple[str, RuleConfig]]] = None,
               cost_bps: float = 19.28) -> pd.DataFrame:
    """跑若干规则变体，返回对比表。"""
    if variants is None:
        base = dict(top_k=30, band=1.0, round_trip_bps=cost_bps)
        variants = [
            ("纯排名（无价格规则）", RuleConfig(**base)),
            ("+ 止损 8%", RuleConfig(stop_loss=0.08, **base)),
            ("+ 止损 15%", RuleConfig(stop_loss=0.15, **base)),
            ("+ 止盈 20%", RuleConfig(take_profit=0.20, tp_partial=1.0, **base)),
            ("+ 止盈 20%(半仓)", RuleConfig(take_profit=0.20, tp_partial=0.5, **base)),
            ("+ 移动止损 10%", RuleConfig(trailing_stop=0.10, **base)),
            ("+ 止损8%+止盈20%", RuleConfig(stop_loss=0.08, take_profit=0.20, **base)),
            ("+ 加仓减仓", RuleConfig(scale_step=0.25, **base)),
            ("+ 进场确认", RuleConfig(entry_confirm=True, **base)),
            ("+ 限持 40 日", RuleConfig(max_hold=40, **base)),
            ("全套规则", RuleConfig(stop_loss=0.15, take_profit=0.30, trailing_stop=0.15,
                                scale_step=0.25, max_hold=60, **base)),
        ]
    rows = []
    for name, cfg in variants:
        r = backtest_rules(scores, close, dates, valid, cfg)
        m = r["metrics"]
        rows.append({"variant": name, "annual": m["annual"], "excess": m["excess_annual"],
                     "ex_sharpe": m["excess_sharpe"], "mdd": m["mdd"],
                     "turnover": m["avg_turnover"], "n_trades": m["n_trades"]})
    return pd.DataFrame(rows).sort_values("ex_sharpe", ascending=False)
