"""
组合回测层（backtest）—— 严格 T+1 开盘成交、非重叠调仓、双边成本可配。

口径（与 SPEC §0.1 / §6 一致）
-------------------------------------------------
* t 日收盘后得到 score → **t+1 开盘**建仓；持有 h 个交易日 → t+1+h 开盘平仓；
  单期收益与引擎契约 r_h(t) = open(t+1+h)/open(t+1) - 1 完全对齐。
* **非重叠调仓**：调仓日序列 t, t+h, t+2h, ...，每期只结算一次，
  因此不存在「同一段行情被重复建仓、成本被重复计」的问题（重叠建仓的
  多空年化会被虚高约 h 倍，这是最容易被忽视的骗人点）。
* 成本：cost_bps 视为**双边（买+卖）成本**，单位 bps。某只股票被整体替换
  （卖出旧仓 + 买入新仓）计一次完整往返 = cost_bps。每期成本
  = cost_bps/1e4 × (多头换手 + 空头换手)，换手 = 被替换比例 ∈ [0,1]。
* 净值：日频 mark-to-market。持仓期内用收盘价估值，期末用开盘价（与 RET
  契约一致）；调仓日同时结算「旧仓到开盘」与「新仓开盘到收盘」两段收益。

输出字典字段见 run_long_short 文档。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:  # 兼容 `python -m qcore.backtest` 与包内导入
    from .metrics import _align_scores, assign_groups
except ImportError:  # pragma: no cover
    from qcore.metrics import _align_scores, assign_groups

__all__ = ["run_long_short", "run_baselines", "compare_table"]

EPS = 1e-12
NAN = float("nan")
TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# 内部：单组合回测
# --------------------------------------------------------------------------- #
def _nan_mean_row(vals: np.ndarray, idx: np.ndarray) -> float:
    """vals[idx] 的等权平均（忽略 NaN）；全 NaN / 空返回 NaN。"""
    if idx.size == 0:
        return NAN
    v = vals[idx]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return NAN
    return float(v.mean())


def _portfolio(scores: np.ndarray, OPEN: np.ndarray, CLOSE: np.ndarray,
               h: int, n_groups: int, cost_bps: float,
               long_only: bool = False) -> dict:
    """
    核心回测：按 score 分组、t+1 开盘等权建仓、持有 h 日、非重叠调仓。

    返回 dict（含逐日因子、净值、绩效、换手、组规模）。数据不足时 ok=False。
    """
    N, T = scores.shape
    h = int(h)
    n_groups = max(1, int(n_groups))
    c = max(0.0, float(cost_bps)) / 1e4

    if T < h + 3 or N < 2:
        return dict(ok=False, reason="insufficient_data", h=h,
                    n_groups=n_groups, cost_bps=float(cost_bps))

    min_names = max(n_groups * 2, 4) if not long_only else max(n_groups, 2)
    valid_cnt = np.isfinite(scores).sum(axis=0)

    # ---- 调仓日：首个可用日 + 每 h 日一次，且必须能完整持有到 exit ----
    cand = np.flatnonzero(valid_cnt >= min_names)
    if cand.size == 0:
        return dict(ok=False, reason="no_valid_scores", h=h,
                    n_groups=n_groups, cost_bps=float(cost_bps))
    first = int(cand[0])
    last_t = T - h - 2
    if first > last_t:
        return dict(ok=False, reason="insufficient_forward_window", h=h,
                    n_groups=n_groups, cost_bps=float(cost_bps))
    rebal = list(range(first, last_t + 1, h))
    if not rebal:
        return dict(ok=False, reason="no_rebalance_date", h=h,
                    n_groups=n_groups, cost_bps=float(cost_bps))

    fac_long = np.ones(T)
    fac_short = np.ones(T)
    fac_ls = np.ones(T)
    to_long_list, to_short_list = [], []
    size_long, size_short = [], []
    prev_long: Optional[np.ndarray] = None
    prev_short: Optional[np.ndarray] = None
    cost_total = 0.0
    last_exit: Optional[int] = None

    for t in rebal:
        groups = assign_groups(scores[:, t], n_groups)
        if groups is None:
            continue
        long_idx = groups[-1]
        short_idx = np.empty(0, dtype=np.int64) if long_only else groups[0]
        if long_only and n_groups > 1:
            long_idx = np.concatenate(groups)  # 多头模式忽略分组，持全池
        if long_idx.size == 0 or (not long_only and short_idx.size == 0):
            continue

        entry = t + 1
        exit_ = t + 1 + h
        if exit_ > T - 1:
            break
        last_exit = exit_

        # ---- 换手（与上一期持仓比较）----
        def _turn(prev, cur):
            if prev is None or prev.size == 0:
                return 1.0
            return float(np.setdiff1d(cur, prev, assume_unique=False).size) / float(max(cur.size, 1))

        tl = _turn(prev_long, long_idx)
        ts = 0.0 if long_only else _turn(prev_short, short_idx)
        prev_long, prev_short = long_idx, short_idx
        to_long_list.append(tl)
        to_short_list.append(ts)
        size_long.append(int(long_idx.size))
        size_short.append(int(short_idx.size))

        # ---- 持仓期内逐日估值 ----
        base = OPEN[:, entry]
        prev_vl, prev_vs = 1.0, 1.0
        for d in range(entry, exit_ + 1):
            px = OPEN[:, d] if d == exit_ else CLOSE[:, d]
            vl = _nan_mean_row(px / base, long_idx)
            if not np.isfinite(vl):
                vl = prev_vl
            rl = vl / prev_vl - 1.0
            prev_vl = vl

            if long_only:
                fac_long[d] *= (1.0 + rl)
                fac_ls[d] *= (1.0 + rl)
                continue

            vs = _nan_mean_row(px / base, short_idx)
            if not np.isfinite(vs):
                vs = prev_vs
            rs = vs / prev_vs - 1.0
            prev_vs = vs
            fac_long[d] *= (1.0 + rl)
            fac_short[d] *= (1.0 + rs)
            fac_ls[d] *= (1.0 + rl - rs)

        # ---- 成本（建仓日一次性扣除，按替换比例）----
        tot = c * (tl + ts)
        cost_total += tot
        fac_ls[entry] *= max(0.0, 1.0 - tot)
        fac_long[entry] *= max(0.0, 1.0 - c * tl)
        if not long_only:
            fac_short[entry] *= max(0.0, 1.0 - c * ts)

    if not to_long_list:
        return dict(ok=False, reason="no_executable_period", h=h,
                    n_groups=n_groups, cost_bps=float(cost_bps))

    # ---- 净值（只覆盖真正有持仓的区间，尾部空仓日不拉平净值）----
    start = rebal[0] + 1
    end = last_exit if last_exit is not None else (T - 1)
    idx = np.arange(start, min(end, T - 1) + 1)
    eq_ls = np.cumprod(fac_ls[idx])
    eq_long = np.cumprod(fac_long[idx])
    eq_short = np.cumprod(fac_short[idx])
    ret_ls = fac_ls[idx] - 1.0
    n_days = int(idx.size)

    def _ann(eq):
        if n_days <= 0 or eq.size == 0 or not np.isfinite(eq[-1]) or eq[-1] <= 0:
            return NAN
        return float(eq[-1] ** (TRADING_DAYS / n_days) - 1.0)

    def _mdd(eq):
        if eq.size == 0:
            return NAN
        peak = np.maximum.accumulate(eq)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = eq / peak - 1.0
        return float(np.nanmin(dd)) if np.isfinite(dd).any() else NAN

    def _sharpe(ret):
        if n_days < 3:
            return NAN
        sd = float(np.std(ret, ddof=1))
        if not np.isfinite(sd) or sd < EPS:
            return NAN
        return float(np.mean(ret) / sd * math.sqrt(TRADING_DAYS))

    ann = _ann(eq_ls)
    mdd = _mdd(eq_ls)
    sharpe = _sharpe(ret_ls)
    calmar = float(ann / abs(mdd)) if (np.isfinite(ann) and np.isfinite(mdd) and abs(mdd) > EPS) else NAN

    to_avg = float(np.mean(np.asarray(to_long_list) + np.asarray(to_short_list)))
    n_periods = len(to_long_list)
    to_ann = to_avg * (TRADING_DAYS / h) if h > 0 else NAN

    dates = None
    gs = dict(long=dict(mean=float(np.mean(size_long)), min=int(np.min(size_long)),
                        max=int(np.max(size_long))))
    if not long_only and size_short:
        gs["short"] = dict(mean=float(np.mean(size_short)), min=int(np.min(size_short)),
                           max=int(np.max(size_short)))

    return dict(
        ok=True, h=h, n_groups=n_groups, cost_bps=float(cost_bps),
        long_only=bool(long_only),
        dates=None,
        day_index=idx,
        equity=eq_ls, equity_long=eq_long, equity_short=eq_short,
        daily_ret=ret_ls,
        n_days=n_days, n_periods=n_periods,
        ann_return=ann, ann_long=_ann(eq_long), ann_short=_ann(eq_short),
        ann_ls=ann,
        ann_vol=float(np.std(ret_ls, ddof=1) * math.sqrt(TRADING_DAYS)) if n_days > 1 else NAN,
        sharpe=sharpe, max_drawdown=mdd, calmar=calmar,
        win_rate=float((ret_ls > 0).mean()) if n_days else NAN,
        turnover_avg=to_avg, turnover_ann=to_ann,
        cost_total_bps=float(cost_total * 1e4),
        cost_ann_bps=float(cost_total / max(n_days, 1) * TRADING_DAYS * 1e4),
        group_sizes=gs,
        rebalance_idx=[int(x) for x in rebal[:n_periods]],
        total_return=float(eq_ls[-1] - 1.0) if eq_ls.size else NAN,
    )


# --------------------------------------------------------------------------- #
# 对外：多空回测
# --------------------------------------------------------------------------- #
def run_long_short(res: dict, h: int, n_groups: int = 5,
                   cost_bps: float = 10.0) -> dict:
    """
    多空组合回测：t 日 score 分 n_groups 组 → t+1 开盘等权建仓 → 持有 h 日
    → 下一期在开盘换仓，非重叠调仓，双边成本 cost_bps。

    返回 dict
    --------
    ok, h, n_groups, cost_bps, long_only,
    dates (np.ndarray 或 None), equity, equity_long, equity_short, daily_ret,
    n_days, n_periods, total_return,
    ann_return / ann_ls / ann_long / ann_short, ann_vol,
    sharpe, max_drawdown, calmar, win_rate,
    turnover_avg（每期两腿合计换手）, turnover_ann, cost_total_bps, cost_ann_bps,
    group_sizes {long:{mean,min,max}, short:{...}}, rebalance_idx
    """
    OPEN = np.asarray(res["OPEN"], dtype=np.float64)
    CLOSE = np.asarray(res["CLOSE"], dtype=np.float64)
    # engine 输出可能是时间主序 (T,N)，这里按 (N,T) 对齐
    scores = _align_scores(res["scores"], OPEN.shape)
    out = _portfolio(scores, OPEN, CLOSE, h, n_groups, cost_bps, long_only=False)
    if out.get("ok") and res.get("dates") is not None:
        d = np.asarray(res["dates"])
        idx = out["day_index"]
        if d.size == scores.shape[1]:
            out["dates"] = d[idx]
    return out


# --------------------------------------------------------------------------- #
# 对外：基线对比
# --------------------------------------------------------------------------- #
def _random_scores(scores: np.ndarray, seed: int = 20240101) -> np.ndarray:
    """同形状随机打分（保持原有的可交易掩码），固定种子可复现。"""
    rng = np.random.default_rng(seed)
    r = rng.uniform(-1.0, 1.0, size=scores.shape)
    return np.where(np.isfinite(scores), r, np.nan)


def run_baselines(res: dict, res_baseline: dict, h: int,
                  cost_bps: float = 10.0, n_groups: int = 5,
                  seed: int = 20240101) -> pd.DataFrame:
    """
    基线对比表：学习器 vs 等权集成 vs 随机打分 vs 买入持有。

    * 学习器      —— res['scores'] 多空
    * 等权集成    —— res_baseline['scores'] 多空（学习器跑不赢它就是无效）
    * 随机打分    —— 同掩码随机分数多空（安慰剂，必须接近 0）
    * 买入持有    —— 全池等权多头，每 h 日再平衡（市场 beta 参照）

    返回 DataFrame，列为中文，行为策略。
    """
    OPEN = np.asarray(res["OPEN"], dtype=np.float64)
    CLOSE = np.asarray(res["CLOSE"], dtype=np.float64)
    scores = _align_scores(res["scores"], OPEN.shape)

    rows: List[Dict[str, Any]] = []

    def _add(name: str, out: dict, note: str = ""):
        if not out.get("ok"):
            rows.append(dict(策略=name, 年化收益=NAN, 年化波动=NAN, Sharpe=NAN,
                             最大回撤=NAN, 胜率=NAN, 平均换手=NAN,
                             累计成本bps=NAN, 期数=0, 备注=out.get("reason", "失败")))
            return
        rows.append(dict(
            策略=name,
            年化收益=out["ann_return"] * 100.0,
            年化波动=out["ann_vol"] * 100.0,
            Sharpe=out["sharpe"],
            最大回撤=out["max_drawdown"] * 100.0,
            胜率=out["win_rate"] * 100.0,
            平均换手=out["turnover_avg"],
            累计成本bps=out["cost_total_bps"],
            期数=out["n_periods"],
            备注=note,
        ))

    _add("学习器(多空)", _portfolio(scores, OPEN, CLOSE, h, n_groups, cost_bps), "主策略")

    bs = res_baseline.get("scores") if isinstance(res_baseline, dict) else None
    if bs is not None:
        try:
            bs_a = _align_scores(bs, OPEN.shape)
        except ValueError:
            bs_a = None
        if bs_a is not None:
            _add("等权集成(多空)", _portfolio(bs_a, OPEN, CLOSE, h, n_groups, cost_bps),
                 "学习器必须战胜它")
        else:
            _add("等权集成(多空)", dict(ok=False, reason="bad_baseline_shape"), "")
    else:
        _add("等权集成(多空)", dict(ok=False, reason="missing_res_baseline"), "")
    _add("随机打分(多空)", _portfolio(_random_scores(scores, seed), OPEN, CLOSE,
                                   h, n_groups, cost_bps), "安慰剂，应≈0")

    # 买入持有：全池等权多头，用 0 分占位但保留可交易掩码
    hold = np.where(np.isfinite(scores), 0.0, np.nan)
    _add("买入持有(等权多头)", _portfolio(hold, OPEN, CLOSE, h, 1, cost_bps, long_only=True),
         "市场beta参照")

    return pd.DataFrame(rows)


def compare_table(bt_main: dict, baselines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """把主策略绩效与基线表拼成一张对比表（report 用）。"""
    rows = []
    if bt_main and bt_main.get("ok"):
        rows.append(dict(
            策略="学习器(多空)", 年化收益=bt_main["ann_return"] * 100.0,
            年化波动=bt_main["ann_vol"] * 100.0, Sharpe=bt_main["sharpe"],
            最大回撤=bt_main["max_drawdown"] * 100.0, 胜率=bt_main["win_rate"] * 100.0,
            平均换手=bt_main["turnover_avg"], 累计成本bps=bt_main["cost_total_bps"],
            期数=bt_main["n_periods"], 备注="主策略",
        ))
    if baselines is not None and len(baselines):
        base = baselines[baselines["策略"] != "学习器(多空)"]
        rows.extend(base.to_dict("records"))
    return pd.DataFrame(rows)
