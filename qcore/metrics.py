"""
评估指标层（metrics）—— 全部为「走样外」口径。

数据契约（由 qcore/engine.py 输出）：
    res['dates']   (T,)   datetime64   主日历
    res['codes']   (N,)   list[str]
    res['scores']  (N,T)  float        综合分 [-1,1]，NaN = 当日不可交易
    res['RET'][h]  (N,T)  float        r_h(t) = open(t+1+h)/open(t+1) - 1
    res['REGIME']  (T,)   int8         市场状态
    res['weights_hist'] (T,S) float    每日信号权重（行和=1）
    res['sids']    (S,)   list[str]

设计要点：
  1. 横截面统计「按日先算、再对日聚合」，避免某天股票多就把该天权重放大；
  2. 只使用 |score| > thresh 的样本（中性分不参与方向统计，避免凑样本）；
  3. 所有分组用「序数秩等分」而非 qcut，天然免疫重复值导致的边界炸裂；
  4. 无 scipy：Spearman 自己实现（平均秩处理 ties）。

注意：本模块只做统计，不做任何全样本标准化 / 分位标定，不引入未来信息。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "hit_rate",
    "rank_ic",
    "weight_concentration",
    "decile_returns",
    "assign_groups",
    "evaluate_all",
    "spearman",
]

EPS = 1e-12
NAN = float("nan")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _arr(x) -> np.ndarray:
    """转 float64 ndarray（NaN 保留）。"""
    a = np.asarray(x, dtype=np.float64)
    return a


def _align_scores(scores, ref_shape) -> np.ndarray:
    """
    把 scores 规整到 ref_shape=(N,T)。

    SPEC 声明 scores 为 (N,T)，但 engine.py 当前实现是时间主序 (T,N)；
    这里按 RET/OPEN 的 (N,T) 形状自动对齐，两种布局都能吃。
    仅当两维形状互换后才匹配时才转置；方形矩阵优先按 (N,T) 解释。
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError("scores must be 2-D")
    if tuple(s.shape) == tuple(ref_shape):
        return s
    if s.T.shape == tuple(ref_shape):
        return np.ascontiguousarray(s.T)
    raise ValueError(f"cannot align scores{tuple(s.shape)} to (N,T)={tuple(ref_shape)}")


def _year_axis(dates, n_cols: int) -> Optional[np.ndarray]:
    """把日期轴转成 (T,) 的年份数组；无效返回 None。"""
    if dates is None:
        return None
    d = np.asarray(dates)
    if d.size != n_cols:
        return None
    try:
        return pd.DatetimeIndex(d).year.to_numpy()
    except Exception:
        return None


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based 平均秩（ties 取平均），用于 Spearman。"""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman 秩相关；任一侧为常量或样本 < 3 时返回 NaN。"""
    a = _arr(a).ravel()
    b = _arr(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return NAN
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa < EPS or sb < EPS:
        return NAN
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def assign_groups(values, n_groups: int) -> Optional[List[np.ndarray]]:
    """
    按横截面序数秩等分成 n_groups 组，返回 list[np.ndarray]（组 0 = 分数最低）。

    用秩而不是分位数切点：重复值不会造成空组 / 边界塌缩。
    有效样本数 < n_groups 时返回 None。
    """
    v = _arr(values).ravel()
    idx = np.flatnonzero(np.isfinite(v))
    k = idx.size
    if n_groups < 1 or k < n_groups:
        return None
    order = np.argsort(v[idx], kind="mergesort")
    rank_pos = np.empty(k, dtype=np.int64)
    rank_pos[order] = np.arange(k)
    g = (rank_pos * n_groups) // k          # 0..n_groups-1，各组数量差 <= 1
    return [idx[g == j] for j in range(n_groups)]


def _group_stat(keys, values, mask) -> Tuple[Dict, Dict]:
    """
    按 keys 分组求 values 的均值与样本数。

    keys   : 与 mask 同形状（通常 (N,T) 广播出来的年份/regime）
    values : 已经按 mask 压平的一维数组
    """
    out: Dict[int, float] = {}
    cnt: Dict[int, int] = {}
    if int(mask.sum()) == 0:
        return out, cnt
    kk = np.asarray(keys)[mask]
    vv = np.asarray(values).ravel()
    for u in np.unique(kk):
        sel = kk == u
        n = int(sel.sum())
        out[int(u)] = float(vv[sel].mean()) if n else NAN
        cnt[int(u)] = n
    return out, cnt


# --------------------------------------------------------------------------- #
# 1. 方向命中率
# --------------------------------------------------------------------------- #
def hit_rate(scores, RET, h: int, dates=None, regime=None,
             thresh: float = 0.0) -> dict:
    """
    方向命中率：sign(score) == sign(r_h) 的比例。

    参数
    ----
    scores : (N,T) 综合分，NaN = 不可交易
    RET    : dict[int, (N,T)] 引擎输出的前瞻收益
    h      : 预测周期（交易日）
    dates  : (T,) 主日历，用于分年
    regime : (T,) 市场状态，用于分 regime
    thresh : 只统计 |score| > thresh 的样本（默认 0，即剔除中性分）

    返回
    ----
    dict: hit_rate(整体)、n_samples、n_days、mean_signed_ret、by_year、by_year_n、
          by_regime、by_regime_n、coverage、h、thresh、insufficient
    零收益样本计为「未命中」（保守口径），并单独给出 n_zero_ret。
    """
    sc = _arr(scores)
    if sc.ndim != 2:
        raise ValueError("scores must be 2-D (N,T)")
    if h not in RET:
        raise KeyError(f"RET has no horizon {h}")
    r = _arr(RET[h])
    sc = _align_scores(sc, r.shape)

    finite = np.isfinite(sc) & np.isfinite(r)
    mask = finite & (np.abs(sc) > thresh)
    n_samples = int(mask.sum())
    base = dict(h=int(h), thresh=float(thresh), n_samples=n_samples,
                n_days=int((mask.sum(axis=0) > 0).sum()),
                coverage=float(finite.sum() / max(int(np.isfinite(sc).sum()), 1)))

    if n_samples == 0:
        base.update(hit_rate=NAN, mean_signed_ret=NAN, n_zero_ret=0,
                    by_year={}, by_year_n={}, by_regime={}, by_regime_n={},
                    insufficient=True)
        return base

    s = sc[mask]
    rr = r[mask]
    hit = (np.sign(s) * np.sign(rr)) > 0          # 零收益 => 未命中
    n_zero = int((rr == 0).sum())

    out = dict(base)
    # 命中率的不确定性：股票-日单元格之间高度相关（同一决策日的截面强相关），
    # 把它们当独立样本会把 SE 低估一个量级。实测 53 天台账：朴素二项 SE ±0.45pp，
    # 按决策日聚类 ±13.5pp（差 30 倍）——于是「48.7% 低于抛硬币 5.5 个标准差」
    # 实际只有 0.18 个标准差。这里同时给出两种口径，并把 insufficient 的
    # 判据从「单元格为 0」改成「独立期数不足」。
    from .thresholds import MIN_TRADES
    p = float(hit.mean())
    n_days = int(base["n_days"])
    se_naive = float(np.sqrt(max(p * (1.0 - p), 1e-12) / max(n_samples, 1)))
    # 重叠收益下，"天数"不是独立观测数：h=5 时 3335 天只有 667 个独立期。
    # 用 n_days 会把 SE 低估 √h 倍（评审 R7）。
    n_indep = int(n_days // max(int(h), 1))
    se_cluster = float(np.sqrt(max(p * (1.0 - p), 1e-12) / max(n_indep, 1)))
    out.update(
        hit_rate=p,
        mean_signed_ret=float((np.sign(s) * rr).mean()),
        n_zero_ret=n_zero,
        se_naive=se_naive,
        se_cluster=se_cluster,
        ci95_naive=[p - 1.96 * se_naive, p + 1.96 * se_naive],
        ci95_cluster=[p - 1.96 * se_cluster, p + 1.96 * se_cluster],
        n_eff=n_indep,                                 # 有效独立观测数（非重叠期）
        n_indep=n_indep,
        insufficient=bool(n_days < MIN_TRADES),
        insufficient_reason=(f"n_days={n_days} < MIN_TRADES={MIN_TRADES}"
                             if n_days < MIN_TRADES else ""),
    )

    yr = _year_axis(dates, sc.shape[1])
    if yr is not None:
        ykeys = np.broadcast_to(yr[None, :], sc.shape)
        out["by_year"], out["by_year_n"] = _group_stat(ykeys, hit.astype(float), mask)
    else:
        out["by_year"], out["by_year_n"] = {}, {}

    if regime is not None:
        rg = np.asarray(regime).ravel()
        if rg.size == sc.shape[1]:
            rkeys = np.broadcast_to(rg[None, :], sc.shape)
            out["by_regime"], out["by_regime_n"] = _group_stat(rkeys, hit.astype(float), mask)
        else:
            out["by_regime"], out["by_regime_n"] = {}, {}
    else:
        out["by_regime"], out["by_regime_n"] = {}, {}

    return out


# --------------------------------------------------------------------------- #
# 2. Rank IC
# --------------------------------------------------------------------------- #
def rank_ic(scores, RET, h: int, dates=None, min_names: int = 5) -> dict:
    """
    每日横截面 Spearman(score, r_h)。

    返回
    ----
    dict: ic (T,) 逐日 IC（无效日为 NaN）、mean、std、icir、icir_ann、
          t_stat、positive_ratio、n_days、by_year、by_year_n、h、min_names
    ICIR = mean/std（日频口径）；icir_ann = ICIR * sqrt(252)。
    """
    sc = _arr(scores)
    if sc.ndim != 2:
        raise ValueError("scores must be 2-D (N,T)")
    if h not in RET:
        raise KeyError(f"RET has no horizon {h}")
    r = _arr(RET[h])
    sc = _align_scores(sc, r.shape)

    T = sc.shape[1]
    ic = np.full(T, np.nan, dtype=np.float64)
    n_used = np.zeros(T, dtype=np.int64)
    for t in range(T):
        m = np.isfinite(sc[:, t]) & np.isfinite(r[:, t])
        n = int(m.sum())
        n_used[t] = n
        if n >= max(int(min_names), 3):
            v = spearman(sc[m, t], r[m, t])
            ic[t] = v

    ok = np.isfinite(ic)
    n_days = int(ok.sum())
    if n_days == 0:
        return dict(h=int(h), ic=ic, n_used=n_used, mean=NAN, std=NAN,
                    icir=NAN, icir_ann=NAN, t_stat=NAN, positive_ratio=NAN,
                    n_days=0, by_year={}, by_year_n={}, min_names=int(min_names),
                    insufficient=True)

    vals = ic[ok]
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n_days > 1 else NAN
    icir = float(mean / std) if (std == std and std > EPS) else NAN
    t_stat = float(mean / (std / math.sqrt(n_days))) if (std == std and std > EPS) else NAN

    out = dict(
        h=int(h), ic=ic, n_used=n_used,
        mean=mean, std=std, icir=icir,
        icir_ann=(icir * math.sqrt(252.0)) if icir == icir else NAN,
        t_stat=t_stat,
        positive_ratio=float((vals > 0).mean()),
        n_days=n_days,
        by_year={}, by_year_n={},
        min_names=int(min_names),
        insufficient=False,
    )

    yr = _year_axis(dates, T)
    if yr is not None:
        by_y, by_yn = {}, {}
        for y in np.unique(yr[ok]):
            sel = ok & (yr == y)
            by_y[int(y)] = float(ic[sel].mean())
            by_yn[int(y)] = int(sel.sum())
        out["by_year"], out["by_year_n"] = by_y, by_yn
    return out


# --------------------------------------------------------------------------- #
# 3. 权重集中度
# --------------------------------------------------------------------------- #
def weight_concentration(weights_hist, sids: Optional[Sequence[str]] = None,
                         dates=None, top_k: int = 5) -> dict:
    """
    信号权重集中度：Herfindahl、Top-K 占比、有效信号数、随时间变化。

    返回
    ----
    dict: hhi (T,)、topk_share (T,)、effective_n (T,)、
          hhi_mean/median、topk_mean、effective_n_mean、n_signals、
          hhi_trend（后半段均值 - 前半段均值）、top_sids_last、bottom_sids_last
    每行先归一到和=1（全零/含 NaN 行记 NaN，不参与统计）。
    """
    w = _arr(weights_hist)
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2:
        raise ValueError("weights_hist must be 2-D (T,S)")
    T, S = w.shape
    k = max(1, min(int(top_k), S))

    ws = np.where(np.isfinite(w), w, np.nan)
    ws = np.clip(ws, 0.0, None)
    row = ws.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        wn = np.where((row > EPS), ws / row, np.nan)

    hhi = np.nansum(wn ** 2, axis=1)
    hhi[np.all(~np.isfinite(wn), axis=1)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        eff_n = np.where(hhi > EPS, 1.0 / hhi, np.nan)
    srt = -np.sort(-np.nan_to_num(wn, nan=0.0), axis=1)
    topk = srt[:, :k].sum(axis=1)
    topk[np.all(~np.isfinite(wn), axis=1)] = np.nan

    ok = np.isfinite(hhi)
    half = T // 2
    trend = NAN
    if ok.sum() >= 4 and half >= 1:
        a = hhi[:half][np.isfinite(hhi[:half])]
        b = hhi[half:][np.isfinite(hhi[half:])]
        if a.size and b.size:
            trend = float(b.mean() - a.mean())

    names = list(sids) if sids is not None and len(sids) == S else [f"s{i}" for i in range(S)]
    top_sids, bottom_sids = [], []
    last = wn[-1] if T else np.full(S, np.nan)
    if np.isfinite(last).any():
        order = np.argsort(-np.nan_to_num(last, nan=-1.0))
        top_sids = [(names[i], float(last[i])) for i in order[:k]]
        bottom_sids = [(names[i], float(last[i])) for i in order[::-1][:k]]

    out = dict(
        n_days=int(T), n_signals=int(S), top_k=int(k),
        hhi=hhi, topk_share=topk, effective_n=eff_n,
        hhi_mean=float(np.nanmean(hhi)) if ok.any() else NAN,
        hhi_median=float(np.nanmedian(hhi)) if ok.any() else NAN,
        hhi_last=float(hhi[-1]) if T else NAN,
        topk_mean=float(np.nanmean(topk)) if ok.any() else NAN,
        effective_n_mean=float(np.nanmean(eff_n)) if ok.any() else NAN,
        hhi_trend=trend,
        top_sids_last=top_sids,
        bottom_sids_last=bottom_sids,
        by_year={},
        insufficient=not bool(ok.any()),
    )

    yr = _year_axis(dates, T)
    if yr is not None:
        by_y = {}
        for y in np.unique(yr[ok]):
            sel = ok & (yr == y)
            if sel.any():
                by_y[int(y)] = dict(hhi=float(np.nanmean(hhi[sel])),
                                    topk=float(np.nanmean(topk[sel])))
        out["by_year"] = by_y
    return out


# --------------------------------------------------------------------------- #
# 4. 分组收益
# --------------------------------------------------------------------------- #
def decile_returns(scores, RET, h: int, n: int = 5, dates=None,
                   thresh: Optional[float] = None) -> pd.DataFrame:
    """
    按 score 分组（默认 5 组，组 1 = 分数最低）的平均前瞻收益。

    先按日算各组等权平均收益，再对「有样本的交易日」等权平均 —— 保证
    每个交易日对结果的贡献相同（不是按股票数加权）。

    返回 DataFrame: group, label, mean_ret, mean_ret_pct, hit_rate,
                    n_obs, n_days
    DataFrame.attrs: monotonicity（组序号与组均收益的 Spearman）、
                     monotone_increasing、top_minus_bottom、h、n_groups
    """
    sc = _arr(scores)
    if sc.ndim != 2:
        raise ValueError("scores must be 2-D (N,T)")
    if h not in RET:
        raise KeyError(f"RET has no horizon {h}")
    r = _arr(RET[h])
    sc = _align_scores(sc, r.shape)

    T = sc.shape[1]
    n = max(2, int(n))
    sums = np.zeros(n)
    cnt_days = np.zeros(n, dtype=np.int64)
    cnt_obs = np.zeros(n, dtype=np.int64)
    hit_sum = np.zeros(n)
    hit_obs = np.zeros(n)

    for t in range(T):
        col = sc[:, t]
        if thresh is not None:
            col = np.where(np.abs(col) > thresh, col, np.nan)
        groups = assign_groups(col, n)
        if groups is None:
            continue
        for g, idx in enumerate(groups):
            rr = r[idx, t]
            ok = np.isfinite(rr)
            if not ok.any():
                continue
            v = rr[ok]
            sums[g] += float(v.mean())
            cnt_days[g] += 1
            cnt_obs[g] += int(ok.sum())
            hit_sum[g] += float(((v > 0).sum()))
            hit_obs[g] += int(ok.size)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ret = np.where(cnt_days > 0, sums / np.maximum(cnt_days, 1), np.nan)
        hr = np.where(hit_obs > 0, hit_sum / np.maximum(hit_obs, 1), np.nan)

    labels = []
    for g in range(n):
        if n == 1:
            labels.append("all")
        elif g == 0:
            labels.append("Q1(low)")
        elif g == n - 1:
            labels.append(f"Q{n}(high)")
        else:
            labels.append(f"Q{g+1}")

    df = pd.DataFrame({
        "group": np.arange(1, n + 1),
        "label": labels,
        "mean_ret": mean_ret,
        "mean_ret_pct": mean_ret * 100.0,
        "hit_rate": hr,
        "n_obs": cnt_obs,
        "n_days": cnt_days,
    })

    finite = np.isfinite(mean_ret)
    if finite.sum() >= 2:
        mono = spearman(np.arange(n)[finite], mean_ret[finite])
    else:
        mono = NAN
    diffs = np.diff(mean_ret)
    mono_inc = bool(np.all(diffs[np.isfinite(diffs)] > 0)) if np.isfinite(diffs).any() else False
    spread = (float(mean_ret[-1] - mean_ret[0])
              if np.isfinite(mean_ret[-1]) and np.isfinite(mean_ret[0]) else NAN)

    df.attrs.update(dict(monotonicity=mono, monotone_increasing=mono_inc,
                         top_minus_bottom=spread, h=int(h), n_groups=n))
    return df


# --------------------------------------------------------------------------- #
# 5. 一站式
# --------------------------------------------------------------------------- #
def evaluate_all(res: dict, horizons: Iterable[int] = (1, 5, 20),
                 n_groups: int = 5, thresh: float = 0.0) -> dict:
    """
    对引擎输出跑全部指标，供 report.render 直接消费。

    返回
    ----
    {'horizons': [...], 'thresh': float,
     'hit_rate': {h: dict}, 'rank_ic': {h: dict},
     'decile': {h: DataFrame}, 'concentration': dict}
    """
    scores = res["scores"]
    RET = res["RET"]
    dates = res.get("dates")
    regime = res.get("REGIME")
    hs = [int(x) for x in horizons if int(x) in RET]

    out: Dict[str, Any] = {"horizons": hs, "thresh": float(thresh),
                           "hit_rate": {}, "rank_ic": {}, "decile": {}}
    for h in hs:
        out["hit_rate"][h] = hit_rate(scores, RET, h, dates=dates,
                                      regime=regime, thresh=thresh)
        out["rank_ic"][h] = rank_ic(scores, RET, h, dates=dates)
        out["decile"][h] = decile_returns(scores, RET, h, n=n_groups, dates=dates)

    wh = res.get("weights_hist")
    if wh is not None:
        out["concentration"] = weight_concentration(
            wh, sids=res.get("sids"), dates=dates)
    else:
        out["concentration"] = {"insufficient": True}
    return out
