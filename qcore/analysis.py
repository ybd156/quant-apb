"""
优势体检 —— 把"这套东西到底行不行"算成一个可证伪的结论。

不做点估计吹牛，只回答四个问题：
  ① 每日横截面 IC 的均值，HAC 标准误多大，t 值多少？（重叠样本已校正）
  ② 扣掉"我试了 73 个信号"的运气之后，Sharpe 还显著吗？（Deflated Sharpe）
  ③ 这个优势是长期稳定的，还是只有某两年？（分年度 + 滚动监控）
  ④ 如果方向是系统性反的，反着做行不行？（对称检验，避免"亏了就反过来说"）

所有函数都只吃"已经算好的分数与收益"，不碰引擎内部状态。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import stats as S


def _rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（处理 ties），无 scipy 依赖。"""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # 处理并列
    sa = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            avg = (i + j + 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def daily_ic_series(scores: np.ndarray, RET: np.ndarray, min_names: int = 5) -> np.ndarray:
    """scores / RET 均为 (N,T)。返回 (T,) 的每日横截面 Spearman。"""
    N, T = scores.shape
    out = np.full(T, np.nan)
    for t in range(T):
        sc, r = scores[:, t], RET[:, t]
        m = np.isfinite(sc) & np.isfinite(r)
        if m.sum() < min_names:
            continue
        out[t] = spearman(sc[m], r[m])
    return out


def long_short_series(scores: np.ndarray, RET: np.ndarray, h: int,
                      n_groups: int = 5) -> np.ndarray:
    """
    非重叠 h 日调仓的多空收益序列（每期一个观测）。
    返回长度约 T/h 的数组，元素为"多头组均收益 − 空头组均收益"。
    """
    N, T = scores.shape
    rets = []
    for t in range(0, T - 1 - h, h):
        sc, r = scores[:, t], RET[:, t]
        m = np.isfinite(sc) & np.isfinite(r)
        if m.sum() < n_groups * 2:
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(sc[idx])]
        g = len(order) // n_groups
        if g < 1:
            continue
        low = order[:g]
        high = order[-g:]
        rets.append(float(r[high].mean() - r[low].mean()))
    return np.asarray(rets, dtype=np.float64)


def edge_report(scores: np.ndarray, RET: np.ndarray, dates: np.ndarray,
                h: int = 5, n_trials: int = 73, label: str = "learned") -> Dict:
    """一份完整的优势体检表。scores/RET 为 (N,T)。"""
    ic = daily_ic_series(scores, RET)
    ic_ok = ic[np.isfinite(ic)]
    summ = S.summarize_edge(ic_ok, dates[-ic_ok.size:], horizon=h, n_trials=n_trials)

    ls = long_short_series(scores, RET, h)
    # 非重叠 h 日收益 → 换算成日频等效
    ls_daily = ls / h if ls.size else ls
    dsr_ls = S.deflated_sharpe(float(np.nanmean(ls_daily)), int(ls_daily.size * h),
                               n_trials) if ls.size > 10 else {"ok": False}
    ls_stat = S.mean_tstat(ls, lags=0)          # 已是非重叠，用朴素 SE
    sr_ls = S.sharpe_ratio(ls, periods=max(1, 252 // max(h, 1)))

    # 反向对称检验：把分数取负再算一遍，避免"亏了就说要反着做"
    ic_inv = -ic
    inv = S.mean_tstat(ic_inv[np.isfinite(ic_inv)], lags=max(h - 1, 0))

    return {
        "label": label, "h": h, "n_trials": n_trials,
        "ic": {"mean": summ["ic_mean"], "hac_se": summ["hac_se"], "t": summ["t"],
               "p": summ["p"], "ci95": summ["ci95"], "boot_ci95": summ["boot_ci95"],
               "n_obs": summ["n_obs"], "effective_n": summ["effective_n"]},
        "ic_verdict": summ["verdict"],
        "long_short": {"mean_per_period": ls_stat["mean"], "t": ls_stat["t"],
                       "p": ls_stat["p"], "sharpe": sr_ls, "n_periods": int(ls.size),
                       "annualized": float(np.nanmean(ls) * 252 / max(h, 1)) if ls.size else None,
                       "dsr": dsr_ls.get("dsr"), "dsr_significant": dsr_ls.get("significant")},
        "inverted_check": {"ic_mean": inv["mean"], "t": inv["t"], "p": inv["p"]},
        "deflated_sharpe": dsr_ls,
        "rolling_edge": {"frac_time_active": summ["frac_time_active"],
                         "final_state": summ["final_state"]},
        "subperiods": summ["subperiods"],
        "conclusion": _conclude(summ, ls_stat, inv),
    }


def _conclude(summ: Dict, ls_stat: Dict, inv: Dict) -> str:
    t = summ.get("t")
    if not np.isfinite(t):
        return "样本不足，无法判断"
    if t >= 2.0 and summ["deflated_sharpe"].get("significant"):
        return "IC 显著为正且通过多重检验 → 有可用优势，可进入模拟盘验证"
    if t <= -2.0:
        return ("IC 显著为负 → 信号方向系统性做反。反着做在统计上成立，"
                "但需独立样本再验证一次，不能直接上线")
    return "IC 与 0 不可区分 → 当前信号集无优势，应停手或换特征，而不是加大权重"


def compare_learned_vs_equal(res: dict, res_eq: dict, p3d, h: int = 5) -> Dict:
    """学习器 vs 等权集成：如果学不赢等权，学习器就是无效的。"""
    a = daily_ic_series(res["scores"], p3d.RET[h])
    b = daily_ic_series(res_eq["scores"], p3d.RET[h])
    sa = S.mean_tstat(a[np.isfinite(a)], lags=max(h - 1, 0))
    sb = S.mean_tstat(b[np.isfinite(b)], lags=max(h - 1, 0))
    diff = a - b
    sd = S.mean_tstat(diff[np.isfinite(diff)], lags=max(h - 1, 0))
    return {"learned_ic": sa["mean"], "learned_t": sa["t"],
            "equal_ic": sb["mean"], "equal_t": sb["t"],
            "diff_ic": sd["mean"], "diff_t": sd["t"], "diff_p": sd["p"],
            "verdict": ("学习器优于等权（t=%.2f）" % sd["t"] if sd["t"] > 2
                        else ("学习器劣于等权（t=%.2f）" % sd["t"] if sd["t"] < -2
                              else "学习器与等权无显著差异 → 学习器无效，应回退等权"))}
