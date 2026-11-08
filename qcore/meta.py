"""
Meta 层（二级学习 / 自我纠错）—— 让系统自己决定"要不要反着用"。

## 为什么必须做这一层

一级集成把 73 个信号加权平均成 score。但"加权平均"有个结构性缺陷：
**权重只能调整"听谁的多"，不能调整"要不要反着听"。**
如果整族信号在当前市场系统性偏反（A 股短周期均值回复），加权平均仍然偏反，
怎么调权重都救不回来。

实测（283 只 / 2013-2026 / h=5）：横截面 Rank IC 均值 −0.018，HAC t = −2.44，
95% 置信区间 [−0.033, −0.003] 整段在 0 以下 —— 即"信号方向系统性做反"。

## 为什么必须用横截面口径

第一版用**混合回归** r ~ score（把所有股票所有日子堆在一起），结果 β 是正的，
和 IC 的负号矛盾。原因：混合回归被**时间序列的 beta** 主导（大盘涨，所有票都涨，
score 高的票恰好是高 beta），把横截面的真实关系淹没了。
选股问题的正确口径是**每日横截面去均值**，这才是 IC 度量的东西。

## 自我纠错的数学

对每个到期日 s 算横截面 Spearman IC，得到 IC 序列 {IC_s}。
用扩张窗口（只含已到期样本）估计均值与 HAC 标准误：

    μ_t = mean(IC_s),  s ≤ t−1−h
    SE_t = Newey-West(IC_s, lags=h−1)
    t_t = μ_t / SE_t

    sign_t = −1  若 t_t ≤ −t_enter   （显著反向 → 反着用）
             +1  若 t_t ≥ +t_enter   （显著正向 → 顺势用）
              0  否则                （不显著 → 不表态，输出 0）

    adj_t = sign_t · score_t

"不表态"是这套机制最重要的一档：**没有统计证据时就不下注**，
而不是把噪声当信号。这正是"对一次两次不算什么"的工程解法。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import stats as S


@dataclass
class MetaConfig:
    t_enter: float = 2.0       # 判定显著的门槛（|t|）
    t_exit: float = 1.0        # 滞回：已翻转后，|t| 掉到此值以下才回到"不表态"
    min_days: int = 250        # 最少到期天数才允许表态
    expanding: bool = True     # True=扩张窗口（稳定）；False=滚动窗口
    window: int = 750          # expanding=False 时的窗口长度


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    sa = a[order]
    i, n = 0, a.size
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return ranks


def _xs_ic(sc: np.ndarray, r: np.ndarray, min_names: int = 5) -> float:
    m = np.isfinite(sc) & np.isfinite(r)
    if m.sum() < min_names:
        return np.nan
    a, b = _rankdata(sc[m]), _rankdata(r[m])
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def learn_meta(scores: np.ndarray, RET: np.ndarray, h: int, warmup: int = 120,
               cfg: MetaConfig = MetaConfig()) -> Dict:
    """
    scores / RET 均为 (T,N)。返回因果的 sign 序列、IC 均值与 t 值历史、调整后分数。
    """
    T, N = scores.shape
    ic_hist = np.full(T, np.nan)          # 当天（到期样本）的 IC
    mu_hist = np.zeros(T)                 # 到期样本的 IC 均值
    t_hist = np.zeros(T)
    sign_hist = np.zeros(T, dtype=np.int8)
    adj = np.zeros((T, N), dtype=np.float32)
    cur = 0

    for t in range(T):
        s = t - 1 - h
        if s >= 0:
            ic_hist[t] = _xs_ic(scores[s], RET[s])

        # 只用已到期的 IC 估计 μ 与 t
        lo = 0 if cfg.expanding else max(0, t - cfg.window)
        win = ic_hist[lo:t - h + 1] if t - h + 1 > lo else np.array([])
        win = win[np.isfinite(win)]
        if win.size >= cfg.min_days:
            st = S.mean_tstat(win, lags=max(h - 1, 0))
            mu_hist[t] = st["mean"]; t_hist[t] = st["t"]
            if cur == 0:
                if st["t"] >= cfg.t_enter:
                    cur = 1
                elif st["t"] <= -cfg.t_enter:
                    cur = -1
            elif cur > 0 and st["t"] < cfg.t_exit:
                cur = 0
            elif cur < 0 and st["t"] > -cfg.t_exit:
                cur = 0
        sign_hist[t] = cur

        if t >= warmup:
            row = scores[t]
            adj[t] = np.where(np.isfinite(row), cur * row, np.nan).astype(np.float32)

    return {
        "adj": adj, "sign": sign_hist, "ic": ic_hist,
        "mu": mu_hist, "t": t_hist,
        "final_sign": int(sign_hist[-1]),
        "final_t": float(t_hist[-1]),
        "final_mu": float(mu_hist[-1]),
        "frac_inverted": float((sign_hist == -1).mean()),
        "frac_silent": float((sign_hist == 0).mean()),
        "verdict": ("反着用" if sign_hist[-1] == -1 else
                    ("顺势用" if sign_hist[-1] == 1 else "不表态")),
    }


def compare_raw_vs_meta(scores: np.ndarray, RET: np.ndarray, meta: Dict) -> Dict:
    """原始 score vs meta 调整后：谁的方向更准。"""
    out = {}
    for name, sc in (("raw", scores), ("meta", meta["adj"])):
        m = np.isfinite(sc) & np.isfinite(RET) & (sc != 0)
        if m.sum() == 0:
            out[name] = {"hit_rate": None, "n": 0}
            continue
        hit = (np.sign(sc[m]) == np.sign(RET[m]))
        out[name] = {"hit_rate": float(hit.mean()), "n": int(m.sum()),
                     "mean_signed_ret": float((np.sign(sc[m]) * RET[m]).mean())}
    return out


def meta_ic(scores: np.ndarray, RET: np.ndarray, meta: Dict, h: int) -> Dict:
    """meta 调整后分数本身的 IC 显著性（用于确认"翻转"是否真的有用）。"""
    adj = meta["adj"]
    T, N = adj.shape
    ics = np.array([_xs_ic(adj[t], RET[t]) for t in range(T)])
    ics = ics[np.isfinite(ics)]
    return S.mean_tstat(ics, lags=max(h - 1, 0))
