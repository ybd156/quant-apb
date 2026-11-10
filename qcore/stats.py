"""
统计严谨层 —— 回答"这到底是本事，还是运气"。

三个必须解决的数学问题（不解决，所有结论都是自欺）：

① **重叠样本**：h 日收益是重叠的（t 与 t+1 的 h 日窗口共享 h-1 天），
   朴素 t 检验会把有效样本数高估约 h 倍，t 值虚高 √h。
   → 用 Newey-West (HAC) 标准误，滞后阶取 h-1；或用 h 日不重叠抽样。

② **多重检验**：73 个信号 × 若干参数变体 × 若干周期，挑最好的那个必然好看。
   → Deflated Sharpe Ratio（Bailey & López de Prado）：把"试了多少次"折进显著性里。
   → PSR：给定 SR、样本长度、偏度峰度，算"真实 SR>0 的概率"。

③ **长期与自纠错**：一次 IC=0.03 没有意义，要看
   - 滚动 IC 的置信区间是否长期在 0 以上
   - 分年度/分市场状态是否符号翻转
   - 什么时候该承认"优势消失了"并停手（kill switch）

全部无 scipy 依赖：正态分布用 erf 与有理逼近实现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

EULER = 0.5772156649015329


# ------------------------------------------------------------------ 分布
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """标准正态分位数（Acklam 有理逼近，误差 < 1.15e-9）。"""
    if not (0.0 < p < 1.0):
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def p_value_two_sided(t: float) -> float:
    if not np.isfinite(t):
        return float("nan")
    return float(2 * (1 - norm_cdf(abs(t))))


# ------------------------------------------------------------------ HAC
def newey_west_se(x: np.ndarray, lags: int = 0) -> float:
    """
    样本均值的 Newey-West (HAC) 标准误。
    lags=0 退化为朴素 SE。重叠 h 日收益建议 lags = h-1。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = (e * e).sum() / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)          # Bartlett 核
        s += 2.0 * w * (e[L:] * e[:-L]).sum() / n
    s = max(s, 0.0)
    return float(math.sqrt(s / n))


def mean_tstat(x: np.ndarray, lags: int = 0) -> Dict:
    """均值 + HAC SE + t 值 + p 值 + 95% 置信区间。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return {"n": int(n), "mean": float("nan"), "se": float("nan"),
                "t": float("nan"), "p": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mu = float(x.mean())
    se = newey_west_se(x, lags)
    t = mu / se if se and se > 0 else float("nan")
    return {"n": int(n), "mean": mu, "se": se, "t": float(t),
            "p": p_value_two_sided(t),
            "lo": mu - 1.96 * se if se else float("nan"),
            "hi": mu + 1.96 * se if se else float("nan")}


# ------------------------------------------------------------------ Bootstrap
def stationary_bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000,
                            avg_block: int = 20, alpha: float = 0.05,
                            seed: int = 42) -> Dict:
    """
    平稳块自助法（Politis-Romano）：保留时间自相关，适合重叠收益/IC 序列。
    返回统计量的百分位置信区间与"统计量>0 的经验概率"。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return {"ok": False, "n": int(n)}
    rng = np.random.default_rng(seed)
    p = 1.0 / max(avg_block, 1)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for j in range(n):
            idx[j] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        vals[b] = stat(x[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"ok": True, "n": int(n), "mean": float(stat(x)),
            "lo": float(lo), "hi": float(hi),
            "p_positive": float((vals > 0).mean()),
            "se_boot": float(vals.std(ddof=1)), "n_boot": int(n_boot)}


# ------------------------------------------------------------------ Sharpe
def sharpe_ratio(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def probabilistic_sharpe(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0,
                         sr_benchmark: float = 0.0) -> float:
    """
    PSR（Bailey & López de Prado 2012）：P(真实 SR > 基准)。
    PSR = Φ( (SR - SR*)·√(n-1) / √(1 - γ3·SR + (γ4-1)/4·SR²) )
    注意 sr 与 n 必须是**同频**（日频 SR 配日数）。
    """
    if not np.isfinite(sr) or n < 3:
        return float("nan")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm_cdf(z))


def deflated_sharpe(sr_daily: float, n_days: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> Dict:
    """
    Deflated Sharpe Ratio：把"试了 n_trials 次"的运气扣掉。
    SR0 = √Var(SR_hat) · [ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)) ]
    其中 Var(SR_hat) ≈ (1 - γ3·SR + (γ4-1)/4·SR²) / (n-1)
    """
    if not np.isfinite(sr_daily) or n_days < 10:
        return {"ok": False}
    N = max(int(n_trials), 1)
    var_sr = (1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2) / (n_days - 1)
    sd = math.sqrt(max(var_sr, 1e-12))
    if N == 1:
        sr0 = 0.0
    else:
        sr0 = sd * ((1 - EULER) * norm_ppf(1 - 1.0 / N) + EULER * norm_ppf(1 - 1.0 / (N * math.e)))
    psr = probabilistic_sharpe(sr_daily, n_days, skew, kurt, sr_benchmark=sr0)
    return {"ok": True, "sr_daily": sr_daily, "sr_annual": sr_daily * math.sqrt(252),
            "n_days": int(n_days), "n_trials": N,
            "sr0_daily": sr0, "sr0_annual": sr0 * math.sqrt(252),
            "dsr": psr, "significant": bool(psr > 0.95)}


def min_track_record_length(sr_daily: float, target_psr: float = 0.95,
                            skew: float = 0.0, kurt: float = 3.0) -> int:
    """要多少天样本，PSR 才能达到 target（回答"还要跑多久才算数"）。"""
    if not np.isfinite(sr_daily) or abs(sr_daily) < 1e-9:
        return -1
    z = norm_ppf(target_psr)
    var = 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2
    return int(math.ceil(1 + var * (z / sr_daily) ** 2))


# ------------------------------------------------------------------ 自纠错
@dataclass
class MonitorConfig:
    window: int = 250          # 滚动窗口（约 1 年）
    lags: int = 4              # HAC 滞后（h-1）
    t_enter: float = 2.0       # 判定"有优势"的 t 门槛
    t_exit: float = 0.5        # 判定"优势消失"的 t 门槛
    min_n: int = 120


def rolling_edge_monitor(ic_series: np.ndarray, cfg: MonitorConfig = MonitorConfig()) -> Dict:
    """
    滚动监控"优势是否还在"：
      - 每个窗口算 HAC t 值
      - 状态机：unknown → active（t>enter）→ dormant（t<exit）
    这就是"自我纠错"的开关：dormant 时下游仓位应归零，而不是继续硬做。
    """
    x = np.asarray(ic_series, dtype=np.float64)
    T = x.size
    t_hist = np.full(T, np.nan)
    state = np.zeros(T, dtype=np.int8)      # 0=unknown 1=active -1=dormant
    cur = 0
    for t in range(T):
        if t + 1 < cfg.min_n:
            continue
        w = x[max(0, t - cfg.window + 1): t + 1]
        st = mean_tstat(w, cfg.lags)
        t_hist[t] = st["t"]
        if np.isfinite(st["t"]):
            if cur <= 0 and st["t"] >= cfg.t_enter:
                cur = 1
            elif cur >= 0 and st["t"] <= cfg.t_exit:
                cur = -1
        state[t] = cur
    frac_active = float((state == 1).mean())
    return {"t_hist": t_hist, "state": state, "frac_active": frac_active,
            "final_state": int(cur),
            "final_t": float(t_hist[-1]) if np.isfinite(t_hist[-1]) else float("nan")}


def subperiod_table(series: np.ndarray, dates: np.ndarray, lags: int = 4) -> List[Dict]:
    """分年度稳定性：任一年符号翻转都要标红。"""
    import pandas as pd
    d = pd.DatetimeIndex(dates)
    yrs = d.year.to_numpy()
    out = []
    for y in sorted(set(yrs.tolist())):
        m = (yrs == y)
        if m.sum() < 30:
            continue
        st = mean_tstat(np.asarray(series)[m], lags)
        st["year"] = int(y)
        out.append(st)
    return out


def summarize_edge(ic_series: np.ndarray, dates: np.ndarray, horizon: int = 5,
                   n_trials: int = 73) -> Dict:
    """一份可以直接写进报告的"优势体检表"。"""
    x = np.asarray(ic_series, dtype=np.float64)
    lags = max(horizon - 1, 0)
    base = mean_tstat(x, lags)
    boot = stationary_bootstrap_ci(x, avg_block=max(horizon, 10))
    # DSR 的 sr 参数必须是「每期 Sharpe」= mean/std。
    # 旧实现传 x.mean()（IC 均值 ≈ -0.005），与 sr0（Sharpe 单位）量纲不一致，
    # 会把 DSR 压在地板值上，使多重检验门禁在构造上不可通过。
    _sd = float(x.std(ddof=1)) if x.size > 2 else 0.0
    sr_pp = float(x.mean() / _sd) if _sd > 0 else float("nan")
    dsr = deflated_sharpe(sr_pp, x.size, n_trials)
    mon = rolling_edge_monitor(x, MonitorConfig(lags=lags))
    return {
        "ic_mean": base["mean"], "hac_se": base["se"], "t": base["t"], "p": base["p"],
        "ci95": [base["lo"], base["hi"]],
        "boot_ci95": [boot.get("lo"), boot.get("hi")],
        "p_positive_boot": boot.get("p_positive"),
        "n_obs": base["n"], "effective_n": int(base["n"] / max(horizon, 1)),
        "sharpe_per_period": sr_pp,
        "deflated_sharpe": dsr,
        "frac_time_active": mon["frac_active"], "final_state": mon["final_state"],
        "verdict": _verdict(base, dsr, mon),
        "subperiods": subperiod_table(x, dates, lags),
    }


def _verdict(base: Dict, dsr: Dict, mon: Dict) -> str:
    if not np.isfinite(base["t"]):
        return "样本不足"
    if base["t"] >= 2.0 and dsr.get("significant"):
        return "有统计显著优势（已扣多重检验）"
    if base["t"] >= 2.0:
        return "看似显著，但扣掉多重检验后不显著"
    if base["t"] <= -2.0:
        return "显著为负（方向系统性做反）"
    return "无统计优势（与随机不可区分）"
