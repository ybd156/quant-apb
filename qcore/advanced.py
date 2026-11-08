"""
进阶机制 —— 针对本项目实测出的三个病灶，各配一味药。

| 病灶（实测） | 机制 | 原理 |
|---|---|---|
| 有效独立信号仅 **12/73**，镜像对 21% | **信号正交化** | 贪心去相关，只留"新方向" |
| 固定 η=0.02、ρ=0.995，环境变了也不变 | **自适应学习率** | AdaGrad 思想：误差大就学慢 |
| 仓位靠经验阈值（回撤>10% 减仓） | **Kelly 分数** | 用胜率与赔率算最优下注 |

## 一、信号正交化

73 个信号里，`rsi_14_ob70` 和 `rsi_21_ob65` 几乎是同一个东西。
把它们都放进集成，等于给同一个观点投两票。

做法（贪心 Gram-Schmidt）：
    1. 按单信号 |IC| 排序
    2. 依次加入：若该信号与**已接受集合**的最大 |相关| < ρ_max，则接受
    3. 否则剔除（它没有带来新信息）

输出是一个"去相关后的信号子集"，有效独立数应显著提升。

## 二、自适应学习率（AdaGrad 式）

Hedge 的学习率 η 决定"多快相信新证据"。
固定 η 的问题是：市场平稳时学得太慢，剧烈变化时又学得太快（被噪声带跑）。

    η_t = η_0 / sqrt(1 + Σ_{s≤t} y_s²)

误差累积越大 → 学得越慢（已经乱的时候别再激进）。
这是 AdaGrad 的在线版本，无需调参、天然稳定。

## 三、Kelly 分数

给定胜率 p 与赔率 b（平均盈利/平均亏损），最优下注比例：

    f* = (p·b − q) / b,   q = 1 − p

性质：
- p·b < q 时 f* ≤ 0 → **不该下注**（这正是我们当前的状态）
- 实践中用 **分数 Kelly**（如 1/4），因为 p、b 都是估出来的，高估会爆仓
- 与 position.py 的阈值法互补：Kelly 给"理论上该下多少"，阈值法给"风险上该封顶多少"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# 一、信号正交化
# ======================================================================
@dataclass
class OrthoConfig:
    rho_max: float = 0.70      # 与已接受集合的最大允许相关
    min_ic: float = 0.005      # 单信号最低 |IC|（太弱的不进候选）
    max_keep: int = 40         # 最多保留多少个
    sample_days: int = 250     # 估算相关时抽样天数（全量 3360 天太慢，250 天足够）


def _sample_idx(T: int, k: int) -> np.ndarray:
    """等间隔抽样，保证覆盖全时段。"""
    if T <= k:
        return np.arange(T)
    return np.linspace(0, T - 1, k).astype(int)


def _daily_ic(x: np.ndarray, y: np.ndarray, min_names: int = 20,
              sample_days: int = 250) -> float:
    """单信号的平均横截面 Spearman（近似：用秩相关）。"""
    T, N = x.shape
    idx = _sample_idx(T, sample_days)
    ics = []
    for t in idx:
        a, b = x[t], y[t]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_names:
            continue
        ra = pd.Series(a[m]).rank().to_numpy().copy()
        rb = pd.Series(b[m]).rank().to_numpy().copy()
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        if d > 0:
            ics.append(float((ra * rb).sum() / d))
    return float(np.mean(ics)) if ics else float("nan")


def _xs_corr(x: np.ndarray, y: np.ndarray, min_names: int = 20,
             sample_days: int = 250) -> float:
    """两个信号的平均横截面相关（衡量"是不是同一个方向"）。"""
    T, N = x.shape
    idx = _sample_idx(T, sample_days)
    cs = []
    for t in idx:
        a, b = x[t], y[t]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_names:
            continue
        av, bv = a[m], b[m]
        if av.std() < 1e-12 or bv.std() < 1e-12:
            continue
        cs.append(float(np.corrcoef(av, bv)[0, 1]))
    return float(np.mean(cs)) if cs else 0.0


def orthogonalize(signals: Dict[str, np.ndarray], RET: np.ndarray,
                  cfg: OrthoConfig = OrthoConfig()) -> Dict:
    """
    贪心去相关：按 |IC| 排序，逐个检查与已接受集合的相关，超过 ρ_max 就剔除。
    signals: {sid: (T,N)}
    """
    names = list(signals)
    ics = {k: _daily_ic(signals[k], RET, sample_days=cfg.sample_days) for k in names}
    ranked = sorted([k for k in names if np.isfinite(ics[k]) and abs(ics[k]) >= cfg.min_ic],
                    key=lambda k: -abs(ics[k]))
    kept, dropped = [], []
    for k in ranked:
        if len(kept) >= cfg.max_keep:
            dropped.append({"sid": k, "reason": "超过 max_keep"})
            continue
        if not kept:
            kept.append(k)
            continue
        corrs = [abs(_xs_corr(signals[k], signals[j], sample_days=cfg.sample_days)) for j in kept]
        worst = max(corrs) if corrs else 0.0
        if worst < cfg.rho_max:
            kept.append(k)
        else:
            dropped.append({"sid": k, "reason": f"与已选信号相关 {worst:.2f} ≥ {cfg.rho_max}"})

    # 有效独立数：用 kept 的相关矩阵算 N_eff = (Σw)²/(wᵀΣw)，等权
    K = len(kept)
    if K >= 2:
        C = np.eye(K)
        for i in range(K):
            for j in range(i + 1, K):
                c = _xs_corr(signals[kept[i]], signals[kept[j]], sample_days=cfg.sample_days)
                C[i, j] = C[j, i] = c
        w = np.full(K, 1.0 / K)
        denom = float(w @ C @ w)
        n_eff = 1.0 / denom if denom > 0 else float(K)
    else:
        n_eff = float(K)

    n_before = min(30, len(names))
    eff_before = _effective_n_all(signals, names[:n_before], cfg.sample_days)
    diversity_before = eff_before / max(n_before, 1)
    diversity_after = n_eff / max(K, 1)
    return {"kept": kept, "dropped": dropped,
            "n_input": len(names), "n_kept": K,
            "effective_n": n_eff,
            "eff_before": eff_before, "n_before": n_before,
            "diversity_before": diversity_before,
            "diversity_after": diversity_after,
            "diversity_gain": (diversity_after / diversity_before - 1.0)
            if diversity_before > 0 else float("nan"),
            "n_eff_before": eff_before,
            "n_before_sampled": n_before,
            "ic_of_kept": {k: ics[k] for k in kept}}


def _effective_n_all(signals: Dict[str, np.ndarray], names: Sequence[str],
                     sample_days: int = 250) -> float:
    K = len(names)
    if K < 2:
        return float(K)
    w = np.full(K, 1.0 / K)
    tot = 0.0
    for i in range(K):
        for j in range(K):
            c = 1.0 if i == j else _xs_corr(signals[names[i]], signals[names[j]],
                                              sample_days=sample_days)
            tot += w[i] * w[j] * c
    return float(1.0 / tot) if tot > 0 else float(K)


# ======================================================================
# 二、自适应学习率
# ======================================================================
@dataclass
class AdaptiveEtaConfig:
    eta0: float = 0.02
    eps: float = 1e-8
    clip: Tuple[float, float] = (0.001, 0.10)


def adaptive_eta_series(payoffs: np.ndarray, cfg: AdaptiveEtaConfig = AdaptiveEtaConfig()) -> np.ndarray:
    """
    AdaGrad 式学习率：η_t = η_0 / sqrt(1 + Σ y²)。
    payoffs: (T,S) 每期标准化收益（NaN=未发声）。
    返回 (T,) 的学习率序列。
    """
    T = payoffs.shape[0]
    acc = 0.0
    out = np.empty(T)
    for t in range(T):
        y = payoffs[t]
        y = y[np.isfinite(y)]
        if y.size:
            acc += float((y * y).sum()) / y.size
        eta = cfg.eta0 / np.sqrt(1.0 + acc)
        out[t] = float(np.clip(eta, cfg.clip[0], cfg.clip[1]))
    return out


# ======================================================================
# 三、Kelly 分数
# ======================================================================
@dataclass
class KellyConfig:
    fraction: float = 0.25         # 分数 Kelly（保守）
    window: int = 250              # 估计 p、b 的窗口
    min_samples: int = 60
    max_fraction: float = 1.0      # 仓位上限（不含杠杆）
    max_leverage: float = 2.0


def kelly_fraction(returns: np.ndarray, cfg: KellyConfig = KellyConfig()) -> Dict:
    """
    用历史收益序列估计胜率 p 与赔率 b，算 Kelly 最优仓位。
    returns: 策略每期收益（1-D）。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)][-cfg.window:]
    if r.size < cfg.min_samples:
        return {"ok": False, "reason": f"样本 {r.size} < {cfg.min_samples}"}
    wins = r[r > 0]
    losses = r[r < 0]
    if wins.size == 0 or losses.size == 0:
        return {"ok": False, "reason": "样本中缺少盈利或亏损，无法估计赔率"}
    p = float(wins.size / r.size)
    q = 1.0 - p
    b = float(wins.mean() / abs(losses.mean()))     # 赔率
    f_star = (p * b - q) / b
    f_used = float(np.clip(f_star * cfg.fraction, 0.0, cfg.max_fraction))
    return {
        "ok": True, "win_rate": p, "payoff_ratio": b,
        "kelly_full": float(f_star), "kelly_used": f_used,
        "fraction": cfg.fraction, "n": int(r.size),
        "verdict": ("期望为负（p·b < q）→ Kelly 判定不该下注" if f_star <= 0 else
                    f"理论仓位 {f_star:.0%}，取 {cfg.fraction:.0%} 分数 Kelly = {f_used:.0%}"),
    }


def kelly_series(returns: np.ndarray, cfg: KellyConfig = KellyConfig()) -> np.ndarray:
    """滚动 Kelly 仓位序列（因果，只用过去数据）。"""
    r = np.asarray(returns, dtype=float)
    T = r.size
    out = np.zeros(T)
    for t in range(T):
        if t < cfg.min_samples:
            continue
        d = kelly_fraction(r[:t], cfg)
        out[t] = d["kelly_used"] if d.get("ok") else 0.0
    return out


# ======================================================================
# 组合：把三个机制接到一起
# ======================================================================
def apply_all(signals: Dict[str, np.ndarray], RET: np.ndarray,
              strategy_returns: np.ndarray,
              ortho_cfg: OrthoConfig = OrthoConfig(),
              kelly_cfg: KellyConfig = KellyConfig()) -> Dict:
    """一次性跑完三套机制，给出可执行的结论。"""
    o = orthogonalize(signals, RET, ortho_cfg)
    etas = adaptive_eta_series(np.column_stack([signals[k] for k in o["kept"]])
                               if o["kept"] else np.zeros((RET.shape[0], 1)),)
    k = kelly_fraction(strategy_returns, kelly_cfg)
    return {
        "orthogonalization": {
            "n_input": o["n_input"], "n_kept": o["n_kept"],
            "n_eff_before": o["n_eff_before"], "effective_n": o["effective_n"],
            "diversity_before": o["diversity_before"],
            "diversity_after": o["diversity_after"],
            "diversity_gain": o["diversity_gain"],
            "kept": o["kept"][:15], "n_dropped": len(o["dropped"]),
        },
        "adaptive_eta": {"final": float(etas[-1]) if etas.size else float("nan"),
                         "min": float(np.min(etas)) if etas.size else float("nan"),
                         "max": float(np.max(etas)) if etas.size else float("nan")},
        "kelly": k,
    }


def format_report(res: Dict) -> str:
    L = ["## 进阶机制实测", ""]
    o = res["orthogonalization"]
    L.append("### 一、信号正交化")
    L.append(f"- 输入 {o['n_input']} 个信号 → 保留 **{o['n_kept']}** 个，剔除 {o['n_dropped']} 个")
    L.append(f"- 每信号多样性（有效独立度/信号数）：**{o['diversity_before']:.2f} → "
             f"{o['diversity_after']:.2f}**（{o['diversity_gain']:+.0%}）")
    L.append(f"- 绝对有效独立数：{o['n_eff_before']:.1f}(抽 {o['n_before_sampled']} 个) → "
             f"{o['effective_n']:.1f}(留 {o['n_kept']} 个)")
    L.append(f"- 保留示例：{', '.join(o['kept'][:8])}")
    L.append("")
    e = res["adaptive_eta"]
    L.append("### 二、自适应学习率（AdaGrad 式）")
    L.append(f"- 末期 η = **{e['final']:.4f}**（区间 {e['min']:.4f} ~ {e['max']:.4f}）")
    L.append("- 误差累积越大 → 学得越慢；无需手调，天然稳定")
    L.append("")
    k = res["kelly"]
    L.append("### 三、Kelly 仓位")
    if k.get("ok"):
        L.append(f"- 胜率 {k['win_rate']:.1%}｜赔率 {k['payoff_ratio']:.2f}｜"
                 f"满 Kelly {k['kelly_full']:+.1%}")
        L.append(f"- **{k['verdict']}**")
    else:
        L.append(f"- 不可用：{k.get('reason')}")
    return "\n".join(L)
