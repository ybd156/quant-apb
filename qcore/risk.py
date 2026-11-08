"""
风险分摊 —— 让"多信号/多股票"真的分散风险，而不是看起来分散。

## 为什么需要这一层

审计已经证明：73 个信号里 **有效独立信号只有 33 个**，21% 是镜像对，
条件相关 p90 = 0.95。也就是说"73 个专家投票"实际是"33 个方向 + 40 维噪声"。
如果按数量平均，等于把风险重复押在同一个方向上。

同理，选 10 只股票也不等于分散——它们可能全在同一个行业、同一波 beta 里。

## 两个层次的分摊

**① 信号层：风险预算（Risk Budgeting）**
不按"信号个数"分配权重，而按"风险贡献"分配：
    每个信号族的风险贡献 RC_i = w_i·(Σw)_i / (wᵀΣw)
    目标：让各族 RC 相等（ERC, Equal Risk Contribution）
用迭代不动点求解：
    w_i ← w_i · (target_i / RC_i)，再归一化

**② 持仓层：逆波动 + 上限约束**
    base_w_i ∝ 1/σ_i          （波动大的少配）
    单票上限 max_weight        （防单点爆雷）
    行业上限 max_sector        （防行业集中）
用注水法（water-filling）迭代满足约束。

## 输出

`diversification_report` 给出：赫芬达尔指数、有效独立数、最大单一敞口，
并直接判断"这个组合到底算不算分散"。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ 工具
def hhi(w: np.ndarray) -> float:
    """赫芬达尔指数（0~1，越大越集中）。"""
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 1.0
    w = w / w.sum()
    return float((w ** 2).sum())


def effective_n_corr(w: np.ndarray, corr: np.ndarray) -> float:
    """
    考虑相关性的有效独立下注数：N_eff = (Σw)² / (wᵀΣw)，Σ 为相关矩阵。

    这是**比 1/HHI 更诚实的分散度指标**。审计实测：73 个信号按权重看 effN≈70，
    但按相关性看只有 33（21% 是镜像对、条件相关 p90=0.95）。
    只看权重会得出"很分散"的错误结论 —— 这正是"表面分散、实际集中"的典型。
    """
    w = np.asarray(w, dtype=np.float64)
    c = np.asarray(corr, dtype=np.float64)
    if w.size != c.shape[0] or w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    denom = float(w @ c @ w)
    return float(1.0 / denom) if denom > 0 else float("nan")


def effective_n(w: np.ndarray) -> float:
    """有效独立数 = 1/HHI。10 个等权 → 10；一个占 90% → 约 1.2。"""
    h = hhi(w)
    return float(1.0 / h) if h > 0 else 1.0


# ------------------------------------------------------------------ ERC
def equal_risk_contribution(cov: np.ndarray, iters: int = 300,
                            tol: float = 1e-8) -> np.ndarray:
    """
    等风险贡献权重（ERC）。cov 为协方差矩阵（k×k）。
    迭代不动点：w_i ← w_i · (target / RC_i)，target = 1/k。
    """
    c = np.asarray(cov, dtype=np.float64)
    k = c.shape[0]
    if k == 0:
        return np.array([])
    if k == 1:
        return np.array([1.0])
    # 数值稳定：加一点对角正则
    c = c + np.eye(k) * max(np.trace(c) / k * 1e-8, 1e-12)
    w = np.full(k, 1.0 / k)
    target = 1.0 / k
    for _ in range(iters):
        mrc = c @ w                      # 边际风险贡献
        port_var = float(w @ mrc)
        if port_var <= 0:
            break
        rc = w * mrc / port_var
        adj = np.where(rc > 0, target / np.maximum(rc, 1e-12), 1.0)
        w_new = w * np.clip(adj, 0.5, 2.0)   # 限步长，防震荡
        w_new = np.maximum(w_new, 1e-8)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def cap_weights(w: np.ndarray, cap: float, iters: int = 100) -> np.ndarray:
    """单资产上限（注水法）：超过上限的部分按比例分给未触顶的资产。"""
    w = np.asarray(w, dtype=np.float64).copy()
    if w.sum() <= 0:
        return np.full_like(w, 1.0 / len(w))
    w /= w.sum()
    if cap * len(w) < 1.0:
        cap = 1.0 / len(w)          # 上限不可行时退化为等权
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over
        if not free.any():
            break
        w[free] += excess * (w[free] / w[free].sum())
    return w / w.sum()


# ------------------------------------------------------------------ 信号层
def family_payoff_matrix(per_signal_payoff: np.ndarray, family_map: Sequence[str]) -> Dict:
    """
    per_signal_payoff: (T, S) 每个信号每期的标准化收益（NaN 表示未发声）
    返回每个信号族的收益序列（族内取均值），用于估协方差。
    """
    fams = sorted(set(family_map))
    out = np.full((per_signal_payoff.shape[0], len(fams)), np.nan)
    for j, f in enumerate(fams):
        idx = [i for i, g in enumerate(family_map) if g == f]
        sub = per_signal_payoff[:, idx]
        with np.errstate(invalid="ignore"):
            out[:, j] = np.nanmean(sub, axis=1) if sub.size else np.nan
    return {"families": fams, "payoffs": out}


def family_risk_budget(family_payoffs: np.ndarray, families: Sequence[str],
                       max_family: float = 0.35, min_family: float = 0.02) -> Dict:
    """
    按 ERC 给信号族分配风险预算，再施加上下限。
    family_payoffs: (T, K)
    """
    p = np.asarray(family_payoffs, dtype=np.float64)
    k = p.shape[1]
    ok = [j for j in range(k) if np.isfinite(p[:, j]).sum() >= 30]
    if len(ok) < 2:
        w = np.full(k, 1.0 / max(k, 1))
        return {"weights": dict(zip(families, w)), "method": "equal_fallback",
                "note": "有效族不足，退化为等权"}

    sub = p[:, ok]
    cov = np.cov(np.nan_to_num(sub, nan=0.0), rowvar=False)
    w_ok = equal_risk_contribution(cov)
    w_ok = cap_weights(w_ok, max_family)

    w = np.zeros(k)
    for pos, j in enumerate(ok):
        w[j] = w_ok[pos]
    w = np.maximum(w, min_family)
    w /= w.sum()

    return {
        "weights": {f: float(wi) for f, wi in zip(families, w)},
        "method": "erc",
        "effective_families": float(effective_n(w)),
        "max_share": float(w.max()),
        "note": f"ERC 风险预算，单族上限 {max_family:.0%}",
    }


def cap_family_weights(signal_weights: np.ndarray, family_map: Sequence[str],
                       max_family: float = 0.30, iters: int = 50) -> np.ndarray:
    """
    信号层的正确用法：**只设上限，不重新分配 alpha**。

    为什么不直接用 ERC：ERC 假设"每个资产的预期收益相同"，只平衡风险。
    但信号权重是 **alpha 权重**（学习器已经判断了谁更准）。
    对 alpha 权重套 ERC，等于把"mom 更准"这个信息扔掉，再按波动率平均分 —— 本末倒置。

    所以这里只做一件事：某个族占比超过上限时，把超出部分按原比例分给其他族。
    alpha 排序不变，只削峰。
    """
    w = np.asarray(signal_weights, dtype=np.float64).copy()
    fams = np.array(family_map, dtype=object)
    if w.sum() <= 0:
        return w
    w /= w.sum()
    for _ in range(iters):
        shares = {f: w[fams == f].sum() for f in set(fams.tolist())}
        over = {f: v for f, v in shares.items() if v > max_family + 1e-9}
        if not over:
            break
        for f, v in over.items():
            m = fams == f
            scale = max_family / v
            freed = w[m].sum() * (1 - scale)
            w[m] *= scale
            other = ~m
            if other.any():
                w[other] += freed * (w[other] / w[other].sum())
    return w / w.sum()


def apply_family_budget(signal_weights: np.ndarray, family_budget: Dict[str, float],
                        family_map: Sequence[str]) -> np.ndarray:
    """
    把学习器给出的信号权重，重标定到族风险预算上：
    族内按原权重比例分配，族间按预算分配。

    ⚠️ 注意：这是"强制风险预算"，会覆盖学习器的 alpha 判断。
    信号层一般应该用 `cap_family_weights`（只削峰），而不是本函数。
    """
    w = np.asarray(signal_weights, dtype=np.float64)
    fams = np.array(family_map)
    out = np.zeros_like(w)
    for f, budget in family_budget.items():
        m = fams == f
        if not m.any():
            continue
        s = w[m].sum()
        out[m] = (w[m] / s * budget) if s > 0 else (budget / m.sum())
    s = out.sum()
    return out / s if s > 0 else w


# ------------------------------------------------------------------ 持仓层
def portfolio_weights(scores: np.ndarray, sigma: np.ndarray,
                      sectors: Optional[Sequence[str]] = None, top_k: int = 10,
                      max_weight: float = 0.15, max_sector: float = 0.35,
                      long_only: bool = True) -> Dict:
    """
    选 top_k 后按逆波动配权，再施加单票/行业上限。
    scores (N,)；sigma (N,) 该股波动率；sectors (N,) 行业标签。
    """
    sc = np.asarray(scores, dtype=np.float64)
    sg = np.asarray(sigma, dtype=np.float64)
    n = sc.size
    valid = np.isfinite(sc) & np.isfinite(sg) & (sg > 0)
    if valid.sum() < 2:
        return {"weights": np.zeros(n), "ok": False, "reason": "insufficient_valid"}

    idx = np.where(valid)[0]
    if long_only:
        idx = idx[sc[idx] > 0]
        if idx.size < 2:
            return {"weights": np.zeros(n), "ok": False, "reason": "no_positive_scores"}
        pick = idx[np.argsort(-sc[idx])][:top_k]
    else:
        pick = idx[np.argsort(-np.abs(sc[idx]))][:top_k]

    inv_vol = 1.0 / sg[pick]
    w = inv_vol / inv_vol.sum()
    w = cap_weights(w, max_weight)

    # 行业上限
    if sectors is not None:
        sec = np.array([sectors[i] for i in pick], dtype=object)
        for _ in range(50):
            over = False
            for s in set(sec.tolist()):
                m = sec == s
                if w[m].sum() > max_sector + 1e-9:
                    scale = max_sector / w[m].sum()
                    freed = w[m].sum() - w[m].sum() * scale
                    w[m] *= scale
                    other = ~m
                    if other.any():
                        w[other] += freed * (w[other] / w[other].sum())
                    over = True
            if not over:
                break
        w = cap_weights(w, max_weight)

    out = np.zeros(n)
    out[pick] = w / w.sum()
    return {"weights": out, "picked": pick.tolist(), "ok": True,
            "n_positions": int(len(pick)),
            "sector_share": _sector_share(out, sectors) if sectors is not None else None}


def _sector_share(w: np.ndarray, sectors: Sequence[str]) -> Dict[str, float]:
    sec = np.array(sectors, dtype=object)
    out = {}
    for s in set(sec.tolist()):
        v = float(w[sec == s].sum())
        if v > 1e-9:
            out[str(s)] = round(v, 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------ 体检
def diversification_report(weights: np.ndarray, family_map: Optional[Sequence[str]] = None,
                           sectors: Optional[Sequence[str]] = None,
                           min_effective: float = 5.0) -> Dict:
    """这个组合/权重到底算不算分散。"""
    w = np.asarray(weights, dtype=np.float64)
    rep = {"hhi": hhi(w), "effective_n": effective_n(w),
           "max_share": float(w.max()) if w.size else 1.0}
    if family_map is not None:
        fams = np.array(family_map, dtype=object)
        fw = {}
        for f in set(fams.tolist()):
            v = float(w[fams == f].sum())
            if v > 1e-9:
                fw[str(f)] = round(v, 4)
        rep["family_share"] = dict(sorted(fw.items(), key=lambda kv: -kv[1]))
        rep["effective_families"] = effective_n(np.array(list(fw.values())))
    if sectors is not None:
        rep["sector_share"] = _sector_share(w, sectors)

    warns = []
    if rep["effective_n"] < min_effective:
        warns.append(f"有效独立数仅 {rep['effective_n']:.1f}，低于阈值 {min_effective}，"
                     "表面分散实则集中")
    if rep["max_share"] > 0.25:
        warns.append(f"最大单一敞口 {rep['max_share']:.0%}，超过 25%")
    if rep.get("effective_families", 99) < 3:
        warns.append(f"有效信号族仅 {rep.get('effective_families', 0):.1f}，"
                     "信号高度同质（审计实测有效独立信号 33/73）")
    rep["warnings"] = warns
    rep["verdict"] = "分散良好" if not warns else f"存在集中风险（{len(warns)} 项）"
    return rep
