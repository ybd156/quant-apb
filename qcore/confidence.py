"""
可信度判断 —— 不是"给个分数装作有把握"，而是**可校准、可检验**的概率。

三层证据，最后压成一个"这句话有多可信"的数字：

  ① 校准层（Calibration）：把综合分分箱，用**只有到期样本**的历史命中率做后验估计
     p_cal = (hits + κ·0.5) / (n + κ)
     这是"过去同样强度的分数，实际上对了多少次"。全部因果计算，绝不看未来。

  ② 一致层（Concordance）：当天发声的信号里，有多少比例的权重站在同一方向。
     一致度低 = 信号打架，即使分数大也不可信。

  ③ 质量层（Data Quality）：该股当日有多少信号真正算出值（预热期/停牌/数据缺失会降低）。

最终：
    p = 0.5 + (p_cal - 0.5) · γ(concord, dq)
γ ∈ [0,1] 是"证据折扣"：信号打架或数据差时，把结论往 0.5 拉回（即"我不知道"）。

输出同时给 Wilson 下界 —— 报告里用下界，而不是点估计，避免小样本吹牛。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ConfConfig:
    n_bins: int = 10
    kappa: float = 30.0          # 校准层的贝叶斯收缩强度（等效先验样本数）
    min_n_bin: int = 50          # 低于此样本量视为"证据不足"
    z_wilson: float = 1.96       # Wilson 下界置信度
    w_concord: float = 0.5       # 一致度对折扣的影响权重
    w_quality: float = 0.5       # 数据质量对折扣的影响权重
    quality_floor: float = 0.6   # 数据质量低于此值直接降级


def _wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return float((centre - adj) / denom)


def _bin_of(score: np.ndarray, n_bins: int) -> np.ndarray:
    """把 [-1,1] 的分数映射到 0..n_bins-1（按绝对值对称分箱）。"""
    s = np.clip(np.abs(score), 0.0, 1.0)
    return np.clip((s * n_bins).astype(int), 0, n_bins - 1)


def calibrate(scores: np.ndarray, RET: np.ndarray, h: int,
              warmup: int = 120, cfg: ConfConfig = ConfConfig()) -> Dict[str, np.ndarray]:
    """
    scores (T,N)；RET (T,N) 为 h 日收益。逐日推进，只用已到期样本更新分箱统计。
    返回 confidence (T,N)、lower (T,N)、n_support (T,N)、bin (T,N)。
    """
    T, N = scores.shape
    nb = cfg.n_bins
    hits = np.zeros(nb); tot = np.zeros(nb)

    conf = np.full((T, N), np.nan, dtype=np.float32)
    lower = np.full((T, N), np.nan, dtype=np.float32)
    nsup = np.zeros((T, N), dtype=np.int32)
    bins = np.full((T, N), -1, dtype=np.int8)

    for t in range(T):
        # ① 先用"到期样本"更新分箱（s = t-1-h 的预测此刻已知结果）
        s = t - 1 - h
        if s >= 0:
            sc, r = scores[s], RET[s]
            m = np.isfinite(sc) & np.isfinite(r) & (np.abs(sc) > 0)
            if m.any():
                b = _bin_of(sc[m], nb)
                hit = (np.sign(sc[m]) == np.sign(r[m])).astype(float)
                for k in range(nb):
                    sel = b == k
                    if sel.any():
                        hits[k] += hit[sel].sum()
                        tot[k] += sel.sum()

        # ② 再给当天打分（此时还没用到当天之后的信息）
        if t >= warmup:
            row = scores[t]
            m = np.isfinite(row)
            if m.any():
                b = _bin_of(row[m], nb)
                p = (hits[b] + cfg.kappa * 0.5) / (tot[b] + cfg.kappa)
                conf[t, m] = p.astype(np.float32)
                nsup[t, m] = tot[b].astype(np.int32)
                bins[t, m] = b.astype(np.int8)
                for i, (pi, ni) in enumerate(zip(p, tot[b])):
                    lower[t, np.where(m)[0][i]] = _wilson_lower(float(pi), int(ni), cfg.z_wilson)

    return {"confidence": conf, "lower": lower, "n_support": nsup, "bin": bins}


def combine(conf_cal: np.ndarray, concord: np.ndarray, quality: np.ndarray,
            cfg: ConfConfig = ConfConfig()) -> np.ndarray:
    """
    把"历史命中率"用"一致度 + 数据质量"做折扣，得到最终可信度。
    折扣 γ = clip(w_c·concord + w_q·quality, 0, 1)，低一致/低质量 → 结论拉回 0.5。
    """
    c = np.nan_to_num(concord, nan=0.5)
    q = np.nan_to_num(quality, nan=0.5)
    gamma = np.clip(cfg.w_concord * c + cfg.w_quality * q, 0.0, 1.0)
    p = 0.5 + (conf_cal - 0.5) * gamma
    return np.clip(p, 0.0, 1.0)


def level_of(p: float, n_support: int, cfg: ConfConfig = ConfConfig()) -> str:
    """可信度等级：给用户看的粗粒度标签，宁严勿宽。"""
    if not np.isfinite(p) or n_support < cfg.min_n_bin:
        return "insufficient"      # 证据不足
    lo = _wilson_lower(p, n_support, cfg.z_wilson)
    if lo >= 0.55:
        return "high"
    if lo >= 0.52:
        return "medium"
    if lo >= 0.48:
        return "low"
    return "no_edge"               # 统计上无优势


# ------------------------------------------------------------------ 校准质量
def calibration_report(p: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> Dict:
    """
    评估"我说的可信度"准不准：
      - Brier score（越低越好，0.25 = 永远说 50%）
      - ECE（期望校准误差：分箱内 |预测概率 − 实际命中率| 的加权平均）
      - 可靠性表（每个分箱的预测 vs 实际）
    """
    m = np.isfinite(p) & np.isfinite(hit)
    if m.sum() < 30:
        return {"ok": False, "reason": "too_few_samples", "n": int(m.sum())}
    pp, hh = p[m], hit[m].astype(float)
    brier = float(np.mean((pp - hh) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pp, edges) - 1, 0, n_bins - 1)
    rows, ece, wsum = [], 0.0, 0
    for k in range(n_bins):
        sel = idx == k
        if not sel.any():
            continue
        pred = float(pp[sel].mean()); act = float(hh[sel].mean()); n = int(sel.sum())
        ece += abs(pred - act) * n
        wsum += n
        rows.append({"bin": f"{edges[k]:.2f}-{edges[k+1]:.2f}", "n": n,
                     "pred": round(pred, 4), "actual": round(act, 4),
                     "gap": round(act - pred, 4)})
    ece = float(ece / max(wsum, 1))
    base = float(hh.mean())
    return {
        "ok": True, "n": int(m.sum()), "brier": round(brier, 5),
        "ece": round(ece, 4), "base_rate": round(base, 4),
        "brier_skill_vs_base": round(1 - brier / max(base * (1 - base), 1e-9), 4),
        "reliability": rows,
    }


def explain(p: float, n_support: int, concord: float, quality: float,
            regime: int, cfg: ConfConfig = ConfConfig()) -> str:
    """给用户一句话解释这个可信度是怎么来的（中文）。"""
    lv = level_of(p, n_support, cfg)
    tag = {"high": "较高", "medium": "中等", "low": "偏低",
           "no_edge": "无统计优势", "insufficient": "证据不足"}[lv]
    reg = {0: "下跌低波", 1: "下跌高波", 2: "上涨低波", 3: "上涨高波"}.get(int(regime), "未知")
    return (f"可信度{tag}（{p:.1%}，n={n_support}，{reg}环境）；"
            f"信号一致度 {concord:.0%}，数据完整度 {quality:.0%}。"
            + ("样本不足，仅供参考。" if lv == "insufficient" else ""))
