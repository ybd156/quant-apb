"""
特征筛选 —— 把 158 个 Alpha158 因子砍到 60 个左右，用来压制噪声与过拟合。

## 为什么需要

Alpha158 里有大量高度重复的因子（KMID / KLEN / KMID2 都是同一根 K 线的形状，
VMA5 / VMA10 / VMA20 / VMA30 / VMA60 是同一动量的五个尺度）。
在横截面 IC 只有 0.03 量级、信噪比极低的场景里，
**多出来的特征主要是在给树模型提供过拟合的自由度**，而不是信息。

两条互补的筛选路线：

1. ``select_by_importance`` —— 用已训练 GBDT 的增益重要性（事后剪枝）。
   便宜，但重要性是**样本内**的，且树模型会把重要性平摊给相关特征组
   （5 个 VMA 各得 1/5，看起来都很重要），所以单独用容易留一堆同质因子。

2. ``select_by_stability`` —— 滚动分折算每个特征的横截面 IC，
   取「|IC| 均值高 且 符号在各折间稳定」的因子（事前筛选）。
   本项目的核心用法：符号不稳定 = 这个因子在样本外会翻脸，直接丢掉。

## 因果性

- 分折是**按时间连续切块**，第 i 折只含第 i 段日期，没有任何跨折信息混入。
- 筛选只允许在训练窗口内做（调用方负责）；测试窗口必须完全不可见。
- 逐日 IC 是横截面统计（第 t 日只用第 t 日的横截面），不违反 INV-1。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

__all__ = ["select_by_importance", "select_by_stability",
           "daily_ic", "stability_scores"]


def daily_ic(X: np.ndarray, y: np.ndarray, min_names: int = 20) -> np.ndarray:
    """逐日横截面 Pearson IC。

    参数
    ----
    X : (T,N,K) 已标准化特征面板
    y : (T,N)   标签
    min_names : 当日有效样本数下限，不足则该日该特征为 NaN

    返回
    ----
    (T,K) 逐日 IC，NaN 表示无效。
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    T, N, K = X.shape
    out = np.full((T, K), np.nan, dtype=np.float64)
    for t in range(T):
        yb = y[t]
        if not np.isfinite(yb).any():
            continue
        blk = X[t]
        m = np.isfinite(blk) & np.isfinite(yb)[:, None]
        cnt = m.sum(axis=0)      # leak-scan: ok 逐日横截面（第 t 日内部统计，无跨日）
        ok = cnt >= min_names
        if not ok.any():
            continue
        xb = np.where(m, blk, 0.0).astype(np.float64)
        yv = np.where(m, yb[:, None], 0.0)
        c = np.maximum(cnt.astype(np.float64), 1.0)   # 空列用 1 兜底，下面会被 bad 屏蔽
        sx = xb.sum(axis=0)      # leak-scan: ok 逐日横截面
        sy = yv.sum(axis=0)      # leak-scan: ok 逐日横截面
        m_xy = (xb * yv).sum(axis=0)     # leak-scan: ok 逐日横截面
        m_xx = (xb * xb).sum(axis=0)     # leak-scan: ok 逐日横截面
        m_yy = (yv * yv).sum(axis=0)     # leak-scan: ok 逐日横截面
        cov = m_xy - sx * sy / c
        vx = m_xx - sx * sx / c
        vy = m_yy - sy * sy / c
        with np.errstate(divide="ignore", invalid="ignore"):
            ic = cov / np.sqrt(vx * vy)
        bad = (vx <= 0) | (vy <= 0) | (cnt == 0)
        ic[bad] = np.nan
        out[t][ok] = ic[ok]
    return out


def _fold_bounds(T: int, n_folds: int) -> List[tuple]:
    n_folds = max(2, int(n_folds))
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)
            if edges[i + 1] > edges[i]]


def _nanmean_cols(a: np.ndarray) -> np.ndarray:
    m = np.isfinite(a)
    c = m.sum(axis=0)        # leak-scan: ok 对"各折 IC"做汇总，不进入特征/标签
    s = np.where(m, a, 0.0).sum(axis=0)   # leak-scan: ok 同上，IC 汇总口径
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def stability_scores(X: np.ndarray, y: np.ndarray, names: List[str],
                     n_folds: int = 5, min_names: int = 20) -> Dict[str, np.ndarray]:
    """滚动分折的因子 IC 稳定性明细。

    返回 dict:
      mean_ic      (K,) 各折 IC 的均值（等权折平均）
      fold_ic      (F,K) 每折的 IC 均值
      consistency  (K,) 与 mean_ic 同号的折占比 ∈ [0,1]
      n_folds_ok   (K,) 有有效 IC 的折数
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    T, N, K = X.shape
    ic = daily_ic(X, y, min_names=min_names)          # (T,K)
    bounds = _fold_bounds(T, n_folds)
    fold_ic = np.full((len(bounds), K), np.nan, dtype=np.float64)
    for i, (a, b) in enumerate(bounds):
        fold_ic[i] = _nanmean_cols(ic[a:b])
    mean_ic = _nanmean_cols(fold_ic)
    with np.errstate(invalid="ignore"):
        sgn = np.sign(mean_ic)
        match = (np.sign(fold_ic) == sgn[None, :]) & np.isfinite(fold_ic)
    n_ok = np.isfinite(fold_ic).sum(axis=0)      # leak-scan: ok 有效折计数
    with np.errstate(divide="ignore", invalid="ignore"):
        consistency = np.where(n_ok > 0, match.sum(axis=0) / np.maximum(n_ok, 1), 0.0)  # leak-scan: ok 符号一致折计数
    return {"mean_ic": mean_ic, "fold_ic": fold_ic,
            "consistency": consistency, "n_folds_ok": n_ok}


def select_by_stability(X: np.ndarray, y: np.ndarray, names: List[str],
                        n_folds: int = 5, top_n: int = 60,
                        min_consistency: float = 0.8,
                        min_names: int = 20) -> List[int]:
    """滚动分折，取 |IC| 均值最高且符号稳定的 top_n 个特征索引。

    参数
    ----
    X : (T,N,K) 已标准化（横截面 rank 正态化）的特征面板
    y : (T,N)
    n_folds : 时间上连续切成的折数
    top_n : 保留的特征数
    min_consistency : 符号一致折占比的门槛（0.8 = 5 折里至少 4 折同号）
    min_names : 当日该特征的有效样本数下限

    返回
    ----
    list[int]，长度 = min(top_n, K)，按 |mean_ic| 降序。

    规则
    ----
    ① 先取「符号一致率 ≥ min_consistency」的特征，按 |mean_ic| 降序排；
    ② 若合格特征不足 top_n，用剩下的按 |mean_ic| 降序补齐（并在报告里注明）。
    这样保证返回长度恒定，下游不用处理变长特征集。
    """
    st = stability_scores(X, y, names, n_folds=n_folds, min_names=min_names)
    mean_ic = st["mean_ic"]
    cons = st["consistency"]
    K = mean_ic.size
    top_n = int(min(max(1, top_n), K))
    abs_ic = np.abs(mean_ic)
    valid = np.isfinite(abs_ic)
    # 无效特征（全 NaN）排最后
    key = np.where(valid, abs_ic, -1.0)
    eligible = valid & (cons >= float(min_consistency))
    order_ok = np.argsort(-key * eligible, kind="stable")
    order_ok = [int(i) for i in order_ok if eligible[i]]
    rest = [int(i) for i in np.argsort(-key, kind="stable") if i not in set(order_ok)]
    return (order_ok + rest)[:top_n]


def select_by_importance(model, names: List[str], top_n: int = 60) -> List[int]:
    """按 GBDT 的归一化增益重要性取 top_n 的特征索引。

    参数
    ----
    model : 已 fit 的 ``qcore.gbdt.GBDT``（需要 ``feature_importance``）
    names : 特征名列表，长度须等于重要性长度
    top_n : 保留数量

    返回
    ----
    list[int]，按重要性降序。

    注意
    ----
    重要性是样本内的，且树模型会在相关特征组之间**平摊**增益：
    5 个 VMA 各得 1/5，看起来都"很重要"。所以这条路线适合做粗剪
    （158 -> 60），不适合做精剪（158 -> 10）。
    """
    imp = np.asarray(getattr(model, "feature_importance"), dtype=np.float64).ravel()
    if imp.size != len(names):
        raise ValueError(f"重要性长度 {imp.size} 与特征名数量 {len(names)} 不一致")
    if imp.size == 0:
        return []
    imp = np.where(np.isfinite(imp), imp, -np.inf)
    top_n = int(min(max(1, top_n), imp.size))
    order = np.argsort(-imp, kind="stable")[:top_n]
    return [int(i) for i in order]


# ---------------------------------------------------------------------- #
# 自检
# ---------------------------------------------------------------------- #
def _selfcheck() -> None:
    rng = np.random.default_rng(20261108)
    T, N, K = 400, 60, 20
    X = rng.standard_normal((T, N, K)).astype(np.float32)
    # 前 3 个特征带信号，其余纯噪声
    y = (0.6 * X[:, :, 0] - 0.4 * X[:, :, 1] + 0.25 * X[:, :, 2]
         + rng.standard_normal((T, N))).astype(np.float32)
    names = [f"f{i}" for i in range(K)]

    sel = select_by_stability(X, y, names, n_folds=5, top_n=5)
    print("select_by_stability ->", [names[i] for i in sel])
    assert set(sel[:3]) == {0, 1, 2}, f"应选出 f0/f1/f2，实际 {sel}"

    st = stability_scores(X, y, names, n_folds=5)
    print(f"  f0 IC={st['mean_ic'][0]:+.4f} 一致率={st['consistency'][0]:.2f}")
    print(f"  f5 IC={st['mean_ic'][5]:+.4f} 一致率={st['consistency'][5]:.2f}")
    assert st["consistency"][0] >= 0.8

    # 重要性筛选
    class _M:
        feature_importance = np.arange(K, dtype=np.float64)[::-1]
    sel2 = select_by_importance(_M(), names, top_n=4)
    assert sel2 == [0, 1, 2, 3], sel2
    print("select_by_importance ->", [names[i] for i in sel2])

    # NaN 特征：全 NaN 的特征必须排在最后，不能崩
    X[:, :, 7] = np.nan
    sel3 = select_by_stability(X, y, names, n_folds=5, top_n=K)
    assert sel3[-1] == 7, f"全 NaN 特征应排最后，实际 {sel3[-3:]}"
    print("全 NaN 特征排最后       ok")
    print("全部断言通过 ✓")


if __name__ == "__main__":
    import sys

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _selfcheck()
