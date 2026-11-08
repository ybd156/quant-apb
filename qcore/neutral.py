"""
行业中性化 —— 把标签或因子对「行业哑变量」回归，取残差。

## 为什么需要

A 股横截面里，行业是最大的共同因子之一。2021-2026 这段样本里，
白酒、半导体、光伏、银行轮动幅度远超个股 alpha 量级：
同一天里"买半导体还是买银行"能解释掉横截面收益方差的 20%-40%。
不剥离行业，模型很容易学成「行业轮动预测器」，而不是「行业内选股器」。

对标签做行业内去均值（= 对行业哑变量做含截距 OLS 后取残差），
等价于问模型：**在同行业的股票里，哪些会涨得更多？**

## 口径（严格因果）

行业中性化是**横截面**运算：第 t 日的残差只用第 t 日的行业归属与该日标签，
不涉及任何跨日统计，因此天然满足 INV-1（无未来函数）。
注意它不是「全样本统计量」—— leak_scan 里的全样本告警针对的是时间维统计。

行业归属本身用**当前**行业分类（东财行业），这是本项目已知的轻微前视：
A 股行业分类极少变动，且不含任何价格信息，影响可忽略（无幸存者偏差放大效应）。
真正带前视的是「用今日市值排名筛历史股票」，那是另一件事（INV-6）。

## 退化情形

- 行业内只有 1 只股票 → 残差恒为 0（含截距回归的必然结果，不是 bug）。
  所以小样本股票池上做行业中性化会**损失**信息，需要看组内样本数。
- 行业未知（映射缺失）→ 归入 "NA" 组，等价于「未知行业之间互相去均值」。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["industry_neutralize", "industry_rank", "group_codes", "group_sizes"]


def group_codes(industry: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """把行业名数组编码成 0..G-1 的整数。

    返回 (codes, G, uniq)。缺失（None / NaN / 空串）统一编码为一个 "NA" 组。
    """
    arr = np.asarray(industry, dtype=object).ravel()
    clean = np.empty(arr.size, dtype=object)
    for i, v in enumerate(arr):
        if v is None:
            clean[i] = "NA"
        elif isinstance(v, float) and not np.isfinite(v):
            clean[i] = "NA"
        else:
            s = str(v).strip()
            clean[i] = s if s and s.lower() not in ("nan", "none", "null") else "NA"
    uniq, codes = np.unique(clean.astype(str), return_inverse=True)
    return codes.astype(np.int64), int(uniq.size), uniq


def group_sizes(industry: np.ndarray) -> np.ndarray:
    """每组（行业）的股票数，按 group_codes 的顺序。"""
    codes, G, _ = group_codes(industry)
    cnt = np.bincount(codes, minlength=G).astype(np.int64)
    return cnt


def _neutralize_1d(y: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    m = np.isfinite(y)
    vals = np.where(m, y, 0.0)
    s = np.bincount(codes, weights=vals, minlength=G)
    c = np.bincount(codes, weights=m.astype(np.float64), minlength=G)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(c > 0, s / np.maximum(c, 1.0), 0.0)
    out = y - mu[codes]
    out[~m] = np.nan
    return out


def _neutralize_2d(y: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    """(T,N)：逐行（逐日）做行业内去均值。

    用「行偏移 + bincount」一次算完 T*G 个组均值，避免 Python 逐日循环。
    """
    T, N = y.shape
    key = (np.arange(T, dtype=np.int64)[:, None] * G + codes[None, :]).ravel()
    m = np.isfinite(y)
    vals = np.where(m, y, 0.0)
    s = np.bincount(key, weights=vals.ravel(), minlength=T * G).reshape(T, G)
    c = np.bincount(key, weights=m.astype(np.float64).ravel(),
                    minlength=T * G).reshape(T, G)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(c > 0, s / np.maximum(c, 1.0), 0.0)
    out = y - mu[:, codes]
    out[~m] = np.nan
    return out


def industry_neutralize(y: np.ndarray, industry: np.ndarray) -> np.ndarray:
    """把标签 y 对行业哑变量回归，取残差（= 行业内去均值）。

    参数
    ----
    y : (N,) 或 (T,N)
        标签。NaN 视为缺失，不参与组均值，且残差保持 NaN。
        二维时第一维是时间、第二维是股票，行业按**同一套股票顺序**对齐。
    industry : (N,)
        行业名数组（str/object），长度须等于 y 的最后一维。

    返回
    ----
    与 y 同形状的残差。组内只有 1 个有效值时残差为 0（含截距 OLS 的必然结果）。

    说明
    ----
    含截距的行业哑变量回归残差 = y - mean(y | industry)，
    所以这里不需要真的构造哑变量矩阵、不需要解线性方程 —— 组内去均值就是闭式解。
    """
    y_arr = np.asarray(y, dtype=np.float64)
    codes, G, _ = group_codes(industry)
    N = y_arr.shape[-1]
    if codes.size != N:
        raise ValueError(f"industry 长度 {codes.size} 与 y 最后一维 {N} 不一致")
    if y_arr.ndim == 1:
        return _neutralize_1d(y_arr, codes, G)
    if y_arr.ndim == 2:
        return _neutralize_2d(y_arr, codes, G)
    raise ValueError(f"y 必须是 (N,) 或 (T,N)，收到 {y_arr.shape}")


def _rank_pct_within_row(vals: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    """单行内：按组分别做 (rank-0.5)/n 的百分位排名，NaN 保持 NaN。

    n 只数**该组内的有效值**（NaN 不占名次）。实现上把 NaN 换成 +inf 后按
    (组, 值) 做一次 lexsort：组内值有序 → 组内名次 = 当前位置 − 该组起点。
    """
    n = vals.size
    m = np.isfinite(vals)
    out = np.full(n, np.nan, dtype=np.float64)
    if not m.any():
        return out
    v = np.where(m, vals, np.inf)
    order = np.lexsort((v, codes))      # 先按组、再按值
    cs = codes[order]                   # 组已连续（组内值升序，NaN 的 inf 排在组尾）
    # 每组在 order 中的起点：cs 升序，用 searchsorted 一次算出
    starts = np.searchsorted(cs, np.arange(G), side="left")
    pos = np.arange(n, dtype=np.int64)
    rank0 = pos - starts[cs]            # 组内 0-based 名次
    cnt = np.bincount(cs[np.isfinite(v[order])], minlength=G).astype(np.float64)  # 只数有效值
    cnt = np.maximum(cnt, 1.0)
    pct = (rank0 + 0.5) / cnt[cs]
    tmp = np.full(n, np.nan, dtype=np.float64)
    tmp[order] = pct
    out[m] = tmp[m]
    return out


def industry_rank(x: np.ndarray, industry: np.ndarray) -> np.ndarray:
    """在每个行业内做横截面 rank(pct)，NaN 保持 NaN。

    参数
    ----
    x : (N,) 或 (T,N)
    industry : (N,)

    返回
    ----
    同形状，取值 (0,1)。采用「中位秩」口径 (rank-0.5)/n：
    与 pandas ``rank(pct=True)``（= rank/n）只差 0.5/n 的常数偏移，
    但永不触及 0/1 端点，避免后续 ndtri / logit 类变换爆到 ±inf。

    与 ``industry_neutralize`` 的区别
    --------------------------------
    - 中性化：线性去均值，保留幅度信息（残差可正可负，量纲同 y）
    - 行业内 rank：只保留组内相对次序，天然抗极值，但丢掉分布形状
    """
    x_arr = np.asarray(x, dtype=np.float64)
    codes, G, _ = group_codes(industry)
    N = x_arr.shape[-1]
    if codes.size != N:
        raise ValueError(f"industry 长度 {codes.size} 与 x 最后一维 {N} 不一致")
    if x_arr.ndim == 1:
        return _rank_pct_within_row(x_arr, codes, G)
    if x_arr.ndim == 2:
        out = np.full_like(x_arr, np.nan)
        for t in range(x_arr.shape[0]):
            out[t] = _rank_pct_within_row(x_arr[t], codes, G)
        return out
    raise ValueError(f"x 必须是 (N,) 或 (T,N)，收到 {x_arr.shape}")


# ---------------------------------------------------------------------- #
# 自检
# ---------------------------------------------------------------------- #
def _selfcheck() -> None:
    ind = np.array(["银行", "银行", "银行", "白酒", "白酒", "NA"], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan])

    r = industry_neutralize(y, ind)
    assert np.allclose(r[:3], [-1.0, 0.0, 1.0]), r
    assert np.allclose(r[3:5], [-5.0, 5.0]), r
    assert np.isnan(r[5])
    print("industry_neutralize 1D  ok:", np.round(r, 4))

    # 2D：逐行独立去均值
    y2 = np.vstack([y, y * 2.0])
    r2 = industry_neutralize(y2, ind)
    assert r2.shape == y2.shape
    assert np.allclose(r2[1], 2.0 * r, equal_nan=True), r2
    print("industry_neutralize 2D  ok: 行间独立")

    # 行业内 rank
    p = industry_rank(np.array([3.0, 1.0, 2.0, 7.0, np.nan, 9.0]), ind)
    assert np.allclose(p[:3], [5 / 6, 1 / 6, 3 / 6]), p
    assert np.isnan(p[4])
    assert p[3] == 0.5, f"组内只有 1 个有效值时 rank 应为 0.5，实际 {p[3]}"
    print("industry_rank           ok:", np.round(p, 4))

    # 单元素行业 -> 残差 0
    r3 = industry_neutralize(np.array([5.0, 1.0, 2.0]), np.array(["X", "Y", "Y"]))
    assert r3[0] == 0.0
    print("单元素行业退化          ok: 残差=0")

    # 与显式 OLS 对比（构造哑变量，最小二乘闭式解）
    rng = np.random.default_rng(0)
    N = 200
    ind2 = rng.choice(["A", "B", "C", "D", "E"], size=N)
    yv = rng.standard_normal(N) + np.repeat([1.0, -2.0, 0.5, 0.0, 3.0], N // 5)
    codes, G, uniq = group_codes(ind2)
    D = np.zeros((N, G))
    D[np.arange(N), codes] = 1.0
    D = np.hstack([np.ones((N, 1)), D[:, 1:]])       # 丢掉一个哑变量避免共线
    beta, *_ = np.linalg.lstsq(D, yv, rcond=None)
    resid = yv - D @ beta
    assert np.allclose(industry_neutralize(yv, ind2), resid, atol=1e-10)
    print("与显式 OLS 残差一致     ok: max|diff| < 1e-10")
    print("全部断言通过 ✓")


if __name__ == "__main__":
    import sys

    for _s in (sys.stdout, sys.stderr):   # Windows 控制台默认 GBK
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _selfcheck()
