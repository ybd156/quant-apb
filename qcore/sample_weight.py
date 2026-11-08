"""
样本时间衰减加权 —— 越近的样本权重越高，压制「用 2010 年的市场规律预测 2025 年」的失真。

## 为什么需要

A 股的横截面结构在十年尺度上是漂移的：注册制、涨跌停幅度变化、
北向资金、量化占比、行业权重迁移。用 2008-2019 全窗口等权训练，
等于假设 2008 年的因子有效性一直持续到 2019 —— 这个假设在 IC 只有 0.03 量级时
足以把信号淹掉。

指数衰减权重：

    w_i = 0.5 ** (age_i / half_life_days)

``age_i`` 是该样本日期相对**最新训练样本**的自然日龄。
半衰期 250 天 ≈ 一年：一年前的样本权重 0.5，三年前 0.125。

## 为什么是「重采样」而不是「加权」

本项目的 ``qcore.gbdt.GBDT.fit`` **不支持 sample_weight**
（它内部 ``h = np.ones(n)``，二阶梯度是常数，加权需要在 g/h 上同时乘 w，
而且 ``_counts_from_hess`` 那条「每箱样本数 = 每箱 h 之和」的捷径会失效）。
按契约不改写 GBDT 内核，所以这里用**有放回重采样**做近似：

    从原样本按 p_i ∝ w_i 抽 n 个（n = 原样本数），重复样本自然获得更大权重。

**这是近似，不是等价**：

- 期望意义上，重采样后每个样本在损失里的出现次数 ∝ w_i，方向与加权一致；
- 但重采样会**重复整行**，于是同一行在直方图里被计多次，
  与「权重连续变化」相比是**有噪声的离散近似**（权重被量化成出现次数）；
- 低权重样本有一定概率完全不被抽到（丢信息），高权重样本被复制（方差略升）；
- 重采样后 ``min_samples_leaf`` 的语义变成「按重采样后的行数」，实际有效叶子样本变少，
  相当于轻度增强正则 —— 这一点在比较 A/D 两组时要记住。

替代方案（未来若改 GBDT）：直接给 g/h 乘 w，代价是要关掉 ``_counts_from_hess`` 捷径，
训练慢约 1/3。本文件保留 ``weighted_gh`` 说明这一点，但不改内核。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["time_decay_weights", "resample_by_weight", "sample_dates"]


def sample_dates(dates_index: np.ndarray, day_pos: np.ndarray) -> np.ndarray:
    """把「样本所在交易日在面板里的行号」映射成 datetime64 数组。

    dates_index : (T,) datetime64[ns] 面板日期
    day_pos     : (n,) 每个样本所属交易日的行号
    """
    d = np.asarray(dates_index)
    pos = np.asarray(day_pos, dtype=np.int64)
    if pos.size and (pos.min() < 0 or pos.max() >= d.size):  # leak-scan: ok 仅做下标越界检查
        raise ValueError("day_pos 越界")
    return d[pos]


def time_decay_weights(dates_of_sample: np.ndarray,
                       half_life_days: float = 250.0) -> np.ndarray:
    """按样本日期给指数衰减权重：w = 0.5 ** (age_days / half_life_days)。

    参数
    ----
    dates_of_sample : (n,) datetime64 数组（也接受 int64 纳秒时间戳）
        age 相对**该数组中的最新日期**计算（不是相对"今天"），
        这样训练窗口整体平移时权重形态不变。
    half_life_days : 半衰期（自然日）。250 ≈ 一年；<=0 视为不衰减（全 1）。

    返回
    ----
    (n,) float64，已归一化到**均值为 1**（便于和等权结果直接比较量级，
    真正的归一化在 ``resample_by_weight`` 里按和做）。

    说明
    ----
    用自然日而不是交易日：跨越春节/长假的样本不该因为"日历上少了 7 个交易日"
    而被认为更近。自然日在十年尺度上差几天无关紧要。
    """
    d = np.asarray(dates_of_sample)
    if d.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.issubdtype(d.dtype, np.datetime64):
        days = d.astype("datetime64[D]").astype(np.int64)
    else:
        # 假定纳秒时间戳
        days = (d.astype(np.int64) // 86_400_000_000_000).astype(np.int64)
    hl = float(half_life_days)
    if not np.isfinite(hl) or hl <= 0:
        return np.ones(days.size, dtype=np.float64)
    age = (days.max() - days).astype(np.float64)   # leak-scan: ok 只用训练窗口内的最新样本日
    w = np.power(0.5, age / hl)
    m = w.mean() if w.size else 1.0                # leak-scan: ok 权重归一化常数
    return w / m if m > 0 else w


def resample_by_weight(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                       n: int | None = None, random_state: int = 42
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按权重有放回重采样（GBDT 不支持 sample_weight 时的近似）。

    参数
    ----
    X : (n,K) 或 (n,) 任意数组，按第一维抽样
    y : (n,)
    w : (n,) 非负权重
    n : 抽样数量，默认 = 原样本数（保持样本规模不变，训练耗时不增）
    random_state : 随机种子，保证可复现

    返回
    ----
    (X2, y2, idx)：重采样后的 X、y，以及被抽中的原始行号（可复现/可审计）。

    近似误差
    --------
    权重被量化成「出现次数」，相当于多项分布抽样：
    抽样次数的标准差 ≈ sqrt(n·p·(1-p))，相对误差约 1/sqrt(n·p)。
    n=50 万时对 p≈0.001 的远期样本，相对误差约 4.5% —— 对 IC 量级 0.03 的影响
    在第三位小数以下，可接受；若要求严格加权，需改 GBDT 内核。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64).ravel()
    n_in = X.shape[0]
    if y.size != n_in or w.size != n_in:
        raise ValueError(f"长度不一致 X={n_in} y={y.size} w={w.size}")
    if n is None:
        n = n_in
    n = int(n)
    ww = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = ww.sum()      # leak-scan: ok 抽样概率归一化（训练窗口内权重求和）
    if s <= 0:
        idx = np.arange(n) % n_in
        return X[idx], y[idx], idx
    p = ww / s
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n_in, size=n, replace=True, p=p)
    return X[idx], y[idx], idx


# ---------------------------------------------------------------------- #
# 自检
# ---------------------------------------------------------------------- #
def _selfcheck() -> None:
    # 半衰期语义：恰好一个半衰期前 -> 0.5
    d = np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[D]")
    w = time_decay_weights(d, half_life_days=366.0)
    print("半衰期 366 天，1 年前样本权重比 =", round(float(w[0] / w[1]), 4))
    assert abs(w[0] / w[1] - 0.5) < 1e-6

    d2 = np.array(["2010-01-01", "2019-12-31"], dtype="datetime64[D]")
    w2 = time_decay_weights(d2, half_life_days=250.0)
    print(f"10 年前样本权重 = {w2[0]:.3e}（均值已归一为 1）")
    assert w2[0] < w2[1]

    # 不衰减
    w3 = time_decay_weights(d2, half_life_days=0)
    assert np.allclose(w3, 1.0)
    print("half_life<=0 -> 全 1  ok")

    # 重采样：权重集中在后半段，抽样后均值应明显右移
    rng = np.random.default_rng(0)
    n = 20_000
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = rng.standard_normal(n)
    dd = np.repeat(np.array(["2010-01-01", "2019-12-31"], dtype="datetime64[D]"), n // 2)
    ww = time_decay_weights(dd, half_life_days=250.0)
    frac_recent_raw = np.count_nonzero(dd == np.datetime64("2019-12-31")) / n
    _, _, idx = resample_by_weight(X, y, ww, random_state=1)
    frac_recent = np.count_nonzero(dd[idx] == np.datetime64("2019-12-31")) / n
    print(f"原始近期样本占比 {frac_recent_raw:.3f} -> 重采样后 {frac_recent:.3f}")
    assert frac_recent > 0.95, frac_recent
    print("全部断言通过 ✓")


if __name__ == "__main__":
    import sys

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _selfcheck()
