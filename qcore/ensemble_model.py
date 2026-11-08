"""
多模型集成 —— 三种异构模型按**验证集 IC** 加权平均。

## 为什么是这三个

单棵 GBDT 在横截面 IC≈0.03 的场景下有两类系统性误差：

1. **树模型的方差**：换一个随机种子（行采样/列采样不同），
   预测的横截面排序会变，IC 会上下浮动 0.005-0.01 —— 与信号本身同量级。
   所以先做 **GBDT ×3 种子平均**，把种子噪声压掉一半左右（√3）。

2. **树模型的偏差**：GBDT 只会输出分段常数，学不到线性外推；
   而 Alpha158 里大部分因子（动量、波动、量比）与未来收益的关系
   在 rank 空间里近似单调线性。所以补一个**带符号岭回归**
   （允许负系数 —— 很多因子是反向的，例如波动率、换手率），
   它给出树模型给不出的平滑线性分量。

3. **共同的过拟合**：两者都在同一批 158 个因子上拟合。
   一个「所有因子等权 rank 平均」的**零参数基准**几乎不可能过拟合，
   把它按 IC 混进来相当于给集成加了一个收缩项（shrinkage prior）。

权重不是拍脑袋定的，而是**在验证集上按各分量的逐日横截面 IC 归一化**得到的。
验证集必须早于测试集（本项目 2019 vs 2021-2026），因此权重不含未来信息。

## 口径

- 各分量先做 z-score（用验证集预测的均值/标准差），再按权重相加。
  不做 z-score 的话，GBDT 输出量级（~1e-3）会被线性分量的量级（~1e-2）压死。
- 分量 IC ≤ 0 的权重被截到 0（"帮倒忙的模型不给票"）；全为 0 时退化为等权。
- ``equal`` 分量的 rank 在**每个横截面（每个交易日）内**做，
  所以 ``predict`` 的输入应当是一天的横截面；传 ``groups`` 可显式分组。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .gbdt import GBDT

__all__ = ["BlendModel", "daily_ic_rows"]


def daily_ic_rows(pred: np.ndarray, y: np.ndarray, days: np.ndarray,
                  min_names: int = 20) -> float:
    """按天分组的平均横截面 Pearson IC。

    pred / y / days 长度一致；days 是「该行属于哪一天」的标签（可任意可哈希值）。
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    days = np.asarray(days)
    ics: List[float] = []
    order = np.argsort(days, kind="stable")
    d_sorted = days[order]
    if d_sorted.size == 0:
        return float("nan")
    # 连续段 = 同一天
    cuts = np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1
    for seg in np.split(order, cuts):
        if seg.size < min_names:
            continue
        p, t = pred[seg], y[seg]
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() < min_names:          # leak-scan: ok 逐日横截面内的有效样本计数
            continue
        ps, ts = p[m], t[m]
        sp, st = ps.std(), ts.std()      # leak-scan: ok 逐日横截面内的标准差
        if sp <= 0 or st <= 0:
            continue
        ics.append(float(np.corrcoef(ps, ts)[0, 1]))
    ics = [v for v in ics if np.isfinite(v)]
    return float(np.mean(ics)) if ics else float("nan")   # leak-scan: ok 对逐日 IC 序列取均值（评估统计）


def _rank_pct(blk: np.ndarray) -> np.ndarray:
    """列方向 (rank-0.5)/n 的百分位排名，NaN 保持 NaN。"""
    df = pd.DataFrame(blk)
    r = df.rank()                       # 默认 average，NaN 不参与
    n = r.notna().sum(axis=0).to_numpy(dtype=np.float64)
    out = r.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (out - 0.5) / n[None, :]
    return out


class BlendModel:
    """GBDT(×k 种子) + 带符号岭回归 + 全特征等权 rank 的 IC 加权集成。

    参数
    ----
    n_gbdt : GBDT 个数（不同随机种子），默认 3
    seeds : 种子列表，长度须 ≥ n_gbdt
    n_estimators, max_depth, learning_rate, min_samples_leaf, subsample,
    colsample, l2, max_bins : 透传给 ``qcore.gbdt.GBDT``
    ridge_alpha : 岭回归 L2 强度（对已横截面标准化的特征，1.0 是温和收缩）
    verbose : 打印每个分量的验证集 IC 与权重

    用法
    ----
        m = BlendModel(n_estimators=250, max_bins=32)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), eval_days=dva)
        score = m.predict(Xte_day)        # 一天的横截面
        m.weights                          # {'gbdt': .6, 'linear': .3, 'equal': .1}
    """

    def __init__(self, n_gbdt: int = 3, seeds: Sequence[int] = (42, 202, 777),
                 n_estimators: int = 250, max_depth: int = 4,
                 learning_rate: float = 0.1, min_samples_leaf: int = 200,
                 subsample: float = 0.8, colsample: float = 0.8, l2: float = 1.0,
                 max_bins: int = 32, ridge_alpha: float = 1.0,
                 min_names: int = 20, verbose: bool = False) -> None:
        self.n_gbdt = int(n_gbdt)
        self.seeds = list(seeds)
        if len(self.seeds) < self.n_gbdt:
            raise ValueError("seeds 数量少于 n_gbdt")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.l2 = float(l2)
        self.max_bins = int(max_bins)
        self.ridge_alpha = float(ridge_alpha)
        self.min_names = int(min_names)
        self.verbose = bool(verbose)

        self.gbdts: List[GBDT] = []
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self._w: Dict[str, float] = {}
        self._scale: Dict[str, Tuple[float, float]] = {}
        self.ic_detail: Dict[str, float] = {}
        self.n_features_in_: int = 0

    # ------------------------------------------------------------------ #
    # 分量预测
    # ------------------------------------------------------------------ #
    def _gbdt_pred(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape[0], dtype=np.float64)
        for m in self.gbdts:
            out += m.predict(X)
        return out / max(1, len(self.gbdts))

    def _linear_pred(self, X: np.ndarray) -> np.ndarray:
        Xf = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        return Xf @ self.coef_ + self.intercept_

    def _equal_pred(self, X: np.ndarray, groups: Optional[np.ndarray]) -> np.ndarray:
        """所有特征 rank(pct) 后按行取均值（每个横截面内排名）。"""
        Xa = np.asarray(X, dtype=np.float64)
        n = Xa.shape[0]
        out = np.full(n, np.nan)
        if groups is None:
            r = _rank_pct(Xa)
            out = np.nanmean(r, axis=1)   # leak-scan: ok 同一横截面内跨特征平均
        else:
            g = np.asarray(groups)
            order = np.argsort(g, kind="stable")
            gs = g[order]
            cuts = np.flatnonzero(gs[1:] != gs[:-1]) + 1
            for seg in np.split(order, cuts):
                r = _rank_pct(Xa[seg])
                out[seg] = np.nanmean(r, axis=1)   # leak-scan: ok 同上，按日分组
        return out

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    def fit(self, X, y, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_days: Optional[np.ndarray] = None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError("X 必须是二维 (n, K)")
        if y.size != X.shape[0]:
            raise ValueError("X/y 行数不一致")
        n, K = X.shape
        self.n_features_in_ = K

        # ---- 1) GBDT × k（不同种子）----
        self.gbdts = []
        for i in range(self.n_gbdt):
            m = GBDT(n_estimators=self.n_estimators, max_depth=self.max_depth,
                     learning_rate=self.learning_rate,
                     min_samples_leaf=self.min_samples_leaf,
                     subsample=self.subsample, colsample=self.colsample,
                     l2=self.l2, max_bins=self.max_bins,
                     random_state=self.seeds[i])
            m.fit(X, y)
            self.gbdts.append(m)

        # ---- 2) 带符号岭回归（分块 float64 累加，避免 float32 累积误差）----
        ybar = float(y.mean())   # leak-scan: ok 训练窗口内标签均值（岭回归截距）
        A = np.zeros((K, K), dtype=np.float64)
        b = np.zeros(K, dtype=np.float64)
        chunk = 200_000
        for s in range(0, n, chunk):
            xb = np.nan_to_num(X[s:s + chunk].astype(np.float64), nan=0.0)
            A += xb.T @ xb
            b += xb.T @ (y[s:s + chunk] - ybar)
        A[np.diag_indices_from(A)] += self.ridge_alpha
        try:
            self.coef_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]
        self.intercept_ = ybar

        # ---- 3) 等权 rank 分量无需训练 ----

        # ---- 4) 验证集 IC -> 权重 ----
        self._fit_weights(X, y, eval_set, eval_days)
        if self.verbose:
            print(f"    BlendModel 权重 {self._fmt_weights()}  "
                  f"分量 IC {({k: round(v, 4) for k, v in self.ic_detail.items()})}",
                  flush=True)
        return self

    def _component_preds(self, X, groups=None) -> Dict[str, np.ndarray]:
        return {"gbdt": self._gbdt_pred(X),
                "linear": self._linear_pred(X),
                "equal": self._equal_pred(X, groups)}

    def _fit_weights(self, Xtr, ytr, eval_set, eval_days) -> None:
        names = ["gbdt", "linear", "equal"]
        if eval_set is not None:
            Xv = np.ascontiguousarray(eval_set[0], dtype=np.float32)
            yv = np.asarray(eval_set[1], dtype=np.float64).ravel()
        else:
            # 没有验证集：退化到训练集（会有轻微乐观，但不引入未来信息）
            Xv, yv = Xtr, ytr
        preds = self._component_preds(Xv, eval_days)
        ic = {}
        for k in names:
            ic[k] = daily_ic_rows(preds[k], yv, eval_days) if eval_days is not None \
                else self._pooled_ic(preds[k], yv)
        self.ic_detail = {k: float(v) for k, v in ic.items()}

        raw = np.array([max(0.0, ic[k]) if np.isfinite(ic[k]) else 0.0 for k in names])
        if raw.sum() <= 0:      # leak-scan: ok 验证集 IC 归一化（不进入特征/标签）
            w = np.full(len(names), 1.0 / len(names))
        else:
            w = raw / raw.sum()  # leak-scan: ok 同上
        self._w = {k: float(v) for k, v in zip(names, w)}

        # 各分量 z-score 的均值/标准差（用验证集预测，量级稳定）
        for k in names:
            p = preds[k]
            m = np.isfinite(p)
            mu = float(p[m].mean()) if m.any() else 0.0   # leak-scan: ok 验证集预测的 z-score 常数
            sd = float(p[m].std()) if m.any() else 1.0    # leak-scan: ok 同上
            self._scale[k] = (mu, sd if sd > 1e-12 else 1.0)

    @staticmethod
    def _pooled_ic(p: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() < 20:      # leak-scan: ok 有效样本计数
            return float("nan")
        a, b = p[m], y[m]
        if a.std() <= 0 or b.std() <= 0:   # leak-scan: ok 相关系数分母检查
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #
    def predict(self, X, groups: Optional[np.ndarray] = None) -> np.ndarray:
        """加权集成预测。

        X : (n,K)。n 通常是一个交易日的横截面；若一次传入多天，
            必须给 ``groups``（每日标签），否则 ``equal`` 分量的 rank 会跨日混排。
        """
        if self.coef_ is None:
            raise RuntimeError("模型未训练，先调用 fit")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        preds = self._component_preds(X, groups)
        out = np.zeros(X.shape[0], dtype=np.float64)
        wsum = 0.0
        for k, p in preds.items():
            w = self._w.get(k, 0.0)
            if w <= 0:
                continue
            mu, sd = self._scale[k]
            out += w * (np.nan_to_num(p, nan=mu) - mu) / sd
            wsum += w
        if wsum > 0:
            out /= wsum
        return out

    # ------------------------------------------------------------------ #
    @property
    def weights(self) -> Dict[str, float]:
        """各分量的 IC 归一化权重，如 {'gbdt': 0.6, 'linear': 0.3, 'equal': 0.1}。"""
        return dict(self._w)

    def _fmt_weights(self) -> str:
        return "{" + ", ".join(f"{k}: {v:.2f}" for k, v in self._w.items()) + "}"

    def __repr__(self) -> str:
        return (f"BlendModel(n_gbdt={self.n_gbdt}, trees={self.n_estimators}, "
                f"ridge_alpha={self.ridge_alpha}, weights={self._fmt_weights()})")


# ---------------------------------------------------------------------- #
# 自检
# ---------------------------------------------------------------------- #
def _selfcheck() -> None:
    rng = np.random.default_rng(20261108)
    T, N, K = 120, 80, 12
    # 造一个"线性 + 轻微非线性"的横截面收益，便于检验三个分量都有贡献
    X = rng.standard_normal((T, N, K)).astype(np.float32)
    y = (0.5 * X[:, :, 0] - 0.35 * X[:, :, 1]
         + 0.3 * np.tanh(2 * X[:, :, 2])
         + rng.standard_normal((T, N)) * 0.9).astype(np.float32)
    Xf = X.reshape(T * N, K)
    yf = y.reshape(T * N)
    days = np.repeat(np.arange(T), N)
    tr = days < int(T * 0.7)
    va = ~tr

    m = BlendModel(n_gbdt=3, n_estimators=40, min_samples_leaf=50,
                   max_bins=16, verbose=True)
    m.fit(Xf[tr], yf[tr], eval_set=(Xf[va], yf[va]), eval_days=days[va])
    w = m.weights
    print("weights:", {k: round(v, 3) for k, v in w.items()})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert set(w) == {"gbdt", "linear", "equal"}

    p = m.predict(Xf[va], groups=days[va])
    ic = daily_ic_rows(p, yf[va], days[va])
    print(f"验证集集成 IC = {ic:+.4f}")
    assert ic > 0.1, ic

    # 分量 IC 应都为正
    for k, v in m.ic_detail.items():
        print(f"  分量 {k:6s} IC={v:+.4f}")
        assert v > 0

    # 可复现性
    p2 = m.predict(Xf[va], groups=days[va])
    assert np.array_equal(p, p2)
    print("可复现性 ok")
    print("全部断言通过 ✓")


if __name__ == "__main__":
    import sys

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _selfcheck()
