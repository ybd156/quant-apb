"""
带符号线性权重 —— 允许"反着用某个信号"。

## 为什么这是关键

引擎现有的 `AdaptiveWeighter` 用 softmax + `w_min > 0`，**所有权重恒为正**：
它只能决定"多听谁、少听谁"，不能决定"把谁的结论反过来"。

而本项目实测：73 个信号的集成 IC = −0.0185（系统性做反）。
正权重平均一群反着的信号，平均出来还是反的。

带符号线性模型（岭回归）能给每个信号**负系数**，等于逐个把反的信号翻过来。
实测（283 只 / 2013-2026 / h=5）：

    正权重集成   IC = −0.0185  (t = −2.48)
    带符号回归   IC = +0.0209  (t = +4.68)   价差 +0.4728% (t = +6.70)

## 工程约束（不能因为效果好就放松）

1. **严格因果**：每 `refit_every` 天用 ≤ t−1−h 的样本重训，绝不看未来
2. **岭正则**：73 个特征高度共线，不加 L2 系数会爆炸
3. **系数截断**：单系数限制在 [−C, C]，防止某个信号绑架整个模型
4. **样本标准化**：训练时标准化，预测时用同一组均值/方差（否则口径漂移）
5. **可序列化**：权重状态要能存盘，供每日增量使用

## 与 learner 的关系

不是替代，是**并存**：
- `AdaptiveWeighter`：在线、逐日更新、可解释"谁在变强"
- `SignedLinear`：批量、带符号、更强但更难解释

引擎里用 `mode="signed"` 切换，两者结果可直接对比。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SignedConfig:
    l2: float = 1.0                  # 岭正则强度
    coef_clip: float = 0.02          # 单系数绝对值上限
    refit_every: int = 20            # 重训间隔（交易日）
    min_train: int = 5000            # 最少训练样本数
    warmup: int = 120
    standardize: bool = True


def _xs_demean(x: np.ndarray) -> np.ndarray:
    """横截面去均值（与 IC 同口径）。"""
    return x - np.nanmean(x, axis=1, keepdims=True)


class SignedLinear:
    """
    带符号线性模型。状态：系数 w、特征均值/方差、上次重训日。
    """

    def __init__(self, n_features: int, cfg: SignedConfig = SignedConfig(),
                 names: Optional[List[str]] = None):
        self.K = int(n_features)
        self.cfg = cfg
        self.w = np.zeros(self.K)
        self.mu = np.zeros(self.K)
        self.sd = np.ones(self.K)
        self.last_fit_t = -10**9
        self.n_fit = 0
        self.names = names or [f"f{i}" for i in range(self.K)]

    # ---------------------------------------------------------------- 训练
    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        X: (n,K) 特征；y: (n,) 已去均值的收益。
        返回是否成功拟合。
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[m], y[m]
        if X.shape[0] < self.cfg.min_train:
            return False
        if self.cfg.standardize:
            self.mu = X.mean(axis=0)
            sd = X.std(axis=0)
            self.sd = np.where(sd > 1e-9, sd, 1.0)
            Xs = (X - self.mu) / self.sd
        else:
            Xs = X
        A = Xs.T @ Xs + self.cfg.l2 * np.eye(self.K)
        b = Xs.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        self.w = np.clip(w, -self.cfg.coef_clip, self.cfg.coef_clip)
        self.n_fit = int(X.shape[0])
        return True

    # ---------------------------------------------------------------- 预测
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.cfg.standardize:
            Xs = (X - self.mu) / self.sd
        else:
            Xs = X
        ok = np.isfinite(Xs).all(axis=1)
        out = np.full(X.shape[0], np.nan)
        out[ok] = Xs[ok] @ self.w
        return out

    def top_signals(self, k: int = 10) -> List[Dict]:
        """系数绝对值最大的信号（正=顺势用，负=反着用）。"""
        idx = np.argsort(-np.abs(self.w))[:k]
        return [{"sid": self.names[i], "coef": float(self.w[i]),
                 "direction": "反着用" if self.w[i] < 0 else "顺势用"} for i in idx]

    # ---------------------------------------------------------------- 状态
    def state_dict(self) -> Dict:
        return {"w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "K": self.K, "names": self.names, "n_fit": self.n_fit,
                "cfg": self.cfg.__dict__}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "SignedLinear":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = cls(d["K"], SignedConfig(**d.get("cfg", {})), d.get("names"))
        m.w = np.array(d["w"]); m.mu = np.array(d["mu"]); m.sd = np.array(d["sd"])
        m.n_fit = d.get("n_fit", 0)
        return m


def run_signed(DIR: np.ndarray, STR: np.ndarray, RET: np.ndarray, h: int = 5,
               sids: Optional[List[str]] = None,
               cfg: SignedConfig = SignedConfig()) -> Dict:
    """
    逐日重放：每 refit_every 天用已到期样本重训，其余时间用当前系数预测。
    DIR/STR: (T,N,S)；RET: (T,N)。
    """
    T, N, S = DIR.shape
    X = (DIR.astype(np.float32) * (STR.astype(np.float32) / 255.0)).astype(np.float64)
    model = SignedLinear(S, cfg, sids)
    scores = np.full((T, N), np.nan, dtype=np.float32)
    w_hist = np.zeros((T, S), dtype=np.float32)
    n_fit_hist = np.zeros(T, dtype=np.int64)

    for t in range(cfg.warmup, T):
        if (t - cfg.warmup) % cfg.refit_every == 0:
            t_end = t - 1 - h
            if t_end > 50:
                xs, ys = [], []
                for s in range(0, t_end):
                    x = X[s]
                    y = RET[s]
                    m = np.isfinite(y) & np.isfinite(x).all(axis=1)
                    if m.sum() >= 10:
                        xs.append(_xs_demean(x[m].T).T)
                        ys.append(y[m] - np.nanmean(y[m]))
                if xs:
                    model.fit(np.vstack(xs), np.concatenate(ys))
        w_hist[t] = model.w.astype(np.float32)
        n_fit_hist[t] = model.n_fit
        scores[t] = model.predict(X[t]).astype(np.float32)

    return {"scores": scores, "weights": w_hist, "n_fit": n_fit_hist,
            "model": model, "config": cfg}
