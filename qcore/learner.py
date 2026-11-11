"""
自学习权重器 —— 本系统的"参数体"。

一句话：把每个 (信号族 × 参数) 当成一位专家，每天按真实结果给它打分，
**对了加权重、错了减权重**，并按市场状态分层记忆，用漂移检测在失效时快速减配。

三套机制叠在一起，各管一件事（见 docs/01_project_survey.md 的取舍结论）：

  1. Hedge / 乘性权重 (MWU)      —— 管"累积对错"：logw += η·y，指数级放大正确专家。
                                    Freund & Schapire 的 sleeping experts：**没发声不惩罚**。
  2. Beta-Bernoulli 后验收缩      —— 管"样本少时别乱信"：R=(a+κ·p0)/(a+b+κ)，n 小则贴近 0.5。
  3. Page-Hinkley 漂移检测        —— 管"失效了要快减"：检测到收益均值下移 → 该专家记忆打折。

所有状态都是 (regime, signal) 矩阵；regime 表通过 γ 与全局表做分层收缩（n 小 → 更信全局）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


@dataclass
class WeightConfig:
    horizon: int = 5
    decay_rho: float = 0.995      # 遗忘因子（半衰期 ~138 次更新）
    kappa: float = 8.0            # 贝叶斯收缩强度（等效先验样本数）
    p0: float = 0.5               # 命中率先验
    eta: float = 0.02             # Hedge 学习率
    beta_R: float = 1.0           # 命中率项系数
    beta_E: float = 0.5           # 收益均值项系数
    beta_H: float = 1.0           # 累积对错项系数
    tau: float = 1.5              # 温度（越大越集中）
    eps_explore: float = 0.03     # 探索率（防止饿死，保留再发现能力）
    w_min: float = 0.005          # 单信号权重下限
    gamma_max: float = 0.6        # regime 表最大占比
    gamma_n0: float = 60.0        # regime 收缩强度
    clip_C: float = 3.0           # 单次标准化收益截断
    lw_decay: float = 0.9995      # 累积信用的每日衰减（近期优先）
    ph_delta: float = 0.02        # PH 容忍度
    ph_lambda: float = 8.0        # PH 报警阈值
    ph_reset: float = 0.35        # 报警时保留的记忆比例
    ph_enabled: bool = True


class AdaptiveWeighter:
    """逐日在线学习；只吃「已兑现」的样本（由调用方保证 embargo）。"""

    def __init__(self, n_signals: int, n_regimes: int = 4,
                 cfg: Optional[WeightConfig] = None, sids: Optional[list] = None):
        self.cfg = cfg or WeightConfig()
        self.S = int(n_signals)
        self.R = int(n_regimes)
        self.sids = list(sids) if sids is not None else [f"s{i}" for i in range(self.S)]

        # ---- 全局表
        self.a_g = np.zeros(self.S); self.b_g = np.zeros(self.S)
        self.E_g = np.zeros(self.S); self.logw_g = np.zeros(self.S)
        self.n_g = np.zeros(self.S)
        # ---- regime 表
        self.a_r = np.zeros((self.R, self.S)); self.b_r = np.zeros((self.R, self.S))
        self.E_r = np.zeros((self.R, self.S)); self.logw_r = np.zeros((self.R, self.S))
        self.n_r = np.zeros((self.R, self.S))
        # ---- Page-Hinkley 状态（只看全局流）
        #: 已学到的样本日（YYYY-MM-DD）。样本级幂等的唯一依据：
        #: 同一个 s 只学一次，重复调用 step_once 不会把同批样本再喂一遍（评审 R45）。
        self.learned_through: Optional[str] = None
        self.ph_mean = np.zeros(self.S); self.ph_sum = np.zeros(self.S)
        self.ph_min = np.zeros(self.S)
        self.alarms = np.zeros(self.S, dtype=int)

    # ------------------------------------------------------------ 读数
    def _hit_rate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        k = self.cfg.kappa
        return (a + k * self.cfg.p0) / (a + b + k)

    def _z_global(self) -> np.ndarray:
        c = self.cfg
        Rg = self._hit_rate(self.a_g, self.b_g)
        return c.beta_R * (2 * Rg - 1) + c.beta_E * self.E_g + c.beta_H * self.logw_g

    def _z_regime(self, regime: int) -> np.ndarray:
        c = self.cfg
        Rr = self._hit_rate(self.a_r[regime], self.b_r[regime])
        return c.beta_R * (2 * Rr - 1) + c.beta_E * self.E_r[regime] + c.beta_H * self.logw_r[regime]

    def weights(self, regime: int) -> np.ndarray:
        """返回 (S,) 归一化权重，含 regime 分层收缩、温度、探索项、下限。"""
        c = self.cfg
        z_g = self._z_global()
        z_r = self._z_regime(int(regime))
        n = self.n_r[int(regime)]
        gamma = c.gamma_max * n / (n + c.gamma_n0)
        z = (1.0 - gamma) * z_g + gamma * z_r
        z = np.clip(z, -30.0, 30.0)

        e = np.exp(c.tau * (z - z.max()))
        p = e / max(e.sum(), 1e-12)
        w = (1.0 - c.eps_explore) * p + c.eps_explore / self.S
        w = np.maximum(w, c.w_min)
        return w / w.sum()

    def mark_learned(self, day: str) -> None:
        d = str(day)[:10]
        if not d:
            return
        if self.learned_through is None or d > self.learned_through:
            self.learned_through = d

    def seen(self, day: str) -> bool:
        return bool(self.learned_through and str(day)[:10] <= self.learned_through)

    # ------------------------------------------------------------ 更新
    def update(self, regime: int, active: np.ndarray, payoff: np.ndarray) -> None:
        """
        active : (S,) bool   当日是否发声
        payoff : (S,) float  标准化收益 y = dir·r/(σ√h)，未发声处可为 NaN
        """
        c = self.cfg
        regime = int(regime)
        m = np.asarray(active, dtype=bool) & np.isfinite(payoff)
        if not m.any():
            self._decay_credit()
            return
        y = np.clip(np.asarray(payoff, dtype=np.float64), -c.clip_C, c.clip_C)

        # 1) 全局：衰减计数 + 命中/未命中 + 收益均值 + 乘性信用
        self.a_g[m] = c.decay_rho * self.a_g[m] + (y[m] > 0)
        self.b_g[m] = c.decay_rho * self.b_g[m] + (y[m] < 0)
        self.E_g[m] = c.decay_rho * self.E_g[m] + (1.0 - c.decay_rho) * y[m]
        self.logw_g[m] += c.eta * y[m]
        self.n_g[m] += 1.0

        # 2) regime 表同理
        self.a_r[regime, m] = c.decay_rho * self.a_r[regime, m] + (y[m] > 0)
        self.b_r[regime, m] = c.decay_rho * self.b_r[regime, m] + (y[m] < 0)
        self.E_r[regime, m] = c.decay_rho * self.E_r[regime, m] + (1.0 - c.decay_rho) * y[m]
        self.logw_r[regime, m] += c.eta * y[m]
        self.n_r[regime, m] += 1.0

        # 3) 漂移检测
        if c.ph_enabled:
            self._page_hinkley(m, y)

        self._decay_credit()

    def _decay_credit(self) -> None:
        self.logw_g *= self.cfg.lw_decay
        self.logw_r *= self.cfg.lw_decay

    def _page_hinkley(self, m: np.ndarray, y: np.ndarray) -> None:
        """单侧 PH：检测收益均值是否显著下移；报警则对记忆打折（快速减配）。"""
        c = self.cfg
        n = self.n_g.copy()
        n[m] += 0.0
        # 更新累计均值
        self.ph_mean[m] += (y[m] - self.ph_mean[m]) / np.maximum(self.n_g[m], 1.0)
        self.ph_sum[m] += (y[m] - self.ph_mean[m] - c.ph_delta)
        self.ph_min[m] = np.minimum(self.ph_min[m], self.ph_sum[m])
        alarm = m & ((self.ph_sum - self.ph_min) > c.ph_lambda)
        if alarm.any():
            k = c.ph_reset
            self.a_g[alarm] *= k; self.b_g[alarm] *= k
            self.E_g[alarm] *= k; self.logw_g[alarm] *= k
            for r in range(self.R):
                self.a_r[r, alarm] *= k; self.b_r[r, alarm] *= k
                self.E_r[r, alarm] *= k; self.logw_r[r, alarm] *= k
            self.ph_sum[alarm] = 0.0; self.ph_min[alarm] = 0.0
            self.alarms[alarm] += 1

    # ------------------------------------------------------------ 诊断
    def diagnostics(self) -> Dict[str, np.ndarray]:
        Rg = self._hit_rate(self.a_g, self.b_g)
        out = {
            "hit_rate": Rg,
            "n_updates": self.n_g.copy(),
            "mean_payoff": self.E_g.copy(),
            "credit": self.logw_g.copy(),
            "alarms": self.alarms.copy(),
            "z": self._z_global(),
        }
        for r in range(self.R):
            out[f"hit_rate_r{r}"] = self._hit_rate(self.a_r[r], self.b_r[r])
            out[f"n_updates_r{r}"] = self.n_r[r].copy()
        return out

    def top(self, k: int = 10, regime: Optional[int] = None) -> list:
        w = self.weights(regime if regime is not None else 0)
        idx = np.argsort(-w)[:k]
        d = self.diagnostics()
        return [{"sid": self.sids[i], "weight": float(w[i]),
                 "hit_rate": float(d["hit_rate"][i]),
                 "n_updates": int(d["n_updates"][i]),
                 "mean_payoff": float(d["mean_payoff"][i])} for i in idx]

    def state_dict(self) -> dict:
        return {
            "cfg": asdict(self.cfg), "S": self.S, "R": self.R, "sids": self.sids,
            "a_g": self.a_g, "b_g": self.b_g, "E_g": self.E_g, "logw_g": self.logw_g,
            "n_g": self.n_g, "a_r": self.a_r, "b_r": self.b_r, "E_r": self.E_r,
            "logw_r": self.logw_r, "n_r": self.n_r, "alarms": self.alarms,
            "learned_through": self.learned_through,
        }
