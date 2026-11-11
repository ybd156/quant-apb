"""
逐日重放引擎 —— 系统的"时间机器"（时间主序 + 零拷贝缓冲版）。

核心纪律（SPEC §0.0）：
    每天 t 严格按此顺序：
      ① 用「只到 t 为止」的信息算综合分 → 当天能拿到的全部
      ② 预测写进台账（封存）
      ③ 只有当 t-1-h 日的预测已到期，才把真实结果喂给学习器（对账）
      ④ 学习器加深/减弱权重 → 进入 t+1

性能设计（N100 目标）：
  - 每天只碰 `DIR[t]` / `STR[t]` 两块连续内存，不跨步；
  - 复用预分配缓冲，避免每日 malloc（5000 天 × 8 次分配会明显拖慢）；
  - 权重用 float32 参与运算，减少一半内存带宽；
  - 台账用向量化构造，不做逐行 append。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset import Panel3D
from .learner import AdaptiveWeighter, WeightConfig


@dataclass
class RunConfig:
    horizon: int = 5
    exec_lag: int = 1
    warmup: int = 120
    weight_cfg: WeightConfig = field(default_factory=WeightConfig)
    mode: str = "learned"            # learned | equal | random | placebo
    seed: int = 42
    score_thresh: float = 0.0
    record_ledger: bool = True
    start_t: int = 0                 # 从第几根 K 线开始打分
    from_t: int = 0                  # 从第几根 K 线开始循环（增量续跑，之前的更新已在状态里）
    n_regimes: int = 4               # 写死，不让全样本统计参与结构决策
    placebo_kind: str = "shift"      # shift=按 sid 循环平移(保留持续性) | sign=按 sid 翻符号


# ------------------------------------------------------------------ 主循环
def run_walkforward(p3d: Panel3D, cfg: RunConfig = RunConfig(),
                    weighter: Optional[AdaptiveWeighter] = None) -> Dict:
    h = cfg.horizon
    T, N, S = p3d.DIR.shape
    rng = np.random.default_rng(cfg.seed)

    DIR = p3d.DIR
    STR = p3d.STR
    is_u8 = (STR.dtype == np.uint8)
    scale = np.float32(1.0 / 255.0)

    if cfg.mode == "placebo":
        # 干净安慰剂：只切断"信号 → 未来收益"的对应关系，保留发声频率与时间自相关。
        if cfg.placebo_kind == "sign":
            sgn = rng.choice(np.array([-1, 1], dtype=np.int8), size=(1, 1, S))
            DIR = (DIR * sgn).astype(np.int8)
        else:
            DIR = np.empty_like(p3d.DIR)
            for si in range(S):
                off = int(rng.integers(max(20, h + 2), max(21, T // 2)))
                DIR[:, :, si] = np.roll(p3d.DIR[:, :, si], off, axis=0)

    RET_h = p3d.RET[h]
    SIGMA = p3d.SIGMA
    sqh = np.sqrt(float(h))

    w_engine = weighter or AdaptiveWeighter(
        S, n_regimes=cfg.n_regimes, cfg=cfg.weight_cfg, sids=p3d.sids)

    scores = np.full((T, N), np.nan, dtype=np.float32)
    concord = np.full((T, N), np.nan, dtype=np.float32)   # 信号一致度
    quality = np.full((T, N), np.nan, dtype=np.float32)   # 数据完整度
    w_hist = np.zeros((T, S), dtype=np.float32)
    n_active_hist = np.zeros(T, dtype=np.int16)

    # 复用缓冲
    buf_s = np.empty((N, S), dtype=np.float32)
    buf_v = np.empty((N, S), dtype=np.float32)
    buf_a = np.empty((N, S), dtype=np.float32)
    tmp = np.empty((N, S), dtype=np.float32)
    denom = np.empty(N, dtype=np.float32)
    num = np.empty(N, dtype=np.float32)
    agree_buf = np.empty((N, S), dtype=bool)

    for t in range(cfg.from_t, T):
        # ① 先预测
        if cfg.mode == "equal":
            w = np.full(S, 1.0 / S, dtype=np.float32)
        elif cfg.mode == "random":
            w = rng.random(S).astype(np.float32)
            w /= w.sum()
        else:
            w = w_engine.weights(int(p3d.REGIME[t])).astype(np.float32)
        w_hist[t] = w

        if t >= cfg.start_t:
            d = DIR[t]
            if is_u8:
                np.multiply(STR[t], scale, out=buf_s, dtype=np.float32)
            else:
                buf_s[:] = STR[t]
            np.multiply(d, buf_s, out=buf_v, dtype=np.float32)     # 投票 = 方向 × 强度
            act = d != 0
            np.multiply(act, w[None, :], out=tmp)
            denom[:] = tmp.sum(axis=1)
            np.multiply(buf_v, w[None, :], out=tmp)
            num[:] = tmp.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sc = np.where(denom > 1e-9, num / np.maximum(denom, 1e-9), np.nan)
            sc = np.where(p3d.VALID[t], sc, np.nan).astype(np.float32)
            scores[t] = sc
            # 一致度：发声信号中，权重站在综合分同方向的比例
            sgn = np.sign(np.nan_to_num(sc, nan=0.0)).astype(np.int8)
            np.equal(d, sgn[:, None], out=agree_buf)
            agree_buf &= act
            np.multiply(agree_buf, w[None, :], out=tmp)
            with np.errstate(invalid="ignore", divide="ignore"):
                cc = np.where(denom > 1e-9, tmp.sum(axis=1) / np.maximum(denom, 1e-9), np.nan)
            concord[t] = cc.astype(np.float32)
            # 数据完整度：可交易 + 至少 MIN_ACTIVE 个信号发声
            nact = act.sum(axis=1).astype(np.float32)
            quality[t] = (0.5 * p3d.VALID[t].astype(np.float32)
                          + 0.5 * np.minimum(1.0, nact / 5.0)).astype(np.float32)
            if t >= cfg.warmup:
                n_active_hist[t] = int(nact.mean())

        # ②③ 对账
        s = t - 1 - h
        if s >= 0 and cfg.mode in ("learned", "placebo"):
            r = RET_h[s]
            sg = SIGMA[s]
            ok = np.isfinite(r) & np.isfinite(sg) & (sg > 1e-9)
            if ok.any():
                d_s = DIR[s][ok]                     # (n,S) int8
                act_m = d_s != 0
                y_all = d_s * (r[ok][:, None] / (sg[ok][:, None] * sqh))
                cnt = act_m.sum(axis=0)
                active = cnt > 0
                payoff = np.full(S, np.nan, dtype=np.float64)
                if active.any():
                    y_m = np.where(act_m, y_all, 0.0)
                    payoff[active] = y_m.sum(axis=0)[active] / cnt[active]
                # 样本级幂等（R45）：已经学过的样本日直接跳过。
                # 没有这道门，`live --loop` 每小时会把同一批到期样本再学一遍。
                if not w_engine.seen(str(p3d.dates[s])[:10]):
                    w_engine.update(int(p3d.REGIME[s]), active, payoff)
                    w_engine.mark_learned(str(p3d.dates[s])[:10])

    diag = w_engine.diagnostics()
    # 对外统一为 (N,T) 朝向（与 SPEC v1.0 / metrics / backtest 一致）；
    # 引擎内部仍是 (T,N) 时间主序，转置只在出口做一次（几 MB）。
    out = {
        "dates": p3d.dates, "codes": p3d.codes, "sids": p3d.sids,
        "scores": np.ascontiguousarray(scores.T),
        "scores_TN": scores,
        "concord": np.ascontiguousarray(concord.T),
        "quality": np.ascontiguousarray(quality.T),
        "weights_hist": w_hist, "n_active_hist": n_active_hist,
        "REGIME": p3d.REGIME,
        "OPEN": np.ascontiguousarray(p3d.OPEN.T),
        "CLOSE": np.ascontiguousarray(p3d.CLOSE.T),
        "RET": {h: np.ascontiguousarray(v.T) for h, v in p3d.RET.items()},
        "SIGMA": np.ascontiguousarray(p3d.SIGMA.T),
        "VALID": np.ascontiguousarray(p3d.VALID.T),
        "diagnostics": diag, "weighter": w_engine,
        "meta": {**p3d.meta, "mode": cfg.mode, "horizon": h, "warmup": cfg.warmup,
                 "weight_cfg": cfg.weight_cfg.__dict__},
    }
    if cfg.record_ledger:
        out["ledger"] = build_ledger(p3d, scores, cfg)
    return out


# ------------------------------------------------------------------ 预测台账
def build_ledger(p3d: Panel3D, scores: np.ndarray, cfg: RunConfig) -> pd.DataFrame:
    """
    向量化构造对账表：预测字段在决策时封存，结果字段到期回填。
    只对 |score| > thresh 的记录入账。
    """
    h = cfg.horizon
    lag = cfg.exec_lag
    T, N = scores.shape
    ret_h = p3d.RET[h]

    mask = np.isfinite(scores) & (np.abs(scores) > cfg.score_thresh)
    if cfg.warmup:
        mask[:cfg.warmup] = False
    mask[T - lag - h:] = False

    ti, ni = np.nonzero(mask)
    if len(ti) == 0:
        return pd.DataFrame(columns=["as_of", "code", "score", "regime", "entry_date",
                                     "entry_price", "exit_date", "exit_price", "ret", "hit", "status"])
    df = pd.DataFrame({
        "as_of": p3d.dates[ti],
        "code": np.asarray(p3d.codes)[ni],
        "score": scores[ti, ni].astype(np.float32),
        "regime": p3d.REGIME[ti],
        "entry_date": p3d.dates[ti + lag],
        "entry_price": p3d.OPEN[ti + lag, ni],
        "exit_date": p3d.dates[ti + lag + h],
        "exit_price": p3d.OPEN[ti + lag + h, ni],
        "ret": ret_h[ti, ni],
    })
    # ret=NaN（未到期 / 停牌缺口）时 hit 必须是「未知」，不能记成 False。
    # 旧写法 np.sign(nan)==np.sign(nan) -> False，于是任何 sum(hit)/len(hit)
    # 都把未到期样本当「没命中」计入分母 —— 与 status="pending" 自相矛盾。
    df["hit"] = (np.sign(df["score"]) == np.sign(df["ret"])).where(
        df["ret"].notna(), other=None)
    df["status"] = np.where(df["ret"].notna(), "scored", "pending")
    return df


# ------------------------------------------------------------------ 模式对照
def run_modes(p3d: Panel3D, cfg: RunConfig = RunConfig()) -> Dict[str, Dict]:
    outs = {}
    for mode in ("learned", "equal", "random", "placebo"):
        c = RunConfig(**{**cfg.__dict__, "mode": mode})
        c.weight_cfg = cfg.weight_cfg
        outs[mode] = run_walkforward(p3d, c)
    return outs

# ------------------------------------------------------------------ 带符号模式
def run_signed_mode(p3d: Panel3D, cfg: RunConfig = RunConfig(),
                    signed_cfg=None) -> Dict:
    """
    带符号线性模式：允许负权重（逐个把反的信号翻过来）。
    与 run_walkforward 返回同样的结构，便于直接对比与接入 pipeline。
    """
    from .signed import SignedConfig, run_signed

    h = cfg.horizon
    T, N, S = p3d.DIR.shape
    sc = signed_cfg or SignedConfig(refit_every=20, warmup=cfg.warmup)
    out = run_signed(p3d.DIR, p3d.STR, p3d.RET[h], h=h, sids=p3d.sids, cfg=sc)

    scores = out["scores"]                     # (T,N)
    scores = np.where(p3d.VALID, scores, np.nan).astype(np.float32)

    # 与主引擎对齐：concord/quality 用简化版（带符号模型没有"投票一致性"概念）
    concord = np.full((T, N), np.nan, dtype=np.float32)
    quality = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T):
        act = p3d.DIR[t] != 0
        nact = act.sum(axis=1).astype(np.float32)
        quality[t] = (0.5 * p3d.VALID[t] + 0.5 * np.minimum(1.0, nact / 5.0)).astype(np.float32)

    diag = {"hit_rate": np.full(S, np.nan), "n_updates": out["n_fit"],
            "mean_payoff": np.zeros(S), "credit": out["model"].w,
            "alarms": np.zeros(S, dtype=int), "z": out["model"].w}

    res = {
        "dates": p3d.dates, "codes": p3d.codes, "sids": p3d.sids,
        "scores": np.ascontiguousarray(scores.T),
        "scores_TN": scores,
        "concord": np.ascontiguousarray(concord.T),
        "quality": np.ascontiguousarray(quality.T),
        "weights_hist": out["weights"],
        "n_active_hist": np.zeros(T, dtype=np.int16),
        "REGIME": p3d.REGIME,
        "OPEN": np.ascontiguousarray(p3d.OPEN.T),
        "CLOSE": np.ascontiguousarray(p3d.CLOSE.T),
        "RET": {k: np.ascontiguousarray(v.T) for k, v in p3d.RET.items()},
        "SIGMA": np.ascontiguousarray(p3d.SIGMA.T),
        "VALID": np.ascontiguousarray(p3d.VALID.T),
        "diagnostics": diag,
        "weighter": out["model"],
        "meta": {**p3d.meta, "mode": "signed", "horizon": h,
                 "signed_cfg": sc.__dict__},
    }
    if cfg.record_ledger:
        res["ledger"] = build_ledger(p3d, scores, cfg)
    return res
