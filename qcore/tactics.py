"""
操盘手法库 —— 把"江湖经验"变成可检验的统计特征。

## 立场

操盘手法（洗盘、诱多、对倒、打板、首阴……）在民间被讲成玄学。
但它们大多是**可观测行为的模式**，只要有明确定义，就能测：
有没有超额收益、t 值多少、五分位价差多大。

本模块只做两件事：
  1. 给出每个手法的**可计算定义**（避免"看图说话"）
  2. 标注它**需要什么数据**——日线能测的标 `data="daily"`，
     必须分钟/盘口才能测的标 `data="intraday"`，**不假装能用日线测出来**

## 诚实边界

- 日线数据看不到"尾盘拉升""大单扫货""对倒"的真实过程，只能用代理变量，
  代理与真实现象的相关系数**未知**，结论只能算线索。
- 所有手法都是**事后归因友好、事前预测困难**——这正是要实测的原因。
- 涨停/跌停造成的截断会污染收益分布，测试时必须用截断感知的口径。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Tactic:
    tid: str
    name: str
    data: str          # daily | intraday
    logic: str         # 手法逻辑（民间说法）
    testable: str      # 可检验的量化定义
    caution: str = ""


TACTICS: List[Tactic] = [
    Tactic("tac_washout", "洗盘结束", "daily",
           "主力打压洗掉浮筹后再拉，特征是缩量回调后放量收阳",
           "近 5 日最大回撤 > 4%，且回调期均量 < 60 日均量 70%；当日放量收阳且涨幅 > 2%",
           "日线看不到盘中打压过程，只能看结果，容易事后归因"),
    Tactic("tac_bull_trap", "诱多假突破", "daily",
           "先突破前高吸引跟风，随后迅速跌回，套住追高者",
           "创 20 日新高后 3 日内收盘跌回突破前高点之下",
           "突破当日与回落当日难以区分，需次日确认"),
    Tactic("tac_first_red", "首阴", "daily",
           "连涨多日后第一根阴线，多空分歧点，常是短线转折",
           "连涨 ≥3 日后首次收阴，且当日跌幅 < 5%（非恐慌）",
           "强趋势中首阴常是加仓点而非卖点，方向不稳定"),
    Tactic("tac_limit_chain", "打板接力", "daily",
           "涨停后次日溢价，接力资金推动连板",
           "当日涨幅 ≥ 9.5% 视为涨停；统计次日开盘溢价与 5 日延续性",
           "A 股涨停有制度截断，收益分布严重右偏，均值会被少数极端值主导"),
    Tactic("tac_gap_fill", "缺口回补", "daily",
           "向上跳空后大概率回补，回补处是支撑/压力",
           "跳空幅度 > 0.5% 且缺口未回补，统计回补所需天数与回补后走势",
           "缺口定义依赖前收盘，除权日会误判，必须先复权"),
    Tactic("tac_ma_squeeze", "均线粘合突破", "daily",
           "多条均线收敛后放量突破，方向选择往往较猛",
           "MA5/MA10/MA20 的极差 / 收盘价 < 3%，且当日放量突破上沿",
           "粘合后也可能向下破位，方向需由突破方向决定"),
    Tactic("tac_vol_dry", "地量见地价", "daily",
           "成交量萎缩到极致说明抛压枯竭，常伴随底部",
           "成交量创 60 日新低，且价格处于 250 日区间下三分之一",
           "地量之后可能还有更地量，时间不可预测"),
    Tactic("tac_upper_shadow", "长上影试盘", "daily",
           "冲高回落留下长上影，是试盘或出货，两种含义相反",
           "上影线长度 > 实体 2 倍，且当日振幅 > 4%",
           "试盘与出货在日线上无法区分，必须结合位置与量能"),
    Tactic("tac_wash_trade", "对倒放量", "daily",
           "自买自卖制造活跃假象，特征是量大价不动",
           "换手率 > 60 日均值 2 倍，但 |涨跌幅| < 1%",
           "日线只能看到'量大价不动'，无法确认是自成交"),
    Tactic("tac_closing_ramp", "尾盘拉升", "intraday",
           "收盘前 30 分钟快速拉高，做收盘价或吸引次日跟风",
           "需分钟数据：最后 30 分钟收益 / 全日收益 > 50%",
           "**日线数据无法测量**，只能用'收盘价接近最高价'做弱代理"),
]


def _roll(s: pd.Series, w: int, fn: str = "mean", minp: Optional[int] = None) -> np.ndarray:
    r = s.rolling(w, min_periods=minp or max(2, w // 3))
    return getattr(r, fn)().to_numpy()


def tactic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入日线（date/open/high/low/close/volume/amount/turnover），
    输出每个手法的**触发标志**（1=触发，0=未触发）与强度（0-1）。
    列名格式：`<tid>`（方向 +1 语义）与 `<tid>__str`（强度）。
    """
    d = df.reset_index(drop=True)
    o = d["open"].to_numpy(float); h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float); c = d["close"].to_numpy(float)
    v = d["volume"].to_numpy(float)
    turn = d["turnover"].to_numpy(float) if "turnover" in d else np.full(len(d), np.nan)

    n = len(d)
    ret = np.full(n, np.nan); ret[1:] = c[1:] / c[:-1] - 1.0
    rng = np.where(h > lo, (h - lo) / np.maximum(c, 1e-9), np.nan)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)

    v60 = _roll(pd.Series(v), 60, "mean")
    v20min = _roll(pd.Series(v), 20, "min")
    ma5 = _roll(pd.Series(c), 5); ma10 = _roll(pd.Series(c), 10); ma20 = _roll(pd.Series(c), 20)
    hi20 = _roll(pd.Series(c), 20, "max"); hi250 = _roll(pd.Series(c), 250, "max")
    lo250 = _roll(pd.Series(c), 250, "min")

    out = pd.DataFrame(index=pd.DatetimeIndex(d["date"]))
    S = lambda x: pd.Series(x)

    # 1) 洗盘结束：近 5 日回撤 + 缩量 + 当日放量收阳
    dd5 = (S(c) / S(c).rolling(5, min_periods=3).max() - 1.0).to_numpy()
    vshrink = (S(v).rolling(5, min_periods=3).mean().to_numpy() < 0.7 * v60)
    up_vol = (ret > 0.02) & (v > 1.2 * v60)
    out["tac_washout"] = ((dd5 < -0.04) & vshrink & up_vol).astype(float)
    out["tac_washout__str"] = np.clip(-dd5 * 5, 0, 1) * out["tac_washout"]

    # 2) 诱多假突破：创 20 日新高后 3 日内跌回
    brk = (c >= hi20)
    back = np.zeros(n)
    for i in range(3, n):
        if brk[i - 3]:
            back[i] = 1.0 if c[i] < hi20[i - 3] else 0.0
    out["tac_bull_trap"] = -back           # 看空
    out["tac_bull_trap__str"] = back * 0.8

    # 3) 首阴：连涨 >=3 日后首次收阴
    streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        streak[i] = streak[i - 1] + 1 if (np.isfinite(ret[i]) and ret[i] > 0) else 0
    first_red = np.zeros(n)
    for i in range(1, n):
        if ret[i] < 0 and streak[i - 1] >= 3 and ret[i] > -0.05:
            first_red[i] = 1.0
    out["tac_first_red"] = -first_red      # 先按"看空"编码，方向由实测决定
    out["tac_first_red__str"] = first_red * 0.6

    # 4) 打板：涨停标志（方向由次日/后续延续性决定，这里给 +1 语义）
    limit_up = (ret >= 0.095).astype(float)
    out["tac_limit_chain"] = limit_up
    out["tac_limit_chain__str"] = limit_up * np.clip(np.nan_to_num(ret) / 0.10, 0, 1)

    # 5) 缺口回补：跳空后仍未回补
    gap_up = np.zeros(n); gap_filled = np.zeros(n)
    prev_c = np.r_[np.nan, c[:-1]]
    gap_up = ((o > prev_c * 1.005)).astype(float)
    for i in range(1, n):
        if gap_up[i - 1] and lo[i] <= prev_c[i - 1]:
            gap_filled[i] = 1.0
    out["tac_gap_fill"] = np.where(gap_up == 1, 1.0, 0.0)
    out["tac_gap_fill__str"] = np.clip(np.abs(o / np.maximum(prev_c, 1e-9) - 1) * 20, 0, 1) * gap_up

    # 6) 均线粘合突破
    spread = np.nanmax(np.vstack([ma5, ma10, ma20]), axis=0) - np.nanmin(np.vstack([ma5, ma10, ma20]), axis=0)
    squeeze = np.where(np.isfinite(spread), spread / np.maximum(c, 1e-9) < 0.03, False)
    brk_up = c > np.nanmax(np.vstack([ma5, ma10, ma20]), axis=0)
    out["tac_ma_squeeze"] = (squeeze & brk_up & (v > 1.2 * v60)).astype(float)
    out["tac_ma_squeeze__str"] = out["tac_ma_squeeze"] * np.clip(v / np.maximum(v60, 1e-9) / 2, 0, 1)

    # 7) 地量见地价
    pos250 = (c - lo250) / np.maximum(hi250 - lo250, 1e-9)
    out["tac_vol_dry"] = ((v <= v20min) & (pos250 < 0.33)).astype(float)
    out["tac_vol_dry__str"] = out["tac_vol_dry"] * np.clip(1 - pos250, 0, 1)

    # 8) 长上影试盘
    us = (upper > 2 * body) & (rng > 0.04)
    out["tac_upper_shadow"] = -us.astype(float)     # 冲高回落按看空编码
    out["tac_upper_shadow__str"] = np.where(us, np.clip(upper / np.maximum(rng * c, 1e-9), 0, 1), 0.0)

    # 9) 对倒放量
    if np.isfinite(turn).any():
        t60 = _roll(pd.Series(turn), 60, "mean")
        wash = (turn > 2 * t60) & (np.abs(np.nan_to_num(ret)) < 0.01)
    else:
        wash = np.zeros(n, dtype=bool)
    out["tac_wash_trade"] = -wash.astype(float)     # 量大价不动按看空编码
    out["tac_wash_trade__str"] = wash.astype(float) * 0.5

    # 10) 尾盘拉升（日线弱代理：收盘接近最高）
    close_pos = (c - lo) / np.maximum(h - lo, 1e-9)
    out["tac_closing_ramp"] = ((close_pos > 0.9) & (v > 1.1 * v60)).astype(float)
    out["tac_closing_ramp__str"] = np.where(close_pos > 0.9, np.clip(close_pos, 0, 1), 0.0)

    return out.replace([np.inf, -np.inf], np.nan)


def list_tactics() -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in TACTICS])


def needs_intraday() -> List[str]:
    return [t.tid for t in TACTICS if t.data == "intraday"]
