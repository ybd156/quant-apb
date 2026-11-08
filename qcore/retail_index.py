"""
散户参与度指数 —— 把"宝妈指数""大学生指数""出租车司机指数"量化。

## 这类民间指标的共同逻辑

它们都不是"预测工具"，而是**参与度温度计**：

    当最不专业、最晚进场的那批人开始讨论股票时，
    说明能进场的钱基本已经进场了 —— 边际买盘耗尽，随后往往是顶部。

所以它们的正确用法是**反向**：参与度极高 → 未来收益偏低；
参与度冰点 → 未来收益偏高。这就是"人多的地方不要去"的量化版。

## 民间指标 vs 可测指标

| 民间说法 | 本质 | 我们能不能测 |
|---|---|---|
| 宝妈指数（宝妈群开始聊股） | 新增散户参与 | ❌ 无数据，只能用代理 |
| 大学生指数（大学生开户） | 新增开户数 | ❌ 中登月度数据，无公开 API |
| 出租车司机荐股 | 全民参与 | ❌ 不可测 |
| 菜市场指数 | 同上 | ❌ |
| **换手率极值** | 交易热度 | ✅ 全市场可算 |
| **涨停家数占比** | 投机情绪 | ✅ 可算 |
| **成交额相对分位** | 资金活跃度 | ✅ 可算 |
| **横截面收益离散度** | 分歧/轮动 | ✅ 可算 |
| **小单净流入占比** | 散户净买入 | ✅ 东财资金流（近 120 日） |
| **新基金发行规模** | 增量资金 | ❌ 需外部数据 |

**本模块只做能测的五个**，并明确写：这是代理，不是原指标。
代理与"宝妈真的在聊股"的相关系数**未知**，结论只能算线索。

## 关键纪律

参与度指数是**市场级**（时间序列）指标，不是个股级（横截面）指标。
它的检验方式与选股因子不同：看的是"指数未来收益"，不是"横截面 IC"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RetailIndexConfig:
    window: int = 250          # 滚动标准化窗口
    min_periods: int = 60
    limit_th: float = 0.095    # 涨停判定阈值（主板 10%，留一点余量）


def _roll_z(x: np.ndarray, win: int, minp: int) -> np.ndarray:
    s = pd.Series(x, dtype="float64")
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-3, 3).to_numpy()


def _roll_pct(x: np.ndarray, win: int, minp: int) -> np.ndarray:
    return pd.Series(x, dtype="float64").rolling(win, min_periods=minp).rank(pct=True).to_numpy()


def market_metrics(p3d, cfg: RetailIndexConfig = RetailIndexConfig()) -> pd.DataFrame:
    """
    从 Panel3D 直接算市场级参与度原始指标（全部因果）。
    返回 index=日期 的 DataFrame。
    """
    T, N = p3d.CLOSE.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.full((T, N), np.nan)
        ret[1:] = p3d.CLOSE[1:] / p3d.CLOSE[:-1] - 1.0

    valid = p3d.VALID
    turn = np.where(valid, np.nan, np.nan)  # 占位：turnover 不在 Panel3D，见下
    out = pd.DataFrame(index=pd.DatetimeIndex(p3d.dates))

    # 1) 涨停家数占比（用收益阈值近似；严格版需涨跌停价，日线够用）
    lim = (ret >= cfg.limit_th) & valid
    denom = valid.sum(axis=1)
    out["limit_up_ratio"] = np.where(denom > 0, lim.sum(axis=1) / np.maximum(denom, 1), np.nan)

    # 2) 横截面收益离散度（分歧度）
    out["dispersion"] = np.nanstd(np.where(valid, ret, np.nan), axis=1)

    # 3) 上涨家数占比（市场宽度）
    up = (ret > 0) & valid
    out["adv_ratio"] = np.where(denom > 0, up.sum(axis=1) / np.maximum(denom, 1), np.nan)

    # 4) 成交额代理：用 SIGMA 的倒数不合适，改由外部传入 amount；此处先用 |ret| 加权活跃度
    out["abs_ret_mean"] = np.nanmean(np.abs(np.where(valid, ret, np.nan)), axis=1)

    return out


def add_turnover(p3d, panel: Dict[str, pd.DataFrame],
                 cfg: RetailIndexConfig = RetailIndexConfig()) -> pd.DataFrame:
    """
    补上换手率与成交额（这两个不在 Panel3D 里，需要从原始日线取）。
    """
    m = market_metrics(p3d, cfg)
    T, N = p3d.CLOSE.shape
    turn = np.full((T, N), np.nan)
    amt = np.full((T, N), np.nan)
    for n, code in enumerate(p3d.codes):
        df = panel.get(code)
        if df is None:
            continue
        pos = np.searchsorted(p3d.dates, df["date"].to_numpy())
        ok = (pos < T) & (p3d.dates[np.clip(pos, 0, T - 1)] == df["date"].to_numpy())
        pos = pos[ok]
        if "turnover" in df:
            turn[pos, n] = df["turnover"].to_numpy(dtype=float)[ok]
        if "amount" in df:
            amt[pos, n] = df["amount"].to_numpy(dtype=float)[ok]

    m["turn_median"] = np.nanmedian(turn, axis=1)
    m["amount_sum"] = np.nansum(amt, axis=1) / 1e8          # 亿元
    m["turn_p90"] = np.nanpercentile(turn, 90, axis=1)      # 尾部热度
    return m


def retail_index(metrics: pd.DataFrame, cfg: RetailIndexConfig = RetailIndexConfig()) -> pd.DataFrame:
    """
    合成参与度指数（每个分项先滚动分位化，再等权平均）。
    返回带分项与综合分的 DataFrame；`rpi` 为综合参与度（0-1 分位尺度）。
    """
    m = metrics.copy()
    cols = []
    spec = {
        "limit_up_ratio": +1,     # 涨停多 = 投机热
        "turn_median": +1,        # 换手高 = 交易热
        "amount_sum": +1,         # 成交额大 = 资金活跃
        "turn_p90": +1,           # 尾部换手 = 题材炒
        "dispersion": +1,         # 分歧大 = 轮动快
        "adv_ratio": +1,          # 上涨家数多 = 情绪好
    }
    for k, sign in spec.items():
        if k in m.columns:
            m[f"{k}_pct"] = _roll_pct(m[k].to_numpy(), cfg.window, cfg.min_periods)
            cols.append(f"{k}_pct")

    if not cols:
        m["rpi"] = np.nan
        return m
    m["rpi"] = np.nanmean(m[cols].to_numpy(), axis=1)
    m["rpi_z"] = _roll_z(m["rpi"].to_numpy(), cfg.window, cfg.min_periods)
    return m


def contrarian_test(rpi: np.ndarray, index_ret: np.ndarray,
                    horizons: List[int] = (5, 20, 60)) -> pd.DataFrame:
    """
    反向指标检验：参与度分位 vs 指数未来收益。
    若为反向指标，高分组（Q5）的未来收益应显著低于低分组（Q1）。
    """
    from . import stats as ST
    rows = []
    n = len(rpi)
    for h in horizons:
        fwd = np.full(n, np.nan)
        if n > h:
            fwd[:n - h] = index_ret[h:] if index_ret.ndim == 1 else np.nan
            # 指数收益需按 h 日累计
            cum = np.full(n, np.nan)
            c = np.cumprod(1 + np.nan_to_num(index_ret))
            cum[:n - h] = c[h:] / c[:n - h] - 1.0
            fwd = cum
        m = np.isfinite(rpi) & np.isfinite(fwd)
        if m.sum() < 100:
            rows.append({"h": h, "ok": False, "reason": "insufficient"})
            continue
        r, y = rpi[m], fwd[m]
        q_lo, q_hi = np.quantile(r, [0.2, 0.8])
        lo = y[r <= q_lo]; hi = y[r >= q_hi]
        st_lo = ST.mean_tstat(lo, lags=max(h - 1, 0))
        st_hi = ST.mean_tstat(hi, lags=max(h - 1, 0))
        rows.append({
            "h": h, "ok": True, "n": int(m.sum()),
            "q1_mean": st_lo["mean"], "q1_t": st_lo["t"],
            "q5_mean": st_hi["mean"], "q5_t": st_hi["t"],
            "q5_minus_q1": st_hi["mean"] - st_lo["mean"],
            "contrarian_holds": bool(st_hi["mean"] < st_lo["mean"]),
        })
    return pd.DataFrame(rows)


def current_state(rpi_df: pd.DataFrame) -> Dict:
    """当前参与度状态（给研判用）。"""
    if "rpi" not in rpi_df or not np.isfinite(rpi_df["rpi"].iloc[-1]):
        return {"ok": False, "reason": "not_enough_history"}
    v = float(rpi_df["rpi"].iloc[-1])
    pct = float(_roll_pct(rpi_df["rpi"].to_numpy(), 250, 60)[-1])
    level = ("极高（历史上 80% 分位以上）" if pct > 0.8 else
             "偏高" if pct > 0.6 else
             "中性" if pct > 0.4 else
             "偏低" if pct > 0.2 else "冰点（历史 20% 分位以下）")
    signal = ("反向看空：参与度极高，边际买盘可能耗尽" if pct > 0.8 else
              "反向看多：参与度冰点，往往对应底部区域" if pct < 0.2 else
              "不表态：参与度中性，无反向信号")
    return {"ok": True, "rpi": v, "percentile": pct, "level": level,
            "signal": signal,
            "note": "这是市场级时间序列指标，不用于个股选择"}
