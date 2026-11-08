"""
经典指标补全（第二辑）—— 把散落在各处的"成功经验"变成可检验的横截面变量。

## 收录原则

只收**有明确出处、有确定公式**的指标，不收"民间玄学"（那些进 `tactics.py`）。

| 类别 | 指标 | 出处 |
|---|---|---|
| 隔夜/日内分解 | overnight, intraday, 隔夜占比 | Lou, Polk & Skouras (2019) JFE |
| 趋势强度 | ADX, DI+/DI− | Wilder (1978) *New Concepts in Technical Trading Systems* |
| 摆动指标 | CCI, MFI, WR, PSY, BIAS, TRIX, ROC | Lambert (1980); Wilder (1978); Lane (1950s) |
| 量价 | OBV 斜率, 价量相关 | Granville (1963); Karpoff (1987) |
| 趋势质量 | Kaufman 效率比 | Kaufman (1995) |
| 价格延迟 | 收益自相关 | Hou & Moskowitz (2005) RFS |
| 风险 | 下行波动, 最大回撤 | Ang et al. (2006); 通用 |
| 量价因子 | Alpha#006, Alpha#012 | Kakushadze (2016) "101 Formulaic Alphas", arXiv:1601.00991 |

## 重要提醒（来自本项目自己的教训）

**IC 显著 ≠ 能赚钱，更 ≠ 有 alpha。**
本项目实测：amihud 原始价差 +0.566%(t=5.94)，行业+市值中性化后 +0.031%(t=0.40)
—— 全部是小市值效应的伪装。

所以本模块**只负责算**，是否有效必须过三道关：
    ① 秩相关 IC 显著（HAC t）
    ② 五分位价差同向且显著
    ③ **行业 + 市值中性化后仍然成立**

第三关不过的，一律视为"没有 alpha"。

## 因果与量纲

- 全部只用 ≤t 的数据
- 全部是相对量/滚动量，**禁止绝对价格阈值**（后复权价格与现价差 10 倍以上）
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# 指标元信息：名称 -> (说明, 文献方向, 出处)
INDICATOR_META: Dict[str, Dict] = {
    "i2_overnight": {"desc": "隔夜收益 open/prev_close − 1（5 日均值）", "sign": 1,
                     "ref": "Lou, Polk & Skouras (2019) JFE"},
    "i2_intraday": {"desc": "日内收益 close/open − 1（5 日均值）", "sign": -1,
                    "ref": "Lou, Polk & Skouras (2019) JFE"},
    "i2_overnight_ratio": {"desc": "20 日隔夜收益占总收益比例", "sign": 1,
                           "ref": "Lou, Polk & Skouras (2019)"},
    "i2_adx14": {"desc": "ADX 趋势强度（14 日）", "sign": 1,
                 "ref": "Wilder (1978)"},
    "i2_di_diff": {"desc": "DI+ − DI−（14 日，方向性）", "sign": 1,
                   "ref": "Wilder (1978)"},
    "i2_cci20": {"desc": "CCI 顺势指标（20 日）", "sign": 1,
                 "ref": "Lambert (1980)"},
    "i2_mfi14": {"desc": "MFI 资金流量指标（14 日）", "sign": 1,
                 "ref": "Wilder (1978)"},
    "i2_obv_slope": {"desc": "OBV 20 日斜率（标准化）", "sign": 1,
                     "ref": "Granville (1963)"},
    "i2_trix12": {"desc": "TRIX 三重指数平滑变化率（12 日）", "sign": 1,
                  "ref": "通用技术分析"},
    "i2_roc20": {"desc": "ROC 变动率（20 日）", "sign": 1,
                 "ref": "通用"},
    "i2_wr14": {"desc": "威廉指标 WR（14 日，0=超买 100=超卖）", "sign": -1,
                "ref": "Lane (1950s)"},
    "i2_psy12": {"desc": "心理线 PSY（12 日上涨天数占比）", "sign": 1,
                 "ref": "通用"},
    "i2_bias20": {"desc": "乖离率 BIAS（20 日）", "sign": -1,
                  "ref": "通用（均值回复）"},
    "i2_eff_ratio": {"desc": "Kaufman 效率比（20 日净变动/路径长度）", "sign": 1,
                     "ref": "Kaufman (1995)"},
    "i2_pv_corr": {"desc": "价量相关系数（20 日）", "sign": 1,
                   "ref": "Karpoff (1987)"},
    "i2_upper_shadow": {"desc": "上影线占振幅比例（20 日均）", "sign": -1,
                        "ref": "通用（冲高回落）"},
    "i2_lower_shadow": {"desc": "下影线占振幅比例（20 日均）", "sign": 1,
                        "ref": "通用（下探回升）"},
    "i2_alpha012": {"desc": "Alpha#012 = sign(Δvolume)·(−Δclose)", "sign": 1,
                    "ref": "Kakushadze (2016) arXiv:1601.00991"},
    "i2_alpha006": {"desc": "Alpha#006 = −corr(open, volume, 10)", "sign": -1,
                    "ref": "Kakushadze (2016)"},
    "i2_autocorr": {"desc": "日收益一阶自相关（20 日）", "sign": -1,
                    "ref": "Hou & Moskowitz (2005) RFS"},
    "i2_downside_vol": {"desc": "下行波动（60 日，仅负收益）", "sign": -1,
                        "ref": "Ang, Chen & Xing (2006)"},
    "i2_maxdd60": {"desc": "60 日最大回撤", "sign": -1, "ref": "通用"},
}


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x, dtype="float64").ewm(span=n, adjust=False).mean().to_numpy()


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder 平滑（等价 ewm alpha=1/n）。"""
    return pd.Series(x, dtype="float64").ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _roll(x: np.ndarray, w: int, how: str = "mean") -> np.ndarray:
    r = pd.Series(x, dtype="float64").rolling(w, min_periods=max(3, w // 3))
    return getattr(r, how)().to_numpy()


def compute_indicators2(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入单只股票日线（date/open/high/low/close/volume/amount），
    输出 index=date、columns=指标名 的 DataFrame（原始值，未标准化）。
    """
    d = df.reset_index(drop=True)
    o = d["open"].to_numpy(float); h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float); c = d["close"].to_numpy(float)
    v = d["volume"].to_numpy(float)
    n = len(d)

    prev_c = np.r_[np.nan, c[:-1]]
    ret = np.full(n, np.nan); ret[1:] = c[1:] / c[:-1] - 1.0
    out = pd.DataFrame(index=pd.DatetimeIndex(d["date"]))

    with np.errstate(invalid="ignore", divide="ignore"):
        # ---- 隔夜 / 日内分解
        on = o / prev_c - 1.0
        it = c / o - 1.0
        out["i2_overnight"] = _roll(on, 5)
        out["i2_intraday"] = _roll(it, 5)
        on20 = _roll(on, 20, "sum"); tot20 = _roll(ret, 20, "sum")
        out["i2_overnight_ratio"] = np.where(np.abs(tot20) > 1e-9, on20 / tot20, np.nan)

        # ---- ADX / DI
        tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
        up_move = np.r_[np.nan, np.diff(h)]
        dn_move = np.r_[np.nan, -np.diff(lo)]
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        atr = _rma(np.nan_to_num(tr), 14)
        pdi = 100 * _rma(plus_dm, 14) / np.maximum(atr, 1e-9)
        mdi = 100 * _rma(minus_dm, 14) / np.maximum(atr, 1e-9)
        dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9)
        out["i2_adx14"] = _rma(dx, 14)
        out["i2_di_diff"] = pdi - mdi

        # ---- CCI
        tp = (h + lo + c) / 3.0
        ma_tp = _roll(tp, 20)
        md = _roll(np.abs(tp - ma_tp), 20)
        out["i2_cci20"] = (tp - ma_tp) / np.maximum(0.015 * md, 1e-9)

        # ---- MFI
        mf = tp * v
        pos_mf = np.where(ret > 0, mf, 0.0); neg_mf = np.where(ret < 0, mf, 0.0)
        mr = _roll(pos_mf, 14, "sum") / np.maximum(_roll(neg_mf, 14, "sum"), 1e-9)
        out["i2_mfi14"] = 100 - 100 / (1 + mr)

        # ---- OBV 斜率
        obv = np.cumsum(np.where(ret > 0, v, np.where(ret < 0, -v, 0.0)))
        obv_s = pd.Series(obv)
        slope = (obv_s - obv_s.shift(20)).to_numpy()
        out["i2_obv_slope"] = slope / np.maximum(_roll(v, 20, "sum"), 1e-9)

        # ---- TRIX / ROC
        e1 = _ema(c, 12); e2 = _ema(e1, 12); e3 = _ema(e2, 12)
        out["i2_trix12"] = (e3 / np.r_[np.nan, e3[:-1]] - 1.0) * 100
        out["i2_roc20"] = (c / np.r_[np.full(20, np.nan), c[:-20]] - 1.0) * 100

        # ---- WR / PSY / BIAS
        hh = _roll(h, 14, "max"); ll = _roll(lo, 14, "min")
        out["i2_wr14"] = 100 * (hh - c) / np.maximum(hh - ll, 1e-9)
        out["i2_psy12"] = 100 * _roll((ret > 0).astype(float), 12)
        ma20 = _roll(c, 20)
        out["i2_bias20"] = (c / ma20 - 1.0) * 100

        # ---- Kaufman 效率比
        net = np.abs(c - np.r_[np.full(20, np.nan), c[:-20]])
        path = _roll(np.abs(np.nan_to_num(ret)) * c, 20, "sum")
        out["i2_eff_ratio"] = net / np.maximum(path, 1e-9)

        # ---- 价量相关
        out["i2_pv_corr"] = (pd.Series(ret).rolling(20, min_periods=8)
                             .corr(pd.Series(np.r_[np.nan, np.diff(np.log(np.maximum(v, 1)))]))
                             .to_numpy())

        # ---- 上下影线
        rng = np.maximum(h - lo, 1e-9)
        out["i2_upper_shadow"] = _roll((h - np.maximum(o, c)) / rng, 20)
        out["i2_lower_shadow"] = _roll((np.minimum(o, c) - lo) / rng, 20)

        # ---- Alpha#012 / Alpha#006
        dvol = np.r_[np.nan, np.diff(v)]
        dclose = np.r_[np.nan, np.diff(c)]
        out["i2_alpha012"] = np.sign(dvol) * (-dclose) / np.maximum(np.abs(c), 1e-9) * 100
        out["i2_alpha006"] = -(pd.Series(o).rolling(10, min_periods=5)
                               .corr(pd.Series(v)).to_numpy())

        # ---- 自相关
        out["i2_autocorr"] = (pd.Series(ret).rolling(20, min_periods=8)
                              .apply(lambda s: pd.Series(s).autocorr(1), raw=False).to_numpy())

        # ---- 下行波动 / 最大回撤
        neg = np.where(ret < 0, ret, np.nan)
        out["i2_downside_vol"] = np.sqrt(_roll(neg ** 2, 60))
        roll_max = pd.Series(c).rolling(60, min_periods=20).max().to_numpy()
        out["i2_maxdd60"] = c / np.maximum(roll_max, 1e-9) - 1.0

    return out.replace([np.inf, -np.inf], np.nan)


def list_indicators() -> List[str]:
    return list(INDICATOR_META)


def describe() -> pd.DataFrame:
    return pd.DataFrame([{"indicator": k, "desc": v["desc"], "sign": v["sign"],
                          "ref": v["ref"]} for k, v in INDICATOR_META.items()])
