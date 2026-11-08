"""
行为金融因子库 —— 把"人性"变成可计算的横截面变量。

## 为什么值得单独做一层

技术指标是公开的、机械的、会被套利掉；人性偏差是慢变量，衰减更慢。
文献里最稳的几条异象，几乎都不是"图形"，而是"人"：

| 因子 | 人性根源 | 代表文献 |
|---|---|---|
| 换手率 | 过度自信 → 过度交易 | Odean (1999) AER; Barber & Odean (2001) QJE |
| MAX | 彩票偏好（小概率大收益被高估） | Bali, Cakici & Whitelaw (2011) JFE |
| IVOL | 特质波动率之谜 | Ang et al. (2006) JF |
| CGO | 处置效应（赚了就跑、亏了死扛） | Grinblatt & Han (2005) JFE |
| 距 52 周高 | 锚定 | George & Hwang (2004) JF |
| 偏度 | 前景理论的概率权重 | Barberis & Huang (2008) |
| 非流动性 | 流动性溢价 | Amihud (2002) |

## 方向说明（文献预测，不是本模块的结论）

本模块**只负责算因子**，不预设它们有效。是否有效必须实测（见 `tests/test_behavioral.py`）。
文献预测方向写在各因子的 `expected_sign` 里，实测若相反，那是实测的结论，不是 bug。

## 全部因果

第 t 行的值只用 ≤t 的数据。参考价递归、滚动统计全部单向。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 因子元信息：名称 -> (中文说明, 文献预测方向, 代表文献)
FACTOR_META: Dict[str, Dict] = {
    "beh_turn_anom": {
        "desc": "异常换手：5 日均换手 / 60 日均换手 − 1",
        "expected_sign": -1,
        "ref": "Odean (1999) AER; Barber & Odean (2001) QJE",
        "logic": "过度自信导致过度交易，交易越频繁未来收益越低",
    },
    "beh_turn_rel": {
        "desc": "换手率相对自身：20 日均换手 / 250 日均换手",
        "expected_sign": -1,
        "ref": "Baker & Wurgler (2006) JF",
        "logic": "换手率是投资者情绪与关注度的代理",
    },
    "beh_max_20": {
        "desc": "过去 20 日最大单日收益",
        "expected_sign": -1,
        "ref": "Bali, Cakici & Whitelaw (2011) JFE",
        "logic": "彩票型股票被高估，未来收益低（MAX 效应）",
    },
    "beh_skew_60": {
        "desc": "过去 60 日收益偏度",
        "expected_sign": -1,
        "ref": "Barberis & Huang (2008); Conrad, Dittmar & Ghysels (2013)",
        "logic": "正偏度被高估",
    },
    "beh_ivol_60": {
        "desc": "60 日对市场回归残差的标准差（特质波动率）",
        "expected_sign": -1,
        "ref": "Ang, Hodrick, Xing & Zhang (2006) JF",
        "logic": "高特质波动股票未来收益低（IVOL 之谜）",
    },
    "beh_cgo": {
        "desc": "资本利得突出量 (P − RP)/P，RP 为换手加权参考价",
        "expected_sign": -1,
        "ref": "Grinblatt & Han (2005) JFE",
        "logic": "浮盈大的股票面临处置效应抛压，未来收益低",
    },
    "beh_dist_high": {
        "desc": "距 52 周高点的距离 (P / High252 − 1)",
        "expected_sign": 1,
        "ref": "George & Hwang (2004) JF",
        "logic": "锚定效应，接近 52 周高的股票继续走强",
    },
    "beh_amihud": {
        "desc": "非流动性：|收益| / 成交额（20 日均值，×1e8）",
        "expected_sign": 1,
        "ref": "Amihud (2002) JFM",
        "logic": "流动性溢价，越不流动要求越高收益",
    },
    "beh_turn_vol": {
        "desc": "换手率的 60 日波动率",
        "expected_sign": -1,
        "ref": "Baker & Wurgler (2006)",
        "logic": "情绪波动大的股票被反复炒作，长期收益低",
    },
    "beh_ret_5d_rev": {
        "desc": "过去 5 日收益（短期反转）",
        "expected_sign": -1,
        "ref": "Jegadeesh (1990); Lehmann (1990)",
        "logic": "短期反转，散户追涨杀跌造成过度反应",
    },
}


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(b) > eps, a / b, np.nan)


def _rolling_skew(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(w, min_periods=max(10, w // 2)).skew().to_numpy()


def _cgo(p: np.ndarray, v: np.ndarray, init_idx: int = 250) -> np.ndarray:
    """
    资本利得突出量（Grinblatt & Han 2005）。
    参考价递归：RP_t = (1 − V_{t−1})·RP_{t−1} + V_{t−1}·P_{t−1}
    其中 V 为换手率（小数），P 为收盘价。
    """
    T = len(p)
    rp = np.full(T, np.nan)
    if T < 2:
        return rp
    cur = p[0]
    rp[0] = cur
    for t in range(1, T):
        vt = v[t - 1]
        if not np.isfinite(vt):
            vt = 0.0
        vt = min(max(vt, 0.0), 1.0)
        cur = (1.0 - vt) * cur + vt * p[t - 1]
        rp[t] = cur
    return rp


def behavioral_factors(df: pd.DataFrame,
                       market_ret: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    输入单只股票日线（含 date/close/high/low/volume/amount/turnover），
    输出 index=date、columns=因子名 的 DataFrame（原始值，未做横截面标准化）。

    market_ret: 与 df 等长的市场日收益（小数）。为 None 时用等权代理（全 0），
                IVOL 退化为总波动，会在 meta 里标注。
    """
    d = df.reset_index(drop=True)
    close = d["close"].to_numpy(dtype=np.float64)
    high = d["high"].to_numpy(dtype=np.float64)
    amount = d["amount"].to_numpy(dtype=np.float64) if "amount" in d else np.full(len(d), np.nan)
    turn = d["turnover"].to_numpy(dtype=np.float64) if "turnover" in d else np.full(len(d), np.nan)
    turn = np.where(np.isfinite(turn) & (turn > 0), turn / 100.0, np.nan)   # % -> 小数

    ret = np.empty(len(d))
    ret[0] = np.nan
    ret[1:] = close[1:] / close[:-1] - 1.0

    out = pd.DataFrame(index=pd.DatetimeIndex(d["date"]))
    s_turn = pd.Series(turn)

    # 1) 异常换手
    t5 = s_turn.rolling(5, min_periods=3).mean()
    t60 = s_turn.rolling(60, min_periods=30).mean()
    out["beh_turn_anom"] = (_safe_div(t5.to_numpy(), t60.to_numpy()) - 1.0)
    # 2) 换手相对自身
    t20 = s_turn.rolling(20, min_periods=10).mean()
    t250 = s_turn.rolling(250, min_periods=60).mean()
    out["beh_turn_rel"] = _safe_div(t20.to_numpy(), t250.to_numpy())
    # 3) MAX
    out["beh_max_20"] = pd.Series(ret).rolling(20, min_periods=10).max().to_numpy()
    # 4) 偏度
    out["beh_skew_60"] = _rolling_skew(ret, 60)
    # 5) IVOL
    if market_ret is not None and len(market_ret) == len(d):
        m = np.asarray(market_ret, dtype=np.float64)
        ok = np.isfinite(ret) & np.isfinite(m)
        resid = np.full(len(d), np.nan)
        # 滚动回归残差（60 日），用协方差/方差公式
        sr = pd.Series(np.where(ok, ret, np.nan))
        sm = pd.Series(np.where(ok, m, np.nan))
        cov = sr.rolling(60, min_periods=30).cov(sm)
        var = sm.rolling(60, min_periods=30).var()
        beta = _safe_div(cov.to_numpy(), var.to_numpy())
        resid = sr.to_numpy() - beta * sm.to_numpy()
        out["beh_ivol_60"] = pd.Series(resid).rolling(60, min_periods=30).std().to_numpy()
    else:
        out["beh_ivol_60"] = pd.Series(ret).rolling(60, min_periods=30).std().to_numpy()
    # 6) CGO
    rp = _cgo(close, np.nan_to_num(turn, nan=0.0))
    out["beh_cgo"] = _safe_div(close - rp, close)
    # 7) 距 52 周高
    hi = pd.Series(close).rolling(250, min_periods=120).max().to_numpy()
    out["beh_dist_high"] = _safe_div(close, hi) - 1.0
    # 8) 非流动性
    amihud = _safe_div(np.abs(ret), amount / 1e8)
    out["beh_amihud"] = pd.Series(amihud).rolling(20, min_periods=10).mean().to_numpy()
    # 9) 换手波动
    out["beh_turn_vol"] = s_turn.rolling(60, min_periods=30).std().to_numpy()
    # 10) 短期反转
    out["beh_ret_5d_rev"] = (pd.Series(close) / pd.Series(close).shift(5) - 1.0).to_numpy()

    return out.replace([np.inf, -np.inf], np.nan)


def list_factors() -> List[str]:
    return list(FACTOR_META)


def describe() -> pd.DataFrame:
    rows = [{"factor": k, "desc": v["desc"], "expected_sign": v["expected_sign"],
             "ref": v["ref"], "logic": v["logic"]} for k, v in FACTOR_META.items()]
    return pd.DataFrame(rows)
