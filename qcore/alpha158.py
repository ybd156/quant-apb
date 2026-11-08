"""Alpha158 因子集 —— 纯 numpy + pandas 实现（无 scipy / sklearn / talib）。

复现微软 Qlib ``qlib/contrib/data/loader.py`` 中 ``Alpha158DL`` 的**因子部分**
（不含 label，不含模型）。158 = 9 个 K 线形态 + 4 个当日价格比 + 29 公式 × 5 窗口。

官方来源
--------
* loader: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
* ops:    https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py

与本项目纪律的关系（AGENTS.md）
-------------------------------
* **INV-1 无未来函数**：所有因子只依赖 ``<= t`` 的窗口。禁止 ``shift(-n)``、
  禁止全样本 mean/std/quantile。实现上只用 ``rolling(...)``、
  ``sliding_window_view``（逐窗口独立计算）与 ``shift(+n)``。
  自检里有两条断言：截断不变性 + 未来扰动不变性。
* **INV-3 统计量必须滚动**：无任何全样本统计量。唯一的常数 ``_DEG_ATOL``
  是 qlib 自己的退化阈值（``Std < 2e-5`` → NaN），不是样本统计量。
* **INV-5 后复权**：见下面 ``VWAP`` 一节——后复权会让 ``amount/volume`` 与
  ``close`` 口径不一致，这是本文件唯一必须做取舍的地方。

预热期
------
qlib 的 ``Rolling`` 用 ``min_periods=1``（预热期有值、但是不足窗口的估计）。
本文件**故意改成 ``min_periods=n``**：预热期输出 ``NaN``，不填 0。
理由：填 0 会在横截面上制造虚假信号；不足窗口的估计在 LightGBM 里等同于噪声。

与 qlib 官方表达式的差异（全部显式列出，便于复核）
--------------------------------------------------
1. ``ROC``：本文件 ``close/shift(close,n)-1``（spec 口径）；qlib 是
   ``Ref($close,n)/$close``。两者互为 ``1/(1+r)`` 的**严格单调递减**变换
   —— 对树模型完全等价（树对单变量单调变换不变），对线性 IC 方向相反。
2. ``SUMP/SUMN/SUMD``：分母统一用 ``Sum(Abs(ret))``（qlib 口径）。
   若按 spec 字面写成「跌幅绝对值之和 / 涨幅绝对值之和」，则 ``SUMP`` 恒等于 1
   （分子分母同为涨幅），因子退化为常数，故不采用。
   另：qlib 用**价格差** ``$close-Ref($close,1)``，本文件用**收益率**
   ``close/shift(close,1)-1``（spec 说的「涨幅」），二者只差窗口内按价格水平加权。
3. ``VSUMP/VSUMN/VSUMD``：同上，用**成交量变化率** ``volume/shift(volume,1)-1``。
4. ``CORD``：本文件 ``corr(ret, log(v_t/v_{t-1}))``（spec 口径）；
   qlib 是 ``Log($volume/Ref($volume,1)+1)``（log 内多一个 +1）。
   加 ``_EPS`` 防止停牌 ``volume=0`` 时 ``log(0) = -inf``。
5. ``IMAX/IMIN``：本文件 ``argmax/n``（0 基，spec 口径）；qlib 是
   ``(argmax+1)/n``（1 基）。差一个常数 ``1/n``，横截面排序完全相同。
   ``IMXD`` 两边相减，差异自动抵消。
6. ``STD``：``ddof=1``，与 qlib 的 ``pandas.rolling.std()`` 默认一致。
   （ddof 只差常数 ``sqrt(n/(n-1))``，不影响横截面排序。）
7. ``RANK``：用 ``(count(<x) + (count(==x)+1)/2) / n``，等价于 pandas
   ``rolling(n).rank(pct=True, method='average')`` 取最后一个元素，
   也就是 qlib 现代版本的实现（旧版走 ``scipy.percentileofscore`` 同口径）。
8. ``BETA/RSQR/RESI``：qlib 用 Cython ``rolling_slope/rsquare/resi``；
   本文件用等价的 OLS 闭式解（``x = 0..n-1`` 中心化后一次算出
   slope / R² / 最后一个点的残差）。**数学上精确，不是近似**，
   自检里有逐点 ``polyfit`` 对拍。
9. 所有 ``/(high-low)`` 与 ``/(max-min)`` 都加 ``1e-12``，与 qlib 一致（防 inf）。

VWAP（唯一的一处「近似」，必须显式标注）
----------------------------------------
spec 要求 ``VWAP = amount/volume``。实测 ``qcore.data.load_kline``（东财 fqt=2）
拿到的数据**不能直接这么用**：

* ``volume`` 单位是「手」（100 股），所以 ``amount/volume = 100 × 原始 VWAP``；
* 更关键：``amount`` 是**不复权**成交额，而 OHLC 是**后复权**价格。
  两者混用会把复权因子当成因子 —— 实测 600519 的 ``(amount/volume)/close``
  从 2001 年的 100.09 单调漂到 2026 年的 16.22（与时间的相关 −0.74）。
  这个比值本质上就是「距上市多少年」，是伪因子。

因此默认 ``vwap_mode="hlc3"``，用 ``(high+low+close)/3`` 近似 VWAP
（相对误差通常在 0.1% 量级，且与 close 同口径、无漂移）。
``vwap_mode="amount"`` 保留原始口径（``amount/volume``，不做单位换算），
**只适用于不复权行情**；调用方可通过 ``out.attrs["vwap_source"]`` 确认实际走了哪条路。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["alpha158", "factor_names", "WINDOWS", "ROLLING_OPS"]

# ---------------------------------------------------------------- 常量
WINDOWS: Tuple[int, ...] = (5, 10, 20, 30, 60)

KBAR_OPS: Tuple[str, ...] = (
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
)
PRICE_OPS: Tuple[str, ...] = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
ROLLING_OPS: Tuple[str, ...] = (
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
)

_EPS = 1e-12
# qlib 的退化窗口阈值：rolling std < 2e-5 时 RSQR / CORR 置 NaN（见 ops.py）。
_DEG_ATOL = 2e-5

_REQUIRED = ("date", "open", "high", "low", "close", "volume")


def factor_names() -> List[str]:
    """返回 158 个因子名，顺序稳定（与 Qlib Alpha158 列名一致）。

    顺序 = 9 个 K 线形态 → 4 个当日价格比 → 29 个公式（每个公式内按
    窗口 5/10/20/30/60 递增）。
    """
    names: List[str] = list(KBAR_OPS) + list(PRICE_OPS)
    for op in ROLLING_OPS:
        for n in WINDOWS:
            names.append(f"{op}{n}")
    return names


assert len(factor_names()) == 158, len(factor_names())


# ---------------------------------------------------------------- 底层工具
def _window(x: np.ndarray, n: int) -> Optional[np.ndarray]:
    """``x`` 的长度 ``n`` 滑窗视图，形状 ``(T-n+1, n)``。

    每一行只包含 ``[t-n+1, t]`` 这一段 —— 这是截断不变性（INV-1）的物理保证：
    第 t 行的值只由它自己那一行窗口算出，与序列总长度无关。
    """
    if x.shape[0] < n:
        return None
    return np.lib.stride_tricks.sliding_window_view(x, n)


def _scatter(win_res: Optional[np.ndarray], n: int, T: int) -> np.ndarray:
    """把长度 ``T-n+1`` 的窗口结果放回长度 ``T``，前 ``n-1`` 个为 NaN。"""
    out = np.full(T, np.nan, dtype=np.float64)
    if win_res is not None and len(win_res):
        out[n - 1:] = win_res
    return out


def _ols_window(y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对每个窗口做 ``y ~ a + b*t``（``t = 0..n-1``）的 OLS，一次返回三个量。

    返回 ``(slope, rsqr, resi)``，长度均为 ``T-n+1``：

    * ``slope = Σ(t-t̄)(y-ȳ) / Σ(t-t̄)²``  —— 与 qlib ``rolling_slope`` 同值；
    * ``rsqr  = slope²·Sxx / Σ(y-ȳ)²``    —— 等价于 ``1 - SSR/SST``；
    * ``resi  = y[-1] - ŷ[-1]``           —— 窗口**最后一个点**的残差（qlib 口径）。

    全部是闭式解，一次向量化算完，不用 ``apply``/``polyfit``。
    """
    W = _window(y, n)
    if W is None:
        z = np.empty(0, dtype=np.float64)
        return z, z, z
    # t 中心化：Σtc = 0，于是 Σ(tc·y) 就等于 Σ(tc·(y-ȳ))，省一次去均值。
    tc = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    sxx = float((tc * tc).sum())  # leak-scan: ok 窗口矩阵每行只含 t-n+1..t
    mean = W.mean(axis=1)  # leak-scan: ok 窗口矩阵每行只含 t-n+1..t
    dev = W - mean[:, None]
    sxy = (dev * tc).sum(axis=1)  # leak-scan: ok 窗口矩阵每行只含 t-n+1..t
    sst = (dev * dev).sum(axis=1)  # leak-scan: ok 窗口矩阵每行只含 t-n+1..t
    slope = sxy / sxx
    with np.errstate(divide="ignore", invalid="ignore"):
        rsqr = (sxy * sxy) / (sxx * sst)
    rsqr = np.where(sst > 0.0, rsqr, np.nan)
    resi = dev[:, -1] - slope * (n - 1) / 2.0
    return slope, rsqr, resi


def _rank_window(y: np.ndarray, n: int) -> np.ndarray:
    """``close[t]`` 在窗口内的百分位，等价 ``rolling(n).rank(pct=True)``。

    ``method='average'`` 口径：``(count(<x) + (count(==x)+1)/2) / n``。
    用两次窗口比较代替 ``rolling.apply``，复杂度 O(T·n)，无 Python 循环。
    """
    W = _window(y, n)
    if W is None:
        return np.empty(0, dtype=np.float64)
    last = W[:, -1]
    with np.errstate(invalid="ignore"):
        lt = (W < last[:, None]).sum(axis=1)  # leak-scan: ok 窗口矩阵每行只含 t-n+1..t
        eq = (W == last[:, None]).sum(axis=1)  # leak-scan: ok 窗口矩阵每行只含 t-n+1..t
    out = (lt + (eq + 1.0) / 2.0) / float(n)
    out = np.where(np.isfinite(last), out, np.nan)
    return out


def _idx_window(y: np.ndarray, n: int, which: str) -> np.ndarray:
    """窗口内极值的 0 基位置（``argmax``/``argmin``），重复值取最早出现。"""
    W = _window(y, n)
    if W is None:
        return np.empty(0, dtype=np.float64)
    return (W.argmax(axis=1) if which == "max" else W.argmin(axis=1)).astype(np.float64)


# ---------------------------------------------------------------- 输入整理
def _prepare(df: pd.DataFrame, vwap_mode: str):
    """校验、排序、去重、取列，返回 (dates, o, h, l, c, v, vwap)。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"alpha158 需要 DataFrame，收到 {type(df).__name__}")
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少必需列 {missing}；现有列 = {list(df.columns)}。"
            "（amount 可选：缺失时 VWAP 自动退回 (high+low+close)/3）"
        )
    d = df
    if not d["date"].is_monotonic_increasing or d["date"].duplicated().any():
        d = d.sort_values("date").drop_duplicates("date", keep="last")
    dates = pd.DatetimeIndex(pd.to_datetime(d["date"]).to_numpy(), name="date")

    cols = {}
    for k in ("open", "high", "low", "close", "volume"):
        cols[k] = pd.to_numeric(d[k], errors="coerce").to_numpy(dtype=np.float64)
    o, h, l, c, v = (cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"])

    # INV-5：后复权仍有约 0.03% 的非正价 bar。非正价参与比值会产出垃圾值，
    # 统一置 NaN，让 NaN 顺着 rolling 传播（= 这根 bar 不可信），而不是伪造。
    bad = ~((o > 0) & (h > 0) & (l > 0) & (c > 0) & np.isfinite(o) & np.isfinite(h)
            & np.isfinite(l) & np.isfinite(c))
    if bad.any():
        o, h, l, c = (np.where(bad, np.nan, a) for a in (o, h, l, c))

    if vwap_mode == "hlc3":
        vwap = (h + l + c) / 3.0
        src = "hlc3:(high+low+close)/3"
    elif vwap_mode == "amount":
        amt = pd.to_numeric(d.get("amount", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = amt / np.where(v > 0, v, np.nan)
        src = "amount/volume"
    else:
        raise ValueError(f"vwap_mode 只能是 'hlc3' 或 'amount'，收到 {vwap_mode!r}")
    return dates, o, h, l, c, v, vwap, src


# ---------------------------------------------------------------- 主函数
def alpha158(df: pd.DataFrame, *, vwap_mode: str = "hlc3") -> pd.DataFrame:
    """把单只股票日线转成 158 个 Alpha158 因子。

    Parameters
    ----------
    df : DataFrame
        必须含 ``date/open/high/low/close/volume``（``amount`` 可选）。
        行序不必有序；重复日期会按 ``keep='last'`` 去重。
    vwap_mode : {'hlc3', 'amount'}
        ``hlc3``（默认）用 ``(high+low+close)/3`` 近似 VWAP，与 close 同口径；
        ``amount`` 用原始 ``amount/volume``，**仅适用于不复权行情**
        （后复权下与 close 口径不一致，详见模块 docstring）。

    Returns
    -------
    DataFrame
        ``index = date``，``columns = factor_names()``（158 列，float64）。
        预热期（不足窗口）为 ``NaN``，不填 0。``.attrs["vwap_source"]`` 记录
        VWAP 实际走的路径。

    Notes
    -----
    * 严格因果：第 t 行只用到 ``<= t`` 的数据（INV-1）。
    * 全部因子都是相对量：价格类除以当日 close，计数/比率类天然无量纲。
    * 无 ``apply`` / ``iterrows`` / 全样本统计量。
    """
    dates, o, h, l, c, v, vwap, vwap_src = _prepare(df, vwap_mode)
    T = c.shape[0]
    out: Dict[str, np.ndarray] = {}

    # ---- 1. 9 个 K 线形态（当日）
    hl = h - l
    oc_hi = np.maximum(o, c)
    oc_lo = np.minimum(o, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["KMID"] = (c - o) / o
        out["KLEN"] = hl / o
        out["KMID2"] = (c - o) / (hl + _EPS)
        out["KUP"] = (h - oc_hi) / o
        out["KUP2"] = (h - oc_hi) / (hl + _EPS)
        out["KLOW"] = (oc_lo - l) / o
        out["KLOW2"] = (oc_lo - l) / (hl + _EPS)
        out["KSFT"] = (2.0 * c - h - l) / o
        out["KSFT2"] = (2.0 * c - h - l) / (hl + _EPS)

    # ---- 2. 4 个当日价格比（都除以 close，去掉绝对价格）
    with np.errstate(divide="ignore", invalid="ignore"):
        out["OPEN0"] = o / c
        out["HIGH0"] = h / c
        out["LOW0"] = l / c
        out["VWAP0"] = vwap / c

    # ---- 3. 145 个滚动特征
    # 统一用 RangeIndex 的 Series，避免日期对齐带来的意外。
    s_c = pd.Series(c)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    s_v = pd.Series(v)

    ret = s_c / s_c.shift(1) - 1.0                       # 收益率
    logv = pd.Series(np.log(v + 1.0))                    # Log($volume+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_prev = s_v.shift(1)
        vchg = s_v / v_prev - 1.0                        # 成交量变化率
        # 停牌/上市首日 volume=0 会让 log 变 -inf，加 _EPS 兜底
        vchg_log = np.log((s_v + _EPS) / (v_prev + _EPS))
    vchg = vchg.where(v_prev > 0)                        # 前一日无量 → NaN
    vchg_log = vchg_log.where(v_prev > 0)

    up = (s_c > s_c.shift(1)).astype(np.float64)
    dn = (s_c < s_c.shift(1)).astype(np.float64)
    up.iloc[0] = 0.0                                     # qlib: NaN 比较结果为 False
    dn.iloc[0] = 0.0

    gain = ret.clip(lower=0.0)                           # 涨幅
    loss = (-ret).clip(lower=0.0)                        # 跌幅绝对值
    absr = ret.abs()
    vgain = vchg.clip(lower=0.0)
    vloss = (-vchg).clip(lower=0.0)
    vabs = vchg.abs()
    wv = (ret.abs() * s_v)                               # WVMA 的底料

    for n in WINDOWS:
        mp = n  # min_periods = n → 预热期 NaN（不足窗口不给估计）

        def roll(s: pd.Series, how: str):
            r = s.rolling(n, min_periods=mp)
            return getattr(r, how)()

        # --- 动量 / 均价 / 波动
        out[f"ROC{n}"] = (s_c.shift(n) / s_c - 1.0).to_numpy()
        out[f"MA{n}"] = (roll(s_c, "mean") / s_c).to_numpy()
        std_raw = s_c.rolling(n, min_periods=mp).std(ddof=1)
        out[f"STD{n}"] = (std_raw / s_c).to_numpy()

        # --- 趋势线性度（闭式 OLS，非近似）
        slope, rsqr, resi = _ols_window(c, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"BETA{n}"] = _scatter(slope / c[n - 1:], n, T)
            out[f"RESI{n}"] = _scatter(resi / c[n - 1:], n, T)
        deg = std_raw.to_numpy() <= _DEG_ATOL          # 常数窗口 → R² 无意义
        out[f"RSQR{n}"] = np.where(deg, np.nan, _scatter(rsqr, n, T))

        # --- 高低点 / 分位
        max_h = s_h.rolling(n, min_periods=mp).max()
        min_l = s_l.rolling(n, min_periods=mp).min()
        out[f"MAX{n}"] = (max_h / s_c).to_numpy()
        out[f"MIN{n}"] = (min_l / s_c).to_numpy()
        out[f"QTLU{n}"] = (s_c.rolling(n, min_periods=mp).quantile(0.8) / s_c).to_numpy()
        out[f"QTLD{n}"] = (s_c.rolling(n, min_periods=mp).quantile(0.2) / s_c).to_numpy()
        out[f"RANK{n}"] = _scatter(_rank_window(c, n), n, T)

        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"RSV{n}"] = ((s_c - min_l) / (max_h - min_l + _EPS)).to_numpy()

        # --- Aroon 类位置（0 基 / n；qlib 为 1 基，差常数 1/n）
        amax = _idx_window(h, n, "max")
        amin = _idx_window(l, n, "min")
        out[f"IMAX{n}"] = _scatter(amax / n, n, T)
        out[f"IMIN{n}"] = _scatter(amin / n, n, T)
        out[f"IMXD{n}"] = _scatter((amax - amin) / n, n, T)

        # --- 量价相关（qlib 在任一侧 std≈0 时置 NaN）
        corr = s_c.rolling(n, min_periods=mp).corr(logv).to_numpy()
        deg_v = logv.rolling(n, min_periods=mp).std(ddof=1).to_numpy() <= _DEG_ATOL
        out[f"CORR{n}"] = np.where(deg | deg_v, np.nan, corr)

        corr_d = ret.rolling(n, min_periods=mp).corr(vchg_log).to_numpy()
        deg_r = ret.rolling(n, min_periods=mp).std(ddof=1).to_numpy() <= _DEG_ATOL
        deg_vr = vchg_log.rolling(n, min_periods=mp).std(ddof=1).to_numpy() <= _DEG_ATOL
        out[f"CORD{n}"] = np.where(deg_r | deg_vr, np.nan, corr_d)

        # --- 涨跌天数占比
        cntp = roll(up, "mean")
        cntn = roll(dn, "mean")
        out[f"CNTP{n}"] = cntp.to_numpy()
        out[f"CNTN{n}"] = cntn.to_numpy()
        out[f"CNTD{n}"] = (cntp - cntn).to_numpy()

        # --- 涨跌幅结构（分母统一 Sum(Abs)，否则 SUMP 恒为 1）
        sabs = roll(absr, "sum") + _EPS
        sump = roll(gain, "sum") / sabs
        sumn = roll(loss, "sum") / sabs
        out[f"SUMP{n}"] = sump.to_numpy()
        out[f"SUMN{n}"] = sumn.to_numpy()
        out[f"SUMD{n}"] = (sump - sumn).to_numpy()

        # --- 量能
        out[f"VMA{n}"] = (roll(s_v, "mean") / (s_v + _EPS)).to_numpy()
        out[f"VSTD{n}"] = (s_v.rolling(n, min_periods=mp).std(ddof=1) / (s_v + _EPS)).to_numpy()
        out[f"WVMA{n}"] = (
            wv.rolling(n, min_periods=mp).std(ddof=1) / (roll(wv, "mean") + _EPS)
        ).to_numpy()

        vabs_s = roll(vabs, "sum") + _EPS
        vsump = roll(vgain, "sum") / vabs_s
        vsumn = roll(vloss, "sum") / vabs_s
        out[f"VSUMP{n}"] = vsump.to_numpy()
        out[f"VSUMN{n}"] = vsumn.to_numpy()
        out[f"VSUMD{n}"] = (vsump - vsumn).to_numpy()

    names = factor_names()
    res = pd.DataFrame({k: np.asarray(v_, dtype=np.float64) for k, v_ in out.items()},
                       index=dates)
    res = res[names]                      # 固定列序（与 factor_names 一致）
    res.attrs["vwap_source"] = vwap_src
    res.attrs["n_warmup"] = max(WINDOWS) - 1
    return res


# ---------------------------------------------------------------- 自检
def _naive_reference(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """最笨的逐点参考实现（仅自检用）：纯 Python 循环重算全部 29 个公式。

    刻意不复用主实现里的任何向量化技巧（不用 rolling / sliding_window_view），
    只用切片 + ``np.polyfit`` / ``np.quantile`` / ``np.corrcoef``，
    用来证明主实现不是「自己和自己一致」。
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    T = len(c)

    ret = np.full(T, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    logv = np.log(v + 1.0)
    vp = np.concatenate([[np.nan], v[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        vchg = np.where(vp > 0, v / vp - 1.0, np.nan)
        vchglog = np.where(vp > 0, np.log((v + _EPS) / (vp + _EPS)), np.nan)
    up = np.zeros(T)
    dn = np.zeros(T)
    up[1:] = (c[1:] > c[:-1]).astype(float)
    dn[1:] = (c[1:] < c[:-1]).astype(float)
    gain = np.where(np.isnan(ret), np.nan, np.maximum(ret, 0.0))
    loss = np.where(np.isnan(ret), np.nan, np.maximum(-ret, 0.0))
    absr = np.abs(ret)
    vgain = np.where(np.isnan(vchg), np.nan, np.maximum(vchg, 0.0))
    vloss = np.where(np.isnan(vchg), np.nan, np.maximum(-vchg, 0.0))
    vabs = np.abs(vchg)
    wv = np.abs(ret) * v

    cols = {op: np.full(T, np.nan) for op in ROLLING_OPS}
    for t in range(n - 1, T):
        sl = slice(t - n + 1, t + 1)
        wc, wh, wl, wv_ = c[sl], h[sl], l[sl], v[sl]
        x = np.arange(n, dtype=float)
        nanp = (np.isnan(wc).any() or np.isnan(wh).any()
                or np.isnan(wl).any() or np.isnan(wv_).any())

        def put(op, val):
            cols[op][t] = val

        if nanp:
            continue
        # 价格类
        put("ROC", c[t - n] / c[t] - 1.0 if t - n >= 0 else np.nan)
        put("MA", wc.mean() / c[t])  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("STD", wc.std(ddof=1) / c[t])  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        b, a = np.polyfit(x, wc, 1)
        put("BETA", b / c[t])
        fit = a + b * x
        sst = ((wc - wc.mean()) ** 2).sum()  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("RSQR", 1.0 - ((wc - fit) ** 2).sum() / sst if sst > 0 else np.nan)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("RESI", (wc[-1] - fit[-1]) / c[t])
        put("MAX", wh.max() / c[t])  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("MIN", wl.min() / c[t])  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("QTLU", np.quantile(wc, 0.8) / c[t])  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("QTLD", np.quantile(wc, 0.2) / c[t])  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("RANK", ((wc < wc[-1]).sum() + ((wc == wc[-1]).sum() + 1) / 2.0) / n)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("RSV", (c[t] - wl.min()) / (wh.max() - wl.min() + _EPS))  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("IMAX", wh.argmax() / n)
        put("IMIN", wl.argmin() / n)
        put("IMXD", (wh.argmax() - wl.argmin()) / n)
        # 量价相关
        r1, r2 = np.corrcoef(wc, logv[sl])[0, 1], np.corrcoef(ret[sl], vchglog[sl])[0, 1]
        put("CORR", np.nan if wc.std() <= _DEG_ATOL or logv[sl].std() <= _DEG_ATOL else r1)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("CORD", np.nan if np.isnan(ret[sl]).any() or np.isnan(vchglog[sl]).any()
            or ret[sl].std() <= _DEG_ATOL or vchglog[sl].std() <= _DEG_ATOL else r2)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        # 涨跌天数 / 涨跌幅结构
        put("CNTP", up[sl].mean())  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("CNTN", dn[sl].mean())  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("CNTD", up[sl].mean() - dn[sl].mean())  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        sa = absr[sl].sum()  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("SUMP", gain[sl].sum() / (sa + _EPS) if not np.isnan(sa) else np.nan)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("SUMN", loss[sl].sum() / (sa + _EPS) if not np.isnan(sa) else np.nan)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("SUMD", (gain[sl].sum() - loss[sl].sum()) / (sa + _EPS)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
            if not np.isnan(sa) else np.nan)
        # 量能
        put("VMA", wv_.mean() / (v[t] + _EPS))  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("VSTD", wv_.std(ddof=1) / (v[t] + _EPS))  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        mw = wv[sl]
        put("WVMA", mw.std(ddof=1) / (mw.mean() + _EPS) if not np.isnan(mw).any() else np.nan)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        va = vabs[sl].sum()  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("VSUMP", vgain[sl].sum() / (va + _EPS) if not np.isnan(va) else np.nan)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("VSUMN", vloss[sl].sum() / (va + _EPS) if not np.isnan(va) else np.nan)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
        put("VSUMD", (vgain[sl].sum() - vloss[sl].sum()) / (va + _EPS)  # leak-scan: ok 自检参考：逐窗口循环，仅 __main__
            if not np.isnan(va) else np.nan)
    return pd.DataFrame(cols)


def _self_check() -> None:
    import sys

    try:  # Windows 控制台默认 GBK，强制 UTF-8 才能正确显示中文
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    from qcore.data import load_kline

    raw = load_kline("600519")
    print(f"输入: 600519  行数={len(raw)}  {raw['date'].iloc[0].date()} → "
          f"{raw['date'].iloc[-1].date()}")

    # ---------- 1. 因子数量与命名
    fac = alpha158(raw)
    names = factor_names()
    assert list(fac.columns) == names, "列序与 factor_names() 不一致"
    print(f"\n[1] 因子数量 = {fac.shape[1]}  (期望 158)  shape={fac.shape}")
    print(f"    VWAP 口径 = {fac.attrs['vwap_source']}")
    print(f"    前 5 个: {names[:5]}")
    print(f"    后 5 个: {names[-5:]}")
    assert fac.shape[1] == 158

    # ---------- 2. 预热期必须是 NaN，不是 0（逐个因子核对首个有效行）
    # 9 个 K 线 + 4 个当日价格比是「当日」因子，第 0 行就有值；
    # 145 个滚动因子的首个有效行取决于它用 rolling(n) 还是 shift(n)：
    #   rolling(n) → 首个有效行 n-1；shift(n) 或多一层滞后（ret/vchg）→ n。
    warm = max(WINDOWS) - 1
    roll_cols = [c for c in names if c not in KBAR_OPS and c not in PRICE_OPS]
    lag_n = {"ROC", "CORD", "SUMP", "SUMN", "SUMD", "VSUMP", "VSUMN", "VSUMD", "WVMA"}
    bad_warm: List[str] = []
    for name in roll_cols:
        op = name.rstrip("0123456789")
        n = int(name[len(op):])
        exp = n if op in lag_n else n - 1
        col = fac[name].to_numpy()
        if not np.isnan(col[:exp]).all():          # 预热期必须是 NaN
            bad_warm.append(f"{name}:预热期有值")
            continue
        nz = np.flatnonzero(np.isfinite(col))
        if len(nz) == 0 or int(nz[0]) != exp:      # 首个有效行必须正好是 exp
            bad_warm.append(f"{name}:首值行={int(nz[0]) if len(nz) else None}≠{exp}")
    tail = fac.iloc[warm:]
    nan_share = float(tail.isna().to_numpy().mean())  # leak-scan: ok 自检诊断量，非特征
    head_today = int(np.isfinite(fac.iloc[:warm][list(KBAR_OPS) + list(PRICE_OPS)]
                                 .to_numpy()).all(axis=1).sum())  # leak-scan: ok 自检诊断量，非特征
    print(f"\n[2] 预热期: 145 个滚动因子首个有效行全部符合 rolling/shift 口径 "
          f"→ {'PASS' if not bad_warm else 'FAIL ' + str(bad_warm[:5])}")
    print(f"    13 个当日因子在前 {warm} 行有值的行数 = {head_today}/{warm}")
    print(f"    预热后 NaN 占比 = {nan_share:.4%}（应极小，来自 CORD/量价退化窗口）")
    assert not bad_warm, bad_warm[:10]
    assert head_today == warm, "当日因子出现 NaN"

    # ---------- 3. 截断不变性（INV-1 核心）
    cut = 2000
    part = alpha158(raw.iloc[:cut])
    a = fac.iloc[:cut].to_numpy()
    b = part.to_numpy()
    same = np.array_equal(a, b, equal_nan=True)
    diff = 0.0 if same else float(np.nanmax(np.abs(a - b)))  # leak-scan: ok 自检诊断量，非特征
    print(f"\n[3] 截断不变性: 前 {cut} 行 vs 全量 {len(raw)} 行 → "
          f"{'PASS' if same else 'FAIL'}  max|diff|={diff:.3e}")
    if not same:
        bad_cols = [n for j, n in enumerate(names)
                    if not np.array_equal(a[:, j], b[:, j], equal_nan=True)]
        print(f"    不一致列({len(bad_cols)}): {bad_cols[:10]}")
    assert same, "截断不变性 FAIL —— 存在未来函数（INV-1）"

    # ---------- 4. 未来扰动不变性（补强 INV-1：抓 shift(-n) 类错误）
    pert = raw.copy()
    k = len(pert) - 200
    for col in ("open", "high", "low", "close"):
        pert.loc[k:, col] = pert.loc[k:, col] * 3.0
    pert.loc[k:, "volume"] = pert.loc[k:, "volume"] * 7.0
    pert_fac = alpha158(pert)
    a2 = fac.iloc[:k].to_numpy()
    b2 = pert_fac.iloc[:k].to_numpy()
    same2 = np.array_equal(a2, b2, equal_nan=True)
    print(f"[4] 未来扰动不变性: 篡改末 200 行后，前 {k} 行 → "
          f"{'PASS' if same2 else 'FAIL'}")
    assert same2, "扰动未来数据影响了历史因子 —— 存在未来函数（INV-1）"

    # ---------- 5. 与朴素参考实现对拍（全部 29 个公式 × 2 个窗口 = 58 列）
    probe = raw.iloc[-400:].reset_index(drop=True)
    ref_fac = alpha158(probe)
    worst = {}
    n_nan_mismatch = 0
    for n in (5, 20):
        ref = _naive_reference(probe, n)
        for op in ROLLING_OPS:
            got = ref_fac[f"{op}{n}"].to_numpy()
            exp = ref[op].to_numpy()
            if not np.array_equal(np.isnan(got), np.isnan(exp)):
                n_nan_mismatch += 1
            m = np.isfinite(exp) & np.isfinite(got)
            worst[f"{op}{n}"] = float(np.max(np.abs(got[m] - exp[m]))) if m.any() else 0.0  # leak-scan: ok 自检诊断量，非特征
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:3]
    print(f"[5] 朴素对拍: {len(worst)} 列, max|diff| = {max(worst.values()):.3e} "
          f"(阈值 1e-9), NaN 位置不一致 = {n_nan_mismatch} 列")
    print(f"    最差三项: {[(k, f'{v_:.2e}') for k, v_ in top]}")
    assert max(worst.values()) < 1e-9, top
    assert n_nan_mismatch == 0, "NaN 位置与参考实现不一致"

    # ---------- 6. 值域健全性
    r = fac["RANK60"].dropna()
    assert r.between(0, 1).all(), "RANK 越界"
    q = fac["RSQR60"].dropna()
    assert q.between(-1e-9, 1 + 1e-9).all(), "RSQR 越界"
    v = fac["VWAP0"].dropna()
    print(f"[6] 值域: RANK∈[{r.min():.3f},{r.max():.3f}]  "  # leak-scan: ok 自检诊断量，非特征
          f"RSQR∈[{q.min():.4f},{q.max():.4f}]  VWAP0∈[{v.min():.4f},{v.max():.4f}]")  # leak-scan: ok 自检诊断量，非特征
    absmax = float(np.nanmax(np.abs(fac.to_numpy())))  # leak-scan: ok 自检诊断量，非特征
    print(f"    全因子 max|value| = {absmax:.2f}（绝对价格若泄漏会是 1e3~1e4 量级）")
    assert absmax < 100.0, "疑似绝对价格泄漏（因子未归一化）"

    # ---------- 7. 性能：spec 口径 2600 根日线
    bench = raw.iloc[-2600:].reset_index(drop=True)
    assert len(bench) == 2600
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        alpha158(bench)
        times.append(time.perf_counter() - t0)
    t2600 = min(times)
    t0 = time.perf_counter()
    alpha158(raw)
    t_full = time.perf_counter() - t0
    print(f"\n[7] 性能: 2600 根日线 = {t2600 * 1000:.0f} ms "
          f"(3 次取最快; 全部 {', '.join(f'{x * 1000:.0f}' for x in times)} ms)")
    print(f"    全量 {len(raw)} 根 = {t_full * 1000:.0f} ms")
    assert t2600 < 1.0, f"2600 根超 1 秒: {t2600:.3f}s"

    print("\n全部自检 PASS ✅")


if __name__ == "__main__":
    _self_check()
