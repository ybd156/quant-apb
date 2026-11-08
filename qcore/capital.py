"""
资金面全景 —— 个股 / ETF / 板块 / 龙虎榜 / 两融 的资金动向汇总。

## 为什么单看个股资金流不够

个股"主力净流入"是个很弱的指标（本项目实测：IC −0.003，t=−0.10，几乎无预测力）。
原因很简单：**主力买所以涨**，它是同期相关的描述，不是领先信号。

真正有信息量的是**跨层级的资金迁移**：
  1. **ETF 份额变化** = 真金白银的申购赎回。份额增加意味着有人拿现金换份额（净买入一篮子），
     这是**增量资金**，比个股主力净流入干净得多（没有拆单干扰）。
  2. **板块资金流** = 行业间的轮动方向。
  3. **龙虎榜** = 游资席位的可见痕迹（唯一能看到"谁"的数据）。
  4. **两融余额** = 杠杆资金的风险偏好。

## 接口可用性（实测，2026-11）

| 数据 | 状态 | 备注 |
|---|---|---|
| ETF 快照（份额/规模） | ✅ | push2 stock/get 的 f84/f85 |
| ETF 资金流 | ✅ | fflow daykline |
| 板块资金流 | ✅ | clist fs=m:90+t:2 的 f62 |
| 龙虎榜 | ✅ | datacenter RPT_DAILYBILLBOARD_DETAILSNEW |
| 个股资金流 | ✅ | 见 qcore/flow.py |
| 两融明细 | ❌ 不稳 | RPT_RZRQ_LSHJ 带 filter 返回空，需换源（交易所官网/Choice） |
| 基金持仓变动 | ❌ | RPT_FUND_MAIN_POSITION 返回空 |

**拿不到的就不假装拿到**：`margin()` 与 `fund_holdings()` 会明确返回 `{"ok": False, "reason": ...}`。
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://data.eastmoney.com/"}
_MIN_INTERVAL = 0.3
_last = [0.0]


def _get(url: str, retries: int = 4, timeout: int = 25) -> str:
    last = None
    for i in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
            _last[0] = time.time()
            return body
        except Exception as e:  # noqa: BLE001
            _last[0] = time.time()
            last = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} :: {last}")


def _secid(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("15", "16", "50", "51", "52", "56", "58")):
        return f"1.{c}" if c[0] == "5" else f"0.{c}"
    return f"1.{c}" if c[0] in "569" else f"0.{c}"


# ------------------------------------------------------------------ ETF
def etf_snapshot(code: str) -> Dict:
    """
    ETF 快照：现价、总份额、流通份额、规模。
    f84 = 总份额（份），f85 = 流通份额，f116 = 总市值，f117 = 流通市值。
    """
    url = ("https://push2.eastmoney.com/api/qt/stock/get"
           f"?secid={_secid(code)}&fltt=2&invt=2&fields=f43,f57,f58,f84,f85,f116,f117,f47,f48")
    d = json.loads(_get(url)).get("data") or {}
    if not d:
        return {"ok": False, "reason": "empty", "code": code}
    return {"ok": True, "code": str(code).zfill(6), "name": d.get("f58"),
            "price": d.get("f43"), "shares_total": d.get("f84"),
            "shares_float": d.get("f85"), "aum": d.get("f116"),
            "aum_float": d.get("f117"), "volume": d.get("f47"), "amount": d.get("f48")}


def etf_flow(code: str) -> pd.DataFrame:
    """ETF 资金流（与个股同口径：超大单/大单/中单/小单）。"""
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           f"?secid={_secid(code)}&fields1=f1,f2,f3,f7"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           "&klt=101&lmt=0")
    d = json.loads(_get(url)).get("data") or {}
    rows = d.get("klines") or []
    if not rows:
        return pd.DataFrame()
    cols = ["date", "main_net", "small_net", "mid_net", "big_net", "huge_net",
            "main_pct", "small_pct", "mid_pct", "big_pct", "huge_pct",
            "close", "pct_chg", "x1", "x2"]
    df = pd.DataFrame([r.split(",") for r in rows], columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = str(code).zfill(6)
    return df.sort_values("date").reset_index(drop=True)


def etf_share_change(code: str, flow: Optional[pd.DataFrame] = None) -> Dict:
    """
    ETF 份额变化估算。
    严格口径需要每日份额历史（东财不直接给），这里用"规模/价格"反推，
    并用资金流净额做交叉校验。返回的是**近似值**，note 里写明。
    """
    snap = etf_snapshot(code)
    if not snap.get("ok"):
        return {"ok": False, "reason": "snapshot_failed"}
    price = snap.get("price") or np.nan
    aum = snap.get("aum") or np.nan
    shares = snap.get("shares_total") or np.nan
    implied = aum / price if (price and np.isfinite(price) and price > 0) else np.nan
    return {"ok": True, "code": snap["code"], "name": snap.get("name"),
            "shares_total": shares, "implied_shares": implied,
            "aum": aum, "price": price,
            "note": "份额历史东财不直接提供，implied_shares 为 规模/价格 近似，"
                    "不可当作官方份额；要做份额变化序列请接交易所/Choice"}


# ------------------------------------------------------------------ 板块
def sector_flow(top: int = 15, board_type: str = "industry") -> pd.DataFrame:
    """板块资金流排行（f62=主力净流入，f3=涨跌幅）。"""
    fs = "m:90+t:2" if board_type == "industry" else "m:90+t:3"
    url = ("https://push2.eastmoney.com/api/qt/clist/get"
           f"?pn=1&pz={top}&po=1&np=1&fltt=2&invt=2&fid=f62&fs={fs}"
           "&fields=f12,f14,f62,f3,f66,f69,f72,f75,f78,f81")
    d = json.loads(_get(url)).get("data") or {}
    rows = d.get("diff") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "f12": "board_code", "f14": "board_name", "f62": "main_net",
        "f3": "pct_chg", "f66": "huge_net", "f69": "huge_pct",
        "f72": "big_net", "f75": "big_pct", "f78": "mid_net", "f81": "mid_pct"})
    df["main_net_yi"] = pd.to_numeric(df["main_net"], errors="coerce") / 1e8
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
    return df[["board_code", "board_name", "main_net_yi", "pct_chg"]]


# ------------------------------------------------------------------ 龙虎榜
def dragon_tiger(date: Optional[str] = None, top: int = 50) -> pd.DataFrame:
    """
    龙虎榜（游资席位唯一可见的痕迹）。
    date 为 None 时取最近一日。字段含成交额占比、上榜原因。
    """
    flt = f"&filter=(TRADE_DATE%3D%27{date}%27)" if date else ""
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
           "?reportName=RPT_DAILYBILLBOARD_DETAILSNEW&columns=ALL"
           f"&pageSize={top}&sortColumns=TRADE_DATE&sortTypes=-1"
           f"&source=WEB&client=WEB{flt}")
    d = json.loads(_get(url))
    rows = (d.get("result") or {}).get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [c for c in ("TRADE_DATE", "SECURITY_CODE", "SECURITY_NAME_ABBR",
                        "CHANGE_RATE", "BILLBOARD_DEAL_AMT", "DEAL_AMOUNT_RATIO",
                        "EXPLAIN", "TURNOVERRATE") if c in df.columns]
    out = df[keep].copy()
    out.columns = [c.lower() for c in out.columns]
    return out


# ------------------------------------------------------------------ 两融 / 基金
def margin(code: str) -> Dict:
    """两融明细。当前东财接口不稳定，明确返回不可用而不是编数。"""
    try:
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
               "?reportName=RPT_RZRQ_LSHJ&columns=ALL&pageSize=10"
               "&sortColumns=DIM_DATE&sortTypes=-1&source=WEB&client=WEB"
               f"&filter=(SCODE%3D%22{str(code).zfill(6)}%22)")
        d = json.loads(_get(url))
        rows = (d.get("result") or {}).get("data") or []
        if not rows:
            return {"ok": False, "reason": "empty_result",
                    "note": "东财 RPT_RZRQ_LSHJ 带 filter 返回空，需换交易所官网或 Choice 数据源"}
        return {"ok": True, "rows": pd.DataFrame(rows)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def fund_holdings(code: str) -> Dict:
    """基金持仓变动（季度）。当前接口返回空，明确标注不可用。"""
    return {"ok": False, "reason": "endpoint_empty",
            "note": "RPT_FUND_MAIN_POSITION 返回空；基金持仓需从基金季报全文解析，"
                    "或接 Choice/Wind。本模块不编造持仓数据。"}


# ------------------------------------------------------------------ 汇总
def capital_panel(codes: List[str], etf_codes: Optional[List[str]] = None) -> Dict:
    """一次拿齐：个股资金流状态 + ETF 快照 + 板块资金流 + 龙虎榜。"""
    out: Dict = {"as_of": time.strftime("%Y-%m-%d"), "stocks": [], "etfs": [],
                 "sector_flow": [], "dragon_tiger": [], "missing": []}
    for c in codes[:20]:
        try:
            from .flow import fetch_flow
            f = fetch_flow(c)
            last = f.iloc[-1]
            out["stocks"].append({
                "code": c, "date": str(pd.Timestamp(last["date"]).date()),
                "main_net_yi": round(float(last["main_net"]) / 1e8, 3),
                "retail_net_yi": round(float(last["small_net"]) / 1e8, 3),
                "pct_chg": float(last["pct_chg"])})
        except Exception as e:  # noqa: BLE001
            out["missing"].append({"code": c, "reason": str(e)[:80]})

    for e in (etf_codes or []):
        s = etf_snapshot(e)
        if s.get("ok"):
            out["etfs"].append(s)
        else:
            out["missing"].append({"code": e, "reason": "etf_snapshot_failed"})

    try:
        out["sector_flow"] = sector_flow(10).to_dict("records")
    except Exception as e:  # noqa: BLE001
        out["missing"].append({"what": "sector_flow", "reason": str(e)[:80]})
    try:
        out["dragon_tiger"] = dragon_tiger(top=10).to_dict("records")
    except Exception as e:  # noqa: BLE001
        out["missing"].append({"what": "dragon_tiger", "reason": str(e)[:80]})
    return out
