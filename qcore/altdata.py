"""
新信息源 —— 分析师预期修正、可转债溢价、期权波动率、期货基差。

## 为什么这些比技术指标更可能有 alpha

技术指标是**公开、机械、零成本**的，被套利掉是必然。
而下面这几类信息有**获取成本或专业门槛**，套利没那么彻底：

| 源 | 为什么可能有 alpha | 学术支撑 |
|---|---|---|
| 分析师预期修正 | 覆盖不足 + 反应迟缓 | Chan, Jegadeesh & Lakonishok (1996) JF |
| 可转债溢价率 | 债股联动、条款复杂、参与者少 | 国内转债研究 |
| 期权隐含波动率/偏度 | 前瞻性风险定价 | Xing, Zhang & Zhao (2010) JF |
| 股指期货基差 | 对冲成本 = 机构情绪的镜像 | Basak & Pavlova 等 |

## 接口实测状态（2026-11）

| 源 | 接口 | 状态 |
|---|---|---|
| 分析师研报 | `reportapi.eastmoney.com/report/list` | ✅ 可用（title/orgName/评级） |
| 盈利预测 | `datacenter RPT_VALUEANALYSIS_DET` | ✅ 可用 |
| 可转债 | `clist fs=b:MK0354` | ✅ 可用（f235 = 转股溢价率） |
| ETF 期权 | `clist m:10` | ❌ 断连，需换源 |
| 股指期货 | `clist m:113` | ❌ 断连，需换源 |
| 财报 MD&A | 巨潮 PDF | ❌ 需 PDF 解析 |

**拿不到的明确返回 ok=False，不编数。**

## 关键约束：这些数据大多**没有长历史**

- 研报接口通常只回最近 1-2 年
- 可转债列表只有当前快照
- 期权/期货同理

所以它们**只能前向积累**，不能回补历史做回测。
这与情绪数据同一性质——本模块在返回里带 `backfillable` 字段明确标注。
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://data.eastmoney.com/"}
_MIN = 0.35
_last = [0.0]


def _get(url: str, retries: int = 3, timeout: int = 25) -> str:
    last = None
    for i in range(retries):
        wait = _MIN - (time.time() - _last[0])
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
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"fetch failed: {url} :: {last}")


# ------------------------------------------------------------------ 分析师
def analyst_reports(code: str, days: int = 365) -> Dict:
    """
    个股研报列表（评级 + 机构）。返回最近 N 天的研报数量与评级分布。
    ⚠️ backfillable=False：接口只回最近窗口。
    """
    end = time.strftime("%Y-%m-%d")
    beg = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
    url = ("https://reportapi.eastmoney.com/report/list?cb=&industryCode=*&pageSize=100"
           f"&industry=*&rating=&ratingChange=&beginTime={beg}&endTime={end}"
           f"&pageNo=1&fields=&qType=0&orgCode=&code={str(code).zfill(6)}&rcode=&p=1&pageNum=1")
    try:
        d = json.loads(_get(url))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "backfillable": False}
    rows = d.get("data") or []
    if not rows:
        return {"ok": False, "reason": "empty", "backfillable": False}
    ratings = pd.Series([r.get("emRatingName") or r.get("sRatingName") or "未评级"
                         for r in rows]).value_counts().to_dict()
    return {"ok": True, "code": str(code).zfill(6), "n_reports": len(rows),
            "ratings": ratings,
            "n_orgs": len({r.get("orgCode") for r in rows if r.get("orgCode")}),
            "latest": str(rows[0].get("publishDate", ""))[:10],
            "backfillable": False,
            "note": "研报数量与评级分布可作为关注度/预期修正的代理"}


def earnings_forecast(code: str) -> Dict:
    """机构盈利预测明细（EPS 预测）。"""
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
           "?reportName=RPT_VALUEANALYSIS_DET&columns=ALL&pageSize=50"
           "&sortColumns=TRADE_DATE&sortTypes=-1&source=WEB&client=WEB"
           f"&filter=(SECURITY_CODE%3D%22{str(code).zfill(6)}%22)")
    try:
        d = json.loads(_get(url))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "backfillable": False}
    rows = (d.get("result") or {}).get("data") or []
    if not rows:
        return {"ok": False, "reason": "empty", "backfillable": False}
    df = pd.DataFrame(rows)
    keep = [c for c in ("TRADE_DATE", "SECURITY_CODE", "EPS", "PE", "ROE",
                        "PREDICT_YEAR", "ORG_NUM") if c in df.columns]
    return {"ok": True, "rows": df[keep].head(20), "n": len(rows),
            "backfillable": False}


# ------------------------------------------------------------------ 可转债
def convertible_list(top: int = 500) -> pd.DataFrame:
    """
    全市场可转债快照。字段：代码/名称/价格/涨幅/正股代码/正股名/**转股溢价率(f235)**。
    """
    url = ("https://push2.eastmoney.com/api/qt/clist/get"
           f"?pn=1&pz={top}&po=1&np=1&fltt=2&invt=2&fid=f3&fs=b:MK0354"
           "&fields=f12,f14,f2,f3,f232,f234,f235")
    d = json.loads(_get(url))
    rows = (d.get("data") or {}).get("diff") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "f12": "cb_code", "f14": "cb_name", "f2": "cb_price", "f3": "cb_pct",
        "f232": "stock_code", "f234": "stock_name", "f235": "premium_pct"})
    for c in ("cb_price", "cb_pct", "premium_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def convertible_premium_by_stock() -> Dict:
    """
    按正股聚合转债溢价率（正股若有转债，其溢价率反映"股债联动"预期）。
    返回 {stock_code: {"premium_pct":..., "n":...}}。
    """
    df = convertible_list()
    if df.empty:
        return {"ok": False, "reason": "empty", "backfillable": False}
    g = df.dropna(subset=["stock_code"]).groupby("stock_code").agg(
        premium_pct=("premium_pct", "mean"), n=("cb_code", "count")).reset_index()
    return {"ok": True,
            "by_stock": {str(r.stock_code).zfill(6): {
                "premium_pct": float(r.premium_pct) if np.isfinite(r.premium_pct) else None,
                "n": int(r.n)} for r in g.itertuples()},
            "n_convertibles": int(len(df)),
            "backfillable": False,
            "note": "溢价率为当前快照，无历史序列"}


# ------------------------------------------------------------------ 期权/期货
def option_snapshot(code: str = "510050") -> Dict:
    """ETF 期权链快照。当前东财接口不稳定，明确返回不可用。"""
    return {"ok": False, "reason": "endpoint_unreachable",
            "note": "东财期权列表接口（clist m:10）实测断连；需换源（上交所/深交所官网或 Choice）",
            "backfillable": False}


def futures_basis(code: str = "IF") -> Dict:
    """股指期货基差（期货 - 现货）。当前接口不可用。"""
    return {"ok": False, "reason": "endpoint_unreachable",
            "note": "东财期货列表接口（clist m:113）实测断连；需换中金所或 Choice",
            "backfillable": False}


def text_sentiment(code: str) -> Dict:
    """财报 MD&A 文本情绪。需要 PDF 解析，当前未实现。"""
    return {"ok": False, "reason": "not_implemented",
            "note": "需从巨潮下载年报 PDF 并解析 MD&A 段落；无 PDF 库，暂不做",
            "backfillable": False}


# ------------------------------------------------------------------ 汇总
def altdata_panel(codes: List[str], max_n: int = 10) -> Dict:
    """一次性拉取可用的另类数据（仅当前快照，不可回测）。"""
    out = {"as_of": time.strftime("%Y-%m-%d"), "analyst": {}, "convertible": None,
           "options": None, "futures": None, "unavailable": []}
    for c in codes[:max_n]:
        a = analyst_reports(c)
        if a.get("ok"):
            out["analyst"][c] = {k: a[k] for k in ("n_reports", "n_orgs", "latest")}
        else:
            out["unavailable"].append({"code": c, "what": "analyst", "reason": a.get("reason")})
    cb = convertible_premium_by_stock()
    out["convertible"] = cb if cb.get("ok") else None
    out["options"] = option_snapshot()
    out["futures"] = futures_basis()
    return out


def capability_report() -> pd.DataFrame:
    """各源可用性一览（诚实版）。"""
    return pd.DataFrame([
        {"source": "分析师研报", "api": "reportapi.eastmoney.com", "status": "可用",
         "backfillable": False, "note": "只回最近窗口"},
        {"source": "盈利预测", "api": "datacenter RPT_VALUEANALYSIS_DET", "status": "可用",
         "backfillable": False, "note": "EPS 预测明细"},
        {"source": "可转债溢价率", "api": "clist fs=b:MK0354 (f235)", "status": "可用",
         "backfillable": False, "note": "当前快照，无历史"},
        {"source": "ETF 期权 IV/偏度", "api": "clist m:10", "status": "不可用",
         "backfillable": False, "note": "接口断连，需换源"},
        {"source": "股指期货基差", "api": "clist m:113", "status": "不可用",
         "backfillable": False, "note": "接口断连，需换源"},
        {"source": "财报 MD&A 情绪", "api": "巨潮 PDF", "status": "未实现",
         "backfillable": True, "note": "需 PDF 解析"},
    ])
