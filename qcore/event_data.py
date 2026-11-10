"""
事件类数据的取数与缓存：大宗交易、龙虎榜。

## 为什么单独一个模块

`flow_factor.py` 管资金流（东财 fflow 接口，上限 120 天）；
这里管两类**事件型**数据，它们共同的特征是「稀疏」——大多数交易日没有，
所以做成信号后多数时候输出 0（sleeping expert），只在事件发生日发声。

## 只用相对量（INV 要求）

绝对金额在不同股票/不同价位之间不可比（后复权价与现价差 10 倍）。
所以这里全部换算成**比例**：
  · 大宗交易：占流通股本比例（TOTAL_SHARES_RATIO）、加权平均溢价率
  · 龙虎榜：  净买入额 / 当日成交额

## 历史深度（实测 2026-11-10）

  · 大宗交易 RPT_DATA_BLOCKTRADE     600519 有 2731 条，跨多年  ✓
  · 龙虎榜   RPT_DAILYBILLBOARD_...   个股稀疏（只在上榜日有）   ✓

## 缓存

落在 `data/events/ext/<code>.csv`，默认 12 小时不重取（事件数据一天更新一次即可）。
取不到就返回空 DataFrame —— 调用方（signals）会把该族输出 0，绝不报错。
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT, "data", "events", "ext")

_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
_BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"

#: 对外承诺的列（信号族只读这些）
COLUMNS = ["date", "bt_shares_ratio", "bt_premium", "bb_net_ratio",
           "mg_net_ratio", "mg_bal_ratio", "mg_short_ratio"]


def _get_json(url: str, timeout: int = 20) -> Dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8", "replace"))


def _datacenter(report: str, code: str, page_size: int = 500,
                key: str = "SECURITY_CODE", max_pages: int = 12,
                sort: str = "TRADE_DATE") -> pd.DataFrame:
    """东财 datacenter 通用取数（**自动翻页**）。返回原始 DataFrame（可能为空）。

    两个实测坑：
      1. 括号不能 URL 编码 —— `quote(..., safe="")` 会把 ( ) 变成 %28/%29，
         东财直接返回空 result，表现为「no_events」，其实是 URL 编码问题。
      2. 单页上限 500 条 —— 融资融券 600519 有 4015 条（9 页），
         不翻页就只拿到最近 500 天，历史凭空少 8 年。
    """
    frames = []
    for page in range(1, max_pages + 1):
        q = urllib.parse.quote(f'({key}="{str(code).zfill(6)}")', safe="()")
        url = (f"{_BASE}?reportName={report}&columns=ALL"
               f"&sortColumns={sort}&sortTypes=-1"
               f"&pageSize={page_size}&pageNumber={page}&filter={q}")
        try:
            d = _get_json(url)
        except Exception:  # noqa: BLE001
            break
        rows = ((d.get("result") or {}) if isinstance(d, dict) else {}).get("data") or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_block_trade(code: str) -> pd.DataFrame:
    """大宗交易 -> 按日聚合 [date, bt_shares_ratio, bt_premium]。

    bt_shares_ratio : 当日大宗成交量占总股本比例（%），相对量，可跨股票比
    bt_premium      : 按成交额加权的溢价率（%，正=溢价成交，负=折价甩卖）
    """
    raw = _datacenter("RPT_DATA_BLOCKTRADE", code)
    if raw.empty:
        return pd.DataFrame(columns=["date", "bt_shares_ratio", "bt_premium"])
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TRADE_DATE"], errors="coerce").dt.normalize(),
        "ratio": pd.to_numeric(raw.get("TOTAL_SHARES_RATIO"), errors="coerce"),
        "prem": pd.to_numeric(raw.get("PREMIUM_RATIO"), errors="coerce"),
        "amt": pd.to_numeric(raw.get("DEAL_AMT"), errors="coerce"),
    }).dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "bt_shares_ratio", "bt_premium"])
    # 同日多笔：比例求和，溢价率按成交额加权
    df["w"] = df["amt"].abs().fillna(0.0) + 1.0        # +1 避免全 0 权重
    # leak-scan: ok 按**同一天**的多笔大宗聚合，不是跨时间的全样本统计
    g = df.groupby("date", as_index=False).apply(  # leak-scan: ok 同日聚合
        lambda x: pd.Series({
            "bt_shares_ratio": float(x["ratio"].fillna(0.0).sum()),  # leak-scan: ok 同日求和
            "bt_premium": float((x["prem"].fillna(0.0) * x["w"]).sum() / x["w"].sum()),  # leak-scan: ok 同日加权
        }), include_groups=False)
    return g.reset_index(drop=True)


def fetch_billboard(code: str) -> pd.DataFrame:
    """龙虎榜 -> 按日聚合 [date, bb_net_ratio]。

    bb_net_ratio : 龙虎榜净买入额 / 当日成交额（%）。正=榜上净买入。
    """
    raw = _datacenter("RPT_DAILYBILLBOARD_DETAILSNEW", code)
    if raw.empty:
        return pd.DataFrame(columns=["date", "bb_net_ratio"])
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TRADE_DATE"], errors="coerce").dt.normalize(),
        "net": pd.to_numeric(raw.get("BILLBOARD_NET_AMT"), errors="coerce"),
        "amt": pd.to_numeric(raw.get("ACCUM_AMOUNT"), errors="coerce"),
    }).dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "bb_net_ratio"])
    # 同一只票一天可能有多条（不同上榜原因），净额求和、成交额取最大
    # leak-scan: ok 同日多条上榜记录聚合，与跨期统计无关
    g = df.groupby("date", as_index=False).agg(  # leak-scan: ok 同日聚合
        net=("net", "sum"), amt=("amt", "max"))
    # pandas 2.x 删了 mode.use_inf_as_na；直接对除零/无穷做清洗
    g["bb_net_ratio"] = (g["net"] / g["amt"].replace(0, pd.NA)) * 100.0
    g["bb_net_ratio"] = g["bb_net_ratio"].replace([float("inf"), float("-inf")], pd.NA)
    return g[["date", "bb_net_ratio"]]


def fetch_margin(code: str) -> pd.DataFrame:
    """融资融券 -> 按日 [date, mg_net_ratio, mg_bal_ratio, mg_short_ratio]。

    实测（2026-11-10，600519）：**4015 条 / 9 页 / 从 2024-10-18 起**。
    注意 filter 字段名是 `SCODE`，不是 `SECURITY_CODE` ——
    用错字段名东财不报错，直接返回空 result（我为此白试了 6 个 reportName）。

    mg_net_ratio   : 融资净买入额 / 当日成交额（%），相对量
    mg_bal_ratio   : 融资余额 / 流通市值（%）—— 接口自带 RZYEZB
    mg_short_ratio : 融券余额 / 融资余额（%），做空压力
    """
    # 这个报表的日期列叫 DATE（不是 TRADE_DATE）；排序列写错东财**不报错**，
    # 直接返回空 result —— 又一个「静默空」的坑。
    raw = _datacenter("RPTA_WEB_RZRQ_GGMX", code, key="SCODE", sort="DATE")
    cols = ["date", "mg_net_ratio", "mg_bal_ratio", "mg_short_ratio"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["DATE"], errors="coerce").dt.normalize(),
        "net": pd.to_numeric(raw.get("RZJME"), errors="coerce"),
        "amt": pd.to_numeric(raw.get("SZ"), errors="coerce"),
        "bal_pct": pd.to_numeric(raw.get("RZYEZB"), errors="coerce"),
        "rzye": pd.to_numeric(raw.get("RZYE"), errors="coerce"),
        "rqye": pd.to_numeric(raw.get("RQYE"), errors="coerce"),
    }).dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["mg_net_ratio"] = (df["net"] / df["amt"].replace(0, pd.NA)) * 100.0
    df["mg_bal_ratio"] = df["bal_pct"]
    df["mg_short_ratio"] = (df["rqye"] / df["rzye"].replace(0, pd.NA)) * 100.0
    for c in cols[1:]:
        df[c] = df[c].replace([float("inf"), float("-inf")], pd.NA)
    return df[cols].sort_values("date").reset_index(drop=True)


def _cache_path(code: str) -> str:
    return os.path.join(EXT_DIR, f"{str(code).zfill(6)}.csv")


def load_events(code: str, refresh: bool = False,
                max_age_hours: float = 12.0) -> pd.DataFrame:
    """读该票的事件数据（大宗交易 + 龙虎榜），带磁盘缓存。

    返回 DataFrame[date, bt_shares_ratio, bt_premium, bb_net_ratio]。
    任何失败都返回**空表**（信号族会输出 0），绝不抛异常中断主流程。
    """
    path = _cache_path(code)
    if (not refresh) and os.path.isfile(path):
        if (time.time() - os.path.getmtime(path)) / 3600.0 < max_age_hours:
            try:
                df = pd.read_csv(path, parse_dates=["date"])
                for c in COLUMNS:
                    if c not in df.columns:
                        df[c] = pd.NA
                return df[COLUMNS]
            except Exception:  # noqa: BLE001
                pass
    try:
        bt = fetch_block_trade(code)
        bb = fetch_billboard(code)
        mg = fetch_margin(code)
        parts = [x for x in (bt, bb, mg) if x is not None and len(x)]
        if not parts:
            out = pd.DataFrame(columns=COLUMNS)
        else:
            out = parts[0]
            for x in parts[1:]:
                out = pd.merge(out, x, on="date", how="outer")
        for c in COLUMNS:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[COLUMNS].sort_values("date").reset_index(drop=True)
        os.makedirs(EXT_DIR, exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8")
        return out
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=COLUMNS)


def available(code: str) -> Dict:
    """诊断用：该票事件数据覆盖情况。"""
    df = load_events(code)
    if df is None or df.empty:
        return {"ok": False, "reason": "no_events", "code": str(code).zfill(6)}
    return {"ok": True, "code": str(code).zfill(6), "rows": int(len(df)),
            "first": str(df["date"].iloc[0].date()),
            "last": str(df["date"].iloc[-1].date()),
            "n_block": int(df["bt_shares_ratio"].notna().sum()),
            "n_billboard": int(df["bb_net_ratio"].notna().sum()),
            "n_margin": int(df["mg_net_ratio"].notna().sum())}


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    print(json.dumps(available(code), ensure_ascii=False, indent=1))
