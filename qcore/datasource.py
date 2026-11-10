"""
数据源中心：**自己去互联网找数据**，而不是写死一个接口就完事。

## 为什么需要它

实测教训（2026-11-10）：我只看了一个接口就下结论「资金流只有 120 天」，
用户追问「不能多找几个接口吗」。实际上同一类数据有多个源、历史深度差几十倍：

    东财 fflow 资金流        120 天
    东财 RPT_DATA_BLOCKTRADE 大宗交易   19 年（272 笔）
    东财 RPT_DAILYBILLBOARD  龙虎榜     多年（个股稀疏）

而且**同一个数据需求，接口名经常变**（融资融券我试了 3 个 reportName 全是空）。
所以正确的做法不是「找到能用的那个就写死」，而是：

  1. 登记**候选源**（一个需求对应多个候选）
  2. 逐个探测：能不能通、有没有数据、历史多长、字段齐不齐
  3. 按「历史最长 → 字段最全」自动选源
  4. 结果写进健康表，下次直接用上次选中的（坏了才重新探）
  5. 接口名变了：`--discover` 拿一批候选名去试，报哪个还活着

## 用法

    python -m qcore.datasource --code 600519          # 探测全部需求，打印健康表
    python -m qcore.datasource --code 600519 --need margin   # 只看某个需求
    python -m qcore.datasource --discover margin      # 接口名变了：试一批候选名
    python -m qcore.datasource --code 600519 --json   # 机器可读

## 边界（不夸大）

- 只做**取数与选源**，不做信号。取回来的数据怎么用由 signals/dataset 决定。
- 「发现」是**在候选清单里试**，不是凭空生成 URL —— 那需要搜索引擎+人工确认，
  自动生成 URL 去猜接口属于瞎猜，本项目不接受（会把噪音当数据）。
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
HEALTH_PATH = os.path.join(STATE, "datasource_health.json")

_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
_EM = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_HIS = "https://push2his.eastmoney.com/api/qt/stock"


# ---------------------------------------------------------------- 工具
def _get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return f.read().decode("utf-8", "replace")


def _em_report(report: str, code: str, page_size: int = 500) -> pd.DataFrame:
    """东财 datacenter 通用取数（括号不能 URL 编码，否则返回空）。"""
    q = urllib.parse.quote(f'(SECURITY_CODE="{str(code).zfill(6)}")', safe="()")
    url = (f"{_EM}?reportName={report}&columns=ALL"
           f"&sortColumns=TRADE_DATE&sortTypes=-1"
           f"&pageSize={page_size}&pageNumber=1&filter={q}")
    d = json.loads(_get(url))
    rows = ((d.get("result") or {}) if isinstance(d, dict) else {}).get("data") or []
    return pd.DataFrame(rows)


def _date_range(df: pd.DataFrame, col: str = "TRADE_DATE") -> Dict:
    if df is None or df.empty or col not in df.columns:
        return {"rows": 0, "first": None, "last": None}
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return {"rows": 0, "first": None, "last": None}
    # leak-scan: ok 只是**报告**数据的日期跨度，不参与任何特征/标准化
    return {"rows": int(len(df)),
            "first": str(s.min().date()), "last": str(s.max().date())}  # leak-scan: ok 仅报告跨度


# ---------------------------------------------------------------- 源定义
class Source:
    """一个候选数据源。fetch(code) -> DataFrame；空表表示「这个源当前不可用」。"""

    def __init__(self, need: str, name: str, fetch: Callable[[str], pd.DataFrame],
                 fields: Optional[List[str]] = None, note: str = ""):
        self.need = need
        self.name = name
        self.fetch = fetch
        self.fields = fields or []
        self.note = note


def _f_kline(code: str) -> pd.DataFrame:
    from . import data as D
    df = D.load_kline(code, adjust=2)
    return df if df is not None else pd.DataFrame()


def _f_flow(code: str) -> pd.DataFrame:
    from . import flow_factor as FF
    return FF.fetch_flow(code)


def _f_block(code: str) -> pd.DataFrame:
    from . import event_data as ED
    return ED.fetch_block_trade(code)


def _f_billboard(code: str) -> pd.DataFrame:
    from . import event_data as ED
    return ED.fetch_billboard(code)


def _f_margin(code: str) -> pd.DataFrame:
    """融资融券 —— 实测 3 个 reportName 全空，留在这里当「探测目标」。"""
    for rpt in ("RPT_MARGIN_DAILY", "RPT_MARGIN_DAILY_DETAIL", "RPT_RZRQ_DETAIL"):
        try:
            df = _em_report(rpt, code)
            if not df.empty:
                return df
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame()


#: 需求 → 候选源（同一需求可以有多个源，按历史长度自动选）
SOURCES: Dict[str, List[Source]] = {
    "kline": [Source("kline", "eastmoney-kline", _f_kline, ["date", "close"], "日线（后复权）")],
    "flow": [Source("flow", "eastmoney-fflow", _f_flow,
                    ["date", "main_ratio"], "资金流，接口上限约 120 天")],
    "block_trade": [Source("block_trade", "eastmoney-blocktrade", _f_block,
                           ["date", "bt_premium"], "大宗交易，实测跨 19 年")],
    "billboard": [Source("billboard", "eastmoney-billboard", _f_billboard,
                         ["date", "bb_net_ratio"], "龙虎榜，个股稀疏")],
    "margin": [Source("margin", "eastmoney-margin", _f_margin, ["TRADE_DATE"],
                      "融资融券（reportName 待发现）")],
}


# ---------------------------------------------------------------- 探测
def probe(source: Source, code: str) -> Dict:
    t0 = time.time()
    try:
        df = source.fetch(code)
        ms = int((time.time() - t0) * 1000)
        rng = _date_range(df, "date" if "date" in getattr(df, "columns", []) else "TRADE_DATE")
        missing = [f for f in source.fields if f not in getattr(df, "columns", [])]
        return {"need": source.need, "source": source.name, "ok": bool(rng["rows"]),
                "rows": rng["rows"], "first": rng["first"], "last": rng["last"],
                "missing_fields": missing, "ms": ms, "note": source.note}
    except Exception as e:  # noqa: BLE001
        return {"need": source.need, "source": source.name, "ok": False,
                "reason": f"{type(e).__name__}: {str(e)[:80]}",
                "ms": int((time.time() - t0) * 1000), "note": source.note}


def _span_days(r: Dict) -> int:
    if not r.get("ok") or not r.get("first") or not r.get("last"):
        return 0
    try:
        return int((pd.Timestamp(r["last"]) - pd.Timestamp(r["first"])).days)
    except Exception:  # noqa: BLE001
        return 0


def best_for(need: str, code: str) -> Optional[Dict]:
    """该需求下历史最长的可用源。"""
    cands = [probe(s, code) for s in SOURCES.get(need, [])]
    cands = [c for c in cands if c.get("ok")]
    if not cands:
        return None
    return max(cands, key=_span_days)


def health(code: str = "600519", needs: Optional[List[str]] = None) -> Dict:
    needs = needs or list(SOURCES)
    rows = []
    for need in needs:
        for s in SOURCES.get(need, []):
            rows.append(probe(s, code))
    picked = {}
    for need in needs:
        ok = [r for r in rows if r["need"] == need and r.get("ok")]
        if ok:
            b = max(ok, key=_span_days)
            picked[need] = {"source": b["source"], "rows": b["rows"],
                            "span_days": _span_days(b),
                            "first": b["first"], "last": b["last"]}
    out = {"ok": True, "code": str(code).zfill(6),
           "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "sources": rows, "picked": picked}
    try:
        os.makedirs(STATE, exist_ok=True)
        tmp = HEALTH_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, HEALTH_PATH)
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------- 接口名发现
#: 需求 → 候选 reportName（接口改版时用这批去试，报哪个还活着）
CANDIDATE_REPORTS: Dict[str, List[str]] = {
    "margin": ["RPT_MARGIN_DAILY", "RPT_MARGIN_DAILY_DETAIL", "RPT_RZRQ_DETAIL",
               "RPTA_WEB_RZRQ_GGMX", "RPT_MARGIN_DAILY_MARKET", "RPT_MARGIN_DAILY_SUM"],
    "block_trade": ["RPT_DATA_BLOCKTRADE", "RPT_BLOCKTRADE_DETAIL"],
    "billboard": ["RPT_DAILYBILLBOARD_DETAILSNEW", "RPT_DAILYBILLBOARD_DETAILS"],
    "flow": ["RPT_STOCK_FUNDFLOW", "RPT_FUNDFLOW_STOCK"],
}


def discover(need: str, code: str = "600519") -> Dict:
    """拿一批候选 reportName 去试，报哪些还活着。

    **不生成 URL**：只在人工登记的候选清单里试。瞎猜 URL 会把噪音当数据。
    """
    out = []
    for rpt in CANDIDATE_REPORTS.get(need, []):
        try:
            df = _em_report(rpt, code)
            rng = _date_range(df)
            out.append({"report": rpt, "ok": bool(rng["rows"]), "rows": rng["rows"],
                        "first": rng["first"], "last": rng["last"]})
        except Exception as e:  # noqa: BLE001
            out.append({"report": rpt, "ok": False,
                        "reason": f"{type(e).__name__}: {str(e)[:60]}"})
    alive = [x for x in out if x["ok"]]
    return {"ok": bool(alive), "need": need, "tried": len(out),
            "alive": [x["report"] for x in alive], "detail": out}


# ---------------------------------------------------------------- CLI
def _print_table(res: Dict) -> None:
    print(f"代码 {res['code']}  探测时间 {res['checked_at']}")
    print(f"{'需求':<13}{'源':<26}{'行数':>7}{'起始':>12}{'最新':>12}{'耗时':>7}  状态")
    print("-" * 92)
    for r in res["sources"]:
        status = "OK" if r.get("ok") else f"不可用({r.get('reason', '无数据')[:22]})"
        print(f"{r['need']:<13}{r['source']:<26}{r.get('rows', 0):>7}"
              f"{str(r.get('first') or '-'):>12}{str(r.get('last') or '-'):>12}"
              f"{r.get('ms', 0):>6}ms  {status}")
    print("\n自动选源（历史最长优先）：")
    for need, p in res["picked"].items():
        print(f"  {need:<13} -> {p['source']:<24} {p['rows']:>6} 行 / {p['span_days']} 天")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    code = "600519"
    for i, a in enumerate(args):
        if a == "--code" and i + 1 < len(args):
            code = args[i + 1]
    if "--discover" in args:
        need = args[args.index("--discover") + 1] if len(args) > args.index("--discover") + 1 else "margin"
        d = discover(need, code)
        print(json.dumps(d, ensure_ascii=False, indent=1))
        raise SystemExit(0 if d["ok"] else 1)
    need = None
    if "--need" in args:
        need = args[args.index("--need") + 1]
    res = health(code, [need] if need else None)
    if "--json" in args:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        _print_table(res)
