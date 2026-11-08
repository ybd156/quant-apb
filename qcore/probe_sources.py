"""
数据接口实测 —— 逐个探活，标注「能不能用」和「有没有历史」。

## 为什么需要这个

AGENTS.md §6 要求：加信息源必须标 `pit_backfillable`。
但更重要的是**先确认接口活着**——过去多次踩坑（期权/期货接口断连、
两融返回 ok:False），都是因为没先探活就写代码。

## 判定口径

- `OK`      返回了非空数据，字段可解析
- `EMPTY`   接口通但返回空（可能非交易日/无数据）
- `FAIL`    网络错误/字段变更/需要鉴权
- `history` 能否取到历史序列（**能回测**）；False 表示只有当前快照
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Dict, List

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://data.eastmoney.com/"}
_MIN = 0.25
_last = [0.0]


def _get(url: str, timeout: int = 15, retries: int = 3) -> str:
    """带重试——实测东财偶发 SSL EOF / 握手超时，单次失败不代表接口坏了。"""
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
    raise last  # type: ignore[misc]


def _probe(name: str, url: str, kind: str = "json",
           history: bool = False, note: str = "") -> Dict:
    try:
        body = _get(url)
        if kind == "json":
            d = json.loads(body)
            # 常见结构：data.diff / data.klines / result.data / data.pool
            data = d.get("data") if isinstance(d, dict) else None
            if isinstance(data, dict):
                n = len(data.get("diff") or data.get("klines") or []) or \
                    len((d.get("result") or {}).get("data") or [])
            elif isinstance(data, list):
                n = len(data)
            else:
                n = 0
            status = "OK" if n > 0 else "EMPTY"
            return {"name": name, "status": status, "rows": n,
                    "history": history, "note": note}
        status = "OK" if len(body) > 100 else "EMPTY"
        return {"name": name, "status": status, "rows": len(body),
                "history": history, "note": note}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "FAIL", "rows": 0, "history": history,
                "note": f"{type(e).__name__}: {str(e)[:60]}"}


PUSH = "https://push2.eastmoney.com/api/qt"
HIS = "https://push2his.eastmoney.com/api/qt/stock"
DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"

CANDIDATES: List[tuple] = [
    # ---------- 行情 ----------
    ("指数实时", f"{PUSH}/ulist.np/get?fltt=2&invt=2&secids=1.000001,0.399001&fields=f1,f2,f3,f12,f14", "json", False, "9 大指数"),
    ("个股日K(后复权)", f"{HIS}/kline/get?secid=1.600519&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=2&beg=20200101&end=20500101&lmt=100", "json", True, "主数据源"),
    ("分钟K(5分)", f"{HIS}/kline/get?secid=1.600519&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56&klt=5&fqt=2&beg=20200101&end=20500101&lmt=500", "json", True, "高频因子"),
    ("全市场快照", f"{PUSH}/clist/get?pn=1&pz=50&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12,f14,f2,f3,f5,f6,f8,f10", "json", False, "选股池"),

    # ---------- 资金 ----------
    ("个股资金流历史", f"{HIS}/fflow/daykline/get?secid=1.600519&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65&lmt=120", "json", True, "★有历史，可做因子"),
    ("板块资金流排行", f"{PUSH}/clist/get?pn=1&pz=20&po=1&np=1&fltt=2&invt=2&fid=f62&fs=m:90+t:2&fields=f12,f14,f3,f62", "json", False, "板块轮动"),
    ("龙虎榜", f"{DC}?reportName=RPT_DAILYBILLBOARD_DETAILSNEW&columns=ALL&pageSize=30&sortColumns=TRADE_DATE&sortTypes=-1&source=WEB&client=WEB", "json", True, "★有历史"),
    ("大宗交易", f"{DC}?reportName=RPT_DATA_BLOCKTRADE&columns=ALL&pageSize=30&sortColumns=TRADE_DATE&sortTypes=-1&source=WEB&client=WEB", "json", True, "★有历史"),
    ("融资融券明细", f"{DC}?reportName=RPTA_WEB_RZRQ_GGMX&columns=ALL&pageSize=30&sortColumns=DATE&sortTypes=-1&source=WEB&client=WEB&filter=(SCODE%3D%22600519%22)", "json", True, "★有历史"),

    # ---------- 基本面/事件 ----------
    ("股东户数", f"{DC}?reportName=RPT_HOLDERNUMLATEST&columns=ALL&pageSize=30&sortColumns=END_DATE&sortTypes=-1&source=WEB&client=WEB&filter=(SECURITY_CODE%3D%22600519%22)", "json", True, "季度"),
    ("机构调研", f"{DC}?reportName=RPT_ORG_SURVEYNEW&columns=ALL&pageSize=30&sortColumns=RECEIVE_START_DATE&sortTypes=-1&source=WEB&client=WEB", "json", True, "事件驱动"),
    ("股权质押", f"{DC}?reportName=RPT_CUSTOM_STOCK_PLEDGE_STATISTICS&columns=ALL&pageSize=30&sortColumns=TRADE_DATE&sortTypes=-1&source=WEB&client=WEB", "json", True, "风险"),
    ("限售解禁", f"{DC}?reportName=RPT_LIFT_STAGE&columns=ALL&pageSize=30&sortColumns=FREE_DATE&sortTypes=-1&source=WEB&client=WEB", "json", True, "风险"),
    ("业绩预告", f"{DC}?reportName=RPT_PUBLIC_OP_NEWPREDICT&columns=ALL&pageSize=30&sortColumns=NOTICE_DATE&sortTypes=-1&source=WEB&client=WEB", "json", True, "事件驱动"),
    ("千股千评", f"{PUSH}/clist/get?pn=1&pz=30&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6&fields=f12,f14,f2,f3,f8,f10,f22", "json", False, "含换手/量比"),

    # ---------- 研报 ----------
    ("研报列表", "https://reportapi.eastmoney.com/report/list?cb=&industryCode=*&pageSize=30&industry=*&rating=&ratingChange=&beginTime=2025-01-01&endTime=2050-01-01&pageNo=1&fields=&qType=0&orgCode=&code=600519&rcode=&p=1&pageNum=1", "json", False, "只回最近窗口"),

    # ---------- 转债 ----------
    ("可转债列表", f"{PUSH}/clist/get?pn=1&pz=50&po=1&np=1&fltt=2&invt=2&fid=f3&fs=b:MK0354&fields=f12,f14,f2,f3,f232,f234,f235", "json", False, "含转股溢价率"),

    # ---------- 宏观 ----------
    ("世行GDP", "https://api.worldbank.org/v2/country/CN/indicator/NY.GDP.MKTP.KD.ZG?format=json&per_page=10", "json", True, "年度"),
]


def main() -> int:
    print("=" * 78)
    print("数据接口实测")
    print("=" * 78)
    rows = []
    for name, url, kind, hist, note in CANDIDATES:
        r = _probe(name, url, kind, hist, note)
        rows.append(r)
        flag = {"OK": "OK  ", "EMPTY": "EMPTY", "FAIL": "FAIL"}[r["status"]]
        h = "有历史" if r["history"] else "快照  "
        print(f"  [{flag}] {r['name']:<16} {h}  rows={r['rows']:<6} {r['note']}")

    ok = [r for r in rows if r["status"] == "OK"]
    hist_ok = [r for r in ok if r["history"]]
    print()
    print(f"可用 {len(ok)}/{len(rows)}；其中**有历史可回测** {len(hist_ok)} 个：")
    for r in hist_ok:
        print(f"    · {r['name']}  ({r['note']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
