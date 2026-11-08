"""
信息快照层：把"当时的形势"采成一条条**带时间戳、带来源、带可信度**的记录。

为什么需要这一层
----------------
系统要同时回答两个问题：
  1. **当时形势** —— 站在 t 日，只看得见 t 日及以前的信息（Point-in-Time）；
  2. **历史形势** —— 当前指标 vs 该股自身历史的相对位置。
第 2 个问题只要历史序列就够；第 1 个问题必须知道"每条信息是什么时候产生的、从哪来的、
有多可信"。所以本模块产出的不是"一个大字典"，而是一组 `InfoItem`：
每条都自带 `as_of`（数据描述的时间）、`source`（源）、`reliability`（0~1 权重）、
`pit_backfillable`（能否回补历史）、`note`（口径/缺陷说明）。

⚠️ 铁律（与 docs/02_SPEC.md §0.0 一致，务必遵守）
------------------------------------------------
* **禁止用 news / announcement 做历史回测。**
  这两类源的 `pit_backfillable = False`：东财新闻搜索只返回"当前可见"的结果集，
  公告接口的 `page_size` 窗口也会随新公告滚动 —— 无法重建"t 日当时可见的新闻/公告集合"，
  历史回补必然引入幸存者/后见偏差。它们**只能做"当下研判"**（今天看的今天的新闻），
  而且必须从部署日起**前向采集**、逐日落盘，才有资格进入复盘统计。
* **禁止把 live 接口的当前值贴到历史日期上。**
  `snapshot(code, as_of=<过去某天>)` 时，只有 `kline` 能用历史 K 线真实截断重建；
  估值/基本面/板块/宽度这些接口只服务"当前"，因此在 backfill 模式下它们会被
  明确写进 `missing`（reason 说明原因），**绝不用今天的 PE 冒充一个月前的 PE**。
* 单源失败不许抛异常：一律进 `missing`，主流程继续。

已实测（2026-11-08，东财）可用性
--------------------------------
| 源 | 接口 | 结果 |
|---|---|---|
| kline | push2his .../kline/get (fqt=2) | ✅ 可用 |
| valuation | push2 .../stock/get?fltt=2&invt=2 (f43/f162/f167/f116/f117/f168/f50) | ✅ 可用 |
| fundamental | datacenter-web RPT_LICO_FN_CPD | ✅ 可用（含 YSTZ/SJLTZ/WEIGHTAVG_ROE/XSMLL/BOARD_CODE） |
| sector | push2 .../stock/get?secid=90.BKxxxx + clist fs=m:90+t:2（榜单取排名） | ✅ 可用 |
| breadth | push2 .../ulist.np/get?secids=1.000001,0.399001 (f104/f105/f106) | ✅ 可用（沪深两市，不含北交所） |
| limit_up | push2ex .../getTopicZTPool | ⚠️ 短窗口：**必须带 date**（不带 rc=102），且只保留最近约 20 个交易日（20261030 有 47 只，20261015 起返回 0 条）→ pit_backfillable=False |
| announcement | np-anotice-stock .../api/security/ann | ✅ 可用 |
| news | search-api-web .../search/jsonp | ✅ 可用（脆弱，正文里混入无关个股，需相关性过滤 + 去重） |

**已测不可用 / 不可用作为历史源**
* `push2 .../stock/get?secid=1.000001&fields=f104,f105,f106`（单只指数取涨跌家数）→ 恒返回 0/0/100，
  必须走 `ulist.np/get` 多标的批量接口。
* `push2ex .../getTopicZTPool` 不带 `date` → rc=102；带远期 date → tc=0（无数据留存），
  即它只是"最近约 20 个交易日"的滚动窗口，见上。
* `clist?fs=m:0+t:6,...&pz=...`（全市场分页统计宽度）→ 首测 RemoteDisconnected，
  重试可通，但 5563 只需 56 页 ≈ 12s/次，已弃用，改用 ulist 的指数聚合字段。
* `datacenter-web .../RPT_F10_FINANCE_MAINFINADATA`（另一套 F10 主指标）→ 可用但字段冗余、
  且 `filter=SECUCODE` 需带交易所后缀，不如 RPT_LICO_FN_CPD 直接，未采用。

可信度为什么是这些数（不许拍脑袋写 1.0）
----------------------------------------
见 `SOURCE_RELIABILITY` 每行注释。原则：**一手 > 二手，法定披露 > 媒体转述，
点值 > 聚合值，日频可核对 > 只能采一次的**。任何源的实际权重还会按"时效/去重/相关性"
做二次折扣（见各 `_xxx_item` 函数），所以最终 `reliability` 是"基准 × 折扣"，不是常数。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qcore import data as qdata

ROOT = qdata.ROOT
TZ = timezone(timedelta(hours=8))          # 东财全部时间戳均为北京时间
INFO_ROOT = os.path.join(ROOT, "data", "info")

# ---------------------------------------------------------------------------
# 源可信度基准
# ---------------------------------------------------------------------------
# 依据（每条都写清楚，别改数值不写理由）：
#   kline         1.00  交易所原始成交回报，唯一"零转述"源；本项目的量价全部由本地
#                      后复权 close 重算，不信任接口自带的 pct_chg。
#   valuation     0.90  东财实时快照的 PE(动态)/PB/市值，日频、可核对，但：①动态 PE 用
#                      最新单季年化，与 TTM 口径不同；②盘中会变；③停牌/极端行情下有延迟。
#   fundamental   0.85  财报口径（RPT_LICO_FN_CPD），权威但季度更新，且**有披露滞后**：
#                      t 日能看到的可能是一个月前甚至更早的报告期，故不给更高分。
#   sector        0.80  行业板块涨幅由成分股加权而来，与个股所属板块的"真实可比性"打折
#                      （个股常跨板块、东财板块口径 ≠ 申万口径）。
#   breadth       0.90  涨跌家数/成交额是聚合值，口径清晰、可复核；扣分项：不含北交所、
#                      只覆盖当前交易日。
#   announcement  0.95  法定信息披露，一手、时间戳准确（notice_date）；扣分项：只覆盖
#                      "已发布"，且**无法回补历史可见集合**（pit_backfillable=False）。
#   news          0.45  二手转述：同一事件常被 5+ 家媒体改写（实测 600519 三季报一条新闻
#                      在 10 条结果里重复 2 次），标题可能张冠李戴，且无法回补历史。
#                      实测还发现搜索结果会混入"顺带提到该股"的大盘综述 → 必须做相关性过滤。
SOURCE_RELIABILITY: Dict[str, float] = {
    "kline": 1.0,          # 交易所原始行情
    "valuation": 0.9,      # 东财估值，日频，可能有延迟
    "fundamental": 0.85,   # 财报口径，季度更新，存在披露滞后
    "sector": 0.8,
    "breadth": 0.9,
    "announcement": 0.95,  # 公告是法定披露，但只覆盖"已发布"
    "news": 0.45,          # 新闻：二手转述、可能重复、无法回补历史
}

#: 能否从历史接口真实重建"t 日当时可见的该源内容"。
#: kline/valuation/fundamental/sector/breadth = True（原则上可回补：K 线可截断重算，
#: PE/PB/财报/板块/宽度都有历史接口或可由历史行情重算）。
#: announcement/news = False（**禁止回测**，只能从部署日起前向采集）。
PIT_BACKFILLABLE: Dict[str, bool] = {
    "kline": True,
    "valuation": True,
    "fundamental": True,
    "sector": True,
    "breadth": True,
    "announcement": False,
    "news": False,
}

_MARKETS = ("CN",)
_MIN_INTERVAL = qdata._MIN_INTERVAL          # 0.22s，与 qcore.data 共用同一个限速器
_last_call = qdata._last_call                # 同一个 list 对象 → 全局限速
_UA = dict(qdata.UA)
_UA["Referer"] = "https://quote.eastmoney.com/"

QUOTE_FIELDS = ("f43,f57,f58,f60,f162,f167,f116,f117,f168,f50,f47,f48,"
                "f169,f170,f171,f127,f128")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class InfoItem:
    """一条信息记录。所有字段都必须能回答"何时/何源/多可信"。"""
    as_of: str            # 数据所描述的时间点（ISO8601，+08:00）；不是抓取时间
    source: str           # SOURCE_RELIABILITY 的键
    kind: str             # 更细的类别，如 price_volume/valuation/fundamental/sector/breadth/limit_up/announcement/news
    payload: dict
    reliability: float    # (0, 1]，基准 × 时效/去重/相关性折扣
    pit_backfillable: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _get_json(url: str, retries: int = 5, timeout: int = 20) -> Optional[dict]:
    """与 qcore.data._get_json 完全相同的限速/退避模式，但**失败返回 None 不抛异常**
    （单源失败绝不能让快照流程中断）。限速器与 qcore.data 共用，全局 0.22s 间隔。"""
    for i in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            _last_call[0] = time.time()
            return json.loads(body.decode("utf-8", errors="replace"))
        except Exception:                      # noqa: BLE001
            _last_call[0] = time.time()
            if i == retries - 1:
                return None
            time.sleep(1.2 * (i + 1))
    return None


def _get_text(url: str, retries: int = 3, timeout: int = 20) -> Optional[str]:
    """给 jsonp 之类非纯 JSON 的端点用。"""
    for i in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            _last_call[0] = time.time()
            return body.decode("utf-8", errors="replace")
        except Exception:                      # noqa: BLE001
            _last_call[0] = time.time()
            if i == retries - 1:
                return None
            time.sleep(1.2 * (i + 1))
    return None


def _now() -> datetime:
    return datetime.now(TZ)


def _iso(dt: datetime) -> str:
    return dt.astimezone(TZ).replace(microsecond=0).isoformat()


def _num(x: Any) -> Optional[float]:
    """东财缺值用 '-' / '' / None 表示 → 统一 None，不当 0。"""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "--", "None", "nan"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _norm_code(code: Any) -> str:
    s = str(code).strip().upper()
    s = s.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    s = re.sub(r"\D", "", s)
    return s.zfill(6) if 0 < len(s) <= 6 else ""


def _parse_as_of(as_of: Any) -> Optional[datetime]:
    if as_of is None or as_of == "":
        return None
    if isinstance(as_of, datetime):
        return as_of if as_of.tzinfo else as_of.replace(tzinfo=TZ)
    if isinstance(as_of, pd.Timestamp):
        dt = as_of.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=TZ)
    s = str(as_of).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=TZ)
        except ValueError:
            continue
    return None


def _reliability(base: float, *factors: float, lo: float = 0.05) -> float:
    """基准 × 折扣，裁剪到 (0, 1]；保留 4 位小数便于比对。"""
    v = float(base)
    for f in factors:
        v *= float(f)
    return round(min(1.0, max(lo, v)), 4)


def _item(source: str, kind: str, as_of: str, payload: dict,
          reliability: float, note: str = "") -> InfoItem:
    return InfoItem(as_of=as_of, source=source, kind=kind, payload=payload,
                    reliability=reliability,
                    pit_backfillable=PIT_BACKFILLABLE.get(source, False),
                    note=note)


# ---------------------------------------------------------------------------
# 1) 行情量价（kline，可回补：历史 K 线截断即真 PIT）
# ---------------------------------------------------------------------------
def _kline_item(code: str, cutoff: Optional[datetime]) -> tuple[InfoItem, str]:
    """返回 (item, 全量K线的最新交易日)。后者用来判定 live/backfill，必须在截断前取。"""
    df = qdata.load_kline(code, max_age_days=1.0)      # 失败会抛，由上层 catch
    if df is None or df.empty:
        raise RuntimeError("empty kline")
    full_last = str(pd.Timestamp(df["date"].max()).date())
    if cutoff is not None:
        df = df[df["date"] <= pd.Timestamp(cutoff.date())]
        if df.empty:
            raise RuntimeError("no bar on/before as_of")
    df = df.reset_index(drop=True)
    last = df.iloc[-1]
    close = float(last["close"])
    amt = df["amount"].astype(float)
    ret = df["close"].astype(float).pct_change()
    tail20 = ret.tail(20)
    high252 = float(df["close"].tail(252).max())
    payload = {
        "date": str(pd.Timestamp(last["date"]).date()),
        "close": round(close, 4),
        "pct_chg": round(float(last["pct_chg"]), 4),
        "amount": float(last["amount"]),
        "volume": float(last["volume"]),
        "turnover": _num(last.get("turnover")),
        "amplitude": _num(last.get("amplitude")),
        "amount_ma20": round(float(amt.tail(20).mean()), 2),
        "vol_ratio_20": round(float(last["amount"]) / float(amt.tail(20).mean()), 4)
        if amt.tail(20).mean() > 0 else None,
        "ret_5d": round(float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100, 4)
        if len(df) > 5 else None,
        "ret_20d": round(float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100, 4)
        if len(df) > 20 else None,
        "ret_60d": round(float(df["close"].iloc[-1] / df["close"].iloc[-61] - 1) * 100, 4)
        if len(df) > 60 else None,
        "vol_20d_annual": round(float(tail20.std(ddof=1)) * np.sqrt(252) * 100, 4)
        if tail20.notna().sum() > 2 else None,
        "high_252d": round(high252, 4),
        "dd_from_252d_high": round((close / high252 - 1) * 100, 4) if high252 > 0 else None,
        "bars": int(len(df)),
        "price_basis": "h2 后复权（乘法复权，本项目唯一口径）",
    }
    bar_date = payload["date"]
    note = ""
    if cutoff is not None and bar_date != str(cutoff.date()):
        note = (f"as_of 非交易日或尚未收盘，数据截至 {bar_date}；"
                f"量比类字段为『当日成交额/20日均额』，非交易所口径量比")
    else:
        note = "量比类字段为『当日成交额/20日均额』，非交易所口径量比"
    # 后复权价与今价不可直接比较，reliability 仍给满：这是唯一零转述源
    return _item("kline", "price_volume", f"{bar_date}T15:00:00+08:00",
                 payload, _reliability(SOURCE_RELIABILITY["kline"]), note), full_last


# ---------------------------------------------------------------------------
# 2) 估值（live only）
# ---------------------------------------------------------------------------
def _fetch_quote(code: str) -> Optional[dict]:
    """实时快照。fltt=2&invt=2 让东财直接返回**已缩放**的浮点值（否则 f43 是分）。"""
    url = (f"https://push2.eastmoney.com/api/qt/stock/get?secid={qdata.secid(code)}"
           f"&fltt=2&invt=2&fields={QUOTE_FIELDS}")
    d = _get_json(url)
    if not d:
        return None
    return d.get("data") or None


def _valuation_item(code: str, quote: dict, kline_close: Optional[float],
                    as_of_iso: str) -> InfoItem:
    pe = _num(quote.get("f162"))
    pb = _num(quote.get("f167"))
    price = _num(quote.get("f43"))
    payload = {
        "code": quote.get("f57") or code,
        "name": quote.get("f58"),
        "price": price,
        "prev_close": _num(quote.get("f60")),
        "pct_chg": _num(quote.get("f170")),
        "amplitude": _num(quote.get("f171")),
        "pe_dynamic": pe,            # 东财 f162 = 市盈率(动态)，单季年化，≠ TTM
        "pb": pb,
        "total_mv": _num(quote.get("f116")),
        "float_mv": _num(quote.get("f117")),
        "turnover": _num(quote.get("f168")),
        "vol_ratio": _num(quote.get("f50")),   # 这个是交易所口径量比（5日）
        "amount": _num(quote.get("f48")),
        "volume": _num(quote.get("f47")),
        "industry": quote.get("f127"),
        "region": quote.get("f128"),
        "pe_basis": "dynamic_annualized",
    }
    factors: List[float] = [1.0]
    notes: List[str] = []
    if pe is None or (pe is not None and pe <= 0):
        notes.append("PE 为负或缺失（当期亏损），估值只能用 PB")
        factors.append(0.95)
    if price is not None and kline_close is not None and price > 0:
        if abs(price - kline_close) / price > 0.005:
            notes.append("实时价与本地K线收盘价不一致（停牌/未收盘/复权口径差异），按折扣计权")
            factors.append(0.9)
    if pb is None:
        notes.append("PB 缺失")
        factors.append(0.95)
    notes.append("PE 为东财动态口径（单季年化），非 TTM；盘中取值会变动")
    return _item("valuation", "valuation", as_of_iso, payload,
                 _reliability(SOURCE_RELIABILITY["valuation"], *factors),
                 "；".join(notes))


# ---------------------------------------------------------------------------
# 3) 基本面（F10，可回补：历史报告期接口）
# ---------------------------------------------------------------------------
_F10_URL = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
            "?reportName=RPT_LICO_FN_CPD&columns=ALL"
            "&filter=(SECURITY_CODE%3D%22{code}%22)"
            "&pageNumber=1&pageSize={n}&sortColumns=REPORTDATE&sortTypes=-1")


def _fetch_f10(code: str, n: int = 6) -> List[dict]:
    d = _get_json(_F10_URL.format(code=code, n=n))
    if not d or not d.get("success"):
        return []
    return ((d.get("result") or {}).get("data") or [])


def _fundamental_item(code: str, rows: List[dict], cutoff: Optional[datetime],
                      as_of_iso: str) -> InfoItem:
    if not rows:
        raise RuntimeError("no F10 rows")
    use = rows
    if cutoff is not None:
        # PIT：只认 as_of 之前已经"披露"的报告期（NOTICE_DATE ≤ as_of）
        use = [r for r in rows
               if (_parse_as_of(str(r.get("NOTICE_DATE") or "")) or datetime(1900, 1, 1, tzinfo=TZ))
               <= cutoff]
        if not use:
            raise RuntimeError("no report disclosed on/before as_of")
    r = use[0]
    notice = _parse_as_of(str(r.get("NOTICE_DATE") or ""))
    report = _parse_as_of(str(r.get("REPORTDATE") or ""))
    lag = int((notice - report).days) if (notice and report) else None
    age = int((_now() - notice).days) if notice else None
    payload = {
        "report_date": str(r.get("REPORTDATE") or "")[:10],
        "notice_date": str(r.get("NOTICE_DATE") or "")[:10],
        "qdate": r.get("QDATE"),
        "revenue": _num(r.get("TOTAL_OPERATE_INCOME")),
        "net_profit": _num(r.get("PARENT_NETPROFIT")),
        "revenue_yoy": _num(r.get("YSTZ")),        # 营收同比 %
        "profit_yoy": _num(r.get("SJLTZ")),        # 归母净利同比 %
        "roe_weighted": _num(r.get("WEIGHTAVG_ROE")),
        "gross_margin": _num(r.get("XSMLL")),
        "eps": _num(r.get("BASIC_EPS")),
        "bps": _num(r.get("BPS")),
        "op_cashflow_ps": _num(r.get("MGJYXJJE")),
        "board_code": r.get("BOARD_CODE"),
        "board_name": r.get("BOARD_NAME"),
        "disclosure_lag_days": lag,
        "age_days_since_notice": age,
        "cumulative": True,                        # 东财该表为年初至今累计口径
    }
    factors = [1.0]
    notes = ["财报为累计口径（年初至今），非单季"]
    if age is not None and age > 120:
        factors.append(0.9)
        notes.append(f"最新已披露报告期距今 {age} 天，信息偏旧")
    if payload["revenue_yoy"] is None or payload["profit_yoy"] is None:
        factors.append(0.95)
        notes.append("同比字段缺失")
    aof = f"{(payload['notice_date'] or '')}T00:00:00+08:00" if payload["notice_date"] else as_of_iso
    return _item("fundamental", "fundamental", aof, payload,
                 _reliability(SOURCE_RELIABILITY["fundamental"], *factors), "；".join(notes))


# ---------------------------------------------------------------------------
# 4) 板块（行业涨幅 + 排名）
# ---------------------------------------------------------------------------
_BOARD_CACHE: Dict[str, Any] = {"ts": 0.0, "rows": [], "complete": False}
_BOARD_TTL = 600.0                                   # 进程内 10 分钟
_BOARD_FILE = os.path.join(ROOT, "data", "cache", "board_table.json")
_BOARD_FILE_TTL = 12 * 3600.0                        # 落盘 12 小时：一天只付一次 5 页的代价


def _board_table(force: bool = False) -> tuple[List[dict], bool]:
    """全部行业板块涨幅（m:90+t:2，实测 total≈496，pz 上限 100 → 必须分 5 页）。

    返回 (rows, complete)。**complete=False 时不许算排名**：少一页就会把名次算错，
    宁可 rank=None + note，也不要一个假名次。
    实测 clist 分页接口会被限流（RemoteDisconnected），所以：内存缓存 10 分钟 +
    落盘 12 小时 + 每页只重试 2 次（快速失败，别让一次快照为排名卡 12 秒）。
    """
    if (not force) and _BOARD_CACHE["rows"] and \
            (time.time() - _BOARD_CACHE["ts"]) < _BOARD_TTL:
        return _BOARD_CACHE["rows"], _BOARD_CACHE["complete"]
    if (not force) and os.path.exists(_BOARD_FILE):
        if (time.time() - os.path.getmtime(_BOARD_FILE)) < _BOARD_FILE_TTL:
            try:
                with open(_BOARD_FILE, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                rows = obj.get("rows") or []
                if rows:
                    _BOARD_CACHE.update(ts=time.time(), rows=rows,
                                        complete=bool(obj.get("complete")))
                    return rows, bool(obj.get("complete"))
            except Exception:               # noqa: BLE001
                pass
    rows: List[dict] = []
    total = None
    complete = False
    for pn in range(1, 7):
        url = ("https://push2.eastmoney.com/api/qt/clist/get"
               f"?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:90+t:2"
               "&fields=f12,f14,f3")
        d = _get_json(url, retries=2)
        data = (d or {}).get("data") or {}
        diff = data.get("diff") or []
        if total is None:
            total = _num(data.get("total"))
        if not diff:
            break
        rows.extend(diff)
        if len(diff) < 100:
            complete = True
            break
        if total is not None and len(rows) >= int(total):
            complete = True
            break
    if rows:
        _BOARD_CACHE.update(ts=time.time(), rows=rows, complete=complete)
        try:
            os.makedirs(os.path.dirname(_BOARD_FILE), exist_ok=True)
            with open(_BOARD_FILE, "w", encoding="utf-8") as f:
                json.dump({"ts": _iso(_now()), "total": total, "complete": complete,
                           "rows": rows}, f, ensure_ascii=False)
        except Exception:                   # noqa: BLE001
            pass
    return rows, complete


def _sector_item(code: str, quote: Optional[dict], board_code: Optional[str],
                 board_name: Optional[str], as_of_iso: str) -> InfoItem:
    bc = board_code
    bn = board_name or (quote or {}).get("f127")
    table, complete = _board_table()
    if not bc and bn:
        for r in table:
            if r.get("f14") == bn:
                bc = r.get("f12")
                break
    if not bc:
        raise RuntimeError("industry board unresolved")
    # 优先从已缓存的板块榜单里取涨幅/名次 —— 榜单里本来就有该板块，没必要再打一次接口
    # （实测单独问 90.BKxxxx 会偶发断连，少一次请求就少一个失败点）。
    row = next((r for r in table if r.get("f12") == bc), None)
    if row is not None:
        pct = _num(row.get("f3"))
        bname = row.get("f14") or bn
        amount = None
        origin = "board_table"
    else:
        d = _get_json(f"https://push2.eastmoney.com/api/qt/stock/get?secid=90.{bc}"
                      f"&fltt=2&invt=2&fields=f43,f57,f58,f170,f136", retries=3)
        bd = (d or {}).get("data") or None
        if not bd:
            raise RuntimeError(f"board quote empty for {bc}")
        pct = _num(bd.get("f170"))
        bname = bd.get("f58") or bn
        amount = _num(bd.get("f136"))
        origin = "board_quote"
    rank = None
    n_board = None
    if table and complete:
        ordered = sorted([r for r in table if _num(r.get("f3")) is not None],
                         key=lambda r: float(_num(r.get("f3"))), reverse=True)
        n_board = len(ordered)
        for i, r in enumerate(ordered, 1):
            if r.get("f12") == bc:
                rank = i
                break
    payload = {
        "board_code": bc,
        "board_name": bname,
        "board_pct_chg": pct,
        "board_amount": amount,
        "board_rank": rank,
        "board_total": n_board,
        "board_table_complete": complete,
        "pct_from": origin,
        "matched_by": "f10_board_code" if board_code else "quote_industry_name",
    }
    factors = [1.0]
    notes = ["板块涨幅为东财行业板块口径，非申万一级"]
    if not board_code:
        factors.append(0.85)
        notes.append("板块由行业名称反查，存在同名/跨板块误匹配风险")
    if rank is None:
        factors.append(0.95)
        notes.append("板块排名缺失（榜单不完整，拒绝给假名次）")
    return _item("sector", "sector", as_of_iso, payload,
                 _reliability(SOURCE_RELIABILITY["sector"], *factors), "；".join(notes))


# ---------------------------------------------------------------------------
# 5) 市场宽度 + 涨停数
# ---------------------------------------------------------------------------
#: 市场级信息（宽度/涨停/板块榜）与个股无关，批量采 300 只时不能每股都打一遍。
#: 进程内 TTL 缓存（默认 5 分钟）把单股快照的请求数从 ~11 降到 ~5。
_MKT_CACHE: Dict[str, Any] = {}
_MKT_TTL = 300.0


def _market_item(key: str, builder, *args) -> InfoItem:
    ent = _MKT_CACHE.get(key)
    if ent and (time.time() - ent[0]) < _MKT_TTL:
        return ent[1]
    it = builder(*args)
    _MKT_CACHE[key] = (time.time(), it)
    return it


def _breadth_item(as_of_iso: str) -> InfoItem:
    """沪深两市涨跌家数。

    两个实测坑：
      ① 单只指数 stock/get 的 f104/f105/f106 恒为 0/0/100（已测不可用），必须用 ulist.np/get；
      ② 一次塞两个 secid 的请求约 20% 概率被 RemoteDisconnected 掉线（5 次实测 4 成功），
         而单 secid 的请求 5/5 稳定 → 所以**分市场各发一次**，一个失败只降级不整块丢。
    """
    members = (("1.000001", "SH", "沪市"), ("0.399001", "SZ", "深市"))
    up = down = flat = 0
    amt = 0.0
    parts: Dict[str, Any] = {}
    failed: List[str] = []
    for sec, tag, _ in members:
        url = ("https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&invt=2"
               f"&secids={sec}&fields=f2,f3,f12,f14,f104,f105,f106,f6")
        d = _get_json(url, retries=3)
        diff = ((d or {}).get("data") or {}).get("diff") or []
        if not diff:
            failed.append(tag)
            continue
        r = diff[0]
        u, dn, fl = (_num(r.get("f104")) or 0), (_num(r.get("f105")) or 0), (_num(r.get("f106")) or 0)
        a = _num(r.get("f6")) or 0.0
        up += int(u); down += int(dn); flat += int(fl); amt += a
        parts[tag] = {"up": int(u), "down": int(dn), "flat": int(fl), "amount": a,
                      "index": r.get("f14")}
    if not parts:
        raise RuntimeError("ulist breadth empty for SH and SZ")
    tot = up + down + flat
    payload = {
        "up": up, "down": down, "flat": flat, "total": tot,
        "adv_dec_ratio": round(up / down, 4) if down else None,
        "up_ratio": round(up / tot * 100, 2) if tot else None,
        "amount_total": amt,
        "by_market": parts,
        "failed_markets": failed,
        "scope": "沪深两市（不含北交所）",
    }
    factors = [1.0 if not failed else 0.9]
    note = "指数聚合字段，不含北交所；只覆盖当前交易日"
    if failed:
        note += f"；{','.join(failed)} 未取到，宽度只覆盖部分市场"
    return _item("breadth", "breadth", as_of_iso, payload,
                 _reliability(SOURCE_RELIABILITY["breadth"], *factors), note)


def _limit_up_item(as_of_iso: str, day: Optional[str] = None) -> InfoItem:
    """涨停池。⚠️ 实测两件事：①不带 `date` 参数直接 rc=102（必须给日期）；
    ②只保留最近约 20 个交易日（传 20261030 还有 47 只，传 20261015 及更早返回 0 条）。
    所以它是**短窗口源**：当下研判可用，历史回补不可靠 → pit_backfillable=False。"""
    if not day:
        raise RuntimeError("need trade date for ztpool (no date -> rc=102)")
    d = _get_json("https://push2ex.eastmoney.com/getTopicZTPool?ut=7eea3edcaed734bea9cbfc24409ed989"
                  "&dpt=wz.ztzt&Pageindex=0&pagesize=10&sort=fbt%3Aasc"
                  f"&date={str(day).replace('-', '')}")
    if not d or d.get("rc") not in (0, None):
        raise RuntimeError(f"ztpool rc={None if not d else d.get('rc')}")
    data = d.get("data") or {}
    pool = data.get("pool") or []
    tc = data.get("tc")
    if not pool and not tc:
        raise RuntimeError(f"date {day} outside ztpool retention window (~20 trading days)")
    payload = {
        "limit_up_count": int(tc or 0),
        "pool_date": str(day),
        "sample": [{"code": p.get("c"), "name": p.get("n"), "boards": p.get("lbc"),
                    "industry": p.get("hybk")} for p in pool[:10]],
    }
    return _item("breadth", "limit_up", as_of_iso, payload,
                 _reliability(SOURCE_RELIABILITY["breadth"], 0.85),
                 "短窗口源：只保留最近约20个交易日（已实测），更早历史不可回补，禁止回测")


# ---------------------------------------------------------------------------
# 6) 公告 / 7) 新闻（pit_backfillable=False）
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"</?em>|</?b>|<[^>]{1,12}>")
_PUNCT_RE = re.compile(r"[\s\u3000·、，。！？；：“”‘’（）()《》〈〉【】\[\]{}<>\"'`~!@#$%^&*\-_=+|\\/,.:;?]+")


def _norm_title(t: Any) -> str:
    s = _TAG_RE.sub("", str(t or ""))
    return _PUNCT_RE.sub("", s)


def _daydiff(a: Any, b: Any) -> int:
    """两个 'YYYY-MM-DD' 相差的天数；任一缺失返回 0（视为同日，不阻断合并）。"""
    try:
        da = datetime.strptime(str(a)[:10], "%Y-%m-%d")
        db = datetime.strptime(str(b)[:10], "%Y-%m-%d")
        return abs((da - db).days)
    except Exception:                       # noqa: BLE001
        return 0


def _ts_num(ts: Any) -> float:
    """'YYYY-MM-DDTHH:MM:SS' → 可比较的时间戳（用于"最新在前"排序）。"""
    try:
        return datetime.strptime(str(ts)[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
    except Exception:                       # noqa: BLE001
        return 0.0


def _dedupe(rows: List[dict], key: str = "title", strict: bool = False,
            max_day_gap: int = 3) -> List[dict]:
    """同一事件被多源转述 → 合成一条，保留**最早发布**的那条（最接近一手），其余进 duplicates。

    判定（必须同时满足"日期接近"，否则同一主题的旧闻会被误合并 —— 实测教训：
    公告标题全部以"XX公司:XX公司关于…"开头，纯前缀匹配会把半年来不同公告揉成一条）：
      strict=True  → 归一化标题完全相同（公告用：只合并真正重复的同一份文件）；
      strict=False → 完全相同 / 互为前缀且长度比 ≥0.85 且 ≥12 字 / 前 16 字相同且数字集合相同。
    日期窗口 max_day_gap 天。
    """
    groups: List[dict] = []
    for r in rows:
        n = _norm_title(r.get(key))
        sig = tuple(re.findall(r"\d+(?:\.\d+)?", n))
        d = r.get("_d") or ""
        hit = None
        for g in groups:
            if _daydiff(d, g["_d"]) > max_day_gap:
                continue
            gn = g["_norm"]
            if n == gn:
                hit = g
                break
            if strict or not n or not gn:
                continue
            a, b = (n, gn) if len(n) <= len(gn) else (gn, n)
            if b.startswith(a) and len(a) >= 12 and len(a) / len(b) >= 0.85:
                hit = g
                break
            if len(n) >= 16 and n[:16] == gn[:16] and sig and sig == g["_sig"]:
                hit = g
                break
        if hit is None:
            groups.append({"_norm": n, "_sig": sig, "_d": d, "main": r, "dups": []})
        else:
            hit["dups"].append(r)
            if str(r.get("_ts", "")) < str(hit["main"].get("_ts", "")):
                hit["dups"].append(hit["main"])
                hit["main"] = r
    return groups


_ANN_URL = ("https://np-anotice-stock.eastmoney.com/api/security/ann"
            "?sr=-1&page_size={n}&page_index=1&ann_type=A&client_source=web&stock_list={code}")


def _announcement_items(code: str, cutoff: Optional[datetime], as_of_iso: str,
                        n: int = 30) -> List[InfoItem]:
    d = _get_json(_ANN_URL.format(n=n, code=code))
    if not d or not d.get("success"):
        raise RuntimeError("ann api failed")
    lst = ((d.get("data") or {}).get("list") or [])
    rows = []
    for r in lst:
        dt = _parse_as_of(str(r.get("notice_date") or r.get("display_time") or "")[:19])
        if dt is None:
            continue
        if cutoff is not None and dt > cutoff + timedelta(days=1):
            continue                      # PIT：未来公告一律丢弃
        rows.append({
            "title": r.get("title") or r.get("title_ch"),
            "notice_date": dt.strftime("%Y-%m-%d"),
            "art_code": r.get("art_code"),
            "columns": ",".join(c.get("column_name", "") for c in (r.get("columns") or [])),
            "url": f"https://data.eastmoney.com/notices/detail/{code}/{r.get('art_code')}.html",
            "_d": dt.strftime("%Y-%m-%d"),
            "_ts": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    if not rows:
        raise RuntimeError("no announcement on/before as_of")
    rows.sort(key=lambda r: r["_ts"], reverse=True)
    out: List[InfoItem] = []
    # 公告用 strict=True：只合并"归一化标题完全相同"的重复文件。实测若用宽松前缀匹配，
    # 半年内不同公告（都以"XX公司:XX公司关于…"开头）会被揉成一条，数量与日期全错。
    for g in _dedupe(rows, strict=True, max_day_gap=2)[:10]:
        m = g["main"]
        ndup = 1 + len(g["dups"])
        payload = {
            "title": _TAG_RE.sub("", str(m["title"])),
            "notice_date": m["notice_date"],
            "art_code": m["art_code"],
            "column": m["columns"],
            "url": m["url"],
            "dup_count": ndup,
            "duplicates": [{"title": _TAG_RE.sub("", str(x["title"])), "notice_date": x["notice_date"]}
                           for x in g["dups"][:5]],
        }
        rel = _reliability(SOURCE_RELIABILITY["announcement"], 1.0 / (1.0 + 0.5 * (ndup - 1)))
        out.append(_item("announcement", "announcement",
                         f"{m['notice_date']}T00:00:00+08:00", payload, rel,
                         "法定披露，一手；仅覆盖已发布公告，无法回补历史可见集合（禁止回测）"))
    out.sort(key=lambda i: i.as_of, reverse=True)      # 最新在前（去重后组序会乱）
    return out


_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp?cb=&param="


def _news_items(code: str, name: Optional[str], cutoff: Optional[datetime],
                as_of_iso: str, page_size: int = 20) -> List[InfoItem]:
    param = json.dumps({
        "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                                       "pageIndex": 1, "pageSize": page_size,
                                       "preTag": "<em>", "postTag": "</em>"}},
    }, ensure_ascii=False)
    txt = _get_text(_NEWS_URL + urllib.parse.quote(param))
    if not txt:
        raise RuntimeError("news api failed")
    try:
        d = json.loads(txt)
    except Exception:                      # noqa: BLE001
        raise RuntimeError("news payload not json")
    res = ((d.get("result") or {}).get("cmsArticleWebOld") or [])
    if not res:
        raise RuntimeError("news result empty")
    kws = [code] + ([str(name)] if name else [])
    rows = []
    for r in res:
        dt = _parse_as_of(str(r.get("date") or "")[:19])
        if dt is None:
            continue
        if cutoff is not None and dt > cutoff:
            continue                      # PIT：未来新闻一律丢弃
        title = _TAG_RE.sub("", str(r.get("title") or ""))
        content = _TAG_RE.sub("", str(r.get("content") or ""))
        in_title = any(k and k in title for k in kws)
        in_body = any(k and k in content for k in kws)
        if not (in_title or in_body):
            continue                      # 与个股无关，丢弃
        rows.append({
            "title": title, "content": content[:200], "media": r.get("mediaName"),
            "url": r.get("url"), "date": dt.strftime("%Y-%m-%d %H:%M"),
            "in_title": in_title, "_d": dt.strftime("%Y-%m-%d"),
            "_ts": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    if not rows:
        raise RuntimeError("no relevant news on/before as_of")
    rows.sort(key=lambda r: r["_ts"], reverse=True)
    out: List[InfoItem] = []
    # 相关性优先：标题命中个股的排前面（最多 8 条），仅正文提及的最多留 3 条 ——
    # 实测搜代码会命中大量"主力动向/机构评级"大盘综述，正文里只是顺带列到该股，
    # 全收进来会把"新闻"变成噪声。
    groups = _dedupe(rows, strict=False, max_day_gap=5)
    groups.sort(key=lambda g: (0 if g["main"]["in_title"] else 1,
                               -_ts_num(g["main"]["_ts"])))
    picked, n_body = [], 0
    for g in groups:
        if not g["main"]["in_title"]:
            if n_body >= 3:
                continue
            n_body += 1
        picked.append(g)
        if len(picked) >= 10:
            break
    for g in picked:
        m = g["main"]
        ndup = 1 + len(g["dups"])
        rel = _reliability(
            SOURCE_RELIABILITY["news"],
            1.0 if m["in_title"] else 0.75,          # 只正文提及 → 相关性折扣
            1.0 / (1.0 + 0.6 * (ndup - 1)),           # 被多源转述 → 单条权重下降
        )
        payload = {
            "title": m["title"],
            "date": m["date"],
            "media": m["media"],
            "url": m["url"],
            "snippet": m["content"],
            "in_title": m["in_title"],
            "dup_count": ndup,
            "duplicates": [{"media": x["media"], "date": x["date"], "title": x["title"]}
                           for x in g["dups"][:5]],
        }
        note = "二手转述，可能被改写/重复；无法回补历史，禁止回测"
        if ndup > 1:
            note += f"；同事件 {ndup} 源转述已合并并降权"
        if not m["in_title"]:
            note += "；仅正文提及，相关性弱"
        out.append(_item("news", "news",
                         m["_ts"] + "+08:00", payload, rel, note))
    out.sort(key=lambda i: i.as_of, reverse=True)
    return out


# ---------------------------------------------------------------------------
# 对外：snapshot
# ---------------------------------------------------------------------------
def _snap_shell(code: str, market: str, as_of: Any) -> dict:
    return {"code": code, "market": market,
            "as_of": _iso(_parse_as_of(as_of) or _now()),
            "fetched_at": _iso(_now()), "mode": "live",
            "items": [], "summary": {}, "missing": [], "timings_ms": {}}


def snapshot(code: str, market: str = "CN", as_of: str | None = None) -> dict:
    """采一次快照，返回 ``{items:[InfoItem...], summary:{...}, missing:[{source,reason}]}``。

    as_of=None       → live 模式：抓当前值，`as_of` 字段记录真实抓取/数据时间。
    as_of=过去某天    → backfill 模式：**只有 kline 用历史 K 线真实截断重建**；
                       估值/基本面/板块/宽度/公告/新闻这些"只服务当前"的接口一律进 missing，
                       绝不把今天的值贴到昨天（§0.0 第一铁律）。

    ⚠️ news / announcement 的 `pit_backfillable=False`：**禁止用它们做历史回测**，
    只能做"当下研判"；要参与复盘统计必须从部署日起逐日前向落盘（`save_snapshot`）。
    """
    t0 = time.perf_counter()
    code = _norm_code(code)
    mk = str(market or "CN").strip().upper()
    snap = _snap_shell(code, mk, as_of)
    missing: List[dict] = snap["missing"]
    items: List[InfoItem] = []

    def miss(src: str, reason: str) -> None:
        missing.append({"source": src, "reason": reason})

    if mk not in _MARKETS:
        for s in SOURCE_RELIABILITY:
            miss(s, f"market {mk} not supported by this module (CN only)")
        snap["summary"] = {"code": code, "market": mk, "n_items": 0, "n_sources": 0}
        snap["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return snap
    if not code:
        for s in SOURCE_RELIABILITY:
            miss(s, f"invalid code {code!r} (need 6 digits)")
        snap["summary"] = {"code": code, "market": mk, "n_items": 0, "n_sources": 0}
        snap["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return snap

    cutoff = _parse_as_of(as_of)
    now_iso = _iso(_now())

    # --- 存在性探针：东财对不存在的 secid 返回 rc=100 / data=null（不是抛异常），
    #     用它做快速校验，避免对错误代码做 5 次退避重试（kline 侧要 ~12s）。
    timings: Dict[str, float] = {}

    def _timed(name: str, fn, *a, **kw):
        """记录每个源的耗时（ms），写进 snap['timings_ms'] —— 限速+重试的锅要能查出来。"""
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            timings[name] = round((time.perf_counter() - t) * 1000, 1)

    quote = _timed("quote", _fetch_quote, code)
    if quote is None and _probe_rc_failed(code):
        for s in SOURCE_RELIABILITY:
            miss(s, f"code {code} not found (eastmoney rc=100, data=null)")
        snap["summary"] = {"code": code, "market": mk, "n_items": 0, "n_sources": 0}
        snap["timings_ms"] = timings
        snap["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return snap

    # --- 1. kline（可回补，且决定"最新交易日"→ 决定 live/backfill）
    kdf_close = None
    full_last = None

    try:
        ki, full_last = _timed("kline", _kline_item, code, cutoff)
        items.append(ki)
        kdf_close = ki.payload.get("close")
        snap["date"] = ki.payload["date"]
    except Exception as e:                  # noqa: BLE001
        miss("kline", f"{type(e).__name__}: {e}")
        snap["date"] = None

    # live 的判定必须用**未截断**的最新交易日：as_of 落在最新交易日或之后 → live；
    # 早于它 → backfill（此时只有 kline 是真的 PIT 重建）。
    if cutoff is None:
        live = True
    elif full_last is not None:
        live = str(cutoff.date()) >= full_last
    else:
        live = snap.get("date") is not None and str(cutoff.date()) >= snap["date"]
    snap["mode"] = "live" if live else "backfill"

    if not live:
        for s in ("valuation", "fundamental", "sector", "breadth", "announcement", "news"):
            miss(s, f"live-only source: cannot reconstruct as-of {str(cutoff.date())}; "
                    f"only kline is truly backfillable here")

    # --- 2. 估值
    f10_rows: List[dict] = []
    if live:
        if quote:
            try:
                items.append(_timed("valuation", _valuation_item,
                                    code, quote, kdf_close, now_iso))
            except Exception as e:          # noqa: BLE001
                miss("valuation", f"{type(e).__name__}: {e}")
        else:
            miss("valuation", "quote endpoint returned no data")

        # --- 3. 基本面
        try:
            f10_rows = _timed("fundamental", _fetch_f10, code)
            items.append(_fundamental_item(code, f10_rows, cutoff, now_iso))
        except Exception as e:              # noqa: BLE001
            miss("fundamental", f"{type(e).__name__}: {e}")

        # --- 4. 板块
        try:
            bc = f10_rows[0].get("BOARD_CODE") if f10_rows else None
            bn = f10_rows[0].get("BOARD_NAME") if f10_rows else None
            items.append(_timed("sector", _sector_item, code, quote, bc, bn, now_iso))
        except Exception as e:              # noqa: BLE001
            miss("sector", f"{type(e).__name__}: {e}")

        # --- 5. 宽度 + 涨停（市场级 → 进程内 TTL 缓存，批量采时每股不再重复请求）
        try:
            items.append(_timed("breadth", _market_item,
                                f"breadth:{snap.get('date')}", _breadth_item, now_iso))
        except Exception as e:              # noqa: BLE001
            miss("breadth", f"{type(e).__name__}: {e}")
        try:
            items.append(_timed("limit_up", _market_item, f"limit_up:{snap.get('date')}",
                                _limit_up_item, now_iso, snap.get("date")))
        except Exception as e:              # noqa: BLE001
            miss("limit_up", f"{type(e).__name__}: {e}")

        # --- 6. 公告
        try:
            items.extend(_timed("announcement", _announcement_items, code, cutoff, now_iso))
        except Exception as e:              # noqa: BLE001
            miss("announcement", f"{type(e).__name__}: {e}")

        # --- 7. 新闻
        try:
            nm = (quote or {}).get("f58")
            items.extend(_timed("news", _news_items, code, nm, cutoff, now_iso))
        except Exception as e:              # noqa: BLE001
            miss("news", f"{type(e).__name__}: {e}")

    snap["items"] = items
    snap["timings_ms"] = timings
    snap["summary"] = _summarize(code, mk, items, missing)
    snap["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return snap


def _probe_rc_failed(code: str) -> bool:
    """再问一次（走缓存般的单请求）拿 rc：东财对无效 secid 返回 rc=100。
    只在 quote 为 None 时调用，用于把"不存在"和"网络抖动"区分开。"""
    d = _get_json(f"https://push2.eastmoney.com/api/qt/stock/get?secid={qdata.secid(code)}"
                  f"&fltt=2&invt=2&fields=f57")
    if not d:
        return False
    return d.get("rc") not in (0, None) and not d.get("data")


def _summarize(code: str, market: str, items: List[InfoItem],
               missing: List[dict]) -> dict:
    """给 UI/报告用的扁平摘要（尽量浅，缺的就不写键）。"""
    by = {}
    for it in items:
        by.setdefault(it.source, []).append(it)
    s: Dict[str, Any] = {
        "code": code, "market": market,
        "n_items": len(items),
        "n_sources": len(by),
        "sources_ok": sorted(by),
        "sources_missing": sorted({m["source"] for m in missing}),
        "reliability_mean": round(float(np.mean([i.reliability for i in items])), 4) if items else None,
        "reliability_min": round(float(min(i.reliability for i in items)), 4) if items else None,
    }
    kv = by.get("kline", [])
    if kv:
        p = kv[0].payload
        s.update({"date": p["date"], "close": p["close"], "pct_chg": p["pct_chg"],
                  "amount": p["amount"], "turnover": p["turnover"],
                  "vol_ratio_20": p["vol_ratio_20"], "ret_20d": p["ret_20d"],
                  "vol_20d_annual": p["vol_20d_annual"],
                  "dd_from_252d_high": p["dd_from_252d_high"]})
    vv = by.get("valuation", [])
    if vv:
        p = vv[0].payload
        s.update({"name": p.get("name"), "pe_dynamic": p.get("pe_dynamic"),
                  "pb": p.get("pb"), "total_mv": p.get("total_mv"),
                  "float_mv": p.get("float_mv"), "industry": p.get("industry"),
                  "vol_ratio": p.get("vol_ratio")})
    fv = by.get("fundamental", [])
    if fv:
        p = fv[0].payload
        s.update({"report_date": p.get("report_date"),
                  "revenue_yoy": p.get("revenue_yoy"),
                  "profit_yoy": p.get("profit_yoy"),
                  "roe_weighted": p.get("roe_weighted")})
    sv = by.get("sector", [])
    if sv:
        p = sv[0].payload
        s.update({"board_name": p.get("board_name"), "board_pct_chg": p.get("board_pct_chg"),
                  "board_rank": p.get("board_rank"), "board_total": p.get("board_total")})
    bv = [i for i in by.get("breadth", []) if i.kind == "breadth"]
    if bv:
        p = bv[0].payload
        s.update({"up": p.get("up"), "down": p.get("down"),
                  "up_ratio": p.get("up_ratio"), "amount_total": p.get("amount_total")})
    lv = [i for i in by.get("breadth", []) if i.kind == "limit_up"]
    if lv:
        s["limit_up_count"] = lv[0].payload.get("limit_up_count")
    av = by.get("announcement", [])
    if av:
        s["ann_latest_date"] = av[0].payload.get("notice_date")
        s["ann_count"] = len(av)
    nv = by.get("news", [])
    if nv:
        s["news_count"] = len(nv)
        s["news_latest"] = nv[0].payload.get("date")
    return s


# ---------------------------------------------------------------------------
# 落盘 / 读取
# ---------------------------------------------------------------------------
def _root_path(root: str) -> str:
    return root if os.path.isabs(root) else os.path.join(ROOT, root)


def _snap_dir(code: str, market: str, root: str = "data/info") -> str:
    d = os.path.join(_root_path(root), market.upper(), _norm_code(code))
    os.makedirs(d, exist_ok=True)
    return d


def save_snapshot(snap: dict, root: str = "data/info") -> str:
    """追加写 `{root}/{market}/{code}/{YYYY}.jsonl`，返回文件路径。

    只追加、不改历史行（与 §0.0 的台账精神一致）；UTF-8 + ensure_ascii=False。
    年度按快照的 `as_of` 归年，便于按年切片读取。
    """
    code = _norm_code(snap.get("code"))
    mk = str(snap.get("market") or "CN").upper()
    year = str(snap.get("as_of") or _iso(_now()))[:4]
    path = os.path.join(_snap_dir(code, mk, root), f"{year}.jsonl")
    rec = dict(snap)
    rec["items"] = [i.to_dict() if isinstance(i, InfoItem) else i for i in snap.get("items", [])]
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def load_snapshots(code: str, market: str, start: Optional[str] = None,
                   end: Optional[str] = None, root: str = "data/info") -> pd.DataFrame:
    """读取历史快照 → **item 级** DataFrame（一行 = 一条 InfoItem），
    并附带该快照的 summary / missing / mode，便于直接按 source/reliability 过滤。

    列：code, market, date, as_of, fetched_at, mode, source, kind, reliability,
        pit_backfillable, note, payload(dict), summary(dict), missing(list)
    """
    code = _norm_code(code)
    mk = str(market or "CN").upper()
    base = os.path.join(_root_path(root), mk, code)
    cols = ["code", "market", "date", "as_of", "fetched_at", "mode", "source", "kind",
            "reliability", "pit_backfillable", "note", "payload", "summary", "missing"]
    if not os.path.isdir(base):
        return pd.DataFrame(columns=cols)
    s = _parse_as_of(start)
    e = _parse_as_of(end)
    rows: List[dict] = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".jsonl"):
            continue
        if s or e:
            yr = fn[:4]
            if s and yr < str(s.year):
                continue
            if e and yr > str(e.year):
                continue
        with open(os.path.join(base, fn), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = str(rec.get("date") or str(rec.get("as_of") or "")[:10])
                if s and d and d < str(s.date()):
                    continue
                if e and d and d > str(e.date()):
                    continue
                for it in rec.get("items", []):
                    rows.append({
                        "code": rec.get("code"), "market": rec.get("market"),
                        "date": d, "as_of": it.get("as_of"),
                        "fetched_at": rec.get("fetched_at"), "mode": rec.get("mode"),
                        "source": it.get("source"), "kind": it.get("kind"),
                        "reliability": it.get("reliability"),
                        "pit_backfillable": it.get("pit_backfillable"),
                        "note": it.get("note"), "payload": it.get("payload"),
                        "summary": rec.get("summary"), "missing": rec.get("missing"),
                    })
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["reliability"] = pd.to_numeric(df["reliability"], errors="coerce")
    return df.sort_values(["date", "fetched_at", "source"]).reset_index(drop=True)


def daily_summary(code: str, market: str = "CN", as_of: Optional[str] = None) -> dict:
    """给 UI/报告用的扁平摘要（单日一条）。

    优先读本地已落盘快照（`load_snapshots`，取该日最新一次抓取）；没有则现采一次
    （as_of 为过去某天时只有 kline 可用，其余进 missing，原因写清）。
    返回 dict，含 `_from` 说明来源（"disk" / "live"），缺数据时含 `_reason`。
    """
    mk = str(market or "CN").upper()
    cd = _norm_code(code)
    day = (_parse_as_of(as_of) or _now()).strftime("%Y-%m-%d")
    df = pd.DataFrame()
    for kw in ({"start": day, "end": day}, {"end": day}):
        try:
            df = load_snapshots(cd, mk, **kw)
        except Exception:                   # noqa: BLE001
            df = pd.DataFrame()
        if isinstance(df, pd.DataFrame) and not df.empty:
            break
    if isinstance(df, pd.DataFrame) and not df.empty:
        last_ts = df["fetched_at"].dropna().max()
        g = df[df["fetched_at"] == last_ts]
        summ = g["summary"].iloc[0] if len(g) else {}
        out = dict(summ) if isinstance(summ, dict) else {}
        out.update({"_from": "disk", "_as_of": str(last_ts),
                    "_date": str(g["date"].iloc[0])[:10] if len(g) else day,
                    "_requested": day})
        return out
    snap = snapshot(cd, mk, as_of)
    out = dict(snap.get("summary") or {})
    out.update({"_from": "live", "_as_of": snap.get("as_of"), "_date": snap.get("date") or day,
                "_mode": snap.get("mode"), "missing": snap.get("missing")})
    return out


if __name__ == "__main__":
    import sys
    c = sys.argv[1] if len(sys.argv) > 1 else "600519"
    t = time.perf_counter()
    sn = snapshot(c)
    dt = time.perf_counter() - t
    print(f"code={c} mode={sn['mode']} items={len(sn['items'])} "
          f"sources={sn['summary'].get('sources_ok')} elapsed={dt:.2f}s")
    for it in sn["items"]:
        print(f"  {it.source:12s} {it.kind:12s} rel={it.reliability:<6} "
              f"pit={it.pit_backfillable} as_of={it.as_of}")
    print("  timings_ms:", sn.get("timings_ms"))
    for m in sn["missing"]:
        print(f"  MISSING {m['source']}: {m['reason'][:120]}")
