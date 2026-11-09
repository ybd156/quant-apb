"""
实时政策与事件快讯 —— 补上硬编码日历的时效缺口。

## 问题

`qcore/policy_calendar.py` 是**人工整理的 56 条**，最后一条 2025-06-18。
今天是 2026-11，已经 1.4 年没更新。UI 上「政策事件」页看起来像实时的，
实际是历史档案。

## 解法：接实时快讯 + 规则分类

两个已探活可用的源：

| 源 | 接口 | 时效 |
|---|---|---|
| 东财 7×24 | `newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_50_1_.html` | 分钟级 |
| 华尔街见闻 | `api-one.wallstcn.com/apiv1/content/lives` | 分钟级 |

**规则分类，不用 LLM**：关键词匹配 + 权重打分。理由是
① 零依赖（不引入模型）② 可解释（能说清为什么判成政策）③ 快（毫秒级）。

## 分类维度

- **type**：policy（政策）/ macro（宏观数据）/ geopolitics（地缘）/ company（公司）/ other
- **direction**：+1 利好 / −1 利空 / 0 中性
- **sectors**：命中板块关键词
- **level**：strong（有明确主体+动作）/ weak（只是提及）

## 输出

`data/state/events_live.json`，结构与 policy_calendar 的 events 兼容，
UI 可以直接拼在一起显示。**保留原文链接**，不编造。

## 去重

东财用 `url_unique`，见闻用 `id`。同一条新闻两个源都有的，按标题相似度去重
（取标题前 20 字做 key）。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
OUT = os.path.join(STATE, "events_live.json")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}

# ---------------------------------------------------------------- 关键词表
# (关键词, 权重)。权重越高越确定是这类事件。
POLICY_KW = [
    ("降准", 3), ("降息", 3), ("加息", 3), ("存款准备金", 3),
    ("证监会", 2), ("国务院", 2), ("国常会", 3), ("央行", 2), ("人民银行", 2),
    ("财政部", 2), ("发改委", 2), ("金融监管总局", 2), ("交易所", 1),
    ("印花税", 3), ("汇金", 3), ("平准基金", 3), ("国家队", 2),
    ("IPO", 1), ("注册制", 2), ("退市", 2), ("减持新规", 3), ("分红", 1),
    ("市值管理", 2), ("并购重组", 2), ("专项债", 2), ("国债", 1),
    ("政策", 1), ("监管", 1), ("试点", 1), ("指导意见", 2), ("征求意见", 1),
    # --- 英文（国外源用）---
    ("Fed ", 3), ("Federal Reserve", 3), ("FOMC", 3), ("rate cut", 3),
    ("rate hike", 3), ("Treasury", 2), ("SEC ", 2), ("stimulus", 3),
    ("regulation", 1), ("regulator", 2), ("subsidy", 2), ("bailout", 3),
    ("central bank", 2), ("policy", 1), ("bill ", 1), ("congress", 2),
]
MACRO_KW = [
    ("CPI", 3), ("PPI", 3), ("GDP", 3), ("PMI", 3), ("社融", 3),
    ("M2", 2), ("出口", 1), ("进口", 1), ("失业率", 2), ("工业增加值", 2),
    ("固定资产投资", 2), ("社零", 2), ("非农", 3), ("通胀", 2),
    # --- 英文 ---
    ("CPI", 3), ("PPI", 3), ("GDP", 3), ("PMI", 3), ("nonfarm", 3),
    ("payrolls", 3), ("inflation", 3), ("unemployment", 2), ("jobless", 2),
    ("retail sales", 2), ("earnings season", 1), ("yield", 1),
]
GEO_KW = [
    ("美联储", 3), ("关税", 3), ("制裁", 3), ("贸易战", 3), ("出口管制", 3),
    ("俄乌", 2), ("中东", 2), ("台海", 3), ("北约", 2), ("欧盟", 1),
    ("白宫", 2), ("特朗普", 2), ("拜登", 2), ("美联储主席", 2),
    # --- 英文 ---
    ("sanction", 3), ("tariff", 3), ("trade war", 3), ("export control", 3),
    ("White House", 2), ("Trump", 2), ("NATO", 2), ("OPEC", 2),
    ("election", 2), ("midterm", 2), ("geopolitic", 3), ("war", 1),
    ("China trade", 3), ("chip ban", 3),
]
COMPANY_KW = [
    ("业绩", 1), ("预告", 2), ("回购", 2), ("增持", 2), ("减持", 2),
    ("停牌", 2), ("复牌", 2), ("立案", 3), ("处罚", 3), ("问询函", 2),
    ("中标", 2), ("订单", 1), ("收购", 2), ("重组", 1), ("退市", 2),
]
BULL_KW = [
    ("降准", 3), ("降息", 3), ("下调", 1), ("宽松", 2), ("支持", 1),
    ("鼓励", 1), ("增持", 2), ("回购", 2), ("超预期", 2), ("回暖", 2),
    ("涨停", 1), ("创新高", 2), ("利好", 3), ("放宽", 2), ("刺激", 2),
    ("减税", 2), ("补贴", 2), ("提速", 1), ("扩大", 1),
    # --- 英文 ---
    ("surge", 2), ("rally", 2), ("beat", 2), ("record high", 2),
    ("upgrade", 2), ("bullish", 3), ("optimism", 2), ("rebound", 2),
    ("rate cut", 3), ("stimulus", 2), ("growth", 1),
]
BEAR_KW = [
    ("加息", 3), ("上调", 1), ("收紧", 2), ("制裁", 3), ("关税", 2),
    ("减持", 2), ("低于预期", 2), ("下滑", 1), ("亏损", 2), ("立案", 3),
    ("处罚", 2), ("跌停", 1), ("创新低", 2), ("利空", 3), ("退市", 2),
    ("暂停", 1), ("调查", 2), ("违约", 3),
    # --- 英文 ---
    ("plunge", 3), ("slump", 2), ("selloff", 3), ("miss", 2),
    ("downgrade", 2), ("bearish", 3), ("recession", 3), ("layoff", 3),
    ("bankruptcy", 3), ("default", 3), ("crash", 3), ("tariff", 2),
]

# 板块关键词 -> 标准板块名
SECTOR_MAP: Dict[str, List[str]] = {
    "银行": ["银行", "商业银行", "信贷"],
    "券商": ["券商", "证券公司", "证券业"],
    "地产": ["地产", "房地产", "楼市", "住房"],
    "半导体": ["半导体", "芯片", "集成电路", "光刻"],
    "新能源": ["新能源", "光伏", "风电", "储能", "锂电"],
    "医药": ["医药", "创新药", "医疗", "疫苗", "生物"],
    "白酒": ["白酒", "食品饮料"],
    "汽车": ["汽车", "新能源车", "整车"],
    "有色金属": ["有色", "铜", "铝", "稀土", "锂"],
    "军工": ["军工", "国防", "航天", "航空"],
    "建筑装饰": ["建筑", "基建", "工程"],
    "公用事业": ["电力", "燃气", "公用事业", "水务"],
    "传媒": ["传媒", "游戏", "影视", "出版"],
    "计算机": ["计算机", "软件", "信创", "AI", "人工智能", "算力"],
    "煤炭": ["煤炭", "焦煤"],
    "石油": ["石油", "原油", "石化"],
}


def _get(url: str, timeout: int = 12, retries: int = 3) -> str:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (i + 1))
    raise last  # type: ignore[misc]


def _score(text: str, table: List[tuple]) -> int:
    s = 0
    for kw, w in table:
        if kw in text:
            s += w
    return s


def classify(title: str, body: str = "") -> Dict:
    """规则分类。返回 {type, direction, sectors, level, scores}。"""
    t = f"{title} {body}"
    sc = {
        "policy": _score(t, POLICY_KW),
        "macro": _score(t, MACRO_KW),
        "geopolitics": _score(t, GEO_KW),
        "company": _score(t, COMPANY_KW),
    }
    typ = max(sc, key=lambda k: sc[k]) if max(sc.values()) > 0 else "other"
    if sc.get(typ, 0) == 0:
        typ = "other"

    bull = _score(t, BULL_KW)
    bear = _score(t, BEAR_KW)
    direction = 1 if bull > bear else (-1 if bear > bull else 0)

    sectors = [name for name, kws in SECTOR_MAP.items()
               if any(k in t for k in kws)]

    level = "strong" if sc.get(typ, 0) >= 3 else ("weak" if sc.get(typ, 0) > 0 else "none")
    return {"type": typ, "direction": direction, "sectors": sectors[:4],
            "level": level, "scores": sc, "bull": bull, "bear": bear}


# ---------------------------------------------------------------- 抓取
def fetch_eastmoney(limit: int = 50) -> List[Dict]:
    url = ("https://newsapi.eastmoney.com/kuaixun/v1/"
           f"getlist_102_ajaxResult_{limit}_1_.html")
    try:
        b = _get(url)
    except Exception:
        return []
    m = re.search(r"=\s*(\{.*\})", b, re.S)
    if not m:
        return []
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    out = []
    for x in (d.get("LivesList") or []):
        title = str(x.get("title") or "").strip()
        if not title:
            continue
        out.append({"id": str(x.get("url_unique") or title[:20]),
                    "source": "eastmoney",
                    "title": title,
                    "digest": str(x.get("digest") or "").strip(),
                    "time": str(x.get("showtime") or ""),
                    "url": str(x.get("url_unique") or "")})
    return out


def fetch_wallstcn(limit: int = 30) -> List[Dict]:
    url = ("https://api-one.wallstcn.com/apiv1/content/lives"
           f"?channel=global-channel&limit={limit}")
    try:
        d = json.loads(_get(url))
    except Exception:
        return []
    out = []
    for x in ((d.get("data") or {}).get("items") or []):
        title = str(x.get("title") or "").strip()
        body = str(x.get("content_text") or "").strip()
        if not title and not body:
            continue
        ts = x.get("display_time")
        try:
            tm = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            tm = ""
        out.append({"id": str(x.get("id") or (title or body)[:20]),
                    "source": "wallstcn",
                    "title": title or body[:60],
                    "digest": body[:300],
                    "time": tm,
                    "url": str(x.get("uri") or "")})
    return out


def _parse_rss(xml: str, source: str, limit: int = 30) -> List[Dict]:
    """
    极简 RSS/Atom 解析（正则，不引第三方库）。
    只取 <item>/<entry> 的 title/link/pubDate，够用即可。
    """
    out = []
    blocks = re.findall(r"<(?:item|entry)[^>]*>(.*?)</(?:item|entry)>", xml, re.S | re.I)
    for b in blocks[:limit]:
        def pick(tag):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", b, re.S | re.I)
            if not m:
                return ""
            v = m.group(1)
            v = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"", v, flags=re.S)
            v = re.sub(r"<[^>]+>", "", v)
            return v.strip()
        title = pick("title")
        link = pick("link") or ""
        if not link:
            m = re.search(r'<link[^>]*href="([^"]+)"', b, re.I)
            link = m.group(1) if m else ""
        date = pick("pubDate") or pick("published") or pick("updated")
        if not title:
            continue
        out.append({"id": link or title[:20], "source": source,
                    "title": title[:160], "digest": pick("description")[:300],
                    "time": _norm_time(date), "url": link})
    return out


def _norm_time(s: str) -> str:
    """把 RSS 的时间格式统一成 YYYY-MM-DD HH:MM:SS。失败则原样返回。"""
    s = (s or "").strip()
    if not s:
        return ""
    fmts = ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S")
    for f in fmts:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
    return s[:19]


def fetch_sina(limit: int = 30) -> List[Dict]:
    """新浪财经 7×24。"""
    url = ("https://zhibo.sina.com.cn/api/zhibo/feed"
           f"?page=1&page_size={limit}&zhibo_id=152&tag_id=0&dire=f&dpc=1")
    try:
        d = json.loads(_get(url))
    except Exception:
        return []
    feed = (((d.get("result") or {}).get("data") or {}).get("feed") or {})
    lst = feed.get("list") or []
    out = []
    for x in lst:
        t = str(x.get("rich_text") or x.get("content") or "").strip()
        t = re.sub(r"<[^>]+>", "", t)
        if not t:
            continue
        out.append({"id": str(x.get("id") or t[:20]), "source": "sina",
                    "title": t[:160], "digest": "",
                    "time": str(x.get("create_time") or ""), "url": ""})
    return out


def fetch_xueqiu(limit: int = 20) -> List[Dict]:
    """雪球热帖（HTML 里嵌 JSON）。"""
    try:
        b = _get("https://xueqiu.com/statuses/hot/listV2.json?since_id=-1&max_id=-1&size=20")
        d = json.loads(b)
    except Exception:
        return []
    out = []
    for x in (d.get("items") or [])[:limit]:
        t = str(x.get("title") or x.get("description") or "").strip()
        t = re.sub(r"<[^>]+>", "", t)
        if not t:
            continue
        out.append({"id": str(x.get("id") or t[:20]), "source": "xueqiu",
                    "title": t[:160], "digest": "", "time": "",
                    "url": "https://xueqiu.com" + str(x.get("target") or "")})
    return out


def fetch_rss_sources(limit: int = 25) -> List[Dict]:
    """
    国外 RSS 源。**只列实测在 NAS 上能通的**：
      CNBC / MarketWatch / ECB —— 这三条已探活
      Reuters / FT / GoogleNews / Investing 在 NAS 上网络不可达，不列。
    """
    feeds = [
        ("cnbc", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("ecb", "https://www.ecb.europa.eu/rss/press.html"),
    ]
    out = []
    for name, url in feeds:
        try:
            xml = _get(url, timeout=10)
        except Exception:
            continue
        out.extend(_parse_rss(xml, name, limit))
    return out


def _dedupe(items: List[Dict]) -> List[Dict]:
    """同一条新闻两个源都有 -> 保留先出现的。key 用标题前 20 字。"""
    seen, out = set(), []
    for it in items:
        k = re.sub(r"\s+", "", it.get("title", ""))[:20]
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def refresh(limit_em: int = 50, limit_wscn: int = 30, verbose: bool = True) -> Dict:
    """抓取 + 分类 + 落盘。"""
    raw = (fetch_eastmoney(limit_em) + fetch_wallstcn(limit_wscn)
           + fetch_sina(limit_em) + fetch_xueqiu(limit_wscn)
           + fetch_rss_sources(limit_wscn))
    if not raw:
        return {"ok": False, "reason": "all_sources_failed"}
    items = _dedupe(raw)

    rows = []
    dropped_irrelevant = 0
    for it in items:
        c = classify(it.get("title", ""), it.get("digest", ""))
        # 相关性过滤：三条至少满足一条才进事件流，否则是噪音
        #   ① 有明确方向（利好/利空）
        #   ② 是政策或宏观（天然与市场相关）
        #   ③ 命中了 A 股板块
        # 实测动机：CNBC 的「Trump offers UFC chief a job」会被判成 geopolitics，
        # 但对 A 股毫无意义，必须挡掉。
        relevant = (c["direction"] != 0
                    or c["type"] in ("policy", "macro")
                    or bool(c["sectors"]))
        if not relevant:
            dropped_irrelevant += 1
            continue
        rows.append({
            "eid": f"LIVE-{it['id']}"[:40],
            "date": (it.get("time") or "")[:10] or str(datetime.now().date()),
            "time": it.get("time"),
            "type": c["type"],
            "direction": c["direction"],
            "sectors": c["sectors"],
            "level": c["level"],
            "title": it["title"],
            "source": it["source"],
            "region": ("CN" if it["source"] in ("eastmoney", "wallstcn", "sina", "xueqiu")
                       else "global"),
            "url": it.get("url") or "",
            "note": (it.get("digest") or "")[:120],
            "reliability": 0.5,      # 快讯未经核实，低于人工整理日历的 1.0
            "pit_backfillable": False,   # 只能前向积累，不能回补
        })
    rows.sort(key=lambda r: r.get("time") or "", reverse=True)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_raw": len(raw), "n_deduped": len(items), "n_events": len(rows),
        "sources": {s: len([x for x in raw if x["source"] == s])
                    for s in sorted({x["source"] for x in raw})},
        "by_region": {"CN": sum(1 for r in rows if r.get("region") == "CN"),
                      "global": sum(1 for r in rows if r.get("region") == "global")},
        "dropped_irrelevant": dropped_irrelevant,
        "by_type": {t: sum(1 for r in rows if r["type"] == t)
                    for t in ("policy", "macro", "geopolitics", "company")},
        "rows": rows,
        "note": ("实时快讯经规则分类（关键词+权重），未经人工核实；"
                 "reliability=0.5，pit_backfillable=False（只能前向积累）"),
    }
    os.makedirs(STATE, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, OUT)
    if verbose:
        print(f"[news_live] 原始 {len(raw)} -> 去重 {len(items)} -> 事件 {len(rows)} "
              f"| {payload['by_type']}", flush=True)
    return {"ok": True, **payload}


def load() -> Dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def merge_into_events(verbose: bool = True) -> Dict:
    """
    把实时事件合并进 events.json 的 `live` 段（不覆盖人工整理的 56 条）。
    UI 可以两段一起显示，并区分 reliability。
    """
    live = load()
    if not live.get("rows"):
        return {"ok": False, "reason": "no_live_events",
                "hint": "先跑 refresh()"}
    p = os.path.join(STATE, "events.json")
    try:
        with open(p, encoding="utf-8") as f:
            ev = json.load(f)
    except Exception:
        ev = {"events": [], "rows": []}
    ev["live"] = live["rows"]
    ev["live_meta"] = {"generated_at": live.get("generated_at"),
                       "n": live.get("n_events"), "by_type": live.get("by_type"),
                       "note": live.get("note")}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)
    if verbose:
        print(f"[news_live] 已合并 {live['n_events']} 条实时事件到 events.json", flush=True)
    return {"ok": True, "merged": live["n_events"], "path": p}


if __name__ == "__main__":
    import sys
    r = refresh()
    if r.get("ok"):
        for x in r["rows"][:8]:
            d = {1: "利好", -1: "利空", 0: "中性"}[x["direction"]]
            print(f"  [{x['type']:<11}][{d}] {x['time']} {x['title'][:52]}")
            if x["sectors"]:
                print(f"      板块: {', '.join(x['sectors'])}")
        if "--merge" in sys.argv:
            print(merge_into_events())
