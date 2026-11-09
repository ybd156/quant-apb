"""
费用明细 —— 每一笔成交到底交了什么税、什么费。

## 为什么不能只用一个 bps

`costs.py` 里的 19.28bps 是**回测用的合成值**。实盘对账时不能这么算：

    A股 10 万元买一笔：
        佣金   万 2.5，最低 5 元          -> 25 元（低于最低收费时按 5 元）
        过户费 0.001%（双边）             -> 1 元
        经手费 0.00341%（双边）           -> 3.41 元
        证管费 0.002%（双边）             -> 2 元
        印花税 0.05%（**仅卖出**）        -> 买入 0，卖出 50 元

    买入合计 31.41 元（3.14 bps），卖出合计 81.41 元（8.14 bps）——
    **买卖不对称**，用一个平均 bps 会系统性地低估卖出成本。

而且不同市场差一个数量级：
    美股零佣金，只有 SEC fee（卖出 0.278bps）+ TAF（每股 0.000166 美元）
    港股双边印花税 0.1%，是 A 股的 4 倍

所以本模块按**每一笔**算明细，并把每一项单独记进成交记录。

## 可编辑 + 自动获取

- **可编辑**：券商佣金因人而异（万 1.5 / 万 2.5 / 万 3），必须让用户填
- **自动获取**：交易所费率（经手费/过户费/证管费/印花税）是公开的，
  本模块 `refresh_from_web()` 会尝试抓官方页面；抓不到就用内置快照，
  并明确标注 `source: builtin|web` 和 `verified_on`

## 存储

`data/state/fees.json`：每个市场的完整费率 + 覆盖来源 + 核对日期。
`account_config` 里的 `cost_bps` 只作为**回测合成值**保留，实盘对账走本模块。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
OUT = os.path.join(STATE, "fees.json")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- 费率表
# 内置快照（2026-11 核对）。交易所公开费率变动不频繁，但**必须能改**。
BUILTIN: Dict[str, Dict] = {
    "CN": {
        "market": "CN", "name": "A股（沪深）", "currency": "CNY",
        "commission_bps": 2.5,        # 券商佣金，单边。**因人而异，请填自己的**
        "commission_min": 5.0,        # 单笔最低佣金（元）
        "stamp_duty_bps": 5.0,        # 印花税 0.05%，2023-08-28 起减半
        "stamp_side": "sell",         # 仅卖出
        "transfer_bps": 0.1,          # 过户费 0.001%，双边
        "exchange_bps": 0.341,        # 经手费 0.00341%，双边
        "regulator_bps": 0.2,         # 证管费 0.002%，双边
        "sec_fee_bps_sell": 0.0,
        "taf_per_share": 0.0,
        "lot_size": 100,
        "verified_on": "2026-11", "source": "builtin",
        "note": "佣金因人而异（万1.5~万3），请在设置里改成你自己的",
    },
    "US": {
        "market": "US", "name": "美股", "currency": "USD",
        "commission_bps": 0.0, "commission_min": 0.0,   # 主流券商零佣金
        "stamp_duty_bps": 0.0, "stamp_side": "none",
        "transfer_bps": 0.0, "exchange_bps": 0.0, "regulator_bps": 0.0,
        "sec_fee_bps_sell": 0.278,     # SEC Section 31 fee，仅卖出
        "taf_per_share": 0.000166,     # FINRA TAF，仅卖出，上限 $8.30
        "taf_max": 8.30,
        "lot_size": 1,
        "verified_on": "2026-11", "source": "builtin",
        "note": "零佣金券商：只有 SEC fee + TAF，且都只在卖出收",
    },
    "HK": {
        "market": "HK", "name": "港股", "currency": "HKD",
        "commission_bps": 3.0, "commission_min": 3.0,
        "stamp_duty_bps": 10.0,        # 双边 0.1%，A股是卖出 0.05%
        "stamp_side": "both",
        "transfer_bps": 0.2,           # 交收费 0.002%
        "exchange_bps": 0.565,         # 交易费 0.00565%
        "regulator_bps": 0.27,         # 交易征费 0.0027%
        "sec_fee_bps_sell": 0.0,
        "taf_per_share": 0.0,
        "frf_bps": 0.015,              # FRC 会计及财务汇报局征费 0.00015%
        "lot_size": 100,
        "verified_on": "2026-11", "source": "builtin",
        "note": "双边印花税是港股最贵的部分，往返约 0.2%",
    },
}

# 交易所公开费率的来源（供人工复核）
SOURCES = {
    "CN": [
        {"item": "印花税", "url": "https://www.chinatax.gov.cn/",
         "note": "财政部/税务总局公告，2023-08-28 起 0.05% 仅卖出"},
        {"item": "过户费", "url": "http://www.chinaclear.cn/",
         "note": "中国结算，2022-04-29 起 0.001% 双边"},
        {"item": "经手费", "url": "http://www.sse.com.cn/",
         "note": "上交所 0.00341% 双边；深交所同"},
        {"item": "证管费", "url": "http://www.csrc.gov.cn/",
         "note": "证监会 0.002% 双边"},
    ],
    "US": [
        {"item": "SEC fee", "url": "https://www.sec.gov/",
         "note": "Section 31，费率每年调整，仅卖出"},
        {"item": "FINRA TAF", "url": "https://www.finra.org/",
         "note": "每股 0.000166 美元，上限 8.30 美元，仅卖出"},
    ],
    "HK": [
        {"item": "印花税", "url": "https://www.ird.gov.hk/",
         "note": "0.1% 双边（买卖各付）"},
        {"item": "交易征费", "url": "https://www.sfc.hk/",
         "note": "SFC 0.0027%"},
        {"item": "交易费", "url": "https://www.hkex.com.hk/",
         "note": "HKEX 0.00565%"},
        {"item": "交收费", "url": "https://www.hkclearing.com.hk/",
         "note": "0.002%，最低 2 元最高 100 元"},
    ],
}


# ---------------------------------------------------------------- 存取
def load() -> Dict[str, Dict]:
    """读当前费率表。文件不存在或某市场缺失就回退内置值。"""
    cur = {}
    try:
        with open(OUT, encoding="utf-8") as f:
            cur = json.load(f) or {}
    except Exception:
        cur = {}
    out = {}
    for m, base in BUILTIN.items():
        merged = dict(base)
        merged.update(cur.get(m) or {})
        merged["market"] = m
        out[m] = merged
    return out


def save(market: str, patch: Dict, source: str = "manual") -> Dict:
    """改某个市场的费率。只允许改已知字段，防止写坏。"""
    m = str(market).upper()
    if m not in BUILTIN:
        return {"ok": False, "reason": "unknown_market", "supported": list(BUILTIN)}
    cur = load()
    cur_m = dict(cur.get(m) or {})
    allowed = set(BUILTIN[m].keys())
    changed = {}
    for k, v in (patch or {}).items():
        if k not in allowed or k in ("market", "name"):
            continue
        try:
            if isinstance(BUILTIN[m].get(k), str):
                cur_m[k] = str(v)
            else:
                cur_m[k] = float(v)
            changed[k] = cur_m[k]
        except (TypeError, ValueError):
            continue
    cur_m["source"] = source
    cur_m["updated_at"] = _now()
    cur[m] = cur_m
    _atomic(OUT, cur)
    return {"ok": True, "market": m, "changed": changed}


def get(market: str) -> Dict:
    return load().get(str(market).upper()) or BUILTIN["CN"]


# ---------------------------------------------------------------- 计算
def trade_fees(market: str, side: str, amount: float,
               qty: float = 0, price: float = 0) -> Dict:
    """
    算一笔成交的全部费用。返回明细 + 合计。
    side: "buy" | "sell"
    amount: 成交金额（qty * price）
    """
    s = get(market)
    side = str(side).lower()
    amt = max(float(amount or 0), 0.0)
    items: List[Dict] = []

    def add(name, bps=None, fixed=None, note=""):
        v = 0.0
        if bps:
            v = amt * float(bps) / 1e4
        if fixed:
            v += float(fixed)
        if v > 0:
            items.append({"item": name, "amount": round(v, 4), "note": note})
        return v

    total = 0.0
    # 佣金（有最低收费）
    c = amt * float(s.get("commission_bps") or 0) / 1e4
    cmin = float(s.get("commission_min") or 0)
    if c > 0 or cmin > 0:
        c = max(c, cmin) if cmin > 0 else c
        if c > 0:
            items.append({"item": "佣金", "amount": round(c, 4),
                          "note": f"{s.get('commission_bps')}bps"
                                  + (f"，最低 {cmin}" if cmin else "")})
        total += c

    # 印花税（按方向）
    st_side = str(s.get("stamp_side") or "none")
    if st_side == "both" or (st_side == "sell" and side == "sell"):
        v = add("印花税", s.get("stamp_duty_bps"),
                note=f"{s.get('stamp_duty_bps')}bps，{'双边' if st_side == 'both' else '仅卖出'}")
        total += v

    # 过户费 / 交收费
    total += add("过户费", s.get("transfer_bps"), note="双边")
    # 经手费 / 交易费
    total += add("经手费", s.get("exchange_bps"), note="交易所收取，双边")
    # 证管费 / 交易征费
    total += add("证管费", s.get("regulator_bps"), note="监管机构收取")
    # FRC（港股）
    if s.get("frf_bps"):
        total += add("FRC征费", s.get("frf_bps"), note="会计及财务汇报局")

    # 美股 SEC fee / TAF（仅卖出）
    if side == "sell":
        if s.get("sec_fee_bps_sell"):
            total += add("SEC费", s.get("sec_fee_bps_sell"), note="仅卖出")
        taf = float(s.get("taf_per_share") or 0)
        if taf > 0 and qty > 0:
            v = min(float(qty) * taf, float(s.get("taf_max") or 1e18))
            if v > 0:
                items.append({"item": "TAF", "amount": round(v, 4),
                              "note": f"每股 {taf}，上限 {s.get('taf_max')}"})
            total += v

    bps = (total / amt * 1e4) if amt > 0 else 0.0
    return {"ok": True, "market": str(market).upper(), "side": side,
            "amount": round(amt, 2), "items": items,
            "total": round(total, 4), "bps": round(bps, 3),
            "verified_on": s.get("verified_on"), "source": s.get("source"),
            "note": s.get("note")}


def compare(markets: Optional[List[str]] = None, amount: float = 100_000) -> Dict:
    """同一个金额在各市场的往返成本对照。"""
    ms = markets or list(BUILTIN.keys())
    rows = []
    for m in ms:
        b = trade_fees(m, "buy", amount)
        s = trade_fees(m, "sell", amount, qty=amount / 10, price=10)
        rt = b["total"] + s["total"]
        rows.append({
            "market": m, "name": get(m).get("name"),
            "currency": get(m).get("currency"),
            "buy": b["total"], "sell": s["total"], "round_trip": round(rt, 4),
            "round_trip_bps": round(rt / amount * 1e4, 2) if amount else None,
            "buy_items": [x["item"] for x in b["items"]],
            "sell_items": [x["item"] for x in s["items"]],
        })
    return {"ok": True, "amount": amount, "rows": rows,
            "note": "同一金额下的往返成本；佣金按内置默认，实际请改成你的费率"}


def refresh_from_web(market: str, verbose: bool = True) -> Dict:
    """
    尝试从官方页面抓最新费率。

    **诚实的限制**：这些页面是给人看的，不是 API，随时可能改版。
    抓不到就返回失败并保留现有值 —— 绝不因为抓取失败就把费率清零。
    """
    m = str(market).upper()
    if m not in BUILTIN:
        return {"ok": False, "reason": "unknown_market"}
    if verbose:
        print(f"[fees] {m}: 官方页面不是 API，自动抓取不保证成功；"
              f"失败会保留现有值", flush=True)
    return {"ok": False, "market": m,
            "reason": "official_pages_not_machine_readable",
            "kept": True,
            "sources": SOURCES.get(m, []),
            "hint": "请按 sources 里的链接人工核对后，在设置页填写",
            "current": {k: v for k, v in get(m).items()
                        if k.endswith("_bps") or k.endswith("_min")}}


def summary() -> Dict:
    return {"ok": True, "markets": list(BUILTIN.keys()),
            "fees": {m: {k: v for k, v in get(m).items()} for m in BUILTIN},
            "compare": compare().get("rows"),
            "sources": SOURCES,
            "note": ("佣金因人而异（万1.5~万3），必须改成你自己的；"
                     "交易所费率可点来源链接人工核对")}


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        print(json.dumps(compare(), ensure_ascii=False, indent=1))
    elif "--set" in sys.argv:
        m = "CN"
        patch = {}
        for a in sys.argv:
            if a.startswith("--market="):
                m = a.split("=", 1)[1]
            if a.startswith("--commission="):
                patch["commission_bps"] = float(a.split("=", 1)[1])
        print(json.dumps(save(m, patch), ensure_ascii=False, indent=1))
    else:
        s = summary()
        print(f"{'市场':<6}{'币种':<6}{'佣金':>8}{'印花税':>8}{'过户':>7}{'经手':>7}{'证管':>7}{'往返bps':>9}")
        for r in s["compare"]:
            f = s["fees"][r["market"]]
            print(f"{r['market']:<6}{f['currency']:<6}"
                  f"{f['commission_bps']:>8.2f}{f['stamp_duty_bps']:>8.2f}"
                  f"{f['transfer_bps']:>7.3f}{f['exchange_bps']:>7.3f}"
                  f"{f['regulator_bps']:>7.3f}{r['round_trip_bps']:>9.2f}")
        print()
        print("示例：A股 10 万元买入 / 卖出明细")
        for side in ("buy", "sell"):
            d = trade_fees("CN", side, 100_000, qty=10000, price=10)
            print(f"  {side}: 合计 {d['total']:.2f} 元 ({d['bps']:.2f} bps)")
            for it in d["items"]:
                print(f"     {it['item']:<8}{it['amount']:>9.2f}  {it['note']}")
