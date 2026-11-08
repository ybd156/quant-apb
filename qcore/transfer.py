"""
数据导入导出 —— 让用户能把数据带走、也能把数据带回来。

## 为什么需要

1. **导出**：用户要备份、要在 Excel 里看、要发给别人、要换机器。
   只给一个 JSON 打包不够——台账和成交表是表格数据，CSV 才有人能用。

2. **导入**：换机器、恢复备份、批量录入实盘成交、从别的系统迁移自选。
   没有导入，数据只能进不能出，等于把用户锁死。

## 设计原则

- **导出不改数值**：原样搬运，不做任何计算（INV-8）
- **导入默认合并，不覆盖**：宁可多一条，不能少一条
- **覆盖必须显式**：`replace=true` 才清空重写
- **校验失败整体拒绝**：不做"部分导入"，避免半残状态
- **每次导入都留报告**：导入了多少、跳过多少、为什么跳过

## 支持范围

| 数据 | 导出 | 导入 | 说明 |
|---|---|---|---|
| 自选 watchlist | JSON/CSV | ✅ 合并 | 按 code 去重 |
| 实盘成交 live_trades | JSON/CSV | ✅ 追加 | 只追加，不改历史（INV-7） |
| 账户配置 account | JSON | ✅ 覆盖 | 走 account_config.save 校验 |
| 推荐 picks | JSON/CSV | ❌ | 由模型生成，不该手改 |
| 预测台账 ledger | JSON/CSV | ❌ | 只追加（INV-7） |
| 因子重要性 | JSON/CSV | ❌ | 由模型生成 |
| 全部 state | JSON | ✅ 覆盖 | 需 replace=true，危险操作 |
"""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "state")
TRADES = os.path.join(STATE, "live_trades.jsonl")


# ------------------------------------------------------------------ 读
def _read_json(name: str):
    try:
        with open(os.path.join(STATE, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[Dict]:
    out = []
    if not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ 导出
def _to_csv(rows: List[Dict]) -> str:
    """行列表 -> CSV 文本。列取所有行的键并集，保持首次出现顺序。"""
    if not rows:
        return ""
    cols: List[str] = []
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        if isinstance(r, dict):
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return buf.getvalue()


# 导出目标：key -> (文件名, 取数函数)
def _sources() -> Dict[str, Tuple[str, callable, str]]:
    def picks():
        d = _read_json("picks.json") or {}
        return d.get("rows") or []

    def ledger():
        d = _read_json("ledger.json")
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for k in ("rows", "items", "data", "records"):
                v = d.get(k)
                if isinstance(v, list):
                    return v
        return []

    def watch():
        d = _read_json("watchlist.json") or {}
        return d.get("items") or []

    def trades():
        return _read_jsonl(TRADES)

    def factors():
        d = _read_json("factor_importance.json") or {}
        return d.get("rows") or []

    def raw(name):
        return lambda: _read_json(name)

    return {
        "picks": ("picks", picks, "csv"),
        "ledger": ("ledger", ledger, "csv"),
        "watch": ("watchlist", watch, "csv"),
        "trades": ("live_trades", trades, "csv"),
        "factors": ("factor_importance", factors, "csv"),
        "account": ("account", raw("account.json"), "json"),
        "status": ("status", raw("status.json"), "json"),
        "signals": ("weights", raw("weights.json"), "json"),
        "regime": ("regime", raw("regime.json"), "json"),
        "performance": ("performance", raw("performance.json"), "json"),
        "paper": ("paper", raw("paper.json"), "json"),
        "model": ("model", raw("model.json"), "json"),
        "names": ("names", raw("names.json"), "json"),
        "events": ("events", raw("events.json"), "json"),
        "market": ("market", raw("market.json"), "json"),
    }


def export(what: str = "all", fmt: str = "auto") -> Dict:
    """
    导出。返回 {"ok","filename","content_type","data"(bytes),"n"}。
    what=all 时打包成单个 JSON；否则按单项导出（表格类默认 CSV）。
    """
    src = _sources()
    if what == "all":
        bundle = {"_meta": {"exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "version": _version(), "app": "APB"}}
        n = 0
        for key, (fname, fn, _f) in src.items():
            try:
                v = fn()
            except Exception:
                continue
            if v is None or (isinstance(v, (list, dict)) and len(v) == 0):
                continue
            bundle[key] = v
            n += 1
        if not n:
            return {"ok": False, "reason": "no_data"}
        body = json.dumps(bundle, ensure_ascii=False, indent=1,
                          default=str, allow_nan=False).encode("utf-8")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return {"ok": True, "filename": f"apb-all-{stamp}.json",
                "content_type": "application/json; charset=utf-8",
                "data": body, "n": n, "what": "all", "format": "json"}

    if what not in src:
        return {"ok": False, "reason": "bad_what",
                "hint": f"what 取 all 或 {', '.join(sorted(src))}"}
    fname, fn, default_fmt = src[what]
    try:
        v = fn()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"[:120]}
    if v is None or (isinstance(v, (list, dict)) and len(v) == 0):
        return {"ok": False, "reason": "no_data", "what": what}

    use_fmt = default_fmt if fmt in ("auto", "") else fmt
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if use_fmt == "csv":
        if not isinstance(v, list):
            v = [v]
        text = _to_csv(v)
        return {"ok": True, "filename": f"apb-{fname}-{stamp}.csv",
                "content_type": "text/csv; charset=utf-8",
                "data": ("\ufeff" + text).encode("utf-8"),   # BOM 让 Excel 正确识别中文
                "n": len(v), "what": what, "format": "csv"}
    body = json.dumps(v, ensure_ascii=False, indent=1,
                      default=str, allow_nan=False).encode("utf-8")
    return {"ok": True, "filename": f"apb-{fname}-{stamp}.json",
            "content_type": "application/json; charset=utf-8",
            "data": body, "n": len(v) if isinstance(v, (list, dict)) else 1,
            "what": what, "format": "json"}


def _version() -> str:
    try:
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ 导入
def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def import_watch(items: List[Dict]) -> Dict:
    """合并自选。按 code 去重，已存在的跳过。"""
    if not isinstance(items, list):
        return {"ok": False, "reason": "watch 必须是数组"}
    path = os.path.join(STATE, "watchlist.json")
    cur = _read_json("watchlist.json") or {}
    have = {str(it.get("code")) for it in (cur.get("items") or []) if isinstance(it, dict)}
    added, skipped = [], []
    for it in items:
        if not isinstance(it, dict) or not it.get("code"):
            skipped.append({"reason": "no_code", "item": str(it)[:40]})
            continue
        c = str(it["code"]).strip()
        if c in have:
            skipped.append({"reason": "duplicate", "code": c})
            continue
        have.add(c)
        added.append({"code": c, "name": it.get("name") or "",
                      "added_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                      "added_price": it.get("added_price"), "last_price": it.get("last_price"),
                      "score": it.get("score"), "rank_pct": it.get("rank_pct"),
                      "pred": it.get("pred"), "ret_since_add": it.get("ret_since_add"),
                      "hit": it.get("hit")})
    cur["items"] = (cur.get("items") or []) + added
    cur["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _atomic_write(path, json.dumps(cur, ensure_ascii=False, indent=1, default=str))
    return {"ok": True, "added": len(added), "skipped": len(skipped),
            "skip_detail": skipped[:10]}


def import_trades(rows: List[Dict]) -> Dict:
    """追加实盘成交。只追加不改历史（INV-7）。按 (date,code,side,price,qty) 去重。"""
    if not isinstance(rows, list):
        return {"ok": False, "reason": "trades 必须是数组"}
    have = set()
    for r in _read_jsonl(TRADES):
        have.add((str(r.get("date")), str(r.get("code")),
                  str(r.get("side")), str(r.get("price")), str(r.get("qty"))))
    added, skipped = 0, 0
    lines = []
    for r in rows:
        if not isinstance(r, dict) or not r.get("code"):
            skipped += 1
            continue
        key = (str(r.get("date")), str(r.get("code")), str(r.get("side")),
               str(r.get("price")), str(r.get("qty")))
        if key in have:
            skipped += 1
            continue
        have.add(key)
        lines.append(json.dumps(r, ensure_ascii=False, default=str))
        added += 1
    if lines:
        os.makedirs(os.path.dirname(TRADES), exist_ok=True)
        with open(TRADES, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return {"ok": True, "added": added, "skipped": skipped}


def import_account(cfg: Dict) -> Dict:
    from . import account_config as AC
    new, err = AC.save(cfg if isinstance(cfg, dict) else {})
    if err:
        return {"ok": False, "reason": "invalid_config", "detail": err}
    return {"ok": True, "config": new}


def import_all(bundle: Dict, replace: bool = False) -> Dict:
    """
    导入整体包（export(what='all') 的产物）。
    replace=false 时只合并 watch/trades/account，其余跳过；
    replace=true 时才覆盖 state 文件（危险）。
    """
    if not isinstance(bundle, dict):
        return {"ok": False, "reason": "body 必须是 JSON 对象"}
    report = {"ok": True, "mode": "replace" if replace else "merge", "actions": {}}
    if "watch" in bundle:
        report["actions"]["watch"] = import_watch(
            (bundle["watch"] or {}).get("items") if isinstance(bundle["watch"], dict)
            else bundle["watch"])
    if "trades" in bundle:
        report["actions"]["trades"] = import_trades(bundle["trades"])
    if "account" in bundle:
        report["actions"]["account"] = import_account(bundle["account"])

    if replace:
        # 危险：覆盖 state 文件。只允许白名单，且必须是 dict/list。
        allow = {"status.json", "weights.json", "regime.json", "ledger.json",
                 "performance.json", "paper.json", "model.json", "names.json",
                 "events.json", "factor_importance.json", "picks.json"}
        mapping = {"status": "status.json", "signals": "weights.json",
                   "regime": "regime.json", "ledger": "ledger.json",
                   "performance": "performance.json", "paper": "paper.json",
                   "model": "model.json", "names": "names.json",
                   "events": "events.json", "factors": "factor_importance.json",
                   "picks": "picks.json"}
        written = []
        for k, fn in mapping.items():
            if k not in bundle or fn not in allow:
                continue
            v = bundle[k]
            if not isinstance(v, (dict, list)):
                continue
            _atomic_write(os.path.join(STATE, fn),
                          json.dumps(v, ensure_ascii=False, indent=1, default=str))
            written.append(fn)
        report["actions"]["replaced"] = written
    return report


def import_payload(body: Dict) -> Dict:
    """统一入口：按 body 里出现的键分派。"""
    if not isinstance(body, dict):
        return {"ok": False, "reason": "body 必须是 JSON 对象"}
    if any(k in body for k in ("status", "signals", "regime", "performance", "paper", "model")) \
            or body.get("_meta"):
        return import_all(body, replace=bool(body.get("_replace")))
    acts = {}
    if "watch" in body:
        acts["watch"] = import_watch(body["watch"] if isinstance(body["watch"], list)
                                     else (body["watch"] or {}).get("items", []))
    if "trades" in body:
        acts["trades"] = import_trades(body["trades"])
    if "account" in body:
        acts["account"] = import_account(body["account"])
    if not acts:
        return {"ok": False, "reason": "nothing_to_import",
                "hint": "支持 watch / trades / account，或整体 state 包"}
    return {"ok": True, "actions": acts}


def summary() -> Dict:
    """导出/导入能力清单，给 UI 展示。"""
    return {"ok": True,
            "export": sorted(_sources().keys()) + ["all"],
            "import": ["watch", "trades", "account", "all(replace=true)"],
            "note": "表格类默认 CSV（带 BOM，Excel 直接打开不乱码）；导入默认合并不覆盖"}


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
    for w in ("picks", "watch", "trades", "account", "all"):
        r = export(w)
        print(f"  {w:<10} ok={r.get('ok')} n={r.get('n')} "
              f"file={r.get('filename')} bytes={len(r.get('data') or b'')}")
