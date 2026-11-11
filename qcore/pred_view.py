"""
统一预测视图（评审 R50，决策 3 的 C 方案）—— **只读合并，不动任何写入方**。

## 问题

同一个"预测"现在散在四处，各自有各自的字段名和口径：

| 来源 | 文件 | 记什么 | 关键字段 |
|---|---|---|---|
| 主台账 | `data/pred_ledger/{year}.jsonl` | 回放期每日打分（INV-7 只追加） | `as_of, code, score, status` |
| 名单 | `data/state/picks*.json` | 今日要买哪些 | `as_of, rows[].code, decision_source` |
| 模拟盘成交 | `data/paper/trades.jsonl` | 实际成交 | `date, code, side, price, price_basis` |
| 判断台账 | `data/claims/claims.jsonl` | 人的结论（不是模型分） | `id, claim, falsifier, review_by` |

于是"某只票某天到底被预测成什么"需要人记四套规则。

## 本模块做什么

只做 **读 + 拼**：给定 (code, as_of) 或给定日期区间，把四处能对上的记录拼成一屏。
**不写任何文件、不改写入方**（所以不存在迁移风险）。

## 明确不做

- 不合并成单一账本（那是 A 方案，要状态迁移 + 双写期）；
- 不重算任何指标（INV-8 精神：只读）；
- 不替口径打架 —— 每一行都带 `source` 与 `basis`，谁是谁一目了然。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_DIR = os.path.join(ROOT, "data", "pred_ledger")
STATE = os.path.join(ROOT, "data", "state")
CLAIMS = os.path.join(ROOT, "data", "claims", "claims.jsonl")
PAPER_TRADES = os.path.join(ROOT, "data", "paper", "trades.jsonl")


def _jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    out: List[Dict] = []
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
                except Exception:  # noqa: BLE001
                    continue
                if limit and len(out) >= limit:
                    break
    except Exception:  # noqa: BLE001
        pass
    return out


def _json(path: str) -> Dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def sources() -> Dict:
    """四处各自有多少条、最后更新时间 —— 先让人看见"账有几本"。"""
    rows = []
    for year in sorted({f[:4] for f in os.listdir(LEDGER_DIR)
                        if f.endswith(".jsonl") and f[:4].isdigit()}) if os.path.isdir(LEDGER_DIR) else []:
        p = os.path.join(LEDGER_DIR, f"{year}.jsonl")
        rows.append({"source": "pred_ledger", "name": f"{year}.jsonl",
                     "n": sum(1 for _ in open(p, encoding="utf-8", errors="replace")) if os.path.isfile(p) else 0,
                     "note": "主台账（INV-7 只追加；含历史重复，读数用 .dedup.jsonl）"})
    for fn, note in (("picks.json", "今日名单（h=1）"), ("picks_h5.json", "今日名单（h=5）")):
        d = _json(os.path.join(STATE, fn))
        if d:
            rows.append({"source": "picks", "name": fn, "n": len(d.get("rows") or []),
                         "as_of": d.get("as_of"), "decision_source": d.get("decision_source"),
                         "note": note})
    t = _jsonl(PAPER_TRADES)
    if t:
        rows.append({"source": "paper_trades", "name": "trades.jsonl", "n": len(t),
                     "note": "模拟盘成交（按 close_T 计价，见 fill_audit）"})
    c = _jsonl(CLAIMS)
    if c:
        rows.append({"source": "claims", "name": "claims.jsonl", "n": len(c),
                     "note": "人的判断（有 falsifier / review_by），与模型分不是一回事"})
    return {"ok": bool(rows), "n_sources": len(rows), "sources": rows,
            "note": "四本账并列，口径各自保留（决策 3 的 C 方案：只读合并，不合并写入）"}


def lookup(code: str, as_of: Optional[str] = None, limit_days: int = 5) -> Dict:
    """某只票最近几天在四本账里各是什么。"""
    c = str(code).strip().zfill(6) if str(code).strip().isdigit() else str(code).strip()
    items: List[Dict] = []

    # 1) 主台账
    if os.path.isdir(LEDGER_DIR):
        files = sorted([f for f in os.listdir(LEDGER_DIR) if f.endswith(".dedup.jsonl")
                        or (f.endswith(".jsonl") and f[:4].isdigit())], reverse=True)
        need_dedup = any(f.endswith(".dedup.jsonl") for f in files)
        for f in files:
            if need_dedup and not f.endswith(".dedup.jsonl"):
                continue                          # 有去重视图就只用它
            for r in _jsonl(os.path.join(LEDGER_DIR, f)):
                if str(r.get("code")) != c:
                    continue
                d = str(r.get("as_of") or "")[:10]
                if as_of and d != str(as_of)[:10]:
                    continue
                items.append({"source": "pred_ledger", "date": d,
                              "score": r.get("score"), "status": r.get("status"),
                              "horizon": r.get("horizon"),
                              "ret": r.get("ret"), "hit": r.get("hit"),
                              "basis": "回放期模型分（score）"})
            break

    # 2) 名单
    for fn, h in (("picks.json", 1), ("picks_h5.json", 5)):
        d = _json(os.path.join(STATE, fn))
        for row in (d.get("rows") or []):
            if str(row.get("code")) != c:
                continue
            items.append({"source": "picks", "date": str(d.get("as_of") or "")[:10],
                          "score": row.get("score") or row.get("gbdt_score"),
                          "rank": row.get("rank"), "held": row.get("held"),
                          "horizon": h,
                          "basis": f"名单（decision_source={d.get('decision_source') or '?'}）"})

    # 3) 模拟盘成交
    for r in _jsonl(PAPER_TRADES):
        if str(r.get("code")) != c:
            continue
        items.append({"source": "paper_trades", "date": str(r.get("date") or "")[:10],
                      "side": r.get("side"), "price": r.get("price"),
                      "price_basis": r.get("price_basis"),
                      "horizon": r.get("horizon"),
                      "basis": "模拟盘成交（close_T 计价）"})

    # 4) 判断台账（按标题/正文里出现的代码模糊匹配，不猜就跳过）
    for r in _jsonl(CLAIMS):
        blob = json.dumps(r, ensure_ascii=False)
        if c in blob:
            items.append({"source": "claims", "date": str(r.get("registered_at") or r.get("date") or "")[:10],
                          "claim": r.get("claim"), "falsifier": r.get("falsifier"),
                          "review_by": r.get("review_by"), "status": r.get("status"),
                          "basis": "人的判断（非模型分）"})

    items.sort(key=lambda x: (x.get("date") or "", x.get("source") or ""))
    by_src: Dict[str, int] = {}
    for it in items:
        by_src[it["source"]] = by_src.get(it["source"], 0) + 1
    return {"ok": bool(items), "code": c, "as_of": as_of,
            "n": len(items), "by_source": by_src, "items": items[-200:],
            "note": "每行都标了 source/basis —— 四本账口径不同，不做跨源加减"}


def summary() -> Dict:
    return {"ok": True, **sources(),
            "hint": "GET /api/predictions_view?code=600519[&as_of=2026-11-11] 查单票"}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        print(json.dumps(lookup(sys.argv[1],
                                sys.argv[2] if len(sys.argv) > 2 else None),
                         ensure_ascii=False, indent=1)[:3000])
    else:
        print(json.dumps(summary(), ensure_ascii=False, indent=1)[:2000])
