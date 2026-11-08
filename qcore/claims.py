"""
判断台账 —— 让"我给你的结论"也可以被对账。

## 为什么需要这个

普通的技术评估有个通病：**说对了没人记，说错了也没人记**。
于是"AI 分析"永远显得很准 —— 因为它只在你记得的那几次里出现。

这个模块做的事很简单，但很硬：
1. **先登记，后验证**（pre-registration）。任何判断在写下来的那一刻，
   就必须同时写清楚：用什么指标检验、阈值多少、什么时候复核。
   不允许事后修改判断内容，只允许追加"验证结果"。
2. **你的纠正优先**。你说"这条不对，因为 XXX"，就记成 refuted 并保留你的理由。
   系统不辩解，只统计。
3. **给我自己算命中率**。confirmed / (confirmed + refuted) —— 这个数字低，
   说明我的"前景判断"不可信，你应该降低对我的信任权重。

这跟预测台账（pred_ledger）是同一个哲学：
**先封存，后对账；不许改历史，只许补结果。**
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIM_DIR = os.path.join(ROOT, "data", "claims")
os.makedirs(CLAIM_DIR, exist_ok=True)
CLAIMS_PATH = os.path.join(CLAIM_DIR, "claims.jsonl")

STATUS = ("open", "confirmed", "refuted", "partial", "expired")


@dataclass
class Claim:
    cid: str
    created_at: str
    subject: str                 # 针对什么（代码/策略/信号）
    statement: str               # 我的判断（一句话，可证伪）
    metric: str                  # 用什么指标检验
    comparator: str              # ">" / "<" / "==" / "abs>"
    threshold: float
    horizon: str                 # 复核时间窗（如 "6个月" / "2027-06-30"）
    review_by: str               # 复核截止日
    status: str = "open"
    evidence: str = ""           # 登记时的证据快照（关键：不可事后修改）
    falsifier: str = ""          # 什么情况下算我错
    resolved_at: str = ""
    actual: Optional[float] = None
    user_feedback: str = ""      # 用户的纠正理由
    source: str = "lead"

    def to_dict(self) -> Dict:
        return asdict(self)


def _append(obj: Dict) -> None:
    with open(CLAIMS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_claims(path: str = CLAIMS_PATH) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def register(subject: str, statement: str, metric: str, threshold: float,
             review_by: str, comparator: str = ">", horizon: str = "",
             evidence: str = "", falsifier: str = "", source: str = "lead") -> Claim:
    """登记一条判断。一旦写入，判断内容不可改（只追加验证结果）。"""
    c = Claim(
        cid=uuid.uuid4().hex[:12],
        created_at=datetime.now().isoformat(timespec="seconds"),
        subject=subject, statement=statement, metric=metric, comparator=comparator,
        threshold=float(threshold), horizon=horizon or review_by,
        review_by=review_by, evidence=evidence, falsifier=falsifier, source=source,
    )
    _append(c.to_dict())
    return c


def _check(actual: float, comparator: str, threshold: float) -> bool:
    if actual != actual:  # NaN
        return False
    if comparator == ">":
        return actual > threshold
    if comparator == "<":
        return actual < threshold
    if comparator == "==":
        return abs(actual - threshold) < 1e-9
    if comparator == "abs>":
        return abs(actual) > abs(threshold)
    if comparator == "abs<":
        return abs(actual) < abs(threshold)
    raise ValueError(f"unknown comparator {comparator}")


def review(cid: str, actual: Optional[float] = None, verdict: Optional[str] = None,
           user_feedback: str = "", source: str = "user") -> Dict:
    """
    复核一条判断。
      - 传 actual：按登记的 comparator/threshold 自动判定
      - 传 verdict：直接采信（用户纠正优先，人工判定覆盖自动判定）
    两者都传时，以 verdict 为准，但 actual 仍记录。
    """
    if verdict is not None and verdict not in STATUS:
        raise ValueError(f"verdict must be one of {STATUS}")
    claims = load_claims()
    target = None
    for c in claims:
        if c["cid"] == cid:
            target = c
            break
    if target is None:
        raise KeyError(f"claim not found: {cid}")

    if verdict is None:
        if actual is None:
            raise ValueError("need actual or verdict")
        verdict = "confirmed" if _check(float(actual), target["comparator"],
                                        target["threshold"]) else "refuted"

    rec = {"cid": cid, "status": verdict, "actual": actual,
           "user_feedback": user_feedback, "resolved_at": datetime.now().isoformat(timespec="seconds"),
           "source": source, "op": "review"}
    _append(rec)
    target.update({k: v for k, v in rec.items() if k != "op"})
    return target


def _fold(claims: List[Dict]) -> List[Dict]:
    """把追加式记录折叠成当前状态。"""
    state: Dict[str, Dict] = {}
    for c in claims:
        cid = c.get("cid")
        if not cid:
            continue
        if c.get("op") == "review" or "status" in c and c.get("resolved_at"):
            state.setdefault(cid, {}).update(c)
        else:
            state.setdefault(cid, {}).update(c)
    return list(state.values())


def score(claims: Optional[List[Dict]] = None) -> Dict:
    """我自己的命中率：这是"你可不可以信我"的唯一硬指标。"""
    claims = _fold(claims if claims is not None else load_claims())
    done = [c for c in claims if c.get("status") in ("confirmed", "refuted", "partial")]
    n_conf = sum(1 for c in done if c["status"] == "confirmed")
    n_ref = sum(1 for c in done if c["status"] == "refuted")
    n_part = sum(1 for c in done if c["status"] == "partial")
    open_ = [c for c in claims if c.get("status") == "open"]
    today = date.today().isoformat()
    overdue = [c for c in open_ if c.get("review_by", "9999") < today]

    # 部分正确按 0.5 计
    denom = n_conf + n_ref + n_part
    hit = (n_conf + 0.5 * n_part) / denom if denom else None
    return {
        "total": len(claims), "resolved": denom, "open": len(open_),
        "confirmed": n_conf, "refuted": n_ref, "partial": n_part,
        "overdue": len(overdue),
        "hit_rate": hit,
        "verdict": _score_verdict(hit, denom),
        "overdue_cids": [c["cid"] for c in overdue][:20],
    }


def _score_verdict(hit: Optional[float], n: int) -> str:
    if hit is None:
        return "还没有已结算的判断，无法评估可信度"
    if n < 10:
        return f"样本仅 {n} 条，命中率 {hit:.0%} 不足以说明问题"
    if hit >= 0.75:
        return f"命中率 {hit:.0%}（n={n}）：我的判断目前可信"
    if hit >= 0.55:
        return f"命中率 {hit:.0%}（n={n}）：中等，需要打折看"
    return f"命中率 {hit:.0%}（n={n}）：我的判断不可信，请以数据为准"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="判断台账")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="登记一条判断")
    a.add_argument("--subject", required=True)
    a.add_argument("--statement", required=True)
    a.add_argument("--metric", required=True)
    a.add_argument("--threshold", type=float, required=True)
    a.add_argument("--comparator", default=">")
    a.add_argument("--review-by", required=True)
    a.add_argument("--evidence", default="")
    a.add_argument("--falsifier", default="")

    r = sub.add_parser("review", help="复核/纠正")
    r.add_argument("--cid", required=True)
    r.add_argument("--actual", type=float, default=None)
    r.add_argument("--verdict", default=None)
    r.add_argument("--reason", default="")

    sub.add_parser("list", help="列出全部判断")
    sub.add_parser("score", help="我的命中率")

    args = ap.parse_args(argv)
    if args.cmd == "add":
        c = register(args.subject, args.statement, args.metric, args.threshold,
                     args.review_by, args.comparator, evidence=args.evidence,
                     falsifier=args.falsifier)
        print(json.dumps(c.to_dict(), ensure_ascii=False, indent=1))
    elif args.cmd == "review":
        print(json.dumps(review(args.cid, args.actual, args.verdict, args.reason),
                         ensure_ascii=False, indent=1))
    elif args.cmd == "list":
        for c in _fold(load_claims()):
            print(f"[{c.get('status','?'):9s}] {c.get('cid','')} {c.get('statement','')[:60]}")
    else:
        print(json.dumps(score(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
