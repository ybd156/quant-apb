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
    """登记一条判断。一旦写入，判断内容不可改（只追加验证结果）。

    **falsifier 与 review_by 现在是强制项**：一条无法被证伪的判断不是判断，
    是一句愿望。历史记录不追溯（只追加），新登记缺这两项直接拒绝。
    """
    if not str(falsifier or "").strip():
        raise ValueError("falsifier 必填：写明「什么证据能推翻这条判断」")
    if not str(review_by or "").strip():
        raise ValueError("review_by 必填：写明复核日期（YYYY-MM-DD）")
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


# ---------------------------------------------------------------- 到期结算
# metric -> (取值函数, 口径说明)。**只映射能从产物单一字段读出、口径无歧义的指标。**
# 读不出或口径有歧义的，一律留给人工，不做猜测 —— 自动结算一个猜出来的数，
# 比不结算更糟。
def _perf() -> Dict:
    try:
        from . import store
        return store.read_json("performance", default={}) or {}
    except Exception:  # noqa: BLE001
        return {}


def _edge5(perf: Dict) -> Dict:
    edge = perf.get("edge") or {}
    return edge.get("5") or edge.get(5) or {}


def _ev_rank_ic_h5() -> Optional[float]:
    return _edge5(_perf()).get("ic", {}).get("mean")


def _ev_dsr_h5() -> Optional[float]:
    return _edge5(_perf()).get("long_short", {}).get("dsr")


def _ev_hit_rate_h5() -> Optional[float]:
    hr = ((_perf().get("metrics") or {}).get("hit_rate") or {})
    return (hr.get("5") or hr.get(5) or {}).get("hit_rate")


def _ev_effective_signals() -> Optional[float]:
    sl = ((_perf().get("risk") or {}).get("signal_level") or {})
    return (sl.get("corr_adjusted") or {}).get("effective_n_corr")


def _ev_return_scale_median() -> Optional[float]:
    """压缩因子中位（audit/basis_probe.py 产出）。<0.95 才说明存在系统性压缩。"""
    try:
        from . import store
        b = store.read_json("basis", default={}) or {}
    except Exception:  # noqa: BLE001
        return None
    return (b.get("compression") or {}).get("median")


EVALUATORS: Dict[str, tuple] = {
    "rank_ic_h5": (_ev_rank_ic_h5, "performance.edge[5].ic.mean"),
    "dsr_h5": (_ev_dsr_h5, "performance.edge[5].long_short.dsr（每期 Sharpe 口径）"),
    "hit_rate_h5": (_ev_hit_rate_h5, "performance.metrics.hit_rate[5].hit_rate"),
    "effective_signals": (_ev_effective_signals,
                          "performance.risk.signal_level.corr_adjusted.effective_n_corr"),
    "return_scale_median": (_ev_return_scale_median,
                            "data/state/basis.json compression.median（audit/basis_probe.py）"),
}


def settle(force: bool = False, dry_run: bool = False,
           today: Optional[str] = None) -> Dict:
    """把**到期**的判断按产物自动结算。

    - 有 evaluator 且能取到值 → review(cid, actual=…) 自动判定 confirmed/refuted
    - 无 evaluator 或取不到值 → 记入 needs_manual（附 falsifier），**不猜**
    - 未到期且未 force → 跳过

    返回值里 `needs_manual` 非空即代表"台账还没走完闭环"，调用方（CI / cron）
    应据此报警 —— 这正是 7 条判断长期 0 结算的根因：没人负责按日期结算。
    """
    today = today or date.today().isoformat()
    claims = _fold(load_claims())
    due, settled, manual, failed = [], [], [], []
    for c in claims:
        if c.get("status") not in (None, "", "open"):
            continue
        rb = str(c.get("review_by") or "9999-99-99")
        if not force and rb > today:
            continue
        cid, metric = c.get("cid"), str(c.get("metric") or "")
        due.append(cid)
        fn = EVALUATORS.get(metric)
        if fn is None:
            manual.append({"cid": cid, "metric": metric,
                           "reason": "无自动取值口径（需人工结算）",
                           "falsifier": c.get("falsifier", ""),
                           "review_by": rb})
            continue
        try:
            actual = fn[0]()
        except Exception as e:  # noqa: BLE001
            failed.append({"cid": cid, "metric": metric, "error": str(e)[:120]})
            continue
        if actual is None:
            manual.append({"cid": cid, "metric": metric,
                           "reason": f"产物缺字段（{fn[1]}）",
                           "falsifier": c.get("falsifier", ""), "review_by": rb})
            continue
        if dry_run:
            verdict = "confirmed" if _check(float(actual), c["comparator"],
                                            c["threshold"]) else "refuted"
            settled.append({"cid": cid, "metric": metric, "actual": float(actual),
                            "verdict": verdict, "dry_run": True})
            continue
        try:
            got = review(cid, actual=float(actual),
                         user_feedback=f"auto-settle({metric}={float(actual):.6g})",
                         source="auto")
            settled.append({"cid": cid, "metric": metric, "actual": float(actual),
                            "verdict": got.get("status")})
        except Exception as e:  # noqa: BLE001
            failed.append({"cid": cid, "metric": metric, "error": str(e)[:120]})
    return {"today": today, "due": len(due), "settled": settled,
            "needs_manual": manual, "failed": failed,
            "coverage": (f"{len(settled)}/{len(due)} 条到期判断已自动结算"
                         if due else "无到期判断")}


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

    st = sub.add_parser("settle", help="按产物自动结算到期判断")
    st.add_argument("--force", action="store_true", help="不看 review_by，全部尝试结算")
    st.add_argument("--dry-run", action="store_true", help="只报告，不写台账")

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
    elif args.cmd == "settle":
        r = settle(force=args.force, dry_run=args.dry_run)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        # 有到期却未结算的 → 非零退出，让 CI / cron 看得见
        return 1 if r["needs_manual"] or r["failed"] else 0
    else:
        print(json.dumps(score(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
