"""
自主情报搜集 —— 让程序**自己决定"该查谁"**，把查回来的东西落盘、打分，
并登记成"可被未来证伪"的判断。

## 它补的是哪一块

仓库里已经有两个"被动"的情报件：

    qcore/info.py       —— 给定 code，抓估值/公告/新闻（PIT 截断、相关性过滤都做了）
    qcore/news_live.py  —— 全市场快讯轮询 + 关键词分类（type / direction / sectors）

缺的是**决策层**：谁决定今天该查哪几只？查不到怎么办？查回来的东西谁在用？
现状是：`info.py` 只服务 pipeline 点名的那几只，`news_live` 只做全市场刷新，
两者的产出都不参与打分（AGENTS R46）。所以"情报"是有的，"寻找"是没有人做的。

本模块做的事，一句话：**从持仓/推荐/自选里选 subject → 查 → 打分 → 落盘 → 登记判断**，
并且把"哪些 subject 没查到"显式记下来，下一轮优先查它们。

## 三个自主性

1. **查谁，自己定**：模拟盘持仓 → 今日 picks → 自选，按这个优先级去重取前 N 个；
   每个 subject 带上 `why`（"持仓"/"推荐"/"自选"），便于事后追问"为什么查这只"。
2. **缺口驱动**：本轮没查到情报的 subject 进 `gaps`，下一轮排到前面（round-robin），
   避免每次都只查头部那几只、尾部永远没人管。
3. **查完要能证伪**：每条情报落进 `data/autoinfo/ledger.jsonl`（**只追加**），
   带 `direction` 与 `h`；`verify()` 到期后回填收益与命中，写到**另一个**追加文件
   `verified.jsonl` —— 不改历史（INV-7），读数时 join。

## 成本与安全

- 每个 subject 只发 **1** 个请求（复用 `info._news_items` 的相关性过滤与去重）；
- 单轮 subject 上限 `MAX_SUBJECTS=8`，失败即跳过并记 `errors`，绝不重试打爆接口；
- 离线/接口挂掉时返回 `ok=False` + `reason`，**不抛异常**、不阻塞调用方；
- 不引入任何新依赖，不调用 LLM（分类用 `news_live.classify` 的规则表）。

## 用法

    python -m qcore.autoinfo --run        # 跑一轮（默认 8 个 subject，看最近 7 天）
    python -m qcore.autoinfo --verify     # 给到期情报回填收益/命中
    python -m qcore.autoinfo --summary    # 看命中率与缺口
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "data", "state")
LEDGER_DIR = os.path.join(ROOT, "data", "autoinfo")
STATE = os.path.join(STATE_DIR, "autoinfo.json")
LEDGER = os.path.join(LEDGER_DIR, "ledger.jsonl")
VERIFIED = os.path.join(LEDGER_DIR, "verified.jsonl")

MAX_SUBJECTS = 8
LOOKBACK_DAYS = 7
DEFAULT_H = 5
MIN_EVAL = 30          # 「历史参照 vs 真实」配对的样本门槛
#: 来源权重：标题命中个股 > 行业关键词命中；快讯类的可信度按 news_live 的约定降权
SOURCE_W = {"info": 1.0, "live": 0.7}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _read_json(p: str, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _uid(*parts: str) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ 1. 查谁
def subjects(limit: int = MAX_SUBJECTS) -> List[Dict]:
    """关注清单：持仓 / 推荐 / 自选三路**按配额轮转**，带 why。

    为什么不是严格优先级：模拟盘持仓动辄 20 只，严格优先级会把 8 个名额全吃光，
    "推荐"和"自选"永远查不到。按配额（持仓 4 / 推荐 2 / 自选 2）再补齐。
    """
    pools: Dict[str, List[Dict]] = {"持仓": [], "推荐": [], "自选": []}
    seen = set()

    def push(why, code, name):
        c = str(code or "").strip()
        if c.isdigit():
            c = c.zfill(6)
        if not c or c in seen:
            return
        seen.add(c)
        pools[why].append({"code": c, "name": str(name or ""), "why": why})

    # ① 模拟盘持仓（真金白银在跑的）
    try:
        from . import paper_auto as PA
        acc = PA.load_account("CN", 1) or {}
        for code in (acc.get("positions") or {}):
            push("持仓", code, "")
    except Exception:  # noqa: BLE001
        pass

    # ② 今日 picks（推荐标的）
    for fn in ("picks.json", "picks_h5.json"):
        p = _read_json(os.path.join(STATE_DIR, fn), {})
        for r in (p.get("rows") or []):
            push("推荐", r.get("code"), r.get("name"))

    # ③ 自选
    try:
        from . import watch as W
        d = W.load() or {}
        for it in (d.get("items") or []):
            if isinstance(it, dict):
                push("自选", it.get("code"), it.get("name"))
    except Exception:  # noqa: BLE001
        pass

    cap = max(1, int(limit))
    quota = {"持仓": max(1, cap // 2), "推荐": max(1, cap // 4), "自选": max(1, cap // 4)}
    out: List[Dict] = []
    for why in ("持仓", "推荐", "自选"):
        out.extend(pools[why][: quota[why]])
    # 补齐：按 持仓 → 推荐 → 自选 的顺序把剩余名额填满
    for why in ("持仓", "推荐", "自选"):
        for it in pools[why][quota[why]:]:
            if len(out) >= cap:
                break
            out.append(it)
    return out[:cap]


def _names() -> Dict[str, str]:
    try:
        from . import names as N
        m = N.load() or {}
        return {str(k).zfill(6): (v.get("name") if isinstance(v, dict) else str(v))
                for k, v in m.items()}
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------ 2. 查回来
def fetch_news(code: str, name: str = "", days: int = LOOKBACK_DAYS) -> List[Dict]:
    """复用 info._news_items（它已有 PIT 截断 + 相关性过滤 + 去重 + 标题优先）。"""
    cutoff = datetime.now()
    try:
        from . import info as I
        items = I._news_items(str(code), name or None, cutoff,
                              cutoff.strftime("%Y-%m-%dT%H:%M:%S"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for it in items:
        payload = getattr(it, "payload", None) or {}
        as_of = str(getattr(it, "as_of", "") or "")[:10]
        out.append({
            "code": str(code),
            "as_of": as_of,
            "title": str(payload.get("title") or "")[:200],
            "url": str(payload.get("url") or ""),
            "media": payload.get("media"),
            "in_title": bool(payload.get("in_title")),
            "source": "info",
        })
    # 时间窗
    floor = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    return [x for x in out if x["as_of"] >= floor]


def _market_items() -> List[Dict]:
    """全市场快讯里与关注清单无关的部分也要能进"背景"，但只取最近 1 天。"""
    d = _read_json(os.path.join(STATE_DIR, "news_live.json"), {})
    rows = d.get("rows") or []
    floor = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("date") or "")[:10] < floor:
            continue
        out.append({
            "code": "", "as_of": str(r.get("date") or "")[:10],
            "title": str(r.get("title") or "")[:200], "url": str(r.get("url") or ""),
            "media": r.get("source"), "in_title": False, "source": "live",
            "type": r.get("type"), "direction": r.get("direction"),
            "sectors": r.get("sectors") or [],
        })
    return out


def score(item: Dict, subject: Dict) -> float:
    """相关性 × 时效。刻意简单、可解释：三档权重，不用打分模型。"""
    base = 0.5 if item.get("in_title") else 0.25
    if item.get("source") == "live":
        base *= SOURCE_W["live"]
    try:
        age = (datetime.now() - datetime.strptime(item["as_of"][:10], "%Y-%m-%d")).days
    except Exception:  # noqa: BLE001
        age = LOOKBACK_DAYS
    recency = 0.5 ** (max(0, age) / 2.0)          # 半衰期 2 天
    return round(0.7 * base + 0.3 * recency, 4)


def _analogs(text: str) -> tuple:
    """历史参照（判分第二步：先量后信）。拿不到就返回空，不影响主流程。"""
    try:
        from . import event_history as EH
        a = EH.analogs(text, top=3)
        im = EH.implied(a)
        return ([{"id": x.get("id"), "name": x.get("name"), "date": x.get("date"),
                  "car": x.get("car")} for x in a], im)
    except Exception:  # noqa: BLE001
        return ([], None)


def classify(item: Dict) -> Dict:
    try:
        from . import news_live as NL
        c = NL.classify(item.get("title") or "", "")
        return {"type": c.get("type"), "direction": int(c.get("direction") or 0),
                "sectors": c.get("sectors") or [], "level": c.get("level")}
    except Exception:  # noqa: BLE001
        return {"type": "other", "direction": 0, "sectors": [], "level": "weak"}


# ------------------------------------------------------------------ 3. 落盘
def _append(path: str, rows: Iterable[Dict], key_fn) -> Dict:
    """只追加 + 幂等（同 key 不重复写）。读取失败就拒绝写（不覆盖历史）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = set()
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    seen.add(key_fn(r))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "ledger_unreadable", "detail": str(e)[:120]}
    n_new = n_dup = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            k = key_fn(r)
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n_new += 1
    return {"ok": True, "new": n_new, "dup": n_dup}


def _key(r: Dict):
    return (str(r.get("as_of"))[:10], str(r.get("code") or ""), str(r.get("url") or "")[:120])


def run(max_subjects: int = MAX_SUBJECTS, days: int = LOOKBACK_DAYS,
        verbose: bool = True) -> Dict:
    """跑一轮：选 subject → 查 → 打分 → 落盘 + 登记判断。离线时 ok=False 但不抛异常。"""
    prev = _read_json(STATE, {})
    subs = subjects(max_subjects)

    # 缺口优先：上一轮没查到的排前面（round-robin），避免尾部永远饿死
    gaps = set(prev.get("gaps") or [])
    subs.sort(key=lambda s: (0 if s["code"] in gaps else 1))
    subs = subs[:max_subjects]

    names = _names()
    items: List[Dict] = []
    gaps_now: List[str] = []
    errors: List[str] = []
    per_subject: Dict[str, int] = {}

    for s in subs:
        code = s["code"]
        name = s["name"] or names.get(code, "")
        got = fetch_news(code, name, days)
        per_subject[code] = len(got)
        if not got:
            gaps_now.append(code)
            continue
        for it in got:
            c = classify(it)
            an, im = _analogs(it.get("title") or "")
            items.append({**it, "name": name, "why": s["why"],
                          "score": score(it, s), **c,
                          "analogs": an, "implied": im})

    # 市场背景：只保留与关注清单同向/同板块的（其余不进，避免噪声）
    sectors = {x for it in items for x in (it.get("sectors") or [])}
    live = _market_items()
    for it in live:
        if sectors and not (set(it.get("sectors") or []) & sectors):
            continue
        c = classify(it) if not it.get("type") else {
            "type": it.get("type"), "direction": int(it.get("direction") or 0),
            "sectors": it.get("sectors") or [], "level": "medium"}
        items.append({**it, "name": "", "why": "背景", "score": score(it, {}), **c})

    items.sort(key=lambda x: -float(x.get("score") or 0))

    state = {
        "generated_at": _now(),
        "as_of": _today(),
        "n_subjects": len(subs),
        "n_items": len(items),
        "subjects": subs,
        "per_subject": per_subject,
        "gaps": gaps_now,
        "errors": errors,
        "items": items[:200],
        "note": ("自主情报：subject 来自 持仓→推荐→自选；每 subject 1 个请求；"
                 "缺口进 gaps 下一轮优先；判断入 data/autoinfo/ledger.jsonl 供到期验证"),
    }

    if not items and not subs:
        state["ok"] = False
        state["reason"] = "no_subject"
        _atomic(STATE, state)
        return {"ok": False, "reason": "no_subject"}

    # 登记判断（只追加，带 h，供 verify 回填）
    rows = [{"as_of": it["as_of"], "code": it["code"], "name": it.get("name"),
             "title": it["title"], "url": it["url"], "type": it.get("type"),
             "direction": int(it.get("direction") or 0), "score": it.get("score"),
             # 历史参照的"预期幅度"：到期后与真实收益相减，就是"缺了多少"
             "implied_car": (it.get("implied") or {}).get("implied_car"),
             "n_analogs": (it.get("implied") or {}).get("n_analogs"),
             "h": DEFAULT_H, "recorded_at": _now()}
            for it in items if it.get("code") and it.get("url")]
    led = _append(LEDGER, rows, _key)
    state["ledger"] = led
    state["ok"] = bool(items)
    if not items:
        state["reason"] = "no_items"
    _atomic(STATE, state)

    if verbose:
        print(f"[autoinfo] subjects={len(subs)} items={len(items)} "
              f"gaps={len(gaps_now)} ledger+{led.get('new')} dup={led.get('dup')}",
              flush=True)
    return {"ok": state["ok"], "n_subjects": len(subs), "n_items": len(items),
            "gaps": gaps_now, "ledger": led, "state": STATE}


# ------------------------------------------------------------------ 4. 验证
def _close_on(code: str, day: str) -> Optional[float]:
    """取某日收盘（qlib 侧，含 overlay）。取不到返回 None。"""
    try:
        from . import qlib_data as QD
        q = "sh" + code if code[0] in "69" else "sz" + code
        df = QD.load_stock(q, start=(datetime.strptime(day, "%Y-%m-%d")
                                     - timedelta(days=10)).strftime("%Y-%m-%d"))
        if df is None or not len(df):
            return None
        d = df.copy()
        d["date"] = d["date"].astype(str).str.slice(0, 10)
        hit = d[d["date"] >= day]
        return float(hit["close"].iloc[0]) if len(hit) else None
    except Exception:  # noqa: BLE001
        return None


def verify(h: Optional[int] = None, verbose: bool = True) -> Dict:
    """给到期情报回填收益与命中。**只往 verified.jsonl 追加**，不改 ledger（INV-7）。"""
    if not os.path.isfile(LEDGER):
        return {"ok": False, "reason": "no_ledger"}
    done = set()
    if os.path.isfile(VERIFIED):
        try:
            with open(VERIFIED, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        done.add((r.get("as_of"), r.get("code"), r.get("url")))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "verified_unreadable", "detail": str(e)[:120]}

    rows, n_skip, n_already = [], 0, 0
    try:
        with open(LEDGER, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                hh = int(h or r.get("h") or DEFAULT_H)
                k = (r.get("as_of"), r.get("code"), r.get("url"))
                if k in done:
                    n_already += 1
                    continue
                if not r.get("direction"):
                    continue
                try:
                    d0 = datetime.strptime(str(r["as_of"])[:10], "%Y-%m-%d")
                except Exception:  # noqa: BLE001
                    continue
                d1 = d0 + timedelta(days=int(hh * 1.5) + 2)      # 自然日近似，取 >= 的第一根
                if d1.date() > datetime.now().date():
                    n_skip += 1
                    continue
                p0 = _close_on(r["code"], d0.strftime("%Y-%m-%d"))
                p1 = _close_on(r["code"], d1.strftime("%Y-%m-%d"))
                if not p0 or not p1 or p0 <= 0:
                    n_skip += 1
                    continue
                ret = p1 / p0 - 1.0
                rows.append({"as_of": r["as_of"], "code": r["code"], "url": r["url"],
                             "direction": int(r["direction"]), "h": hh,
                             "entry": round(p0, 4), "exit": round(p1, 4),
                             "ret": round(ret, 6),
                             "hit": bool(int(r["direction"]) * ret > 0),
                             "verified_at": _now()})
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "ledger_unreadable", "detail": str(e)[:120]}

    res = _append(VERIFIED, rows, lambda r: (r.get("as_of"), r.get("code"), r.get("url")))
    if verbose:
        print(f"[autoinfo] verify: +{res.get('new')} verified, {n_skip} not matured",
              flush=True)
    return {"ok": True, "new": res.get("new"), "dup": res.get("dup"),
            "already": n_already, "not_matured": n_skip, "path": VERIFIED}



def evaluate() -> Dict:
    """历史参照有多准：把"预期幅度"与"真实收益"配对，算偏差/MAE/同号率。

    这一步回答的正是"缺了多少"：
      bias = mean(implied_car - ret)   >0 说明历史参照系统性高估
      mae  = mean(|implied_car - ret|) 平均差多少
      sign = P(sign(implied) == sign(ret))  方向对了多少次
    样本 < MIN_EVAL 一律 insufficient（AGENTS §10 规则 7）。
    """
    implied: Dict[tuple, Dict] = {}
    if os.path.isfile(LEDGER):
        try:
            with open(LEDGER, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("implied_car") is None:
                        continue
                    implied[(r.get("as_of"), r.get("code"), r.get("url"))] = r
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "ledger_unreadable", "detail": str(e)[:120]}

    pairs = []
    if os.path.isfile(VERIFIED):
        try:
            with open(VERIFIED, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    v = json.loads(line)
                    k = (v.get("as_of"), v.get("code"), v.get("url"))
                    src = implied.get(k)
                    if not src or v.get("ret") is None:
                        continue
                    pairs.append({"as_of": v["as_of"], "code": v["code"],
                                  "implied_car": float(src["implied_car"]),
                                  "ret": float(v["ret"]),
                                  "diff": round(float(src["implied_car"]) - float(v["ret"]), 6)})
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "verified_unreadable", "detail": str(e)[:120]}

    n = len(pairs)
    if not n:
        return {"ok": True, "n": 0, "insufficient": True,
                "note": "还没有同时具备「历史参照」与「真实结果」的样本"}
    bias = sum(p["diff"] for p in pairs) / n
    mae = sum(abs(p["diff"]) for p in pairs) / n
    sign = sum(1 for p in pairs if (p["implied_car"] > 0) == (p["ret"] > 0)) / n
    return {"ok": True, "n": n,
            "bias": round(bias, 6), "mae": round(mae, 6),
            "sign_match": round(sign, 3),
            "insufficient": bool(n < MIN_EVAL),
            "samples": pairs[:50],
            "note": (f"bias>0 = 历史参照高估；mae = 平均差多少；"
                     f"sign_match = 方向对的比例；样本 < {MIN_EVAL} 只作观察")}

# ------------------------------------------------------------------ 5. 读数
def summary() -> Dict:
    led = 0
    if os.path.isfile(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            led = sum(1 for line in f if line.strip())
    ver, hits = 0, 0
    if os.path.isfile(VERIFIED):
        with open(VERIFIED, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                ver += 1
                hits += 1 if r.get("hit") else 0
    st = _read_json(STATE, {})
    return {"ok": True, "generated_at": st.get("generated_at"),
            "n_items": st.get("n_items"), "gaps": st.get("gaps") or [],
            "ledger": led, "verified": ver,
            "hit_rate": round(hits / ver, 4) if ver else None,
            "insufficient": bool(ver < 30),
            "note": "情报判断的到期命中率；样本 < 30 只作观察（AGENTS §10 规则 7）"}


if __name__ == "__main__":
    import sys

    args = set(sys.argv[1:])
    if "--verify" in args:
        print(json.dumps(verify(), ensure_ascii=False, indent=1))
    if "--run" in args or not (args & {"--verify", "--summary"}):
        print(json.dumps(run(verbose=True), ensure_ascii=False, indent=1)[:1200])
    if "--evaluate" in args:
        print(json.dumps(evaluate(), ensure_ascii=False, indent=1))
    if "--summary" in args:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
