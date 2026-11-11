"""
告警通知 —— 把"这台测量仪是不是还活着"变成可推送的信号。

## 为什么需要它

AGENTS.md 第 0 节说得很清楚：本项目**不是**印钞机，是测量仪。
测量仪最危险的失效方式不是"算错"，而是**悄悄停止工作**：

  · 模型 9 天没重训 → 分数还是老市场的，但 UI 上看不出来
  · 数据源连续失败 → 缺数据被当成"没有信号"，静默空仓
  · picks.json 3 天没更新 → 页面还在展示上周的名单
  · 自选命中率掉到 30% → 策略已经失效，但没人盯
  · 实盘滑点 40bps → 回测的 19bps 假设是假的，净值曲线是幻觉

这五件事的共同点：**不告警就不会被发现**。所以告警模块的职责不是"报告好消息"，
而是把「该停手了」这件事推到人眼前。

## 设计要点

1. **只读、不修数据**：本模块不重训模型、不修数据源、不改 picks。
   它是纯粹的观测者 —— 观测者一旦会改状态，就不再是观测者了。
2. **永不抛异常**：任何一条规则炸了，其他规则照常评估（缺文件/坏 JSON 都算"规则不满足"）。
   告警系统自己挂掉是最糟的失败模式。
3. **去重窗口 1 小时**：同一个 key 1 小时内只发一次，否则每 3 秒轮询会把人淹没。
4. **通道可插拔**：file（零依赖、可事后审计） / webhook（钉钉/飞书/Telegram）。

## 落盘

    data/state/alerts.jsonl       告警流水（只追加，审计用）
    data/state/alert_state.json   {key: 上次发送时间}，用于去重

## 隔离测试

目录可用 `APB_STATE_DIR` / `APB_MODEL_DIR` 覆盖（`__main__` 自检就是这么做的），
滑点规则走 `slippage` 模块的 `APB_SLIPPAGE_DIR`。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 复用 slippage 的落盘原语：同一套"原子写 + 容错读 + 时间解析"语义，
# 两处各写一份迟早会漂移（例如一处补了 fsync、另一处忘了）。
from .slippage import _append_jsonl, _atomic_write, _parse_ts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_STATE = os.path.join(ROOT, "data", "state")
DEF_MODEL = os.path.join(ROOT, "data", "model")

# 阈值集中在这里 —— 改阈值应该改常量，不要改规则体（否则没人找得到）
TH_MODEL_AGE_DAYS = 7.0        # 模型几天没重训就认为过期
TH_SOURCE_FAIL = 3             # 数据源连续失败几次算"挂了"
TH_PICKS_AGE_DAYS = 3.0        # picks.json 几天没更新算陈旧
TH_HIT_RATE = 0.40             # 自选命中率下限
TH_HIT_MIN_N = 10              # 命中率样本下限（少于这个数不做判断）
TH_SLIP_BPS = 30.0             # 平均滑点上限（bps，单边）
TH_DRAWDOWN = -0.15            # 回撤阈值
DEDUP_HOURS = 1.0              # 同一 key 的去重窗口
STATE_KEEP_DAYS = 7            # alert_state 保留天数（防无限增长）

LEVEL_ORDER = {"crit": 0, "warn": 1, "info": 2}

# 最近一次 check_rules() 里各规则的异常（诊断用，不影响返回值契约）
LAST_ERRORS: Dict[str, str] = {}


# ------------------------------------------------------------------ 基础设施
def _now() -> str:
    """全项目统一的时间口径：本地时区、秒级精度。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _state_dir() -> str:
    return os.environ.get("APB_STATE_DIR") or DEF_STATE


def _model_dir() -> str:
    return os.environ.get("APB_MODEL_DIR") or DEF_MODEL


def _p(name: str) -> str:
    return os.path.join(_state_dir(), name)


def _read_json(path: str) -> Optional[Dict]:
    """容错读 JSON。缺文件/坏 JSON/类型不对 → None（调用方当作"数据缺失"）。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _age_days(ts) -> Optional[float]:
    """把一个时间戳（iso 或纯日期）换算成"距今多少天"。解析失败返回 None。"""
    t = _parse_ts(ts)
    if t is None:
        return None
    return (datetime.now().astimezone() - t).total_seconds() / 86400.0


def _alert(level: str, key: str, msg: str, **extra) -> Dict:
    d = {"level": level, "key": key, "msg": msg, "at": _now()}
    d.update(extra)
    return d


# ------------------------------------------------------------------ 规则实现
def _model_age_days() -> Optional[float]:
    """
    模型年龄。优先用 live_model.model_age_days()（项目既有口径）；
    隔离测试模式（设了 APB_MODEL_DIR）或调用失败时，直接读 meta.json。

    经济含义：GBDT 学的是「当前市场结构下特征→收益」的映射。
    结构漂移后老模型的分数就是**过期的判断**，而它看起来和新鲜判断一模一样 ——
    这是最隐蔽的失效。
    """
    d = os.environ.get("APB_MODEL_DIR")
    if not d:
        try:
            from . import live_model as LM
            return LM.model_age_days()
        except Exception:
            pass
        d = DEF_MODEL
    meta = _read_json(os.path.join(d, "meta.json")) or {}
    return _age_days(meta.get("saved_at"))


def _health_sources() -> Optional[Dict[str, Dict]]:
    """
    读数据源健康度，兼容两种历史格式：
      {"as_of":..., "sources": {name: {...}}}   ← health.check_all() 写的
      {name: {...}}                              ← 早期版本/手工维护的扁平格式
    """
    raw = _read_json(_p("source_health.json"))
    if not raw:
        return None
    src = raw.get("sources")
    if isinstance(src, dict):
        return {str(k): v for k, v in src.items() if isinstance(v, dict)}
    # 扁平格式：把顶层里长得像源记录的值挑出来
    flat = {str(k): v for k, v in raw.items()
            if isinstance(v, dict) and ("fail" in v or "last_error" in v)}
    return flat or None


def _watch_items() -> List[Dict]:
    """
    自选列表。默认走 watch.load()（组件自带容错）；
    隔离测试模式直接读文件，避免读到真实路径。
    """
    if not os.environ.get("APB_STATE_DIR"):
        try:
            from . import watch as W
            return [it for it in ((W.load() or {}).get("items") or [])
                    if isinstance(it, dict)]
        except Exception:
            pass
    d = _read_json(_p("watchlist.json")) or {}
    return [it for it in (d.get("items") or []) if isinstance(it, dict)]


def rule_model_stale() -> List[Dict]:
    """
    规则 1 · model_stale —— 模型过期。

    经济含义：模型学的是"什么特征在什么市场状态下有效"。市场换了（波动率抬升、
    风格切换），老权重会把噪声当信号。超过 7 天没重训，分数就该被当成"待验证"而不是"结论"。
    """
    age = _model_age_days()
    if age is None:
        return []            # 从未训练 / meta 缺失 → 不告警（首装不该天天喊）
    if age > TH_MODEL_AGE_DAYS:
        return [_alert("warn", "model_stale",
                       f"模型已 {age:.1f} 天未重训（阈值 {TH_MODEL_AGE_DAYS:.0f} 天）",
                       age_days=round(float(age), 2))]
    return []


def rule_source_down() -> List[Dict]:
    """
    规则 2 · source_down —— 数据源挂了。

    经济含义：缺数据在本项目里会被**静默**当成"没有信号"（信号预热期输出 0）。
    也就是说数据源死了，系统不会报错，只会安静地不再说话。
    连续失败 ≥3 次基本可以判定不是瞬时抖动。
    """
    src = _health_sources()
    if not src:
        return []
    bad = []
    for name, v in src.items():
        err = v.get("last_error") or ""
        try:
            fail = int(v.get("fail") or 0)
        except Exception:
            fail = 0
        if str(err).strip() and fail >= TH_SOURCE_FAIL:
            bad.append((name, fail, str(err)[:60]))
    if not bad:
        return []
    bad.sort(key=lambda x: -x[1])
    detail = "；".join(f"{n} 连续失败 {f} 次（{e}）" for n, f, e in bad)
    return [_alert("crit", "source_down",
                   f"数据源异常：{detail}",
                   sources=[{"name": n, "fail": f, "last_error": e}
                            for n, f, e in bad])]


def rule_picks_stale() -> List[Dict]:
    """
    规则 3 · picks_stale —— 选股名单陈旧。

    经济含义：picks.json 是 UI 和下游唯一看到的"今日名单"。
    它不更新时页面照样好看，只是展示的是上周的名单 ——
    **看起来在运行，实际已经停了**。这是最容易被忽略的失效。
    """
    d = _read_json(_p("picks.json"))
    if not d:
        return []
    age = _age_days(d.get("as_of"))
    if age is None:
        return [_alert("warn", "picks_stale", "picks.json 存在但 as_of 无法解析",
                       as_of=d.get("as_of"))]
    if age > TH_PICKS_AGE_DAYS:
        n = len(d.get("rows") or [])
        return [_alert("warn", "picks_stale",
                       f"picks.json 已 {age:.1f} 天未更新（阈值 "
                       f"{TH_PICKS_AGE_DAYS:.0f} 天），当前 {n} 只名单是旧的",
                       age_days=round(float(age), 2), as_of=d.get("as_of"))]
    return []


def rule_watch_hit_drop() -> List[Dict]:
    """
    规则 4 · watch_hit_drop —— 自选命中率下滑。

    经济含义：命中率是**实盘口径**的准确率（预测方向 vs 真实收益），
    比回测 IC 诚实得多。样本 ≥10 才有统计意义；低于 40% 意味着
    "按这个方向做，还不如抛硬币"，也就是策略已失效。
    """
    items = _watch_items()
    if not items:
        return []
    hits = [it.get("hit") for it in items if isinstance(it.get("hit"), bool)]
    n = len(hits)
    if n < TH_HIT_MIN_N:
        return []
    rate = float(np.mean([1.0 if h else 0.0 for h in hits]))
    if rate < TH_HIT_RATE:
        return [_alert("warn", "watch_hit_drop",
                       f"自选命中率 {rate:.1%}（阈值 {TH_HIT_RATE:.0%}，样本 {n}）",
                       hit_rate=round(rate, 4), n=n)]
    return []


def rule_slippage_high() -> List[Dict]:
    """
    规则 5 · slippage_high —— 实盘滑点过高。

    经济含义：这是回测可信度的**直接反证**。回测假设单边约 9.6bps
    （往返 19.28bps 的一半），实测均值超过 30bps 说明真实成交环境差得多，
    回测那条漂亮的净值曲线在实盘上要打掉一大截。
    """
    try:
        from . import slippage as S
    except Exception:
        return []
    r = S.analyze(days=30)
    if not r.get("ok"):
        return []
    mean = r.get("mean_slippage_bps")
    if mean is None:
        return []
    if float(mean) > TH_SLIP_BPS:
        return [_alert("warn", "slippage_high",
                       f"近 30 天平均滑点 {float(mean):.1f}bps（阈值 "
                       f"{TH_SLIP_BPS:.0f}bps，样本 {r.get('n_pairs')} 笔），"
                       f"回测成本假设偏乐观",
                       mean_bps=round(float(mean), 2), n_pairs=r.get("n_pairs"))]
    return []


def rule_panel_lag() -> List[Dict]:
    """面板落后于行情（评审 R14）：数据到了 11-10、模型只算到 11-06。

    只看两个日期，不做任何计算：`data_sync.local_date()` 是行情最后一天，
    `status.as_of` 是最后一次算到的行情日。
    """
    try:
        from . import data_sync as DS
        d_to = DS.local_date()
    except Exception:  # noqa: BLE001
        return []
    st = _read_json(_p("status")) or {}
    as_of = st.get("as_of")
    if not d_to or not as_of or str(as_of)[:10] >= str(d_to)[:10]:
        return []
    try:
        from . import calendar_cn as CAL
        d0 = _dt.date.fromisoformat(str(as_of)[:10]); d1 = _dt.date.fromisoformat(str(d_to)[:10])
        n, cur = 0, d0
        while cur < d1 and n < 60:
            cur = CAL.next_trading_day(cur); n += 1
    except Exception:  # noqa: BLE001
        n = 0
    if n <= 0:
        return []
    return [_alert("warn", "panel_lag",
                   f"面板落后行情 {n} 个交易日（算到 {as_of} / 行情到 {d_to}）",
                   lag_trading_days=n, as_of=str(as_of), data_to=str(d_to))]


def rule_drawdown() -> List[Dict]:
    """
    规则 6（附加）· drawdown —— 模拟盘/实盘回撤超阈值。

    经济含义：AGENTS.md 的结论是"当前信号集没有可用 alpha"。
    回撤超 -15% 不是"运气差"，而是"该停手了"的量化信号 ——
    继续加仓的期望收益为负。
    """
    d = _read_json(_p("paper.json"))
    if not d:
        return []
    nav = d.get("live_nav") or d.get("sim_nav") or []
    vals = []
    for v in nav if isinstance(nav, list) else []:
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            vals.append(f)
    if len(vals) < 2:
        return []
    arr = np.asarray(vals, dtype=float)
    peak = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, arr / peak - 1.0, 0.0)
    mdd = float(np.min(dd))
    if mdd < TH_DRAWDOWN:
        return [_alert("warn", "drawdown",
                       f"模拟盘回撤 {mdd:.1%}（阈值 {TH_DRAWDOWN:.0%}）",
                       max_drawdown=round(mdd, 4), n=len(vals))]
    return []


# (key, 函数) —— 顺序即输出顺序（同一 level 内按此排）
RULES: List[Tuple[str, Callable[[], List[Dict]]]] = [
    ("panel_lag", rule_panel_lag),
    ("source_down", rule_source_down),      # crit 排最前：数据源挂了，其他结论都不可信
    ("model_stale", rule_model_stale),
    ("picks_stale", rule_picks_stale),
    ("watch_hit_drop", rule_watch_hit_drop),
    ("slippage_high", rule_slippage_high),
    ("drawdown", rule_drawdown),
]


# ------------------------------------------------------------------ 检查
def check_rules() -> List[Dict]:
    """
    检查预置规则，返回触发的告警列表（按 level: crit → warn → info 排序）。

    **永不抛异常**：任何一条规则内部出错只记进 LAST_ERRORS 并跳过。
    告警系统自己崩掉是最糟的失败模式 —— 你会以为"没消息就是好消息"。
    """
    LAST_ERRORS.clear()
    out: List[Dict] = []
    for key, fn in RULES:
        try:
            got = fn() or []
        except Exception as e:  # noqa: BLE001
            LAST_ERRORS[key] = f"{type(e).__name__}: {str(e)[:80]}"
            continue
        for a in got:
            if isinstance(a, dict) and a.get("level") and a.get("key"):
                out.append(a)
    out.sort(key=lambda a: (LEVEL_ORDER.get(a["level"], 9), a["key"]))
    return out


# ------------------------------------------------------------------ 发送
def _webhook_payload(text: str, alerts: List[Dict]) -> Dict:
    """
    一份 JSON 同时喂三家，靠"多余字段被忽略"的容错：
      钉钉   {"msgtype":"text","text":{"content":...}}
      飞书   {"msg_type":"text","content":{"text":...}}
      Telegram {"chat_id":..., "text":...}
    """
    p: Dict = {
        "msgtype": "text", "text": {"content": text},
        "msg_type": "text", "content": {"text": text},
        "alerts": alerts,
    }
    chat = os.environ.get("APB_WEBHOOK_CHAT_ID")
    if chat:
        p["chat_id"] = chat
    return p


def dispatch(alerts: List[Dict], channel: str = "file") -> Dict:
    """
    发送告警。

    channel:
      "file"    追加到 data/state/alerts.jsonl（默认，零依赖、可事后审计）
      "webhook" POST 到环境变量 APB_WEBHOOK_URL（urllib，兼容钉钉/飞书/Telegram）

    返回 {"ok":True,"sent":n}；失败不抛异常，返回 ok=False + reason。
    """
    alerts = [a for a in (alerts or []) if isinstance(a, dict)]
    if not alerts:
        return {"ok": True, "sent": 0, "channel": channel}

    ch = str(channel or "file").strip().lower()
    if ch == "file":
        try:
            path = _p("alerts.jsonl")
            for a in alerts:
                _append_jsonl(path, a)
            return {"ok": True, "sent": len(alerts), "channel": "file", "path": path}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "write_failed",
                    "detail": f"{type(e).__name__}: {str(e)[:80]}", "sent": 0}

    if ch == "webhook":
        url = os.environ.get("APB_WEBHOOK_URL", "").strip()
        if not url:
            return {"ok": False, "reason": "no_webhook_url", "sent": 0,
                    "hint": "设置环境变量 APB_WEBHOOK_URL"}
        text = "【量化预知告警】\n" + "\n".join(
            f"[{a.get('level','info').upper()}] {a.get('key')}: {a.get('msg')}"
            for a in alerts)
        body = json.dumps(_webhook_payload(text, alerts),
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8",
                     "User-Agent": "apb-alert/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                code = getattr(r, "status", 200)
            ok = 200 <= int(code) < 300
            return {"ok": ok, "sent": len(alerts) if ok else 0, "channel": "webhook",
                    "status": int(code)}
        except urllib.error.HTTPError as e:
            return {"ok": False, "reason": "http_error", "status": int(e.code),
                    "sent": 0}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "network_error",
                    "detail": f"{type(e).__name__}: {str(e)[:80]}", "sent": 0}

    return {"ok": False, "reason": "bad_channel", "sent": 0,
            "hint": "channel 只支持 file / webhook"}


# ------------------------------------------------------------------ 去重
def _load_state() -> Dict[str, str]:
    d = _read_json(_p("alert_state.json")) or {}
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[str(k)] = v
        elif isinstance(v, dict) and isinstance(v.get("at"), str):
            out[str(k)] = v["at"]
    return out


def _save_state(st: Dict[str, str]) -> None:
    """写去重状态，顺便清掉过期条目（否则一年后会积累几万条 key）。"""
    now = datetime.now().astimezone()
    keep = {}
    for k, v in st.items():
        t = _parse_ts(v)
        if t is None:
            continue
        if (now - t).total_seconds() < STATE_KEEP_DAYS * 86400:
            keep[k] = v
    _atomic_write(_p("alert_state.json"),
                  json.dumps(keep, ensure_ascii=False, indent=1, sort_keys=True))


def _dedupe(alerts: List[Dict], st: Dict[str, str],
            window_h: float = DEDUP_HOURS) -> Tuple[List[Dict], List[Dict]]:
    """把 1 小时内已经发过的 key 过滤掉。返回 (要发的, 被压制的)。"""
    now = datetime.now().astimezone()
    fresh, suppressed = [], []
    for a in alerts:
        k = str(a.get("key"))
        last = _parse_ts(st.get(k))
        if last is not None and (now - last).total_seconds() < window_h * 3600:
            suppressed.append(a)
        else:
            fresh.append(a)
    return fresh, suppressed


def run_once(channel: str = "file") -> Dict:
    """
    检查 + 发送 + 去重（同一 key 1 小时内不重复发）。

    这是给 cron / 常驻循环调用的唯一入口：幂等、可重复调用、不抛异常。
    """
    alerts = check_rules()
    st = _load_state()
    fresh, suppressed = _dedupe(alerts, st)
    res = dispatch(fresh, channel=channel)
    if res.get("ok") and fresh:
        now = _now()
        for a in fresh:
            st[str(a["key"])] = now
        try:
            _save_state(st)
        except Exception as e:  # noqa: BLE001
            LAST_ERRORS["_state_write"] = f"{type(e).__name__}: {str(e)[:80]}"
    return {"ok": bool(res.get("ok")),
            "triggered": len(alerts),
            "sent": int(res.get("sent") or 0),
            "suppressed": len(suppressed),
            "channel": channel,
            "dispatch": res,
            "alerts": alerts,
            "rule_errors": dict(LAST_ERRORS)}


# ------------------------------------------------------------------ 自检
if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 74)
    print("告警模块自检 —— 5 条规则在「数据缺失」「数据异常」下都不崩")
    print("=" * 74)

    tmp = tempfile.mkdtemp(prefix="apb_alert_")
    old = {k: os.environ.get(k) for k in
           ("APB_STATE_DIR", "APB_MODEL_DIR", "APB_SLIPPAGE_DIR", "APB_WEBHOOK_URL")}
    empty = os.path.join(tmp, "empty")
    broken = os.path.join(tmp, "broken")
    slip = os.path.join(tmp, "slip")
    for d in (empty, broken, slip):
        os.makedirs(d, exist_ok=True)

    def set_env(state: str, model: str, slipdir: str) -> None:
        os.environ["APB_STATE_DIR"] = state
        os.environ["APB_MODEL_DIR"] = model
        os.environ["APB_SLIPPAGE_DIR"] = slipdir

    try:
        # ============================================================ 场景 A
        print("\n【场景 A】数据缺失（目录全空，无任何 json）")
        set_env(empty, os.path.join(tmp, "no_model"), slip)
        a = check_rules()
        print(f"  check_rules() -> {len(a)} 条（期望 0，且不抛异常）")
        print(f"  rule_errors   -> {LAST_ERRORS or '{}'}")
        r = run_once("file")
        print(f"  run_once(file)-> ok={r['ok']} triggered={r['triggered']} "
              f"sent={r['sent']} suppressed={r['suppressed']}")
        ok_a = (len(a) == 0 and r["ok"] and r["sent"] == 0 and not LAST_ERRORS)

        # ============================================================ 场景 B
        print("\n【场景 B】数据异常（时间戳乱写 / 类型错乱 / 滑点爆表 / 源全挂）")
        # 模型 meta：saved_at 是 20 天前 → model_stale
        os.makedirs(os.path.join(tmp, "model_old"), exist_ok=True)
        old_ts = (datetime.now().astimezone() - timedelta(days=20)).isoformat(timespec="seconds")
        with open(os.path.join(tmp, "model_old", "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"saved_at": old_ts}, f)
        # 源健康：一个挂死、一个正常、一个字段类型乱写（fail 是字符串）
        with open(os.path.join(broken, "source_health.json"), "w", encoding="utf-8") as f:
            json.dump({"as_of": "2026-11-08T10:00:00+08:00",
                       "sources": {"eastmoney.kline": {"ok": 0, "fail": 5,
                                                       "last_error": "timeout"},
                                   "tencent.tick": {"ok": 9, "fail": 0, "last_error": ""},
                                   "weird.src": {"ok": None, "fail": "abc",
                                                 "last_error": None}}}, f)
        # picks：as_of 是 10 天前的纯日期 → picks_stale
        p10 = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        with open(os.path.join(broken, "picks.json"), "w", encoding="utf-8") as f:
            json.dump({"as_of": p10, "rows": [{"code": "600519"}]}, f)
        # 自选：12 条，命中 3 条 = 25% < 40% → watch_hit_drop（混入 None/字符串脏值）
        items = [{"code": f"60000{i}", "hit": (i < 3)} for i in range(12)]
        items += [{"code": "999999", "hit": None}, {"code": "888888", "hit": "maybe"},
                  {"code": "777777"}]
        with open(os.path.join(broken, "watchlist.json"), "w", encoding="utf-8") as f:
            json.dump({"items": items, "updated_at": _now()}, f)
        # 模拟盘：净值从 1.0 跌到 0.80 → 回撤 -20% < -15%
        nav = [1.0, 1.05, 0.98, 0.90, 0.85, 0.80]
        with open(os.path.join(broken, "paper.json"), "w", encoding="utf-8") as f:
            json.dump({"dates": [f"2026-10-{d:02d}" for d in range(1, 7)],
                       "sim_nav": nav, "live_nav": [1.0, "bad", None] + nav[3:]}, f)
        # 滑点账本：信号日 D-1 收盘 100 → 成交日 D 的 100.4 = +40bps > 30bps → slippage_high
        # （信号日必须早于成交日，否则因果判据不配对，见 slippage._event_time）
        os.makedirs(slip, exist_ok=True)
        sig_day = (datetime.now() - timedelta(days=11)).strftime("%Y-%m-%d")
        with open(os.path.join(slip, "signals.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"recorded_at": _now(), "as_of": sig_day,
                                "rows": [{"code": "600519", "ref_price": 100.0,
                                          "score": 0.9}]}, ensure_ascii=False) + "\n")
        with open(os.path.join(slip, "fills.jsonl"), "w", encoding="utf-8") as f:
            for hh in (10, 11, 14):
                f.write(json.dumps({"recorded_at": _now(), "code": "600519",
                                    "price": 100.4,
                                    "ts": f"{p10}T{hh:02d}:30:00+08:00",
                                    "source": "manual", "side": "buy"},
                                   ensure_ascii=False) + "\n")

        set_env(broken, os.path.join(tmp, "model_old"), slip)
        b = check_rules()
        print(f"  check_rules() -> {len(b)} 条")
        for x in b:
            print(f"    [{x['level']:<4}] {x['key']:<15} {x['msg']}")
        got_keys = {x["key"] for x in b}
        want = {"model_stale", "source_down", "picks_stale",
                "watch_hit_drop", "slippage_high", "drawdown"}
        ok_b = (want <= got_keys) and not LAST_ERRORS
        print(f"  期望覆盖 5 条预置规则 + drawdown -> 实得 {sorted(got_keys)}")
        print(f"  rule_errors -> {LAST_ERRORS or '{}'}")

        # ============================================================ 场景 C
        print("\n【场景 C】发送 + 去重（同一 key 1 小时内不重复发）")
        r1 = run_once("file")
        r2 = run_once("file")
        print(f"  第 1 次 run_once: triggered={r1['triggered']} sent={r1['sent']} "
              f"suppressed={r1['suppressed']}")
        print(f"  第 2 次 run_once: triggered={r2['triggered']} sent={r2['sent']} "
              f"suppressed={r2['suppressed']}（期望 sent=0）")
        n_lines = 0
        ap = os.path.join(broken, "alerts.jsonl")
        if os.path.isfile(ap):
            with open(ap, encoding="utf-8") as f:
                n_lines = sum(1 for ln in f if ln.strip())
        print(f"  alerts.jsonl 行数 = {n_lines}（期望 {r1['sent']}，即第二次没重复写）")
        st = _load_state()
        print(f"  alert_state.json 记录 {len(st)} 个 key：{sorted(st)}")
        ok_c = (r1["sent"] == r1["triggered"] and r2["sent"] == 0
                and r2["suppressed"] == r2["triggered"] and n_lines == r1["sent"])

        # ============================================================ 场景 D
        print("\n【场景 D】通道异常（webhook 无 URL / 非法 channel）")
        d1 = dispatch(b, channel="webhook")
        d2 = dispatch(b, channel="smoke-signal")
        d3 = dispatch([], channel="file")
        print(f"  webhook 无 URL -> {d1}")
        print(f"  非法 channel   -> {d2}")
        print(f"  空告警列表     -> {d3}")
        ok_d = (d1.get("reason") == "no_webhook_url" and d2.get("reason") == "bad_channel"
                and d3.get("ok") is True and d3.get("sent") == 0)

        print("\n" + "=" * 74)
        for name, ok in (("A 数据缺失", ok_a), ("B 数据异常", ok_b),
                         ("C 去重", ok_c), ("D 通道异常", ok_d)):
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"自检结果：{'全部 PASS' if all((ok_a, ok_b, ok_c, ok_d)) else '存在 FAIL'}")
        print("=" * 74)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)
