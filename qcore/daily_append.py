"""
收盘后自行补齐当日 K 线 —— 不依赖第三方数据包的发布节奏。

## 为什么需要

上游数据包（chenditc/investment_data）每天只发布一次，通常在北京时间凌晨。
实测：2026-11-09 19:47（周一，A股已收盘 4 小时），
本地数据最后一天还是 2026-11-06（周五）——
因为最新发布包（tag 2026-11-08）里只含周五收盘。

用户看到「估值日 2026-11-06」会认为系统卡住了。
实际上系统没错，是**它没有自己去拿当天数据**。

## 做法

1. 收盘后（15:05 以后），从东财拉每只票最近几天的日线（后复权 fqt=2）
2. 把「目标日相对前一日的比值」存到 overlay：
       data/qlib/overlay/20261109.json
       {"date": "2026-11-09", "bars": {"sh600519": {"oo":1.01,"oc":1.02,
                                                     "ol":0.99,"cc":1.005,
                                                     "volume":..., "amount":...}}}
3. `qlib_data.load_stock` 在读完 qlib 原始数据后，把 overlay 的比值
   乘到最后一根收盘价上，追加成新行

## 为什么存比值而不是绝对价

qlib_bin 里的 close 是**归一化价**（保留比值关系，不是真实成交价）。
直接追加东财的绝对价会和历史断档。存比值再乘回去，收益序列是连续的 ——
因子只需要连续收益，不需要绝对价位。
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLAY_DIR = os.path.join(ROOT, "data", "qlib", "overlay")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def overlay_path(day: str) -> str:
    return os.path.join(OVERLAY_DIR, f"{str(day).replace('-', '')}.json")


def load_overlay(day: str) -> Optional[Dict]:
    p = overlay_path(day)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_overlays() -> List[str]:
    if not os.path.isdir(OVERLAY_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(OVERLAY_DIR)):
        if fn.endswith(".json") and len(fn) == 13:
            out.append(f"{fn[:4]}-{fn[4:6]}-{fn[6:8]}")
    return out


def _atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _bare(code: str) -> str:
    """sh600519 / sz000001 -> 600519 / 000001（东财接口要纯 6 位）。"""
    c = str(code).strip().lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c[2:]
    return c


def _fetch_recent(code: str, days: int = 8) -> Optional[pd.DataFrame]:
    """
    拉最近几天的日线（后复权）。

    注意：qlib 的代码带 sh/sz 前缀，东财接口要纯 6 位。
    不转换的话 _em_secid 会拼出 "0.sh600519" 这种无效 secid，
    请求返回空 —— 表现为「no_bars_fetched」，但其实是代码格式问题。
    """
    try:
        from . import sources as SRC
        df = SRC.fetch_eastmoney(_bare(code), adjust=2)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("date")
    return df.tail(days)


def append_day(target_day: Optional[str] = None, codes: Optional[List[str]] = None,
               verbose: bool = True) -> Dict:
    """
    抓取 target_day（默认今天）的日线并写入 overlay。

    只写「相对前一日的比值」，供 qlib_data 追加。
    """
    from . import qlib_data as QD

    day = target_day or date.today().isoformat()
    cal = QD.load_calendar()
    if cal is None or len(cal) == 0:
        return {"ok": False, "reason": "no_calendar"}
    last_local = str(pd.Timestamp(cal[-1]).date())

    tgt = pd.Timestamp(day)
    if tgt <= pd.Timestamp(last_local):
        return {"ok": True, "action": "skip", "reason": "already_have",
                "last_local": last_local, "target": day}

    if codes is None:
        # 必须传 start：csi300.txt 是「历史所有成分」（20998 行），
        # 不传 start 会把全部历史成分都抓一遍（几小时）。
        # 传 last_local 只拿当前成分（约 300 只）。
        try:
            codes = QD.universe_codes("csi300", start=last_local)
        except Exception:
            codes = []
        if not codes:
            try:
                codes = QD.universe_codes("csi300", start=day)
            except Exception:
                codes = []

    if verbose:
        print(f"[daily] 目标 {day}（本地最后 {last_local}），"
              f"抓 {len(codes)} 只 ...", flush=True)

    bars: Dict[str, Dict] = {}
    ok_n = 0
    for i, c in enumerate(codes, 1):
        df = _fetch_recent(c)
        if df is None or len(df) < 2:
            continue
        # 找目标日那一行；没有就跳过（可能停牌/未收盘）
        rows = df[df["date"] <= tgt]
        if len(rows) < 2:
            continue
        cur = rows.iloc[-1]
        prev = rows.iloc[-2]
        if str(pd.Timestamp(cur["date"]).date()) != day:
            continue                      # 没有目标日的数据
        pc = float(prev["close"])
        if pc <= 0:
            continue
        try:
            bars[str(c).lower()] = {
                # 记下「这根比值是相对哪一天」—— 下游要用它校验连续性，
                # 否则断档日会被静默吞掉（实测 11-09 缺一天，+5% 直接消失）
                "prev": str(_pd.Timestamp(prev["date"]).date()),
                "oo": round(float(cur["open"]) / pc, 8),
                "oc": round(float(cur["close"]) / pc, 8),
                "ol": round(float(cur["low"]) / pc, 8),
                "oh": round(float(cur["high"]) / pc, 8),
                "volume": float(cur.get("volume") or 0),
                "amount": float(cur.get("amount") or 0),
            }
            ok_n += 1
        except Exception:
            continue
        if verbose and i % 100 == 0:
            print(f"  ... {i}/{len(codes)}（已拿到 {ok_n}）", flush=True)

    if not bars:
        return {"ok": False, "reason": "no_bars_fetched", "target": day,
                "last_local": last_local,
                "hint": "东财还没出当日数据，或全部停牌"}

    payload = {"date": day, "generated_at": _now(),
               "last_local": last_local, "n": len(bars), "bars": bars}
    _atomic(overlay_path(day), payload)

    # ---- 把日期追加到 qlib 日历 ----
    # 关键：overlay 只给个股加了 K 线，日历还是旧的那一天。
    # pipeline / live / 台账全用日历判断「最新交易日」，
    # 不追加的话它们仍然认为最后一天是旧的（实测 data_to 又变回 11-06）。
    try:
        from . import qlib_data as _QD
        import os as _os
        cal_path = _os.path.join(_QD.base_dir(), "calendars", "day.txt")
        if _os.path.isfile(cal_path):
            with open(cal_path, encoding="utf-8") as f:
                days = [ln.strip() for ln in f if ln.strip()]
            if days and days[-1] < day:
                with open(cal_path, "a", encoding="utf-8") as f:
                    f.write(day + "\n")
                if verbose:
                    print(f"[daily] 日历追加 {day}（原最后 {days[-1]}）", flush=True)
                # 清日历缓存，让同进程后续读取看到新日期
                try:
                    _QD._cache.pop("calendar", None)
                except Exception:
                    pass
    except Exception as _e:  # noqa: BLE001
        if verbose:
            print(f"[daily] 日历追加失败（不影响 overlay）: {type(_e).__name__}", flush=True)

    # 同步更新 status.json 的 data_to —— 否则界面上「数据截至」还停在旧日期，
    # 明明预测已经用上新数据了。
    try:
        from . import store as _store
        st = _store.read_json("status", default={}) or {}
        if str(st.get("data_to") or "") < day:
            st["data_to"] = day
            st["data_to_source"] = "daily_append"
            st["data_to_at"] = _now()
            _store.write_json("status", st)
    except Exception:
        pass
    if verbose:
        print(f"[daily] 写入 {len(bars)} 只的当日比值 -> {overlay_path(day)}", flush=True)
    return {"ok": True, "action": "written", "target": day,
            "last_local": last_local, "n": len(bars)}


if __name__ == "__main__":
    import sys
    d = None
    for a in sys.argv[1:]:
        if a.startswith("--day="):
            d = a.split("=", 1)[1]
    r = append_day(target_day=d, verbose=True)
    print(json.dumps({k: v for k, v in r.items() if k != "bars"},
                     ensure_ascii=False, indent=1))
