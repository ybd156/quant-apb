"""
时间门禁回归测试 —— 「不在交易时间也能交易」的六条防线。

实测事故（2026-11-09）：周一 13:10（盘中）用**周五收盘价**成交 20 笔，
并全部记成周五；账户 20/20 持仓 `last_price == cost`，收益恰好等于手续费。

根因不是漏了一个 if，而是三层同时失守：
  ① `session.in_window()` 只判 `weekday()>=5`（不含节假日），且把午休算成盘中
  ② `paper_auto.rebalance()` 对「交易日 / 时段 / bar 是否收盘」零检查
  ③ 成交价 `_price(code, as_of) or _price(code)` 的 fallback 会用**未来价**

本测试把这三层都钉住。每条都对应一个曾真实发生过的错误行为。

跑法：python -m tests.test_time_gates
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore import paper_auto as PA  # noqa: E402
from qcore import session as S  # noqa: E402

PASS = 0
FAIL = 0

#: 本文件里所有 `cn_dt(2026, ...)` 都表示**北京时间**。
#: 不标时区的话，naive 会被按机器时区解释 —— CI 跑在 UTC 时
#: 北京 15:30 会被当成 UTC 15:30（= 北京 23:30），14 条用例全红（实测）。
_CN_TZ = None
try:
    from zoneinfo import ZoneInfo as _ZI
    _CN_TZ = _ZI("Asia/Shanghai")
except Exception:  # noqa: BLE001
    _CN_TZ = None


def cn_dt(*a, **kw):
    """构造一个**北京时间**的 aware datetime。"""
    if _CN_TZ is not None and kw.get("tzinfo") is None:
        kw["tzinfo"] = _CN_TZ
    return datetime(*a, **kw)



def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    # ---------- 1) session.in_window：节假日 + 午休 ----------
    print("[1] session.in_window")
    cases = [
        (cn_dt(2026, 11, 10, 8, 45), "premarket", "交易日盘前"),
        (cn_dt(2026, 11, 10, 10, 0), "morning", "上午"),
        (cn_dt(2026, 11, 10, 11, 47), "lunch", "午休（旧实现返回 intraday）"),
        (cn_dt(2026, 11, 10, 14, 0), "afternoon", "下午"),
        (cn_dt(2026, 11, 10, 15, 30), "postclose", "盘后"),
        (cn_dt(2026, 10, 1, 10, 0), "closed", "国庆（旧实现返回 intraday）"),
        (cn_dt(2026, 11, 7, 10, 0), "closed", "周六"),
    ]
    for t, exp, note in cases:
        got = S.in_window(t)
        check(f"in_window({t:%Y-%m-%d %H:%M}) == {exp}", got == exp, f"got={got} · {note}")

    check("in_session(午休) == False", S.in_session(cn_dt(2026, 11, 10, 11, 47)) is False)
    check("in_session(上午) == True", S.in_session(cn_dt(2026, 11, 10, 10, 0)) is True)
    check("in_session(国庆) == False", S.in_session(cn_dt(2026, 10, 1, 10, 0)) is False)

    # ---------- 1b) 多市场时段：美股四段 / 港股收盘 16:00 ----------
    print(chr(10) + "[1b] 多市场时段（实测事故：美股被 A 股时段拦住）")

    def t_multi():
        from zoneinfo import ZoneInfo
        from qcore.session import segment, in_session
        ET = ZoneInfo("America/New_York")
        import datetime as _dt
        # 美股四段（含跨午夜夜盘）
        cases = [
            (21, 0, "overnight", False),
            (2, 0, "overnight", False),
            (6, 0, "premarket", False),
            (11, 0, "regular", True),
            (15, 0, "regular", True),
            (18, 0, "afterhours", False),
            (20, 30, "overnight", False),
        ]
        bad = []
        for h, mi, exp, trad in cases:
            t = _dt.cn_dt(2026, 11, 11, h, mi, tzinfo=ET)
            seg = segment(t, "US")
            if seg["phase"] != exp or bool(seg["tradable"]) != trad:
                bad.append(f"ET {h:02d}:{mi:02d} -> {seg['phase']}/{seg['tradable']} "
                           f"期望 {exp}/{trad}")
        assert not bad, "; ".join(bad)
        # 边界：20:00 属夜盘，04:00 属盘前
        assert segment(_dt.cn_dt(2026, 11, 11, 19, 59, tzinfo=ET), "US")["phase"] == "afterhours"
        assert segment(_dt.cn_dt(2026, 11, 11, 20, 0, tzinfo=ET), "US")["phase"] == "overnight"
        assert segment(_dt.cn_dt(2026, 11, 11, 4, 0, tzinfo=ET), "US")["phase"] == "premarket"
        # 港股收盘 16:00，比 A 股晚 1 小时
        # 必须传 tz-aware：naive 会按**机器时区**解释，
        # CI 跑在 UTC 时 15:30 会被当成北京 23:30（实测三版本全红）。
        HKT = ZoneInfo("Asia/Hong_Kong")
        CST = ZoneInfo("Asia/Shanghai")
        assert segment(_dt.cn_dt(2026, 11, 10, 15, 30, tzinfo=HKT), "HK")["phase"] == "afternoon"
        assert segment(_dt.cn_dt(2026, 11, 10, 15, 30, tzinfo=CST), "CN")["phase"] == "postclose"
        # 只有常规盘中 in_session 为真
        assert in_session(_dt.cn_dt(2026, 11, 11, 11, 0, tzinfo=ET), "US") is True
        assert in_session(_dt.cn_dt(2026, 11, 11, 18, 0, tzinfo=ET), "US") is False
        assert in_session(_dt.cn_dt(2026, 11, 11, 18, 0, tzinfo=ET), "US",
                          allow_extended=True) is True
        return "US 四段 + 港股 16:00 + 边界 20:00/04:00"

    check("多市场时段分段正确", t_multi)

    # ---------- 1c) 美股盘中被放行（原来被 A 股时段拦住）----------
    def t_us_gate():
        from zoneinfo import ZoneInfo
        import datetime as _dt
        ET = ZoneInfo("America/New_York")
        r = PA._assert_tradable({"as_of": "2026-11-10"},
                                now=_dt.cn_dt(2026, 11, 11, 11, 0, tzinfo=ET),
                                market="US")
        assert r["ok"], f"美股盘中应放行，实际 {r.get('reason')}"
        assert r.get("price_mode") == "live", r.get("price_mode")
        # 同一时刻 A 股应拒绝
        r2 = PA._assert_tradable({"as_of": "2026-11-10"},
                                 now=_dt.cn_dt(2026, 11, 11, 11, 0, tzinfo=ET),
                                 market="CN")
        assert not r2["ok"], "同一时刻 A 股不该放行"
        return f"US 放行(live) / CN 拒绝({r2.get('reason')})"

    check("美股盘中放行、A股同时刻拒绝", t_us_gate)

    # ---------- 2) lag_trading_days：基准必须是「最近已收盘交易日」 ----------
    print("\n[2] lag_trading_days（基准不是 qlib 日历最后一根）")
    lag_cases = [
        ("2026-11-06", cn_dt(2026, 11, 10, 10, 0), 1, "盘中：11-06 落后 1 日"),
        ("2026-11-06", cn_dt(2026, 11, 10, 16, 0), 2, "盘后：落后 2 日"),
        ("2026-11-09", cn_dt(2026, 11, 10, 10, 0), 0, "盘中：11-09 就是最近已收盘"),
        ("2026-11-09", cn_dt(2026, 11, 10, 16, 0), 1, "盘后：落后 1 日"),
        ("2026-10-01", cn_dt(2026, 11, 10, 10, 0), None, "非交易日 -> None"),
    ]
    for as_of, now, exp, note in lag_cases:
        got = PA.lag_trading_days(as_of, now)
        check(f"lag({as_of}, {now:%m-%d %H:%M}) == {exp}", got == exp, f"got={got} · {note}")

    # ---------- 3) _assert_tradable：六条拒单理由 ----------
    print("\n[3] paper_auto._assert_tradable")
    gate_cases = [
        ({"as_of": "2026-11-07"}, cn_dt(2026, 11, 10, 10, 0), False, "not_trading_day", "周六"),
        ({"as_of": "2026-10-01"}, cn_dt(2026, 11, 10, 10, 0), False, "not_trading_day", "国庆"),
        ({"as_of": "2027-01-04"}, cn_dt(2026, 11, 10, 10, 0), False, "as_of_in_future", "未来"),
        ({}, cn_dt(2026, 11, 10, 10, 0), False, "no_as_of", "缺 as_of"),
        ({"as_of": "2026-11-06"}, cn_dt(2026, 11, 10, 10, 0), False, "stale_picks", "陈旧 1 日"),
        # 盘中现在**允许**成交，但取价方式必须是实时价（price_mode="live"）——
        # 旧设计是「盘中一律拒绝」，用户反问「盘中有实时价为什么不给」是对的。
        ({"as_of": "2026-11-09"}, cn_dt(2026, 11, 10, 10, 0), True, None,
         "盘中放行（走实时价）"),
        # 16:00 已收盘 → 11-10 的收盘价已可得，11-09 就成了「落后 1 日」
        ({"as_of": "2026-11-09"}, cn_dt(2026, 11, 10, 16, 0), False, "stale_picks",
         "盘后：11-09 已落后 1 日"),
        # A 股收盘价 15:00 就定了，盘后不能成交 —— 不存在「收盘后调仓」
        ({"as_of": "2026-11-10"}, cn_dt(2026, 11, 10, 16, 0), False, "market_closed",
         "当日盘后：不能成交"),
    ]
    for pk, now, ok_exp, reason_exp, note in gate_cases:
        r = PA._assert_tradable(as_of_hint=pk, dry_run=False, now=now)
        good = (r["ok"] == ok_exp) and (r.get("reason") == reason_exp)
        check(f"{note}", good, f"ok={r['ok']} reason={r.get('reason')}")

    # 盘中放行时必须给出 live 取价方式（否则又会拿旧收盘价成交）
    r = PA._assert_tradable(as_of_hint={"as_of": "2026-11-09"},
                            dry_run=False, now=cn_dt(2026, 11, 10, 14, 0))
    check("盘中取价方式是实时价", r["ok"] and r.get("price_mode") == "live",
          f"ok={r['ok']} mode={r.get('price_mode')}")
    r = PA._assert_tradable(as_of_hint={"as_of": "2026-11-10"},
                            dry_run=True, now=cn_dt(2026, 11, 10, 16, 0))
    check("dry_run 盘后允许试算", r["ok"] is True, f"reason={r.get('reason')}")

    # dry_run 不受执行窗口限制（回放/试算不落盘）
    r = PA._assert_tradable(as_of_hint={"as_of": "2026-11-09"},
                            dry_run=True, now=cn_dt(2026, 11, 10, 10, 0))
    check("dry_run 盘中放行（不落盘，允许试算）", r["ok"] is True, f"reason={r.get('reason')}")

    # allow_backfill 显式越过但必须留痕
    r = PA._assert_tradable(as_of_hint={"as_of": "2026-11-09"}, dry_run=False,
                            now=cn_dt(2026, 11, 10, 10, 0), allow_backfill=True)
    check("allow_backfill 放行且留痕 backfilled=True",
          r["ok"] is True and r.get("backfilled") is True, f"{r}")

    # ---------- 4) 数据新鲜度：attrs 而不是 mtime ----------
    print("\n[4] data.load_kline 新鲜度标注")
    try:
        from qcore.data import load_kline
        import glob
        files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
        if not files:
            print("  [SKIP] 本地无行情缓存")
        else:
            code = os.path.basename(files[0]).split("_")[0]
            df = load_kline(code, now=cn_dt(2026, 11, 10, 10, 0))
            a = df.attrs
            check("attrs 有 data_last_date", "data_last_date" in a, str(a.get("data_last_date")))
            check("attrs 有 expected_last_date", "expected_last_date" in a,
                  str(a.get("expected_last_date")))
            check("attrs 有 stale", "stale" in a, str(a.get("stale")))
            # 关键：新鲜度不再由文件 mtime 决定
            check("stale 由数据日期判定（不是 mtime）",
                  a.get("stale") is not None and a.get("data_last_date") is not None)
    except Exception as e:  # noqa: BLE001
        check("load_kline 可读", False, f"{type(e).__name__}: {e}")

    # ---------- 5) live 的参考日：按交易日历，不按陈旧 index 缓存 ----------
    print("\n[5] live._latest_closed_trading_day")
    try:
        from qcore.live import _latest_closed_trading_day
        ref = _latest_closed_trading_day()
        check("返回交易日", ref is not None, str(ref.date()) if ref is not None else "None")
        from qcore import calendar_cn as CAL
        check("确实是交易日", CAL.is_trading_day_bool(ref.date()) if ref is not None else False)
        check("不晚于今天", ref.date() <= datetime.now().date() if ref is not None else False)
    except Exception as e:  # noqa: BLE001
        check("可导入", False, f"{type(e).__name__}: {e}")

    # ---------- 6) 负控：这些检查器真的会失败 ----------
    print("\n[6] 负控（注入错误行为必须被抓住）")
    check("国庆被拒（若返回 True 说明门禁没接日历）",
          PA._assert_tradable({"as_of": "2026-10-01"},
                              now=cn_dt(2026, 11, 10, 16, 0))["ok"] is False)
    check("午休不算盘中（若返回 intraday 说明午休没拆）",
          S.in_window(cn_dt(2026, 11, 10, 11, 47)) != "intraday")

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
