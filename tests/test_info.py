"""tests/test_info.py —— 信息快照层验收（可直接 `python -m tests.test_info` 运行）

覆盖：
  1. 真实快照：600519 至少 4 个源成功；kline/valuation/sector/breadth 至少中 3 个
  2. 每条 InfoItem 都有 as_of / source / reliability，且 reliability ∈ (0,1]
  3. news / announcement 的 pit_backfillable 必须为 False（禁止回测的硬标记）
  4. 语义：不存在的代码 → 返回 missing 而不是抛异常；非法 market 同理
  5. 落盘/读取往返一致（save → load → 字段对得上）
  6. backfill 模式：as_of 为过去某天时，只有 kline 可用，live-only 源进 missing
  7. 可信度不是常数：至少出现 3 个不同取值，且 news < announcement < kline
  8. daily_summary 扁平摘要可用

注意：控制台为 GBK，**print 一律 ASCII**，中文只进数据不进口志。
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import traceback

import pandas as pd

from qcore.info import (PIT_BACKFILLABLE, ROOT, SOURCE_RELIABILITY, InfoItem,
                        daily_summary, load_snapshots, save_snapshot, snapshot)

CODE = "600519"
_PASS = 0
_FAIL = 0
_FAILS = []


def _ok(name: str, detail: str = "") -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))


def _bad(name: str, detail: str = "") -> None:
    global _FAIL
    _FAIL += 1
    _FAILS.append(f"{name} :: {detail}")
    print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def check(name: str, fn) -> None:
    try:
        detail = fn()
        _ok(name, "" if detail is None else str(detail))
    except AssertionError as e:
        _bad(name, str(e) or "assertion failed")
    except Exception as e:                  # noqa: BLE001
        _bad(name, f"{type(e).__name__}: {e}")
        traceback.print_exc()


def main() -> int:
    print("=" * 78)
    print("test_info: information snapshot layer")
    print("=" * 78)

    # ---------------- 采一次真实快照（全程只采一次，避免重复打接口） ----------------
    print("\n[1] live snapshot of %s" % CODE)
    t0 = time.perf_counter()
    snap = snapshot(CODE)
    dt = time.perf_counter() - t0
    items = snap["items"]
    by_src = {}
    for it in items:
        by_src.setdefault(it.source, []).append(it)
    print("  elapsed=%.2fs items=%d sources=%s missing=%s"
          % (dt, len(items), sorted(by_src), [m["source"] for m in snap["missing"]]))

    def t_sources():
        assert isinstance(snap, dict), "snapshot must return dict"
        for k in ("items", "summary", "missing"):
            assert k in snap, f"missing key {k}"
        n_ok = len(by_src)
        assert n_ok >= 4, f"only {n_ok} sources ok: {sorted(by_src)}"
        core = {"kline", "valuation", "sector", "breadth"}
        hit = core & set(by_src)
        assert len(hit) >= 3, f"core sources only {sorted(hit)}"
        assert snap["summary"]["n_sources"] == n_ok
        return f"{n_ok} sources, core hit {len(hit)}/4 {sorted(hit)}"
    check("at least 4 sources, >=3 of kline/valuation/sector/breadth", t_sources)

    def t_fields():
        assert len(items) > 0, "no items"
        for it in items:
            assert isinstance(it, InfoItem), f"bad type {type(it)}"
            assert isinstance(it.as_of, str) and len(it.as_of) >= 10, f"bad as_of {it.as_of!r}"
            assert it.as_of[:4].isdigit(), f"as_of not ISO {it.as_of!r}"
            assert it.source in SOURCE_RELIABILITY, f"unknown source {it.source!r}"
            assert isinstance(it.reliability, float), f"reliability not float {it.reliability!r}"
            assert 0.0 < it.reliability <= 1.0, f"reliability out of range {it.reliability}"
            assert isinstance(it.payload, dict) and it.payload, f"empty payload {it.source}"
            assert isinstance(it.kind, str) and it.kind, "empty kind"
            assert isinstance(it.pit_backfillable, bool), "pit flag not bool"
        return f"{len(items)} items all carry as_of/source/reliability"
    check("every item has as_of/source/reliability in (0,1]", t_fields)

    def t_pit_news_ann():
        for it in items:
            if it.source in ("news", "announcement"):
                assert it.pit_backfillable is False, \
                    f"{it.source} must be pit_backfillable=False, got {it.pit_backfillable}"
        for s in ("news", "announcement"):
            assert PIT_BACKFILLABLE[s] is False, f"{s} flag in table must be False"
        for s in ("kline", "valuation", "fundamental", "sector", "breadth"):
            assert PIT_BACKFILLABLE[s] is True, f"{s} flag in table must be True"
        n_news = len(by_src.get("news", []))
        n_ann = len(by_src.get("announcement", []))
        return f"news={n_news} ann={n_ann} all pit_backfillable=False; core sources True"
    check("news/announcement pit_backfillable is False", t_pit_news_ann)

    def t_reliability_varies():
        rels = sorted({round(i.reliability, 4) for i in items})
        assert len(rels) >= 3, f"reliability looks constant: {rels}"
        news_rel = [i.reliability for i in items if i.source == "news"]
        ann_rel = [i.reliability for i in items if i.source == "announcement"]
        kline_rel = [i.reliability for i in items if i.source == "kline"]
        assert kline_rel and max(kline_rel) == 1.0, "kline should be 1.0 (exchange raw)"
        if ann_rel:
            assert max(ann_rel) <= 0.95 + 1e-9, "announcement base should be 0.95"
        if news_rel:
            assert max(news_rel) < 0.45 + 1e-9, \
                f"news should be discounted below 0.45 base, got {max(news_rel)}"
        return f"distinct reliabilities={len(rels)} min={min(rels)} max={max(rels)}"
    check("reliability is graded, not all 1.0", t_reliability_varies)

    def t_dedupe_news():
        news = by_src.get("news", [])
        if not news:
            return "no news item (source missing) - skipped"
        for it in news:
            p = it.payload
            assert "dup_count" in p, "news payload lacks dup_count"
            assert p["dup_count"] >= 1
            if p["dup_count"] > 1:
                assert p["duplicates"], "dup_count>1 but duplicates empty"
                assert it.reliability < SOURCE_RELIABILITY["news"], \
                    "duplicated news must be down-weighted below base"
        titles = [i.payload["title"] for i in news]
        assert len(set(titles)) == len(titles), "duplicate titles survived dedupe"
        return f"{len(news)} news kept, max dup_count={max(i.payload['dup_count'] for i in news)}"
    check("news dedupe + per-item down-weight", t_dedupe_news)

    # ---------------- 语义：坏代码不抛异常 ----------------
    print("\n[2] graceful degradation")

    def t_bad_code():
        t = time.perf_counter()
        bad = snapshot("999999")
        el = time.perf_counter() - t
        assert isinstance(bad, dict), "must return dict"
        assert bad["items"] == [], f"bad code should yield no items, got {len(bad['items'])}"
        assert len(bad["missing"]) >= 5, f"missing too short: {bad['missing']}"
        for m in bad["missing"]:
            assert "source" in m and "reason" in m, f"bad missing entry {m}"
        assert bad["summary"]["n_items"] == 0
        return f"no exception, {len(bad['missing'])} missing entries, {el:.2f}s"
    check("nonexistent code -> missing, no exception", t_bad_code)

    def t_bad_market():
        r = snapshot(CODE, market="US")
        assert isinstance(r, dict) and r["items"] == []
        assert any("market" in m["reason"] for m in r["missing"])
        return f"{len(r['missing'])} missing entries for US"
    check("unsupported market -> missing, no exception", t_bad_market)

    def t_weird_input():
        for bad in ("", None, "abc", 600519):
            r = snapshot(bad)
            assert isinstance(r, dict) and isinstance(r["missing"], list)
        r = snapshot(CODE, as_of="not-a-date")
        assert isinstance(r, dict), "unparseable as_of must not raise"
        return "empty/None/alpha/int/unparseable-as_of all survived"
    check("weird inputs never raise", t_weird_input)

    # ---------------- 落盘 / 读取往返 ----------------
    print("\n[3] save -> load round trip")

    # 临时目录必须落在工作区内：沙箱只允许写 workspace，系统 temp 会 PermissionError
    tmp = os.path.join(ROOT, "data", "_test_info_tmp")

    def t_roundtrip():
        path = save_snapshot(snap, root=tmp)
        assert os.path.exists(path), f"no file at {path}"
        assert path.endswith(".jsonl"), path
        assert os.sep + "CN" + os.sep + CODE + os.sep in path, f"bad layout {path}"
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        assert raw.count("\n") == 1, f"expected 1 jsonl line, got {raw.count(chr(10))}"
        assert "\\u" not in raw[:2000], "jsonl not written with ensure_ascii=False"
        df = load_snapshots(CODE, "CN", root=tmp)
        assert isinstance(df, pd.DataFrame) and not df.empty, "load returned empty"
        assert len(df) == len(items), f"rows {len(df)} != items {len(items)}"
        for col in ("code", "market", "date", "as_of", "source", "kind",
                    "reliability", "pit_backfillable", "note", "payload", "summary"):
            assert col in df.columns, f"missing column {col}"
        a = sorted((i.as_of, i.source, i.kind, round(i.reliability, 6),
                    bool(i.pit_backfillable)) for i in items)
        b = sorted((r.as_of, r.source, r.kind, round(float(r.reliability), 6),
                    bool(r.pit_backfillable)) for r in df.itertuples())
        assert a == b, "round trip mismatch between snapshot items and loaded frame"
        assert df["code"].nunique() == 1 and df["code"].iloc[0] == CODE
        assert df["market"].iloc[0] == "CN"
        assert (df["reliability"] > 0).all() and (df["reliability"] <= 1).all()
        return f"{path.split(os.sep)[-3:]} rows={len(df)} cols={len(df.columns)}"
    check("save -> load keeps as_of/source/kind/reliability/pit", t_roundtrip)

    def t_roundtrip_payload():
        df = load_snapshots(CODE, "CN", root=tmp)
        row = df[df["source"] == "kline"].iloc[0]
        orig = [i for i in items if i.source == "kline"][0]
        assert row["payload"]["date"] == orig.payload["date"]
        assert abs(row["payload"]["close"] - orig.payload["close"]) < 1e-9
        assert isinstance(row["summary"], dict) and row["summary"]["code"] == CODE
        return f"payload keys match, summary.n_items={row['summary']['n_items']}"
    check("payload/summary survive json round trip", t_roundtrip_payload)

    def t_append_and_filter():
        n_before = len(load_snapshots(CODE, "CN", root=tmp))
        save_snapshot(snap, root=tmp)
        n_after = len(load_snapshots(CODE, "CN", root=tmp))
        assert n_after == 2 * n_before, f"append broken: {n_before} -> {n_after}"
        d = str(snap.get("date") or "")[:10]
        sub = load_snapshots(CODE, "CN", start=d, end=d, root=tmp)
        assert not sub.empty, "date filter returned nothing"
        far = load_snapshots(CODE, "CN", start="1990-01-01", end="1990-12-31", root=tmp)
        assert far.empty, "future/past filter broken"
        empty = load_snapshots("000000", "CN", root=tmp)
        assert isinstance(empty, pd.DataFrame) and empty.empty
        return f"append ok ({n_before}->{n_after}), date filter ok, unknown code -> empty"
    check("append-only + date filtering", t_append_and_filter)

    def t_save_real():
        p = save_snapshot(snap)                       # 默认根 data/info，真实归档一条
        assert os.path.exists(p), f"no file at {p}"
        rel = os.path.relpath(p, ROOT)
        assert rel.startswith(os.path.join("data", "info", "CN", CODE)), rel
        return rel
    check("save to default data/info archive", t_save_real)

    def t_daily_summary():
        s = daily_summary(CODE, "CN")
        assert isinstance(s, dict) and s, "daily_summary empty"
        assert s.get("_from") in ("disk", "live"), f"bad _from {s.get('_from')}"
        assert s.get("code") == CODE
        assert "n_items" in s and s["n_items"] >= 1
        return f"_from={s['_from']} n_items={s['n_items']} sources={s.get('n_sources')}"
    check("daily_summary returns flat dict", t_daily_summary)
    print("\n[4] point-in-time backfill mode")

    def t_backfill():
        past = "2026-06-30"
        r = snapshot(CODE, as_of=past)
        srcs = {i.source for i in r["items"]}
        assert r["mode"] == "backfill", f"mode={r['mode']} (expected backfill)"
        assert "kline" in srcs, f"kline must be backfilled, got {srcs}"
        assert srcs == {"kline"}, f"live-only sources leaked into backfill: {srcs}"
        assert r["date"] is not None and r["date"] <= past, \
            f"kline not truncated: {r['date']} > {past}"
        miss_srcs = {m["source"] for m in r["missing"]}
        for s in ("valuation", "fundamental", "sector", "breadth", "announcement", "news"):
            assert s in miss_srcs, f"{s} should be reported missing in backfill mode"
        return f"date={r['date']} only kline, {len(miss_srcs)} sources refused (no future leakage)"
    check("as_of in the past -> kline only, live sources refused", t_backfill)

    def t_no_future_items():
        now = pd.Timestamp.now()
        for it in items:
            ts = pd.Timestamp(it.as_of.replace("+08:00", ""))
            assert ts <= now + pd.Timedelta(days=1), f"future-dated item {it.source} {it.as_of}"
        return "no item is dated in the future"
    check("no future-dated items in live snapshot", t_no_future_items)

    # ---------------- 汇总 ----------------
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 78)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed   (snapshot elapsed {dt:.2f}s)")
    if _FAILS:
        for f in _FAILS:
            print("  FAILED:", f)
        print("=" * 78)
        return 1
    print("ALL GREEN")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
