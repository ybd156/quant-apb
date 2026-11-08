"""Cold-start measurement of auto mode + pool (ASCII only)."""
import qcore.us_data as ud

print("== auto mode, cold cache ==")
tot = 0
for t in ("AAPL", "MSFT", "NVDA", "TSLA", "AMD", "BABA"):
    d = ud.load_us_kline(t)
    rep = [r["date"] for r in d.attrs.get("repaired", [])]
    ver = [r["date"] for r in d.attrs.get("verified_extreme", [])]
    n = d.attrs.get("verify_requests", 0)
    tot += n
    print(f"{t:5s} bars={len(d):5d} repaired={rep} verified={ver} extra_req={n}")
for c in ("00700", "09988"):
    d = ud.load_hk_kline(c)
    rep = [r["date"] for r in d.attrs.get("repaired", [])]
    ver = [r["date"] for r in d.attrs.get("verified_extreme", [])]
    n = d.attrs.get("verify_requests", 0)
    tot += n
    print(f"{c} bars={len(d):5d} repaired={rep} verified={ver} extra_req={n}")
print("total extra requests:", tot)

print("\n== pool ==")
p = ud.get_pool_us(80, refresh=True)
print("n=", len(p), "mktcap_cov=", round(float(p["mktcap"].notna().mean()), 4),
      "source=", p.attrs.get("source"))
print("excluded:", p.attrs.get("excluded"))
print("deduped:", p.attrs.get("deduped"))
print("has SPCX:", "SPCX" in set(p["code"]),
      "has GOOG:", "GOOG" in set(p["code"]), "has GOOGL:", "GOOGL" in set(p["code"]),
      "has BRK_A:", "BRK_A" in set(p["code"]), "has BRK_B:", "BRK_B" in set(p["code"]))
print("top10:", p["code"].head(10).tolist())
