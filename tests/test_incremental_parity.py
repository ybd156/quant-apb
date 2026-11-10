"""
增量 = 批量：同一历史日必须得到同一个 regime / 同一份分数。

实测问题（Backend Architect 逐位探针）：
    full (3360,283,73) vs inc (610,283,73)，重叠 610 天：
      DIR / STR  : 0 差异（信号层是稳的）
      REGIME     : 57/610 天不同（9.3%）  <-- 决定用哪套权重
      SIGMA      : 6,621 格；UNIV: 2,388 格
根因：`build_regime` 拿 start 之后的切片算 vol_pct(250)，窗口前 250 天全 NaN，
`fillna(0)` 把它伪装成真实 regime 0。现在改为恒用完整指数历史计算。

本测试用 6 只票的小面板，断言：
  1. REGIME 在重叠区间**逐位相等**；
  2. 预热期之后（skip=260 天）DIR / STR / UNIV / SIGMA 也逐位相等；
  3. 预热期之内的差异只允许出现在窗口开头，且被明确报告（不静默）。

跑法：python -m tests.test_incremental_parity
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qcore.dataset import build_dataset  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" :: {detail}" if detail else ""))


def _codes(n: int = 6):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "kline", "*_h2.csv")))
    return [os.path.basename(p).split("_")[0] for p in files[:n]]


def main() -> int:
    codes = _codes()
    if len(codes) < 3:
        print("[SKIP] 本地行情不足（data/raw/kline/*_h2.csv < 3）")
        return 0
    cutoff = "2024-06-28"
    print(f"[info] codes={codes} cutoff={cutoff}")

    full = build_dataset(codes, start="2018-01-01", horizon=5, cutoff=cutoff,
                         extra_horizons=[1, 20], verbose=False)
    inc = build_dataset(codes, start="2023-01-01", horizon=5, cutoff=cutoff,
                        extra_horizons=[1, 20], verbose=False)
    print(f"[info] full T={full.shape[0]}  inc T={inc.shape[0]}  N={full.shape[1]}")

    # 对齐重叠日期
    f_dates = np.asarray(full.dates)
    i_dates = np.asarray(inc.dates)
    common = np.intersect1d(f_dates, i_dates)
    if common.size < 100:
        print(f"[SKIP] 重叠区间太短（{common.size} 天）")
        return 0
    fi = np.searchsorted(f_dates, common)
    ii = np.searchsorted(i_dates, common)
    print(f"[info] overlapping dates = {common.size}")

    # ---- 1) REGIME 逐位相等（这是本次修复的目标）
    r_f = full.REGIME[fi]
    r_i = inc.REGIME[ii]
    n_diff = int((r_f != r_i).sum())
    check("REGIME 逐位相等", n_diff == 0,
          f"diff={n_diff}/{common.size}")
    if n_diff:
        bad = common[r_f != r_i][:5]
        print(f"       前 5 个不同日期: {[str(x)[:10] for x in bad]}")

    # ---- 2) 预热期之后，信号与掩码也必须逐位相等
    skip = 260
    if common.size > skip + 50:
        sl = slice(skip, None)
        d_dir = int((full.DIR[fi][sl] != inc.DIR[ii][sl]).sum())
        d_str = int((full.STR[fi][sl] != inc.STR[ii][sl]).sum())
        d_univ = int((full.UNIV[fi][sl] != inc.UNIV[ii][sl]).sum())
        d_sig = int((~np.isclose(full.SIGMA[fi][sl], inc.SIGMA[ii][sl],
                                 rtol=1e-6, atol=0, equal_nan=True)).sum())
        check("DIR 逐位相等（skip 预热后）", d_dir == 0, f"diff={d_dir}")
        check("STR 逐位相等（skip 预热后）", d_str == 0, f"diff={d_str}")
        check("UNIV 逐位相等（skip 预热后）", d_univ == 0, f"diff={d_univ}")
        check("SIGMA 逐位相等（skip 预热后）", d_sig == 0, f"diff={d_sig}")

        # ---- 3) 预热期内的差异必须被量化报告，而不是静默
        pre_dir = int((full.DIR[fi][:skip] != inc.DIR[ii][:skip]).sum())
        print(f"  [info] 预热期（前 {skip} 天）DIR 差异 = {pre_dir} 格"
              f"（窗口起点不同导致，属预期，不计为失败）")

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
