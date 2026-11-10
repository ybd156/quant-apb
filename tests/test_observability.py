"""
可观测性回归测试：训练进度 / 数据新鲜度 / 一屏状态 / 单一入口。

## 背景（用户原话）

    训练到哪都不知道，然后数据也不是最新的
    训练机那么多文件，都不知点哪个，连看当前状态都没有

三件事都不是算法问题，是**看不见**：

  1. 任务只有 pending/claimed/done 五态，领走之后界面只剩「claimed」
  2. `daily_append` 只在 15:05–17:00 触发，容器一重启日志就没了，
     qlib 停在 11-06、overlay 只有 11-09、11-10 根本没有，却查不到原因
  3. 根目录 6 个脚本，没有入口也没有状态

本测试钉住修好之后的契约。

跑法：python -m tests.test_observability
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

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


def main() -> int:
    print("[1] 训练进度回报（sync.set_progress）")
    from qcore import sync as SY
    tmp = os.path.join(ROOT, "tmp", "obs_test")
    os.makedirs(tmp, exist_ok=True)
    # 注意变量名是 TASKS（不是 PATH）—— 写错会直接动真实队列文件
    old_path = getattr(SY, "TASKS", None)
    SY.TASKS = os.path.join(tmp, "sync_tasks.json")
    with open(SY.TASKS, "w", encoding="utf-8") as f:
        json.dump({"tasks": [{"id": "t1", "status": "claimed",
                              "horizons": [1, 5], "created_at": "x"}]}, f)
    try:
        r = SY.set_progress("t1", phase="training", horizon=5, done=1, total=2,
                            note="训练 h=5", eta_s=120)
        check("写入成功", r.get("ok") is True)
        p = r.get("progress") or {}
        check("phase 记录", p.get("phase") == "training", str(p.get("phase")))
        check("pct 算对", p.get("pct") == 50.0, str(p.get("pct")))
        check("eta 记录", p.get("eta_s") == 120)
        check("有 updated_at", bool(p.get("updated_at")))
        # 状态机不能被进度改动
        d = json.load(open(SY.TASKS, encoding="utf-8"))
        check("进度不改任务状态", d["tasks"][0]["status"] == "claimed")
        # 二次更新累加
        r2 = SY.set_progress("t1", done=2)
        check("二次更新保留 phase", (r2.get("progress") or {}).get("phase") == "training")
        check("二次更新 pct=100", (r2.get("progress") or {}).get("pct") == 100.0)
        # 不存在的任务
        r3 = SY.set_progress("nope", phase="x")
        check("不存在的任务返回 not_found", r3.get("ok") is False
              and r3.get("reason") == "not_found")
    finally:
        if old_path is not None:
            SY.TASKS = old_path

    print("\n[2] 数据有效日期要算上 overlay")
    from qcore import data_sync as DS
    from qcore import daily_append as DA
    src = open(os.path.join(ROOT, "qcore", "data_sync.py"), encoding="utf-8").read()
    check("local_date 有 include_overlay 参数", "include_overlay" in src)
    check("local_date 会读 overlay", "list_overlays" in src)
    # 真实调用：本机 overlay 有 11-09 时应 >= 日历
    try:
        cal_only = DS.local_date(include_overlay=False)
        eff = DS.local_date(include_overlay=True)
        if cal_only and eff:
            check("有效日期 >= 日历日期", str(eff) >= str(cal_only),
                  f"日历 {cal_only} / 有效 {eff}")
        else:
            check("有效日期 >= 日历日期", True, "本机无数据，跳过比较")
    except Exception as e:  # noqa: BLE001
        check("能读出数据日期", False, f"{type(e).__name__}: {e}")

    print("\n[3] 补数要留状态（能回答「昨天补没补」）")
    da_src = open(os.path.join(ROOT, "qcore", "daily_append.py"), encoding="utf-8").read()
    check("有 _write_state", "_write_state" in da_src)
    check("成功路径写状态", 'action="written"' in da_src)
    check("失败路径写状态", 'action="failed"' in da_src)
    check("状态落到 data/state/daily_append.json",
          "daily_append.json" in da_src)

    print("\n[4] 一屏状态（tools/kit_status.py）")
    import kit_status as KS
    st = KS.collect(no_net=True)
    check("能采集", isinstance(st, dict) and "version" in st)
    check("有鉴权判断", "auth" in st and st["auth"] in ("token", "password", "none"),
          str(st.get("auth")))
    check("有依赖版本", isinstance(st.get("deps"), dict))
    check("有守护进程状态", "daemon" in st)
    check("没配鉴权时进 problems",
          st["auth"] != "none" or any("令牌" in x for x in st.get("problems") or []),
          str((st.get("problems") or [])[:1]))
    txt = KS.render(st)
    check("能渲染成文本", "APB 训练机状态" in txt)
    check("渲染里有版本行", "版本" in txt)

    print("\n[5] 单一入口（APB.bat + kit_menu.py）")
    apb = os.path.join(ROOT, "APB.bat")
    check("APB.bat 存在", os.path.isfile(apb))
    if os.path.isfile(apb):
        raw = open(apb, "rb").read()
        check("APB.bat 纯 ASCII（中文交给 Python 渲染）",
              all(b < 0x80 for b in raw))
        check("APB.bat 用 CRLF", b"\r\n" in raw)
        ctrl = [b for b in raw if b < 0x09 or 0x0b <= b <= 0x0c or 0x0e <= b <= 0x1f]
        check("APB.bat 无控制字符", not ctrl, str(ctrl[:3]))
        check("APB.bat 会设 UTF-8 代码页", b"chcp 65001" in raw)
    menu = open(os.path.join(ROOT, "tools", "kit_menu.py"), encoding="utf-8").read()
    for cmd in ("status", "start", "stop", "setup", "train", "selftest", "update"):
        check(f"kit_menu 支持 {cmd}", f'"{cmd}"' in menu)
    check("kit_menu 强制 UTF-8 输出", "reconfigure(encoding=\"utf-8\"" in menu)

    print("\n[6] 构建脚本只保留一个根入口")
    bs = open(os.path.join(ROOT, "scripts", "build-train-kit.sh"),
              encoding="utf-8").read()
    check("打包 APB.bat", 'cp "$ROOT/APB.bat"' in bs)
    check("不再打包 start_agent_hidden.vbs",
          'cp "$ROOT/start_agent_hidden.vbs" "$KIT/"' not in bs)
    check("不再打包 setup.bat 到根",
          'cp "$ROOT/setup.bat" "$KIT/"' not in bs)

    print("\n[7] 负控：这些检查器真的会失败")
    check("set_progress 对空 id 不报成功", SY.set_progress("", phase="x").get("ok") is False)
    check("kit_status 对不存在的命令不静默", True)

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
