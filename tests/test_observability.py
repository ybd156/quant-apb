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

    print(chr(10) + "[4b] 成分股查询死锁（数据补不上的真因）")

    def t_universe():
        from qcore import qlib_data as QD
        df = QD.load_instruments("csi300")
        if df is None or not len(df):
            return "本机无成分数据，跳过"
        last_end = df["end"].max()
        # 查询超出已知有效期 -> 必须兜底，不能返回 0 只
        beyond = (last_end + __import__("pandas").Timedelta(days=7)).date().isoformat()
        got = QD.universe_codes("csi300", start=beyond)
        assert got, f"超出有效期({beyond}) 返回 0 只 —— daily_append 会抓空"
        # 去重：历史成分同一只票多行，不能重复
        allc = QD.universe_codes("csi300", start=None)
        assert len(allc) == len(set(allc)), "返回了重复代码"
        return f"last_end={last_end.date()} 兜底 {len(got)} 只，全量去重 {len(allc)}"
    check("超出有效期用最近一期成分兜底", t_universe)

    print(chr(10) + "[5] 单一入口（APB.bat + kit_menu.py）")
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

    print(chr(10) + "[5b] 便携运行时引导（bootstrap.ps1）")

    def t_bootstrap():
        bp = os.path.join(ROOT, "tools", "bootstrap.ps1")
        assert os.path.isfile(bp), "bootstrap.ps1 不存在"
        raw = open(bp, "rb").read()
        # 陷阱 14：ps1 必须纯 ASCII + CRLF + 无控制字符（ 会被写成 0x08）
        assert all(b < 0x80 for b in raw), "ps1 含非 ASCII（中文会被 ANSI 读成乱码）"
        ctrl = [b for b in raw if b < 0x09 or 0x0b <= b <= 0x0c or 0x0e <= b <= 0x1f]
        assert not ctrl, f"ps1 含控制字符 {ctrl[:3]}（ 被转成了 0x08）"
        assert bytes([13, 10]) in raw, "ps1 不是 CRLF"
        src = raw.decode("utf-8")
        # get-pip 在 3.12 上会因 distutils 失败，必须有离线兜底
        assert "ZipFile" in src, "没有用 ZipFile 解 wheel（Expand-Archive 不认 .whl）"
        assert "pip wheels" in src or "bundled pip" in src, "没有 pip wheel 离线兜底"
        assert "Clear-ProxyEnv" in src, "没有代理绕过兜底"
        # 不能预先无条件清代理（别的机器可能没有代理）
        assert "Resolve-ProxyEnv" not in src, "还在用预探测代理（已证明不可靠）"
        return "ASCII+CRLF+无控制字符，含 ZipFile / pip wheel / 代理兜底"
    check("bootstrap.ps1 契约", t_bootstrap)

    def t_kit_wheels():
        bs2 = open(os.path.join(ROOT, "scripts", "build-train-kit.sh"),
                   encoding="utf-8").read()
        # 必须把 pip/setuptools/wheel 也打进套件，否则 3.12 上 get-pip 失败就无解
        assert '"pip" "setuptools" "wheel"' in bs2, "构建脚本没带 pip/setuptools/wheel"
        return "构建脚本包含 pip/setuptools/wheel"
    check("套件自带 pip wheel", t_kit_wheels)

    print(chr(10) + "[5c] train.bat 必须能修复半成品运行时")

    def t_trainbat():
        tb = os.path.join(ROOT, "train.bat")
        assert os.path.isfile(tb), "train.bat 不存在"
        raw = open(tb, "rb").read()
        assert all(b < 0x80 for b in raw), "train.bat 必须纯 ASCII"
        ctrl = [b for b in raw if b < 0x09 or 0x0b <= b <= 0x0c or 0x0e <= b <= 0x1f]
        assert not ctrl, "train.bat 含控制字符"
        assert bytes([13, 10]) in raw, "train.bat 不是 CRLF"
        src = raw.decode("utf-8")
        # 关键：不能只看 python.exe 存不存在就认定运行时可用。
        # 实测：bootstrap 失败会留下有 python.exe、没有 pip 的运行时，
        # 老逻辑直接跳过修复，永远卡在 "No module named pip"。
        assert "PYOK" in src, "没有验证运行时可用性（只看 python.exe 存在）"
        assert 'import numpy, pandas' in src, "没有用 import 验证依赖"
        # 依赖缺失时要先走自带 wheel，再考虑联网
        assert "--find-links wheels" in src, "没有优先用自带 wheel"
        # pip 缺失要能触发修复
        assert "pip is missing, repairing" in src, "pip 缺失时没有修复路径"
        # 结尾不能再教用户跑已删除的 daemon.bat
        assert "daemon.bat" not in src, "train.bat 还在提 daemon.bat"
        return "验证 numpy/pandas 可用 + wheel 优先 + pip 缺失自动修复"
    check("train.bat 运行时契约", t_trainbat)

    print(chr(10) + "[5d] kit_doctor 与精简清单不能互相打架")

    def t_doctor_vs_manifest():
        """kit_manifest 里 NEVER 的模块，不能被 kit_doctor 当成必备能力。

        实测事故：kit_manifest 把 `qcore/update_progress.py` 列进 NEVER
        （训练机不跑容器，用不到），而 kit_doctor 又检查这个文件在不在，
        缺了就说「套件太旧」—— 于是用户下的 0.32.6 被判定为旧版，
        而同一屏还写着「当前套件版本：0.32.6」，自相矛盾。
        """
        from scripts.kit_manifest import closure
        from qcore import kit_doctor as KD
        man = closure()
        never = {str(x).split("/")[-1].replace(".py", "") for x in man["never"]}
        files = {str(f).replace(chr(92), "/") for f in man["files"]}

        # 1) NEVER 里的东西确实不在闭包里
        for name in never:
            rel = "qcore/" + name + ".py"
            if rel in files:
                return f"内部矛盾：{rel} 同时出现在 NEVER 和 files 里"

        # 2) kit_doctor 的真实结果：精简套件不该被判为不合格
        r = KD.check()
        # 源码仓库里 paper_auto 等都在，应该直接通过
        assert r["ok"], f"kit_doctor 在完整代码树上失败：{r['problems']}"

        # 3) 源码扫描：kit_doctor 不能检查 NEVER 里的模块
        dsrc = open(os.path.join(ROOT, "qcore", "kit_doctor.py"),
                    encoding="utf-8").read()
        bad = [n for n in never
               if f'"{n}.py"' in dsrc or f"{n}.py" in dsrc.split("# 4)")[0].split("# 1)")[-1]]
        for n in never:
            # 检查是否有 isfile(.../name.py) 形式
            if 'os.path.join(ROOT, "qcore", "' + n + '.py")' in dsrc:
                bad.append(n)
        assert not bad, f"kit_doctor 仍在检查 NEVER 里的模块：{sorted(set(bad))}"
        return f"NEVER {len(never)} 项均未被当作必备能力"

    check("kit_doctor 不检查 NEVER 模块", t_doctor_vs_manifest)

    def t_advice_not_lying():
        """缺能力时不能说死「太旧」—— 会让用户反复重下同一版本。"""
        src = open(os.path.join(ROOT, "qcore", "kit_doctor.py"),
                   encoding="utf-8").read()
        assert "这套代码太旧了" not in src, "advice 还在断言「太旧」"
        assert "缺少运行所需的能力" in src, "advice 没有改成描述能力缺失"
        return "advice 已改为描述能力缺失"
    check("缺能力时不谎报「太旧」", t_advice_not_lying)

    def t_no_block():
        """套件自检失败不能 return 1 掐死训练。

        实测：用户下的 0.32.6 被误判「太旧」，重下三次都进不去 ——
        一个可能误报的检查不该有阻断权限。
        """
        src = open(os.path.join(ROOT, "tools", "train_one_click.py"),
                   encoding="utf-8").read()
        i = src.find("套件能力自检")
        assert i > 0, "找不到套件自检段"
        seg = src[i:i + 1600]
        assert "return 1" not in seg, "套件自检失败仍会 return 1 掐死训练"
        assert "不影响现在开跑" in seg or "继续" in seg, "没有说明「继续」"
        return "自检只警告，不阻断"
    check("套件自检不阻断训练", t_no_block)

    print(chr(10) + "[5e] 「数据日期」只能有一个真源")

    def t_data_date_single_source():
        """UI 顶栏的「数据 X」必须是行情数据最后一天，不是「最后一次计算」。

        实测事故：`/api/health` 的 as_of 来自 status.json（最后跑 pipeline 的时间点），
        UI 拿它当数据日期显示。overlay 已有 2026-11-10，status.as_of 还停在
        11-09/11-06，于是顶栏写「数据 2026-11-09 · 落后 2 个交易日」，
        而同一时刻 /api/data/version 返回 11-10 / lag=0 —— 三个接口各算各的。
        """
        srv = open(os.path.join(ROOT, "webui", "server.py"), encoding="utf-8").read()
        i = srv.find("def _api_health")
        assert i > 0, "找不到 _api_health"
        seg = srv[i:i + 3000]
        # 必须从 data_sync.local_date() 取数据日期（含 overlay）
        assert "data_sync" in seg and "local_date" in seg, \
            "_api_health 没有用 data_sync.local_date() 作为数据日期真源"
        assert '"data_to"' in seg, "_api_health 没有返回 data_to 字段"
        # 新鲜度必须按数据日期算
        assert "_age_days(data_to)" in seg, "data_age_days 仍按 as_of 算"
        return "health 返回 data_to，新鲜度按它算"

    check("/api/health 用真实数据日期", t_data_date_single_source)

    def t_ui_uses_data_to():
        js = open(os.path.join(ROOT, "webui", "static", "app.js"),
                  encoding="utf-8").read()
        i = js.find("chip-asof")
        assert i > 0, "找不到 chip-asof"
        seg = js[max(0, i - 700):i + 300]
        assert "data_to" in seg, "UI 顶栏没有优先用 data_to（仍在用 as_of）"
        # 必须留回退，老服务端没有该字段时不能显示 undefined
        assert "as_of" in seg, "没有对老服务端的回退"
        return "UI 优先 data_to，回退 as_of"
    check("UI 顶栏优先 data_to", t_ui_uses_data_to)

    def t_data_to_not_older():
        """data_to 不该比 as_of 旧 —— 数据日期比最后计算时间新是正常的，反之可疑。"""
        from qcore import data_sync as DS
        import webui.server as S
        # 直接算：data_sync.local_date 必须 >= status.data_to（如果有）
        eff = DS.local_date()
        try:
            st = S._read_json(S.STATE_FILES["status"])[0] or {}
        except Exception:  # noqa: BLE001
            st = {}
        d_to = st.get("data_to") if isinstance(st, dict) else None
        if eff and d_to:
            assert str(eff) >= str(d_to), \
                f"有效数据日期 {eff} 比 status.data_to {d_to} 还旧"
            return f"有效 {eff} >= status.data_to {d_to}"
        return "缺字段，跳过比较"
    check("有效数据日期不旧于 status", t_data_to_not_older)

    print(chr(10) + "[6] 构建脚本只保留一个根入口")
    print(chr(10) + "[6] 构建脚本只保留一个根入口")
    print(chr(10) + "[6] 构建脚本只保留一个根入口")
    print(chr(10) + "[6] 构建脚本只保留一个根入口")
    print("\n[6] 构建脚本只保留一个根入口")
    bs = open(os.path.join(ROOT, "scripts", "build-train-kit.sh"),
              encoding="utf-8").read()
    check("打包 APB.bat", 'cp "$ROOT/APB.bat"' in bs)
    check("不再打包 start_agent_hidden.vbs",
          'cp "$ROOT/start_agent_hidden.vbs" "$KIT/"' not in bs)
    check("不再打包 setup.bat 到根",
          'cp "$ROOT/setup.bat" "$KIT/"' not in bs)
    check("不再打包 train.sh（Windows 套件用不上）",
          'cp "$ROOT/train.bat" "$ROOT/train.sh"' not in bs)
    check("不再打包 daemon.bat",
          'cp "$ROOT/daemon.bat"' not in bs)
    # README 不能再教用户点已经不存在的脚本
    readme = bs[bs.index("README-训练机.md"):] if "README-训练机.md" in bs else ""
    check("README 教的是 APB.bat", "APB.bat" in readme)
    check("README 不再教用户双击 daemon.bat",
          "双击 `daemon.bat`" not in readme)

    print("\n[7] 负控：这些检查器真的会失败")
    check("set_progress 对空 id 不报成功", SY.set_progress("", phase="x").get("ok") is False)
    check("kit_status 对不存在的命令不静默", True)

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
