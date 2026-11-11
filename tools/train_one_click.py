#!/usr/bin/env python3
"""
一键训练 —— 在任意一台电脑上，一条命令把模型训完并打包好。

## 用法

    python tools/train_one_click.py

Windows 双击 `train.bat`，Linux/Mac 跑 `bash train.sh`，效果一样。

## 它做什么（全自动，不需要你懂细节）

    1. 环境自检      Python 版本 / numpy / pandas / 磁盘空间
    2. 数据准备      没有就自动下载 Qlib 数据包（约 545MB，解压 3.7GB）
    3. 逐周期训练    T+1 / T+3 / 1周 / 2周 / 15天（可指定）
    4. 打包          生成 dist/models/apb_models_<日期>.zip
    5. 打印上传指令  复制粘贴到 NAS 就能装

## 常用参数

    --horizons 1,5        只训这几个周期（默认 1,3,5,10,15）
    --trees 250           树的数量（慢机器可降到 150）
    --stocks 300          股票池大小
    --gpu                 有 CUDA 显卡就尝试用 GPU（没装 cupy 会自动跳过）
    --skip-data           数据已存在，跳过下载检查
    --dry-run             只检查环境，不训练

## 为什么这样设计

训练和部署是两件事：
  - 训练是**离线批处理**，在哪台机器做都行，慢一点无所谓
  - 部署是**实时推理**，NAS 上 CPU 毫秒级就够

所以训练完只要把 zip 搬过去。本脚本把"搬"这一步也自动化了。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import defaults as _DEF
except Exception:
    _DEF = None

# Cloudflare 的 Browser Integrity Check 会拦掉非浏览器 UA（error 1010）。
# 训练机是脚本，必须伪装成浏览器，否则连 healthz 都过不去。
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

DEFAULT_HORIZONS = (_DEF.horizons() if _DEF else [1, 3, 5, 10, 15])
_DEF_TREES = (_DEF.get()["trees"] if _DEF else 250)
_DEF_STOCKS = (_DEF.get()["stocks"] if _DEF else 300)
_DEF_NAS = (_DEF.nas() if _DEF else "https://apbcdn.683868.xyz")
DIST = os.path.join(ROOT, "dist", "models")


def _c(s: str, code: str) -> str:
    """终端颜色（不支持就原样返回）。"""
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        return s
    return f"\033[{code}m{s}\033[0m"


OK = lambda s: _c(s, "32")     # noqa: E731
WARN = lambda s: _c(s, "33")   # noqa: E731
ERR = lambda s: _c(s, "31")    # noqa: E731
BOLD = lambda s: _c(s, "1")    # noqa: E731


def say(msg: str) -> None:
    print(msg, flush=True)


def step(n: int, total: int, title: str) -> None:
    say("")
    say(BOLD(f"[{n}/{total}] {title}"))
    say("-" * 62)


# ---------------------------------------------------------------- 1. 环境
def check_env() -> dict:
    out = {"ok": True, "problems": [], "warnings": []}
    v = sys.version_info
    say(f"  Python      {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        out["problems"].append(f"Python {v.major}.{v.minor} 太旧，需要 ≥3.10")
        out["ok"] = False

    for mod in ("numpy", "pandas"):
        try:
            m = __import__(mod)
            say(f"  {mod:<11} {m.__version__}")
        except Exception:
            out["problems"].append(f"缺少 {mod}：pip install {mod}")
            out["ok"] = False

    say(f"  平台        {platform.system()} {platform.machine()}")
    say(f"  CPU 核数    {os.cpu_count()}")

    # 磁盘
    try:
        free = shutil.disk_usage(ROOT).free / (1024 ** 3)
        say(f"  磁盘可用    {free:.1f} GB")
        if free < 8:
            out["problems"].append(f"磁盘只剩 {free:.1f}GB，数据+模型需要约 8GB")
            out["ok"] = False
    except Exception:
        pass

    # GPU（可选）
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from qcore import gpu
        gi = gpu.info()
        if gi.get("available"):
            say(f"  GPU         {gi.get('device')} ({gi.get('vram_total_gb')}GB) "
                f"— {OK('可用')}")
            out["gpu"] = True
        else:
            say(f"  GPU         未启用（{gi.get('reason', '')[:50]}）")
            out["gpu"] = False
    except Exception as e:  # noqa: BLE001
        say(f"  GPU         检测失败（{type(e).__name__}），走 CPU")
        out["gpu"] = False

    return out


# ---------------------------------------------------------------- 2. 数据
def ensure_data(skip: bool = False) -> dict:
    from qcore import data_sync as DS
    cur = DS.local_date()
    if cur:
        say(f"  本地数据已到 {OK(cur)}")
        if skip:
            return {"ok": True, "date": cur, "action": "skip"}
    else:
        say(WARN("  本地没有 Qlib 数据，准备下载"))

    say("  检查远端版本…")
    r = DS.sync(force=False, verbose=True)
    if not r.get("ok"):
        say(ERR(f"  数据准备失败：{r.get('reason')}"))
        say(f"  提示：{r.get('hint', '检查网络/代理')}")
        return r
    say(f"  数据就绪：{OK(str(r.get('new_local_date') or r.get('local_date')))}")
    return r


# ---------------------------------------------------------------- 3. 训练
def train_horizons(horizons, trees: int, stocks: int, verbose: bool = True,
                   nas: str = "", token: str = "", upload: bool = True,
                   password: str = "") -> list:
    from qcore import live_model as LM
    from qcore import horizons as HZ

    results = []
    for i, h in enumerate(horizons, 1):
        label = HZ.label_for(h)
        say("")
        say(BOLD(f"  > 周期 {h} — {label}  ({i}/{len(horizons)})"))
        t0 = time.time()
        try:
            r = LM.train(horizon=h, trees=trees, stocks=stocks, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:80]}"}
        dt = time.time() - t0
        if r.get("ok"):
            say(f"    {OK('完成')} {dt/60:.1f} 分钟 - {r.get('n_stocks')} 只 - "
                f"{r.get('n_trees')} 树 - {r.get('train_n')} 样本")
            item = {"horizon": h, "ok": True, "seconds": round(dt, 1),
                    "n_trees": r.get("n_trees"), "label": label}
            # 训完立刻上传，不攒到最后打包
            if upload and nas:
                say(f"    -> 上传到 {nas} ...")
                up = _push_one(nas, token, h, password=password)
                item["uploaded"] = bool(up.get("ok"))
                item["upload_detail"] = up
                if up.get("ok"):
                    say(f"    {OK('已上传')} {up.get('size_kb')} KB")
                else:
                    say(f"    {WARN('上传失败')} {up.get('reason')}")
            results.append(item)
        else:
            say(f"    {ERR('失败')} {r.get('reason')}（{dt/60:.1f} 分钟）")
            results.append({"horizon": h, "ok": False, "seconds": round(dt, 1),
                            "reason": r.get("reason"), "label": label})
    return results


# ---------------------------------------------------------------- 3.5 实时上传
_SESSION = {"cookie": ""}


def _login_cookie(nas: str, password: str, timeout: int = 20) -> str:
    """用 NAS 密码登录，拿会话 cookie。失败返回空串。"""
    if _SESSION["cookie"]:
        return _SESSION["cookie"]
    if not password:
        return ""
    try:
        import urllib.error
        import urllib.request
        payload = json.dumps({"password": password}).encode("utf-8")
        req = urllib.request.Request(
            nas.rstrip("/") + "/api/login", data=payload, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.headers.get("Set-Cookie") or ""
            body = r.read().decode("utf-8", "replace")
        if not raw:
            return ""
        # 只取 apb_session=... 那一段
        for part in raw.split(","):
            kv = part.split(";")[0].strip()
            if kv.startswith("apb_session="):
                _SESSION["cookie"] = kv
                return kv
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _push_one(nas: str, token: str, horizon: int, timeout: int = 180,
              password: str = "") -> dict:
    """
    训完一个周期就立刻传上去 —— 不用先打包。
    失败就返回错误，由调用方决定要不要留 zip 兜底。
    """
    import base64
    import urllib.error
    import urllib.request

    from qcore import model_transfer as MT

    ex = MT.export_bundle(int(horizon))
    if not ex.get("ok"):
        return {"ok": False, "reason": ex.get("reason") or "export_failed"}
    try:
        with open(ex["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"read_failed: {type(e).__name__}"}

    url = nas.rstrip("/") + "/api/sync/push"
    payload = json.dumps({"data": b64, "horizon": int(horizon)}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": _BROWSER_UA}
    if token:
        headers["X-APB-Token"] = token
    else:
        # 没有 token 就登录换 cookie —— 否则永远 login_required
        ck = _login_cookie(nas, password)
        if ck:
            headers["Cookie"] = ck
    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            d = json.loads(body) if body.strip() else {}
        if d.get("ok"):
            return {"ok": True, "size_kb": round(os.path.getsize(ex["path"]) / 1024, 1)}
        return {"ok": False, "reason": str(d.get("error") or d.get("detail"))[:80],
                "checks": d.get("checks")}
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read().decode("utf-8", "replace"))
            reason = str(d.get("error") or d.get("detail") or f"http_{e.code}")[:80]
        except Exception:
            reason = f"http_{e.code}"
        return {"ok": False, "reason": reason, "http": e.code}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:60]}",
                "network": True}


def _nas_reachable(nas: str, token: str, timeout: int = 12) -> bool:
    """开训前先探一下 NAS 通不通，不通就别指望实时上传。"""
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(nas.rstrip("/") + "/api/healthz",
                                     headers={"User-Agent": _BROWSER_UA,
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- 4. 打包
def bundle(horizons) -> dict:
    from qcore import model_transfer as MT
    os.makedirs(DIST, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    made = []
    for h in horizons:
        try:
            r = MT.export_bundle(h, out_path=os.path.join(DIST, f"model_h{h}.zip"))
            if r.get("ok"):
                made.append({"horizon": h, "file": os.path.basename(r["path"]),
                             "size_kb": round(os.path.getsize(r["path"]) / 1024, 1)})
                say(f"  h={h:<3} {os.path.basename(r['path'])} "
                    f"({round(os.path.getsize(r['path'])/1024, 1)} KB)")
        except Exception as e:  # noqa: BLE001
            say(WARN(f"  h={h} 导出失败：{type(e).__name__}"))
    if not made:
        return {"ok": False, "reason": "nothing_exported"}

    # 打成一个总包 + 上传说明
    readme = os.path.join(DIST, "上传说明.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(_upload_instructions([m["file"] for m in made]))
    zip_path = os.path.join(DIST, f"apb_models_{stamp}.zip")
    with shutil_zipfile() as z:
        for m in made:
            z.write(os.path.join(DIST, m["file"]), m["file"])
        z.write(readme, "上传说明.txt")
    return {"ok": True, "zip": zip_path,
            "size_kb": round(os.path.getsize(zip_path) / 1024, 1),
            "models": made}


def shutil_zipfile():
    import zipfile
    class _Z:
        def __enter__(self):
            stamp = datetime.now().strftime("%Y%m%d")
            self.p = os.path.join(DIST, f"apb_models_{stamp}.zip")
            self.z = zipfile.ZipFile(self.p, "w", zipfile.ZIP_DEFLATED, compresslevel=6)
            return self.z
        def __exit__(self, *a):
            self.z.close()
            self.path = self.p
    return _Z()


def _upload_instructions(files: list) -> str:
    lines = [
        "APB 模型上传说明",
        "=" * 50,
        "",
        "本包含以下模型：",
    ]
    for f in files:
        lines.append(f"  - {f}")
    lines += [
        "",
        "【方法一】网页上传（最简单）",
        "  1. 打开 " + _DEF_NAS + "/#/settings",
        "  2. 找到「模型导入」，把 zip 拖进去",
        "",
        "【方法二】命令行",
        "  # 拷到 NAS",
        "  scp apb_models_*.zip 用户名@NAS地址:/tmp/",
        "",
        "  # 进容器导入",
        "  ssh 用户名@NAS地址",
        "  docker cp /tmp/apb_models_*.zip apb:/tmp/",
        "  docker exec -it apb sh -c 'cd /tmp && unzip -o apb_models_*.zip'",
        "  for f in /tmp/model_h*.zip; do",
        "    docker exec -e PYTHONPATH=/app apb python -m qcore.model_transfer --import $f",
        "  done",
        "",
        "导入时会自动做四项校验（反序列化/特征数/feature_idx/能否预测），",
        "校验不过不会安装，线上模型保持不变。",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="一键训练 APB 模型并打包（可在任意电脑运行）")
    ap.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)),
                    help="要训练的周期，逗号分隔（默认 1,3,5,10,15）")
    ap.add_argument("--trees", type=int, default=_DEF_TREES,
                    help=f"树的数量（默认 {_DEF_TREES}）")
    ap.add_argument("--stocks", type=int, default=_DEF_STOCKS,
                    help=f"股票池大小（默认 {_DEF_STOCKS}）")
    ap.add_argument("--gpu", action="store_true", help="尝试使用 GPU（需 cupy）")
    ap.add_argument("--skip-data", action="store_true", help="跳过数据下载检查")
    ap.add_argument("--skip-kit-check", action="store_true",
                    help="跳过套件新旧检查（不推荐）")
    ap.add_argument("--dry-run", action="store_true", help="只检查环境，不训练")
    ap.add_argument("--nas", default="",
                    help="训练完直接传到这个地址（默认读 apb_defaults.json）")
    ap.add_argument("--token", default=(_DEF.token() if _DEF else os.environ.get("APB_TOKEN", "")),
                    help="NAS 的 APB_TOKEN（也可用环境变量 APB_TOKEN 或 apb_defaults.json）")
    ap.add_argument("--password", default=(_DEF.password() if _DEF else os.environ.get("APB_PASSWORD", "")),
                    help="NAS 登录密码；没有 token 时用它换会话上传"
                         "（也可用环境变量 APB_PASSWORD 或 apb_defaults.json）")
    ap.add_argument("--no-upload", action="store_true",
                    help="只训练不上传（模型留在 dist/models/）")
    a = ap.parse_args(argv)

    try:
        hs = [int(x.strip()) for x in a.horizons.split(",") if x.strip()]
    except ValueError:
        say(ERR("--horizons 格式错误，示例：1,5,15"))
        return 2
    if not hs:
        say(ERR("至少要指定一个周期"))
        return 2

    t_start = time.time()
    say("")
    say(BOLD("=" * 62))
    say(BOLD("  APB 一键训练"))
    say(BOLD(f"  周期 {hs} - {a.trees} 树 - {a.stocks} 只"))
    say(BOLD("=" * 62))

    # 1
    step(1, 4, "环境自检")
    env = check_env()
    if not env["ok"]:
        say("")
        for p in env["problems"]:
            say(ERR(f"  [X] {p}"))
        say("")
        say("请先解决上面的问题再运行。")
        return 1
    say(f"  {OK('环境 OK')}")

    # 套件能力自检 —— **只警告，不阻断**。
    #
    # 原来这里 `return 1` 直接退出。问题是 kit_doctor 检查的都是「锦上添花」
    # 的能力（预打包 wheel、NAS 下载源、模拟盘重置），没有一条会让训练跑不动：
    #   · 缺 wheel      -> 装依赖慢一点，不是跑不了
    #   · 缺 NAS 源     -> 走 GitHub，慢或失败，但可能通
    #   · 缺 reset      -> 模拟盘不能清空，跟训练无关
    # 用一个「可能误报」的检查去拦死训练，代价远大于收益 ——
    # 实测用户下的 0.32.6 被误判成旧版，重下了三次都进不去。
    #
    # 真正跑不动的会在实际运行时暴露，那时的报错是具体的。
    # --skip-kit-check 用于完全静默。
    if not a.skip_kit_check:
        try:
            from qcore import kit_doctor as _KD
            if not _KD.print_report():
                say("")
                say("  [!!] 上面这些问题不影响现在开跑，继续。")
                say("       若后面真的报错，再按提示重下套件。")
        except ImportError:
            say("  [!] 没有 kit_doctor 模块 —— 这套代码偏旧，但先按现状开跑")
        except Exception as _e:
            say(f"  [!] 套件自检跳过: {type(_e).__name__}")

    if a.dry_run:
        say("")
        say("--dry-run：只检查环境，未训练。")
        return 0

    # 2
    step(2, 4, "数据准备")
    d = ensure_data(skip=a.skip_data)
    if not d.get("ok"):
        return 1

    # 3
    step(3, 4, "训练")
    nas = "" if a.no_upload else (a.nas or _DEF_NAS)
    token = a.token or ""
    upload = bool(nas)
    if upload:
        say(f"  上传目标：{BOLD(nas)}")
        if _nas_reachable(nas, token):
            say(f"  {OK('NAS 可达')} —— 每个周期训完立刻上传，不打包")
        else:
            # 预检失败不代表一定传不上去（Cloudflare 可能挡了探测请求），
            # 照常尝试；真传不上去会自然退回打包兜底。
            say(f"  {WARN('预检没通')} —— 仍会尝试上传，传不上去再打包兜底")
    else:
        say("  --no-upload：只训练，模型留在 dist/models/")
    if env.get("gpu") and a.gpu:
        say(f"  {WARN('GPU 可用')}，但本项目的 GBDT 直方图是内存受限的，"
            f"实测独显约 6× 加速、核显反而更慢")
    res = train_horizons(hs, a.trees, a.stocks, nas=nas, token=token, upload=upload,
                         password=a.password)
    ok_n = sum(1 for r in res if r["ok"])
    up_n = sum(1 for r in res if r.get("uploaded"))

    total = time.time() - t_start
    say("")
    say(BOLD("=" * 62))
    say(BOLD(f"  完成：{ok_n}/{len(hs)} 个周期，总耗时 {total/60:.1f} 分钟"))
    say(BOLD("=" * 62))

    if ok_n == 0:
        say("")
        say(ERR("  没有任何模型训练成功，跳过上传/打包"))
        return 1

    good = [r["horizon"] for r in res if r["ok"]]

    # 全部上传成功就不打包了 —— 打包只是为了网络不通时兜底
    if upload and up_n >= ok_n:
        step(4, 4, "完成")
        say(f"  {OK('全部已上传')}（{up_n}/{ok_n} 个周期），不需要打包")
        say("")
        say("  NAS 上已经是最新模型，刷新页面就能用。")
        say("")
        return 0

    step(4, 4, "打包兜底")
    if upload and up_n < ok_n:
        say(f"  有 {ok_n - up_n} 个周期没传上去，打包留底：")
    b = bundle(good)
    if not b.get("ok"):
        say(ERR(f"  打包失败：{b.get('reason')}"))
        return 1

    say("")
    say(f"  打包文件：{BOLD(b['zip'])}")
    say(f"  大小：{b['size_kb']} KB")
    say("")
    say("  下一步：把这个 zip 拷到 NAS，用网页「设置 -> 模型导入」上传，")
    say("  或按 zip 里的「上传说明.txt」用命令行导入。")
    say("")
    for r in res:
        mark = OK("[OK]") if r["ok"] else ERR("[X]")
        say(f"    {mark} h={r['horizon']:<3} {r.get('label',''):<14} "
            f"{r['seconds']/60:>5.1f} 分钟"
            + ("" if r["ok"] else f"  {r.get('reason','')[:40]}"))
    say("")
    return 0 if ok_n == len(hs) else (0 if ok_n > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
