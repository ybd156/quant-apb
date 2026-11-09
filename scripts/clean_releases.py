"""清理旧 Release，只保留最近 N 个。

## 为什么需要

每次发版都往 Releases 塞 23MB（含预打包 wheels），
15 个版本就是 250MB。仓库会越来越臃肿，用户也分不清该下哪个。

## 策略

保留最近 KEEP 个（默认 2：当前版 + 上一个回退版），其余全删。
同时删掉对应的 git tag（tag 本身很小，但会让 `git tag` 输出过长，
且旧 tag 对应的代码已经不在维护）。
"""
import json
import os
import subprocess
import sys
import urllib.request

TOKEN = os.environ.get("GH_TOKEN", "")
REPO = "ybd156/quant-apb"
KEEP = int(os.environ.get("APB_KEEP_RELEASES", "2"))


def api(method: str, path: str):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}{path}",
        headers={"Authorization": f"token {TOKEN}",
                 "Accept": "application/vnd.github+json"},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:200]}


def ver_key(tag: str):
    try:
        return tuple(int(x) for x in tag.lstrip("v").split("."))
    except Exception:
        return (0,)


def main():
    if not TOKEN:
        print("缺 GH_TOKEN")
        return 1

    rels = api("GET", "/releases?per_page=100")
    if not isinstance(rels, list):
        print("拉取失败:", rels)
        return 1

    rels.sort(key=lambda r: ver_key(r["tag_name"]), reverse=True)
    print(f"共 {len(rels)} 个 Release，保留最近 {KEEP} 个\n")

    keep = rels[:KEEP]
    drop = rels[KEEP:]

    print("保留：")
    for r in keep:
        mb = sum(a["size"] for a in (r.get("assets") or [])) / 1024 / 1024
        print(f"  {r['tag_name']:<10} {mb:>6.1f} MB")

    if not drop:
        print("\n没有需要删除的")
        return 0

    print(f"\n删除 {len(drop)} 个：")
    freed = 0.0
    for r in drop:
        tag = r["tag_name"]
        mb = sum(a["size"] for a in (r.get("assets") or [])) / 1024 / 1024
        res = api("DELETE", f"/releases/{r['id']}")
        if res.get("_error"):
            print(f"  {tag:<10} 失败 {res['_error']}")
            continue
        freed += mb
        # 删对应 tag
        subprocess.run(
            ["git", "push", "--delete", "origin", tag],
            capture_output=True, text=True, timeout=60,
            env={**os.environ,
                 "GIT_TERMINAL_PROMPT": "0"})
        print(f"  {tag:<10} {mb:>6.1f} MB  已删")

    print(f"\n释放 {freed:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
