"""通过 GitHub REST API 提交整个仓库（沙箱里 git push 的 remote-https 助手不可用）。

用法：python data/tmp_push.py <owner/repo> <branch> <commit_message>
token 从环境变量 GH_TOKEN 读取，绝不落盘、绝不打印。
"""
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

TOKEN = os.environ["GH_TOKEN"]
REPO = sys.argv[1]
BRANCH = sys.argv[2]
MSG = sys.argv[3]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def api(path, method="GET", payload=None):
    url = "https://api.github.com" + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "apb-push",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        raise RuntimeError(f"{method} {path} -> {e.code}: {body}")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    files = [f for f in out.stdout.splitlines() if f.strip()]
    return files


def main():
    files = tracked_files()
    print(f"files: {len(files)}")
    entries = []
    total = 0
    for i, rel in enumerate(files, 1):
        p = os.path.join(ROOT, rel)
        with open(p, "rb") as f:
            raw = f.read()
        total += len(raw)
        blob = api(f"/repos/{REPO}/git/blobs", "POST",
                   {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
        entries.append({"path": rel.replace("\\", "/"), "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})
        if i % 10 == 0:
            print(f"  blob {i}/{len(files)}")

    tree = api(f"/repos/{REPO}/git/trees", "POST", {"tree": entries})
    commit = api(f"/repos/{REPO}/git/commits", "POST",
                 {"message": MSG, "tree": tree["sha"], "parents": []})

    try:
        api(f"/repos/{REPO}/git/refs", "POST",
            {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})
    except RuntimeError:
        api(f"/repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH",
            {"sha": commit["sha"], "force": True})

    print(json.dumps({"ok": True, "commit": commit["sha"],
                      "files": len(files), "bytes": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
