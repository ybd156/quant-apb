"""
散户情绪 —— 从股吧评论里读出"大部分人怎么想"，并区分"说的多数"与"被看见的多数"。

## 三个层次的输出

1. **数量口径**（每条算一票）：`bull_ratio`、`consensus`
2. **关注度口径**（按阅读数加权）：`bull_ratio_w`、`consensus_w`
   —— 这个更接近"市场真正接收到的观点"，因为一条 1.7 万阅读的帖子
   和一条 12 阅读的帖子，对情绪的影响完全不同。
3. **两者背离**（`attention_gap`）：说的多数 ≠ 被看见的多数时，
   说明**少数派掌握了话语权**（大 V、热门帖），这往往比简单多数更有信息量。

## 情绪打分：词典法 + 否定 + 程度副词

朴素词典法最大的漏洞是"不涨"被判成看多。本版加了：
  - **否定词窗口**：情绪词前 3 字内出现 不/没/别/非/无/未 → 极性翻转
  - **程度副词**：很/非常/极/超 → ×1.5；略/稍/有点 → ×0.5
  - **反问/黑话**：保留原始词表，不做过度规则化（宁可漏判，不可错判）

## 数据与边界（必须先看）

1. 股吧**只给当前列表，不能按历史日期翻页** ⇒ 情绪数据**只能前向采集、不能回测**。
   任何用今天情绪解释昨天涨跌的做法都是自欺。
2. 页面是 JS 渲染，但**内嵌了完整 JSON**（含标题/时间/阅读数/回复数），
   用正则从中提取，比解析 DOM 稳。
3. `gbapi.eastmoney.com` 的 JSON 接口实测返回空（缺签名），**不可用**，已放弃。
4. 词典法测的是"字面情绪"，不是"观点"。反讽会漏判，这是已知上限。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://guba.eastmoney.com/"}
_MIN_INTERVAL = 0.3
_last = [0.0]

# ---------------- 词典 ----------------
BULL_WORDS = [
    "涨停", "大涨", "暴涨", "起飞", "牛市", "抄底", "买入", "加仓", "满仓", "梭哈",
    "突破", "新高", "利好", "强势", "拉升", "翻倍", "看多", "上涨", "冲", "干",
    "反弹", "反转", "底部", "机会", "上车", "持股", "红包", "赚", "红盘", "放量上涨",
    "增持", "回购", "业绩超预期", "订单", "中标", "涨", "牛", "赚了", "看好", "加仓干",
]
BEAR_WORDS = [
    "跌停", "大跌", "暴跌", "跳水", "熊市", "割肉", "卖出", "减仓", "空仓", "清仓",
    "破位", "新低", "利空", "弱势", "砸盘", "腰斩", "看空", "下跌", "崩", "跑",
    "套牢", "顶部", "风险", "出货", "亏损", "绿盘", "放量下跌", "退市", "爆雷", "垃圾",
    "减持", "质押", "商誉", "立案", "问询", "跌", "熊", "亏了", "爆仓", "看淡", "不看好",
]
NEGATIONS = ["不", "没", "没有", "别", "非", "无", "未", "莫", "勿", "难以", "不会", "不能"]
DEGREE_UP = ["很", "非常", "特别", "极", "超", "巨", "十分", "格外", "疯狂"]
DEGREE_DOWN = ["略", "稍", "有点", "略微", "小幅", "些许"]
NEG_WINDOW = 3


@dataclass
class SentimentResult:
    code: str
    as_of: str
    n_posts: int
    bull: int
    bear: int
    neutral: int
    bull_ratio: float
    consensus: float
    dispersion: float
    intensity: float
    bull_ratio_w: float = float("nan")
    consensus_w: float = float("nan")
    attention_gap: float = float("nan")
    extreme_bull: bool = False
    extreme_bear: bool = False
    sample_titles: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def _get(url: str, retries: int = 3, timeout: int = 25) -> str:
    last = None
    for i in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
            _last[0] = time.time()
            return body
        except Exception as e:  # noqa: BLE001
            _last[0] = time.time()
            last = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"guba fetch failed: {url} :: {last}")


# 内嵌 JSON 中，一条帖子内字段顺序稳定：title → last_time → click → comment
# 四个字段在页面内各自出现次数一致（实测各 86 次），但相邻关系不固定，
# 所以分开提取再按位置配对，比"整段相邻"正则稳得多。
_TITLE_RE = re.compile(r'"post_title":"([^"]{1,200})"')
_TIME_RE = re.compile(r'"post_last_time":"([^"]{4,30})"')
_CLICK_RE = re.compile(r'"post_click_count":(\d+)')
_CMT_RE = re.compile(r'"post_comment_count":(\d+)')


def fetch_posts(code: str, pages: int = 3) -> List[Dict]:
    """抓股吧帖子（标题/时间/阅读数/回复数）。pages 越多覆盖越广（每页约 80 条）。"""
    code = str(code).zfill(6)
    out: List[Dict] = []
    for p in range(1, pages + 1):
        url = f"https://guba.eastmoney.com/list,{code},f.html" if p == 1 \
            else f"https://guba.eastmoney.com/list,{code},f_{p}.html"
        try:
            html = _get(url)
        except Exception:
            continue
        titles = _TITLE_RE.findall(html)
        times = _TIME_RE.findall(html)
        clicks = _CLICK_RE.findall(html)
        cmts = _CMT_RE.findall(html)
        n = min(len(titles), len(times))
        for i in range(n):
            out.append({"title": titles[i], "time": times[i],
                        "click": int(clicks[i]) if i < len(clicks) else 0,
                        "comment": int(cmts[i]) if i < len(cmts) else 0})
    # 去重（同一帖子可能出现在多页）
    seen, uniq = set(), []
    for p in out:
        k = (p["title"], p["time"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def score_text(text: str) -> Tuple[float, int]:
    """
    返回 (情绪分 ∈ [-1,1], 强度计数)。
    规则：命中情绪词 → 检查前 NEG_WINDOW 字是否有否定 → 翻转；
          检查前后是否有程度副词 → 放大/缩小。
    """
    t = str(text)
    score = 0.0
    hits = 0
    for words, sign in ((BULL_WORDS, +1), (BEAR_WORDS, -1)):
        for w in words:
            idx = t.find(w)
            while idx >= 0:
                prefix = t[max(0, idx - NEG_WINDOW):idx]
                negated = any(n in prefix for n in NEGATIONS)
                mult = 1.0
                ctx = t[max(0, idx - 4):idx + len(w) + 4]
                if any(d in ctx for d in DEGREE_UP):
                    mult = 1.5
                elif any(d in ctx for d in DEGREE_DOWN):
                    mult = 0.5
                score += sign * mult * (-1.0 if negated else 1.0)
                hits += 1
                idx = t.find(w, idx + len(w))
    if hits == 0:
        return 0.0, 0
    norm = score / max(abs(score), 1.0) if abs(score) <= 1 else score / abs(score)
    intensity = hits + t.count("!") + t.count("！")
    for w in ("必", "一定", "肯定", "绝对"):
        if w in t:
            intensity += 2
    return float(np.clip(norm, -1, 1)), int(intensity)


def analyze(code: str, pages: int = 3, as_of: Optional[str] = None) -> SentimentResult:
    posts = fetch_posts(code, pages)
    if not posts:
        return SentimentResult(str(code).zfill(6), as_of or "", 0, 0, 0, 0,
                               float("nan"), float("nan"), float("nan"), float("nan"),
                               note="no_posts")
    bull = bear = neutral = 0
    w_bull = w_bear = 0.0
    inten, samples = [], []
    for p in posts:
        sc, it = score_text(p["title"])
        w = 1.0 + float(p.get("click", 0)) ** 0.5      # 阅读数开方加权，抑制极端爆款
        if sc > 0.15:
            bull += 1; w_bull += w
        elif sc < -0.15:
            bear += 1; w_bear += w
        else:
            neutral += 1
        inten.append(it)
        if len(samples) < 8 and abs(sc) > 0.15:
            samples.append(f"{p['title'][:36]}（阅读{p.get('click',0)}）")

    n = len(posts)
    con = abs(bull - bear) / max(bull + bear, 1)
    con_w = abs(w_bull - w_bear) / max(w_bull + w_bear, 1e-9) if (w_bull + w_bear) > 0 else float("nan")
    gap = (con_w - con) if np.isfinite(con_w) else float("nan")
    return SentimentResult(
        code=str(code).zfill(6), as_of=as_of or time.strftime("%Y-%m-%d"),
        n_posts=n, bull=bull, bear=bear, neutral=neutral,
        bull_ratio=bull / n if n else float("nan"),
        consensus=float(con), dispersion=float(1 - con),
        intensity=float(np.mean(inten)) if inten else 0.0,
        bull_ratio_w=w_bull / max(w_bull + w_bear, 1e-9) if (w_bull + w_bear) > 0 else float("nan"),
        consensus_w=float(con_w), attention_gap=float(gap),
        extreme_bull=bool(con > 0.6 and bull > bear * 2 and n >= 30),
        extreme_bear=bool(con > 0.6 and bear > bull * 2 and n >= 30),
        sample_titles=samples,
        note="数量口径 vs 关注度口径；attention_gap>0 表示少数派掌握了话语权")


def save(res: SentimentResult, root: str = "data/sentiment") -> str:
    """前向采集台账：只能追加，不能回补历史。"""
    d = os.path.join(root, res.as_of[:7])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{res.as_of}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
    return path


def contrarian_signal(res: SentimentResult) -> float:
    """
    反向指标（先验，未验证）：极端一致看多 → 负；极端一致看空 → 正。
    用**关注度口径**的一致性判断（更接近市场真实接收到的观点）。
    """
    if res.n_posts < 30:
        return 0.0
    con = res.consensus_w if np.isfinite(res.consensus_w) else res.consensus
    if con <= 0.6:
        return 0.0
    strength = min(1.0, (con - 0.6) * 3 + 0.3)
    if res.extreme_bull:
        return -strength
    if res.extreme_bear:
        return strength
    return 0.0
