"""
散户情绪 —— 从股吧评论里读出"大部分人怎么想"，并量化"一致 vs 分歧"。

## 为什么盯"一致性"而不是"看多看空"

单看"70% 看多"信息量很低——牛市里几乎所有股票都 70% 看多。
真正有区分度的是**极端一致性**：当散户意见高度统一时，
往往意味着**边际买盘已经耗尽**（该买的都买了），这是反向指标的核心逻辑。

所以本模块输出三个数：
    bull_ratio   看多占比
    consensus    一致性 = |多 − 空| / 总数       （0=分歧，1=完全一致）
    dispersion   分歧度 = 1 − consensus

## 数据与边界（必须先看）

1. 股吧页面**只给当前列表，不能按历史日期翻**。
   所以情绪数据**只能前向采集、不能回测** —— 与分钟数据同一性质。
   任何用今天情绪去解释昨天涨跌的做法都是自欺。
2. 评论标题短、噪声大、有大量情绪化刷屏与机器人。
   本模块用**词典法**（可解释、无训练、无泄漏），不用模型。
3. 词典法测的是"字面情绪"，不是"观点"。反讽、黑话（"yyds""药丸"）会漏判，
   这是已知上限，不要假装它是 NLP 大模型。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
      "Referer": "https://guba.eastmoney.com/"}
_MIN_INTERVAL = 0.3
_last = [0.0]

# 情绪词典（可扩展；每个词按强度 1 计）
BULL_WORDS = [
    "涨停", "大涨", "暴涨", "起飞", "牛市", "抄底", "买入", "加仓", "满仓", "梭哈",
    "突破", "新高", "利好", "强势", "拉升", "翻倍", "看多", "上涨", "冲", "干",
    "反弹", "反转", "底部", "机会", "上车", "持股", "红包", "赚", "红盘", "放量上涨",
]
BEAR_WORDS = [
    "跌停", "大跌", "暴跌", "跳水", "熊市", "割肉", "卖出", "减仓", "空仓", "清仓",
    "破位", "新低", "利空", "弱势", "砸盘", "腰斩", "看空", "下跌", "崩", "跑",
    "套牢", "顶部", "风险", "出货", "亏损", "绿盘", "放量下跌", "退市", "爆雷", "垃圾",
]
INTENSIFIERS = ["必", "一定", "肯定", "绝对", "全力", "死", "梭哈", "all in", "ALL IN"]


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
    intensity: float          # 平均情绪强度（含感叹号/加强词）
    extreme_bull: bool
    extreme_bear: bool
    sample_titles: List[str]

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


_TITLE_RE = re.compile(r'"post_title"\s*:\s*"([^"]{2,120})"')
_TIME_RE = re.compile(r'"post_publish_time"\s*:\s*"([^"]{4,30})"')


def fetch_posts(code: str, pages: int = 3) -> List[Dict]:
    """
    抓股吧帖子标题。pages 越多覆盖越广（每页约 80 条）。
    只返回标题与时间，不做正文抓取（正文请求量×N，且噪声更大）。
    """
    code = str(code).zfill(6)
    out: List[Dict] = []
    for p in range(1, pages + 1):
        url = (f"https://guba.eastmoney.com/list,{code}"
               + (f"_f{p}.html" if p > 1 else ".html"))
        try:
            html = _get(url)
        except Exception:
            continue
        titles = _TITLE_RE.findall(html)
        times = _TIME_RE.findall(html)
        for i, t in enumerate(titles):
            out.append({"title": t,
                        "time": times[i] if i < len(times) else ""})
    return out


def score_text(text: str) -> Tuple[float, int]:
    """返回 (情绪分 ∈ [-1,1], 强度计数)。"""
    t = str(text)
    b = sum(1 for w in BULL_WORDS if w in t)
    s = sum(1 for w in BEAR_WORDS if w in t)
    if b == 0 and s == 0:
        return 0.0, 0
    score = (b - s) / max(b + s, 1)
    intensity = (b + s) + t.count("!") + t.count("！")
    for w in INTENSIFIERS:
        if w in t:
            intensity += 2
    return float(np.clip(score, -1, 1)), int(intensity)


def analyze(code: str, pages: int = 3, as_of: Optional[str] = None) -> SentimentResult:
    """抓取并汇总某只股票的散户情绪。"""
    posts = fetch_posts(code, pages)
    if not posts:
        return SentimentResult(str(code).zfill(6), as_of or "", 0, 0, 0, 0,
                               float("nan"), float("nan"), float("nan"),
                               float("nan"), False, False, [])
    bull = bear = neutral = 0
    inten = []
    samples = []
    for p in posts:
        sc, it = score_text(p["title"])
        if sc > 0.15:
            bull += 1
        elif sc < -0.15:
            bear += 1
        else:
            neutral += 1
        inten.append(it)
        if len(samples) < 8 and abs(sc) > 0.15:
            samples.append(p["title"][:40])

    n = len(posts)
    con = abs(bull - bear) / max(bull + bear, 1)
    return SentimentResult(
        code=str(code).zfill(6),
        as_of=as_of or time.strftime("%Y-%m-%d"),
        n_posts=n, bull=bull, bear=bear, neutral=neutral,
        bull_ratio=bull / n if n else float("nan"),
        consensus=float(con),
        dispersion=float(1 - con),
        intensity=float(np.mean(inten)) if inten else 0.0,
        extreme_bull=bool(con > 0.6 and bull > bear * 2 and n >= 30),
        extreme_bear=bool(con > 0.6 and bear > bull * 2 and n >= 30),
        sample_titles=samples,
    )


def save(res: SentimentResult, root: str = "data/sentiment") -> str:
    """前向采集台账：只能追加，不能回补历史。"""
    import os
    d = os.path.join(root, res.as_of[:7])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{res.as_of}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
    return path


def contrarian_signal(res: SentimentResult) -> float:
    """
    反向指标（先验，未验证）：极端一致看多 → 负；极端一致看空 → 正。
    返回 ∈ [-1,1]，0 表示不表态。
    """
    if res.n_posts < 30:
        return 0.0
    if res.extreme_bull:
        return -min(1.0, (res.consensus - 0.6) * 3 + 0.3)
    if res.extreme_bear:
        return min(1.0, (res.consensus - 0.6) * 3 + 0.3)
    return 0.0
