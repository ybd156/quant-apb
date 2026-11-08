"""
图表渲染 —— 纯 Pillow，零额外依赖。

为什么不用 matplotlib：
  目标运行环境（小主机/NAS）装 matplotlib 要拖一堆编译依赖；
  本项目只需要折线、条形、K 线三种图，Pillow 的 ImageDraw 足够，
  且输出 PNG 体积小、无字体缓存问题。

中文字体：优先找系统 CJK 字体，找不到就退回默认字体（英文仍可读）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# 主题
BG = (17, 20, 26)
PANEL = (24, 28, 36)
GRID = (45, 52, 64)
FG = (226, 232, 240)
MUTED = (140, 152, 170)
UP = (239, 83, 80)        # A 股习惯：红涨
DOWN = (38, 166, 154)     # 绿跌
ACCENT = (96, 165, 250)
WARN = (250, 204, 21)

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
_font_cache: Dict[int, ImageFont.ImageFont] = {}


def font(size: int = 14) -> ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                return f
            except Exception:
                continue
    try:
        f = ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()
    _font_cache[size] = f
    return f


def _nice_range(lo: float, hi: float, pad: float = 0.05) -> Tuple[float, float]:
    if not (lo < hi):
        return lo - 1, hi + 1
    d = hi - lo
    return lo - d * pad, hi + d * pad


def _axes(d: ImageDraw.ImageDraw, box: Tuple[int, int, int, int],
          ylim: Tuple[float, float], yticks: int = 5, fmt: str = "{:.2f}") -> None:
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=GRID)
    lo, hi = ylim
    for i in range(yticks + 1):
        y = y0 + (y1 - y0) * i / yticks
        v = hi - (hi - lo) * i / yticks
        d.line([(x0, y), (x1, y)], fill=GRID)
        d.text((x0 - 6, y), fmt.format(v), font=font(11), fill=MUTED, anchor="rm")


def line_png(path: str, series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
             title: str = "", subtitle: str = "", width: int = 1000, height: int = 380,
             fmt: str = "{:.2f}", colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
             hline: Optional[float] = None) -> str:
    """
    series: {名称: (xs, ys)}。xs 可为空序列，此时按索引。
    """
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 14), title, font=font(18), fill=FG)
    if subtitle:
        d.text((20, 40), subtitle, font=font(12), fill=MUTED)

    box = (72, 70, width - 20, height - 46)
    allx, ally = [], []
    for xs, ys in series.values():
        ally += [v for v in ys if v == v]
        allx += list(range(len(ys))) if (xs is None or len(xs) == 0) else list(xs)
    if not ally:
        d.text((box[0], box[1] + 40), "no data", font=font(14), fill=MUTED)
        img.save(path)
        return path
    ylim = _nice_range(min(ally), max(ally))
    _axes(d, box, ylim, fmt=fmt)

    if hline is not None and ylim[0] <= hline <= ylim[1]:
        y = box[1] + (box[3] - box[1]) * (ylim[1] - hline) / (ylim[1] - ylim[0])
        for x in range(box[0], box[2], 12):
            d.line([(x, y), (x + 6, y)], fill=MUTED)

    palette = [ACCENT, UP, DOWN, WARN, (167, 139, 250), (45, 212, 191)]
    for i, (name, (xs, ys)) in enumerate(series.items()):
        col = (colors or {}).get(name) or palette[i % len(palette)]
        n = len(ys)
        if n < 2:
            continue
        pts = []
        for j, v in enumerate(ys):
            if v != v:
                continue
            x = box[0] + (box[2] - box[0]) * j / max(n - 1, 1)
            y = box[1] + (box[3] - box[1]) * (ylim[1] - v) / (ylim[1] - ylim[0])
            pts.append((x, y))
        if len(pts) > 1:
            d.line(pts, fill=col, width=2, joint="curve")
        # 图例
        lx = box[0] + 10 + i * 170
        d.line([(lx, 56), (lx + 18, 56)], fill=col, width=3)
        d.text((lx + 24, 56), name, font=font(12), fill=FG, anchor="lm")

    d.text((20, height - 24), f"last={ally[-1]:.4f}  min={min(ally):.4f}  max={max(ally):.4f}",
           font=font(11), fill=MUTED)
    img.save(path)
    return path


def bar_png(path: str, labels: Sequence[str], values: Sequence[float],
            title: str = "", subtitle: str = "", width: int = 1000, height: int = 420,
            fmt: str = "{:+.4f}", pos_color=UP, neg_color=DOWN) -> str:
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 14), title, font=font(18), fill=FG)
    if subtitle:
        d.text((20, 40), subtitle, font=font(12), fill=MUTED)

    n = len(values)
    if n == 0:
        img.save(path)
        return path
    box = (150, 70, width - 30, height - 40)
    lo, hi = _nice_range(min(min(values), 0), max(max(values), 0), 0.08)
    _axes(d, box, (lo, hi), fmt="{:.3f}")

    zero_y = box[1] + (box[3] - box[1]) * (hi - 0) / (hi - lo)
    bh = (box[3] - box[1]) / max(n, 1) * 0.62
    for i, (lab, v) in enumerate(zip(labels, values)):
        cy = box[1] + (box[3] - box[1]) * (i + 0.5) / n
        x = box[0] + (box[2] - box[0]) * (0 - lo) / (hi - lo)
        w = (box[2] - box[0]) * (v - 0) / (hi - lo)
        col = pos_color if v >= 0 else neg_color
        d.rectangle([x, cy - bh / 2, x + w, cy + bh / 2], fill=col)
        d.text((box[0] - 8, cy), str(lab)[:18], font=font(11), fill=FG, anchor="rm")
        d.text((x + w + (6 if v >= 0 else -6), cy), fmt.format(v), font=font(10),
               fill=MUTED, anchor="lm" if v >= 0 else "rm")
    d.line([(box[0] + (box[2] - box[0]) * (0 - lo) / (hi - lo), box[1]),
            (box[0] + (box[2] - box[0]) * (0 - lo) / (hi - lo), box[3])], fill=MUTED)
    img.save(path)
    return path


def candle_png(path: str, dates: Sequence[str], o: Sequence[float], h: Sequence[float],
               l: Sequence[float], c: Sequence[float], v: Optional[Sequence[float]] = None,
               ma: Optional[Dict[str, Sequence[float]]] = None,
               title: str = "", subtitle: str = "",
               width: int = 1000, height: int = 520, tail: int = 120) -> str:
    """K 线 + 均线 + 成交量（当前形态用）。"""
    n = min(tail, len(c))
    o, h, l, c = o[-n:], h[-n:], l[-n:], c[-n:]
    v = list(v[-n:]) if v is not None else None
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 14), title, font=font(18), fill=FG)
    if subtitle:
        d.text((20, 40), subtitle, font=font(12), fill=MUTED)

    vol_h = 90 if v else 0
    box = (70, 70, width - 20, height - 40 - vol_h)
    lo = min(l); hi = max(h)
    ylim = _nice_range(lo, hi, 0.06)
    _axes(d, box, ylim, fmt="{:.2f}")

    cw = (box[2] - box[0]) / max(n, 1)
    for i in range(n):
        x = box[0] + cw * (i + 0.5)
        def yv(val):
            return box[1] + (box[3] - box[1]) * (ylim[1] - val) / (ylim[1] - ylim[0])
        col = UP if c[i] >= o[i] else DOWN
        d.line([(x, yv(h[i])), (x, yv(l[i]))], fill=col, width=1)
        d.rectangle([x - cw * 0.32, yv(max(o[i], c[i])), x + cw * 0.32, yv(min(o[i], c[i]))], fill=col)

    for name, series in (ma or {}).items():
        ys = list(series)[-n:]
        pts = [(box[0] + cw * (i + 0.5), box[1] + (box[3] - box[1]) * (ylim[1] - ys[i]) / (ylim[1] - ylim[0]))
               for i in range(len(ys)) if ys[i] == ys[i]]
        if len(pts) > 1:
            d.line(pts, fill=ACCENT, width=1)
            d.text((box[2] - 10, pts[-1][1]), name, font=font(10), fill=ACCENT, anchor="rm")

    if v:
        vb = (box[0], box[3] + 16, box[2], height - 30)
        vmax = max([x for x in v if x == x] or [1])
        d.rectangle(vb, outline=GRID)
        for i in range(n):
            x = box[0] + cw * (i + 0.5)
            hh = (vb[3] - vb[1]) * (v[i] / vmax if v[i] == v[i] else 0)
            col = UP if c[i] >= o[i] else DOWN
            d.rectangle([x - cw * 0.32, vb[3] - hh, x + cw * 0.32, vb[3]], fill=col)

    if dates:
        for frac in (0, 0.5, 1.0):
            i = int((n - 1) * frac)
            d.text((box[0] + cw * (i + 0.5), height - 18), str(dates[i])[:10],
                   font=font(10), fill=MUTED, anchor="mm")
    img.save(path)
    return path


def regime_band_png(path: str, dates: Sequence[str], regimes: Sequence[int],
                    width: int = 1000, height: int = 150) -> str:
    """市场状态色带。"""
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), "市场状态色带（0=跌低波 1=跌高波 2=涨低波 3=涨高波）",
           font=font(15), fill=FG)
    box = (70, 46, width - 20, height - 34)
    colors = {0: DOWN, 1: (13, 90, 84), 2: (180, 60, 58), 3: UP}
    n = len(regimes)
    if n == 0:
        img.save(path); return path
    cw = (box[2] - box[0]) / n
    for i, r in enumerate(regimes):
        x = box[0] + cw * i
        d.rectangle([x, box[1], x + max(cw, 1), box[3]], fill=colors.get(int(r), MUTED))
    d.text((20, height - 24), f"{dates[0][:10]} → {dates[-1][:10]}  ({n} 个交易日)",
           font=font(11), fill=MUTED)
    img.save(path)
    return path
