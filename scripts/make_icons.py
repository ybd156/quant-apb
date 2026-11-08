"""生成 fnOS 应用图标（PNG）。在 CI 里也会调用，保证可复现。

输出：deploy/fnos/fpk/ICON.PNG (64x64)、ICON_256.PNG (256x256)
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "deploy", "fnos", "fpk")

BG = (17, 20, 26)
ACCENT = (59, 130, 246)
FG = (226, 232, 240)


def _font(size: int) -> ImageFont.ImageFont:
    for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make(size: int, path: str) -> str:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(size * 0.22)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG)
    # 左侧强调条
    d.rounded_rectangle([0, 0, int(size * 0.12), size - 1], radius=r, fill=ACCENT)
    d.rectangle([int(size * 0.06), 0, int(size * 0.12), size - 1], fill=ACCENT)
    # 文字
    txt = "APB"
    f = _font(int(size * 0.30))
    bbox = d.textbbox((0, 0), txt, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size + size * 0.10 - tw) / 2, (size - th) / 2 - bbox[1]), txt, font=f, fill=FG)
    # 底部小点（三个，象征三层证据）
    for i, x in enumerate((0.40, 0.52, 0.64)):
        cy = int(size * 0.74)
        rad = max(2, int(size * 0.022))
        d.ellipse([int(size * x) - rad, cy - rad, int(size * x) + rad, cy + rad],
                  fill=ACCENT if i == 1 else (90, 110, 140))
    img.save(path, "PNG")
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s, name in ((64, "ICON.PNG"), (256, "ICON_256.PNG")):
        p = make(s, os.path.join(OUT, name))
        print("wrote", p, os.path.getsize(p), "bytes")
