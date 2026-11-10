"""
样本量门槛的唯一真源（AGENTS §7.5 规则六）。

历史上这三个数字散落在不同模块里，语义还不一致：
    qcore/events.py:85            MIN_N = 5        <- 其实是「横截面最少股票数」
    qcore/impact.py:138           min_n = 8        <- 事件影响的最小样本
    qcore/intraday_online.py:90   MIN_TRADES = 30  <- 单组最少交易笔数

结果：events.json 里 9 行影响因子全部 n<30，却全部 insufficient=False，
被 UI 原样渲染成 9 行正经结论。任何「最优 X」必须先过 MIN_TRADES。

本模块零依赖、无状态，任何模块都可以 import（不会造成循环引用）。
"""
from __future__ import annotations

# 单组最少独立观测数（笔数 / 独立期数）。低于此一律 insufficient。
MIN_TRADES: int = 30

# 横截面最少股票数：某日截面不足这么多只，当日统计量不出点估计。
MIN_XS_NAMES: int = 5

# 市场模型估参窗口最少有效观测数。
MIN_EST: int = 40

__all__ = ["MIN_TRADES", "MIN_XS_NAMES", "MIN_EST"]
