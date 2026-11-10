"""
事件驱动影响因子层
==================

把"政策/公告/新闻"翻译成可回测的数字：一个事件对某个板块在 h 日内的
**平均异常收益（direction-signed CAR）**，以及这个平均值有多可信
（t 值 / 样本数 / 置信区间 / 分年度稳定性）。

四问框架（每个事件都必须能回答）
--------------------------------
1. **分类**   `Event.type` ∈ {policy, earnings, order, capital, regulation, geopolitics, macro}
2. **影响路径** `POLICY_TRANSMISSION`: subtype -> [(板块, 方向, 传导逻辑)]
   基线（事件前市场已知什么）→ 增量（新事实改变了什么）→ 传导（销量/价格/成本/
   现金流/份额/估值/仅情绪）→ 首个验证变量 → 失效条件
3. **市场反应** `event_study`: 事件前 CAR(-5,-1) 判断是否已 price-in；
   CAR(0,h) 看首次反应与后续接受（延续 / 回补 / 跌回）
4. **证据等级** `Event.evidence_level` ∈ {strong, medium, weak, conflict}

核心数学（市场模型 / event study）
--------------------------------
    r_i,t      = alpha + beta * r_m,t + eps     估参窗口 [-120,-20]，**绝不含事件后**
    AR_i,t     = r_i,t - (alpha_hat + beta_hat * r_m,t)
    CAR(t1,t2) = sum_{t=t1..t2} AR_i,t

横截面（跨事件）t 值复用 `qcore.stats.mean_tstat(car_list, lags=...)`，
事件日重叠时用 Newey-West HAC（lags = 重叠跨度）。

口径警告（血泪教训，勿改）
--------------------------
* 收益率一律为**小数**（0.01 = 1%），不是百分数。传百分数则所有 CAR 放大 100 倍。
* 事件日 = 市场**首次可交易日**。盘后/周末公告请把 `date` 写成下一个交易日，
  或调用时传 `lag=1`（本模块用"事件日或其后第一个交易日"定位，不做事后回填）。
* 板块收益优先用东财行业板块指数（`fetch_board_returns`）；拿不到时用个股等权
  聚合近似（`build_sector_returns`），口径差异见各自 docstring。
* 事件研究是**相关不是因果**：同一窗口内可能有别的事件，CAR 是"净效应 + 噪声"。

泄漏自查（audit/leak_scan.py 对本文件的 MEDIUM 命中，逐条说明）
-------------------------------------------------------------
1. `_agg()` 里的 mean/median/min/max/sum 是**横截面**统计（跨事件汇总 CAR），
   不是时间序列上的全样本标准化，不参与任何"当日信号"计算。
2. `impact_factor()` 同理：它产出的是**估计表**（研究结果），不是逐日打分输入。
   若要用它做预测/回测，必须传 `as_of=t`，只允许 `date <= t` 的事件进表
   —— 这正是 SPEC §0.0 的要求；不传 as_of 得到的是全样本表，只能用于"看历史规律"。
3. `build_sector_returns()` 的等权平均是**当日横截面**平均（每日独立），
   不含任何未来信息。
4. `event_study()` 的估参窗口固定为 [-120,-20]，事件窗口 [-5,+10]，
   两者都不含事件后数据；`tests/test_events.py::test_estimation_window_has_no_lookahead`
   用"事件后数据整体 ×5 但 alpha/beta 必须逐位不变"证明这一点。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from qcore.stats import mean_tstat  # 复用现成 HAC / t 值，不重写统计层

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_DIR = os.path.join(ROOT, "data", "events")
EVENTS_PATH = os.path.join(EVENTS_DIR, "events.jsonl")

# ------------------------------------------------------------------ 枚举
EVENT_TYPES: Tuple[str, ...] = (
    "policy",        # 政策（货币/财政/产业/资本市场制度）
    "earnings",      # 业绩（预告/快报/年报季报/指引）
    "order",         # 订单（大单/中标/产能投放/涨价函）
    "capital",       # 资本运作（增发/回购/并购/减持/分红）
    "regulation",    # 监管（立案/处罚/退市/问询/停牌核查）
    "geopolitics",   # 地缘（制裁/出口管制/关税/冲突）
    "macro",         # 宏观（CPI/PMI/社融/汇率/海外议息）
)
EVENT_TYPE_CN: Dict[str, str] = {
    "policy": "政策类", "earnings": "业绩类", "order": "订单类",
    "capital": "资本运作", "regulation": "监管类", "geopolitics": "地缘",
    "macro": "宏观",
}
SCOPES: Tuple[str, ...] = ("market", "sector", "stock")
EVIDENCE_LEVELS: Tuple[str, ...] = ("strong", "medium", "weak", "conflict")

from .thresholds import MIN_TRADES, MIN_XS_NAMES

# 横截面样本下限：某日截面不足这么多只，不给点估计。
# 注意：这不是「单组最少交易笔数」—— 那个是 MIN_TRADES=30（AGENTS §7.5 规则六）。
# 两个语义共用一个旧名 MIN_N，正是「n=20 却 insufficient=False」的来源。
MIN_N = MIN_XS_NAMES
MIN_EST = 40       # 市场模型估参窗口最少有效观测数


# ------------------------------------------------------------------ 数据模型
@dataclass
class Event:
    """一个可回测的事件。字段口径见模块 docstring 的四问框架。"""

    eid: str
    date: str                       # 可观测日 YYYY-MM-DD（首次可交易日）
    type: str                       # EVENT_TYPES
    subtype: str = ""               # 细分：rrr_cut / stamp_tax_cut / collection_bid ...
    title: str = ""                 # 一句话事实（不含观点）
    source: str = ""                # 来源 URL（政府网/央行/证监会/交易所/新华社）
    source_reliability: float = 0.8  # 0..1，1=原文可核
    scope: str = "sector"           # market|sector|stock
    direction: int = 0              # +1 利多 / -1 利空 / 0 方向不明
    sectors: List[str] = field(default_factory=list)
    evidence_level: str = "medium"  # strong|medium|weak|conflict
    pit_backfillable: bool = True   # 该事件当时是否可点对点观测（事后补录=False）
    catalyst_window: int = 20       # 催化窗口（交易日）
    note: str = ""                  # 失效条件 / 首个验证变量 / 备注
    lag: int = 0                    # date -> 首个可交易日 的偏移；盘后公告=1

    def __post_init__(self) -> None:
        self.date = _to_date_str(self.date)
        self.eid = str(self.eid)
        self.type = str(self.type)
        self.subtype = str(self.subtype or "")
        self.sectors = [str(s) for s in (self.sectors or [])]
        try:
            d = int(self.direction)
        except (TypeError, ValueError):
            d = 0
        self.direction = d if d in (-1, 0, 1) else (1 if d > 0 else -1)
        try:
            r = float(self.source_reliability)
        except (TypeError, ValueError):
            r = 0.5
        self.source_reliability = float(min(1.0, max(0.0, r)))
        try:
            self.catalyst_window = max(1, int(self.catalyst_window))
        except (TypeError, ValueError):
            self.catalyst_window = 20
        self.scope = self.scope if self.scope in SCOPES else "sector"
        self.evidence_level = (self.evidence_level
                               if self.evidence_level in EVIDENCE_LEVELS else "weak")
        self.pit_backfillable = bool(self.pit_backfillable)
        try:
            self.lag = int(self.lag)
        except (TypeError, ValueError):
            self.lag = 0
        self.lag = min(5, max(0, self.lag))

    @property
    def year(self) -> int:
        return int(self.date[:4])

    def as_dict(self) -> Dict:
        return asdict(self)


def validate_event(ev: Event) -> List[str]:
    """返回问题清单（纯 ASCII，便于写入任何编码的日志）。空列表 = 通过。"""
    bad: List[str] = []
    if not ev.eid:
        bad.append("empty eid")
    if not (len(ev.date) == 10 and ev.date[4] == "-" and ev.date[7] == "-"):
        bad.append(f"bad date: {ev.date!r}")
    if ev.type not in EVENT_TYPES:
        bad.append(f"unknown type: {ev.type!r}")
    if ev.scope not in SCOPES:
        bad.append(f"unknown scope: {ev.scope!r}")
    if ev.evidence_level not in EVIDENCE_LEVELS:
        bad.append(f"unknown evidence_level: {ev.evidence_level!r}")
    if ev.direction not in (-1, 0, 1):
        bad.append(f"bad direction: {ev.direction!r}")
    if not (0.0 <= ev.source_reliability <= 1.0):
        bad.append(f"reliability out of range: {ev.source_reliability!r}")
    if not (0 <= ev.lag <= 5):
        bad.append(f"bad lag: {ev.lag!r}")
    if ev.scope in ("sector", "stock") and not ev.sectors:
        bad.append("scope requires sectors")
    if not ev.source.startswith(("http://", "https://")):
        bad.append("source is not a URL")
    return bad


# ------------------------------------------------------------------ 传导表
# subtype -> [(板块, 方向, 传导逻辑)]
# 方向是**板块**方向，不是市场方向。板块名用通用行业名，落地时经
# expand_sector() 模糊映射到东财/申万的实际行业字段（见 SECTOR_ALIASES）。
POLICY_TRANSMISSION: Dict[str, List[Tuple[str, int, str]]] = {
    # ---------------- 货币政策
    "rrr_cut": [
        ("银行", 1, "降准释放长期低成本资金，压低负债成本、放大可用头寸；净息差压力与规模扩张对冲，方向偏正但幅度温和"),
        ("地产", 1, "按揭与开发贷额度宽松，销售端预期改善；失效条件=销售面积同比仍负"),
        ("券商", 1, "流动性宽松抬升风险偏好与成交量，经纪/两融收入弹性最大"),
        ("建筑装饰", 1, "基建配套融资改善，订单落地率提升"),
        ("成长", 1, "无风险利率下行抬升长久期资产估值（分母端），对盈利无直接作用"),
    ],
    "rate_cut": [
        ("银行", -1, "贷款端先降、存款端后降，净息差收窄；仅在负债成本同步下调时转为中性"),
        ("地产", 1, "按揭利率下行直接改善购买力，是地产链最灵敏的单一变量"),
        ("券商", 1, "分母端估值 + 交易活跃度双击"),
        ("公用事业", 1, "高股息资产相对吸引力上升（类债券）"),
        ("有色金属", 1, "实际利率下行利多贵金属；工业金属看需求不看利率"),
    ],
    "rrr_hike": [
        ("银行", -1, "抽紧银行体系流动性，可用头寸与信贷投放受限；紧缩预期同时压制估值"),
        ("地产", -1, "开发贷与按揭额度收紧，销售与拿地双双走弱"),
        ("券商", -1, "流动性收缩压低成交量与风险偏好，高贝塔板块首当其冲"),
        ("建筑装饰", -1, "配套融资受限，基建订单落地率下降"),
    ],
    "rate_hike": [
        ("银行", 1, "贷款端先调、存款端滞后，短期净息差扩张；但紧缩预期会压制估值，方向可能反转"),
        ("地产", -1, "按揭利率上行直接削弱购买力，是地产链最灵敏的单一变量"),
        ("券商", -1, "无风险利率上行压缩估值，成交量下滑"),
        ("公用事业", -1, "高股息资产相对吸引力下降（类债券定价）"),
        ("有色金属", -1, "实际利率上行压制贵金属；需求端预期同步走弱"),
    ],
    "rate_liberalization": [
        ("银行", -1, "存款利率上限放开抬升负债成本，中小行息差压力更大"),
        ("券商", 0, "利率市场化本身对券商无直接传导"),
    ],
    "mortgage_rate_cut": [
        ("地产", 1, "取消房贷利率下限 + 首付比例下调，直接改善购买力与置换需求"),
        ("建筑材料", 1, "竣工与开工链需求预期改善"),
        ("家用电器", 1, "地产后周期需求，滞后 1-2 个季度验证"),
        ("银行", -1, "按揭重定价压缩资产端收益率，存量房贷利率下调直接让利"),
    ],
    "housing_fund_rate": [
        ("地产", 1, "公积金贷款利率下调降低刚需月供，对二三线成交弹性更明显"),
        ("建筑材料", 1, "销售改善传导至开工需要 1-2 个季度"),
    ],
    "buyback_loan": [
        ("券商", 1, "回购增持再贷款 + 互换便利直接给券商自营与两融提供子弹"),
        ("银行", 1, "高股息大行是回购增持的主力标的"),
        ("沪深300", 1, "两项工具定向支持宽基权重股，托底指数"),
    ],
    "fund_industry_support": [
        ("券商", 1, "公募扩容直接增厚代销与托管收入，头部集中度提升"),
        ("沪深300", 1, "权益基金规模扩张带来被动增量资金"),
    ],
    "soe_mktcap": [
        ("沪深300", 1, "央企市值管理纳入考核，回购分红与破净修复抬升权重股"),
        ("银行", 1, "破净央企大行是最典型的市值管理对象"),
        ("建筑装饰", 1, "建筑央企破净比例高，估值修复弹性大"),
    ],
    "bse_support": [
        ("券商", 1, "北交所扩容与做市业务增厚投行/做市收入"),
        ("次新股", 1, "多层次市场定位明确，中小市值稀缺性抬升（情绪驱动）"),
    ],
    "securities_law": [
        ("券商", 1, "注册制与投资者保护制度完善，长期利好投行与财富管理"),
        ("次新股", -1, "信息披露与退市约束收紧，壳价值折价"),
    ],
    "regulatory_tighten": [
        ("券商", -1, "合规成本与业务限制上升，短期情绪受压"),
        ("计算机", -1, "数据与信披合规成本上升，中小市值题材股承压"),
    ],
    "capital_market_reform": [
        ("券商", 1, "投行/财富管理/衍生品扩容，头部集中度提升"),
        ("银行", 1, "强调分红回购与长期资金入市，高股息蓝筹受益"),
        ("公用事业", 1, "红利资产在长期资金配置中的权重上升"),
    ],
    "lpr_reform": [
        ("银行", -1, "利率并轨压缩存贷利差，中长期净息差承压"),
        ("地产", 1, "房贷利率锚切换，按揭成本下行"),
    ],
    # ---------------- 资本市场制度
    "stamp_tax_cut": [
        ("券商", 1, "交易成本下降直接提升换手率与经纪收入，历史单日弹性最大的政策之一"),
        ("成长", 1, "高换手小盘更受益于成本下降与情绪修复"),
    ],
    "stamp_tax_hike": [
        ("券商", -1, "交易成本上升抑制换手，经纪业务与两融同步受损"),
    ],
    "ipo_slowdown": [
        ("券商", -1, "投行收入与直投退出节奏放缓"),
        ("次新股", 1, "供给收缩抬升存量小盘稀缺性（情绪驱动，持续性弱）"),
    ],
    "registration_reform": [
        ("券商", 1, "投行项目储备与跟投收益扩张，头部集中度提升"),
        ("创投", 1, "退出通道拓宽，PE/VC 账面价值重估"),
        ("次新股", -1, "壳价值与稀缺性溢价消失，小盘估值中枢下移"),
    ],
    "delisting_strict": [
        ("微盘股", -1, "退市风险显性化，壳价值折价"),
        ("券商", 0, "投行与并购重组业务此消彼长，净影响不明"),
    ],
    "reduce_restriction": [
        ("券商", 1, "减持受限降低大股东抛压，改善市场供需"),
        ("次新股", 1, "解禁冲击后移，短期供给压力缓解"),
    ],
    "refinance_tighten": [
        ("券商", -1, "再融资与并购配套融资规模收缩，投行收入下降"),
        ("次新股", -1, "外延扩张路径受限，成长故事折价"),
    ],
    "long_money_inflow": [
        ("银行", 1, "长钱偏好高股息低波，银行/公用是最直接受益池"),
        ("公用事业", 1, "保险与社保增配高股息资产"),
        ("沪深300", 1, "宽基 ETF 增量资金抬升权重股"),
    ],
    "national_team_buy": [
        ("银行", 1, "汇金增持四大行是最直接、最可核的护盘动作"),
        ("沪深300", 1, "宽基 ETF 申购托底指数"),
        ("券商", 1, "情绪修复的贝塔品种"),
    ],
    # ---------------- 财政 / 产业
    "vat_reform": [
        ("建筑装饰", 1, "营改增后进项抵扣改善，规范企业税负下降"),
        ("交通运输", 1, "可抵扣进项增加，实际税负下降"),
    ],
    "infra_stimulus": [
        ("建筑装饰", 1, "专项债与基建订单直接落地，收入确认领先其他板块"),
        ("建筑材料", 1, "水泥/玻璃需求随开工回升"),
        ("工程机械", 1, "设备开工小时数与销量滞后 1-2 个季度验证"),
        ("钢铁", 1, "需求端预期改善，但需铁矿石成本不涨"),
    ],
    "consumption_voucher": [
        ("商贸零售", 1, "补贴直接拉动可选消费流水"),
        ("家用电器", 1, "以旧换新补贴对白电/厨电销量弹性最直接"),
        ("汽车", 1, "购置税减免/置换补贴对销量弹性高"),
        ("食品饮料", 0, "必选消费弹性小，更多是情绪"),
    ],
    "new_energy_subsidy": [
        ("电力设备", 1, "装机与招标量直接受益（光伏/风电/储能）"),
        ("汽车", 1, "新能源车补贴与购置税优惠拉动销量"),
        ("有色金属", 1, "锂/钴/铜需求预期上修"),
    ],
    "subsidy_phaseout": [
        ("电力设备", -1, "补贴退坡压缩单瓦盈利，抢装后需求真空"),
        ("汽车", -1, "新能源车补贴退坡抬升终端价格，销量增速下台阶"),
    ],
    "collection_bid": [   # 药品/器械集采
        ("医药", -1, "中标价大幅下降压缩仿制药/耗材毛利，量增难补价跌；失效条件=创新药占比高则免疫"),
    ],
    "medical_insurance_negotiation": [
        ("医药", -1, "医保谈判降价换量，创新药放量但单价下修"),
    ],
    "innovation_drug_support": [
        ("医药", 1, "审评加速 + 医保倾斜，创新药收入曲线前移"),
    ],
    "property_tighten": [
        ("地产", -1, "限购限贷加码直接压制销售与拿地"),
        ("建筑材料", -1, "开工与竣工链需求同步走弱"),
        ("家用电器", -1, "地产后周期需求滞后下行"),
        ("银行", -1, "开发贷与按揭资产质量担忧上升"),
    ],
    "property_loosen": [
        ("地产", 1, "首付/利率/限购三支箭，销售弹性最大的是高杠杆民企"),
        ("建筑材料", 1, "竣工链先于开工链修复"),
        ("家用电器", 1, "后周期需求改善，滞后 1-2 个季度"),
        ("银行", 1, "资产质量担忧缓解，估值修复"),
    ],
    "three_red_lines": [
        ("地产", -1, "有息负债增速被硬约束，高杠杆房企融资断流"),
        ("银行", -1, "涉房敞口质量担忧，但幅度取决于按揭占比"),
    ],
    "land_finance_tighten": [
        ("地产", -1, "土地出让两集中抬高地价不确定性"),
    ],
    # ---------------- 监管 / 地缘
    "export_control": [
        ("半导体", -1, "设备/EDA/先进制程断供，资本开支与技术迭代受阻"),
        ("计算机", -1, "高端算力芯片受限，AI 训练成本上升"),
        ("信创", 1, "国产替代逻辑被强化，订单与政策资源倾斜（事件驱动弹性最大）"),
        ("稀土", 1, "反制手段之一，出口管制抬升海外价格"),
    ],
    "tariff_hike": [
        ("电子", -1, "出口占比高的消费电子/PCB 直接受损"),
        ("家用电器", -1, "对美出口占比高的白电/小家电受损"),
        ("纺织服饰", -1, "订单转移至东南亚，代工毛利压缩"),
        ("农林牧渔", 1, "进口替代与国内涨价逻辑（大豆/猪肉）"),
    ],
    "sanction_entity_list": [
        ("半导体", -1, "被列实体清单的公司供应链与客户双受限"),
        ("国防军工", 1, "自主可控与装备列装加速"),
    ],
    "environmental_limit": [
        ("钢铁", 1, "限产收缩供给，吨钢毛利扩张（需求不崩时）"),
        ("煤炭", 1, "供给收缩抬价，但需需求端不崩；失效条件=下游需求同步下滑"),
        ("基础化工", 1, "高耗能品种（电石/纯碱/黄磷）涨价"),
        ("电力", 0, "发电量下降但电价上行，净影响不明"),
    ],
    "carbon_target": [
        ("电力设备", 1, "新能源装机长期空间打开"),
        ("煤炭", -1, "长期需求天花板下移；短期靠供给收缩对冲，失效条件=煤价先跌"),
        ("钢铁", -1, "碳成本内部化抬升吨钢成本"),
    ],
    "semiconductor_fund": [
        ("半导体", 1, "大基金三期注册资本与投向直接决定设备/材料订单"),
    ],
    "data_security": [
        ("计算机", -1, "数据合规成本上升，平台型公司商业模式受限"),
        ("信创", 1, "安全合规采购抬升国产软硬件需求"),
    ],
    "education_tighten": [
        ("教育", -1, "学科类培训营利性被禁止，收入模型直接失效（强证据、路径唯一）"),
        ("传媒", 0, "情绪波及，基本面无直接传导"),
    ],
    "platform_economy_tighten": [
        ("传媒", -1, "平台反垄断与合规成本上升，广告与游戏流水承压"),
        ("商贸零售", -1, "社区团购/电商补贴受限，GMV 增速下台阶"),
    ],
    "platform_economy_support": [
        ("传媒", 1, "平台监管常态化，估值压制解除"),
        ("商贸零售", 1, "促消费与平台经济政策转向"),
    ],
    # ---------------- 宏观
    "cpi_up": [
        ("食品饮料", 1, "提价能力强的白酒/调味品受益于通胀预期"),
        ("农林牧渔", 1, "猪价/粮价上行直接改善养殖与种植利润"),
    ],
    "pmi_up": [
        ("机械设备", 1, "制造业景气回升，订单与开工率同步改善"),
        ("有色金属", 1, "工业金属需求预期上修"),
    ],
    "rmb_depreciate": [
        ("家用电器", 1, "出口占比高，汇兑收益 + 价格竞争力"),
        ("纺织服饰", 1, "出口订单与汇兑收益双升"),
        ("航空", -1, "美元负债重估损失 + 油价成本上升"),
    ],
    "fed_hike": [
        ("成长", -1, "全球无风险利率上行压制高估值成长分母端"),
        ("有色金属", -1, "美元走强压制金属价格"),
        ("银行", 0, "息差与资产质量对冲，方向不明"),
    ],
    "fed_cut": [
        ("成长", 1, "贴现率下行抬升成长股估值"),
        ("有色金属", 1, "美元走弱利多贵金属"),
    ],
    "oil_spike": [
        ("石油石化", 1, "上游勘探开采直接受益于油价"),
        ("航空", -1, "燃油成本占营业成本 30%+"),
        ("基础化工", -1, "原料成本上行，传导滞后"),
    ],
}

# 通用板块名 -> 东财/申万行业字段的可能写法（用于落地时模糊匹配）
SECTOR_ALIASES: Dict[str, List[str]] = {
    "银行": ["银行"],
    "券商": ["证券", "多元金融"],
    "保险": ["保险"],
    "地产": ["房地产开发", "房地产服务", "房地产"],
    "医药": ["化学制药", "中药", "生物制品", "医疗器械", "医疗服务", "医药商业", "医药"],
    "半导体": ["半导体", "电子元件", "光学光电子", "消费电子", "电子"],
    "电子": ["电子", "半导体", "消费电子", "元件", "光学光电子"],
    "计算机": ["计算机", "软件开发", "IT服务", "计算机设备"],
    "信创": ["软件开发", "计算机设备", "IT服务", "通信设备"],
    "传媒": ["传媒", "游戏", "广告营销", "影视院线", "出版"],
    "电力设备": ["电池", "光伏设备", "风电设备", "电网设备", "电机", "电力设备"],
    "汽车": ["汽车整车", "汽车零部件", "乘用车", "商用车", "汽车"],
    "有色金属": ["有色金属", "工业金属", "贵金属", "小金属", "能源金属"],
    "稀土": ["小金属", "有色金属"],
    "钢铁": ["钢铁", "普钢", "特钢"],
    "煤炭": ["煤炭", "煤炭开采", "焦炭"],
    "基础化工": ["化学原料", "化学制品", "化学纤维", "塑料", "橡胶", "农化制品"],
    "石油石化": ["石油石化", "炼化及贸易", "油气开采"],
    "公用事业": ["电力", "燃气", "公用事业"],
    "电力": ["电力", "电网设备"],
    "交通运输": ["航运港口", "航空机场", "物流", "铁路公路", "交通运输"],
    "航空": ["航空机场"],
    "建筑装饰": ["建筑装饰", "基础建设", "专业工程", "房屋建设"],
    "建筑材料": ["建筑材料", "水泥", "玻璃玻纤", "装修建材"],
    "工程机械": ["工程机械", "专用设备", "通用设备"],
    "机械设备": ["机械设备", "通用设备", "专用设备", "自动化设备"],
    "家用电器": ["白色家电", "黑色家电", "小家电", "厨卫电器", "家用电器"],
    "食品饮料": ["白酒", "食品加工", "饮料乳品", "休闲食品", "调味发酵品", "食品饮料"],
    "农林牧渔": ["养殖业", "种植业", "饲料", "农产品加工", "农林牧渔"],
    "商贸零售": ["一般零售", "专业连锁", "互联网电商", "贸易", "商贸零售"],
    "纺织服饰": ["纺织制造", "服装家纺", "纺织服饰"],
    "社会服务": ["旅游及景区", "酒店餐饮", "教育", "专业服务", "社会服务"],
    "教育": ["教育", "社会服务"],
    "国防军工": ["航天装备", "航空装备", "地面兵装", "航海装备", "军工电子"],
    "环保": ["环境治理", "环保设备", "环保"],
    "通信": ["通信服务", "通信设备"],
    "微盘股": ["微盘股"],
    "次新股": ["次新股"],
    "成长": ["创业板", "科创50"],
    "沪深300": ["沪深300"],
}


def expand_sector(name: str, available: Iterable[str]) -> List[str]:
    """
    把通用板块名映射到实际行业字段。先精确命中，再按 SECTOR_ALIASES 模糊包含。
    找不到就原样返回（下游按"缺数据"降级，不静默替换成别的板块）。
    """
    avail = [str(a) for a in available]
    if name in avail:
        return [name]
    keys = SECTOR_ALIASES.get(name, [name])
    hit = [a for a in avail if any(k in a for k in keys)]
    return hit if hit else [name]


def _resolve_series(sec: str, sector_ret: Dict[str, pd.Series]
                    ) -> Optional[Tuple[str, pd.Series, str]]:
    """
    把日历里的通用板块名落到实际可用的行业序列上。

    返回 (用于取数的 key, 序列, 说明)：
      * 精确命中 -> 直接用
      * 模糊命中多个行业（如 医药 -> 化学制药+医疗服务）-> **等权平均**成一条合成序列
      * 一个都没命中 -> None（调用方按"缺数据"降级，绝不偷换成别的板块）
    """
    if not isinstance(sector_ret, dict) or not sector_ret:
        return None
    if sec in sector_ret:
        return sec, sector_ret[sec], sec
    hits = [k for k in expand_sector(sec, sector_ret.keys()) if k in sector_ret]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0], sector_ret[hits[0]], hits[0]
    mats = [_to_ret_series(sector_ret[k]) for k in hits]
    mats = [m for m in mats if m is not None and len(m)]
    if not mats:
        return None
    comp = pd.concat(mats, axis=1).mean(axis=1, skipna=True).dropna()
    if len(comp) < 30:
        return None
    comp.name = sec
    return sec, comp, "+".join(hits)


# ------------------------------------------------------------------ 小工具
def _to_date_str(d) -> str:
    ts = pd.Timestamp(d)
    return ts.strftime("%Y-%m-%d")


def _to_ret_series(x) -> Optional[pd.Series]:
    """把 Series/单列 DataFrame/ndarray 统一成 float Series（DatetimeIndex, 升序, 无重复）。"""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return None
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == 0:
            return None
        s = pd.Series(arr)
    try:
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    except (TypeError, ValueError):
        return None
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _align(sector_ret, market_ret) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """按日期内连接并剔除 NaN。任一侧不可用返回 None（调用方优雅降级）。"""
    s = _to_ret_series(sector_ret)
    m = _to_ret_series(market_ret)
    if s is None or m is None or len(s) == 0 or len(m) == 0:
        return None
    idx = s.index.intersection(m.index)
    if len(idx) < 30:
        return None
    idx = idx.sort_values()
    sv = s.reindex(idx).to_numpy(dtype=float)
    mv = m.reindex(idx).to_numpy(dtype=float)
    ok = np.isfinite(sv) & np.isfinite(mv)
    if ok.sum() < 30:
        return None
    return sv[ok], mv[ok], idx[ok]


def _pos_on_or_after(idx: pd.DatetimeIndex, d, lag: int = 0) -> Optional[int]:
    """事件日 -> 索引位置（事件日或其后第一个交易日）。越界返回 None。"""
    ts = pd.Timestamp(d).normalize()
    pos = int(idx.searchsorted(ts, side="left"))
    pos += int(lag)
    if pos < 0 or pos >= len(idx):
        return None
    return pos


def _ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, int]:
    """y = a + b*x 的最小二乘。返回 (a, b, n)。样本不足或 x 无方差时返回 nan。"""
    n = int(np.isfinite(y).sum())
    if n < 3:
        return float("nan"), float("nan"), n
    X = np.column_stack([np.ones(n), x])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), n
    a, b = float(coef[0]), float(coef[1])
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan"), float("nan"), n
    return a, b, n


def _event_payload(sv: np.ndarray, mv: np.ndarray, pos: int,
                   window: Tuple[int, int], est_window: Tuple[int, int],
                   min_est: int = MIN_EST,
                   max_off: Optional[int] = None) -> Optional[Dict]:
    """
    单个事件的市场模型 + 异常收益路径。
    AR 覆盖 [window0, max(window1, max_off)]，因此任意 h <= max_off 的 CAR(0,h) 都能取到。
    """
    T = len(sv)
    e0, e1 = int(est_window[0]), int(est_window[1])
    lo, hi = pos + e0, pos + e1
    if lo < 0 or hi >= T or hi <= lo:
        return None
    ys, xs = sv[lo:hi + 1], mv[lo:hi + 1]
    a, b, n = _ols(ys, xs)
    if not np.isfinite(a) or not np.isfinite(b) or n < min_est:
        return None
    w0, w1 = int(window[0]), int(window[1])
    if max_off is not None:
        w1 = max(w1, int(max_off))
    ar: Dict[int, float] = {}
    missing = 0
    for off in range(w0, w1 + 1):
        p = pos + off
        if p < 0 or p >= T:
            missing += 1
            continue
        ar[off] = float(sv[p] - (a + b * mv[p]))
    if not ar:
        return None
    return {"alpha": a, "beta": b, "n_est": n, "ar": ar, "missing": missing}


def _sum_ar(ar: Dict[int, float], o0: int, o1: int) -> Tuple[float, int]:
    vals = [ar[o] for o in range(o0, o1 + 1) if o in ar]
    if not vals:
        return float("nan"), 0
    return float(np.sum(vals)), len(vals)


def _agg(vals: Sequence[float], lags: int = 0, min_n: int = MIN_N) -> Dict:
    """一组 CAR 的横截面统计。n < min_n -> insufficient，仍给点估计但明确标注。"""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = int(v.size)
    out = {"n": n, "mean": float("nan"), "median": float("nan"), "se": float("nan"),
           "t": float("nan"), "p": float("nan"), "lo": float("nan"), "hi": float("nan"),
           "win_rate": float("nan"), "min": float("nan"), "max": float("nan"),
           "insufficient": bool(n < min_n)}
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    out["median"] = float(np.median(v))
    out["min"] = float(v.min())
    out["max"] = float(v.max())
    out["win_rate"] = float((v > 0).mean())
    if n >= 3:
        st = mean_tstat(v, lags=int(max(0, lags)))
        for k in ("se", "t", "p", "lo", "hi"):
            out[k] = st[k]
    return out


def _auto_lags(positions: Sequence[int], span: int) -> int:
    """事件日重叠时用 HAC（lags=窗口跨度），不重叠则 lags=0。"""
    p = sorted(int(x) for x in positions)
    if len(p) < 2:
        return 0
    gap = min(b - a for a, b in zip(p[:-1], p[1:]))
    return int(max(0, span)) if gap <= max(span, 1) else 0


# ------------------------------------------------------------------ 事件研究
def event_study(sector_ret, market_ret, event_dates: Sequence,
                window: Tuple[int, int] = (-5, 10),
                est_window: Tuple[int, int] = (-120, -20),
                lag: int = 0, min_est: int = MIN_EST,
                lags: Optional[int] = None, min_n: int = MIN_N) -> Dict:
    """
    单个板块的事件研究。

    参数
    ----
    sector_ret : Series  板块日收益（小数，DatetimeIndex）
    market_ret : Series  市场日收益（基准指数，小数）
    event_dates: 事件日序列（YYYY-MM-DD / Timestamp）。取"该日或其后的第一个交易日"。
    window     : (w0, w1) 事件窗口，默认 (-5, +10)。CAR 全窗 = CAR(-5,+10)。
    est_window : (e0, e1) 估参窗口，默认 (-120, -20)，**绝不含事件后**。
    lag        : 事件日再后移几天（盘后公告传 1）
    lags       : HAC 滞后；None=自动（事件日重叠则取窗口跨度）

    返回
    ----
    dict：
      status            ok | insufficient | degraded
      n_events/n_valid  输入事件数 / 成功估计数
      car               全窗 CAR 的横截面统计（mean/median/t/p/ci95/win_rate/n）
      pre_car           CAR(w0,-1)：事件前是否已 price-in
      post_car          CAR(0,w1)：事件后净反应
      price_in          {pre_mean, post_mean, same_sign, conflict, verdict}
      horizons          {h: CAR(0,h) 统计}，h ∈ 1,2,3,5,10,20（按窗口可得性）
      daily             逐日横截面 AR 与 HAC t 值
      per_event         每个事件的明细（alpha/beta/car/pre/post）
      warnings          降级原因（ASCII）
    """
    out: Dict = {
        "sector": getattr(sector_ret, "name", None),
        "window": (int(window[0]), int(window[1])),
        "est_window": (int(est_window[0]), int(est_window[1])),
        "n_events": len(list(event_dates)) if event_dates is not None else 0,
        "n_valid": 0, "status": "insufficient",
        "car": _agg([]), "pre_car": _agg([]), "post_car": _agg([]),
        "price_in": {}, "horizons": {}, "daily": [], "per_event": [],
        "warnings": [],
    }
    if not event_dates:
        out["warnings"].append("no event dates")
        return out
    al = _align(sector_ret, market_ret)
    if al is None:
        out["warnings"].append("sector_ret/market_ret unavailable or too short (<30 obs)")
        out["status"] = "degraded"
        return out
    sv, mv, idx = al
    if len(idx) < 200:
        out["warnings"].append(f"short history: {len(idx)} obs (<200), est window may fail")

    max_h = max([20] + [h for h in (1, 2, 3, 5, 10, 20) if h <= max(window[1], 20)])
    car_list, pre_list, post_list = [], [], []
    hz: Dict[int, List[float]] = {h: [] for h in (1, 2, 3, 5, 10, 20)}
    positions: List[int] = []
    daily_ar: Dict[int, List[float]] = {off: [] for off in range(int(window[0]), int(window[1]) + 1)}

    for d in event_dates:
        pos = _pos_on_or_after(idx, d, lag=lag)
        if pos is None:
            out["warnings"].append(f"event {d} out of data range")
            continue
        pl = _event_payload(sv, mv, pos, window, est_window, min_est, max_off=max_h)
        if pl is None:
            out["warnings"].append(f"event {d}: estimation window unusable")
            continue
        positions.append(pos)
        ar = pl["ar"]
        car, ncar = _sum_ar(ar, int(window[0]), int(window[1]))
        pre, _ = _sum_ar(ar, int(window[0]), -1)
        post, _ = _sum_ar(ar, 0, int(window[1]))
        if np.isfinite(car):
            car_list.append(car)
        if np.isfinite(pre):
            pre_list.append(pre)
        if np.isfinite(post):
            post_list.append(post)
        for h in hz:
            v, _ = _sum_ar(ar, 0, h)
            if np.isfinite(v):
                hz[h].append(v)
        for off in daily_ar:
            if off in ar:
                daily_ar[off].append(ar[off])
        out["per_event"].append({
            "date": str(pd.Timestamp(d).strftime("%Y-%m-%d")),
            "trade_date": str(idx[pos].strftime("%Y-%m-%d")),
            "pos": int(pos), "alpha": pl["alpha"], "beta": pl["beta"],
            "n_est": pl["n_est"], "car": None if not np.isfinite(car) else car,
            "pre_car": None if not np.isfinite(pre) else pre,
            "post_car": None if not np.isfinite(post) else post,
            "n_ar": int(ncar), "missing": pl["missing"],
        })

    nv = len(positions)
    out["n_valid"] = nv
    if nv == 0:
        out["status"] = "degraded"
        return out

    L = int(lags) if lags is not None else _auto_lags(positions, window[1] - window[0])
    out["lags"] = L
    out["car"] = _agg(car_list, lags=L, min_n=min_n)
    out["pre_car"] = _agg(pre_list, lags=0, min_n=min_n)
    out["post_car"] = _agg(post_list, lags=L, min_n=min_n)
    out["horizons"] = {h: _agg(v, lags=(h - 1 if h > 1 else 0), min_n=min_n)
                       for h, v in hz.items() if v}
    out["daily"] = [dict({"offset": off}, **_agg(v, lags=0, min_n=min_n))
                    for off, v in sorted(daily_ar.items()) if v]
    out["price_in"] = _price_in(out["pre_car"], out["post_car"])
    out["status"] = "ok" if (nv >= min_n and not out["car"]["insufficient"]) else "insufficient"
    if out["status"] == "insufficient":
        out["warnings"].append(f"n_valid={nv} < min_n={min_n}: point estimate is not evidence")
    return out


def _price_in(pre: Dict, post: Dict) -> Dict:
    """事件前是否已 price-in：CAR(-5,-1) 与 CAR(0,+h) 反号 -> conflict。"""
    pm, sm = pre.get("mean", float("nan")), post.get("mean", float("nan"))
    same = bool(np.isfinite(pm) and np.isfinite(sm) and (pm == 0 or sm == 0 or np.sign(pm) == np.sign(sm)))
    conflict = bool(np.isfinite(pm) and np.isfinite(sm)
                    and pm != 0 and sm != 0 and np.sign(pm) != np.sign(sm))
    enough = bool(pre.get("n", 0) >= MIN_N and post.get("n", 0) >= MIN_N)
    if not enough:
        verdict = "insufficient"
    elif conflict:
        verdict = "conflict_pre_priced"
    elif np.isfinite(pm) and abs(pm) > 0.01 and same:
        verdict = "leaked_before_event"
    elif np.isfinite(sm) and abs(sm) < 0.005:
        verdict = "no_reaction"
    else:
        verdict = "clean"
    return {"pre_mean": pm, "post_mean": sm, "same_sign": same,
            "conflict": conflict, "verdict": verdict, "n_pre": pre.get("n", 0),
            "n_post": post.get("n", 0)}


# ------------------------------------------------------------------ 影响因子表
def sector_direction(ev: Event, sector: str) -> int:
    """板块方向：优先传导表（同一政策对不同板块方向可以相反），否则用事件整体方向。"""
    for s, d, _logic in POLICY_TRANSMISSION.get(ev.subtype, []):
        if s == sector:
            return int(d)
    return int(ev.direction)


def impact_factor(events: Sequence[Event], sector_ret, market_ret,
                  horizons: Sequence[int] = (1, 5, 20),
                  window: Tuple[int, int] = (-5, 10),
                  est_window: Tuple[int, int] = (-120, -20),
                  lag: int = 0, min_n: int = MIN_N,
                  pre_window: Tuple[int, int] = (-5, -1),
                  as_of: Optional[str] = None) -> pd.DataFrame:
    """
    影响因子估计：按 (type, sector, horizon) 聚合 direction-signed CAR。

    口径（关键）
    -----------
    * 对每个 (事件, 板块)：`signed_car = direction * CAR(0,h)`，
      direction 取 `sector_direction()`（传导表优先）。于是
      `mean_car > 0` 表示"按该事件的方向押注，平均赚"。
    * `mean_car_raw` 是不乘方向的原始 CAR（用于看绝对幅度与符号）。
    * direction == 0 的事件**不入表**（没有方向假设就无法定义 signed CAR），
      只在 warnings 里计数。
    * 每个事件用**自己的** `Event.lag`（盘后公告=1）；传进来的 `lag` 只作为
      事件未指定时的默认值。否则一批"周末公告 + 盘后公告"混在一起会错位一天。
    * 同一 (type, sector, horizon) 内事件日可能重叠 -> HAC lags = h-1。
    * n < min_n 的行仍保留但 `insufficient=True`。

    as_of : 点对点截断日。给定时**只用 date <= as_of 的事件**估计，
            用于"站在 t 日只拿历史事件建表"（否则全样本表就是未来函数，见 SPEC §0.0）。
    sector_ret : dict[str, Series] 或单个 Series（单板块场景）
                 字典的键是**实际行业字段**（东财 f100，如 银行Ⅱ/证券Ⅱ/化学制药）；
                 事件里的通用板块名（银行/券商/医药）经 `_resolve_series` 模糊映射，
                 命中多个行业时按**等权平均**合成一条序列（口径见 expand_sector）。
    返回 DataFrame，列：
      type, sector, horizon, n, mean_car, mean_car_raw, median_car, se, t, p,
      ci_lo, ci_hi, win_rate, mean_pre_car, conflict_rate, stability, n_years,
      insufficient, level  (level: type_sector | type_all)
    """
    warnings: List[str] = []
    evs = list(events or [])
    if as_of is not None:
        cut = _to_date_str(as_of)
        n0 = len(evs)
        evs = [e for e in evs if e.date <= cut]
        warnings.append(f"as_of={cut}: {n0 - len(evs)} future events dropped (PIT)")
    if not evs:
        df0 = _empty_factor_table()
        df0.attrs.update({"warnings": ["no events"], "n_skip_direction": 0,
                          "n_skip_data": 0, "n_events": 0})
        return df0

    single = not isinstance(sector_ret, dict)
    cache: Dict[Tuple[str, str], Optional[Dict]] = {}
    rows: List[Dict] = []
    n_skip_dir = n_skip_data = 0
    max_h = int(max(list(horizons) + [1]))

    for ev in evs:
        ev_lag = int(getattr(ev, "lag", 0) or lag)
        secs = list(ev.sectors) if ev.sectors else []
        if not secs and single:
            secs = [str(getattr(sector_ret, "name", "SECTOR") or "SECTOR")]
        if not secs:
            n_skip_data += 1
            warnings.append(f"{ev.eid}: no sectors and no single-sector series")
            continue
        for sec in secs:
            d = sector_direction(ev, sec)
            if d == 0:
                n_skip_dir += 1
                continue
            if single:
                ser, used = sector_ret, str(getattr(sector_ret, "name", sec) or sec)
            else:
                res = _resolve_series(sec, sector_ret)
                if res is None:
                    n_skip_data += 1
                    warnings.append(f"{ev.eid}/{sec}: sector series missing "
                                    f"(no exact/alias match in sector_ret keys)")
                    continue
                _rk, ser, used = res
            key = (sec, ev.date, ev_lag)
            if key not in cache:
                al = _align(ser, market_ret)
                if al is None:
                    cache[key] = None
                else:
                    sv, mv, idx = al
                    pos = _pos_on_or_after(idx, ev.date, lag=ev_lag)
                    cache[key] = None if pos is None else _event_payload(
                        sv, mv, pos, window, est_window, MIN_EST, max_off=max_h)
                if cache[key] is None:
                    warnings.append(f"{ev.eid}/{sec}: no usable window")
            pl = cache[key]
            if pl is None:
                n_skip_data += 1
                continue
            ar = pl["ar"]
            pre, _ = _sum_ar(ar, int(pre_window[0]), int(pre_window[1]))
            for h in horizons:
                car, nobs = _sum_ar(ar, 0, int(h))
                if not np.isfinite(car) or nobs < int(h) + 1:
                    continue
                rows.append({
                    "type": ev.type, "subtype": ev.subtype, "sector": sec,
                    "horizon": int(h), "eid": ev.eid, "date": ev.date,
                    "year": ev.year, "direction": d,
                    "car": float(car), "signed_car": float(d * car), "series": used,
                    "pre_car": float(pre) if np.isfinite(pre) else float("nan"),
                })

    if not rows:
        warnings.append(f"no usable (event, sector) rows (skip_dir={n_skip_dir}, "
                        f"skip_data={n_skip_data})")
        df = _empty_factor_table()
        df.attrs.update({"warnings": warnings, "n_skip_direction": n_skip_dir,
                         "n_skip_data": n_skip_data, "n_events": len(evs)})
        return df

    raw = pd.DataFrame(rows)
    recs: List[Dict] = []
    for (typ, sec, h), g in raw.groupby(["type", "sector", "horizon"], sort=True):
        recs.append(_factor_row(g, typ, sec, int(h), min_n, level="type_sector"))
    for (typ, h), g in raw.groupby(["type", "horizon"], sort=True):
        recs.append(_factor_row(g, typ, "*", int(h), min_n, level="type_all"))
    df = pd.DataFrame(recs)
    df = df.sort_values(["level", "type", "sector", "horizon"],
                        ascending=[False, True, True, True]).reset_index(drop=True)
    df.attrs["warnings"] = warnings
    df.attrs["n_skip_direction"] = n_skip_dir
    df.attrs["n_skip_data"] = n_skip_data
    df.attrs["n_events"] = len(evs)
    return df


def _factor_row(g: pd.DataFrame, typ: str, sec: str, h: int,
                min_n: int, level: str) -> Dict:
    v = g["signed_car"].to_numpy(dtype=float)
    st = _agg(v, lags=max(h - 1, 0), min_n=min_n)
    pre = g["pre_car"].to_numpy(dtype=float)
    pre_ok = pre[np.isfinite(pre)]
    # 分年度稳定性：符号一致年份占比（至少 2 个年份才计算）
    stab = float("nan")
    n_years = 0
    if len(g) > 0:
        ym = g.groupby("year")["signed_car"].mean()
        n_years = int(ym.size)
        if n_years >= 2 and np.isfinite(st["mean"]) and st["mean"] != 0:
            stab = float((np.sign(ym.to_numpy()) == np.sign(st["mean"])).mean())
    conflict = float("nan")
    if len(pre_ok) and np.isfinite(st["mean"]) and st["mean"] != 0:
        conflict = float((np.sign(pre_ok) != np.sign(st["mean"])).mean())
    series_used = ("+".join(sorted(set(g["series"]))) if "series" in g.columns else "")
    return {
        "type": typ, "sector": sec, "horizon": int(h), "n": st["n"],
        "series": series_used,
        "mean_car": st["mean"], "mean_car_raw": float(g["car"].mean()),
        "median_car": st["median"], "se": st["se"], "t": st["t"], "p": st["p"],
        "ci_lo": st["lo"], "ci_hi": st["hi"], "win_rate": st["win_rate"],
        "mean_pre_car": float(pre_ok.mean()) if pre_ok.size else float("nan"),
        "conflict_rate": conflict, "stability": stab, "n_years": n_years,
        "insufficient": bool(st["insufficient"]), "level": level,
    }


def _empty_factor_table() -> pd.DataFrame:
    cols = ["type", "sector", "horizon", "n", "series", "mean_car", "mean_car_raw", "median_car",
            "se", "t", "p", "ci_lo", "ci_hi", "win_rate", "mean_pre_car",
            "conflict_rate", "stability", "n_years", "insufficient", "level"]
    return pd.DataFrame(columns=cols)


# ------------------------------------------------------------------ 预测
def _confidence(n: int, t: float, mean: float, min_n: int = MIN_N) -> float:
    """
    置信度 = 样本量因子 × t 因子，∈[0,1]。刻意保守（宁可说"不知道"）：
      n_factor = log1p(n)/log1p(30) 截断到 1（n=5 -> 0.52，n=30 -> 1）
      t_factor = clip((|t|-1)/2, 0, 1)  （|t|<=1 -> 0，|t|=2 -> 0.5，|t|>=3 -> 1）
      n < min_n -> 0（样本不够就不给置信度）
    之所以不用 Φ(|t|)：那会给出 t=0.8 -> 0.79 的"高置信"，与"不显著"自相矛盾。
    """
    if n < min_n or not np.isfinite(t) or not np.isfinite(mean):
        return 0.0
    n_f = min(1.0, math.log1p(n) / math.log1p(30))
    t_f = float(min(1.0, max(0.0, (abs(t) - 1.0) / 2.0)))
    return float(min(1.0, max(0.0, n_f * t_f)))


def _lookup(table: pd.DataFrame, typ: str, sec: str, h: int) -> Optional[pd.Series]:
    if table is None or len(table) == 0:
        return None
    m = table[(table["type"] == typ) & (table["sector"] == sec) & (table["horizon"] == h)]
    if len(m) == 0:
        m = table[(table["type"] == typ) & (table["sector"] == "*") & (table["horizon"] == h)]
    if len(m) == 0:
        return None
    return m.iloc[0]


def predict_impact(event: Event, table: pd.DataFrame, horizon: int = 5,
                   min_n: int = MIN_N) -> Dict:
    """
    新事件 -> 查影响因子表 -> 预测。

    expected_car 是**该板块的实际预期收益**（已乘事件方向）：
      expected_car = direction * mean_car(direction-signed)
    查不到历史样本时返回 expected_car=None 且 reason='no_history'，绝不外推。
    """
    out: Dict = {
        "eid": event.eid, "date": event.date, "type": event.type,
        "subtype": event.subtype, "scope": event.scope,
        "evidence_level": event.evidence_level,
        "source_reliability": event.source_reliability,
        "horizon": int(horizon), "sectors": [],
        "catalyst_clock": {"window": int(event.catalyst_window),
                           "peak_h": None, "curve": {}},
        "warnings": [],
    }
    secs = list(event.sectors) if event.sectors else []
    if not secs:
        out["warnings"].append("event has no sectors: nothing to predict")
        return out
    peak: Dict[int, float] = {}
    for sec in secs:
        d = sector_direction(event, sec)
        row = _lookup(table, event.type, sec, int(horizon))
        if row is None:
            out["sectors"].append({
                "sector": sec, "direction": d, "expected_car": None,
                "expected_car_signed": None, "confidence": 0.0, "n": 0,
                "t": float("nan"), "insufficient": True,
                "source_level": None, "reason": "no_history"})
            continue
        n = int(row["n"])
        mean_signed = float(row["mean_car"]) if pd.notna(row["mean_car"]) else float("nan")
        t = float(row["t"]) if pd.notna(row["t"]) else float("nan")
        insuff = bool(row["insufficient"]) or n < max(int(min_n), MIN_TRADES)
        conf = _confidence(n, t, mean_signed, min_n=min_n)
        # 样本不足时**不给点预测**（"不硬报"）：只保留统计量供人看
        expected = (None if (insuff or d == 0 or not np.isfinite(mean_signed))
                    else float(d * mean_signed))
        reason = "ok"
        if insuff:
            reason = "insufficient_samples"
        elif d == 0:
            reason = "no_direction"
        elif not np.isfinite(t) or abs(t) < 2.0:
            reason = "weak_t"
        out["sectors"].append({
            "sector": sec, "direction": d, "expected_car": expected,
            "expected_car_signed": None if not np.isfinite(mean_signed) else mean_signed,
            "confidence": conf, "n": n, "t": t, "insufficient": insuff,
            "source_level": str(row["level"]), "reason": reason,
            "ci95": [None if pd.isna(row["ci_lo"]) else float(row["ci_lo"]),
                     None if pd.isna(row["ci_hi"]) else float(row["ci_hi"])],
            "stability": None if pd.isna(row["stability"]) else float(row["stability"]),
        })
        # 催化时钟：用同 (type, sector) 各 horizon 的 signed CAR 拼一条粗曲线
        sub = table[(table["type"] == event.type) & (table["sector"] == sec)]
        for _, r in sub.iterrows():
            if pd.notna(r["mean_car"]):
                peak[int(r["horizon"])] = float(r["mean_car"])
    if peak:
        hh = max(peak, key=lambda k: abs(peak[k]))
        out["catalyst_clock"]["peak_h"] = int(hh)
        out["catalyst_clock"]["curve"] = {int(k): round(float(v), 6) for k, v in sorted(peak.items())}
        if event.catalyst_window < hh:
            out["warnings"].append(
                f"catalyst_window={event.catalyst_window} < peak_h={hh}")
    confs = [s["confidence"] for s in out["sectors"]]
    out["confidence_max"] = max(confs) if confs else 0.0
    out["has_history"] = any(s["reason"] != "no_history" for s in out["sectors"])
    if not out["has_history"]:
        out["warnings"].append("no historical samples for this (type, sector): no forecast")
    if event.evidence_level in ("weak", "conflict"):
        out["warnings"].append(f"evidence_level={event.evidence_level}: treat as hypothesis only")
    return out


# ------------------------------------------------------------------ 持久化
def load_events(path: str = EVENTS_PATH) -> List[Event]:
    """读 JSONL。文件不存在返回 []（不抛异常）。坏行跳过并计数。"""
    if not path or not os.path.exists(path):
        return []
    out: List[Event] = []
    fields = set(Event.__dataclass_fields__.keys())
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            kw = {k: v for k, v in d.items() if k in fields}
            if "eid" not in kw or "date" not in kw or "type" not in kw:
                continue
            try:
                out.append(Event(**kw))
            except TypeError:
                continue
    return out


def save_event(ev: Event, path: str = EVENTS_PATH, append: bool = True) -> str:
    """
    追加一条事件到 JSONL（只追加，不改历史 —— 与预测台账同一条铁律）。
    返回写入的绝对路径。
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(ev.as_dict(), ensure_ascii=False) + "\n")
    return os.path.abspath(path)


def save_events(evs: Sequence[Event], path: str = EVENTS_PATH) -> str:
    """批量写（覆盖）。日历重建用；回测用的事件流请走 save_event 追加。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in evs:
            f.write(json.dumps(ev.as_dict(), ensure_ascii=False) + "\n")
    return os.path.abspath(path)


# ------------------------------------------------------------------ 板块收益
def fetch_board_returns(board_code: str, beg: str = "20100101",
                        end: str = "20500101") -> pd.Series:
    """
    东财行业板块日线 -> 板块日收益（小数）。board_code 形如 'BK0475'。
    走 qcore.data 的 _get_json（纯标准库 urllib），无网络/被限流时抛异常，
    由调用方降级到 build_sector_returns。
    """
    from qcore.data import KLINE_COLS, KLINE_FIELDS, _get_json  # 局部导入，避免硬依赖

    code = board_code.upper().replace("90.", "")
    if not code.startswith("BK"):
        code = f"BK{code}"
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid=90.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2={KLINE_FIELDS}"
           f"&klt=101&fqt=1&beg={beg}&end={end}&lmt=100000")
    d = _get_json(url)
    data = d.get("data") or {}
    rows = [ln.split(",") for ln in (data.get("klines") or [])]
    if not rows:
        raise RuntimeError(f"no board kline for {board_code}")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    for c in KLINE_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    s = df.set_index("date")["close"].pct_change()
    s.name = code
    return s.dropna()


def build_sector_returns(panel: Dict[str, pd.DataFrame], pool: pd.DataFrame,
                         min_names: int = 3, industry_col: str = "industry",
                         price_col: str = "close") -> Dict[str, pd.Series]:
    """
    ⚠️ 近似口径：用 `qcore.data.get_pool()` 的 industry 字段把个股聚合成
    **行业等权日收益指数**（每日等权，不调权、不剔除当日停牌外的其他偏差）。

    与东财行业板块指数的差异：
      1) 板块指数是市值加权（部分为自由流通加权），等权会放大小盘弹性；
      2) 板块指数含全部成分股，pool 只含 top_n（大市值），行业覆盖偏头部；
      3) 成分股调整时点不同 -> 长周期上会累积偏离。
    因此用本函数估计的 CAR 只能用于**方向与量级的粗判断**，
    正式回测请优先 fetch_board_returns。

    返回 {行业名: Series(日收益小数)}，样本数 < min_names 的行业丢弃。
    """
    if panel is None or pool is None or len(pool) == 0:
        return {}
    if industry_col not in pool.columns:
        return {}
    ind = dict(zip(pool["code"].astype(str).str.zfill(6),
                   pool[industry_col].astype(str)))
    buckets: Dict[str, List[pd.Series]] = {}
    for code, df in panel.items():
        c = str(code).zfill(6)
        if c not in ind or df is None or len(df) < 30:
            continue
        if price_col not in df.columns or "date" not in df.columns:
            continue
        s = pd.Series(pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float),
                      index=pd.DatetimeIndex(pd.to_datetime(df["date"]))).sort_index()
        r = s.pct_change()
        r = r[np.isfinite(r)]
        if len(r) < 30:
            continue
        buckets.setdefault(ind[c], []).append(r)
    out: Dict[str, pd.Series] = {}
    for k, lst in buckets.items():
        if len(lst) < min_names:
            continue
        mat = pd.concat(lst, axis=1)
        out[k] = mat.mean(axis=1, skipna=True).dropna()
        out[k].name = k
    return out


__all__ = [
    "Event", "EVENT_TYPES", "EVENT_TYPE_CN", "SCOPES", "EVIDENCE_LEVELS",
    "POLICY_TRANSMISSION", "SECTOR_ALIASES", "MIN_N",
    "validate_event", "expand_sector", "sector_direction",
    "event_study", "impact_factor", "predict_impact",
    "load_events", "save_event", "save_events",
    "fetch_board_returns", "build_sector_returns",
]
