"""
A 股政策事件日历（2010-2026）—— 政策层回测的骨架
=================================================

**真实性声明（读这一节再读代码）**
本文件的每一条 `date` 都来自该条 `source` 页面上**自带的日期**，不是凭印象写的：
  * 中国人民银行公告 -> `pbc.gov.cn` 正文页 `<meta name="PubDate">`（公告日，非生效日）
  * 国务院/国办/部委文件 -> `gov.cn` 政策文件库条目（`pubtimeStr`，即发布日）
  * 新华社/中国政府网新闻 -> `gov.cn` 新闻页首行时间戳
凡是找不到"页面自带日期 == 事件日期"的候选事件，**一律不写**（见文件末尾 DELETED 清单）。
宁可日历短，也不要一条假日期把整个政策层的回测结论带偏。

**口径（三条，别改）**
1. `date` = **公告/发布日**（可核事实）。
2. `lag`  = 公告日到**首个可交易日**的偏移：盘后/收盘后发布的公告 `lag=1`，
   非交易日（周末/节假日）发布的 `lag=0`（引擎会自动滚到下一个交易日），
   盘中/盘前发布 `lag=0`。**每个事件单独标**，因为混用会错位一天。
3. `direction` = 该事件对 `sectors` 里板块的方向；同一条政策对不同板块方向可以相反，
   所以 `qcore.events.sector_direction()` 会优先查 `POLICY_TRANSMISSION`。

**板块名**
用通用行业名（银行/地产/券商/医药/半导体...），落地时用
`qcore.events.expand_sector(name, pool['industry'])` 模糊映射到东财行业字段。
少数事件天然作用于风格/宽基（沪深300/次新股），这些**不是**行业板块，
需要单独准备指数序列（见 `qcore.events.build_sector_returns` 的说明）。

**已知缺口（不要以为是漏了）**
* 2017 年：再融资新规（2/17）与减持新规（5/27）都很重要，但 gov.cn 上只有
  《国务院公报》转载页（公报日期 2017-09-30 / 2017-11-30），拿不到"公告日 == 页面日期"
  的原文页，按上面第 1 条口径只能留空。
* 2019-08-17 LPR 形成机制改革、2013-07-19 全面放开贷款利率管制：央行站内检索
  拿不到对应公告页，删除。
* 2020-08-20 "三道红线"：只有座谈会报道、无公开文件，删除。
* 2015-09-07 股指期货管控、2016-01-07 熔断暂停：交易所/证监会公告页未核实，删除。
"""
from __future__ import annotations

from typing import Dict, List

from qcore.events import Event

# ---------------------------------------------------------------------------
# 来源前缀（便于核对；完整 URL 写在每条里）
#   PBC  = 中国人民银行 www.pbc.gov.cn
#   GOV  = 中国政府网 www.gov.cn
# ---------------------------------------------------------------------------

CALENDAR: List[Event] = [
    # ==================== 2010 ====================
    Event(eid="P20100112", date="2010-01-12", type="policy", subtype="rrr_hike", lag=1,
          title="央行上调存款准备金率0.5个百分点（本轮紧缩周期首次提准，1月18日实施）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125434/125798/2843677/index.html",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="基线：2009年信贷狂飙后市场已知政策将转向；增量=时点早于预期。"
               "首个验证变量=1月新增信贷与银行间回购利率；失效=信贷不降则紧缩继续加码"),
    Event(eid="P20100417", date="2010-04-17", type="policy", subtype="property_tighten", lag=0,
          title="国务院《关于坚决遏制部分城市房价过快上涨的通知》（国发〔2010〕10号，新国十条）",
          source="https://www.gov.cn/zhengce/zhengceku/2010-04/17/content_4639.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["地产", "银行", "建筑材料", "家用电器"], evidence_level="strong",
          catalyst_window=40,
          note="首个全国性差别化信贷+限购框架；失效条件=地方细则打折执行、成交未降"),
    Event(eid="P20101019", date="2010-10-19", type="policy", subtype="rate_hike", lag=1,
          title="央行上调存贷款基准利率0.25个百分点（2007年以来首次加息）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2848542/index.html",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["银行", "地产", "券商", "公用事业"], evidence_level="strong",
          catalyst_window=20,
          note="加息周期开启的确认信号；银行方向存在'息差扩张 vs 估值压缩'冲突，属可证伪点"),
    Event(eid="P20101225", date="2010-12-25", type="policy", subtype="rate_hike", lag=0,
          title="央行年内第二次加息0.25个百分点（12月26日起）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2881546/index.html",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="周末公告，反应落在12月27日（周一）"),

    # ==================== 2011 ====================
    Event(eid="P20110127", date="2011-01-27", type="policy", subtype="property_tighten", lag=0,
          title="国务院办公厅《关于进一步做好房地产市场调控工作有关问题的通知》（国办发〔2011〕1号，新国八条）",
          source="https://www.gov.cn/zhengce/zhengceku/2011-01/27/content_4593.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["地产", "银行", "建筑材料"], evidence_level="strong",
          catalyst_window=40,
          note="二套房首付60%、房产税试点扩容、限购扩面；增量=行政手段升级"),
    Event(eid="P20110208", date="2011-02-08", type="policy", subtype="rate_hike", lag=0,
          title="央行上调存贷款基准利率0.25个百分点（春节假期最后一日公告）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2850197/index.html",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="公告日为春节假期（非交易日），反应落在2月9日开市"),
    Event(eid="P20110405", date="2011-04-05", type="policy", subtype="rate_hike", lag=0,
          title="央行上调存贷款基准利率0.25个百分点（清明节假期公告）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2876411/index.html",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20, note="非交易日公告，反应落在4月6日"),
    Event(eid="P20110706", date="2011-07-06", type="policy", subtype="rate_hike", lag=1,
          title="央行年内第三次加息0.25个百分点（本轮最后一次加息）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2852986/index.html",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20, note="通胀顶+加息末段的典型'利空出尽'候选，可检验方向反转"),
    Event(eid="P20111130", date="2011-11-30", type="policy", subtype="rrr_cut", lag=1,
          title="央行下调存款准备金率0.5个百分点（2008年以来首次降准，12月5日实施）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125434/125798/2855694/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="货币政策转向的标志；当日恰逢全球六大央行联合提供美元流动性"),

    # ==================== 2012 ====================
    Event(eid="P20120607", date="2012-06-07", type="policy", subtype="rate_cut", lag=1,
          title="央行降息0.25个百分点并扩大存贷款利率浮动区间（利率市场化实质一步）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2861207/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["地产", "券商", "银行", "公用事业"], evidence_level="strong",
          catalyst_window=20,
          note="存款利率上浮10%，银行息差双向受压；地产是当年最直接受益板块"),
    Event(eid="P20120705", date="2012-07-05", type="policy", subtype="rate_cut", lag=1,
          title="央行再次降息0.25个百分点（一个月内第二次，非对称）",
          source="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/125835/2862280/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["地产", "券商", "银行"], evidence_level="strong",
          catalyst_window=20, note="连续降息=稳增长压力显性化，需区分'宽松利多'与'需求差利空'"),

    # ==================== 2013 ====================
    Event(eid="P20130304", date="2013-03-04", type="policy", subtype="property_tighten", lag=0,
          title="国务院办公厅《关于继续做好房地产市场调控工作的通知》（国办发〔2013〕17号，新国五条）",
          source="https://www.gov.cn/zhengce/zhengceku/2013-03/04/content_4564.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["地产", "银行", "建筑材料"], evidence_level="strong",
          catalyst_window=40,
          note="3月1日印发、3月4日首个交易日；二手房转让征20%个税是最强增量"),

    # ==================== 2014 ====================
    Event(eid="P20141121", date="2014-11-21", type="policy", subtype="rate_cut", lag=1,
          title="央行非对称降息（贷款基准利率降0.4个百分点，存款降0.25，扩大浮动区间）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212540037549/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "地产", "银行", "建筑装饰"], evidence_level="strong",
          catalyst_window=40,
          note="2014-2015 大牛市的第一根火柴；券商板块随后两周涨幅居前"),

    # ==================== 2015 ====================
    Event(eid="P20150228", date="2015-02-28", type="policy", subtype="rate_cut", lag=0,
          title="央行降息0.25个百分点并扩大存款利率浮动区间（3月1日起）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212541969623/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "地产", "银行"], evidence_level="strong",
          catalyst_window=20, note="周末公告，反应落在3月2日"),
    Event(eid="G20150228", date="2015-02-28", type="policy", subtype="collection_bid", lag=0,
          title="国务院办公厅《关于完善公立医院药品集中采购工作的指导意见》（国办发〔2015〕7号）",
          source="https://www.gov.cn/zhengce/zhengceku/2015-02/28/content_9502.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["医药"], evidence_level="strong",
          catalyst_window=40,
          note="省级招标+分类采购框架，是后续集采的制度起点；首个验证变量=中标价降幅"),
    Event(eid="P20150510", date="2015-05-10", type="policy", subtype="rate_cut", lag=0,
          title="央行降息0.25个百分点（5月11日起）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212540210029/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "地产", "银行"], evidence_level="strong",
          catalyst_window=20, note="周日公告，反应落在5月11日"),
    Event(eid="P20150627", date="2015-06-27", type="policy", subtype="rate_cut", lag=0,
          title="央行定向降准并同时降息0.25个百分点（股灾首轮暴跌后的周末宽松）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212541919056/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "银行", "地产"], evidence_level="strong",
          catalyst_window=10,
          note="典型'政策底≠市场底'样本：6月29日高开后继续暴跌，可检验失效条件"),
    Event(eid="P20150825", date="2015-08-25", type="policy", subtype="rate_cut", lag=1,
          title="央行降息0.25个百分点并降准0.5个百分点（全球黑色星期一之后）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212542946700/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "银行", "地产", "建筑装饰"], evidence_level="strong",
          catalyst_window=20,
          note="外部冲击下的对冲式宽松，方向可正可负，属高不确定样本"),
    Event(eid="P20151023", date="2015-10-23", type="policy", subtype="rate_cut", lag=1,
          title="央行双降（降息0.25+降准0.5）并放开存款利率浮动上限（利率市场化收官）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212543080959/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "银行", "地产"], evidence_level="strong",
          catalyst_window=20,
          note="存款利率上限放开对银行是长期利空、短期靴子落地，方向存在冲突"),

    # ==================== 2016 ====================
    Event(eid="P20160229", date="2016-02-29", type="policy", subtype="rrr_cut", lag=1,
          title="央行下调存款准备金率0.5个百分点（2016年唯一一次全面降准）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212543432205/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="熔断后的信心修复窗口；同日为2月最后一个交易日"),

    # ==================== 2018 ====================
    Event(eid="P20180417", date="2018-04-17", type="policy", subtype="rrr_cut", lag=1,
          title="央行定向降准1个百分点置换中期借贷便利（释放约4000亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212545070579/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="中美贸易摩擦升级背景下的对冲，需区分'宽松利多'与'风险偏好利空'"),
    Event(eid="P20181007", date="2018-10-07", type="policy", subtype="rrr_cut", lag=0,
          title="央行降准1个百分点置换中期借贷便利（国庆假期最后一日公告）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212545397257/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="假期公告，10月8日开盘反应；当日上证大跌3.7%，是'利好出尽/外围补跌'样本"),

    # ==================== 2019 ====================
    Event(eid="P20190104", date="2019-01-04", type="policy", subtype="rrr_cut", lag=1,
          title="央行全面降准1个百分点（分两次实施，释放约1.5万亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212545520568/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商", "建筑装饰"], evidence_level="strong",
          catalyst_window=20,
          note="2019年'春季躁动'起点；公告当晚美股大涨，属共振样本"),
    Event(eid="G20190117", date="2019-01-17", type="policy", subtype="collection_bid", lag=0,
          title="国务院办公厅《国家组织药品集中采购和使用试点方案》（国办发〔2019〕2号，4+7）",
          source="https://www.gov.cn/zhengce/zhengceku/2019-01/17/content_5358604.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["医药"], evidence_level="strong",
          catalyst_window=40,
          note="4+7 中标价平均降幅52%，是医药板块估值体系重估的起点"),
    Event(eid="P20190906", date="2019-09-06", type="policy", subtype="rrr_cut", lag=1,
          title="央行全面降准0.5个百分点（9月16日实施）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212550014257/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20, note="降准落地前市场已部分预期（国常会先吹风），可检验 price-in"),
    Event(eid="G20190930", date="2019-09-30", type="policy", subtype="collection_bid", lag=0,
          title="国家医保局等九部门《关于国家组织药品集中采购和使用试点扩大区域范围的意见》（医保发〔2019〕56号）",
          source="https://www.gov.cn/zhengce/zhengceku/2019-09/30/content_5456439.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["医药"], evidence_level="strong",
          catalyst_window=40, note="集采从4+7城市扩至全国，量价博弈升级"),

    # ==================== 2020 ====================
    Event(eid="P20200101", date="2020-01-01", type="policy", subtype="rrr_cut", lag=0,
          title="央行全面降准0.5个百分点（1月6日实施，释放约8000亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212550341272/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20, note="元旦公告，1月2日开盘反应"),
    Event(eid="G20200116", date="2020-01-16", type="policy", subtype="collection_bid", lag=0,
          title="国家医保局等五部门《关于开展第二批国家组织药品集中采购和使用的通知》（医保发〔2020〕2号）",
          source="https://www.gov.cn/zhengce/zhengceku/2020-01/16/content_5562282.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["医药"], evidence_level="strong",
          catalyst_window=40, note="33个品种、平均降幅53%，规则常态化"),
    Event(eid="G20200229", date="2020-02-29", type="policy", subtype="securities_law", lag=0,
          title="国务院办公厅《关于贯彻实施修订后的证券法有关工作的通知》（国办发〔2020〕5号）",
          source="https://www.gov.cn/zhengce/zhengceku/2020-02/29/content_5485074.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "次新股"], evidence_level="strong",
          catalyst_window=40,
          note="新证券法3月1日施行（注册制+严监管）；文件为配套落地安排"),

    # ==================== 2021 ====================
    Event(eid="G20210128", date="2021-01-28", type="policy", subtype="collection_bid", lag=0,
          title="国务院办公厅《关于推动药品集中带量采购工作常态化制度化开展的意见》（国办发〔2021〕2号）",
          source="https://www.gov.cn/zhengce/zhengceku/2021-01/28/content_5583305.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["医药"], evidence_level="strong",
          catalyst_window=40, note="集采制度化=降价预期长期化，压制仿制药估值中枢"),
    Event(eid="G20210707", date="2021-07-07", type="regulation", subtype="regulatory_tighten", lag=0,
          title="中办国办《关于依法从严打击证券违法活动的意见》发布",
          source="https://www.gov.cn/zhengce/2021-07/07/content_5622911.htm",
          source_reliability=1.0, scope="market", direction=-1,
          sectors=["券商", "计算机"], evidence_level="strong",
          catalyst_window=20,
          note="首个由中办国办联合发布的证券执法文件；对小盘题材股冲击大于指数"),
    Event(eid="P20210709", date="2021-07-09", type="policy", subtype="rrr_cut", lag=1,
          title="央行全面降准0.5个百分点（7月15日实施，释放约1万亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212551625265/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商", "建筑装饰"], evidence_level="strong",
          catalyst_window=20,
          note="国常会7月7日已提'降准'，属半 price-in 样本：可对比 CAR(-5,-1) 与 CAR(0,h)"),
    Event(eid="G20210724", date="2021-07-24", type="policy", subtype="education_tighten", lag=0,
          title="中办国办印发《关于进一步减轻义务教育阶段学生作业负担和校外培训负担的意见》（双减）",
          source="https://www.gov.cn/xinwen/2021-07/24/content_5627132.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["教育", "传媒"], evidence_level="strong",
          catalyst_window=40,
          note="学科类培训一律不得上市融资，商业模式直接失效；7月26日（周一）教育股腰斩"),
    Event(eid="G20211031", date="2021-10-31", type="policy", subtype="bse_support", lag=0,
          title="证监会发布《北京证券交易所向不特定合格投资者公开发行股票注册管理办法（试行）》",
          source="https://www.gov.cn/zhengce/zhengceku/2021-10/31/content_5647940.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "次新股"], evidence_level="strong",
          catalyst_window=20,
          note="北交所11月15日开市；周日发布，反应落在11月1日"),
    Event(eid="P20211206", date="2021-12-06", type="policy", subtype="rrr_cut", lag=1,
          title="央行全面降准0.5个百分点（12月15日实施，释放约1.2万亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212552059003/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="总理12月3日已提及'适时降准'，price-in 程度高"),

    # ==================== 2022 ====================
    Event(eid="P20220415", date="2022-04-15", type="policy", subtype="rrr_cut", lag=1,
          title="央行降准0.25个百分点（4月25日实施，释放约5300亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212552480200/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="国常会4月13日已预告；幅度低于预期，是'预期落空'样本"),
    Event(eid="P20221125", date="2022-11-25", type="policy", subtype="rrr_cut", lag=1,
          title="央行降准0.25个百分点（12月5日实施，释放约5000亿）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212552812807/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="与地产'三支箭'、防疫优化共振，事件窗口内混杂因素多"),

    # ==================== 2023 ====================
    Event(eid="G20230202", date="2023-02-02", type="policy", subtype="registration_reform", lag=0,
          title="全面实行股票发行注册制改革正式启动（证监会2月1日发布制度规则）",
          source="https://www.gov.cn/zhengce/2023-02/02/content_5739696.htm",
          source_reliability=1.0, scope="market", direction=1,
          sectors=["券商", "次新股"], evidence_level="strong",
          catalyst_window=40,
          note="主板注册制落地，券商投行逻辑兑现；次新股壳价值折价"),
    Event(eid="P20230317", date="2023-03-17", type="policy", subtype="rrr_cut", lag=1,
          title="央行降准0.25个百分点（3月27日实施）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212553138609/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="两会后落地，时点早于市场预期（原本预期4月）"),
    Event(eid="G20230827", date="2023-08-27", type="policy", subtype="stamp_tax_cut", lag=0,
          title="财政部税务总局公告2023年第39号：证券交易印花税减半征收（8月28日起）",
          source="https://www.gov.cn/zhengce/zhengceku/202308/content_6900443.htm",
          source_reliability=1.0, scope="market", direction=1,
          sectors=["券商", "计算机"], evidence_level="strong",
          catalyst_window=20,
          note="周日公告，8月28日大幅高开后全天回落（典型高开低走），是'情绪政策'教科书样本"),
    Event(eid="G20230901", date="2023-09-01", type="policy", subtype="bse_support", lag=0,
          title="证监会《关于高质量建设北京证券交易所的意见》（深改19条）",
          source="https://www.gov.cn/zhengce/zhengceku/202309/content_6901599.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "次新股"], evidence_level="strong",
          catalyst_window=20, note="北交所扩容与转板机制明确"),
    Event(eid="P20230914", date="2023-09-14", type="policy", subtype="rrr_cut", lag=1,
          title="央行降准0.25个百分点（9月15日实施）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212553497620/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20, note="降准当日MLF超额续作，属组合拳"),

    # ==================== 2024 ====================
    Event(eid="P20240124", date="2024-01-24", type="policy", subtype="rrr_cut", lag=0,
          title="央行降准0.5个百分点并下调支农支小再贷款、再贴现利率0.25个百分点",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212553729065/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商", "建筑装饰"], evidence_level="strong",
          catalyst_window=20,
          note="国新办发布会盘中宣布，当日上证涨1.8%、两日后继续上攻，是'盘中政策'样本"),
    Event(eid="G20240412", date="2024-04-12", type="policy", subtype="capital_market_reform", lag=1,
          title="国务院《关于加强监管防范风险推动资本市场高质量发展的若干意见》（国发〔2024〕10号，新国九条）",
          source="https://www.gov.cn/zhengce/zhengceku/202404/content_6944878.htm",
          source_reliability=1.0, scope="market", direction=1,
          sectors=["券商", "银行", "公用事业"], evidence_level="strong",
          catalyst_window=40,
          note="周五盘后发布，4月15-16日小盘股大跌、红利上涨：结构分化远大于指数方向"),
    Event(eid="G20240413", date="2024-04-13", type="policy", subtype="delisting_strict", lag=0,
          title="证监会《关于严格执行退市制度的意见》（配合新国九条）",
          source="https://www.gov.cn/zhengce/zhengceku/202404/content_6945065.htm",
          source_reliability=1.0, scope="market", direction=-1,
          sectors=["券商", "次新股"], evidence_level="strong",
          catalyst_window=40,
          note="退市标准收紧=壳价值折价；对小市值是直接利空（无纯行业代理，用次新股近似）"),
    Event(eid="P20240517", date="2024-05-17", type="policy", subtype="mortgage_rate_cut", lag=0,
          title="央行取消全国首套/二套房贷利率政策下限（同日首付比例降至15%、公积金利率降0.25）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554062364/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["地产", "建筑材料", "家用电器", "银行"], evidence_level="strong",
          catalyst_window=40,
          note="盘中发布（午间），当日地产链午后直线拉升，是最干净的'政策脉冲'样本"),
    Event(eid="P20240927", date="2024-09-27", type="policy", subtype="rrr_cut", lag=0,
          title="央行下调金融机构存款准备金率0.5个百分点（9·24一揽子政策落地）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554238383/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商"], evidence_level="strong",
          catalyst_window=20,
          note="9月24日已宣布、9月27日落地；真正的价格反应在9月24-30日，本条为执行公告"),
    Event(eid="P20241018", date="2024-10-18", type="policy", subtype="buyback_loan", lag=0,
          title="央行设立股票回购增持再贷款（首期3000亿）并正式启动证券基金保险公司互换便利（SFISF）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554316920/index.html",
          source_reliability=1.0, scope="market", direction=1,
          sectors=["券商", "银行"], evidence_level="strong",
          catalyst_window=20,
          note="金融街论坛盘中宣布；两项工具是9·24的落地，属'二次确认'事件"),
    Event(eid="G20241113", date="2024-11-13", type="policy", subtype="property_loosen", lag=0,
          title="财政部税务总局住建部公告2024年第16号：房地产税收新政（契税降至1%、取消普宅标准）",
          source="https://www.gov.cn/zhengce/zhengceku/202411/content_6986750.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["地产", "建筑材料", "家用电器"], evidence_level="strong",
          catalyst_window=40, note="交易环节税负下降，利好二手房成交与置换链"),
    Event(eid="G20241118", date="2024-11-18", type="geopolitics", subtype="export_control", lag=0,
          title="商务部等四部门公布《中华人民共和国两用物项出口管制清单》（12月1日实施）",
          source="https://www.gov.cn/zhengce/zhengceku/202411/content_6987846.htm",
          source_reliability=1.0, scope="sector", direction=-1,
          sectors=["半导体", "有色金属"], evidence_level="strong",
          catalyst_window=20,
          note="对美反制工具制度化；对稀土/小金属是涨价逻辑（方向冲突，可检验）"),
    Event(eid="G20241217", date="2024-12-17", type="policy", subtype="soe_mktcap", lag=0,
          title="国务院国资委《关于改进和加强中央企业控股上市公司市值管理工作的若干意见》",
          source="https://www.gov.cn/zhengce/zhengceku/202412/content_6993174.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "建筑装饰"], evidence_level="strong",
          catalyst_window=20,
          note="市值管理纳入央企负责人考核，破净央企估值修复逻辑"),

    # ==================== 2025 ====================
    Event(eid="G20250122", date="2025-01-22", type="policy", subtype="long_money_inflow", lag=1,
          title="六部门印发《关于推动中长期资金入市工作的实施方案》",
          source="https://www.gov.cn/zhengce/202501/content_7000556.htm",
          source_reliability=1.0, scope="market", direction=1,
          sectors=["银行", "公用事业", "券商"], evidence_level="strong",
          catalyst_window=40,
          note="明确公募/保险/社保的入市比例与长周期考核；次日国新办发布会"),
    Event(eid="P20250507", date="2025-05-07", type="policy", subtype="rrr_cut", lag=0,
          title="央行降准0.5个百分点并下调政策利率10bp（一揽子金融政策，国新办9时发布会）",
          source="https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554653911/index.html",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["银行", "地产", "券商", "建筑装饰"], evidence_level="strong",
          catalyst_window=20,
          note="开盘前发布，当日高开；与同日证监会/金监总局政策构成组合拳"),
    Event(eid="G20250507", date="2025-05-07", type="policy", subtype="fund_industry_support", lag=0,
          title="证监会《推动公募基金高质量发展行动方案》（证监发〔2025〕21号）",
          source="https://www.gov.cn/zhengce/zhengceku/202505/content_7022806.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "银行"], evidence_level="strong",
          catalyst_window=40,
          note="浮动费率+长周期考核，压制'风格漂移'，间接利好低波红利"),
    Event(eid="P20250507B", date="2025-05-07", type="policy", subtype="housing_fund_rate", lag=0,
          title="央行下调个人住房公积金贷款利率0.25个百分点",
          source="https://www.gov.cn/zhengce/zhengceku/202505/content_7022708.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["地产", "建筑材料"], evidence_level="strong",
          catalyst_window=40,
          note="与降准同日发布，属同一政策包，单独计量会高估其独立效应"),
    Event(eid="G20250618", date="2025-06-18", type="policy", subtype="registration_reform", lag=0,
          title="证监会《关于在科创板设置科创成长层 增强制度包容性适应性的意见》",
          source="https://www.gov.cn/zhengce/zhengceku/202506/content_7028576.htm",
          source_reliability=1.0, scope="sector", direction=1,
          sectors=["券商", "半导体"], evidence_level="strong",
          catalyst_window=40,
          note="陆家嘴论坛宣布；未盈利科技企业上市通道拓宽，利好半导体/创新药融资"),
]

# 2026：本仓库的数据与检索都到 2026-11，但**没有**找到"页面自带日期"可核的重大
# 金融/产业政策原文页（2026 年的条目多为部委常规通知），因此按口径留空，不硬凑。

BY_YEAR: Dict[int, int] = {}
for _ev in CALENDAR:
    BY_YEAR[_ev.year] = BY_YEAR.get(_ev.year, 0) + 1

# 明确删除的候选（找不到可核来源，宁可不写）
DELETED: List[str] = [
    "2017-02-17 再融资新规（证监会公告〔2017〕5号）—— gov.cn 只有公报转载页(2017-09-30)",
    "2017-05-27 减持新规（证监会公告〔2017〕9号）—— 同上，公报转载页(2017-11-30)",
    "2016-03-24 营改增全面推开（财税〔2016〕36号）—— 只有公报转载页(2016-07-10)",
    "2018-04-27 资管新规（银发〔2018〕106号）—— 只有公报转载页(2018-09-20)",
    "2019-01-30 科创板设立实施意见 —— 只有公报转载页(2019-10-18)",
    "2019-08-17 LPR形成机制改革 —— 央行站内检索无对应公告页",
    "2013-07-19 全面放开贷款利率管制 —— 同上",
    "2020-08-20 房企'三道红线' —— 仅座谈会报道，无公开文件",
    "2015-09-07 股指期货交易管控 / 2016-01-07 熔断暂停 —— 交易所公告页未核实",
    "2021-10-23 房地产税改革试点授权决定 —— gov.cn 政策库无条目",
    "2025-02-17 民营企业座谈会 —— 仅地方网站转载新华社稿，无一手页",
    "2010-2026 各类'某某规划/指导意见' —— 与二级市场传导路径太远，不收",
]


def get_events(start: str = "2010-01-01", end: str = "2099-12-31",
               types: List[str] | None = None,
               subtypes: List[str] | None = None) -> List[Event]:
    """按区间/类型过滤（含端点）。日期比较用字符串，ISO 格式可直接比大小。"""
    out = []
    for ev in CALENDAR:
        if not (start <= ev.date <= end):
            continue
        if types and ev.type not in types:
            continue
        if subtypes and ev.subtype not in subtypes:
            continue
        out.append(ev)
    return out


def to_jsonl(path: str = "data/events/policy_calendar.jsonl") -> str:
    """把日历落成 JSONL（回测直接读，不依赖 Python 模块）。"""
    from qcore.events import save_events
    return save_events(CALENDAR, path)


def to_state_json(path: str | None = None, impact=None) -> str:
    """
    导出为 Web UI 直接可读的 data/state/events.json。
    （server 的 /api/events 优先读这个文件）
    """
    import json as _json
    import os as _os
    if path is None:
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(root, "data", "state", "events.json")
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    rows = []
    for e in CALENDAR:
        d = e.__dict__.copy() if hasattr(e, "__dict__") else dict(e)
        rows.append(d)
    payload = {"generated_at": __import__("datetime").datetime.now().astimezone()
               .isoformat(timespec="seconds"),
               "n": len(rows), "source": "qcore.policy_calendar",
               "events": rows, "rows": rows}

    # ---- 保留其他模块写进同一文件的内容 ----
    # news_live.merge_into_events() 会写 live / live_meta，
    # 这里如果直接覆盖，实时事件就全丢了（实测踩过：重算 impact 后
    # events.json 里 live 段消失，页面上 2026 年又变回空白）。
    try:
        if _os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                old = _json.load(f)
            for k in ("live", "live_meta"):
                if isinstance(old, dict) and old.get(k):
                    payload[k] = old[k]
    except Exception:
        pass

    if impact is not None:
        payload["impact"] = impact
        try:
            from .impact import summary as _imp_sum
            payload["impact_summary"] = _imp_sum(impact)
        except Exception:
            pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    _os.replace(tmp, path)
    return path


if __name__ == "__main__":
    import sys as _sys
    if "--to-json" in _sys.argv:
        p = to_state_json()
        print(f"wrote {p} ({len(CALENDAR)} events)")
        raise SystemExit(0)
    # 自查：条数 + 按年分布 + 字段完整性（全部 ASCII 输出，GBK 控制台安全）
    from qcore.events import validate_event
    print(f"total events: {len(CALENDAR)}")
    print(f"deleted candidates: {len(DELETED)}")
    years = sorted(BY_YEAR)
    print(f"year range: {years[0]}..{years[-1]}  years covered: {len(years)}")
    for y in years:
        print(f"  {y}: {BY_YEAR[y]:2d}  {'#' * BY_YEAR[y]}")
    missing = [y for y in range(years[0], years[-1] + 1) if y not in BY_YEAR]
    print(f"years with no event: {missing}")
    ids = [e.eid for e in CALENDAR]
    assert len(ids) == len(set(ids)), "duplicate eid"
    dates = [e.date for e in CALENDAR]
    assert dates == sorted(dates), "CALENDAR must be sorted by date"
    bad = []
    for e in CALENDAR:
        probs = validate_event(e)
        if probs:
            bad.append((e.eid, probs))
    print(f"validation failures: {len(bad)}")
    for eid, probs in bad[:10]:
        print("  ", eid, probs)
    by_type: Dict[str, int] = {}
    for e in CALENDAR:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    print("by type:", by_type)
    by_sub: Dict[str, int] = {}
    for e in CALENDAR:
        by_sub[e.subtype] = by_sub.get(e.subtype, 0) + 1
    print("by subtype (top10):", sorted(by_sub.items(), key=lambda x: -x[1])[:10])
    print("sources: pbc=%d gov=%d other=%d" % (
        sum("pbc.gov.cn" in e.source for e in CALENDAR),
        sum("gov.cn" in e.source for e in CALENDAR),
        sum(("pbc.gov.cn" not in e.source and "gov.cn" not in e.source) for e in CALENDAR)))
    print("all sources unique:", len({e.source for e in CALENDAR}) == len(CALENDAR))
    print("lag=1 events:", sum(e.lag == 1 for e in CALENDAR))
