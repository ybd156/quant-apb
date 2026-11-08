/* ==========================================================================
 * APB 控制台 — app.js
 * 原生 JS，零依赖：无 CDN、无框架、无构建步骤、无图表库（图表全部手绘 SVG）。
 *
 * 铁律：前端只做 IO 与呈现，**不计算任何指标**。
 *   所有数值原样来自 /api/* -> data/state/*.json（由 qcore 产出）；
 *   这里只做「取字段 / 单位标注 / 排序 / 筛选 / 排版」。
 *   - 排序、筛选 = 选择行，不产生新数值；
 *   - 图表坐标 = 像素映射，不改变数值含义。
 * ========================================================================== */
(function () {
  'use strict';

  /* ============================== 基础工具 ============================== */

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.prototype.slice.call((root || document).querySelectorAll(sel));

  const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (v) => String(v === null || v === undefined ? '' : v)
    .replace(/[&<>"']/g, (c) => ESC_MAP[c]);

  const isNil = (v) => v === null || v === undefined || v === '';
  const toNum = (v) => {
    if (isNil(v) || typeof v === 'boolean') return null;
    const n = Number(v);
    return isFinite(n) ? n : null;
  };
  function firstNum(list) {
    for (let i = 0; i < list.length; i++) {
      const n = toNum(list[i]);
      if (n !== null) return n;
    }
    return null;
  }
  const num = (v, d) => {
    const n = toNum(v);
    return n === null ? '—' : n.toFixed(d === undefined ? 3 : d);
  };
  /** 比率：|v|<=1 视为分数（0.53→53.00%），否则视为已是百分数 */
  function pct(v, d) {
    const n = toNum(v);
    if (n === null) return '—';
    const x = Math.abs(n) <= 1.0000001 ? n * 100 : n;
    return x.toFixed(d === undefined ? 2 : d) + '%';
  }
  function signedPct(v, d) {
    const n = toNum(v);
    if (n === null) return '—';
    const x = Math.abs(n) <= 1.0000001 ? n * 100 : n;
    return (x > 0 ? '+' : '') + x.toFixed(d === undefined ? 2 : d) + '%';
  }
  function signClass(v) {
    const n = toNum(v);
    if (n === null || n === 0) return '';
    return n > 0 ? 'up-text' : 'down-text';
  }
  /** 引擎在基线/成本表里已按百分数给值（14.21 表示 14.21%） */
  function fmtByKey(key, v) {
    const n = toNum(v);
    if (n === null) return isNil(v) ? '—' : esc(v);
    if (/收益|回撤|胜率|波动|成本|占比|价差/.test(key)) return n.toFixed(2) + '%';
    if (/期数|笔数|次数/.test(key)) return n.toFixed(0);
    return n.toFixed(3);
  }

  const REGIMES = {
    0: { name: '下跌低波', color: 'var(--rg0)' },
    1: { name: '下跌高波', color: 'var(--rg1)' },
    2: { name: '上涨低波', color: 'var(--rg2)' },
    3: { name: '上涨高波', color: 'var(--rg3)' }
  };
  const regimeInfo = (v) => {
    const n = toNum(v);
    return n === null ? { name: '未知', color: 'var(--rgx)' } : (REGIMES[n] || { name: 'regime ' + n, color: 'var(--rgx)' });
  };

  const EVENT_TYPES = {
    policy: '政策', earnings: '业绩', order: '订单', capital: '资本运作',
    regulation: '监管', geo: '地缘', macro: '宏观'
  };

  /* ================================ 请求层 ============================== */

  const _cache = new Map();
  const clearCache = () => _cache.clear();

  async function api(url, options) {
    const opts = Object.assign({ headers: { Accept: 'application/json' } }, options || {});
    let res;
    try {
      res = await fetch(url, opts);
    } catch (e) {
      setHealth('down');
      return { ok: false, error: 'network_error', detail: String((e && e.message) || e) };
    }
    let body = null;
    try { body = await res.json(); } catch (e) { body = null; }
    if (!body || typeof body !== 'object') {
      return { ok: false, error: 'bad_response', detail: 'HTTP ' + res.status };
    }
    body._status = res.status;
    if (body.ok !== true && !body.error) body.error = 'http_' + res.status;
    return body;
  }

  /** ttl 内复用同一 GET（performance.json 近 1MB，避免翻页重复解析） */
  function apiCached(url, ttl) {
    const now = Date.now();
    const hit = _cache.get(url);
    if (hit && now - hit.t < (ttl === undefined ? 5000 : ttl)) return hit.p;
    const p = api(url);
    _cache.set(url, { t: now, p: p });
    return p;
  }

  /* ============================== 提示 / 占位 ============================ */

  let toastTimer = null;
  function toast(msg, kind) {
    const el = $('#toast');
    el.textContent = msg;
    el.className = 'toast show' + (kind ? ' ' + kind : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 3800);
  }

  const ICON = (name, cls) => '<svg class="ico ' + (cls || '') + '" aria-hidden="true"><use href="#i-' + name + '"></use></svg>';

  function emptyBox(title, desc, icon) {
    return '<div class="empty">' + ICON(icon || 'dash') +
      '<div class="t">' + esc(title) + '</div>' +
      (desc ? '<div class="d">' + desc + '</div>' : '') + '</div>';
  }

  /** 数据未就绪 / 出错的统一占位 */
  function notReadyBox(res, what) {
    const err = (res && res.error) || 'unknown';
    if (err === 'not_ready') {
      const file = res.missing ? '<code>data/state/' + esc(res.missing) + '</code>' : '数据文件';
      return emptyBox('未就绪：等待引擎首次运行',
        '缺少 ' + file + '。<br>先运行 <code>python -m qcore.live --once</code>' +
        '（首次需 <code>--rebuild</code>），生成后点右上角「刷新」。', 'alert');
    }
    if (err === 'bad_json') return emptyBox('数据文件解析失败', '文件可能正在写入，稍后点「刷新」重试。', 'alert');
    if (err === 'network_error') return emptyBox('无法连接服务', '服务进程可能已退出，请检查终端。', 'alert');
    return emptyBox('加载失败：' + esc(err), what ? esc(what) : '', 'alert');
  }

  /* =============================== 组件 ================================ */

  function card(opt) {
    opt = opt || {};
    const tools = opt.tools ? '<div class="card-tools">' + opt.tools + '</div>' : '';
    const head = (opt.title || tools)
      ? '<div class="card-head"><h2>' + esc(opt.title || '') + '</h2>' +
        (opt.sub ? '<span class="sub">' + opt.sub + '</span>' : '') + tools + '</div>'
      : '';
    return '<section class="card ' + (opt.cls || '') + '"' + (opt.id ? ' id="' + opt.id + '"' : '') + '>' +
      head + (opt.body || '') + '</section>';
  }

  function kpi(o) {
    return '<div class="kpi tone-' + (o.tone || 'brand') + '">' +
      '<span class="k">' + esc(o.k) + '</span>' +
      '<span class="v ' + (o.sm ? 'sm ' : '') + (o.cls || '') + '">' + esc(o.v) + '</span>' +
      (o.s ? '<span class="s">' + o.s + '</span>' : '') + '</div>';
  }

  const kvItem = (k, v, cls, sub) => '<div class="item"><span class="k">' + esc(k) + '</span>' +
    '<span class="v ' + (cls || '') + '">' + esc(v) + '</span>' +
    (sub ? '<span class="s">' + esc(sub) + '</span>' : '') + '</div>';

  const kv = (items) => '<div class="kv">' + items.join('') + '</div>';

  function banner(o) {
    return '<div class="banner ' + (o.tone || '') + '">' +
      '<div class="b-ico">' + ICON(o.icon || 'alert') + '</div>' +
      '<div><div class="b-title">' + esc(o.title) + '</div>' +
      (o.text ? '<div class="b-text">' + o.text + '</div>' : '') +
      (o.meta ? '<div class="b-meta">' + esc(o.meta) + '</div>' : '') + '</div></div>';
  }

  function table(opt) {
    const cols = opt.cols || [];
    const rows = opt.rows || [];
    const head = '<tr>' + cols.map((c) => {
      const cls = (c.num ? 'num ' : '') + (c.sort ? 'sortable ' : '') +
        (opt.sort && opt.sort.key === c.sort ? 'sorted' : '');
      const arrow = c.sort ? '<span class="arrow">' +
        (opt.sort && opt.sort.key === c.sort ? (opt.sort.dir > 0 ? '▲' : '▼') : '↕') + '</span>' : '';
      return '<th class="' + cls + '"' + (c.sort ? ' data-sort="' + c.sort + '"' : '') + '>' +
        esc(c.label) + arrow + '</th>';
    }).join('') + '</tr>';

    let body;
    if (!rows.length) {
      body = '<tr class="row-empty"><td colspan="' + cols.length + '">' +
        (opt.empty || emptyBox('暂无数据', '')) + '</td></tr>';
    } else {
      body = rows.map((r) => '<tr>' + cols.map((c) => {
        const v = c.render ? c.render(r) : esc(isNil(r[c.key]) ? '—' : r[c.key]);
        return '<td class="' + (c.num ? 'num ' : '') + (c.mono ? 'mono' : '') + '">' + v + '</td>';
      }).join('') + '</tr>').join('');
    }
    return '<div class="table-wrap"><table class="table"><thead>' + head +
      '</thead><tbody>' + body + '</tbody></table></div>';
  }

  /* =============================== 图表 ================================ */

  const chartWidth = () => (window.innerWidth < 768 ? 520 : window.innerWidth < 1200 ? 760 : 960);

  function fmtAxis(v) {
    const a = Math.abs(v);
    if (a >= 10000) return (v / 1000).toFixed(0) + 'k';
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    return v.toFixed(3);
  }

  /**
   * 手绘折线图。series: [{values, color, label, area, dashed}]
   * opt: {height, labels:{first,last}, aria, tipLabels:[]}
   * 每个数据点带 <title>，悬停即显示「日期 + 数值」（原生 tooltip，无 JS 依赖）。
   */
  function chartSVG(series, opt) {
    opt = opt || {};
    const W = chartWidth();
    const H = opt.height || 250;
    const padL = W < 700 ? 44 : 56;
    const padR = 14;
    const padT = series.length > 1 ? 26 : 14;
    const padB = 26;
    const iw = W - padL - padR;
    const ih = H - padT - padB;

    const all = [];
    series.forEach((s) => (s.values || []).forEach((v) => {
      const n = toNum(v);
      if (n !== null) all.push(n);
    }));
    if (!all.length) return '';

    let lo = Math.min.apply(null, all);
    let hi = Math.max.apply(null, all);
    if (!(hi > lo)) { const pad = Math.abs(hi) * 0.02 || 1; lo -= pad; hi += pad; }
    const span = hi - lo;
    const n = Math.max.apply(null, series.map((s) => (s.values || []).length));
    const px = (i) => padL + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
    const py = (v) => padT + ih - ((v - lo) / span) * ih;

    let out = '';
    for (let k = 0; k <= 4; k++) {
      const y = padT + (ih * k) / 4;
      const val = hi - (span * k) / 4;
      out += '<line class="grid-line" x1="' + padL + '" y1="' + y.toFixed(1) +
        '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) + '"/>';
      out += '<text class="axis-text" x="' + (padL - 7) + '" y="' + (y + 3.5).toFixed(1) +
        '" text-anchor="end">' + esc(fmtAxis(val)) + '</text>';
    }
    if (lo < 0 && hi > 0) {
      const y0 = py(0).toFixed(1);
      out += '<line class="zero-line" x1="' + padL + '" y1="' + y0 +
        '" x2="' + (W - padR) + '" y2="' + y0 + '"/>';
    }
    if (opt.labels && (opt.labels.first || opt.labels.last)) {
      out += '<text class="axis-text" x="' + padL + '" y="' + (H - 8) + '">' + esc(opt.labels.first || '') + '</text>';
      out += '<text class="axis-text" x="' + (W - padR) + '" y="' + (H - 8) +
        '" text-anchor="end">' + esc(opt.labels.last || '') + '</text>';
    }

    const tips = opt.tipLabels || [];
    series.forEach((s, si) => {
      const vals = s.values || [];
      let d = '';
      let started = false;
      const pts = [];
      for (let i = 0; i < vals.length; i++) {
        const v = toNum(vals[i]);
        if (v === null) { started = false; continue; }
        d += (started ? 'L' : 'M') + px(i).toFixed(1) + ' ' + py(v).toFixed(1) + ' ';
        started = true;
        pts.push([i, v]);
      }
      if (!d) return;
      if (s.area && vals.length > 1) {
        const first = px(0).toFixed(1);
        const last = px(vals.length - 1).toFixed(1);
        out += '<path d="' + d + 'L' + last + ' ' + (padT + ih) + ' L' + first + ' ' +
          (padT + ih) + ' Z" fill="' + s.color + '" opacity="0.10" stroke="none"/>';
      }
      out += '<path d="' + d + '" fill="none" stroke="' + s.color +
        '" stroke-width="' + (si === 0 ? 1.9 : 1.5) + '" stroke-linejoin="round" stroke-linecap="round"' +
        (s.dashed ? ' stroke-dasharray="5 4"' : '') + '/>';
      const step = pts.length > 160 ? Math.ceil(pts.length / 160) : 1;
      for (let j = 0; j < pts.length; j += step) {
        const i = pts[j][0];
        const v = pts[j][1];
        out += '<circle class="pt" cx="' + px(i).toFixed(1) + '" cy="' + py(v).toFixed(1) +
          '" r="' + (step > 1 ? 4 : 2.6) + '" fill="' + s.color + '" opacity="' + (step > 1 ? '0.001' : '0.9') + '">' +
          '<title>' + esc((s.label ? s.label + ' · ' : '') + (tips[i] ? tips[i] + ' ' : '') + num(v, 4)) + '</title></circle>';
      }
    });

    if (series.length > 1) {
      let lx = padL;
      series.forEach((s) => {
        out += '<rect x="' + lx + '" y="6" width="9" height="9" rx="2" fill="' + s.color + '"/>';
        out += '<text class="axis-text" x="' + (lx + 13) + '" y="14">' + esc(s.label || '') + '</text>';
        lx += 18 + String(s.label || '').length * 7;
      });
    }
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
      esc(opt.aria || '折线图') + '">' + out + '</svg>';
  }

  /** 水平条形（权重 Top/Bottom）。rows: [{label, value, sid, title, cls, sel}] */
  function hbars(rows, opt) {
    opt = opt || {};
    if (!rows.length) return emptyBox(opt.emptyTitle || '暂无数据', opt.emptyDesc || '');
    const maxAbs = opt.max || rows.reduce((m, r) => Math.max(m, Math.abs(toNum(r.value) || 0)), 0) || 1;
    return '<div class="hbars">' + rows.map((r) => {
      const v = toNum(r.value);
      const w = v === null ? 0 : Math.max(1.5, (Math.abs(v) / maxAbs) * 100);
      return '<button type="button" class="hbar' + (r.sel ? ' is-sel' : '') + '"' +
        (r.sid ? ' data-sid="' + esc(r.sid) + '"' : '') +
        (r.title ? ' title="' + esc(r.title) + '"' : '') + '>' +
        '<span class="track"><i class="fill ' + (r.cls || '') + '" style="width:' + w.toFixed(1) + '%"></i>' +
        '<span class="name">' + esc(r.label) + '</span></span>' +
        '<span class="val">' + esc(opt.fmt ? opt.fmt(v) : num(v, 4)) + '</span></button>';
    }).join('') + '</div>';
  }

  /** 双向条形（从 0 向两侧生长）：IC / 价差 / 分年度 IC */
  function dbars(rows, opt) {
    opt = opt || {};
    if (!rows.length) return emptyBox(opt.emptyTitle || '暂无数据', opt.emptyDesc || '');
    const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(toNum(r.value) || 0)), 0) || 1;
    return '<div class="dbars">' + rows.map((r) => {
      const v = toNum(r.value);
      const w = v === null ? 0 : (Math.abs(v) / maxAbs) * 50;
      const side = v === null ? '' : v >= 0 ? 'pos' : 'neg';
      return '<div class="dbar" title="' + esc((r.label || '') + ' ' + (v === null ? '—' : num(v, 4))) + '">' +
        '<span class="lbl">' + esc(r.label || '') + '</span>' +
        '<span class="mid">' + (side ? '<i class="' + side + '" style="width:' + w.toFixed(1) + '%"></i>' : '') + '</span>' +
        '<span class="val">' + esc(opt.fmt ? opt.fmt(v) : num(v, 4)) + '</span></div>';
    }).join('') + '</div>';
  }

  /** regime 色带 */
  function band(rows) {
    if (!rows.length) return { html: emptyBox('暂无 regime 序列', ''), legend: '' };
    const html = '<div class="band">' + rows.map((r) => {
      const info = regimeInfo(r.regime);
      return '<span style="background:' + info.color + '" title="' +
        esc((r.date || '') + ' · ' + info.name) + '"></span>';
    }).join('') + '</div>';
    const legend = '<div class="legend">' + Object.keys(REGIMES).map((k) =>
      '<span class="lg"><i class="sw" style="background:' + REGIMES[k].color + '"></i>' +
      esc(k + ' = ' + REGIMES[k].name) + '</span>').join('') + '</div>';
    return { html: html, legend: legend };
  }

  /** 政策事件时间线 */
  function timeline(rows) {
    if (!rows.length) return emptyBox('暂无事件', '');
    return '<ul class="timeline">' + rows.map((e) => {
      const dir = toNum(e.direction);
      const cls = dir === null ? '' : dir > 0 ? 'dir-pos' : dir < 0 ? 'dir-neg' : '';
      const type = EVENT_TYPES[e.type] || e.type || '事件';
      const sectors = Array.isArray(e.sectors) ? e.sectors : (e.sectors ? [e.sectors] : []);
      const dirTxt = dir === null ? '—' : dir > 0 ? '利好' : dir < 0 ? '利空' : '中性';
      return '<li class="tl-item ' + cls + '">' +
        '<div class="tl-head"><span class="tl-date">' + esc(e.date || '—') + '</span>' +
        '<span class="tag info">' + esc(type) + '</span>' +
        '<span class="tag ' + (dir > 0 ? 'hit' : dir < 0 ? 'miss' : 'na') + '">' + esc(dirTxt) + '</span>' +
        (e.evidence_level ? '<span class="tag na">证据 ' + esc(e.evidence_level) + '</span>' : '') +
        (e.lag ? '<span class="tag na">lag ' + esc(e.lag) + '</span>' : '') +
        '</div>' +
        '<div class="tl-title">' + esc(e.title || '（无标题）') + '</div>' +
        (sectors.length ? '<div class="tl-meta">' + sectors.slice(0, 8).map((s) =>
          '<span class="tag info">' + esc(s) + '</span>').join('') + '</div>' : '') +
        (e.note ? '<div class="tl-note">' + esc(e.note) + '</div>' : '') +
        '</li>';
    }).join('') + '</ul>';
  }

  /* ============================ 形态适配（不产数值） ==================== */

  function seriesValues(arr) {
    if (!Array.isArray(arr)) return [];
    return arr.map((x) => {
      if (x === null || x === undefined) return null;
      if (typeof x === 'number') return x;
      if (typeof x === 'string') return toNum(x);
      if (typeof x === 'object') {
        const keys = ['nav', 'value', 'y', 'equity', 'close', 'ret'];
        for (let i = 0; i < keys.length; i++) {
          const v = toNum(x[keys[i]]);
          if (v !== null) return v;
        }
      }
      return null;
    });
  }
  function seriesDates(arr) {
    if (!Array.isArray(arr)) return [];
    return arr.map((x) => (x && typeof x === 'object' ? (x.date || x.as_of || '') : ''));
  }
  function findCurve(obj) {
    if (!obj || typeof obj !== 'object') return null;
    const keys = ['curve', 'equity', 'nav_curve', 'navs', 'nav_history', 'history'];
    for (let i = 0; i < keys.length; i++) {
      const v = obj[keys[i]];
      if (Array.isArray(v) && v.length) return v;
    }
    return null;
  }

  function adaptPerformance(d) {
    const out = {
      hits: [], ic: [], ls: null, baselines: [], curve: null, labels: null,
      edge: null, meta: null, confidence: null, risk: null, costs: null, learned: null, dataset: null
    };
    if (!d || typeof d !== 'object') return out;
    const metrics = (d.metrics && typeof d.metrics === 'object') ? d.metrics : {};
    const backtest = (d.backtest && typeof d.backtest === 'object') ? d.backtest : {};
    out.meta = d.meta || null;
    out.confidence = d.confidence || null;
    out.risk = d.risk || null;
    out.costs = d.costs || null;
    out.learned = d.learned_vs_equal || null;
    out.edge = d.edge || null;
    out.dataset = d.dataset || null;

    const hr = (d.hit_rate && typeof d.hit_rate === 'object') ? d.hit_rate
      : (metrics.hit_rate && typeof metrics.hit_rate === 'object') ? metrics.hit_rate : null;
    if (hr) {
      Object.keys(hr).forEach((k) => {
        const v = hr[k];
        let val = null;
        if (typeof v === 'number') val = v;
        else if (v && typeof v === 'object') val = toNum(v.hit_rate);
        if (val === null) return;
        const m = String(k).match(/(\d+)/);
        out.hits.push({ h: m ? Number(m[1]) : 0, label: 'h=' + (m ? m[1] : k), value: val });
      });
      out.hits.sort((a, b) => a.h - b.h);
    }

    const ic = (d.ic && typeof d.ic === 'object') ? d.ic
      : (metrics.rank_ic && typeof metrics.rank_ic === 'object') ? metrics.rank_ic : null;
    if (ic) {
      if (toNum(ic.mean) !== null) {
        out.ic.push({ label: 'mean IC', value: toNum(ic.mean), kind: 'num' });
        out.ic.push({ label: 'ICIR', value: firstNum([ic.ir, ic.icir]), kind: 'num' });
        out.ic.push({ label: 'IC 胜率', value: firstNum([ic.win_rate, ic.positive_ratio]), kind: 'pct' });
      } else {
        Object.keys(ic).forEach((k) => {
          const v = ic[k];
          if (!v || typeof v !== 'object') return;
          const m = String(k).match(/(\d+)/);
          const tag = 'h=' + (m ? m[1] : k);
          out.ic.push({ label: tag + ' mean IC', value: toNum(v.mean), kind: 'num' });
          out.ic.push({ label: tag + ' ICIR', value: firstNum([v.icir, v.ir]), kind: 'num' });
          out.ic.push({ label: tag + ' IC 胜率', value: firstNum([v.positive_ratio, v.win_rate]), kind: 'pct' });
        });
      }
    }

    const lsRaw = (d.long_short && typeof d.long_short === 'object') ? d.long_short
      : (backtest.long_short && typeof backtest.long_short === 'object') ? backtest.long_short : null;
    if (lsRaw) {
      const ls = {
        annual: firstNum([lsRaw.annual, lsRaw.ann_return, lsRaw.ann_ls]),
        sharpe: toNum(lsRaw.sharpe),
        mdd: firstNum([lsRaw.mdd, lsRaw.max_drawdown]),
        win_rate: toNum(lsRaw.win_rate),
        n_periods: toNum(lsRaw.n_periods)
      };
      if (ls.annual !== null || ls.sharpe !== null || ls.mdd !== null) out.ls = ls;
      const curve = findCurve(lsRaw);
      if (curve) {
        out.curve = seriesValues(curve);
        const dd = seriesDates(curve).filter(Boolean);
        if (dd.length) out.labels = { first: dd[0], last: dd[dd.length - 1] };
      }
    }

    if (Array.isArray(backtest.baselines)) out.baselines = backtest.baselines.filter((r) => r && typeof r === 'object');
    else if (d.baseline && typeof d.baseline === 'object') {
      out.baselines = Object.keys(d.baseline).map((k) => {
        const v = d.baseline[k];
        return (v && typeof v === 'object') ? Object.assign({ 策略: k }, v) : { 策略: k, 值: v };
      });
    }
    return out;
  }

  /* ============================== 全局状态 ============================= */

  const S = {
    page: 'overview',
    version: null,
    health: null,
    sort: { signals: { key: 'weight', dir: -1 } },
    filter: {
      family: 'all', q: '', result: 'all', from: '', to: '', days: 30,
      evType: 'all', evDir: 'all'
    },
    selSid: null,
    update: null,
    updating: false,
    names: {},
    namesHint: '',
    account: null
  };

  /* ============================== 页面定义 ============================= */

  const PAGES = [
    { id: 'overview', label: '总览', icon: 'overview', title: '总览', sub: '这台测量仪现在读数是多少' },
    { id: 'market', label: '市场', icon: 'market', title: '实时行情', sub: '指数 / 情绪温度 / 市场宽度' },
    { id: 'signals', label: '信号', icon: 'signals', title: '信号权重', sub: '每个「信号 × 参数」专家当前权重' },
    { id: 'model', label: '模型', icon: 'model', title: '模型', sub: 'Alpha158 + GBDT · 当前推荐持仓' },
    { id: 'ledger', label: '预测台账', icon: 'ledger', title: '预测台账', sub: '先封存，后对账' },
    { id: 'paper', label: '模拟实盘', icon: 'paper', title: '模拟实盘', sub: '模拟盘 vs 实盘对账' },
    { id: 'behavior', label: '行为因子', icon: 'behavior', title: '行为因子', sub: '10 个行为金融因子的横截面表现' },
    { id: 'events', label: '政策事件', icon: 'events', title: '政策事件', sub: '政策日历与传导影响' },
    { id: 'risk', label: '风险体检', icon: 'risk', title: '风险体检', sub: '不变量 / 成本 / 数据源 / 集中度' },
    { id: 'settings', label: '设置', icon: 'settings', title: '设置', sub: '版本、目录、更新与导出' }
  ];
  const pageMeta = (id) => PAGES.filter((p) => p.id === id)[0] || PAGES[0];

  /* ------------------------------ 顶栏状态 ---------------------------- */

  const setSub = (text) => { $('#page-sub').textContent = text || ''; };

  function setHealth(state, text) {
    const el = $('#health-dot');
    el.classList.remove('ok', 'stale', 'down');
    if (state) el.classList.add(state);
    el.querySelector('b').textContent = text || (state === 'ok' ? '正常' : state === 'stale' ? '陈旧' : state === 'down' ? '异常' : '—');
    el.title = '连接状态：' + (state === 'ok' ? '服务正常，数据新鲜'
      : state === 'stale' ? '服务正常，但数据超过 1 天未更新'
        : state === 'down' ? '无法连接服务或数据文件缺失' : '检测中');
  }

  async function loadVersion() {
    const res = await api('/api/version');
    if (res.ok && res.data) {
      S.version = res.data;
      const v = 'v' + (res.data.version || '—');
      $('#chip-version').textContent = v;
      $('#side-version').textContent = v;
    }
  }

  async function loadHealth() {
    const res = await api('/api/health');
    if (!res.ok || !res.data) { setHealth('down'); return; }
    S.health = res.data;
    const age = toNum(res.data.data_age_days);
    const missing = (res.data.missing_files || []).length;
    if (missing) setHealth('down', '缺 ' + missing + ' 文件');
    else if (age !== null && age > 1) setHealth('stale', '陈旧 ' + age.toFixed(0) + ' 天');
    else setHealth('ok');
    const asof = res.data.as_of || '—';
    $('#chip-asof').textContent = '数据 ' + asof;
    $('#chip-asof').title = '最新数据日 ' + asof + (age === null ? '' : '（' + age + ' 天前）');
  }

  /* ======================= 版本更新（应用内一键更新） ==================== */

  // 后端错误码 → 用户可读提示（分类处理，不把裸错误码丢给用户）
  const UPDATE_ERRORS = {
    token_not_configured: {
      text: '服务端未设置 APB_TOKEN，无法在应用内更新',
      hint: '请在容器环境变量里设置 APB_TOKEN 后重启服务，或改用命令行更新',
      clearToken: true
    },
    unauthorized: {
      text: 'X-Token 与 APB_TOKEN 不匹配',
      hint: '请重新输入服务端环境变量里的 APB_TOKEN',
      clearToken: true
    },
    docker_socket_unavailable: {
      text: '容器未挂载 docker.sock，无法应用内更新',
      hint: '请在宿主机执行：docker compose pull && docker compose up -d'
    },
    already_latest: { text: '已是最新版本', hint: '' },
    import_error: { text: '服务端缺少 docker_api 模块', hint: '请改用命令行更新' },
    helper_create_failed: { text: '辅助更新容器创建失败', hint: '请查看服务端日志或改用命令行更新' },
    helper_start_failed: { text: '辅助更新容器启动失败', hint: '请查看服务端日志或改用命令行更新' },
    network_error: { text: '无法连接服务（可能正在重启）', hint: '稍等 1–3 分钟后刷新页面' },
    not_ready: { text: '服务端还没有更新信息', hint: '等服务器完成一次版本检查后重试' }
  };

  /** token 只存本机 localStorage（跨标签页只填一次） */
  function getToken() {
    let t = '';
    try { t = localStorage.getItem('apb.token') || ''; } catch (e) { /* ignore */ }
    if (!t) { try { t = sessionStorage.getItem('apb_token') || ''; } catch (e) { /* ignore */ } }
    return t;
  }
  function setToken(t) {
    try { localStorage.setItem('apb.token', t || ''); } catch (e) { /* ignore */ }
    try { sessionStorage.setItem('apb_token', t || ''); } catch (e) { /* ignore */ }
  }

  /** 语义化版本比较（仅数字段）。返回 -1 / 0 / 1 —— 纯字符串处理，不涉及任何指标 */
  function cmpVersion(a, b) {
    const pa = String(a || '').split('.');
    const pb = String(b || '').split('.');
    const n = Math.max(pa.length, pb.length);
    for (let i = 0; i < n; i++) {
      const va = parseInt(pa[i], 10);
      const vb = parseInt(pb[i], 10);
      const x = isFinite(va) ? va : 0;
      const y = isFinite(vb) ? vb : 0;
      if (x !== y) return x > y ? 1 : -1;
    }
    return 0;
  }

  /** 只有远端「严格更新」才算有新版本：后端 has_update 是 latest != current，
      会把「远端比本地旧」也标成可更新，这里必须再判一次，避免把降级当成升级。 */
  function isNewer(latest, current) {
    return !!latest && !!current && cmpVersion(latest, current) > 0;
  }

  async function loadUpdate(force) {
    const res = await apiCached('/api/update', force ? 0 : 5000);
    S.update = (res.ok && res.data) ? res.data : null;
    renderUpdateBadge();
    return S.update;
  }

  function renderUpdateBadge() {
    const u = S.update;
    const badge = $('#badge-update');
    const text = $('#badge-update-text');
    if (!badge) return;
    if (S.updating) {
      badge.hidden = false;
      badge.disabled = true;
      if (text) text.textContent = '更新中…';
      return;
    }
    const latest = u && u.latest;
    const cur = (u && u.current) || (S.version && S.version.version) || '';
    if (u && u.has_update && isNewer(latest, cur)) {
      badge.hidden = false;
      badge.disabled = false;
      if (text) text.textContent = '有新版本 v' + latest;
      badge.title = '当前 v' + cur + ' → v' + latest + '，点击更新';
    } else {
      badge.hidden = true;
      badge.disabled = false;
      badge.title = '当前 v' + (cur || '?') + (latest ? '（远端 v' + latest + ' 不高于本地）' : '（尚未检查远端）');
    }
  }

  /* ------------------------------- 弹窗 ------------------------------- */

  function openModal(o) {
    $('#modal-title').textContent = o.title || '';
    $('#modal-body').innerHTML = o.body || '';
    $('#modal-foot').innerHTML = o.foot || '';
    $('#modal-mask').hidden = false;
    $$('#modal-foot [data-close]').forEach((btn) => btn.addEventListener('click', closeModal));
    const input = $('#modal-body input');
    if (input && !input.value) input.focus();
  }

  function closeModal() { $('#modal-mask').hidden = true; }

  /* --------------------------- 应用内更新流程 -------------------------- */

  function openUpdateModal() {
    const u = S.update || {};
    const cur = u.current || (S.version && S.version.version) || '—';
    const latest = u.latest || '—';
    openModal({
      title: '更新到 v' + latest,
      body:
        '<div class="kv" style="margin-bottom:12px">' +
        kvItem('当前版本', 'v' + cur) +
        kvItem('目标版本', 'v' + latest) +
        kvItem('最近检查', u.checked_at || '未检查') +
        '</div>' +
        '<p>更新会在后台拉取新镜像并重启容器，<b>约 1–3 分钟</b>内本页面不可用，随后自动恢复。</p>' +
        '<p class="muted" style="margin-top:6px">内网访问可直接更新（服务端未设 APB_TOKEN 时只放行内网）；' +
        '如需公网访问请设置 <code>APB_TOKEN</code>。也可用命令行：<code>' + esc(u.command || 'docker compose pull && docker compose up -d') + '</code></p>' +
        '<p class="form-msg" id="upd-msg" style="margin-top:8px"></p>',
      foot: '<button class="btn ghost" type="button" data-close="1">取消</button>' +
        '<button class="btn primary" id="upd-confirm" type="button">' + ICON('download') + '确认更新</button>'
    });
    const btn = $('#upd-confirm');
    if (btn) btn.addEventListener('click', applyUpdate);
  }

  async function applyUpdate() {
    const msg = $('#upd-msg');
    const btn = $('#upd-confirm');
    // 内网默认免 token：没设 APB_TOKEN 时服务端只放行内网。
    // 若服务端确实设了 token，会返回 unauthorized，这里给出提示而不是强制填。
    const token = (getToken() || '').trim();
    if (btn) btn.disabled = true;
    if (msg) { msg.className = 'form-msg'; msg.textContent = '正在提交…'; }

    const res = await api('/api/update/apply', {
      method: 'POST',
      headers: token ? { 'X-Token': token, Accept: 'application/json' }
                     : { Accept: 'application/json' }
    });
    if (btn) btn.disabled = false;

    if (!res.ok) {
      const info = UPDATE_ERRORS[res.error] || { text: '更新失败：' + (res.error || 'unknown'), hint: '' };
      if (info.clearToken) setToken('');
      if (msg) {
        msg.className = 'form-msg err';
        msg.innerHTML = esc(info.text) +
          (info.hint ? '<br><span class="muted">' + esc(info.hint) + '</span>' : '') +
          (res.hint && res.hint !== info.hint ? '<br><span class="muted">' + esc(res.hint) + '</span>' : '');
      }
      toast(info.text, 'err');
      if (res.error === 'already_latest') { loadUpdate(true); }
      return;
    }

    const to = (res.data && res.data.to) || (S.update && S.update.latest) || '';
    startUpdateWatch(to, (res.data && res.data.note) || '');
  }

  /** 提交成功后：显示更新中，并每 10 秒轮询 /api/version 直到版本变化 */
  function startUpdateWatch(to, note) {
    S.updating = true;
    renderUpdateBadge();
    const body = $('#modal-body');
    const foot = $('#modal-foot');
    if (body) {
      body.innerHTML =
        '<div style="text-align:center;padding:6px 0 2px">' +
        '<div class="spinner"></div>' +
        '<p style="margin-top:12px">更新中，目标版本 <b>v' + esc(to || '?') + '</b></p>' +
        '<p class="muted" style="margin-top:6px">' + esc(note || '更新在后台进行，约 1-3 分钟后刷新页面') + '</p>' +
        '<p class="mono" id="upd-watch" style="margin-top:12px;color:var(--text-3);font-size:11px">等待首次检查…</p>' +
        '</div>';
    }
    if (foot) {
      foot.innerHTML = '<button class="btn ghost" type="button" data-close="1">后台继续</button>';
      $$('#modal-foot [data-close]').forEach((btn) => btn.addEventListener('click', closeModal));
    }
    const from = (S.version && S.version.version) || (S.update && S.update.current) || '';
    let tries = 0;
    const timer = setInterval(async () => {
      tries += 1;
      const line = $('#upd-watch');
      if (line) line.textContent = '第 ' + tries + ' 次检查 · ' + new Date().toLocaleTimeString();
      const res = await api('/api/version');
      if (res.ok && res.data && res.data.version && res.data.version !== from) {
        clearInterval(timer);
        S.updating = false;
        S.update = null;
        if (line) line.textContent = '已更新到 v' + res.data.version;
        toast('已更新到 v' + res.data.version + '，正在刷新…', 'ok');
        setTimeout(() => { try { location.reload(); } catch (e) { /* ignore */ } }, 2500);
        return;
      }
      if (tries >= 30) {           // 10s × 30 = 5 分钟
        clearInterval(timer);
        S.updating = false;
        renderUpdateBadge();
        if (line) line.textContent = '等待超时：请手动刷新页面确认版本';
        toast('等待超时，请手动刷新页面', 'err');
      }
    }, 10000);
  }

  /* ------------------------- 设置页：检查更新卡片 ---------------------- */

  function updateCard(u) {
    const cur = (u && u.current) || (S.version && S.version.version) || '—';
    const latest = u && u.latest;
    const newer = isNewer(latest, cur);
    const same = !!latest && cmpVersion(latest, cur) === 0;
    let stateTag;
    if (S.updating) stateTag = '<span class="tag info">更新中…</span>';
    else if (!latest) stateTag = '<span class="tag na">尚未检查</span>';
    else if (newer) stateTag = '<span class="tag warn">有新版本</span>';
    else if (same) stateTag = '<span class="tag ok">已是最新</span>';
    else stateTag = '<span class="tag na">远端更旧，已忽略</span>';

    return card({
      title: '版本与更新', sub: '来自 /api/update',
      body: kv([
        kvItem('当前版本', 'v' + cur),
        kvItem('远端最新', latest ? 'v' + latest : '未知'),
        kvItem('最近检查', (u && u.checked_at) || '未检查'),
        '<div class="item"><span class="k">状态</span><span class="v">' + stateTag + '</span></div>'
      ]) +
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px">' +
        '<button class="btn" id="btn-check-update" type="button">' + ICON('refresh') + '检查更新</button>' +
        (newer
          ? '<button class="btn primary" id="btn-do-update" type="button">' + ICON('download') + '立即更新到 v' + esc(latest) + '</button>'
          : '') +
        '</div>' +
        '<p class="muted" style="margin-top:10px">命令行等价操作：<code>' +
        esc((u && u.command) || 'docker compose pull && docker compose up -d') + '</code></p>'
    });
  }

  /* ======================= 代码 → 名称（展示映射） ===================== */

  /** names[code].name；查不到返回空串，由调用方兜底显示代码（绝不渲染 undefined） */
  function nameOf(code) {
    const raw = String(code === null || code === undefined ? '' : code).trim();
    if (!raw) return '';
    const bare = bareCode(raw);                       // sh600519 -> 600519
    const tries = [raw, raw.toUpperCase(), bare, bare.toUpperCase()];
    for (let i = 0; i < tries.length; i++) {
      const rec = S.names[tries[i]];
      if (rec && rec.name) return String(rec.name);
    }
    return '';
  }

  function industryOf(code) {
    const raw = String(code === null || code === undefined ? '' : code).trim();
    if (!raw) return '';
    const bare = bareCode(raw);
    const tries = [raw, raw.toUpperCase(), bare, bare.toUpperCase()];
    for (let i = 0; i < tries.length; i++) {
      const rec = S.names[tries[i]];
      if (rec && rec.industry) return String(rec.industry);
    }
    return '';
  }

  /** 统一渲染「名称(代码)」；无名称时只显示代码 */
  function codeCell(code) {
    const c = String(code === null || code === undefined ? '' : code).trim();
    if (!c) return '<span class="mono">—</span>';
    const nm = nameOf(c);
    if (!nm || nm === c) return '<span class="mono">' + esc(c) + '</span>';
    return '<span class="name-cell"><span class="nm">' + esc(nm) + '</span>' +
      '<span class="cd">(' + esc(c) + ')</span></span>';
  }

  async function loadNames() {
    const res = await api('/api/names');
    if (res.ok && res.data && res.data.names && typeof res.data.names === 'object') {
      S.names = res.data.names;
      S.namesHint = '';
    } else {
      S.names = {};
      S.namesHint = (res.ok && res.data && res.data.hint) || '';
    }
  }

  const namesEmpty = () => !S.names || !Object.keys(S.names).length;

  /* ===================== 模拟账户配置（可编辑本金） =================== */

  const rawVal = (v) => (isNil(v) ? '' : String(v));

  const fmtMoney = (v) => {
    const n = toNum(v);
    return n === null ? '—' : n.toLocaleString('zh-CN') + ' 元';
  };

  /** 0.15 -> "15"（百分数显示）。仅单位换算，不取整、不纠正 */
  function pctOf(v) {
    const n = toNum(v);
    return n === null ? '' : String(+(n * 100).toFixed(4));
  }

  function accountCardHtml(cfg) {
    const c = cfg || {};
    const sub = 'data/state/account.json · ' +
      (c.updated_at ? '更新于 ' + esc(c.updated_at) : '尚未保存过');
    return card({
      title: '账户设置',
      sub: sub,
      tools: '<span class="tag na" title="未设 APB_TOKEN 时只允许内网修改">内网免 token</span>',
      body: '<form class="form" id="acct-form" autocomplete="off" novalidate>' +
        '<label class="field"><span>初始资金（元）</span>' +
        '<input type="number" name="initial_capital" step="1000" value="' + esc(rawVal(c.initial_capital)) +
        '" placeholder="1000000">' +
        '<span class="hintline" id="acct-cap-hint">' + esc(fmtMoney(c.initial_capital)) + '</span></label>' +
        '<label class="field"><span>持仓数 top_k</span>' +
        '<input type="number" name="top_k" step="1" value="' + esc(rawVal(c.top_k)) + '" placeholder="10"></label>' +
        '<label class="field"><span>单票上限（%）</span>' +
        '<input type="number" name="max_weight_pct" step="1" value="' + esc(pctOf(c.max_weight)) + '" placeholder="15">' +
        '<span class="hintline">填百分数：15 表示 15%</span></label>' +
        '<label class="field"><span>成本 cost_bps</span>' +
        '<input type="number" name="cost_bps" step="1" value="' + esc(rawVal(c.cost_bps)) + '" placeholder="10"></label>' +
        '<label class="field"><span>持有期 hold（交易日）</span>' +
        '<input type="number" name="hold" step="1" value="' + esc(rawVal(c.hold)) + '" placeholder="5"></label>' +
        '<div class="form-actions">' +
        '<button class="btn primary" type="submit">' + ICON('check') + '保存</button>' +
        '<span class="form-msg" id="acct-msg"></span></div>' +
        '</form>' +
        '<p class="acct-note">' + ICON('alert') +
        '<span>改本金只影响<b>下一次引擎运行</b>，不会重算历史净值与台账。</span></p>'
    });
  }

  /** 提交账户配置：越界一律交给后端判定，前端不做自动纠正 */
  async function submitAccount(ev) {
    ev.preventDefault();
    const form = ev.target;
    const msg = $('#acct-msg');
    const read = (field) => {
      const el = form.elements[field];
      if (!el) return null;
      const raw = String(el.value || '').trim();
      if (raw === '') return null;              // 留空 = 不修改这一项
      const n = Number(raw);
      return isFinite(n) ? n : raw;             // 非法输入原样交给后端判定
    };

    const body = {};
    const cap = read('initial_capital');
    const topK = read('top_k');
    const mwPct = read('max_weight_pct');
    const cost = read('cost_bps');
    const hold = read('hold');
    if (cap !== null) body.initial_capital = cap;
    if (topK !== null) body.top_k = topK;
    if (mwPct !== null) body.max_weight = (typeof mwPct === 'number') ? mwPct / 100 : mwPct;
    if (cost !== null) body.cost_bps = cost;
    if (hold !== null) body.hold = hold;

    if (!Object.keys(body).length) {
      msg.className = 'form-msg err';
      msg.textContent = '没有可提交的字段';
      return;
    }

    const btn = form.querySelector('button[type="submit"]');
    if (btn) btn.disabled = true;
    msg.className = 'form-msg';
    msg.textContent = '保存中…';

    const res = await api('/api/account/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body)
    });
    if (btn) btn.disabled = false;

    if (res.ok) {
      S.account = res.data;
      msg.className = 'form-msg ok';
      msg.textContent = '已保存';
      toast('已保存，下次运行引擎生效', 'ok');
      clearCache();
      navigate('paper', false);
    } else {
      // 越界/非法：原样显示后端 detail（不做自动纠正）
      msg.className = 'form-msg err';
      msg.textContent = res.detail || res.hint || ('保存失败：' + (res.error || 'unknown'));
      toast('保存失败：' + (res.detail || res.error || 'unknown'), 'err');
    }
  }

  /* =============================== 模型 ================================ */

  /** sh600519 / SZ000001 -> 600519（仅用于展示与名称查表） */
  function bareCode(code) {
    const c = String(code === null || code === undefined ? '' : code).trim();
    const m = c.match(/^(sh|sz|bj)(\d{6})$/i);
    return m ? m[2] : c;
  }

  /** 推荐行：优先用后端给的 name，其次 names.json，最后只显示代码 */
  function pickCell(row) {
    const code = String((row && row.code) || '').trim();
    const nm = (row && row.name && String(row.name).trim()) || nameOf(code);
    const bare = bareCode(code);
    if (!nm || nm === code) return '<span class="mono">' + esc(bare || '—') + '</span>';
    return '<span class="name-cell"><span class="nm">' + esc(nm) + '</span>' +
      '<span class="cd">(' + esc(bare) + ')</span></span>';
  }

  /** 回测四项是否还是占位（全 0 / 缺失 → 视为「待回测」） */
  function backtestPending(bt) {
    if (!bt || typeof bt !== 'object') return true;
    const keys = ['annual', 'sharpe', 'mdd', 'turnover'];
    return keys.every((k) => {
      const v = toNum(bt[k]);
      return v === null || v === 0;
    });
  }

  async function pageModel() {
    const [modelRes, picksRes, impRes] = await Promise.all([
      apiCached('/api/model'),
      apiCached('/api/picks?limit=500'),
      apiCached('/api/factors/importance?top=200')
    ]);

    const m = (modelRes.ok && modelRes.data) ? modelRes.data : null;
    const picks = (picksRes.ok && picksRes.data) ? picksRes.data : null;
    const imp = (impRes.ok && impRes.data) ? impRes.data : null;

    if (!m) {
      setSub('模型尚未训练');
      return notReadyBox(modelRes, '模型元信息缺失') + '<div style="height:14px"></div>' + banner({
        tone: 'warn', icon: 'alert',
        title: '模型尚未训练，请先运行回测脚本',
        text: '期望 <code>data/state/model.json</code>（模型元信息）、<code>picks.json</code>（推荐持仓）、' +
          '<code>factor_importance.json</code>（因子重要性）。三者缺失时本页只显示占位，其他页面不受影响。',
        meta: '契约：{model, version, n_factors, n_trees, train_period, test_period, ic, ic_t, rank_ic, universe, n_stocks, backtest{...}}'
      });
    }

    setSub('v' + (m.version || '—') + ' · ' + (m.universe || '—') + ' · ' +
      (m.n_factors || '—') + ' 因子 × ' + (m.n_trees || '—') + ' 棵树');

    /* --- 顶部模型卡 --- */
    const ic = toNum(m.ic);
    const icT = toNum(m.ic_t);
    const ric = toNum(m.rank_ic);
    const icCls = ic === null ? '' : ic > 0 ? 'down-text' : ic < 0 ? 'up-text' : '';

    const heroCard = card({
      title: (m.model || '模型') + (m.version ? ' · ' + m.version : ''),
      sub: m.updated_at ? '更新于 ' + esc(m.updated_at) : '',
      tools: '<span class="tag info">' + esc(m.universe || '—') + '</span>' +
        '<span class="tag na">' + esc(num(m.n_stocks, 0)) + ' 只</span>',
      body:
        '<div class="grid two" style="gap:18px;align-items:center">' +
        '<div>' + kv([
          kvItem('训练期', m.train_period || '—'),
          kvItem('样本外', m.test_period || '—'),
          kvItem('因子数', num(m.n_factors, 0)),
          kvItem('树数', num(m.n_trees, 0))
        ]) + '</div>' +
        '<div class="hero-metrics">' +
        '<div class="hm"><span class="hm-k">IC</span>' +
        '<span class="hm-v ' + icCls + '">' + (ic === null ? '—' : (ic > 0 ? '+' : '') + num(ic, 4)) + '</span></div>' +
        '<div class="hm"><span class="hm-k">t 值</span>' +
        '<span class="hm-v ' + icCls + '">' + (icT === null ? '—' : (icT > 0 ? '+' : '') + num(icT, 2)) + '</span></div>' +
        '<div class="hm"><span class="hm-k">RankIC</span>' +
        '<span class="hm-v ' + (ric === null ? '' : ric > 0 ? 'down-text' : ric < 0 ? 'up-text' : '') + '">' +
        (ric === null ? '—' : (ric > 0 ? '+' : '') + num(ric, 4)) + '</span></div>' +
        '</div></div>'
    });

    /* --- 回测摘要（6 个 KPI） --- */
    const bt = m.backtest || {};
    const pending = backtestPending(bt);
    const btKpis = [
      kpi({ k: '净年化', v: pending ? '—' : pct(bt.annual), cls: pending ? '' : signClass(bt.annual), tone: pending ? 'neutral' : (toNum(bt.annual) > 0 ? 'good' : 'bad') }),
      kpi({ k: '夏普', v: pending ? '—' : num(bt.sharpe, 2), tone: pending ? 'neutral' : 'brand' }),
      kpi({ k: '最大回撤', v: pending ? '—' : pct(bt.mdd), tone: pending ? 'neutral' : 'warn' }),
      kpi({ k: '平均换手', v: pending ? '—' : pct(bt.turnover), tone: pending ? 'neutral' : 'brand' }),
      kpi({ k: '持有期', v: num(bt.hold_days, 0) + ' 日', sm: true, tone: 'neutral' }),
      kpi({ k: '持仓数', v: num(bt.top_k, 0) + ' 只', sm: true, tone: 'neutral' })
    ].join('');
    const btCard = card({
      title: '回测摘要', sub: 'backtest · 样本外',
      tools: pending ? '<span class="tag warn">待回测</span>' : '<span class="tag ok">已回测</span>',
      body: '<div class="grid four" style="gap:12px">' + btKpis + '</div>'
    });

    /* --- 推荐持仓 --- */
    const rows = (picks && Array.isArray(picks.rows)) ? picks.rows : [];
    const picksCard = card({
      title: '当前推荐持仓',
      sub: picks
        ? 'as_of ' + esc(picks.as_of || '—') + ' · 共 ' + esc(num(picks.n, 0)) + ' 只' +
          (picks.top_k ? ' · top_k ' + esc(num(picks.top_k, 0)) : '') +
          (picks.hold_days ? ' · 持有 ' + esc(num(picks.hold_days, 0)) + ' 日' : '')
        : '未就绪',
      cls: 'flush',
      tools: '<span class="tag ok">持有中 = 缓冲带内不动</span>',
      body: '<div style="padding:0 16px 16px">' + table({
        cols: [
          { label: '#', key: 'rank', num: true, render: (r) => esc(num(r.rank, 0)) },
          { label: '名称（代码）', key: 'code', render: (r) => pickCell(r) },
          {
            label: '分数', key: 'score', num: true,
            render: (r) => '<span class="' + signClass(r.score) + '">' + esc(num(r.score, 4)) + '</span>'
          },
          {
            label: '涨跌', key: 'chg_pct', num: true,
            render: (r) => '<span class="' + signClass(r.chg_pct) + '">' + esc(signedPct(r.chg_pct)) + '</span>'
          },
          { label: '权重', key: 'weight', num: true, render: (r) => esc(pct(r.weight)) },
          {
            label: '状态', key: 'held',
            render: (r) => (r.held === true
              ? '<span class="tag hit">持有中</span>'
              : '<span class="tag info">新进</span>')
          },
          { label: '已持有', key: 'days_held', num: true, render: (r) => (isNil(r.days_held) ? '—' : esc(num(r.days_held, 0)) + ' 日') }
        ],
        rows: rows,
        empty: picks
          ? emptyBox('暂无推荐', 'picks.rows 为空。')
          : notReadyBox(picksRes, '推荐持仓缺失')
      }) + (picks && picks.note
        ? '<p class="muted" style="margin-top:8px">' + esc(picks.note) + '</p>' : '') + '</div>'
    });

    /* --- 因子重要性 top 20 --- */
    const impRows = (imp && Array.isArray(imp.rows)) ? imp.rows.filter((r) => r && !isNil(r.factor)) : [];
    const top20 = impRows.slice().sort((a, b) => (toNum(b.importance) || 0) - (toNum(a.importance) || 0)).slice(0, 20);
    const impCard = card({
      title: '因子重要性 Top 20', sub: imp ? 'GBDT feature_importance · 共 ' + esc(num(imp.n, 0)) + ' 个因子' : '未就绪',
      body: top20.length
        ? hbars(top20.map((r) => ({
          label: r.factor,
          value: r.importance,
          title: r.factor + ' · ' + num(r.importance, 4)
        })), { fmt: (v) => num(v, 4) })
        : notReadyBox(impRes, '因子重要性缺失')
    });

    return '<div style="margin-bottom:14px">' + heroCard + '</div>' +
      '<div style="margin-bottom:14px">' + btCard + '</div>' +
      '<div style="margin-bottom:14px">' + picksCard + '</div>' +
      impCard;
  }

  /* --------------------- 总览页：模型状态条 --------------------------- */

  async function modelBarHtml(modelRes) {
    const m = (modelRes && modelRes.ok && modelRes.data) ? modelRes.data : null;
    if (!m) {
      return '<button type="button" class="model-bar is-empty" data-page="model">' +
        '<span class="mb-name">模型</span><span class="mb-sep">·</span>' +
        '<span class="mb-ic">尚未就绪</span><span class="mb-sep">·</span>' +
        '<span class="mb-oos">等待训练 / 打分</span>' +
        '<span class="mb-go">查看 →</span></button>';
    }
    const ic = toNum(m.ic);
    const icT = toNum(m.ic_t);
    const icCls = ic === null ? 'is-na' : ic > 0 ? 'is-pos' : ic < 0 ? 'is-neg' : 'is-na';
    const icTxt = ic === null ? 'IC —' : 'IC ' + (ic > 0 ? '+' : '') + num(ic, 4) +
      (icT === null ? '' : ' (t=' + (icT > 0 ? '+' : '') + num(icT, 2) + ')');
    return '<button type="button" class="model-bar" data-page="model">' +
      '<span class="mb-name">' + esc(m.model || '模型') + '</span><span class="mb-sep">·</span>' +
      '<span class="mb-ic ' + icCls + '">' + esc(icTxt) + '</span><span class="mb-sep">·</span>' +
      '<span class="mb-oos">样本外 ' + esc(m.test_period || '—') + '</span>' +
      '<span class="mb-go">查看推荐 →</span></button>';
  }

  /* =============================== 总览 ================================ */

  async function pageOverview() {
    const [perfRes, regimeRes, paperRes, statusRes, modelRes] = await Promise.all([
      apiCached('/api/performance'), apiCached('/api/regime'),
      apiCached('/api/paper'), apiCached('/api/status'), apiCached('/api/model')
    ]);
    const perf = perfRes.ok ? adaptPerformance(perfRes.data) : null;
    const status = statusRes.ok ? statusRes.data : null;
    const regimeRows = (regimeRes.ok && regimeRes.data && regimeRes.data.rows) || [];
    const paper = (paperRes.ok && paperRes.data) || null;

    setSub(status
      ? (status.data_from || '—') + ' → ' + (status.data_to || '—') + ' · ' +
        (status.n_stocks || '—') + ' 只 × ' + (status.n_signals || '—') + ' 信号 · h=' + (status.horizon || '—')
      : (perfRes.error === 'not_ready' ? '未就绪：等待引擎首次运行' : '数据不可用'));

    /* --- 4 个 KPI --- */
    const lastReg = regimeRows.length ? regimeRows[regimeRows.length - 1] : null;
    const regInfo = regimeInfo(lastReg && lastReg.regime);
    const regUp = lastReg && toNum(lastReg.regime) >= 2;

    const conf = perf && perf.confidence;
    const edge5 = perf && perf.edge && (perf.edge['5'] || perf.edge[5]);
    const icMean = edge5 && edge5.ic ? toNum(edge5.ic.mean) : null;
    const icT = edge5 && edge5.ic ? toNum(edge5.ic.t) : null;
    const costs = perf && perf.costs;
    const drag = costs && costs.drag ? toNum(costs.drag.annual_cost_pct) : null;

    const kpis = [
      kpi({
        k: '市场状态 regime', v: regInfo.name, tone: lastReg ? (regUp ? 'bad' : 'good') : 'neutral', sm: true,
        s: lastReg ? esc(lastReg.date || '') + ' · 0跌低 / 1跌高 / 2涨低 / 3涨高' : '暂无序列'
      }),
      kpi({
        k: '可信度 Brier', v: conf ? num(conf.brier, 4) : '—', tone: 'neutral',
        s: conf ? 'ECE ' + num(conf.ece, 4) + ' · skill ' + num(conf.brier_skill_vs_base, 4) : '未就绪'
      }),
      kpi({
        k: 'Rank IC (h=5)', v: icMean === null ? '—' : num(icMean, 4),
        cls: signClass(icMean), tone: icMean === null ? 'neutral' : (icMean > 0 ? 'good' : 'bad'),
        s: icT === null ? '未就绪'
          : 'HAC t = ' + num(icT, 2) + (edge5 && edge5.ic_verdict ? ' · ' + esc(edge5.ic_verdict) : '')
      }),
      kpi({
        k: '成本拖累 / 年', v: drag === null ? '—' : num(drag, 2) + '%',
        tone: drag === null ? 'neutral' : 'warn',
        s: costs ? esc(costs.market) + ' · 双边 ' + num(costs.round_trip_bps, 2) + ' bps' : '未就绪'
      })
    ].join('');

    /* --- 结论横幅（字段全部来自引擎判定，前端只做取舍与配色） --- */
    let bannerHtml = '';
    if (perf && (perf.meta || perf.learned)) {
      const lv = perf.learned || {};
      const tone = (icMean !== null && icMean < 0) ? 'bad' : (icMean !== null && icMean > 0.02 ? 'good' : 'warn');
      bannerHtml = banner({
        tone: tone,
        icon: tone === 'good' ? 'check' : 'alert',
        title: (perf.meta && perf.meta.verdict) || '当前结论未标注',
        text: '学习器 IC ' + num(lv.learned_ic, 4) + '（t=' + num(lv.learned_t, 2) + '） vs 等权 IC ' +
          num(lv.equal_ic, 4) + '（t=' + num(lv.equal_t, 2) + '） · 差 ' + num(lv.diff_ic, 4) +
          '（p=' + num(lv.diff_p, 3) + '）' + (lv.verdict ? '<br>' + esc(lv.verdict) : ''),
        meta: 'DSR ' + num(edge5 && edge5.long_short && edge5.long_short.dsr, 4) +
          ' · 相关性有效信号 ' + num(perf.risk && perf.risk.signal_level &&
            perf.risk.signal_level.corr_adjusted && perf.risk.signal_level.corr_adjusted.effective_n_corr, 1) +
          ' · 生成于 ' + ((perfRes.data && perfRes.data.generated_at) || '—')
      });
    } else if (!perf) {
      bannerHtml = notReadyBox(perfRes);
    }

    /* --- 净值曲线：优先多空 equity，其次模拟/实盘净值 --- */
    let navCard;
    if (perf && perf.curve && perf.curve.length) {
      navCard = card({
        title: '多空组合净值', sub: 'backtest.long_short.equity',
        body: '<div class="chart">' + chartSVG(
          [{ values: perf.curve, color: 'var(--brand)', area: true, label: '多空净值' }],
          { height: 260, labels: perf.labels || {}, aria: '多空组合净值曲线' }) + '</div>'
      });
    } else if (paper && Array.isArray(paper.sim_nav)) {
      const dates = Array.isArray(paper.dates) ? paper.dates : [];
      navCard = card({
        title: '净值曲线', sub: 'paper.json · sim_nav / live_nav',
        body: '<div class="chart">' + chartSVG([
          { values: seriesValues(paper.sim_nav), color: 'var(--brand)', label: '模拟盘' },
          { values: seriesValues(paper.live_nav), color: 'var(--warn)', label: '实盘' }
        ], {
          height: 260, tipLabels: dates,
          labels: { first: dates[0] || '', last: dates[dates.length - 1] || '' },
          aria: '模拟盘与实盘净值曲线'
        }) + '</div><p class="muted" style="margin-top:8px">引擎未导出多空 equity 序列，此处展示模拟/实盘净值。</p>'
      });
    } else {
      navCard = card({ title: '净值曲线', body: notReadyBox(perfRes.ok ? paperRes : perfRes) });
    }

    /* --- regime 色带 --- */
    const bd = band(regimeRows);
    const bandCard = card({
      title: '市场状态 regime', sub: regimeRows.length + ' 个交易日',
      body: bd.html + bd.legend
    });

    /* --- 分年度 IC（真实字段：edge[h].subperiods） --- */
    let subperiodCard = '';
    if (edge5 && Array.isArray(edge5.subperiods) && edge5.subperiods.length) {
      const rows = edge5.subperiods.map((s) => ({
        label: String(s.year || '—'), value: toNum(s.mean),
        title: 'n=' + s.n + ' t=' + num(s.t, 2)
      }));
      subperiodCard = card({
        title: '分年度 IC（h=5）', sub: 'edge["5"].subperiods · 悬停看 t 值',
        body: dbars(rows, { fmt: (v) => num(v, 4) })
      });
    }

    /* --- 命中率 / IC 明细 --- */
    const hitItems = perf && perf.hits.length
      ? perf.hits.map((it) => kvItem(it.label + ' 命中率', pct(it.value)))
      : [];
    const icItems = perf && perf.ic.length
      ? perf.ic.map((it) => kvItem(it.label,
        it.kind === 'pct' ? pct(it.value) : num(it.value, 4),
        it.kind === 'pct' ? '' : signClass(it.value)))
      : [];
    const detailCard = card({
      title: '命中率与 IC 明细', sub: 'metrics.hit_rate / metrics.rank_ic',
      body: (hitItems.length || icItems.length)
        ? kv(hitItems) + '<div style="height:12px"></div>' + kv(icItems)
        : emptyBox('暂无明细', 'performance.json 里没有 hit_rate / rank_ic。')
    });

    return '<div style="margin-bottom:14px">' + (await modelBarHtml(modelRes)) + '</div>' +
      '<div class="grid four" style="margin-bottom:14px">' + kpis + '</div>' +
      '<div style="margin-bottom:14px">' + bannerHtml + '</div>' +
      '<div class="grid two" style="margin-bottom:14px">' + navCard + bandCard + '</div>' +
      '<div class="grid two">' + (subperiodCard || detailCard) + detailCard + '</div>';
  }

  /* =============================== 市场 ================================ */

  // 情绪分档 → 配色（只做「等级文字 → 颜色」的映射，温度数值原样来自 /api/market）
  const TEMP_LEVELS = {
    '冰点': { cls: 't-ice', color: '#3b82f6' },
    '偏冷': { cls: 't-cold', color: 'var(--down)' },
    '中性': { cls: 't-neutral', color: '#94a3b8' },
    '偏热': { cls: 't-warm', color: 'var(--warn)' },
    '过热': { cls: 't-hot', color: 'var(--up)' }
  };

  /** level 文字优先；没有 level 时才按温度区间兜底（纯展示映射） */
  function tempTone(level, temperature) {
    if (level && TEMP_LEVELS[level]) return TEMP_LEVELS[level];
    const t = toNum(temperature);
    if (t === null) return { cls: 'temp-unknown', color: 'var(--text-3)' };
    if (t < 20) return TEMP_LEVELS['冰点'];
    if (t < 40) return TEMP_LEVELS['偏冷'];
    if (t < 60) return TEMP_LEVELS['中性'];
    if (t < 80) return TEMP_LEVELS['偏热'];
    return TEMP_LEVELS['过热'];
  }

  /** 从 ISO 时间戳取 HH:MM:SS（字符串切片，不做时区换算） */
  function clockOf(iso) {
    const str = String(iso || '');
    const m = str.match(/T(\d{2}:\d{2}:\d{2})/);
    return m ? m[1] : (str || '—');
  }

  function indexCard(it) {
    const chg = toNum(it.chg_pct);
    const cls = chg === null ? 'is-flat' : chg > 0 ? 'is-up' : chg < 0 ? 'is-down' : 'is-flat';
    const arrow = chg === null ? '' : chg > 0 ? '▲ ' : chg < 0 ? '▼ ' : '— ';
    const chgTxt = chg === null ? '—' : (chg > 0 ? '+' : '') + num(chg, 2) + '%';
    const pts = toNum(it.change);
    const ptsTxt = pts === null ? '' : (pts > 0 ? '+' : '') + num(pts, 2);
    return '<div class="idx ' + cls + '" title="' + esc((it.name || '') + ' ' + (it.code || '')) + '">' +
      '<div class="idx-name"><span>' + esc(it.name || '—') + '</span>' +
      '<span class="idx-code">' + esc(it.code || '') + '</span></div>' +
      '<div class="idx-price">' + esc(num(it.price, 2)) + '</div>' +
      '<div class="idx-chg">' + esc(arrow + chgTxt) + (ptsTxt ? ' · ' + esc(ptsTxt) : '') + '</div>' +
      '</div>';
  }

  async function pageMarket() {
    const res = await apiCached('/api/market', 0);
    if (!res.ok || !res.data) {
      setSub('采集未启动');
      return notReadyBox(res) + '<div style="height:14px"></div>' + banner({
        tone: 'warn', icon: 'alert',
        title: '行情采集尚未启动',
        text: '期望 <code>data/state/market.json</code>。启动采集后本页每 60 秒自动刷新，无需手动操作。',
        meta: '字段：as_of / indices[] / breadth / limit_up / amount_yi / sentiment'
      });
    }

    const d = res.data;
    const indices = Array.isArray(d.indices) ? d.indices : [];
    const b = (d.breadth && typeof d.breadth === 'object') ? d.breadth : {};
    const sent = (d.sentiment && typeof d.sentiment === 'object') ? d.sentiment : {};
    const missing = Array.isArray(d.missing) ? d.missing : [];

    // 温度：ok=false 时显示「—」，绝不显示 0
    const tempOk = sent.ok === true && toNum(sent.temperature) !== null;
    const tempVal = toNum(sent.temperature);
    const tone = tempTone(sent.level, tempOk ? tempVal : null);
    const width = tempOk ? Math.max(0, Math.min(100, tempVal)) : 0;

    setSub('采集于 ' + clockOf(d.as_of) +
      (toNum(d.elapsed_ms) !== null ? ' · 耗时 ' + num(d.elapsed_ms, 0) + 'ms' : '') +
      ' · 每 60 秒自动刷新');

    /* --- 情绪温度 --- */
    const tempCard = card({
      title: '情绪温度', sub: 'sentiment.temperature · 0–100',
      cls: tone.cls,
      body:
        '<div class="big-num">' + (tempOk ? esc(num(tempVal, 1)) : '—') +
        (tempOk ? '<span class="unit">/100</span>' : '') + '</div>' +
        '<div style="margin-top:4px">' +
        (tempOk && sent.level ? '<span class="tag info">' + esc(sent.level) + '</span>' : '') +
        (!tempOk ? '<span class="tag na">暂不计算</span>' : '') + '</div>' +
        '<div class="temp-scale">' +
        (tempOk ? '<i style="width:' + width.toFixed(1) + '%"></i>' : '') +
        '<b style="left:25%"></b><b style="left:50%"></b><b style="left:75%"></b></div>' +
        '<div class="temp-ticks"><span>0 冰点</span><span>25</span><span>50 中性</span><span>75</span><span>100 过热</span></div>' +
        (!tempOk
          ? '<p class="muted" style="margin-top:9px">' + esc(sent.reason || '历史样本不足，温度暂不可用') + '</p>'
          : (sent.parts
            ? '<p class="muted" style="margin-top:9px">分项（引擎原始值）：涨跌比 ' + num(sent.parts.up_ratio_pct, 2) +
              ' · 涨停 ' + num(sent.parts.limit_up_pct, 2) + ' · 成交额 ' + num(sent.parts.amount_pct, 2) + '</p>'
            : ''))
    });

    /* --- 涨停 / 成交额 --- */
    const limitCard = card({
      title: '涨停家数', sub: 'limit_up',
      body: '<div class="big-num">' + esc(num(d.limit_up, 0)) + '<span class="unit">家</span></div>' +
        '<p class="muted" style="margin-top:6px">当日封板家数（含连板）</p>'
    });
    const amountCard = card({
      title: '两市成交额', sub: 'amount_yi · 亿元',
      body: '<div class="big-num">' + esc(num(d.amount_yi, 1)) + '<span class="unit">亿元</span></div>' +
        '<p class="muted" style="margin-top:6px">沪深两市合计</p>'
    });

    /* --- 市场宽度：flex-grow 让 CSS 分配比例，JS 不做除法 --- */
    const up = toNum(b.up);
    const down = toNum(b.down);
    const flat = toNum(b.flat);
    const hasBreadth = up !== null || down !== null || flat !== null;
    const breadthCard = card({
      title: '市场宽度', sub: 'breadth · 上涨 / 平盘 / 下跌 家数占比',
      body: hasBreadth
        ? '<div class="breadth-bar">' +
          '<i class="b-up" style="flex:' + (up || 0) + ' 1 0" title="上涨 ' + (up || 0) + ' 家"></i>' +
          '<i class="b-flat" style="flex:' + (flat || 0) + ' 1 0" title="平盘 ' + (flat || 0) + ' 家"></i>' +
          '<i class="b-down" style="flex:' + (down || 0) + ' 1 0" title="下跌 ' + (down || 0) + ' 家"></i>' +
          '</div>' +
          '<div class="breadth-legend">' +
          '<span class="bl"><i class="sw" style="background:var(--up)"></i>上涨 <span class="num">' + esc(num(up, 0)) + '</span></span>' +
          '<span class="bl"><i class="sw" style="background:var(--text-3);opacity:.45"></i>平盘 <span class="num">' + esc(num(flat, 0)) + '</span></span>' +
          '<span class="bl"><i class="sw" style="background:var(--down)"></i>下跌 <span class="num">' + esc(num(down, 0)) + '</span></span>' +
          '<span class="bl">上涨占比 <span class="num">' + esc(pct(b.up_ratio)) + '</span></span>' +
          '</div>' +
          (b.by_market && typeof b.by_market === 'object'
            ? '<div class="market-foot">' + Object.keys(b.by_market).map((k) => {
              const m = b.by_market[k] || {};
              return '<span>' + esc(k) + ' 涨 ' + esc(num(m.up, 0)) + ' / 平 ' + esc(num(m.flat, 0)) +
                ' / 跌 ' + esc(num(m.down, 0)) + '</span>';
            }).join('') + '</div>'
            : '')
        : emptyBox('暂无宽度数据', 'breadth 字段缺失。')
    });

    /* --- 指数网格 --- */
    const idxCard = card({
      title: '指数行情', sub: '共 ' + indices.length + ' 个',
      body: indices.length
        ? '<div class="idx-grid">' + indices.map(indexCard).join('') + '</div>'
        : emptyBox('暂无指数数据', 'indices 为空。')
    });

    /* --- 采集脚注 --- */
    const foot = '<div class="market-foot">' +
      '<span>as_of ' + esc(d.as_of || '—') + '</span>' +
      '<span>最后更新 ' + esc(clockOf(d.as_of)) + '</span>' +
      (toNum(d.elapsed_ms) !== null ? '<span>耗时 ' + esc(num(d.elapsed_ms, 0)) + ' ms</span>' : '') +
      (missing.length ? '<span style="color:var(--warn)">缺 ' + esc(missing.join(', ')) + '</span>' : '<span>数据完整</span>') +
      '</div>';

    return '<div class="grid" style="margin-bottom:14px">' + tempCard + limitCard + amountCard + '</div>' +
      '<div style="margin-bottom:14px">' + breadthCard + '</div>' +
      idxCard + foot;
  }

  /** 市场页自动刷新：只替换视图，不重置滚动位置 */
  async function refreshMarket() {
    if (S.page !== 'market' || document.hidden) return;
    try {
      const html = await pageMarket();
      if (S.page !== 'market') return;
      $('#view').innerHTML = html;
    } catch (e) { /* 静默失败，下个周期再试 */ }
  }

  /* =============================== 信号 ================================ */

  async function pageSignals() {
    const res = await apiCached('/api/signals?top=200');
    if (!res.ok || !res.data) { setSub('未就绪'); return notReadyBox(res); }

    let rows = (res.data.top || []).concat(res.data.bottom || []);
    const seen = new Set();
    rows = rows.filter((r) => {
      const k = r && r.sid;
      if (!k || seen.has(k)) return false;
      seen.add(k);
      return true;
    });

    const families = Array.from(new Set(rows.map((r) => r.family).filter(Boolean))).sort();
    setSub('共 ' + (res.data.total || rows.length) + ' 个信号 · ' + families.length + ' 个族');

    const fam = S.filter.family;
    const q = (S.filter.q || '').trim().toLowerCase();
    const view = rows.filter((r) => {
      if (fam !== 'all' && r.family !== fam) return false;
      if (q && !(String(r.sid).toLowerCase().includes(q) || String(r.family).toLowerCase().includes(q))) return false;
      return true;
    });

    const s = S.sort.signals;
    const sorted = view.slice().sort((a, b) => {
      const va = toNum(a[s.key]);
      const vb = toNum(b[s.key]);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      return (va - vb) * s.dir;
    });

    const topN = sorted.slice(0, 20);
    const botN = sorted.slice(-20).reverse();
    const maxTop = toNum(topN[0] && topN[0].weight) || 0;

    const barsCard = card({
      title: '权重 Top 20 / Bottom 20', sub: 'weights.json · 条形长度 = 相对最大权重',
      body: '<div class="grid two" style="gap:18px">' +
        '<div><div class="card-head" style="margin-bottom:8px"><h3 style="font-size:12px;color:var(--up)">Top 20（权重最高）</h3></div>' +
        hbars(topN.map((r) => ({
          label: r.sid, value: r.weight, sid: r.sid, sel: S.selSid === r.sid,
          title: r.sid + ' · ' + (r.family || '') + ' · 命中率 ' + pct(r.hit_rate) + ' · 更新 ' + num(r.n_updates, 0) + ' 次'
        })), { fmt: (v) => num(v, 4) }) + '</div>' +
        '<div><div class="card-head" style="margin-bottom:8px"><h3 style="font-size:12px;color:var(--down)">Bottom 20（权重最低）</h3></div>' +
        hbars(botN.map((r) => ({
          label: r.sid, value: r.weight, sid: r.sid, cls: 'neg', sel: S.selSid === r.sid,
          title: r.sid + ' · ' + (r.family || '') + ' · 命中率 ' + pct(r.hit_rate) + ' · 更新 ' + num(r.n_updates, 0) + ' 次'
        })), { fmt: (v) => num(v, 4), max: maxTop }) + '</div>' +
        '</div><p class="muted" style="margin-top:10px">点任意条形查看该信号明细（表格同步高亮）。</p>'
    });

    const sel = rows.filter((r) => r.sid === S.selSid)[0];
    const detail = sel ? card({
      title: '信号明细', sub: esc(sel.sid),
      body: kv([
        kvItem('家族 family', sel.family || '—'),
        kvItem('权重 weight', num(sel.weight, 4)),
        kvItem('命中率 hit_rate', pct(sel.hit_rate)),
        kvItem('更新次数 n_updates', num(sel.n_updates, 0)),
        kvItem('平均收益 mean_payoff', num(sel.mean_payoff, 4), signClass(sel.mean_payoff)),
        kvItem('发声次数 alarms', num(sel.alarms, 0))
      ])
    }) : '';

    const tableCard = card({
      title: '信号全表', sub: '点表头排序', cls: 'flush',
      tools: '<label class="field inline"><span>族</span><select id="f-family">' +
        '<option value="all">全部（' + rows.length + '）</option>' +
        families.map((f) => '<option value="' + esc(f) + '"' + (fam === f ? ' selected' : '') + '>' +
          esc(f) + '</option>').join('') + '</select></label>' +
        '<label class="field inline"><span>搜索</span><input id="f-q" type="search" placeholder="sid / family" value="' +
        esc(S.filter.q) + '"></label>' +
        '<span class="muted">' + sorted.length + ' 行</span>',
      body: '<div style="padding:0 16px 16px">' + table({
        cols: [
          { label: '信号 ID', key: 'sid', mono: true, sort: 'sid' },
          { label: '族', key: 'family', render: (r) => '<span class="tag info">' + esc(r.family || '—') + '</span>' },
          { label: '权重', key: 'weight', num: true, sort: 'weight', render: (r) => esc(num(r.weight, 5)) },
          { label: '命中率', key: 'hit_rate', num: true, sort: 'hit_rate', render: (r) => esc(pct(r.hit_rate)) },
          { label: '更新次数', key: 'n_updates', num: true, sort: 'n_updates', render: (r) => esc(num(r.n_updates, 0)) },
          {
            label: '平均收益', key: 'mean_payoff', num: true, sort: 'mean_payoff',
            render: (r) => '<span class="' + signClass(r.mean_payoff) + '">' + esc(num(r.mean_payoff, 4)) + '</span>'
          },
          { label: '发声', key: 'alarms', num: true, sort: 'alarms', render: (r) => esc(num(r.alarms, 0)) }
        ],
        rows: sorted.slice(0, 500),
        sort: S.sort.signals,
        empty: emptyBox('没有匹配的信号', '换个族或清空搜索词。')
      }) + (sorted.length > 500 ? '<p class="muted" style="margin-top:8px">仅渲染前 500 行（共 ' + sorted.length + ' 行）</p>' : '') + '</div>'
    });

    const legacyBanner = banner({
      tone: 'warn', icon: 'alert',
      title: '这一页是旧版手写信号（实测 IC −0.0185，无预测能力），保留仅供参考',
      text: '有效模型请见 <button type="button" class="link-btn" data-page="model">模型</button> 页' +
        '（Alpha158+GBDT，IC +0.0260，样本外验证通过）。',
      meta: '73 个信号 · 相关性调整后有效独立信号约 12 个'
    });

    return '<div style="margin-bottom:14px">' + legacyBanner + '</div>' +
      '<div style="margin-bottom:14px">' + barsCard + '</div>' +
      (detail ? '<div style="margin-bottom:14px">' + detail + '</div>' : '') + tableCard;
  }

  /* ============================== 预测台账 ============================= */

  const isHit = (r) => r.hit === true || r.hit === 1 || r.hit === '1' || r.hit === 'true' || r.hit === 'True';

  function verdictTag(r) {
    const st = String(r.status || '').toLowerCase();
    if (st === 'pending' || isNil(r.hit)) return '<span class="tag pending">pending</span>';
    return isHit(r) ? '<span class="tag hit">命中</span>' : '<span class="tag miss">未中</span>';
  }

  async function pageLedger() {
    const res = await apiCached('/api/ledger?days=' + S.filter.days);
    if (!res.ok || !res.data) { setSub('未就绪'); return notReadyBox(res); }

    const all = (res.data.rows || []).filter((r) => r && typeof r === 'object');
    const q = (S.filter.q || '').trim().toUpperCase();
    const from = S.filter.from;
    const to = S.filter.to;
    const rf = S.filter.result;

    const rows = all.filter((r) => {
      const d = String(r.as_of || '');
      if (from && d < from) return false;
      if (to && d > to) return false;
      if (q && !String(r.code || '').toUpperCase().includes(q)) return false;
      const pending = String(r.status || '').toLowerCase() === 'pending' || isNil(r.hit);
      if (rf === 'pending') return pending;
      if (rf === 'hit') return !pending && isHit(r);
      if (rf === 'miss') return !pending && !isHit(r);
      return true;
    });

    setSub('最近 ' + S.filter.days + ' 个交易日 · 共 ' + all.length + ' 条 · 筛选后 ' + rows.length + ' 条');

    return card({
      title: '预测台账', sub: '台账只追加、不回改；pending = h 日未到，尚未对账', cls: 'flush',
      tools: '<label class="field inline"><span>最近</span><select id="f-days">' +
        [10, 30, 60, 120].map((d) => '<option value="' + d + '"' + (S.filter.days === d ? ' selected' : '') + '>' +
          d + ' 天</option>').join('') + '</select></label>' +
        '<div class="segmented" id="f-result">' +
        [['all', '全部'], ['hit', '命中'], ['miss', '未中'], ['pending', '待定']].map(([v, l]) =>
          '<button type="button" class="seg' + (rf === v ? ' is-active' : '') + '" data-f="' + v + '">' + l + '</button>').join('') +
        '</div>',
      body: '<div style="padding:0 16px 4px"><div class="form" style="gap:10px;margin-bottom:12px">' +
        '<label class="field"><span>代码搜索</span><input id="f-code" type="search" placeholder="600519 / 688981" value="' + esc(S.filter.q) + '"></label>' +
        '<label class="field"><span>起（as_of ≥）</span><input id="f-from" type="date" value="' + esc(S.filter.from) + '"></label>' +
        '<label class="field"><span>止（as_of ≤）</span><input id="f-to" type="date" value="' + esc(S.filter.to) + '"></label>' +
        '</div></div>' +
        '<div style="padding:0 16px 16px">' + table({
          cols: [
            { label: '决策日', key: 'as_of', mono: true },
            { label: '名称（代码）', key: 'code', render: (r) => codeCell(r.code) },
            { label: '综合分', key: 'score', num: true, render: (r) => esc(num(r.score, 4)) },
            { label: 'regime', key: 'regime', render: (r) => esc(regimeInfo(r.regime).name) },
            { label: '入场价', key: 'entry_price', num: true, render: (r) => esc(num(r.entry_price, 2)) },
            { label: '出场价', key: 'exit_price', num: true, render: (r) => esc(num(r.exit_price, 2)) },
            { label: '收益', key: 'ret', num: true, render: (r) => '<span class="' + signClass(r.ret) + '">' + esc(signedPct(r.ret)) + '</span>' },
            { label: '结果', key: 'hit', render: verdictTag }
          ],
          rows: rows.slice(0, 500),
          empty: emptyBox('没有匹配的记录', '试试放宽日期范围或清空代码搜索。')
        }) + (rows.length > 500 ? '<p class="muted" style="margin-top:8px">仅渲染最近 500 行（共 ' + rows.length + ' 行）。</p>' : '') +
        '</div>'
    });
  }

  /* ============================== 模拟实盘 ============================= */

  async function pagePaper() {
    const [res, tradesRes, acctRes] = await Promise.all([
      apiCached('/api/paper'), apiCached('/api/paper/trades?limit=20'),
      apiCached('/api/account/config', 0)
    ]);
    const trades = (tradesRes.ok && tradesRes.data) ? tradesRes.data : { rows: [], n: 0 };
    const acct = (acctRes.ok && acctRes.data) ? acctRes.data : null;

    let body = '';
    if (!res.ok || !res.data) {
      setSub('未就绪');
      body = notReadyBox(res);
    } else {
      const p = res.data;
      const cmp = (p.compare && typeof p.compare === 'object') ? p.compare
        : (p.diff && typeof p.diff === 'object') ? p.diff : {};
      const simNav = seriesValues(p.sim_nav || []);
      const liveNav = seriesValues(p.live_nav || []);
      const dates = Array.isArray(p.dates) ? p.dates : [];
      const gap = toNum(cmp.pnl_gap);
      setSub('对比 ' + num(cmp.n_days_compared, 0) + ' 天 · 模拟期末 ' + num(cmp.sim_final, 0) +
        ' · 实盘期末 ' + num(cmp.live_final, 0));

      const kpis = [
        kpi({
          k: '模拟盘期末', v: num(cmp.sim_final, 0), tone: 'brand',
          s: '年化 ' + pct(cmp.sim_annual) + ' · 命中率 ' + pct(cmp.sim_hit_rate)
        }),
        kpi({ k: '实盘期末', v: num(cmp.live_final, 0), tone: 'brand', s: '成交 ' + num(cmp.live_trades, 0) + ' 笔' }),
        kpi({
          k: '净值差 pnl_gap', v: signedPct(cmp.pnl_gap), cls: signClass(cmp.pnl_gap),
          tone: gap === null ? 'neutral' : gap < 0 ? 'bad' : 'good', s: '模拟 − 实盘'
        }),
        kpi({ k: '跟踪误差(年化)', v: pct(cmp.tracking_error_ann), tone: 'warn', s: '滑点 ' + num(cmp.exec_slippage_bps, 1) + ' bps' })
      ].join('');

      const chartCard = card({
        title: '净值双线', sub: dates.length ? dates[0] + ' → ' + dates[dates.length - 1] : 'paper.json',
        body: (simNav.some((v) => v !== null) || liveNav.some((v) => v !== null))
          ? '<div class="chart">' + chartSVG([
            { values: simNav, color: 'var(--brand)', label: '模拟盘' },
            { values: liveNav, color: 'var(--warn)', label: '实盘' }
          ], {
            height: 280, tipLabels: dates,
            labels: { first: dates[0] || '', last: dates[dates.length - 1] || '' },
            aria: '模拟盘与实盘净值对比'
          }) + '</div>'
          : emptyBox('暂无净值曲线', 'paper.json 未包含 sim_nav / live_nav 序列。')
      });

      const diffCard = card({
        title: '差异指标', sub: 'compare',
        body: kv([
          kvItem('跟踪误差(年化)', pct(cmp.tracking_error_ann)),
          kvItem('净值差', signedPct(cmp.pnl_gap), signClass(cmp.pnl_gap)),
          kvItem('执行滑点', num(cmp.exec_slippage_bps, 1) + ' bps'),
          kvItem('对比天数', num(cmp.n_days_compared, 0) + ' 天'),
          kvItem('模拟年化', pct(cmp.sim_annual), signClass(cmp.sim_annual)),
          kvItem('实盘年化', pct(cmp.live_annual), signClass(cmp.live_annual)),
          kvItem('模拟命中率', pct(cmp.sim_hit_rate)),
          kvItem('引擎备注', isNil(cmp.note) ? '—' : String(cmp.note))
        ])
      });

      body = '<div class="grid four" style="margin-bottom:14px">' + kpis + '</div>' +
        '<div class="grid two" style="margin-bottom:14px">' + chartCard + diffCard + '</div>';
    }

    const formCard = card({
      title: '录入实盘成交', sub: 'POST /api/paper/trade → data/state/live_trades.jsonl',
      body: '<form class="form" id="trade-form" autocomplete="off">' +
        '<label class="field"><span>日期</span><input type="date" name="date" required></label>' +
        '<label class="field"><span>代码</span><input type="text" name="code" placeholder="600519 / 00700.HK" required maxlength="24">' +
        '<span class="hintline" id="trade-name-hint"></span></label>' +
        '<label class="field"><span>方向</span><select name="side"><option value="buy">买入 buy</option><option value="sell">卖出 sell</option></select></label>' +
        '<label class="field"><span>成交价</span><input type="number" name="price" step="0.001" min="0.001" placeholder="1520.5" required></label>' +
        '<label class="field"><span>数量</span><input type="number" name="qty" step="1" min="1" placeholder="100" required></label>' +
        '<label class="field wide"><span>备注</span><input type="text" name="note" placeholder="可选" maxlength="200"></label>' +
        '<div class="form-actions"><button class="btn primary" type="submit">' + ICON('check') +
        '提交成交</button><span class="form-msg" id="trade-msg"></span></div></form>'
    });

    const tradesCard = card({
      title: '最近成交',
      sub: '共 ' + (trades.n || 0) + ' 笔 · 最新在前' + (namesEmpty() ? ' · 名称映射未导出' : ''),
      cls: 'flush',
      body: '<div style="padding:0 16px 16px">' + table({
        cols: [
          { label: '日期', key: 'date', mono: true },
          { label: '名称（代码）', key: 'code', render: (r) => codeCell(r.code) },
          {
            label: '方向', key: 'side',
            render: (r) => {
              const sd = String(r.side || '').toLowerCase();
              return '<span class="' + (sd === 'buy' ? 'up-text' : sd === 'sell' ? 'down-text' : '') + '">' +
                esc(sd === 'buy' ? '买入' : sd === 'sell' ? '卖出' : r.side || '—') + '</span>';
            }
          },
          { label: '价格', key: 'price', num: true, render: (r) => esc(num(r.price, 3)) },
          { label: '数量', key: 'qty', num: true, render: (r) => esc(num(r.qty, 0)) },
          { label: '备注', key: 'note', render: (r) => esc(r.note || '') }
        ],
        rows: trades.rows || [],
        empty: emptyBox('还没有录入成交', '用左边表单记录第一笔真实成交，下次引擎运行会计入实盘净值。')
      }) + '</div>'
    });

    return '<div style="margin-bottom:14px">' + accountCardHtml(acct) + '</div>' +
      body + '<div class="grid two">' + formCard + tradesCard + '</div>';
  }

  /* ============================== 行为因子 ============================= */

  // 因子说明（来自 qcore/behavioral.py FACTOR_META，纯文档标签，不含任何数值）
  const BEHAVIOR_META = {
    beh_turn_anom: { desc: '异常换手：5 日均换手 / 60 日均换手 − 1', sign: -1, ref: 'Odean (1999); Barber & Odean (2001)', logic: '过度自信导致过度交易，交易越频繁未来收益越低' },
    beh_turn_rel: { desc: '换手率相对自身：20 日均换手 / 250 日均换手', sign: -1, ref: 'Baker & Wurgler (2006)', logic: '换手率是投资者情绪与关注度的代理' },
    beh_max_20: { desc: '过去 20 日最大单日收益', sign: -1, ref: 'Bali, Cakici & Whitelaw (2011)', logic: '彩票型股票被高估，未来收益低（MAX 效应）' },
    beh_skew_60: { desc: '过去 60 日收益偏度', sign: -1, ref: 'Barberis & Huang (2008); Conrad et al. (2013)', logic: '正偏度被高估' },
    beh_ivol_60: { desc: '60 日对市场回归残差的标准差（特质波动率）', sign: -1, ref: 'Ang, Hodrick, Xing & Zhang (2006)', logic: '高特质波动股票未来收益低（IVOL 之谜）' },
    beh_cgo: { desc: '资本利得突出量 (P − RP)/P，RP 为换手加权参考价', sign: -1, ref: 'Grinblatt & Han (2005)', logic: '浮盈大的股票面临处置效应抛压，未来收益低' },
    beh_dist_high: { desc: '距 52 周高点的距离 (P / High252 − 1)', sign: 1, ref: 'George & Hwang (2004)', logic: '锚定效应，接近 52 周高的股票继续走强' },
    beh_amihud: { desc: '非流动性：|收益| / 成交额（20 日均值，×1e8）', sign: 1, ref: 'Amihud (2002)', logic: '流动性溢价，越不流动要求越高收益' },
    beh_turn_vol: { desc: '换手率的 60 日波动率', sign: -1, ref: 'Baker & Wurgler (2006)', logic: '情绪波动大的股票被反复炒作，长期收益低' },
    beh_ret_5d_rev: { desc: '过去 5 日收益（短期反转）', sign: -1, ref: 'Jegadeesh (1990); Lehmann (1990)', logic: '短期反转，散户追涨杀跌造成过度反应' }
  };

  async function pageBehavior() {
    const res = await apiCached('/api/behavior');
    const keys = Object.keys(BEHAVIOR_META);
    let rows = [];

    if (res.ok && res.data) {
      const b = res.data.behavior;
      if (Array.isArray(b)) rows = b;
      else if (b && Array.isArray(b.factors)) rows = b.factors;
      else if (b && typeof b === 'object') {
        rows = Object.keys(b).map((k) => (b[k] && typeof b[k] === 'object')
          ? Object.assign({ factor: k }, b[k]) : { factor: k, ic: b[k] });
      }
    }

    if (!rows.length) {
      setSub('引擎尚未导出 behavior 段（10 个因子定义已就绪）');
      return (res.ok ? '' : notReadyBox(res)) +
        banner({
          tone: 'warn', icon: 'alert',
          title: '行为因子数据未就绪',
          text: '页面已就绪，只等引擎把 <code>behavior</code> 段写进 <code>data/state/performance.json</code>。' +
            '期望结构（数组或 <code>{factors:[...]}</code>）：<br>' +
            '<code>[{factor, ic, t, spread, ic_neutral, note}]</code>' +
            ' · <code>ic_neutral</code> 用于标注「中性化后失效」。',
          meta: 'qcore/behavioral.py · list_factors() 共 10 个'
        }) + '<div style="height:14px"></div>' + card({
          title: '10 个行为因子（定义）', sub: '来自 qcore/behavioral.py FACTOR_META',
          body: table({
            cols: [
              { label: '因子', key: 'factor', mono: true },
              { label: '定义', key: 'desc', render: (r) => '<span style="white-space:normal">' + esc(r.desc) + '</span>' },
              {
                label: '预期方向', key: 'sign',
                render: (r) => '<span class="tag ' + (r.sign > 0 ? 'hit' : 'miss') + '">' +
                  (r.sign > 0 ? '正向' : '负向') + '</span>'
              },
              { label: '文献', key: 'ref', render: (r) => '<span class="muted" style="white-space:normal">' + esc(r.ref) + '</span>' },
              { label: '逻辑', key: 'logic', render: (r) => '<span class="muted" style="white-space:normal">' + esc(r.logic) + '</span>' }
            ],
            rows: keys.map((k) => Object.assign({ factor: k }, BEHAVIOR_META[k]))
          })
        });
    }

    setSub('共 ' + rows.length + ' 个因子 · ' + ((res.data && res.data.generated_at) || ''));

    const norm = rows.map((r) => {
      const f = r.factor || r.factor_id || r.sid || r.name || '—';
      const meta = BEHAVIOR_META[f] || {};
      const ic = firstNum([r.ic, r.ic_mean, r.mean_ic]);
      const icNeutral = firstNum([r.ic_neutral, r.neutral_ic, r.ic_after_neutral]);
      const t = firstNum([r.t, r.t_stat, r.hac_t]);
      const spread = firstNum([r.spread, r.spread_pct, r.ls_spread, r.long_short]);
      let note = r.note || '';
      if (!note && icNeutral !== null && ic !== null && Math.abs(icNeutral) < Math.abs(ic) * 0.5) note = '中性化后失效';
      return { factor: f, meta, ic, icNeutral, t, spread, note };
    });

    return card({
      title: '因子 IC / 多空价差', sub: '双向条形，中轴为 0',
      body: '<div class="grid two" style="gap:18px">' +
        '<div><div class="card-head" style="margin-bottom:8px"><h3 style="font-size:12px">Rank IC</h3></div>' +
        dbars(norm.map((r) => ({ label: r.factor, value: r.ic })), { fmt: (v) => num(v, 4) }) + '</div>' +
        '<div><div class="card-head" style="margin-bottom:8px"><h3 style="font-size:12px">多空价差</h3></div>' +
        dbars(norm.map((r) => ({ label: r.factor, value: r.spread })), { fmt: (v) => num(v, 3) }) + '</div>' +
        '</div>'
    }) + '<div style="height:14px"></div>' + card({
      title: '因子明细', sub: '含「中性化后失效」标注',
      body: table({
        cols: [
          { label: '因子', key: 'factor', mono: true },
          { label: '定义', key: 'desc', render: (r) => '<span class="muted" style="white-space:normal">' + esc(r.meta.desc || '') + '</span>' },
          {
            label: '预期', key: 'sign',
            render: (r) => r.meta.sign ? '<span class="tag ' + (r.meta.sign > 0 ? 'hit' : 'miss') + '">' +
              (r.meta.sign > 0 ? '正' : '负') + '</span>' : '—'
          },
          { label: 'IC', key: 'ic', num: true, render: (r) => '<span class="' + signClass(r.ic) + '">' + esc(num(r.ic, 4)) + '</span>' },
          { label: 'HAC t', key: 't', num: true, render: (r) => esc(num(r.t, 2)) },
          { label: '多空价差', key: 'spread', num: true, render: (r) => '<span class="' + signClass(r.spread) + '">' + esc(num(r.spread, 3)) + '</span>' },
          { label: '中性化后 IC', key: 'icNeutral', num: true, render: (r) => esc(num(r.icNeutral, 4)) },
          {
            label: '结论', key: 'note',
            render: (r) => r.note ? '<span class="tag warn">' + esc(r.note) + '</span>' : '<span class="tag na">—</span>'
          }
        ],
        rows: norm
      })
    });
  }

  /* ============================== 政策事件 ============================= */

  async function pageEvents() {
    const res = await apiCached('/api/events?limit=2000');
    if (!res.ok || !res.data) {
      setSub('未就绪');
      return notReadyBox(res) + '<div style="height:14px"></div>' + banner({
        tone: 'warn', icon: 'alert',
        title: '政策日历未就绪',
        text: '期望 <code>data/state/events.json</code>（或 <code>data/events/policy_calendar.jsonl</code>）。' +
          '生成：<code>python -m qcore.policy_calendar</code>。',
        meta: '字段：date / type / subtype / title / direction / sectors[] / evidence_level / source / note'
      });
    }

    const rows = (res.data.rows || []).filter((r) => r && typeof r === 'object');
    const types = Array.from(new Set(rows.map((r) => r.type).filter(Boolean)));
    const years = Array.from(new Set(rows.map((r) => String(r.date || '').slice(0, 4)).filter(Boolean))).sort();
    setSub('共 ' + rows.length + ' 条 · ' + types.length + ' 类 · ' +
      (years.length ? years[0] + '–' + years[years.length - 1] : '—'));

    const tf = S.filter.evType;
    const df = S.filter.evDir;
    const view = rows.filter((r) => {
      if (tf !== 'all' && r.type !== tf) return false;
      const d = toNum(r.direction);
      if (df === 'pos' && !(d > 0)) return false;
      if (df === 'neg' && !(d < 0)) return false;
      return true;
    }).slice().sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));

    const tl = card({
      title: '政策日历时间线', sub: '最新在前 · 共 ' + view.length + ' 条', cls: 'flush',
      tools: '<label class="field inline"><span>类型</span><select id="f-evtype"><option value="all">全部</option>' +
        types.map((t) => '<option value="' + esc(t) + '"' + (tf === t ? ' selected' : '') + '>' +
          esc(EVENT_TYPES[t] || t) + '</option>').join('') + '</select></label>' +
        '<div class="segmented" id="f-evdir">' +
        [['all', '全部'], ['pos', '利好'], ['neg', '利空']].map(([v, l]) =>
          '<button type="button" class="seg' + (df === v ? ' is-active' : '') + '" data-d="' + v + '">' + l + '</button>').join('') +
        '</div>',
      body: '<div style="padding:0 16px 16px;max-height:760px;overflow:auto">' + timeline(view) + '</div>'
    });

    const impact = Array.isArray(res.data.impact) ? res.data.impact
      : (res.data.impact && Array.isArray(res.data.impact.rows)) ? res.data.impact.rows : [];

    const impactCard = card({
      title: '影响因子表', sub: 'events.impact_factor 输出',
      body: impact.length ? table({
        cols: Object.keys(impact[0]).slice(0, 7).map((k) => ({
          label: k, key: k, num: toNum(impact[0][k]) !== null,
          render: (r) => esc(fmtByKey(k, r[k]))
        })),
        rows: impact
      }) : emptyBox('暂无影响因子表',
        '需要 events.json 里带 <code>impact</code> 数组（type / sector / horizon / mean_car / t / n）。')
    });

    return '<div style="margin-bottom:14px">' + tl + '</div>' + impactCard;
  }

  /* ============================== 风险体检 ============================= */

  const INVARIANTS = [
    { id: 'INV-1', name: '无未来函数', check: 'tests/test_no_lookahead.py' },
    { id: 'INV-2', name: '先预测后对账', check: 'engine 循环顺序' },
    { id: 'INV-3', name: '统计量必须滚动', check: 'audit/leak_scan.py' },
    { id: 'INV-4', name: '重叠收益用 HAC', check: 'lags = h-1' },
    { id: 'INV-5', name: '后复权且非零风险', check: '数据守卫' },
    { id: 'INV-6', name: '股票池点对点', check: 'UNIV 掩码' },
    { id: 'INV-7', name: '台账只追加', check: 'pred_ledger / claims' },
    { id: 'INV-8', name: 'UI 不做计算', check: 'webui/ 只读 JSON' }
  ];

  async function pageRisk() {
    const [perfRes, healthRes] = await Promise.all([apiCached('/api/performance'), apiCached('/api/health')]);
    const perf = perfRes.ok ? adaptPerformance(perfRes.data) : null;
    const health = healthRes.ok ? healthRes.data : null;
    setSub(perf ? '生成于 ' + ((perfRes.data && perfRes.data.generated_at) || '—') : '未就绪');

    /* 不变量 */
    const inv = health && health.invariants;
    const invMap = {};
    if (Array.isArray(inv)) inv.forEach((x) => { if (x && x.id) invMap[x.id] = x; });
    else if (inv && typeof inv === 'object') Object.keys(inv).forEach((k) => { invMap[k] = inv[k]; });

    const invRows = INVARIANTS.map((it) => {
      const got = invMap[it.id];
      let status = '<span class="tag na">未提供</span>';
      let detail = it.check;
      if (got) {
        const st = String(got.status || got.result || '').toLowerCase();
        const ok = st === 'pass' || st === 'ok' || got.ok === true;
        status = ok ? '<span class="tag ok">PASS</span>'
          : '<span class="tag bad">' + esc((st || 'fail').toUpperCase()) + '</span>';
        detail = got.detail || got.note || it.check;
      }
      return { id: it.id, name: it.name, status, detail };
    });

    const invCard = card({
      title: '不变量检查',
      sub: inv ? '来自 invariants.json' : '未提供机读结果',
      cls: 'flush',
      body: '<div style="padding:0 16px 16px">' + table({
        cols: [
          { label: '编号', key: 'id', mono: true },
          { label: '不变量', key: 'name' },
          { label: '状态', key: 'status', render: (r) => r.status },
          { label: '依据', key: 'detail', render: (r) => '<span class="muted" style="white-space:normal">' + esc(r.detail) + '</span>' }
        ],
        rows: invRows
      }) + (!inv ? '<p class="muted" style="margin-top:8px">跑完 <code>tests/</code> 与 <code>audit/</code> 后把结果写成 ' +
        '<code>data/state/invariants.json</code>：[{id, name, status:"pass", detail}]</p>' : '') + '</div>'
    });

    /* 数据源健康度 */
    const sh = health && health.source_health;
    const shCard = (sh && typeof sh === 'object') ? card({
      title: '数据源健康度', sub: 'data/state/source_health.json',
      body: table({
        cols: [
          { label: '源', key: 'name', mono: true },
          { label: '状态', key: 'tag', render: (r) => r.tag },
          { label: '成功', key: 'ok', num: true, render: (r) => esc(num(r.ok, 0)) },
          {
            label: '失败', key: 'fail', num: true,
            render: (r) => '<span class="' + (toNum(r.fail) > 0 ? 'up-text' : '') + '">' + esc(num(r.fail, 0)) + '</span>'
          },
          { label: '延迟 ms', key: 'latency', num: true, render: (r) => esc(num(r.latency, 1)) },
          { label: '最后错误', key: 'err', render: (r) => '<span class="muted">' + esc(r.err || '—') + '</span>' }
        ],
        rows: Object.keys(sh).map((k) => {
          const v = sh[k] || {};
          const failN = toNum(v.fail);
          return {
            name: k, ok: toNum(v.ok), fail: failN, latency: toNum(v.latency_ms),
            err: v.last_error || '',
            tag: (failN !== null && failN > 0) ? '<span class="tag bad">有失败</span>' : '<span class="tag ok">正常</span>'
          };
        })
      })
    }) : card({
      title: '数据源健康度',
      body: emptyBox('暂无数据源健康度', '需要 <code>data/state/source_health.json</code>（qcore/sources.py 写入）。')
    });

    /* 成本明细 */
    const costs = perf && perf.costs;
    let costCard;
    if (costs) {
      const bd = costs.breakdown || {};
      const items = [];
      Object.keys(bd).forEach((side) => {
        const part = bd[side] || {};
        Object.keys(part).forEach((k) => {
          const v = toNum(part[k]);
          if (v !== null && v !== 0) items.push({ side: side === 'buy' ? '买入' : '卖出', item: k, value: v });
        });
      });
      const maxV = items.reduce((m, x) => Math.max(m, x.value), 0) || 1;
      const breakdownHtml = items.length
        ? '<div class="kv" style="grid-template-columns:repeat(auto-fit,minmax(168px,1fr))">' + items.map((x) =>
          '<div class="item"><span class="k">' + esc(x.side + ' · ' + x.item) + '</span>' +
          '<span class="v">' + esc(num(x.value, 2)) + '<span class="muted" style="font-size:10px"> 元/百万</span></span>' +
          '<div class="meter" style="margin-top:5px"><i style="width:' + ((x.value / maxV) * 100).toFixed(1) + '%"></i></div></div>'
        ).join('') + '</div>'
        : emptyBox('暂无成本拆解', '');

      const scen = Array.isArray(costs.scenarios) ? costs.scenarios : [];
      costCard = card({
        title: '成本明细',
        sub: esc(costs.market) + ' · 双边 ' + num(costs.round_trip_bps, 2) + ' bps · 核对于 ' + esc(costs.verified_on || '—'),
        body: kv([
          kvItem('年化拖累', num(costs.drag && costs.drag.annual_cost_pct, 2) + '%'),
          kvItem('年换手次数', num(costs.drag && costs.drag.periods_per_year, 1)),
          kvItem('双边成本', num(costs.round_trip_bps, 2) + ' bps')
        ]) + '<div style="height:14px"></div>' + breakdownHtml +
          (scen.length ? '<div style="height:14px"></div>' + table({
            cols: [
              { label: '市场', key: 'market', mono: true },
              { label: '场景', key: 'name' },
              { label: '名义金额', key: 'notional', num: true, render: (r) => esc(num(r.notional, 0)) },
              { label: '双边 bps', key: 'round_trip_bps', num: true, render: (r) => esc(num(r.round_trip_bps, 2)) }
            ],
            rows: scen.slice(0, 15)
          }) : '')
      });
    } else {
      costCard = card({ title: '成本明细', body: emptyBox('暂无成本数据', '需要 performance.json 的 <code>costs</code> 段。') });
    }

    /* 集中度 */
    const risk = perf && perf.risk;
    let riskCard;
    if (risk) {
      const sl = risk.signal_level || {};
      const pl = risk.portfolio_level || {};
      const ca = sl.corr_adjusted || {};
      const fam = sl.family_share || {};
      const famRows = Object.keys(fam).map((k) => ({ label: k, value: toNum(fam[k]) * 100 }));
      riskCard = card({
        title: '集中度', sub: risk.note || 'HHI / 有效独立数',
        body: kv([
          kvItem('信号 HHI', num(sl.hhi, 4)),
          kvItem('权重有效数', num(sl.effective_n, 2)),
          kvItem('相关性有效数', num(ca.effective_n_corr, 2), '', 'effective_n_corr'),
          kvItem('最大单权重', pct(sl.max_share)),
          kvItem('组合 HHI', num(pl.hhi, 4)),
          kvItem('组合有效数', num(pl.effective_n, 2))
        ]) +
          ((sl.warnings && sl.warnings.length)
            ? '<div style="margin-top:12px">' + sl.warnings.map((w) =>
              '<div class="tag warn" style="margin:2px 4px 2px 0;white-space:normal">' + esc(w) + '</div>').join('') + '</div>'
            : '') +
          (famRows.length
            ? '<div style="height:14px"></div><div class="muted" style="margin-bottom:6px">族权重占比（%）</div>' +
              hbars(famRows, { fmt: (v) => num(v, 2) + '%' })
            : '')
      });
    } else {
      riskCard = card({ title: '集中度', body: emptyBox('暂无集中度数据', '需要 performance.json 的 <code>risk</code> 段。') });
    }

    /* 可信度校准 */
    const conf = perf && perf.confidence;
    const relRows = conf && Array.isArray(conf.reliability) ? conf.reliability : [];
    const relCard = card({
      title: '可信度校准',
      sub: conf ? 'Brier ' + num(conf.brier, 4) + ' · ECE ' + num(conf.ece, 4) + ' · 基准率 ' + pct(conf.base_rate) : '未就绪',
      body: relRows.length ? table({
        cols: [
          { label: '分箱', key: 'bin', mono: true },
          { label: '样本', key: 'n', num: true, render: (r) => esc(num(r.n, 0)) },
          { label: '预测', key: 'pred', num: true, render: (r) => esc(num(r.pred, 4)) },
          { label: '实际', key: 'actual', num: true, render: (r) => esc(num(r.actual, 4)) },
          { label: '偏差', key: 'gap', num: true, render: (r) => '<span class="' + signClass(r.gap) + '">' + esc(num(r.gap, 4)) + '</span>' }
        ],
        rows: relRows
      }) : emptyBox('暂无可靠性曲线', '需要 confidence.reliability。')
    });

    /* 漂移监控 */
    const mon = perf && perf.meta && perf.meta.monitor;
    const monCard = card({
      title: '漂移监控', sub: 'meta.monitor（Page-Hinkley）',
      body: mon ? kv([
        kvItem('活跃时间占比', pct(mon.frac_time_active)),
        kvItem('当前状态', toNum(mon.final_state) === -1 ? '停手 (−1)' : toNum(mon.final_state) === 1 ? '可用 (+1)' : '中性 (0)'),
        kvItem('当前 t 值', num(mon.final_t, 3), signClass(mon.final_t))
      ]) : emptyBox('暂无漂移监控', '')
    });

    return '<div class="grid two" style="margin-bottom:14px">' + invCard + shCard + '</div>' +
      '<div style="margin-bottom:14px">' + costCard + '</div>' +
      '<div class="grid two" style="margin-bottom:14px">' + riskCard + relCard + '</div>' +
      monCard;
  }

  /* ================================ 设置 ============================== */

  async function pageSettings() {
    const [verRes, healthRes, updRes] = await Promise.all([
      apiCached('/api/version', 0), apiCached('/api/health', 0), apiCached('/api/update', 0)
    ]);
    const v = (verRes.ok && verRes.data) || {};
    const h = (healthRes.ok && healthRes.data) || {};
    const u = (updRes.ok && updRes.data) || S.update || null;
    setSub('v' + (v.version || '—') + (v.python ? ' · Python ' + v.python : ''));

    const files = h.files || {};
    const fileRows = Object.keys(files).map((name) => {
      const f = files[name] || {};
      return {
        name, exists: f.exists, size: f.size, mtime: f.mtime,
        tag: f.exists ? '<span class="tag ok">存在</span>' : '<span class="tag bad">缺失</span>'
      };
    });
    const opt = h.optional_files || {};
    Object.keys(opt).forEach((name) => {
      const f = opt[name] || {};
      fileRows.push({
        name: name + '（可选）', exists: f.exists, size: f.size, mtime: f.mtime,
        tag: f.exists ? '<span class="tag ok">存在</span>' : '<span class="tag na">未生成</span>'
      });
    });

    const infoCard = card({
      title: '版本与运行环境', sub: '来自 /api/version',
      body: kv([
        kvItem('版本', 'v' + (v.version || '—')),
        kvItem('构建时间', v.build_time || '—'),
        kvItem('Python', v.python || '—'),
        kvItem('平台', v.platform || '—'),
        kvItem('数据目录', h.state_dir || '—'),
        kvItem('静态目录', h.static_dir || '—')
      ])
    });

    const filesCard = card({
      title: '数据文件',
      sub: (h.missing_files && h.missing_files.length) ? '缺 ' + h.missing_files.length + ' 个' : '齐备',
      cls: 'flush',
      body: '<div style="padding:0 16px 16px">' + table({
        cols: [
          { label: '文件', key: 'name', mono: true },
          { label: '状态', key: 'tag', render: (r) => r.tag },
          {
            label: '大小', key: 'size', num: true,
            render: (r) => r.exists ? esc((r.size / 1024).toFixed(1)) + ' KB' : '—'
          },
          { label: '修改时间', key: 'mtime', mono: true, render: (r) => esc(r.mtime || '—') }
        ],
        rows: fileRows
      }) + '</div>'
    });

    const runCard = card({
      title: '触发一次更新', sub: 'POST /api/run → python -m qcore.live --once（需 X-Token）',
      body: '<div class="form" style="grid-template-columns:minmax(0,1fr) auto;align-items:end">' +
        '<label class="field"><span>APB_TOKEN（存本机 localStorage，只填一次）</span>' +
        '<input id="set-token" type="password" placeholder="服务端环境变量 APB_TOKEN 的值" autocomplete="off"></label>' +
        '<button class="btn primary" id="btn-run" type="button">' + ICON('play') + '运行一次</button>' +
        '</div><p class="form-msg" id="run-msg" style="margin-top:10px">' +
        (h.run_enabled === false ? '服务端已用 --no-run 禁用该功能。'
          : '触发后在后台运行，结果会轮询显示在下方。') + '</p>' +
        '<div id="run-status" style="margin-top:10px"></div>'
    });

    const exportCard = card({
      title: '导出数据', sub: 'GET /api/export（JSON 附件）',
      body: '<div style="display:flex;gap:8px;flex-wrap:wrap">' +
        '<a class="btn" href="/api/export?what=all" download>' + ICON('download') + '全部状态文件</a>' +
        '<a class="btn ghost" href="/api/export?what=ledger" download>台账</a>' +
        '<a class="btn ghost" href="/api/export?what=signals" download>权重</a>' +
        '<a class="btn ghost" href="/api/export?what=performance" download>评估</a>' +
        '</div><p class="muted" style="margin-top:10px">导出内容 = data/state 下已存在文件的原样拼接，不做任何加工。</p>'
    });

    const themeCard = card({
      title: '外观', sub: '存 localStorage，默认暗色',
      body: '<div class="segmented" id="set-theme">' +
        '<button type="button" class="seg' + (document.documentElement.getAttribute('data-theme') !== 'light' ? ' is-active' : '') + '" data-t="dark">暗色</button>' +
        '<button type="button" class="seg' + (document.documentElement.getAttribute('data-theme') === 'light' ? ' is-active' : '') + '" data-t="light">亮色</button>' +
        '</div><p class="muted" style="margin-top:10px">侧栏折叠状态同样会记住（点侧栏右上角图标切换）。</p>'
    });

    return '<div class="grid two" style="margin-bottom:14px">' + infoCard + filesCard + '</div>' +
      '<div style="margin-bottom:14px">' + updateCard(u) + '</div>' +
      '<div class="grid two" style="margin-bottom:14px">' + runCard + exportCard + '</div>' +
      themeCard;
  }

  /* =============================== 路由 =============================== */

  const RENDERERS = {
    overview: pageOverview,
    market: pageMarket,
    signals: pageSignals,
    model: pageModel,
    ledger: pageLedger,
    paper: pagePaper,
    behavior: pageBehavior,
    events: pageEvents,
    risk: pageRisk,
    settings: pageSettings
  };

  function buildNav() {
    $('#side-nav').innerHTML = PAGES.map((p) =>
      '<button type="button" class="nav-item" data-page="' + p.id + '" title="' + esc(p.label) + '">' +
      ICON(p.icon) + '<span class="nav-label">' + esc(p.label) + '</span></button>').join('');
    $('#bottom-nav').innerHTML = PAGES.map((p) =>
      '<button type="button" class="bn-item" data-page="' + p.id + '">' +
      ICON(p.icon) + '<span>' + esc(p.label) + '</span></button>').join('');
  }

  const markNav = (id) => $$('.nav-item, .bn-item').forEach((b) =>
    b.classList.toggle('is-active', b.getAttribute('data-page') === id));

  function skeleton() {
    return '<div class="grid four" style="margin-bottom:14px">' +
      Array(4).fill('<div class="kpi"><span class="k"><span class="skeleton" style="display:inline-block;width:52px"></span></span>' +
        '<span class="v"><span class="skeleton" style="display:inline-block;width:96px;height:22px"></span></span></div>').join('') +
      '</div><div class="card"><div class="skeleton" style="width:100%;height:120px"></div></div>';
  }

  async function navigate(id, push) {
    if (!RENDERERS[id]) id = 'overview';
    S.page = id;
    const meta = pageMeta(id);
    $('#page-title').textContent = meta.title;
    setSub(meta.sub);
    markNav(id);
    document.body.classList.remove('nav-open');
    $('#scrim').hidden = true;
    if (push !== false) {
      try { location.hash = '#/' + id; } catch (e) { /* ignore */ }
      try { localStorage.setItem('apb.page', id); } catch (e) { /* ignore */ }
    }
    const view = $('#view');
    view.innerHTML = skeleton();
    let html;
    try {
      html = await RENDERERS[id]();
    } catch (e) {
      html = card({ title: '渲染失败', body: emptyBox('页面渲染出错', esc(String((e && e.message) || e)), 'alert') });
    }
    if (S.page !== id) return;      // 期间已切页
    view.innerHTML = html;
    window.scrollTo(0, 0);
    bindPageEvents(id);
  }

  /* --------------------------- 页内事件绑定 --------------------------- */

  function bindPageEvents(id) {
    if (id === 'signals') {
      const fam = $('#f-family');
      if (fam) fam.addEventListener('change', () => { S.filter.family = fam.value; navigate('signals', false); });
      const q = $('#f-q');
      if (q) {
        let t = null;
        q.addEventListener('input', () => {
          clearTimeout(t);
          t = setTimeout(() => { S.filter.q = q.value; navigate('signals', false); }, 220);
        });
      }
      $$('.hbar[data-sid]').forEach((b) => b.addEventListener('click', () => {
        S.selSid = b.getAttribute('data-sid');
        navigate('signals', false);
      }));
      $$('th.sortable').forEach((th) => th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort');
        const s = S.sort.signals;
        if (s.key === key) s.dir = -s.dir;
        else { s.key = key; s.dir = key === 'sid' ? 1 : -1; }
        navigate('signals', false);
      }));
    }

    if (id === 'ledger') {
      const days = $('#f-days');
      if (days) days.addEventListener('change', () => { S.filter.days = Number(days.value) || 30; navigate('ledger', false); });
      const code = $('#f-code');
      if (code) {
        let t = null;
        code.addEventListener('input', () => {
          clearTimeout(t);
          t = setTimeout(() => { S.filter.q = code.value; navigate('ledger', false); }, 220);
        });
      }
      const from = $('#f-from');
      if (from) from.addEventListener('change', () => { S.filter.from = from.value; navigate('ledger', false); });
      const to = $('#f-to');
      if (to) to.addEventListener('change', () => { S.filter.to = to.value; navigate('ledger', false); });
      $$('#f-result .seg').forEach((b) => b.addEventListener('click', () => {
        S.filter.result = b.getAttribute('data-f');
        navigate('ledger', false);
      }));
    }

    if (id === 'events') {
      const ty = $('#f-evtype');
      if (ty) ty.addEventListener('change', () => { S.filter.evType = ty.value; navigate('events', false); });
      $$('#f-evdir .seg').forEach((b) => b.addEventListener('click', () => {
        S.filter.evDir = b.getAttribute('data-d');
        navigate('events', false);
      }));
    }

    if (id === 'paper') {
      const form = $('#trade-form');
      if (form) {
        const d = form.elements.date;
        if (d && !d.value) {
          const now = new Date();
          const pad = (x) => String(x).padStart(2, '0');
          d.value = now.getFullYear() + '-' + pad(now.getMonth() + 1) + '-' + pad(now.getDate());
        }
        form.addEventListener('submit', submitTrade);

        // 输入代码 → 实时显示匹配到的名称（查不到提示「未知代码」，仍可提交）
        const codeInput = form.elements.code;
        const nameHint = $('#trade-name-hint');
        if (codeInput && nameHint) {
          codeInput.addEventListener('input', () => {
            const c = String(codeInput.value || '').trim();
            if (!c) { nameHint.className = 'hintline'; nameHint.textContent = ''; return; }
            const nm = nameOf(c);
            const ind = industryOf(c);
            if (nm) {
              nameHint.className = 'hintline';
              nameHint.textContent = '→ ' + nm + (ind ? ' · ' + ind : '');
            } else {
              nameHint.className = 'hintline warn';
              nameHint.textContent = '未知代码（仍可提交）';
            }
          });
        }
      }

      // 账户设置表单
      const acctForm = $('#acct-form');
      if (acctForm) {
        const capEl = acctForm.elements.initial_capital;
        const capHint = $('#acct-cap-hint');
        if (capEl && capHint) {
          capEl.addEventListener('input', () => {
            const n = toNum(capEl.value);
            capHint.textContent = n === null ? '　' : fmtMoney(n);
          });
        }
        acctForm.addEventListener('submit', submitAccount);
      }
    }

    if (id === 'settings') {
      const tk = $('#set-token');
      if (tk) {
        tk.value = getToken();
        tk.addEventListener('change', () => setToken(tk.value.trim()));
      }
      const run = $('#btn-run');
      if (run) run.addEventListener('click', runOnce);
      const chk = $('#btn-check-update');
      if (chk) chk.addEventListener('click', async () => {
        chk.disabled = true;
        await loadUpdate(true);
        navigate('settings', false);
        toast('已重新读取更新信息', 'ok');
      });
      const doUpd = $('#btn-do-update');
      if (doUpd) doUpd.addEventListener('click', openUpdateModal);
      $$('#set-theme .seg').forEach((b) => b.addEventListener('click', () => setTheme(b.getAttribute('data-t'))));
      refreshRunStatus();
    }
  }

  /* ---------------------------- 成交提交 ------------------------------ */

  async function submitTrade(ev) {
    ev.preventDefault();
    const form = ev.target;
    const msg = $('#trade-msg');
    const payload = {
      date: String(form.elements.date.value || '').trim(),
      code: String(form.elements.code.value || '').trim(),
      side: form.elements.side.value,
      price: Number(String(form.elements.price.value || '').trim()),
      qty: Number(String(form.elements.qty.value || '').trim()),
      note: String(form.elements.note.value || '').trim()
    };
    if (!payload.date) { msg.className = 'form-msg err'; msg.textContent = '请填写日期'; return; }
    if (!payload.code) { msg.className = 'form-msg err'; msg.textContent = '请填写代码'; return; }
    if (!isFinite(payload.price) || payload.price <= 0) { msg.className = 'form-msg err'; msg.textContent = '成交价必须为正数'; return; }
    if (!isFinite(payload.qty) || payload.qty <= 0 || payload.qty % 1 !== 0) { msg.className = 'form-msg err'; msg.textContent = '数量必须为正整数'; return; }

    const btn = form.querySelector('button[type="submit"]');
    if (btn) btn.disabled = true;
    msg.className = 'form-msg';
    msg.textContent = '提交中…';
    const res = await api('/api/paper/trade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload)
    });
    if (btn) btn.disabled = false;
    if (res.ok) {
      msg.className = 'form-msg ok';
      msg.textContent = '已写入：' + payload.code + ' ' + payload.side + ' ' + payload.qty + ' @ ' + payload.price;
      toast('成交已记录', 'ok');
      clearCache();
      navigate('paper', false);
    } else {
      msg.className = 'form-msg err';
      msg.textContent = '失败：' + (res.error || 'unknown') + (res.hint ? ' — ' + res.hint : '');
      toast('录入失败：' + (res.error || 'unknown'), 'err');
    }
  }

  /* ----------------------------- 运行一次 ----------------------------- */

  async function refreshRunStatus() {
    const box = $('#run-status');
    if (!box) return;
    const res = await api('/api/run/status');
    if (!res.ok) { box.innerHTML = ''; return; }
    const s = res.data || {};
    if (s.state === 'idle') { box.innerHTML = '<span class="muted">尚未触发过运行。</span>'; return; }
    const tone = s.state === 'done' ? 'ok' : s.state === 'running' ? 'info' : 'bad';
    box.innerHTML = kv([
      kvItem('状态', s.state),
      kvItem('开始', s.started_at || '—'),
      kvItem('耗时', (s.duration_s === null || s.duration_s === undefined) ? '—' : s.duration_s + ' s'),
      kvItem('返回码', (s.returncode === null || s.returncode === undefined) ? '—' : String(s.returncode))
    ]) + (s.error
      ? '<div class="tag bad" style="margin-top:8px;white-space:normal">' + esc(s.error) + '</div>' : '') +
      (s.tail
        ? '<pre class="mono" style="margin-top:8px;max-height:180px;overflow:auto;font-size:11px;color:var(--text-3);white-space:pre-wrap">' +
          esc(String(s.tail).slice(-1200)) + '</pre>' : '') +
      '<div class="tag ' + tone + '" style="margin-top:8px">' + esc(s.state) + '</div>';
  }

  async function runOnce() {
    const msg = $('#run-msg');
    let token = getToken();
    if (!token) token = window.prompt('请输入服务端环境变量 APB_TOKEN 的值', '') || '';
    if (!token) { toast('未提供 X-Token，已取消', 'err'); return; }
    setToken(token);
    const btn = $('#btn-run');
    if (btn) btn.disabled = true;
    const res = await api('/api/run', { method: 'POST', headers: { 'X-Token': token } });
    if (btn) btn.disabled = false;
    if (!res.ok) {
      if (msg) {
        msg.className = 'form-msg err';
        msg.textContent = '触发失败：' + res.error + (res.hint ? ' — ' + res.hint : '');
      }
      toast('触发失败：' + res.error, 'err');
      return;
    }
    if (msg) { msg.className = 'form-msg ok'; msg.textContent = '已触发，后台运行中…'; }
    toast('已触发 python -m qcore.live --once');
    refreshRunStatus();
    let tries = 200;
    const timer = setInterval(async () => {
      tries -= 1;
      const st = await api('/api/run/status');
      refreshRunStatus();
      const stt = (st.ok && st.data) ? st.data.state : 'idle';
      if (stt !== 'running' || tries <= 0) {
        clearInterval(timer);
        if (stt === 'done') { clearCache(); toast('运行完成，正在刷新', 'ok'); navigate(S.page, false); }
      }
    }, 3000);
  }

  /* ------------------------------- 主题 ------------------------------- */

  function setTheme(t) {
    const light = t === 'light';
    document.documentElement.setAttribute('data-theme', light ? 'light' : 'dark');
    try { localStorage.setItem('apb.theme', light ? 'light' : 'dark'); } catch (e) { /* ignore */ }
    const use = $('#theme-ico use');
    if (use) use.setAttribute('href', light ? '#i-sun' : '#i-moon');
    $$('#set-theme .seg').forEach((b) => b.classList.toggle('is-active', b.getAttribute('data-t') === (light ? 'light' : 'dark')));
  }

  const toggleTheme = () => setTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light');

  function toggleRail() {
    const cur = document.documentElement.getAttribute('data-rail') === '1' ? '0' : '1';
    document.documentElement.setAttribute('data-rail', cur);
    try { localStorage.setItem('apb.rail', cur); } catch (e) { /* ignore */ }
  }

  /* ------------------------------- 绑定 ------------------------------- */

  function bindGlobal() {
    $('#btn-refresh').addEventListener('click', async () => {
      clearCache();
      toast('刷新中…');
      await Promise.all([loadVersion(), loadHealth()]);
      navigate(S.page, false);
    });
    $('#badge-update').addEventListener('click', () => { if (!S.updating) openUpdateModal(); });
    $('#chip-version').addEventListener('click', () => navigate('settings'));
    $('#modal-close').addEventListener('click', closeModal);
    $('#modal-mask').addEventListener('click', (ev) => {
      if (ev.target === $('#modal-mask')) closeModal();
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && !$('#modal-mask').hidden) closeModal();
    });
    $('#btn-theme').addEventListener('click', toggleTheme);
    $('#btn-rail').addEventListener('click', toggleRail);
    $('#btn-menu').addEventListener('click', () => {
      const open = document.body.classList.toggle('nav-open');
      $('#scrim').hidden = !open;
    });
    $('#scrim').addEventListener('click', () => {
      document.body.classList.remove('nav-open');
      $('#scrim').hidden = true;
    });
    document.addEventListener('click', (ev) => {
      const btn = ev.target.closest('[data-page]');
      if (btn) navigate(btn.getAttribute('data-page'));
    });
    window.addEventListener('hashchange', () => {
      const id = (location.hash || '').replace(/^#\/?/, '') || 'overview';
      if (id !== S.page) navigate(id, false);
    });
    let rt = null;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => { if (S.page === 'overview' || S.page === 'paper') navigate(S.page, false); }, 260);
    });
    setInterval(() => { if (!document.hidden) loadHealth(); }, 60000);
    // 市场页：页面可见时每 60 秒拉一次 /api/market
    setInterval(() => { refreshMarket(); }, 60000);
  }

  /* ------------------------------- 启动 ------------------------------- */

  async function boot() {
    buildNav();
    bindGlobal();
    const fromHash = (location.hash || '').replace(/^#\/?/, '');
    let last = null;
    try { last = localStorage.getItem('apb.page'); } catch (e) { /* ignore */ }
    const start = RENDERERS[fromHash] ? fromHash : (RENDERERS[last] ? last : 'overview');
    const use = $('#theme-ico use');
    if (use) use.setAttribute('href', document.documentElement.getAttribute('data-theme') === 'light' ? '#i-sun' : '#i-moon');
    await Promise.all([loadVersion(), loadHealth(), loadUpdate(), loadNames()]);
    await navigate(start, true);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
