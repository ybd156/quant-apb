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
  /**
   * 明确按「百分点」格式化：输入 0.83 表示 0.83%，不做 ×100 猜测。
   * signedPct 有歧义启发式（|v|≤1 就当小数），对已经换算好的百分数会翻 100 倍。
   * 凡是后端已经给出百分数的字段，一律用这个。
   */
  function pctPoints(v, d) {
    const n = toNum(v);
    if (n === null) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(d === undefined ? 2 : d) + '%';
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
    // 写请求自动带 token：服务端设了 APB_TOKEN 时，不带就是 403。
    // 调用方显式传的 X-Token 优先（比如「触发一次更新」卡片里的输入框）。
    const _m = String(opts.method || 'GET').toUpperCase();
    if (_m !== 'GET' && _m !== 'HEAD') {
      opts.headers = Object.assign({}, opts.headers);
      const has = Object.keys(opts.headers).some((k) => k.toLowerCase() === 'x-token');
      const t = getToken();
      if (!has && t) opts.headers['X-Token'] = t;
    }
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
    if ((body.error === 'unauthorized' || body.error === 'forbidden')
        && !body.hint2) {
      body.hint2 = '服务端设了 APB_TOKEN。去「设置 → 访问令牌」填一次，浏览器会记住。';
    }
    // 会话过期/未登录：直接送去登录页，不要停在一个到处报错的界面
    if (body.error === 'login_required') {
      if (!window.__apbRedirecting) {
        window.__apbRedirecting = true;
        location.replace('/login');
      }
    }
    return body;
  }

  /** ttl 内复用同一 GET（performance.json 近 1MB，避免翻页重复解析） */
  // 非 GET 成功后统一清缓存。
  // 之前只在 4 个地方手动 clearCache()，新加的 POST（加自选、生成预测、
  // 清空重设）全都漏了 —— 用户点完按钮页面纹丝不动，以为按钮坏了。
  const _apiRaw = api;
  api = async function (url, options) {
    const r = await _apiRaw(url, options);
    const m = String((options && options.method) || 'GET').toUpperCase();
    if (m !== 'GET' && m !== 'HEAD' && r && r.ok) _cache.clear();
    return r;
  };

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
    return '<div class="kpi tone-' + (o.tone || 'brand') + '"' +
      (o.title ? ' title="' + esc(o.title) + '"' : '') + '>' +
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
      body = rows.map((r, idx) => {
        // beforeRow：在这一行前面插入任意 HTML（用于「按天分组」的日期小标题）
        const pre = (typeof opt.beforeRow === 'function')
          ? (opt.beforeRow(r, idx, rows) || '') : '';
        const trCls = (typeof opt.rowClass === 'function')
          ? (opt.rowClass(r, idx) || '') : '';
        return pre + '<tr' + (trCls ? ' class="' + trCls + '"' : '') + '>' +
          cols.map((c) => {
            const v = c.render ? c.render(r) : esc(isNil(r[c.key]) ? '—' : r[c.key]);
            return '<td class="' + (c.num ? 'num ' : '') + (c.mono ? 'mono' : '') + '">' + v + '</td>';
          }).join('') + '</tr>';
      }).join('');
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
        (e._live ? '<span class="tag" title="实时抓取，未经人工核实">实时</span> ' : '') +
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
      page: 1, pageSize: 100, perDay: 20,
      evType: 'all', evDir: 'all'
    },
    selSid: null,
    update: null,
    updating: false,
    names: {},
    namesHint: '',
    account: null,
    paperMarket: 'CN',
    costMap: null
  };

  /* ============================== 页面定义 ============================= */

  const PAGES = [
    { id: 'overview', label: '总览', icon: 'overview', title: '总览', sub: '这台测量仪现在读数是多少' },
    { id: 'market', label: '市场', icon: 'market', title: '实时行情', sub: '指数 / 情绪温度 / 市场宽度' },
    { id: 'chart', label: '个股', icon: 'search', title: '个股行情', sub: 'K线 / 均线 / 成交量 · 含自选' },
    { id: 'watch', label: '自选', icon: 'star', title: '自选股', sub: '分组跟踪 · 加进来就打分' },
    { id: 'hotspot', label: '风口', icon: 'events', title: '当日风口', sub: '涨停梯队 / 板块强度 / 主力资金' },
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
    // 后端 _age_days 返回的是**交易日**数，不是自然日 —— 写成「天」会夸大
    // （周五→周一差 3 个自然日，只隔 1 个交易日）。
    else if (age !== null && age > 1) setHealth('stale', '落后 ' + age.toFixed(0) + ' 个交易日');
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

  /**
   * 检查版本。
   *
   * force=true 时带 ?force=1 —— 服务端会**立刻**去 GitHub 查一次，
   * 而不是读 6 小时前的缓存。
   *
   * 实测踩过：v0.23.1 的 Release 15:32 就建好了，但定时任务 15:26 跑过，
   * 缓存里还是 0.23.0，用户反复点检查都看不到更新。「检查更新」不真的
   * 检查，是最让人火大的假功能。
   */
  async function loadUpdate(force) {
    const url = force ? '/api/update?force=1' : '/api/update';
    const res = await apiCached(url, force ? 0 : 5000);
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

  /**
   * 更新进度看板。
   *
   * 之前的实现只有一个转圈 + 「等待首次检查…」，用户完全不知道：
   *   · 开始了吗
   *   · 在拉镜像还是重建容器
   *   · 卡住了还是正常
   *
   * 现在读 /api/update/progress，显示：
   *   阶段名 + 进度条 + 已耗时 + 预计剩余 + 每一步的时间线
   *
   * 关键：服务重启期间请求会失败 —— 那不是错误，是正常现象，
   * 明确提示「服务正在重启」，避免用户以为挂了。
   */
  function startUpdateWatch(to, note) {
    S.updating = true;
    renderUpdateBadge();
    const body = $('#modal-body');
    const foot = $('#modal-foot');
    if (body) {
      body.innerHTML =
        '<div id="upd-panel" style="min-height:150px">' +
        '<div style="text-align:center;padding:4px 0 2px">' +
        '<p>更新中，目标版本 <b>v' + esc(to || '?') + '</b></p>' +
        '<p class="muted" style="margin-top:4px;font-size:12px">' +
          esc(note || '进度会实时刷新，服务重启期间页面会短暂断开') + '</p>' +
        '</div>' +
        '<div class="upd-bar" style="margin:14px 0 8px;height:6px;border-radius:3px;' +
          'background:var(--surface-3,#1c2434);overflow:hidden">' +
          '<div id="upd-fill" style="height:100%;width:6%;background:var(--brand,#4a7dff);' +
            'transition:width .5s"></div>' +
        '</div>' +
        '<div style="display:flex;justify-content:space-between;font-size:12px">' +
          '<span id="upd-stage" class="muted">排队中</span>' +
          '<span id="upd-eta" class="mono muted">—</span>' +
        '</div>' +
        '<div id="upd-steps" class="mono" style="margin-top:12px;font-size:11px;' +
          'color:var(--text-3);line-height:1.7;max-height:150px;overflow:auto"></div>' +
        '</div>';
    }
    if (foot) {
      foot.innerHTML = '<button class="btn ghost" type="button" data-close="1">后台继续</button>';
      $$('#modal-foot [data-close]').forEach((btn) => btn.addEventListener('click', closeModal));
    }

    const from = (S.version && S.version.version) || (S.update && S.update.current) || '';
    let tries = 0;
    let servRestart = false;

    const paint = (d) => {
      const fill = $('#upd-fill');
      const stage = $('#upd-stage');
      const eta = $('#upd-eta');
      const steps = $('#upd-steps');
      const pct = Math.max(3, Math.min(100, Number(d.percent) || 0));
      if (fill) fill.style.width = pct + '%';
      if (stage) {
        stage.textContent = (d.stage_label || d.stage || '进行中') + ' · ' + pct + '%';
      }
      if (eta) {
        const el = Number(d.elapsed_seconds);
        const rem = Number(d.eta_seconds);
        let t = '';
        if (isFinite(el) && el > 0) t += '已用 ' + Math.round(el) + 's';
        if (isFinite(rem) && rem > 0 && d.state === 'running') {
          t += (t ? ' · ' : '') + '预计还需 ' + Math.round(rem) + 's';
        }
        eta.textContent = t || '—';
      }
      if (steps && Array.isArray(d.steps)) {
        steps.innerHTML = d.steps.slice(-8).map((x) => {
          const mark = x.ok === false ? '✗' : '✓';
          const t = String(x.at || '').slice(11, 19);
          return '<div>' + mark + ' ' + t + ' ' + esc(x.label || x.stage) +
            (x.detail ? ' <span style="opacity:.7">' + esc(String(x.detail).slice(0, 60)) + '</span>' : '') +
            '</div>';
        }).join('');
      }
    };

    const timer = setInterval(async () => {
      tries += 1;

      // 1) 先看进度
      const pr = await api('/api/update/progress');
      if (pr && pr.ok && pr.data) {
        servRestart = false;
        const d = pr.data;
        paint(d);

        if (d.state === 'done') {
          clearInterval(timer);
          S.updating = false;
          S.update = null;
          toast('更新完成：' + (d.target || ''), 'ok');
          setTimeout(() => { try { location.reload(); } catch (e) { /* ignore */ } }, 2500);
          return;
        }
        if (d.state === 'failed' || d.state === 'rolled_back') {
          clearInterval(timer);
          S.updating = false;
          renderUpdateBadge();
          const stage = $('#upd-stage');
          if (stage) {
            stage.className = 'err-text';
            stage.textContent = (d.stage_label || '更新失败') +
              (d.error ? '：' + String(d.error).slice(0, 80) : '');
          }
          toast(d.state === 'rolled_back' ? '更新失败，已回滚' : '更新失败', 'err');
          return;
        }
        if (d.state === 'stale') {
          const stage = $('#upd-stage');
          if (stage) {
            stage.className = 'warn-text';
            stage.textContent = '超过 10 分钟没有进展 —— 辅助容器可能已退出，请检查日志';
          }
        }
      } else {
        // 请求失败 = 服务正在重启，这是更新流程的正常一环
        servRestart = true;
        const stage = $('#upd-stage');
        if (stage) stage.textContent = '服务正在重启（页面暂时断开）…';
      }

      // 2) 再确认版本是否已变
      const res = await api('/api/version');
      if (res && res.ok && res.data && res.data.version &&
          from && res.data.version !== from) {
        clearInterval(timer);
        S.updating = false;
        S.update = null;
        const stage = $('#upd-stage');
        if (stage) stage.textContent = '已更新到 v' + res.data.version;
        toast('已更新到 v' + res.data.version + '，正在刷新…', 'ok');
        setTimeout(() => { try { location.reload(); } catch (e) { /* ignore */ } }, 2000);
        return;
      }

      // 3) 超时保护：3 分钟
      if (tries >= 60) {
        clearInterval(timer);
        S.updating = false;
        renderUpdateBadge();
        const stage = $('#upd-stage');
        if (stage) {
          stage.className = 'warn-text';
          stage.textContent = servRestart
            ? '服务仍未恢复 —— 请在 NAS 上查看：docker logs apb'
            : '等待超时 —— 请手动刷新页面确认版本';
        }
      }
    }, 3000);
  }

  /* ------------------------- 设置页：检查更新卡片 ---------------------- */

  /**
   * 「远端最新」这个标签不能直接显示缓存值。
   *
   * 实测：页面上写着「远端最新 v0.0.5」，而当前版本是 v0.28.4 ——
   * 那是 update.json 里一个过期的占位值。把「比当前还旧/无法解析」的值
   * 标成「最新」，比不显示更糟：用户会以为服务端只到 0.0.5。
   */
  /**
   * 把成交时刻标成「盘中 / 盘前 / 盘后 / 非交易日」。
   *
   * 只做字符串到标签的映射，不判断「现在」—— 交易时段规则由后端
   * `qcore/session.py` 唯一定义，这里只解释一个**已发生的时刻**。
   * 目的很具体：让「盘中用收盘价成交」这种异常一眼可见。
   */
  function _sessionTag(iso) {
    const s0 = String(iso || '');
    const m = s0.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
    if (!m) return '';
    const d = new Date(m[1] + '-' + m[2] + '-' + m[3] + 'T00:00:00');
    if (isNaN(d.getTime())) return '';
    const wd = d.getDay();
    if (wd === 0 || wd === 6) return '非交易日';
    const hm = Number(m[4]) * 60 + Number(m[5]);
    if (hm < 9 * 60 + 30) return '盘前';
    if (hm <= 11 * 60 + 30) return '盘中';
    if (hm < 13 * 60) return '午休';
    if (hm <= 15 * 60) return '盘中';
    if (hm <= 17 * 60) return '盘后';
    return '非交易时段';
  }

  function latestLabel(latest, cur) {
    if (!latest) return '未知（未成功检查）';
    const ok = /^v?\d+\.\d+\.\d+$/.test(String(latest));
    if (!ok) return '未知（服务端返回非法版本号 ' + latest + '）';
    if (cmpVersion(latest, cur) < 0) {
      return 'v' + latest + '（比当前旧，已忽略）';
    }
    return 'v' + latest;
  }

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
        kvItem('远端最新', latestLabel(latest, cur)),
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

  /**
   * 把后端错误码翻译成人话。
   *
   * 后端返回 {error: "market_mismatch", hint: "..."}，
   * 之前直接显示 error，用户看到「调仓失败: market_mismatch」——
   * 完全不知道该怎么办。
   *
   * 优先级：hint（后端已写好的解释）> 映射表 > 原始码
   */
  function errText(r) {
    if (!r) return '未知错误';
    if (r.hint) return r.hint;
    const M = {
      market_mismatch: '这份预测属于另一个市场，不能给当前账户下单',
      picks_market_mismatch: '这份预测的股票代码不属于当前市场',
      no_picks: '这个周期还没有预测，需要先生成',
      no_model: '这个周期还没有模型，需要先在训练机训练',
      picks_empty: '预测列表是空的',
      already_rebalanced_today: '今天已经调过仓了，不重复交易',
      hold_period_not_elapsed: '持有期还没到，现在不该调仓',
      not_initialized: '账户还没初始化',
      bad_capital: '本金金额无效',
      reset_failed: '清空失败',
      unauthorized: '没有权限（需要登录或令牌）',
      internal_error: '服务内部错误',
    };
    const k = String(r.error || '');
    return M[k] || k || (r.detail ? String(r.detail).slice(0, 120) : '未知错误');
  }

  /**
   * 数据日期显示成「2026-11-06 周五」并提示落后几个交易日。
   *
   * 用户看到「估值日 2026-11-06」会以为卡住了 —— 但今天是 11-09（周一），
   * 11-07/08 是周末，11-06 就是最新交易日。把星期和落后天数写出来，
   * 一眼就能判断是「正常」还是「数据没同步」。
   */
  function dataAsOf(d, lagDays) {
    if (!d) return '—';
    const s0 = String(d).slice(0, 10);
    const dt = new Date(s0 + 'T00:00:00');
    if (isNaN(dt.getTime())) return s0;
    const wd = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'][dt.getDay()];
    // 落后几个工作日（粗略：只数周一~周五）
    const now = new Date();
    let back = 0;
    const cur = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const tgt = new Date(dt.getTime());
    while (tgt < cur) {
      tgt.setDate(tgt.getDate() + 1);
      const w = tgt.getDay();
      if (w !== 0 && w !== 6) back += 1;
    }
    // 滞后天数优先用后端给的（真实交易日历），前端只做兜底估算
    if (typeof lagDays === 'number' && isFinite(lagDays)) back = lagDays;
    let note = '';
    if (back <= 0) note = '（今天）';
    else if (back === 1) note = '（落后 1 个交易日）';
    else if (back > 3) note = '（落后 ' + back + ' 个交易日，检查同步）';
    else note = '（落后 ' + back + ' 个交易日）';
    return s0 + ' ' + wd + note;
  }

  /** 交易原因中文化：rebalance_in -> 新进入 top-k */
  function reasonText(r) {
    const M = {
      rebalance_in: '新进入 top-k',
      rebalance_out: '掉出 top-k',
      take_profit: '止盈',
      stop_loss: '止损',
      trailing_stop: '移动止损',
      add: '加仓',
      reduce: '减仓',
      manual: '手工录入',
      hold_expired: '持有到期',
    };
    const k = String(r || '');
    return M[k] || k || '—';
  }

  const fmtMoney = (v, cur) => {
    const n = toNum(v);
    if (n === null) return '—';
    return n.toLocaleString('zh-CN') + (cur ? ' ' + cur : ' 元');
  };

  /** 0.15 -> "15"（百分数显示）。仅单位换算，不取整、不纠正 */
  function pctOf(v) {
    const n = toNum(v);
    return n === null ? '' : String(+(n * 100).toFixed(4));
  }

  /* ===================== 模拟实盘：市场隔离（CN/US/HK） ================= */

  const MKTS = ['CN', 'US', 'HK'];
  const MKT_LABEL = { CN: 'A股', US: '美股', HK: '港股' };
  const mktLabel = (m) => MKT_LABEL[m] || String(m || '—');

  /** 本金压缩显示（万 / 亿）——纯单位换算 */
  function compactMoney(v) {
    const n = toNum(v);
    if (n === null) return '—';
    const abs = Math.abs(n);
    if (abs >= 1e8) return (+(n / 1e8).toFixed(2)) + '亿';
    if (abs >= 1e4) return (+(n / 1e4).toFixed(2)) + '万';
    return String(Math.round(n));
  }

  /** 各市场实测往返成本（从 performance.costs.scenarios 里挑 10 万名义那一行，不做计算） */
  async function loadCostMap() {
    if (S.costMap) return S.costMap;
    const map = {};
    const res = await apiCached('/api/performance');
    const scen = res.ok && res.data && res.data.costs && res.data.costs.scenarios;
    if (Array.isArray(scen)) {
      scen.forEach((r) => {
        if (!r || !r.market) return;
        const bps = toNum(r.round_trip_bps);
        if (bps === null) return;
        const notional = toNum(r.notional);
        if (!(r.market in map) || notional === 100000) {
          map[r.market] = { bps: bps, notional: notional };
        }
      });
    }
    S.costMap = map;
    return map;
  }

  /** 只有「响应自报属于该市场」才用；A 股直接用共享的 paper.json */
  function paperForMarket(res, mkt) {
    if (!res || !res.ok || !res.data) return null;
    const d = res.data;
    if (mkt === 'CN') return d;
    const declared = String(d.market || '').toUpperCase();
    if (declared === mkt) return (d.markets && d.markets[mkt]) || d;
    return null;
  }

  function marketTabsHtml(summary, current) {
    const markets = (summary && summary.markets) || {};
    return '<div class="mkt-tabs" role="tablist" aria-label="市场切换">' + MKTS.map((m) => {
      const cfg = markets[m] || null;
      const cap = cfg ? compactMoney(cfg.initial_capital) : '—';
      const cur = (cfg && cfg.currency) || '';
      const on = m === current;
      return '<button type="button" class="mkt-tab' + (on ? ' is-active' : '') + '" role="tab"' +
        ' data-mkt="' + m + '" aria-selected="' + (on ? 'true' : 'false') + '"' +
        ' title="' + esc(mktLabel(m) + (cur ? ' · ' + cur : '') + (cfg ? ' · 本金 ' + cap : ' · 配置缺失')) + '">' +
        '<span class="mt-name">' + esc(mktLabel(m)) + '</span>' +
        '<span class="mt-dot">·</span>' +
        '<span class="mt-cap">' + esc(cap) + '</span>' +
        '</button>';
    }).join('') + '</div>';
  }

  function marketPlaceholder(mkt, res) {
    const isCN = mkt === 'CN';
    const cmd = 'python -m qcore.pipeline' + (isCN ? '' : ' --market ' + mkt);
    const why = (res && res.error === 'not_ready')
      ? '缺少 <code>data/state/' + esc(res.missing || 'paper.json') + '</code>，'
      : '';
    const tail = isCN
      ? 'A 股模拟盘由 <code>paper.json</code> 提供；跑一次全流程即可生成。'
      : '<code>paper.json</code> 只包含 A 股模拟盘；' + esc(mktLabel(mkt)) +
        '需要单独跑一次 pipeline 才会产出 <code>paper_' + esc(mkt) + '.json</code>。';
    return card({
      title: mktLabel(mkt) + ' · 尚无模拟盘数据',
      tools: '<span class="tag na">' + esc(mkt) + '</span>',
      body: '<div class="empty">' + ICON('alert') +
        '<div class="t">该市场尚无模拟盘数据</div>' +
        '<div class="d">' + why + '先运行 <code>' + esc(cmd) + '</code> 生成。' +
        '<br>在生成之前，这里不会显示任何来自其他市场的数据。</div></div>' +
        '<p class="muted" style="margin-top:10px">' + tail +
        ' 实盘成交录入（<code>live_trades.jsonl</code>）只在 A 股对账中生效。</p>'
    });
  }

  function accountCardHtml(cfg, mkt, costMap, cur) {
    const c = cfg || {};
    const market = mkt || c.market || 'CN';
    const label = mktLabel(market);
    const currency = cur || c.currency || '';
    const measured = (costMap && costMap[market]) ? costMap[market] : null;
    const costHint = measured
      ? '实测往返 ' + num(measured.bps, 2) + ' bps' +
        (measured.notional ? '（' + compactMoney(measured.notional) + ' 名义）' : '')
      : '实测往返成本见「风险体检 → 成本明细」';

    const sub = 'data/state/account.json · ' + esc(label) +
      (currency ? '（' + esc(currency) + '）' : '') + ' · ' +
      (c.updated_at ? '更新于 ' + esc(c.updated_at) : '尚未保存过');

    return card({
      title: '账户设置 · ' + label + (currency ? '（' + currency + '）' : ''),
      sub: sub,
      tools: '<span class="tag na" title="未设 APB_TOKEN 时只允许内网修改">内网免 token</span>',
      body: '<form class="form" id="acct-form" autocomplete="off" novalidate>' +
        '<input type="hidden" name="market" value="' + esc(market) + '">' +
        '<label class="field"><span>初始资金（' + esc(currency || '元') + '）</span>' +
        '<input type="number" name="initial_capital" step="1000" value="' + esc(rawVal(c.initial_capital)) +
        '" placeholder="1000000">' +
        '<span class="hintline" id="acct-cap-hint">' + esc(fmtMoney(c.initial_capital, currency)) + '</span></label>' +
        '<label class="field"><span>持仓数 top_k</span>' +
        '<input type="number" name="top_k" step="1" value="' + esc(rawVal(c.top_k)) + '" placeholder="20"></label>' +
        '<label class="field"><span>单票上限（%）</span>' +
        '<input type="number" name="max_weight_pct" step="1" value="' + esc(pctOf(c.max_weight)) + '" placeholder="15">' +
        '<span class="hintline">填百分数：15 表示 15%</span></label>' +
        '<label class="field"><span>成本 cost_bps</span>' +
        '<input type="number" name="cost_bps" step="0.01" value="' + esc(rawVal(c.cost_bps)) + '" placeholder="10">' +
        '<span class="hintline">' + esc(costHint) + '</span></label>' +
        '<label class="field"><span>持有期 hold（交易日）</span>' +
        '<input type="number" name="hold" step="1" value="' + esc(rawVal(c.hold)) + '" placeholder="5"></label>' +
        '<div class="form-actions">' +
        '<button class="btn primary" type="submit">' + ICON('check') + '保存</button>' +
        '<span class="form-msg" id="acct-msg"></span></div>' +
        '</form>' +
        // 每个周期单独设本金（在 form 之外，单独一个保存按钮）
        '<div style="margin-top:18px;padding-top:16px;border-top:1px solid var(--line,#26303f)">' +
        '<div style="font-weight:600;margin-bottom:6px">各周期本金（' + esc(currency || '元') + '）</div>' +
        '<p class="muted" style="margin:0 0 10px;font-size:12px">' +
        '不同周期的最优资金规模不一样：T+1 换手极快，本金大了成本吃光收益；' +
        '15 天可以放更多。留空表示用上面的「初始资金」。</p>' +
        '<div class="form" style="gap:10px">' +
        [1, 3, 5, 10, 15].map(function (h) {
          const lbl = ({ 1: 'T+1 短线', 3: 'T+3 中短线', 5: '1 周 中线',
                         10: '2 周', 15: '15 天 中长线' })[h] || ('h' + h);
          const v = (c.horizon_capital && c.horizon_capital[String(h)] != null)
            ? c.horizon_capital[String(h)] : '';
          return '<label class="field"><span>' + esc(lbl) + '</span>' +
            '<input type="number" class="hz-cap" data-h="' + h + '" step="1000" ' +
            'value="' + esc(v === '' ? '' : String(v)) + '" placeholder="' +
            esc(rawVal(c.initial_capital)) + '"></label>';
        }).join('') +
        '</div>' +
        '<div class="form-actions" style="margin-top:12px">' +
        '<button class="btn primary" type="button" id="hz-cap-save">' +
        ICON('check') + '保存各周期本金</button>' +
        '<span class="form-msg" id="hz-cap-msg"></span></div></div>' +
        '<p class="acct-note">' + ICON('alert') +
        '<span>改本金只影响<b>下一次引擎运行</b>，不会重算历史净值与台账；' +
        '三个市场各自独立，互不影响。</span></p>'
    });
  }

  /** 保存各周期本金（独立于基础设置） */
  async function submitHorizonCapital(market) {
    const msg = $('#hz-cap-msg');
    const hc = {};
    $$('.hz-cap').forEach(function (el) {
      const raw = String(el.value || '').trim();
      if (raw === '') return;                  // 留空 = 不设置这一项
      const n = Number(raw);
      if (!isFinite(n) || n <= 0) return;
      hc[String(el.getAttribute('data-h'))] = n;
    });
    if (!Object.keys(hc).length) {
      if (msg) { msg.className = 'form-msg err'; msg.textContent = '没有填写任何金额'; }
      return;
    }
    const r = await api('/api/account/config', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ market: market, horizon_capital: hc })
    });
    if (r && r.ok) {
      if (msg) { msg.className = 'form-msg ok'; msg.textContent = '已保存'; }
      toast('各周期本金已保存', 'ok');
      rerender();
    } else {
      const t = (r && (r.hint || r.error)) || '保存失败';
      if (msg) { msg.className = 'form-msg err'; msg.textContent = String(t).slice(0, 120); }
      toast('保存失败：' + errText(r), 'err');
    }
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

    const body = { market: (form.elements.market && form.elements.market.value) || S.paperMarket || 'CN' };
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

    if (Object.keys(body).length <= 1) {          // 只有 market，没有实际字段
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
      const savedMkt = body.market;
      const savedCur = (res.data && res.data.currency) || '';
      msg.className = 'form-msg ok';
      msg.textContent = '已保存 ' + mktLabel(savedMkt) + (savedCur ? '（' + savedCur + '）' : '');
      toast('已保存 ' + mktLabel(savedMkt) + (savedCur ? '（' + savedCur + '）' : '') +
        ' 配置，下次运行引擎生效', 'ok');
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
    // 周期选择：默认 T+1。切换会重新拉该周期的 picks。
    const H = (S.modelHorizon == null) ? 1 : S.modelHorizon;
    const [modelRes, picksRes, impRes, hzRes] = await Promise.all([
      apiCached('/api/model'),
      apiCached('/api/picks?limit=500' + (H !== 1 ? '&horizon=' + H : '')),
      apiCached('/api/factors/importance?top=200'),
      apiCached('/api/horizons', 8000)
    ]);
    const hz = (hzRes && hzRes.ok && hzRes.data) ? hzRes.data : {};
    const hzList = hz.horizons || [];

    const m = (modelRes.ok && modelRes.data) ? modelRes.data : null;
    const picks = (picksRes.ok && picksRes.data) ? picksRes.data : null;
    const _hzInfo = (hzList.find((x) => Number(x.horizon) === H)) || {};
    const _hzNoModel = !_hzInfo.model_exists;
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
    // 配色必须与同文件 signClass() 一致：正 = 好。旧写法把正 IC 染成「跌」色，
    // 与 modelBarHtml 的 is-pos/is-neg 相反 —— 同一页两套相反映射。
    const icCls = ic === null ? '' : signClass(ic);

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
        '<span class="hm-v ' + (ric === null ? '' : signClass(ric)) + '">' +
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
      kpi({ k: '回测持有期', v: num(bt.hold_days, 0) + ' 日' +
          (num(bt.hold_days, 0) === Number(H) ? '' : '（该回测按 h=' +
            num(bt.hold_days, 0) + ' 跑，与当前所选 h=' + Number(H) + ' 不同）'), sm: true,
           tone: 'neutral',
           title: '这是模型训练/回测时用的持有天数，不是上面选中的周期' }),
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
        ? 'as_of ' + esc(picks.as_of || '—') +
          ' · ' + (picks.horizon_label
            ? '周期 ' + esc(picks.horizon_label)
            : (picks.horizon ? '周期 h' + esc(num(picks.horizon, 0)) : '周期 —')) +
          ' · 共 ' + esc(num(picks.n, 0)) + ' 只' +
          (picks.top_k ? ' · top_k ' + esc(num(picks.top_k, 0)) : '')
        : '未就绪',
      cls: 'flush',
      tools: (picks
        ? '<span class="tag ok">持有中 = 缓冲带内不动</span>'
        : '<button type="button" class="btn ghost sm" id="md-predict">' +
          ICON('play') + '生成该周期预测</button>') +
        '<button type="button" class="btn ghost sm" id="md-predict-any" style="margin-left:6px">' +
        ICON('play') + '补齐全部周期</button>',
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
            render: (r) => '<span class="' + signClass(r.chg_pct) + '">' + esc(pctPoints(r.chg_pct)) + '</span>'
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

    // ---- 周期切换 + 各周期战绩 ----
    // 只显示该市场支持的周期。T+0 在 A 股不存在（T+1 制度），
    // 之前把它也列出来，点进去就是「缺少 picks_h0.json」，很困惑。
    const hzUsable = hzList.filter((x) => x.market_supported !== false);
    const hzTabs = hzUsable.map((x) => {
      const on = Number(x.horizon) === H;
      return '<button type="button" class="seg' + (on ? ' is-active' : '') +
        (x.model_exists ? '' : ' is-dim') + '" data-hz="' + x.horizon + '">' +
        esc(x.label || ('h' + x.horizon)) +
        (x.model_exists ? '' : ' <span style="opacity:.6">·</span>') + '</button>';
    }).join('');

    const perfRows = hzUsable.map((x) => {
      const pf = x.perf || {};
      const n = pf.n || 0;
      const hr = pf.hit_rate;
      const ex = pf.excess_mean;
      const tone = ex == null ? '' : (ex > 0 ? 'is-up' : 'is-down');
      return '<tr' + (Number(x.horizon) === H ? ' class="row-on"' : '') + '>' +
        '<td>' + esc(x.label || ('h' + x.horizon)) + '</td>' +
        '<td>' + (x.model_exists ? '<span class="tag hit">已训</span>'
                                 : '<span class="tag pending">未训</span>') + '</td>' +
        '<td class="num">' + (n || '—') + '</td>' +
        '<td class="num">' + (hr == null ? '—' : (hr * 100).toFixed(1) + '%') + '</td>' +
        '<td class="num ' + tone + '">' + (ex == null ? '—' :
          (ex > 0 ? '+' : '') + (ex * 100).toFixed(2) + '%') + '</td>' +
        '<td class="num">' + (pf.pending || 0) + '</td>' +
        '<td class="num muted">' + (x.cost_drag_annual == null ? '—' :
          x.cost_drag_annual.toFixed(1) + '%') + '</td>' +
        '</tr>';
    }).join('');

    const horizonCard = card({
      title: '持有周期', sub: '每个周期一套独立模型、一本独立台账',
      tools: '<div class="segmented" id="hz-tabs">' + hzTabs + '</div>',
      body:
        '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
          '<th>周期</th><th>模型</th><th>已对账</th><th>命中率</th>' +
          '<th>超额</th><th>待对账</th><th>年成本</th>' +
        '</tr></thead><tbody>' + perfRows + '</tbody></table></div>' +
        '<p class="muted" style="margin-top:10px">' +
          '命中率 = top 三分之一跑赢全体平均的期数占比；' +
          '超额 = top 组平均收益 − 全体平均。样本 &lt; 30（MIN_TRADES）一律标 insufficient，只能当观察。' +
        '</p>'
    });

    return '<div style="margin-bottom:14px">' + horizonCard + '</div>' +
      '<div style="margin-bottom:14px">' + heroCard + '</div>' +
      '<div style="margin-bottom:14px">' + btCard + '</div>' +
      '<div style="margin-bottom:14px">' +
        (_hzNoModel
          ? banner({ tone: 'warn', icon: 'alert',
              title: (esc(_hzInfo.label || ('h' + H))) + ' 这个周期还没训练',
              text: '上面「持有周期」表里该行显示「未训」。训练完并跑一次预测后才有推荐。',
              meta: '命令：python -m qcore.live_model --train --horizon=' + H +
                    ' 然后 --predict --horizon=' + H })
          : picksCard) + '</div>' +
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
    const [perfRes, regimeRes, paperRes, statusRes, modelRes, onlineRes] = await Promise.all([
      apiCached('/api/performance'), apiCached('/api/regime'),
      apiCached('/api/paper'), apiCached('/api/status'), apiCached('/api/model'),
      apiCached('/api/online/learn', 5000)
    ]);
    const perf = perfRes.ok ? adaptPerformance(perfRes.data) : null;
    const status = statusRes.ok ? statusRes.data : null;
    const regimeRows = (regimeRes.ok && regimeRes.data && regimeRes.data.rows) || [];
    const paper = (paperRes.ok && paperRes.data) || null;

    // 样本量必须取 meta.n_stocks：顶层 n_stocks 曾经把 scores.shape[1]（= 天数）
    // 当成股票数写进去，页面显示「2633 只」而真值 112 只，错 23.5 倍。
    // 广度是 IR ≈ IC·√BR 的分子，显示错值会直接误导「有多大优势」的判断。
    const nStk = (status && status.meta && status.meta.n_stocks) ||
      (perf && perf.dataset && perf.dataset.N) || null;
    setSub(status
      ? (status.data_from || '—') + ' → ' + (status.data_to || '—') + ' · ' +
        (nStk === null ? 'N —' : 'N=' + nStk + ' 只') + ' × ' +
        (status.n_signals || '—') + ' 信号 · T=' + (status.n_days || '—') +
        ' 天 · h=' + (status.horizon || '—')
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
          : 'HAC t = ' + num(icT, 2) +
            ' · n=' + num(edge5 && edge5.ic && edge5.ic.n_obs, 0) +
            ' · lags=' + (edge5 && edge5.h ? edge5.h - 1 : 4) +
            (edge5 && edge5.ic_verdict ? ' · ' + esc(edge5.ic_verdict) : '')
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
        // 引擎自己的判定句（edge[h].conclusion）比 meta.verdict（常常是「不表态」）
        // 更具体，优先展示；缺失时回退并显式标注。
        title: (edge5 && edge5.conclusion) || (perf.meta && perf.meta.verdict) || '引擎未给结论',
        text: '学习器 IC ' + num(lv.learned_ic, 4) + '（t=' + num(lv.learned_t, 2) + '） vs 等权 IC ' +
          num(lv.equal_ic, 4) + '（t=' + num(lv.equal_t, 2) + '） · 差 ' + num(lv.diff_ic, 4) +
          '（p=' + num(lv.diff_p, 3) + '）' + (lv.verdict ? '<br>' + esc(lv.verdict) : ''),
        meta: 'Deflated Sharpe ' + num(edge5 && edge5.long_short && edge5.long_short.dsr, 4) +
          (edge5 && edge5.long_short && edge5.long_short.dsr_significant === false ? '（未过 0.95）' : '') +
          ' · n_trials=' + num(edge5 && edge5.n_trials, 0) +
          ' · 相关性有效信号 ' + num(perf.risk && perf.risk.signal_level &&
            perf.risk.signal_level.corr_adjusted && perf.risk.signal_level.corr_adjusted.effective_n_corr, 1) +
          ' · 生成于 ' + ((perfRes.data && perfRes.data.generated_at) || '—')
      });
    } else if (!perf) {
      bannerHtml = notReadyBox(perfRes);
    }

    /* --- 在线学习：校准参数 a/b 与更新历史 --- */
    // 这是「自学习」真正在跑的地方：用已到期样本学 ret ≈ a*score + b。
    // a 为负说明模型方向反了；b 是整体偏移修正。每次更新都记历史。
    let onlineCard = '';
    const OL = (onlineRes && onlineRes.ok && onlineRes.data) ? onlineRes.data : null;
    if (OL && Array.isArray(OL.horizons)) {
      const learned = OL.horizons.filter((x) => (x.n_updates || 0) > 0);
      const rows = OL.horizons.map((x) => {
        const has = (x.n_updates || 0) > 0;
        const aCls = has ? (x.a < 0 ? 'is-down' : 'is-up') : '';
        return '<tr>' +
          '<td>h' + esc(x.horizon) + '</td>' +
          '<td class="num ' + aCls + '">' + (has ? num(x.a, 4) : '—') + '</td>' +
          '<td class="num">' + (has ? num(x.b, 6) : '—') + '</td>' +
          '<td class="num">' + (x.n_samples || 0) +
            (x.n_pending ? '<span class="muted" style="margin-left:4px">+' +
              x.n_pending + ' 待到期</span>' : '') + '</td>' +
          '<td class="num">' + (x.n_updates || 0) + '</td>' +
          '<td class="muted">' + esc((x.updated_at || '—').slice(5, 16)) + '</td>' +
          '</tr>';
      }).join('');
      // 样本为 0 的周期必须说明原因，否则用户看到的就是「在线学习没做好」。
      // 实测：5 个周期只有 h=5 有样本，原因各不相同（没模型 / 没到期 / 主台账）。
      const zeroWhy = OL.horizons
        .filter((x) => (x.n_samples || 0) === 0 && x.why_zero)
        .map((x) => 'h' + esc(x.horizon) + '：' + esc(x.why_zero));
      const zeroHtml = zeroWhy.length
        ? '<p class="muted" style="margin-top:10px">样本为 0 的原因 —— ' +
          zeroWhy.join('；') + '</p>'
        : '';
      const hist = learned.length
        ? learned[learned.length - 1].history || []
        : [];
      const histHtml = hist.length ? (
        '<div style="margin-top:14px"><div style="font-weight:600;margin-bottom:6px">' +
        '最近一次调整</div><div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>时间</th><th>样本</th><th>a</th><th>b</th><th>MSE</th><th>方向</th>' +
        '</tr></thead><tbody>' + hist.slice().reverse().map((e) =>
          '<tr><td class="mono">' + esc(String(e.at || '').slice(5, 16)) + '</td>' +
          '<td class="num">' + e.n_samples + '</td>' +
          '<td class="num mono">' + num(e.a_before, 4) + ' → ' + num(e.a_after, 4) + '</td>' +
          '<td class="num mono">' + num(e.b_before, 6) + ' → ' + num(e.b_after, 6) + '</td>' +
          '<td class="num mono">' + num(e.mse_before, 8) + ' → ' + num(e.mse_after, 8) + '</td>' +
          '<td>' + esc(e.direction || '—') + '</td></tr>').join('') +
        '</tbody></table></div></div>') : '';

      onlineCard = card({
        title: '在线学习', sub: 'qcore/online_calib · ret ≈ a × score + b',
        tools: '<button type="button" class="btn ghost sm" id="ol-run">' +
          ICON('play') + '立即学习一轮</button>',
        body: '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
          '<th>周期</th><th>a（区分力）</th><th>b（偏移）</th>' +
          '<th>样本</th><th>更新次数</th><th>最近更新</th>' +
          '</tr></thead><tbody>' + rows + '</tbody></table></div>' +
          zeroHtml +
          (learned.length
            ? '<p class="muted" style="margin-top:10px">a &lt; 0 表示模型给的分数越高、' +
              '实际收益反而越低（方向反了，系统会自己把分数翻过来）；' +
              'b 是整体偏移修正。每学到新样本更新一次，全部历史可查。</p>'
            : '<p class="muted" style="margin-top:10px">还没有可用的已到期样本。' +
              '等台账累积到 30 个样本以上，点「立即学习一轮」即可。</p>') +
          histHtml
      });
    }

    /* --- 学习器为什么无效：逐条给原因，不只说「无显著差异」 --- */
    let diagCard = '';
    const dg = (perf && perf.learned && perf.learned.diagnosis) || null;
    if (dg && dg.ok !== false && ((dg.reasons || []).length || (dg.checks || []).length)) {
      const chk = (dg.checks || []).map((c) =>
        '<tr><td>' + esc(c.name) + '</td>' +
        '<td class="num mono">' + esc(c.value == null ? '—' : String(c.value)) + '</td>' +
        '<td class="muted">' + esc(c.detail || '') + '</td></tr>').join('');
      const reasons = (dg.reasons || []).map((r) =>
        '<li style="margin:6px 0">' + esc(r) + '</li>').join('');
      const paths = (dg.paths || []).map((r) =>
        '<li style="margin:6px 0">' + esc(r) + '</li>').join('');
      diagCard = card({
        title: '为什么无效', sub: 'qcore/analysis.py · diagnose_learner',
        body:
          (chk ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
            '<th>检查项</th><th>实测</th><th>说明</th></tr></thead><tbody>' +
            chk + '</tbody></table></div>' : '') +
          (reasons ? '<div style="margin-top:14px"><div style="font-weight:600;margin-bottom:6px">' +
            '原因</div><ul style="margin:0;padding-left:20px">' + reasons + '</ul></div>' : '') +
          (paths ? '<div style="margin-top:14px"><div style="font-weight:600;margin-bottom:6px">' +
            '可以试的方向</div><ul style="margin:0;padding-left:20px">' + paths + '</ul></div>' : '') +
          (dg.sensitivity_skipped
            ? '<p class="muted" style="margin-top:10px">参数敏感性扫描（ρ×κ 共 6 组）未运行' +
              '（要跑 6 次完整回测，几分钟）。需要时跑 ' +
              '<code>python -m qcore.pipeline --diagnose</code></p>' : '')
      });
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
      (onlineCard ? '<div style="margin-bottom:14px">' + onlineCard + '</div>' : '') +
      (diagCard ? '<div style="margin-bottom:14px">' + diagCard + '</div>' : '') +
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

  /** 从 ISO 时间戳取 MM-DD HH:MM（字符串切片，不做时区换算）。
   *
   * 旧实现只返回 HH:MM:SS —— 实测把「2026-11-08 周日 18:48」渲染成
   * 「采集于 18:48:08 · 每 60 秒自动刷新」，用户以为那是今天刚采的。
   * 时间必须带日期，否则「几点的」这件事无法判断。
   */
  function clockOf(iso) {
    const str = String(iso || '');
    if (!str) return '—';
    const m = str.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})/);
    if (m) return m[2] + '-' + m[3] + ' ' + m[4];
    const only = str.match(/T(\d{2}:\d{2}:\d{2})/);
    return only ? only[1] : str;
  }

  /** 采集时刻 + 它离最近交易日有多远 —— 单看时刻会被误读成「刚刚」。 */
  function collectedText(iso, lagDays) {
    const base = clockOf(iso);
    const lag = toNum(lagDays);
    if (lag === null) return '采集于 ' + base;
    if (lag <= 0) return '采集于 ' + base;
    return '采集于 ' + base + '（落后 ' + lag + ' 个交易日）';
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

    setSub(collectedText(d.as_of, d.data_age_days) +
      (toNum(d.elapsed_ms) !== null ? ' · 耗时 ' + num(d.elapsed_ms, 0) + 'ms' : '') +
      ' · 每 60 秒自动刷新');

    /* --- 情绪温度：三个市场分别算 --- */
    // A 股用完整口径（涨跌家数+涨停+成交额分位），
    // 港股/美股拿不到逐只家数，用主要指数涨跌做简化映射（界面会标注口径）。
    const sb = d.sentiment_by_market || {};
    const mkOrder = ['CN', 'HK', 'US'];
    const mkNames = { CN: 'A股', HK: '港股', US: '美股' };

    const tempRow = (mk) => {
      const v = sb[mk] || {};
      const ok = v.ok && v.temperature != null;
      const tv = toNum(v.temperature);
      const w = ok ? Math.max(2, Math.min(100, tv)) : 0;
      return '<div style="padding:10px 0;border-bottom:1px solid var(--line,#26303f)">' +
        '<div style="display:flex;justify-content:space-between;align-items:baseline">' +
          '<span style="font-weight:600">' + esc(mkNames[mk] || mk) + '</span>' +
          (ok
            ? '<span><b class="big-num" style="font-size:22px">' + esc(num(tv, 1)) + '</b>' +
              '<span class="unit" style="font-size:12px">/100</span> ' +
              (v.level ? '<span class="tag info">' + esc(v.level) + '</span>' : '') + '</span>'
            : '<span class="tag na">暂不计算</span>') +
        '</div>' +
        (ok
          ? '<div class="temp-scale" style="margin-top:6px"><i style="width:' + w.toFixed(1) + '%"></i>' +
            '<b style="left:25%"></b><b style="left:50%"></b><b style="left:75%"></b></div>'
          : '') +
        '<p class="muted" style="margin-top:5px;font-size:11px">' +
          esc(v.basis || '') +
          (!ok && v.reason ? ' · ' + esc(v.reason) : '') +
        '</p>' +
      '</div>';
    };

    const tempCard = card({
      title: '情绪温度', sub: 'A股完整口径 · 港美股用指数简化',
      cls: tone.cls,
      body: (Object.keys(sb).length
        ? mkOrder.map(tempRow).join('')
        : '<div class="big-num">' + (tempOk ? esc(num(tempVal, 1)) : '—') + '</div>' +
          (!tempOk ? '<p class="muted">' + esc(sent.reason || '历史不足') + '</p>' : ''))
    });

    /* --- 涨停 / 成交额 --- */
    const limitCard = card({
      title: '涨停家数 · A股', sub: 'limit_up · 沪深（含连板）',
      body: '<div class="big-num">' + esc(num(d.limit_up, 0)) + '<span class="unit">家</span></div>' +
        '<p class="muted" style="margin-top:6px">当日封板家数（含连板）</p>'
    });
    const amountCard = card({
      title: '两市成交额 · A股', sub: 'amount_yi · 沪深合计（亿元）',
      body: '<div class="big-num">' + esc(num(d.amount_yi, 1)) + '<span class="unit">亿元</span></div>' +
        '<p class="muted" style="margin-top:6px">沪深两市合计</p>'
    });

    /* --- 市场宽度：flex-grow 让 CSS 分配比例，JS 不做除法 --- */
    const up = toNum(b.up);
    const down = toNum(b.down);
    const flat = toNum(b.flat);
    const hasBreadth = up !== null || down !== null || flat !== null;
    const breadthCard = card({
      title: '市场宽度', sub: 'A股 · breadth · 上涨 / 平盘 / 下跌 家数占比',
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
      '<span>采集时刻 ' + esc(clockOf(d.as_of)) + '</span>' +
      '<span>行情数据日 ' + (d.data_date
        ? esc(dataAsOf(d.data_date, d.data_age_days))
        : '<span title="market.json 未写 data_date（旧产物）">未标注</span>') + '</span>' +
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
    // 两个信号栈并列，不做「主 / 归档」的价值排序 —— 谁是有效模型由统计判定，
    // 不由页面布局判定。页面上所有 IC / t 都从产物读，不许硬编码
    // （AGENTS §7.5 规则五：UI 显示的数字必须有写入方）。
    const fi = await apiCached('/api/factors/importance');
    const perfS = await apiCached('/api/performance');
    const perfD = perfS && perfS.ok ? perfS.data : null;
    const edgeS = perfD && perfD.edge && (perfD.edge['5'] || perfD.edge[5]);
    const sigIc = edgeS && edgeS.ic ? toNum(edgeS.ic.mean) : null;
    const sigT = edgeS && edgeS.ic ? toNum(edgeS.ic.t) : null;
    const sigN = edgeS && edgeS.ic ? toNum(edgeS.ic.n_obs) : null;
    const sigEff = perfD && perfD.risk && perfD.risk.signal_level &&
      perfD.risk.signal_level.corr_adjusted
      ? toNum(perfD.risk.signal_level.corr_adjusted.effective_n_corr) : null;
    const fRows = ((fi && fi.ok && fi.data && fi.data.rows) || [])
      .filter((r) => r && typeof r === 'object')
      .slice()
      .sort((a, b) => toNum(b.importance) - toNum(a.importance));
    const topF = fRows.slice(0, 20);
    const maxImp = topF.length ? toNum(topF[0].importance) : 1;

    const factorCard = topF.length ? card({
      title: 'Alpha158 因子重要性', sub: '来自当前部署的 GBDT 模型 · 共 ' + fRows.length + ' 个因子',
      body: '<div class="bars">' + topF.map((r) => {
        const v = toNum(r.importance) || 0;
        const w = maxImp ? Math.max(2, (v / maxImp) * 100) : 2;
        return '<div class="bar-row">' +
          '<span class="bar-label">' + esc(r.factor || '') + '</span>' +
          '<span class="bar-track"><span class="bar-fill" style="width:' + w.toFixed(1) + '%"></span></span>' +
          '<span class="bar-val">' + (v * 100).toFixed(2) + '%</span></div>';
      }).join('') + '</div>',
      foot: '这是模型真正在用的因子。重要性 = GBDT 分裂增益占比，不代表单因子收益。'
    }) : '';

    const res = await apiCached('/api/signals?top=200');
    if (!res.ok || !res.data) {
      setSub('Alpha158 ' + fRows.length + ' 个因子');
      return '<div style="margin-bottom:14px">' + factorCard + '</div>' + notReadyBox(res);
    }

    let rows = (res.data.top || []).concat(res.data.bottom || []);
    const seen = new Set();
    rows = rows.filter((r) => {
      const k = r && r.sid;
      if (!k || seen.has(k)) return false;
      seen.add(k);
      return true;
    });

    const families = Array.from(new Set(rows.map((r) => r.family).filter(Boolean))).sort();
    setSub('Alpha158 ' + fRows.length + ' 个因子 · 手写信号 ' + (res.data.total || rows.length) + ' 个（并列两栈，不可直接比较）');

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

    // 73 个手写信号是「量化预知」的测量本体，不是归档 —— 不折叠、不降级。
    // 不利结论可以显示得不好看，但不能被藏起来。
    const legacyBanner = banner({
      tone: (sigT !== null && Math.abs(sigT) < 2) ? 'warn' : 'bad',
      icon: 'alert',
      title: sigIc === null
        ? '73 个手写信号：IC 未就绪（缺 /api/performance）'
        : '73 个手写信号：Rank IC ' + num(sigIc, 4) +
          '（HAC t ' + num(sigT, 2) + '，n=' + num(sigN, 0) + '）',
      text: '这是本项目的测量对象本身。上方 Alpha158+GBDT 是另一套并行模型，' +
        '池、区间、因子数都不同，<b>两者不可直接比较</b>；各自的样本量与口径见各自卡片。',
      meta: '73 个信号 · 相关性调整后有效独立信号 ' +
        (sigEff === null ? '—' : num(sigEff, 1)) + ' / 73 · 判定：' +
        (sigIc === null ? '未就绪' : (Math.abs(sigT) < 2 ? '与 0 不可区分' : '显著'))
    });

    const legacyBlock =
      '<div style="margin:14px 0">' + legacyBanner + '</div>' +
      '<div style="margin-bottom:14px">' + barsCard + '</div>' +
      (detail ? '<div style="margin-bottom:14px">' + detail + '</div>' : '') +
      tableCard;

    return '<div style="margin-bottom:14px">' + factorCard + '</div>' + legacyBlock;
  }

  /* ============================== 当日风口 ============================= */

  async function pageHotspot() {
    const res = await apiCached('/api/hotspot', 20000);
    if (!res.ok || !res.data) {
      setSub('未就绪');
      return notReadyBox(res, '当日风口数据缺失');
    }
    const d = res.data;
    const s = d.sentiment || {};
    const zt = d.limit_up || {};
    const dt = d.limit_down || {};
    const ind = d.industry || {};
    const con = d.concept || {};
    const fund = d.fund_flow || {};
    setSub((d.date || '') + ' · 涨停 ' + (s.limit_up || 0) +
      ' / 跌停 ' + (s.limit_down || 0) + ' · ' + (s.read || '—'));

    // ---- 情绪读数 ----
    const sentCard = card({
      title: '情绪读数', sub: d.date || '',
      tools: '<button type="button" class="btn ghost sm" id="hs-refresh">' +
        ICON('refresh') + '重新抓取</button>',
      body:
        '<div class="quote-grid" style="grid-template-columns:repeat(auto-fit,minmax(110px,1fr))">' +
          [['涨停', s.limit_up, s.limit_up >= 60 ? 'is-up' : ''],
           ['跌停', s.limit_down, s.limit_down >= 20 ? 'is-down' : ''],
           ['最高连板', s.max_board ? s.max_board + ' 板' : '—', ''],
           ['涨跌停比', s.ratio, s.ratio >= 5 ? 'is-up' : s.ratio < 1 ? 'is-down' : ''],
           ['读数', s.read, s.read === '强' ? 'is-up' : s.read === '弱' ? 'is-down' : '']]
            .map(([k, v, tone]) => '<div class="qg-item"><span class="qg-k">' + k +
              '</span><span class="qg-v ' + tone + '">' + (v == null ? '—' : v) +
              '</span></div>').join('') +
        '</div>'
    });

    // ---- 涨停梯队 ----
    const ladder = d.ladder || [];
    const ladderCard = ladder.length ? card({
      title: '涨停梯队', sub: '按连板高度排列 · 最高 ' + (s.max_board || 0) + ' 板',
      body: ladder.map((x) => {
        const hot = x.boards >= 3;
        return '<div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid rgba(120,140,180,.10)">' +
          '<div style="min-width:64px"><span class="tag ' + (hot ? 'is-up' : '') + '">' +
            x.boards + ' 板</span></div>' +
          '<div style="min-width:48px" class="muted">' + x.n + ' 家</div>' +
          '<div style="flex:1">' + (x.names || []).map((n) =>
            '<span class="chip">' + esc(n) + '</span>').join('') +
            (x.n > (x.names || []).length ? '<span class="muted"> …</span>' : '') +
          '</div></div>';
      }).join('')
    }) : '';

    // ---- 主线板块 ----
    const mainCard = (d.main_sectors || []).length ? card({
      title: '主线板块', sub: '按涨停家数排 · 这才是资金真去的地方',
      body: '<div class="bars">' + (d.main_sectors || []).map((m) => {
        const max = Math.max.apply(null, (d.main_sectors || []).map((x) => x.limit_up_n).concat([1]));
        const w = Math.max(4, (m.limit_up_n / max) * 100);
        return '<div class="bar-row"><span class="bar-label">' + esc(m.industry) +
          '</span><span class="bar-track"><span class="bar-fill" style="width:' +
          w.toFixed(1) + '%"></span></span><span class="bar-val">' + m.limit_up_n +
          ' 家</span></div>';
      }).join('') + '</div>'
    }) : '';

    // ---- 板块涨幅榜 ----
    const sectorTable = (rows, title, sub) => card({
      title: title, sub: sub,
      body: rows.length ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>板块</th><th>涨幅</th><th>主力净流入</th><th>涨/跌</th><th>领涨股</th>' +
        '</tr></thead><tbody>' + rows.slice(0, 12).map((r) => {
          const tone = r.chg_pct > 0 ? 'is-up' : 'is-down';
          const ftone = r.main_in_yi > 0 ? 'is-up' : 'is-down';
          return '<tr><td>' + esc(r.name) + '</td>' +
            '<td class="num ' + tone + '">' + (r.chg_pct > 0 ? '+' : '') + r.chg_pct + '%</td>' +
            '<td class="num ' + ftone + '">' + (r.main_in_yi > 0 ? '+' : '') +
              r.main_in_yi + ' 亿</td>' +
            '<td class="num muted">' + r.up + '/' + r.down + '</td>' +
            '<td class="muted">' + esc(r.leader || '—') + '</td></tr>';
        }).join('') + '</tbody></table></div>'
        : emptyBox('暂无数据', '')
    });

    const indCard = sectorTable((ind.rows || []), '行业板块涨幅',
      '主力净流入 = 大单+超大单，比涨幅更能说明资金真实意图');
    const conCard = sectorTable((con.rows || []), '概念板块涨幅', '题材热度');

    // ---- 主力资金 ----
    const fundRows = (fund.rows || []).slice(0, 15);
    const fundCard = fundRows.length ? card({
      title: '主力净流入排行', sub: '今日大单+超大单净买入',
      body: '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>代码</th><th>名称</th><th>涨幅</th><th>净流入</th><th>占比</th>' +
        '</tr></thead><tbody>' + fundRows.map((r) => {
          const tone = r.chg_pct > 0 ? 'is-up' : 'is-down';
          return '<tr><td class="mono"><a class="link-btn" data-goto="' + esc(r.code) +
            '">' + esc(r.code) + '</a></td>' +
            '<td>' + esc(r.name) + '</td>' +
            '<td class="num ' + tone + '">' + (r.chg_pct > 0 ? '+' : '') + r.chg_pct + '%</td>' +
            '<td class="num ' + (r.main_in_yi > 0 ? 'is-up' : 'is-down') + '">' +
              r.main_in_yi + ' 亿</td>' +
            '<td class="num muted">' + r.main_pct + '%</td></tr>';
        }).join('') + '</tbody></table></div>'
    }) : '';

    // ---- 跌停 ----
    const dtRows = (dt.rows || []).slice(0, 10);
    const dtCard = dtRows.length ? card({
      title: '跌停', sub: '情绪的反面 · 共 ' + (dt.n || 0) + ' 家',
      body: '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>代码</th><th>名称</th><th>跌幅</th><th>封单</th><th>行业</th>' +
        '</tr></thead><tbody>' + dtRows.map((r) =>
          '<tr><td class="mono"><a class="link-btn" data-goto="' + esc(r.code) + '">' +
          esc(r.code) + '</a></td><td>' + esc(r.name) + '</td>' +
          '<td class="num is-down">' + r.chg_pct + '%</td>' +
          '<td class="num">' + r.fund_yi + ' 亿</td>' +
          '<td class="muted">' + esc(r.industry || '—') + '</td></tr>').join('') +
        '</tbody></table></div>'
    }) : '';

    return '<div style="margin-bottom:14px">' + sentCard + '</div>' +
      '<div class="grid two" style="margin-bottom:14px">' + mainCard + ladderCard + '</div>' +
      '<div style="margin-bottom:14px">' + indCard + '</div>' +
      '<div style="margin-bottom:14px">' + conCard + '</div>' +
      '<div style="margin-bottom:14px">' + fundCard + '</div>' +
      (dtCard ? '<div style="margin-bottom:14px">' + dtCard + '</div>' : '');
  }

  /* ============================== 自选股 ============================= */

  async function pageWatch() {
    const res = await apiCached('/api/watch', 4000);
    const d = (res && res.ok && res.data) ? res.data : {};
    const items = d.items || [];
    const groups = d.groups || [];
    const sum = d.summary || {};
    setSub(items.length + ' 只 · ' + groups.length + ' 个分组');

    const newGroupInput =
      '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px">' +
        '<label class="field" style="flex:1;min-width:200px"><span>添加代码（逗号或空格分隔）</span>' +
          '<input id="w-codes" class="ck-input" placeholder="600519, 000001, 00700.HK" ' +
          'style="text-transform:uppercase"></label>' +
        '<label class="field" style="max-width:160px"><span>分组</span>' +
          '<input id="w-group" placeholder="核心 / 观察 / 短线" maxlength="20"></label>' +
        '<button type="button" class="btn" id="w-add">' + ICON('plus') + '加入自选</button>' +
      '</div>';

    if (!items.length) {
      return card({
        title: '自选股', sub: '加进来后会立刻打分并记录进场价',
        body: newGroupInput + emptyBox('还没有自选',
          '在「个股」页看到想跟踪的票，点「加自选」；也可以在上面直接输入代码。')
      });
    }

    // 按分组渲染
    const byGroup = {};
    items.forEach((it) => {
      const g = (it.group || '').trim() || '未分类';
      (byGroup[g] = byGroup[g] || []).push(it);
    });
    const order = Object.keys(byGroup).sort((a, b) =>
      (byGroup[b].length - byGroup[a].length) || a.localeCompare(b));

    const groupCards = order.map((g) => {
      const rows = byGroup[g];
      const tbody = rows.map((it) => {
        const ret = toNum(it.ret_since_add);
        const tone = ret == null ? '' : (ret > 0 ? 'is-up' : ret < 0 ? 'is-down' : '');
        const sc = toNum(it.score);
        return '<tr>' +
          '<td class="mono"><a class="link-btn" data-goto="' + esc(it.code) + '">' +
            esc(it.code) + '</a></td>' +
          '<td>' + esc(it.name || '—') + '</td>' +
          '<td class="num">' + (it.last_price != null ? Number(it.last_price).toFixed(2) : '—') + '</td>' +
          '<td class="num ' + tone + '">' + (ret == null ? '—' :
            (ret > 0 ? '+' : '') + (ret * 100).toFixed(2) + '%') + '</td>' +
          '<td class="num">' + (sc == null ? '—' : sc.toFixed(4)) + '</td>' +
          '<td class="muted">' + esc(String(it.added_at || '').slice(5, 16)) + '</td>' +
          '<td><button type="button" class="link-btn w-del" data-code="' + esc(it.code) +
            '">移除</button></td>' +
          '</tr>';
      }).join('');
      return card({
        title: g, sub: rows.length + ' 只',
        tools: '<button type="button" class="link-btn w-move" data-group="' + esc(g) +
          '">整组改分组</button>',
        body: '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
          '<th>代码</th><th>名称</th><th>现价</th><th>自加入</th><th>分数</th><th>加入时间</th><th></th>' +
          '</tr></thead><tbody>' + tbody + '</tbody></table></div>'
      });
    }).join('<div style="height:14px"></div>');

    return card({
      title: '自选股', sub: '加进来后会立刻打分并记录进场价',
      body: newGroupInput
    }) + '<div style="height:14px"></div>' + groupCards;
  }

  /* ============================== 个股行情 ============================= */

  function renderCandles(d) {
    const bars = d.bars || [];
    if (!bars.length) return emptyBox('无 K 线数据', '换一个代码或周期试试');
    const ma = d.ma || {};
    const W = 1120, H = 420, PAD_L = 8, PAD_R = 62, PAD_T = 14;
    const volH = 84, gap = 18;
    const priceH = H - PAD_T - volH - gap;
    const n = bars.length;
    const slot = (W - PAD_L - PAD_R) / Math.max(n, 1);
    const bw = Math.max(1.2, Math.min(slot * 0.68, 12));

    // 价格区间（含均线，避免均线跑出画面）
    let hi = -Infinity, lo = Infinity;
    bars.forEach((b) => {
      if (b.h != null) hi = Math.max(hi, b.h);
      if (b.l != null) lo = Math.min(lo, b.l);
    });
    Object.keys(ma).forEach((k) => (ma[k] || []).forEach((v) => {
      if (v != null) { hi = Math.max(hi, v); lo = Math.min(lo, v); }
    }));
    if (!isFinite(hi) || !isFinite(lo)) return emptyBox('价格数据异常', '');
    const pad = (hi - lo) * 0.04 || hi * 0.01 || 1;
    hi += pad; lo -= pad;

    const yOf = (p) => PAD_T + (hi - p) / (hi - lo) * priceH;
    const xOf = (i) => PAD_L + slot * (i + 0.5);

    // 网格 + 右侧价格刻度
    let grid = '';
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const p = lo + (hi - lo) * i / ticks;
      const y = yOf(p);
      grid += '<line x1="' + PAD_L + '" y1="' + y.toFixed(1) + '" x2="' + (W - PAD_R) +
        '" y2="' + y.toFixed(1) + '" stroke="rgba(120,140,180,.12)" stroke-width="1"/>';
      grid += '<text x="' + (W - PAD_R + 6) + '" y="' + (y + 3.5).toFixed(1) +
        '" fill="#7d8794" font-size="10.5">' + p.toFixed(2) + '</text>';
    }

    // 蜡烛
    let candles = '';
    bars.forEach((b, i) => {
      if (b.o == null || b.c == null) return;
      const up = b.c >= b.o;
      const col = up ? '#e8544a' : '#2bb673';
      const x = xOf(i);
      const yO = yOf(b.o), yC = yOf(b.c);
      const top = Math.min(yO, yC);
      const bh = Math.max(1, Math.abs(yC - yO));
      if (b.h != null && b.l != null) {
        candles += '<line x1="' + x.toFixed(1) + '" y1="' + yOf(b.h).toFixed(1) +
          '" x2="' + x.toFixed(1) + '" y2="' + yOf(b.l).toFixed(1) +
          '" stroke="' + col + '" stroke-width="1"/>';
      }
      candles += '<rect x="' + (x - bw / 2).toFixed(1) + '" y="' + top.toFixed(1) +
        '" width="' + bw.toFixed(1) + '" height="' + bh.toFixed(1) +
        '" fill="' + (up ? col : col) + '" opacity="' + (up ? 0.95 : 0.95) + '"/>';
    });

    // 均线
    const MA_COLORS = { ma5: '#f5c542', ma10: '#5b9dff', ma20: '#c77dff', ma60: '#38d39f', ma120: '#ff8f5b' };
    let maPath = '';
    Object.keys(ma).forEach((k) => {
      const arr = ma[k] || [];
      const col = MA_COLORS[k] || '#888';
      let seg = '', started = false;
      arr.forEach((v, i) => {
        if (v == null) { started = false; return; }
        seg += (started ? 'L' : 'M') + xOf(i).toFixed(1) + ' ' + yOf(v).toFixed(1) + ' ';
        started = true;
      });
      if (seg) maPath += '<path d="' + seg + '" fill="none" stroke="' + col +
        '" stroke-width="1.3" opacity="0.9"/>';
    });

    // 成交量
    const vols = bars.map((b) => b.v || 0);
    const vmax = Math.max.apply(null, vols.concat([1]));
    const volTop = PAD_T + priceH + gap;
    let volBars = '';
    bars.forEach((b, i) => {
      const v = b.v || 0;
      if (!v) return;
      const h = (v / vmax) * (volH - 8);
      const up = (b.c != null && b.o != null) ? b.c >= b.o : true;
      volBars += '<rect x="' + (xOf(i) - bw / 2).toFixed(1) + '" y="' + (volTop + volH - h).toFixed(1) +
        '" width="' + bw.toFixed(1) + '" height="' + h.toFixed(1) +
        '" fill="' + (up ? '#e8544a' : '#2bb673') + '" opacity="0.55"/>';
    });
    let volLabel = '<text x="' + PAD_L + '" y="' + (volTop + 11) +
      '" fill="#7d8794" font-size="10.5">成交量 ' + fmtVol(vmax) + '</text>';

    // X 轴日期（取 6 个）
    let xlab = '';
    const step = Math.max(1, Math.floor(n / 6));
    for (let i = 0; i < n; i += step) {
      xlab += '<text x="' + xOf(i).toFixed(1) + '" y="' + (H - 3) +
        '" fill="#7d8794" font-size="10" text-anchor="middle">' +
        esc(String(bars[i].d || '').slice(2)) + '</text>';
    }

    return '<div class="kline-wrap"><svg viewBox="0 0 ' + W + ' ' + H +
      '" preserveAspectRatio="none" class="kline-svg">' +
      grid + volBars + volLabel + candles + maPath + xlab + '</svg></div>';
  }

  function fmtVol(v) {
    const n = toNum(v);
    if (n == null) return '—';
    if (n >= 1e8) return (n / 1e8).toFixed(2) + '亿';
    if (n >= 1e4) return (n / 1e4).toFixed(2) + '万';
    return n.toFixed(0);
  }


  async function pageChart() {
    const code = (S.chartCode || '').trim();
    const period = S.chartPeriod || 'daily';
    const adjust = (S.chartAdjust == null ? 0 : S.chartAdjust);

    // 自选条必须在**所有** return 分支里都出现 ——
    // 之前只拼在最后一个 return 上，而 !code 的早退分支先返回，
    // 导致自选条永远不显示（实测就是「只有搜索框 + 还没选股票」）。
    // 自选条：用启动时缓存的 S.watchItems 同步渲染。
    // 之前走 await api() 拿数据，实测在 headless/某些环境下这条分支不生效
    // （页面能渲染但没有自选条）。同步数据最稳。
    const wl = (S.watchItems || []);
    let watchStrip = '';
    if (wl.length) {
      const byG = {};
      wl.forEach((it) => {
        const g = it.group || '未分组';
        (byG[g] = byG[g] || []).push(it);
      });
      watchStrip = '<div class="card" style="margin-bottom:14px"><div class="card-head" ' +
        'style="padding:14px 16px 0"><h2>自选股</h2>' +
        '<span class="sub">' + wl.length + ' 只 · 点一下看 K 线</span></div>' +
        '<div style="padding:10px 16px 14px">' +
        Object.keys(byG).map((g) => {
          return '<div style="margin-bottom:8px;display:flex;gap:8px;' +
            'flex-wrap:wrap;align-items:center">' +
            '<span class="muted" style="min-width:52px;font-size:12px">' + esc(g) + '</span>' +
            byG[g].map((it) => {
              const on = String(it.code) === String(code);
              return '<button type="button" class="btn ghost sm wl-chip' +
                (on ? ' is-active' : '') + '" data-code="' + esc(it.code) + '"' +
                ' title="' + esc((it.name || '') + ' ' + it.code) + '">' +
                esc(it.name || it.code) + '</button>';
            }).join('') + '</div>';
        }).join('') + '</div></div>';
    } else {
      watchStrip = '<div class="card" style="margin-bottom:14px"><div class="card-head" ' +
        'style="padding:14px 16px 0"><h2>自选股</h2>' +
        '<span class="sub">还没有自选</span></div>' +
        '<div style="padding:10px 16px 14px"><div class="muted" style="font-size:13px">' +
        '在上面的搜索框输入代码，点「加自选」就会出现在这里。</div></div></div>';
    }

    const wrap = (inner) => watchStrip + inner;


    const periods = [['daily', '日K'], ['weekly', '周K'], ['monthly', '月K'],
                     ['quarterly', '季K'], ['halfyear', '半年K'], ['yearly', '年K']];
    const adjusts = [[0, '不复权'], [1, '前复权'], [2, '后复权']];

    const searchBar = '<div class="card" style="margin-bottom:14px"><div class="card-head" style="padding:14px 16px 0">' +
      '<h2>个股行情</h2><span class="sub">输入代码：600519 / 00700.HK / AAPL</span></div>' +
      '<div style="padding:12px 16px 16px">' +
      '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">' +
        '<input id="ck-code" class="ck-input" placeholder="股票代码" value="' + esc(code) + '">' +
        '<button type="button" class="btn" id="ck-go">' + ICON('search') + '查询</button>' +
        '<button type="button" class="btn ghost" id="ck-watch">' + ICON('plus') + '加自选</button>' +
        '<div class="segmented" id="ck-period">' + periods.map(([v, l]) =>
          '<button type="button" class="seg' + (period === v ? ' is-active' : '') +
          '" data-p="' + v + '">' + l + '</button>').join('') + '</div>' +
        '<div class="segmented" id="ck-adjust">' + adjusts.map(([v, l]) =>
          '<button type="button" class="seg' + (adjust === v ? ' is-active' : '') +
          '" data-a="' + v + '">' + l + '</button>').join('') + '</div>' +
      '</div></div></div>';

    if (!code) {
      setSub('输入代码开始');
      return wrap(searchBar + emptyBox('还没选股票',
        '在上面的输入框填代码，比如 <code>600519</code>、<code>00700.HK</code>、<code>AAPL</code>'));
    }

    setSub(code + ' · 加载中');
    const q = '/api/kline?code=' + encodeURIComponent(code) +
      '&period=' + encodeURIComponent(period) + '&adjust=' + adjust + '&count=240';
    const res = await apiCached(q, 15000);
    if (!res.ok || !res.data) {
      setSub(code + ' · 取不到');
      return wrap(searchBar + notReadyBox(res));
    }
    const d = res.data;
    const qt = d.quote || {};
    const st = d.stats || {};
    const chg = toNum(qt.chg_pct);
    const tone = chg == null ? '' : (chg > 0 ? 'is-up' : chg < 0 ? 'is-down' : '');
    setSub((d.name || code) + ' · ' + d.period_label + ' · ' + d.count + ' 根');

    const quoteCard = card({
      title: (d.name || code) + '  ' + code, sub: d.period_label + ' · ' + (d.adjust_note || ''),
      body:
        '<div class="quote-head">' +
          '<div class="quote-price ' + tone + '">' + (qt.price != null ? qt.price.toFixed(2) : '—') + '</div>' +
          '<div class="quote-chg ' + tone + '">' +
            (chg == null ? '—' : (chg > 0 ? '+' : '') + chg.toFixed(2) + '%') + '</div>' +
          (qt.stale ? '<span class="tag pending">非实时</span>' : '<span class="tag hit">实时</span>') +
        '</div>' +
        '<div class="quote-grid">' +
          [['今开', qt.open], ['最高', qt.high], ['最低', qt.low], ['昨收', qt.prev_close],
           ['成交量', qt.volume != null ? fmtVol(qt.volume) : null],
           ['5日', st.chg_5 != null ? st.chg_5 + '%' : null],
           ['20日', st.chg_20 != null ? st.chg_20 + '%' : null],
           ['60日', st.chg_60 != null ? st.chg_60 + '%' : null]]
            .map(([k, v]) => '<div class="qg-item"><span class="qg-k">' + k + '</span>' +
              '<span class="qg-v">' + (v == null ? '—' : (typeof v === 'number' ? v.toFixed(2) : v)) + '</span></div>')
            .join('') +
        '</div>'
    });

    const maLast = (d.ma_windows || []).map((w) => {
      const arr = (d.ma['ma' + w] || []);
      const v = arr.length ? arr[arr.length - 1] : null;
      const col = { 5: '#f5c542', 10: '#5b9dff', 20: '#c77dff', 60: '#38d39f', 120: '#ff8f5b' }[w] || '#888';
      return '<span class="ma-chip"><i style="background:' + col + '"></i>MA' + w + ' ' +
        (v == null ? '—' : v.toFixed(2)) + '</span>';
    }).join('');

    const chartCard = card({
      title: 'K 线', sub: d.count + ' 根 · ' + (d.adjust_note || ''),
      tools: '<div class="ma-chips">' + maLast + '</div>',
      body: renderCandles(d)
    });

    return wrap(searchBar + '<div style="margin-bottom:14px">' + quoteCard + '</div>' +
      '<div style="margin-bottom:14px">' + chartCard + '</div>');
  }

  /* ============================== 预测台账 ============================= */

  const isHit = (r) => r.hit === true || r.hit === 1 || r.hit === '1' || r.hit === 'true' || r.hit === 'True';

  function verdictTag(r) {
    const st = String(r.status || '').toLowerCase();
    if (st === 'pending' || isNil(r.hit)) return '<span class="tag pending">pending</span>';
    return isHit(r) ? '<span class="tag hit">命中</span>' : '<span class="tag miss">未中</span>';
  }

  async function pageLedger() {
    // 带 horizon 读**实时**台账（pred_ledger）；predict 只写那里，
    // 不带 horizon 读到的是 walk-forward 回放，点「生成预测」不会有变化。
    const _hz = S.modelHorizon || 1;
    const res = await apiCached('/api/ledger?days=' + S.filter.days + '&horizon=' + _hz);
    // 主台账（data/pred_ledger/*.jsonl）长期只写不读：14MB / 4.7 万行没有任何
    // 读取方，于是 5.85× 重复写入无人发现。这里把它的真实总量与重复率摊开。
    const _fRes = await apiCached('/api/ledger?source=file&limit=1', 60000);
    const _fd = (_fRes && _fRes.ok && _fRes.data) ? _fRes.data : null;
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

    const _dates = Array.from(new Set(all.map((r) => String(r.as_of || '').slice(0, 10))))
      .filter(Boolean).sort();
    const _span = _dates.length
      ? _dates[0] + ' ~ ' + _dates[_dates.length - 1] + '（' + _dates.length + ' 个交易日）'
      : '—';
    // ---- 按「天」分页 ----
    // 之前按行分页：默认按日期降序，最新一天正好 100 行占满第一页，
    // 用户翻第一页永远只看到一个交易日 —— 加了分页和柱状图也没解决。
    // 改成：每页固定 N 个交易日，每天最多 M 条。
    const DAYS_PER_PAGE = 5;          // 每页几个交易日
    const PER_DAY = Number(S.filter.perDay || 20);   // 每天最多几条

    // 先按日期分组（rows 已按日期降序）
    const byDay = [];
    let curDay = null, bucket = null;
    rows.forEach((r) => {
      const d = String(r.as_of || '').slice(0, 10);
      if (d !== curDay) { curDay = d; bucket = []; byDay.push([d, bucket]); }
      if (bucket.length < PER_DAY) bucket.push(r);
    });

    const total = rows.length;
    const dayCount = byDay.length;
    const pages = Math.max(1, Math.ceil(dayCount / DAYS_PER_PAGE));
    const page = Math.min(Math.max(1, Number(S.filter.page || 1)), pages);
    const slice = byDay.slice((page - 1) * DAYS_PER_PAGE, page * DAYS_PER_PAGE);
    const pageRows = [].concat(...slice.map((x) => x[1]));
    const shownDays = slice.map((x) => x[0]);
    setSub('共 ' + all.length + ' 条 · ' + _span + ' · 筛选后 ' + total + ' 条');

    // 日期分布条：每个交易日的条数，点一下筛选那天。
    // 之前默认按日期降序 + 每页 100 行，最新一天正好占满第一页，
    // 看起来像「只有 10-29 这一天」—— 加了这条用户能一眼看到 30 个交易日。
    const _perDay = {};
    all.forEach((r) => {
      const d = String(r.as_of || '').slice(0, 10);
      if (d) _perDay[d] = (_perDay[d] || 0) + 1;
    });
    const _dayKeys = Object.keys(_perDay).sort().reverse();
    const _maxN = Math.max.apply(null, _dayKeys.map((k) => _perDay[k]).concat([1]));
    const dayBar = _dayKeys.length > 1 ? (
      '<div style="padding:0 16px 10px;display:flex;gap:3px;flex-wrap:wrap;align-items:flex-end">' +
      _dayKeys.map((k) => {
        const n = _perDay[k];
        const h = Math.max(10, Math.round((n / _maxN) * 30));
        const on = S.filter.from === k && S.filter.to === k;
        return '<div class="day-bar' + (on ? ' is-on' : '') + '" data-day="' + k +
          '" title="' + k + ' · ' + n + ' 条（点击只看这天）" ' +
          'style="width:13px;height:' + h + 'px"></div>';
      }).join('') +
      '</div>' +
      '<div style="padding:0 16px 10px;display:flex;gap:10px;align-items:center">' +
        '<span class="muted">' + _dayKeys.length + ' 个交易日 · 柱子=当日条数</span>' +
        (S.filter.from && S.filter.from === S.filter.to
          ? '<button type="button" class="link-btn" id="f-clear-day">清除日期筛选</button>' : '') +
      '</div>'
    ) : '';


    return card({
      title: '预测台账', sub: '台账只追加、不回改；pending = h 日未到，尚未对账', cls: 'flush',
      tools: '<label class="field inline"><span>最近</span><select id="f-days">' +
        [10, 30, 60, 120].map((d) => '<option value="' + d + '"' + (S.filter.days === d ? ' selected' : '') + '>' +
          d + ' 天</option>').join('') + '</select></label>' +
        '<div class="segmented" id="f-result">' +
        [['all', '全部'], ['hit', '命中'], ['miss', '未中'], ['pending', '待定']].map(([v, l]) =>
          '<button type="button" class="seg' + (rf === v ? ' is-active' : '') + '" data-f="' + v + '">' + l + '</button>').join('') +
        '</div>',
      body: dayBar +
        '<div style="padding:0 16px 4px"><div class="form" style="gap:10px;margin-bottom:12px">' +
        '<label class="field"><span>代码搜索</span><input id="f-code" type="search" placeholder="600519 / 688981" value="' + esc(S.filter.q) + '"></label>' +
        '<label class="field"><span>起（as_of ≥）</span><input id="f-from" type="date" value="' + esc(S.filter.from) + '"></label>' +
        '<label class="field"><span>止（as_of ≤）</span><input id="f-to" type="date" value="' + esc(S.filter.to) + '"></label>' +
        '</div></div>' +
        '<div style="padding:0 16px 8px">' + table({
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
          rows: pageRows,
          rowClass: (r) => '',
          empty: emptyBox('没有匹配的记录', '试试放宽日期范围或清空代码搜索。'),
          // 在每天的第一行前面插入一条「日期小标题」
          beforeRow: (r, idx, arr) => {
            const d = String(r.as_of || '').slice(0, 10);
            const prev = idx > 0 ? String(arr[idx - 1].as_of || '').slice(0, 10) : null;
            if (d === prev) return '';
            const n = byDay.find((x) => x[0] === d);
            const cnt = n ? n[1].length : 0;
            return '<tr class="day-sep"><td colspan="8">' + esc(d) +
              '<span class="muted" style="margin-left:8px">' + cnt + ' 条（最多显示 ' +
              PER_DAY + ' 条）</span></td></tr>';
          }
        }) + '</div>' +
        '<div style="padding:0 16px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">' +
          '<span class="muted">第 ' + page + ' / ' + pages + ' 页（每页 ' +
            DAYS_PER_PAGE + ' 个交易日）· 共 ' + dayCount + ' 个交易日 / ' +
            total + ' 条 · 每天最多 ' +
            '<select id="f-psize" style="margin:0 4px">' +
            [10, 20, 50, 100].map(function (n) {
              return '<option value="' + n + '"' + (PER_DAY === n ? ' selected' : '') + '>' + n + '</option>';
            }).join('') +
            '</select> 条</span>' +
          '<div class="segmented" id="f-page">' +
            '<button type="button" class="seg" data-pg="1">« 首页</button>' +
            '<button type="button" class="seg" data-pg="' + Math.max(1, page - 1) + '">‹ 上页</button>' +
            '<button type="button" class="seg" data-pg="' + Math.min(pages, page + 1) + '">下页 ›</button>' +
            '<button type="button" class="seg" data-pg="' + pages + '">末页 »</button>' +
          '</div>' +
          '<span class="muted">命中 = 预测方向与实际一致（预测跌而实际跌也算命中），不是"涨了才算命中"</span>' +
        '</div>' +'</div>'
    }) + (_fd ? card({
      title: '主台账（文件）',
      sub: 'data/pred_ledger/' + (_fd.file || '—') +
        (_fd.dedup ? ' · 去重视图（每 (决策日,代码,周期) 保留最后一条）'
                   : ' · 原始台账（含历史重复，只追加不可回改）'),
      body: '<div style="padding:0 16px 16px">' +
        kv([
          kvItem('文件总行数', num(_fd.total_records, 0), '', '物理行数'),
          kvItem('唯一预测数', num(_fd.total_deduped, 0), '', '去重后的 (日,代码,周期) 键数'),
          kvItem('历史重复行',
            _fd.total_records !== null && _fd.total_deduped !== null
              ? num(_fd.total_records - _fd.total_deduped, 0) : '—',
            (_fd.total_records && _fd.total_deduped &&
             _fd.total_records > _fd.total_deduped) ? 'warn-text' : '',
            _fd.total_records && _fd.total_deduped
              ? '重复率 ' + num((1 - _fd.total_deduped / _fd.total_records) * 100, 1) + '%'
              : '')
        ]) +
        '<p class="muted" style="margin:12px 0 0">' +
        '幂等写入已生效（append_ledger 按 (as_of, code, horizon) 去重）；' +
        '历史重复受 INV-7 约束不可回改，读数请用去重视图。' +
        '</p></div>'
    }) : '');
  }

  /* ============================== 模拟实盘 ============================= */

  async function pagePaper() {
    const mkt = MKTS.indexOf(S.paperMarket) >= 0 ? S.paperMarket : 'CN';
    const label = mktLabel(mkt);

    const CUR_H = Number(S.paHorizon || 1);
    const [sumRes, cfgRes, paperRes, tradesRes, autoRes, periodRes] = await Promise.all([
      apiCached('/api/account/config', 0),
      apiCached('/api/account/config?market=' + mkt, 0),
      apiCached('/api/paper?market=' + mkt, 0),
      mkt === 'CN'
        ? apiCached('/api/paper/trades?limit=20', 0)
        : Promise.resolve({ ok: true, data: { rows: [], n: 0 } }),
      apiCached('/api/paper/auto?market=' + mkt + '&by_horizon=1', 3000),
      apiCached('/api/paper/period?market=' + mkt + '&bucket=' +
                (S.paBucket || 'day') + '&horizon=' + CUR_H, 3000)
    ]);

    const summary = (sumRes.ok && sumRes.data) ? sumRes.data : null;
    const cfg = (cfgRes.ok && cfgRes.data) ? cfgRes.data : null;
    const trades = (tradesRes.ok && tradesRes.data) ? tradesRes.data : { rows: [], n: 0 };
    const costMap = await loadCostMap();
    const currency = (cfg && cfg.currency) || '';
    const paperData = paperForMarket(paperRes, mkt);
    const tag = label + (currency ? '（' + currency + '）' : '');

    let body = '';
    if (!paperData) {
      // 未就绪市场：明确说明，绝不显示别的市场的数据
      setSub(tag + ' · 尚无模拟盘数据');
      body = '<div style="margin-bottom:14px">' + marketPlaceholder(mkt, paperRes) + '</div>';
    } else {
      const p = paperData;
      const cmp = (p.compare && typeof p.compare === 'object') ? p.compare
        : (p.diff && typeof p.diff === 'object') ? p.diff : {};
      const simNav = seriesValues(p.sim_nav || []);
      const liveNav = seriesValues(p.live_nav || []);
      const dates = Array.isArray(p.dates) ? p.dates : [];
      const gap = toNum(cmp.pnl_gap);
      setSub(tag + ' · 对比 ' + num(cmp.n_days_compared, 0) + ' 天 · 模拟期末 ' +
        num(cmp.sim_final, 0) + ' · 实盘期末 ' + num(cmp.live_final, 0));

      const kpis = [
        kpi({
          k: '模拟盘期末 · ' + label, v: num(cmp.sim_final, 0), tone: 'brand',
          s: esc(currency) + ' · 年化 ' + pct(cmp.sim_annual) + ' · 命中率 ' + pct(cmp.sim_hit_rate)
        }),
        kpi({
          k: '实盘期末 · ' + label, v: num(cmp.live_final, 0), tone: 'brand',
          s: esc(currency) + ' · 成交 ' + num(cmp.live_trades, 0) + ' 笔'
        }),
        kpi({
          k: '净值差 · ' + label, v: signedPct(cmp.pnl_gap), cls: signClass(cmp.pnl_gap),
          tone: gap === null ? 'neutral' : gap < 0 ? 'bad' : 'good', s: '模拟 − 实盘'
        }),
        kpi({
          k: '跟踪误差 · ' + label, v: pct(cmp.tracking_error_ann), tone: 'warn',
          s: '滑点 ' + num(cmp.exec_slippage_bps, 1) + ' bps'
        })
      ].join('');

      const chartCard = card({
        title: '净值双线 · ' + tag,
        sub: dates.length ? dates[0] + ' → ' + dates[dates.length - 1] : 'paper.json',
        body: (simNav.some((v) => v !== null) || liveNav.some((v) => v !== null))
          ? '<div class="chart">' + chartSVG([
            { values: simNav, color: 'var(--brand)', label: '模拟盘' },
            { values: liveNav, color: 'var(--warn)', label: '实盘' }
          ], {
            height: 280, tipLabels: dates,
            labels: { first: dates[0] || '', last: dates[dates.length - 1] || '' },
            aria: label + ' 模拟盘与实盘净值对比'
          }) + '</div>'
          : emptyBox('暂无净值曲线', 'paper.json 未包含 sim_nav / live_nav 序列。')
      });

      const diffCard = card({
        title: '差异指标 · ' + label, sub: 'compare',
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

    // 成交录入只对 A 股开放：live_trades.jsonl 没有市场字段，混记会污染 A 股对账
    let forms = '';
    if (mkt === 'CN') {
      const formCard = card({
        title: '录入实盘成交 · A股',
        sub: 'POST /api/paper/trade → data/state/live_trades.jsonl',
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

      forms = '<div class="grid two">' + formCard + tradesCard + '</div>';
    } else {
      forms = card({
        title: '实盘成交录入 · ' + label,
        body: emptyBox('该市场暂不支持成交录入',
          '实盘对账文件 <code>live_trades.jsonl</code> 没有市场字段，目前只用于 A 股；' +
          '为避免污染 A 股对账，' + esc(label) + ' 暂不提供录入入口。', 'alert')
      });
    }

    // 配置接口失败时给占位，不渲染一个空表单假装能改
    const acctCard = cfg
      ? accountCardHtml(cfg, mkt, costMap, currency)
      : card({
        title: '账户设置 · ' + label,
        body: emptyBox('账户配置不可用',
          '接口返回 <code>' + esc(cfgRes.error || 'unknown') + '</code>' +
          (cfgRes.detail ? '：' + esc(cfgRes.detail) : '') +
          '。<br>配置来自 <code>data/state/account.json</code>（由 qcore 读取），请检查 qcore 是否可用。',
          'alert')
      });

    // ================= 按周期分账户（每个周期独立本金/成本/成交）=================
    const HB = (autoRes && autoRes.ok && autoRes.data) ? autoRes.data : {};
    const hbRows = HB.rows || [];
    const curLabel = (hbRows.find((x) => Number(x.horizon) === CUR_H) || {}).label || ('h' + CUR_H);

    const hzAccountCard = card({
      title: '按周期分账户', sub: 'T+1 / T+3 / 1周 / 2周 / 15天 各自独立，不走同一个池子',
      tools: '<button type="button" class="btn ghost sm" id="pa-all">' +
        ICON('play') + '全周期调仓</button>',
      body: hbRows.length
        ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
          '<th>周期</th><th>状态</th><th>净值</th><th>累计收益</th>' +
          '<th>持仓</th><th>成交</th><th>调仓</th><th>费用</th>' +
          '</tr></thead><tbody>' + hbRows.map((r) => {
            const on = Number(r.horizon) === CUR_H;
            const tr = toNum(r.total_return);
            const tone = tr == null ? '' : (tr > 0 ? 'is-up' : tr < 0 ? 'is-down' : '');
            const noData = !r.ok || r.equity == null;
            return '<tr' + (on ? ' class="row-on"' : '') + '>' +
              '<td><a class="link-btn" data-pahz="' + r.horizon + '">' +
                esc(r.label || ('h' + r.horizon)) + '</a></td>' +
              '<td>' + (noData ? '<span class="tag pending">未建仓</span>'
                               : '<span class="tag hit">在跑</span>') + '</td>' +
              '<td class="num">' + (r.equity == null ? '—' : fmtMoney(r.equity)) +
                (r.capital_mismatch
                  ? '<span class="tag warn" style="margin-left:6px" title="账户本金 ' +
                    esc(fmtMoney(r.account_capital)) + '，而「各周期本金」配置是 ' +
                    esc(fmtMoney(r.capital_configured)) +
                    '。点「清空重设」可按配置本金重开">本金不符</span>'
                  : '') + '</td>' +
              '<td class="num ' + tone + '">' + (tr == null ? '—' :
                (tr > 0 ? '+' : '') + (tr * 100).toFixed(2) + '%') + '</td>' +
              '<td class="num">' + (r.n_positions == null ? '—' : r.n_positions) + '</td>' +
              '<td class="num">' + (r.n_trades == null ? '—' : r.n_trades) + '</td>' +
              '<td class="num">' + (r.trade_periods == null ? '—' : r.trade_periods) + '</td>' +
              '<td class="num">' + (r.fee_total == null ? '—' : r.fee_total.toFixed(2)) + '</td>' +
              '</tr>';
          }).join('') + '</tbody></table></div>' +
          (HB.verdict ? '<p class="muted" style="margin-top:10px">' + esc(HB.verdict) + '</p>' : '')
        : emptyBox('还没有周期账户', '点右上角「全周期调仓」为每个已训练周期建仓')
    });

    // ================= 当前周期的自动交易账户 =================
    const AU = (periodRes && periodRes.ok && periodRes.data) ? periodRes.data : {};
    const A = (AU.account && AU.account.account) ? AU.account.account : {};
    S.paperAcct = A;          // 事件处理器要用（那里拿不到这里的局部变量）
    const auTrades = (AU.account && AU.account.recent_trades) || [];
    const PG = AU.period || {};
    const FS = AU.fees || {};

    const autoCard = card({
      title: '自动交易账户 · ' + curLabel,
      sub: '模拟盘自动记账（不下真实委托）· 按该周期推荐',
      tools: '<div class="segmented" id="pa-bucket">' +
        [['day', '按日'], ['week', '按周'], ['month', '按月']].map(([v, l]) =>
          '<button type="button" class="seg' + ((S.paBucket || 'day') === v ? ' is-active' : '') +
          '" data-b="' + v + '">' + l + '</button>').join('') +
        '</div><button type="button" class="btn ghost sm" id="pa-rebal" style="margin-left:8px">' +
        ICON('play') + '本周期调仓</button>' +
        '<button type="button" class="btn ghost sm" id="pa-reset" style="margin-left:6px">' +
        ICON('trash') + '清空重设</button>',
      body: (A.equity != null
        ? '<div class="quote-grid" style="grid-template-columns:repeat(auto-fit,minmax(104px,1fr));margin-bottom:14px">' +
            [['净值', fmtMoney(A.equity)], ['现金', fmtMoney(A.cash)],
             ['持仓', (A.n_positions || 0) + ' 只'],
             ['累计收益', (A.total_return == null ? '—'
               : (A.total_return > 0 ? '+' : '') + (A.total_return * 100).toFixed(2) + '%')],
             ['成交笔数', (AU.account && AU.account.n_trades) || 0],
             ['数据截至', dataAsOf(A.as_of, AU.as_of_lag_trading_days)]]
              .map(([k, v]) => '<div class="qg-item"><span class="qg-k">' + k +
                '</span><span class="qg-v">' + esc(String(v)) + '</span></div>').join('') +
          '</div>'
        : '') +
        (auTrades.length
          ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
            '<th>成交日</th><th>成交时刻</th><th>方向</th><th>名称（代码）</th>' +
            '<th>数量</th><th>价格</th><th>金额</th><th>费用</th><th>原因</th>' +
            '</tr></thead><tbody>' + auTrades.slice().reverse().slice(0, 20).map((t) =>
              '<tr><td class="mono">' + esc(String(t.date || '').slice(5)) + '</td>' +
              // `at` 是真实执行时刻（写入时一直有），旧表只渲染 date → 用户
              // 看不出这笔是盘中还是盘后成交，才会问「不在交易时间怎么也能交易」。
              '<td class="mono">' + (t.at
                ? esc(String(t.at).replace('T', ' ').slice(0, 19)) +
                  '<span class="muted" style="margin-left:4px">' + esc(_sessionTag(t.at)) + '</span>'
                : '<span class="muted" title="该笔为历史记录，写入时未记录执行时刻">无时刻</span>') + '</td>' +
              '<td class="' + (t.side === 'buy' ? 'is-up' : 'is-down') + '">' +
                (t.side === 'buy' ? '买入' : '卖出') + '</td>' +
              '<td><a class="link-btn" data-goto="' + esc(t.code) + '">' +
                codeCell(t.code) + '</a></td>' +
              '<td class="num">' + esc(t.qty) + '</td>' +
              '<td class="num">' + (Number(t.price) || 0).toFixed(2) + '</td>' +
              '<td class="num">' + fmtMoney(t.amount) + '</td>' +
              '<td class="num">' + (t.cost == null ? '—' : Number(t.cost).toFixed(2)) + '</td>' +
              '<td class="muted">' + esc(reasonText(t.reason)) + '</td></tr>').join('') +
            '</tbody></table></div>'
          : emptyBox('该周期还没有成交', '点右上角「本周期调仓」建仓；该周期没模型时需先训练+预测')) +
        (PG.buckets && PG.buckets.length
          ? '<div style="margin-top:16px"><div class="muted" style="margin-bottom:8px">分时段（' +
            ({ day: '日', week: '周', month: '月' }[PG.bucket] || '日') + '）</div>' +
            '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
            '<th>时段</th><th>笔数</th><th>买</th><th>卖</th><th>买入额</th><th>卖出额</th><th>费用</th>' +
            '</tr></thead><tbody>' + PG.buckets.slice(0, 10).map((b) =>
              '<tr><td class="mono">' + esc(b.period) + '</td>' +
              '<td class="num">' + b.n + '</td>' +
              '<td class="num is-up">' + b.buys + '</td>' +
              '<td class="num is-down">' + b.sells + '</td>' +
              '<td class="num">' + fmtMoney(b.buy_amount) + '</td>' +
              '<td class="num">' + fmtMoney(b.sell_amount) + '</td>' +
              '<td class="num">' + b.fee.toFixed(2) + '</td></tr>').join('') +
            '</tbody></table></div></div>'
          : '') +
        (FS.items && FS.items.length
          ? '<div style="margin-top:16px"><div class="muted" style="margin-bottom:8px">费用构成（合计 ' +
            (FS.total || 0).toFixed(2) + '，' + (FS.avg_bps == null ? '—' : FS.avg_bps + ' bps') + '）</div>' +
            '<div class="bars">' + FS.items.map((it) =>
              '<div class="bar-row"><span class="bar-label">' + esc(it.item) +
              '</span><span class="bar-track"><span class="bar-fill" style="width:' +
              Math.max(2, (it.share || 0) * 100).toFixed(1) + '%"></span></span>' +
              '<span class="bar-val">' + it.amount.toFixed(2) + '</span></div>').join('') +
            '</div></div>'
          : '')
    });

    return '<div style="margin-bottom:12px">' + marketTabsHtml(summary, mkt) + '</div>' +
      '<div style="margin-bottom:14px">' + hzAccountCard + '</div>' +
      '<div style="margin-bottom:14px">' + autoCard + '</div>' +
      '<div style="margin-bottom:14px">' + acctCard + '</div>' +
      body + forms;
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
    const live = (res.data.live || []).filter((r) => r && typeof r === 'object');
    const liveMeta = res.data.live_meta || {};
    const types = Array.from(new Set(rows.map((r) => r.type).filter(Boolean)));
    const years = Array.from(new Set(rows.map((r) => String(r.date || '').slice(0, 4)).filter(Boolean))).sort();
    // 人工整理的事件止于 curatedUntil（实测 2025-06），之后只能靠实时抓取。
    // 必须在 setSub 之前定义 —— const 有暂时性死区，先用后定义会抛
    // 「Cannot access 'curatedUntil' before initialization」
    const curatedUntil = years.length ? years[years.length - 1] : '';
    setSub('人工 ' + rows.length + ' 条（截至 ' + (curatedUntil || '—') + '）'
      + (live.length ? ' + 实时 ' + live.length + ' 条' : '')
      + ' · ' + types.length + ' 类 · '
      + (years.length ? years[0] + '–' + years[years.length - 1] : '—'));

    const tf = S.filter.evType;
    const df = S.filter.evDir;

    // 把 live 也并进时间线，否则 2026 年那一段是空的 —— 看起来像"系统坏了"。
    const liveAsEvents = live.map((r) => ({
      date: String(r.date || r.time || '').slice(0, 10),
      type: r.type, direction: r.direction, sectors: r.sectors || [],
      level: r.level, title: r.title, source: r.source,
      note: r.note, url: r.url,
      _live: true,
      evidence_level: r.level === 'strong' ? 'strong' : 'weak',
      subtype: null, eid: r.eid,
    }));

    const merged = rows.concat(liveAsEvents);
    const view = merged.filter((r) => {
      if (tf !== 'all' && r.type !== tf) return false;
      const d = toNum(r.direction);
      if (df === 'pos' && !(d > 0)) return false;
      if (df === 'neg' && !(d < 0)) return false;
      return true;
    }).slice().sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));

    const tl = card({
      title: '政策日历时间线',
      sub: '最新在前 · 共 ' + view.length + ' 条（人工 + 实时合并）', cls: 'flush',
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

    // 实时快讯（规则分类，reliability 0.5）—— 与人工整理的历史日历分开呈现
    const liveCard = live.length ? card({
      title: '实时快讯', cls: 'flush',
      sub: (liveMeta.generated_at ? '更新于 ' + String(liveMeta.generated_at).slice(11, 19) + ' · ' : '') +
        '共 ' + live.length + ' 条 · 规则分类，未经人工核实',
      body: '<div style="padding:0 16px 16px;max-height:420px;overflow:auto">' +
        live.map((r) => {
          const d = toNum(r.direction);
          const tag = d > 0 ? '利好' : d < 0 ? '利空' : '中性';
          const cls = d > 0 ? 'is-up' : d < 0 ? 'is-down' : '';
          return '<div class="tl-item">' +
            '<div class="tl-head"><span class="muted">' + esc(String(r.time || '').slice(5, 16)) + '</span>' +
            ' <span class="tag">' + esc(EVENT_TYPES[r.type] || r.type || '其他') + '</span>' +
            ' <span class="tag ' + cls + '">' + tag + '</span>' +
            (r.level === 'strong' ? ' <span class="tag">强</span>' : '') + '</div>' +
            '<div class="tl-title">' + esc(r.title || '') + '</div>' +
            ((r.sectors || []).length ? '<div class="tl-sectors">' +
              r.sectors.map((s) => '<span class="chip">' + esc(s) + '</span>').join('') + '</div>' : '') +
            '</div>';
        }).join('') + '</div>'
    }) : '';

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

    return (liveCard ? '<div style="margin-bottom:14px">' + liveCard + '</div>' : '') +
      '<div style="margin-bottom:14px">' + tl + '</div>' + impactCard;
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

  // 数据源健康度：扁平（现行）与 {sources:{...}}（旧假设）两种结构都吃。
  const _SRC_META = ['as_of', 'generated_at', 'sources', 'summary', 'updated_at', 'note'];
  function _srcMap(sh) {
    if (!sh || typeof sh !== 'object') return {};
    if (sh.sources && typeof sh.sources === 'object') return sh.sources;
    const out = {};
    Object.keys(sh).forEach((k) => {
      if (_SRC_META.indexOf(k) >= 0) return;
      if (sh[k] && typeof sh[k] === 'object') out[k] = sh[k];
    });
    return out;
  }

  async function pageRisk() {
    const [perfRes, healthRes, onlineRes] = await Promise.all([
      apiCached('/api/performance'), apiCached('/api/health'),
      apiCached('/api/online/learn', 5000)]);
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
          { label: '用途', key: 'desc', render: (r) => '<span class="muted">' + esc(r.desc || '—') + '</span>' },
          { label: '状态', key: 'tag', render: (r) => r.tag },
          { label: '成功', key: 'ok', num: true, render: (r) => esc(num(r.ok, 0)) },
          {
            label: '失败', key: 'fail', num: true,
            render: (r) => '<span class="' + (toNum(r.fail) > 0 ? 'up-text' : '') + '">' + esc(num(r.fail, 0)) + '</span>'
          },
          { label: '延迟 ms', key: 'latency', num: true, render: (r) => esc(num(r.latency, 1)) },
          { label: '最后错误', key: 'err', render: (r) => '<span class="muted">' + esc(r.err || '—') + '</span>' }
        ],
        // source_health.json 的结构是 {as_of, sources:{...}, summary}。
        // 之前直接遍历顶层，把 as_of / sources / summary 当成"源"显示出来了。
        // qcore/sources.py 写的是**扁平**结构 {eastmoney:{...}, sina:{...}}，
        // 而这里过去读 sh.sources —— 一个从来没有写入方的嵌套键，
        // 于是「数据源健康度」表恒为空（陷阱 18：只改读取方）。
        // 兼容两版：有 sources 就用，没有就把顶层对象当源表，跳过元信息键。
        rows: Object.keys(_srcMap(sh)).map((k) => {
          const v = _srcMap(sh)[k] || {};
          const failN = toNum(v.fail);
          const okN = toNum(v.ok);
          const bad = (failN !== null && failN > 0 && (okN === null || okN === 0));
          return {
            name: k,
            desc: v.desc || '',
            ok: okN, fail: failN, latency: toNum(v.latency_ms),
            err: v.last_error || '',
            tag: bad ? '<span class="tag bad">失败</span>'
                     : (failN !== null && failN > 0)
                       ? '<span class="tag warn">有失败</span>'
                       : '<span class="tag ok">正常</span>'
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
    try {
      const sy = await apiCached('/api/sync', 3000);
      S.sync = (sy && sy.ok && sy.data) ? sy.data : {};
    } catch (e) { S.sync = {}; }
    try {
      const au = await apiCached('/api/auth', 3000);
      S.auth = (au && au.ok && au.data) ? au.data : {};
    } catch (e) { S.auth = {}; }
    try {
      const fe = await apiCached('/api/fees', 30000);
      S.fees = (fe && fe.ok && fe.data) ? fe.data : {};
    } catch (e) { S.fees = {}; }
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

    // ---- 训练任务：NAS 下发，训练机领取 ----
    const SY = S.sync || {};
    const tasks = SY.recent || SY.tasks || [];
    const counts = SY.counts || {};
    const stTone = { pending: 'pending', claimed: 'is-active', done: 'hit', failed: 'miss',
                     expired: 'miss', cancelled: 'pending' };
    const stLabel = { pending: '待领取', claimed: '训练中', done: '已完成',
                      failed: '失败', expired: '超时', cancelled: '已取消' };

    const taskRows = tasks.slice(0, 12).map((t) => {
      const st = String(t.status || 'pending');
      return '<tr>' +
        '<td class="mono">' + esc(String(t.id || '').slice(0, 8)) + '</td>' +
        '<td>' + (t.horizons || []).map((h) => '<span class="chip">h' + esc(h) + '</span>').join('') + '</td>' +
        '<td class="num">' + esc(t.trees || '—') + '</td>' +
        '<td><span class="tag ' + (stTone[st] || '') + '">' + (stLabel[st] || st) + '</span></td>' +
        '<td class="muted">' + esc(t.claimed_by || '—') + '</td>' +
        '<td class="muted">' + esc(String(t.created_at || '').slice(5, 16)) + '</td>' +
        '</tr>';
    }).join('');

    const horizonChips = (S.horizonsList || [1, 3, 5, 10, 15]).map((h) =>
      '<label class="chip-check"><input type="checkbox" class="tk-h" value="' + h + '"' +
      (h === 1 || h === 5 ? ' checked' : '') + '> h' + h + '</label>').join('');

    // ---- 交易费用（分市场，可编辑）----
    const FE = S.fees || {};
    // 每个数值都是可直接编辑的输入框 —— 之前是静态文本，
    // 界面上写着「佣金可改成你自己的费率」却根本点不了。
    const feeInp = (mkt, field, v, step) =>
      '<input class="fee-inp" type="number" step="' + (step || '0.001') + '"' +
      ' data-mkt="' + esc(mkt) + '" data-field="' + field + '"' +
      ' value="' + (v == null ? '' : v) + '" placeholder="—">';

    const feeRows = (FE.compare || []).map((r) => {
      const f = (FE.fees || {})[r.market] || {};
      const mkt = r.market;
      return '<tr><td>' + esc(r.name || mkt) + '</td>' +
        '<td class="muted">' + esc(r.currency || '') + '</td>' +
        '<td class="num">' + feeInp(mkt, 'commission_bps', f.commission_bps, '0.1') + '</td>' +
        '<td class="num">' + feeInp(mkt, 'stamp_duty_bps', f.stamp_duty_bps, '0.1') +
          (f.stamp_side === 'sell' ? '<span class="muted" style="font-size:10px"> 卖</span>' :
           f.stamp_side === 'both' ? '<span class="muted" style="font-size:10px"> 双</span>' : '') + '</td>' +
        '<td class="num">' + feeInp(mkt, 'transfer_bps', f.transfer_bps) + '</td>' +
        '<td class="num">' + feeInp(mkt, 'exchange_bps', f.exchange_bps) + '</td>' +
        '<td class="num">' + feeInp(mkt, 'regulator_bps', f.regulator_bps) + '</td>' +
        '<td class="num ' + ((r.round_trip_bps || 0) > 20 ? 'is-down' : '') + '">' +
          (r.round_trip_bps == null ? '—' : r.round_trip_bps) + '</td></tr>';
    }).join('');

    const feeCard = card({
      title: '交易费用', sub: '按市场拆分 · 佣金可改成你自己的费率',
      tools: '<button type="button" class="btn ghost sm" id="fee-edit">' +
        ICON('settings') + '改佣金</button>',
      body: (feeRows
        ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
          '<th>市场</th><th>币种</th><th>佣金</th><th>印花税</th>' +
          '<th>过户</th><th>经手</th><th>证管</th><th>往返bps</th>' +
          '</tr></thead><tbody>' + feeRows + '</tbody></table></div>'
        : emptyBox('费用表未就绪', '')) +
        '<p class="muted" style="margin-top:10px">' +
          '单位 bps（万分之一）。佣金因人而异（万1.5~万3），**必须改成你自己的**；' +
          '印花税 A股仅卖出、港股双边。交易所费率可点下方来源核对。</p>'
    });

    // ---- 登录状态 ----
    const AU = S.auth || {};
    const authCard = AU.enabled ? card({
      title: '登录状态', sub: '会话 ' + (AU.session_days || 7) + ' 天，滑动续期',
      body:
        '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">' +
          '<span class="tag ' + (AU.logged_in ? 'hit' : 'miss') + '">' +
            (AU.logged_in ? '已登录' : '未登录') + '</span>' +
          '<span class="muted">当前会话数 ' + (AU.active_sessions || 0) + '</span>' +
          '<span class="muted">你的 IP ' + esc(AU.you_are || '—') + '</span>' +
          '<button type="button" class="btn ghost" id="au-logout">退出登录</button>' +
        '</div>' +
        '<p class="muted" style="margin-top:10px">退出后需要重新输入密码。' +
          '所有会话可在服务端用 <code>python -m qcore.auth</code> 查看。</p>'
    }) : card({
      title: '登录状态', sub: '未启用',
      cls: 'flush',
      body: banner({ tone: 'warn', icon: 'alert',
        title: '任何人都能打开这个页面看到你的持仓和策略',
        text: '服务端<strong>没有</strong>生效的密码。当前所有页面和 API 都是公开的 —— '
            + '包括持仓、成交记录、推荐名单、策略参数。',
        meta: '这是真风险，不是提示性警告。' }) +
        '<div style="padding:0 16px 16px" class="muted">' +
        '<p style="margin:12px 0 6px"><strong>怎么修</strong>（在 NAS 上执行）：</p>' +
        '<pre class="mono" style="white-space:pre-wrap;background:var(--surface-3,#1c2434);' +
        'padding:10px;border-radius:6px;font-size:11px;line-height:1.7">' +
        esc('# 1. 删掉当前容器（数据在挂载里，不会丢）\n' +
            'docker rm -f apb\n\n' +
            '# 2. 带上密码重新创建\n' +
            'docker run -d --name apb --restart unless-stopped \\\n' +
            '  -p 8765:8765 \\\n' +
            '  -v /usr/local/apps/@appdata/apb/data:/app/data \\\n' +
            '  -v /usr/local/apps/@appdata/apb/logs:/app/logs \\\n' +
            '  -v /var/run/docker.sock:/var/run/docker.sock \\\n' +
            '  -e APB_PASSWORD=你的密码 \\\n' +
            '  -e APB_TOKEN=你的令牌 \\\n' +
            '  -e APB_ALLOWED_HOSTS=你的域名 \\\n' +
            '  -e APB_PUBLIC_URL=https://你的域名 \\\n' +
            '  ghcr.io/ybd156/quant-apb:latest') +
        '</pre>' +
        '<p style="margin:10px 0 0">⚠ 用 <code>docker restart</code> 修不好 —— ' +
        'restart 不会改变环境变量，必须 <code>rm</code> 后重新 <code>run</code>。</p>' +
        '<p style="margin:6px 0 0">设好之后，容器会把关键变量记到 ' +
        '<code>data/state/env_snapshot.json</code>，以后应用内更新重建容器会自动带上，' +
        '不会因为一次遗漏而永久丢失。</p>' +
        '</div>'
    });

    // ---- 访问令牌 ----
    const curTok = getToken();
    const tokenCard = card({
      title: '访问令牌', sub: '服务端设了 APB_TOKEN 后，所有写操作都要带它',
      body:
        '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end">' +
          '<label class="field" style="flex:1;min-width:260px"><span>令牌</span>' +
            '<input type="password" id="tok-input" placeholder="粘贴 APB_TOKEN 的值" value="' +
            esc(curTok) + '"></label>' +
          '<button type="button" class="btn" id="tok-save">' + ICON('check') + '保存</button>' +
          '<button type="button" class="btn ghost" id="tok-clear">清除</button>' +
        '</div>' +
        '<p class="muted" style="margin-top:10px">当前状态：' +
          (curTok ? '<span class="tag hit">已设置</span>' : '<span class="tag miss">未设置</span>') +
          ' · 只存在本机 localStorage，不上传任何地方。</p>'
    });

    const trainCard = card({
      title: '训练任务', sub: 'NAS 下发 → 训练机自动领取 → 训练完自动上传',
      tools: '<button type="button" class="btn ghost sm" id="tk-refresh">' + ICON('refresh') + '刷新</button>',
      body:
        '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px">' +
          '<div><div class="muted" style="margin-bottom:6px">周期</div>' +
            '<div class="chip-row">' + horizonChips + '</div></div>' +
          '<label class="field" style="max-width:110px"><span>树数</span>' +
            '<input type="number" id="tk-trees" value="250" min="50" max="1000"></label>' +
          '<label class="field" style="max-width:110px"><span>股票池</span>' +
            '<input type="number" id="tk-stocks" value="300" min="50" max="800"></label>' +
          '<button type="button" class="btn" id="tk-add">' + ICON('plus') + '排任务</button>' +
        '</div>' +
        '<div class="muted" style="margin-bottom:10px">' +
          '队列：' + ['pending', 'claimed', 'done', 'failed', 'expired']
            .filter((k) => counts[k]).map((k) => (stLabel[k] || k) + ' ' + counts[k]).join(' · ') +
          (Object.keys(counts).length ? '' : '空') +
        '</div>' +
        (taskRows
          ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
            '<th>ID</th><th>周期</th><th>树</th><th>状态</th><th>执行机器</th><th>创建</th>' +
            '</tr></thead><tbody>' + taskRows + '</tbody></table></div>'
          : emptyBox('队列为空', '上面选好周期点「排任务」，训练机上跑 ' +
              '<code>python tools/sync_agent.py --nas http://&lt;NAS&gt;:8765 --loop 60</code> 就会自动领活')) +
        '<p class="muted" style="margin-top:10px">' +
          '训练机命令：<code>python tools/sync_agent.py --nas http://你的NAS:8765 --once</code>；' +
          '也可以在本机直接 <code>python tools/train_one_click.py</code>。' +
        '</p>'
    });

    const autoOn = S.autoRefresh !== false;
    const autoCard = card({
      title: '自动刷新', sub: '行情类页面自动更新，不用手点刷新',
      body: '<div class="segmented" id="set-auto">' +
        '<button type="button" class="seg' + (autoOn ? ' is-active' : '') + '" data-v="1">开启</button>' +
        '<button type="button" class="seg' + (!autoOn ? ' is-active' : '') + '" data-v="0">关闭</button>' +
        '</div><p class="muted" style="margin-top:10px">' +
        '交易时段 30 秒刷新一次，非交易时段 5 分钟。只在「总览 / 市场 / 风口 / 个股 / 自选」生效；' +
        '设置页不刷新（会把正在填的表单冲掉）。弹窗打开时也暂停。</p>'
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
      '<div style="margin-bottom:14px">' + feeCard + '</div>' +
      '<div class="grid two" style="margin-bottom:14px">' + authCard + autoCard + '</div>' +
      '<div style="margin-bottom:14px">' + tokenCard + '</div>' +
      '<div style="margin-bottom:14px">' + trainCard + '</div>' +
      '<div class="grid two" style="margin-bottom:14px">' + runCard + exportCard + '</div>' +
      themeCard;
  }

  /* ============================== 自动刷新 ============================= */

  /**
   * 行情类页面自动刷新。交易时段 30 秒，非交易时段 5 分钟。
   * 只在「行情/风口/个股/自选」页生效 —— 设置页刷新会把用户正在填的表单冲掉。
   */
  const AUTO_PAGES = { market: 1, hotspot: 1, chart: 1, watch: 1, overview: 1 };
  let _autoTimer = null;

  function _isTradingTime() {
    const d = new Date();
    const wd = d.getDay();
    if (wd === 0 || wd === 6) return false;
    const hm = d.getHours() * 60 + d.getMinutes();
    return (hm >= 9 * 60 + 15 && hm <= 11 * 60 + 35) ||
           (hm >= 12 * 60 + 55 && hm <= 15 * 60 + 10);
  }

  function startAutoRefresh(page) {
    stopAutoRefresh();
    if (!AUTO_PAGES[page]) return;
    if (S.autoRefresh === false) return;
    const ms = _isTradingTime() ? 30000 : 300000;
    _autoTimer = setTimeout(() => {
      // 弹窗打开时不刷，避免打断操作
      const mask = document.getElementById('modal-mask');
      const modalOpen = mask && !mask.hasAttribute('hidden');
      if (!modalOpen) navigate(S.page, false, true);
      else startAutoRefresh(page);
    }, ms);
  }

  function stopAutoRefresh() {
    if (_autoTimer) { clearTimeout(_autoTimer); _autoTimer = null; }
  }

  /* =============================== 路由 =============================== */

  const RENDERERS = {
    overview: pageOverview,
    chart: pageChart,
    hotspot: pageHotspot,
    watch: pageWatch,
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

  /** 切换模拟实盘的市场（记住上次选择） */
  function switchMarket(m) {
    const next = MKTS.indexOf(m) >= 0 ? m : 'CN';
    if (next === S.paperMarket && S.page === 'paper') return;
    S.paperMarket = next;
    try { localStorage.setItem('apb.paper.market', next); } catch (e) { /* ignore */ }
    clearCache();
    if (S.page === 'paper') navigate('paper', false);
  }

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

  /** 重新渲染当前页（不切换路由、不压历史栈）。 */
  function rerender() {
    try {
      return navigate(S.page || 'overview', false, true);
    } catch (e) {
      return null;
    }
  }

  async function navigate(id, push, silent) {
    if (!RENDERERS[id]) id = 'overview';
    stopAutoRefresh();
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
    const _scrollY = silent ? window.scrollY : 0;
    if (!silent) view.innerHTML = skeleton();
    let html;
    try {
      html = await RENDERERS[id]();
    } catch (e) {
      html = card({ title: '渲染失败', body: emptyBox('页面渲染出错', esc(String((e && e.message) || e)), 'alert') });
    }
    if (S.page !== id) return;      // 期间已切页
    view.innerHTML = html;
    if (silent) {
      window.scrollTo(0, _scrollY);   // 自动刷新不能把用户拽回顶部
    } else {
      window.scrollTo(0, 0);
    }
    bindPageEvents(id);
    startAutoRefresh(id);
  }

  /* --------------------------- 页内事件绑定 --------------------------- */

  function bindPageEvents(id) {
    if (id === 'hotspot') {
      const r = $('#hs-refresh');
      if (r) r.addEventListener('click', async () => {
        r.disabled = true;
        const res = await api('/api/hotspot?refresh=1');
        r.disabled = false;
        if (res && res.ok) { toast('已重新抓取', 'ok'); rerender(); }
        else toast('抓取失败', 'err');
      });
      $$('[data-goto]').forEach((a) => a.addEventListener('click', () => {
        S.chartCode = a.getAttribute('data-goto');
        navigate('chart', false);
      }));
      return;
    }
    if (id === 'model') {
      $$('#hz-tabs .seg').forEach((b) => b.addEventListener('click', () => {
        S.modelHorizon = Number(b.getAttribute('data-hz'));
        navigate('model', false);
      }));
      // 生成该周期预测（纯推理，秒级）
      const pOne = $('#md-predict');
      if (pOne) pOne.addEventListener('click', async () => {
        const h = S.modelHorizon || 1;
        pOne.disabled = true;
        const r = await api('/api/predict', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ horizon: h })
        });
        pOne.disabled = false;
        if (r && r.ok) {
          const d = r.data || {};
          toast('已生成 h' + h + ' 预测：' + (d.n || 0) + ' 只（' + (d.as_of || '') + '）', 'ok');
          rerender();
        } else {
          toast('生成失败：' + errText(r), 'err');
        }
      });
      // 补齐全部周期：逐个调 predict，缺模型的会明确跳过
      const pAll = $('#md-predict-any');
      if (pAll) pAll.addEventListener('click', async () => {
        pAll.disabled = true;
        const hs = [1, 3, 5, 10, 15];
        const done = [], skip = [];
        for (const h of hs) {
          const r = await api('/api/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizon: h })
          });
          if (r && r.ok) done.push(h);
          else skip.push(h);
        }
        pAll.disabled = false;
        toast('已生成 ' + done.length + ' 个周期' +
              (skip.length ? '，' + skip.length + ' 个缺模型（需在训练机训练）' : ''), 
              done.length ? 'ok' : 'err');
        rerender();
      });
      return;
    }
    if (id === 'chart') {
      $$('.wl-chip').forEach((b) => b.addEventListener('click', () => {
        S.chartCode = b.getAttribute('data-code');
        navigate('chart', false);
      }));
      const go = $('#ck-go');
      const inp = $('#ck-code');
      const doQuery = () => {
        const v = ((inp || {}).value || '').trim().toUpperCase();
        if (!v) { toast('请输入股票代码', 'warn'); return; }
        S.chartCode = v;
        navigate('chart', false);
      };
      if (go) go.addEventListener('click', doQuery);
      if (inp) {
        inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') doQuery(); });
        // 首次进入时聚焦，省一次点击
        if (!S.chartCode) setTimeout(() => inp.focus(), 60);
      }
      $$('#ck-period .seg').forEach((b) => b.addEventListener('click', () => {
        S.chartPeriod = b.getAttribute('data-p');
        navigate('chart', false);
      }));
      $$('#ck-adjust .seg').forEach((b) => b.addEventListener('click', () => {
        S.chartAdjust = Number(b.getAttribute('data-a'));
        navigate('chart', false);
      }));
      const cw = $('#ck-watch');
      if (cw) cw.addEventListener('click', async () => {
        if (!S.chartCode) { toast('先查一只股票', 'warn'); return; }
        const g = window.prompt('加到哪个分组？（可留空）', '观察');
        if (g === null) return;
        const r = await api('/api/watch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ add: [S.chartCode], group: g || '' })
        });
        toast(r && r.ok ? '已加入自选' : ('失败：' + ((r && (r.error || r.hint)) || '未知')),
              r && r.ok ? 'ok' : 'err');
      });
      return;
    }
    if (id === 'watch') {
      const add = $('#w-add');
      if (add) add.addEventListener('click', async () => {
        const raw = (($('#w-codes') || {}).value || '').trim();
        if (!raw) { toast('请输入代码', 'warn'); return; }
        const codes = raw.split(/[,，\s]+/).map((x) => x.trim().toUpperCase()).filter(Boolean);
        const g = (($('#w-group') || {}).value || '').trim();
        add.disabled = true;
        const r = await api('/api/watch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ add: codes, group: g })
        });
        add.disabled = false;
        if (r && r.ok) { toast('已加入 ' + codes.length + ' 只', 'ok'); rerender(); }
        else toast('失败：' + ((r && (r.error || r.hint)) || '未知'), 'err');
      });
      $$('.w-del').forEach((b) => b.addEventListener('click', async () => {
        const code = b.getAttribute('data-code');
        const r = await api('/api/watch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ remove: [code] })
        });
        if (r && r.ok) { toast('已移除 ' + code, 'ok'); rerender(); }
      }));
      $$('.w-move').forEach((b) => b.addEventListener('click', async () => {
        const g0 = b.getAttribute('data-group');
        const g1 = window.prompt('把「' + g0 + '」整组改到哪个分组？', g0);
        if (g1 === null) return;
        const codes = $$('.w-del').map((x) => x.getAttribute('data-code'));
        // 只改本组的：从表格里取
        const rows = b.closest('.card') ? b.closest('.card').querySelectorAll('.w-del') : [];
        const list = Array.prototype.map.call(rows, (x) => x.getAttribute('data-code'));
        const r = await api('/api/watch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ set_group: list, group: g1 })
        });
        if (r && r.ok) { toast('已改分组', 'ok'); rerender(); }
      }));
      $$('[data-goto]').forEach((a) => a.addEventListener('click', () => {
        S.chartCode = a.getAttribute('data-goto');
        navigate('chart', false);
      }));
      return;
    }
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
      if (days) days.addEventListener('change', () => {
        S.filter.days = Number(days.value) || 30;
        S.filter.page = 1;          // 换范围回第一页
        navigate('ledger', false);
      });
      // ---- 分页 ----
      $$('#f-page [data-pg]').forEach(function (b) {
        b.addEventListener('click', function () {
          S.filter.page = Number(b.getAttribute('data-pg')) || 1;
          navigate('ledger', false);
        });
      });
      $$('.day-bar[data-day]').forEach(function (el) {
        el.addEventListener('click', function () {
          const d = el.getAttribute('data-day');
          if (S.filter.from === d && S.filter.to === d) {
            S.filter.from = ''; S.filter.to = '';
          } else {
            S.filter.from = d; S.filter.to = d;
          }
          S.filter.page = 1;
          navigate('ledger', false);
        });
      });
      const clearDay = $('#f-clear-day');
      if (clearDay) clearDay.addEventListener('click', function () {
        S.filter.from = ''; S.filter.to = ''; S.filter.page = 1;
        navigate('ledger', false);
      });
      const psize = $('#f-psize');
      if (psize) psize.addEventListener('change', function () {
        S.filter.perDay = Number(psize.value) || 20;
        S.filter.page = 1;
        navigate('ledger', false);
      });
      const code = $('#f-code');
      if (code) {
        let t = null;
        code.addEventListener('input', () => {
          clearTimeout(t);
          t = setTimeout(() => { S.filter.q = code.value; S.filter.page = 1; navigate('ledger', false); }, 220);
        });
      }
      const from = $('#f-from');
      if (from) from.addEventListener('change', () => { S.filter.from = from.value; S.filter.page = 1; navigate('ledger', false); });
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

    if (id === 'overview' || id === 'risk') {
      // 「在线学习」卡片渲染在总览页，但两个页面都可能出现这个按钮
      const ol = $('#ol-run');
      if (ol) ol.addEventListener('click', async () => {
        ol.disabled = true;
        const r = await api('/api/online/learn?run=1');
        ol.disabled = false;
        if (r && r.ok) {
          const d = r.data || {};
          const n = (d.run && d.run.ok_count) || 0;
          const rec = (d.reconcile && d.reconcile.done) || 0;
          toast('对账 ' + rec + ' 条，' + n + ' 个周期完成校准', 'ok');
          rerender();
        } else {
          toast('学习失败：' + errText(r), 'err');
        }
      });
      if (id === 'overview') return;
    }
    if (id === 'paper') {
      $$('#pa-bucket .seg').forEach((b) => b.addEventListener('click', () => {
        S.paBucket = b.getAttribute('data-b');
        navigate('paper', false);
      }));
      $$('[data-pahz]').forEach((a) => a.addEventListener('click', () => {
        S.paHorizon = Number(a.getAttribute('data-pahz'));
        navigate('paper', false);
      }));
      const allBtn = $('#pa-all');
      if (allBtn) allBtn.addEventListener('click', async () => {
        if (!window.confirm('对所有已训练周期各调一次仓？（每个周期独立账户）')) return;
        allBtn.disabled = true;
        const r = await api('/api/paper/rebalance', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ market: S.paperMarket || 'CN', all_horizons: true })
        });
        allBtn.disabled = false;
        if (r && r.ok) {
          const res = (r.data && r.data.results) || [];
          const acted = res.filter((x) => x.ok && !x.skipped_all).length;
          const noPicks = res.filter((x) => x.reason === 'no_picks').length;
          toast('处理 ' + res.length + ' 个周期：' + acted + ' 个有动作' +
                (noPicks ? '，' + noPicks + ' 个还没预测' : ''), 'ok');
          rerender();
        } else toast('调仓失败：' + errText(r), 'err');
      });
      // 「清空重设」——用页面内弹窗，不用 window.prompt。
      // 实测：prompt 被浏览器拦截时返回 null，处理器直接 return，
      // 用户点了完全没反应（后端其实是好的）。
      const rsb = $('#pa-reset');
      if (rsb) rsb.addEventListener('click', () => {
        const mktNow = S.paperMarket || 'CN';
        const hzNow = S.paHorizon || 1;
        const _acc = S.paperAcct || {};
        const curNav = (_acc.equity != null) ? _acc.equity
                     : (_acc.initial_capital || 0);
        // 重投的默认本金必须来自「各周期本金」配置，不是当前净值 ——
        // 旧实现预填 curNav，用户设了 20 万、净值 19.8 万，点确认就变成 19.8 万。
        const _cfgCap = Number(_acc.capital_configured || _acc.initial_capital || 0);
        const _cfgExplicit = _acc.capital_explicit != null;
        const hzLabel = ({ 1: 'T+1 短线', 3: 'T+3 中短线', 5: '1 周 中线',
                           10: '2 周', 15: '15 天 中长线' })[hzNow] || ('h' + hzNow);
        openModal({
          title: '清空重设 · ' + mktNow + ' / ' + hzLabel,
          body:
            '<p>当前净值 <b>' + esc(fmtMoney(curNav)) + '</b>，' +
            '持仓 <b>' + esc(String(_acc.n_positions || 0)) + '</b> 只。</p>' +
            '<p class="muted" style="margin-top:6px">' +
            '会删除该账户的全部成交记录（其他市场/周期不受影响），' +
            '并备份到 <code>data/paper/backup/</code>。</p>' +
            '<label class="field" style="margin-top:14px"><span>新的本金</span>' +
            '<input type="number" id="rs-cap" step="1000" value="" placeholder="' +
            esc(String(Math.round(Number(_cfgCap) || 5000))) + '"></label>' +
            '<p class="muted" style="margin-top:4px">留空 = 用该周期配置本金 <b>' +
            esc(fmtMoney(_cfgCap || 5000)) + '</b>' +
            (_cfgExplicit ? '（你在「各周期本金」里设过）' : '（未单独设置，取市场默认本金）') +
            '</p>' +
            '<p class="form-msg" id="rs-msg"></p>',
          foot: '<button class="btn ghost" type="button" data-close="1">取消</button>' +
                '<button class="btn primary" type="button" id="rs-go">' +
                ICON('check') + '确认清空</button>'
        });
        const go = $('#rs-go');
        const msg = $('#rs-msg');
        if (!go) return;
        go.addEventListener('click', async () => {
          const raw = String(($('#rs-cap') || {}).value || '').trim();
          const cap = raw === '' ? null : Number(raw);
          if (cap !== null && (!isFinite(cap) || cap <= 0)) {
            if (msg) { msg.className = 'form-msg err'; msg.textContent = '金额无效'; }
            return;
          }
          go.disabled = true;
          if (msg) { msg.className = 'form-msg'; msg.textContent = '处理中…'; }
          const r = await api('/api/paper/reset', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ market: mktNow, horizon: hzNow, capital: cap })
          });
          go.disabled = false;
          if (r && r.ok) {
            const d = r.data || {};
            closeModal();
            toast('已清空 ' + mktNow + ' ' + hzLabel + '：删除 ' +
                  (d.cleared_trades || 0) + ' 条成交，本金 ' +
                  Number(d.capital || 0).toLocaleString(), 'ok');
            rerender();
          } else {
            const t = errText(r);
            if (msg) { msg.className = 'form-msg err'; msg.textContent = String(t).slice(0, 120); }
            toast('清空失败：' + t, 'err');
          }
        });
      });
      const rb = $('#pa-rebal');
      if (rb) rb.addEventListener('click', async () => {
        if (!window.confirm('按当前周期推荐调仓？会真实写入成交记录。')) return;
        rb.disabled = true;
        const r = await api('/api/paper/rebalance', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ market: S.paperMarket || 'CN',
                                 horizon: S.paHorizon || 1 })
        });
        rb.disabled = false;
        if (r && r.ok) {
          const o = (r.data && r.data.orders) || [];
          const sk = (r.data && r.data.skipped) || [];
          toast('已调仓：' + o.length + ' 笔' + (sk.length ? '，跳过 ' + sk.length + ' 只' : ''), 'ok');
          rerender();
        } else {
          toast('调仓失败：' + errText(r), 'err');
        }
      });
      $$('[data-goto]').forEach((a) => a.addEventListener('click', () => {
        S.chartCode = a.getAttribute('data-goto');
        navigate('chart', false);
      }));
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

      // 各周期本金保存
      const hzBtn = $('#hz-cap-save');
      if (hzBtn) hzBtn.addEventListener('click', () => {
        const mf = $('#acct-form');
        const mk = (mf && mf.elements.market) ? mf.elements.market.value
                                              : (S.paperMarket || 'CN');
        submitHorizonCapital(mk);
      });

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
      // 费用表单元格直接编辑：改完失焦即保存
      // 之前界面上写着「佣金可改成你自己的费率」，但单元格是静态文本点不了。
      $$('.fee-inp').forEach(function (inp) {
        inp.addEventListener('change', async function () {
          const mkt = inp.getAttribute('data-mkt');
          const field = inp.getAttribute('data-field');
          const val = inp.value === '' ? null : Number(inp.value);
          if (val !== null && (!isFinite(val) || val < 0)) {
            toast('费率必须是非负数', 'err'); return;
          }
          const patch = {}; patch[field] = val;
          inp.disabled = true;
          const r = await api('/api/fees/save', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ market: mkt, patch: patch })
          });
          inp.disabled = false;
          if (r && r.ok) { toast(mkt + ' 费率已保存', 'ok'); rerender(); }
          else { toast('保存失败：' + errText(r), 'err'); }
        });
      });
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

      // ---- 交易费用（分市场，可编辑）----
    const FE = S.fees || {};
    const feeRows = (FE.compare || []).map((r) => {
      const f = (FE.fees || {})[r.market] || {};
      return '<tr><td>' + esc(r.name || r.market) + '</td>' +
        '<td class="muted">' + esc(r.currency || '') + '</td>' +
        '<td class="num">' + (f.commission_bps == null ? '—' : f.commission_bps) + '</td>' +
        '<td class="num">' + (f.stamp_duty_bps == null ? '—' : f.stamp_duty_bps) +
          (f.stamp_side === 'sell' ? '<span class="muted"> 卖出</span>' :
           f.stamp_side === 'both' ? '<span class="muted"> 双边</span>' : '') + '</td>' +
        '<td class="num">' + (f.transfer_bps == null ? '—' : f.transfer_bps) + '</td>' +
        '<td class="num">' + (f.exchange_bps == null ? '—' : f.exchange_bps) + '</td>' +
        '<td class="num">' + (f.regulator_bps == null ? '—' : f.regulator_bps) + '</td>' +
        '<td class="num ' + ((r.round_trip_bps || 0) > 20 ? 'is-down' : '') + '">' +
          (r.round_trip_bps == null ? '—' : r.round_trip_bps) + '</td></tr>';
    }).join('');

    const feeCard = card({
      title: '交易费用', sub: '按市场拆分 · 佣金可改成你自己的费率',
      tools: '<button type="button" class="btn ghost sm" id="fee-edit">' +
        ICON('settings') + '改佣金</button>',
      body: (feeRows
        ? '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
          '<th>市场</th><th>币种</th><th>佣金</th><th>印花税</th>' +
          '<th>过户</th><th>经手</th><th>证管</th><th>往返bps</th>' +
          '</tr></thead><tbody>' + feeRows + '</tbody></table></div>'
        : emptyBox('费用表未就绪', '')) +
        '<p class="muted" style="margin-top:10px">' +
          '单位 bps（万分之一）。佣金因人而异（万1.5~万3），**必须改成你自己的**；' +
          '印花税 A股仅卖出、港股双边。交易所费率可点下方来源核对。</p>'
    });

    // ---- 登录状态 ----
    const AU = S.auth || {};
    const authCard = AU.enabled ? card({
      title: '登录状态', sub: '会话 ' + (AU.session_days || 7) + ' 天，滑动续期',
      body:
        '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">' +
          '<span class="tag ' + (AU.logged_in ? 'hit' : 'miss') + '">' +
            (AU.logged_in ? '已登录' : '未登录') + '</span>' +
          '<span class="muted">当前会话数 ' + (AU.active_sessions || 0) + '</span>' +
          '<span class="muted">你的 IP ' + esc(AU.you_are || '—') + '</span>' +
          '<button type="button" class="btn ghost" id="au-logout">退出登录</button>' +
        '</div>' +
        '<p class="muted" style="margin-top:10px">退出后需要重新输入密码。' +
          '所有会话可在服务端用 <code>python -m qcore.auth</code> 查看。</p>'
    }) : card({
      title: '登录状态', sub: '未启用',
      body: banner({ tone: 'warn', icon: 'alert',
        title: '任何人都能打开这个页面看到你的持仓和策略',
        text: '服务端没设 <code>APB_PASSWORD</code>。暴露到公网前必须设置。',
        meta: '设置后重启容器生效；设了之后所有页面和 API 都需要登录' })
    });

    // ---- 访问令牌 ----
      const tokSave = $('#tok-save');
      if (tokSave) {
        tokSave.addEventListener('click', () => {
          const v = (($('#tok-input') || {}).value || '').trim();
          setToken(v);
          toast(v ? '令牌已保存' : '令牌已清空', v ? 'ok' : 'warn');
          rerender();
        });
      }
      const tokClear = $('#tok-clear');
      if (tokClear) {
        tokClear.addEventListener('click', () => {
          setToken('');
          toast('令牌已清空', 'warn');
          rerender();
        });
      }

      // ---- 训练任务 ----
      const tkAdd = $('#tk-add');
      if (tkAdd) {
        tkAdd.addEventListener('click', async () => {
          const hs = $$('.tk-h').filter((x) => x.checked).map((x) => Number(x.value));
          if (!hs.length) { toast('至少选一个周期', 'warn'); return; }
          const trees = Number(($('#tk-trees') || {}).value || 250);
          const stocks = Number(($('#tk-stocks') || {}).value || 300);
          tkAdd.disabled = true;
          const r = await api('/api/sync/task', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizons: hs, trees, stocks })
          });
          tkAdd.disabled = false;
          if (r && r.ok) {
            toast('已排任务：h' + hs.join('/h'), 'ok');
            rerender();
          } else {
            toast('排任务失败：' + ((r && (r.error || r.hint)) || '未知'), 'err');
          }
        });
      }
      const tkRef = $('#tk-refresh');
      if (tkRef) tkRef.addEventListener('click', () => rerender());

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
      const mktBtn = ev.target.closest ? ev.target.closest('[data-mkt]') : null;
      if (mktBtn) { switchMarket(mktBtn.getAttribute('data-mkt')); return; }
      const btn = ev.target.closest ? ev.target.closest('[data-page]') : null;
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

  // 全局兜底：任何脚本错误都弹出来。
  // 之前点击处理器里出现 ReferenceError 时，界面毫无反应 ——
  // 用户只能看到「点了没用」，完全不知道发生了什么。
  window.addEventListener('error', (e) => {
    try {
      const msg = (e && e.message) ? String(e.message) : '未知脚本错误';
      if (/favicon/i.test(msg)) return;
      toast('页面脚本错误：' + msg.slice(0, 120), 'err');
    } catch (_) { /* 忽略 */ }
  });
  window.addEventListener('unhandledrejection', (e) => {
    try {
      const r = e && e.reason;
      const msg = (r && (r.message || r.error)) ? String(r.message || r.error) : String(r || '');
      if (msg) toast('请求失败：' + msg.slice(0, 120), 'err');
    } catch (_) { /* 忽略 */ }
  });

  async function boot() {
    // 自选股预加载：个股页要同步渲染，不能等异步
    try {
      const wr = await api('/api/watch');
      S.watchItems = (wr && wr.ok && wr.data && wr.data.items) || [];
    } catch (e) { S.watchItems = []; }
    buildNav();
    bindGlobal();
    const fromHash = (location.hash || '').replace(/^#\/?/, '');
    let last = null;
    try { last = localStorage.getItem('apb.page'); } catch (e) { /* ignore */ }
  try {
    const _ar = localStorage.getItem('apb.autoRefresh');
    S.autoRefresh = (_ar == null) ? true : (_ar === '1');
  } catch (e) { S.autoRefresh = true; }
    const start = RENDERERS[fromHash] ? fromHash : (RENDERERS[last] ? last : 'overview');
    try {
      const pm = localStorage.getItem('apb.paper.market');
      if (MKTS.indexOf(pm) >= 0) S.paperMarket = pm;
    } catch (e) { /* ignore */ }
    const use = $('#theme-ico use');
    if (use) use.setAttribute('href', document.documentElement.getAttribute('data-theme') === 'light' ? '#i-sun' : '#i-moon');
    await Promise.all([loadVersion(), loadHealth(), loadUpdate(), loadNames()]);
    await navigate(start, true);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
