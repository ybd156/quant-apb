/* ==========================================================================
 * APB Web UI — app.js
 * 原生 JS，零依赖（无 CDN / 无框架 / 完全离线可用）。
 * 铁律：前端只做 IO 与呈现，**不计算任何指标**。
 *   所有数值原样来自 /api/* -> data/state/*.json（由 qcore 产出）；
 *   这里只做「取字段 / 单位标注 / 排版」，不派生年化、不回算命中率、不重算净值。
 *
 * 数据契约（以 qcore/store.py + qcore/pipeline.py 落盘为准，并兼容 SPEC 简写形态）：
 *   status.json       {as_of, data_from, data_to, n_stocks, n_signals, n_days, mode, horizon, updated_at}
 *   weights.json      [{sid, family, weight, hit_rate, n_updates, mean_payoff, alarms}]
 *   regime.json       [{date, regime}]
 *   ledger.json       [{as_of, code, score, regime, entry_price, exit_price, ret, hit, status}]
 *   performance.json  {metrics:{hit_rate:{h:{hit_rate,...}}, rank_ic:{h:{mean,icir,...}}},
 *                      backtest:{long_short:{ann_return,sharpe,max_drawdown,...}, baselines:[...]}}
 *                     （兼容 SPEC 简写 {hit_rate:{h1..}, ic:{mean,ir,win_rate}, long_short:{annual,sharpe,mdd,curve}}）
 *   paper.json        {dates:[], sim_nav:[], live_nav:[], compare:{...}}
 *                     （兼容 SPEC 简写 {sim:{nav,ret}, live:{nav,ret}, diff:{...}}）
 * ========================================================================== */
(function () {
  'use strict';

  /* ----------------------------- 基础工具 ----------------------------- */

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.prototype.slice.call((root || document).querySelectorAll(sel));

  const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  function esc(v) {
    return String(v === null || v === undefined ? '' : v)
      .replace(/[&<>"']/g, (c) => ESC_MAP[c]);
  }

  function isNil(v) { return v === null || v === undefined || v === ''; }

  function toNum(v) {
    if (isNil(v) || typeof v === 'boolean') return null;
    const n = Number(v);
    return isFinite(n) ? n : null;
  }

  /** 取第一个可用数值（字段名在不同落盘形态间可能不同） */
  function firstNum(list) {
    for (let i = 0; i < list.length; i++) {
      const n = toNum(list[i]);
      if (n !== null) return n;
    }
    return null;
  }

  /** 原样数值 */
  function num(v, d) {
    const n = toNum(v);
    if (n === null) return '—';
    return n.toFixed(d === undefined ? 3 : d);
  }

  /** 比率：|v|<=1 视为分数（0.53 -> 53.00%），否则视为已是百分数 */
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

  /** 正负着色（A 股口径：红涨绿跌），只用于数值列 */
  function signClass(v) {
    const n = toNum(v);
    if (n === null || n === 0) return '';
    return n > 0 ? 'up-text' : 'down-text';
  }

  /** 基线表里引擎已按百分数给值（年化收益=14.21 表示 14.21%） */
  function fmtByKey(key, v) {
    const n = toNum(v);
    if (n === null) return isNil(v) ? '—' : String(v);
    if (/收益|回撤|胜率|波动/.test(key)) return n.toFixed(2) + '%';
    if (/期数|笔数/.test(key)) return n.toFixed(0);
    return n.toFixed(3);
  }

  const REGIMES = {
    0: { name: '下跌低波', color: 'var(--rg0)' },
    1: { name: '下跌高波', color: 'var(--rg1)' },
    2: { name: '上涨低波', color: 'var(--rg2)' },
    3: { name: '上涨高波', color: 'var(--rg3)' }
  };
  function regimeInfo(v) {
    const n = toNum(v);
    if (n === null) return { name: '未知', color: 'var(--rgx)' };
    return REGIMES[n] || { name: 'regime ' + n, color: 'var(--rgx)' };
  }

  let toastTimer = null;
  function toast(msg, kind) {
    const el = $('#toast');
    el.textContent = msg;
    el.className = 'toast show' + (kind ? ' ' + kind : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 3800);
  }

  function emptyBox(title, desc) {
    return '<div class="empty"><div class="ico">◌</div>' +
      '<div class="t">' + esc(title) + '</div>' +
      (desc ? '<div class="d">' + desc + '</div>' : '') + '</div>';
  }

  /** 数据未就绪 / 出错的统一占位（desc 允许内联 <code>） */
  function notReadyBox(res, what) {
    const err = (res && res.error) || 'unknown';
    if (err === 'not_ready') {
      const file = res.missing
        ? '<code>data/state/' + esc(res.missing) + '</code>' : '数据文件';
      return emptyBox('未就绪：等待引擎首次运行',
        '缺少 ' + file + '。先运行 <code>python -m qcore.live --once</code>' +
        '（首次需 <code>--rebuild</code>），生成后点「刷新」即可。');
    }
    if (err === 'bad_json') {
      return emptyBox('数据文件解析失败', '文件可能正在写入，稍后点「刷新」重试。');
    }
    if (err === 'network_error') {
      return emptyBox('无法连接服务', '服务进程可能已退出，请检查终端。');
    }
    return emptyBox('加载失败：' + esc(err), what ? esc(what) : '');
  }

  /* ------------------------------ 请求层 ------------------------------ */

  async function api(url, options) {
    const opts = Object.assign({ headers: { Accept: 'application/json' } }, options || {});
    let res;
    try {
      res = await fetch(url, opts);
    } catch (e) {
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

  /* ------------------------------ 图表 ------------------------------- */

  function chartWidth() { return window.innerWidth < 700 ? 540 : 960; }

  function fmtAxis(v) {
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    return v.toFixed(3);
  }

  /**
   * 手绘折线图（纯 SVG，无图表库）。
   * series: [{ values:[], color, label, area }]
   * opt:    { height, labels:{first,last}, aria }
   */
  function chartSVG(series, opt) {
    opt = opt || {};
    const W = chartWidth();
    const H = opt.height || 260;
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
    const ticks = 4;
    for (let k = 0; k <= ticks; k++) {
      const y = padT + (ih * k) / ticks;
      const val = hi - (span * k) / ticks;
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
      out += '<text class="axis-text" x="' + padL + '" y="' + (H - 8) + '">' +
        esc(opt.labels.first || '') + '</text>';
      out += '<text class="axis-text" x="' + (W - padR) + '" y="' + (H - 8) +
        '" text-anchor="end">' + esc(opt.labels.last || '') + '</text>';
    }

    series.forEach((s, si) => {
      const vals = s.values || [];
      let d = '';
      let started = false;
      for (let i = 0; i < vals.length; i++) {
        const v = toNum(vals[i]);
        if (v === null) { started = false; continue; }
        d += (started ? 'L' : 'M') + px(i).toFixed(1) + ' ' + py(v).toFixed(1) + ' ';
        started = true;
      }
      if (!d) return;
      if (s.area && vals.length > 1) {
        const first = px(0).toFixed(1);
        const last = px(vals.length - 1).toFixed(1);
        out += '<path d="' + d + 'L' + last + ' ' + (padT + ih) + ' L' + first + ' ' +
          (padT + ih) + ' Z" fill="' + s.color + '" opacity="0.10" stroke="none"/>';
      }
      out += '<path d="' + d + '" fill="none" stroke="' + s.color +
        '" stroke-width="' + (si === 0 ? 1.9 : 1.5) + '" stroke-linejoin="round" ' +
        'stroke-linecap="round"' + (s.dashed ? ' stroke-dasharray="5 4"' : '') + '/>';
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

  /* --------------------------- 序列抽取助手 --------------------------- */

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

  /* -------------------- 形态适配（不产生新数值） ---------------------- */

  /** performance.json -> 归一展示结构 */
  function adaptPerformance(d) {
    const out = { hits: [], ic: [], ls: null, baselines: [], curve: null, labels: null };
    if (!d || typeof d !== 'object') return out;

    const metrics = (d.metrics && typeof d.metrics === 'object') ? d.metrics : {};
    const backtest = (d.backtest && typeof d.backtest === 'object') ? d.backtest : {};

    // ---- 命中率 ----
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
        out.hits.push({ h: m ? Number(m[1]) : 0, label: 'h = ' + (m ? m[1] : k) + ' 日', value: val });
      });
      out.hits.sort((a, b) => a.h - b.h);
    }

    // ---- IC ----
    const ic = (d.ic && typeof d.ic === 'object') ? d.ic
      : (metrics.rank_ic && typeof metrics.rank_ic === 'object') ? metrics.rank_ic : null;
    if (ic) {
      if (toNum(ic.mean) !== null) {
        out.ic.push({ label: 'mean IC', value: toNum(ic.mean), kind: 'num' });
        out.ic.push({ label: 'ICIR', value: firstNum([ic.ir, ic.icir]), kind: 'num' });
        out.ic.push({ label: 'IC 胜率', value: firstNum([ic.win_rate, ic.positive_ratio]), kind: 'pct' });
        if (toNum(ic.t_stat) !== null) out.ic.push({ label: 't 值', value: toNum(ic.t_stat), kind: 'num' });
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

    // ---- 多空组合 ----
    const lsRaw = (d.long_short && typeof d.long_short === 'object') ? d.long_short
      : (backtest.long_short && typeof backtest.long_short === 'object') ? backtest.long_short : null;
    if (lsRaw) {
      const ls = {
        annual: firstNum([lsRaw.annual, lsRaw.ann_return, lsRaw.ann_ls]),
        sharpe: toNum(lsRaw.sharpe),
        mdd: firstNum([lsRaw.mdd, lsRaw.max_drawdown]),
        win_rate: toNum(lsRaw.win_rate),
        n_days: toNum(lsRaw.n_days),
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

    // ---- 基线 ----
    if (Array.isArray(backtest.baselines)) {
      out.baselines = backtest.baselines.filter((r) => r && typeof r === 'object');
    } else if (d.baseline && typeof d.baseline === 'object') {
      out.baselines = Object.keys(d.baseline).map((k) => {
        const v = d.baseline[k];
        return (v && typeof v === 'object') ? Object.assign({ 策略: k }, v) : { 策略: k, 值: v };
      });
    }
    return out;
  }

  /** paper.json -> 归一展示结构 */
  function adaptPaper(d) {
    const out = {
      dates: [], simNav: [], liveNav: [],
      simItems: [], liveItems: [], diffItems: [], cmp: {}
    };
    if (!d || typeof d !== 'object') return out;

    const cmp = (d.compare && typeof d.compare === 'object') ? d.compare
      : (d.diff && typeof d.diff === 'object') ? d.diff : {};
    out.cmp = cmp;

    // 曲线：真实形态是顶层 dates/sim_nav/live_nav；兼容 SPEC 的 sim.curve/live.curve
    if (Array.isArray(d.sim_nav) || Array.isArray(d.live_nav)) {
      out.dates = Array.isArray(d.dates) ? d.dates : [];
      out.simNav = seriesValues(d.sim_nav || []);
      out.liveNav = seriesValues(d.live_nav || []);
    } else {
      const sc = findCurve(d.sim);
      const lc = findCurve(d.live);
      if (sc) {
        out.simNav = seriesValues(sc);
        const dd = seriesDates(sc).filter(Boolean);
        if (dd.length) out.dates = dd;
      }
      if (lc) {
        out.liveNav = seriesValues(lc);
        if (!out.dates.length) {
          const dd = seriesDates(lc).filter(Boolean);
          if (dd.length) out.dates = dd;
        }
      }
    }

    const sim = (d.sim && typeof d.sim === 'object') ? d.sim : {};
    const live = (d.live && typeof d.live === 'object') ? d.live : {};

    // 卡片：优先用引擎给的 compare.*，其次 SPEC 的 sim/live 字段
    const simNav = firstNum([cmp.sim_final, sim.nav]);
    const liveNav = firstNum([cmp.live_final, live.nav]);
    if (simNav !== null) out.simItems.push({ k: '期末净值', v: num(simNav, 4) });
    const simAnnual = firstNum([cmp.sim_annual]);
    if (simAnnual !== null) out.simItems.push({ k: '年化', v: pct(simAnnual) });
    const simRet = firstNum([sim.ret]);
    if (simRet !== null && simAnnual === null) out.simItems.push({ k: '累计收益', v: signedPct(simRet) });
    const simHit = firstNum([cmp.sim_hit_rate]);
    if (simHit !== null) out.simItems.push({ k: '命中率', v: pct(simHit) });

    if (liveNav !== null) out.liveItems.push({ k: '期末净值', v: num(liveNav, 4) });
    const liveAnnual = firstNum([cmp.live_annual]);
    if (liveAnnual !== null) out.liveItems.push({ k: '年化', v: pct(liveAnnual) });
    const liveRet = firstNum([live.ret]);
    if (liveRet !== null && liveAnnual === null) out.liveItems.push({ k: '累计收益', v: signedPct(liveRet) });
    const liveTrades = firstNum([cmp.live_trades]);
    if (liveTrades !== null) out.liveItems.push({ k: '实盘成交', v: num(liveTrades, 0) + ' 笔' });

    // 差异指标
    const te = firstNum([cmp.tracking_error_ann, cmp.tracking_error]);
    if (te !== null) out.diffItems.push({ k: '跟踪误差(年化)', v: pct(te) });
    const pnlGap = firstNum([cmp.pnl_gap]);
    if (pnlGap !== null) out.diffItems.push({ k: '净值差 pnl_gap', v: signedPct(pnlGap), cls: signClass(pnlGap) });
    const hitGap = firstNum([cmp.hit_gap]);
    if (hitGap !== null) out.diffItems.push({ k: '命中率差 hit_gap', v: signedPct(hitGap), cls: signClass(hitGap) });
    const slip = firstNum([cmp.exec_slippage_bps]);
    if (slip !== null) out.diffItems.push({ k: '执行滑点', v: num(slip, 1) + ' bps', cls: signClass(-slip) });
    const nCmp = firstNum([cmp.n_days_compared]);
    if (nCmp !== null) out.diffItems.push({ k: '对比天数', v: num(nCmp, 0) + ' 天' });
    const simFinal = firstNum([cmp.sim_final]);
    if (simFinal !== null && liveNav !== null) {
      out.diffItems.push({ k: '模拟期末净值', v: num(simFinal, 4) });
    }
    if (!isNil(cmp.note)) {
      out.diffItems.push({ k: '引擎备注 note', v: String(cmp.note) });
    }
    return out;
  }

  /* ---------------------------- 展示小件 ----------------------------- */

  function kvItem(k, v, cls) {
    return '<div class="item"><span class="k">' + esc(k) + '</span>' +
      '<span class="v ' + (cls || '') + '">' + esc(v) + '</span></div>';
  }

  function statCard(k, v, sm) {
    return '<div class="stat"><span class="k">' + esc(k) + '</span>' +
      '<span class="v' + (sm ? ' sm' : '') + '">' + esc(v) + '</span></div>';
  }

  /* ============================ 状态栏 ============================= */

  async function loadBadges() {
    const res = await api('/api/status');
    const asofEl = $('#badge-asof');
    const modeEl = $('#badge-mode');
    if (res.ok && res.data && typeof res.data === 'object') {
      const d = res.data;
      asofEl.textContent = '数据日 ' + (d.as_of || '—');
      modeEl.textContent = '模式 ' + (d.mode || '—');
      $('#foot-stamp').textContent = 'updated_at ' + (d.updated_at || '—');
    } else {
      asofEl.textContent = '数据日 —';
      modeEl.textContent = res.error === 'not_ready' ? '未就绪' : '不可用';
    }
  }

  /* ============================ 总览 =============================== */

  function renderStatusCards(res) {
    const host = $('#status-cards');
    if (!res.ok || !res.data || typeof res.data !== 'object') {
      host.innerHTML = notReadyBox(res);
      return;
    }
    const d = res.data;
    let html = '';
    html += statCard('数据日 as_of', d.as_of || '—');
    html += statCard('股票数', num(d.n_stocks, 0));
    html += statCard('信号数', num(d.n_signals, 0));
    html += statCard('交易日数', num(d.n_days, 0));
    if (!isNil(d.horizon)) html += statCard('预测周期 h', num(d.horizon, 0) + ' 日');
    html += statCard('数据区间', (d.data_from || '—') + ' → ' + (d.data_to || '—'), true);
    html += statCard('最后更新', d.updated_at || '—', true);
    host.innerHTML = html;
  }

  function renderHitRates(perf) {
    const host = $('#hitrates');
    if (!perf) { host.innerHTML = notReadyBox(perfRes); return; }
    if (!perf.hits.length) {
      host.innerHTML = emptyBox('暂无命中率', 'performance.json 里没有 hit_rate / metrics.hit_rate。');
      return;
    }
    host.innerHTML = perf.hits.map((it) => {
      const x = Math.abs(it.value) <= 1.0000001 ? it.value * 100 : it.value;
      const width = Math.max(0, Math.min(100, x));
      return '<div class="gauge"><div class="lbl">' + esc(it.label) + '</div>' +
        '<div class="val">' + esc(pct(it.value)) + '</div>' +
        '<div class="bar"><i style="width:' + width.toFixed(1) + '%"></i></div></div>';
    }).join('');
  }

  function renderIC(perf) {
    const host = $('#ic-block');
    if (!perf) { host.innerHTML = notReadyBox(perfRes); return; }
    if (!perf.ic.length) {
      host.innerHTML = emptyBox('暂无 IC', 'performance.json 里没有 ic / metrics.rank_ic。');
      return;
    }
    host.innerHTML = perf.ic.map((it) => kvItem(
      it.label,
      it.kind === 'pct' ? pct(it.value) : num(it.value, 4),
      it.kind === 'pct' ? '' : signClass(it.value)
    )).join('');
  }

  function renderLongShort(perf) {
    const host = $('#ls-chart');
    const meta = $('#ls-meta');
    if (!perf) {
      meta.textContent = '—';
      host.innerHTML = notReadyBox(perfRes);
      return;
    }
    const ls = perf.ls;
    if (!ls) {
      meta.textContent = '—';
      host.innerHTML = emptyBox('暂无多空组合', 'performance.json 里没有 backtest.long_short。');
      return;
    }
    const bits = [];
    if (ls.annual !== null) bits.push('年化 ' + pct(ls.annual));
    if (ls.sharpe !== null) bits.push('Sharpe ' + num(ls.sharpe, 2));
    if (ls.mdd !== null) bits.push('最大回撤 ' + pct(ls.mdd));
    if (ls.win_rate !== null) bits.push('胜率 ' + pct(ls.win_rate));
    if (ls.n_periods !== null) bits.push('期数 ' + num(ls.n_periods, 0));
    meta.textContent = bits.join(' · ') || '—';

    if (!perf.curve || !perf.curve.length) {
      host.innerHTML = emptyBox('暂无净值曲线',
        'performance.json 的 <code>backtest.long_short</code> 未含 <code>equity</code> 序列' +
        '（引擎导出时剔除了 ndarray）。年化 / Sharpe / 回撤见上方标题行。');
      return;
    }
    host.innerHTML = chartSVG(
      [{ values: perf.curve, color: '#7c9cff', area: true, label: '多空净值' }],
      { height: 250, labels: perf.labels || {}, aria: '多空组合净值曲线' }
    );
  }

  function renderRegime(res) {
    const band = $('#regime-band');
    const legend = $('#regime-legend');
    const meta = $('#regime-meta');
    if (!res.ok || !res.data) {
      meta.textContent = '—';
      band.classList.add('is-empty');
      band.innerHTML = notReadyBox(res);
      legend.innerHTML = '';
      return;
    }
    band.classList.remove('is-empty');
    const rows = (res.data.rows || []).filter((r) => r && typeof r === 'object');
    const last = rows.length ? rows[rows.length - 1] : null;
    meta.textContent = '共 ' + rows.length + ' 个交易日' +
      (last ? ' · 最新 ' + (last.date || '—') + ' ' + regimeInfo(last.regime).name : '');
    if (!rows.length) {
      band.innerHTML = emptyBox('暂无 regime 序列', 'regime.json 为空。');
      legend.innerHTML = '';
      return;
    }
    band.innerHTML = rows.map((r) => {
      const info = regimeInfo(r.regime);
      return '<span style="background:' + info.color + '" title="' +
        esc((r.date || '') + ' · ' + info.name) + '"></span>';
    }).join('');
    legend.innerHTML = Object.keys(REGIMES).map((k) => (
      '<span class="lg"><i class="sw" style="background:' + REGIMES[k].color + '"></i>' +
      esc(k + ' = ' + REGIMES[k].name) + '</span>'
    )).join('');
  }

  function renderBaseline(perf) {
    const card = $('#baseline-card');
    const host = $('#baseline-block');
    if (!perf || !perf.baselines.length) { card.hidden = true; return; }
    const rows = perf.baselines;
    const cols = [];
    rows.forEach((r) => Object.keys(r).forEach((k) => {
      if (cols.indexOf(k) < 0 && cols.length < 7) cols.push(k);
    }));
    card.hidden = false;
    let html = '<div class="table-wrap"><table class="table"><thead><tr>' +
      cols.map((c) => '<th>' + esc(c) + '</th>').join('') + '</tr></thead><tbody>';
    rows.forEach((r) => {
      html += '<tr>' + cols.map((c) => (
        '<td class="' + (toNum(r[c]) !== null ? 'num' : '') + '">' + esc(fmtByKey(c, r[c])) + '</td>'
      )).join('') + '</tr>';
    });
    html += '</tbody></table></div>';
    host.innerHTML = html;
  }

  let perfRes = null;

  async function loadOverview() {
    const [status, performance, regime] = await Promise.all([
      api('/api/status'), api('/api/performance'), api('/api/regime')
    ]);
    perfRes = performance;
    renderStatusCards(status);
    const perf = performance.ok ? adaptPerformance(performance.data) : null;
    renderHitRates(perf);
    renderIC(perf);
    renderLongShort(perf);
    renderBaseline(perf);
    renderRegime(regime);
    loadBadges();
  }

  /* ============================ 信号 =============================== */

  function renderBars(host, rows, maxW, side) {
    if (!rows || !rows.length) {
      host.innerHTML = emptyBox('暂无信号', '');
      return;
    }
    host.innerHTML = rows.map((r) => {
      const w = toNum(r.weight);
      const width = (w === null || !maxW) ? 0 : Math.max(1.5, (w / maxW) * 100);
      const cls = side === 'bottom' ? 'neg' : '';
      return '<button type="button" class="bar-row" data-sid="' + esc(r.sid || '') + '">' +
        '<span class="bar-track"><i class="bar-fill ' + cls + '" style="width:' +
        width.toFixed(1) + '%"></i><span class="bar-name">' + esc(r.sid || '—') +
        '</span></span><span class="bar-val">' + esc(num(w, 4)) + '</span></button>';
    }).join('');
  }

  function renderSignalDetail(row) {
    const host = $('#signal-detail');
    const sid = $('#signal-detail-sid');
    if (!row) {
      sid.textContent = '未选择';
      host.innerHTML = emptyBox('点击左侧条形查看明细', '');
      return;
    }
    sid.textContent = row.sid || '—';
    let html = '';
    html += kvItem('家族 family', row.family || '—');
    html += kvItem('权重 weight', num(row.weight, 4));
    html += kvItem('命中率 hit_rate', pct(row.hit_rate));
    html += kvItem('更新次数 n_updates', num(row.n_updates, 0));
    html += kvItem('平均收益 mean_payoff', num(row.mean_payoff, 4), signClass(row.mean_payoff));
    if (!isNil(row.alarms)) html += kvItem('发声次数 alarms', num(row.alarms, 0));
    host.innerHTML = html;
  }

  async function loadSignals() {
    const top = $('#topn').value || 20;
    const res = await api('/api/signals?top=' + encodeURIComponent(top));
    const meta = $('#signals-meta');
    if (!res.ok || !res.data) {
      meta.textContent = '—';
      $('#bars-top').innerHTML = notReadyBox(res);
      $('#bars-bottom').innerHTML = notReadyBox(res);
      renderSignalDetail(null);
      return;
    }
    const topRows = res.data.top || [];
    const botRows = res.data.bottom || [];
    meta.textContent = '共 ' + (res.data.total || 0) + ' 个信号 · 每侧 ' +
      (res.data.top_n || topRows.length) + ' 条';
    const maxW = topRows.length ? toNum(topRows[0].weight) || 0 : 0;
    renderBars($('#bars-top'), topRows, maxW, 'top');
    renderBars($('#bars-bottom'), botRows, maxW, 'bottom');

    const all = topRows.concat(botRows);
    let keep = all.filter((r) => r && r.sid === state.selSid)[0];
    if (!keep && all[0]) { state.selSid = all[0].sid; keep = all[0]; }
    renderSignalDetail(keep || null);
    $$('.bar-row').forEach((b) => b.classList.toggle('is-sel',
      b.getAttribute('data-sid') === state.selSid));
  }

  /* ============================ 台账 =============================== */

  // 只影响渲染条数（面向 N100 这类小主机的性能护栏），不做任何统计
  const LEDGER_MAX_ROWS = 500;

  function isHit(r) {
    return r.hit === true || r.hit === 1 || r.hit === '1' || r.hit === 'true' || r.hit === 'True';
  }

  function ledgerVerdict(row) {
    const st = String(row.status || '').toLowerCase();
    if (st === 'pending' || isNil(row.hit)) return '<span class="tag pending">pending</span>';
    if (isHit(row)) return '<span class="tag hit">命中</span>';
    if (row.hit === false || row.hit === 0 || row.hit === '0' ||
        row.hit === 'false' || row.hit === 'False') {
      return '<span class="tag miss">未中</span>';
    }
    return '<span class="tag na">' + esc(String(row.hit)) + '</span>';
  }

  function renderLedger(res) {
    const body = $('#ledger-body');
    const meta = $('#ledger-meta');
    if (!res.ok || !res.data) {
      meta.textContent = '—';
      body.innerHTML = '<tr><td colspan="8">' + notReadyBox(res) + '</td></tr>';
      return;
    }
    const rows = (res.data.rows || []).filter((r) => r && typeof r === 'object');
    const f = state.ledgerFilter;
    const shown = rows.filter((r) => {
      const pending = String(r.status || '').toLowerCase() === 'pending' || isNil(r.hit);
      if (f === 'all') return true;
      if (f === 'pending') return pending;
      if (pending) return false;
      return f === 'hit' ? isHit(r) : !isHit(r);
    });
    meta.textContent = '显示 ' + Math.min(shown.length, LEDGER_MAX_ROWS) + ' / ' + rows.length + ' 条' +
      (shown.length > LEDGER_MAX_ROWS ? '（已截断，缩小天数查看更早）' : '');
    if (!shown.length) {
      body.innerHTML = '<tr><td colspan="8">' + emptyBox('该筛选下没有记录', '') + '</td></tr>';
      return;
    }
    body.innerHTML = shown.slice(0, LEDGER_MAX_ROWS).map((r) => {
      const ret = toNum(r.ret);
      return '<tr>' +
        '<td class="mono">' + esc(r.as_of || '—') + '</td>' +
        '<td class="mono">' + esc(r.code || '—') + '</td>' +
        '<td class="num">' + esc(num(r.score, 4)) + '</td>' +
        '<td>' + esc(regimeInfo(r.regime).name) + '</td>' +
        '<td class="num">' + esc(num(r.entry_price, 2)) + '</td>' +
        '<td class="num">' + esc(num(r.exit_price, 2)) + '</td>' +
        '<td class="num ' + (ret === null ? '' : (ret > 0 ? 'up-text' : ret < 0 ? 'down-text' : '')) +
        '">' + esc(signedPct(r.ret)) + '</td>' +
        '<td>' + ledgerVerdict(r) + '</td></tr>';
    }).join('');
  }

  async function loadLedger() {
    const days = $('#ledger-days').value || 30;
    const res = await api('/api/ledger?days=' + encodeURIComponent(days));
    state.ledger = res;
    renderLedger(res);
  }

  /* ========================== 模拟实盘 ============================= */

  function renderPaper(res) {
    const simNav = $('#sim-nav');
    const liveNav = $('#live-nav');
    const simKv = $('#sim-kv');
    const liveKv = $('#live-kv');
    const chartHost = $('#paper-chart');
    const chartMeta = $('#paper-chart-meta');
    const diffHost = $('#paper-diff');

    if (!res.ok || !res.data) {
      simNav.textContent = '—'; liveNav.textContent = '—';
      simNav.className = 'nav-value'; liveNav.className = 'nav-value';
      simKv.innerHTML = notReadyBox(res);
      liveKv.innerHTML = '';
      chartMeta.textContent = '—';
      chartHost.innerHTML = notReadyBox(res);
      diffHost.innerHTML = notReadyBox(res);
      return;
    }

    const p = adaptPaper(res.data);
    const simFinal = firstNum([p.cmp.sim_final]);
    const liveFinal = firstNum([p.cmp.live_final]);

    simNav.textContent = simFinal !== null ? num(simFinal, 4) : '—';
    simNav.className = 'nav-value ' + signClass(firstNum([p.cmp.sim_annual]));
    liveNav.textContent = liveFinal !== null ? num(liveFinal, 4) : '—';
    liveNav.className = 'nav-value ' + signClass(firstNum([p.cmp.live_annual]));

    simKv.innerHTML = p.simItems.length
      ? p.simItems.map((it) => kvItem(it.k, it.v)).join('')
      : emptyBox('暂无模拟盘指标', 'paper.json 未提供 sim 相关字段。');
    liveKv.innerHTML = p.liveItems.length
      ? p.liveItems.map((it) => kvItem(it.k, it.v)).join('')
      : emptyBox('暂无实盘指标', '录入实盘成交后，引擎重跑即会填充。');

    // 净值双线
    const hasCurve = p.simNav.some((v) => v !== null) || p.liveNav.some((v) => v !== null);
    if (!hasCurve) {
      chartMeta.textContent = '无曲线';
      chartHost.innerHTML = emptyBox('暂无净值曲线',
        'paper.json 未包含 <code>sim_nav</code> / <code>live_nav</code> 序列。');
    } else {
      const dates = p.dates || [];
      chartMeta.textContent = (dates.length ? dates[0] + ' → ' + dates[dates.length - 1] + ' · ' : '') +
        p.simNav.length + ' / ' + p.liveNav.length + ' 点';
      chartHost.innerHTML = chartSVG([
        { values: p.simNav, color: '#4d9cf0', label: '模拟盘', area: false },
        { values: p.liveNav, color: '#e3b341', label: '实盘', area: false }
      ], {
        height: 270,
        labels: { first: dates[0] || '', last: dates[dates.length - 1] || '' },
        aria: '模拟盘与实盘净值对比'
      });
    }

    diffHost.innerHTML = p.diffItems.length
      ? p.diffItems.map((it) => kvItem(it.k, it.v, it.cls)).join('')
      : emptyBox('暂无差异指标', 'paper.json 未提供 compare 字段。');
  }

  function renderTrades(res) {
    const body = $('#trades-body');
    const meta = $('#trades-meta');
    if (!res.ok || !res.data) {
      meta.textContent = '—';
      body.innerHTML = '<tr><td colspan="6">' + notReadyBox(res) + '</td></tr>';
      return;
    }
    const rows = res.data.rows || [];
    meta.textContent = '共 ' + (res.data.n || 0) + ' 笔';
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="6">' + emptyBox('还没有录入成交', '') + '</td></tr>';
      return;
    }
    body.innerHTML = rows.map((r) => {
      const side = String(r.side || '').toLowerCase();
      const sideTxt = side === 'buy' ? '买入' : side === 'sell' ? '卖出' : (r.side || '—');
      const sideCls = side === 'buy' ? 'up-text' : side === 'sell' ? 'down-text' : '';
      return '<tr>' +
        '<td class="mono">' + esc(r.date || '—') + '</td>' +
        '<td class="mono">' + esc(r.code || '—') + '</td>' +
        '<td class="' + sideCls + '">' + esc(sideTxt) + '</td>' +
        '<td class="num">' + esc(num(r.price, 3)) + '</td>' +
        '<td class="num">' + esc(num(r.qty, 0)) + '</td>' +
        '<td>' + esc(r.note || '') + '</td></tr>';
    }).join('');
  }

  async function loadPaper() {
    const [paper, trades] = await Promise.all([
      api('/api/paper'), api('/api/paper/trades?limit=20')
    ]);
    renderPaper(paper);
    renderTrades(trades);
  }

  /* --------------------------- 成交录入 --------------------------- */

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
    if (!isFinite(payload.price) || payload.price <= 0) {
      msg.className = 'form-msg err'; msg.textContent = '成交价必须为正数'; return;
    }
    if (!isFinite(payload.qty) || payload.qty <= 0 || payload.qty % 1 !== 0) {
      msg.className = 'form-msg err'; msg.textContent = '数量必须为正整数'; return;
    }

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
      msg.textContent = '已写入 live_trades.jsonl：' + payload.code + ' ' +
        payload.side + ' ' + payload.qty + ' @ ' + payload.price;
      form.elements.code.value = '';
      form.elements.price.value = '';
      form.elements.qty.value = '';
      form.elements.note.value = '';
      toast('成交已记录', 'ok');
      renderTrades(await api('/api/paper/trades?limit=20'));
    } else {
      msg.className = 'form-msg err';
      msg.textContent = '失败：' + (res.error || 'unknown') + (res.hint ? ' — ' + res.hint : '');
      toast('录入失败：' + (res.error || 'unknown'), 'err');
    }
  }

  /* ---------------------------- 运行一次 --------------------------- */

  function getToken() {
    let t = sessionStorage.getItem('apb_token') || '';
    if (!t) {
      t = window.prompt('请输入服务端环境变量 APB_TOKEN 的值（仅保存在本标签页会话中）', '') || '';
      if (t) sessionStorage.setItem('apb_token', t);
    }
    return t;
  }

  async function pollRun(tries) {
    const res = await api('/api/run/status');
    if (!res.ok) { toast('运行状态查询失败：' + res.error, 'err'); return; }
    const s = res.data || {};
    if (s.state === 'running') {
      if (tries > 0) setTimeout(() => pollRun(tries - 1), 3000);
      return;
    }
    if (s.state === 'done') {
      toast('运行完成，耗时 ' + (s.duration_s || '?') + 's', 'ok');
      refreshActive();
    } else if (s.state === 'error') {
      toast('运行失败：' + (s.error || ('exit ' + s.returncode)), 'err');
    }
  }

  async function runOnce() {
    const token = getToken();
    if (!token) { toast('未提供 X-Token，已取消', 'err'); return; }
    const btn = $('#btn-run');
    btn.disabled = true;
    const res = await api('/api/run', { method: 'POST', headers: { 'X-Token': token } });
    btn.disabled = false;
    if (!res.ok) {
      if (res.error === 'unauthorized') sessionStorage.removeItem('apb_token');
      toast('触发失败：' + res.error + (res.hint ? ' — ' + res.hint : ''), 'err');
      return;
    }
    toast('已触发 python -m qcore.live --once，后台运行中…');
    pollRun(200);
  }

  /* ---------------------------- 页签切换 --------------------------- */

  const state = { tab: 'overview', selSid: null, ledgerFilter: 'all', ledger: null, busy: false };

  const LOADERS = { overview: loadOverview, signals: loadSignals, ledger: loadLedger, paper: loadPaper };

  function switchTab(name) {
    if (!LOADERS[name]) return;
    state.tab = name;
    $$('.tab').forEach((b) => {
      const on = b.getAttribute('data-tab') === name;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    $$('.panel').forEach((p) => { p.classList.toggle('is-active', p.id === 'panel-' + name); });
    LOADERS[name]();
  }

  function refreshActive() {
    if (state.busy) return;
    state.busy = true;
    Promise.resolve(LOADERS[state.tab]()).then(
      () => { state.busy = false; },
      () => { state.busy = false; }
    );
  }

  /* ------------------------------ 绑定 ----------------------------- */

  function bind() {
    $$('.tab').forEach((b) => b.addEventListener('click', () => switchTab(b.getAttribute('data-tab'))));
    $('#btn-refresh').addEventListener('click', () => { toast('刷新中…'); refreshActive(); loadBadges(); });
    $('#btn-run').addEventListener('click', runOnce);

    $('#topn').addEventListener('change', () => { state.selSid = null; loadSignals(); });
    $('#ledger-days').addEventListener('change', loadLedger);

    $('#ledger-filter').addEventListener('click', (ev) => {
      const btn = ev.target.closest('.seg');
      if (!btn) return;
      state.ledgerFilter = btn.getAttribute('data-f');
      $$('#ledger-filter .seg').forEach((b) => b.classList.toggle('is-active', b === btn));
      if (state.ledger) renderLedger(state.ledger);
    });

    ['#bars-top', '#bars-bottom'].forEach((sel) => {
      $(sel).addEventListener('click', (ev) => {
        const row = ev.target.closest('.bar-row');
        if (!row) return;
        state.selSid = row.getAttribute('data-sid');
        loadSignals();
      });
    });

    $('#trade-form').addEventListener('submit', submitTrade);

    const dateInput = $('#trade-form').elements.date;
    if (dateInput && !dateInput.value) {
      const d = new Date();
      const pad = (x) => String(x).padStart(2, '0');
      dateInput.value = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
    }

    let rt = null;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => {
        if (state.tab === 'overview' || state.tab === 'paper') refreshActive();
      }, 250);
    });

    setInterval(() => { if (!document.hidden) refreshActive(); }, 90000);
  }

  function boot() {
    bind();
    renderSignalDetail(null);
    loadOverview();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
