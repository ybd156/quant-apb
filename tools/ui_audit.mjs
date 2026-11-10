/**
 * 无头浏览器界面审计 —— 让「界面上数字自相矛盾」由脚本发现，而不是由用户发现。
 *
 * 用法：
 *   node tools/ui_audit.mjs                       # 默认 http://127.0.0.1:8766
 *   node tools/ui_audit.mjs --port 8765           # 指定端口
 *   node tools/ui_audit.mjs --page paper          # 只审一页
 *
 * 它做三件事：
 *   1. 用 Chrome DevTools Protocol 打开每个页面，等渲染完成
 *   2. 抓「可见文本 + 关键数值」做**跨字段一致性断言**（不是截图给人看）
 *   3. 有任何一条不成立 → 退出码 1
 *
 * 为什么必须这样：本项目的 bug 大多不是算法错，是「同一件事在两处写法不同」。
 * 静态读代码看不见，肉眼看截图会漏，只有把渲染后的 DOM 拿出来比才可靠。
 */
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const argOf = (k, d) => {
  const i = args.indexOf('--' + k);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};
const PORT = Number(argOf('port', 8766));
const ONLY = argOf('page', '');
const BASE = `http://127.0.0.1:${PORT}`;
const CHROME = process.env.CHROME_BIN
  || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = path.resolve('tmp/chrome-audit');
const SHOTS = path.resolve('tmp/shots');

const PAGES = ['overview', 'market', 'signals', 'model', 'ledger', 'paper', 'settings'];

let pass = 0, fail = 0;
const problems = [];
function check(name, ok, detail = '') {
  if (ok) { pass++; console.log(`  [PASS] ${name}` + (detail ? ` :: ${detail}` : '')); }
  else {
    fail++; console.log(`  [FAIL] ${name}` + (detail ? ` :: ${detail}` : ''));
    problems.push({ name, detail });
  }
}

/** 极简 CDP 客户端（Node 内置 WebSocket，不引依赖） */
class CDP {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.id = 0;
    this.waiting = new Map();
    this.ready = new Promise((res, rej) => {
      this.ws.addEventListener('open', () => res());
      this.ws.addEventListener('error', (e) => rej(e));
    });
    this.ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.id && this.waiting.has(msg.id)) {
        const { resolve, reject } = this.waiting.get(msg.id);
        this.waiting.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      setTimeout(() => {
        if (this.waiting.has(id)) { this.waiting.delete(id); reject(new Error('cdp timeout: ' + method)); }
      }, 30000);
    });
  }
  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

async function getWsUrl(port) {
  for (let i = 0; i < 40; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json/list`);
      const tabs = await r.json();
      const page = tabs.find((t) => t.type === 'page');
      if (page && page.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch { /* retry */ }
    await sleep(250);
  }
  throw new Error('Chrome CDP 未就绪');
}

/** 在页面里执行一段表达式并返回值 */
async function evaluate(cdp, expr) {
  const r = await cdp.send('Runtime.evaluate', {
    expression: expr, returnByValue: true, awaitPromise: true,
  });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text || 'eval error');
  return r.result?.value;
}

/** 等待渲染：轮询「body 里有内容且不再变化」 */
async function waitRendered(cdp) {
  let last = '', stable = 0;
  for (let i = 0; i < 60; i++) {
    const n = await evaluate(cdp, 'document.body ? document.body.innerText.length : 0');
    if (n > 200 && n === last) { stable++; if (stable >= 3) return; }
    else { stable = 0; last = n; }
    await sleep(200);
  }
}

const textOf = (cdp) => evaluate(cdp, 'document.body.innerText');
const numsIn = (s) => (s.match(/-?\d[\d,]*\.?\d*/g) || [])
  .map((x) => Number(x.replace(/,/g, ''))).filter((x) => Number.isFinite(x));

async function main() {
  // 每次用全新 profile：静态资源缓存会让审计对着上一版界面报结论
  // （实测：改了 app.js 但浏览器用旧文件，审计连续三次给出相反结果）。
  try { fs.rmSync(PROFILE, { recursive: true, force: true }); } catch { /* ignore */ }
  fs.mkdirSync(PROFILE, { recursive: true });
  fs.mkdirSync(SHOTS, { recursive: true });
  const dbgPort = 9334;
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
    '--no-first-run', '--disable-extensions',
    `--remote-debugging-port=${dbgPort}`,
    `--user-data-dir=${PROFILE}`,
    '--window-size=1600,2000', 'about:blank',
  ], { stdio: 'ignore' });

  let cdp;
  try {
    const wsUrl = await getWsUrl(dbgPort);
    cdp = new CDP(wsUrl);
    await cdp.ready;
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    // 必须关缓存：静态资源带缓存头，改了 app.js 但浏览器还用旧的，
    // 审计就会对着上一版界面报「通过」或「失败」——两边都不可信。
    try {
      await cdp.send('Network.enable');
      await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
    } catch { /* 老版本 Chrome 没有这个域，忽略 */ }

    for (const page of PAGES) {
      if (ONLY && page !== ONLY) continue;
      console.log(`\n=== ${page} ===`);
      await cdp.send('Page.navigate', { url: `${BASE}/#/${page}` });
      await sleep(1500);
      await waitRendered(cdp);
      const txt = await textOf(cdp);

      // 每页都要成立的通则
      check('页面有内容（不是空白/报错页）', txt.length > 300, `${txt.length} 字符`);
      check('没有 JS 报错文案', !/undefined|NaN|\[object Object\]/.test(txt),
        (txt.match(/undefined|NaN|\[object Object\]/g) || []).slice(0, 3).join(','));

      if (page === 'paper') {
        // ① 各周期净值 vs 顶部市场本金
        const mktCap = numsIn((txt.match(/A股\s*[\d.]+万/) || [''])[0]);
        const rowCaps = [...txt.matchAll(/(T\+1 短线|T\+3 中短线|1 周 中线|2 周|15 天 中长线)\s+在跑\s+([\d,]+(?:\.\d+)?)/g)]
          .map((m) => ({ name: m[1], nav: Number(m[2].replace(/,/g, '')) }));
        check('能读到各周期账户行', rowCaps.length >= 3, `${rowCaps.length} 行`);
        // ② 成交表必须显示时刻。判据要**只看成交表**，不能在整页里找时间字符串 ——
        //    上一版就是这么写的，结果「数据截至 2026-11-06」让检查假通过（陷阱 19）。
        const tradeRow = await evaluate(cdp, `(() => {
          const tbl = [...document.querySelectorAll('table.tbl')].find(
            (t) => /成交时刻/.test(t.innerText));
          if (!tbl) return null;
          const rows = [...tbl.querySelectorAll('tbody tr')];
          if (!rows.length) return { header: true, n: 0, withTime: 0 };
          // 用 textContent 而不是 innerText：innerText 受布局影响，
          // 元素在折叠容器里会返回空串 —— 那会让「有时刻」被误判成「没时刻」。
          const withTime = rows.filter((r) => (r.textContent || '').indexOf(':') >= 0).length;
          const first = (rows[0] && rows[0].textContent || '').slice(0, 120);
          return { header: true, n: rows.length, withTime, first };
        })()`);
        check('成交表有「成交时刻」列', Boolean(tradeRow && tradeRow.header),
          tradeRow ? '' : '表头没有「成交时刻」');
        check('成交行确实带时刻（不是只有日期）',
          Boolean(tradeRow) && (tradeRow.n === 0 || tradeRow.withTime === tradeRow.n),
          tradeRow ? `${tradeRow.withTime}/${tradeRow.n} 行带时刻 | 首行: ${tradeRow.first}` : '');
        // ③ 「天」当「交易日」用
        const dayMisuse = /陈旧\s*\d+\s*天/.test(txt);
        check('陈旧度单位是「交易日」不是「天」', !dayMisuse,
          dayMisuse ? '把交易日数写成「天」' : '');
        // ④ 模拟盘不许说成真实下单
        const impliesReal = /真在记账|自动下单（不是回测/.test(txt);
        check('模拟盘不暗示真实下单', !impliesReal,
          impliesReal ? '文案暗示真实委托' : '');
        // ⑤ 各周期净值应等于该周期配置本金（未成交时）
        const cfgCap = await evaluate(cdp,
          `(async()=>{try{const r=await fetch('/api/account/config?market=CN');` +
          `const j=await r.json();const d=j.data||{};return {init:d.initial_capital,` +
          `hc:d.horizon_capital||null};}catch(e){return null}})()`);
        if (cfgCap && cfgCap.init) {
          // 判据不是「数字必须等于配置」，而是「不一致必须被标出来」。
          // 历史账户不能回改（INV-7），所以界面的责任是让人看见差异。
          const flat = rowCaps.filter((r) => r.nav === 1000000);
          const flagged = /本金不符/.test(txt);
          check('各周期本金与配置不符时已标注', flat.length === 0 || flagged,
            flat.length && !flagged
              ? `${flat.map((r) => r.name).join('/')} = 1,000,000 且页面无「本金不符」标记`
              : (flagged ? '页面已标「本金不符」' : ''));
        }
      }

      if (page === 'settings') {
        // 版本与更新：远端最新必须是一个合法 semver 且不早于当前
        const m = txt.match(/远端最新\s*v?(\d+\.\d+\.\d+)/);
        const cur = (txt.match(/当前版本\s*v?(\d+\.\d+\.\d+)/) || [])[1];
        check('能读到当前版本', Boolean(cur), cur || '');
        // 缓存里可能是过期/占位值；只要页面如实标「已忽略/未知」就算诚实。
        const honest = /已忽略|未知|非法版本号/.test(txt);
        check('远端最新要么是真版本，要么被标为忽略/未知',
          (Boolean(m) && m[1] !== '0.0.5') || honest,
          m ? m[1] + (honest ? '（已标注）' : '（未标注，会被误读为最新）') : '未显示');
      }

      if (page === 'model') {
        // 选中 T+1 时，回测摘要的持有期不能是 20 日
        const tabH1 = /T\+1 短线/.test(txt);
        const hold20 = /回测持有期[\s\S]{0,40}?20\s*日/.test(txt);
        const annotated = /与当前所选/.test(txt);
        check('持有期与所选周期一致（或已标注不一致）',
          !(tabH1 && hold20) || annotated,
          tabH1 && hold20 && !annotated ? '选中 T+1 却显示持有期 20 日且无说明' : '');
        // 在线学习：样本 0 的周期必须给出原因
        const zeroRows = (txt.match(/样本\s*0/g) || []).length;
        check('样本为 0 时有解释（不是干巴巴的 0）',
          zeroRows === 0 || /未生成|无台账|待生成/.test(txt),
          zeroRows ? `${zeroRows} 行样本 0` : '');
      }

      const shot = path.join(SHOTS, `audit_${page}.png`);
      const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(shot, Buffer.from(data, 'base64'));
    }
  } finally {
    if (cdp) cdp.close();
    chrome.kill();
  }

  console.log(`\nRESULT: ${pass} passed, ${fail} failed`);
  if (fail) {
    console.log('\n发现的问题：');
    for (const p of problems) console.log(`  · ${p.name}${p.detail ? ' — ' + p.detail : ''}`);
  }
  process.exit(fail ? 1 : 0);
}

main().catch((e) => { console.error('audit error:', e); process.exit(2); });
