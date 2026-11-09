// Orb neuron explorer: the page logic. Data comes from data/*.json and data/fires.bin (built by build/*);
// a local server (server.py) adds live Orb runs, pushes, saved names and the Claude tab.
// Panel layout and interaction after lyramakesmusic/neuron-explorer (MIT).
import { createMap } from "./map3d.js?v=1";
import { createCrystal, elementColour } from "./crystal3d.js?v=1";

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const DATA = "data/";
const D = { arch: null, crystals: null, all: null, layout: null, notes: {}, units: {}, links: {}, fires: null, names: {} };
const state = { layer: 4, focus: null, layoutMode: "writes", colour: "kurt", web: true, webZ: 6, qMode: "lens", sort: "kurt",
  server: false, live: false, claude: false, xtal: { c: null }, results: null, runC: null, chat: null, busy: false };
const fmt = (v, d = 2) => v == null || Number.isNaN(v) ? "—" : (+v).toFixed(d);
const pct = v => v == null ? "—" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

async function getJSON(name) { const r = await fetch(DATA + name); if (!r.ok) throw new Error(`${name}: ${r.status}`); return r.json(); }
async function units(l) { const k = D.arch.layers[l].key; if (!D.units[k]) D.units[k] = await getJSON(`units_${k}.json`); return D.units[k]; }
async function links(l) { const k = D.arch.layers[l].key; if (!D.links[k]) D.links[k] = await getJSON(`links_${k}.json`); return D.links[k]; }
async function fires() {
  if (D.fires) return D.fires;
  const prev = $("#status").textContent; setStatus("loading fires index…");
  const want = D.arch.n_atoms_all * L() * K() * 2 * 2;  // bytes: uint16 [atoms, layers, k, 2]
  let buf = null;
  try { const r = await fetch(DATA + "fires.bin"); if (r.ok) { const b = await r.arrayBuffer(); if (b.byteLength === want) buf = b; } } catch {}
  if (!buf) {  // a host that serves no binary files gets the same bytes base64-encoded in a json file
    const j = await getJSON("fires.b64.json"), s = atob(j.b64); buf = new ArrayBuffer(s.length); const u = new Uint8Array(buf);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  }
  D.fires = new Uint16Array(buf); setStatus(prev); return D.fires;
}
const L = () => D.arch.layers.length, N = () => D.arch.N, K = () => D.arch.fires_k;
const lay = l => D.arch.layers[l];
const short = l => { const y = lay(l); return y.block === 0 ? `Enc L${y.h}` : `B${y.block} L${y.h}`; };
const nid = (l, n) => `${short(l)} #${n}`;
const key = (l, n) => `${lay(l).key}/${n}`;
const nameOf = (l, n) => D.names[key(l, n)] || "";
const autoName = (l, n) => D.all[lay(l).key].name[n];
const shown = (l, n) => nameOf(l, n) || autoName(l, n);
function setStatus(t) { $("#status").textContent = t; }

// ---------- map ----------
let map, xtal;
function levels() {
  const out = new Float32Array(L() * N()), dead = new Uint8Array(L() * N()), named = new Uint8Array(L() * N());
  for (let l = 0; l < L(); l++) {
    const a = D.all[lay(l).key], base = l * N();
    const vals = new Float64Array(N()); let have = [];
    for (let n = 0; n < N(); n++) {
      if (!a.live[n]) { dead[base + n] = 1; continue; }
      if (nameOf(l, n) || D.notes[key(l, n)]) named[base + n] = 1;
      let v = 0;
      if (state.colour === "kurt") v = a.kurt[n] ?? -1e9;
      else if (state.colour === "on") v = a.on[n] ?? 0;
      vals[n] = v; have.push(n);
    }
    have.sort((x, y) => vals[x] - vals[y]);
    have.forEach((n, i) => { const r = i / Math.max(1, have.length - 1); out[base + n] = Math.pow(r, 5); });
  }
  return { out, dead, named };
}
function recolour() { const { out, dead, named } = levels(); map.setColours(out, dead, named, state.layer); }

// ---------- lists ----------
function sortVal(u, k) {
  if (k === "n") return -u.n;
  if (k === "best") return u.best ? Math.abs(u.best[1]) : -1;
  if (k === "kurt") return u.kurt ?? -1e9;
  return u[k] ?? -1e9;
}
function sortText(u, k) {
  if (k === "n") return ""; if (k === "kurt") return fmt(u.kurt, 1); if (k === "max") return fmt(u.max, 2); if (k === "on") return pct(u.on);
  if (k === "eta2") return fmt(u.eta2, 2); if (k === "best") return u.best ? `${u.best[0]} ${fmt(u.best[1], 2)}` : "—";
  return "";
}
async function renderList() {
  if (state.results) return renderResults();
  const us = await units(state.layer), q = $("#filter").value.trim().toLowerCase(), k = state.sort;
  const rows = us.filter(u => !q || shown(state.layer, u.n).toLowerCase().includes(q) || String(u.n) === q.replace("#", ""))
    .sort((a, b) => (b.live - a.live) || (sortVal(b, k) - sortVal(a, k)));
  $("#rows").innerHTML = rows.slice(0, 1100).map(u => `<li data-n="${u.n}" class="${u.live ? "" : "dead"}${state.focus && state.focus.layer === state.layer && state.focus.n === u.n ? " sel" : ""}">
    <span class="n">#${u.n}</span><span class="z">${sortText(u, k)}</span><span class="w">${nameOf(state.layer, u.n) ? `<b>${esc(nameOf(state.layer, u.n))}</b> ` : ""}${esc(u.name)}</span></li>`).join("");
}
function renderResults() {
  const r = state.results;
  $("#rows").innerHTML = `<li class="head">${r.length} neurons match “${esc(r.q)}”</li>` + r.slice(0, 400).map(x => `<li class="res" data-l="${x.l}" data-n="${x.n}">
    <span class="n">${short(x.l)} #${x.n}</span><span class="z">${x.score}</span><span class="w">${nameOf(x.l, x.n) ? `<b>${esc(nameOf(x.l, x.n))}</b> ` : ""}${esc(autoName(x.l, x.n))}</span></li>`).join("");
}
function runSearch() {
  const q = $("#q").value.trim(); if (!q) { state.results = null; renderList(); return; }
  const ql = q.toLowerCase(), res = [], isEl = /^[A-Z][a-z]?$/.test(q);
  for (let l = 0; l < L(); l++) {
    const a = D.all[lay(l).key];
    for (let n = 0; n < N(); n++) {
      if (!a.live[n]) continue;
      const nm = (D.names[key(l, n)] ? D.names[key(l, n)] + " " : "") + a.name[n];
      const lo = nm.toLowerCase(); let s = 0;
      if (isEl) { const m = nm.match(new RegExp(`(^|[ ,;])${q}(?=[ ,;]|$)`)); if (m) s = nm.indexOf(q) < nm.indexOf(";") ? 3 : 1; const f = nm.match(new RegExp(`\\b${q}(?=[A-Z0-9(]|\\b)`, "g")); if (f && !s) s = 1; }
      else if (lo.includes(ql)) s = lo.indexOf(ql) < (lo.indexOf(";") < 0 ? 999 : lo.indexOf(";")) ? 2 : 1;
      if (D.names[key(l, n)] && D.names[key(l, n)].toLowerCase().includes(ql)) s = 4;
      if (s) res.push({ l, n, score: s, k: a.kurt[n] ?? 0 });
    }
  }
  res.sort((a, b) => b.score - a.score || b.k - a.k); res.q = q; state.results = res; renderResults();
}
$("#q").addEventListener("input", () => { if (state.qMode === "lens") runSearch(); else runHits($("#q").value, $("#run-hits")); });
$("#q").addEventListener("keydown", e => { if (e.key === "Escape") { e.target.value = ""; state.results = null; renderList(); } });
$("#filter").addEventListener("input", renderList);
$("#sort").addEventListener("change", e => { state.sort = e.target.value; renderList(); });
for (const ol of [$("#rows"), $("#run-rows")]) {
  ol.addEventListener("click", e => { const li = e.target.closest("li"); if (!li || li.dataset.n == null) return; openNeuron(li.dataset.l != null ? +li.dataset.l : state.layer, +li.dataset.n, { crystal: li.dataset.c != null ? +li.dataset.c : undefined }); });
  ol.addEventListener("pointerover", e => { const li = e.target.closest("li"); if (li && li.dataset.n != null) map.hover(li.dataset.l != null ? +li.dataset.l : state.layer, +li.dataset.n); });
  ol.addEventListener("pointerleave", () => map.hover(null));
}

// ---------- crystals: pick one, see what fires on its atoms ----------
function findCrystals(q) {
  q = q.trim().toLowerCase(); if (!q) return [];
  const out = [];
  D.crystals.forEach((c, i) => { const f = c.f.toLowerCase(); if (c.id === q || f === q) out.unshift({ i, s: 3 }); else if (f.startsWith(q) || c.id.startsWith(q)) out.push({ i, s: 2 }); else if (f.includes(q)) out.push({ i, s: 1 }); });
  return out.sort((a, b) => b.s - a.s).slice(0, 30);
}
function runHits(q, box) {
  const hits = findCrystals(q);
  box.innerHTML = hits.map(h => { const c = D.crystals[h.i]; return `<div data-c="${h.i}"><b>${esc(c.f)}</b><span class="id">${esc(c.id)} · ${esc(c.sg || "")}</span>${c.x ? `<span class="id">extra</span>` : ""}</div>`; }).join("");
  box.classList.toggle("on", hits.length > 0);
}
$("#run-q").addEventListener("input", e => runHits(e.target.value, $("#run-hits")));
$("#run-q").addEventListener("keydown", e => { if (e.key === "Enter") { const h = findCrystals(e.target.value)[0]; if (h) runCrystal(h.i); } });
$("#run-hits").addEventListener("click", e => { const d = e.target.closest("[data-c]"); if (d) runCrystal(+d.dataset.c); });
$("#run-layers").addEventListener("change", () => state.runC != null && renderRun());
$("#run-min").addEventListener("change", () => state.runC != null && renderRun());
document.addEventListener("click", e => { if (!e.target.closest("#run-head") && !e.target.closest("#q")) $("#run-hits").classList.remove("on"); });
async function runCrystal(ci) {
  state.runC = ci; $("#run-hits").classList.remove("on"); showLeft("run");
  const c = D.crystals[ci]; $("#run-q").value = c.f;
  $("#run-info").innerHTML = `<b>${esc(c.f)}</b> ${esc(c.id)} · ${esc(c.sg || "")} · ${c.ns} atoms${c.x ? " · outside the main set (shown, never pooled)" : ""}`;
  await fires(); renderRun();
  if (!state.focus) showCrystal(ci, null, 1);
  history.replaceState(null, "", `#c-${c.id}`);
}
function firesOf(atom, l) { const F = D.fires, out = [], mx = D.all[lay(l).key].max; const o = (atom * L() + l) * K() * 2; for (let k = 0; k < K(); k++) { const n = F[o + k * 2]; if (mx[n] != null && mx[n] >= 0.1) out.push([n, F[o + k * 2 + 1] / 10]); } return out; }
function renderRun() {
  const c = D.crystals[state.runC], lf = $("#run-layers").value, zmin = +$("#run-min").value || 0, rows = [];
  for (let j = 0; j < c.ns; j++) for (let l = 0; l < L(); l++) {
    if (lf !== "all" && !lay(l).key.startsWith(lf)) continue;
    for (const [n, z] of firesOf(c.a0 + j, l)) if (z >= zmin) rows.push({ l, n, z, j });
  }
  rows.sort((a, b) => b.z - a.z);
  $("#run-rows").innerHTML = `<li class="head">${rows.length} neuron-atom pairs at z ≥ ${zmin} on ${esc(c.f)} (z: activation in units of the neuron's spread over the ${D.arch.n_crystals}-crystal set; only each atom's top ${K()} per layer are indexed)</li>` +
    rows.slice(0, 600).map(r => `<li class="res" data-l="${r.l}" data-n="${r.n}" data-c="${state.runC}"><span class="n">${short(r.l)} #${r.n}</span><span class="sg up">${r.z.toFixed(1)}</span><span class="w"><span class="tk">${esc(c.sites[r.j][0])} #${r.j}</span>${esc(shown(r.l, r.n))}</span></li>`).join("");
}
// activations of neuron (l, n) on crystal ci: exact from the neuron's top crystals or the server, else rebuilt from the fires index (z to 0.1)
async function actsOn(l, n, ci) {
  const u = (await units(l))[n], c = D.crystals[ci];
  const t = u.tc && u.tc.find(x => x[0] === ci); if (t) return { acts: t[1], exact: true };
  if (state.server) { try { const r = await fetch(`api/acts?layer=${l}&n=${n}&c=${ci}`); if (r.ok) return { acts: (await r.json()).acts, exact: true }; } catch {} }
  await fires(); const acts = [];
  for (let j = 0; j < c.ns; j++) { const hit = firesOf(c.a0 + j, l).find(x => x[0] === n); acts.push(hit ? u.mean + hit[1] * u.sd : null); }
  return { acts, exact: false };
}

// ---------- neuron panel ----------
const focusEl = $("#focus");
function isFocus(l, n) { return state.focus && state.focus.layer === l && state.focus.n === n; }
async function openNeuron(l, n, opts = {}) {
  state.focus = { layer: l, n }; saveFocus();
  if (map.layerMode !== null && map.layerMode !== l) map.enterLayer(l);
  focusEl.classList.add("on"); $("#list").classList.add("has-focus"); setFocusCollapsed(false);
  $("#f-id").textContent = nid(l, n); $("#f-meta").textContent = ""; setNameField(nameOf(l, n)); $("#f-cov").textContent = "";
  $("#tab-examples").innerHTML = ""; $("#tab-details").innerHTML = "";
  if (l !== state.layer) await setLayer(l, false);
  if (!isFocus(l, n)) return;
  map.select(l, n, opts.fly !== false); renderList(); renderAgentActions();
  const u = (await units(l))[n]; if (!isFocus(l, n)) return;
  $("#f-meta").textContent = u.live ? `on ${pct(u.on)} · max ${fmt(u.max, 2)} · selective ${fmt(u.kurt, 1)}` : "dead";
  $("#f-name").placeholder = u.name;
  $("#f-cov").innerHTML = u.live && u.top.length ? `label from the ${u.top.length} strongest atoms: the named elements cover ${Math.round(u.cov * 100)}% of them · element η² ${fmt(u.eta2, 2)} (beats ${fmt(u.eta2p, 0)}% of random directions)` : "";
  $("#net").innerHTML = netHTML(l, n, u);
  renderExamples(l, n, u); renderDetails(l, n, u);
  const ci = opts.crystal ?? (u.tc.length ? u.tc[0][0] : null);
  if (ci != null) { const { acts, exact } = await actsOn(l, n, ci); if (isFocus(l, n)) showCrystal(ci, acts, Math.max(u.p999, 1e-3), exact); }
  if (opts.crystal != null) setTab("crystal");
  await refreshWeb(); await loadLinksPanel(l, n, u);
}
function closeNeuron() { state.focus = null; saveFocus(); focusEl.classList.remove("on"); $("#list").classList.remove("has-focus"); $("#net").innerHTML = ""; map.select(null); map.setWeb([]); renderList(); syncEdges(); }
$("#f-close").addEventListener("click", closeNeuron);
function chipsN(list, side) {
  if (!list || !list.length) return `<span class="dim">none</span>`;
  return list.map(([ol, on, z, s]) => `<span class="chip ${s > 0 ? "up" : "down"}" data-l="${ol}" data-n="${on}" title="${esc(shown(ol, on))}">${short(ol)} #${on}<i>${s > 0 ? "+" : "−"}${z.toFixed(1)}</i></span>`).join("");
}
function netHTML(l, n, u) {
  if (!u.live) return `<div class="note">This unit is constant on every atom of the set (dead). ${u.max ? `Its value is ${fmt(u.max, 3)}.` : ""}</div>`;
  const cr = i => D.crystals[i];
  let h = "";
  const note = D.notes[key(l, n)];
  if (note) h += `<div class="wsum note"><div class="lab">note</div>${note.text} <span class="dim">(${esc(note.source)})</span></div>`;
  h += `<div class="wsum"><div class="lab">strongest on</div><div class="chips">${u.els.map(([e, c]) => `<span class="chip" data-q="${e}" style="border-left:2px solid ${elementColour(e)}">${e}<i>${c}</i></span>`).join("")}</div></div>`;
  h += `<div id="web-sum"><div class="wsum"><div class="lab"><span class="sw up"></span>fed by</div><div class="chips dim">loading…</div></div></div>`;
  return h;
}
async function loadLinksPanel(l, n, u) {
  const lk = await links(l); if (!isFocus(l, n)) return;
  const up = lk.up[n] || [], dn = lk.down[n] || [];
  const up_pos = up.filter(x => x[3] > 0), up_neg = up.filter(x => x[3] < 0), dn_pos = dn.filter(x => x[3] > 0), dn_neg = dn.filter(x => x[3] < 0);
  const title = lay(l).h === 1 ? (lay(l).block === 0 ? "the encoder's first layer reads the element embedding only" : "reads the residual stream (node slice) written by earlier layer-2 units; message slices not shown") : "reads this block's layer-1 units";
  $("#web-sum").innerHTML = `<div class="wsum"><div class="lab">web · direct weight paths · ${esc(title)}</div>
    <div class="lrow"><span class="sgn up">+</span><div class="chips">${chipsN(up_pos)}</div></div><div class="lrow"><span class="sgn down">−</span><div class="chips">${chipsN(up_neg)}</div></div>
    <div class="lab" style="margin-top:8px"><span class="sw down"></span>feeds ${lay(l).key === "L5_h2" ? "the energy head (not drawn)" : ""}</div>
    <div class="lrow"><span class="sgn up">+</span><div class="chips">${chipsN(dn_pos)}</div></div><div class="lrow"><span class="sgn down">−</span><div class="chips">${chipsN(dn_neg)}</div></div></div>`;
  map.setLinks([...up.slice(0, state.webZ).map(x => ({ layer: x[0], n: x[1], z: x[2], sign: x[3] })), ...dn.slice(0, state.webZ).map(x => ({ layer: x[0], n: x[1], z: x[2], sign: x[3] }))]);
}
$("#net").addEventListener("click", e => {
  const ch = e.target.closest(".chip"); if (!ch) return;
  if (ch.dataset.l != null) return openNeuron(+ch.dataset.l, +ch.dataset.n);
  if (ch.dataset.c != null) { const f = state.focus; return actsOn(f.layer, f.n, +ch.dataset.c).then(({ acts, exact }) => units(f.layer).then(us => { showCrystal(+ch.dataset.c, acts, Math.max(us[f.n].p999, 1e-3), exact); setTab("crystal"); })); }
  if (ch.dataset.q) { $("#q").value = ch.dataset.q; setQMode("lens"); runSearch(); }
});
$("#net").addEventListener("pointerover", e => { const ch = e.target.closest(".chip[data-l]"); if (ch) map.hover(+ch.dataset.l, +ch.dataset.n); });
$("#net").addEventListener("pointerleave", () => map.hover(null));
async function refreshWeb() {
  const f = state.focus;
  if (!f || !state.web) { map.setWeb([]); return; }
  const edges = [], k1 = state.webZ, k2 = 3;
  const lk = await links(f.layer); if (!isFocus(f.layer, f.n)) return;
  const up = (lk.up[f.n] || []).slice(0, k1), dn = (lk.down[f.n] || []).slice(0, k1);
  for (const [ol, on, z, s] of up) edges.push([ol, on, f.layer, f.n, s, z, 1]);
  for (const [ol, on, z, s] of dn) edges.push([f.layer, f.n, ol, on, s, z, 1]);
  for (const [ol, on] of up.slice(0, k2)) { const l2 = await links(ol); for (const [pl, pn, z, s] of (l2.up[on] || []).slice(0, k2)) edges.push([pl, pn, ol, on, s, z, 2]); }
  for (const [ol, on] of dn.slice(0, k2)) { const l2 = await links(ol); for (const [pl, pn, z, s] of (l2.down[on] || []).slice(0, k2)) edges.push([ol, on, pl, pn, s, z, 2]); }
  if (isFocus(f.layer, f.n)) map.setWeb(edges);
}

function renderExamples(l, n, u) {
  if (!u.live || !u.tc.length) { $("#tab-examples").innerHTML = `<div class="note">No strong atoms.</div>`; return; }
  const scale = Math.max(u.p999, 1e-3);
  const topSet = new Map(u.top.map(([i, v]) => [i, v]));
  const h = u.tc.map(([ci, acts], rank) => {
    const c = D.crystals[ci], mx = Math.max(...acts), site = acts.indexOf(mx);
    const atoms = acts.map((a, j) => { const w = Math.max(0, Math.min(1, a / scale)); return `<span style="background:rgba(255,181,71,${(0.08 + 0.7 * w).toFixed(2)});color:${w > 0.5 ? "#111" : "inherit"}">${esc(c.sites[j][0])} ${a.toFixed(2)}</span>`; }).join("");
    const cn = c.cn ? c.cn[site] : null, vd = c.void ? c.void[site] : null;
    return `<div class="ex" data-c="${ci}"><div class="h"><b>${esc(c.f)}</b><span class="id">${esc(c.id)} · ${esc(c.sg || "")}</span>${c.x ? `<span class="id">extra</span>` : ""}<span class="v">${mx.toFixed(2)}</span></div>
      <div class="atoms">${atoms}</div><div class="meta">peak on ${esc(c.sites[site][0])} #${site}${cn != null ? ` · ${cn} neighbours` : ""}${vd != null ? ` · void ${vd.toFixed(2)} Å` : ""}${c.gap != null ? ` · gap ${fmt(c.gap, 2)} eV` : ""}${c.eh != null ? ` · E<sub>hull</sub> ${fmt(c.eh, 3)}` : ""}${c.e_orb != null ? ` · Orb energy ${fmt(c.e_orb, 3)} eV/atom` : ""}</div></div>`;
  }).join("");
  $("#tab-examples").innerHTML = `<div class="note" style="margin-bottom:10px">The ${u.top.length} strongest atoms in the ${D.arch.n_crystals}-crystal set (symmetry copies merged), grouped by crystal; every atom of each crystal is shaded by this neuron's activation (full shade at the 99.9th percentile, ${scale.toFixed(2)}). Click a crystal to see it in 3D.</div>` + h;
}
$("#tab-examples").addEventListener("click", async e => { const x = e.target.closest(".ex"); if (!x || !state.focus) return; const f = state.focus, u = (await units(f.layer))[f.n]; const { acts, exact } = await actsOn(f.layer, f.n, +x.dataset.c); showCrystal(+x.dataset.c, acts, Math.max(u.p999, 1e-3), exact); setTab("crystal"); });

function renderDetails(l, n, u) {
  const y = lay(l), lab = D.arch.labels;
  let h = `<h3>${nid(l, n)} · ${esc(y.name)}</h3><table class="dtable"><tr><th>max</th><th>99.9th pct</th><th>on (&gt; 0.1)</th><th>mean</th><th>sd</th><th>excess kurtosis</th></tr>
    <tr><td class="num">${fmt(u.max, 3)}</td><td class="num">${fmt(u.p999, 3)}</td><td class="num">${pct(u.on)}</td><td class="num">${fmt(u.mean, 3)}</td><td class="num">${fmt(u.sd, 3)}</td><td class="num">${fmt(u.kurt, 1)}</td></tr></table>`;
  if (!u.live) { $("#tab-details").innerHTML = h; return; }
  h += `<h3>What the activation lines up with</h3><div class="note" style="margin-bottom:6px">Spearman correlation with each atom fact, both centred within element, over the ${D.arch.n_atoms.toLocaleString()} atoms; the percentile says how many of ${D.arch.random_dirs || 256} random directions in this layer it beats.</div>
    <table class="dtable"><tr><th>fact</th><th class="num">ρ</th><th class="num">beats random</th></tr>
    <tr><td>element (η², share of variance)</td><td class="num">${fmt(u.eta2, 3)}</td><td class="num">${fmt(u.eta2p, 0)}%</td></tr>` +
    Object.entries(u.lab || {}).sort((a, b) => Math.abs(b[1][0]) - Math.abs(a[1][0])).map(([k, [r, p]]) => `<tr><td>${esc(lab[k] || k)}</td><td class="num">${fmt(r, 3)}</td><td class="num">${p == null ? "—" : fmt(p, 0) + "%"}</td></tr>`).join("") + `</table>`;
  h += `<h3>Direct links</h3><div id="d-links" class="note">loading…</div>`;
  $("#tab-details").innerHTML = h;
  links(l).then(lk => { if (!isFocus(l, n)) return;
    const row = (x, side) => `<tr class="con" data-l="${x[0]}" data-n="${x[1]}"><td>${short(x[0])} #${x[1]}</td><td class="num ${x[3] > 0 ? "up" : "down"}">${x[3] > 0 ? "+" : "−"}${x[2].toFixed(1)}</td><td>${esc(shown(x[0], x[1]))}</td></tr>`;
    $("#d-links").innerHTML = `<div class="note" style="margin-bottom:6px">|weight| as a z-score among all units on the other side; sign = the direction of the push. Within a block: layer 1 → layer 2 weights. Across blocks: layer-2 write direction (RMSNorm gain × output column) · layer-1 read row on the node slice.</div>
      <table class="dtable"><tr><th colspan="3">fed by</th></tr>${(lk.up[n] || []).map(x => row(x, "up")).join("") || "<tr><td colspan=3 class=dim>none</td></tr>"}</table>
      <table class="dtable"><tr><th colspan="3">feeds</th></tr>${(lk.down[n] || []).map(x => row(x, "down")).join("") || "<tr><td colspan=3 class=dim>none in the node MLPs</td></tr>"}</table>`; });
}
$("#tab-details").addEventListener("click", e => { const tr = e.target.closest("tr.con"); if (tr) openNeuron(+tr.dataset.l, +tr.dataset.n); });

// ---------- crystal tab ----------
function showCrystal(ci, acts, scale, exact = true) {
  const c = D.crystals[ci]; state.xtal = { c: ci, acts, scale };
  $("#xtal-title").textContent = `${c.f} · ${c.id} · ${c.sg || ""}`;
  $("#xtal-scale").textContent = acts ? (exact ? `shade: full at ${scale.toFixed(2)}` : `shade rebuilt from the fires index (atoms where this neuron is not in the top ${K()} stay grey)`) : "";
  xtal.show(c, acts, scale); markFresh("crystal");
  renderAtomRows(ci, acts);
}
async function renderAtomRows(ci, acts) {
  const c = D.crystals[ci]; await fires();
  const rows = c.sites.map((s, j) => {
    const best = [];
    for (let l = 0; l < L(); l++) for (const [n, z] of firesOf(c.a0 + j, l).slice(0, 3)) best.push([l, n, z]);
    best.sort((a, b) => b[2] - a[2]);
    const a = acts ? acts[j] : null, hot = a != null && state.xtal.scale && a / state.xtal.scale > 0.5;
    return `<div class="atomrow" data-j="${j}"><span class="a ${hot ? "hot" : ""}" style="color:${hot ? "" : elementColour(s[0])}">${esc(s[0])} #${j}${a != null ? `<br><span class="dim">${a.toFixed(2)}</span>` : ""}</span>
      <div class="chips">${best.slice(0, 6).map(([l, n, z]) => `<span class="chip" data-l="${l}" data-n="${n}" title="${esc(shown(l, n))}">${short(l)} #${n}<i>z ${z.toFixed(1)}</i></span>`).join("")}</div></div>`;
  }).join("");
  $("#xtal-atoms").innerHTML = `<div class="lab">what fires on each atom (top 6 across layers, by z)</div>` + rows;
}
$("#xtal-atoms").addEventListener("click", e => { const ch = e.target.closest(".chip[data-l]"); if (ch) openNeuron(+ch.dataset.l, +ch.dataset.n, { crystal: state.xtal.c, fly: true }); });
$("#xtal-atoms").addEventListener("pointerover", e => { const ch = e.target.closest(".chip[data-l]"); if (ch) map.hover(+ch.dataset.l, +ch.dataset.n); });
$$("#xtal-cell button").forEach(b => b.addEventListener("click", () => { $$("#xtal-cell button").forEach(x => x.classList.toggle("on", x === b)); xtal.setCell(+b.dataset.s); }));
$("#xtal-shade").addEventListener("change", e => xtal.setShade(e.target.checked));

// ---------- tabs, panels, header ----------
function setTab(t) { $$("#tabs button[data-tab]").forEach(b => b.classList.toggle("on", b.dataset.tab === t)); $$(".tab").forEach(x => x.classList.toggle("on", x.id === `tab-${t}`)); const b = $(`#tabs button[data-tab="${t}"]`); b && b.classList.remove("fresh"); if (t === "crystal") xtal.resize(); }
function markFresh(t) { const b = $(`#tabs button[data-tab="${t}"]`); if (b && !b.classList.contains("on")) b.classList.add("fresh"); }
$$("#tabs button[data-tab]").forEach(b => b.addEventListener("click", () => setTab(b.dataset.tab)));
function showLeft(t) { $$("#ltabs button[data-lt]").forEach(b => b.classList.toggle("on", b.dataset.lt === t)); $$(".lpane").forEach(x => x.classList.toggle("on", x.id === `lt-${t}`)); }
$$("#ltabs button[data-lt]").forEach(b => b.addEventListener("click", () => showLeft(b.dataset.lt)));
function setListHidden(h) { $("#list").classList.toggle("hidden", h); document.body.classList.toggle("list-hidden", h); syncEdges(); }
function setFocusCollapsed(c) { focusEl.classList.toggle("collapsed", c); syncEdges(); }
function syncEdges() {
  const lh = $("#list").classList.contains("hidden"), fc = focusEl.classList.contains("collapsed") && focusEl.classList.contains("on");
  $("#list-tab").classList.toggle("on", lh); $("#focus-tab").classList.toggle("on", fc); $("#focus-tab").textContent = state.focus ? `‹ ${nid(state.focus.layer, state.focus.n)}` : "‹";
  const fo = focusEl.classList.contains("on") && !focusEl.classList.contains("collapsed");
  map.setInsets(lh ? 44 : 404, fo ? focusEl.offsetWidth + 12 : 0);
}
$("#list-hide").addEventListener("click", () => setListHidden(true)); $("#list-tab").addEventListener("click", () => setListHidden(false));
$("#f-collapse").addEventListener("click", () => setFocusCollapsed(true)); $("#focus-tab").addEventListener("click", () => setFocusCollapsed(false));
$("#layer").addEventListener("change", e => setLayer(+e.target.value));
async function setLayer(l, outline = true) {
  state.layer = l; $("#layer").value = l; await units(l);
  if (outline) recolour(); else recolour();
  renderList();
}
$$("#layout-mode button").forEach(b => b.addEventListener("click", () => { $$("#layout-mode button").forEach(x => x.classList.toggle("on", x === b)); state.layoutMode = b.dataset.k; map.setLayout(state.layoutMode === "writes" ? D.arch.layers.map(y => D.layout[y.key]) : null); }));
$$("#colour-mode button").forEach(b => b.addEventListener("click", () => { $$("#colour-mode button").forEach(x => x.classList.toggle("on", x === b)); state.colour = b.dataset.c; recolour(); }));
$$("#web-mode button").forEach(b => b.addEventListener("click", () => { $$("#web-mode button").forEach(x => x.classList.toggle("on", x === b)); state.web = b.dataset.w === "on"; refreshWeb(); }));
$("#web-z").addEventListener("change", e => { state.webZ = Math.max(1, Math.min(16, +e.target.value || 6)); e.target.value = state.webZ; refreshWeb(); if (state.focus) units(state.focus.layer).then(us => loadLinksPanel(state.focus.layer, state.focus.n, us[state.focus.n])); });
function setQMode(m) { state.qMode = m; $$("#q-mode button").forEach(x => x.classList.toggle("on", x.dataset.q === m)); $("#q").placeholder = m === "lens" ? "search labels: O, Cu, NaCl…" : "crystal: formula or id"; if (m === "lens") { $("#run-hits").classList.remove("on"); runSearch(); } else { state.results = null; renderList(); runHits($("#q").value, $("#run-hits")); } }
$$("#q-mode button").forEach(b => b.addEventListener("click", () => setQMode(b.dataset.q)));
$("#q").addEventListener("keydown", e => { if (e.key === "Enter" && state.qMode === "fires") { const h = findCrystals(e.target.value)[0]; if (h) runCrystal(h.i); } });
function parseJump(t) {
  t = t.trim(); let m;
  if ((m = t.match(/^(?:enc|e)\s*l?\s*(\d)\s*#?\s*(\d+)$/i))) return [D.arch.layers.findIndex(y => y.block === 0 && y.h === +m[1]), +m[2]];
  if ((m = t.match(/^b\s*(\d)\s*l?\s*(\d)\s*#?\s*(\d+)$/i))) return [D.arch.layers.findIndex(y => y.block === +m[1] && y.h === +m[2]), +m[3]];
  if ((m = t.match(/^L(\d)_h(\d)\s*#?\s*(\d+)$/i))) return [D.arch.layers.findIndex(y => y.block === +m[1] && y.h === +m[2]), +m[3]];
  if ((m = t.match(/^(\d)\s*\/\s*(\d)\s*\/\s*(\d+)$/))) return [D.arch.layers.findIndex(y => y.block === +m[1] && y.h === +m[2]), +m[3]];
  if ((m = t.match(/^#?(\d+)$/))) return [state.layer, +m[1]];
  return null;
}
$("#jump").addEventListener("keydown", e => { if (e.key !== "Enter") return; const j = parseJump(e.target.value); if (j && j[0] >= 0 && j[1] >= 0 && j[1] < N()) openNeuron(j[0], j[1]); });
$("#layer-back").addEventListener("click", () => map.exitLayer());
addEventListener("keydown", e => { if (e.key === "Escape" && !e.target.matches("input,textarea")) { if (map.layerMode !== null) map.exitLayer(); else if ($("#about-box").classList.contains("on")) $("#about-box").classList.remove("on"); } });

// names: kept in this browser; the local server also keeps them in data/names.json
function setNameField(v) { const t = $("#f-name"); t.value = v; t.style.height = "auto"; t.style.height = `${t.scrollHeight}px`; }
$("#f-name").addEventListener("input", () => setNameField($("#f-name").value));
$("#f-name").addEventListener("change", async () => {
  const f = state.focus; if (!f) return; const v = $("#f-name").value.trim(), k = key(f.layer, f.n);
  if (v) D.names[k] = v; else delete D.names[k];
  try { localStorage.setItem("orbx.names", JSON.stringify(D.names)); } catch {}
  if (state.server) { try { await fetch("api/name", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ key: k, name: v }) }); } catch {} }
  $("#f-name").classList.add("saved"); setTimeout(() => $("#f-name").classList.remove("saved"), 900); recolour(); renderList();
});
$("#f-name").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } });
function saveFocus() { try { localStorage.setItem("orbx.focus", JSON.stringify(state.focus)); } catch {} if (state.focus) history.replaceState(null, "", `#${lay(state.focus.layer).key.replace("_", "")}-${state.focus.n}`); }

// ---------- hover tip ----------
const tip = $("#tip");
function onHover(p, x, y) {
  if (!p) { tip.style.display = "none"; return; }
  let html = "";
  if (p.type === "neuron") { const a = D.all[lay(p.layer).key]; html = `<div class="lab">${nid(p.layer, p.n)}</div>${nameOf(p.layer, p.n) ? `<b>${esc(nameOf(p.layer, p.n))}</b><br>` : ""}<span class="tw">${esc(a.name[p.n])}</span>` + (a.live[p.n] ? `<br><span class="tw">on ${pct(a.on[p.n])} · selective ${fmt(a.kurt[p.n], 1)}</span>` : ""); }
  else if (p.type === "layer") { const y = lay(p.layer); html = `<div class="lab">${esc(y.name)}</div><span class="tw">${y.live} of ${N()} units live · median selectivity ${fmt(y.kurt_median, 1)}</span><br><span class="tw">click to face this sheet</span>`; }
  else if (p.type === "slab") html = `<div class="lab">block ${p.block} · message pass</div><span class="tw">edge MLP (two hidden layers of 1,024 per directed edge) and send/receive gates; not shown</span>`;
  tip.innerHTML = html; tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight; tip.style.left = `${Math.min(innerWidth - w - 8, x + 14)}px`; tip.style.top = `${Math.min(innerHeight - h - 8, y + 14)}px`;
}

// ---------- Claude tab (local server only) ----------
function renderAgentActions() {
  const f = state.focus, box = $("#a-actions");
  if (!state.claude) { box.innerHTML = ""; return; }
  box.innerHTML = (f ? [`explain ${nid(f.layer, f.n)}`, `what does its web do?`, `push it on NaCl`] : [`which neurons fire on oxygen?`, `what is special about block 2?`]).map(t => `<button>${esc(t)}</button>`).join("");
}
$("#a-actions").addEventListener("click", e => { if (e.target.tagName === "BUTTON") { $("#a-input").value = e.target.textContent; $("#a-form").requestSubmit(); } });
$("#a-form").addEventListener("submit", async e => {
  e.preventDefault(); const msg = $("#a-input").value.trim(); if (!msg) return;
  if (!state.claude) { $("#a-thread").innerHTML += `<div class="a-err">The Claude tab needs the local server (python explorer/server.py) with the claude CLI installed and logged in. ${state.server ? "This server found no claude CLI." : "This page is served without a server."}</div>`; return; }
  $("#a-input").value = ""; const th = $("#a-thread"); th.querySelector(".a-empty")?.remove();
  th.innerHTML += `<div class="a-user">${esc(msg)}</div>`; th.scrollTop = th.scrollHeight;
  const f = state.focus; state.busy = true; $("#a-send").disabled = true;
  try {
    const r = await fetch("api/agent/send", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ chat: state.chat, message: msg, layer: f?.layer, n: f?.n, crystal: state.runC }) });
    const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "", cur = null;
    while (true) { const { value, done } = await rd.read(); if (done) break; buf += dec.decode(value, { stream: true }); let i;
      while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 1); if (!line.trim()) continue; const ev = JSON.parse(line);
        if (ev.t === "text") { if (!cur) { cur = document.createElement("div"); cur.className = "a-text"; th.appendChild(cur); } cur.textContent += ev.text; }
        else if (ev.t === "tool") { cur = null; const d = document.createElement("div"); d.className = "a-tool"; d.textContent = `⚙ ${ev.name} ${ev.input ? JSON.stringify(ev.input).slice(0, 120) : ""}`; th.appendChild(d); if (ev.name === "open_neuron" && ev.input) openNeuron(+ev.input.layer, +ev.input.n); if (ev.name === "open_crystal" && ev.input) { const h = findCrystals(String(ev.input.id || ev.input.formula || ""))[0]; if (h) runCrystal(h.i); } }
        else if (ev.t === "error") { const d = document.createElement("div"); d.className = "a-err"; d.textContent = ev.text; th.appendChild(d); }
        th.scrollTop = th.scrollHeight; } }
  } catch (err) { th.innerHTML += `<div class="a-err">${esc(err.message)}</div>`; }
  state.busy = false; $("#a-send").disabled = false;
});
$("#a-input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#a-form").requestSubmit(); } });

// ---------- about ----------
function aboutHTML() {
  const a = D.arch;
  return `<p>A browsable map of the node neurons of <b>Orb-v3 (conservative-inf-omat)</b>: the encoder and five message-passing blocks, each with two hidden layers of 1,024 SiLU units (12,288 per atom). Dots are neurons; click one to see what it fires on, what pushing it does, who feeds it and whom it feeds.</p>
  <p><b>Data.</b> Activations of every unit on every atom of ${a.n_crystals.toLocaleString()} textbook crystals (${a.n_atoms.toLocaleString()} atoms): metals, salts, semiconductors, oxides and layered crystals, built with ASE and relaxed with this same Orb-v3 model. Labels and statistics are computed from these atoms alone, so with a set this small they illustrate the tool rather than characterise the model. The web and the “writes” layout are read from the checkpoint's weights. Built ${esc(a.built)}. To explore your own crystals, run the build scripts on any structure file (see the README).</p>
  <p><b>Reading the numbers.</b> <i>selective</i> = excess kurtosis of the unit over atoms (spiky units fire on few atoms). <i>z</i> = activation minus the unit's mean, over its spread, on the ${a.n_crystals}-crystal set. A push (local server with Orb) adds a multiple of the unit's spread to its output on every atom and reports the change in energy and forces. All of it is exploratory and not independently checked; a label is a description of the strongest atoms, not a verified function.</p>
  <p><b>Not here.</b> Edge-MLP neurons and the attention gates.</p>
  <p><b>Credits.</b> Design after <a href="https://github.com/lyramakesmusic/neuron-explorer" target="_blank" rel="noopener">lyramakesmusic/neuron-explorer</a> (MIT): the 3D map, glow points, panels and the Claude-over-MCP tab are theirs in spirit and partly in code. Orb-v3: Orbital Materials (Apache-2.0). Three.js (MIT).</p>
  <p class="dim">${state.server ? (state.live ? "Local server with Orb loaded: live runs, pushes and names are on." : "Local server without Orb: static data only.") : "Static page: no live runs; names stay in this browser."}</p>`;
}
$("#about").addEventListener("click", e => { e.preventDefault(); $("#about-text").innerHTML = aboutHTML(); $("#about-box").classList.add("on"); });
$("#about-close").addEventListener("click", () => $("#about-box").classList.remove("on"));

// ---------- start ----------
async function main() {
  setStatus("loading…");
  [D.arch, D.crystals, D.all, D.layout] = await Promise.all([getJSON("arch.json"), getJSON("crystals.json"), getJSON("all_units.json"), getJSON("layout.json")]);
  try { D.notes = await getJSON("notes.json"); } catch { D.notes = {}; }
  try { D.names = JSON.parse(localStorage.getItem("orbx.names") || "{}"); } catch {}
  try { const r = await fetch("api/status"); if (r.ok) { const s = await r.json(); state.server = true; state.live = !!s.live; state.claude = !!s.claude; if (s.names) D.names = { ...D.names, ...s.names }; } } catch {}
  state.chat = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));
  $("#layer").innerHTML = D.arch.layers.map((y, i) => `<option value="${i}">${esc(y.name)}</option>`).join("");
  map = createMap($("#map"), $("#layer-labels"), D.arch, {
    onHover, onPick: (l, n) => openNeuron(l, n, { fly: false }),
    onPickLayer: l => { setLayer(l); map.enterLayer(l); },
    onPickSlab: () => {},
    onLayerMode: l => { $("#layer-back").classList.toggle("on", l !== null); $("#layer-back").textContent = l === null ? "" : `‹ back from ${short(l)}`; },
  });
  xtal = createCrystal($("#xtal"), $("#xtal-tip"), {});
  window.orbx = { map, xtal, state, D, openNeuron, runCrystal };  // for the console
  map.setLayout(D.arch.layers.map(y => D.layout[y.key]));
  const l75 = D.arch.layers.findIndex(y => y.key === "L2_h1");
  await setLayer(l75 >= 0 ? l75 : 0);
  syncEdges(); renderAgentActions();
  setStatus(`${D.arch.n_crystals.toLocaleString()} crystals · ${D.arch.n_atoms.toLocaleString()} atoms · ${state.server ? (state.live ? "live Orb" : "server, no Orb") : "static"}`);
  const h = location.hash.slice(1);
  let m;
  if ((m = h.match(/^(L\d)(h\d)-(\d+)$/))) { const l = D.arch.layers.findIndex(y => y.key === `${m[1]}_${m[2]}`); if (l >= 0) openNeuron(l, +m[3], { fly: false }); }
  else if ((m = h.match(/^c-(.+)$/))) { const ci = D.crystals.findIndex(c => c.id === m[1]); if (ci >= 0) runCrystal(ci); }
  else { let f = null; try { f = JSON.parse(localStorage.getItem("orbx.focus") || "null"); } catch {} if (f && f.layer < L()) openNeuron(f.layer, f.n, { fly: false }); else if (l75 >= 0) openNeuron(l75, 75, { fly: false }); }
}
main().catch(e => { setStatus(`error: ${e.message}`); console.error(e); });
