// 3D map of Orb-v3's node MLPs: the encoder and five message-passing blocks run left to right; each has two hidden
// layers of 1,024 SiLU units, drawn as vertical sheets of dots (32 x 32). Between blocks a translucent slab stands for
// the message pass (edge MLP and attention gates) whose neurons are not shown.
// Structure, glow-point shader and web drawing follow lyramakesmusic/neuron-explorer's map3d.js (MIT, 2026), reduced
// to what Orb needs: no attention heads, no sign regimes, one dot per neuron.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const SHEET = 4.2, GAP = 1.05, SLAB = 1.75, SHEET_Y0 = 0.7;
const PAL = {
  bg: 0x000000,
  idle: new THREE.Color(0.05, 0.065, 0.075), lo: new THREE.Color(0.30, 0.22, 0.08), hi: new THREE.Color(1.0, 0.72, 0.30),
  named: new THREE.Color(1, 1, 1), dead: new THREE.Color(0.02, 0.025, 0.03),
  wire: 0x1c2a30, slab: 0x7fa0b0, slabEdge: 0x4a6470, end: 0x6a8592,
};
const BRIGHT = Number(new URLSearchParams(location.search).get("bright")) || 2.2;
for (const k of ["wire", "slabEdge", "end"]) PAL[k] = new THREE.Color(PAL[k]).multiplyScalar(BRIGHT);
const WEB_UP = new THREE.Color(1.0, 0.82, 0.2), WEB_DOWN = new THREE.Color(1.0, 0.25, 0.85);

function glowPoints(sizePx, maxScale = 3) {
  return new THREE.RawShaderMaterial({
    uniforms: { size: { value: sizePx * Math.min(devicePixelRatio, 2) }, ref: { value: 1 }, maxScale: { value: maxScale }, tdist: { value: 1 },
      focusX: { value: 0 }, fadeW: { value: 1 }, fadeAmt: { value: 0 } },
    vertexShader: `precision highp float;
      uniform mat4 modelViewMatrix, projectionMatrix; uniform float size, ref, maxScale, tdist, focusX, fadeW, fadeAmt;
      attribute vec3 position; attribute vec3 color; varying vec3 vC;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        float d = (position.x - focusX) / fadeW;
        float keep = mix(1.0, exp(-d * d), fadeAmt * 0.8);
        float near = clamp(sqrt(ref / tdist), 1.0, 3.0);
        float sc = clamp(sqrt(ref / -mv.z), 0.6, maxScale), sc0 = clamp(sqrt(ref / tdist), 0.6, maxScale);
        vC = color * keep * near * min(1.0, sc0 / sc);
        gl_PointSize = size * sc; }`,
    fragmentShader: `precision highp float; varying vec3 vC;
      void main() { float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard;
        float edge = 1.0 - smoothstep(0.36, 0.5, r); gl_FragColor = vec4(vC * edge, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

export function createMap(canvas, labelsDiv, arch, cb) {
  const L = arch.layers.length, N = arch.N, side = arch.side;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(PAL.bg);
  const scene = new THREE.Scene();
  // x positions: encoder pair, then (slab, h1, h2) per block
  const layerX = [], slabX = [];
  let x = 0;
  for (const ly of arch.layers) {
    if (ly.h === 1 && ly.block > 0) { x += SLAB; slabX.push(x - SLAB * 0.55); }
    layerX.push(x); x += GAP;
  }
  const X0 = -1.3, X1 = x - GAP + 1.6;
  const insets = { left: 404, right: 0 };
  const camera = new THREE.PerspectiveCamera(32, 1, 0.02, 2000);
  let homeDist = 40, fitted = false;
  const home = { target: new THREE.Vector3((X0 + X1) / 2, SHEET_Y0 + SHEET / 2 - 0.3, 0), dir: new THREE.Vector3(0.58, 0.36, 1).normalize() };
  camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(home.target); controls.enableDamping = false; controls.zoomToCursor = true;
  controls.minDistance = 1.2; controls.maxDistance = 300; controls.zoomSpeed = 0.6; controls.update();

  let layoutCells = null;  // per layer: array of grid cells, or null for index order
  const neuronPos = (l, n) => {
    const c = layoutCells ? layoutCells[l][n] : n, i = c % side, j = Math.floor(c / side);
    return new THREE.Vector3(layerX[l], SHEET_Y0 + (j / (side - 1)) * SHEET, (i / (side - 1) - 0.5) * SHEET);
  };
  const pos = new Float32Array(L * N * 3), col = new Float32Array(L * N * 3), base = new Float32Array(L * N * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pointMat = glowPoints(2.5, 5);
  const points = new THREE.Points(geo, pointMat); scene.add(points);
  function placePoints() {
    for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) { const p = neuronPos(l, n), k = (l * N + n) * 3; pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z; }
    geo.attributes.position.needsUpdate = true; geo.computeBoundingSphere();
  }
  placePoints();
  for (let i = 0; i < L * N; i++) PAL.idle.toArray(base, i * 3);
  col.set(base);

  // structure: a spine, sheet outlines, message slabs, an "atoms in" marker and the energy head at the end
  const wires = [], V = (x, y, z) => new THREE.Vector3(x, y, z);
  const seg = (a, b) => wires.push(a.x, a.y, a.z, b.x, b.y, b.z);
  const corners = x => { const h = SHEET / 2; return [V(x, SHEET_Y0, -h), V(x, SHEET_Y0, h), V(x, SHEET_Y0 + SHEET, h), V(x, SHEET_Y0 + SHEET, -h)]; };
  seg(V(X0, 0, 0), V(X1, 0, 0));
  for (let l = 0; l < L; l++) { const cs = corners(layerX[l]); for (let i = 0; i < 4; i++) seg(cs[i], cs[(i + 1) % 4]); seg(V(layerX[l], 0, 0), V(layerX[l], SHEET_Y0, 0)); }
  const slabs = [];
  slabX.forEach((sx, b) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.26, SHEET * 0.92, SHEET * 0.92), new THREE.MeshBasicMaterial({ color: PAL.slab, transparent: true, opacity: 0.06 * BRIGHT, depthWrite: false }));
    m.position.set(sx, SHEET_Y0 + SHEET / 2, 0); m.userData = { block: b + 1 }; scene.add(m); slabs.push(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: PAL.slabEdge, transparent: true }));
    e.position.copy(m.position); scene.add(e); m.userData.edges = e;
  });
  for (const [xx, w, h, d] of [[X0, 0.14, 1.4, 2.2], [X1 - 0.3, 0.3, 2.2, 2.2]]) {
    const m = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)), new THREE.LineBasicMaterial({ color: PAL.end, transparent: true }));
    m.position.set(xx, xx > 0 ? SHEET_Y0 + SHEET / 2 : 0, 0); scene.add(m);
  }
  const wg = new THREE.BufferGeometry(); wg.setAttribute("position", new THREE.Float32BufferAttribute(wires, 3));
  const wireMat = new THREE.LineBasicMaterial({ color: PAL.wire, transparent: true });
  scene.add(new THREE.LineSegments(wg, wireMat));
  const curOutline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x3a5a66 }));
  curOutline.frustumCulled = false; scene.add(curOutline);

  // highlights: hovered / selected / linked dots, direct link lines, and the web (fat lines)
  const hiGeo = new THREE.BufferGeometry(), hiPts = new THREE.Points(hiGeo, glowPoints(5, 2)); hiPts.frustumCulled = false; hiPts.renderOrder = 3; scene.add(hiPts);
  const conGeo = new THREE.BufferGeometry();
  const conLines = new THREE.LineSegments(conGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  conLines.frustumCulled = false; conLines.renderOrder = 2; scene.add(conLines);
  const webMat = new LineMaterial({ vertexColors: true, linewidth: 2, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const webLines = new LineSegments2(new LineSegmentsGeometry(), webMat); webLines.frustumCulled = false; webLines.renderOrder = 1; webLines.visible = false; scene.add(webLines);
  const webPtGeo = new THREE.BufferGeometry(), webPts = new THREE.Points(webPtGeo, glowPoints(4.5, 2.5)); webPts.frustumCulled = false; webPts.renderOrder = 2; webPts.visible = false; scene.add(webPts);
  let selected = null, hovered = null, links = [], webEdges = [];
  function refreshHighlights() {
    const p = [], c = [], add = (l, n, color) => { const v = neuronPos(l, n); p.push(v.x, v.y, v.z); c.push(color.r, color.g, color.b); };
    const zmax = Math.max(1e-6, ...links.map(k => k.z));
    for (const k of links) add(k.layer, k.n, (k.sign > 0 ? WEB_UP : WEB_DOWN).clone().multiplyScalar(0.3 + 0.7 * k.z / zmax));
    if (hovered) add(hovered.layer, hovered.n, PAL.named.clone().multiplyScalar(0.7));
    if (selected) add(selected.layer, selected.n, PAL.named);
    hiGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3)); hiGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
    const lp = [], lc = [];
    if (selected) {
      const a = neuronPos(selected.layer, selected.n);
      for (const k of links) {
        const b = neuronPos(k.layer, k.n), cl = (k.sign > 0 ? WEB_UP : WEB_DOWN).clone().multiplyScalar(0.6 * (0.2 + 0.8 * k.z / zmax));
        lp.push(a.x, a.y, a.z, b.x, b.y, b.z); lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    conGeo.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3)); conGeo.setAttribute("color", new THREE.Float32BufferAttribute(lc, 3));
    render();
  }
  function buildWeb() {
    if (!webEdges.length) { webLines.visible = false; webPts.visible = false; render(); return; }
    const m = webEdges.length, p = new Float32Array(m * 6), c = new Float32Array(m * 6), tp = [], tc = [], seen = new Set();
    const zhi = Math.max(1e-6, ...webEdges.map(e => Math.abs(e[5])));
    webEdges.forEach((e, i) => {
      const a = neuronPos(e[0], e[1]), b = neuronPos(e[2], e[3]), k = e[4] > 0 ? WEB_UP : WEB_DOWN;
      const al = 0.5 * (0.5 + 0.5 * Math.abs(e[5]) / zhi) * Math.pow(0.8, Math.max(0, (e[6] || 1) - 1));
      p.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6); c.set([k.r * al, k.g * al, k.b * al, k.r * al, k.g * al, k.b * al], i * 6);
      for (const [v, key] of [[a, e[0] * N + e[1]], [b, e[2] * N + e[3]]]) if (!seen.has(key)) { seen.add(key); tp.push(v.x, v.y, v.z); tc.push(k.r * 0.8, k.g * 0.8, k.b * 0.8); }
    });
    webPtGeo.setAttribute("position", new THREE.Float32BufferAttribute(tp, 3)); webPtGeo.setAttribute("color", new THREE.Float32BufferAttribute(tc, 3));
    webLines.geometry.dispose(); webLines.geometry = new LineSegmentsGeometry().setPositions(p).setColors(c);
    webLines.visible = true; webPts.visible = true; render();
  }

  // labels: one per sheet, in its plane above the top-left corner; click to look at that sheet face-on
  const labels = [];
  const LABEL_H = 0.22;
  arch.layers.forEach((ly, l) => {
    const txt = ly.block === 0 ? `Enc·L${ly.h}` : `B${ly.block}·L${ly.h}`;
    const cv = document.createElement("canvas"), ctx = cv.getContext("2d"), font = "600 60px ui-monospace, Menlo, monospace";
    ctx.font = font; const tw = Math.ceil(ctx.measureText(txt).width) + 10; cv.width = tw; cv.height = 76;
    ctx.font = font; ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(txt, 5, 40);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    const w = LABEL_H * tw / 76;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide, color: 0xc8d4da, opacity: 0.85 });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, LABEL_H), mat);
    m.rotation.y = Math.PI / 2; m.position.set(layerX[l], SHEET_Y0 + SHEET + 0.08 + LABEL_H / 2, -SHEET / 2 + w / 2); m.userData = { layer: l };
    scene.add(m); labels.push(m);
  });
  // block captions above the slabs
  slabX.forEach((sx, b) => {
    const txt = `block ${b + 1}: messages`;
    const cv = document.createElement("canvas"), ctx = cv.getContext("2d"), font = "500 44px ui-monospace, Menlo, monospace";
    ctx.font = font; const tw = Math.ceil(ctx.measureText(txt).width) + 10; cv.width = tw; cv.height = 60;
    ctx.font = font; ctx.fillStyle = "#9fb4bf"; ctx.textBaseline = "middle"; ctx.fillText(txt, 5, 31);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const w = 0.16 * tw / 60, m = new THREE.Mesh(new THREE.PlaneGeometry(w, 0.16), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide, opacity: 0.7 }));
    m.rotation.y = Math.PI / 2; m.position.set(sx, SHEET_Y0 - 0.25, -SHEET / 2 + w / 2 + 0.1); scene.add(m);
  });

  let pending = false;
  const curDist = () => camera.position.distanceTo(controls.target);
  function render() {
    if (pending) return; pending = true;
    requestAnimationFrame(() => {
      pending = false;
      for (const m of [pointMat, hiPts.material, webPts.material]) { m.uniforms.ref.value = homeDist; m.uniforms.tdist.value = curDist(); }
      applyFade(); renderer.render(scene, camera);
    });
  }
  const bounds = [];
  for (const xx of [X0, X1]) for (const y of [-0.8, SHEET_Y0 + SHEET + 0.4]) for (const z of [-SHEET / 2, SHEET / 2]) bounds.push(V(xx, y, z));
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); webMat.resolution.set(w, h);
    const narrow = w <= 760, li = narrow ? 0 : insets.left, ri = narrow ? 0 : insets.right, availW = Math.max(200, w - li - ri);
    camera.aspect = w / h;
    // fit the whole stack into the free area; re-place the camera while it still sits at home (the user has not moved it)
    const atHome = !fitted || camera.position.distanceTo(home.target.clone().addScaledVector(home.dir, homeDist)) < 1e-3;
    const r = Math.max(...bounds.map(b => b.distanceTo(home.target)));
    const vHalf = THREE.MathUtils.degToRad(camera.fov / 2), hHalf = Math.atan(Math.tan(vHalf) * availW / h);
    homeDist = r / Math.sin(Math.min(vHalf, hHalf)) * 0.8;
    if (atHome && layerMode === null) { camera.position.copy(home.target).addScaledVector(home.dir, homeDist); controls.target.copy(home.target); controls.update(); }
    fitted = true;
    camera.setViewOffset(w, h, w / 2 - (li + availW / 2), 0, w, h); camera.updateProjectionMatrix(); render();
  }
  new ResizeObserver(resize).observe(canvas);
  controls.addEventListener("change", render);

  // fade: zooming towards one sheet dims the others
  const fade = { fx: 0, fw: 1, amt: 0 };
  let layerMode = null, saved = null;
  function applyFade() {
    if (layerMode !== null) {
      pointMat.uniforms.focusX.value = layerX[layerMode]; pointMat.uniforms.fadeW.value = 0.05; pointMat.uniforms.fadeAmt.value = 1;
      for (const s of slabs) { s.material.opacity = 0; s.userData.edges.material.opacity = 0; }
      labels.forEach((m, l) => m.material.opacity = l === layerMode ? 0.9 : 0); wireMat.opacity = 0.15; return;
    }
    const dist = curDist(), far = Math.log(homeDist * 0.7), near = Math.log(homeDist * 0.08);
    const amt = THREE.MathUtils.smoothstep(THREE.MathUtils.clamp((far - Math.log(dist)) / (far - near), 0, 1), 0, 1);
    const tv = controls.target.clone().project(camera), pv = new THREE.Vector3();
    let fx = controls.target.x, best = Infinity;
    for (let l = 0; l < L; l++) { pv.set(layerX[l], SHEET_Y0 + SHEET / 2, 0).project(camera); const dd = Math.hypot((pv.x - tv.x) * camera.aspect, pv.y - tv.y); if (pv.z < 1 && dd < best) { best = dd; fx = layerX[l]; } }
    const fw = GAP * (1.2 + 3 * (1 - amt));
    pointMat.uniforms.focusX.value = fx; pointMat.uniforms.fadeW.value = fw; pointMat.uniforms.fadeAmt.value = amt;
    for (const s of slabs) { const d = (s.position.x - fx) / fw, f = 1 - amt * (1 - Math.exp(-d * d)); s.material.opacity = 0.06 * BRIGHT * f; s.userData.edges.material.opacity = f; }
    labels.forEach((m, l) => { const d = (layerX[l] - fx) / fw; m.material.opacity = 0.85 * (1 - 0.5 * amt * (1 - Math.exp(-d * d))); });
    wireMat.opacity = 1 - 0.85 * amt;
  }
  function enterLayer(l) {
    if (layerMode === null) saved = { pos: camera.position.clone(), tgt: controls.target.clone() };
    layerMode = l;
    const c = V(layerX[l], SHEET_Y0 + SHEET / 2, 0), dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12;
    controls.target.copy(c); camera.position.copy(c).add(V(-dist, 0, 0));
    controls.enableRotate = false; controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(l);
  }
  function exitLayer() {
    if (layerMode === null) return;
    layerMode = null; controls.enableRotate = true; controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    camera.position.copy(saved.pos); controls.target.copy(saved.tgt); controls.update(); render(); cb.onLayerMode && cb.onLayerMode(null);
  }
  function flyTo(target, dist) {
    const from = controls.target.clone(), to = target.clone(), dir = camera.position.clone().sub(controls.target), d0 = dir.length(); dir.normalize();
    const t0 = performance.now();
    (function step(t) { const k = Math.min(1, (t - t0) / 450), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, to, e); camera.position.copy(controls.target).addScaledVector(dir, d0 + (dist - d0) * e); controls.update(); render();
      if (k < 1) requestAnimationFrame(step); })(t0);
  }
  function viewLayer(l) {
    const c = V(layerX[l], SHEET_Y0 + SHEET / 2, 0), dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
    const from = controls.target.clone(), p0 = camera.position.clone(), p1 = c.clone().add(V(dist, 0.0001, 0)), t0 = performance.now();
    (function step(t) { const k = Math.min(1, (t - t0) / 600), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, c, e); camera.position.lerpVectors(p0, p1, e); controls.update(); render(); if (k < 1) requestAnimationFrame(step); })(t0);
  }

  // picking
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.05; const mouse = new THREE.Vector2();
  const layerVisibility = l => 1 - 0.8 * (layerMode !== null ? (l === layerMode ? 0 : 1) : pointMat.uniforms.fadeAmt.value * (1 - Math.exp(-Math.pow((layerX[l] - pointMat.uniforms.focusX.value) / pointMat.uniforms.fadeW.value, 2))));
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(mouse, camera);
    const hits = ray.intersectObject(points);
    if (hits.length) {
      const layers = [...new Set(hits.map(h => Math.floor(h.index / N)))];
      const front = layers.find(l => layerVisibility(l) > 0.5) ?? layers[0];
      let best = null;
      for (const h of hits) if (Math.floor(h.index / N) === front && (!best || h.distanceToRay < best.distanceToRay)) best = h;
      return { type: "neuron", layer: front, n: best.index % N };
    }
    const lb = ray.intersectObjects(labels)[0]; if (lb) return { type: "layer", layer: lb.object.userData.layer };
    const sb = ray.intersectObjects(slabs)[0]; if (sb) return { type: "slab", block: sb.object.userData.block };
    return null;
  }
  let moveQueued = null, downAt = null;
  canvas.addEventListener("pointermove", ev => {
    if (moveQueued) { moveQueued = ev; return; } moveQueued = ev;
    requestAnimationFrame(() => { const e = moveQueued; moveQueued = null; const p = pick(e);
      const h = p && p.type === "neuron" ? { layer: p.layer, n: p.n } : null;
      if (JSON.stringify(h) !== JSON.stringify(hovered)) { hovered = h; refreshHighlights(); }
      cb.onHover(p, e.clientX, e.clientY); });
  });
  canvas.addEventListener("pointerleave", () => { hovered = null; refreshHighlights(); cb.onHover(null); });
  canvas.addEventListener("pointerdown", ev => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener("pointerup", ev => {
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
    const p = pick(ev);
    if (p && p.type === "neuron") cb.onPick(p.layer, p.n);
    else if (p && p.type === "layer") cb.onPickLayer && cb.onPickLayer(p.layer);
    else if (p && p.type === "slab") cb.onPickSlab && cb.onPickSlab(p.block);
  });

  return {
    neuronPos, layerX, get layerMode() { return layerMode; }, enterLayer, exitLayer, viewLayer,
    home() { exitLayer(); flyTo(home.target, homeDist); },
    // per-point levels 0..1 (Float32Array L*N), dead flags, named flags; the current layer's sheet gets an outline
    setColours(levels, dead, named, current) {
      const c = new THREE.Color();
      for (let i = 0; i < L * N; i++) {
        if (named && named[i]) c.copy(PAL.named);
        else if (dead && dead[i]) c.copy(PAL.dead);
        else { const v = levels ? levels[i] : 0; if (v <= 0) c.copy(PAL.idle); else c.copy(PAL.lo).lerp(PAL.hi, Math.min(1, v)); }
        c.toArray(col, i * 3);
      }
      geo.attributes.color.needsUpdate = true;
      const cs = corners(layerX[current]), dl = [];
      for (let i = 0; i < 4; i++) dl.push(...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
      curOutline.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dl, 3));
      render();
    },
    select(layer, n, fly = true) {
      selected = layer == null ? null : { layer, n }; links = []; refreshHighlights();
      if (!fly || layer == null || layerMode !== null) return;
      const v = neuronPos(layer, n).project(camera);
      if (v.z > 1 || Math.abs(v.x) > 0.9 || Math.abs(v.y) > 0.9) flyTo(neuronPos(layer, n), Math.min(curDist(), homeDist * 0.7));
    },
    hover(layer, n) { hovered = layer == null ? null : { layer, n }; refreshHighlights(); },
    setLinks(list) { links = list || []; refreshHighlights(); },
    setWeb(edges) { webEdges = edges || []; buildWeb(); },
    setLayout(cells) { layoutCells = cells; placePoints(); buildWeb(); refreshHighlights(); },
    setInsets(l, r) { insets.left = l; insets.right = r; resize(); },
    project(layer, n) { const v = neuronPos(layer, n).project(camera); return { x: (v.x + 1) / 2 * canvas.clientWidth, y: (1 - v.y) / 2 * canvas.clientHeight, z: v.z }; },
  };
}
