// A small crystal viewer: unit cell (or 2x2x2), atoms as spheres coloured by element, optionally shaded towards a hot
// colour by a neuron's activation on each atom (the counterpart of shading each token by its activation).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Jmol/CPK colours and Cordero covalent radii (A) for the elements that occur in the set; others fall back to grey, 1.5 A
const EL = {
  H: ["#f0f0f0", .31], Li: ["#cc80ff", 1.28], Be: ["#c2ff00", .96], B: ["#ffb5b5", .84], C: ["#909090", .76], N: ["#3050f8", .71], O: ["#ff0d0d", .66], F: ["#90e050", .57],
  Na: ["#ab5cf2", 1.66], Mg: ["#8aff00", 1.41], Al: ["#bfa6a6", 1.21], Si: ["#f0c8a0", 1.11], P: ["#ff8000", 1.07], S: ["#ffff30", 1.05], Cl: ["#1ff01f", 1.02],
  K: ["#8f40d4", 2.03], Ca: ["#3dff00", 1.76], Sc: ["#e6e6e6", 1.70], Ti: ["#bfc2c7", 1.60], V: ["#a6a6ab", 1.53], Cr: ["#8a99c7", 1.39], Mn: ["#9c7ac7", 1.39],
  Fe: ["#e06633", 1.32], Co: ["#f090a0", 1.26], Ni: ["#50d050", 1.24], Cu: ["#c88033", 1.32], Zn: ["#7d80b0", 1.22], Ga: ["#c28f8f", 1.22], Ge: ["#668f8f", 1.20],
  As: ["#bd80e3", 1.19], Se: ["#ffa100", 1.20], Br: ["#a62929", 1.20], Rb: ["#702eb0", 2.20], Sr: ["#00ff00", 1.95], Y: ["#94ffff", 1.90], Zr: ["#94e0e0", 1.75],
  Nb: ["#73c2c9", 1.64], Mo: ["#54b5b5", 1.54], Tc: ["#3b9e9e", 1.47], Ru: ["#248f8f", 1.46], Rh: ["#0a7d8c", 1.42], Pd: ["#006985", 1.39], Ag: ["#c0c0c0", 1.45],
  Cd: ["#ffd98f", 1.44], In: ["#a67573", 1.42], Sn: ["#668080", 1.39], Sb: ["#9e63b5", 1.39], Te: ["#d47a00", 1.38], I: ["#940094", 1.39], Cs: ["#57178f", 2.44],
  Ba: ["#00c900", 2.15], La: ["#70d4ff", 2.07], Ce: ["#ffffc7", 2.04], Pr: ["#d9ffc7", 2.03], Nd: ["#c7ffc7", 2.01], Pm: ["#a3ffc7", 1.99], Sm: ["#8fffc7", 1.98],
  Eu: ["#61ffc7", 1.98], Gd: ["#45ffc7", 1.96], Tb: ["#30ffc7", 1.94], Dy: ["#1fffc7", 1.92], Ho: ["#00ff9c", 1.92], Er: ["#00e675", 1.89], Tm: ["#00d452", 1.90],
  Yb: ["#00bf38", 1.87], Lu: ["#00ab24", 1.87], Hf: ["#4dc2ff", 1.75], Ta: ["#4da6ff", 1.70], W: ["#2194d6", 1.62], Re: ["#267dab", 1.51], Os: ["#266696", 1.44],
  Ir: ["#175487", 1.41], Pt: ["#d0d0e0", 1.36], Au: ["#ffd123", 1.36], Hg: ["#b8b8d0", 1.32], Tl: ["#a6544d", 1.45], Pb: ["#575961", 1.46], Bi: ["#9e4fb5", 1.48],
  Ac: ["#70abfa", 2.15], Th: ["#00baff", 2.06], Pa: ["#00a1ff", 2.00], U: ["#008fff", 1.96], Np: ["#0080ff", 1.90], Pu: ["#006bff", 1.87],
};
export const elementColour = e => (EL[e] || ["#9a9a9a"])[0];
const HOT = new THREE.Color("#ffd27a");

export function createCrystal(canvas, tipEl, cb = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.3); key.position.set(3, 5, 6); scene.add(key);
  const fill = new THREE.DirectionalLight(0x9fc4ff, 0.5); fill.position.set(-5, -2, -4); scene.add(fill);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 500); camera.position.set(0, 0, 20);
  const controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.dampingFactor = 0.12; controls.zoomSpeed = 0.7;
  let mesh = null, cellLines = null, cur = null, k = 1, shade = true, acts = null, scale = 1, hovered = -1, running = false;
  const sphere = new THREE.SphereGeometry(1, 24, 16);
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.1 });
  const siteOf = id => cur ? id % cur.sites.length : -1;
  function colourOf(site, instance) {
    const base = new THREE.Color(elementColour(cur.sites[site][0]));
    if (!shade || !acts) return base;
    const a = acts[site]; if (a == null) return base.multiplyScalar(0.35);
    const w = Math.max(0, Math.min(1, a / (scale || 1)));
    return base.multiplyScalar(0.42).lerp(HOT, w);
  }
  function build() {
    if (mesh) { scene.remove(mesh); mesh.dispose(); mesh = null; }
    if (cellLines) { scene.remove(cellLines); cellLines.geometry.dispose(); cellLines = null; }
    if (!cur) { render(); return; }
    const M = cur.lat, ns = cur.sites.length, reps = [];
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) for (let l = 0; l < k; l++) reps.push([i, j, l]);
    const cart = (f) => new THREE.Vector3(f[0] * M[0][0] + f[1] * M[1][0] + f[2] * M[2][0], f[0] * M[0][1] + f[1] * M[1][1] + f[2] * M[2][1], f[0] * M[0][2] + f[1] * M[1][2] + f[2] * M[2][2]);
    const centre = cart([k / 2, k / 2, k / 2]);
    mesh = new THREE.InstancedMesh(sphere, mat, ns * reps.length);
    const m4 = new THREE.Matrix4(); let id = 0; const pts = [];
    for (const r of reps) for (let s = 0; s < ns; s++) {
      const site = cur.sites[s], p = cart([site[1] + r[0], site[2] + r[1], site[3] + r[2]]).sub(centre);
      const rad = (EL[site[0]] || [0, 1.5])[1] * 0.42;
      m4.makeScale(rad, rad, rad).setPosition(p); mesh.setMatrixAt(id, m4); mesh.setColorAt(id, colourOf(s, id)); pts.push(p); id++;
    }
    mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true; scene.add(mesh);
    // cell edges of the k x k x k block
    const C = [[0, 0, 0], [k, 0, 0], [k, k, 0], [0, k, 0], [0, 0, k], [k, 0, k], [k, k, k], [0, k, k]].map(f => cart(f).sub(centre));
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]], arr = [];
    for (const [a, b] of E) arr.push(C[a].x, C[a].y, C[a].z, C[b].x, C[b].y, C[b].z);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
    cellLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x4a5a64, transparent: true, opacity: 0.9 })); scene.add(cellLines);
    // fit
    let r = 0; for (const p of [...pts, ...C]) r = Math.max(r, p.length()); r += 1.5;
    const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.95;
    const dir = camera.position.clone().sub(controls.target).normalize(); if (!dir.length()) dir.set(0.4, 0.3, 1).normalize();
    controls.target.set(0, 0, 0); camera.position.copy(dir.multiplyScalar(dist)); camera.near = dist / 100; camera.far = dist * 10; camera.updateProjectionMatrix(); controls.update();
    render();
  }
  function recolour() { if (!mesh) return; const ns = cur.sites.length; for (let i = 0; i < mesh.count; i++) mesh.setColorAt(i, colourOf(i % ns, i)); mesh.instanceColor.needsUpdate = true; render(); }
  let pending = false;
  function render() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; renderer.render(scene, camera); }); }
  function loop() { if (!running) return; controls.update(); render(); requestAnimationFrame(loop); }
  function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); render(); }
  new ResizeObserver(resize).observe(canvas);
  controls.addEventListener("change", render);
  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
  canvas.addEventListener("pointermove", ev => {
    if (!mesh) return;
    const r = canvas.getBoundingClientRect(); mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObject(mesh)[0]; const s = hit ? siteOf(hit.instanceId) : -1;
    if (s !== hovered) { hovered = s; cb.onHover && cb.onHover(s); }
    if (s >= 0) { const site = cur.sites[s]; const a = acts ? acts[s] : null;
      tipEl.innerHTML = `<b>${site[0]}</b> <span class="dim">#${s}</span>` + (a == null ? "" : ` · act ${a.toFixed(2)}`) + (cur.cn && cur.cn[s] != null ? ` · ${cur.cn[s]} neighbours` : "") + (cur.void && cur.void[s] != null ? ` · void ${cur.void[s].toFixed(2)} Å` : "");
      tipEl.style.display = "block"; tipEl.style.left = `${ev.clientX - r.left + 12}px`; tipEl.style.top = `${ev.clientY - r.top + 12}px`; }
    else tipEl.style.display = "none";
  });
  canvas.addEventListener("pointerleave", () => { tipEl.style.display = "none"; hovered = -1; cb.onHover && cb.onHover(-1); });
  canvas.addEventListener("click", ev => { if (hovered >= 0 && cb.onClick) cb.onClick(hovered); });
  return {
    show(crystal, activations = null, actScale = 1) { cur = crystal; acts = activations; scale = actScale; build(); if (!running) { running = true; loop(); } },
    setActs(activations, actScale) { acts = activations; scale = actScale; recolour(); },
    setCell(kk) { k = kk; build(); }, setShade(on) { shade = on; recolour(); },
    get crystal() { return cur; }, resize, stop() { running = false; },
  };
}
