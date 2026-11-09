"""Build the explorer's data folder from per-atom activations alone (no external labels or tables).

    python build/build_data.py --acts <acts folder> --structures structures_demo.jsonl --ckpt <orb-v3 .ckpt> --out data

Inputs: <acts>/acts_all.npy, atoms_all.csv, layers.json from build/extract_atoms.py, and the same structures file.
Per atom it computes two facts from the structure itself: coordination number (pymatgen CrystalNN) and nearest-neighbour
distance. Per unit: activation statistics, the strongest atoms, an auto label from them, the share of variance explained
by the element (eta^2), and Spearman correlations with the two facts, each compared against random directions in the
same layer. Outputs: arch.json, crystals.json, all_units.json, units_<layer>.json (12), fires.bin.

Every number here is descriptive: a label describes the strongest atoms, it is not a verified function.
"""
import argparse, datetime, hashlib, json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import kurtosis, rankdata
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True); ap.add_argument("--structures", required=True)
ap.add_argument("--ckpt", default=None, help="checkpoint file, for its sha256 in arch.json")
ap.add_argument("--out", required=True)
ap.add_argument("--topk-fires", type=int, default=12)
ap.add_argument("--top-atoms", type=int, default=20)
ap.add_argument("--random-dirs", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
names = json.load(open(os.path.join(a.acts, "layers.json")))
X = np.load(os.path.join(a.acts, "acts_all.npy"))
meta = pd.read_csv(os.path.join(a.acts, "atoms_all.csv"))
NA = len(meta)
LAYER_NAMES = {"L0": "Encoder", "L1": "Block 1", "L2": "Block 2", "L3": "Block 3", "L4": "Block 4", "L5": "Block 5"}
LABS = [("cn", "coordination number"), ("nn", "nearest-neighbour distance")]

# ---------- crystals and per-atom facts ----------
rows = [json.loads(l) for l in open(a.structures)]; rows.sort(key=lambda r: r["material_id"])
assert [r["material_id"] for r in rows] == list(dict.fromkeys(meta.material_id)), "structure order must match the atoms table"
cnn = CrystalNN()
crystals, cn_all, nn_all = [], [], []
a0 = 0
for r in rows:
    s = Structure.from_dict(r["structure"])
    try: sga = SpacegroupAnalyzer(s, symprec=0.1); sg, sgn = sga.get_space_group_symbol(), sga.get_space_group_number()
    except Exception: sg, sgn = None, None
    cn = []
    for j in range(len(s)):
        try: cn.append(int(cnn.get_cn(s, j)))
        except Exception: cn.append(None)
    dm = s.distance_matrix + np.eye(len(s)) * 1e9 if len(s) > 1 else None
    nn = []
    for j in range(len(s)):
        d = min([x.nn_distance for x in s.get_neighbors(s[j], 6.0)] or [np.nan])
        nn.append(round(float(d), 3))
    rec = dict(id=r["material_id"], f=r["formula"], sg=sg, sgn=sgn, ns=len(s), gap=None, ef=None, eh=None, mag=None,
               dens=round(float(s.density), 2), th=False, a0=a0, x=False, el=None,
               e_orb=r.get("energy_eV_per_atom"),
               lat=[[round(v, 4) for v in row] for row in s.lattice.matrix.tolist()],
               sites=[[site.specie.symbol, *[round(float(v), 4) for v in site.frac_coords]] for site in s],
               cn=cn, nn=nn)
    crystals.append(rec); cn_all += cn; nn_all += nn; a0 += len(s)
assert a0 == NA
json.dump(crystals, open(os.path.join(a.out, "crystals.json"), "w"), separators=(",", ":"))
atom_cidx = np.concatenate([[i] * c["ns"] for i, c in enumerate(crystals)])
el = meta.element.values
facts = {"cn": np.array([np.nan if v is None else v for v in cn_all], float), "nn": np.array(nn_all, float)}
print("crystals", len(crystals), "atoms", NA, flush=True)

# within-element centring and helpers
groups = [np.where(el == e)[0] for e in np.unique(el)]
def centre(v):
    v = v.astype(float).copy()
    for g in groups: v[g] -= np.nanmean(v[g]) if np.isfinite(v[g]).any() else 0
    return v
def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12: return np.nan
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1])
def eta2(v):
    tot = ((v - v.mean()) ** 2).sum()
    if tot < 1e-12: return np.nan
    return float(sum(len(g) * (v[g].mean() - v.mean()) ** 2 for g in groups) / tot)
fc = {k: centre(v) for k, v in facts.items()}

# ---------- per layer ----------
fires = np.zeros((NA, len(names), a.topk_fires, 2), np.uint16)   # [atom, layer, k, (neuron, z*10)]
all_units = {}; arch_layers = []
for li, nm in enumerate(names):
    M = X[:, li, :].astype(np.float64)
    mean = M.mean(0); sd = M.std(0); live = sd > 1e-4
    kurt = np.full(1024, np.nan); kurt[live] = kurtosis(M[:, live], axis=0, fisher=True)
    mx = M.max(0); p999 = np.percentile(M, 99.9, axis=0); on = (M > 0.1).mean(0)
    on_units = live & (mx >= 0.1)  # units that never switch on get huge z from a tiny spread: keep them out
    Z = (M - mean) / np.where(on_units, sd, np.inf)
    k = a.topk_fires
    part = np.argpartition(-Z, k, axis=1)[:, :k]
    zz = np.take_along_axis(Z, part, 1); order = np.argsort(-zz, axis=1)
    part = np.take_along_axis(part, order, 1); zz = np.take_along_axis(zz, order, 1)
    fires[:, li, :, 0] = part; fires[:, li, :, 1] = np.clip(np.round(zz * 10), 0, 65535)
    # random directions in this layer: the yardstick for eta^2 and the correlations
    Ml = M[:, live] - mean[live]
    R = rng.standard_normal((Ml.shape[1], a.random_dirs)); R /= np.linalg.norm(R, axis=0)
    P = Ml @ R
    rnd_eta = np.array([eta2(P[:, i]) for i in range(P.shape[1])])
    rnd_rho = {kf: np.abs([spearman(centre(P[:, i]), fc[kf]) for i in range(P.shape[1])]) for kf in fc}
    pct = lambda v, ref: None if not np.isfinite(v) else round(float((ref[np.isfinite(ref)] < abs(v)).mean() * 100), 1)
    recs = []; compact = {"name": [], "on": [], "kurt": [], "max": [], "live": [], "eta2": [], "best": []}
    eta_live = []
    for n in range(1024):
        r = dict(n=n, live=bool(live[n]), max=round(float(mx[n]), 3), p999=round(float(p999[n]), 3), on=round(float(on[n]), 4),
                 mean=round(float(mean[n]), 4), sd=round(float(sd[n]), 4), kurt=None if not live[n] else round(float(kurt[n]), 1))
        if live[n]:
            v = M[:, n]; e2 = eta2(v); eta_live.append(e2)
            r["eta2"] = None if not np.isfinite(e2) else round(e2, 3); r["eta2p"] = pct(e2, rnd_eta)
            labs = {}; best = None; vc = centre(v)
            for key_, desc in LABS:
                rho = spearman(vc, fc[key_])
                if not np.isfinite(rho): continue
                labs[key_] = [round(rho, 3), pct(rho, rnd_rho[key_])]
                if best is None or abs(rho) > abs(best[1]): best = [key_, round(rho, 3), labs[key_][1]]
            r["lab"] = labs; r["best"] = best
            # strongest atoms, symmetry copies merged (same crystal, element and activation)
            top, seen_keys = [], set()
            for i in np.argsort(-v):
                kk = (int(atom_cidx[i]), el[i], round(float(v[i]), 3))
                if kk in seen_keys: continue
                seen_keys.add(kk); top.append([int(i), round(float(v[i]), 3)])
                if len(top) == a.top_atoms: break
            r["top"] = top
            seen, tc = [], []
            for i, _ in top:
                ci = int(atom_cidx[i])
                if ci in seen: continue
                seen.append(ci); c = crystals[ci]
                tc.append([ci, [round(float(x), 3) for x in M[c["a0"]:c["a0"] + c["ns"], n]]])
                if len(tc) == 12: break
            r["tc"] = tc
            cnt = {}
            for i, _ in top: cnt[el[i]] = cnt.get(el[i], 0) + 1
            els = sorted(cnt.items(), key=lambda kv: -kv[1]); r["els"] = els
            cover = 0; named = []
            for e, c_ in els:
                named.append(e); cover += c_
                if cover >= 0.75 * len(top) or len(named) == 3: break
            r["cov"] = round(cover / max(1, len(top)), 2)
            cns = [facts["cn"][i] for i, _ in top if el[i] in named and np.isfinite(facts["cn"][i])]
            cn_txt = ""
            if cns:
                vals, counts = np.unique(cns, return_counts=True); j = counts.argmax()
                if counts[j] >= 0.7 * len(cns): cn_txt = f" with {int(vals[j])} neighbours"
            fset = []
            for ci, _ in tc:
                if crystals[ci]["f"] not in fset: fset.append(crystals[ci]["f"])
            if r["max"] < 0.1: name = f"never switches on (max {r['max']:.2f})"
            else: name = f"{', '.join(named)} atoms{cn_txt}; strongest in {', '.join(fset[:4])}"
            r["name"] = name
        else:
            r["name"] = "dead (constant)" if abs(r["max"]) < 1e-3 else "dead (constant, non-zero)"
            r["top"] = []; r["tc"] = []
        recs.append(r)
        for kk in compact: compact[kk].append(r.get(kk))
    json.dump(recs, open(os.path.join(a.out, f"units_{nm}.json"), "w"), separators=(",", ":"))
    all_units[nm] = compact
    blk, h = nm.split("_")
    arch_layers.append(dict(key=nm, idx=li, block=int(blk[1]), h=int(h[1]), name=f"{LAYER_NAMES[blk]} · layer {h[1]}", live=int(live.sum()),
                            eta_units=round(float(np.nanmedian(eta_live)), 3) if eta_live else None,
                            eta_rand=round(float(np.nanmedian(rnd_eta)), 3),
                            kurt_median=round(float(np.nanmedian(kurt)), 1) if live.any() else None))
    print(nm, "live", int(live.sum()), "median kurt", arch_layers[-1]["kurt_median"], flush=True)
fires.astype("<u2").tofile(os.path.join(a.out, "fires.bin"))
json.dump(all_units, open(os.path.join(a.out, "all_units.json"), "w"), separators=(",", ":"))
sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest() if a.ckpt else None
arch = dict(model="orb-v3-conservative-inf-omat-20250404", checkpoint_sha256=sha, N=1024, side=32, layers=arch_layers, fires_k=a.topk_fires,
            n_crystals=len(crystals), n_extra=0, n_atoms=NA, n_atoms_all=NA, ckpts=[],
            built=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
            labels={k: d for k, d in LABS}, random_dirs=a.random_dirs,
            notes=["Activations: Orb-v3 conservative-inf-omat, compile=False, float32-high, CPU, on the demo crystals (built with ASE, relaxed with the same model).",
                   "Statistics (top atoms, z-scores, labels, kurtosis) are over the demo atoms only; with a small set they are illustrative.",
                   "Descriptive only; a label describes the strongest atoms, not a verified function."])
json.dump(arch, open(os.path.join(a.out, "arch.json"), "w"), indent=1)
print("done", flush=True)
