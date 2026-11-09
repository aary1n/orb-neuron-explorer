"""Per-atom node hidden units of an Orb-v3 checkpoint for a set of crystals, post-SiLU.

Settings: compile=False, float32-high, CPU. Layers: L0 = encoder node MLP, L1..L5 = gnn_stacks.0..4 node MLP; h1 = Act-0,
h2 = Act-1 (1024 units each), so 12 x 1024 numbers per atom.

    python build/extract_atoms.py --ckpt <orb-v3 .ckpt> --structures structures.jsonl --out <folder>

Writes <out>/acts_all.npy (float16 [n_atoms, 12, 1024]), <out>/atoms_all.csv (material_id, atom, element) and
<out>/layers.json. Crystals are sorted by material_id, atoms in site order.
"""
import argparse, json, os, time
import numpy as np, torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from orb_models.forcefield import pretrained, atomic_system

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--structures", required=True, help="jsonl: {material_id, structure: pymatgen dict}")
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=int(os.environ.get("NT", "4")))
ap.add_argument("--chunk", type=int, default=100)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
CH = os.path.join(a.out, "chunks"); os.makedirs(CH, exist_ok=True)

reg = pretrained.orb_v3_conservative_inf_omat(weights_path=a.ckpt, device="cpu", precision="float32-high", compile=False)
reg.eval()
names, mods = [], []
for L, prefix in enumerate(["model._encoder._node_fn.mlp"] + [f"model.gnn_stacks.{i}._node_mlp.mlp" for i in range(5)]):
    for h, act in (("h1", "Act-0"), ("h2", "Act-1")):
        names.append(f"L{L}_{h}"); mods.append(dict(reg.named_modules())[f"{prefix}.{act}"])
buf = {}
def mk(i):
    def hook(m, inp, out): buf[i] = out.detach().float().numpy()
    return hook
for i, m in enumerate(mods): m.register_forward_hook(mk(i))

rows = [json.loads(l) for l in open(a.structures)]
rows.sort(key=lambda r: r["material_id"])
tags = []
for c0 in range(0, len(rows), a.chunk):
    tag = f"{c0:05d}"; tags.append(tag)
    if os.path.exists(f"{CH}/acts_{tag}.npy"): continue
    A, meta = [], []
    t = time.time()
    for r in rows[c0:c0 + a.chunk]:
        s = Structure.from_dict(r["structure"])
        atoms = AseAtomsAdaptor.get_atoms(s)
        g = atomic_system.ase_atoms_to_atom_graphs(atoms, system_config=reg.system_config, device="cpu")
        buf.clear()
        with torch.no_grad():
            reg.model(g)
        x = np.stack([buf[i] for i in range(12)], axis=1)  # [n, 12, 1024]
        assert x.shape[0] == len(s), (r["material_id"], x.shape)
        A.append(x.astype(np.float16))
        for j, site in enumerate(s):
            meta.append(f'{r["material_id"]},{j},{site.specie.symbol}')
    np.save(f"{CH}/acts_{tag}.npy", np.concatenate(A))
    open(f"{CH}/atoms_{tag}.csv", "w").write("material_id,atom,element\n" + "\n".join(meta) + "\n")
    print(tag, len(meta), f"{time.time() - t:.1f}s", flush=True)
# join
X = np.concatenate([np.load(f"{CH}/acts_{t}.npy") for t in tags])
np.save(os.path.join(a.out, "acts_all.npy"), X)
with open(os.path.join(a.out, "atoms_all.csv"), "w") as f:
    f.write("material_id,atom,element\n")
    for t in tags:
        f.write("".join(open(f"{CH}/atoms_{t}.csv").readlines()[1:]))
json.dump(names, open(os.path.join(a.out, "layers.json"), "w"))
print("done", X.shape, flush=True)
