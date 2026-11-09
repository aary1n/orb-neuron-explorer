"""Demo crystal set for the public explorer: textbook crystals built with ASE from approximate room-temperature
lattice constants, then relaxed (cell and positions) with Orb-v3 conservative-inf-omat itself, so every structure is
at the model's own minimum. No database structures are used.

    python build/make_demo_structures.py --ckpt <orb-v3 .ckpt> --out structures_demo.jsonl

Writes one JSON line per crystal: {material_id, formula, structure (pymatgen dict), energy_eV_per_atom, relax_steps}.
"""
import argparse, json
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.io.ase import AseAtomsAdaptor
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fmax", type=float, default=0.01)
ap.add_argument("--steps", type=int, default=400)
a = ap.parse_args()

# (name, formula, crystal structure, a [A], extra kwargs); approximate lattice constants, relaxed below
SPECS = [
    ("Cu", "Cu", "fcc", 3.61, {}), ("Ag", "Ag", "fcc", 4.09, {}), ("Au", "Au", "fcc", 4.08, {}), ("Al", "Al", "fcc", 4.05, {}),
    ("Ni", "Ni", "fcc", 3.52, {}), ("Pt", "Pt", "fcc", 3.92, {}),
    ("Fe", "Fe", "bcc", 2.87, {}), ("W", "W", "bcc", 3.16, {}), ("Mo", "Mo", "bcc", 3.15, {}), ("Na", "Na", "bcc", 4.23, {}),
    ("K", "K", "bcc", 5.23, {}), ("V", "V", "bcc", 3.03, {}),
    ("Mg", "Mg", "hcp", 3.21, {"c": 5.21}), ("Ti", "Ti", "hcp", 2.95, {"c": 4.68}), ("Zn", "Zn", "hcp", 2.66, {"c": 4.95}),
    ("Si", "Si", "diamond", 5.43, {}), ("Ge", "Ge", "diamond", 5.66, {}), ("C-diamond", "C", "diamond", 3.57, {}),
    ("NaCl", "NaCl", "rocksalt", 5.64, {}), ("KCl", "KCl", "rocksalt", 6.29, {}), ("LiF", "LiF", "rocksalt", 4.03, {}),
    ("MgO", "MgO", "rocksalt", 4.21, {}), ("CaO", "CaO", "rocksalt", 4.81, {}), ("PbS", "PbS", "rocksalt", 5.94, {}),
    ("CsCl", "CsCl", "cesiumchloride", 4.12, {}), ("NiAl", "NiAl", "cesiumchloride", 2.88, {}),
    ("GaAs", "GaAs", "zincblende", 5.65, {}), ("ZnS", "ZnS", "zincblende", 5.41, {}), ("InP", "InP", "zincblende", 5.87, {}),
    ("SiC", "SiC", "zincblende", 4.36, {}),
    ("ZnO", "ZnO", "wurtzite", 3.25, {"c": 5.21}), ("GaN", "GaN", "wurtzite", 3.19, {"c": 5.19}), ("AlN", "AlN", "wurtzite", 3.11, {"c": 4.98}),
    ("CaF2", "CaF2", "fluorite", 5.46, {}), ("CeO2", "CeO2", "fluorite", 5.41, {}), ("Li2O", "OLi2", "fluorite", 4.61, {}),
]
# (name, symbols, basis, spacegroup, cellpar); approximate experimental structures, relaxed below
SPG = [
    ("SrTiO3", ["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], 221, [3.905, 3.905, 3.905, 90, 90, 90]),
    ("BaTiO3", ["Ba", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], 221, [4.00, 4.00, 4.00, 90, 90, 90]),
    ("KMgF3", ["K", "Mg", "F"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], 221, [3.99, 3.99, 3.99, 90, 90, 90]),
    ("TiO2-rutile", ["Ti", "O"], [(0, 0, 0), (0.305, 0.305, 0)], 136, [4.594, 4.594, 2.959, 90, 90, 90]),
    ("SnO2", ["Sn", "O"], [(0, 0, 0), (0.307, 0.307, 0)], 136, [4.737, 4.737, 3.186, 90, 90, 90]),
    ("MoS2", ["Mo", "S"], [(1 / 3, 2 / 3, 0.25), (1 / 3, 2 / 3, 0.621)], 194, [3.16, 3.16, 12.29, 90, 90, 120]),
    ("graphite", ["C", "C"], [(0, 0, 0.25), (1 / 3, 2 / 3, 0.25)], 194, [2.46, 2.46, 6.71, 90, 90, 120]),
    ("hBN", ["B", "N"], [(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.25)], 194, [2.50, 2.50, 6.66, 90, 90, 120]),
    ("Al2O3", ["Al", "O"], [(0, 0, 0.3523), (0.3064, 0, 0.25)], 167, [4.759, 4.759, 12.991, 90, 90, 120]),
    ("MgAl2O4", ["Mg", "Al", "O"], [(0.125, 0.125, 0.125), (0.5, 0.5, 0.5), (0.2624, 0.2624, 0.2624)], 227, [8.08, 8.08, 8.08, 90, 90, 90]),  # origin choice 2
    ("CaCO3", ["Ca", "C", "O"], [(0, 0, 0), (0, 0, 0.25), (0.2567, 0, 0.25)], 167, [4.99, 4.99, 17.06, 90, 90, 120]),
    ("LiCoO2", ["Li", "Co", "O"], [(0, 0, 0.5), (0, 0, 0), (0, 0, 0.2396)], 166, [2.816, 2.816, 14.05, 90, 90, 120]),
]
reg = pretrained.orb_v3_conservative_inf_omat(weights_path=a.ckpt, device="cpu", precision="float32-high", compile=False)
calc = ORBCalculator(reg, device="cpu")
with open(a.out, "w") as f:
    jobs = [(n, (lambda f=f, c=c, l=l, k=k: bulk(f, c, a=l, **k))) for n, f, c, l, k in SPECS]
    jobs += [(n, (lambda sy=sy, b=b, g=g, cp=cp: crystal(sy, basis=b, spacegroup=g, setting=2 if g == 227 else 1, cellpar=cp, primitive_cell=True))) for n, sy, b, g, cp in SPG]
    for name, build in jobs:
        at = build()
        at.calc = calc
        opt = FIRE(FrechetCellFilter(at), logfile=None)
        conv = opt.run(fmax=a.fmax, steps=a.steps)
        fmax = float(np.abs(at.get_forces()).max())
        s = AseAtomsAdaptor.get_structure(at)
        rec = dict(material_id=f"demo-{name}", formula=s.composition.reduced_formula, structure=s.as_dict(),
                   energy_eV_per_atom=round(float(at.get_potential_energy()) / len(at), 4), relax_steps=int(opt.nsteps),
                   converged=bool(conv), fmax=round(fmax, 4))
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(name, len(at), "atoms", "steps", opt.nsteps, "conv", conv, "a", round(at.cell.cellpar()[0], 3), flush=True)
