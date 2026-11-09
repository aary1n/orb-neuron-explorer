"""Direct-path links between Orb-v3 node neurons, read from the weights alone (no data), for the explorer's web.

Pattern from lyramakesmusic/neuron-explorer's build_connections.py (MIT), adapted to Orb-v3's residual stream:
- within a block, layer-1 neuron n feeds layer-2 neuron m with weight NN-1[m, n];
- layer-2 neuron m of the encoder or block i writes the direction w_m = gain_i * NN-2[:, m] into the 256-channel
  node residual (RMSNorm gain applied, the 1/rms scalar left out); layer-1 neuron n of a later block j reads the
  residual through the node slice of its input, r_n = NN-0[n, :256]; the virtual weight is P[n, m] = r_n . w_m.
Each neuron keeps its k strongest links each way, |P| z-scored over the other side, with the sign of P.
Not covered: the message paths (received / sent slices of NN-0 read the edge MLPs, which also read the residual).

    python build/build_connections.py --ckpt <orb-v3 .ckpt> --out data

Writes links_<layer>.json: {"up": [[[layer_index, n, z, sign], ...] per neuron], "down": [...]} (z to 1 decimal).
"""
import argparse, json, os
import numpy as np, torch
from orb_models.forcefield import pretrained

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
reg = pretrained.orb_v3_conservative_inf_omat(weights_path=a.ckpt, device="cpu", precision="float32-high", compile=False)
sd = {k: v.detach().double().numpy() for k, v in reg.state_dict().items()}
pre = ["model._encoder._node_fn"] + [f"model.gnn_stacks.{i}._node_mlp" for i in range(5)]
names = [f"L{L}_{h}" for L in range(6) for h in ("h1", "h2")]
W0 = [sd[p + ".mlp.NN-0.weight"] for p in pre]      # [1024, in]
W1 = [sd[p + ".mlp.NN-1.weight"] for p in pre]      # [1024, 1024]: h2 <- h1
W2 = [sd[p + ".mlp.NN-2.weight"] for p in pre]      # [256, 1024]: residual <- h2
G = [sd[p + ".layer_norm.weight"] for p in pre]     # RMSNorm gain [256]
write = [W2[b] * G[b][:, None] for b in range(6)]   # [256, 1024]: column m = what h2 neuron m adds to the residual
K = a.k
best = {nm: {"up": [[] for _ in range(1024)], "down": [[] for _ in range(1024)]} for nm in names}

def keep(nm, side, P_abs_z, P_sign, other_layer_idx, axis):
    """axis 0: rows are this layer's neurons (links over columns); axis 1: columns are this layer's neurons."""
    Z = P_abs_z if axis == 0 else P_abs_z.T
    S = P_sign if axis == 0 else P_sign.T
    idx = np.argsort(-Z, axis=1)[:, :K]
    for n in range(Z.shape[0]):
        for j in idx[n]:
            best[nm][side][n].append((other_layer_idx, int(j), float(Z[n, j]), int(S[n, j])))

def zrows(M):  # z-score |M| over each row
    A = np.abs(M); return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
def zcols(M):
    A = np.abs(M); return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-9)

for b in range(6):
    h1, h2 = names[2 * b], names[2 * b + 1]
    P = W1[b]                                   # [m (h2), n (h1)]
    keep(h2, "up", zrows(P), np.sign(P), 2 * b, 0)        # h2 neuron m: strongest h1 feeders
    keep(h1, "down", zcols(P), np.sign(P), 2 * b + 1, 1)  # h1 neuron n: strongest h2 readers
for j in range(1, 6):                           # block j reads the residual written by the encoder and blocks < j
    R = W0[j][:, :256]                          # [n (h1 of block j), 256]
    for i in range(0, j):
        P = R @ write[i]                        # [n (h1 of j), m (h2 of i)]
        keep(names[2 * j], "up", zrows(P), np.sign(P), 2 * i + 1, 0)
        keep(names[2 * i + 1], "down", zcols(P), np.sign(P), 2 * j, 1)
for nm in names:
    out = {}
    for side in ("up", "down"):
        rows = []
        for n in range(1024):
            L = sorted(best[nm][side][n], key=lambda t: -t[2])[:K]
            rows.append([[l, m, round(z, 1), s] for l, m, z, s in L])
        out[side] = rows
    json.dump(out, open(os.path.join(a.out, f"links_{nm}.json"), "w"), separators=(",", ":"))
    print(nm, "up", len(out["up"][0]), "down", len(out["down"][0]), flush=True)
print("done")
