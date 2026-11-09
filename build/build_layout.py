"""Sheet layouts for the map: each layer's 1,024 neurons on a 32 x 32 grid so that neurons with similar write weights
sit together (pattern from lyramakesmusic/neuron-explorer's build_layout.py, MIT). A layer-1 neuron's write is its
column of NN-1 (what it adds to layer 2); a layer-2 neuron's write is gain * NN-2 column (what it adds to the residual).
Cosine-normalised, PCA to 50, UMAP to 2D, snapped to the grid by rows of 32. Weights only, no data.

    python build/build_layout.py --ckpt <orb-v3 .ckpt> --out data

Writes layout.json: {layer: [cell for each neuron]} with cell = row * 32 + col.
"""
import argparse, json, os
import numpy as np, torch
from orb_models.forcefield import pretrained

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--side", type=int, default=32)
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=20)
a = ap.parse_args()
import umap
os.makedirs(a.out, exist_ok=True)
reg = pretrained.orb_v3_conservative_inf_omat(weights_path=a.ckpt, device="cpu", precision="float32-high", compile=False)
sd = {k: v.detach().double().numpy() for k, v in reg.state_dict().items()}
pre = ["model._encoder._node_fn"] + [f"model.gnn_stacks.{i}._node_mlp" for i in range(5)]
side = a.side; assert side * side == 1024
out = {}
for b, p in enumerate(pre):
    for h, X in (("h1", sd[p + ".mlp.NN-1.weight"].T), ("h2", (sd[p + ".mlp.NN-2.weight"] * sd[p + ".layer_norm.weight"][:, None]).T)):
        nm = f"L{b}_{h}"
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        X = X - X.mean(0)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        Z = U[:, :a.pca] * S[:a.pca]
        xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
        cells = np.zeros(1024, np.int32)
        by_y = np.argsort(xy[:, 1], kind="stable")
        for r in range(side):
            band = by_y[r * side:(r + 1) * side]
            band = band[np.argsort(xy[band, 0], kind="stable")]
            cells[band] = r * side + np.arange(side)
        out[nm] = cells.tolist()
        print(nm, flush=True)
json.dump(out, open(os.path.join(a.out, "layout.json"), "w"), separators=(",", ":"))
print("done")
