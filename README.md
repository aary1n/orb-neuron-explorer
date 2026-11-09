# Orb neuron explorer

A browsable 3D map of the node neurons inside **Orb-v3**, a machine-learned interatomic potential from Orbital Materials (checkpoint `orb-v3-conservative-inf-omat-20250404`). The encoder and five message-passing blocks are drawn as sheets of 1,024 glowing dots, 12,288 units per atom. Click a unit to see which atoms and crystals it fires on, which units feed it and which it feeds (read from the weights), and how its activation lines up with simple atom facts. Pick a crystal to see what fires on each of its atoms in 3D.

**Live demo:** https://aaryans.page/orb-neuron-explorer/

> **What this repository is.** This is the tool, shipped with a small demo data set so it runs out of the box. The demo crystals are textbook structures chosen to show how the explorer works; they are not the data set of my research and the labels they produce are not research findings. The research results built with this tool are being written up for publication and are deliberately not released here.

## Run it

The page is static: no server, GPU or Orb install needed.

```
python -m http.server -d docs 8000
# open http://127.0.0.1:8000
```

It can also be served by GitHub Pages from the `docs/` folder.

For live runs (exact activations on any crystal of the set, live "what fires here", pushes that add to a unit's output and report the change in energy and forces), saved neuron names and a Claude tab, run the local server:

```
pip install fastapi "uvicorn[standard]" orb-models==0.5.5     # torch on CPU is fine
ORB_CKPT=path/to/orb-v3-conservative-inf-omat-20250404.ckpt python server.py
# open http://127.0.0.1:8792
```

The Claude tab needs the `claude` CLI (Claude Code) on PATH; each message is one `claude -p` turn with the read-only tools in `agent_mcp.py` and the prompt in `agent_prompt.md`. Everything runs on the machine that starts the server.

## The demo data

`docs/data/` holds 48 crystals (148 atoms): fcc, bcc and hcp metals, diamond-structure semiconductors, rock-salt and caesium-chloride salts, zinc-blende and wurtzite compounds, fluorites, perovskites, rutiles, corundum, spinel, calcite, LiCoO2 and the layered MoS2, graphite and hexagonal BN. Each was built with ASE from approximate lattice constants and then relaxed (cell and positions, forces below 0.01 eV/Å) with the same Orb-v3 model, so every structure sits at the model's own minimum. The structures are in `docs/data/structures_demo.jsonl`.

With so few atoms the statistics are illustrative. A unit's label describes its strongest atoms in this set; it is not a verified function.

## Build your own data

Any set of crystals works: write one JSON line per crystal, `{"material_id": ..., "formula": ..., "structure": <pymatgen Structure.as_dict()>}`.

```
python build/make_demo_structures.py --ckpt <ckpt> --out structures.jsonl          # or your own file
python build/extract_atoms.py      --ckpt <ckpt> --structures structures.jsonl --out work/acts
python build/build_data.py         --acts work/acts --structures structures.jsonl --ckpt <ckpt> --out docs/data
python build/build_connections.py  --ckpt <ckpt> --out docs/data
python build/build_layout.py       --ckpt <ckpt> --out docs/data
```

The checkpoint downloads from Orbital Materials' public bucket (see `orb_models.forcefield.pretrained`). The build runs on a laptop CPU; the demo set takes under two minutes.

## Reading the page

- **selective** = excess kurtosis of the unit over atoms (spiky units fire on few atoms). **on** = share of atoms above 0.1.
- **z** = activation minus the unit's mean, over its spread, across the set.
- **element η²** = share of the unit's variance explained by the element; the percentile says how many random directions in the same layer it beats.
- **atom facts** = Spearman correlation with coordination number (pymatgen CrystalNN) and nearest-neighbour distance, both centred within element, again against random directions.
- **web** = |weight| z-scored among all units on the other side. Within a block: layer 1 → layer 2. Across blocks: the layer-2 write direction (RMSNorm gain × output column) read by a later layer-1 row on the node slice. Message passing (edge MLPs, attention gates) is not drawn.
- **layout**: "writes" places units with similar write weights together (PCA then UMAP, snapped to a 32 × 32 grid); "index" places them by number.

## Credits and licence

- Design after [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (MIT), with its author's permission: the 3D stack of layer sheets, the glow-point shader, the panel layout, the search and fires modes, the weight web, the "writes" layout and the Claude-over-MCP tab are theirs in spirit and partly in code. Their licence is in `THIRD_PARTY_NOTICES.md`.
- Orb-v3 model and `orb-models` by Orbital Materials (Apache License 2.0). The weights are downloaded at build time and are not in this repository.
- three.js (MIT), loaded from jsDelivr. ASE (LGPL) and pymatgen (MIT) are used by the build scripts.
- This repository is released under the MIT License (see `LICENSE`).
