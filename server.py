"""Local server for the Orb neuron explorer.

    python server.py                      # static data only (no Orb): http://127.0.0.1:8792
    ORB_CKPT=path/to/orb-v3-conservative-inf-omat-20250404.ckpt python server.py   # + live Orb runs and pushes

Serves docs/ (the page) and docs/data/, keeps neuron names in docs/data/names.json, and, with orb-models installed
and a checkpoint given, runs Orb on any crystal of the set (exact activations, "what fires here", pushes). The Claude tab
runs the `claude` CLI (Claude Code) as a subprocess with the tools in agent_mcp.py, the way lyramakesmusic/neuron-explorer
does (MIT); it is on when `claude` is on PATH. Everything runs on the machine that starts this file.
"""
import asyncio, json, os, shutil, sys, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "docs"); DATA = os.path.join(STATIC, "data")
PORT = int(os.environ.get("PORT", "8792"))
NAMES = os.path.join(DATA, "names.json")
CHAT_DIR = os.path.join(DATA, "chats")  # claude's cwd: nothing of the repo's settings applies there
app = FastAPI()
_cache = {}


def J(name):
    if name not in _cache:
        with open(os.path.join(DATA, name)) as f:
            _cache[name] = json.load(f)
    return _cache[name]


ARCH = J("arch.json")
L = ARCH["layers"]
KEY = {y["key"]: i for i, y in enumerate(L)}


def units(l): return J(f"units_{L[l]['key']}.json")
def links(l): return J(f"links_{L[l]['key']}.json")
def crystals(): return J("crystals.json")


@app.middleware("http")
async def local_only(request, call_next):
    if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
        return JSONResponse({"error": "local only"}, status_code=403)
    r = await call_next(request)
    r.headers["cache-control"] = "no-store"
    return r


# ---------- live Orb (optional) ----------
_orb = {"reg": None, "mods": None, "err": None}


def orb():
    """The regressor and its 12 hook points, loaded once; None when orb-models or the checkpoint is missing."""
    if _orb["reg"] is not None or _orb["err"]:
        return _orb["reg"]
    ck = os.environ.get("ORB_CKPT", "")
    if not ck or not os.path.exists(ck):
        _orb["err"] = "set ORB_CKPT to the orb-v3-conservative-inf-omat checkpoint"
        return None
    try:
        import torch
        from orb_models.forcefield import pretrained
        torch.set_num_threads(int(os.environ.get("NT", "4")))
        reg = pretrained.orb_v3_conservative_inf_omat(weights_path=ck, device="cpu", precision="float32-high", compile=False)
        reg.eval()
        mods = dict(reg.named_modules())
        hooks = []
        for Lx, prefix in enumerate(["model._encoder._node_fn.mlp"] + [f"model.gnn_stacks.{i}._node_mlp.mlp" for i in range(5)]):
            for act in ("Act-0", "Act-1"):
                hooks.append(mods[f"{prefix}.{act}"])
        _orb["reg"], _orb["mods"] = reg, hooks
    except Exception as e:  # noqa
        _orb["err"] = f"{type(e).__name__}: {e}"
    return _orb["reg"]


def atoms_of(c):
    """ASE Atoms from a crystals.json record (lattice rows in Å, fractional sites)."""
    from ase import Atoms
    return Atoms([s[0] for s in c["sites"]], scaled_positions=[s[1:4] for s in c["sites"]], cell=c["lat"], pbc=True)


def run_orb(atoms, push=None):
    """Forward pass; returns (acts [n,12,1024] float32, energy eV, forces [n,3]). push = (layer, n, delta) adds delta to that
    unit's post-SiLU output on every atom."""
    import numpy as np, torch
    from orb_models.forcefield import atomic_system
    reg = orb()
    g = atomic_system.ase_atoms_to_atom_graphs(atoms, system_config=reg.system_config, device="cpu")
    buf, handles = {}, []
    for i, m in enumerate(_orb["mods"]):
        def mk(i):
            def hook(mod, inp, out):
                if push and push[0] == i:
                    out = out.clone(); out[:, push[1]] = out[:, push[1]] + push[2]
                buf[i] = out.detach().float().numpy()
                return out if push and push[0] == i else None
            return hook
        handles.append(m.register_forward_hook(mk(i)))
    try:
        out = reg.predict(g)
    finally:
        for h in handles: h.remove()
    e = out["energy"]
    f = out.get("grad_forces", out.get("forces"))  # orb-models 0.5.5 names the conservative forces grad_forces
    acts = np.stack([buf[i] for i in range(12)], axis=1)
    return acts, float(torch.as_tensor(e).reshape(-1)[0]), torch.as_tensor(f).detach().cpu().numpy()


def find_crystal(q):
    q = str(q).strip().lower()
    cs = crystals()
    for i, c in enumerate(cs):
        if c["id"] == q: return i
    for i, c in enumerate(cs):
        if c["f"].lower() == q: return i
    return None


# ---------- API ----------
@app.get("/api/status")
def status():
    names = json.load(open(NAMES)) if os.path.exists(NAMES) else {}
    live = orb() is not None
    return {"live": live, "orb_error": None if live else _orb["err"], "claude": bool(shutil.which("claude")), "names": names, "built": ARCH["built"]}


@app.post("/api/name")
async def set_name(req: Request):
    b = await req.json()
    names = json.load(open(NAMES)) if os.path.exists(NAMES) else {}
    if b.get("name"): names[b["key"]] = b["name"]
    else: names.pop(b["key"], None)
    json.dump(names, open(NAMES, "w"), indent=0, ensure_ascii=False)
    return {"ok": True, "n": len(names)}


@app.get("/api/overview")
def overview():
    return dict(ARCH)


@app.get("/api/neuron")
def neuron(layer: int, n: int):
    u = dict(units(layer)[n]); cs = crystals()
    u["layer"] = L[layer]["key"]
    u["top_crystals"] = [{"id": cs[ci]["id"], "formula": cs[ci]["f"], "spacegroup": cs[ci]["sg"], "acts_by_atom": [[s[0], round(a, 3)] for s, a in zip(cs[ci]["sites"], acts)]} for ci, acts in u.pop("tc", [])]
    u.pop("top", None)
    lk = links(layer)
    nm = lambda l, m: (json.load(open(NAMES)) if os.path.exists(NAMES) else {}).get(f"{L[l]['key']}/{m}") or J("all_units.json")[L[l]["key"]]["name"][m]
    u["fed_by"] = [{"unit": f"{L[l]['key']} #{m}", "z": z, "sign": s, "label": nm(l, m)} for l, m, z, s in lk["up"][n]]
    u["feeds"] = [{"unit": f"{L[l]['key']} #{m}", "z": z, "sign": s, "label": nm(l, m)} for l, m, z, s in lk["down"][n]]
    notes = J("notes.json") if os.path.exists(os.path.join(DATA, "notes.json")) else {}
    u["note"] = notes.get(f"{L[layer]['key']}/{n}")
    return u


@app.get("/api/search")
def search(q: str, limit: int = 60):
    ql = q.lower(); al = J("all_units.json"); names = json.load(open(NAMES)) if os.path.exists(NAMES) else {}
    out = []
    for l, y in enumerate(L):
        a = al[y["key"]]
        for n in range(ARCH["N"]):
            if not a["live"][n]: continue
            nm = (names.get(f"{y['key']}/{n}", "") + " " + a["name"][n])
            if ql in nm.lower():
                out.append({"unit": f"{y['key']} #{n}", "layer": l, "n": n, "label": nm.strip(), "kurt": a["kurt"][n], "on": a["on"][n]})
    out.sort(key=lambda r: -(r["kurt"] or 0))
    return out[:limit]


def fires_rows(ci, top=6):
    import numpy as np
    F = _cache.setdefault("fires", np.fromfile(os.path.join(DATA, "fires.bin"), dtype="<u2").reshape(-1, len(L), ARCH["fires_k"], 2))
    c = crystals()[ci]; out = []; al = J("all_units.json")
    for j, s in enumerate(c["sites"]):
        best = []
        for l in range(len(L)):
            for n, z in F[c["a0"] + j, l, :4]:
                if (al[L[l]["key"]]["max"][int(n)] or 0) >= 0.1: best.append((float(z) / 10, l, int(n)))
        best.sort(reverse=True)
        out.append({"atom": f"{s[0]} #{j}", "fires": [{"unit": f"{L[l]['key']} #{n}", "z": z} for z, l, n in best[:top]]})
    return out


@app.get("/api/crystal")
def crystal(q: str):
    ci = find_crystal(q)
    if ci is None: return {"error": f"no crystal {q!r} in the set (formula or id, exact)"}
    c = dict(crystals()[ci]); c.pop("sites", None); c.pop("lat", None)
    c["atoms"] = fires_rows(ci)
    return c


@app.get("/api/acts")
def acts(layer: int, n: int, c: int):
    cs = crystals()[c]
    for ci, a in units(layer)[n].get("tc", []):
        if ci == c: return {"acts": a, "source": "stored"}
    if orb() is None: return JSONResponse({"error": _orb["err"]}, status_code=503)
    A, e, f = run_orb(atoms_of(cs))
    return {"acts": [round(float(x), 4) for x in A[:, layer, n]], "source": "live"}


@app.post("/api/fires")
async def fires_live(req: Request):
    """{id|formula} -> top-k firing units per atom per layer from a live run (z against the stored unit stats)."""
    import numpy as np
    b = await req.json(); ci = find_crystal(b.get("id") or b.get("formula", ""))
    if ci is None: return {"error": "no such crystal"}
    if orb() is None: return JSONResponse({"error": _orb["err"]}, status_code=503)
    A, e, f = run_orb(atoms_of(crystals()[ci]))
    out = []
    for l in range(len(L)):
        us = units(l); mean = np.array([u["mean"] or 0 for u in us]); sd = np.array([u["sd"] or 1 for u in us]); sd[sd == 0] = 1
        z = (A[:, l, :] - mean) / sd
        for j in range(z.shape[0]):
            idx = np.argsort(-z[j])[:int(b.get("k", 8))]
            out.append({"atom": j, "layer": L[l]["key"], "top": [[int(n), round(float(z[j, n]), 2), round(float(A[j, l, n]), 3)] for n in idx]})
    return {"energy_eV": e, "n_atoms": int(A.shape[0]), "rows": out}


@app.post("/api/push")
async def push(req: Request):
    """{c (index) | id, layer, n, dose (in sd of the unit, default 1)} -> energy and force change from adding dose*sd on every atom."""
    import numpy as np
    b = await req.json()
    ci = b.get("c") if b.get("c") is not None else find_crystal(b.get("id", ""))
    if ci is None: return {"error": "no such crystal"}
    if orb() is None: return JSONResponse({"error": _orb["err"]}, status_code=503)
    l, n, dose = int(b["layer"]), int(b["n"]), float(b.get("dose", 1.0))
    u = units(l)[n]; delta = dose * (u["sd"] or 0)
    at = atoms_of(crystals()[ci])
    A0, e0, f0 = run_orb(at); A1, e1, f1 = run_orb(at, push=(l, n, delta))
    df = np.linalg.norm(f1 - f0, axis=1)
    return {"crystal": crystals()[ci]["f"], "unit": f"{L[l]['key']} #{n}", "dose_sd": dose, "delta_act": delta,
            "dE_meV_per_atom": round(1000 * (e1 - e0) / len(at), 3), "max_force_change_eV_A": round(float(df.max()), 4),
            "force_change_by_atom": [[s, round(float(x), 4)] for s, x in zip(at.get_chemical_symbols(), df)]}


# ---------- Claude tab: one `claude -p` turn per message, resumed by session id ----------
def claude_cmd(chat, prompt):
    mcp = {"mcpServers": {"explorer": {"command": sys.executable, "args": [os.path.join(HERE, "agent_mcp.py")],
                                       "env": {"EXPLORER_URL": f"http://127.0.0.1:{PORT}"}}}}
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--system-prompt-file", os.path.join(HERE, "agent_prompt.md"), "--mcp-config", json.dumps(mcp), "--strict-mcp-config",
           "--tools", "", "--allowedTools", "mcp__explorer", "--setting-sources", ""]
    if os.environ.get("EXPLORER_CLAUDE_MODEL"): cmd += ["--model", os.environ["EXPLORER_CLAUDE_MODEL"]]
    seen = os.path.join(CHAT_DIR, f"{chat}.started")
    cmd += ["--resume", chat] if os.path.exists(seen) else ["--session-id", chat]
    return cmd, seen


@app.post("/api/agent/send")
async def agent_send(req: Request):
    b = await req.json()
    chat = str(uuid.UUID(str(b.get("chat") or uuid.uuid4())))
    msg = b["message"]
    ctx = []
    if b.get("layer") is not None and b.get("n") is not None: ctx.append(f"the neuron {L[int(b['layer'])]['key']} #{int(b['n'])} open (tool arguments: layer={int(b['layer'])}, n={int(b['n'])})")
    if b.get("crystal") is not None: ctx.append(f"the crystal {crystals()[int(b['crystal'])]['f']} ({crystals()[int(b['crystal'])]['id']}) open")
    if ctx: msg = f"[The person has {' and '.join(ctx)} in the explorer.]\n\n{msg}"
    os.makedirs(CHAT_DIR, exist_ok=True)
    if not shutil.which("claude"):
        return StreamingResponse(iter([json.dumps({"t": "error", "text": "claude CLI not found on PATH"}) + "\n"]), media_type="application/x-ndjson")
    cmd, seen = claude_cmd(chat, msg)

    async def gen():
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=CHAT_DIR, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        open(seen, "w").write(str(time.time()))
        async for line in proc.stdout:
            try: m = json.loads(line)
            except ValueError: continue
            if m.get("type") == "stream_event":
                e = m["event"]
                if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta":
                    yield json.dumps({"t": "text", "text": e["delta"]["text"]}) + "\n"
            elif m.get("type") == "assistant":
                for blk in m["message"].get("content", []):
                    if blk.get("type") == "tool_use":
                        yield json.dumps({"t": "tool", "name": blk["name"].removeprefix("mcp__explorer__"), "input": blk.get("input")}) + "\n"
            elif m.get("type") == "result":
                if m.get("is_error"): yield json.dumps({"t": "error", "text": str(m.get("result"))[:600]}) + "\n"
                yield json.dumps({"t": "done", "cost": m.get("total_cost_usd")}) + "\n"
        err = (await proc.stderr.read()).decode(errors="replace").strip()
        code = await proc.wait()
        if code not in (0, None) and err: yield json.dumps({"t": "error", "text": err[-600:]}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------- files ----------
app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"Orb neuron explorer: http://127.0.0.1:{PORT}  (Orb: {'on' if orb() else 'off: ' + str(_orb['err'])}; claude CLI: {'found' if shutil.which('claude') else 'not found'})")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
