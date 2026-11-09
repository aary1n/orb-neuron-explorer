"""Stdio MCP server for the explorer's Claude tab: reads neuron data through the local server's HTTP API, runs Orb when it is
loaded, and tells the page what to open. Standard library only; started by `claude -p --mcp-config ...` from server.py.
Pattern after lyramakesmusic/neuron-explorer's agent_mcp.py (MIT)."""
import json, os, sys, urllib.parse, urllib.request

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8792")
LN = {"layer": {"type": ["integer", "string"], "description": "layer index 0-11 (0,1 = encoder h1,h2; 2,3 = block 1; 4,5 = block 2; 6,7 = block 3; 8,9 = block 4; 10,11 = block 5) or its key such as 'L2_h1' (block 2, hidden layer 1 = index 4)"},
      "n": {"type": "integer", "description": "unit number 0-1023"}}
TOOLS = [
    {"name": "overview", "description": "The model, the data set, the 12 layers (live units, median selectivity) and the label keys.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "neuron", "description": "Everything stored for one unit: stats, auto label, label correlations, strongest crystals with per-atom activations, direct weight links each way, and a note if one exists.", "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "search", "description": "Units whose label (auto or saved name) contains the text, e.g. an element symbol or a formula.", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}, "limit": {"type": "integer", "default": 60}}, "required": ["q"]}},
    {"name": "crystal", "description": "A crystal of the set by exact formula or id: facts and the top firing units on each atom (from the stored index).", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
    {"name": "acts", "description": "One unit's activation on every atom of a crystal (stored for the unit's top crystals, otherwise a live Orb run on this machine).", "inputSchema": {"type": "object", "properties": {**LN, "q": {"type": "string", "description": "formula or id"}}, "required": ["layer", "n", "q"]}},
    {"name": "fires", "description": "Live Orb run on a crystal: the top units per atom per layer by z. Needs Orb loaded on this machine.", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}, "k": {"type": "integer", "default": 8}}, "required": ["q"]}},
    {"name": "push", "description": "Live push: add dose (in sd of the unit) to the unit's output on every atom of a crystal; returns the energy change per atom and force changes. Needs Orb loaded.", "inputSchema": {"type": "object", "properties": {**LN, "q": {"type": "string"}, "dose": {"type": "number", "default": 1.0}}, "required": ["layer", "n", "q"]}},
    {"name": "open_neuron", "description": "Open a unit in the person's explorer page.", "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "open_crystal", "description": "Open a crystal (formula or id) in the person's explorer page.", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]


def get(path, **q):
    with urllib.request.urlopen(f"{URL}{path}?{urllib.parse.urlencode(q)}", timeout=600) as r: return json.load(r)


def post(path, body):
    req = urllib.request.Request(f"{URL}{path}", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.load(r)


KEYS = [f"L{i}_h{h}" for i in range(6) for h in (1, 2)]


def _layer(v):
    """Layer index from an index or a key such as 'L2_h1' (block 2, hidden layer 1 = index 4)."""
    if isinstance(v, str) and v.strip() in KEYS: return KEYS.index(v.strip())
    return int(v)


def call(name, a):
    if "layer" in a: a = {**a, "layer": _layer(a["layer"])}
    if name == "overview": return get("/api/overview")
    if name == "neuron": return get("/api/neuron", layer=a["layer"], n=a["n"])
    if name == "search": return get("/api/search", q=a["q"], limit=a.get("limit", 60))
    if name == "crystal": return get("/api/crystal", q=a["q"])
    if name == "acts":
        c = get("/api/crystal", q=a["q"])
        if "error" in c: return c
        cs = get("/api/crystal", q=a["q"])
        idx = _index(a["q"])
        return get("/api/acts", layer=a["layer"], n=a["n"], c=idx)
    if name == "fires": return post("/api/fires", {"id": a["q"], "k": a.get("k", 8)})
    if name == "push": return post("/api/push", {"id": a["q"], "layer": a["layer"], "n": a["n"], "dose": a.get("dose", 1.0)})
    if name in ("open_neuron", "open_crystal"): return {"ok": True, "note": "the page opens it when it sees this tool call"}
    raise ValueError(f"unknown tool {name}")


def _index(q):
    with urllib.request.urlopen(f"{URL}/data/crystals.json", timeout=60) as r: cs = json.load(r)
    ql = q.strip().lower()
    for i, c in enumerate(cs):
        if c["id"] == ql: return i
    for i, c in enumerate(cs):
        if c["f"].lower() == ql: return i
    raise ValueError(f"no crystal {q!r}")


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()


for line in sys.stdin:
    if not line.strip(): continue
    m = json.loads(line); method, id_ = m.get("method"), m.get("id")
    if id_ is None: continue
    if method == "initialize":
        reply(id_, {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}}, "serverInfo": {"name": "explorer", "version": "0.1"}})
    elif method == "tools/list": reply(id_, {"tools": TOOLS})
    elif method == "tools/call":
        p = m["params"]
        try: reply(id_, {"content": [{"type": "text", "text": json.dumps(call(p["name"], p.get("arguments") or {}), ensure_ascii=False)}]})
        except Exception as e: reply(id_, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    elif method == "ping": reply(id_, {})
    else: reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
