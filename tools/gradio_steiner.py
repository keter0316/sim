# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, math, io, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# ---------- bootstrap project root ----------
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- project outputs ----------
try:
    from tools.run_flows import BASE
except Exception:
    BASE = Path("outputs")

MATRIX_DIR = BASE / "matrix"
EDGES_ALL  = MATRIX_DIR / "edges_all.csv"
ADJ_SW     = MATRIX_DIR / "adjacency_switches.csv"

# ---------- QoS (for headroom tie-break) ----------
from models.queues import load_qos, build_reservations

# ---------- Steiner/SPT helpers (你的檔) ----------
from routing.steiner import (
    steiner_tree_mst, root_spt_tree,
    choose_root_from_candidates, choose_root_by_medoid
)

# -------------------- utils --------------------
def _norm(x: str) -> str:
    return (x or "").strip().lower()

def _is_host(n: str) -> bool:
    s = str(n).lower()
    return s.startswith("h") and s[1:].isdigit()

def _is_sw(n: str) -> bool:
    s = str(n).lower()
    return s.startswith("s") and s[1:].isdigit()

def _num_key(n: str) -> int:
    return int(re.sub(r"^[a-z]+", "", str(n).lower()) or 0)

# -------------------- graph IO --------------------
def load_graph_from_files() -> Tuple[nx.Graph, Dict[str, str]]:
    """
    以 edges_all.csv 建圖；若有 adjacency_switches.csv 會補 s-s 邊。
    回傳 (G, host_attach)；host_attach[h]=s
    """
    if not EDGES_ALL.exists():
        raise FileNotFoundError(f"not found: {EDGES_ALL}")

    G = nx.Graph()
    host_attach: Dict[str, str] = {}

    with EDGES_ALL.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            u = _norm(r.get("src_switch") or r.get("src") or r.get("u"))
            v = _norm(r.get("dst_switch") or r.get("dst") or r.get("v"))
            if not u or not v:
                continue
            if u.startswith("h") and v.startswith("h"):
                continue
            if not ((_is_sw(u) and _is_sw(v)) or (_is_sw(u) and _is_host(v)) or (_is_host(u) and _is_sw(v))):
                continue

            try: w = float(r.get("weight", 1.0))
            except: w = 1.0
            try: bw = float(r.get("bw_mbps", 10_000.0))
            except: bw = 10_000.0

            G.add_edge(u, v, w=w, bw_mbps=bw, type=_norm(r.get("type", "")))

            t = _norm(r.get("type", ""))
            if t == "host-access" or (_is_host(u) and _is_sw(v)) or (_is_host(v) and _is_sw(u)):
                h = u if _is_host(u) else v
                s = v if _is_sw(v) else u
                host_attach[h] = s

    if ADJ_SW.exists():
        rows = list(csv.reader(ADJ_SW.open("r", encoding="utf-8")))
        if rows:
            header = [_norm(x) for x in rows[0]]
            has_row = header and (header[0] in {"", "node", "id", "switch"})
            names = header[1:] if has_row else header
            for i in range(1, len(rows)):
                row = rows[i]
                if not row: continue
                rname = _norm(row[0]) if has_row else names[i-1]
                for j in range(1 if has_row else (i+1), len(row)):
                    try: val = float(row[j])
                    except: val = 0.0
                    if val <= 0: continue
                    cname = names[j-1 if has_row else j]
                    u, v = _norm(rname), _norm(cname)
                    if not (_is_sw(u) and _is_sw(v)): 
                        continue
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, w=1.0, bw_mbps=10_000.0, type="access-access")
    return G, host_attach

# -------------------- deterministic layout --------------------
def ring_positions(nodes: List[str], radius: float, anchor: Optional[str]) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    if not nodes:
        return pos
    order = list(nodes)
    if anchor in order:
        i0 = order.index(anchor)
        order = order[i0:] + order[:i0]
    n = len(order)
    for i, name in enumerate(order):
        th = 2.0 * math.pi * (i / n)
        pos[name] = (radius * math.cos(th), radius * math.sin(th))
    return pos

def compute_positions(G: nx.Graph, host_attach: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    switches = sorted([n for n in G.nodes if _is_sw(n)], key=_num_key)
    hosts    = sorted([n for n in G.nodes if _is_host(n)], key=_num_key)
    access_sw = sorted(set(host_attach.values()), key=_num_key)
    core_sw   = [s for s in switches if s not in access_sw]

    pos = {}
    pos.update(ring_positions(access_sw, radius=1.0,  anchor=("s1" if "s1" in access_sw else (access_sw[0] if access_sw else None))))
    pos.update(ring_positions(core_sw,   radius=0.55, anchor=(core_sw[0] if core_sw else None)))
    for h, s in host_attach.items():
        if s in pos:
            x, y = pos[s]
            r = math.hypot(x, y) or 1.0
            scale = 1.25 / r
            pos[h] = (x * scale, y * scale)
    return pos

# -------------------- 摘要 --------------------
def summarize_topo(G: nx.Graph, host_attach: Dict[str,str]) -> str:
    switches = sorted([n for n in G.nodes if _is_sw(n)], key=_num_key)
    hosts    = sorted([n for n in G.nodes if _is_host(n)], key=_num_key)
    access_sw = sorted(set(host_attach.values()), key=_num_key)
    core_sw   = [s for s in switches if s not in access_sw]
    lines = []
    lines.append(f"Switches: {len(switches)}   Hosts: {len(hosts)}")
    lines.append(f"Access: {len(access_sw)}  Core: {len(core_sw)}")
    lines.append(f"edges_all: {EDGES_ALL.as_posix()}")
    lines.append(f"adjacency: {ADJ_SW.as_posix() if ADJ_SW.exists() else '(none)'}")
    lines.append("Host attachments:")
    for h in hosts:
        lines.append(f"{h} -> {host_attach.get(h, '?')}")
    return "\n".join(lines)

# -------------------- 畫圖 --------------------
def draw_topology(G: nx.Graph,
                  pos: Dict[str, Tuple[float,float]],
                  tree: nx.Graph | None,
                  root: Optional[str],
                  participants: Iterable[str]) -> Image.Image:
    plt.figure(figsize=(8.8, 8.2))
    ax = plt.gca()
    ax.axis("off")
    nx.draw_networkx_edges(G, pos, width=1.3, edge_color="#d0d0d0", alpha=0.9)
    if tree and tree.number_of_edges() > 0:
        nx.draw_networkx_edges(tree, pos, edgelist=list(tree.edges()), width=4.0, edge_color="#2b6cff")
    switches = [n for n in G.nodes if _is_sw(n)]
    hosts    = [n for n in G.nodes if _is_host(n)]
    nx.draw_networkx_nodes(G, pos, nodelist=switches, node_color="#ffffff", edgecolors="#333", node_size=720, linewidths=1.2)
    nx.draw_networkx_nodes(G, pos, nodelist=hosts,    node_color="#ffffff", edgecolors="#333", node_size=520, linewidths=1.2)
    parts = list(participants or [])
    if parts:
        nx.draw_networkx_nodes(G, pos, nodelist=parts, node_color="#f3c93f", edgecolors="#333", node_size=520, linewidths=1.2)
    if root and root in pos:
        nx.draw_networkx_nodes(G, pos, nodelist=[root], node_color="#e45757", edgecolors="#333", node_size=760, linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# -------------------- 一次更新 --------------------
def run_once(participants: List[str],
             gradual_k: int,
             mode: str,            # "SPT" or "Steiner"
             root_mode: str,       # "AUTO(all+headroom)" or "MANUAL"
             root_pick: Optional[str]):

    G, ha = load_graph_from_files()
    pos = compute_positions(G, ha)

    # 參與者
    hosts_all = sorted([n for n in G.nodes if _is_host(n)], key=_num_key)
    selected = [h for h in hosts_all if h in (participants or [])]
    if gradual_k and gradual_k > 0:
        selected = selected[:min(gradual_k, len(selected))]

    # 候選 root = **所有 switch**（不再限 core）
    switches = sorted([n for n in G.nodes if _is_sw(n)], key=_num_key)
    candidates = switches

    # QoS reserved/usage（tie-break 用 EF 餘額）
    qos = load_qos()
    reserved, usage = build_reservations(G, qos)

    # 選 root
    if root_mode.startswith("MANUAL"):
        root = root_pick if root_pick else (candidates[0] if candidates else None)
    else:  # AUTO(all+headroom)
        root = choose_root_from_candidates(
            G, candidates=candidates, terminals=selected,
            reserved=reserved, usage=usage, pool_cls="EF", weight="w"
        )
        if root is None:
            root = choose_root_by_medoid(G, candidates or switches, weight="w") if (candidates or switches) else None

    # 生成樹
    if mode == "Steiner":
        terms = selected + ([root] if root else [])
        T = steiner_tree_mst(G, terminals=terms, weight="w")
    else:
        T = root_spt_tree(G, root=root, terminals=selected, weight="w") if root else nx.Graph()

    img = draw_topology(G, pos, T, root, selected)
    tree_edges = [tuple(sorted(e)) for e in T.edges]
    info = summarize_topo(G, ha)
    return img, (root or "—"), json.dumps(tree_edges, ensure_ascii=False, indent=2), info

# -------------------- Gradio UI --------------------
def build_ui():
    G, ha = load_graph_from_files()
    hosts = sorted([n for n in G.nodes if _is_host(n)], key=_num_key)
    switches = sorted([n for n in G.nodes if _is_sw(n)], key=_num_key)

    with gr.Blocks(title="Steiner / SPB-SPT 會議樹視覺化（Gradio）") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                hsel = gr.CheckboxGroup(choices=hosts, label="選擇參與者 (h1~hH)")
                gk = gr.Slider(0, len(hosts), value=0, step=1, label="Gradual（前 k 位參與者）")
                mode  = gr.Radio(["SPT", "Steiner"], value="SPT", label="樹型（SPT=SPB per-root）")
                root_mode = gr.Radio(["AUTO(all+headroom)", "MANUAL"], value="AUTO(all+headroom)", label="Root 選擇")
                root_pick = gr.Dropdown(choices=switches, value=(switches[0] if switches else None),
                                        label="手動 Root（MANUAL 時生效）", allow_custom_value=True)
                btn = gr.Button("更新", variant="primary")
            with gr.Column(scale=3):
                canvas = gr.Image(label="Topology & Selected Tree", interactive=False)
            with gr.Column(scale=1, min_width=300):
                info = gr.Textbox(label="拓樸摘要（檢檢）", value=summarize_topo(*load_graph_from_files()), lines=16)

        root_out = gr.Textbox(label="Root / 匯聚點", value="—")
        tree_edges = gr.Textbox(label="Tree Edges", value="[]")

        def _update(hs, k, m, rm, rp):
            return run_once(hs or [], int(k or 0), m, rm, rp)

        btn.click(_update, [hsel, gk, mode, root_mode, root_pick], [canvas, root_out, tree_edges, info])
        demo.load(_update, [hsel, gk, mode, root_mode, root_pick], [canvas, root_out, tree_edges, info])

    return demo

if __name__ == "__main__":
    build_ui().launch()
