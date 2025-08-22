# models/topology.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import csv
import networkx as nx
import matplotlib.pyplot as plt


# ------------------------- Data classes -------------------------

@dataclass
class EdgeMetrics:
    delay_ms: float = 1.0
    loss: float = 0.0
    bw_mbps: float = 1000.0
    etx: float = 1.0
    rssi: float = -50.0
    snr: float = 30.0


# ------------------------- Topology object -------------------------

@dataclass
class Topology:
    G: nx.Graph = field(default_factory=nx.Graph)
    reserved: Dict[Tuple[str,str], Dict[str,float]] = field(default_factory=dict)
    usage:    Dict[Tuple[str,str], Dict[str,float]] = field(default_factory=dict)

    @staticmethod
    def from_csv(p: Path) -> "Topology":
        """僅讀 edges_all.csv；邊屬性統一用 'weight'。"""
        topo = Topology()
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u, v = row["src_switch"].strip(), row["dst_switch"].strip()
                w = float(row.get("weight", 1.0))
                bw = float(row.get("bw_mbps", 1000.0))
                topo.G.add_edge(
                    u, v,
                    weight=w,                              # <<< 統一用 weight
                    metrics=EdgeMetrics(
                        delay_ms=float(row.get("delay_ms", 1.0) or 1.0),
                        loss=float(row.get("loss", 0.0) or 0.0),
                        bw_mbps=bw,
                        etx=float(row.get("etx", 1.0) or 1.0),
                        rssi=float(row.get("rssi", -50.0) or -50.0),
                        snr=float(row.get("snr", 30.0) or 30.0),
                    )
                )
        return topo
    
    @staticmethod
    def from_edges_and_nodes(edges_csv: Path, nodes_csv: Optional[Path] = None) -> "Topology":
        """（選）同時讀 edges + nodes，把 role 存成 node 屬性 ntype。"""
        topo = Topology.from_csv(edges_csv)
        if nodes_csv and nodes_csv.exists():
            with nodes_csv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n = row["node_id"].strip()
                    role = (row.get("role") or "").strip()
                    if n:
                        topo.G.add_node(n, ntype=role or topo.G.nodes.get(n, {}).get("ntype"))
        return topo
    
    def edge_key(self, u, v):
        return (u, v) if u <= v else (v, u)
    

    def load_nodes_csv(self, nodes_csv: Path) -> None:
        """Attach node roles from nodes.csv (node_id,role)."""
        with nodes_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = row["node_id"].strip()
                role = row.get("role", "").strip() or None
                if nid not in self.G:
                    # If nodes.csv contains nodes not present in edges, still add them
                    self.G.add_node(nid)
                if role:
                    self.G.nodes[nid]["role"] = role

    # ---------- Helpers ----------

    @staticmethod
    def edge_key(u: str, v: str) -> Tuple[str, str]:
        """Normalize undirected edge key (lower lexical first)."""
        return (u, v) if u <= v else (v, u)

    def get_metrics(self, u: str, v: str) -> EdgeMetrics:
        """Return EdgeMetrics for edge (u, v)."""
        data = self.G.get_edge_data(u, v, default=None)
        if not data:
            raise KeyError(f"edge ({u},{v}) not found")
        em = data.get("metrics")
        if isinstance(em, EdgeMetrics):
            return em
        # If stored as dict accidentally, coerce back
        return EdgeMetrics(**(em or {}))

    # ---------- Visualization ----------

    def save_graphviz(
        self,
        out_png: Path,
        *,
        with_labels: bool = True,
        figsize: Tuple[float, float] = (9.0, 9.0),
    ) -> None:
        """
        Save a PNG of the topology with reasonable styling for demo.

        - Node color by role (host/access/core/other)
        - Edge color by type (host-access/access-access/access-core/core-core/other)
        - Works even if nodes don't have 'role' (falls back to spring layout/colors)
        """
        out_png.parent.mkdir(parents=True, exist_ok=True)

        G = self.G
        roles = nx.get_node_attributes(G, "role")

        # Node sets
        hosts  = [n for n, r in roles.items() if r == "host"]
        access = [n for n, r in roles.items() if r == "access"]
        core   = [n for n, r in roles.items() if r == "core"]
        others = [n for n in G.nodes if n not in hosts and n not in access and n not in core]

        # Try to place nicely if roles exist, else fall back to spring_layout
        if access and (hosts or core):
            # Circle layout for access, inner circle for core, hosts near their access
            pos = {}
            pos.update(nx.circular_layout(access, scale=1.0))
            # hosts near their paired access: assume hX—sX if present else scatter
            for a in access:
                ax, ay = pos[a]
                hx = f"h{a[1:]}" if a.startswith("s") else None
                if hx in G:
                    pos[hx] = (ax * 1.25, ay * 1.25)
            if core:
                pos.update(nx.circular_layout(core, scale=0.6))
            # Others: place around
            rem = [n for n in others if n not in pos]
            if rem:
                pos.update(nx.spring_layout(G.subgraph(rem), seed=1))
        else:
            pos = nx.spring_layout(G, seed=1)

        # Edge groups by type
        def edges_of(t: str) -> Iterable[Tuple[str, str]]:
            return [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == t]

        e_host_access = edges_of("host-access")
        e_acc_acc     = edges_of("access-access")
        e_acc_core    = edges_of("access-core")
        e_core_core   = edges_of("core-core")
        e_other       = [(u, v) for u, v in G.edges if (u, v) not in set(e_host_access + e_acc_acc + e_acc_core + e_core_core)]

        # Draw
        plt.figure(figsize=figsize)
        # edges
        if e_acc_acc:
            nx.draw_networkx_edges(G, pos, edgelist=e_acc_acc, width=1.8)
        if e_acc_core:
            nx.draw_networkx_edges(G, pos, edgelist=e_acc_core, width=1.8)
        if e_core_core:
            nx.draw_networkx_edges(G, pos, edgelist=e_core_core, width=2.6)
        if e_host_access:
            nx.draw_networkx_edges(G, pos, edgelist=e_host_access, width=1.2, style="dashed")
        if e_other:
            nx.draw_networkx_edges(G, pos, edgelist=e_other, width=1.5, alpha=0.6)

        # nodes
        if access:
            nx.draw_networkx_nodes(G, pos, nodelist=access, node_size=650)
        if hosts:
            nx.draw_networkx_nodes(G, pos, nodelist=hosts, node_size=420)
        if core:
            nx.draw_networkx_nodes(G, pos, nodelist=core, node_size=720)
        if others:
            nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=520)

        if with_labels:
            nx.draw_networkx_labels(G, pos, font_size=9)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()

    # ---------- Convenience ----------

    def ensure_reserve_entry(self, u: str, v: str, cls: str) -> None:
        """Ensure reserved/usage dicts have entries for (u,v), cls."""
        ek = self.edge_key(u, v)
        if ek not in self.reserved:
            self.reserved[ek] = {}
        if ek not in self.usage:
            self.usage[ek] = {}
        self.reserved[ek].setdefault(cls, 0.0)
        self.usage[ek].setdefault(cls, 0.0)

    def set_reserve(self, u: str, v: str, cls: str, mbps: float) -> None:
        self.ensure_reserve_entry(u, v, cls)
        self.reserved[self.edge_key(u, v)][cls] = float(mbps)

    def add_usage(self, u: str, v: str, cls: str, mbps: float) -> None:
        self.ensure_reserve_entry(u, v, cls)
        self.usage[self.edge_key(u, v)][cls] += float(mbps)

    def get_residual(self, u: str, v: str, cls: str) -> float:
        self.ensure_reserve_entry(u, v, cls)
        ek = self.edge_key(u, v)
        return self.reserved[ek][cls] - self.usage[ek][cls]
