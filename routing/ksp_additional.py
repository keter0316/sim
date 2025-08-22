# routing/ksp_additional.py
from __future__ import annotations

import argparse
import json
from heapq import heappush, heappop
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx


# ------------------------- Utilities -------------------------

def _edge_cost(G: nx.Graph, u: str, v: str, weight: str) -> float:
    """Get edge weight with safe fallback to 1.0 if attribute missing."""
    data = G.get_edge_data(u, v, default={})
    if data is None:
        # MultiGraph not expected; simple fallback
        return 1.0
    w = data.get(weight)
    if w is None:
        w = data.get("weight", 1.0)
    return float(w)


def path_cost(G: nx.Graph, path: List[str], weight: str = "weight") -> float:
    """Sum of weights along path on the ORIGINAL graph G."""
    if len(path) < 2:
        return 0.0
    return sum(_edge_cost(G, path[i], path[i + 1], weight) for i in range(len(path) - 1))


# ------------------------- Yen's KSP (robust) -------------------------

def yen_ksp(G: nx.Graph, src: str, dst: str, K: int = 3, weight: str = "weight") -> List[List[str]]:
    """
    Yen's algorithm for K shortest loopless paths.
    - Works on an undirected weighted graph.
    - Does NOT mutate the original graph (spur computations use a copy).
    - Stable candidate ordering via (cost, tuple(path)).
    """
    if src == dst:
        return [[src]]

    # First shortest path
    try:
        p0 = nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []

    A: List[List[str]] = [p0]     # finalized paths
    B: List[Tuple[float, Tuple[str, ...]]] = []  # candidates heap: (cost, path_tuple)
    seen: set[Tuple[str, ...]] = {tuple(p0)}

    for k in range(1, max(1, K)):
        prev_path = A[-1]
        # Spur at each node of the previous path (except the last)
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            # Build a modified graph H for this spur
            H = G.copy()

            # Remove the edges that would create the same root_path prefix in any previously accepted path
            for p in A:
                if len(p) > i and p[: i + 1] == root_path:
                    u, v = p[i], p[i + 1]
                    if H.has_edge(u, v):
                        H.remove_edge(u, v)

            # Remove nodes in the root_path except spur_node to avoid loops
            for n in root_path[:-1]:
                if H.has_node(n):
                    H.remove_node(n)

            # Compute spur path in H
            try:
                spur_path = nx.shortest_path(H, spur_node, dst, weight=weight)
            except nx.NetworkXNoPath:
                continue

            candidate = root_path[:-1] + spur_path
            cand_t = tuple(candidate)
            if cand_t in seen:
                continue
            seen.add(cand_t)

            cost = path_cost(G, candidate, weight=weight)
            heappush(B, (cost, cand_t))

        # No candidates left
        if not B:
            break

        # Pop the lowest cost candidate that is not already chosen
        cost, cand = heappop(B)
        A.append(list(cand))

    # Cap to K paths (in case duplicates filtered reduced count)
    return A[:K]


# ------------------------- All-pairs wrapper -------------------------

def ksp_all_pairs(
    G: nx.Graph,
    K: int = 3,
    *,
    weight: str = "weight",
    limit_dst: Optional[int] = None,
    sources: Optional[Iterable[str]] = None,
) -> Dict[str, List[List[str]]]:
    """
    Compute K shortest paths for all source-destination pairs and return:
      {"src->dst": [[p1...],[p2...], ...], ...}
    - limit_dst: for each source, only compute the first N destinations (sorted order) to reduce runtime
    - sources: optional iterable of sources to restrict computation
    """
    nodes = sorted(G.nodes())
    src_list = sorted(sources) if sources is not None else nodes

    out: Dict[str, List[List[str]]] = {}
    for src in src_list:
        # Choose destination list
        dsts = [n for n in nodes if n != src]
        if limit_dst is not None:
            dsts = dsts[: max(0, int(limit_dst))]

        for dst in dsts:
            paths = yen_ksp(G, src, dst, K=K, weight=weight)
            out[f"{src}->{dst}"] = paths

    return out


# ------------------------- I/O helpers & CLI -------------------------

def save_ksp_json(ksp_map: Mapping[str, List[List[str]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ksp_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] KSP (all-pairs) saved -> {out_path}")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute all-pairs Yen K-shortest paths")
    ap.add_argument("--edges", type=str, required=True, help="Path to outputs/matrix/edges_all.csv")
    ap.add_argument("--weight", type=str, default="weight", help="Edge weight attr (default: weight)")
    ap.add_argument("--K",      type=int, default=3, help="Number of paths per (src,dst)")
    ap.add_argument("--limit-dst", type=int, default=0, help="Limit number of destinations per source (0 = all)")
    ap.add_argument("--sources", type=str, default="", help="Comma-separated list of sources; default all nodes")
    ap.add_argument("--out", type=str, default="outputs/paths/ksp_all_pairs.json", help="Output JSON path")
    return ap


def main() -> None:
    args = build_cli().parse_args()
    edges_csv = Path(args.edges)
    out_path  = Path(args.out)
    weight    = args.weight
    K         = max(1, int(args.K))
    limit_dst = int(args.limit_dst) if int(args.limit_dst) > 0 else None
    sources   = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None

    # Load graph (lightweight, just weight attr)
    import csv as _csv
    G = nx.Graph()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            u = row["src_switch"].strip()
            v = row["dst_switch"].strip()
            w = float(row.get("weight", 1.0))
            G.add_edge(u, v, weight=w)

    ksp_map = ksp_all_pairs(G, K=K, weight=weight, limit_dst=limit_dst, sources=sources)
    save_ksp_json(ksp_map, out_path)


if __name__ == "__main__":
    main()
