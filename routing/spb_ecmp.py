# routing/spb_ecmp.py
from collections import defaultdict
import networkx as nx
from __future__ import annotations
from typing import List
import networkx as nx
import hashlib

def ecmp_fdb(G: nx.Graph, weight="w"):
    """
    回傳 fdb[src][dst] = [next_hop1, next_hop2, ...]
    """
    fdb = defaultdict(dict)
    for dst in G.nodes:
        dist, preds = nx.dijkstra_predecessor_and_distance(G, dst, weight=weight)
        for src in G.nodes:
            if src == dst:
                continue
            # 沒路就跳過
            if src not in preds or not preds[src]:
                continue
            # 對無向圖：preds[src] 就是「往 dst 的等價下一跳」
            fdb[src][dst] = preds[src]          # <<< 關鍵：寫 preds，不是 dist
    return fdb

def _hash_flow(src: str, dst: str, salt: str = "") -> int:
    h = hashlib.sha256(f"{src}->{dst}|{salt}".encode()).hexdigest()
    return int(h, 16)

def pick_ecmp_path(G: nx.Graph, src: str, dst: str, policy: str = "hash", seed: int = 42, k: int = 8) -> List[str]:
    """在最短路徑集合（等長）中挑一條。
    policy = "hash" | "lex"
    """
    if src not in G or dst not in G:
        raise KeyError(f"unknown nodes: {src},{dst}")
    # 所有最短等長路徑
    try:
        splen = nx.shortest_path_length(G, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return []
    paths = [p for p in nx.all_shortest_paths(G, src, dst, weight="weight")]
    if not paths:
        return []
    if policy == "lex":
        return sorted(paths)[0]
    # hash：保持 flow-consistent
    h = _hash_flow(src, dst, salt=str(seed))
    idx = h % len(paths)
    return paths[idx]
