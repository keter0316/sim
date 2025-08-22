# routing/spb_ecmp.py
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping

import networkx as nx

# 我們的 Topology 物件（用來讀 edges_all.csv / nodes.csv）
try:
    from models.topology import Topology
except Exception:
    Topology = None  # 允許僅依賴 NetworkX 的情況（測試時可直接傳 G 進來）


# ------------------------- Core ECMP logic -------------------------

def ecmp_fdb(G: nx.Graph, *, weight: str = "weight") -> Dict[str, Dict[str, List[str]]]:
    """
    建立即時轉發表 FDB：fdb[src][dst] = [next_hop1, next_hop2, ...]
    - 使用 Dijkstra predecessor，source 設為 dst（在無向圖下 preds[src] 即朝向 dst 的等價下一跳）
    - next_hops 做字典序排序，確保穩定輸出
    """
    fdb: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    nodes = list(G.nodes)
    for dst in nodes:
        # 注意：dijkstra_predecessor_and_distance(G, source=dst, ...) 回傳所有節點的前驅集合
        preds, _dist = nx.dijkstra_predecessor_and_distance(G, source=dst, weight=weight)
        for src in nodes:
            if src == dst:
                continue
            nh = preds.get(src, [])
            if not nh:
                continue  # unreachable
            # 穩定排序，避免 Python 集合/內部順序造成抖動
            fdb[src][dst] = sorted(nh)
    return fdb


def _hash_flow(src: str, dst: str, *, seed: int = 0, salt: str = "") -> int:
    """
    以 SHA256 產生可重現的整數 hash。seed/salt 任一更動，結果都會改變。
    """
    s = f"{src}->{dst}|{seed}|{salt}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def pick_ecmp_path(
    G: nx.Graph,
    src: str,
    dst: str,
    *,
    weight: str = "weight",
    policy: str = "hash",
    seed: int = 42,
) -> List[str]:
    """
    在所有「加權最短路」集合中挑一條路徑（以節點列表回傳）。
    policy:
      - "lex"  ：以字典序挑第一條（穩定且可讀）
      - "hash" ：用可重現 hash 對等價集合做 index，flow-consistent
    """
    if src not in G or dst not in G:
        raise KeyError(f"unknown nodes: {src},{dst}")

    try:
        _ = nx.shortest_path_length(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []

    # 取得所有最短路並做穩定排序
    paths = list(nx.all_shortest_paths(G, src, dst, weight=weight))
    if not paths:
        return []

    # 以 tuple 轉成可排序再排序（確保同集合每次順序一致）
    paths = sorted(paths, key=lambda p: tuple(p))

    if policy == "lex":
        return paths[0]

    # policy == "hash"
    h = _hash_flow(src, dst, seed=seed)
    idx = h % len(paths)
    return paths[idx]


# ------------------------- I/O helpers -------------------------

def _fdb_to_keymap(fdb: Mapping[str, Mapping[str, List[str]]]) -> Dict[str, List[str]]:
    """
    轉成 {"src->dst": [next_hops...]}；並確保 next_hops 排序、鍵也排序輸出。
    """
    out: Dict[str, List[str]] = {}
    for src, inner in fdb.items():
        for dst, nhs in inner.items():
            out[f"{src}->{dst}"] = sorted(nhs)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def save_fdb_json(fdb: Mapping[str, Mapping[str, List[str]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keymap = _fdb_to_keymap(fdb)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(keymap, f, ensure_ascii=False, indent=2)
    print(f"[OK] ECMP FDB saved -> {out_path}")


# ------------------------- CLI -------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute SPB/ECMP FDB from edges_all.csv")
    ap.add_argument("--edges", type=str, required=True, help="Path to outputs/matrix/edges_all.csv")
    ap.add_argument("--nodes", type=str, default="", help="Optional nodes.csv for roles (improves layout if later plotted)")
    ap.add_argument("--weight", type=str, default="weight", help="Edge weight attribute name (default: weight)")
    ap.add_argument("--out", type=str, default="outputs/paths/fdb_spb_ecmp.json", help="Output JSON path for FDB")
    ap.add_argument("--seed", type=int, default=1, help="Seed for reproducible hashing in demos (not used for FDB ordering)")
    return ap


def main() -> None:
    args = build_cli().parse_args()
    edges_csv = Path(args.edges)
    nodes_csv = Path(args.nodes) if args.nodes else None
    out_path  = Path(args.out)

    if Topology is not None:
        topo = Topology.from_csv(edges_csv, nodes_csv=nodes_csv)
        G = topo.G
    else:
        # 後備：若無 Topology 類別（獨立測試時），直接讀邊
        import csv
        G = nx.Graph()
        with edges_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u, v = row["src_switch"].strip(), row["dst_switch"].strip()
                w = float(row.get("weight", 1.0))
                G.add_edge(u, v, weight=w)

    fdb = ecmp_fdb(G, weight=args.weight)
    save_fdb_json(fdb, out_path)


if __name__ == "__main__":
    main()
