# tools/step2_3_build_paths.py
"""
產生 FDB 和 KSP 路徑
"""
import csv, json, os
from pathlib import Path
import networkx as nx
from collections import defaultdict
from heapq import heappush, heappop

BASE = Path("outputs")
EDGE_EL = BASE / "matrix/edges_all.csv"
ADJ_MAT = BASE / "matrix/adjacency_switches.csv"
OUT_DIR = BASE / "paths"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BW = 10000.0  # Mbps

def load_edge_list_csv(path: Path) -> nx.Graph:
    G = nx.Graph()
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)

        for r in rdr:
            # 讀固定欄位：src_switch / dst_switch
            try:
                u = r["src_switch"].strip()
                v = r["dst_switch"].strip()
            except KeyError as e:
                raise KeyError(
                    f"缺少欄位 {e}. 請確認 edges_all.csv 表頭為 "
                    "'src_switch,dst_switch,weight,bw_mbps,type'"
                )

            if not u or not v:
                continue  # 跳過空列

            # 權重與頻寬（給預設）
            try:
                w = float(r.get("weight", 1.0))
            except Exception:
                w = 1.0
            try:
                bw = float(r.get("bw_mbps", 10000.0))
            except Exception:
                bw = 10000.0

            G.add_edge(u, v, w=w, bw_mbps=bw)

    return G



def load_adj_matrix_csv(path: Path) -> nx.Graph:
    G = nx.Graph()
    rows = list(csv.reader(path.open(encoding="utf-8")))
    hdr = [h.strip() for h in rows[0][1:]]
    for i, row in enumerate(rows[1:]):
        u = row[0].strip()
        for j, cell in enumerate(row[1:]):
            v = hdr[j]
            if u == v: 
                continue
            cell = cell.strip()
            if not cell or cell.upper() in ("NA", "INF", "NAN"): 
                continue
            w = float(cell)
            if w == 0: 
                continue
            if G.has_edge(u, v): 
                continue
            G.add_edge(u, v,
                       w=w,
                       bw_mbps=DEFAULT_BW,
                       delay_ms=w,
                       loss=0.0)
    return G

def load_topology() -> nx.Graph:
    if EDGE_EL.exists():
        return load_edge_list_csv(EDGE_EL)
    elif ADJ_MAT.exists():
        return load_adj_matrix_csv(ADJ_MAT)
    else:
        raise SystemExit("No topology file found.")

def health(G: nx.Graph):
    comps = list(nx.connected_components(G))
    print(f"#nodes={G.number_of_nodes()}  #edges={G.number_of_edges()}  #components={len(comps)}")
    if len(comps) > 1:
        print("WARN: topology is not fully connected:", [len(c) for c in comps])
    ws = [d["w"] for *_, d in G.edges(data=True)]
    print(f"w min/avg/max: {min(ws):.3f}/{sum(ws)/len(ws):.3f}/{max(ws):.3f}")

def is_switch(n: str) -> bool:
    return n.startswith("s")

def ecmp_fdb(G: nx.Graph, weight="w"):
    from collections import defaultdict
    fdb = defaultdict(dict)

    # 只針對 switch，並固定順序讓輸出可重現
    nodes = sorted([n for n in G.nodes if is_switch(n)])
    SG = G.subgraph(nodes).copy()

    for dst in nodes:
        # 注意回傳順序：preds 在前、dist 在後！
        preds, dist = nx.dijkstra_predecessor_and_distance(SG, dst, weight=weight)

        for src in nodes:
            if src == dst:
                continue
            nh = preds.get(src, [])        # list of next-hops
            if not nh:
                continue
            fdb[src][dst] = sorted(set(nh))  # 排序去重，輸出穩定
    return fdb


def yen_ksp(G: nx.Graph, src: str, dst: str, K=3, weight="w"):
    try:
        p0 = nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []
    A = [p0]; B = []
    def path_cost(p): 
        return sum(G[p[i]][p[i+1]][weight] for i in range(len(p)-1))
    for _ in range(1, K):
        root = A[-1]
        for i in range(len(root) - 1):
            spur_node = root[i]
            root_path = root[:i+1]
            removed = []
            for p in A:
                if len(p) > i and p[:i+1] == root_path:
                    u, v = p[i], p[i+1]
                    if G.has_edge(u, v):
                        attr = dict(G[u][v])       # <<< 一定要 copy
                        G.remove_edge(u, v)
                        removed.append((u, v, attr))
            try:
                spur = nx.shortest_path(G, spur_node, dst, weight=weight)
                cand = root_path[:-1] + spur
                heappush(B, (path_cost(cand), cand))
            except nx.NetworkXNoPath:
                pass
            finally:
                for u, v, attr in removed:
                    G.add_edge(u, v, **attr)
        if not B: 
            break
        _, path = heappop(B)
        A.append(path)
    return A


def to_plain(d):
    # defaultdict -> dict（遞迴）
    return {k: (to_plain(v) if isinstance(v, dict) else v) for k, v in d.items()}

def main():
    G = load_topology()
    for u, v in G.edges:
        G[u][v]["bw_mbps"] = DEFAULT_BW
    health(G)

    # 只針對 switch
    nodes = sorted([n for n in G.nodes if n.startswith("s")])

    # Step 2: FDB
    fdb = ecmp_fdb(G, weight="w")
    (OUT_DIR / "fdb_spb_ecmp.json").write_text(
        json.dumps(to_plain(fdb), indent=2, ensure_ascii=False)
    )
    print("[OK] wrote", OUT_DIR / "fdb_spb_ecmp.json")

    # Step 3: KSP（純 switch 組合）
    K = int(os.environ.get("KSP_K", 3))
    all_ksp = {}
    for i, s in enumerate(nodes):
        for t in nodes[i+1:]:
            paths = yen_ksp(G, s, t, K=K, weight="w")
            if paths:
                all_ksp[f"{s}->{t}"] = paths
    (OUT_DIR / "ksp_all_pairs.json").write_text(
        json.dumps(all_ksp, indent=2, ensure_ascii=False)
    )
    print("[OK] wrote", OUT_DIR / "ksp_all_pairs.json")


if __name__ == "__main__":
    main()
