# routing/steiner.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import networkx as nx

# 可選：若要做實際資源入場控管，從 models.queues 匯入；不存在就跳過
try:
    from models.queues import admit_edge, release_edge, norm_class  # optional
except Exception:  # pragma: no cover
    admit_edge = release_edge = norm_class = None  # type: ignore

# ------------------------------
# 小工具
# ------------------------------

def _edge_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)

def _path_cost(G: nx.Graph, path: List[str], weight: str = "weight") -> float:
    return sum(float(G[path[i]][path[i+1]].get(weight, 1.0)) for i in range(len(path)-1))

def _prune_leaves(T: nx.Graph, terminals: set):
    # 修剪度=1 且非終端的葉子
    changed = True
    while changed:
        changed = False
        for n in list(T.nodes):
            if n not in terminals and T.degree(n) == 1:
                T.remove_node(n)
                changed = True

# ------------------------------
# 1) Root SPT（保留：有些場景會用；本題主角是 MST）
# ------------------------------

def root_spt_tree(G: nx.Graph, root: str, terminals: List[str], weight: str = "weight") -> nx.Graph:
    T = nx.Graph()
    T.add_node(root)
    for t in terminals:
        if t == root:
            continue
        try:
            p = nx.shortest_path(G, root, t, weight=weight)
            T.add_nodes_from(p)
            T.add_edges_from((p[i], p[i+1]) for i in range(len(p)-1))
        except nx.NetworkXNoPath:
            pass
    return T

# ------------------------------
# 2) Steiner 近似：Metric-closure MST
# ------------------------------

def steiner_tree_mst(G: nx.Graph, terminals: List[str], weight: str = "weight") -> nx.Graph:
    Tset = list(dict.fromkeys(terminals))  # 保序去重
    if len(Tset) <= 1:
        T = nx.Graph(); T.add_nodes_from(Tset); return T

    # metric closure
    sp_cache: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}
    K = nx.Graph()
    for i in range(len(Tset)):
        for j in range(i+1, len(Tset)):
            s, t = Tset[i], Tset[j]
            try:
                p = nx.shortest_path(G, s, t, weight=weight)
                c = _path_cost(G, p, weight)
                sp_cache[(s, t)] = (c, p)
                sp_cache[(t, s)] = (c, list(reversed(p)))
                K.add_edge(s, t, weight=c)
            except nx.NetworkXNoPath:
                continue

    if K.number_of_edges() == 0:
        T = nx.Graph(); T.add_nodes_from(Tset); return T

    mst = nx.minimum_spanning_tree(K, weight="weight")

    T = nx.Graph()
    for u, v in mst.edges:
        _, p = sp_cache[(u, v)]
        T.add_nodes_from(p)
        T.add_edges_from((p[k], p[k+1]) for k in range(len(p)-1))

    _prune_leaves(T, set(Tset))
    return T

def steiner_tree_cost(G: nx.Graph, T: nx.Graph, weight: str = "weight") -> float:
    total = 0.0
    for u, v in T.edges:
        total += float(G[u][v].get(weight, 1.0))
    return total

# ------------------------------
# 3) 增量 Greedy Attach
# ------------------------------

def steiner_extend_greedy(G: nx.Graph, terminals: List[str], weight: str = "weight") -> nx.Graph:
    """從第一個 terminal 起步，逐一把其餘節點接到現有樹的最近點。"""
    if not terminals:
        return nx.Graph()
    T = nx.Graph()
    T.add_node(terminals[0])
    present = {terminals[0]}
    remain = [t for t in terminals[1:] if t not in present]

    while remain:
        best = None
        best_cost = float("inf")
        attach = None
        for t in remain:
            for v in T.nodes:
                try:
                    p = nx.shortest_path(G, v, t, weight=weight)
                    c = _path_cost(G, p, weight)
                    if c < best_cost or (c == best_cost and tuple(p) < tuple(best or ())):
                        best_cost = c; best = p; attach = t
                except nx.NetworkXNoPath:
                    continue
        if best is None:
            # 不可達：跳過該點
            remain.pop(0)
            continue
        T.add_nodes_from(best)
        T.add_edges_from((best[i], best[i+1]) for i in range(len(best)-1))
        present.add(attach)
        remain = [t for t in remain if t != attach]

    _prune_leaves(T, set(terminals))
    return T

# ------------------------------
# 4) Root 選擇（以 MST 成本最小作為 root 候選）
# ------------------------------

def choose_root_by_mst_cost(G: nx.Graph, terminals: List[str], candidates: List[str], weight: str = "weight") -> Optional[str]:
    best_root, best_cost = None, float("inf")
    Tset = list(dict.fromkeys(terminals))
    for r in sorted(candidates):
        terms_aug = sorted(set(Tset + [r]))
        T = steiner_tree_mst(G, terms_aug, weight=weight)
        c = steiner_tree_cost(G, T, weight=weight)
        if c < best_cost or (c == best_cost and (best_root is None or r < best_root)):
            best_cost, best_root = c, r
    return best_root

# ------------------------------
# 5) 局部邊替換（single-edge swap）
# ------------------------------

@dataclass
class SwapLog:
    add_edge: Tuple[str, str]
    drop_edge: Tuple[str, str]
    delta_cost: float   # 正值代表成本下降多少
    new_total: float

def local_edge_swaps(
    G: nx.Graph,
    T: nx.Graph,
    *,
    weight: str = "weight",
    tau_pct: float = 0.03,   # 至少改善 3%（相對於「被移除邊」或整體）
    max_iter: int = 64
) -> Tuple[nx.Graph, List[SwapLog]]:
    """
    嘗試以單一非樹邊 e' 取代樹上的某一邊 e（位於同一環上），若 w(e') < w(e)*(1-τ) 則替換。
    以避免抖動，要求明顯改善。
    """
    T = T.copy()
    logs: List[SwapLog] = []

    def edge_w(u, v): return float(G[u][v].get(weight, 1.0))

    it = 0
    improved = True
    while improved and it < max_iter:
        it += 1
        improved = False
        total_before = steiner_tree_cost(G, T, weight=weight)

        # 嘗試所有非樹邊
        for u, v in G.edges:
            if T.has_edge(u, v):
                continue
            # 加入形成唯一環
            T.add_edge(u, v)
            try:
                cycle = nx.find_cycle(T, source=u)
            except nx.NetworkXNoCycle:
                T.remove_edge(u, v)
                continue

            # 在環上尋找可以移除的「最貴」樹邊（但不可是剛加的 e'）
            cyc_edges = [(a, b) for (a, b) in cycle]
            candidate_drop = None
            drop_w = -1.0
            for a, b in cyc_edges:
                if (a == u and b == v) or (a == v and b == u):
                    continue
                w_ab = edge_w(a, b)
                if w_ab > drop_w:
                    drop_w = w_ab
                    candidate_drop = (a, b)

            if candidate_drop is not None:
                add_w = edge_w(u, v)
                # 改善條件：新邊顯著比舊邊便宜
                if add_w < drop_w * (1.0 - tau_pct) - 1e-12:
                    # 接受替換
                    T.remove_edge(*candidate_drop)
                    new_total = steiner_tree_cost(G, T, weight=weight)
                    delta = total_before - new_total
                    logs.append(SwapLog(add_edge=_edge_key(u, v),
                                        drop_edge=_edge_key(*candidate_drop),
                                        delta_cost=max(0.0, delta),
                                        new_total=new_total))
                    improved = True
                else:
                    # 還原
                    T.remove_edge(u, v)
            else:
                T.remove_edge(u, v)

        # 可能經過一輪仍無改善
    _prune_leaves(T, set(T.nodes))  # 只會修剪非終端葉；終端集合在外層控制
    return T, logs

# ------------------------------
# 6) Admission（雙向 SFU 模型；可選）
# ------------------------------

def tree_edges_list(T: nx.Graph) -> List[Tuple[str,str]]:
    return [_edge_key(u, v) for u, v in T.edges]

def tree_from_edges(edges: Iterable[Tuple[str,str]]) -> nx.Graph:
    T = nx.Graph()
    for u, v in edges:
        T.add_edge(u, v)
    return T

# ------------------------------
# 7) I/O & CLI：讀 flows，為每個多播流建樹並輸出 CSV
# ------------------------------

def read_edges_graph(edges_csv: Path, weight: str = "weight") -> nx.Graph:
    G = nx.Graph()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = row["src_switch"].strip()
            v = row["dst_switch"].strip()
            w = float(row.get("weight", 1.0))
            G.add_edge(u, v, weight=w)
    return G

def read_flows(flows_csv: Path) -> List[dict]:
    flows = []
    with flows_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row: 
                continue
            flows.append(row)
    return flows

def parse_subs(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]

def build_tree_for_flow(
    G: nx.Graph,
    pub: str,
    subs: List[str],
    *,
    method: str = "mst",
    weight: str = "weight",
    swap_pct: float = 0.03,
    swap_iter: int = 64
) -> Tuple[nx.Graph, List[SwapLog]]:
    terminals = [pub] + [x for x in subs if x != pub]
    if method == "greedy":
        T = steiner_extend_greedy(G, terminals, weight=weight)
    elif method == "hybrid":
        # 先 MST，再以 greedy 的方式（其實已隱含在 swaps）做補強
        T = steiner_tree_mst(G, terminals, weight=weight)
    else:  # "mst"
        T = steiner_tree_mst(G, terminals, weight=weight)

    # 邊替換（local swap）
    T2, logs = local_edge_swaps(G, T, weight=weight, tau_pct=swap_pct, max_iter=swap_iter)
    # 修剪非終端葉
    _prune_leaves(T2, set(terminals))
    return T2, logs

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Steiner tree (MST / Greedy attach) for Pub/Sub multicast")
    ap.add_argument("--edges", type=str, required=True, help="outputs/matrix/edges_all.csv")
    ap.add_argument("--flows", type=str, required=True, help="configs/flows.csv（需含 subs）")
    ap.add_argument("--out-trees", type=str, default="outputs/flows/trees.csv")
    ap.add_argument("--out-reroutes", type=str, default="outputs/flows/reroutes.csv")
    ap.add_argument("--weight", type=str, default="weight", help="邊權重欄位（預設 weight）")
    ap.add_argument("--mode", choices=["mst","greedy","hybrid"], default="mst", help="建樹策略")
    ap.add_argument("--swap-pct", type=float, default=0.03, help="local swap 改善門檻（百分比）")
    ap.add_argument("--swap-iters", type=int, default=64, help="local swap 最多迭代次數")
    return ap

def main() -> None:
    args = build_cli().parse_args()
    edges_csv = Path(args.edges)
    flows_csv = Path(args.flows)
    out_trees = Path(args.out_trees); out_trees.parent.mkdir(parents=True, exist_ok=True)
    out_rr    = Path(args.out_reroutes); out_rr.parent.mkdir(parents=True, exist_ok=True)

    G = read_edges_graph(edges_csv, weight=args.weight)
    flows = read_flows(flows_csv)

    # trees.csv
    with out_trees.open("w", newline="", encoding="utf-8") as ft, out_rr.open("w", newline="", encoding="utf-8") as fr:
        wt = csv.writer(ft)
        wr = csv.writer(fr)
        wt.writerow(["flow_id","method","pub","subs","num_edges","total_cost","edge_list","swap_count"])
        wr.writerow(["flow_id","step","add_edge","drop_edge","delta_cost","new_total_cost"])

        for row in flows:
            fid  = (row.get("id") or "").strip() or f"flow{len(row)}"
            pub  = (row.get("src") or "").strip()
            topic= (row.get("topic") or "").strip()
            subs = parse_subs(row.get("subs"))

            # 非多播流（或 subs 為空）就跳過
            if not subs:
                continue
            terms = [pub] + subs

            T, logs = build_tree_for_flow(
                G, pub, subs,
                method=args.mode,
                weight=args.weight,
                swap_pct=float(args.swap_pct),
                swap_iter=int(args.swap_iters)
            )
            # 統計
            cost = steiner_tree_cost(G, T, weight=args.weight)
            edges_ser = ";".join(f"{u}-{v}" for u, v in sorted(tree_edges_list(T)))
            wt.writerow([fid, args.mode, pub, ";".join(subs), T.number_of_edges(), f"{cost:.6f}", edges_ser, len(logs)])

            for i, ev in enumerate(logs, 1):
                wr.writerow([
                    fid,
                    i,
                    f"{ev.add_edge[0]}-{ev.add_edge[1]}",
                    f"{ev.drop_edge[0]}-{ev.drop_edge[1]}",
                    f"{ev.delta_cost:.6f}",
                    f"{ev.new_total:.6f}",
                ])

    print(f"[OK] trees written   -> {out_trees}")
    print(f"[OK] reroutes written-> {out_rr}")


if __name__ == "__main__":
    main()
