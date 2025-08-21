# routing/steiner.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable
import networkx as nx

# 可選：若你要直接在這支檔內做 admission
try:
    from models.queues import admit_edge, release_edge, norm_class  # 已存在於你的專案
except Exception:
    admit_edge = release_edge = norm_class = None  # 允許純演算法使用


# ------------------------------
# 基本工具
# ------------------------------
def _path_cost(G: nx.Graph, path: List[str], weight: str = "w") -> float:
    return sum(G[path[i]][path[i+1]].get(weight, 1.0) for i in range(len(path)-1))

def _prune_leaves(T: nx.Graph, terminals: set):
    prunable = True
    while prunable:
        prunable = False
        for n in list(T.nodes):
            if n not in terminals and T.degree(n) == 1:
                T.remove_node(n)
                prunable = True


# ------------------------------
# root SPT（保留，給 SPT 需求用；本題用 MST）
# ------------------------------
def root_spt_tree(G: nx.Graph, root: str, terminals: List[str], weight: str = "w") -> nx.Graph:
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
# Steiner 近似：metric-closure MST
# ------------------------------
def steiner_tree_mst(G: nx.Graph, terminals: List[str], weight: str = "w") -> nx.Graph:
    Tset = list(dict.fromkeys(terminals))
    if len(Tset) <= 1:
        T = nx.Graph(); T.add_nodes_from(Tset); return T

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
                K.add_edge(s, t, w=c)
            except nx.NetworkXNoPath:
                continue

    if K.number_of_edges() == 0:
        T = nx.Graph(); T.add_nodes_from(Tset); return T

    mst = nx.minimum_spanning_tree(K, weight="w")

    T = nx.Graph()
    for u, v in mst.edges:
        _, p = sp_cache[(u, v)]
        T.add_nodes_from(p)
        T.add_edges_from((p[k], p[k+1]) for k in range(len(p)-1))

    _prune_leaves(T, set(Tset))
    return T

def steiner_tree_cost(G: nx.Graph, terminals: List[str], weight: str = "w") -> float:
    """回傳 Steiner-MST 的總邊權重（展開後的原圖邊，去重後求和）。"""
    T = steiner_tree_mst(G, terminals, weight=weight)
    total = 0.0
    for u, v in T.edges:
        total += float(G[u][v].get(weight, 1.0))
    return total


# ------------------------------
# 逐人加入（增量）
# ------------------------------
def steiner_extend_greedy(G: nx.Graph, T: nx.Graph, new_terms: Iterable[str], weight: str = "w") -> nx.Graph:
    if T.number_of_nodes() == 0:
        return steiner_tree_mst(G, list(new_terms), weight=weight)

    T2 = T.copy()
    for t in new_terms:
        if t in T2:
            continue
        best_cost, best_path = None, None
        for v in T2.nodes:
            try:
                p = nx.shortest_path(G, v, t, weight=weight)
                c = _path_cost(G, p, weight)
                if best_cost is None or c < best_cost:
                    best_cost, best_path = c, p
            except nx.NetworkXNoPath:
                continue
        if best_path:
            T2.add_nodes_from(best_path)
            T2.add_edges_from((best_path[i], best_path[i+1]) for i in range(len(best_path)-1))
    return T2


# ------------------------------
# Root 選擇（MST成本最小）
# ------------------------------
def choose_root_by_medoid(G: nx.Graph, terminals: List[str], weight: str = "w") -> str:
    best, best_sum = terminals[0], float("inf")
    for r in terminals:
        s = 0.0
        ok = True
        for t in terminals:
            if t == r: 
                continue
            try:
                p = nx.shortest_path(G, r, t, weight=weight)
                s += _path_cost(G, p, weight)
            except nx.NetworkXNoPath:
                ok = False; break
        if ok and s < best_sum:
            best, best_sum = r, s
    return best

def choose_root_by_mst_cost(
    G: nx.Graph,
    terminals: List[str],
    candidate_switches: List[str],
    weight: str = "w"
) -> Optional[str]:
    """
    在 candidate_switches（通常為所有 s*）中，挑一個 root，使
      cost(root) = cost(SteinerMST( terminals ∪ {root} ))
    最小。若皆不可達，回傳 None。
    """
    best_root, best_cost = None, float("inf")
    Tset = list(dict.fromkeys(terminals))  # 去重
    for r in candidate_switches:
        terms_aug = sorted(set(Tset + [r]))
        try:
            c = steiner_tree_cost(G, terms_aug, weight=weight)
        except Exception:
            continue
        if c < best_cost:
            best_cost, best_root = c, r
        elif c == best_cost and best_root is not None and str(r) < str(best_root):
            # 穩定化 tie-break：名字小者
            best_root = r
    return best_root


# ------------------------------
# 擁塞感知權重
# ------------------------------
def congestion_weighted_copy(
    G: nx.Graph,
    reserved: Dict[Tuple[str,str], Dict[str,float]],
    usage:    Dict[Tuple[str,str], Dict[str,float]],
    pool_cls: str = "EF",
    alpha: float = 2.0,
    base_weight_attr: str = "w",
    out_attr: str = "w_ca"
) -> nx.Graph:
    H = G.copy()
    for u, v, d in H.edges(data=True):
        ekey = tuple(sorted((u, v)))
        base_w = float(d.get(base_weight_attr, 1.0))
        r_pool = float(reserved.get(ekey, {}).get(pool_cls, 0.0))
        u_pool = float(usage.get(ekey, {}).get(pool_cls, 0.0))
        util = (u_pool / r_pool) if r_pool > 0 else 0.0
        if util < 0: util = 0.0
        d[out_attr] = base_w * (1.0 + alpha * util)
    return H


# ------------------------------
# 載荷計算（SFU 雙向）
# ------------------------------
def tree_edge_loads_bidir(
    T: nx.Graph,
    root: str,
    participants: List[str],
    rate_up_mbps: float,
    rate_down_mbps: float
) -> Dict[Tuple[str, str], float]:
    P = set(participants)
    parent = {root: None}
    order = [root]
    for u in order:
        for v in T.neighbors(u):
            if v in parent: continue
            parent[v] = u
            order.append(v)

    cnt = {n: (1 if n in P else 0) for n in T.nodes}
    for n in reversed(order):
        p = parent[n]
        if p is not None:
            cnt[p] = cnt.get(p, 0) + cnt.get(n, 0)

    loads: Dict[Tuple[str, str], float] = {}
    for n in T.nodes:
        p = parent.get(n)
        if p is None: continue
        ekey = tuple(sorted((n, p)))
        subtree_count = cnt[n]
        loads[ekey] = float(subtree_count) * (float(rate_up_mbps) + float(rate_down_mbps))
    return loads


# ------------------------------
# Admission（初次 / 增量）
# ------------------------------
def _admit_edges_demands(
    cls: str,
    demands: Dict[Tuple[str,str], float],
    reserved,
    usage
) -> Tuple[bool, List[Tuple[str,str]]]:
    if admit_edge is None or release_edge is None:
        raise RuntimeError("admit_edge/release_edge 未可用；請在 models/queues 中匯出後再使用。")

    taken: List[Tuple[str,str]] = []
    pool = "AF" if norm_class and norm_class(cls).startswith("AF") else (norm_class(cls) if norm_class else cls)
    for e, need in demands.items():
        if need <= 0:
            continue
        ok = admit_edge(e, pool, float(need), reserved, usage)
        if not ok:
            for ek in taken:
                release_edge(ek, pool, float(demands.get(ek, 0.0)), reserved, usage)
            return False, taken
        taken.append(e)
    return True, taken

def _release_edges_demands(cls: str, demands: Dict[Tuple[str,str], float], reserved, usage):
    if release_edge is None:
        return
    pool = "AF" if norm_class and norm_class(cls).startswith("AF") else (norm_class(cls) if norm_class else cls)
    for e, need in demands.items():
        if need > 0:
            release_edge(e, pool, float(need), reserved, usage)

def admit_tree_bidir_initial(
    T: nx.Graph,
    cls: str,
    root: str,
    participants: List[str],
    rate_up_mbps: float,
    rate_down_mbps: float,
    reserved,
    usage
) -> Tuple[bool, Dict[Tuple[str,str], float]]:
    loads = tree_edge_loads_bidir(T, root, participants, rate_up_mbps, rate_down_mbps)
    ok, taken = _admit_edges_demands(cls, loads, reserved, usage)
    if not ok:
        return False, loads
    return True, loads

def admit_tree_bidir_incremental(
    T_old: nx.Graph,
    loads_old: Dict[str, float],  # {"u-v": Mbps}
    T_new: nx.Graph,
    cls: str,
    root: str,
    participants: List[str],
    rate_up_mbps: float,
    rate_down_mbps: float,
    reserved,
    usage
) -> Tuple[bool, Dict[Tuple[str,str], float], Dict[Tuple[str,str], float], Dict[Tuple[str,str], float]]:

    loads_old_t: Dict[Tuple[str,str], float] = {}
    for k, v in (loads_old or {}).items():
        a, b = k.split("-", 1)
        ekey = tuple(sorted((a, b)))
        loads_old_t[ekey] = float(v)

    loads_new_t = tree_edge_loads_bidir(T_new, root, participants, rate_up_mbps, rate_down_mbps)

    edges = set(list(T_new.edges()))
    edges |= set(loads_old_t.keys())

    delta_pos: Dict[Tuple[str,str], float] = {}
    delta_neg: Dict[Tuple[str,str], float] = {}

    for e in edges:
        newv = float(loads_new_t.get(e, 0.0))
        oldv = float(loads_old_t.get(e, 0.0))
        if newv > oldv:
            delta_pos[e] = newv - oldv
        elif oldv > newv:
            delta_neg[e] = oldv - newv

    ok, taken = _admit_edges_demands(cls, delta_pos, reserved, usage)
    if not ok:
        return False, loads_new_t, delta_pos, delta_neg

    _release_edges_demands(cls, delta_neg, reserved, usage)
    return True, loads_new_t, delta_pos, delta_neg


# ------------------------------
# Pinned Tree：序列化/還原
# ------------------------------
def tree_edges_list(T: nx.Graph) -> List[Tuple[str,str]]:
    return [tuple(sorted(e)) for e in T.edges]

def tree_from_edges(edges: Iterable[Tuple[str,str]]) -> nx.Graph:
    T = nx.Graph()
    for u, v in edges:
        T.add_edge(u, v)
    return T
