# routing/ksp_additional.py
import networkx as nx
from heapq import heappush, heappop

def yen_ksp(G: nx.Graph, src: str, dst: str, K=3, weight="w"):
    # 取最短路
    try:
        path0 = nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []
    A = [path0]; B = []
    for k in range(1, K):
        for i in range(len(A[-1]) - 1):
            spur_node = A[-1][i]
            root_path = A[-1][:i+1]
            removed = []
            for p in A:
                if len(p) > i and p[:i+1] == root_path:
                    u,v = p[i], p[i+1]
                    if G.has_edge(u,v):
                        attr = G[u][v]
                        G.remove_edge(u,v)
                        removed.append((u,v,attr))
            try:
                spur_path = nx.shortest_path(G, spur_node, dst, weight=weight)
                candidate = root_path[:-1] + spur_path
                # cost
                cost = sum(G[candidate[j]][candidate[j+1]][weight] for j in range(len(candidate)-1))
                heappush(B, (cost, candidate))
            except nx.NetworkXNoPath:
                pass
            for u,v,attr in removed:
                G.add_edge(u,v,**attr)
        if not B: break
        cost, path = heappop(B)
        A.append(path)
    return A
