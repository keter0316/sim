# routing/spb_ecmp.py
from collections import defaultdict
import networkx as nx

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
