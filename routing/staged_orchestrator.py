# routing/staged_orchestrator.py
from __future__ import annotations
from typing import List, Dict, Tuple, Any
import networkx as nx
from pathlib import Path
import json

from routing.grouping import group_vc_multicast
from models.queues import classify  # 你現有的分類函式

# 你現有的演算法
from routing.steiner import steiner_tree_mst, choose_root_by_medoid, tree_edges_list
from routing.spb_ecmp import pick_ecmp_path
from routing.ksp_additional import get_additional_paths

#讀 YAML 策略
def load_staged_policy(path: str | Path) -> Dict[str,Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

#查一路徑是否還有 class=cls 的剩餘保留，若夠則「鎖定」占用。
def admit_path(path: List[str], cls: str, rate: float,
               reserved, usage) -> bool:
    # 直接沿用你現有的 admission；這裡給簡化版
    for i in range(len(path)-1):
        e = tuple(sorted((path[i], path[i+1])))
        cap = float(reserved.get(e, {}).get(cls, 1e30))
        use = float(usage.get((e[0],e[1],cls), 0.0))
        if use + rate > cap: return False
    for i in range(len(path)-1):
        e = tuple(sorted((path[i], path[i+1])))
        usage[(e[0],e[1],cls)] = usage.get((e[0],e[1],cls), 0.0) + rate
    return True

#對一組邊做同樣的檢查與鎖定（給樹用）。
def admit_edges(edges: List[Tuple[str,str]], cls: str, rate: float,
                reserved, usage) -> bool:
    for (u,v) in edges:
        e = tuple(sorted((u,v)))
        cap = float(reserved.get(e, {}).get(cls, 1e30))
        use = float(usage.get((e[0],e[1],cls), 0.0))
        if use + rate > cap: return False
    for (u,v) in edges:
        e = tuple(sorted((u,v)))
        usage[(e[0],e[1],cls)] = usage.get((e[0],e[1],cls), 0.0) + rate
    return True

#在已經生成的樹內，取 src→dst 的最短路徑。
def paths_within_tree(tree: nx.Graph, src: str, dsts: List[str]) -> List[List[str]]:
    paths = []
    for d in dsts:
        paths.append(nx.shortest_path(tree, src, d))
    return paths

def run_staged_routing(G: nx.Graph,
                       flows: List[Dict[str,Any]],
                       ksp: Dict[str,List[List[str]]],
                       reserved, usage,
                       policy: Dict[str,Any],
                       pins_out: Path | None = None) -> List[Dict[str,Any]]:
    results: List[Dict[str,Any]] = [None] * len(flows)  # 每筆 flow 的結果
    stages = policy["stages"]

    # -------- Stage 1：VC 多播（Steiner）先處理並鎖資源 --------
    stage1 = next((s for s in stages if s["algo"]=="steiner_mst_pinned"), None)
    if stage1:
        vc_groups = group_vc_multicast(flows)
        pinned = []
        for key, g in vc_groups.items():
            pubs = list(g["publishers"])
            subs = list(g["subscribers"])
            total_rate = g["rate_mbps"]
            # root
            root = pubs[0]
            rm = stage1.get("root_mode","publisher_access")
            if rm == "auto_medoid":
                root = choose_root_by_medoid(G, pubs + subs)
            elif rm == "core_candidates":
                # 簡單挑距離和最小
                cand = stage1.get("core_candidates", [])
                if cand: 
                    best = None; best_cost = 1e30
                    for c in cand:
                        cost = sum(nx.shortest_path_length(G, c, x) for x in pubs+subs if nx.has_path(G,c,x))
                        if cost < best_cost: best, best_cost = c, cost
                    root = best or root

            # 生成 Steiner tree
            tree = steiner_tree_mst(G, root, subs + pubs)  # 含所有 pub/sub
            edges = tree_edges_list(tree)
            # admission 以「整棵樹」鎖資源（可改為 per-edge 帶寬加總模型）
            if admit_edges(edges, "EF", total_rate, reserved, usage):
                # 樹內每條 flow 的實際 path = tree 中 src->dst 最短路徑
                for i in g["flows"]:
                    f = flows[i]
                    dsts = f.get("dsts") or ([f["dst"]] if f.get("dst") else [])
                    ps = paths_within_tree(tree, f["src"], dsts)
                    results[i] = {"result":"OK","algo":"steiner","class":"EF","paths":ps,"rate":f["rate_mbps"]}
                pinned.append({"session": key, "edges": edges})
            else:
                # 鎖不下，讓這些 flow 留給後續階段嘗試（降級/單播）
                pass

        if stage1.get("pin", False) and pins_out:
            pins_out.write_text(json.dumps(pinned, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------- Stage 2/3：其餘依序處理；依匹配規則選 ECMP 或 Additional --------
    for s in stages:
        if s.get("algo") == "steiner_mst_pinned":
            continue
        for i, f in enumerate(flows):
            if results[i] is not None:
                continue  # 已經在前面處理好了
            # 條件匹配
            topic = f.get("topic",""); cls = classify(topic)
            dsts  = f.get("dsts") or ([f["dst"]] if f.get("dst") else [])
            match = False
            if s.get("match",{}).get("any"): match = True
            if "classes" in s.get("match",{}):
                match = match or (cls in s["match"]["classes"])
            if "topics_prefix" in s.get("match",{}):
                match = match or any(topic.startswith(pfx) for pfx in s["match"]["topics_prefix"])
            if not match: 
                continue

            rate = float(f.get("rate_mbps", 1.0))
            chosen_paths = []
            ok_all = True

            for d in dsts:
                p = None
                if s["algo"] == "spb_ecmp" and pick_ecmp_path:
                    p = pick_ecmp_path(G, f["src"], d, s.get("ecmp",{}))
                    if p and not admit_path(p, cls, rate, reserved, usage):
                        p = None
                if not p:
                    # 先試 KSP（你預先算好的最短路徑集合）
                    key = f"{f['src']}->{d}"
                    for cand in ksp.get(key, []):
                        if admit_path(cand, cls, rate, reserved, usage):
                            p = cand; break
                if not p and s["algo"] == "additional_paths":
                    for cand in get_additional_paths(G, f["src"], d, k=s.get("additional",{}).get("k",6)):
                        if admit_path(cand, cls, rate, reserved, usage):
                            p = cand; break

                if not p:
                    ok_all = False
                    break
                chosen_paths.append(p)

            if ok_all and chosen_paths:
                results[i] = {"result":"OK","algo":s["algo"],"class":cls,"paths":chosen_paths,"rate":rate}

    # 最後補上還沒成功的（標 BLOCKED 或做降級鏈）
    for i, f in enumerate(flows):
        if results[i] is None:
            results[i] = {"result":"BLOCKED","algo":"none","class":classify(f.get("topic","")), "paths":[], "rate":f.get("rate_mbps",1.0)}
    return results
