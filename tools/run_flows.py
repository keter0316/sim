# tools/run_flows.py
# --- bootstrapping sys.path so this file works when run directly ---
from __future__ import annotations

import sys, re, json, csv
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]   # .../sim
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- end bootstrap ---

from models.queues import (
    load_qos, classify, build_reservations,
    admit_edge, release_edge, DOWNGRADE_CHAIN
)
from routing.steiner import (
    steiner_tree_mst, steiner_extend_greedy,
    choose_root_by_mst_cost,   # 用 MST 成本挑 root
    congestion_weighted_copy,
    admit_tree_bidir_initial, admit_tree_bidir_incremental,
    tree_from_edges, tree_edges_list
)
from routing.spb_ecmp import pick_ecmp_path            # ECMP / SPB
from routing.ksp_additional import get_additional_paths # 其他候選路徑

BASE = Path("outputs")
EDGE_CSV = BASE / "matrix" / "edges_all.csv"
KSP_JSON = BASE / "paths" / "ksp_all_pairs.json"
PIN_FILE = BASE / "paths" / "pinned_paths.json"        # 單路徑 pin（目前未使用）
PIN_TREE_FILE = BASE / "paths" / "pinned_trees.json"   # 會議樹 pin 檔


# ------------------------------
# Graph 與路徑 Admission（保留既有邏輯）
# ------------------------------
def load_graph(path: Path) -> nx.Graph:
    G = nx.Graph()
    with path.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            u, v = r["src_switch"].strip(), r["dst_switch"].strip()
            w  = float(r.get("weight", 1.0))
            bw = float(r.get("bw_mbps", 10_000.0))
            G.add_edge(u, v, w=w, bw_mbps=bw)
    return G

def try_admit_path(G, path, cls, rate, reserved, usage):
    taken = []
    for i in range(len(path)-1):
        ekey = tuple(sorted((path[i], path[i+1])))
        ok = admit_edge(ekey, cls, rate, reserved, usage)
        if not ok:
            for ek in taken:
                release_edge(ek, cls, rate, reserved, usage)
            return False
        taken.append(ekey)
    return True

def _get_ksp_paths(ksp: dict, src: str, dst: str):
    key_fwd = f"{src}->{dst}"
    if key_fwd in ksp:
        return ksp[key_fwd]
    key_rev = f"{dst}->{src}"
    if key_rev in ksp:
        return [list(reversed(p)) for p in ksp[key_rev]]
    return []


# ------------------------------
# （新版）一般 flow 的挑路邏輯：ECMP → KSP → Additional（含降級鏈）
# ------------------------------
def pick_path_for_flow(G, ksp, flow, qos, reserved, usage, pins):
    src, dst, topic = flow["src"], flow["dst"], flow["topic"]
    rate = float(flow.get("rate_mbps", 0.0))

    if rate <= 0:
        return {"result":"OK", "class":None, "path":[], "rate":rate, "pinned":False, "reason":"zero_rate"}

    base_cls = classify(topic, qos)
    cand_paths = _get_ksp_paths(ksp, src, dst)  # KSP 備援

    # 1) 先試 ECMP（若函式可用）
    def _try_ecmp(cls: str):
        if 'pick_ecmp_path' in globals() and pick_ecmp_path:
            p = pick_ecmp_path(G, src, dst)  # 需要帶參數可自行擴充
            if p and try_admit_path(G, p, cls, rate, reserved, usage):
                return p
        return None

    # 2) 再試 KSP
    def _try_ksp(cls: str):
        for p in cand_paths:
            if try_admit_path(G, p, cls, rate, reserved, usage):
                return p
        return None

    # 3) 最後試 Additional paths
    def _try_additional(cls: str):
        if 'get_additional_paths' in globals() and get_additional_paths:
            for p in get_additional_paths(G, src, dst, k=6):
                if try_admit_path(G, p, cls, rate, reserved, usage):
                    return p
        return None

    # 依 class + 降級鏈嘗試（每級皆走 ECMP→KSP→Additional）
    for cls in [base_cls] + DOWNGRADE_CHAIN.get(base_cls, []):
        p = _try_ecmp(cls) or _try_ksp(cls) or _try_additional(cls)
        if p:
            return {"result":"OK", "class":cls, "path":p, "rate":rate, "pinned":False, "reason":"ok"}

    return {"result":"BLOCKED", "class":base_cls, "path":[], "rate":rate, "pinned":False, "reason":"insufficient_reservation"}


# ------------------------------
# Qca pinned（單路徑）：目前未啟用，但保留 I/O 相容
# ------------------------------
def load_pins():
    if PIN_FILE.exists():
        return json.loads(PIN_FILE.read_text(encoding="utf-8"))
    return {}

def save_pins(pins: dict):
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(json.dumps(pins, indent=2, ensure_ascii=False))


# ------------------------------
# 會議房 Pinned Steiner MST（雙向 + 增量 + 擁塞感知）
# ------------------------------
def parse_room_id(topic: str) -> str | None:
    m = re.match(r"^vc/([^/]+)/", topic or "")
    return m.group(1) if m else None

def load_pinned_trees() -> dict:
    if PIN_TREE_FILE.exists():
        return json.loads(PIN_TREE_FILE.read_text(encoding="utf-8"))
    return {}

def save_pinned_trees(pins: dict):
    PIN_TREE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_TREE_FILE.write_text(json.dumps(pins, indent=2, ensure_ascii=False))

def _ensure_root_in_tree(G: nx.Graph, T: nx.Graph, root: str, weight: str = "w"):
    """保險：若 T 內沒有 root，接一條 root→T 最近點的最短路，把路徑併入 T。"""
    if root in T:
        return T
    best = None
    for v in T.nodes or []:
        try:
            p = nx.shortest_path(G, root, v, weight=weight)
            c = sum(G[p[i]][p[i+1]][weight] for i in range(len(p)-1))
            if best is None or c < best[0]:
                best = (c, p)
        except nx.NetworkXNoPath:
            continue
    if best:
        _, p = best
        T.add_nodes_from(p)
        T.add_edges_from((p[i], p[i+1]) for i in range(len(p)-1))
    else:
        T.add_node(root)
    return T

def admit_vc_room_as_tree_pinned(
    G: nx.Graph,
    room: str,
    terminals: list[str],
    rate_up_mbps: float,
    rate_down_mbps: float,
    reserved, usage,
    alpha: float = 2.5,          # 擁塞感知強度
    prefer_incremental: bool = True
):
    """
    初次：在「所有 switch（名稱以 s 開頭）」中挑 root，使 Steiner MST(terms ∪ {root}) 的總權重最小；
          權重採用 w_ca = w * (1 + alpha*util_EF) 做擁塞感知。
    已 pinned：成員變動時，對舊樹做增量（只押 Δ+、釋放 Δ-），維持樹穩定。
    """
    # 擁塞感知
    G_ca = congestion_weighted_copy(G, reserved, usage, pool_cls="EF", alpha=alpha,
                                    base_weight_attr="w", out_attr="w_ca")

    pins = load_pinned_trees()
    key = f"vc|{room}"
    def _k(e): return f"{e[0]}-{e[1]}"

    # 已有 pinned？
    if key in pins:
        rec = pins[key]
        root = rec.get("root")
        old_edges = [tuple(sorted(tuple(x))) for x in rec.get("edges", [])]
        T_old = tree_from_edges(old_edges)
        # 保險，確保 root 在樹中
        T_old = _ensure_root_in_tree(G_ca, T_old, root, weight="w_ca")

        old_participants = sorted(set(rec.get("participants", [])))
        loads_old = rec.get("loads", {})
        new_participants = sorted(set(terminals))

        if new_participants == old_participants and not prefer_incremental:
            ok, loads_t = admit_tree_bidir_initial(
                T_old, "EF", root, new_participants, rate_up_mbps, rate_down_mbps, reserved, usage
            )
            return {
                "ok": ok,
                "root": root,
                "tree_edges": tree_edges_list(T_old),
                "loads": { _k(e): loads_t.get(e, 0.0) for e in loads_t },
                "reason": "ok" if ok else "pinned_tree_admission_failed",
                "pinned": True
            }

        added = [p for p in new_participants if p not in old_participants]
        T_new = T_old
        if added:
            T_new = steiner_extend_greedy(G_ca, T_old, added, weight="w_ca")

        ok, loads_new_t, dpos, dneg = admit_tree_bidir_incremental(
            T_old, loads_old, T_new, "EF", root, new_participants,
            rate_up_mbps, rate_down_mbps, reserved, usage
        )
        if not ok:
            return {
                "ok": False,
                "root": root,
                "tree_edges": tree_edges_list(T_new),
                "loads": { _k(e): loads_new_t.get(e, 0.0) for e in loads_new_t },
                "reason": "incremental_admission_failed",
                "pinned": True
            }

        pins[key] = {
            "root": root,
            "edges": tree_edges_list(T_new),
            "participants": new_participants,
            "rate_up": float(rate_up_mbps),
            "rate_down": float(rate_down_mbps),
            "loads": { _k(e): loads_new_t.get(e, 0.0) for e in loads_new_t }
        }
        save_pinned_trees(pins)
        return {
            "ok": True,
            "root": root,
            "tree_edges": tree_edges_list(T_new),
            "loads": { _k(e): loads_new_t.get(e, 0.0) for e in loads_new_t },
            "reason": "ok_incremental",
            "pinned": True
        }

    # 尚未 pinned：從「所有 s*」挑 root，使 MST 成本最小；把 root 也放入 terminals 再建樹
    terms_list = sorted(set(terminals))
    if not terms_list:
        return {"ok": False, "reason": "no_participants", "tree_edges": [], "pinned": False}

    all_switches = sorted([n for n in G.nodes if str(n).startswith("s")])
    root = choose_root_by_mst_cost(G_ca, terms_list, all_switches, weight="w_ca")
    if root is None:
        # 不可達
        return {"ok": False, "reason": "no_reachable_root", "tree_edges": [], "pinned": False}

    terms_aug = sorted(set(terms_list + [root]))
    T = steiner_tree_mst(G_ca, terms_aug, weight="w_ca")

    ok, loads_t = admit_tree_bidir_initial(
        T, "EF", root, terms_list, rate_up_mbps, rate_down_mbps, reserved, usage
    )
    if not ok:
        return {
            "ok": False,
            "root": root,
            "tree_edges": tree_edges_list(T),
            "loads": { f"{e[0]}-{e[1]}": loads_t.get(e, 0.0) for e in loads_t },
            "reason": "insufficient_reservation",
            "pinned": False
        }

    pins = load_pinned_trees()
    pins[f"vc|{room}"] = {
        "root": root,
        "edges": tree_edges_list(T),
        "participants": terms_list,
        "rate_up": float(rate_up_mbps),
        "rate_down": float(rate_down_mbps),
        "loads": { f"{e[0]}-{e[1]}": loads_t.get(e, 0.0) for e in loads_t }
    }
    save_pinned_trees(pins)
    return {
        "ok": True,
        "root": root,
        "tree_edges": tree_edges_list(T),
        "loads": { f"{e[0]}-{e[1]}": loads_t.get(e, 0.0) for e in loads_t },
        "reason": "ok",
        "pinned": False
    }


# ------------------------------
# main：先處理會議樹，再處理一般 flows
# ------------------------------
def main():
    G = load_graph(EDGE_CSV)
    qos = load_qos()
    reserved, usage = build_reservations(G, qos)
    ksp = json.loads((KSP_JSON).read_text(encoding="utf-8")) if KSP_JSON.exists() else {}
    pins = load_pins()  # 目前未用於一般流

    # 讀 flows
    IN_FLOWS = Path("inputs/flows.json")
    if IN_FLOWS.exists():
        flows = json.loads(IN_FLOWS.read_text(encoding="utf-8"))
    else:
        print("[WARN] inputs/flows.json not found")
        flows = []

    # 正規化大小寫
    for f in flows:
        if "src" in f and isinstance(f["src"], str): f["src"] = f["src"].lower()
        if "dst" in f and isinstance(f["dst"], str): f["dst"] = f["dst"].lower()
        if "dsts" in f and isinstance(f["dsts"], list):
            f["dsts"] = [d.lower() for d in f["dsts"] if isinstance(d, str)]

    # ---------- 會議房聚合（支援多 dsts；以房間為單位） ----------
    # 只要「房間參與者 >= 3（至少 pub + 2 subs 或多 pub/sub）」才走 Steiner；否則回到一般流
    room_terms: dict[str, set[str]] = {}
    room_rate_up:   dict[str, float] = {}
    room_rate_down: dict[str, float] = {}
    room_flows: dict[str, list[dict]] = {}

    normal_flows: list[dict] = []

    for f in flows:
        room = parse_room_id(f.get("topic", ""))
        if not room:
            normal_flows.append(f)
            continue

        src = f.get("src")
        dsts = f.get("dsts") or ([f["dst"]] if f.get("dst") else [])
        r = float(f.get("rate_mbps", 0.0))

        room_flows.setdefault(room, []).append(f)
        if src: room_terms.setdefault(room, set()).add(src)
        room_terms.setdefault(room, set()).update(dsts)

        # 速率估計：上行總和（pub->tree）；下行估計為每 flow 的 r 乘以其 dst 數量
        if r > 0:
            room_rate_up[room]    = room_rate_up.get(room, 0.0) + r
            room_rate_down[room]  = room_rate_down.get(room, 0.0) + r * max(1, len(dsts))

    results = []
    edge_flow_rows = []

    # ---------- Stage 1：先處理每個房（Steiner MST + pinned + 擁塞感知） ----------
    processed_rooms: set[str] = set()
    for room, terms in room_terms.items():
        terms_list = sorted(terms)

        # 只有參與者數量 >= 3 才視為「多點會議」用樹；否則視作一般流
        if len(terms_list) < 3:
            normal_flows.extend(room_flows.get(room, []))
            continue

        rate_up   = float(room_rate_up.get(room, 0.0))
        rate_down = float(room_rate_down.get(room, 0.0))

        tree_res = admit_vc_room_as_tree_pinned(
            G, room, terms_list,
            rate_up_mbps=rate_up, rate_down_mbps=rate_down,
            reserved=reserved, usage=usage, alpha=2.5, prefer_incremental=True
        )

        results.append({
            "type": "steiner_tree",
            "topic": f"vc/{room}/#",
            "class": "EF",
            "participants": terms_list,
            "result": "OK" if tree_res["ok"] else "BLOCKED",
            "root": tree_res.get("root"),
            "tree_edges": tree_res.get("tree_edges", []),
            "loads": tree_res.get("loads", {}),
            "reason": tree_res.get("reason", ""),
            "pinned": tree_res.get("pinned", False)
        })

        print(f"[{'OK' if tree_res['ok'] else 'BLOCKED'}][TREE] vc/{room}/# "
              f"class=EF participants={len(terms_list)} root={tree_res.get('root')} "
              f"edges={len(tree_res.get('tree_edges', []))} reason={tree_res.get('reason')}")

        # 把樹邊承載寫入明細（方便後續圖表）
        for e_str, load in tree_res.get("loads", {}).items():
            a, b = sorted(e_str.split("-"))
            edge_flow_rows.append({
                "edge": f"{a}-{b}",
                "class": "EF",
                "topic": f"vc/{room}/#",
                "src": tree_res.get("root", ""),
                "dst": "*",
                "rate_mbps": float(load),
                "flow_index": f"room:{room}",
                "path": "<steiner_mst>"
            })

        processed_rooms.add(room)

    # ---------- Stage 2/3：再處理一般 flows（ECMP → KSP → Additional，含降級鏈） ----------
    for idx, f in enumerate(normal_flows):
        r = pick_path_for_flow(G, ksp, f, qos, reserved, usage, pins)
        results.append({**f, **r})
        print(f"[{r['result']}] {f.get('src','?')}→{f.get('dst','?')} {f.get('topic','')} "
              f"class={r.get('class')} path={r.get('path')} rate={r.get('rate')} reason={r.get('reason')}")

        if r["result"] == "OK" and r["path"]:
            cls = r["class"] or "BE"
            pool = "AF" if isinstance(cls, str) and cls.startswith("AF") else cls
            path = r["path"]; rate = float(r["rate"])
            for i in range(len(path) - 1):
                u, v = sorted((path[i], path[i + 1]))
                edge_flow_rows.append({
                    "edge": f"{u}-{v}",
                    "class": pool,
                    "topic": f.get("topic",""),
                    "src": f.get("src",""),
                    "dst": f.get("dst",""),
                    "rate_mbps": rate,
                    "flow_index": idx,
                    "path": "->".join(path),
                })

    # -------------------- 輸出 --------------------
    out = BASE / "paths" / "flows_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("[OK] wrote", out)

    out_res = BASE / "reservation" / "reserved.csv"
    out_res.parent.mkdir(parents=True, exist_ok=True)
    with out_res.open("w", newline='', encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["edge", "class", "reserved_mbps"])
        for edge, cls_map in reserved.items():
            edge_str = f"{edge[0]}-{edge[1]}"
            for cls, cap in cls_map.items():
                wr.writerow([edge_str, cls, cap])

    out_usage_tot = BASE / "reservation" / "edge_usage_totals.csv"
    with out_usage_tot.open("w", newline='', encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["edge", "class", "usage_mbps"])
        for edge, cls_map in usage.items():
            edge_str = f"{edge[0]}-{edge[1]}"
            for cls, used in cls_map.items():
                wr.writerow([edge_str, cls, used])

    out_usage_detail = BASE / "reservation" / "edge_usage_detail.csv"
    with out_usage_detail.open("w", newline='', encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["edge", "class", "topic", "src", "dst", "rate_mbps", "flow_index", "path"])
        for row in edge_flow_rows:
            wr.writerow([
                row["edge"], row["class"], row["topic"],
                row["src"], row["dst"], row["rate_mbps"],
                row["flow_index"], row["path"],
            ])

    print("[OK] wrote", out_res, out_usage_tot, out_usage_detail)


if __name__ == "__main__":
    main()
