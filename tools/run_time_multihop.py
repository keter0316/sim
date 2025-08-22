#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse, csv, json, hashlib
import numpy as np
import networkx as nx

from models.queue_runtime import Packet
from models.link_manager import LinkManager

# ---------- ECMP: 一致雜湊挑一條等長最短路 ----------
def _hash32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) & 0xffffffff

def pick_ecmp_path(G: nx.Graph, src: str, dst: str, seed: int = 42) -> List[str]:
    if not (src in G and dst in G): return []
    try:
        _ = nx.shortest_path_length(G, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return []
    paths = [p for p in nx.all_shortest_paths(G, src, dst, weight="weight")]
    if not paths: return []
    idx = _hash32(f"{src}->{dst}|{seed}") % len(paths)
    return paths[idx]

# ---------- 讀取 paths-json（容忍多種格式/KSP） ----------
def normalize_paths_json(p: Path, seed: int = 42) -> Dict[str, List[str]]:
    if not p or not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    norm: Dict[str, List[str]] = {}

    def choose_from_list(key: str, val: Any):
        # val 可能是 ["S1","S2","S3"] 或 [["S1","S2","S3"], ["S1","S4","S3"], ...]
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            norm[key] = val
        elif isinstance(val, list) and val and isinstance(val[0], list):
            idx = _hash32(key) % len(val)
            norm[key] = val[idx]
        elif isinstance(val, dict):
            # 常見鍵：paths / ksp / candidates
            for k in ("paths","ksp","candidates"):
                if k in val and isinstance(val[k], list) and val[k]:
                    lst = val[k]
                    if lst and isinstance(lst[0], list):
                        idx = _hash32(key) % len(lst)
                        norm[key] = lst[idx]
                        break
        # 其他格式略過

    if isinstance(raw, dict):
        for key, val in raw.items():
            # 鍵可能是 "S1->S3" 或 巢狀 {"S1":{"S3":[...]}}
            if "->" in key:
                choose_from_list(key, val)
            else:
                # 巢狀：第一層 src，第二層 dst
                sub = val
                if isinstance(sub, dict):
                    for dst, v in sub.items():
                        choose_from_list(f"{key}->{dst}", v)
    return norm

# ---------- Flow 規格 ----------
@dataclass
class FlowSpec:
    src: str
    dst: str
    qid: int
    rate_mbps: float
    mean_bytes: int
    path: List[str]

def read_flows(path: Path) -> List[FlowSpec]:
    flows: List[FlowSpec] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        # 支援：list of objects [{src,dst,qid,rate_mbps,mean_bytes}, ...]
        # 或 dict {"flows":[...]}
        items = data.get("flows", data) if isinstance(data, dict) else data
        for r in items:
            flows.append(FlowSpec(
                src=str(r["src"]), dst=str(r["dst"]),
                qid=int(r.get("qid",7)),
                rate_mbps=float(r["rate_mbps"]),
                mean_bytes=int(r.get("mean_bytes",800)),
                path=[]
            ))
    else:
        with path.open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                flows.append(FlowSpec(
                    src=str(r["src"]), dst=str(r["dst"]),
                    qid=int(r.get("qid",7)),
                    rate_mbps=float(r["rate_mbps"]),
                    mean_bytes=int(r.get("mean_bytes",800)),
                    path=[]
                ))
    return flows

def main():
    ap = argparse.ArgumentParser(description="Multi-hop queue+buffer time-engine")
    ap.add_argument("--edges", type=Path, required=True, help="edges.csv (src,dst[,capacity_mbps])")
    ap.add_argument("--flows", type=Path, required=True, help="flows.csv 或 flows.json")
    ap.add_argument("--paths-json", type=Path, default=None, help="可選：路徑 JSON（單一路徑或 KSP 皆可）")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", choices=["all","deq","none"], default="deq")
    ap.add_argument("--log-sample", type=int, default=10)
    ap.add_argument("--perhop-csv", type=Path, default=Path("outputs/runtime/multihop_perhop.csv"))
    ap.add_argument("--summary-csv", type=Path, default=Path("outputs/runtime/multihop_summary.csv"))
    ap.add_argument("--qdepth-csv", type=Path, default=Path("outputs/runtime/multihop_qdepth.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) LinkManager + 圖
    LM = LinkManager(args.edges)
    G = nx.Graph()
    seen = set()
    for (u,v) in LM.links.keys():
        if (v,u) in seen: continue
        G.add_edge(u, v, weight=1.0)
        seen.add((u,v)); seen.add((v,u))

    # 2) flows
    flows = read_flows(args.flows)

    # 3) 路徑
    pre_paths = normalize_paths_json(args.paths_json, seed=args.seed) if args.paths_json else {}
    for f in flows:
        key = f"{f.src}->{f.dst}"
        if key in pre_paths:
            f.path = pre_paths[key]
        else:
            f.path = pick_ecmp_path(G, f.src, f.dst, seed=args.seed)
        if not f.path or len(f.path) < 2:
            raise RuntimeError(f"No path for {key}")

    # 4) 模擬
    now = 0.0
    pkt_id = 0
    out_rows: List[List[object]] = []
    next_sample_t = 0.0

    e2e_done = 0
    e2e_bytes = 0
    e2e_sojourn_sum = 0.0

    lam_bytes = {i: f.rate_mbps * 1e6 * args.dt / 8 for i, f in enumerate(flows)}

    while now < args.duration:
        # 4.1 注入
        for i, f in enumerate(flows):
            bytes_this_dt = rng.poisson(lam_bytes[i])
            remain = bytes_this_dt
            while remain > 0:
                sz = max(64, int(rng.normal(f.mean_bytes, f.mean_bytes*0.2)))
                pkt = Packet(pkt_id=pkt_id, size_bytes=sz, qid=f.qid)
                pkt.meta["t_ingress"] = now
                pkt.meta["path"] = f.path
                pkt.meta["hop"] = 0
                u, v = f.path[0], f.path[1]
                LM.get(u,v).enqueue(pkt, now)
                pkt_id += 1
                remain -= sz

        # 4.2 服務 + 推進
        for (u,v), link in LM.links.items():
            sent = link.service(now, args.dt)
            for p in sent:
                path: List[str] = p.meta["path"]
                hop = int(p.meta["hop"])
                if args.log in ("all","deq") and (p.pkt_id % args.log_sample == 0):
                    out_rows.append([
                        now, "deq", f"{u}-{v}", p.qid, p.pkt_id, p.size_bytes,
                        f"qdelay={(p.start_service_ts or now)-p.enqueue_ts:.6f}|"
                        f"sojourn={(p.depart_ts or now)-p.enqueue_ts:.6f}|"
                        f"ecn={int(p.marks.get('ecn',False))}"
                    ])
                if hop + 1 < len(path) - 1:
                    p.start_service_ts = None
                    p.enqueue_ts = now
                    p.meta["hop"] = hop + 1
                    uu, vv = path[hop+1], path[hop+2]
                    LM.get(uu, vv).enqueue(p, now)
                else:
                    e2e_done += 1
                    e2e_bytes += p.size_bytes
                    t_ingress = float(p.meta.get("t_ingress", now))
                    e2e_sojourn_sum += (now - t_ingress)

        # 4.3 取樣 queue depth
        if now + 1e-12 >= next_sample_t:
            for link in LM.all_links():
                link.sample_depth(now)
            next_sample_t += args.dt

        now += args.dt

    # 5) 輸出
    args.perhop_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.perhop_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["t","event","link","qid","pkt_id","size","extra"]); w.writerows(out_rows)

    total_enq = sum(l.stats["enq"] for l in LM.all_links())
    total_deq = sum(l.stats["deq"] for l in LM.all_links())
    total_drop = sum(l.stats["drop"] for l in LM.all_links())
    total_mark = sum(l.stats["mark"] for l in LM.all_links())
    total_served = sum(l.stats["bytes_served"] for l in LM.all_links())
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["enq","deq","drop","ecn_mark","bytes_served_b","duration_s",
                    "served_rate_mbps","e2e_done","e2e_rate_mbps","e2e_avg_s"])
        w.writerow([total_enq, total_deq, total_drop, total_mark, total_served, args.duration,
                    f"{8*total_served/args.duration/1e6:.2f}", e2e_done,
                    f"{8*e2e_bytes/args.duration/1e6:.2f}", f"{(e2e_sojourn_sum/max(1,e2e_done)):.6f}"])

    with args.qdepth_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["t","link","qid","bytes_in_q"])
        for link in LM.all_links():
            w.writerows(link.depth_rows)

if __name__ == "__main__":
    main()
