#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
參數化拓撲產生器

外圍 H 台 access switches ↔ H 台 hosts（1:1），其餘為 core：
- 總 switch = N 命名 s1..sN
- 外圍 access = s1..sH，h1..hH 各自對接 s1..sH
- 內部 core   = s(H+1)..sN
- Access 之間可用 k-近鄰環 (k=0 不互連）
- Access → Core uplinks=1 或 2（輪詢分配到 core）
- Core 拓撲：ring / clique / ws（小世界）

輸出：
- {out}/topo_diagram/topo_auto.png
- {out}/matrix/adjacency_switches.csv      （只含所有 switch 的鄰接矩陣）
- {out}/matrix/edges_all.csv               （全網邊；欄位: src_switch,dst_switch,weight,bw_mbps,type,delay_ms,loss,etx,rssi,snr）
- {out}/matrix/nodes.csv                   （node_id, role）

注意：所有邊的 bw_mbps 預設均為 10000（可調）。
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import List

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------- CLI -------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="N switches (s1..sN) with H outer access switches (s1..sH) and hosts (h1..hH)")
    ap.add_argument("--switches", type=int, default=12, help="總 switch 數 N (>=1)，名稱 s1..sN")
    ap.add_argument("--hosts",    type=int, default=8,  help="外圍 host/switch 對數 H (0 <= H <= N)")

    # Access 層（外圍 s1..sH）
    ap.add_argument("--access-k", type=int, default=1,
                    help="Access 間 k-近鄰環（每邊連 k 個左右鄰居；0=不互連）")

    # Core 層（內部 s(H+1)..sN）
    ap.add_argument("--core-mode", choices=["none","ring","clique","ws"], default="ring",
                    help="Core 連線型態（當 N>H 有 core 時才有意義）")
    ap.add_argument("--core-k", type=int, default=2, help="Core 環/小世界的近鄰度（偶數佳）")
    ap.add_argument("--core-rewire", type=float, default=0.1, help="小世界重接概率（ws 模式）")

    # Access → Core 接法
    ap.add_argument("--uplinks", type=int, default=1, choices=[1,2],
                    help="每個 access switch 連到幾個 core（1 或 2），core 不存在則忽略")

    # Link 參數（全部預設 10000）
    ap.add_argument("--bw-access", type=float, default=10_000.0, help="Access/Host/Access-Core 預設頻寬 (Mbps)")
    ap.add_argument("--bw-core",   type=float, default=10_000.0, help="Core-Core 預設頻寬 (Mbps)")
    ap.add_argument("--weight",    type=float, default=1.0,      help="路徑計算權重（預設 1.0）")

    # 新增：seed / out 目錄 / 是否輸出無線指標
    ap.add_argument("--seed",      type=int,   default=1,        help="隨機種子（重現用）")
    ap.add_argument("--out-dir",   type=str,   default="outputs",help="輸出根目錄（預設 outputs）")
    ap.add_argument("--radio",     action="store_true",          help="為 access/host 相關鏈路產生 ETX/RSSI/SNR")
    return ap


# ------------------------- Metrics sampler -------------------------

# 預設延遲範圍（毫秒）
DELAY_MS = {
    "host-access":  (0.3, 0.7),
    "access-access":(0.7, 1.2),
    "access-core":  (0.8, 1.5),
    "core-core":    (0.5, 1.0),
}

def sample_link_metrics(edge_type: str, rng: np.random.Generator, emit_radio: bool):
    lo, hi = DELAY_MS.get(edge_type, (0.8, 1.5))
    delay_ms = float(rng.uniform(lo, hi))
    # 丟包維持極低（可視為基線/雜訊）
    loss = float(max(0.0, rng.normal(0.0001, 0.0001)))

    if emit_radio and edge_type in ("host-access", "access-access", "access-core"):
        etx  = float(max(1.0, rng.normal(1.05, 0.05)))
        # 粗略給個型態感（可再調）：
        base_rssi = {"host-access": -58, "access-access": -65, "access-core": -62}[edge_type]
        base_snr  = {"host-access":  25, "access-access":  20, "access-core":  22}[edge_type]
        rssi = float(rng.normal(base_rssi, 4.0))
        snr  = float(rng.normal(base_snr,  3.0))
    else:
        etx, rssi, snr = (np.nan, np.nan, np.nan)

    return dict(delay_ms=delay_ms, loss=loss, etx=etx, rssi=rssi, snr=snr)


# ------------------------- Topology builders -------------------------

def build_names(N: int, H: int) -> tuple[list[str], list[str], list[str]]:
    """回傳 (all_switches, access_switches, core_switches) 名稱列表。"""
    switches = [f"s{i}" for i in range(1, N+1)]
    access   = switches[:H]          # s1..sH
    core     = switches[H:]          # s(H+1)..sN
    return switches, access, core


def add_access_and_hosts(G: nx.Graph, access_sw: List[str], H: int,
                         k: int, bw_access: float, weight: float,
                         rng: np.random.Generator, emit_radio: bool) -> List[str]:
    """建立外圍 access switches 與 hosts（h1..hH），加上 access 間近鄰環與 host 對接。"""
    # 標記 access switches（同步設定 role 與相容欄位 ntype）
    G.add_nodes_from((s, {"role": "access", "ntype": "access"}) for s in access_sw)
    # 建 host
    hosts = [f"h{i}" for i in range(1, H+1)]
    G.add_nodes_from((h, {"role": "host", "ntype": "host"}) for h in hosts)

    # host 一對一對接 access：hᵢ ↔ sᵢ
    for i in range(H):
        h, s = hosts[i], access_sw[i]
        G.add_edge(h, s,
                   type="host-access", bw_mbps=bw_access, weight=weight,
                   **sample_link_metrics("host-access", rng, emit_radio))
    # access 間 k-近鄰環
    A = len(access_sw)
    if k > 0 and A >= 2:
        for i in range(A):
            u = access_sw[i]
            for d in range(1, k+1):
                v = access_sw[(i + d) % A]
                if not G.has_edge(u, v):
                    G.add_edge(u, v,
                               type="access-access", bw_mbps=bw_access, weight=weight,
                               **sample_link_metrics("access-access", rng, emit_radio))
    return hosts


def add_core_layer(G: nx.Graph, cores: List[str], mode: str, core_k: int,
                   rewire_p: float, bw_core: float, weight: float,
                   rng: np.random.Generator, emit_radio: bool):
    """在 core 節點之間建立連線（ring/clique/ws）。"""
    if not cores:
        return
    # 標記 core 節點（同步設定 role 與相容欄位 ntype）
    G.add_nodes_from((c, {"role": "core", "ntype": "core"}) for c in cores)

    C = len(cores)
    if mode in ("ring","ws"):
        k = max(2, core_k + (core_k % 2)) if C > 2 else 1
        # 基礎環
        for i in range(C):
            u, v = cores[i], cores[(i+1) % C]
            if not G.has_edge(u, v):
                G.add_edge(u, v,
                           type="core-core", bw_mbps=bw_core, weight=weight,
                           **sample_link_metrics("core-core", rng, emit_radio))
        # 補鄰近 d=2..k/2
        for d in range(2, (k//2)+1):
            for i in range(C):
                u, v = cores[i], cores[(i + d) % C]
                if not G.has_edge(u, v):
                    G.add_edge(u, v,
                               type="core-core", bw_mbps=bw_core, weight=weight,
                               **sample_link_metrics("core-core", rng, emit_radio))
        # 小世界重接
        if mode == "ws" and C >= 4 and rewire_p > 0:
            edges = [(u, v) for u, v, data in G.edges(cores, data=True) if data.get("type") == "core-core"]
            for (u, v) in list(edges):
                if not G.has_edge(u, v):
                    continue
                if rng.random() < rewire_p:
                    G.remove_edge(u, v)
                    candidates = [x for x in cores if x != u and not G.has_edge(u, x)]
                    if candidates:
                        w = rng.choice(candidates)
                        G.add_edge(u, w,
                                   type="core-core", bw_mbps=bw_core, weight=weight,
                                   **sample_link_metrics("core-core", rng, emit_radio))

    elif mode == "clique":
        for i in range(C):
            for j in range(i+1, C):
                if not G.has_edge(cores[i], cores[j]):
                    G.add_edge(cores[i], cores[j],
                               type="core-core", bw_mbps=bw_core, weight=weight,
                               **sample_link_metrics("core-core", rng, emit_radio))

    elif mode == "none":
        pass
    else:
        raise ValueError(f"unsupported core mode: {mode}")


def add_uplinks(G: nx.Graph, access_sw: List[str], core_sw: List[str],
                uplinks: int, bw_access: float, weight: float,
                rng: np.random.Generator, emit_radio: bool):
    """Access → Core 上行連線（核心存在時才做）。"""
    if not core_sw or not access_sw or uplinks <= 0:
        return
    C = len(core_sw)
    for i, s in enumerate(access_sw):
        if uplinks == 1:
            c = core_sw[i % C]
            G.add_edge(s, c,
                       type="access-core", bw_mbps=bw_access, weight=weight,
                       **sample_link_metrics("access-core", rng, emit_radio))
        else:  # uplinks == 2
            c1 = core_sw[i % C]
            c2 = core_sw[(i + (C//2 or 1)) % C] if C > 1 else c1
            G.add_edge(s, c1,
                       type="access-core", bw_mbps=bw_access, weight=weight,
                       **sample_link_metrics("access-core", rng, emit_radio))
            if c2 != c1:
                G.add_edge(s, c2,
                           type="access-core", bw_mbps=bw_access, weight=weight,
                           **sample_link_metrics("access-core", rng, emit_radio))


# ------------------------- Layout & Export -------------------------

def layout_positions(access_sw: List[str], hosts: List[str], core_sw: List[str]):
    pos = {}
    # 外圈 access switches
    if access_sw:
        pos_access = nx.circular_layout(access_sw, scale=1.0)
        pos.update(pos_access)
    # 對應 host：沿 access 再往外推
    for s in access_sw:
        h = "h" + s[1:]  # s1 -> h1
        v = np.array(pos[s])
        pos[h] = (v * 1.25).tolist()
    # 內圈 core
    if core_sw:
        pos_core = nx.circular_layout(core_sw, scale=0.55)
        pos.update(pos_core)
    return pos


def export_outputs(G: nx.Graph, all_sw: List[str], core_sw: List[str], out_dir: str):
    # 目錄
    root = Path(out_dir)
    img_dir = root / "topo_diagram"; img_dir.mkdir(parents=True, exist_ok=True)
    mat_dir = root / "matrix";       mat_dir.mkdir(parents=True, exist_ok=True)

    # 以 role 為主（相容 ntype）
    def nodes_with(role: str) -> List[str]:
        return [n for n, d in G.nodes(data=True) if d.get("role") == role or d.get("ntype") == role]

    access_nodes = nodes_with("access")
    host_nodes   = nodes_with("host")
    core_nodes   = nodes_with("core")

    pos = layout_positions(access_nodes, host_nodes, core_sw)

    # 畫圖
    plt.figure(figsize=(8.5, 8.5))
    def draw_edges(edge_type, width=1.8):
        es = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == edge_type]
        if es:
            nx.draw_networkx_edges(G, pos, edgelist=es, width=width)
    draw_edges("access-access", 1.8)
    draw_edges("access-core", 1.8)
    draw_edges("core-core", 2.6)
    draw_edges("host-access", 1.2)

    if access_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=access_nodes, node_size=650)
    if host_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=host_nodes, node_size=420)
    if core_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_size=720)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.axis("off")
    fig_path = img_dir / "topo_auto.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close()

    # 鄰接矩陣（所有 switch）
    SG = G.subgraph(all_sw).copy()
    A = nx.to_numpy_array(SG, nodelist=all_sw, dtype=int)
    pd.DataFrame(A, index=all_sw, columns=all_sw).to_csv(mat_dir / "adjacency_switches.csv")

    # 全網 edges.csv（擴充欄位）
    edges_csv = mat_dir / "edges_all.csv"
    with edges_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_switch","dst_switch","weight","bw_mbps","type",
                    "delay_ms","loss","etx","rssi","snr"])
        for u, v, d in G.edges(data=True):
            w.writerow([
                u, v,
                d.get("weight", 1.0),
                d.get("bw_mbps", 10_000.0),
                d.get("type",""),
                d.get("delay_ms",""),
                d.get("loss",""),
                d.get("etx",""),
                d.get("rssi",""),
                d.get("snr",""),
            ])

    # 節點角色表（以 role 為主；若無則回退 ntype；仍缺則以名稱推斷）
    nodes_csv = mat_dir / "nodes.csv"
    with nodes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id","role"])
        for n in sorted(G.nodes, key=str):
            d = G.nodes[n]
            role = d.get("role") or d.get("ntype")
            if not role:
                if str(n).startswith("h"):
                    role = "host"
                elif str(n).startswith("s"):
                    # 盡量推斷：若連到 host 視為 access，否則 core
                    neigh = list(G.neighbors(n))
                    role = "access" if any(str(x).startswith("h") for x in neigh) else "core"
                else:
                    role = "other"
            w.writerow([n, role])

    print(f"[OK] topology image     -> {fig_path}")
    print(f"[OK] adjacency (switch) -> {mat_dir / 'adjacency_switches.csv'}")
    print(f"[OK] edges csv          -> {edges_csv}")
    print(f"[OK] nodes csv          -> {nodes_csv}")


# ------------------------- Main -------------------------

def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    N = max(1, args.switches)
    H = max(0, min(args.hosts, N))   # 0 <= H <= N

    # 名稱切分
    all_sw, access_sw, core_sw = build_names(N, H)

    G = nx.Graph()

    # 外圍：access switches + hosts（嚴格 hᵢ ↔ sᵢ）
    _hosts = add_access_and_hosts(
        G, access_sw, H,
        k=max(0, args.access_k),
        bw_access=args.bw_access, weight=args.weight,
        rng=rng, emit_radio=args.radio
    )

    # 內部：core switches 連線
    add_core_layer(
        G, core_sw,
        mode=("none" if not core_sw else args.core_mode),
        core_k=args.core_k, rewire_p=args.core_rewire,
        bw_core=args.bw_core, weight=args.weight,
        rng=rng, emit_radio=args.radio
    )

    # Access → Core uplinks
    add_uplinks(
        G, access_sw, core_sw,
        uplinks=args.uplinks, bw_access=args.bw_access, weight=args.weight,
        rng=rng, emit_radio=args.radio
    )

    export_outputs(G, all_sw, core_sw, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
