#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
examples/build_paths.py

一鍵示範：gen_topology -> ECMP FDB -> KSP(all-pairs) -> quick_verify
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import networkx as nx

# 我們的演算法模組
from routing.spb_ecmp import ecmp_fdb, save_fdb_json
from routing.ksp_additional import ksp_all_pairs, save_ksp_json


# ------------------------- helpers -------------------------

def run_gen_topology(args) -> None:
    """呼叫 tools/gen_topology.py 產生拓樸與 edges_all.csv。"""
    cmd = [
        sys.executable, "tools/gen_topology.py",
        "--switches", str(args.switches),
        "--hosts", str(args.hosts),
        "--access-k", str(args.access_k),
        "--core-mode", args.core_mode,
        "--core-k", str(args.core_k),
        "--uplinks", str(args.uplinks),
        "--seed", str(args.seed),
        "--out-dir", args.out_dir,
    ]
    if args.radio:
        cmd.append("--radio")
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_graph_from_edges(edges_csv: Path, weight_key: str = "weight") -> nx.Graph:
    G = nx.Graph()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = row["src_switch"].strip()
            v = row["dst_switch"].strip()
            w = float(row.get(weight_key, 1.0))
            G.add_edge(u, v, **{weight_key: w})
    return G


# ------------------------- CLI -------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Demo pipeline: gen_topology -> ECMP/KSP -> quick_verify")
    # Topology
    ap.add_argument("--no-topo", action="store_true", help="跳過拓樸產生（直接使用既有 outputs/matrix/edges_all.csv）")
    ap.add_argument("--switches", type=int, default=10)
    ap.add_argument("--hosts", type=int, default=8)
    ap.add_argument("--access-k", type=int, default=1)
    ap.add_argument("--core-mode", choices=["none", "ring", "clique", "ws"], default="ring")
    ap.add_argument("--core-k", type=int, default=2)
    ap.add_argument("--uplinks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radio", action="store_true", help="為 access/host 相關鏈路產生 ETX/RSSI/SNR")
    ap.add_argument("--out-dir", type=str, default="outputs")

    # KSP
    ap.add_argument("--K", type=int, default=3, help="K-shortest paths per (src,dst)")
    ap.add_argument("--limit-dst", type=int, default=0, help="每個 src 僅取前 N 個目的地（0=全對）")

    # verify
    ap.add_argument("--theta", type=float, default=0.05, help="驗證用 Δcost 門檻（5%）")
    return ap


def main() -> None:
    args = build_cli().parse_args()
    out_dir = Path(args.out_dir)
    edges_csv = out_dir / "matrix" / "edges_all.csv"
    paths_dir = out_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)

    # 1) 拓樸
    if not args.no_topo:
        run_gen_topology(args)
    if not edges_csv.exists():
        raise FileNotFoundError(f"找不到 {edges_csv}，請先產生拓樸或移除 --no-topo")

    # 載入圖（使用 'weight' 欄位）
    G = load_graph_from_edges(edges_csv, weight_key="weight")

    # 2) ECMP FDB
    fdb = ecmp_fdb(G, weight="weight")
    save_fdb_json(fdb, paths_dir / "fdb_spb_ecmp.json")

    # 3) KSP all pairs
    limit = args.limit_dst if args.limit_dst > 0 else None
    ksp_map = ksp_all_pairs(G, K=args.K, weight="weight", limit_dst=limit)
    save_ksp_json(ksp_map, paths_dir / "ksp_all_pairs.json")

    # 4) quick_verify（用 CLI 呼叫，順便繪圖）
    cmd = [
        sys.executable, "quick_verify.py",
        "--edges", str(edges_csv),
        "--fdb", str(paths_dir / "fdb_spb_ecmp.json"),
        "--ksp", str(paths_dir / "ksp_all_pairs.json"),
        "--reports-dir", "reports",
        "--plots-dir", "plots",
        "--theta", str(args.theta),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\n[DONE] ECMP/KSP 已產生並驗證：")
    print(" - FDB :", paths_dir / "fdb_spb_ecmp.json")
    print(" - KSP :", paths_dir / "ksp_all_pairs.json")
    print(" - 報表:", Path("reports").resolve())
    print(" - 圖檔:", Path("plots").resolve())


if __name__ == "__main__":
    main()
