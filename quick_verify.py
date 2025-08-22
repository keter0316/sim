#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quick_verify.py

檢查與產物：
1) ECMP vs KSP 等價性（最短路集合的第一跳 ⊆ FDB 的下一跳集合）
   -> reports/ecmp_ksp_check.csv

2) 成本單調性
   - 若存在 outputs/flows/reroutes.csv：檢查 new_cost <= old_cost*(1-θ)
   - 否則：對 per_flow 的 WECMP 候選做 Δcost ≤ θ 檢查（以 edges_all.csv 計算成本）
   -> reports/cost_trend.csv

3) Admission 殘額：usage ≤ reserve*(1-headroom)
   -> reports/admission_check.csv

並輸出圖：
- plots/path_len_hist.png（以 per_flow 的 path 或 KSP 第一條製作）
- plots/cost_cdf.png（以 per_flow 的 path_cost 製作）
- plots/residual_cdf.png（以 per-class per-edge 殘額 residual=limit-usage 製作）
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import yaml


# ------------------------- I/O helpers -------------------------

def load_edges_graph(edges_csv: Path, weight_key: str = "weight") -> nx.Graph:
    G = nx.Graph()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["src_switch"].strip()
            v = row["dst_switch"].strip()
            w = float(row.get("weight", 1.0))
            G.add_edge(
                u, v,
                weight=w,
                delay_ms=float(row.get("delay_ms", 1.0) or 1.0),
                loss=float(row.get("loss", 0.0) or 0.0),
                bw_mbps=float(row.get("bw_mbps", 1000.0) or 1000.0),
            )
    return G


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_per_flow(per_flow_csv: Path) -> List[dict]:
    if not per_flow_csv.exists():
        return []
    rows = []
    with per_flow_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ------------------------- Utilities -------------------------

def ekey(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)

# 兼容先前用法（修正 Pylance 報 ek 未定義）
def ek(u: str, v: str) -> Tuple[str, str]:
    return ekey(u, v)

def parse_path_string(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x for x in s.split("->") if x]


def path_cost(G: nx.Graph, path: List[str], weight: str = "weight") -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        total += float(G[u][v].get(weight, 1.0))
    return total


def compute_reserve_per_edge(bw_mbps: float, reservations_cfg: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls, cfg in (reservations_cfg or {}).items():
        if not isinstance(cfg, dict):
            continue
        if "min_mbps" in cfg and cfg["min_mbps"] is not None:
            out[cls] = float(cfg["min_mbps"])
        elif "share" in cfg and cfg["share"] is not None:
            out[cls] = float(cfg["share"]) * float(bw_mbps)
        else:
            out[cls] = 0.0
    return out


# ------------------------- Checks -------------------------

def check_ecmp_vs_ksp(G: nx.Graph, fdb_obj, ksp_obj, reports_dir: Path) -> Tuple[int, int]:
    """
    回傳 (checked_pairs, mismatches)
    """
    # 支援兩種 FDB 結構：
    # 1) {"s->d": [nh,...]}
    # 2) {"s": {"d": [nh,...]}, ...}
    if all("->" in k for k in fdb_obj.keys()):
        def fdb_nh(s, d): return set(fdb_obj.get(f"{s}->{d}", []))
        pairs = [tuple(k.split("->", 1)) for k in fdb_obj.keys()]
    else:
        def fdb_nh(s, d): return set((fdb_obj.get(s, {}) or {}).get(d, []))
        pairs = [(s, d) for s, mp in fdb_obj.items() for d in (mp or {}).keys()]

    rows = []
    mismatches = 0
    checked = 0

    for s, d in sorted(pairs):
        paths = ksp_obj.get(f"{s}->{d}", [])
        if not paths:
            continue
        # 在 KSP 回傳的集合裡，挑出「成本最小」的子集，再取其第一跳集合
        costs = [path_cost(G, p, weight="weight") for p in paths]
        if not costs:
            continue
        minc = min(costs)
        first_hops = set()
        for p, c in zip(paths, costs):
            if len(p) >= 2 and abs(c - minc) <= 1e-9:
                first_hops.add(p[1])

        nh_set = fdb_nh(s, d)
        ok = first_hops.issubset(nh_set) and len(first_hops) > 0
        if not ok:
            mismatches += 1
        checked += 1
        rows.append({
            "src": s,
            "dst": d,
            "ksp_first_hops": ";".join(sorted(first_hops)) if first_hops else "",
            "fdb_next_hops":  ";".join(sorted(nh_set)) if nh_set else "",
            "status": "OK" if ok else "MISMATCH",
        })

    out_csv = reports_dir / "ecmp_ksp_check.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src","dst","ksp_first_hops","fdb_next_hops","status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return checked, mismatches


def check_admission(G: nx.Graph, qos: dict, per_flow_rows: List[dict], reports_dir: Path) -> Tuple[int, int, List[float]]:
    """
    以 per_flow.csv 的 admitted flows 聚合每條邊的 usage（依 class），
    與 qos.yaml 的 reservations 比對，確認 usage ≤ reserve*(1-headroom)。
    回傳 (checked_edges, violations, residual_list)
    residual = limit - usage，其中 limit = reserve*(1-headroom)
    """
    headroom = float(qos.get("headroom", 0.10))
    reservations_cfg = qos.get("reservations", {}) or {}

    # 預備：每邊 per-class 的保留
    reserve: Dict[Tuple[str, str], Dict[str, float]] = {}
    for u, v, d in G.edges(data=True):
        rmap = compute_reserve_per_edge(float(d.get("bw_mbps", 1000.0)), reservations_cfg)
        reserve[ek(u, v)] = rmap

    # 聚合 usage
    usage: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in per_flow_rows:
        if row.get("admitted") not in ("1", 1, "True", "true"):
            continue
        cls = (row.get("class_admitted") or row.get("class_req") or "").strip()
        rate = float(row.get("rate_mbps", 0.0) or 0.0)
        path = parse_path_string(row.get("path", ""))
        for i in range(len(path) - 1):
            e = ek(path[i], path[i+1])
            usage.setdefault(e, {})
            usage[e][cls] = usage[e].get(cls, 0.0) + rate

    # 檢查
    rows = []
    violations = 0
    checked = 0
    residuals: List[float] = []

    for e, rmap in reserve.items():
        u, v = e
        ucls = usage.get(e, {})
        all_classes = sorted(set(list(rmap.keys()) + list(ucls.keys())))
        for cls in all_classes:
            cap = float(rmap.get(cls, 0.0))
            use = float(ucls.get(cls, 0.0))
            limit = cap * (1.0 - headroom)
            residual = limit - use
            residuals.append(residual)
            ok = (use <= limit + 1e-9)
            if not ok:
                violations += 1
            checked += 1
            rows.append({
                "edge": f"{u}-{v}",
                "class": cls,
                "reserve_mbps": f"{cap:.6f}",
                "usage_mbps": f"{use:.6f}",
                "limit_mbps": f"{limit:.6f}",
                "residual_mbps": f"{residual:.6f}",
                "status": "OK" if ok else "VIOLATION",
            })

    out_csv = reports_dir / "admission_check.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["edge","class","reserve_mbps","usage_mbps","limit_mbps","residual_mbps","status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return checked, violations, residuals


def check_cost_trend(
    G: nx.Graph,
    per_flow_rows: List[dict],
    reports_dir: Path,
    *,
    theta: float = 0.05,
    reroutes_csv: Optional[Path] = None
) -> Tuple[int, int]:
    """
    若提供 reroutes_csv，檢查每次切換是否 >= 5% 改善；
    否則用 per_flow 的 WECMP 候選做 Δcost ≤ theta 檢查。
    回傳 (checked_count, violations)
    """
    rows = []
    violations = 0
    checked = 0

    if reroutes_csv and reroutes_csv.exists():
        with reroutes_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fid = r.get("flow_id", "")
                oldc = r.get("old_cost", "") or r.get("old_total_cost", "")
                newc = r.get("new_cost", "") or r.get("new_total_cost", "")
                if not oldc or not newc:
                    continue
                oldc = float(oldc); newc = float(newc)
                ok = (newc <= oldc * (1.0 - theta) + 1e-12)
                if not ok:
                    violations += 1
                checked += 1
                rows.append({
                    "flow_id": fid,
                    "old_cost": f"{oldc:.6f}",
                    "new_cost": f"{newc:.6f}",
                    "improvement_pct": f"{(oldc-newc)/max(oldc,1e-9):.4f}",
                    "status": "OK" if ok else "VIOLATION",
                })
    else:
        # 靜態檢查：以 per_flow 的最佳路徑成本 vs 候選成本差距
        for r in per_flow_rows:
            if not r.get("path"):
                continue
            fid = r.get("flow_id") or r.get("id") or ""
            best_path = parse_path_string(r.get("path", ""))
            best_cost = float(r.get("path_cost", 0.0) or 0.0)
            if best_cost <= 0:
                best_cost = path_cost(G, best_path, weight="weight")

            cand_ser = (r.get("wecmp_candidates") or "").strip()
            if not cand_ser:
                continue
            cand_paths = []
            for seg in cand_ser.split("|"):
                p = parse_path_string(seg)
                if p:
                    cand_paths.append(p)

            for p in cand_paths:
                c = path_cost(G, p, weight="weight")
                ok = (c <= best_cost * (1.0 + theta) + 1e-12)
                if not ok:
                    violations += 1
                checked += 1
                rows.append({
                    "flow_id": fid,
                    "best_cost": f"{best_cost:.6f}",
                    "candidate_cost": f"{c:.6f}",
                    "delta_pct": f"{(c-best_cost)/max(best_cost,1e-9):.4f}",
                    "status": "OK" if ok else "VIOLATION",
                })

    out_csv = reports_dir / "cost_trend.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if reroutes_csv and reroutes_csv.exists():
            fields = ["flow_id","old_cost","new_cost","improvement_pct","status"]
        else:
            fields = ["flow_id","best_cost","candidate_cost","delta_pct","status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    return checked, violations


# ------------------------- Plots -------------------------

def plot_path_len_hist(per_flow_rows: List[dict], ksp_obj: dict, plots_dir: Path):
    lens: List[int] = []
    if per_flow_rows:
        for r in per_flow_rows:
            p = parse_path_string(r.get("path", ""))
            if p:
                lens.append(max(0, len(p) - 1))
    elif ksp_obj:
        for key, paths in ksp_obj.items():
            if paths:
                lens.append(max(0, len(paths[0]) - 1))
    if not lens:
        return

    plt.figure()
    plt.hist(lens, bins=min(20, max(5, len(set(lens)))))
    plt.xlabel("Path hops")
    plt.ylabel("Count")
    plt.title("Path length histogram")
    out = plots_dir / "path_len_hist.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_cost_cdf(per_flow_rows: List[dict], G: nx.Graph, plots_dir: Path):
    costs: List[float] = []
    for r in per_flow_rows:
        p = parse_path_string(r.get("path", ""))
        if p:
            c = float(r.get("path_cost", 0.0) or 0.0)
            if c <= 0:
                c = path_cost(G, p, weight="weight")
            costs.append(c)
    if not costs:
        return
    xs = sorted(costs)
    ys = [i / len(xs) for i in range(1, len(xs) + 1)]

    plt.figure()
    plt.plot(xs, ys, drawstyle="steps-post")
    plt.xlabel("Path cost")
    plt.ylabel("CDF")
    plt.title("Per-flow path cost CDF")
    out = plots_dir / "cost_cdf.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_residual_cdf(residuals: List[float], plots_dir: Path):
    """
    以 residual = reserve*(1-headroom) - usage 的集合畫 CDF。
    含負值（代表違規）；可直接觀察 tail 風險。
    """
    if not residuals:
        return
    xs = sorted(residuals)
    ys = [i / len(xs) for i in range(1, len(xs) + 1)]

    plt.figure()
    plt.plot(xs, ys, drawstyle="steps-post")
    plt.xlabel("Residual Mbps (limit - usage)")
    plt.ylabel("CDF")
    plt.title("Per-edge per-class residual CDF")
    out = plots_dir / "residual_cdf.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


# ------------------------- CLI -------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Verify ECMP/KSP consistency, Admission, and Cost monotonicity")
    ap.add_argument("--edges", type=str, default="outputs/matrix/edges_all.csv")
    ap.add_argument("--fdb",   type=str, default="outputs/paths/fdb_spb_ecmp.json")
    ap.add_argument("--ksp",   type=str, default="outputs/paths/ksp_all_pairs.json")
    ap.add_argument("--flows", type=str, default="outputs/flows/per_flow.csv")
    ap.add_argument("--qos",   type=str, default="configs/qos.yaml")
    ap.add_argument("--reroutes", type=str, default="outputs/flows/reroutes.csv",
                    help="若存在則用於成本切換檢查；否則用候選 Δcost 檢查")
    ap.add_argument("--reports-dir", type=str, default="reports")
    ap.add_argument("--plots-dir",   type=str, default="plots")
    ap.add_argument("--theta", type=float, default=0.05, help="等價/改善門檻（5%）")
    return ap


def main():
    args = build_cli().parse_args()

    edges_csv = Path(args.edges)
    fdb_json  = Path(args.fdb)
    ksp_json  = Path(args.ksp)
    flows_csv = Path(args.flows)
    qos_yaml  = Path(args.qos)
    reroutes_csv = Path(args.reroutes) if args.reroutes else None

    reports_dir = Path(args.reports_dir)
    plots_dir   = Path(args.plots_dir)
    ensure_dirs(reports_dir, plots_dir)

    # Load data
    G = load_edges_graph(edges_csv)
    fdb_obj = load_json(fdb_json) if fdb_json.exists() else {}
    ksp_obj = load_json(ksp_json) if ksp_json.exists() else {}
    per_flow_rows = load_per_flow(flows_csv)
    qos = load_yaml(qos_yaml)

    # 1) ECMP vs KSP
    if fdb_obj and ksp_obj:
        checked, mismatches = check_ecmp_vs_ksp(G, fdb_obj, ksp_obj, reports_dir)
        print(f"[ECMP/KSP] checked={checked}, mismatches={mismatches} -> {reports_dir/'ecmp_ksp_check.csv'}")
    else:
        print("[ECMP/KSP] skip (missing FDB or KSP JSON)")

    # 2) Admission
    residuals: List[float] = []
    if qos and per_flow_rows:
        checked, violations, residuals = check_admission(G, qos, per_flow_rows, reports_dir)
        print(f"[Admission] checked={checked}, violations={violations} -> {reports_dir/'admission_check.csv'}")
    else:
        print("[Admission] skip (missing qos.yaml or per_flow.csv)")

    # 3) Cost monotonicity / Δcost
    checked, violations = check_cost_trend(G, per_flow_rows, reports_dir, theta=float(args.theta), reroutes_csv=reroutes_csv)
    print(f"[Cost] checked={checked}, violations={violations} -> {reports_dir/'cost_trend.csv'}")

    # Plots
    plot_path_len_hist(per_flow_rows, ksp_obj, plots_dir)
    plot_cost_cdf(per_flow_rows, G, plots_dir)
    plot_residual_cdf(residuals, plots_dir)
    print(f"[Plots] saved to {plots_dir}/path_len_hist.png, {plots_dir}/cost_cdf.png and {plots_dir}/residual_cdf.png")


if __name__ == "__main__":
    main()
