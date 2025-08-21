#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt

BASE = Path("outputs") / "reservation"
RESERVED_CSV = BASE / "reserved.csv"
USAGE_TOT_CSV = BASE / "edge_usage_totals.csv"
DETAIL_CSV = BASE / "edge_usage_detail.csv"

OUT = Path("outputs") / "charts"
OUT.mkdir(parents=True, exist_ok=True)

def read_reserved():
    # returns: dict[(edge)][class] = reserved_mbps
    res = {}
    with RESERVED_CSV.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            e = r["edge"]; c = r["class"]; v = float(r["reserved_mbps"])
            res.setdefault(e, {})[c] = v
    return res

def read_usage():
    # returns: dict[(edge)][class] = usage_mbps
    use = {}
    with USAGE_TOT_CSV.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            e = r["edge"]; c = r["class"]; v = float(r["usage_mbps"])
            use.setdefault(e, {})[c] = v
    return use

def read_detail():
    rows = []
    with DETAIL_CSV.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            r["rate_mbps"] = float(r["rate_mbps"])
            rows.append(r)
    return rows

def compute_util(reserved, usage):
    # returns: util[(edge)][class] = usage/reserved (0~1); also writes CSV
    classes = sorted({c for e in reserved for c in reserved[e].keys()} |
                     {c for e in usage    for c in usage[e].keys()})
    edges = sorted(set(reserved.keys()) | set(usage.keys()))
    util = {}
    out_csv = OUT / "edge_utilization.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["edge","class","usage_mbps","reserved_mbps","util_ratio"])
        for e in edges:
            util[e] = {}
            for c in classes:
                r = reserved.get(e, {}).get(c, 0.0)
                u = usage.get(e, {}).get(c, 0.0)
                ratio = (u / r) if r > 0 else 0.0
                util[e][c] = ratio
                wr.writerow([e, c, u, r, ratio])
    return util

def plot_heatmap(util):
    # edges × classes heatmap
    classes = sorted({c for e in util for c in util[e].keys()})
    edges = sorted(util.keys())
    mat = []
    for e in edges:
        row = [util[e].get(c, 0.0) for c in classes]
        mat.append(row)

    fig = plt.figure(figsize=(max(6, len(classes)*0.6), max(6, len(edges)*0.3)))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(edges)))
    ax.set_yticklabels(edges)
    ax.set_title("Edge × Class Utilization (usage/reserved)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = OUT / "heatmap_edge_class_util.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[OK] wrote", p)

def plot_topn_per_class(util, reserved, usage, topn=10):
    classes = sorted({c for e in util for c in util[e].keys()})
    for c in classes:
        items = []
        for e, m in util.items():
            ratio = m.get(c, 0.0)
            items.append((ratio, e))
        items.sort(reverse=True)
        top = items[:topn]

        ratios = [x[0] for x in top]
        edges  = [x[1] for x in top]

        fig = plt.figure(figsize=(10, max(4, len(top)*0.4)))
        ax = plt.gca()
        ax.barh(range(len(top)), ratios)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(edges)
        ax.invert_yaxis()
        ax.set_xlabel("utilization (usage/reserved)")
        ax.set_title(f"Top {topn} edges by utilization — class {c}")
        fig.tight_layout()
        p = OUT / f"top{topn}_edges_util_{c}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("[OK] wrote", p)

def plot_edge_detail(edge_name, reserved, usage):
    # bar chart: per-class usage vs reserved on a single edge
    classes = sorted({c for c in reserved.get(edge_name, {})} |
                     {c for c in usage.get(edge_name, {})})
    u = [usage.get(edge_name, {}).get(c, 0.0) for c in classes]
    r = [reserved.get(edge_name, {}).get(c, 0.0) for c in classes]
    x = list(range(len(classes)))

    fig = plt.figure(figsize=(max(6, len(classes)*0.6), 5))
    ax = plt.gca()
    ax.bar(x, r, alpha=0.5, label="reserved (Mbps)")
    ax.bar(x, u, alpha=0.8, label="usage (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_title(f"Edge {edge_name}: usage vs reserved")
    ax.legend()
    fig.tight_layout()
    p = OUT / f"edge_detail_{edge_name.replace('/', '_')}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[OK] wrote", p)

def main():
    if not (RESERVED_CSV.exists() and USAGE_TOT_CSV.exists()):
        print("[ERR] missing CSVs under", BASE)
        return
    reserved = read_reserved()
    usage = read_usage()
    # detail = read_detail()  # 若之後要做 flow 級別挑邊分析可用

    util = compute_util(reserved, usage)
    plot_heatmap(util)
    plot_topn_per_class(util, reserved, usage, topn=10)

    # 再挑幾條邊做「usage vs reserved」的對照圖
    # 這裡挑利用率最高的 3 條（針對全 class 的 max ratio）
    tops = []
    for e, m in util.items():
        mx = max(m.values()) if m else 0.0
        tops.append((mx, e))
    tops.sort(reverse=True)
    for _, e in tops[:3]:
        plot_edge_detail(e, reserved, usage)

if __name__ == "__main__":
    main()
