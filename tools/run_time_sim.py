#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Queue+Buffer time-engine simulator
"""

# --- bootstrapping sys.path so this file works when run directly ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # .../sim
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- end bootstrap ---

import argparse, csv
import numpy as np

from models.queue_runtime import Packet, ECNProfile, Link
from models.schedulers import HybridSP_DRR

DEF_WEIGHTS = {1:12, 2:10, 3:8, 4:6, 5:4, 6:2, 7:1}

def build_link(cap_gbps: float = 1.0) -> Link:
    cap_bps = int(cap_gbps * 1_000_000_000)
    buf = {q: 512_000 for q in range(8)}  # 512KB/queue；可改
    ecn = {q: ECNProfile(kmin=128_000, kmax=384_000, pmax=0.1) for q in range(8)}
    # Q0 為 SP；其餘以權重換算量子（近似 WFQ）
    base_q = 1500
    quanta = {q: base_q * DEF_WEIGHTS.get(q, 1) for q in range(1, 8)}
    sched = HybridSP_DRR(sp_set=[0], wfq_set=[1,2,3,4,5,6,7], quantums=quanta)
    return Link("L0", cap_bps, buf, ecn, sched)

def main():
    ap = argparse.ArgumentParser(description="Queue+Buffer time-engine simulator")
    ap.add_argument("--duration", type=float, default=1.0, help="模擬秒數，例如 10.0")
    ap.add_argument("--dt", type=float, default=0.001, help="步長秒（預設 1ms）")
    ap.add_argument("--rate-mbps", type=float, default=800, help="到達平均速率 (Mbps)")
    ap.add_argument("--mean-bytes", type=int, default=800, help="平均封包大小（bytes）")
    ap.add_argument("--burstiness", type=float, default=1.5, help="越大越 bursty")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=Path, default=Path("outputs/runtime/perhop_demo.csv"))
    ap.add_argument("--log", choices=["all","deq","none"], default="all",
                    help="記錄全部事件/只記deq/不記錄事件")
    ap.add_argument("--log-sample", type=int, default=1,
                    help="每 N 筆事件才寫一筆（降採樣）")
    ap.add_argument("--summary", type=Path, default=Path("outputs/runtime/summary.csv"))
    ap.add_argument("--sample-dt", type=float, default=0.001, help="佇列深度取樣間隔秒")
    ap.add_argument("--qdepth-csv", type=Path, default=Path("outputs/runtime/queue_depth.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    link = build_link(cap_gbps=1.0)

    now = 0.0
    pkt_id = 0
    out_rows = []
    next_sample_t = 0.0  # 下一次 depth 取樣時間

    # 平均 rate → 每步期望到達 bytes
    mean_bps = args.rate_mbps * 1_000_000
    lam_bytes_per_dt = mean_bps * args.dt / 8

    while now < args.duration:
        # 到達（Poisson）+ burst（Gamma）
        bytes_this_dt = rng.poisson(lam_bytes_per_dt)
        if args.burstiness > 1:
            bytes_this_dt = int(bytes_this_dt * rng.gamma(args.burstiness, 1/args.burstiness))

        remain = bytes_this_dt
        while remain > 0:
            sz = max(64, int(rng.normal(args.mean_bytes, args.mean_bytes*0.2)))
            pkt = Packet(pkt_id=pkt_id, size_bytes=sz, qid=rng.integers(0,8))
            ok, act = link.enqueue(pkt, now)
            if args.log == "all" and (pkt_id % args.log_sample == 0):
                out_rows.append([now, "enq", link.name, pkt.qid, pkt_id, sz, act])
            pkt_id += 1
            remain -= sz

        # 服務
        sent = link.service(now, args.dt)
        for p in sent:
            qdelay = (p.start_service_ts or now) - p.enqueue_ts
            sojourn = (p.depart_ts or now) - p.enqueue_ts
            if args.log in ("all","deq") and (p.pkt_id % args.log_sample == 0):
                out_rows.append([now, "deq", link.name, p.qid, p.pkt_id, p.size_bytes,
                                 f"qdelay={qdelay:.6f}|sojourn={sojourn:.6f}|ecn={int(p.marks.get('ecn', False))}"])

        # 取樣 queue 深度（可與 dt 不同步）
        if now + 1e-12 >= next_sample_t:
            link.sample_depth(now)
            next_sample_t += args.sample_dt

        now += args.dt

    # --- 摘要統計輸出 ---
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as fsum:
        sw = csv.writer(fsum)
        sw.writerow([
            "enq","deq","drop","ecn_mark",
            "bytes_served_b","duration_s","served_rate_mbps","arrive_rate_mbps_est"
        ])
        served_rate = 8 * link.stats["bytes_served"] / args.duration / 1e6
        arrive_bytes_est = int(args.mean_bytes) * link.stats["enq"]
        arrive_rate_est = 8 * arrive_bytes_est / args.duration / 1e6
        sw.writerow([
            link.stats["enq"], link.stats["deq"], link.stats["drop"], link.stats["mark"],
            link.stats["bytes_served"], args.duration, f"{served_rate:.2f}", f"{arrive_rate_est:.2f}"
        ])

    # 事件輸出
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["t","event","link","qid","pkt_id","size","extra"])
        cw.writerows(out_rows)

    # 佇列深度輸出
    args.qdepth_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.qdepth_csv.open("w", newline="", encoding="utf-8") as fqd:
        cw = csv.writer(fqd)
        cw.writerow(["t","link","qid","bytes_in_q"])
        cw.writerows(link.depth_rows)

if __name__ == "__main__":
    main()
