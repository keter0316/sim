#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小動態模擬（不改 run_flows.py）
- 每個 tick 重新做一次 Admission，觀察 EF（vc/#）在流量升高時是否被擋下
- 輸出每個 tick 的結果與 reserved/usage 快照（可被 JSON 正確序列化）
"""

from __future__ import annotations

# --- bootstrapping sys.path so this file works when run directly ---
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # .../sim
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- end bootstrap ---

# 直接沿用 run_flows.py 裡的工具與常數（用絕對匯入）
from tools.run_flows import (
    load_graph, pick_path_for_flow, load_pins,
    EDGE_CSV, KSP_JSON, BASE
)
from models.queues import load_qos, build_reservations

OUT_DIR = BASE / "dynamic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_flows(t: int):
    """
    簡單的時間變化：
    - vc/#：每個 tick +5 Mbps（測 EF 保留是否足夠）
    - telemetry/#：固定 2 Mbps
    - bulk/#：固定 50 Mbps
    """
    return [
        {"src": "s1", "dst": "s5", "topic": "vc/#",         "rate_mbps": 10 + 5*t},
        {"src": "s2", "dst": "s9", "topic": "telemetry/#",  "rate_mbps": 2},
        {"src": "s8", "dst": "s3", "topic": "bulk/#",       "rate_mbps": 50},
    ]

def _edges_to_str_keys(d: dict) -> dict:
    """把 {(u,v): {...}} 改成 {'u-v': {...}}，可被 json 序列化。"""
    out = {}
    for k, v in d.items():
        if isinstance(k, tuple) and len(k) == 2:
            out[f"{k[0]}-{k[1]}"] = v
        else:
            out[str(k)] = v
    return out

def main():
    G    = load_graph(EDGE_CSV)
    qos  = load_qos()
    ksp  = json.loads((KSP_JSON).read_text())
    pins = load_pins()

    TICKS = 6  # 跑 6 個時間點：t=0..5

    all_results = []
    last_reserved = {}
    last_usage = {}

    for t in range(TICKS):
        # 每個 tick 重建 reservation/usage（簡單版）
        reserved, usage = build_reservations(G, qos)

        flows = make_flows(t)
        tick_results = []
        print(f"\n=== tick {t} ===")
        for f in flows:
            f["src"] = f["src"].lower()
            f["dst"] = f["dst"].lower()
            r = pick_path_for_flow(G, ksp, f, qos, reserved, usage, pins)
            rec = {**f, **r, "tick": t}
            tick_results.append(rec)
            print(f"[{r['result']}] {f['src']}→{f['dst']} {f['topic']} "
                  f"class={r['class']} rate={f['rate_mbps']} path={r['path']} reason={r.get('reason')}")

        # 輸出每個 tick 的結果與快照（edge key 轉成字串）
        (OUT_DIR / f"results_t{t}.json").write_text(json.dumps(tick_results, indent=2, ensure_ascii=False))
        (OUT_DIR / f"snapshot_t{t}.json").write_text(json.dumps({
            "reserved": _edges_to_str_keys(reserved),
            "usage":    _edges_to_str_keys(usage),
        }, indent=2, ensure_ascii=False))

        all_results.extend(tick_results)
        last_reserved, last_usage = reserved, usage

    # 總表
    (OUT_DIR / "results_all.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    # 簡單 KPI：各類 class 的 OK/BLOCKED 統計
    kpi = {}
    for r in all_results:
        key = (r["class"], r["result"])
        kpi[key] = kpi.get(key, 0) + 1
    (OUT_DIR / "kpi.json").write_text(json.dumps(
        {f"{k[0]}::{k[1]}": v for k, v in kpi.items()}, indent=2, ensure_ascii=False
    ))

    print("\n[OK] dynamic outputs ->", OUT_DIR)

if __name__ == "__main__":
    main()
