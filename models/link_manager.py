from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable
import csv

from models.queue_runtime import Link, ECNProfile
from models.schedulers import HybridSP_DRR

DEF_WEIGHTS = {1:12, 2:10, 3:8, 4:6, 5:4, 6:2, 7:1}

def _make_link(name: str, cap_mbps: float = 1000.0) -> Link:
    cap_bps = int(cap_mbps * 1e6)
    buf = {q: 512_000 for q in range(8)}
    ecn = {q: ECNProfile(kmin=128_000, kmax=384_000, pmax=0.1) for q in range(8)}
    base_q = 1500
    quanta = {q: base_q * DEF_WEIGHTS.get(q, 1) for q in range(1, 8)}
    sched = HybridSP_DRR(sp_set=[0], wfq_set=[1,2,3,4,5,6,7], quantums=quanta)
    return Link(name=name, capacity_bps=cap_bps, buf_caps=buf, ecn=ecn, scheduler=sched)

_SRC_KEYS = {"src","source","u","from","node_u","node1","a","s"}
_DST_KEYS = {"dst","destination","v","to","node_v","node2","b","t"}
_CAP_KEYS = {"capacity_mbps","cap_mbps","mbps","bandwidth_mbps","bw_mbps","capacity","bandwidth"}

class LinkManager:
    """
    從 edges.csv 建立「有向」Link：
    - 自動辨識欄位：src/dst（大小寫不敏感，多種別名）
    - capacity_mbps 可省略（預設 1000）
    """
    def __init__(self, edges_csv: Path, default_mbps: float = 1000.0):
        self.links: Dict[Tuple[str, str], Link] = {}

        with Path(edges_csv).open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames is None:
                raise RuntimeError(f"No header found in {edges_csv}")

            # 欄位名小寫化
            fields = [c.strip() for c in rd.fieldnames]
            lower_map = {c.lower(): c for c in fields}

            # 嘗試映射 src/dst/cap 欄位
            def pick(cands):
                for k in cands:
                    if k in lower_map:
                        return lower_map[k]
                return None

            col_src = pick(_SRC_KEYS)
            col_dst = pick(_DST_KEYS)
            col_cap = pick(_CAP_KEYS)

            # 萬一對不上，就用前兩欄當 src/dst
            if col_src is None or col_dst is None:
                col_src = fields[0]
                col_dst = fields[1]

            for row in rd:
                u = str(row[col_src]).strip()
                v = str(row[col_dst]).strip()
                if not u or not v:
                    continue
                mbps = default_mbps
                if col_cap and row.get(col_cap, "") not in (None, "", "NaN"):
                    try:
                        mbps = float(row[col_cap])
                    except ValueError:
                        mbps = default_mbps
                # 雙向
                self.links[(u, v)] = _make_link(f"{u}-{v}", cap_mbps=mbps)
                self.links[(v, u)] = _make_link(f"{v}-{u}", cap_mbps=mbps)

    def get(self, u: str, v: str) -> Link:
        return self.links[(u, v)]

    def all_links(self) -> Iterable[Link]:
        return self.links.values()
