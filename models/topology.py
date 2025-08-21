# models/topology.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import csv, networkx as nx

@dataclass
class EdgeMetrics:
    delay_ms: float = 1.0
    loss: float = 0.0
    bw_mbps: float = 1000.0
    etx: float = 1.0
    rssi: float = -50.0
    snr: float = 30.0

@dataclass
class Topology:
    G: nx.Graph = field(default_factory=nx.Graph)
    # 每條邊、每個 Class 的保留與佔用（做 Queue/Admission 用）
    reserved: Dict[Tuple[str,str], Dict[str,float]] = field(default_factory=dict)
    usage:    Dict[Tuple[str,str], Dict[str,float]] = field(default_factory=dict)

    @staticmethod
    def from_csv(p: Path) -> "Topology":
        topo = Topology()
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u, v = row["src_switch"].strip(), row["dst_switch"].strip()
                w = float(row.get("weight", 1.0))
                bw = float(row.get("bw_mbps", 1000.0))
                topo.G.add_edge(u, v,
                    w=w,
                    metrics=EdgeMetrics(
                        delay_ms=float(row.get("delay_ms", 1.0)),
                        loss=float(row.get("loss", 0.0)),
                        bw_mbps=bw,
                        etx=float(row.get("etx", 1.0)),
                        rssi=float(row.get("rssi", -50.0)),
                        snr=float(row.get("snr", 30.0)),
                    )
                )
        return topo

    def edge_key(self, u, v):
        return (u, v) if u <= v else (v, u)
