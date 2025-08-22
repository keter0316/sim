from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque
import random

if TYPE_CHECKING:
    from models.schedulers import BaseScheduler, QueueState

# ----------------- Data models -----------------

@dataclass
class Packet:
    pkt_id: int
    size_bytes: int
    ecn_capable: bool = True
    enqueue_ts: float = 0.0
    start_service_ts: Optional[float] = None
    depart_ts: Optional[float] = None
    remaining_bytes: int = 0
    topic: str = ""
    qid: int = 7
    marks: Dict[str, bool] = field(default_factory=dict)
    meta: Dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining_bytes == 0:
            self.remaining_bytes = self.size_bytes


@dataclass
class ECNProfile:
    kmin: int
    kmax: int
    pmax: float = 0.1  # 0~1


@dataclass
class Link:
    name: str
    capacity_bps: int                 # e.g. 1_000_000_000 (1Gbps)
    buf_caps: Dict[int, int]          # per-queue buffer cap (bytes)
    ecn: Dict[int, ECNProfile]
    scheduler: "BaseScheduler"

    def __post_init__(self):
        from models.schedulers import QueueState  # late import to avoid cycles
        self.queues: Dict[int, QueueState] = {qid: QueueState(qid=qid, pkts=deque()) for qid in range(8)}
        # 目前佇列中剩餘的 bytes（包含 HOL 封包未送完的剩餘部分）
        self.bytes_in_q: Dict[int, int] = {qid: 0 for qid in range(8)}

        # 全域統計
        self.stats = {"enq": 0, "deq": 0, "drop": 0, "mark": 0, "bytes_served": 0}

        # per-queue 統計
        self.perq_stats: Dict[int, Dict[str, int]] = {
            qid: {"enq": 0, "deq": 0, "drop": 0, "mark": 0, "served_b": 0} for qid in range(8)
        }

        # 佇列深度時間序列（給外部寫 CSV）
        self.depth_rows: List[List[object]] = []  # [t, link, qid, bytes_in_q]

    # ----------------- Operations -----------------

    # 入佇列（含 WRED/ECN 與 Drop-tail）
    def enqueue(self, pkt: Packet, now: float) -> Tuple[bool, str]:
        qid = pkt.qid
        qbytes = self.bytes_in_q[qid]
        cap = self.buf_caps.get(qid, 1_000_000)

        # 滿了 → drop-tail
        if qbytes + pkt.size_bytes > cap:
            self.stats["drop"] += 1
            self.perq_stats[qid]["drop"] += 1
            return False, "drop_tail"

        # ECN/WRED 線性標記
        prof = self.ecn.get(qid)
        if prof and pkt.ecn_capable:
            if qbytes >= prof.kmax:
                pkt.marks["ecn"] = True
                self.stats["mark"] += 1
                self.perq_stats[qid]["mark"] += 1
            elif qbytes > prof.kmin:
                p = prof.pmax * (qbytes - prof.kmin) / max(1, (prof.kmax - prof.kmin))
                if random.random() < p:
                    pkt.marks["ecn"] = True
                    self.stats["mark"] += 1
                    self.perq_stats[qid]["mark"] += 1

        pkt.enqueue_ts = now
        self.queues[qid].pkts.append(pkt)
        self.bytes_in_q[qid] += pkt.size_bytes
        self.stats["enq"] += 1
        self.perq_stats[qid]["enq"] += 1
        return True, "enq"

    # 在 Δt 內可服務的 bytes，採 byte-level 逐步消耗（正確反映 queue depth）
    def service(self, now: float, delta_t: float) -> List[Packet]:
        budget = int(self.capacity_bps * delta_t / 8)
        sent: List[Packet] = []

        while budget > 0:
            qid = self.scheduler.pick(self.queues)
            if qid is None:
                break  # 無封包

            qs = self.queues[qid]
            if not qs.pkts:
                continue

            pkt = qs.pkts[0]
            if pkt.start_service_ts is None:
                pkt.start_service_ts = now

            take = min(pkt.remaining_bytes, budget)
            pkt.remaining_bytes -= take
            budget -= take

            # 即時更新：服務出去的 bytes 立刻離開佇列深度
            self.bytes_in_q[qid] -= take
            if self.bytes_in_q[qid] < 0:
                self.bytes_in_q[qid] = 0  # 安全防呆（理論上不會 < 0）

            self.stats["bytes_served"] += take
            self.perq_stats[qid]["served_b"] += take

            # 若是 DRR/WFQ，更新 deficit
            if hasattr(qs, "deficit"):
                qs.deficit -= take

            # 封包完成
            if pkt.remaining_bytes == 0:
                qs.pkts.popleft()
                # 最後一段 take 的序列化時間：take / (capacity_bytes_per_sec)
                pkt.depart_ts = now + (take / max(1, self.capacity_bps / 8))
                sent.append(pkt)
                self.stats["deq"] += 1
                self.perq_stats[qid]["deq"] += 1

            if budget <= 0:
                break

        return sent

    # 由外部時間引擎在固定間隔呼叫，取樣各 queue 深度
    def sample_depth(self, now: float) -> None:
        for qid in range(8):
            self.depth_rows.append([now, self.name, qid, self.bytes_in_q[qid]])
