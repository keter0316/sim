# models/schedulers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.queue_runtime import Packet

@dataclass
class QueueState:
    qid: int
    pkts: Deque["Packet"]
    deficit: int = 0
    quantum: int = 1500  # bytes

class BaseScheduler:
    def pick(self, queues: Dict[int, QueueState]) -> Optional[int]:
        raise NotImplementedError("BaseScheduler.pick not implemented")

class StrictPriorityScheduler(BaseScheduler):
    def __init__(self, sp_set: List[int]):
        # 保留呼叫者給的順序；前面的優先
        self.order = list(sp_set)

    def pick(self, queues: Dict[int, QueueState]) -> Optional[int]:
        for qid in self.order:
            qs = queues.get(qid)
            if qs and qs.pkts:
                return qid
        return None

class DRRScheduler(BaseScheduler):
    def __init__(self, serve_set: List[int], quantums: Dict[int,int]):
        self.order = list(serve_set)
        self.q = dict(quantums)
        self.idx = 0

    def pick(self, queues: Dict[int, QueueState]) -> Optional[int]:
        n = len(self.order)
        if n == 0:
            return None
        for _ in range(n):
            qid = self.order[self.idx]
            self.idx = (self.idx + 1) % n
            qs = queues.get(qid)
            if not qs or not qs.pkts:
                continue
            qs.deficit += self.q.get(qid, qs.quantum)
            head = qs.pkts[0]
            if head.remaining_bytes <= qs.deficit:
                return qid
        return None

class HybridSP_DRR(BaseScheduler):
    def __init__(self, sp_set: List[int], wfq_set: List[int], quantums: Dict[int,int]):
        self.sp = StrictPriorityScheduler(sp_set)
        self.drr = DRRScheduler(wfq_set, quantums)

    def pick(self, queues: Dict[int, QueueState]) -> Optional[int]:
        qid = self.sp.pick(queues)
        if qid is not None:
            return qid
        return self.drr.pick(queues)
