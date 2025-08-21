# routing/grouping.py
from __future__ import annotations
from typing import List, Dict, Any
import re
"""
同一場會議/主題要先群組（把多個 pub+多個 sub 當一群，一次算一棵 Steiner-tree),不然會各自算,浪費路徑共享。
"""

def topic_session_key(topic: str) -> str:
    # 例：vc/room123/userA -> "vc/room123"
    if topic.startswith("vc/"):
        m = re.match(r"^(vc/[^/]+)", topic)
        if m: return m.group(1)
    return topic

def group_vc_multicast(flows: List[Dict[str,Any]]) -> Dict[str, Dict[str,Any]]:
    """
    將同一場 vc（相同 session_key）的流量合併：
    回傳 {session_key: {"publishers": set(), "subscribers": set(), "rate_mbps": total_rate, "flows": [idx...]}}
    """
    groups: Dict[str, Dict[str,Any]] = {}
    for i, f in enumerate(flows):
        topic = f.get("topic","")
        if not topic.startswith("vc/"): 
            continue
        dsts = f.get("dsts") or ([f["dst"]] if f.get("dst") else [])
        if len(dsts) < 2: 
            continue
        key = topic_session_key(topic)
        g = groups.setdefault(key, {"publishers": set(), "subscribers": set(), "rate_mbps": 0.0, "flows": []})
        g["publishers"].add(f["src"])
        g["subscribers"].update(dsts)
        g["rate_mbps"] += float(f.get("rate_mbps", 1.0))
        g["flows"].append(i)
    return groups
