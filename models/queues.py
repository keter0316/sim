# models/queues.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import yaml

# 統一類別名稱的小工具（別名➡標準）
CLASS_ALIAS = {
    "CS0": "BE",   # 你的 topic_map 用 CS0；程式內部統一用 BE
}
def norm_class(cls: str) -> str:
    return CLASS_ALIAS.get(cls, cls)

# 佇列/類別對應（供查詢時使用；不強制要求 YAML 一致）
QUEUE_OF = {
    "EF":   "Q0",
    "CS5":  "Q1",
    "AF41": "Q2",
    "AF31": "Q3",
    "AF21": "Q4",
    "CS3":  "Q5",
    "AF11": "Q6",
    "BE":   "Q7",
}

# 降級鏈（Admission 失敗時依序降級）
DOWNGRADE_CHAIN = {
    "EF":   ["CS5", "AF41", "AF31", "AF21", "AF11", "BE"],
    "CS5":  ["AF41", "AF31", "AF21", "AF11", "BE"],
    "AF41": ["AF31", "AF21", "AF11", "BE"],
    "AF31": ["AF21", "AF11", "BE"],
    "AF21": ["AF11", "BE"],
    "AF11": ["BE"],
    "BE":   []
}

@dataclass
class QoSConfig:
    # reservations 允許兩種型態：
    #   - 比例: {"EF":0.15, "CS5":0.10, ...}
    #   - 絕對: {"EF":{"min_mbps":100,...}, "CS5":{"min_mbps":50,...}, ...}
    reservations: Dict[str, Any]
    # topic_map 統一轉成 [{"match":"vc/#","class":"EF","queue":0,...}, ...]
    topic_map: List[Dict[str, Any]]

def load_qos(path="configs/qos.yaml") -> QoSConfig:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    # 將 topic_map（原本是 dict）轉成列表，並把 class 正規化
    tm_list: List[Dict[str, Any]] = []
    for pat, attrs in (y.get("topic_map") or {}).items():
        cls = norm_class(str(attrs.get("qos", "BE")))
        tm_list.append({
            "match": pat,
            "class": cls,
            "queue": attrs.get("queue"),
            "dscp":  attrs.get("dscp"),
            "pcp":   attrs.get("pcp"),
            "prio":  attrs.get("prio"),
            "importance": attrs.get("importance"),
        })
    return QoSConfig(reservations=y.get("reservations", {}), topic_map=tm_list)

def classify(topic: str, qos: QoSConfig) -> str:
    # 只看到 '#' 的前綴
    for rule in qos.topic_map:
        pat = rule["match"].split("#", 1)[0]
        if topic.startswith(pat):
            return rule["class"]
    return "BE"

def _is_ratio_reservations(resv: Dict[str, Any]) -> bool:
    """
    判斷 reservations 是比例型(純數值或<=1.2的小數) 還是絕對型(dict 含 min_mbps)。
    """
    # 如果值是 dict（含 min_mbps），明顯是絕對
    if any(isinstance(v, dict) for v in resv.values()):
        return False
    # 都是數字，且總和落在(0, 1.2]，視為比例
    try:
        total = sum(float(v) for v in resv.values())
        return 0 < total <= 1.2
    except Exception:
        return False

def build_reservations(G, qos: QoSConfig):
    """
    回傳 (reserved, usage):
      reserved[(u,v)][class] = 以 Mbps 為單位的保留值(AF 類別會匯總到 'AF' 池）
      usage[(u,v)][class]    = 以 Mbps 為單位的已用值

    同時支援：
      1) 比例:EF: 0.15   或  EF: {share: 0.15, weight:..., queue:...}
      2) 絕對:EF: {min_mbps: 100, ...}

    另外支援 headroom(預設 0.10)，即僅以 (1 - headroom) * bw_mbps 作為可分配底盤。
    """
    def _get_share(spec):
        """回傳 share(float) 或 None;接受純數字或 dict{'share': x}。"""
        if isinstance(spec, (int, float, str)):
            try:
                return float(spec)
            except Exception:
                return None
        if isinstance(spec, dict):
            v = spec.get("share", None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    def _get_min_mbps(spec):
        """回傳 min_mbps(float) 或 None;只在 dict 中讀 'min_mbps'。"""
        if isinstance(spec, dict) and "min_mbps" in spec:
            try:
                return float(spec["min_mbps"])
            except Exception:
                return None
        return None

    # 判定是否為比例模式：只要任一類別具有可解析的 share，就視為比例模式
    res_cfg = getattr(qos, "reservations", {}) or {}
    any_share = any(_get_share(spec) is not None for spec in res_cfg.values())
    any_min   = any(_get_min_mbps(spec) is not None for spec in res_cfg.values())
    is_ratio  = any_share and not any_min  # 若同時存在，以 min_mbps 優先（可依需求調整）

    headroom = float(getattr(qos, "headroom", 0.10) or 0.0)  # 預設 10%
    headroom = max(0.0, min(headroom, 0.9))  # 簡單夾限

    reserved, usage = {}, {}

    for u, v, d in G.edges(data=True):
        bw = float(d.get("bw_mbps", 10_000.0))           # 邊容量（Mbps）
        eff_bw = bw * (1.0 - headroom)                   # 可分配底盤（Mbps）
        key = tuple(sorted((u, v)))
        r: Dict[str, float] = {}

        if is_ratio:
            # 比例模式：讀 share（純數或 dict.share）
            for cls, spec in res_cfg.items():
                cls_n = norm_class(cls)
                share = _get_share(spec)
                if share is None:
                    continue
                val = eff_bw * float(share)              # Mbps
                pool = "AF" if cls_n.startswith("AF") else cls_n
                r[pool] = r.get(pool, 0.0) + val
        else:
            # 絕對模式：讀 min_mbps（dict.min_mbps 或 數字也接受當 min_mbps）
            for cls, spec in res_cfg.items():
                cls_n = norm_class(cls)
                mm = _get_min_mbps(spec)
                if mm is None:
                    # 也接受「直接給數字」當成 min_mbps
                    if isinstance(spec, (int, float, str)):
                        try:
                            mm = float(spec)
                        except Exception:
                            mm = None
                val = float(mm or 0.0)                   # Mbps
                pool = "AF" if cls_n.startswith("AF") else cls_n
                r[pool] = r.get(pool, 0.0) + val

            # 避免總和超過可分配底盤 eff_bw
            total = sum(r.values())
            if total > eff_bw and total > 0.0:
                scale = eff_bw / total
                for k in list(r.keys()):
                    r[k] *= scale

        reserved[key] = r
        usage[key] = {k: 0.0 for k in r.keys()}

        # 確保常見池都有 key（避免查不到時 KeyError）
        for k in ["EF", "CS5", "AF", "CS3", "BE"]:
            reserved[key].setdefault(k, 0.0)
            usage[key].setdefault(k, 0.0)

    return reserved, usage


def admit_edge(edge_key: Tuple[str, str], cls: str, demand_mbps: float,
               reserved: Dict[Tuple[str, str], Dict[str, float]],
               usage:    Dict[Tuple[str, str], Dict[str, float]]) -> bool:
    r = reserved[edge_key]
    u = usage[edge_key]
    cls_n = norm_class(cls)
    pool_cls = "AF" if cls_n.startswith("AF") else cls_n
    ok = (r.get(pool_cls, 0.0) - u.get(pool_cls, 0.0)) >= demand_mbps
    if ok:
        u[pool_cls] = u.get(pool_cls, 0.0) + demand_mbps
    return ok

def release_edge(edge_key, cls, demand_mbps, reserved, usage):
    r = reserved[edge_key]
    u = usage[edge_key]
    cls_n = norm_class(cls)
    pool_cls = "AF" if cls_n.startswith("AF") else cls_n
    u[pool_cls] = max(0.0, u.get(pool_cls, 0.0) - demand_mbps)