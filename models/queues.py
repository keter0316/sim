# models/queues.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import csv
import yaml

# =========================
# 類別標準化與對應
# =========================

# 統一名稱：以 CS0 為標準；接受 BE 為別名
CLASS_ALIAS = {"BE": "CS0"}
CLASSES_STD = {"EF", "CS5", "AF41", "AF31", "AF21", "AF11", "CS3", "CS0"}

def norm_class(cls: str) -> str:
    s = (cls or "").strip().upper()
    s = CLASS_ALIAS.get(s, s)
    return s if s in CLASSES_STD else "CS0"

def pool_for(cls: str) -> str:
    """Admission 池規則：AF* 進同一個 'AF'；其餘用原 class。"""
    c = norm_class(cls)
    return "AF" if c.startswith("AF") else c

# 佇列對應（展示用）
QUEUE_OF = {
    "EF":   "Q0",  # 嚴格優先（預設）
    "CS5":  "Q1",
    "AF41": "Q2",
    "AF31": "Q3",
    "AF21": "Q4",
    "CS3":  "Q5",
    "AF11": "Q6",
    "CS0":  "Q7",
}

# 降級鏈（Admission 失敗時）
DOWNGRADE_CHAIN = {
    "EF":   ["CS5", "AF41", "AF31", "AF21", "AF11", "CS0"],
    "CS5":  ["AF41", "AF31", "AF21", "AF11", "CS0"],
    "AF41": ["AF31", "AF21", "AF11", "CS0"],
    "AF31": ["AF21", "AF11", "CS0"],
    "AF21": ["AF11", "CS0"],
    "AF11": ["CS0"],
    "CS0":  [],
}

# =========================
# QoS 設定載入
# =========================

@dataclass
class QoSConfig:
    reservations: Dict[str, Any]           # 每 class 的保留設定（min_mbps/share/weight/queue…）
    topic_map: List[Dict[str, Any]]        # [{match, class, queue, dscp, ...}]
    headroom: float = 0.10                 # Admission 安全餘裕（0..0.9）
    scheduler: str = "SP_WFQ"              # "SP_WFQ" | "WFQ" | "DRR" | "CBS"
    avg_pkt_bytes: int = 1000              # 平均封包大小（估算序列化時間用）
    max_queue_delay_ms: float = 100.0      # 壅塞時上限延遲（避免發散）

def load_qos(path: str = "configs/qos.yaml") -> QoSConfig:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # topic_map 支援新版 class 與舊版 qos
    tm_list: List[Dict[str, Any]] = []
    for pat, attrs in (y.get("topic_map") or {}).items():
        cls_key = attrs.get("class", attrs.get("qos", "CS0"))
        cls = norm_class(str(cls_key))
        tm_list.append({
            "match": pat,
            "class": cls,
            "queue": attrs.get("queue", None),
            "dscp":  attrs.get("dscp"),
            "pcp":   attrs.get("pcp"),
            "prio":  attrs.get("prio"),
            "importance": attrs.get("importance"),
        })

    headroom = float(y.get("headroom", 0.10) or 0.0)
    headroom = max(0.0, min(headroom, 0.9))

    scheduler = str(y.get("scheduler", "SP_WFQ")).upper()
    if scheduler not in {"SP_WFQ", "WFQ", "DRR", "CBS"}:
        scheduler = "SP_WFQ"

    avg_pkt_bytes = int(y.get("avg_pkt_bytes", 1000) or 1000)
    max_q_ms = float(y.get("max_queue_delay_ms", 100.0) or 100.0)

    return QoSConfig(
        reservations=y.get("reservations", {}) or {},
        topic_map=tm_list,
        headroom=headroom,
        scheduler=scheduler,
        avg_pkt_bytes=avg_pkt_bytes,
        max_queue_delay_ms=max_q_ms,
    )

def classify(topic: str, qos: QoSConfig) -> str:
    """以 topic 前綴匹配 '#' 前的部分；回傳標準化 class。"""
    for rule in qos.topic_map:
        pat = (rule.get("match") or "").split("#", 1)[0]
        if topic.startswith(pat):
            return norm_class(rule.get("class"))
    return "CS0"

# =========================
# 建立 reserved / usage
# =========================

def _get_share(spec: Any) -> Optional[float]:
    if isinstance(spec, (int, float, str)):
        try: return float(spec)
        except Exception: return None
    if isinstance(spec, dict):
        v = spec.get("share", None)
        if v is not None:
            try: return float(v)
            except Exception: return None
    return None

def _get_min_mbps(spec: Any) -> Optional[float]:
    if isinstance(spec, dict) and "min_mbps" in spec:
        try: return float(spec["min_mbps"])
        except Exception: return None
    return None

def build_reservations(G, qos: QoSConfig):
    """
    回傳 (reserved, usage)：
      reserved[(u,v)][pool] = Mbps
      usage[(u,v)][pool]    = Mbps

    - 支援比例（share）與絕對（min_mbps）；若兩者同時存在，min_mbps 優先
    - AF* 共同歸入 'AF' 資源池（Admission 共用）
    - 以 headroom 套用到底盤：eff_bw = bw_mbps * (1 - headroom)
    """
    res_cfg = qos.reservations or {}
    any_share = any(_get_share(spec) is not None for spec in res_cfg.values())
    any_min   = any(_get_min_mbps(spec) is not None or isinstance(spec, (int,float,str)) for spec in res_cfg.values())
    is_ratio  = any_share and not any_min

    reserved: Dict[Tuple[str, str], Dict[str, float]] = {}
    usage:    Dict[Tuple[str, str], Dict[str, float]] = {}

    for u, v, d in G.edges(data=True):
        bw = float(d.get("bw_mbps", 10_000.0))
        eff_bw = bw * (1.0 - float(qos.headroom))
        key = tuple(sorted((u, v)))
        r: Dict[str, float] = {}

        if is_ratio:
            for cls, spec in res_cfg.items():
                cls_n = norm_class(cls)
                share = _get_share(spec)
                if share is None: continue
                val = eff_bw * float(share)
                r[pool_for(cls_n)] = r.get(pool_for(cls_n), 0.0) + val
        else:
            for cls, spec in res_cfg.items():
                cls_n = norm_class(cls)
                mm = _get_min_mbps(spec)
                if mm is None and isinstance(spec, (int, float, str)):
                    try: mm = float(spec)
                    except Exception: mm = None
                val = float(mm or 0.0)
                r[pool_for(cls_n)] = r.get(pool_for(cls_n), 0.0) + val

            total = sum(r.values())
            if total > eff_bw and total > 0.0:
                scale = eff_bw / total
                for k in list(r.keys()):
                    r[k] *= scale

        reserved[key] = r
        usage[key] = {k: 0.0 for k in r.keys()}
        # 確保常見池存在
        for k in ["EF", "CS5", "AF", "CS3", "CS0"]:
            reserved[key].setdefault(k, 0.0)
            usage[key].setdefault(k, 0.0)

    return reserved, usage

# =========================
# Admission API
# =========================

def admit_edge(edge_key: Tuple[str, str], cls: str, demand_mbps: float,
               reserved: Dict[Tuple[str, str], Dict[str, float]],
               usage:    Dict[Tuple[str, str], Dict[str, float]]) -> bool:
    r = reserved[edge_key]; u = usage[edge_key]
    pool = pool_for(cls)
    if demand_mbps <= 0: return True
    ok = (r.get(pool, 0.0) - u.get(pool, 0.0)) >= demand_mbps - 1e-12
    if ok:
        u[pool] = u.get(pool, 0.0) + float(demand_mbps)
    return ok

def release_edge(edge_key: Tuple[str, str], cls: str, demand_mbps: float,
                 reserved: Dict[Tuple[str, str], Dict[str, float]],
                 usage:    Dict[Tuple[str, str], Dict[str, float]]) -> None:
    u = usage[edge_key]; pool = pool_for(cls)
    if demand_mbps > 0:
        u[pool] = max(0.0, u.get(pool, 0.0) - float(demand_mbps))

# =========================
# 服務率分配（排程模式）
# =========================

def _weights_from_qos(qos: QoSConfig) -> Dict[str, float]:
    """從 reservations 讀 weight（沒有就給缺省）。回傳 pool->weight。"""
    defaults = {"EF": 1e9, "CS5": 12.0, "AF": 8.0, "CS3": 4.0, "CS0": 1.0}  # EF 非常大近似 SP
    out: Dict[str, float] = {"EF": defaults["EF"], "CS5":12.0, "AF":8.0, "CS3":4.0, "CS0":1.0}
    for cls, spec in (qos.reservations or {}).items():
        p = pool_for(cls)
        if isinstance(spec, dict) and "weight" in spec and spec["weight"] is not None:
            try: out[p] = float(spec["weight"])
            except Exception: pass
    return out

def _strict_set(qos: QoSConfig) -> set:
    """誰是嚴格優先？預設 EF。若你在 YAML 中為某 class 設 weight: null，亦視為 SP。"""
    strict = {"EF"}
    for cls, spec in (qos.reservations or {}).items():
        if isinstance(spec, dict) and spec.get("weight", None) is None and norm_class(cls) != "CS0":
            strict.add(pool_for(cls))
    return strict

def _share_wfq(weights: Dict[str, float], pools: List[str], cap_rem: float) -> Dict[str, float]:
    total_w = sum(max(0.0, float(weights.get(p, 0.0))) for p in pools) or 1.0
    return {p: cap_rem * float(weights.get(p, 0.0)) / total_w for p in pools}

def service_rates_mbps(
    link_capacity_mbps: float,
    qos: QoSConfig,
    arrivals_mbps: Dict[str, float],
) -> Dict[str, float]:
    """
    回傳各 pool 的服務率 μ（有效份額，Mbps）。預設 SP+WFQ：
    - 先滿足嚴格優先（EF），剩餘頻寬再按權重分配給其他 pool。
    - WFQ/DRR 模式：不做 SP，直接按權重（DRR 近似）。
    - CBS：以 WFQ 近似（如要更精確需額外 credit 模型）。
    """
    pools = ["EF", "CS5", "AF", "CS3", "CS0"]
    W = _weights_from_qos(qos)
    strict = _strict_set(qos) if qos.scheduler == "SP_WFQ" else set()
    mu: Dict[str, float] = {p: 0.0 for p in pools}

    C = max(1e-6, float(link_capacity_mbps))
    A = {p: max(0.0, float(arrivals_mbps.get(p, 0.0))) for p in pools}

    if qos.scheduler == "WFQ" or qos.scheduler == "DRR" or qos.scheduler == "CBS":
        shares = _share_wfq(W, pools, C)
        for p in pools:
            mu[p] = max(1e-6, shares.get(p, 0.0))  # 份額即服務率上限
        return mu

    # SP_WFQ：先扣嚴格優先
    cap_rem = C
    if strict:
        A_strict = sum(A.get(p, 0.0) for p in strict)
        use_sp = min(cap_rem, A_strict)  # 保證 EF/其他 SP 至少可達其到達率（上界為 C）
        # 給嚴格優先一個「接近 C」的 μ，避免誤算；亦可設為 min(cap_rem, C)
        for p in strict:
            # 若有多個 SP，就按到達率比例分布（也可改用平均）
            frac = (A.get(p, 0.0) / A_strict) if A_strict > 0 else (1.0 / len(strict))
            mu[p] = max(1e-6, use_sp * frac)
        cap_rem = max(0.0, cap_rem - use_sp)

    rest = [p for p in pools if p not in strict]
    if cap_rem > 0 and rest:
        shares = _share_wfq(W, rest, cap_rem)
        for p in rest:
            mu[p] = max(1e-6, shares.get(p, 0.0))

    return mu

# =========================
# 排隊延遲估算
# =========================

def estimate_queue_delay_ms_for_pool(
    pool: str,
    arrivals_mbps: float,
    service_mbps: float,
    *,
    avg_pkt_bytes: int = 1000,
    max_queue_delay_ms: float = 100.0,
) -> float:
    """
    簡化 M/M/1 近似：Wq ≈ (ρ/(1-ρ)) * t_pkt，t_pkt ≈ 8*B / μ (ms)，μ 以 Mbps。
    若 λ≥μ，回傳上限延遲（壅塞告警）。
    """
    lam = max(0.0, float(arrivals_mbps))
    mu  = max(1e-9, float(service_mbps))
    if lam >= mu:
        return max_queue_delay_ms
    rho = lam / mu
    t_pkt_ms = (8.0 * float(avg_pkt_bytes)) / (mu * 1000.0)  # 8*B bits / (μ*1e6 bps) * 1000
    return min(max_queue_delay_ms, (rho / max(1e-9, 1.0 - rho)) * t_pkt_ms)

def estimate_queue_delay_ms(
    G,
    edge_key: Tuple[str, str],
    cls: str,
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    qos: QoSConfig,
) -> float:
    """
    以目前 usage 聚合（各 pool 的 λ），算出各 pool 的 μ，再回傳指定 cls 所屬 pool 的 Wq（ms）。
    """
    u, v = edge_key
    d = G[u][v]
    C = float(d.get("bw_mbps", 10_000.0))

    # 各 pool 到達率（Mbps）
    arrivals = {p: float(usage.get(edge_key, {}).get(p, 0.0)) for p in ["EF","CS5","AF","CS3","CS0"]}

    # 各 pool 服務率（Mbps）
    mu = service_rates_mbps(C, qos, arrivals)

    p = pool_for(cls)
    return estimate_queue_delay_ms_for_pool(
        p, arrivals.get(p, 0.0), mu.get(p, 1e-6),
        avg_pkt_bytes=qos.avg_pkt_bytes,
        max_queue_delay_ms=qos.max_queue_delay_ms,
    )

def effective_link_delay_ms(
    G,
    edge_key: Tuple[str, str],
    cls: str,
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    qos: QoSConfig,
) -> float:
    """鏈路延遲（delay_ms）+ 排隊延遲（估算）。供 CSPF 使用。"""
    u, v = edge_key
    base = None
    # 支援兩種記法：edge["metrics"].delay_ms 或 edge["delay_ms"]
    d = G[u][v]
    if "metrics" in d and getattr(d["metrics"], "delay_ms", None) is not None:
        base = float(d["metrics"].delay_ms)
    else:
        base = float(d.get("delay_ms", 1.0))
    q_ms = estimate_queue_delay_ms(G, edge_key, cls, reserved, usage, qos)
    return base + q_ms

# =========================
# 匯出：edge_usage.csv（可視化用）
# =========================

def export_edge_usage_csv(
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    path: str = "outputs/flows/edge_usage.csv"
) -> None:
    rows: List[dict] = []
    for e, rmap in reserved.items():
        u, v = e
        umap = usage.get(e, {})
        keys = sorted(set(list(rmap.keys()) + list(umap.keys())))
        for k in keys:
            rv = float(rmap.get(k, 0.0))
            uv = float(umap.get(k, 0.0))
            rows.append({
                "edge": f"{u}-{v}",
                "pool": k,
                "reserve_mbps": f"{rv:.6f}",
                "usage_mbps": f"{uv:.6f}",
                "residual_mbps": f"{(rv-uv):.6f}",
            })

    # 輸出 CSV
    import os
    os.makedirs("outputs/flows", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["edge","pool","reserve_mbps","usage_mbps","residual_mbps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
