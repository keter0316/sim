# routing/cspf.py
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import yaml


# 可選：使用既有的 Yen KSP；不存在時退化為只找最短一路
try:
    from routing.ksp_additional import yen_ksp
except Exception:
    yen_ksp = None  # type: ignore


# ------------------------- Dataclasses -------------------------

@dataclass
class ClassParams:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.8
    delta: float = 0.8
    maxDelay_ms: Optional[float] = None
    maxLoss: Optional[float] = None


# ------------------------- Topic → Class mapping -------------------------

def _match_topic(pattern: str, topic: str) -> bool:
    """
    超輕量 MQTT 風格比對：'foo/#' 視為 startswith('foo/')
    其餘用等值比對。
    """
    pat = pattern.strip()
    if pat.endswith("/#"):
        return topic.startswith(pat[:-2])
    return topic == pat


def topic_to_class(topic: str, qos_cfg: dict) -> Optional[str]:
    """
    從 qos.yaml 的 topic_map 找 class；支援 key 'class' 或 'qos'。
    """
    tmap = qos_cfg.get("topic_map", {}) or {}
    for pat, val in tmap.items():
        if _match_topic(pat, topic):
            if isinstance(val, dict):
                return val.get("class") or val.get("qos")
            if isinstance(val, str):
                return val
    return None


# ------------------------- Reservations -------------------------

def compute_reserve_per_edge(bw_mbps: float, reservations_cfg: dict) -> Dict[str, float]:
    """
    將 qos.yaml 的 reservations 轉成「每 class 的保留 Mbps」。
    支援：
      - min_mbps: 絕對保留
      - share: 按鏈路帶寬比例（0..1）
    若同時存在，min_mbps 優先。
    """
    out: Dict[str, float] = {}
    for cls, cfg in (reservations_cfg or {}).items():
        if not isinstance(cfg, dict):
            continue
        if "min_mbps" in cfg and cfg["min_mbps"] is not None:
            out[cls] = float(cfg["min_mbps"])
        elif "share" in cfg and cfg["share"] is not None:
            out[cls] = float(cfg["share"]) * float(bw_mbps)
        else:
            out[cls] = 0.0
    return out


# ------------------------- Cost / Admission -------------------------

def edge_cost(
    e_data: dict,
    cls: str,
    usage_mbps: float,
    reserve_mbps: float,
    params: ClassParams,
    eps: float = 1e-3,
) -> float:
    """
    w = α·delay + β·loss + γ·lambda + δ·rho
    """
    delay = float(e_data.get("delay_ms", 1.0))
    loss = float(e_data.get("loss", 0.0))

    if reserve_mbps <= 0:
        # 沒保留時：若 usage>0 仍給一個大懲罰；否則只看 delay/loss
        lam = 1e6 if usage_mbps > 0 else 0.0
        rho = 1e6
    else:
        u_ratio = usage_mbps / reserve_mbps
        lam = max(0.0, u_ratio - 1.0)
        rho = 1.0 / (eps + max(0.0, 1.0 - u_ratio))

    return (
        params.alpha * delay +
        params.beta  * loss  +
        params.gamma * lam   +
        params.delta * rho
    )


def path_sla_ok(
    G: nx.Graph,
    path: List[str],
    cls_params: ClassParams,
) -> bool:
    if not path or len(path) < 2:
        return True
    if cls_params.maxDelay_ms is None and cls_params.maxLoss is None:
        return True
    total_delay = 0.0
    total_loss = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        d = G[u][v]
        total_delay += float(d.get("delay_ms", 1.0))
        total_loss  += float(d.get("loss", 0.0))
    if cls_params.maxDelay_ms is not None and total_delay > cls_params.maxDelay_ms:
        return False
    if cls_params.maxLoss is not None and total_loss > cls_params.maxLoss:
        return False
    return True


# ------------------------- Main CSPF selection per flow -------------------------

def cspf_pick_path(
    G_base: nx.Graph,
    cls: str,
    rate_mbps: float,
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    cls_params: ClassParams,
    *,
    headroom: float = 0.10,
    theta: float = 0.05,  # Δcost ≤ 5% 視為等價
    tau: float = 1.0,     # WECMP softmax 溫度
    K: int = 8,           # 取前 K 最短候選
) -> Tuple[List[str], float, List[List[str]], List[float]]:
    """
    在「可行邊」上計算對應 class 的成本，回傳：
      (best_path, best_cost, wecmp_paths, wecmp_weights)
    wecmp_paths 只保留 Δcost ≤ theta 的候選；權重為 softmax(exp(-tau*cost))
    """
    # 1) 構造可行子圖（Admission 硬限制）
    H = nx.Graph()
    for u, v, d in G_base.edges(data=True):
        ek = (u, v) if u <= v else (v, u)
        cap = float(reserved.get(ek, {}).get(cls, 0.0))
        use = float(usage.get(ek, {}).get(cls, 0.0))
        # Admission：usage + rate ≤ cap*(1-headroom)
        feasible = (cap * (1.0 - headroom) - use) >= rate_mbps - 1e-9
        if not feasible:
            continue
        # 2) 邊成本
        cost = edge_cost(d, cls, use, cap, cls_params)
        H.add_edge(u, v, cspf_cost=cost, delay_ms=d.get("delay_ms", 1.0), loss=d.get("loss", 0.0))

    # 3) Dijkstra（最短成本路）
    try:
        # 先用未指定 src/dst 的情境；呼叫者會指定
        pass
    except Exception:
        pass  # placeholder，實際選路在 pick_for_pair 內進行

    # 由於 src/dst 會因 flow 而異，實際計算放在內部 helper
    def pick_for_pair(src: str, dst: str) -> Tuple[List[str], float, List[List[str]], List[float]]:
        if src not in H or dst not in H:
            return [], math.inf, [], []

        try:
            P_best = nx.shortest_path(H, src, dst, weight="cspf_cost")
        except nx.NetworkXNoPath:
            return [], math.inf, [], []

        best_cost = sum(H[P_best[i]][P_best[i+1]]["cspf_cost"] for i in range(len(P_best)-1))
        if not path_sla_ok(H, P_best, cls_params):
            return [], math.inf, [], []

        # 4) KSP 候選與 WECMP
        paths = [P_best]
        if yen_ksp is not None and K > 1:
            ksp = yen_ksp(H, src, dst, K=K, weight="cspf_cost")
            # 去掉第一條（已包含），再合併
            for p in ksp:
                if p != P_best:
                    paths.append(p)

        # 只保留 Δcost ≤ theta
        costs = []
        keep_paths = []
        for p in paths:
            c = sum(H[p[i]][p[i+1]]["cspf_cost"] for i in range(len(p)-1))
            if c <= best_cost * (1.0 + theta) + 1e-12 and path_sla_ok(H, p, cls_params):
                keep_paths.append(p)
                costs.append(c)

        # softmax(exp(-τ·cost))
        if not keep_paths:
            return P_best, best_cost, [], []
        exps = [math.exp(-tau * (c - best_cost)) for c in costs]  # 以 best 作基準避免 underflow
        s = sum(exps)
        weights = [x / s for x in exps] if s > 0 else [1.0/len(exps)] * len(exps)
        return keep_paths[0], best_cost, keep_paths, weights

    return pick_for_pair  # type: ignore


# ------------------------- CSV I/O helpers -------------------------

def read_edges_graph(edges_csv: Path) -> nx.Graph:
    G = nx.Graph()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = row["src_switch"].strip()
            v = row["dst_switch"].strip()
            w = float(row.get("weight", 1.0))
            G.add_edge(
                u, v,
                weight=w,
                delay_ms=float(row.get("delay_ms", 1.0) or 1.0),
                loss=float(row.get("loss", 0.0) or 0.0),
                bw_mbps=float(row.get("bw_mbps", 1000.0) or 1000.0),
            )
    return G


def load_qos(qos_yaml: Path) -> dict:
    with qos_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_flows(flows_csv: Path) -> List[dict]:
    flows = []
    with flows_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or (row.get("id") or "").startswith("#"):
                continue
            flows.append(row)
    return flows


# ------------------------- Admission bookkeeping -------------------------

def ek(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def ensure_edge_class(d: Dict[Tuple[str, str], Dict[str, float]], e: Tuple[str, str], cls: str) -> None:
    if e not in d:
        d[e] = {}
    d[e].setdefault(cls, 0.0)


# ------------------------- CLI main -------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CSPF path selection with Admission and WECMP")
    ap.add_argument("--edges", type=str, required=True, help="outputs/matrix/edges_all.csv")
    ap.add_argument("--qos",   type=str, required=True, help="configs/qos.yaml")
    ap.add_argument("--flows", type=str, required=True, help="configs/flows.csv")
    ap.add_argument("--out",   type=str, default="outputs/flows/per_flow.csv")
    # 參數
    ap.add_argument("--headroom", type=float, default=0.10)
    ap.add_argument("--theta",    type=float, default=0.05, help="WECMP 等價門檻 Δcost")
    ap.add_argument("--tau",      type=float, default=1.0,  help="WECMP softmax 溫度")
    ap.add_argument("--K",        type=int,   default=8,    help="KSP 候選條數")
    return ap


def main() -> None:
    args = build_cli().parse_args()
    edges_csv = Path(args.edges)
    qos_yaml  = Path(args.qos)
    flows_csv = Path(args.flows)
    out_csv   = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) 載入
    G = read_edges_graph(edges_csv)
    qos_cfg = load_qos(qos_yaml)
    flows   = load_flows(flows_csv)

    # 2) 準備 class 參數與降級鏈
    classes_cfg = qos_cfg.get("classes", {}) or {}
    class_params: Dict[str, ClassParams] = {}
    for cls, cfg in classes_cfg.items():
        if not isinstance(cfg, dict):
            continue
        sla = cfg.get("sla") or {}
        class_params[cls] = ClassParams(
            alpha=float(cfg.get("alpha", 0.5)),
            beta =float(cfg.get("beta",  0.3)),
            gamma=float(cfg.get("gamma", 0.8)),
            delta=float(cfg.get("delta", 0.8)),
            maxDelay_ms=sla.get("maxDelay_ms"),
            maxLoss=sla.get("maxLoss"),
        )
    # 預設降級鏈（可在 yaml 以 downgrade_chain 覆寫）
    default_chain = {
        "EF":   ["AF41", "AF21", "CS0"],
        "CS5":  ["AF41", "AF21", "CS0"],
        "AF41": ["AF21", "CS0"],
        "AF21": ["CS0"],
        "CS0":  [],
    }
    downgrade_chain = qos_cfg.get("downgrade_chain", default_chain)

    # 3) 為每條邊建立 per-class 保留/用量
    reserved: Dict[Tuple[str, str], Dict[str, float]] = {}
    usage:    Dict[Tuple[str, str], Dict[str, float]] = {}
    res_cfg = qos_cfg.get("reservations", {}) or {}
    for u, v, d in G.edges(data=True):
        bw = float(d.get("bw_mbps", 1000.0))
        rmap = compute_reserve_per_edge(bw, res_cfg)
        reserved[ek(u, v)] = dict(rmap)
        usage[ek(u, v)] = {cls: 0.0 for cls in rmap.keys()}

    # 4) 逐流處理
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "flow_id","src","dst","topic",
            "class_req","class_admitted","rate_mbps","admitted","reason",
            "path","path_cost","sum_delay_ms","sum_loss",
            "wecmp_candidates","wecmp_weights"
        ])

        for row in flows:
            fid   = row.get("id", "").strip() or f"flow{len(row)}"
            src   = (row.get("src") or "").strip()
            dst   = (row.get("dst") or "").strip()
            topic = (row.get("topic") or "").strip()
            rate  = float(row.get("rate_mbps", 0.0) or 0.0)

            # 忽略多播（有 subs 就註記跳過）
            subs = (row.get("subs") or "").strip()
            if subs:
                w.writerow([fid, src, dst, topic, "", "", rate, 0, "multicast_not_supported_in_cspf.py",
                            "", "", "", "", ""])
                continue

            # 解析 class：flows.csv 的 'class' 優先生效；否則從 topic_map 推出
            cls_req = (row.get("class") or "").strip()
            if not cls_req:
                cls_req = topic_to_class(topic, qos_cfg) or ""

            if not cls_req:
                w.writerow([fid, src, dst, topic, "", "", rate, 0, "no_class_for_topic",
                            "", "", "", "", ""])
                continue
            if cls_req not in class_params:
                # 用預設參數也能跑；但標記一下
                class_params.setdefault(cls_req, ClassParams())
            params = class_params[cls_req]

            # 準備挑路器（回傳一個針對 (src,dst) 的挑選函式）
            pick_for_pair = cspf_pick_path(
                G, cls_req, rate, reserved, usage, params,
                headroom=float(args.headroom),
                theta=float(args.theta),
                tau=float(args.tau),
                K=int(args.K),
            )

            # 主邏輯：嘗試原 class，失敗就依降級鏈嘗試
            tried_classes = [cls_req] + list(downgrade_chain.get(cls_req, []))
            admitted = 0
            reason = "blocked"
            sel_path: List[str] = []
            sel_cost = math.inf
            wecmp_paths: List[List[str]] = []
            wecmp_weights: List[float] = []
            cls_ok = ""

            for cls_try in tried_classes:
                params_try = class_params.get(cls_try, ClassParams())
                best, cost, cands, weights = pick_for_pair(src, dst)
                if not best:
                    reason = "no_feasible_path"
                    continue
                # Admission 已在子圖過濾，這裡正式鎖定用量
                ok_all = True
                sum_delay = 0.0
                sum_loss  = 0.0
                for i in range(len(best)-1):
                    u, v = best[i], best[i+1]
                    d = G[u][v]
                    sum_delay += float(d.get("delay_ms", 1.0))
                    sum_loss  += float(d.get("loss", 0.0))
                    ekey = ek(u, v)
                    # 再次保險檢查
                    cap = float(reserved.get(ekey, {}).get(cls_try, 0.0))
                    use = float(usage.get(ekey, {}).get(cls_try, 0.0))
                    if use + rate > cap * (1.0 - float(args.headroom)) + 1e-9:
                        ok_all = False
                        reason = "admission_conflict"
                        break
                if not ok_all:
                    continue

                # 鎖定用量
                for i in range(len(best)-1):
                    u, v = best[i], best[i+1]
                    ekey = ek(u, v)
                    usage.setdefault(ekey, {}).setdefault(cls_try, 0.0)
                    usage[ekey][cls_try] += rate

                # 成功
                sel_path, sel_cost = best, cost
                wecmp_paths, wecmp_weights = cands, weights
                admitted = 1
                reason = "ok"
                cls_ok = cls_try
                # 寫出並跳出降級嘗試
                w.writerow([
                    fid, src, dst, topic,
                    cls_req, cls_ok, rate, admitted, reason,
                    "->".join(sel_path), f"{sel_cost:.6f}", f"{sum_delay:.3f}", f"{sum_loss:.6f}",
                    "|".join("->".join(p) for p in wecmp_paths),
                    "|".join(f"{x:.4f}" for x in wecmp_weights)
                ])
                break

            if not admitted:
                w.writerow([
                    fid, src, dst, topic,
                    cls_req, "", rate, 0, reason,
                    "", "", "", "", ""
                ])

    print(f"[OK] per-flow written -> {out_csv}")


if __name__ == "__main__":
    main()
