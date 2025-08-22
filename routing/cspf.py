# routing/cspf.py
from __future__ import annotations

import argparse
import csv
import math,time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

# —— 佇列/保留/排隊延遲 與 QoS 設定，全走 models/queues 的單一口徑 ——
from models.queues import (
    load_qos,            # 讀 qos.yaml，含 headroom/scheduler/weights 等
    classify,            # topic -> class（支援新版 class 與舊版 qos 欄位）
    build_reservations,  # 依 qos.yaml 建 reserved/usage，含 AF 共池 + headroom
    admit_edge, release_edge,
    pool_for,            # AF* -> 'AF'，其餘 class 原名
    effective_link_delay_ms,  # 鏈路延遲 + 排隊延遲（ms）
    export_edge_usage_csv,    # 匯出 outputs/flows/edge_usage.csv
)

# 可選：Yen KSP（不存在時退化為單一路徑）
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


# ------------------------- CSV I/O -------------------------

def read_edges_graph(edges_csv: Path) -> nx.Graph:
    G = nx.Graph()
    with edges_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = row["src_switch"].strip()
            v = row["dst_switch"].strip()
            G.add_edge(
                u, v,
                weight=float(row.get("weight", 1.0) or 1.0),
                delay_ms=float(row.get("delay_ms", 1.0) or 1.0),
                loss=float(row.get("loss", 0.0) or 0.0),
                bw_mbps=float(row.get("bw_mbps", 10_000.0) or 10_000.0),
            )
    return G

def write_edge_usage_csv(
    G: nx.Graph,
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    out_csv: Path,
    *,
    headroom: Optional[float] = None
) -> None:
    """
    輸出每邊每級的 reserve/usage/residual 到 CSV。
    欄位：
      u,v,edge,bw_mbps,class,reserve_mbps,usage_mbps,residual_mbps,utilization,
      cap_with_headroom,headroom_violated
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "u","v","edge","bw_mbps","class",
            "reserve_mbps","usage_mbps","residual_mbps","utilization",
            "cap_with_headroom","headroom_violated"
        ])

        for (u, v), rmap in reserved.items():
            # 取邊的 bw（圖是無向，兩向都試）
            if G.has_edge(u, v):
                bw = float(G[u][v].get("bw_mbps", ""))
            elif G.has_edge(v, u):
                bw = float(G[v][u].get("bw_mbps", ""))
            else:
                bw = ""

            umap = usage.get((u, v), {})
            classes = sorted(set(rmap.keys()) | set(umap.keys()))
            for cls in classes:
                r_val = float(rmap.get(cls, 0.0))
                u_val = float(umap.get(cls, 0.0))
                residual = r_val - u_val
                util = (u_val / r_val) if r_val > 0 else (0.0 if u_val == 0 else float("inf"))

                cap_head = ""
                violated = ""
                if headroom is not None:
                    cap_head = r_val * (1.0 - float(headroom))
                    violated = (u_val > cap_head + 1e-9)

                w.writerow([
                    u, v, f"{u}-{v}", bw, cls,
                    f"{r_val:.6f}", f"{u_val:.6f}", f"{residual:.6f}",
                    (f"{util:.6f}" if util != float("inf") else "inf"),
                    (f"{cap_head:.6f}" if cap_head != "" else ""),
                    (int(violated) if violated != "" else "")
                ])



def load_flows(flows_csv: Path) -> List[dict]:
    flows: List[dict] = []
    with flows_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or (row.get("id") or "").startswith("#"):
                continue
            flows.append(row)
    return flows


def ek(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


# ------------------------- Cost / SLA -------------------------

def edge_cost(
    G: nx.Graph,
    ekey: Tuple[str, str],
    cls: str,
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    qos_cfg,                 # models.queues.QoSConfig
    params: ClassParams,
    eps: float = 1e-3,
) -> float:
    """
    w = α·(delay_link + delay_queue) + β·loss + γ·lambda + δ·rho
        where lambda = max(0, u/r - 1)
              rho    = 1 / (eps + max(0, 1 - u/r))
    u/r 計算以「pool_for(cls)」為準（AF 共池）。
    """
    u, v = ekey
    d = G[u][v]
    # 1) 延遲（含排隊）
    delay_ms = effective_link_delay_ms(G, ekey, cls, reserved, usage, qos_cfg)
    # 2) 丟包
    loss = float(d.get("loss", 0.0))
    # 3) 擁塞項（u/r）
    pool = pool_for(cls)
    r = float(reserved.get(ekey, {}).get(pool, 0.0))
    uval = float(usage.get(ekey, {}).get(pool, 0.0))
    if r <= 0:
        lam = (1e6 if uval > 0 else 0.0)
        rho = 1e6
    else:
        ratio = uval / r
        lam = max(0.0, ratio - 1.0)
        rho = 1.0 / (eps + max(0.0, 1.0 - ratio))

    return (
        params.alpha * delay_ms +
        params.beta  * loss     +
        params.gamma * lam      +
        params.delta * rho
    )


def eval_path_sla(
    G: nx.Graph,
    path: List[str],
    cls: str,
    reserved, usage, qos_cfg,
    cls_params: ClassParams,
) -> Tuple[bool, float, float, str]:
    """
    以「鏈路延遲+排隊延遲」做 delay 加總；loss 直接相加。
    回傳 (ok, sum_delay_ms, sum_loss, reason)
    """
    if not path or len(path) < 2:
        return True, 0.0, 0.0, ""
    total_delay = 0.0
    total_loss = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        total_delay += effective_link_delay_ms(G, ek(u, v), cls, reserved, usage, qos_cfg)
        total_loss  += float(G[u][v].get("loss", 0.0))
    reasons: List[str] = []
    if cls_params.maxDelay_ms is not None and total_delay > cls_params.maxDelay_ms:
        reasons.append(f"delay>{cls_params.maxDelay_ms:.3f}({total_delay:.3f})")
    if cls_params.maxLoss is not None and total_loss > cls_params.maxLoss:
        reasons.append(f"loss>{cls_params.maxLoss:.6f}({total_loss:.6f})")
    ok = (len(reasons) == 0)
    return ok, total_delay, total_loss, ";".join(reasons)


# ------------------------- CSPF（一次嘗試：固定 class） -------------------------

def cspf_pick_once(
    G: nx.Graph,
    src: str,
    dst: str,
    cls: str,
    rate_mbps: float,
    reserved: Dict[Tuple[str, str], Dict[str, float]],
    usage:    Dict[Tuple[str, str], Dict[str, float]],
    qos_cfg,                 # models.queues.QoSConfig
    cls_params: ClassParams,
    *,
    headroom: float = 0.10,
    theta: float = 0.05,
    tau: float = 1.0,
    K: int = 8,
) -> Tuple[List[str], float, List[List[str]], List[float], List[str]]:
    """
    在「可行子圖」上跑 Dijkstra；取 KSP 候選，過濾 Δcost ≤ theta 與 SLA，
    回傳：best_path, best_cost, wecmp_paths, wecmp_weights, filtered_sla_info
    """
    pool = pool_for(cls)

    # 1) 可行子圖（Admission headroom）
    H = nx.Graph()
    for u, v, _ in G.edges(data=True):
        e = ek(u, v)
        cap = float(reserved.get(e, {}).get(pool, 0.0))
        use = float(usage.get(e, {}).get(pool, 0.0))
        if use + rate_mbps > cap * (1.0 - headroom) + 1e-9:
            continue
        # 邊成本：用 queues 模型算出的延遲 + 擁塞項
        cost = edge_cost(G, e, cls, reserved, usage, qos_cfg, cls_params)
        H.add_edge(u, v, cspf_cost=cost)

    if src not in H or dst not in H:
        return [], math.inf, [], [], []

    # 2) 最短一路
    try:
        best = nx.shortest_path(H, src, dst, weight="cspf_cost")
    except nx.NetworkXNoPath:
        return [], math.inf, [], [], []

    best_cost = sum(H[best[i]][best[i+1]]["cspf_cost"] for i in range(len(best)-1))
    ok_best, _, _, reason_best = eval_path_sla(G, best, cls, reserved, usage, qos_cfg, cls_params)
    if not ok_best:
        return [], math.inf, [], [], [f"{'->'.join(best)}|viol={reason_best}"]

    # 3) KSP 候選
    all_paths: List[List[str]] = [best]
    if yen_ksp is not None and K > 1:
        for p in yen_ksp(H, src, dst, K=K, weight="cspf_cost") or []:
            if p and p not in all_paths:
                all_paths.append(p)

    # 4) Δcost 與 SLA 過濾
    keep_paths: List[List[str]] = []
    costs: List[float] = []
    filtered_sla: List[str] = []
    for p in all_paths:
        c = sum(H[p[i]][p[i+1]]["cspf_cost"] for i in range(len(p)-1))
        ok_sla, dsum, lsum, reason = eval_path_sla(G, p, cls, reserved, usage, qos_cfg, cls_params)
        if c <= best_cost * (1.0 + theta) + 1e-12 and ok_sla:
            keep_paths.append(p); costs.append(c)
        elif not ok_sla:
            filtered_sla.append(f"{'->'.join(p)}|delay={dsum:.3f}|loss={lsum:.6f}|viol={reason}")

    if not keep_paths:
        return [], math.inf, [], [], filtered_sla

    # 5) WECMP 權重（softmax on cost）
    exps = [math.exp(-tau * (c - min(costs))) for c in costs]
    s = sum(exps) or 1.0
    weights = [x/s for x in exps]
    return keep_paths[0], costs[0], keep_paths, weights, filtered_sla


# ------------------------- CLI -------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CSPF path selection with Admission/WECMP and queue-aware delay")
    ap.add_argument("--edges", type=str, required=True, help="outputs/matrix/edges_all.csv")
    ap.add_argument("--qos",   type=str, required=True, help="configs/qos.yaml")
    ap.add_argument("--flows", type=str, required=True, help="configs/flows.csv")
    ap.add_argument("--out",   type=str, default="outputs/flows/per_flow.csv")
    # 參數
    ap.add_argument("--headroom", type=float, default=0.10)
    ap.add_argument("--theta",    type=float, default=0.05, help="WECMP 等價門檻 Δcost")
    ap.add_argument("--tau",      type=float, default=1.0,  help="WECMP softmax 溫度")
    ap.add_argument("--K",        type=int,   default=8,    help="KSP 候選條數")
    ap.add_argument("--watch", type=float, default=0.0,help="每 N 秒重算一次；0 表示只跑一次")
    return ap


# ------------------------- main -------------------------

def _read_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main() -> None:
    args = build_cli().parse_args()
    edges_csv = Path(args.edges)
    flows_csv = Path(args.flows)
    qos_yaml  = Path(args.qos)
    out_csv   = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def run_once():
        # 1) 載入圖與 QoS（統一用 models/queues 的 load_qos）
        G = read_edges_graph(edges_csv)
        from models.queues import load_qos, build_reservations, classify, admit_edge, release_edge, export_edge_usage_csv
        qos_cfg = load_qos(str(qos_yaml))  # -> QoSConfig
        headroom = float(args.headroom)

        # 2) 由 QoS 配置建立 reserved/usage（AF 類別會匯總到 "AF" 池）
        reserved, usage = build_reservations(G, qos_cfg)

        # 3) 讀 flows
        flows = load_flows(flows_csv)

        # 4) 讀 classes 參數（αβγδ 與 SLA）——從 YAML 的 classes 區塊
        yall = _read_yaml(qos_yaml)
        classes_blk = (yall.get("classes") or {})
        class_params: Dict[str, ClassParams] = {}
        for cls, cfg in classes_blk.items():
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

        # 5) 降級鏈（yaml 可覆寫）
        default_chain = {
            "EF":   ["AF41", "AF21", "CS0"],
            "CS5":  ["AF41", "AF21", "CS0"],
            "AF41": ["AF21", "CS0"],
            "AF21": ["CS0"],
            "CS0":  [],
        }
        downgrade_chain = yall.get("downgrade_chain", default_chain)

        # 6) 逐流處理
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "flow_id","src","dst","topic",
                "class_req","class_admitted","rate_mbps","admitted","reason",
                "path","path_cost","sum_delay_ms","sum_loss",
                "sla_maxDelay_ms","sla_maxLoss",
                "wecmp_candidates","wecmp_weights","wecmp_filtered_sla"
            ])

            for row in flows:
                fid   = row.get("id", "").strip() or f"flow{len(row)}"
                src   = (row.get("src") or "").strip()
                dst   = (row.get("dst") or "").strip()
                topic = (row.get("topic") or "").strip()
                rate  = float(row.get("rate_mbps", 0.0) or 0.0)
                subs  = (row.get("subs") or "").strip()

                # 多播交給 steiner.py
                if subs:
                    w.writerow([fid, src, dst, topic, "", "", rate, 0, "multicast_not_supported_in_cspf.py",
                                "", "", "", "", "", "", "", ""])
                    continue

                # 指定 class > topic_map 推導
                cls_req = (row.get("class") or row.get("qos") or "").strip()
                if not cls_req:
                    cls_req = classify(topic, qos_cfg) or ""

                if not cls_req:
                    w.writerow([fid, src, dst, topic, "", "", rate, 0, "no_class_for_topic",
                                "", "", "", "", "", "", "", ""])
                    continue

                # 沒提供參數就用預設
                params_req = class_params.get(cls_req, ClassParams())
                sla_dmax = params_req.maxDelay_ms if params_req.maxDelay_ms is not None else ""
                sla_lmax = params_req.maxLoss if params_req.maxLoss is not None else ""

                # 嘗試原 class 及降級鏈
                tried = [cls_req] + list(downgrade_chain.get(cls_req, []))
                admitted_flag = 0
                reason = "blocked"
                sel_path: List[str] = []
                sel_cost = math.inf
                wecmp_paths: List[List[str]] = []
                wecmp_weights: List[float] = []
                filtered_sla_info: List[str] = []
                cls_ok = ""
                sum_delay = 0.0
                sum_loss  = 0.0

                for cls_try in tried:
                    params_try = class_params.get(cls_try, ClassParams())

                    # ⚠️ 正確的呼叫（不要帶 qos_cfg）
                    best, cost, cands, weights, filtered_sla = cspf_pick_once(
                        G, src, dst, cls_try, rate, reserved, usage, qos_cfg, params_try,
                        headroom=headroom, theta=float(args.theta), tau=float(args.tau), K=int(args.K),
                    )

                    if filtered_sla:
                        filtered_sla_info.extend(f"[{cls_try}] " + s for s in filtered_sla)

                    if not best:
                        reason = "no_feasible_path_or_sla"
                        continue

                    # Admission：逐邊鎖定（失敗則回退）
                    taken: List[Tuple[Tuple[str,str], float]] = []
                    ok = True
                    for i in range(len(best)-1):
                        e = ek(best[i], best[i+1])
                        if not admit_edge(e, cls_try, rate, reserved, usage):
                            ok = False
                            reason = "admission_conflict"
                            break
                        taken.append((e, rate))
                    if not ok:
                        for e, r_need in taken:
                            release_edge(e, cls_try, r_need, reserved, usage)
                        continue

                    # 成功：計算 SLA（用現有 eval_path_sla 簽名）
                    ok_sla, sum_delay, sum_loss, _ = eval_path_sla(
                        G, best, cls_try, reserved, usage, qos_cfg, params_try
                    )
                    if not ok_sla:
                        for e, r_need in taken:
                            release_edge(e, cls_try, r_need, reserved, usage)
                        reason = "sla_violation_after_admit"
                        continue

                    sel_path, sel_cost = best, cost
                    wecmp_paths, wecmp_weights = cands, weights
                    admitted_flag = 1
                    reason = "ok"
                    cls_ok = cls_try
                    break

                # 輸出一筆 flow
                if admitted_flag:
                    w.writerow([
                        fid, src, dst, topic,
                        cls_req, cls_ok, rate, 1, reason,
                        "->".join(sel_path), f"{sel_cost:.6f}", f"{sum_delay:.3f}", f"{sum_loss:.6f}",
                        sla_dmax, sla_lmax,
                        "|".join("->".join(p) for p in wecmp_paths),
                        "|".join(f"{x:.4f}" for x in wecmp_weights),
                        " || ".join(filtered_sla_info)
                    ])
                else:
                    w.writerow([
                        fid, src, dst, topic,
                        cls_req, "", rate, 0, reason,
                        "", "", "", "",
                        sla_dmax, sla_lmax,
                        "", "", " || ".join(filtered_sla_info)
                    ])

        print(f"[OK] per-flow written -> {out_csv}")

        # 匯出邊用量（名稱以你 models/queues 為準；若沒有 export_edge_usage_csv，請改回 write_edge_usage_csv）
        try:
            export_edge_usage_csv(reserved, usage, path=str(out_csv.parent / "edge_usage.csv"))
        except Exception:
            # 後備：用 cspf.py 內建的 helper（若你有）
            write_edge_usage_csv(G, reserved, usage, out_csv.parent / "edge_usage.csv", headroom=float(args.headroom))
        print(f"[OK] edge-usage written -> {out_csv.parent / 'edge_usage.csv'}")

    # 單次 or 監看模式
    if getattr(args, "watch", 0):
        while True:
            run_once()
            time.sleep(float(args.watch))
    else:
        run_once()



if __name__ == "__main__":
    main()
