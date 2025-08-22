# examples/live_view.py
from __future__ import annotations
import  time, math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ---------- 路徑設定 ----------
EDGES_CSV  = Path("outputs/matrix/edges_all.csv")
FLOWS_CSV  = Path("outputs/flows/per_flow.csv")        # routing/cspf.py 產出
USAGE_CSV  = Path("outputs/flows/edge_usage.csv")      # models/queues.export_edge_usage_csv
TREES_CSV  = Path("outputs/flows/trees.csv")           # routing/steiner.py（若有）

# ---------- 顏色/樣式 ----------
POOL_COLOR = {
    "EF":  "#e74c3c",  # 紅
    "CS5": "#c57936",  # 橙
    "AF":  "#3498db",  # 藍 (AF41/31/21/11 聚合池)
    "CS3": "#9b59b6",  # 紫
    "CS0": "#7f8c8d",  # 灰
}
NODE_COLOR = {
    "host":  "#2ecc71",
    "access":"#34495e",
    "core":  "#1abc9c",
    "other": "#95a5a6",
}

# ---------- 小工具 ----------
def ek(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def load_edges() -> pd.DataFrame:
    df = safe_read_csv(EDGES_CSV)
    if df is None:
        st.warning(f"找不到 {EDGES_CSV}")
        return pd.DataFrame(columns=["src_switch","dst_switch","type","bw_mbps","delay_ms","loss","weight"])
    # 填預設
    for c, v in [("type",""),("bw_mbps",10000.0),("delay_ms",1.0),("loss",0.0),("weight",1.0)]:
        if c not in df.columns: df[c] = v
    return df

def load_per_flow() -> pd.DataFrame:
    df = safe_read_csv(FLOWS_CSV)
    if df is None:
        return pd.DataFrame(columns=[
            "flow_id","src","dst","topic","class_req","class_admitted","rate_mbps","admitted",
            "path","path_cost","sum_delay_ms","sum_loss",
            "sla_maxDelay_ms","sla_maxLoss","wecmp_candidates","wecmp_weights","wecmp_filtered_sla"
        ])
    # 清理 admitted/rate
    if "admitted" in df.columns:
        df["admitted"] = pd.to_numeric(df["admitted"], errors="coerce").fillna(0).astype(int)
    if "rate_mbps" in df.columns:
        df["rate_mbps"] = pd.to_numeric(df["rate_mbps"], errors="coerce").fillna(0.0)
    return df

def load_edge_usage() -> Optional[pd.DataFrame]:
    df = safe_read_csv(USAGE_CSV)
    if df is None:
        return None
    # reserve/usage/residual 轉為數值
    for c in ("reserve_mbps","usage_mbps","residual_mbps"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def load_trees() -> Optional[pd.DataFrame]:
    df = safe_read_csv(TREES_CSV)
    return df

def parse_path_str(path_str: str) -> List[str]:
    if not isinstance(path_str, str) or not path_str.strip():
        return []
    return [x.strip() for x in path_str.split("->") if x.strip()]

def parse_edges_str(edges_str: str) -> List[Tuple[str,str]]:
    """將 'u-v|x-y|...' 轉成 [(u,v), ...]"""
    if not isinstance(edges_str, str) or not edges_str.strip():
        return []
    out = []
    for seg in edges_str.split("|"):
        seg = seg.strip()
        if not seg: continue
        if "-" in seg:
            a,b = seg.split("-",1)
            out.append(ek(a.strip(), b.strip()))
    return out

def detect_roles(df_edges: pd.DataFrame) -> Dict[str, str]:
    """
    依邊 type 推測 node role: host/access/core。
    """
    role: Dict[str,str] = {}
    for _, r in df_edges.iterrows():
        u = str(r["src_switch"]); v = str(r["dst_switch"])
        typ = str(r.get("type",""))
        for n in (u, v):
            role.setdefault(n, "other")
        if typ == "host-access":
            # 找出 host 與 access
            # 假設 h* 與 s* 的命名
            if u.startswith("h"): role[u] = "host"
            if v.startswith("h"): role[v] = "host"
            if u.startswith("s"): role[u] = "access" if role.get(u) != "core" else role[u]
            if v.startswith("s"): role[v] = "access" if role.get(v) != "core" else role[v]
        elif typ == "access-core":
            # s*-s* 邊，但其中一個是 core
            if u.startswith("s"): role[u] = "access" if role.get(u) != "core" else role[u]
            if v.startswith("s"): role[v] = "access" if role.get(v) != "core" else role[v]
        elif typ == "core-core":
            if u.startswith("s"): role[u] = "core"
            if v.startswith("s"): role[v] = "core"
    return role

def fixed_positions(nodes: List[str], roles: Dict[str,str]) -> Dict[str, Tuple[float,float]]:
    """
    位置固定：access 圓環、host 外環、core 內環（若資料不足就 fallback spring_layout）
    """
    access = [n for n in nodes if roles.get(n) == "access"]
    hosts  = [n for n in nodes if roles.get(n) == "host"]
    cores  = [n for n in nodes if roles.get(n) == "core"]
    pos: Dict[str, Tuple[float,float]] = {}

    if access:
        pos_access = nx.circular_layout(access, scale=300)
        for n, p in pos_access.items():
            pos[n] = (float(p[0]), float(p[1]))
    for a in access:
        # 推算對應 host：h{same index}
        if a[0] == "s" and a[1:].isdigit():
            h = "h" + a[1:]
            if h in hosts and a in pos:
                x, y = pos[a]
                pos[h] = (x*1.25, y*1.25)
    if cores:
        pos_core = nx.circular_layout(cores, scale=180)
        for n, p in pos_core.items():
            pos[n] = (float(p[0]), float(p[1]))
    # 其它節點（fallback）
    others = [n for n in nodes if n not in pos]
    if others:
        fall = nx.spring_layout(nx.Graph([(n, n) for n in others]), seed=42, scale=280)
        for n, p in fall.items():
            pos[n] = (float(p[0]), float(p[1]))
    return pos

def pool_of_class(cls: str) -> str:
    cls = (cls or "").strip().upper()
    if cls.startswith("AF"): return "AF"
    if cls in ("EF","CS5","CS3","CS0"): return cls
    return "CS0"

def aggregate_usage_from_flows(df_flows: pd.DataFrame) -> Dict[Tuple[str,str], Dict[str,float]]:
    """
    若沒有 edge_usage.csv，從 per_flow.csv 推估各邊各池的 usage（僅採已 admitted 的流）。
    """
    usage: Dict[Tuple[str,str], Dict[str,float]] = {}
    if df_flows is None or df_flows.empty: return usage
    for _, r in df_flows.iterrows():
        if int(r.get("admitted",0)) != 1: continue
        cls = str(r.get("class_admitted") or r.get("class_req") or "CS0")
        pool = pool_of_class(cls)
        rate = float(r.get("rate_mbps",0.0) or 0.0)
        path = parse_path_str(r.get("path",""))
        for i in range(len(path)-1):
            e = ek(path[i], path[i+1])
            usage.setdefault(e, {})
            usage[e][pool] = usage[e].get(pool, 0.0) + rate
    return usage

def edge_key_str(u: str, v: str) -> str:
    a, b = ek(u, v); return f"{a}-{b}"

def color_mix(base: str, t: float) -> str:
    """
    讓顏色隨負載加深：t ∈ [0,1]，0=淡，1=基色
    """
    base = base.lstrip("#")
    r = int(base[0:2], 16); g = int(base[2:4], 16); b = int(base[4:6], 16)
    # 與白色混合
    r2 = int((1-t)*255 + t*r); g2 = int((1-t)*255 + t*g); b2 = int((1-t)*255 + t*b)
    return f"#{r2:02x}{g2:02x}{b2:02x}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI-Switch Live View", layout="wide")
st.title("AI-Switch 同步可視化（ECMP / CSPF / Steiner）")# 自動刷新（每 1 秒）




# Sidebar controls
with st.sidebar:
    st.header("篩選 / 顯示")
    autorefresh = st.checkbox("每秒自動刷新", value=True)
    # 手動刷新按鈕
    if st.button("手動刷新", disabled=autorefresh, use_container_width=True):
        st.rerun()
    selected_pools = st.multiselect("顯示的類別池 (Pool)", ["EF","CS5","AF","CS3","CS0"], default=["EF","CS5","AF","CS3","CS0"])
    topic_filter = st.text_input("Topic 關鍵字（含子字串）", value="")
    show_multicast = st.checkbox("疊加多播樹 (trees.csv)", value=True)
    show_wecmp = st.checkbox("顯示 WECMP 候選與權重（選單下方）", value=False)
    st.divider()
    st.subheader("視覺強化")
    edge_gain = st.slider("邊寬放大倍率", 1.0, 50.0, 12.0, 0.5)
    color_boost = st.slider("顏色強度", 0.0, 1.0, 0.7, 0.05)
    ratio_mode = st.radio("負載比基準", ["usage / reserve", "usage / bw"], index=0)
    
    st.caption("EF/CS5 會以較醒目顏色呈現；邊寬依 usage/reserve 比例或 usage/bw 比例。")

    st.divider()
    st.subheader("Edge 檢視")
    # Edge 選擇器稍後填充（等載入完資料）

# 讀資料
df_edges = load_edges()
df_flows = load_per_flow()
df_usage = load_edge_usage()
df_trees = load_trees()

# 角色與圖
nodes = sorted(set(df_edges["src_switch"].astype(str)).union(set(df_edges["dst_switch"].astype(str))))
roles = detect_roles(df_edges)
pos = fixed_positions(nodes, roles)

# usage/reserve 來源：
reserve_map: Dict[Tuple[str,str], Dict[str,float]] = {}
usage_map:   Dict[Tuple[str,str], Dict[str,float]] = {}

# ...在 df_usage 解析的這段改一下...
if df_usage is not None and not df_usage.empty:
    for _, r in df_usage.iterrows():
        u_v = str(r.get("edge",""))
        if "-" not in u_v:
            continue
        u, v = u_v.split("-",1)
        e = ek(u.strip(), v.strip())
        # 兼容兩種欄位名稱：優先用 class，沒有再用 pool
        pool = str(r.get("class", r.get("pool", "CS0"))).upper()

        rv = float(r.get("reserve_mbps", 0.0))
        uv = float(r.get("usage_mbps",   0.0))
        reserve_map.setdefault(e, {})[pool] = rv
        usage_map.setdefault(e, {})[pool]   = uv
else:
    # 沒有 edge_usage.csv，就用 flows 推估 usage，reserve 用 bw 的一部分（僅用於視覺化）
    usage_map = aggregate_usage_from_flows(df_flows)
    for _, r in df_edges.iterrows():
        e = ek(str(r["src_switch"]), str(r["dst_switch"]))
        bw = float(r.get("bw_mbps", 10000.0))
        # 簡單估：EF:15%, CS5:10%, AF:30%, CS3:10%, CS0:35%（僅視覺化 fallback）
        reserve_map.setdefault(e, {})
        reserve_map[e].setdefault("EF", 0.15*bw)
        reserve_map[e].setdefault("CS5",0.10*bw)
        reserve_map[e].setdefault("AF", 0.30*bw)
        reserve_map[e].setdefault("CS3",0.10*bw)
        reserve_map[e].setdefault("CS0",0.35*bw)

# topic 過濾
if topic_filter and "topic" in df_flows.columns:
    df_flows = df_flows[df_flows["topic"].fillna("").str.contains(topic_filter, case=False)]

# --- 高亮 Flow & Edge 檢視（在建圖前定義好變數） ---
with st.sidebar:
    st.subheader("高亮 Flow")
    flow_ids_all = []
    if df_flows is not None and not df_flows.empty and "admitted" in df_flows.columns:
        flow_ids_all = df_flows[df_flows["admitted"] == 1]["flow_id"].astype(str).tolist()
    hl_flow  = st.selectbox("選一條已 Admitted 的 Flow", ["(無)"] + flow_ids_all, index=0)
    hl_width = st.slider("高亮線寬", 6, 30, 16)
    hl_arrows = st.checkbox("顯示方向箭頭", value=True)

    st.subheader("Edge 檢視（詳情）")
    edge_opts = sorted({edge_key_str(str(r["src_switch"]), str(r["dst_switch"])) for _, r in df_edges.iterrows()}) if df_edges is not None else []
    sel_edge = st.selectbox("選擇 Edge", edge_opts if edge_opts else ["(none)"])
    if sel_edge and sel_edge != "(none)":
        a, b = sel_edge.split("-", 1)
        e = ek(a, b)
        bw = None
        ty = ""
        row = df_edges[(df_edges["src_switch"].astype(str) == a) & (df_edges["dst_switch"].astype(str) == b)]
        if row.empty:
            row = df_edges[(df_edges["src_switch"].astype(str) == b) & (df_edges["dst_switch"].astype(str) == a)]
        if not row.empty:
            bw = float(row.iloc[0].get("bw_mbps", 10000.0))
            ty = str(row.iloc[0].get("type", ""))
        st.markdown(f"**{sel_edge}**  • type=`{ty}`  • bw=`{bw if bw is not None else ''}` Mbps")
        table_rows = []
        for p in ["EF","CS5","AF","CS3","CS0"]:
            rv = float(reserve_map.get(e,{}).get(p, 0.0))
            uv = float(usage_map.get(e,{}).get(p, 0.0))
            util = (uv/rv*100.0) if rv>0 else (uv/(bw or 1.0)*100.0)
            table_rows.append({"pool": p, "reserve_mbps": rv, "usage_mbps": uv, "util_%": round(util,1)})
        st.table(pd.DataFrame(table_rows))


# 準備 PyVis 網路
net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#2c3e50", notebook=False, directed=False)
net.barnes_hut(gravity=-20000, central_gravity=0.0, spring_length=200, spring_strength=0.0, damping=0.9, overlap=0)
net.toggle_physics(False)  # 固定位置

# 加節點
for n in nodes:
    x, y = pos.get(n, (0.0, 0.0))
    role = roles.get(n, "other")
    color = NODE_COLOR.get(role, NODE_COLOR["other"])
    size = 22 if role=="core" else (18 if role=="access" else 14)
    net.add_node(n, label=n, x=float(x), y=float(y), color=color, size=size, physics=False, title=f"{n} ({role})")

# 邊的樣式（疊代所有 pools，取符合篩選者）
edge_style: Dict[Tuple[str,str], Dict[str, float]] = {}  # width/color per edge（聚合）
for _, r in df_edges.iterrows():
    u = str(r["src_switch"]); v = str(r["dst_switch"])
    e = ek(u, v)
    # 基礎顯示：使用者選的 pool 中，挑負載最高的那個來決定顏色與寬度（也可改成加總）
    best_t = 0.0
    best_color = "#bdc3c7"
    # 計算 usage ratio 或 usage/bw
    bw = float(r.get("bw_mbps", 10000.0))
    title_lines = [f"{u}-{v}", f"bw={bw:.0f} Mbps", f"type={str(r.get('type',''))}"]
    for p in selected_pools:
        rv = float(reserve_map.get(e, {}).get(p, 0.0))
        uv = float(usage_map.get(e, {}).get(p, 0.0))

        # 基準切換：usage/reserve 或 usage/bw
        if ratio_mode == "usage / reserve" and rv > 0:
            raw = uv / rv
        else:
            raw = uv / max(1.0, bw)
        raw = max(0.0, raw)

        # 非線性映射，讓小流量也看得出差異
        t = 1.0 - math.exp(-5.0 * float(color_boost) * raw)  # t ∈ [0,1)

        if t > best_t:
            best_t = t
            base = POOL_COLOR.get(p, "#7f8c8d")
            best_color = color_mix(base, t)
        if rv > 0 or uv > 0:
            util_pct = (uv / rv * 100.0) if rv > 0 else (uv / bw * 100.0)
            title_lines.append(f"{p}: reserve={rv:.2f}, usage={uv:.2f}, util={util_pct:.1f}%")

    # 邊寬放大倍率
    width = 1.0 + float(edge_gain) * best_t
    edge_style[e] = {"width": width, "color": best_color, "title": "<br/>".join(title_lines)}


# 疊加多播樹（著色加深/加粗）
if show_multicast and df_trees is not None and not df_trees.empty and "edges" in df_trees.columns:
    for _, r in df_trees.iterrows():
        edges_list = parse_edges_str(r.get("edges",""))
        for e in edges_list:
            if e in edge_style:
                edge_style[e]["width"] = max(edge_style[e]["width"], 7.5)
                edge_style[e]["color"] = color_mix("#2c3e50", 0.9)  # 深藍灰

# 加邊到圖
for _, r in df_edges.iterrows():
    u = str(r["src_switch"]); v = str(r["dst_switch"])
    e = ek(u, v)
    style = edge_style.get(e, {"width": 1.0, "color": "#bdc3c7", "title": f"{u}-{v}"})
    net.add_edge(u, v, value=style["width"], width=style["width"], color=style["color"], title=style["title"])


# --- 疊加所有 admitted flows（弱化線 + 速率對應寬度 + 偽動畫） ---
show_all_paths = st.sidebar.checkbox("顯示所有已 admitted flow 路徑（動態）", value=True)
if show_all_paths and df_flows is not None and not df_flows.empty:
    df_adm = df_flows[df_flows.get("admitted", 0) == 1]
    # 讓顏色隨時間「呼吸」：0.4~1.0 之間變化
    phase = time.time() % 1.0
    blink = 0.4 + 0.6 * 0.5 * (1 + math.sin(2 * math.pi * phase))

    for _, rr in df_adm.iterrows():
        path = parse_path_str(str(rr.get("path", "")))
        if len(path) < 2:
            continue
        cls = str(rr.get("class_admitted") or rr.get("class_req") or "CS0")
        pool = pool_of_class(cls)
        base = POOL_COLOR.get(pool, "#7f8c8d")
        color = color_mix(base, blink)
        rate = float(rr.get("rate_mbps", 0.0) or 0.0)
        width = max(2, min(12, 2 + rate * 1.5))   # 2..12px，與速率成正比

        for i in range(len(path) - 1):
            net.add_edge(
                path[i], path[i+1],
                width=width, color=color, arrows='to', physics=False
            )

# --- 疊加高亮 Flow（一定要在 generate_html 之前） ---
if 'hl_flow' in locals() and hl_flow and hl_flow != "(無)" and not df_flows.empty and "flow_id" in df_flows.columns:
    row_hl = df_flows[(df_flows["flow_id"].astype(str) == str(hl_flow)) & (df_flows["admitted"] == 1)]
    if not row_hl.empty:
        cls_hl = str(row_hl.iloc[0].get("class_admitted") or row_hl.iloc[0].get("class_req") or "CS0")
        pool_hl = pool_of_class(cls_hl)
        color_hl = POOL_COLOR.get(pool_hl, "#e74c3c")
        path_hl = parse_path_str(str(row_hl.iloc[0].get("path", "")))
        for i in range(len(path_hl) - 1):
            u_hl, v_hl = path_hl[i], path_hl[i+1]
            net.add_edge(
                u_hl, v_hl,
                width=int(hl_width),
                color=color_hl,
                arrows='to' if hl_arrows else '',
                physics=False
            )


# 右側：網路圖
with st.container():
    st.subheader("即時網路拓樸（邊寬/顏色 ≈ 使用率）")
    html = net.generate_html()
    st.components.v1.html(html, height=800, scrolling=True)

# 下方：流量列表 + 候選
st.subheader("Per-flow 狀態（已 admitted）")
if df_flows is None or df_flows.empty:
    st.info("尚無 per_flow.csv 或無資料。")
else:
    df_show = df_flows[df_flows["admitted"]==1].copy()
    if topic_filter:
        df_show = df_show[df_show["topic"].fillna("").str.contains(topic_filter, case=False)]
    st.dataframe(df_show[["flow_id","topic","class_admitted","rate_mbps","path","wecmp_candidates","wecmp_weights"]], use_container_width=True, height=260)

    if show_wecmp:
        st.markdown("**WECMP 候選與權重（選一個 Flow）**")
        flow_ids = df_show["flow_id"].tolist()
        if flow_ids:
            sel = st.selectbox("選擇 flow_id", flow_ids)
            row = df_show[df_show["flow_id"]==sel].iloc[0]
            cands = str(row.get("wecmp_candidates",""))
            weights = str(row.get("wecmp_weights",""))
            cand_list = [p for p in cands.split("|") if p]
            w_list = [float(x) if x else 0.0 for x in weights.split("|")] if weights else []
            if cand_list:
                st.write(pd.DataFrame({"candidate_path": cand_list, "weight": w_list}))
            else:
                st.info("該 Flow 沒有候選路徑資訊。")


# --- 自動刷新（放在所有 render 之後） ---
if autorefresh:
    time.sleep(1)
    st.rerun()
