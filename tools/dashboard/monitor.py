#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 監控儀表板（新版 API）
讀取 outputs/paths/dynamic/ 下列檔案：
  - results_t*.json（每個 tick 的 Admission 結果）
  - snapshot_t*.json（每個 tick 的 reserved/usage 快照）
  - results_all.json、kpi_summary.json（總表與 KPI 摘要）

使用方式：
  streamlit run tools/dashboard/monitor.py
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ===== 路徑設定（可在右側 UI 變更） =====
DEFAULT_BASE = Path("outputs") / "dynamic"

# ===== 小工具 =====
def _list_sorted(base: Path, glob_pat: str) -> List[Path]:
    files = list(base.glob(glob_pat))
    files.sort(key=lambda p: p.stat().st_mtime)
    return files

def _latest(base: Path, pattern: str) -> Path | None:
    files = _list_sorted(base, pattern)
    return files[-1] if files else None

def _read_json(p: Path) -> dict | list:
    return json.loads(p.read_text(encoding="utf-8"))

def _edge_key_fmt(edge_key: Any) -> str:
    # edge_key 可能是 "('s1', 's2')" 或 ["s1","s2"] 或 "s1|s2"
    if isinstance(edge_key, (list, tuple)) and len(edge_key) == 2:
        return f"{edge_key[0]}—{edge_key[1]}"
    s = str(edge_key).strip()
    # 嘗試解析 "('s1', 's2')" 形式
    m = re.findall(r"'([^']+)'", s)
    if len(m) == 2:
        return f"{m[0]}—{m[1]}"
    if "|" in s:
        return s.replace("|", "—")
    return s

def _extract_tick_num(p: Path) -> int:
    m = re.search(r"_t(\d+)\.json$", p.name)
    return int(m.group(1)) if m else -1

def parse_snapshot(p: Path) -> pd.DataFrame:
    data = _read_json(p)
    reserved: Dict[str, Dict[str, float]] = data["reserved"]
    usage:    Dict[str, Dict[str, float]] = data["usage"]

    rows = []
    for edge, pools in reserved.items():
        pools_u = usage.get(edge, {})
        for pool, r_mbps in pools.items():
            u_mbps = float(pools_u.get(pool, 0.0))
            rows.append({
                "edge": _edge_key_fmt(edge),
                "pool": pool,
                "reserved_mbps": float(r_mbps),
                "used_mbps": u_mbps,
                "utilization": (u_mbps / r_mbps if r_mbps > 0 else 0.0),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["utilization", "edge", "pool"], ascending=[False, True, True])
    return df

def parse_results(p: Path) -> pd.DataFrame:
    arr: List[dict] = _read_json(p)
    rows = []
    for r in arr:
        rows.append({
            "tick":     r.get("tick"),
            "src":      r.get("src"),
            "dst":      r.get("dst"),
            "topic":    r.get("topic"),
            "rate":     r.get("rate_mbps"),
            "class":    r.get("class"),
            "result":   r.get("result"),
            "pinned":   bool(r.get("pinned", False)),
            "reason":   r.get("reason"),
            "path":     "→".join(r.get("path", [])) if r.get("path") else "",
        })
    return pd.DataFrame(rows)

def parse_kpi(p: Path) -> pd.DataFrame:
    d: Dict[str, int] = _read_json(p)  # {"AF21:OK": 3, "EF:BLOCKED":2, ...}
    rows = []
    for k, v in d.items():
        if ":" in k:
            cls, res = k.split(":", 1)
        else:
            cls, res = k, ""
        rows.append({"class": cls, "result": res, "count": int(v)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["class", "result"])
    return df

# ===== UI 開始 =====
st.set_page_config(page_title="Sim QoS Monitor", layout="wide")
st.title("🛰️ Sim QoS Monitor (Streamlit)")

# 右側設定欄
with st.sidebar:
    st.header("⚙️ 設定")
    base_in = st.text_input("動態輸出資料夾", str(DEFAULT_BASE))
    BASE = Path(base_in)
    st.caption("預設：outputs/paths/dynamic/")
    interval = st.slider("自動刷新（秒）", 1, 10, 2)
    auto = st.toggle("自動刷新", value=True)
    if auto:
        # 每 interval 秒自動刷新
        st_autorefresh(interval=interval * 1000, key="auto_refresh_key")

# 主要內容
col1, col2 = st.columns([3, 2])

# 讀取最新檔案指標
snap_p = _latest(BASE, "snapshot_t*.json")
res_p  = _latest(BASE, "results_t*.json")
allres_p = BASE / "results_all.json"
kpi_p    = BASE / "kpi_summary.json"

if not BASE.exists():
    st.error(f"找不到資料夾：{BASE}")
elif not snap_p or not res_p:
    st.warning("尚未找到 snapshot_t*.json / results_t*.json。\n\n請先執行：`python tools/dynamic_demo.py` 產生資料。")
else:
    tick = _extract_tick_num(res_p)
    st.subheader(f"⏱️ 最新 Tick：t={tick}")
    st.caption(f"results: {res_p.name} ｜ snapshot: {snap_p.name}")

    # 區塊 1：當前 flows 結果
    with col1:
        st.markdown("### 📦 當前 Admission 結果")
        df_res = parse_results(res_p)
        if df_res.empty:
            st.info("當前結果為空。")
        else:
            def _row_style(row):
                if row["result"] == "BLOCKED":
                    return ["background-color: #ffe5e5"] * len(row)
                if row["pinned"]:
                    return ["background-color: #e8f7ff"] * len(row)
                return ["" for _ in row]
            st.dataframe(df_res.style.apply(_row_style, axis=1), use_container_width=True, height=360)

    # 區塊 2：各池利用率（邊×池）
    with col2:
        st.markdown("### 🧮 保留池利用（依利用率排序）")
        df_snap = parse_snapshot(snap_p)
        if df_snap.empty:
            st.info("尚無利用率資料。")
        else:
            st.dataframe(df_snap.head(200), use_container_width=True, height=360)
            hot = df_snap[df_snap["utilization"] >= 0.9]
            if not hot.empty:
                st.error(f"⚠️ 有 {len(hot)} 個邊×池利用率 ≥ 90%（可能即將阻擋/降級）。")
            else:
                st.success("✅ 目前沒有邊×池超過 90%。")

    # 區塊 3：累計 KPI
    st.markdown("### 📊 累計 KPI")
    if kpi_p.exists():
        df_kpi = parse_kpi(kpi_p)
        if not df_kpi.empty:
            chart_df = df_kpi.pivot_table(
                index="class", columns="result", values="count", aggfunc="sum"
            ).fillna(0)
            st.bar_chart(chart_df)
            st.dataframe(df_kpi, use_container_width=True, height=280)
        else:
            st.info("KPI 摘要為空。")
    else:
        st.info("尚無 kpi_summary.json（跑過多個 tick 後就會出現）。")

    # 區塊 4：歷史 Tick 檢視
    st.markdown("### 🧭 歷史 Tick 檢視")
    all_res_files = _list_sorted(BASE, "results_t*.json")
    if all_res_files:
        names = [p.name for p in all_res_files]
        sel = st.selectbox("選擇 tick 檔", names, index=len(names)-1)
        p = BASE / sel
        st.caption(str(p))
        st.dataframe(parse_results(p), use_container_width=True, height=280)
    else:
        st.info("沒有可選的歷史 results_t*.json。")
