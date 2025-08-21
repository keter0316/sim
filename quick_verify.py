# quick_verify.py（隨機抽樣版）
"""
檢查 FDB 與 KSP 的一致性
"""
import json, random
from pathlib import Path

p = Path("outputs/paths")
fdb = json.loads((p/"fdb_spb_ecmp.json").read_text())
ksp = json.loads((p/"ksp_all_pairs.json").read_text())

# --- 1) 隨機抽 5 對 (src,dst)，輸出下一跳數量 ---
pairs = [(s, d) for s, mp in fdb.items() for d in mp.keys()]
k = min(5, len(pairs))
samples = random.sample(pairs, k)          # 完全隨機抽樣
print("sample next-hops:",
      [(s, d, len(fdb[s][d])) for (s, d) in samples])
# 如果想看實際 next-hops，可改為：
# print("sample next-hops:",
#       [(s, d, fdb[s][d], len(fdb[s][d])) for (s, d) in samples])

# --- 2) 隨機抽 10 對檢查 KSP 與 FDB 一致性（第一跳 ∈ FDB 下一跳） ---
check = random.sample(pairs, k=min(10, len(pairs)))
bad = []
for s, d in check:
    paths = ksp.get(f"{s}->{d}", [])
    if not paths:
        continue
    first = paths[0][1] if len(paths[0]) >= 2 else None
    nh = set(fdb.get(s, {}).get(d, []))
    if first and nh and first not in nh:
        bad.append((f"{s}->{d}", first, list(nh)))
print("mismatch count:", len(bad)) 
