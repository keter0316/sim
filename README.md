# 0) 建環境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # networkx, numpy, matplotlib, pyyaml, pandas 等

# 1) 產生拓樸 + ECMP/KSP + 檢核
python examples/build_paths.py \
  --switches 10 --hosts 8 --access-k 1 --core-mode ring --uplinks 2 \
  --seed 1 --radio --K 3

# 2) 單播 CSPF + Admission + WECMP
python -m routing.cspf \
  --edges outputs/matrix/edges_all.csv \
  --qos   configs/qos.yaml \
  --flows configs/flows.csv \
  --out   outputs/flows/per_flow.csv

# 3) 多播（Pub/Sub）樹 + 邊替換（可選）
python -m routing.steiner \
  --edges outputs/matrix/edges_all.csv \
  --flows configs/flows.csv \
  --mode mst --swap-pct 0.03 --swap-iters 64 \
  --out-trees   outputs/flows/trees.csv \
  --out-reroutes outputs/flows/reroutes.csv

# 4) 綜合檢核（報表 + 圖）
python quick_verify.py \
  --edges outputs/matrix/edges_all.csv \
  --fdb   outputs/paths/fdb_spb_ecmp.json \
  --ksp   outputs/paths/ksp_all_pairs.json \
  --flows outputs/flows/per_flow.csv \
  --qos   configs/qos.yaml \
  --theta 0.05
