# sim.py（節選：建 FDB、KSP 候選、輸出 JSON）
import json
from pathlib import Path
from models.topology import Topology
from routing.spb_ecmp import ecmp_fdb
from routing.ksp_additional import yen_ksp

if __name__ == "__main__":
    topo = Topology.from_csv(Path("sim/configs/edges.csv"))
    fdb = ecmp_fdb(topo.G, weight="w")
    # 範例：為 (S8→S2) 取 K=3 條候選附加路徑
    ksp = yen_ksp(topo.G, "S8", "S2", K=3, weight="w")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/fdb_spb_ecmp.json").write_text(json.dumps(fdb, indent=2, ensure_ascii=False))
    Path("outputs/ksp_S8_S2.json").write_text(json.dumps(ksp, indent=2, ensure_ascii=False))
    print("[OK] FDB & KSP exported.")
