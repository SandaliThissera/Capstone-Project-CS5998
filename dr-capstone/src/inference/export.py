"""Pack the final ensemble into one small file (weights only, no optimiser state)."""
import argparse, json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cutoff-json", default="reports/test_final_ensemble.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sds = [{k: v.cpu() for k, v in torch.load(Path(a.runs_dir) / r / "last.pt", map_location="cpu")["model"].items()}
           for r in a.runs]
    cutoff = json.load(open(a.cutoff_json))["cutoff_from_val"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "efficientnet_b0", "size": 512, "cutoff": float(cutoff),
                "runs": list(a.runs), "state_dicts": sds}, a.out)
    print("saved", a.out, f"| {len(sds)} models | cutoff {cutoff:.4f}")


if __name__ == "__main__":
    main()
