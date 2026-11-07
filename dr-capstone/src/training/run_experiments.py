import argparse, subprocess, sys, tempfile
from pathlib import Path

import yaml


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="optional: one run per seed, name gets a _s<seed> suffix")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    for cfg_path in a.configs:
        base = yaml.safe_load(open(cfg_path))
        for seed in (a.seeds or [base["seed"]]):
            cfg = dict(base)
            cfg["seed"] = seed
            if a.seeds:
                cfg["name"] = f"{base['name']}_s{seed}"
            run_dir = Path(a.out_dir) / cfg["name"]
            tmp_cfg = tmp / f"{cfg['name']}.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg))

            if (run_dir / "summary.json").exists():
                print(f"[skip training] {cfg['name']} already finished", flush=True)
            else:
                run([sys.executable, "-m", "src.training.train", "--config", str(tmp_cfg),
                     "--out-dir", a.out_dir, "--resume"])

            for ckpt, fname in (("last", "probs_val_last.csv"), ("best", "probs_val.csv")):
                if not (run_dir / fname).exists():
                    run([sys.executable, "-m", "src.evaluation.evaluate", "--run-dir", str(run_dir),
                         "--split", "val", "--ckpt", ckpt])


if __name__ == "__main__":
    main()
