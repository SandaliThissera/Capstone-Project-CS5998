import argparse, csv, json, random, time
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset, build_transforms, class_weights
from src.evaluation.metrics import compute_metrics


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    ys, ps, total, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        total += criterion(logits.float(), y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy()); ps.append(logits.argmax(1).cpu().numpy())
    return total / n, np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="parent folder for runs (overrides config)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(a.out_dir or cfg["output"]["dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    tr_ds = DRDataset(d["splits"], "train", d["cache_dir"], build_transforms(True, d["image_size"]), d["enhance"])
    va_ds = DRDataset(d["splits"], "val", d["cache_dir"], build_transforms(False, d["image_size"]), d["enhance"])
    tr_dl = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                       num_workers=t["num_workers"], pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=t["batch_size"] * 2, shuffle=False,
                       num_workers=t["num_workers"], pin_memory=True)

    model = timm.create_model(m["arch"], pretrained=m["pretrained"], num_classes=m["num_classes"]).to(device)
    weights = class_weights(tr_ds.df, m["num_classes"]).to(device) if t["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t["amp"])

    start_epoch, best_qwk = 0, -1.0
    last = run_dir / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_qwk = ck["epoch"] + 1, ck["best_qwk"]
        print(f"Resumed: continuing from epoch {start_epoch + 1}")

    fields = ["epoch", "train_loss", "val_loss", "val_acc", "val_qwk", "val_macro_f1",
              "recall_per_class", "lr", "seconds"]
    log_path = run_dir / "log.csv"
    new_log = start_epoch == 0 or not log_path.exists()
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.DictWriter(log_f, fieldnames=fields)
    if new_log:
        log.writeheader()

    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(y); n += len(y)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_loss, yt, yp = evaluate(model, va_dl, device, criterion)
        mt = compute_metrics(yt, yp, m["num_classes"])
        secs = time.time() - t0

        if mt["qwk"] > best_qwk:
            best_qwk = mt["qwk"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_qwk": best_qwk}, last)

        log.writerow({"epoch": epoch + 1, "train_loss": round(run_loss / n, 4), "val_loss": round(val_loss, 4),
                      "val_acc": round(mt["acc"], 4), "val_qwk": round(mt["qwk"], 4),
                      "val_macro_f1": round(mt["macro_f1"], 4),
                      "recall_per_class": json.dumps(mt["recall_per_class"]),
                      "lr": f"{lr:.2e}", "seconds": round(secs, 1)})
        log_f.flush()
        print(f"epoch {epoch + 1:02d}/{t['epochs']} | train {run_loss / n:.3f} | val {val_loss:.3f} | "
              f"acc {mt['acc']:.3f} | qwk {mt['qwk']:.3f} | f1 {mt['macro_f1']:.3f} | {secs:.0f}s")

    log_f.close()
    (run_dir / "summary.json").write_text(json.dumps({"best_val_qwk": best_qwk, "epochs": t["epochs"]}, indent=2))
    print("done | best val QWK:", round(best_qwk, 4), "| run dir:", run_dir)


if __name__ == "__main__":
    main()
