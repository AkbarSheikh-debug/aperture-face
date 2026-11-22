"""Task 2 - train the ResNet face embedder.

Two-stage recipe (configs/):
  stage 1  pretrain_digiface.yaml : ImageNet ResNet50 -> DigiFace-1M synthetic identities
  stage 2  finetune_lfw.yaml      : stage-1 weights   -> LFW *train* identities  -> checkpoints/best_model.pth

Model selection uses ONLY the LFW validation identities (all-vs-all val pairs, TAR@FAR=1%, AUC tie-break).
Test identities are never loaded here.

Usage:
  python train.py --config configs/finetune_lfw.yaml [--quick] [--set loss.kind=arcface epochs=10 ...]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.data import FaceDataset, PKSampler, digiface_train_set, lfw_split, lfw_train_set, train_transform
from common.embed import embed_paths
from common.io import RESULTS, ROOT, load_yaml, save_json
from common.losses import CombinedLoss
from common.metrics import auc, eer, tar_at_far
from common.seed import seed_everything
from model import build_model, load_checkpoint, save_checkpoint


def parse_overrides(items):
    import yaml

    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def apply_overrides(cfg, overrides):
    for key, v in overrides.items():
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


# ------------------------------------------------------------------------------------------- validation
def val_pairs_all(df):
    """All-vs-all pairs among validation images (excluding self pairs and near-duplicate pairs)."""
    n = len(df)
    iu, ju = np.triu_indices(n, k=1)
    ids = df.identity.to_numpy()
    dup = df.dup_cluster.to_numpy()
    keep = dup[iu] != dup[ju]
    iu, ju = iu[keep], ju[keep]
    return iu, ju, (ids[iu] == ids[ju]).astype(np.int8)


@torch.no_grad()
def validate(model, device, val_df, pairs, tta=True, num_workers=0):
    emb = embed_paths(model, val_df.path.tolist(), device, tta=tta, num_workers=num_workers)
    iu, ju, lab = pairs
    s = (emb[iu] * emb[ju]).sum(1)
    tar1, _ = tar_at_far(lab, s, 1e-2)
    tar01, _ = tar_at_far(lab, s, 1e-3)
    return {"auc": auc(lab, s), "eer": eer(lab, s)[0], "tar@far1e-2": tar1, "tar@far1e-3": tar01,
            "n_genuine": int(lab.sum()), "n_impostor": int((lab == 0).sum())}


def score_key(m):
    return (round(m["tar@far1e-2"], 4), m["auc"])


# ------------------------------------------------------------------------------------------------ train
def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true", help="smoke test: tiny data, 1 epoch")
    ap.add_argument("--set", nargs="*", help="override config entries, e.g. loss.kind=arcface epochs=5")
    args = ap.parse_args()
    cfg = apply_overrides(load_yaml(args.config), parse_overrides(args.set))
    if args.quick:
        cfg.update(epochs=1, max_batches=15, num_workers=2)
        cfg["digiface_max_ids"] = 100
    seed_everything(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["name"]
    out_path = ROOT / cfg["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] device={device}  config={cfg}")

    # ---- data
    if cfg["dataset"] == "digiface":
        paths, labels, n_classes = digiface_train_set(cfg.get("digiface_max_ids"))
    elif cfg["dataset"] == "digiface+lfw":
        # joint stage 1: synthetic identities + the real LFW *train* identities as extra classes (label offset)
        p1, l1, n1 = digiface_train_set(cfg.get("digiface_max_ids"))
        p2, l2, n2 = lfw_train_set()
        paths, labels, n_classes = p1 + p2, np.concatenate([l1, l2 + n1]), n1 + n2
    elif cfg["dataset"] == "lfw":
        paths, labels, n_classes = lfw_train_set()
    else:
        raise ValueError(cfg["dataset"])
    print(f"[{name}] train images={len(paths)} identities={n_classes}")
    ds = FaceDataset(paths, labels, train_transform(cfg.get("augment", "heavy")))
    P, K = cfg["sampler"]["P"], cfg["sampler"]["K"]
    n_batches = cfg.get("max_batches") or cfg.get("batches_per_epoch") or max(1, len(paths) // (P * K))
    sampler = PKSampler(labels, P, K, n_batches=n_batches, seed=cfg.get("seed", 42))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.get("num_workers", 8), pin_memory=True,
                        persistent_workers=cfg.get("num_workers", 8) > 0)
    val_df = lfw_split("val")
    val_pairs = val_pairs_all(val_df)

    # ---- model + loss
    if cfg.get("init_checkpoint"):
        model, _ = load_checkpoint(ROOT / cfg["init_checkpoint"])
        print(f"[{name}] initialised from {cfg['init_checkpoint']}")
    else:
        model = build_model(cfg["model"], pretrained=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    L = cfg["loss"]
    criterion = CombinedLoss(L["kind"], model.config["emb_dim"], n_classes, s=L.get("s", 64), m=L.get("m", 0.5),
                             triplet_margin=L.get("triplet_margin", 0.3),
                             triplet_weight=L.get("triplet_weight", 0.1)).to(device)

    O = cfg["optim"]
    body_params = list(model.body.parameters())
    body_ids = {id(p) for p in body_params}
    neck_params = [p for p in model.parameters() if id(p) not in body_ids]

    def groups(params, lr):
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        return [{"params": decay, "lr": lr, "weight_decay": O["weight_decay"]},
                {"params": no_decay, "lr": lr, "weight_decay": 0.0}]

    param_groups = (groups(body_params, O["lr"] * O.get("backbone_lr_mult", 1.0))
                    + groups(neck_params, O["lr"])
                    + groups(list(criterion.parameters()), O["lr"] * O.get("head_lr_mult", 1.0)))
    param_groups = [g for g in param_groups if g["params"]]
    for g in param_groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.SGD(param_groups, momentum=O.get("momentum", 0.9), nesterov=True)
    total_steps = cfg["epochs"] * n_batches
    warm = int(O.get("warmup_epochs", 1) * n_batches)

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    writer = SummaryWriter(str(RESULTS / "runs" / name))
    history = {"config": cfg, "epochs": []}
    best, best_metrics = None, None

    def maybe_save(epoch, metrics):
        nonlocal best, best_metrics
        if best is None or score_key(metrics) > best:
            best, best_metrics = score_key(metrics), dict(metrics, epoch=epoch)
            save_checkpoint(out_path, model, extra={"val": best_metrics, "train_config": cfg,
                                                    "n_train_classes": n_classes})
            print(f"[{name}]   -> new best, saved {out_path.name}")

    if cfg.get("init_checkpoint") and not args.quick:   # epoch 0 = "no fine-tuning" reference point
        m0 = validate(model, device, val_df, val_pairs)
        print(f"[{name}] epoch 0 (before training) val: {m0}")
        history["epochs"].append({"epoch": 0, "val": m0})
        maybe_save(0, m0)

    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); criterion.train()
        if epoch <= cfg.get("freeze_bn_epochs", 0):
            set_bn_eval(model.body)
        margin = L.get("m", 0.5) * min(1.0, epoch / max(1, L.get("margin_warmup_epochs", 0) + 1)) \
            if L.get("margin_warmup_epochs", 0) else None
        t0, agg = time.time(), {}
        for x, y in loader:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step)
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                emb = model.forward_unnormalized(x)
            loss, logs = criterion(emb.float(), y, margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 5.0)
            opt.step()
            logs["loss"] = loss.item()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            if step % 50 == 0:
                for k, v in logs.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
            step += 1
        train_logs = {k: v / n_batches for k, v in agg.items()}
        dt = time.time() - t0
        metrics = validate(model, device, val_df, val_pairs)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[{name}] epoch {epoch}/{cfg['epochs']} {dt:.0f}s ({n_batches * P * K / dt:.0f} img/s) "
              f"train={ {k: round(v, 4) for k, v in train_logs.items()} } val={ {k: round(v, 4) for k, v in metrics.items()} }")
        history["epochs"].append({"epoch": epoch, "train": train_logs, "val": metrics, "seconds": dt,
                                  "margin": margin})
        maybe_save(epoch, metrics)
        save_json(history, RESULTS / "train_logs" / f"{name}.json")

    history["best"] = best_metrics
    save_json(history, RESULTS / "train_logs" / f"{name}.json")
    writer.close()
    print(f"[{name}] done. best val: {best_metrics}")


if __name__ == "__main__":
    main()
