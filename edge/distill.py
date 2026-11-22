"""Knowledge distillation ResNet50 (teacher) -> ResNet18 (student) for edge devices.

Loss = w_cos * (1 - cos(s_i, t_i))  +  w_rel * || S_s - S_t ||^2
  where S = E E^T is the in-batch cosine-similarity matrix. The second term transfers the *relative* geometry
  (who is close to whom), which is exactly what verification thresholds operate on.
The student sees the same augmented image as the teacher. Model selection on LFW val identities only.

Usage: python -m edge.distill [--config configs/distill_r18.yaml] [--quick]
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from common.data import FaceDataset, digiface_train_set, lfw_split, lfw_train_set, train_transform
from common.io import DIGIFACE_META, RESULTS, ROOT, load_yaml, save_json
from common.seed import seed_everything
from model import build_model, load_checkpoint, save_checkpoint
from train import score_key, val_pairs_all, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill_r18.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.quick:
        cfg.update(epochs=1, images_per_epoch=256 * 10, num_workers=2)
    seed_everything(cfg["seed"])
    device = "cuda"

    paths, labels = [], []
    if "digiface" in cfg["data"] and DIGIFACE_META.exists():
        p, l, _ = digiface_train_set(); paths += p; labels += list(l)
    if "lfw_train" in cfg["data"]:
        p, l, _ = lfw_train_set(); paths += p; labels += list(l)
    print(f"[distill] training images: {len(paths)}")
    ds = FaceDataset(paths, np.zeros(len(paths)), train_transform(cfg.get("augment", "light")))
    sampler = RandomSampler(ds, replacement=False, num_samples=min(cfg["images_per_epoch"], len(ds)))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"],
                        pin_memory=True, drop_last=True, persistent_workers=True)

    teacher, _ = load_checkpoint(ROOT / cfg["teacher"], device)
    teacher = teacher.eval().to(memory_format=torch.channels_last)
    student = build_model(cfg["student"], pretrained=True).to(device).to(memory_format=torch.channels_last)
    O = cfg["optim"]
    decay = [p for p in student.parameters() if p.ndim > 1]
    no_decay = [p for p in student.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": O["weight_decay"]},
                           {"params": no_decay, "weight_decay": 0.0}], lr=O["lr"], momentum=O["momentum"], nesterov=True)
    steps_per_epoch = len(loader)
    total, warm = cfg["epochs"] * steps_per_epoch, int(O["warmup_epochs"] * steps_per_epoch)

    val_df = lfw_split("val")
    val_pairs = val_pairs_all(val_df)
    t_val = validate(teacher, device, val_df, val_pairs)
    print(f"[distill] teacher val: {t_val}")
    history, best, step = {"config": cfg, "teacher_val": t_val, "epochs": []}, None, 0
    out = ROOT / cfg["out"]
    for epoch in range(1, cfg["epochs"] + 1):
        student.train()
        t0, agg = time.time(), {"cos": 0.0, "rel": 0.0}
        for x, _ in loader:
            lr = O["lr"] * ((step + 1) / warm if step < warm else
                            0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t = F.normalize(teacher.forward_unnormalized(x).float(), dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = student.forward_unnormalized(x)
            s = F.normalize(s.float(), dim=1)
            l_cos = (1 - (s * t).sum(1)).mean()
            l_rel = F.mse_loss(s @ s.t(), t @ t.t())
            loss = cfg["loss"]["cosine_weight"] * l_cos + cfg["loss"]["relational_weight"] * l_rel
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg["cos"] += l_cos.item(); agg["rel"] += l_rel.item(); step += 1
        logs = {k: v / steps_per_epoch for k, v in agg.items()}
        m = validate(student, device, val_df, val_pairs)
        print(f"[distill] epoch {epoch}/{cfg['epochs']} {time.time() - t0:.0f}s loss={logs} val={ {k: round(v, 4) for k, v in m.items()} }",
              flush=True)
        history["epochs"].append({"epoch": epoch, "train": logs, "val": m})
        if best is None or score_key(m) > best:
            best = score_key(m)
            save_checkpoint(out, student, extra={"val": dict(m, epoch=epoch), "distill_config": cfg})
            print(f"[distill]   -> saved {out.name}")
        save_json(history, RESULTS / "train_logs" / f"{cfg['name']}.json")


if __name__ == "__main__":
    main()
