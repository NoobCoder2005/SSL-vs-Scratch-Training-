import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running this file directly: `python evaluation/eval_segmentation.py ...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_loader import CTDatasetSegmentation
from models.unet import UNet
from utils.seg_vis import save_seg_triplet


def _read_patients_list(path: str | None):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


@dataclass
class DiceResult:
    num_samples: int
    num_batches: int
    dice_per_class: list[float]
    mean_fg_dice: float


@torch.no_grad()
def evaluate_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    amp: bool,
):
    model.eval()
    use_amp = bool(amp and device == "cuda")

    # accumulate intersection/union over whole dataset for stability
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)

    n_batches = 0
    for img, mask in tqdm(loader, desc="eval", leave=True):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img)
        pred = logits.argmax(dim=1)

        for c in range(num_classes):
            p = (pred == c)
            t = (mask == c)
            inter[c] += (p & t).sum().to(torch.float64)
            union[c] += (p.sum() + t.sum()).to(torch.float64)

        n_batches += 1

    dice = ((2.0 * inter + 1e-5) / (union + 1e-5)).detach().cpu().tolist()
    if num_classes > 1:
        mean_fg = float(sum(dice[1:]) / (num_classes - 1))
    else:
        mean_fg = float(dice[0])
    return DiceResult(
        num_samples=len(loader.dataset),
        num_batches=n_batches,
        dice_per_class=[float(x) for x in dice],
        mean_fg_dice=mean_fg,
    )


@torch.no_grad()
def save_example_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    out_dir: str,
    num_classes: int,
    max_images: int,
    amp: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    if max_images <= 0:
        return

    model.eval()
    use_amp = bool(amp and device == "cuda")
    saved = 0
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img)
        b = img.shape[0]
        for i in range(b):
            if saved >= max_images:
                return
            out_path = os.path.join(out_dir, f"test_pred_{saved:03d}.png")
            save_seg_triplet(
                img[i].cpu(),
                mask[i],
                logits[i].cpu(),
                out_path,
                num_classes=num_classes,
                title=f"test_pred_{saved:03d}",
            )
            saved += 1


def main():
    p = argparse.ArgumentParser(description="Evaluate UNet segmentation on a patient split.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--test_list", type=str, required=True, help="Path to test patients .txt")
    p.add_argument("--ckpt", type=str, default="checkpoints/unet.pth")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_dir", type=str, default="results/eval")
    p.add_argument("--max_images", type=int, default=12, help="How many qualitative PNGs to save.")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    patients = _read_patients_list(args.test_list)
    ds = CTDatasetSegmentation(root_dir=args.data_root, dummy=False, patients_list=patients)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda" and args.num_workers > 0),
    )

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = UNet(num_classes=args.num_classes).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate_dice(
        model=model,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        amp=args.amp,
    )

    # Write JSON
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    # Write CSV
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["num_samples", metrics.num_samples])
        w.writerow(["num_batches", metrics.num_batches])
        w.writerow(["mean_fg_dice", metrics.mean_fg_dice])
        for i, d in enumerate(metrics.dice_per_class):
            w.writerow([f"dice_class_{i}", d])

    # Qualitative images
    save_example_predictions(
        model=model,
        loader=loader,
        device=device,
        out_dir=os.path.join(args.out_dir, "predictions"),
        num_classes=args.num_classes,
        max_images=args.max_images,
        amp=args.amp,
    )

    print("Evaluation complete.")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Split: {args.test_list}")
    print(f"  MeanFgDice: {metrics.mean_fg_dice:.4f}")
    print(f"  Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()

