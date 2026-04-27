import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.projection_head import ProjectionHead
from losses.contrastive_loss import NTXentLoss
from data.dataset_loader import CTDatasetSSL
from utils.augmentations import get_simclr_augmentations


def _encoder_to_projection_vec(encoder_out):
    """Encoder returns skip tensors; SimCLR uses GAP on the deepest map (512, H', W')."""
    x4 = encoder_out[-1]
    return torch.mean(x4, dim=(2, 3))


def _read_patients_list(path: str | None):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


@torch.no_grad()
def _eval_ssl(encoder, projector, loader, criterion, device: str):
    encoder.eval()
    projector.eval()
    total = 0.0
    n = 0
    for v1, v2 in loader:
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        z1 = projector(_encoder_to_projection_vec(encoder(v1)))
        z2 = projector(_encoder_to_projection_vec(encoder(v2)))
        loss = criterion(z1, z2)
        total += float(loss.item())
        n += 1
    return (total / max(n, 1)), n


def train_ssl(
    data_root=None,
    dummy=False,
    epochs=10,
    batch_size=32,
    num_workers=4,
    train_list_path=None,
    val_list_path=None,
    log_every=0,
    device="auto",
    amp=False,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_patients = _read_patients_list(train_list_path)
    val_patients = _read_patients_list(val_list_path)

    train_ds = CTDatasetSSL(
        root_dir=data_root,
        transform=get_simclr_augmentations(),
        dummy=dummy,
        patients_list=train_patients,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda" and num_workers > 0),
    )

    val_loader = None
    if (not dummy) and val_patients:
        val_ds = CTDatasetSSL(
            root_dir=data_root,
            transform=get_simclr_augmentations(),
            dummy=False,
            patients_list=val_patients,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(device == "cuda" and num_workers > 0),
        )

    encoder = Encoder().to(device)
    projector = ProjectionHead().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=1e-3
    )

    criterion = NTXentLoss()
    use_amp = bool(amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        encoder.train()
        projector.train()

        last_loss = None
        epoch_start = time.perf_counter()
        step_start = epoch_start
        for step, (v1, v2) in enumerate(train_loader, start=1):
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                z1 = projector(_encoder_to_projection_vec(encoder(v1)))
                z2 = projector(_encoder_to_projection_vec(encoder(v2)))
                loss = criterion(z1, z2)
            last_loss = loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if log_every and (step % int(log_every) == 0):
                now = time.perf_counter()
                dt = now - step_start
                step_start = now
                print(
                    f"Epoch {epoch} Step {step} "
                    f"TrainLoss: {loss.item():.4f} "
                    f"({dt:.1f}s/{int(log_every)} steps)"
                )

        msg = (
            f"Epoch {epoch} TrainLoss(last_batch): "
            f"{last_loss.item() if last_loss is not None else float('nan')}"
            f" | EpochTime: {time.perf_counter() - epoch_start:.1f}s"
        )
        if val_loader is not None:
            val_loss, val_steps = _eval_ssl(
                encoder=encoder,
                projector=projector,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            msg += f" | ValLoss(avg): {val_loss:.4f} (steps={val_steps})"
        print(msg)

    torch.save(encoder.state_dict(), "checkpoints/encoder.pth")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SimCLR pretraining on CT slices")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Directory of patient folders (each with ct.nii.gz). Required unless --dummy.",
    )
    p.add_argument("--dummy", action="store_true", help="Use random data (smoke test)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_list", type=str, default=None, help="Path to train patients .txt")
    p.add_argument("--val_list", type=str, default=None, help="Path to val patients .txt")
    p.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="Print progress every N training steps (0 = only per-epoch).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on. Use 'cuda' for GPU if available; 'auto' picks cuda when possible.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (CUDA only). Usually faster and uses less VRAM.",
    )
    args = p.parse_args()
    if not args.dummy and not args.data_root:
        p.error("Provide --data_root or use --dummy")
    train_ssl(
        data_root=args.data_root,
        dummy=args.dummy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_list_path=args.train_list,
        val_list_path=args.val_list,
        log_every=args.log_every,
        device=args.device,
        amp=args.amp,
    )