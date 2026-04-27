import argparse
import os

import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from losses.dice_loss import DiceLoss
from data.dataset_loader import CTDatasetSegmentation
from utils.seg_metrics import evaluate_seg_batch_losses
from utils.seg_vis import save_seg_triplet


def _read_patients_list(path: str | None):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def _normalize_encoder_state_dict(raw):
    """Accept state dicts from SSL (flat Encoder) or nested / prefixed saves."""
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if isinstance(raw, dict) and "encoder" in raw and isinstance(raw["encoder"], dict):
        return raw["encoder"]
    if isinstance(raw, dict) and any(k.startswith("encoder.") for k in raw):
        return {k[len("encoder.") :]: v for k, v in raw.items() if k.startswith("encoder.")}
    if isinstance(raw, dict) and any(k.startswith("module.encoder.") for k in raw):
        return {
            k[len("module.encoder.") :]: v
            for k, v in raw.items()
            if k.startswith("module.encoder.")
        }
    return raw


def load_encoder_into_unet(model: UNet, encoder_ckpt: str | None, device: str) -> bool:
    """Load SSL encoder weights into ``model.encoder``. Returns True if weights were loaded."""
    if not encoder_ckpt or not str(encoder_ckpt).strip():
        print("Training from scratch (no SSL)")
        return False
    encoder_ckpt = os.path.expanduser(str(encoder_ckpt).strip())
    if not os.path.isfile(encoder_ckpt):
        print("Training from scratch (no SSL)")
        return False

    raw = torch.load(encoder_ckpt, map_location=device, weights_only=True)
    state = _normalize_encoder_state_dict(raw)
    target = model.encoder
    target_keys = set(target.state_dict().keys())
    loaded_keys = set(state.keys()) if isinstance(state, dict) else set()
    overlap = target_keys & loaded_keys
    incomp = target.load_state_dict(state, strict=False)
    print(
        f"Loaded encoder into UNet from {encoder_ckpt}: "
        f"{len(overlap)}/{len(target_keys)} parameter tensors matched by name."
    )
    if incomp.missing_keys:
        mk = incomp.missing_keys[:12]
        extra = f" ... (+{len(incomp.missing_keys) - 12})" if len(incomp.missing_keys) > 12 else ""
        print(f"  missing_keys: {mk}{extra}")
    if incomp.unexpected_keys:
        uk = incomp.unexpected_keys[:12]
        extra = f" ... (+{len(incomp.unexpected_keys) - 12})" if len(incomp.unexpected_keys) > 12 else ""
        print(f"  unexpected_keys: {uk}{extra}")
    return True


def _decoder_params(model: UNet):
    return (
        list(model.up4.parameters())
        + list(model.up3.parameters())
        + list(model.up2.parameters())
        + list(model.up1.parameters())
        + list(model.final.parameters())
    )


def _set_encoder_trainable(model: UNet, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def _save_val_visuals(
    model: UNet,
    val_loader: DataLoader,
    device: str,
    vis_dir: str,
    tag: str,
    num_classes: int,
    max_samples: int,
):
    if max_samples <= 0 or len(val_loader.dataset) == 0:
        return
    model.eval()
    saved = 0
    for img, mask in val_loader:
        img = img.to(device, non_blocking=True)
        out = model(img)
        b = img.shape[0]
        for i in range(b):
            if saved >= max_samples:
                return
            path = os.path.join(vis_dir, f"{tag}_sample{saved}.png")
            save_seg_triplet(
                img[i].cpu(),
                mask[i],
                out[i].cpu(),
                path,
                num_classes=num_classes,
                title=tag,
            )
            saved += 1


def train_seg(
    data_root=None,
    dummy=False,
    batch_size=8,
    num_workers=4,
    encoder_ckpt=None,
    save_path="checkpoints/unet.pth",
    train_list_path=None,
    val_list_path=None,
    epochs_head=5,
    epochs_finetune=10,
    lr_head=1e-3,
    lr_finetune=1e-4,
    vis_dir="results/seg_vis",
    vis_max_samples=4,
    num_classes=5,
    device="auto",
    amp=False,
    cache_patients=1,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_patients = _read_patients_list(train_list_path)
    val_patients = _read_patients_list(val_list_path)

    train_ds = CTDatasetSegmentation(
        root_dir=data_root,
        dummy=dummy,
        patients_list=train_patients,
        cache_patients=cache_patients,
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
        val_ds = CTDatasetSegmentation(
            root_dir=data_root,
            dummy=False,
            patients_list=val_patients,
            cache_patients=cache_patients,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(device == "cuda" and num_workers > 0),
        )
    elif dummy and val_patients is None:
        val_loader = DataLoader(
            CTDatasetSegmentation(root_dir=data_root, dummy=True, patients_list=None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(device == "cuda" and num_workers > 0),
        )

    model = UNet(num_classes=num_classes).to(device)
    load_encoder_into_unet(model, encoder_ckpt, device)

    dice_loss = DiceLoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    use_amp = bool(amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    global_epoch = 0

    def run_validation(phase_tag: str):
        if val_loader is None:
            return
        ce_v, dice_v, fg_dice = evaluate_seg_batch_losses(
            model, val_loader, device, dice_loss, ce_loss, num_classes
        )
        print(
            f"  [{phase_tag}] Val  CE: {ce_v:.4f}  DiceLoss: {dice_v:.4f}  "
            f"MeanFgDice: {fg_dice:.4f}"
        )
        _save_val_visuals(
            model,
            val_loader,
            device,
            vis_dir,
            tag=f"e{global_epoch:03d}_{phase_tag}",
            num_classes=num_classes,
            max_samples=vis_max_samples,
        )

    # --- Phase 1: decoder only, frozen encoder ---
    if epochs_head > 0:
        _set_encoder_trainable(model, False)
        opt = torch.optim.Adam(_decoder_params(model), lr=lr_head)
        print(
            f"Phase 1 (encoder frozen): {epochs_head} epoch(s), lr={lr_head}. "
            f"Decoder params: {sum(p.numel() for p in _decoder_params(model))}"
        )
        for _ in range(epochs_head):
            model.train()
            last_loss = None
            for img, mask in train_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(img)
                    loss = dice_loss(out, mask) + ce_loss(out, mask)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                last_loss = loss
            print(f"Epoch {global_epoch} (head) Train loss: {last_loss.item():.4f}")
            run_validation("head")
            global_epoch += 1

    # --- Phase 2: full UNet, lower LR ---
    if epochs_finetune > 0:
        _set_encoder_trainable(model, True)
        opt = torch.optim.Adam(model.parameters(), lr=lr_finetune)
        print(f"Phase 2 (full model): {epochs_finetune} epoch(s), lr={lr_finetune}")
        for _ in range(epochs_finetune):
            model.train()
            last_loss = None
            for img, mask in train_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(img)
                    loss = dice_loss(out, mask) + ce_loss(out, mask)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                last_loss = loss
            print(f"Epoch {global_epoch} (finetune) Train loss: {last_loss.item():.4f}")
            run_validation("finetune")
            global_epoch += 1

    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}  Visuals: {vis_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="UNet segmentation (optional SSL init)")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Directory of patient folders (ct.nii.gz + segmentations/). Required unless --dummy.",
    )
    p.add_argument("--dummy", action="store_true", help="Use random data (smoke test)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Optional path to SSL encoder weights; omit to train from scratch.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/unet.pth",
        help="Where to save UNet weights after training.",
    )
    p.add_argument("--train_list", type=str, default=None)
    p.add_argument("--val_list", type=str, default=None)
    p.add_argument("--epochs_head", type=int, default=5, help="Train decoder with encoder frozen")
    p.add_argument("--epochs_finetune", type=int, default=10, help="Fine-tune full UNet")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-4)
    p.add_argument("--vis_dir", type=str, default="results/seg_vis")
    p.add_argument("--vis_max_samples", type=int, default=4)
    p.add_argument(
        "--cache_patients",
        type=int,
        default=1,
        help="How many patients to keep cached in RAM per DataLoader worker (0 disables cache).",
    )
    args = p.parse_args()
    if not args.dummy and not args.data_root:
        p.error("Provide --data_root or use --dummy")
    train_seg(
        data_root=args.data_root,
        dummy=args.dummy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder_ckpt=args.encoder_ckpt,
        save_path=args.save_path,
        train_list_path=args.train_list,
        val_list_path=args.val_list,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        vis_dir=args.vis_dir,
        vis_max_samples=args.vis_max_samples,
        cache_patients=args.cache_patients,
    )
