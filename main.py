import argparse

from training.pretrain_ssl import train_ssl
from training.train_segmentation import train_seg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SSL pretrain or segmentation train")
    p.add_argument("mode", choices=["ssl", "seg"], help="ssl = SimCLR, seg = UNet")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Patient root (see CTDatasetSSL / CTDatasetSegmentation). Required unless --dummy.",
    )
    p.add_argument("--dummy", action="store_true", help="Random data smoke test")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
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
    p.add_argument(
        "--train_list",
        type=str,
        default=None,
        help="ssl/seg: Path to train patients .txt (optional).",
    )
    p.add_argument(
        "--val_list",
        type=str,
        default=None,
        help="ssl/seg: Path to val patients .txt (optional; enables val Dice + vis).",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="(ssl) Print progress every N training steps (0 = only per-epoch).",
    )
    p.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="seg: optional path to SSL encoder weights; omit to train encoder from scratch.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/unet.pth",
        help="seg: path to save UNet weights after training.",
    )
    p.add_argument(
        "--epochs_head",
        type=int,
        default=5,
        help="seg: epochs with encoder frozen (decoder only).",
    )
    p.add_argument(
        "--epochs_finetune",
        type=int,
        default=None,
        help="seg: full-model fine-tune epochs (default: same as --epochs).",
    )
    p.add_argument("--lr_head", type=float, default=1e-3, help="seg: LR when encoder frozen.")
    p.add_argument(
        "--lr_finetune",
        type=float,
        default=1e-4,
        help="seg: LR when training full UNet.",
    )
    p.add_argument(
        "--vis_dir",
        type=str,
        default="results/seg_vis",
        help="seg: where to save CT|GT|Pred PNGs after each val.",
    )
    p.add_argument(
        "--vis_max_samples",
        type=int,
        default=4,
        help="seg: max validation images to save per val pass.",
    )
    p.add_argument(
        "--cache_patients",
        type=int,
        default=1,
        help="seg: patients to cache in RAM per DataLoader worker (0 disables cache).",
    )
    args = p.parse_args()
    if not args.dummy and not args.data_root:
        p.error("Provide --data_root or use --dummy")
    if args.mode == "ssl":
        bs = args.batch_size if args.batch_size is not None else 32
        train_ssl(
            data_root=args.data_root,
            dummy=args.dummy,
            epochs=args.epochs,
            batch_size=bs,
            num_workers=args.num_workers,
            train_list_path=args.train_list,
            val_list_path=args.val_list,
            log_every=args.log_every,
            device=args.device,
            amp=args.amp,
        )
    else:
        bs = args.batch_size if args.batch_size is not None else 8
        epochs_ft = args.epochs_finetune if args.epochs_finetune is not None else args.epochs
        train_seg(
            data_root=args.data_root,
            dummy=args.dummy,
            batch_size=bs,
            num_workers=args.num_workers,
            encoder_ckpt=args.encoder_ckpt,
            save_path=args.save_path,
            train_list_path=args.train_list,
            val_list_path=args.val_list,
            epochs_head=args.epochs_head,
            epochs_finetune=epochs_ft,
            lr_head=args.lr_head,
            lr_finetune=args.lr_finetune,
            vis_dir=args.vis_dir,
            vis_max_samples=args.vis_max_samples,
            device=args.device,
            amp=args.amp,
            cache_patients=args.cache_patients,
        )
