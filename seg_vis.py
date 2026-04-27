import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _label_overlay(rgb: np.ndarray, labels: np.ndarray, num_classes: int, alpha: float = 0.45):
    """Blend discrete label map onto RGB background."""
    cmap = plt.get_cmap("tab10")
    h, w = labels.shape
    for c in range(1, num_classes):
        m = labels == c
        if not m.any():
            continue
        color = np.array(cmap(c % 10)[:3], dtype=np.float32)
        for ch in range(3):
            rgb[:, :, ch] = np.where(
                m,
                (1 - alpha) * rgb[:, :, ch] + alpha * color[ch],
                rgb[:, :, ch],
            )
    return rgb


@torch.no_grad()
def save_seg_triplet(
    img: torch.Tensor,
    mask_gt: torch.Tensor,
    logits: torch.Tensor,
    out_path: str,
    num_classes: int = 5,
    title: str = "",
):
    """
    img: (1,H,W) or (H,W), mask_gt: (H,W) int, logits: (C,H,W).
    Saves one figure: CT | GT overlay | Pred overlay.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if img.dim() == 3:
        x = img[0].cpu().numpy()
    else:
        x = img.cpu().numpy()
    x = np.clip(x, 0, 1)

    gt = mask_gt.cpu().numpy().astype(np.int64)
    pred = logits.argmax(dim=0).cpu().numpy().astype(np.int64)

    base = np.stack([x, x, x], axis=-1)
    gt_rgb = _label_overlay(base.copy(), gt, num_classes)
    pr_rgb = _label_overlay(base.copy(), pred, num_classes)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(x, cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")
    axes[1].imshow(gt_rgb)
    axes[1].set_title("GT")
    axes[1].axis("off")
    axes[2].imshow(pr_rgb)
    axes[2].set_title("Pred")
    axes[2].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
