import torch


@torch.no_grad()
def dice_per_class(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1e-5):
    """Mean Dice per class over batch (macro over spatial dims, micro over batch)."""
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        d = (2 * inter + smooth) / (union + smooth)
        dices.append(d.mean())
    return torch.stack(dices)


@torch.no_grad()
def mean_foreground_dice(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1e-5):
    """Average Dice over classes 1..C-1 (exclude background 0)."""
    per = dice_per_class(logits, targets, num_classes, smooth=smooth)
    if num_classes <= 1:
        return per[0]
    return per[1:].mean()


@torch.no_grad()
def evaluate_seg_batch_losses(model, loader, device, dice_loss, ce_loss, num_classes: int):
    model.eval()
    n = 0
    sum_ce = 0.0
    sum_dice_loss = 0.0
    sum_fg_dice = 0.0
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        out = model(img)
        sum_ce += float(ce_loss(out, mask).item())
        sum_dice_loss += float(dice_loss(out, mask).item())
        sum_fg_dice += float(mean_foreground_dice(out, mask, num_classes).item())
        n += 1
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return sum_ce / n, sum_dice_loss / n, sum_fg_dice / n
