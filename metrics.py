import torch


def dice_score(pred, target, num_classes=5):
    pred = torch.argmax(pred, dim=1)

    dice = 0
    for cls in range(1, num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()

        intersection = (p * t).sum()
        union = p.sum() + t.sum()

        dice += (2 * intersection) / (union + 1e-5)

    return dice / (num_classes - 1)