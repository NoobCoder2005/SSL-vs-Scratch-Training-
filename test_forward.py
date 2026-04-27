import torch
from data.dataset_loader import CTDatasetSSL, CTDatasetSegmentation
from utils.augmentations import get_simclr_augmentations
from models.encoder import Encoder
from models.projection_head import ProjectionHead
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# -------------------------------
# 1. TEST SSL DATASET + AUGMENTATIONS
# -------------------------------
print("\n[1] Testing SSL Dataset...")

ssl_dataset = CTDatasetSSL(dummy=True, transform=get_simclr_augmentations())

v1, v2 = ssl_dataset[0]

print("View1 shape:", v1.shape)
print("View2 shape:", v2.shape)

assert v1.shape == v2.shape, "Views mismatch!"
assert len(v1.shape) == 3, "Expected (C,H,W)"

# Add batch dimension
v1 = v1.unsqueeze(0).to(device)
v2 = v2.unsqueeze(0).to(device)

# -------------------------------
# 2. TEST ENCODER + PROJECTION HEAD
# -------------------------------
print("\n[2] Testing Encoder + Projection Head...")

encoder = Encoder().to(device)
projector = ProjectionHead().to(device)

with torch.no_grad():
    f1 = encoder(v1)
    f2 = encoder(v2)

    # Take final feature map (x4)
    f1_final = f1[-1]
    f2_final = f2[-1]

    print("Encoder final feature shape:", f1_final.shape)

    # Global Average Pooling → (N, 512)
    f1_vec = torch.mean(f1_final, dim=(2, 3))
    f2_vec = torch.mean(f2_final, dim=(2, 3))

    print("Pooled feature shape:", f1_vec.shape)

    # Projection
    z1 = projector(f1_vec)
    z2 = projector(f2_vec)

    print("Projection output shape:", z1.shape)

# -------------------------------
# 3. TEST SEGMENTATION DATASET
# -------------------------------
print("\n[3] Testing Segmentation Dataset...")

seg_dataset = CTDatasetSegmentation(dummy=True)

img, mask = seg_dataset[0]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)

img = img.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

# -------------------------------
# 4. TEST UNET
# -------------------------------
print("\n[4] Testing UNet...")

model = UNet().to(device)

with torch.no_grad():
    out = model(img)

print("UNet output shape:", out.shape)
assert out.shape[2:] == img.shape[2:], "Seg logits must match input spatial size"

print("\n✅ ALL TESTS PASSED!")

from losses.contrastive_loss import NTXentLoss

loss_fn = NTXentLoss()

loss = loss_fn(z1, z2)
print("Loss:", loss.item())