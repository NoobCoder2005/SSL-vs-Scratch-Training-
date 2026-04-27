import io
import tempfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import streamlit as st
import torch
from PIL import Image

from models.unet import UNet


NUM_CLASSES = 5
INPUT_SIZE = 224
CLASS_NAMES = {
    0: "Background",
    1: "Liver",
    2: "Spleen",
    3: "Kidney Left",
    4: "Kidney Right",
}
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 99, 132),
    2: (54, 162, 235),
    3: (255, 206, 86),
    4: (75, 192, 192),
}


def normalize_01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > img.min():
        return (img - img.min()) / (img.max() - img.min())
    return np.zeros_like(img, dtype=np.float32)


def normalize_ct_hu(img: np.ndarray) -> np.ndarray:
    # Match training normalization used in dataset loader.
    img = np.clip(img.astype(np.float32), -1000, 1000)
    return (img + 1000.0) / 2000.0


def ct_window_for_display(img_hu: np.ndarray, center: float = 50.0, width: float = 400.0) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    win = np.clip(img_hu.astype(np.float32), lo, hi)
    return (win - lo) / max(hi - lo, 1e-6)


def preprocess_2d_array(img: np.ndarray, target_size: int) -> tuple[np.ndarray, torch.Tensor]:
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
    return img_resized, tensor


def preprocess_image(uploaded_bytes: bytes, target_size: int) -> tuple[np.ndarray, torch.Tensor]:
    pil_img = Image.open(io.BytesIO(uploaded_bytes)).convert("L")
    img = normalize_01(np.array(pil_img))
    return preprocess_2d_array(img, target_size=target_size)


def load_nifti_volume(uploaded_file) -> np.ndarray:
    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    nii = nib.load(tmp_path)
    vol = nii.get_fdata().astype(np.float32)
    return vol


def choose_best_slice_index(vol: np.ndarray, axis: int = 2, num_candidates: int = 32) -> int:
    n = vol.shape[axis]
    if n <= 1:
        return 0
    if n <= num_candidates:
        candidates = np.arange(n, dtype=int)
    else:
        candidates = np.linspace(0, n - 1, num_candidates).astype(int)

    best_idx = int(candidates[len(candidates) // 2])
    best_score = -1.0
    for idx in candidates:
        if axis == 0:
            sl = vol[idx, :, :]
        elif axis == 1:
            sl = vol[:, idx, :]
        else:
            sl = vol[:, :, idx]

        # Prefer slices that have meaningful tissue variability (avoid air-only slices).
        sl_norm = normalize_ct_hu(sl)
        score = float(np.std(sl_norm) + 0.5 * np.mean(sl_norm > 0.15))
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx, :, :]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


@torch.no_grad()
def infer_best_slice_for_segmentation(
    model: torch.nn.Module,
    vol: np.ndarray,
    device: str,
    axis: int = 2,
    num_candidates: int = 32,
):
    n = vol.shape[axis]
    if n <= num_candidates:
        candidates = np.arange(n, dtype=int)
    else:
        candidates = np.linspace(0, n - 1, num_candidates).astype(int)

    best = {
        "idx": int(candidates[len(candidates) // 2]),
        "pred": None,
        "model_img": None,
        "display_img": None,
        "score": -1.0,
    }

    for idx in candidates:
        slice_img = get_slice(vol, axis=axis, idx=int(idx))
        model_img = normalize_ct_hu(slice_img)
        display_img = ct_window_for_display(slice_img, center=50.0, width=400.0)
        model_img_resized, x = preprocess_2d_array(model_img, target_size=INPUT_SIZE)
        display_resized, _ = preprocess_2d_array(display_img, target_size=INPUT_SIZE)
        x = x.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Prefer slices with confident non-background predictions.
        non_bg_conf = float(1.0 - probs[:, 0, :, :].mean().item())
        non_bg_area = float(np.mean(pred > 0))
        score = non_bg_conf + 0.75 * non_bg_area

        if score > best["score"]:
            best.update(
                {
                    "idx": int(idx),
                    "pred": pred,
                    "model_img": model_img_resized,
                    "display_img": display_resized,
                    "score": score,
                }
            )

    return best


def label_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = color
    return out


def blend_overlay(gray_img: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    base = (np.clip(gray_img, 0, 1) * 255).astype(np.uint8)
    base_rgb = np.stack([base, base, base], axis=-1)
    return cv2.addWeighted(base_rgb, 1.0 - alpha, color_mask, alpha, 0)


@st.cache_resource
def load_model(checkpoint_path: str, device: str):
    model = UNet(num_classes=NUM_CLASSES).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    st.set_page_config(page_title="CT Segmentation App", layout="wide")
    st.title("CT Segmentation App (2D + 3D NIfTI)")
    st.write(
        "Upload either a 2D CT slice image (`.png`/`.jpg`) or a 3D CT volume "
        "(`.nii` / `.nii.gz`). The app auto-selects a representative slice for 3D volumes."
    )

    default_ckpt = "checkpoints/unet_from_scratch.pth"
    ckpt_path = st.sidebar.text_input("Checkpoint path", value=default_ckpt)
    device_opt = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)
    alpha = st.sidebar.slider("Overlay alpha", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    if device_opt == "cuda" and not torch.cuda.is_available():
        st.sidebar.warning("CUDA selected but unavailable. Falling back to CPU.")
        device_opt = "cpu"

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        st.error(f"Checkpoint not found: `{ckpt_path}`")
        st.stop()

    uploaded = st.file_uploader("Upload CT image/volume", type=["png", "jpg", "jpeg", "nii", "nii.gz"])
    if uploaded is None:
        st.info("Please upload a `.png`/`.jpg` or `.nii`/`.nii.gz` file to run segmentation.")
        return

    with st.spinner("Loading model..."):
        model = load_model(str(ckpt), device_opt)

    file_name = uploaded.name.lower()
    if file_name.endswith((".png", ".jpg", ".jpeg")):
        img_np, x = preprocess_image(uploaded.getvalue(), target_size=INPUT_SIZE)
        source_note = "2D image mode"
    elif file_name.endswith((".nii", ".nii.gz")):
        with st.spinner("Loading NIfTI volume..."):
            vol = load_nifti_volume(uploaded)
        if vol.ndim != 3:
            st.error(f"Expected a 3D volume, but got shape {vol.shape}.")
            st.stop()

        axis = 2
        with st.spinner("Running multi-slice inference (auto-selecting best slice)..."):
            best = infer_best_slice_for_segmentation(
                model=model,
                vol=vol,
                device=device_opt,
                axis=axis,
                num_candidates=32,
            )
        slice_idx = best["idx"]
        pred = best["pred"]
        img_np = best["model_img"]
        display_np = best["display_img"]
        source_note = (
            f"3D NIfTI mode | volume shape={vol.shape} | auto axis={axis} | best slice={slice_idx}"
        )
    else:
        st.error("Unsupported file type. Please upload `.png`, `.jpg`, `.nii`, or `.nii.gz`.")
        st.stop()

    if not file_name.endswith((".nii", ".nii.gz")):
        x = x.to(device_opt)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_mask = label_to_color(pred)
    if file_name.endswith((".nii", ".nii.gz")):
        show_img = display_np
    else:
        show_img = img_np
    overlay = blend_overlay(show_img, color_mask, alpha=alpha)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input Slice (normalized, resized)")
        st.image(show_img, clamp=True, use_container_width=True)
    with col2:
        st.subheader("Predicted Mask")
        st.image(color_mask, use_container_width=True)
    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)

    st.caption(source_note)

    st.markdown("### Class Legend")
    for cls_idx in range(NUM_CLASSES):
        c = CLASS_COLORS[cls_idx]
        st.markdown(
            f"- `{cls_idx}`: **{CLASS_NAMES[cls_idx]}** "
            f"(RGB: `{c[0]}, {c[1]}, {c[2]}`)"
        )

    present = np.unique(pred).tolist()
    present_names = ", ".join([f"{p} ({CLASS_NAMES[p]})" for p in present])
    if present == [0]:
        st.warning(
            "Prediction is background-only for this slice. This can happen on non-organ slices "
            "or out-of-distribution images. Try another CT image/volume."
        )
    else:
        st.success(f"Classes predicted in this slice: {present_names}")


if __name__ == "__main__":
    main()
