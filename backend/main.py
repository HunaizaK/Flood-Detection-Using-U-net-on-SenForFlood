from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from io import BytesIO
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import uvicorn

from segmentation_models_pytorch import Unet
from captum.attr import IntegratedGradients

import rasterio
from rasterio.io import MemoryFile
from scipy.stats import wasserstein_distance


# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Training distribution stats (hardcoded)
# -----------------------
TRAINING_STATS = {
    "S1_before_flood": {"mean": 1735.8619384765625, "std": 3036.968505859375},
    "S1_after_flood":  {"mean": 1803.71923828125,   "std": 3154.760009765625},
    "Terrain":         {"mean": 24.5716609954834,   "std": 53.64265823364258},
    "LULC":            {"mean": 40.98617935180664,  "std": 18.569223403930664},
}

Z_THRESH = 2.0
WD_THRESH = 1.0


def compute_stats(arr: np.ndarray):
    """Compute mean/std on raw values."""
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def drift_metrics(train_mean, train_std, inf_mean, inf_std):
    """Return z-score shifts for mean and std."""
    z_mean = abs(inf_mean - train_mean) / (train_std + 1e-6)
    z_std = abs(inf_std - train_std) / (train_std + 1e-6)
    return float(z_mean), float(z_std)


def load_raw_tiff_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Read TIFF from bytes as raw float32 (C, H, W)."""
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            arr = src.read()
    return arr.astype(np.float32)


def severity_from_metrics(z_m: float, z_s: float, wd: float) -> str:
    """
    Map metric magnitudes to severity levels:
      - none:   very small deviations (< 0.5 * threshold)
      - low:    mild deviations    (< 1.0 * threshold)
      - medium: clear drift        (< 2.0 * threshold)
      - high:   strong drift       (>= 2.0 * threshold)
    Uses the worst of {z_m, z_s, wd_norm}.
    """
    # Normalize metrics against their thresholds
    z_m_norm = z_m / Z_THRESH if Z_THRESH > 0 else 0.0
    z_s_norm = z_s / Z_THRESH if Z_THRESH > 0 else 0.0
    wd_norm = wd / WD_THRESH if WD_THRESH > 0 else 0.0

    max_norm = max(z_m_norm, z_s_norm, wd_norm)

    if max_norm < 0.5:
        return "none"
    elif max_norm < 1.0:
        return "low"
    elif max_norm < 2.0:
        return "medium"
    else:
        return "high"


def compute_drift_from_raw(
    s1_before_raw: np.ndarray,
    s1_after_raw: np.ndarray,
    terrain_raw: np.ndarray,
    lulc_raw: np.ndarray,
):
    """
    Compute drift metrics for 4 groups:
    - S1_before_flood: channels [0:4]
    - S1_after_flood:  channels [4:8]
    - Terrain:         channels [8:10]
    - LULC:            channels [10:11]
    Assumes input shapes:
      s1_before_raw: (4, H, W)
      s1_after_raw:  (4, H, W)
      terrain_raw:   (2, H, W)
      lulc_raw:      (1, H, W)

    Returns:
      report: dict per group
      overall_alert: bool
      overall_severity: "none" | "low" | "medium" | "high"
      messages: list of human-readable alerts
    """
    x = np.concatenate([s1_before_raw, s1_after_raw, terrain_raw, lulc_raw], axis=0)
    # safety check
    if x.shape[0] != 11:
        raise ValueError(f"[DRIFT] Expected 11 channels in concatenated raw data, got {x.shape[0]}")

    groups = {
        "S1_before_flood": x[0:4],
        "S1_after_flood":  x[4:8],
        "Terrain":         x[8:10],
        "LULC":            x[10:11],
    }

    report = {}
    messages = []
    severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    overall_severity = "none"

    for key, arr in groups.items():
        inf_mean, inf_std = compute_stats(arr)
        tr_mean = TRAINING_STATS[key]["mean"]
        tr_std = TRAINING_STATS[key]["std"]

        z_m, z_s = drift_metrics(tr_mean, tr_std, inf_mean, inf_std)

        # Wasserstein distance vs a constant array at training mean (normalized)
        arr_norm = arr.flatten() / (tr_std + 1e-6)
        ref_norm = np.full(arr_norm.shape, tr_mean / (tr_std + 1e-6), dtype=np.float32)
        wd = float(wasserstein_distance(arr_norm, ref_norm))

        # Severity
        severity = severity_from_metrics(z_m, z_s, wd)
        drift_detected = severity in ("medium", "high")

        # Track overall severity
        if severity_order[severity] > severity_order[overall_severity]:
            overall_severity = severity

        # Human-readable message per group
        if severity == "none":
            msg = f"{key}: No meaningful drift detected. Input distribution closely matches training data."
        elif severity == "low":
            msg = (
                f"{key}: Mild distribution shift detected. Monitor this input, "
                f"but model predictions are likely still stable."
            )
        elif severity == "medium":
            msg = (
                f"{key}: Noticeable drift detected. Model predictions for this input "
                f"may be less reliable; consider reviewing data or retraining."
            )
        else:  # high
            msg = (
                f"{key}: Strong drift detected. Input distribution differs significantly "
                f"from training; downstream predictions may be unreliable."
            )

        messages.append(msg)

        report[key] = {
            "inference_mean": inf_mean,
            "inference_std": inf_std,
            "z_mean_shift": z_m,
            "z_std_shift": z_s,
            "wasserstein_distance": wd,
            "severity": severity,
            "drift_detected": drift_detected,
        }

    overall_alert = overall_severity in ("medium", "high")
    return report, overall_alert, overall_severity, messages


# -----------------------
# Model definition + load
# -----------------------
model = Unet(
    encoder_name="resnet50",
    encoder_weights=None,        # fine at inference; we load our own weights
    in_channels=11,
    classes=1,
    decoder_use_batchnorm=True,
    decoder_block_type="transpose",
).to(device)

state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()


# -----------------------
# XAI helpers
# -----------------------
def saliency_map(model: nn.Module, img: torch.Tensor) -> np.ndarray:
    """
    Gradient saliency on input.
    img: (1, C, H, W) on device
    returns: (H, W) numpy
    """
    model.zero_grad()
    img = img.clone().detach().to(device)
    img.requires_grad_(True)

    out = model(img)        
    out.sum().backward()

    sal = img.grad.abs().squeeze(0).detach().cpu()  
    sal = sal.sum(0).numpy()                        
    return sal


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        """
        img: (1, C, H, W) on device
        returns: (H, W) numpy
        """
        self.model.zero_grad()
        img = img.clone().detach().to(device)
        img.requires_grad_(True)

        out = self.model(img)      
        out.sum().backward()

        grads = self.gradients         
        acts = self.activations       

        w = grads.mean(dim=(2, 3), keepdim=True)       
        cam = (w * acts).sum(dim=1, keepdim=True)       
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=img.shape[2:],       
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()      
        return cam


# GradCAM target: last block of encoder.layer4
target_layer = model.encoder.layer4[-1]
gradcam = GradCAM(model, target_layer)


class SegmentationWrapper(nn.Module):
    """
    Wrap segmentation model to return a scalar per image by averaging
    the output mask, so IG has a single scalar target.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)           # (N, 1, H, W)
        out = out.mean(dim=(2, 3))         # (N, 1)
        return out


wrapped_model = SegmentationWrapper(model).to(device)
ig_explainer = IntegratedGradients(wrapped_model)


def integrated_gradients_map(img: torch.Tensor) -> np.ndarray:
    """
    Compute Integrated Gradients attribution map.
    img: (1, C, H, W) on device
    returns: (H, W) numpy
    """
    wrapped_model.zero_grad()
    baseline = torch.zeros_like(img).to(device)

    attr = ig_explainer.attribute(
        img,
        baselines=baseline,
        target=0,   # single "class"
        n_steps=5,
    )                          # (1, C, H, W)

    ig_map = attr.squeeze(0).abs().sum(0).detach().cpu().numpy()  # (H, W)
    return ig_map


# -----------------------
# Preprocessing (for model, not for drift)
# -----------------------
def preprocess_tiff(file_bytes: bytes, patch_size: int = 512) -> torch.Tensor:
    """
    Reads and preprocesses TIFF to match training preprocessing.
    Returns tensor of shape (C, patch_size, patch_size).
    """
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img_data = src.read()  # (C, H, W)

    img_data = img_data.astype(np.float32)

    # Per-channel z-score normalization
    for c in range(img_data.shape[0]):
        mean = np.nanmean(img_data[c])
        std = np.nanstd(img_data[c]) + 1e-6
        img_data[c] = (img_data[c] - mean) / std

    # Center crop to (patch_size, patch_size)
    _, H, W = img_data.shape
    if H < patch_size or W < patch_size:
        raise ValueError(
            f"Image smaller than patch size {patch_size}x{patch_size}: got {H}x{W}"
        )

    top = (H - patch_size) // 2
    left = (W - patch_size) // 2
    img_data = img_data[:, top:top + patch_size, left:left + patch_size]

    return torch.from_numpy(img_data)      # (C, patch_size, patch_size)


# -----------------------
# Endpoint
# -----------------------
@app.post("/predict")
async def predict(
    s1_before_flood: UploadFile = File(...),
    s1_after_flood: UploadFile = File(...),
    terrain: UploadFile = File(...),
    lulc: UploadFile = File(...),
):
    # type check
    if not (
        s1_before_flood.content_type == "image/tiff"
        and s1_after_flood.content_type == "image/tiff"
        and terrain.content_type == "image/tiff"
        and lulc.content_type == "image/tiff"
    ):
        raise HTTPException(status_code=400, detail="All files must be of type TIFF")

    try:
        patch_size = 512

        # Read bytes (once)
        s1_before_bytes = await s1_before_flood.read()
        s1_after_bytes = await s1_after_flood.read()
        terrain_bytes = await terrain.read()
        lulc_bytes = await lulc.read()

        # -------- Data Drift (use RAW TIFFs, no normalization) --------
        s1_before_raw = load_raw_tiff_from_bytes(s1_before_bytes)  # expect (4, H, W)
        s1_after_raw = load_raw_tiff_from_bytes(s1_after_bytes)    # expect (4, H, W)
        terrain_raw = load_raw_tiff_from_bytes(terrain_bytes)      # expect (2, H, W)
        lulc_raw = load_raw_tiff_from_bytes(lulc_bytes)            # expect (1, H, W)

        drift_report, drift_alert, drift_overall_severity, drift_messages = compute_drift_from_raw(
            s1_before_raw, s1_after_raw, terrain_raw, lulc_raw
        )

        # -------- Preprocess for model (normalized + cropped) --------
        s1_before = preprocess_tiff(s1_before_bytes, patch_size)
        s1_after = preprocess_tiff(s1_after_bytes, patch_size)
        terrain_img = preprocess_tiff(terrain_bytes, patch_size)
        lulc_img = preprocess_tiff(lulc_bytes, patch_size)

        # Numpy copies (normalized, cropped) for visualization
        s1_before_np = s1_before.numpy()
        s1_after_np = s1_after.numpy()
        terrain_np = terrain_img.numpy()
        lulc_np = lulc_img.numpy()

        # Stack channels for model input: (C_total, H, W)
        stacked = torch.cat([s1_before, s1_after, terrain_img, lulc_img], dim=0)

        if stacked.shape[0] != 11:
            raise ValueError(f"Expected 11 channels, got {stacked.shape[0]}")

        # Add batch: (1, 11, H, W)
        input_tensor = stacked.unsqueeze(0).to(device)

        # -------- Model prediction --------
        with torch.no_grad():
            logits = model(input_tensor)        # (1, 1, H, W)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)

        flood_mask = (probs > 0.5).astype(np.uint8) * 255      # (H, W), uint8

        # -------- XAI --------
        sal_map = saliency_map(model, input_tensor)            # (H, W)
        cam_map = gradcam(input_tensor)                        # (H, W)
        ig_map = integrated_gradients_map(input_tensor)        # (H, W)

        # For XAI background, use first channel of s1_before (normalized, cropped)
        orig_image = s1_before_np[0]                           # (H, W)

        # -------- PNG #1: inputs + prediction --------
        fig1, axs1 = plt.subplots(1, 5, figsize=(25, 5))

        axs1[0].imshow(s1_before_np[0], cmap="gray")
        axs1[0].set_title("S1 Before Flood (Ch 0)")
        axs1[0].axis("off")

        axs1[1].imshow(s1_after_np[0], cmap="gray")
        axs1[1].set_title("S1 During/After Flood (Ch 0)")
        axs1[1].axis("off")

        axs1[2].imshow(terrain_np[0], cmap="terrain")
        axs1[2].set_title("Terrain")
        axs1[2].axis("off")

        axs1[3].imshow(lulc_np[0], cmap="tab20")
        axs1[3].set_title("LULC")
        axs1[3].axis("off")

        axs1[4].imshow(flood_mask, cmap="Reds")
        axs1[4].set_title("Predicted Flood Map")
        axs1[4].axis("off")

        plt.tight_layout()

        buf_pred = BytesIO()
        fig1.savefig(buf_pred, format="PNG", bbox_inches="tight")
        buf_pred.seek(0)
        plt.close(fig1)

        # -------- PNG #2: XAI (orig + 3 maps) --------
        fig2, axs2 = plt.subplots(1, 4, figsize=(20, 5))

        axs2[0].imshow(orig_image, cmap="gray")
        axs2[0].set_title("Original (Before Flood Ch 0)")
        axs2[0].axis("off")

        axs2[1].imshow(orig_image, cmap="gray")
        axs2[1].imshow(sal_map, cmap="hot", alpha=0.5)
        axs2[1].set_title("Saliency (Grad)")
        axs2[1].axis("off")

        axs2[2].imshow(orig_image, cmap="gray")
        axs2[2].imshow(cam_map, cmap="hot", alpha=0.5)
        axs2[2].set_title("GradCAM")
        axs2[2].axis("off")

        axs2[3].imshow(orig_image, cmap="gray")
        axs2[3].imshow(ig_map, cmap="hot", alpha=0.5)
        axs2[3].set_title("Integrated Gradients")
        axs2[3].axis("off")

        plt.tight_layout()

        buf_xai = BytesIO()
        fig2.savefig(buf_xai, format="PNG", bbox_inches="tight")
        buf_xai.seek(0)
        plt.close(fig2)

        # -------- Encode both PNGs into base64 and return dict --------
        inputs_pred_b64 = base64.b64encode(buf_pred.getvalue()).decode("utf-8")
        xai_b64 = base64.b64encode(buf_xai.getvalue()).decode("utf-8")

        result = {
            "inputs_and_prediction": inputs_pred_b64,
            "xai_maps": xai_b64,
            "drift_report": drift_report,                # per-group metrics & severity
            "drift_alert": drift_alert,                  # True if severity >= medium
            "drift_overall_severity": drift_overall_severity,  # none/low/medium/high
            "drift_messages": drift_messages,            # clean English sentences
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}"},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
