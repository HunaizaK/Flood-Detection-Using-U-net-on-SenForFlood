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

    out = model(img)        # (1, 1, H, W)
    out.sum().backward()

    sal = img.grad.abs().squeeze(0).detach().cpu()  # (C, H, W)
    sal = sal.sum(0).numpy()                        # (H, W)
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

        out = self.model(img)      # (1, 1, H, W)
        out.sum().backward()

        grads = self.gradients         # (1, C_l, H_l, W_l)
        acts = self.activations        # (1, C_l, H_l, W_l)

        w = grads.mean(dim=(2, 3), keepdim=True)        # (1, C_l, 1, 1)
        cam = (w * acts).sum(dim=1, keepdim=True)       # (1, 1, H_l, W_l)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=img.shape[2:],        # upsample to input size
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()      # (H, W)
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
# Preprocessing
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

        # Read bytes
        s1_before_bytes = await s1_before_flood.read()
        s1_after_bytes = await s1_after_flood.read()
        terrain_bytes = await terrain.read()
        lulc_bytes = await lulc.read()

        # Preprocess each -> (C, H, W)
        s1_before = preprocess_tiff(s1_before_bytes, patch_size)
        s1_after = preprocess_tiff(s1_after_bytes, patch_size)
        terrain_img = preprocess_tiff(terrain_bytes, patch_size)
        lulc_img = preprocess_tiff(lulc_bytes, patch_size)

        # Numpy copies (normalized) for visualization
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

        # For XAI background, use first channel of s1_before
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
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}"},
        )


if __name__ == "__main__":
 
    uvicorn.run(app, host="127.0.0.1", port=8000)
