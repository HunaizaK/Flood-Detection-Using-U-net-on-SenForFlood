from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segmentation_models_pytorch import Unet
from fastapi.responses import JSONResponse
import rasterio
from rasterio.io import MemoryFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the React app's URL, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load model (must match training exactly) ----
model = Unet(
    encoder_name="resnet50",
    encoder_weights=None,        # IMPORTANT: weights come from your .pth, don't re-load ImageNet here
    in_channels=11,  # Your model expects 11 channels
    classes=1,
    decoder_use_batchnorm=True,
    decoder_block_type="transpose",
).to(device)

state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ---- Preprocessing function ----
def preprocess_tiff(file_bytes: bytes, patch_size=512):
    """
    Reads and preprocesses the uploaded TIFF image files to match the training preprocessing pipeline.
    """
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img_data = src.read()  # (C, H, W)
    
    # Convert to numpy array
    img_data = img_data.astype(np.float32)
    
    # Normalize the image per channel
    for c in range(img_data.shape[0]):
        mean, std = np.nanmean(img_data[c]), np.nanstd(img_data[c]) + 1e-6
        img_data[c] = (img_data[c] - mean) / std
    
    # Center crop to the patch size (512x512)
    _, H, W = img_data.shape
    top = (H - patch_size) // 2
    left = (W - patch_size) // 2
    img_data = img_data[:, top:top + patch_size, left:left + patch_size]  # (C, patch_size, patch_size)
    
    return torch.from_numpy(img_data)  # Shape: (C, H, W)


@app.post("/predict")
async def predict(
    s1_before_flood: UploadFile = File(...),
    s1_after_flood: UploadFile = File(...),
    terrain: UploadFile = File(...),
    lulc: UploadFile = File(...),
):
    # Validate that the uploaded files are TIFF
    if not (s1_before_flood.content_type == 'image/tiff' and
            s1_after_flood.content_type == 'image/tiff' and
            terrain.content_type == 'image/tiff' and
            lulc.content_type == 'image/tiff'):
        raise HTTPException(status_code=400, detail="All files must be of type TIFF")

    try:
        patch_size = 512  # Set patch size to 512 for consistency
        
        # Read each of the uploaded files and preprocess them
        s1_before_bytes = await s1_before_flood.read()
        s1_after_bytes = await s1_after_flood.read()
        terrain_bytes = await terrain.read()
        lulc_bytes = await lulc.read()

        # Preprocess the images (resize, normalize, crop)
        s1_before = preprocess_tiff(s1_before_bytes, patch_size)
        s1_after = preprocess_tiff(s1_after_bytes, patch_size)
        terrain_img = preprocess_tiff(terrain_bytes, patch_size)
        lulc_img = preprocess_tiff(lulc_bytes, patch_size)

        # Stack the 4 modalities into a single tensor (shape: (4, C, H, W))
        input_tensor = torch.cat([s1_before, s1_after, terrain_img, lulc_img], dim=0).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)  # output shape: (1, 1, patch_size, patch_size)
            probs = torch.sigmoid(logits)[0, 0]  # (patch_size, patch_size)

        # Convert prediction to binary (flooded area or not)
        flood_mask = (probs > 0.5).cpu().numpy()  # Convert to NumPy array
        flood_mask = (flood_mask * 255).astype(np.uint8)  # Convert to 0 or 255 (uint8)

        # ---- Visualization ----
        # Convert to numpy for visualization
        x_np = s1_before.numpy()

        # Create a plot with 2 subplots (no ground truth)
        plt.figure(figsize=(15,5))

        # S1 Before Flood (VV)
        plt.subplot(1, 2, 1)
        plt.title("S1 Before Flood - VV (Channel 0)")
        plt.imshow(x_np[0], cmap="gray")

        # Predicted Flood Map
        plt.subplot(1, 2, 2)
        plt.title("Predicted Flood Map")
        plt.imshow(flood_mask, cmap="Reds")

        # Save the visualization as PNG image
        buffer = BytesIO()
        plt.savefig(buffer, format="PNG")
        buffer.seek(0)
        plt.close()

        # Return the PNG image as response
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
