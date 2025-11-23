from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import io

# Import the trained U-Net model
from segmentation_models_pytorch import Unet

app = FastAPI()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your U-Net model (use the same architecture as the one used for training)
# Model with pre-trained weights, using default decoder channels
model = Unet(
    encoder_name="resnet50",  # Using ResNet50 as the encoder
    encoder_weights="imagenet",  # Load pre-trained weights from ImageNet
    in_channels=11,
    classes=1,  # For binary segmentation (1 class output)
    decoder_use_batchnorm=True,  # Use batch normalization in the decoder
    decoder_block_type='transpose',  # Using transposed convolution in the decoder
  
).to(device)

# Load the model weights (model.pth should be saved in the same directory or provide the full path)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Preprocessing pipeline (matching with your training code)
def preprocess_image(image: Image.Image):
    # Resize the image to 256x256
    image = image.resize((256, 256))

    # Convert to numpy array
    image_np = np.array(image).astype(np.float32)

    # Normalize the image (same as in your training code)
    for c in range(image_np.shape[2]):  # Assume the image has 3 channels
        mean, std = np.nanmean(image_np[:, :, c]), np.nanstd(image_np[:, :, c]) + 1e-6
        image_np[:, :, c] = (image_np[:, :, c] - mean) / std

    # Convert to PyTorch tensor (C, H, W)
    image_tensor = torch.tensor(image_np.transpose(2, 0, 1))  # Convert HWC to CHW
    return image_tensor

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image data from the file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))

        # Preprocess the image
        input_image = preprocess_image(image).unsqueeze(0).to(device)  # Add batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(input_image)
            pred = torch.sigmoid(output).cpu().numpy()[0][0]  # Get the prediction for the first channel

        # Convert prediction to binary (flooded area or not)
        flood_mask = (pred > 0.5).astype(np.uint8)

        # Convert the flood mask back to an image to send to frontend (as base64 or simple image)
        mask_image = Image.fromarray(flood_mask * 255)  # Convert to an image (0 or 255 for flood or no-flood)
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Return the mask as a response (could be base64 encoded or simply image data)
        return JSONResponse(content={"flood_mask": buffer.getvalue().decode("latin1")})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing image: {str(e)}"})

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn (start the backend server)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
