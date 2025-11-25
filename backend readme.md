# Flood Detection Using U-Net on SenForFlood

A comprehensive flood detection system leveraging deep learning (U-Net architecture) with Explainable AI (XAI) capabilities for satellite imagery analysis. This project consists of a FastAPI backend for model inference and a React frontend for user interaction.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Backend](#backend)
- [Frontend](#frontend)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)

## 🌟 Overview

This application performs flood detection using a U-Net segmentation model trained on the SenForFlood dataset. It processes Sentinel-1 SAR imagery along with terrain and land use/land cover (LULC) data to predict flood-affected areas. The system includes advanced Explainable AI (XAI) techniques to provide interpretable results.

## 🏗️ Architecture

The project follows a client-server architecture:

- **Backend**: FastAPI-based REST API for model inference and XAI computations
- **Frontend**: React + Vite application with styled-components for UI
- **Model**: U-Net with ResNet50 encoder for semantic segmentation
- **XAI Methods**: Saliency Maps, GradCAM, and Integrated Gradients

## ✨ Features

### Core Functionality
- **Multi-modal Input Processing**: Handles 11-channel input (S1 before, S1 during, terrain, LULC)
- **Deep Learning Inference**: U-Net model with ResNet50 backbone for flood segmentation
- **Real-time Predictions**: Fast API endpoint for immediate flood mask generation
- **Explainable AI**: Three XAI techniques for model interpretability

### XAI Techniques
1. **Saliency Maps**: Gradient-based visualization of input importance
2. **GradCAM**: Class Activation Mapping for spatial feature importance
3. **Integrated Gradients**: Attribution method for robust feature importance

### User Interface
- Modern, responsive React interface
- Drag-and-drop file upload for TIFF images
- Real-time visualization of predictions
- Separate views for predictions and XAI explanations

## 📁 Project Structure

```
Flood-Detection-Using-U-net-on-SenForFlood/
├── backend/
│   ├── main.py              # FastAPI application with inference endpoint
│   ├── xai_inference.py     # XAI experimentation notebook/script
│   └── model.pth            # Trained U-Net model weights (not in repo)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── landingpage.jsx    # Main application interface
│   │   │   └── dashboard.jsx       # Alternative dashboard view
│   │   ├── App.jsx                 # Root application component
│   │   ├── main.jsx                # React entry point
│   │   └── assets/                 # Static assets
│   ├── public/                     # Public assets
│   ├── package.json                # Frontend dependencies
│   ├── vite.config.js              # Vite configuration
│   ├── eslint.config.js            # ESLint configuration
│   └── index.html                  # HTML entry point
└── README.md                       # This file
```

## 🔧 Backend

### Overview
The backend is built with **FastAPI** and provides a RESTful API for flood detection inference with integrated XAI capabilities.

### Key Components

#### `main.py`
The main FastAPI application containing:

**Model Architecture:**
- U-Net with ResNet50 encoder
- 11 input channels (S1 VV/VH before, S1 VV/VH during, terrain, LULC)
- Single output channel for binary flood segmentation
- Trained weights loaded from `model.pth`

**XAI Implementations:**
1. **Saliency Map (`saliency_map`)**: 
   - Computes gradient-based attribution
   - Returns (H, W) numpy array showing input importance
   - Uses absolute gradients summed across channels

2. **GradCAM (`GradCAM` class)**:
   - Implements Gradient-weighted Class Activation Mapping
   - Targets `encoder.layer4[-1]` (last ResNet50 block)
   - Provides spatial heatmap of model attention
   - Upsamples to input resolution

3. **Integrated Gradients (`integrated_gradients_map`)**:
   - Uses Captum library implementation
   - Computes attribution via path integral from baseline
   - 5-step approximation for efficiency
   - Returns aggregated channel importance

**Preprocessing (`preprocess_tiff`)**:
- Reads TIFF files using `rasterio`
- Per-channel z-score normalization
- Center crop to 512×512 patches
- Returns PyTorch tensor ready for inference

**API Endpoint:**
```
POST /predict
```
- **Inputs**: 4 TIFF files (multipart/form-data)
  - `s1_before_flood`: Sentinel-1 before flood event
  - `s1_after_flood`: Sentinel-1 during/after flood
  - `terrain`: Digital Elevation Model (DEM)
  - `lulc`: Land Use/Land Cover classification
- **Outputs**: JSON with two base64-encoded PNG images
  - `inputs_and_prediction`: Visualization of inputs + flood mask
  - `xai_maps`: Three XAI heatmaps overlaid on original image

**Image Generation:**
- Uses matplotlib for visualization
- Creates two comprehensive PNG outputs
- Base64 encoding for easy frontend consumption
- First PNG: 5-panel view (4 inputs + prediction)
- Second PNG: 4-panel XAI view (original + 3 attribution maps)

#### `xai_inference.py`
Experimental script/notebook for XAI development:
- Google Colab-based development environment
- Similar implementations of saliency, GradCAM, and IG
- Used for prototyping before production integration
- Includes visualization and testing code

### Technical Requirements

**Python Version**: 3.8+

**Key Dependencies**:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `torch`: Deep learning framework
- `segmentation-models-pytorch`: Pre-built U-Net implementation
- `captum`: Facebook's XAI library for Integrated Gradients
- `rasterio`: Geospatial raster I/O
- `matplotlib`: Visualization
- `numpy`: Numerical operations

### Running the Backend

```bash
cd backend
python main.py
```

Server starts at: `http://127.0.0.1:8000`

API documentation available at: `http://127.0.0.1:8000/docs`

## 💻 Frontend

### Overview
Modern React-based web application built with Vite for fast development and optimized production builds.

### Key Components

#### `App.jsx`
Root component implementing routing:
- Uses React Router for navigation
- Styled-components for CSS-in-JS
- Container with scroll management
- Single route to `LandingPage` component

#### `landingpage.jsx`
Main application interface with comprehensive styling:

**Layout Features:**
- Radial gradient dark theme
- Card-based design with glassmorphism effects
- Responsive grid layout for file uploads
- Professional color scheme (dark blues, teals, greens)

**Upload Interface:**
- 4 upload boxes for TIFF files:
  1. S1 Before Flood (VV/VH backscatter)
  2. S1 During Flood (flood-time acquisition)
  3. Terrain (DEM/elevation)
  4. LULC (land use/land cover)
- Visual feedback on file selection
- Styled upload zones with hover effects

**Inference Workflow:**
1. User uploads 4 required TIFF files
2. Clicks "Run Inference" button
3. Frontend sends multipart form data to backend
4. Displays loading state during processing
5. Renders two result images upon completion

**Results Display:**
- **Inputs & Predicted Flood Map**: 5-panel visualization
- **XAI Maps**: Saliency, GradCAM, and Integrated Gradients
- Base64 PNG images decoded and displayed inline
- Styled result sections with tags and borders

**Error Handling:**
- Validation for all 4 required files
- User-friendly error messages
- Loading states and status updates
- Backend error capture and display

#### `dashboard.jsx`
Alternative dashboard implementation:
- Simpler UI with basic styling
- Similar functionality to landing page
- May be used for different workflow or testing

### Technical Stack

**Core Framework:**
- React 19.2.0
- React Router DOM 7.9.6
- Vite 7.2.4 (build tool)

**Styling:**
- styled-components 6.1.19
- CSS-in-JS approach
- Theme: Dark mode with gradient backgrounds

**HTTP Client:**
- Axios 1.13.2 for API communication
- Multipart form data support

**Icons:**
- lucide-react 0.554.0

**Development Tools:**
- ESLint for code quality
- Vite plugin for React
- TypeScript type definitions

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Development server starts at: `http://localhost:5173` (default Vite port)

**Build for Production:**
```bash
npm run build
npm run preview
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ and npm
- CUDA-capable GPU (optional, for faster inference)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install fastapi uvicorn torch torchvision segmentation-models-pytorch captum rasterio matplotlib numpy

# Place your trained model file
# Ensure model.pth is in the backend directory

# Start the server
python main.py
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 📖 Usage

1. **Start Backend Server**:
   ```bash
   cd backend
   python main.py
   ```
   Backend runs on `http://127.0.0.1:8000`

2. **Start Frontend Application**:
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend runs on `http://localhost:5173`

3. **Upload Data**:
   - Open the frontend in your browser
   - Upload 4 TIFF files:
     - Sentinel-1 before flood
     - Sentinel-1 during/after flood
     - Terrain/DEM
     - LULC classification

4. **Run Inference**:
   - Click "Run Inference" button
   - Wait for processing (typically 5-15 seconds)
   - View results:
     - Input visualizations + predicted flood mask
     - XAI explanation maps

5. **Interpret Results**:
   - **Flood Mask**: Red areas indicate predicted flooding
   - **Saliency Map**: Highlights input pixels contributing to predictions
   - **GradCAM**: Shows spatial regions model focuses on
   - **Integrated Gradients**: Attribution scores for feature importance

## 🔬 Technical Details

### Model Specifications

**Architecture**: U-Net with ResNet50 Encoder
- **Input**: 11 channels (512×512)
  - 2 channels: S1 VV/VH before flood
  - 2 channels: S1 VV/VH during flood
  - 1 channel: Terrain/elevation
  - 1 channel: LULC classification
  - (Note: actual channel count may vary based on data)
- **Output**: 1 channel (512×512) - binary flood mask
- **Activation**: Sigmoid for probability output
- **Threshold**: 0.5 for binary classification

### Data Processing Pipeline

1. **Input**: GeoTIFF files from Sentinel-1 and ancillary data
2. **Normalization**: Per-channel z-score normalization
3. **Spatial Processing**: Center crop to 512×512
4. **Inference**: Forward pass through U-Net
5. **Post-processing**: Sigmoid activation + thresholding
6. **Visualization**: Matplotlib-generated PNG outputs

### XAI Methods Explained

1. **Saliency Maps**:
   - Gradient of output w.r.t. input
   - Shows which pixels influence prediction most
   - Fast to compute, pixel-level granularity

2. **GradCAM**:
   - Weighted combination of feature maps
   - Provides spatial localization of important regions
   - Works on deep layer activations (ResNet layer4)

3. **Integrated Gradients**:
   - Path-based attribution method
   - Theoretically grounded (satisfies axioms)
   - Computes attribution via baseline integration

## 📦 Dependencies

### Backend Dependencies

```
fastapi>=0.100.0
uvicorn>=0.23.0
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.0
captum>=0.6.0
rasterio>=1.3.0
matplotlib>=3.7.0
numpy>=1.24.0
python-multipart>=0.0.6
```

### Frontend Dependencies

```json
{
  "dependencies": {
    "axios": "^1.13.2",
    "lucide-react": "^0.554.0",
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "react-router-dom": "^7.9.6",
    "styled-components": "^6.1.19"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^5.1.1",
    "eslint": "^9.39.1",
    "vite": "^7.2.4"
  }
}
```

## 🔍 API Reference

### POST /predict

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `s1_before_flood`: TIFF file
  - `s1_after_flood`: TIFF file
  - `terrain`: TIFF file
  - `lulc`: TIFF file

**Response:**
```json
{
  "inputs_and_prediction": "base64_encoded_png_string",
  "xai_maps": "base64_encoded_png_string"
}
```

**Error Response:**
```json
{
  "message": "Error message description"
}
```

## 🛠️ Configuration

### Backend Configuration
- **Host**: `127.0.0.1`
- **Port**: `8000`
- **CORS**: Enabled for all origins (adjust for production)
- **Device**: Auto-detects CUDA GPU or falls back to CPU

### Frontend Configuration
- **API Endpoint**: `http://127.0.0.1:8000/predict`
- **File Types**: `.tif`, `.tiff`
- **Build Tool**: Vite with React plugin

## 📝 Notes

- Ensure `model.pth` is present in the backend directory before running
- TIFF files must match the expected spatial resolution and band configuration
- For production deployment, update CORS settings in `main.py`
- GPU acceleration significantly improves inference speed
- Frontend expects exact field names in FormData

## 🤝 Contributing

This project is part of research on flood detection using deep learning and XAI techniques. Contributions, issues, and feature requests are welcome.

## 📄 License

Please refer to the repository license for usage terms.

## 🙏 Acknowledgments

- SenForFlood Dataset
- Segmentation Models PyTorch library
- Captum (Facebook AI Research) for XAI implementations
- FastAPI and React communities

---

**Repository**: [Flood-Detection-Using-U-net-on-SenForFlood](https://github.com/HunaizaK/Flood-Detection-Using-U-net-on-SenForFlood)  
**Branch**: hunaiza
