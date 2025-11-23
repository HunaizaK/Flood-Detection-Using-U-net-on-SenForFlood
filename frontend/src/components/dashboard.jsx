import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const Container = styled.div`
  font-family: 'Arial', sans-serif;
  background-color: #f5f5f5;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
`;

const Title = styled.h1`
  color: #2f4f4f;
  margin-bottom: 30px;
`;

const Input = styled.input`
  margin: 10px 0;
`;

const Button = styled.button`
  padding: 12px 24px;
  background-color: #2f4f4f;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  margin-top: 20px;

  &:hover {
    background-color: #4f6d6d;
  }
`;

const LoadingMessage = styled.p`
  font-size: 18px;
  color: #4f6d6d;
`;

const ImageContainer = styled.div`
  margin-top: 20px;
`;

const Image = styled.img`
  max-width: 500px;
  margin-top: 20px;
  border: 2px solid #ccc;
  border-radius: 8px;
`;

const App = () => {
  const [s1BeforeFloodFile, setS1BeforeFloodFile] = useState(null);
  const [s1AfterFloodFile, setS1AfterFloodFile] = useState(null);
  const [terrainFile, setTerrainFile] = useState(null);
  const [lulcFile, setLulcFile] = useState(null);
  const [floodMask, setFloodMask] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle form submission
  const handleSubmit = async () => {
    // Check if all files are uploaded
    if (!s1BeforeFloodFile || !s1AfterFloodFile || !terrainFile || !lulcFile) {
      alert('Please upload all 4 required images: S1 Before Flood, S1 After Flood, Terrain, and LULC.');
      return;
    }

    const formData = new FormData();
    formData.append('s1_before_flood', s1BeforeFloodFile);
    formData.append('s1_after_flood', s1AfterFloodFile);
    formData.append('terrain', terrainFile);
    formData.append('lulc', lulcFile);

    try {
      setLoading(true);
      setError(null);

      // Send the form data to the FastAPI backend
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Assuming the backend returns the flood mask as a URL or base64-encoded string
      setFloodMask(response.data.flood_mask);
      setLoading(false);
    } catch (err) {
      console.error('Error during prediction', err);
      setError('Error occurred while processing the files.');
      setLoading(false);
    }
  };

  return (
    <Container>
      <Title>Flood Prediction Using U-Net</Title>

      {/* S1 Before Flood */}
      <label>S1 Before Flood:</label>
      <Input
        type="file"
        accept=".tif,.tiff"
        onChange={(e) => setS1BeforeFloodFile(e.target.files[0])}
      />
      {s1BeforeFloodFile && <p>{s1BeforeFloodFile.name}</p>}

      {/* S1 After Flood */}
      <label>S1 After Flood:</label>
      <Input
        type="file"
        accept=".tif,.tiff"
        onChange={(e) => setS1AfterFloodFile(e.target.files[0])}
      />
      {s1AfterFloodFile && <p>{s1AfterFloodFile.name}</p>}

      {/* Terrain */}
      <label>Terrain:</label>
      <Input
        type="file"
        accept=".tif,.tiff"
        onChange={(e) => setTerrainFile(e.target.files[0])}
      />
      {terrainFile && <p>{terrainFile.name}</p>}

      {/* LULC */}
      <label>LULC:</label>
      <Input
        type="file"
        accept=".tif,.tiff"
        onChange={(e) => setLulcFile(e.target.files[0])}
      />
      {lulcFile && <p>{lulcFile.name}</p>}

      {/* Submit Button */}
      <Button onClick={handleSubmit}>Submit All</Button>

      {loading && <LoadingMessage>Processing files...</LoadingMessage>}
      {error && <LoadingMessage>{error}</LoadingMessage>}

      {/* Show Uploaded Images */}
      {s1BeforeFloodFile && (
        <ImageContainer>
          <h3>S1 Before Flood - VV (Channel 0):</h3>
          <Image src={URL.createObjectURL(s1BeforeFloodFile)} alt="S1 Before Flood" />
        </ImageContainer>
      )}

      {s1AfterFloodFile && (
        <ImageContainer>
          <h3>S1 After Flood - VV (Channel 0):</h3>
          <Image src={URL.createObjectURL(s1AfterFloodFile)} alt="S1 After Flood" />
        </ImageContainer>
      )}

      {terrainFile && (
        <ImageContainer>
          <h3>Terrain:</h3>
          <Image src={URL.createObjectURL(terrainFile)} alt="Terrain" />
        </ImageContainer>
      )}

      {lulcFile && (
        <ImageContainer>
          <h3>LULC:</h3>
          <Image src={URL.createObjectURL(lulcFile)} alt="LULC" />
        </ImageContainer>
      )}

      {/* Show Predicted Flood Mask */}
      {floodMask && (
        <ImageContainer>
          <h3>Predicted Flood Mask:</h3>
          <Image src={floodMask} alt="Predicted Flood Mask" />
        </ImageContainer>
      )}
    </Container>
  );
};

export default App;
