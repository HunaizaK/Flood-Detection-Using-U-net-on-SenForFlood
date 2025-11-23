import React, { useState } from "react";
import styled from "styled-components";
import axios from "axios";

// ---------- styled components ----------

const PageWrapper = styled.div`
  min-height: 100vh;
  background: radial-gradient(circle at top, #1f2933 0, #0b1120 45%, #020617 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  color: #e5e7eb;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
`;

const Card = styled.div`
  width: 100%;
  max-width: 1100px;
  background: rgba(15, 23, 42, 0.95);
  border-radius: 24px;
  padding: 28px 32px 32px;
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.6),
    0 0 0 1px rgba(148, 163, 184, 0.15);
  backdrop-filter: blur(16px);
`;

const Header = styled.div`
  margin-bottom: 24px;
`;

const Title = styled.h1`
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0.03em;
  color: #f9fafb;
  margin-bottom: 6px;
`;

const Subtitle = styled.p`
  font-size: 14px;
  color: #9ca3af;
`;

const UploadGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 18px;
  margin-top: 20px;
`;

const UploadBox = styled.label`
  border-radius: 18px;
  border: 1px dashed rgba(148, 163, 184, 0.55);
  padding: 16px 14px;
  background: radial-gradient(circle at top left, #111827 0, #020617 60%);
  cursor: pointer;
  transition: all 0.18s ease;
  display: flex;
  flex-direction: column;
  gap: 10px;

  &:hover {
    border-color: #38bdf8;
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.5);
    transform: translateY(-1px);
  }
`;

const UploadTitle = styled.div`
  font-size: 14px;
  font-weight: 600;
  color: #e5e7eb;
`;

const UploadHint = styled.div`
  font-size: 12px;
  color: #9ca3af;
`;

const HiddenInput = styled.input`
  display: none;
`;

const FileName = styled.div`
  font-size: 12px;
  color: #a5b4fc;
  margin-top: 4px;
  word-break: break-all;
`;

const ActionsRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 22px;
  flex-wrap: wrap;
`;

const RunButton = styled.button`
  border: none;
  outline: none;
  padding: 10px 20px;
  border-radius: 999px;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  cursor: pointer;
  background: linear-gradient(120deg, #22c55e, #16a34a);
  color: #f9fafb;
  box-shadow:
    0 12px 25px rgba(22, 163, 74, 0.5),
    0 0 0 1px rgba(34, 197, 94, 0.4);
  transition: all 0.16s ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow:
      0 18px 35px rgba(22, 163, 74, 0.55),
      0 0 0 1px rgba(34, 197, 94, 0.5);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const StatusText = styled.div`
  font-size: 13px;
  color: ${({ error }) => (error ? "#fca5a5" : "#9ca3af")};
`;

const ResultSection = styled.div`
  margin-top: 26px;
  border-radius: 18px;
  background: radial-gradient(circle at top, #020617 0, #020617 65%);
  border: 1px solid rgba(148, 163, 184, 0.35);
  padding: 14px 14px 18px;
`;

const ResultTitleRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
`;

const ResultTitle = styled.div`
  font-size: 14px;
  font-weight: 600;
  color: #e5e7eb;
`;

const ResultTag = styled.span`
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(56, 189, 248, 0.12);
  color: #7dd3fc;
  border: 1px solid rgba(56, 189, 248, 0.35);
`;

const ResultImageWrapper = styled.div`
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(55, 65, 81, 0.9);
  background: #020617;
  max-height: 480px;
  display: flex;
  justify-content: center;
`;

const ResultImage = styled.img`
  width: 100%;
  object-fit: contain;
`;

// ---------- component ----------

const FloodDetectionPage = () => {
  const [s1BeforeFloodFile, setS1BeforeFloodFile] = useState(null);
  const [s1AfterFloodFile, setS1AfterFloodFile] = useState(null);
  const [terrainFile, setTerrainFile] = useState(null);
  const [lulcFile, setLulcFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [resultImageUrl, setResultImageUrl] = useState(null);

  const handleSubmit = async () => {
    if (!s1BeforeFloodFile || !s1AfterFloodFile || !terrainFile || !lulcFile) {
      setError("Please upload all 4 required TIFF images before running inference.");
      return;
    }

    setError("");
    setStatus("Uploading and running inference...");
    setLoading(true);

    const formData = new FormData();
    // field names MUST match FastAPI: s1_before_flood, s1_after_flood, terrain, lulc
    formData.append("s1_before_flood", s1BeforeFloodFile);
    formData.append("s1_after_flood", s1AfterFloodFile);
    formData.append("terrain", terrainFile);
    formData.append("lulc", lulcFile);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Accept: "image/png",
        },
        responseType: "blob",
      });

      // revoke old URL if any
      if (resultImageUrl) {
        URL.revokeObjectURL(resultImageUrl);
      }

      const imageUrl = URL.createObjectURL(response.data);
      setResultImageUrl(imageUrl);
      setStatus("Inference completed.");
    } catch (err) {
      console.error(err);
      setError("Error during prediction. Check backend logs.");
      setStatus("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <Card>
        <Header>
          <Title>Flood Detection Using U-Net</Title>
          <Subtitle>
            Upload Sentinel-1 (before & during flood), terrain, and LULC tiles to generate a flood prediction mask.
          </Subtitle>
        </Header>

        <UploadGrid>
          <UploadBox>
            <UploadTitle>S1 Before Flood</UploadTitle>
            <UploadHint>.tif / .tiff • VV/VH backscatter</UploadHint>
            <HiddenInput
              type="file"
              accept=".tif,.tiff"
              onChange={(e) => setS1BeforeFloodFile(e.target.files[0] || null)}
            />
            <div style={{ fontSize: 12, marginTop: 6, opacity: 0.85 }}>
              Click to select file
            </div>
            {s1BeforeFloodFile && (
              <FileName>Selected: {s1BeforeFloodFile.name}</FileName>
            )}
          </UploadBox>

          <UploadBox>
            <UploadTitle>S1 During Flood</UploadTitle>
            <UploadHint>.tif / .tiff • Flood-time acquisition</UploadHint>
            <HiddenInput
              type="file"
              accept=".tif,.tiff"
              onChange={(e) => setS1AfterFloodFile(e.target.files[0] || null)}
            />
            <div style={{ fontSize: 12, marginTop: 6, opacity: 0.85 }}>
              Click to select file
            </div>
            {s1AfterFloodFile && (
              <FileName>Selected: {s1AfterFloodFile.name}</FileName>
            )}
          </UploadBox>

          <UploadBox>
            <UploadTitle>Terrain</UploadTitle>
            <UploadHint>.tif / .tiff • DEM / elevation</UploadHint>
            <HiddenInput
              type="file"
              accept=".tif,.tiff"
              onChange={(e) => setTerrainFile(e.target.files[0] || null)}
            />
            <div style={{ fontSize: 12, marginTop: 6, opacity: 0.85 }}>
              Click to select file
            </div>
            {terrainFile && <FileName>Selected: {terrainFile.name}</FileName>}
          </UploadBox>

          <UploadBox>
            <UploadTitle>LULC</UploadTitle>
            <UploadHint>.tif / .tiff • Land use / land cover</UploadHint>
            <HiddenInput
              type="file"
              accept=".tif,.tiff"
              onChange={(e) => setLulcFile(e.target.files[0] || null)}
            />
            <div style={{ fontSize: 12, marginTop: 6, opacity: 0.85 }}>
              Click to select file
            </div>
            {lulcFile && <FileName>Selected: {lulcFile.name}</FileName>}
          </UploadBox>
        </UploadGrid>

        <ActionsRow>
          <RunButton onClick={handleSubmit} disabled={loading}>
            {loading ? "Running..." : "Run Inference"}
          </RunButton>
          {status && !error && <StatusText>{status}</StatusText>}
          {error && <StatusText error>{error}</StatusText>}
        </ActionsRow>

        {resultImageUrl && (
          <ResultSection>
            <ResultTitleRow>
              <ResultTitle>Model Output</ResultTitle>
              <ResultTag>PNG • Visualization</ResultTag>
            </ResultTitleRow>
            <ResultImageWrapper>
              <ResultImage
                src={resultImageUrl}
                alt="S1 before flood and predicted flood mask"
              />
            </ResultImageWrapper>
          </ResultSection>
        )}
      </Card>
    </PageWrapper>
  );
};

export default FloodDetectionPage;
