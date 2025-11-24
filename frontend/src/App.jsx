import React, { useState, useEffect } from "react";
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

// Toast (simple, not messing with main layout)
const DriftToast = styled.div`
  position: fixed;
  right: 24px;
  bottom: 24px;
  border-radius: 16px;
  padding: 10px 14px;
  font-size: 12px;
  max-width: 320px;
  box-shadow:
    0 10px 30px rgba(0, 0, 0, 0.7),
    0 0 0 1px rgba(15, 23, 42, 0.9);
  z-index: 50;
`;

// ---------- helpers ----------

const getSeverityColors = (severity) => {
  switch (severity) {
    case "high":
      return {
        bg: "rgba(239, 68, 68, 0.18)",
        border: "rgba(239, 68, 68, 0.6)",
        text: "#fecaca",
      };
    case "medium":
      return {
        bg: "rgba(245, 158, 11, 0.18)",
        border: "rgba(245, 158, 11, 0.6)",
        text: "#fed7aa",
      };
    case "low":
      return {
        bg: "rgba(56, 189, 248, 0.14)",
        border: "rgba(56, 189, 248, 0.55)",
        text: "#7dd3fc",
      };
    case "none":
    default:
      return {
        bg: "rgba(34, 197, 94, 0.14)",
        border: "rgba(34, 197, 94, 0.55)",
        text: "#bbf7d0",
      };
  }
};

// ---------- component ----------

const FloodDetectionPage = () => {
  const [s1BeforeFloodFile, setS1BeforeFloodFile] = useState(null);
  const [s1AfterFloodFile, setS1AfterFloodFile] = useState(null);
  const [terrainFile, setTerrainFile] = useState(null);
  const [lulcFile, setLulcFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  // result images
  const [inputsAndPredictionUrl, setInputsAndPredictionUrl] = useState(null);
  const [xaiMapsUrl, setXaiMapsUrl] = useState(null);

  // drift info
  const [driftAlert, setDriftAlert] = useState(null); // bool or null
  const [driftSeverity, setDriftSeverity] = useState(null); // "none"/"low"/"medium"/"high"
  const [driftReport, setDriftReport] = useState(null); // per-channel
  const [driftMessages, setDriftMessages] = useState([]); // list of strings
  const [driftExpanded, setDriftExpanded] = useState(false);
  const [showDriftToast, setShowDriftToast] = useState(false);

  const handleSubmit = async () => {
    if (!s1BeforeFloodFile || !s1AfterFloodFile || !terrainFile || !lulcFile) {
      setError("Please upload all 4 required TIFF images before running inference.");
      return;
    }

    setError("");
    setStatus("Uploading and running inference...");
    setLoading(true);

    const formData = new FormData();
    formData.append("s1_before_flood", s1BeforeFloodFile);
    formData.append("s1_after_flood", s1AfterFloodFile);
    formData.append("terrain", terrainFile);
    formData.append("lulc", lulcFile);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Accept: "application/json",
        },
      });

      const {
        inputs_and_prediction,
        xai_maps,
        drift_report,
        drift_alert,
        drift_overall_severity,
        drift_messages,
      } = response.data || {};

      if (!inputs_and_prediction || !xai_maps) {
        throw new Error("Backend response missing expected image keys.");
      }

      const inputsUrl = `data:image/png;base64,${inputs_and_prediction}`;
      const xaiUrl = `data:image/png;base64,${xai_maps}`;

      setInputsAndPredictionUrl(inputsUrl);
      setXaiMapsUrl(xaiUrl);

      setDriftReport(drift_report || null);
      setDriftAlert(typeof drift_alert === "boolean" ? drift_alert : null);
      setDriftSeverity(drift_overall_severity || null);
      setDriftMessages(Array.isArray(drift_messages) ? drift_messages : []);
      setDriftExpanded(false); // collapse on new run

      setStatus("Inference completed.");
    } catch (err) {
      console.error(err);
      setError("Error during prediction. Check backend logs.");
      setStatus("");
      setInputsAndPredictionUrl(null);
      setXaiMapsUrl(null);
      setDriftAlert(null);
      setDriftSeverity(null);
      setDriftReport(null);
      setDriftMessages([]);
      setDriftExpanded(false);
    } finally {
      setLoading(false);
    }
  };

  // Show toast when drift is detected
  useEffect(() => {
    if (driftAlert === true) {
      setShowDriftToast(true);
      const timer = setTimeout(() => setShowDriftToast(false), 5000);
      return () => clearTimeout(timer);
    } else {
      setShowDriftToast(false);
    }
  }, [driftAlert]);

  const severityColors = getSeverityColors(driftSeverity || "none");

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

        {inputsAndPredictionUrl && (
          <ResultSection>
            <ResultTitleRow>
              <ResultTitle>Inputs & Predicted Flood Map</ResultTitle>
              <ResultTag>PNG • Visualization</ResultTag>
            </ResultTitleRow>
            <ResultImageWrapper>
              <ResultImage
                src={inputsAndPredictionUrl}
                alt="Inputs and predicted flood mask"
              />
            </ResultImageWrapper>
          </ResultSection>
        )}

        {xaiMapsUrl && (
          <ResultSection>
            <ResultTitleRow>
              <ResultTitle>XAI Maps (Saliency, GradCAM, IG)</ResultTitle>
              <ResultTag>PNG • Visualization</ResultTag>
            </ResultTitleRow>
            <ResultImageWrapper>
              <ResultImage
                src={xaiMapsUrl}
                alt="XAI maps: saliency, GradCAM, integrated gradients"
              />
            </ResultImageWrapper>
          </ResultSection>
        )}

        {driftReport && (
          <ResultSection>
            <ResultTitleRow>
              <ResultTitle>Data Drift Monitoring</ResultTitle>
              <ResultTag
                style={{
                  background: severityColors.bg,
                  borderColor: severityColors.border,
                  color: severityColors.text,
                }}
              >
                {driftSeverity
                  ? `${driftSeverity.toUpperCase()} drift`
                  : "No drift"}
              </ResultTag>
            </ResultTitleRow>

            {/* Alert messages */}
            {driftMessages.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                {driftMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: "rgba(15,23,42,0.85)",
                      borderRadius: 10,
                      border: `1px solid ${severityColors.border}`,
                      padding: "6px 10px",
                      fontSize: 12,
                      marginBottom: 6,
                      color: severityColors.text,
                    }}
                  >
                    {msg}
                  </div>
                ))}
              </div>
            )}

            {/* Collapsible details */}
            <div
              style={{
                fontSize: 11,
                color: "#9ca3af",
                cursor: "pointer",
                marginBottom: driftExpanded ? 6 : 0,
                textDecoration: "underline",
                width: "fit-content",
              }}
              onClick={() => setDriftExpanded((prev) => !prev)}
            >
              {driftExpanded ? "Hide drift details" : "Show drift details"}
            </div>

            {driftExpanded && (
              <ul
                style={{
                  marginTop: 4,
                  fontSize: 12,
                  color: "#9ca3af",
                  paddingLeft: 18,
                }}
              >
                {Object.entries(driftReport).map(([key, val]) => {
                  const v = val || {};
                  const groupColors = getSeverityColors(v.severity || "none");
                  return (
                    <li key={key} style={{ marginBottom: 4 }}>
                      <span
                        style={{
                          color: "#e5e7eb",
                          fontWeight: 600,
                          marginRight: 4,
                        }}
                      >
                        {key}
                      </span>
                      <span
                        style={{
                          padding: "2px 8px",
                          borderRadius: 999,
                          fontSize: 10,
                          marginRight: 6,
                          background: groupColors.bg,
                          border: `1px solid ${groupColors.border}`,
                          color: groupColors.text,
                        }}
                      >
                        {v.severity ? v.severity.toUpperCase() : "NONE"}
                      </span>
                      <span>
                        {`z_mean=${Number(v.z_mean_shift || 0).toFixed(
                          2
                        )}, z_std=${Number(v.z_std_shift || 0).toFixed(
                          2
                        )}, wd=${Number(
                          v.wasserstein_distance || 0
                        ).toFixed(2)}`}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </ResultSection>
        )}
      </Card>

      {/* Toast for drift alert */}
      {showDriftToast && driftSeverity && (
        <DriftToast
          style={{
            background: severityColors.bg,
            border: `1px solid ${severityColors.border}`,
            color: severityColors.text,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 12 }}>
            {driftSeverity.toUpperCase()} data drift detected
          </div>
          <div style={{ fontSize: 11 }}>
            {driftMessages && driftMessages.length > 0
              ? driftMessages[0]
              : "Inputs differ from training distribution. Review data before relying on predictions."}
          </div>
        </DriftToast>
      )}
    </PageWrapper>
  );
};

export default FloodDetectionPage;
