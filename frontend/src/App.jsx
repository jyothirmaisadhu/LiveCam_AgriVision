
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const CROPS = [
  { id: "tomato", name: "Tomato", emoji: "🍅" },
  { id: "corn", name: "Corn", emoji: "🌽" },
  { id: "paddy", name: "Paddy", emoji: "🌾" },
  { id: "chilli", name: "Chilli", emoji: "🌶️" },
  { id: "cotton", name: "Cotton", emoji: "🌿" },
];

function App() {
  const [crop, setCrop] = useState("tomato");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [cameraOn, setCameraOn] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [preview]);

  function chooseCrop(nextCrop) {
    setCrop(nextCrop);
    setResult(null);
    setError("");
  }

  function selectFile(selectedFile) {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("The image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }

  async function startCamera() {
    setError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera access is unavailable in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });

      streamRef.current = stream;
      setCameraOn(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setError("Could not access the camera. Check browser permissions.");
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraOn(false);
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setError("The camera is not ready yet.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Could not capture the image.");
        return;
      }

      selectFile(
        new File([blob], "camera-leaf.jpg", { type: "image/jpeg" })
      );
      stopCamera();
    }, "image/jpeg", 0.92);
  }

  async function predictDisease(event) {
    event.preventDefault();

    if (!file) {
      setError("Upload or capture a leaf image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/api/predict/${crop}`,
        { method: "POST", body: formData }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the API. Check that FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function speakResult() {
    if (!result || !("speechSynthesis" in window)) {
      setError("Voice output is not supported by this browser.");
      return;
    }

    window.speechSynthesis.cancel();

    const message = new SpeechSynthesisUtterance(
      `Crop: ${result.crop_name}. Prediction: ${result.predicted_class}. ` +
      `Model confidence: ${result.confidence} percent. ` +
      `This is an AI prediction, not a confirmed diagnosis.`
    );

    window.speechSynthesis.speak(message);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home">
          <span className="brand-icon">🌱</span>
          <span>AgriVision<span className="brand-dot">.</span></span>
        </a>
        <span className="api-status">
          <span className="status-dot" />
          AI Plant Classifier
        </span>
      </header>

      <main id="home" className="main-content">
        <section className="hero">
          <span className="eyebrow">SMART AGRICULTURE · AI ASSISTANT</span>
          <h1>Healthy plants.<br /><span>Better harvests.</span></h1>
          <p>
            Analyze a leaf image with your trained AI models and get a
            plant-classification prediction in seconds.
          </p>
        </section>

        <section className="workspace">
          <div className="panel input-panel">
            <div className="section-heading">
              <span className="step-number">01</span>
              <div>
                <h2>Choose your crop</h2>
                <p>Select the crop shown in your image.</p>
              </div>
            </div>

            <div className="crop-grid">
              {CROPS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`crop-option ${crop === item.id ? "selected" : ""}`}
                  onClick={() => chooseCrop(item.id)}
                  aria-pressed={crop === item.id}
                >
                  <span className="crop-emoji">{item.emoji}</span>
                  <span>{item.name}</span>
                  {crop === item.id && <span className="selected-mark">✓</span>}
                </button>
              ))}
            </div>

            <div className="section-heading image-heading">
              <span className="step-number">02</span>
              <div>
                <h2>Add a leaf image</h2>
                <p>Use an image or capture one with your camera.</p>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => selectFile(event.target.files?.[0])}
            />

            {cameraOn ? (
              <div className="camera-box">
                <video ref={videoRef} autoPlay playsInline muted />
                <div className="camera-actions">
                  <button type="button" className="primary-button" onClick={capturePhoto}>
                    Capture photo
                  </button>
                  <button type="button" className="secondary-button" onClick={stopCamera}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : preview ? (
              <div className="preview-box">
                <img src={preview} alt="Selected leaf preview" />
                <div className="preview-caption">
                  <span>Image ready for analysis</span>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Change image
                  </button>
                </div>
              </div>
            ) : (
              <button
                type="button"
                className="upload-box"
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="upload-icon">↑</span>
                <strong>Click to upload a leaf image</strong>
                <span>JPG, PNG or other browser-supported image · Max 10 MB</span>
              </button>
            )}

            {!cameraOn && (
              <button
                type="button"
                className="camera-button"
                onClick={startCamera}
              >
                <span>◎</span> Use camera
              </button>
            )}

            <button
              type="button"
              className="analyze-button"
              onClick={predictDisease}
              disabled={loading || !file}
            >
              {loading ? "Analyzing image..." : "✦  Analyze leaf"}
              {!loading && <span>→</span>}
            </button>

            {error && <div className="error-message" role="alert">{error}</div>}

            <p className="privacy-note">
              Images are processed by your local development API.
            </p>
          </div>

          <div className="panel result-panel">
            <div className="section-heading">
              <span className="step-number">03</span>
              <div>
                <h2>Analysis results</h2>
                <p>Your prediction will appear here.</p>
              </div>
            </div>

            {loading ? (
              <div className="empty-state">
                <div className="loader" />
                <h3>Analyzing your leaf</h3>
                <p>Your selected crop model is processing the image.</p>
              </div>
            ) : result ? (
              <div className="result-content">
                <div className="result-success">✓ Analysis complete</div>
                <div className="prediction-label">PREDICTED CLASS</div>
                <h3 className="prediction-name">{result.predicted_class.replaceAll("_", " ")}</h3>
                <div className="confidence-row">
                  <span>Model confidence</span>
                  <strong>{result.confidence.toFixed(2)}%</strong>
                </div>
                <div className="confidence-track">
                  <div style={{ width: `${Math.min(result.confidence, 100)}%` }} />
                </div>

                <h4>Top predictions</h4>
                <div className="prediction-list">
                  {result.top_predictions.map((prediction) => (
                    <div className="prediction-item" key={prediction.class_name}>
                      <span>{prediction.class_name.replaceAll("_", " ")}</span>
                      <strong>{prediction.confidence.toFixed(2)}%</strong>
                    </div>
                  ))}
                </div>

                <div className="inference-time">
                  <span>Inference time</span>
                  <strong>{result.inference_time_ms} ms</strong>
                </div>

                <button type="button" className="secondary-button voice-button" onClick={speakResult}>
                  ♫ Read result aloud
                </button>

                <p className="disclaimer">{result.notice}</p>
              </div>
            ) : (
              <div className="empty-state">
                <div className="leaf-art">🌿</div>
                <h3>Ready when you are</h3>
                <p>
                  Select a crop and upload a clear leaf image. Your AI
                  prediction and confidence scores will show up here.
                </p>
                <div className="empty-features">
                  <span>✓ 5 crop models</span>
                  <span>✓ Top predictions</span>
                  <span>✓ Voice output</span>
                </div>
              </div>
            )}
          </div>
        </section>

        <footer className="footer">
          <span>🌱 LiveCam AgriVision</span>
          <span>AI-assisted plant classification</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
