"use client";

import React, { useState, useRef, useEffect } from "react";
import styles from "./page.module.css";
import { MetricsBar } from "../components/dashboard/metrics-bar";
import { AnomalyPanel, Anomaly } from "../components/dashboard/anomaly-panel";
import { AnomaliesSummary } from "../components/dashboard/anomalies-summary";
import { IdentificationSummary } from "../components/dashboard/identification-summary";
import { AnomalyListPanel } from "../components/dashboard/anomaly-list-panel";
import { AnomaliesVisualization } from "../components/dashboard/anomalies-visualization";
import { ProductChart } from "../components/dashboard/product-chart";
import { OCRResultsTable } from "../components/dashboard/ocr-results-table";
import { useSettings } from "./settings-context";

interface ProcessedResult {
  filename: string;
  originalImage: string | null;
  processedImage: string | null;
  anomalyImage: string | null;
  metrics: {
    totalProducts: number;
    identifiedProducts: number;
    outOfStock: number;
    planogramAnomalies: number;
    countsByName: Record<string, number>;
  };
  anomalies: Anomaly[];
  ocrResults: any[];
}

export default function Home() {
  const { confidence, ocrEnabled } = useSettings();
  
  const [results, setResults] = useState<ProcessedResult[]>([]);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [maximizedImage, setMaximizedImage] = useState<string | null>(null);
  
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);

  const [hardwareDevice, setHardwareDevice] = useState<string | null>(null);

  // Rehydrate results from session storage when returning to the dashboard
  useEffect(() => {
    const saved = sessionStorage.getItem("dashboard_results");
    if (saved) {
      try {
        setResults(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse saved dashboard results", e);
      }
    }
  }, []);

  // Save results to session storage whenever they change
  useEffect(() => {
    if (results.length > 0) {
      sessionStorage.setItem("dashboard_results", JSON.stringify(results));
    } else {
      sessionStorage.removeItem("dashboard_results");
    }
  }, [results]);

  // Fetch API health and current device
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then(res => res.json())
      .then(data => {
        if (data && data.device) {
          setHardwareDevice(data.device.toUpperCase());
        }
      })
      .catch(err => console.error("Failed to fetch device status:", err));
  }, []);

  // Stop camera when unmounting
  useEffect(() => {
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    setIsCameraActive(true);
    setResults([]);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert("Failed to access camera. Please check permissions.");
      setIsCameraActive(false);
    }
  };

  function stopCamera() {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const captureCamera = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        stopCamera();
        processImages([file]);
      }
    }, "image/jpeg");
  };

  const handleSingleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setPendingFiles([e.target.files[0]]);
    }
  };

  const handleBatchUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setPendingFiles(Array.from(e.target.files));
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setPendingFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const processImages = async (files: File[]) => {
    setIsProcessing(true);
    setProgress({ current: 0, total: files.length });
    setResults([]);
    setPendingFiles([]); // clear so they don't show up again on reset

    const newResults: ProcessedResult[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setProgress({ current: i + 1, total: files.length });
      
      const formData = new FormData();
      formData.append("file", file);
      formData.append("confidence", confidence.toString());
      formData.append("ocr_enabled", ocrEnabled.toString());
      formData.append("detect_anomalies", "true");

      try {
        const response = await fetch("http://localhost:8000/detect", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          
          const newAnomalies: Anomaly[] = [];
          let oos = 0;
          let plano = 0;

          if (data.anomalies) {
            data.anomalies.forEach((a: any, idx: number) => {
              let type: "danger" | "warning" | "info" = "warning";
              if (a.severity === "high") type = "danger";
              else if (a.severity === "low") type = "info";
              
              let title = "Anomaly Detected";
              if (a.type === "dl_anomaly") {
                title = "Visual Damage / Wrong Facings";
              } else if (a.type === "planogram_violation") {
                title = "Planogram Violation";
                plano++;
              } else if (a.type === "empty_shelf" || a.type === "low_stock") {
                title = "Out of Stock Gap";
                oos++;
              } else if (a.type === "misplaced") {
                title = "Misplaced Product";
              } else if (a.type === "price_tag_missing") {
                title = "Price Tag Missing";
              }
              
              newAnomalies.push({ id: `backend-${i}-${idx}`, type, title, description: a.description });
            });
          }

          newResults.push({
            filename: file.name,
            originalImage: data.original_image_path ? data.original_image_path : URL.createObjectURL(file),
            processedImage: data.processed_image_path ? data.processed_image_path : (data.image_b64 ? `data:image/jpeg;base64,${data.image_b64}` : null),
            anomalyImage: data.anomaly_image_path ? data.anomaly_image_path : (data.image_anomaly_b64 ? `data:image/jpeg;base64,${data.image_anomaly_b64}` : null),
            metrics: {
              totalProducts: data.total_products,
              identifiedProducts: data.product_inventory?.total_identified || 0,
              outOfStock: oos,
              planogramAnomalies: plano,
              countsByName: data.product_inventory?.counts_by_name || {},
            },
            anomalies: newAnomalies,
            ocrResults: data.product_inventory?.products || [],
          });
        }
      } catch (error) {
        console.error("Failed to process", file.name, error);
      }
    }
    
    setResults(newResults);
    setIsProcessing(false);
  };

  // Calculate aggregates
  const aggMetrics = results.reduce((acc, curr) => ({
    totalProducts: acc.totalProducts + curr.metrics.totalProducts,
    identifiedProducts: acc.identifiedProducts + curr.metrics.identifiedProducts,
    outOfStock: acc.outOfStock + curr.metrics.outOfStock,
    planogramAnomalies: acc.planogramAnomalies + curr.metrics.planogramAnomalies,
  }), { totalProducts: 0, identifiedProducts: 0, outOfStock: 0, planogramAnomalies: 0 });

  const aggAnomalies = results.flatMap(r => r.anomalies);
  

  return (
    <div>
      <div className={styles.pageHeader} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h1 className={styles.pageTitle} style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            Shelf Analysis
            {hardwareDevice && (
              <span style={{
                fontSize: "0.75rem",
                padding: "0.25rem 0.75rem",
                borderRadius: "999px",
                backgroundColor: hardwareDevice === "CUDA" ? "var(--success)" : "var(--primary)",
                color: "white",
                fontWeight: 600,
                letterSpacing: "0.05em",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.3rem"
              }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
                {hardwareDevice === "CUDA" ? "GPU ACTIVE" : "CPU ACTIVE"}
              </span>
            )}
          </h1>
          <p className={styles.pageSubtitle}>Upload an image or use the camera to detect products and gaps.</p>
        </div>
        
        <div style={{ display: "flex", gap: "1rem" }}>
          <button className={styles.toggleBtn} onClick={() => fileInputRef.current?.click()} disabled={isProcessing}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Upload
          </button>
          <button className={styles.toggleBtn} onClick={() => batchInputRef.current?.click()} disabled={isProcessing}>
             <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            Batch Upload
          </button>
          <button className={styles.uploadButton} onClick={startCamera} disabled={isProcessing}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
            Live Camera
          </button>
          
          {results.length > 0 && !isProcessing && (
            <button className={styles.toggleBtn} onClick={() => setResults([])} style={{ borderColor: "var(--danger)", color: "var(--danger)" }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
              Clear All
            </button>
          )}
          
          <input type="file" ref={fileInputRef} onChange={handleSingleUpload} accept="image/*" style={{ display: "none" }} />
          <input type="file" ref={batchInputRef} onChange={handleBatchUpload} accept="image/*" multiple style={{ display: "none" }} />
        </div>
      </div>
      {!isProcessing && !isCameraActive && (
        <div 
          className={styles.settingsPanel} 
          style={{ 
            textAlign: "center", 
            padding: "4rem 2rem", 
            marginTop: "2rem",
            border: pendingFiles.length === 0 ? "2px dashed var(--border)" : "1px solid var(--border)",
            cursor: pendingFiles.length === 0 ? "pointer" : "default",
            backgroundColor: "var(--bg-secondary)",
            transition: "all 0.2s ease"
          }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => { if (pendingFiles.length === 0) batchInputRef.current?.click(); }}
        >
          {pendingFiles.length === 0 ? (
            <>
              <div style={{ marginBottom: "1rem", color: "var(--primary)" }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "0.5rem" }}>Drag & Drop an image here</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", maxWidth: "400px", margin: "0 auto" }}>
                or click to browse from your computer. You can also use the live camera or upload a batch of images.
              </p>
            </>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.5rem" }}>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 600 }}>Ready to analyze</h3>
              
              <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", justifyContent: "center" }}>
                {pendingFiles.map((file, idx) => (
                  <div key={idx} style={{ 
                    display: "flex", alignItems: "center", gap: "0.75rem", 
                    padding: "0.75rem 1rem", backgroundColor: "var(--bg-primary)", 
                    borderRadius: "var(--radius-md)", border: "1px solid var(--border)",
                    boxShadow: "var(--shadow-sm)"
                  }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{color: "var(--primary)"}}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    <div style={{ textAlign: "left" }}>
                      <div style={{ fontSize: "0.875rem", fontWeight: 500, color: "var(--text-primary)", maxWidth: "150px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                    </div>
                    <button 
                      onClick={(e) => { e.stopPropagation(); setPendingFiles(pendingFiles.filter((_, i) => i !== idx)); }}
                      style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: "0.25rem" }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: "1rem" }}>
                <button 
                  className={styles.uploadButton} 
                  style={{ padding: "0.75rem 2.5rem", fontSize: "1rem" }}
                  onClick={() => processImages(pendingFiles)}
                >
                  Run Detection
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {!isProcessing && results.length > 0 && (
          <MetricsBar 
            totalProducts={aggMetrics.totalProducts} 
            identifiedProducts={aggMetrics.identifiedProducts} 
            outOfStock={aggMetrics.outOfStock} 
            anomaliesCount={aggMetrics.planogramAnomalies} 
          />
      )}

      {isProcessing && (
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          padding: "5rem 2rem", backgroundColor: "var(--bg-secondary)", borderRadius: "var(--radius-lg)",
          border: "1px solid var(--border)", margin: "2rem 0", boxShadow: "var(--shadow-sm)"
        }}>
          <div style={{ position: "relative", width: "80px", height: "80px", marginBottom: "2rem" }}>
             {/* Spinner animation */}
             <div style={{
               position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
               border: "4px solid var(--bg-primary)", borderRadius: "50%",
             }}></div>
             <div style={{
               position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
               border: "4px solid var(--primary)", borderRadius: "50%",
               borderTopColor: "transparent", animation: "spin 1s linear infinite"
             }}></div>
             <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          </div>
          
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.5rem", color: "var(--text-primary)" }}>
            Analyzing Shelf Data...
          </h2>
          <p style={{ color: "var(--text-secondary)", marginBottom: "2rem" }}>
            Processing {progress.current} of {progress.total} images using deep learning pipelines.
          </p>

          <div style={{ width: "100%", maxWidth: "400px", height: "6px", backgroundColor: "var(--bg-primary)", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ width: `${(progress.current / progress.total) * 100}%`, height: "100%", backgroundColor: "var(--primary)", transition: "width 0.3s ease" }} />
          </div>
        </div>
      )}

      {isCameraActive && (
        <div className={styles.settingsPanel} style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "1rem" }}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: 600 }}>Live Camera</h3>
            <button onClick={stopCamera} style={{ background: "transparent", border: "none", cursor: "pointer", color: "var(--text-secondary)" }}>
               <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
          <div style={{ position: "relative", width: "100%", backgroundColor: "#000", borderRadius: "8px", overflow: "hidden", display: "flex", justifyContent: "center" }}>
            <video ref={videoRef} autoPlay playsInline style={{ maxHeight: "60vh", maxWidth: "100%" }} />
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>
          <div style={{ display: "flex", justifyContent: "center", marginTop: "1rem" }}>
            <button className={styles.uploadButton} onClick={captureCamera}>Capture & Analyze</button>
          </div>
        </div>
      )}

      {!isProcessing && results.length > 0 && (
        <>
          <div className={styles.mainGrid}>
            <div className={styles.leftColumn}>
              {results.map((result, idx) => (
                <div key={idx} style={{ marginBottom: "1.5rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem", color: "var(--text-primary)" }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    <h2 style={{ fontSize: "1.125rem", fontWeight: 600, flex: 1 }}>{result.filename}</h2>
                    <button 
                      onClick={() => setResults(results.filter((_, i) => i !== idx))}
                      style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer", padding: "0.25rem", display: "flex", alignItems: "center", borderRadius: "50%", transition: "background 0.2s" }}
                      title="Remove from history"
                      onMouseOver={(e) => e.currentTarget.style.backgroundColor = "rgba(239, 68, 68, 0.1)"}
                      onMouseOut={(e) => e.currentTarget.style.backgroundColor = "transparent"}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                  </div>
                  
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                    <div className={styles.imageViewer}>
                      <div className={styles.imageHeader}><div className={styles.imageTitle}>Original Input</div></div>
                      <div className={styles.imageContainer} style={{ maxHeight: "350px", cursor: "pointer" }} onClick={() => result.originalImage && setMaximizedImage(result.originalImage)}>
                        {result.originalImage && <img src={result.originalImage} alt="Original Input" style={{ objectFit: "contain" }} />}
                      </div>
                    </div>
                    <div className={styles.imageViewer}>
                      <div className={styles.imageHeader}><div className={styles.imageTitle}>AI Analysis Output (Detections)</div></div>
                      <div className={styles.imageContainer} style={{ maxHeight: "350px", cursor: "pointer" }} onClick={() => result.processedImage && setMaximizedImage(result.processedImage)}>
                        {result.processedImage && <img src={result.processedImage} alt="Processed Output" style={{ objectFit: "contain" }} />}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className={styles.rightColumn}>
              <AnomaliesSummary anomalies={aggAnomalies} />
            </div>
          </div>

          <div className={styles.bottomGrid}>
             <IdentificationSummary totalDetected={aggMetrics.totalProducts} identified={aggMetrics.identifiedProducts} />
             <AnomalyListPanel 
                title="Out of Stock Gaps" 
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{marginRight: "0.5rem", color: "var(--warning)"}}><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>} 
                anomalies={aggAnomalies.filter(a => a.type === "danger" && (a.title.includes("Out of Stock") || a.title.includes("Low Stock")))} 
             />
             <AnomalyListPanel 
                title="Planogram Anomalies" 
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{marginRight: "0.5rem", color: "var(--danger)"}}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>} 
                anomalies={aggAnomalies.filter(a => a.title.includes("Planogram"))} 
             />
             {results[0] && (
               <div onClick={() => results[0].anomalyImage && setMaximizedImage(results[0].anomalyImage)} style={{ cursor: "pointer" }}>
                 <AnomaliesVisualization imageSrc={results[0].anomalyImage} />
               </div>
             )}
             {results.length > 0 && <OCRResultsTable ocrResults={results.flatMap(r => r.ocrResults)} counts={results.reduce((acc, r) => {
                Object.entries(r.metrics.countsByName).forEach(([name, count]) => {
                  acc[name] = (acc[name] || 0) + count;
                });
                return acc;
              }, {} as Record<string, number>)} />}
          </div>
        </>
      )}

      {maximizedImage && (
        <div 
          style={{
            position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh",
            backgroundColor: "rgba(0,0,0,0.85)", zIndex: 9999,
            display: "flex", alignItems: "center", justifyContent: "center",
            padding: "2rem", cursor: "zoom-out"
          }}
          onClick={() => setMaximizedImage(null)}
        >
          <div style={{ position: "relative", maxWidth: "95%", maxHeight: "95%" }}>
            <img src={maximizedImage} alt="Maximized view" style={{ maxWidth: "100%", maxHeight: "90vh", objectFit: "contain", borderRadius: "8px", boxShadow: "0 25px 50px -12px rgba(0,0,0,0.5)" }} />
            <button 
              style={{ position: "absolute", top: "-1rem", right: "-1rem", background: "var(--danger)", color: "white", border: "none", borderRadius: "50%", width: "32px", height: "32px", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }}
              onClick={(e) => { e.stopPropagation(); setMaximizedImage(null); }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
