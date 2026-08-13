import React from "react";
import styles from "./AnomalyPanel.module.css";

interface AnomaliesVisualizationProps {
  imageSrc: string | null;
}

export function AnomaliesVisualization({ imageSrc }: AnomaliesVisualizationProps) {
  return (
    <div className={styles.panel} style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div className={styles.panelHeader}>
        <div className={styles.title} style={{ color: "var(--text-primary)" }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
          Anomalies Visualization
        </div>
      </div>
      
      <div style={{ flex: 1, padding: "0.5rem", display: "flex", alignItems: "center", justifyContent: "center", backgroundColor: "#000", borderRadius: "0 0 var(--radius-md) var(--radius-md)", overflow: "hidden", position: "relative" }}>
        {imageSrc ? (
          <>
            <img src={imageSrc} alt="Anomalies Visualization" style={{ maxWidth: "100%", maxHeight: "250px", objectFit: "contain" }} />
            {/* We could dynamically overlay badges here if backend provided exact absolute bounding boxes for all anomaly types, 
                but rendering the backend's annotated image is perfectly accurate. */}
          </>
        ) : (
          <span style={{ color: "#475569", fontSize: "0.875rem" }}>No visualization available</span>
        )}
      </div>
    </div>
  );
}
