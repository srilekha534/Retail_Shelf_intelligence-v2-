"use client";

import React from "react";
import { useSettings } from "../settings-context";
import styles from "../page.module.css";

export default function SettingsPage() {
  const { confidence, setConfidence, ocrEnabled, setOcrEnabled } = useSettings();

  return (
    <div>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Settings</h1>
        <p className={styles.pageSubtitle}>Configure global analysis parameters for the Retail AI.</p>
      </div>

      <div className={styles.settingsPanel} style={{ maxWidth: "600px" }}>
        <h3 className={styles.uploadTitle} style={{ marginBottom: "1.5rem" }}>Analysis Settings</h3>
        
        <div style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
            <label style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)" }}>
              Confidence Threshold
            </label>
            <span style={{ fontSize: "0.875rem", color: "var(--primary)", fontWeight: 600 }}>
              {confidence.toFixed(2)}
            </span>
          </div>
          <input 
            type="range" 
            min="0.1" 
            max="0.95" 
            step="0.05" 
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            style={{ width: "100%", accentColor: "var(--primary)", cursor: "pointer" }}
          />
          <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.5rem" }}>
            Lower values detect more products but may increase false positives. Higher values are stricter.
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
          <input 
            type="checkbox" 
            id="ocrToggle" 
            checked={ocrEnabled}
            onChange={(e) => setOcrEnabled(e.target.checked)}
            style={{ marginTop: "0.25rem", width: "16px", height: "16px", accentColor: "var(--primary)", cursor: "pointer" }}
          />
          <div>
            <label htmlFor="ocrToggle" style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)", cursor: "pointer" }}>
              Enable OCR Brand Recognition
            </label>
            <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
              Turn off for faster analysis if you only need bounding boxes and gap detection.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
