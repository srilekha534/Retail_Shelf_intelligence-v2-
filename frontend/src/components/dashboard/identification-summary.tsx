import React from "react";
import styles from "./AnomalyPanel.module.css";

interface IdentificationSummaryProps {
  totalDetected: number;
  identified: number;
}

export function IdentificationSummary({ totalDetected, identified }: IdentificationSummaryProps) {
  const rate = totalDetected > 0 ? (identified / totalDetected) * 100 : 0;
  const unidentified = totalDetected - identified;
  const dashArray = 283; // 2 * pi * r (approx for r=45)
  const dashOffset = dashArray - (dashArray * rate) / 100;

  return (
    <div className={styles.panel} style={{ height: "100%" }}>
      <div className={styles.panelHeader}>
        <div className={styles.title}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          Identification Summary
        </div>
      </div>
      
      <div style={{ padding: "1.5rem", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
        
        {/* Stats List */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem", flex: 1, minWidth: "120px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{width:"16px", height:"16px", display:"flex", alignItems:"center", justifyContent:"center", color:"var(--text-secondary)"}}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            </span>
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 500 }}>Total Detected</div>
              <div style={{ fontSize: "1rem", fontWeight: 600 }}>{totalDetected}</div>
            </div>
          </div>
          
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{width:"16px", height:"16px", display:"flex", alignItems:"center", justifyContent:"center", color:"var(--success)"}}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </span>
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--success)", fontWeight: 500 }}>Identified</div>
              <div style={{ fontSize: "1rem", fontWeight: 600 }}>{identified}</div>
            </div>
          </div>
          
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{width:"16px", height:"16px", display:"flex", alignItems:"center", justifyContent:"center", color:"var(--danger)"}}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            </span>
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--danger)", fontWeight: 500 }}>Unidentified</div>
              <div style={{ fontSize: "1rem", fontWeight: 600 }}>{unidentified}</div>
            </div>
          </div>
        </div>

        {/* Circular Progress */}
        <div style={{ position: "relative", width: "120px", height: "120px", flexShrink: 0, margin: "0 auto" }}>
          <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)" }}>
            <circle cx="50" cy="50" r="45" fill="none" stroke="var(--border)" strokeWidth="10" />
            <circle 
              cx="50" cy="50" r="45" 
              fill="none" 
              stroke="var(--primary)" 
              strokeWidth="10" 
              strokeDasharray={dashArray}
              strokeDashoffset={dashOffset}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 1s ease" }}
            />
          </svg>
          <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary)" }}>{rate.toFixed(2)}%</div>
            <div style={{ fontSize: "0.6rem", color: "var(--text-secondary)", textTransform: "uppercase", fontWeight: 600 }}>ID Rate</div>
          </div>
        </div>

      </div>
    </div>
  );
}
