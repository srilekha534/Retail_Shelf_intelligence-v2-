import React from "react";
import styles from "./AnomalyPanel.module.css";
import { Anomaly } from "./anomaly-panel";

interface AnomalyListPanelProps {
  title: string;
  icon: React.ReactNode;
  anomalies: Anomaly[];
}

export function AnomalyListPanel({ title, icon, anomalies }: AnomalyListPanelProps) {
  return (
    <div className={styles.panel} style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div className={styles.panelHeader}>
        <div className={styles.title} style={{ color: "var(--text-primary)" }}>
          {icon}
          {title}
        </div>
        <div className={styles.badge} style={{ backgroundColor: "var(--bg-primary)", color: "var(--primary)" }}>{anomalies.length}</div>
      </div>
      
      <div style={{ flex: 1, overflowY: "auto", maxHeight: "250px" }}>
        {anomalies.length === 0 ? (
          <div className={styles.emptyState} style={{ padding: "2rem 1rem", textAlign: "center" }}>
            <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>None detected</span>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column" }}>
            {anomalies.map((anomaly, idx) => (
              <div key={idx} style={{ padding: "1rem", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                  <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.25rem" }}>
                    {anomaly.title.includes("Out of Stock") ? `Gap ${idx + 1}` : `Violation ${idx + 1}`}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                    {anomaly.description}
                  </div>
                </div>
                
                <div style={{ 
                  fontSize: "0.65rem", 
                  fontWeight: 600,
                  padding: "0.25rem 0.5rem",
                  borderRadius: "100px",
                  textTransform: "uppercase",
                  backgroundColor: anomaly.type === "danger" ? "#fef2f2" : anomaly.type === "warning" ? "#fffbeb" : "#f0fdf4",
                  color: anomaly.type === "danger" ? "var(--danger)" : anomaly.type === "warning" ? "var(--warning)" : "var(--success)"
                }}>
                  {anomaly.type === "danger" ? "High" : anomaly.type === "warning" ? "Medium" : "Low"}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {anomalies.length > 0 && (
        <div style={{ padding: "1rem", borderTop: "1px solid var(--border)", textAlign: "left" }}>
          <button style={{ background: "none", border: "none", color: "var(--primary)", fontSize: "0.75rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.25rem", cursor: "pointer", padding: 0 }}>
            View all <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </button>
        </div>
      )}
    </div>
  );
}
