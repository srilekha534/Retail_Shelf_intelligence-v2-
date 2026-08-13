import React from "react";
import styles from "./AnomalyPanel.module.css";
import { Anomaly } from "./anomaly-panel";

interface AnomaliesSummaryProps {
  anomalies: Anomaly[];
}

export function AnomaliesSummary({ anomalies }: AnomaliesSummaryProps) {
  const counts = {
    oos: anomalies.filter(a => a.type === "danger" && (a.title.includes("Out of Stock") || a.title.includes("Low Stock"))).length,
    planogram: anomalies.filter(a => a.title.includes("Planogram")).length,
    misplaced: anomalies.filter(a => a.title.includes("Misplaced") && !a.title.includes("Visual Damage")).length,
    damaged: anomalies.filter(a => a.title.includes("Visual Damage")).length,
    priceTag: anomalies.filter(a => a.title.includes("Price Tag")).length,
  };

  const total = anomalies.length;

  return (
    <div className={styles.panel} style={{ height: "100%" }}>
      <div className={styles.panelHeader}>
        <div className={styles.title}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          Anomalies Summary
        </div>
        <div className={styles.badge}>{total}</div>
      </div>
      
      <div style={{ padding: "1rem" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left" }}>
              <th style={{ padding: "0.5rem 0", color: "var(--text-secondary)", fontWeight: 600 }}>Anomaly Type</th>
              <th style={{ padding: "0.5rem 0", color: "var(--text-secondary)", fontWeight: 600, textAlign: "right" }}>Count</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "0.75rem 0", display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{width:"8px", height:"8px", borderRadius:"50%", backgroundColor:"var(--danger)"}}></span>Out of Stock Gap</td>
              <td style={{ padding: "0.75rem 0", textAlign: "right" }}>{counts.oos}</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "0.75rem 0", display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{width:"8px", height:"8px", borderRadius:"50%", backgroundColor:"#ef4444"}}></span>Planogram Violation</td>
              <td style={{ padding: "0.75rem 0", textAlign: "right" }}>{counts.planogram}</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "0.75rem 0", display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{width:"8px", height:"8px", borderRadius:"50%", backgroundColor:"#a855f7"}}></span>Misplaced Product</td>
              <td style={{ padding: "0.75rem 0", textAlign: "right" }}>{counts.misplaced}</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "0.75rem 0", display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{width:"8px", height:"8px", borderRadius:"50%", backgroundColor:"#3b82f6"}}></span>Damaged / Wrong Facings</td>
              <td style={{ padding: "0.75rem 0", textAlign: "right" }}>{counts.damaged}</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "0.75rem 0", display: "flex", alignItems: "center", gap: "0.5rem" }}><span style={{width:"8px", height:"8px", borderRadius:"50%", backgroundColor:"#22c55e"}}></span>Price Tag Missing</td>
              <td style={{ padding: "0.75rem 0", textAlign: "right" }}>{counts.priceTag}</td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td style={{ padding: "1rem 0", fontWeight: 600, color: "var(--primary)" }}>Total Anomalies</td>
              <td style={{ padding: "1rem 0", fontWeight: 600, color: "var(--primary)", textAlign: "right" }}>{total}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
