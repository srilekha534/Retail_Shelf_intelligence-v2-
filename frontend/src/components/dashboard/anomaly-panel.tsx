import React from "react";
import styles from "./AnomalyPanel.module.css";

export interface Anomaly {
  id: string;
  type: "warning" | "danger" | "info";
  title: string;
  description: string;
}

interface AnomalyPanelProps {
  anomalies: Anomaly[];
}

export function AnomalyPanel({ anomalies }: AnomalyPanelProps) {
  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <div className={styles.title}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          Alerts
        </div>
        <div className={styles.badge}>{anomalies.length}</div>
      </div>
      
      <div className={styles.list}>
        {anomalies.length === 0 ? (
          <div className={styles.emptyState}>
            No anomalies detected on the shelf.
          </div>
        ) : (
          anomalies.map((anomaly) => (
            <div key={anomaly.id} className={styles.item}>
              <div className={`${styles.itemIcon} ${
                anomaly.type === "danger" ? styles.iconDanger :
                anomaly.type === "warning" ? styles.iconWarning : styles.iconInfo
              }`}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              </div>
              <div className={styles.itemContent}>
                <div className={styles.itemTitle}>{anomaly.title}</div>
                <div className={styles.itemDesc}>{anomaly.description}</div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
