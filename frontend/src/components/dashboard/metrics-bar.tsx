import React from "react";
import styles from "./MetricsBar.module.css";

interface MetricsBarProps {
  totalProducts: number;
  identifiedProducts: number;
  outOfStock: number;
  anomaliesCount: number;
}

export function MetricsBar({ totalProducts, identifiedProducts, outOfStock, anomaliesCount }: MetricsBarProps) {
  const identificationRate = totalProducts > 0 ? Math.round((identifiedProducts / totalProducts) * 100) : 0;
  
  return (
    <div className={styles.metricsGrid}>
      <div className={styles.metricCard}>
        <div className={styles.metricTitle}>
          Total Detected
          <svg className={styles.metricIcon} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
        </div>
        <div className={styles.metricValue}>{totalProducts}</div>
        <div className={`${styles.metricTrend} ${styles.trendNeutral}`}>Products on shelf</div>
      </div>

      <div className={styles.metricCard}>
        <div className={styles.metricTitle}>
          Identified
          <svg className={styles.metricIcon} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        </div>
        <div className={styles.metricValue}>{identifiedProducts}</div>
        <div className={`${styles.metricTrend} ${identificationRate >= 90 ? styles.trendUp : styles.trendDown}`}>
          {identificationRate}% Identification Rate
        </div>
      </div>

      <div className={styles.metricCard}>
        <div className={styles.metricTitle}>
          Out of Stock Gaps
          <svg className={styles.metricIcon} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <div className={styles.metricValue}>{outOfStock}</div>
        <div className={`${styles.metricTrend} ${outOfStock > 0 ? styles.trendDown : styles.trendUp}`}>
          {outOfStock > 0 ? "Requires restocking" : "Stock is healthy"}
        </div>
      </div>

      <div className={styles.metricCard}>
        <div className={styles.metricTitle}>
          Planogram Anomalies
          <svg className={styles.metricIcon} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        </div>
        <div className={styles.metricValue}>{anomaliesCount}</div>
        <div className={`${styles.metricTrend} ${anomaliesCount > 0 ? styles.trendDown : styles.trendUp}`}>
          {anomaliesCount > 0 ? "Check misplacements" : "Perfect compliance"}
        </div>
      </div>
    </div>
  );
}
