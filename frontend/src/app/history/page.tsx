"use client";

import React, { useEffect, useState } from "react";
import styles from "../page.module.css";

interface HistoryRecord {
  id: number;
  timestamp: string;
  total_products: number;
  total_identified: number;
  avg_confidence: number;
  processing_time_ms: number;
  original_image_path: string | null;
  processed_image_path: string | null;
  anomalies: any[];
  inventory: any[];
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch("/api/history");
      if (!response.ok) throw new Error("Failed to fetch history");
      const data = await response.json();
      setHistory(data.history);
    } catch (err) {
      console.error(err);
      setError("Could not load historical analyses.");
    } finally {
      setLoading(false);
    }
  };

  const clearAllHistory = async () => {
    if (!confirm("Are you sure you want to permanently delete all history?")) return;
    try {
      const response = await fetch("/api/history", { method: "DELETE" });
      if (response.ok) {
        setHistory([]);
        setExpandedId(null);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to clear history");
    }
  };

  const deleteRecord = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!confirm("Delete this analysis record?")) return;
    try {
      const response = await fetch(`/api/history/${id}`, { method: "DELETE" });
      if (response.ok) {
        setHistory(prev => prev.filter(r => r.id !== id));
        if (expandedId === id) setExpandedId(null);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to delete record");
    }
  };

  const getIdentificationRate = (identified: number, total: number) => {
    if (total === 0) return 0;
    return ((identified / total) * 100).toFixed(1);
  };

  if (loading) return <div className={styles.pageHeader}>Loading history...</div>;
  if (error) return <div className={styles.pageHeader} style={{color: "var(--danger)"}}>{error}</div>;

  return (
    <div>
      <div className={styles.pageHeader} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <h1 className={styles.pageTitle}>Analysis History</h1>
          <p className={styles.pageSubtitle}>Review previous shelf analyses and their results.</p>
        </div>
        {history.length > 0 && (
          <button 
            className={styles.toggleBtn} 
            onClick={clearAllHistory} 
            style={{ borderColor: "var(--danger)", color: "var(--danger)" }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            Clear All History
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className={styles.settingsPanel} style={{ textAlign: "center", padding: "4rem 2rem" }}>
          <p style={{ color: "var(--text-secondary)" }}>No history available. Upload an image to start.</p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {history.map((record) => {
            const isExpanded = expandedId === record.id;
            const oosGaps = record.anomalies.filter((a) => a.type === "empty_shelf").length;
            const planogramAnomalies = record.anomalies.filter((a) => a.type === "planogram_violation").length;
            
            return (
              <div key={record.id} className={styles.settingsPanel} style={{ marginBottom: 0, transition: "all 0.2s" }}>
                <div 
                  style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }}
                  onClick={() => setExpandedId(isExpanded ? null : record.id)}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
                    <div style={{ width: "80px", height: "60px", backgroundColor: "var(--bg-primary)", borderRadius: "var(--radius-sm)", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      {record.processed_image_path ? (
                        <img src={record.processed_image_path.replace('data/history_images/', '/history-images/')} alt="Thumbnail" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "cover" }} />
                      ) : (
                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>No Image</span>
                      )}
                    </div>
                    <div>
                      <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.25rem" }}>
                        {new Date(record.timestamp + "Z").toLocaleString()}
                      </h4>
                      <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
                        {record.total_products} detected • {getIdentificationRate(record.total_identified, record.total_products)}% identified
                      </p>
                    </div>
                  </div>
                  
                  <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
                    {oosGaps > 0 && (
                      <span style={{ fontSize: "0.75rem", backgroundColor: "#fef2f2", color: "var(--danger)", padding: "0.25rem 0.5rem", borderRadius: "100px", fontWeight: 600 }}>
                        {oosGaps} OOS
                      </span>
                    )}
                    {planogramAnomalies > 0 && (
                      <span style={{ fontSize: "0.75rem", backgroundColor: "#fffbeb", color: "var(--warning)", padding: "0.25rem 0.5rem", borderRadius: "100px", fontWeight: 600 }}>
                        {planogramAnomalies} Misplaced
                      </span>
                    )}
                    <button 
                      onClick={(e) => deleteRecord(e, record.id)}
                      style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer", padding: "0.25rem", borderRadius: "50%", transition: "background 0.2s" }}
                      title="Delete record"
                      onMouseOver={(e) => e.currentTarget.style.backgroundColor = "rgba(239, 68, 68, 0.1)"}
                      onMouseOut={(e) => e.currentTarget.style.backgroundColor = "transparent"}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                    <svg style={{ transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
                  </div>
                </div>

                {isExpanded && (
                  <div style={{ marginTop: "1.5rem", paddingTop: "1.5rem", borderTop: "1px solid var(--border)", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
                    <div>
                      <h5 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>Image Result</h5>
                      {record.processed_image_path ? (
                         <img src={record.processed_image_path.replace('data/history_images/', '/history-images/')} alt="Processed Result" style={{ width: "100%", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }} />
                      ) : (
                         <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>Image not saved for this run.</p>
                      )}
                    </div>
                    <div>
                      <h5 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>Details</h5>
                      <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}><span>Processing Time</span> <span>{record.processing_time_ms.toFixed(0)} ms</span></div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}><span>Avg Confidence</span> <span>{(record.avg_confidence * 100).toFixed(1)}%</span></div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}><span>Total Products</span> <span>{record.total_products}</span></div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}><span>Identified Products</span> <span>{record.total_identified}</span></div>
                      </div>
                      
                      {record.anomalies.length > 0 && (
                        <>
                          <h5 style={{ fontSize: "0.875rem", fontWeight: 600, marginTop: "1.5rem", marginBottom: "0.75rem" }}>Anomalies</h5>
                          <ul style={{ fontSize: "0.875rem", color: "var(--text-secondary)", paddingLeft: "1.25rem", margin: 0 }}>
                            {record.anomalies.map((a, i) => (
                              <li key={i} style={{ marginBottom: "0.25rem" }}>
                                <strong>{a.type}:</strong> {a.description}
                              </li>
                            ))}
                          </ul>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
