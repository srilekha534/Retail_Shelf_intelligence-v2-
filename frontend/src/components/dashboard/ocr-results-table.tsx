import React, { useState } from "react";
import styles from "./AnomalyPanel.module.css";

interface OCRProduct {
  name: string;
  ocr_confidence: number;
  detection_confidence: number;
  bbox: number[];
  all_texts: string[];
}

interface OCRResultsTableProps {
  ocrResults: OCRProduct[];
}

export function OCRResultsTable({ ocrResults }: OCRResultsTableProps) {
  const [filter, setFilter] = useState("all");

  const filteredResults = ocrResults.filter(r => {
    if (filter === "identified") return r.name !== "Unknown" && r.name !== "";
    if (filter === "unidentified") return r.name === "Unknown" || r.name === "";
    return true;
  });

  return (
    <div className={styles.panel} style={{ gridColumn: "1 / -1", overflow: "hidden", marginTop: "1.5rem" }}>
      <div className={styles.panelHeader} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div className={styles.title}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg>
          OCR Extraction Results
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button 
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem", borderRadius: "4px", border: "1px solid var(--border)", background: filter === "all" ? "var(--primary)" : "transparent", color: filter === "all" ? "white" : "var(--text-primary)", cursor: "pointer" }}
            onClick={() => setFilter("all")}
          >All</button>
          <button 
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem", borderRadius: "4px", border: "1px solid var(--border)", background: filter === "identified" ? "var(--primary)" : "transparent", color: filter === "identified" ? "white" : "var(--text-primary)", cursor: "pointer" }}
            onClick={() => setFilter("identified")}
          >Identified</button>
          <button 
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem", borderRadius: "4px", border: "1px solid var(--border)", background: filter === "unidentified" ? "var(--primary)" : "transparent", color: filter === "unidentified" ? "white" : "var(--text-primary)", cursor: "pointer" }}
            onClick={() => setFilter("unidentified")}
          >Unidentified</button>
        </div>
      </div>
      
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", backgroundColor: "var(--bg-secondary)", textAlign: "left" }}>
              <th style={{ padding: "0.75rem 1rem", fontWeight: 600, color: "var(--text-secondary)" }}>Detected Product</th>
              <th style={{ padding: "0.75rem 1rem", fontWeight: 600, color: "var(--text-secondary)" }}>Extracted OCR Text</th>
              <th style={{ padding: "0.75rem 1rem", fontWeight: 600, color: "var(--text-secondary)" }}>Confidence</th>
              <th style={{ padding: "0.75rem 1rem", fontWeight: 600, color: "var(--text-secondary)" }}>Bounding Box</th>
            </tr>
          </thead>
          <tbody>
            {filteredResults.length === 0 ? (
              <tr>
                <td colSpan={4} style={{ padding: "2rem", textAlign: "center", color: "var(--text-muted)" }}>
                  No OCR results available.
                </td>
              </tr>
            ) : (
              filteredResults.map((result, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid var(--border)", backgroundColor: idx % 2 === 0 ? "transparent" : "var(--bg-secondary)" }}>
                  <td style={{ padding: "0.75rem 1rem", fontWeight: 500, color: (result.name === "Unknown" || result.name === "") ? "var(--text-muted)" : "var(--text-primary)" }}>
                    {result.name || "Unknown"}
                  </td>
                  <td style={{ padding: "0.75rem 1rem", color: "var(--text-primary)", maxWidth: "300px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {result.all_texts && result.all_texts.length > 0 ? (
                      <div style={{ display: "flex", gap: "0.25rem", flexWrap: "wrap" }}>
                        {result.all_texts.map((text, i) => (
                          <span key={i} style={{ background: "var(--bg-primary)", padding: "0.1rem 0.4rem", borderRadius: "4px", border: "1px solid var(--border)", fontSize: "0.75rem" }}>{text}</span>
                        ))}
                      </div>
                    ) : (
                      <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>No text readable</span>
                    )}
                  </td>
                  <td style={{ padding: "0.75rem 1rem" }}>
                    {result.ocr_confidence > 0 ? (
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{ width: "40px", height: "4px", backgroundColor: "var(--border)", borderRadius: "2px", overflow: "hidden" }}>
                          <div style={{ width: `${result.ocr_confidence * 100}%`, height: "100%", backgroundColor: result.ocr_confidence > 0.7 ? "var(--success)" : "var(--warning)" }}></div>
                        </div>
                        <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>{(result.ocr_confidence * 100).toFixed(0)}%</span>
                      </div>
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>-</span>
                    )}
                  </td>
                  <td style={{ padding: "0.75rem 1rem", fontFamily: "monospace", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                    [{result.bbox.map(b => Math.round(b)).join(", ")}]
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
