import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

interface ProductChartProps {
  counts: Record<string, number>;
}

export function ProductChart({ counts }: ProductChartProps) {
  // Convert object to array and sort by count descending
  const data = Object.entries(counts)
    .filter(([name]) => name !== "Unknown") // Filter out unknown if desired
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  if (data.length === 0) {
    return (
      <div style={{
        padding: "2rem",
        textAlign: "center",
        backgroundColor: "var(--bg-primary)",
        borderRadius: "var(--radius-lg)",
        border: "1px solid var(--border)",
        color: "var(--text-secondary)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "300px"
      }}>
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ marginBottom: "1rem", opacity: 0.5 }}>
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
          <polyline points="7.5 4.21 12 6.81 16.5 4.21" />
          <polyline points="7.5 19.79 7.5 14.6 3 12" />
          <polyline points="21 12 16.5 14.6 16.5 19.79" />
          <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
          <line x1="12" y1="22.08" x2="12" y2="12" />
        </svg>
        <p>No identified products to display yet.</p>
      </div>
    );
  }

  // Generate some aesthetic colors
  const colors = ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"];

  return (
    <div style={{
      padding: "1.5rem",
      backgroundColor: "var(--bg-primary)",
      borderRadius: "var(--radius-lg)",
      border: "1px solid var(--border)",
      boxShadow: "var(--shadow-sm)",
      marginTop: "1.5rem"
    }}>
      <div style={{ marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: "var(--primary)" }}>
          <line x1="18" y1="20" x2="18" y2="10" />
          <line x1="12" y1="20" x2="12" y2="4" />
          <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
        <h3 style={{ fontSize: "1.125rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
          Identified Product Quantities
        </h3>
      </div>
      
      <div style={{ width: "100%", height: 300 }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.1)" />
            <XAxis type="number" hide />
            <YAxis 
              type="category" 
              dataKey="name" 
              width={120} 
              axisLine={false} 
              tickLine={false} 
              tick={{ fill: "var(--text-secondary)", fontSize: 12 }} 
            />
            <Tooltip 
              cursor={{ fill: "rgba(255,255,255,0.05)" }}
              contentStyle={{ 
                backgroundColor: "var(--bg-secondary)", 
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--text-primary)"
              }}
              itemStyle={{ color: "var(--primary)", fontWeight: "bold" }}
            />
            <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={24}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
