"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface SettingsContextType {
  confidence: number;
  setConfidence: (val: number) => void;
  ocrEnabled: boolean;
  setOcrEnabled: (val: boolean) => void;
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [confidence, setConfidence] = useState<number>(0.25);
  const [ocrEnabled, setOcrEnabled] = useState<boolean>(true);

  // Load from local storage on mount
  useEffect(() => {
    const savedConf = localStorage.getItem("retail_ai_confidence");
    const savedOcr = localStorage.getItem("retail_ai_ocr");
    if (savedConf) setConfidence(parseFloat(savedConf));
    if (savedOcr) setOcrEnabled(savedOcr === "true");
  }, []);

  // Save to local storage on change
  useEffect(() => {
    localStorage.setItem("retail_ai_confidence", confidence.toString());
    localStorage.setItem("retail_ai_ocr", ocrEnabled.toString());
  }, [confidence, ocrEnabled]);

  return (
    <SettingsContext.Provider value={{ confidence, setConfidence, ocrEnabled, setOcrEnabled }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error("useSettings must be used within a SettingsProvider");
  }
  return context;
}
