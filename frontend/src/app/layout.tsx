import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import styles from "./Layout.module.css";
import Sidebar from "../components/layout/sidebar";
import { SettingsProvider } from "./settings-context";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Retail Intelligence",
  description: "Shelf Analytics & Stock Optimization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <SettingsProvider>
          <div className={styles.layoutWrapper}>
            <Sidebar />
            
            {/* Main Content Area */}
            <div className={styles.mainContent}>
              <header className={styles.header}>
                <div className={styles.headerTitle}>Shelf Overview</div>
              </header>
              
              <main className={styles.pageContainer}>
                {children}
              </main>
            </div>
          </div>
        </SettingsProvider>
      </body>
    </html>
  );
}
