"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "../../app/Layout.module.css";

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <span className={styles.logoIcon}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 10h16M4 14h16M4 18h16M4 6h16"/></svg>
        </span>
        Retail AI
      </div>
      <nav>
        <Link href="/" className={`${styles.navItem} ${pathname === "/" ? styles.navItemActive : ""}`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
          Dashboard
        </Link>
        <Link href="/history" className={`${styles.navItem} ${pathname === "/history" ? styles.navItemActive : ""}`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
          History
        </Link>
        <Link href="/settings" className={`${styles.navItem} ${pathname === "/settings" ? styles.navItemActive : ""}`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/></svg>
          Settings
        </Link>
      </nav>
    </aside>
  );
}
