# src/anomaly/rules.py
#
# PURPOSE:
#   Detect shelf anomalies using simple rule-based logic on ShelfStats.
#   No ML required. Fast, explainable, debuggable.
#
#   Three anomaly types:
#     1. EMPTY_SHELF   — a zone has almost no products
#     2. LOW_STOCK     — a zone has fewer products than expected
#     3. MISPLACED     — a product is far from the cluster of its class
#
# USAGE:
#   from src.anomaly.rules import AnomalyDetector
#   detector = AnomalyDetector()
#   anomalies = detector.detect(shelf_stats)

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg
from src.detection.counter import ShelfStats
from src.detection.detector import Detection


# ── Anomaly types ─────────────────────────────────────────────────────────────

class AnomalyType(str, Enum):
    EMPTY_SHELF         = "empty_shelf"
    LOW_STOCK           = "low_stock"
    MISPLACED           = "misplaced"
    FALLEN_PRODUCT      = "fallen_product"
    PLANOGRAM_VIOLATION = "planogram_violation"


@dataclass
class Anomaly:
    """Represents a detected shelf anomaly."""
    anomaly_type: AnomalyType
    severity:     str            # "low", "medium", "high"
    description:  str
    zone_id:      int = -1       # -1 = whole image
    detection:    Optional[Detection] = None  # the specific product (for misplaced)

    def to_dict(self) -> dict:
        return {
            "type":        self.anomaly_type.value,
            "severity":    self.severity,
            "description": self.description,
            "zone_id":     self.zone_id,
        }


# ── Detector ─────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Rule-based anomaly detector for retail shelves.

    Thresholds are read from config but can be overridden per-instance.
    """

    def __init__(
        self,
        empty_shelf_threshold: Optional[int] = None,
        low_stock_threshold:   Optional[int] = None,
    ):
        self.empty_threshold    = empty_shelf_threshold or cfg.EMPTY_SHELF_MAX_PRODUCTS
        self.low_stock_threshold = low_stock_threshold or cfg.LOW_STOCK_MAX_PRODUCTS

    def detect(self, stats: ShelfStats) -> List[Anomaly]:
        """
        Run all anomaly checks on a ShelfStats object.
        Returns a list of Anomaly objects (may be empty if shelf is fine).
        """
        anomalies = []
        anomalies.extend(self._check_empty_shelves_and_fallen(stats))
        anomalies.extend(self._check_low_stock(stats))
        anomalies.extend(self._check_misplaced(stats))
        return anomalies

    # ── Check 1: Empty shelf & Fallen Products ──────────────────────────────

    def _check_empty_shelves_and_fallen(self, stats: ShelfStats) -> List[Anomaly]:
        """
        Detect large horizontal gaps (Out of Stock) between products.
        Also detects fallen products by comparing aspect ratios within the shelf.
        """
        anomalies = []
        for zone in stats.zones:
            if not zone.detections:
                anomalies.append(Anomaly(
                    anomaly_type = AnomalyType.EMPTY_SHELF,
                    severity     = "high",
                    description  = f"Zone {zone.zone_id} is completely empty.",
                    zone_id = zone.zone_id,
                    detection = Detection(box=[zone.zone_box[0], zone.zone_box[1], zone.zone_box[2], zone.zone_box[3]], class_id=-1, confidence=1.0, class_name="gap")
                ))
                continue
                
            # Cluster detections into horizontal shelves based on their Y-centers
            # We assume products on the same shelf have Y-centers within roughly half a product height of each other.
            avg_height = sum((d.y2 - d.y1) for d in zone.detections) / len(zone.detections)
            y_tolerance = avg_height * 0.6
            
            # Sort by Y-center
            dets_by_y = sorted(zone.detections, key=lambda d: d.center[1])
            
            shelves = []
            current_shelf = [dets_by_y[0]]
            
            for d in dets_by_y[1:]:
                if abs(d.center[1] - current_shelf[-1].center[1]) < y_tolerance:
                    current_shelf.append(d)
                else:
                    shelves.append(current_shelf)
                    current_shelf = [d]
            shelves.append(current_shelf)
            
            # For each shelf, check gaps and fallen products
            for shelf in shelves:
                if len(shelf) < 2:
                    continue
                    
                dets_by_x = sorted(shelf, key=lambda d: d.x1)
                
                # Check for Fallen Products based on aspect ratio
                aspect_ratios = [((d.x2 - d.x1) / max(1, d.y2 - d.y1)) for d in dets_by_x]
                median_ar = sorted(aspect_ratios)[len(aspect_ratios)//2]
                
                for idx, d in enumerate(dets_by_x):
                    ar = aspect_ratios[idx]
                    # If this product is significantly wider relative to its height than the median on this shelf
                    if ar > median_ar * 2.5 and ar > 1.2:
                        anomalies.append(Anomaly(
                            anomaly_type = AnomalyType.FALLEN_PRODUCT,
                            severity     = "medium",
                            description  = "Product appears to have fallen over.",
                            zone_id = zone.zone_id,
                            detection = d
                        ))
                
                # Check for Gaps
                avg_width = sum((d.x2 - d.x1) for d in dets_by_x) / len(dets_by_x)
                
                for i in range(len(dets_by_x) - 1):
                    gap = dets_by_x[i+1].x1 - dets_by_x[i].x2
                    # If gap is larger than 1.5x average product width, it's an out-of-stock gap
                    if gap > max(avg_width * 1.5, 50): # At least 50px gap
                        anomalies.append(Anomaly(
                            anomaly_type = AnomalyType.EMPTY_SHELF,
                            severity     = "high",
                            description  = "Out of stock gap detected on shelf.",
                            zone_id = zone.zone_id,
                            detection = Detection(box=[dets_by_x[i].x2, dets_by_x[i].y1, dets_by_x[i+1].x1, dets_by_x[i].y2], class_id=-1, confidence=1.0, class_name="gap")
                        ))
        return anomalies

    # ── Check 2: Low stock ────────────────────────────────────────────────────

    def _check_low_stock(self, stats: ShelfStats) -> List[Anomaly]:
        """
        Flag zones that are below the low-stock threshold but not empty.
        """
        anomalies = []
        for zone in stats.zones:
            if self.empty_threshold < zone.count <= self.low_stock_threshold:
                anomalies.append(Anomaly(
                    anomaly_type = AnomalyType.LOW_STOCK,
                    severity     = "medium",
                    description  = (
                        f"Zone {zone.zone_id} has low stock "
                        f"({zone.count} products detected)."
                    ),
                    zone_id = zone.zone_id,
                ))
        return anomalies

    # ── Check 3: Misplaced products ───────────────────────────────────────────

    def _check_misplaced(self, stats: ShelfStats) -> List[Anomaly]:
        """
        Detect products that are far from the centroid of their class cluster.
        """
        anomalies = []

        # Group detections by class
        by_class: dict = {}
        all_detections = [d for zone in stats.zones for d in zone.detections]
        for d in all_detections:
            by_class.setdefault(d.class_name, []).append(d)

        # In a single-class setup (where all items are labeled "product"),
        # centroid-based clustering is not meaningful and creates false alerts.
        if len(by_class) <= 1:
            return []

        threshold_px = stats.image_width * 0.60  # 60% of image width

        max_misplaced = 5  # Only report the worst offenders

        for class_name, detections in by_class.items():
            if len(detections) < 3:
                continue

            # Compute centroid
            cx_mean = sum(d.center[0] for d in detections) / len(detections)
            cy_mean = sum(d.center[1] for d in detections) / len(detections)

            # Collect candidates with their distances
            candidates = []
            for d in detections:
                dx = d.center[0] - cx_mean
                dy = d.center[1] - cy_mean
                distance = (dx**2 + dy**2) ** 0.5

                if distance > threshold_px:
                    candidates.append((distance, d))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for distance, d in candidates[:max_misplaced]:
                anomalies.append(Anomaly(
                    anomaly_type = AnomalyType.MISPLACED,
                    severity     = "low",
                    description  = (
                        f"'{class_name}' product may be misplaced "
                        f"(distance from cluster centroid: {distance:.0f}px)."
                    ),
                    zone_id   = -1,
                    detection = d,
                ))

        return anomalies
    def format_report(self, anomalies: List[Anomaly]) -> str:
        """Return a human-readable anomaly report string."""
        if not anomalies:
            return "No anomalies detected. Shelf looks healthy."

        lines = [f"{len(anomalies)} anomaly/anomalies detected:"]
        for a in anomalies:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(a.severity, "⚪")
            lines.append(f"  {icon} [{a.anomaly_type.value}] {a.description}")
        return "\n".join(lines)
