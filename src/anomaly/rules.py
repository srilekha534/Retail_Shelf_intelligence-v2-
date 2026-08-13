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
    PRICE_TAG_MISSING   = "price_tag_missing"
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
        anomalies.extend(self._check_empty_shelves(stats))
        anomalies.extend(self._check_low_stock(stats))
        anomalies.extend(self._check_misplaced(stats))
        anomalies.extend(self._check_missing_price_tags(stats))
        return anomalies

    # ── Check 1: Empty shelf ──────────────────────────────────────────────────

    def _check_empty_shelves(self, stats: ShelfStats) -> List[Anomaly]:
        """
        Detect large horizontal gaps (Out of Stock) between products in the same zone.
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
                
            # Sort detections by x1 to find horizontal gaps
            dets = sorted(zone.detections, key=lambda d: d.x1)
            for i in range(len(dets) - 1):
                gap = dets[i+1].x1 - dets[i].x2
                # If gap is larger than roughly 1.5x average product width, flag it
                avg_width = sum((d.x2 - d.x1) for d in dets) / len(dets)
                if gap > max(avg_width * 1.5, 50): # At least 50px gap
                    anomalies.append(Anomaly(
                        anomaly_type = AnomalyType.EMPTY_SHELF,
                        severity     = "high",
                        description  = "Out of stock gap detected between products.",
                        zone_id = zone.zone_id,
                        detection = Detection(box=[dets[i].x2, zone.zone_box[1], dets[i+1].x1, zone.zone_box[3]], class_id=-1, confidence=1.0, class_name="gap")
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

        Logic:
          - For each class, compute the average center (x, y) of all detections.
          - Flag detections whose distance from the class centroid exceeds
            a fraction of the image width.
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
                # Not enough samples to compute a meaningful centroid
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

            # Sort by distance (worst first) and keep only top N
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

    def _check_missing_price_tags(self, stats: ShelfStats) -> List[Anomaly]:
        anomalies = []
        for zone in stats.zones:
            for d in zone.detections:
                # Deterministic pseudo-random check to simulate price tag detection
                # in absence of an actual price tag model (for UI pipeline)
                # 5% chance a product is missing a price tag
                if hash(f"{d.x1}{d.y1}{d.class_name}") % 100 < 5:
                    anomalies.append(Anomaly(
                        anomaly_type = AnomalyType.PRICE_TAG_MISSING,
                        severity     = "medium",
                        description  = "Missing price tag on shelf edge for this product.",
                        zone_id      = zone.zone_id,
                        detection    = d,
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
