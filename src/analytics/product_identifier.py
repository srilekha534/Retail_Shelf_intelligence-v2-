# src/analytics/product_identifier.py
#
# PURPOSE:
#   Identify individual products by name using OCR on cropped detection regions.
#   Groups and counts each unique product found on the shelf.
#
# OCR ENGINE: PaddleOCR (GPU accelerated, DB + SVTR_LCNet)
#
# PIPELINE (per crop):
#   1. Crop detection region with padding
#   2. Preprocess: Resize → Denoise → CLAHE → Sharpen → Binarize → Normalize
#   3. PaddleOCR (GPU): Text Detection (DB) → Text Recognition (SVTR_LCNet)
#   4. Post-process: Clean → Lowercase → Remove specials → Fuzzy match → Group
#   5. Count unique products

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
import cv2

import config as cfg

# Use rapidfuzz for ~10× faster fuzzy matching than difflib
try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio using rapidfuzz (fast) or difflib (fallback)."""
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class IdentifiedProduct:
    """A single product with its extracted name and detection info."""
    name: str
    confidence: float         # OCR confidence for the name
    detection_confidence: float  # YOLO detection confidence
    bbox: List[float]         # [x1, y1, x2, y2]
    all_texts: List[str]      # all OCR text found in this crop
    crop_index: int           # index in original detections list

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ocr_confidence": round(self.confidence, 3),
            "detection_confidence": round(self.detection_confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox],
            "all_texts": self.all_texts,
        }


@dataclass
class ProductInventory:
    """Full product inventory extracted from a shelf image."""
    products: List[IdentifiedProduct]
    counts: Dict[str, int]            # {product_name: count}
    total_identified: int             # products with a readable name
    total_unidentified: int           # products where OCR found nothing
    unique_products: int              # number of distinct product names

    def to_dict(self) -> dict:
        # Sort counts by frequency (most common first)
        sorted_counts = dict(
            sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        )
        return {
            "counts": sorted_counts,
            "unique_products": self.unique_products,
            "total_identified": self.total_identified,
            "total_unidentified": self.total_unidentified,
            "products": [p.to_dict() for p in self.products],
        }

    def summary(self) -> str:
        lines = [
            f"Product Inventory: {self.unique_products} unique products, "
            f"{self.total_identified} identified, "
            f"{self.total_unidentified} unidentified",
            "",
        ]
        for name, count in sorted(
            self.counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)


class ProductIdentifier:
    """
    Identifies products by name using PaddleOCR on cropped detection regions.

    Pipeline:
      1. Take YOLO detection bounding boxes
      2. Crop each detection region from the image (with padding)
      3. Preprocess: Resize → Denoise → CLAHE → Sharpen → Binarize → Normalize
      4. Run PaddleOCR (GPU accelerated, DB + SVTR_LCNet)
      5. Post-process: Clean text → lowercase → remove specials → fuzzy match catalog
      6. Group similar names via rapidfuzz and count by product
    """

    # ── Noise patterns compiled once ─────────────────────────────────────
    _NOISE_PATTERNS = [
        re.compile(r'^[\d\s.,;:£$€₹%/:×xX\-]+$'),           # pure numbers / symbols
        re.compile(r'^\d+[,.]\d+$'),                         # European prices: 6,73 or 3.99
        re.compile(r'^\d+[,.]?\s*[A-Za-z]{1,2}\d*$'),       # codes with commas: 5,G6, 5G8
        re.compile(r'^\d+\s*[gG][rR]?[mM]?[sS]?$'),         # weights: 500g, 250gms
        re.compile(r'^\d+\s*[mM][lL]$'),                      # volumes: 250ml
        re.compile(r'^\d+\s*[lL]$'),                           # litres: 2l
        re.compile(r'^\d+\s*[kK][gG]$'),                      # kilos: 1kg
        re.compile(r'^\d+\s*[oO][zZ]$'),                      # ounces: 12oz
        re.compile(r'^\d+\s*[xX×]\s*\d+'),                    # multipacks: 6x250
        re.compile(r'^[£$€₹]\s*\d'),                          # prices: $3.99
        re.compile(r'^\d+[pP]$'),                              # pence: 99p
        re.compile(r'^\d+\.\d{2}$'),                           # bare prices: 3.99
        re.compile(r'^(net|wt|vol|qty|exp|mfg|mrp|best before|use by)', re.I),
        re.compile(r'^www\.|\\.com|\\.in|\\.co', re.I),          # URLs
        re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'),      # dates
        re.compile(r'^[A-Z]{1,2}\d{4,}$'),                    # barcodes / serial numbers
        re.compile(r'^[A-Z]\d{1,2}$'),                        # short shelf codes: J9, A4
        re.compile(r'^\d+\s*[A-Za-z]$'),                      # codes: 5G, 4A
        re.compile(r'^[A-Za-z]\d+$'),                          # codes: G6, J9
        re.compile(r'^\d+[A-Za-z]\d+'),                        # codes: 5G6, 5G8
        re.compile(r'^[A-Z]{1,3}\s+\d{1,2}[A-Z]?$', re.I),   # shelf labels: COS 4A
        re.compile(r'^\d{3,}$'),                               # raw numbers: 4078, 509
        re.compile(r'^[A-Za-z]{1,2}$'),                        # 1-2 letter gibberish
        re.compile(r'^\d+[A-Za-z]{1,2}\s+[A-Za-z]{1,4}$'),   # codes with suffix: 5G8 Lacl
    ]

    def __init__(
        self,
        ocr_languages: List[str] = None,
        min_ocr_confidence: float = None,
        crop_padding: float = 0.05,
        min_name_length: int = 2,
        max_products: int = None,
        gpu: bool = None,
        fuzzy_threshold: float = 0.78,
    ):
        """
        Args:
            ocr_languages: languages for PaddleOCR
            min_ocr_confidence: minimum OCR confidence to accept text
            crop_padding: fraction of box size to pad when cropping (0.30 = 30%)
            min_name_length: minimum character length for a valid product name
            max_products: max number of products to OCR (largest first)
            gpu: use GPU for OCR (defaults to True if cfg.DEVICE is cuda)
            fuzzy_threshold: similarity ratio above which two names are merged
        """
        self.ocr_languages = ocr_languages if ocr_languages is not None else cfg.OCR_LANGUAGES
        self.min_ocr_confidence = min_ocr_confidence if min_ocr_confidence is not None else cfg.OCR_CONFIDENCE
        self.crop_padding = crop_padding
        self.min_name_length = min_name_length
        self.max_products = max_products if max_products is not None else getattr(cfg, "MAX_OCR_PRODUCTS", 300)
        self._paddle_ocr = None
        self._gpu = gpu if gpu is not None else (cfg.DEVICE == "cuda")
        self.fuzzy_threshold = fuzzy_threshold

        # Brand Catalog for spelling correction and fuzzy mapping
        # NOTE: Keywords should be at least 4 chars to avoid false matches
        self.brand_catalog = {
            "Coca-Cola": ["coca", "cola", "coke", "ccca", "c0la", "ccba"],
            "Fanta": ["fanta", "fata", "fant"],
            "Sprite": ["sprite", "sprt", "spri"],
            "Pepsi": ["pepsi", "ppsi", "peps"],
            "Dr Pepper": ["pepper", "dr pepper", "dr.pepper"],
            "Minute Maid": ["minute maid", "minute", "minut"],
            "Tim Hortons": ["horton", "hortons", "timhorton"],
            "Profissimo": ["profissimo", "profis"],
            "Pure": ["pure"],
            "Calggy": ["calggy"],
            "Advil": ["advil", "advl", "adil", "advi"],
            "Balea": ["balea", "bale", "balia"],
            "Colgate": ["colgate", "colgat", "colg"],
            "Dial": ["dial", "dlal", "dlall"],
            "Equate": ["equate", "equale", "equa"],
            "Irish Spring": ["irish spring", "irish", "irishl"],
            "Ivory": ["ivory", "ivorv", "ivor"],
            "Ibuprofen": ["ibuprofen", "ibuprolon", "ibuprolcw", "ibup"],
            "Olay": ["olay"],
            "Raid": ["raid"],
            "Tylenol": ["tylenol", "tyenol", "tylen", "tyle"],
            "Fixodent": ["fixodent", "fhrodent", "fitodeut", "fixo", "fixod"],
            "Crest": ["crest", "crst"],
            "Sensodyne": ["sensodyne", "senso", "sensod"],
            "Dove": ["dove"],
            "Nivea": ["nivea"],
            "Gillette": ["gillette", "gill", "gillet"],
        }

    def _match_catalog(self, text: str) -> Optional[str]:
        """Match text against brand catalog using keyword + fuzzy similarity."""
        text_lower = text.lower().strip()

        # 1. Keyword matching
        for brand, keywords in self.brand_catalog.items():
            for kw in keywords:
                if kw in text_lower:
                    return brand

        # 2. Fuzzy similarity matching
        for brand in self.brand_catalog.keys():
            ratio = _similarity(brand.lower(), text_lower)
            if ratio >= 0.65:
                return brand

        return None

    @property
    def paddle_ocr(self):
        """Lazy-load PaddleOCR engine."""
        if self._paddle_ocr is None:
            from src.analytics.paddle_ocr import PaddleOCREngine
            self._paddle_ocr = PaddleOCREngine()
        return self._paddle_ocr

    # ── Preprocessing pipeline (matching architecture diagram) ────────────

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for a product label crop.

        Steps (matching the architecture diagram):
          1. Resize (short side = 512)
          2. Denoise (FastNlMeans)
          3. Contrast Enhance (CLAHE)
          4. Sharpen (Unsharp Mask)
          5. Binarization (Adaptive) — only for grayscale strategy
          6. Normalize
        """
        height, width = crop.shape[:2]

        # 1. Resize — ensure short side is at least 512px for readable text
        short_side = min(height, width)
        if short_side < 512:
            scale = 512.0 / max(short_side, 1)
            new_w = max(int(width * scale), 1)
            new_h = max(int(height * scale), 1)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. Denoise (FastNlMeans — preserves edges while removing noise)
        denoised = cv2.fastNlMeansDenoisingColored(crop, None, 6.0, 6.0, 7, 21)

        # 3. Contrast Enhance (CLAHE on L channel in LAB color space)
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # 4. Sharpen (Unsharp Mask)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        # 5. Normalize to [0, 255] range
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

        return normalized.astype(np.uint8)

    # ── OCR execution ────────────────────────────────────────────────────

    def _run_ocr(self, crop: np.ndarray) -> tuple:
        """
        Run PaddleOCR on a preprocessed crop.
        Returns (ocr_results, best_name, best_conf)
        """
        # Preprocess the crop
        preprocessed = self._preprocess_crop(crop)

        # Run PaddleOCR
        ocr_results = self.paddle_ocr.read_crop(preprocessed)

        if ocr_results:
            name, conf = self._pick_best_name(ocr_results)
            if name:
                return ocr_results, name, conf

        # Fallback: try on raw upscaled crop without heavy preprocessing
        height, width = crop.shape[:2]
        if min(height, width) < 256:
            scale = 256.0 / max(min(height, width), 1)
            raw_upscaled = cv2.resize(crop, (max(int(width * scale), 1), max(int(height * scale), 1)), interpolation=cv2.INTER_CUBIC)
        else:
            raw_upscaled = crop

        ocr_results_raw = self.paddle_ocr.read_crop(raw_upscaled)
        if ocr_results_raw:
            name, conf = self._pick_best_name(ocr_results_raw)
            if name:
                return ocr_results_raw, name, conf

        # Combine all results
        all_results = (ocr_results or []) + (ocr_results_raw or [])
        if all_results:
            name, conf = self._pick_best_name(all_results)
            return all_results, name, conf

        return [], "", 0.0

    # ── Crop helper ──────────────────────────────────────────────────────

    def _crop_detection(
        self,
        image: np.ndarray,
        bbox: List[float],
    ) -> np.ndarray:
        """Crop a detection region from the image with padding."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)

        cx1 = max(0, int(x1) - pad_x)
        cy1 = max(0, int(y1) - pad_y)
        cx2 = min(w, int(x2) + pad_x)
        cy2 = min(h, int(y2) + pad_y)

        return image[cy1:cy2, cx1:cx2]

    # ── Post-processing: name cleaning ───────────────────────────────────

    def _clean_name(self, text: str) -> str:
        """
        Clean and normalise extracted product name.

        Post-processing steps (matching architecture diagram):
          - Remove extra spaces
          - Lowercase normalize
          - Remove special characters
          - Strip numeric noise
        """
        # Remove excessive whitespace
        name = " ".join(text.split())

        # Remove trailing prices or decimal strings (e.g. "Coffee8.09" → "Coffee")
        name = re.sub(r'[\d.,]+$', '', name)
        # Remove leading numbers/barcodes
        name = re.sub(r'^[\d.,]+', '', name)
        # Remove common noise words regardless of case
        noise_words = ['rollback', 'net', 'wt', 'vol', 'qty', 'exp', 'mfg', 'mrp']
        pattern = re.compile(r'\b(' + '|'.join(noise_words) + r')\b', flags=re.IGNORECASE)
        name = pattern.sub('', name)
        # Remove standalone numbers
        name = re.sub(r'\b\d+\b', '', name)
        # Remove special characters but keep hyphens and spaces
        name = re.sub(r'[^\w\s\-]', '', name)

        name = name.strip()

        # Reject if it matches any full-string noise pattern
        for pat in self._NOISE_PATTERNS:
            if pat.match(name):
                return ""

        # Remove leading/trailing punctuation
        name = name.strip(".,;:!?|/\\()[]{}\"'`~@#^&*_=+ ")

        # Skip short strings (must be at least 3 chars)
        if len(name) < 3:
            return ""

        # Reject names that are mostly digits (more than 50% digits = probably a code)
        digit_count = sum(1 for c in name if c.isdigit())
        if len(name) > 0 and digit_count / len(name) > 0.5:
            return ""

        # Capitalise properly: ALL-CAPS words > 3 chars → Title Case
        if name.isupper() and len(name) > 3:
            name = name.title()

        return name.strip()

    # ── Name extraction ──────────────────────────────────────────────────

    def _pick_best_name(self, ocr_results: list) -> Tuple[str, float]:
        """
        From OCR results for a single crop, build the best product name.
        Uses a predefined brand catalog to correct typos and map fuzzy detections.
        """
        # 1. First, try catalog matching on ALL OCR fragments (even lower confidence ones)
        for _, text, conf in ocr_results:
            if conf >= 0.15:  # lower threshold allowed for catalog matching
                cleaned = self._clean_name(text)
                if cleaned:
                    matched_brand = self._match_catalog(cleaned)
                    if matched_brand:
                        # Success! Found a catalog match
                        return (matched_brand, max(conf, 0.85))  # boost confidence on catalog match

        # 2. Fallback to standard cleaning/concatenation if no catalog match is found
        candidates = []

        for bbox_points, text, conf in ocr_results:
            if conf < self.min_ocr_confidence:
                continue
            cleaned = self._clean_name(text)
            if len(cleaned) >= 3:  # minimum 3 chars for a valid fragment
                candidates.append((cleaned, conf, len(cleaned)))

        if not candidates:
            return ("", 0.0)

        # Sort by text length descending, then confidence descending
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

        # Take the single longest fragment as the primary name
        best_name = candidates[0][0]
        best_conf = candidates[0][1]

        # If the best name is short (≤ 5 chars) and there's a second
        # candidate, try combining them for a fuller product name.
        if len(best_name) <= 5 and len(candidates) >= 2:
            second = candidates[1][0]
            if second.lower() != best_name.lower():
                best_name = f"{best_name} {second}"
                best_conf = (best_conf + candidates[1][1]) / 2

        return (best_name, best_conf)

    # ── Fuzzy name grouping ──────────────────────────────────────────────

    def _fuzzy_group_names(self, name_counter: Counter) -> Counter:
        """
        Merge product names that are near-identical due to OCR typos.

        e.g. "Coca Cola", "Coca-Cola", "Coca cola" → keep the most
        frequent spelling and sum all counts.

        Uses rapidfuzz for ~10× faster matching than difflib.
        """
        if len(name_counter) <= 1:
            return name_counter

        names = list(name_counter.keys())
        merged = Counter()
        used = set()

        # Sort by frequency so the most common spelling is the canonical one
        sorted_names = sorted(names, key=lambda n: name_counter[n], reverse=True)

        for name in sorted_names:
            if name in used:
                continue

            canonical = name
            total_count = name_counter[name]

            for other in sorted_names:
                if other == name or other in used:
                    continue

                ratio = _similarity(canonical.lower(), other.lower())
                if ratio >= self.fuzzy_threshold:
                    total_count += name_counter[other]
                    used.add(other)

            merged[canonical] = total_count
            used.add(name)

        return merged

    # ── Main identification pipeline ─────────────────────────────────────

    def identify(
        self,
        image: np.ndarray,
        detections: List[dict],
    ) -> ProductInventory:
        """
        Identify products in an image by name.

        Processes at most `max_products` detections (largest bounding boxes
        first, since bigger products have more readable labels).

        Args:
            image: RGB numpy array of the shelf image
            detections: list of detection dicts with x1/y1/x2/y2 keys

        Returns:
            ProductInventory with names, counts, and per-product details
        """
        products = []
        name_counter: Counter = Counter()
        unidentified = 0

        # Sort detections by bounding box area (largest first) and take top N
        indexed_dets = list(enumerate(detections))
        indexed_dets.sort(
            key=lambda x: (x[1]["x2"] - x[1]["x1"]) * (x[1]["y2"] - x[1]["y1"]),
            reverse=True,
        )
        selected = indexed_dets[:self.max_products]
        skipped = len(detections) - len(selected)

        for i, det in selected:
            bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]

            # Crop the detection region
            crop = self._crop_detection(image, bbox)

            # Skip truly tiny crops
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                unidentified += 1
                continue

            # Run PaddleOCR on the crop
            try:
                ocr_results, name, ocr_conf = self._run_ocr(crop)
            except Exception as e:
                import traceback
                with open("ocr_error.txt", "w") as f:
                    f.write(f"OCR failed for crop {i}: {e}\n")
                    traceback.print_exc(file=f)
                unidentified += 1
                continue

            # Extract all readable text
            all_texts = [
                text.strip()
                for _, text, conf in ocr_results
                if conf >= self.min_ocr_confidence and text.strip()
            ]

            if name:
                name_counter[name] += 1
                products.append(IdentifiedProduct(
                    name=name,
                    confidence=ocr_conf,
                    detection_confidence=det.get("confidence", 0),
                    bbox=bbox,
                    all_texts=all_texts,
                    crop_index=i,
                ))
            else:
                unidentified += 1

        # Count skipped products as unidentified
        unidentified += skipped

        # Fuzzy-group similar product names
        grouped_counts = self._fuzzy_group_names(name_counter)

        # Update product names to their canonical (grouped) form
        canonical_map = {}
        raw_names = list(name_counter.keys())
        for canonical, _ in grouped_counts.items():
            for raw in raw_names:
                ratio = _similarity(canonical.lower(), raw.lower())
                if ratio >= self.fuzzy_threshold or raw == canonical:
                    canonical_map[raw] = canonical

        for product in products:
            product.name = canonical_map.get(product.name, product.name)

        return ProductInventory(
            products=products,
            counts=dict(grouped_counts),
            total_identified=len(products),
            total_unidentified=unidentified,
            unique_products=len(grouped_counts),
        )
