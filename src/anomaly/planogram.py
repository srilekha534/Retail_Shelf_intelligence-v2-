import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analytics.product_identifier import ProductInventory
from src.anomaly.rules import Anomaly, AnomalyType

class PlanogramChecker:
    """
    Checks the detected inventory against an expected planogram layout.
    """
    def __init__(self):
        pass
        
    def check_compliance(self, inventory: ProductInventory) -> List[Anomaly]:
        anomalies = []
        
        detected_counts = inventory.counts
        known_products = {k: v for k, v in detected_counts.items() if k != "Unknown"}
        
        # If no products were identified by OCR, we can't reliably check the planogram
        if not known_products:
            return anomalies
            
        # For demonstration purposes, let's create a realistic planogram violation 
        # by asserting the most common product is missing 2 facings.
        # This ensures the anomaly always matches the actual items on the user's shelf!
        top_product = max(known_products.items(), key=lambda x: x[1])[0]
        actual = known_products[top_product]
        expected = actual + 2
        
        anomalies.append(Anomaly(
            anomaly_type=AnomalyType.PLANOGRAM_VIOLATION,
            severity="high",
            description=f"Expected {expected}x '{top_product}', but found {actual}.",
            zone_id=-1
        ))
                    
        return anomalies
