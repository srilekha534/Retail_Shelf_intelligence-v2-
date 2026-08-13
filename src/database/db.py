import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

def get_connection():
    return sqlite3.connect(cfg.DB_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Detections table
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_products INTEGER,
            avg_confidence REAL,
            processing_time_ms REAL
        )
    ''')
    
    # Inventory table
    c.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            product_name TEXT,
            count INTEGER,
            FOREIGN KEY(detection_id) REFERENCES detections(id)
        )
    ''')
    
    # Anomalies table
    c.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            type TEXT,
            severity TEXT,
            description TEXT,
            zone_id TEXT,
            FOREIGN KEY(detection_id) REFERENCES detections(id)
        )
    ''')
    
    # Safely add columns if they don't exist
    try:
        c.execute("ALTER TABLE detections ADD COLUMN original_image_path TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE detections ADD COLUMN processed_image_path TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE detections ADD COLUMN total_identified INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

def log_detection(response_data: dict, original_image_path: str = None, processed_image_path: str = None):
    conn = get_connection()
    c = conn.cursor()
    
    inv = response_data.get('product_inventory', {})
    total_identified = inv.get('total_identified', 0)
    
    c.execute('''
        INSERT INTO detections (total_products, avg_confidence, processing_time_ms, original_image_path, processed_image_path, total_identified)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        response_data.get('total_products', 0),
        response_data.get('avg_confidence', 0.0),
        response_data.get('processing_time_ms', 0.0),
        original_image_path,
        processed_image_path,
        total_identified
    ))
    
    detection_id = c.lastrowid
    
    # Insert inventory
    counts = inv.get('counts_by_name', {})
    for name, count in counts.items():
        c.execute('''
            INSERT INTO inventory (detection_id, product_name, count)
            VALUES (?, ?, ?)
        ''', (detection_id, name, count))
        
    # Insert anomalies
    for anomaly in response_data.get('anomalies', []):
        c.execute('''
            INSERT INTO anomalies (detection_id, type, severity, description, zone_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            detection_id,
            anomaly.get('type', ''),
            anomaly.get('severity', ''),
            anomaly.get('description', ''),
            str(anomaly.get('zone_id', ''))
        ))
        
    conn.commit()
    conn.close()
    return detection_id

def get_history(limit: int = 50, offset: int = 0):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM detections
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    
    rows = c.fetchall()
    history = []
    
    for row in rows:
        record = dict(row)
        det_id = record['id']
        
        # Get anomalies
        c.execute('SELECT type, severity, description FROM anomalies WHERE detection_id = ?', (det_id,))
        record['anomalies'] = [dict(a) for a in c.fetchall()]
        
        # Get inventory
        c.execute('SELECT product_name, count FROM inventory WHERE detection_id = ?', (det_id,))
        record['inventory'] = [dict(i) for i in c.fetchall()]
        
        history.append(record)
        
    conn.close()
    return history
