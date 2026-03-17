#!/usr/bin/env python3
"""
Service de prédiction régime de marché — process persistant.

Communique avec Node.js via stdin/stdout (JSON lines).
Charge le modèle XGBoost une seule fois au démarrage.
Chaque requête : query DB → pivot features → normalise → prédit.

Protocole:
  stdin:  {"action": "predict", "horizon": "15m"}
  stdout: {"regime": "actif", "proba": 0.87, "timestamp": "2026-03-15T12:00:00Z", "stale": false}
"""

import sys
import json
import os
import xgboost as xgb
import numpy as np

try:
    import psycopg2
except ImportError:
    # pg8000 as fallback
    import pg8000
    psycopg2 = None

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://trade:trade_secret@172.17.0.1:5432/trade')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '../../notebooks/models'))

SYMBOLS = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SHIBUSDT']
SYMBOL_PREFIXES = ['sol', 'btc', 'eth', 'doge', 'shib']
FEATURE_NAMES = [
    'd1_short', 'd2_short', 'd1_mid', 'd2_mid', 'd1_long', 'd2_long',
    'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
    'atr', 'ma_deviation', 'integral_deviation'
]

# 70 features in the exact order the model was trained on
FEATURE_COLS = []
for prefix in SYMBOL_PREFIXES:
    for feat in FEATURE_NAMES:
        FEATURE_COLS.append(f'{prefix}_{feat}')

# Max staleness before we consider data too old (minutes)
MAX_STALE_MINUTES = 10

models = {}
norm_stats = {}  # {feature_name: {mean, std}}


def connect_db():
    """Connect to PostgreSQL with timeout."""
    if psycopg2:
        return psycopg2.connect(DATABASE_URL, connect_timeout=5)
    else:
        from urllib.parse import urlparse
        parsed = urlparse(DATABASE_URL)
        return pg8000.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip('/'),
            timeout=5
        )


def load_models():
    """Load XGBoost regime models for all horizons."""
    for horizon in ['5m', '15m', '1h']:
        path = os.path.join(MODELS_DIR, f'regime_{horizon}.json')
        if os.path.exists(path):
            model = xgb.XGBClassifier()
            model.load_model(path)
            models[horizon] = model
            log(f'Model loaded: regime_{horizon}')
        else:
            log(f'Model not found: {path}')


def load_norm_stats():
    """Load normalization stats from JSON file or DB."""
    global norm_stats

    # Try JSON file first (faster, works without DB)
    norm_path = os.path.join(MODELS_DIR, 'norm_stats.json')
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_stats = json.load(f)
        log(f'Norm stats loaded from file: {len(norm_stats)} features')
        return

    # Fallback to DB
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute('SELECT feature, mean, std FROM dataset_wide_5_norm')
        rows = cur.fetchall()
        for row in rows:
            norm_stats[row[0]] = {'mean': float(row[1]), 'std': float(row[2])}
        cur.close()
        conn.close()
        log(f'Norm stats loaded from DB: {len(norm_stats)} features')

        # Cache to file for next time
        with open(norm_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        log(f'Norm stats cached to {norm_path}')
    except Exception as e:
        log(f'Failed to load norm stats: {e} — predictions will use raw features')


def get_latest_features():
    """
    Query the latest features from DB for all 5 cryptos and pivot into 70 columns.
    Returns (feature_vector, timestamp, stale).
    """
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Get the latest time that has data for all 5 symbols
        cur.execute("""
            SELECT f.time,
                   f.symbol, f.d1_short, f.d2_short, f.d1_mid, f.d2_mid,
                   f.d1_long, f.d2_long, f.rsi, f.macd, f.macd_signal,
                   f.bollinger_upper, f.bollinger_lower, f.atr,
                   f.ma_deviation, f.integral_deviation
            FROM features f
            WHERE f.time = (
                SELECT MAX(f1.time) FROM features f1
                WHERE f1.symbol = 'SOLUSDT'
                AND EXISTS (SELECT 1 FROM features f2 WHERE f2.time = f1.time AND f2.symbol = 'BTCUSDT')
                AND EXISTS (SELECT 1 FROM features f3 WHERE f3.time = f1.time AND f3.symbol = 'ETHUSDT')
                AND EXISTS (SELECT 1 FROM features f4 WHERE f4.time = f1.time AND f4.symbol = 'DOGEUSDT')
                AND EXISTS (SELECT 1 FROM features f5 WHERE f5.time = f1.time AND f5.symbol = 'SHIBUSDT')
            )
            AND f.symbol IN ('SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'SHIBUSDT')
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if len(rows) < 5:
            return None, None, True

        # Organize by symbol
        by_symbol = {}
        timestamp = None
        for row in rows:
            time_val = row[0]
            symbol = row[1]
            features = [float(v) if v is not None else 0.0 for v in row[2:]]
            by_symbol[symbol] = features
            if timestamp is None:
                timestamp = str(time_val)

        # Build 70-feature vector in correct order
        vector = []
        for symbol in SYMBOLS:
            if symbol in by_symbol:
                vector.extend(by_symbol[symbol])
            else:
                vector.extend([0.0] * 14)

        # Check staleness
        from datetime import datetime, timezone
        try:
            if hasattr(rows[0][0], 'timestamp'):
                ts = rows[0][0].timestamp()
            else:
                ts = datetime.fromisoformat(str(rows[0][0]).replace('Z', '+00:00')).timestamp()
            age_minutes = (datetime.now(timezone.utc).timestamp() - ts) / 60
            stale = age_minutes > MAX_STALE_MINUTES
        except Exception:
            stale = False

        return vector, timestamp, stale

    except Exception as e:
        log(f'DB error: {e}')
        return None, None, True


def normalize(vector):
    """Normalize feature vector using stored mean/std."""
    if not norm_stats:
        return vector

    normalized = []
    for i, col_name in enumerate(FEATURE_COLS):
        val = vector[i]
        if col_name in norm_stats:
            mean = norm_stats[col_name]['mean']
            std = norm_stats[col_name]['std']
            if std > 0:
                val = (val - mean) / std
            else:
                val = 0.0
        normalized.append(val)
    return normalized


def predict(horizon):
    """Run full prediction pipeline for a given horizon."""
    if horizon not in models:
        return {'error': f'Model not loaded for {horizon}'}

    vector, timestamp, stale = get_latest_features()
    if vector is None:
        return {'error': 'No features available', 'regime': 'actif', 'proba': 0.5, 'stale': True}

    # Normalize
    normalized = normalize(vector)

    # Predict
    X = np.array([normalized], dtype=np.float32)
    proba = models[horizon].predict_proba(X)[0]
    pred = int(models[horizon].predict(X)[0])

    regime = 'calme' if pred == 0 else 'actif'
    confidence = float(proba[pred])

    return {
        'regime': regime,
        'proba': round(confidence, 4),
        'proba_calme': round(float(proba[0]), 4),
        'proba_actif': round(float(proba[1]), 4),
        'horizon': horizon,
        'timestamp': timestamp,
        'stale': stale,
    }


def log(msg):
    """Log to stderr (stdout is for JSON protocol)."""
    print(f'[predict] {msg}', file=sys.stderr, flush=True)


def main():
    log('Starting predict service...')
    load_models()
    load_norm_stats()
    log(f'Ready. Models: {list(models.keys())}')

    # Signal ready
    print(json.dumps({'status': 'ready', 'models': list(models.keys())}), flush=True)

    # Read commands from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
            action = cmd.get('action', 'predict')

            if action == 'predict':
                horizon = cmd.get('horizon', '15m')
                result = predict(horizon)
                print(json.dumps(result), flush=True)

            elif action == 'predict_all':
                results = {}
                for h in models:
                    results[h] = predict(h)
                print(json.dumps(results), flush=True)

            elif action == 'ping':
                print(json.dumps({'status': 'ok'}), flush=True)

            else:
                print(json.dumps({'error': f'Unknown action: {action}'}), flush=True)

        except json.JSONDecodeError:
            print(json.dumps({'error': 'Invalid JSON'}), flush=True)
        except Exception as e:
            log(f'Error: {e}')
            print(json.dumps({'error': str(e)}), flush=True)


if __name__ == '__main__':
    main()
