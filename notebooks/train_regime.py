#!/usr/bin/env python3
"""
Training XGBoost — prédiction du RÉGIME DE MARCHÉ (calme vs actif).

Pivot clé : les features (ATR, Bollinger) prédisent la volatilité, pas la direction.
On utilise cette force pour classifier le régime de marché, puis les règles algo D1/D2
décident la direction de trading.

Labels régime (basés sur volatilité future) :
  0 = calme — ATR futur bas, pas d'opportunité → ne pas trader
  1 = actif — ATR futur élevé → laisser D1/D2 décider

Usage:
    python train_regime.py                     # Lance tout (3 horizons)
    python train_regime.py --horizon 15m       # Un seul horizon
    python train_regime.py --verbose           # Logs détaillés par epoch

Rapide : ~3 minutes pour 3 horizons.
"""

import numpy as np
import os
import sys
import argparse
import subprocess
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LOCAL_DIR, 'data')
MODELS_DIR = os.path.join(LOCAL_DIR, 'models')
GCS_BUCKET = 'gs://trade-ml-bot-data'
HORIZONS = ['5m', '15m', '1h']


def download_data():
    """Télécharge les .npy depuis GCS si pas en local."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(os.path.join(DATA_DIR, 'features.npy')):
        print('✅ Données déjà en local')
        return
    print('📥 Téléchargement depuis GCS...')
    os.system(f'gsutil -m cp {GCS_BUCKET}/prepared/*.npy {DATA_DIR}/')


def load_raw_prices():
    """Charge les prix SOL bruts alignés avec le dataset."""
    prices_path = os.path.join(DATA_DIR, 'sol_prices.csv')
    if not os.path.exists(prices_path):
        print('❌ sol_prices.csv manquant dans data/. Exporte depuis le VPS.')
        sys.exit(1)
    prices = np.loadtxt(prices_path, dtype=np.float32)
    return prices


def compute_future_atr(prices, window, horizon_minutes):
    """
    Calcule l'ATR futur sur une fenêtre donnée.
    Proxy : moyenne des |close[t] - close[t-1]| sur la fenêtre future.
    """
    n = len(prices)
    future_atr = np.full(n, np.nan, dtype=np.float32)

    # True Range proxy : |close[t] - close[t-1]|
    tr = np.abs(np.diff(prices))
    tr = np.concatenate([[0], tr])

    for i in range(n - horizon_minutes - window):
        start = i + horizon_minutes
        end = start + window
        if end <= n:
            future_atr[i] = np.mean(tr[start:end])

    return future_atr


def build_regime_labels(prices, horizon_minutes, atr_window=30):
    """
    Crée des labels binaires de régime de marché :
      0 = calme — ATR futur < médiane → ne pas trader
      1 = actif — ATR futur >= médiane → opportunité, laisser D1/D2 décider
    """
    future_atr = compute_future_atr(prices, atr_window, horizon_minutes)

    # Médiane sur les valeurs valides
    valid_mask = ~np.isnan(future_atr)
    valid_atr = future_atr[valid_mask]
    median_atr = np.median(valid_atr)

    n = len(prices)
    labels = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        if np.isnan(future_atr[i]):
            continue
        labels[i] = 0 if future_atr[i] < median_atr else 1

    return labels, median_atr


def horizon_to_minutes(h):
    if h == '5m':
        return 5
    elif h == '15m':
        return 15
    elif h == '1h':
        return 60
    return 15


def train_horizon(horizon, features_np, feature_cols, prices, verbose):
    """Entraîne XGBoost régime binaire pour un horizon."""
    import xgboost as xgb

    print(f'\n{"="*60}')
    print(f'  XGBoost RÉGIME {horizon} (calme vs actif)')
    print(f'{"="*60}')

    minutes = horizon_to_minutes(horizon)
    labels, median_atr = build_regime_labels(prices, minutes)

    print(f'Seuil ATR (médiane): {median_atr:.6f}')

    # Filtrer les labels invalides (-1)
    valid = labels >= 0
    X = features_np[valid]
    y = labels[valid]

    # Distribution
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    class_names = ['Calme', 'Actif']
    print(f'Distribution:')
    for u, c in zip(unique, counts):
        print(f'  {class_names[u]}: {c:,} ({c/total*100:.1f}%)')

    # Split chronologique
    n = len(X)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    X_train, y_train = X[:t_end], y[:t_end]
    X_val, y_val = X[t_end:v_end], y[t_end:v_end]
    X_test, y_test = X[v_end:], y[v_end:]

    print(f'Train: {len(X_train):,} — Val: {len(X_val):,} — Test: {len(X_test):,}')

    # Poids pour équilibrer
    sample_weights = compute_sample_weight('balanced', y_train)

    # XGBoost binaire
    verbosity = 1 if verbose else 0
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        n_jobs=-1,
        verbosity=verbosity,
    )

    print(f'\nTraining...')
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=10 if verbose else 50,
    )

    # Évaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f'\n--- Résultats Test {horizon} (calme vs actif) ---')
    print(classification_report(y_test, y_pred, target_names=class_names))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # AUC
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f'\nAUC: {auc:.4f}')
    except Exception as e:
        auc = 0
        print(f'AUC non calculable: {e}')

    # Feature importance — top 20
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    print(f'\nTop 20 features:')
    for rank, idx in enumerate(top_idx):
        print(f'  {rank+1:2d}. {feature_cols[idx]:30s} importance: {importance[idx]:.4f}')

    # Sauvegarder
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'regime_{horizon}.json')
    model.save_model(model_path)
    print(f'\n✅ Modèle sauvé: {model_path}')

    # Upload GCS
    gcs_path = f'{GCS_BUCKET}/models/regime_{horizon}.json'
    result = subprocess.run(['gsutil', 'cp', model_path, gcs_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f'✅ Uploadé sur GCS: {gcs_path}')

    # Sauvegarder résultats + seuil (nécessaire pour l'inférence)
    results = {
        'horizon': horizon,
        'model_type': 'regime_binary',
        'classes': ['calme', 'actif'],
        'test_accuracy': float(np.mean(y_pred == y_test)),
        'test_auc': float(auc),
        'threshold': {
            'median_atr': float(median_atr),
        },
        'distribution': {
            'calme': int(counts[0]),
            'actif': int(counts[1]),
        },
        'top_features': [
            {'name': str(feature_cols[idx]), 'importance': float(importance[idx])}
            for idx in top_idx[:20]
        ],
        'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else 0,
        'per_class_metrics': {},
    }

    # Métriques par classe
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_pred = (y_pred == i)
            class_true = mask
            tp = int((class_pred & class_true).sum())
            fp = int((class_pred & ~class_true).sum())
            fn = int((~class_pred & class_true).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['per_class_metrics'][name.lower()] = {
                'precision': float(precision),
                'recall': float(recall),
                'support': int(mask.sum()),
            }

    results_path = os.path.join(MODELS_DIR, f'regime_{horizon}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'✅ Résultats sauvés: {results_path}')

    return results


def main():
    parser = argparse.ArgumentParser(description='Training XGBoost régime de marché (calme vs actif)')
    parser.add_argument('--horizon', choices=['5m', '15m', '1h'], help='Un seul horizon')
    parser.add_argument('--verbose', action='store_true', help='Logs détaillés')
    args = parser.parse_args()

    try:
        import xgboost
        print(f'XGBoost: {xgboost.__version__}')
    except ImportError:
        print('❌ XGBoost pas installé. Lance: pip install xgboost')
        sys.exit(1)

    # Données
    download_data()

    print('\n📂 Chargement des données...')
    features_np = np.load(os.path.join(DATA_DIR, 'features.npy'))
    feature_cols = np.load(os.path.join(DATA_DIR, 'feature_cols.npy'), allow_pickle=True)
    print(f'Features: {features_np.shape}')

    print('📂 Chargement des prix SOL...')
    prices = load_raw_prices()
    print(f'Prix: {len(prices)} points')

    # Alignement
    if len(prices) != len(features_np):
        print(f'⚠️ Alignement: prices={len(prices)}, features={len(features_np)}')
        min_len = min(len(prices), len(features_np))
        prices = prices[:min_len]
        features_np = features_np[:min_len]

    # Entraîner
    horizons = [args.horizon] if args.horizon else HORIZONS
    all_results = []

    for h in horizons:
        results = train_horizon(h, features_np, feature_cols, prices, args.verbose)
        all_results.append(results)

    # Résumé
    print(f'\n{"="*60}')
    print(f'  RÉSUMÉ — Régime marché (calme vs actif)')
    print(f'{"="*60}')
    for r in all_results:
        print(f'  {r["horizon"]:3s} — Accuracy: {r["test_accuracy"]:.4f}, AUC: {r["test_auc"]:.4f}')
        for cls, m in r['per_class_metrics'].items():
            print(f'    {cls:7s} — precision: {m["precision"]:.2f}, recall: {m["recall"]:.2f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
