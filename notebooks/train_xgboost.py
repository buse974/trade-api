#!/usr/bin/env python3
"""
Training XGBoost — classification direction SOL avec seuil de mouvement significatif.

Usage:
    python train_xgboost.py                     # Lance tout
    python train_xgboost.py --threshold 0.5     # Seuil 0.5% (défaut)
    python train_xgboost.py --threshold 1.0     # Seuil 1%

Rapide : ~5 minutes pour 3 horizons.
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


def build_labels_with_threshold(prices, horizon_minutes, threshold_pct):
    """
    Crée des labels 3 classes basés sur un seuil de mouvement :
    0 = baisse significative (< -threshold%)
    1 = neutre (entre -threshold% et +threshold%)
    2 = hausse significative (> +threshold%)
    """
    n = len(prices)
    labels = np.full(n, -1, dtype=np.int32)

    for i in range(n - horizon_minutes):
        future_price = prices[i + horizon_minutes]
        current_price = prices[i]
        if current_price == 0:
            continue
        pct_change = ((future_price - current_price) / current_price) * 100

        if pct_change < -threshold_pct:
            labels[i] = 0  # baisse
        elif pct_change > threshold_pct:
            labels[i] = 2  # hausse
        else:
            labels[i] = 1  # neutre

    return labels


def horizon_to_minutes(h):
    if h == '5m':
        return 5
    elif h == '15m':
        return 15
    elif h == '1h':
        return 60
    return 15


def train_horizon(horizon, features_np, feature_cols, prices, threshold, train_end, val_end):
    """Entraîne XGBoost pour un horizon."""
    import xgboost as xgb

    print(f'\n{"="*60}')
    print(f'  XGBoost {horizon} — seuil {threshold}%')
    print(f'{"="*60}')

    minutes = horizon_to_minutes(horizon)
    labels = build_labels_with_threshold(prices, minutes, threshold)

    # Filtrer les labels invalides (-1)
    valid = labels >= 0
    X = features_np[valid]
    y = labels[valid]

    # Distribution
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f'Distribution: baisse={counts[0]} ({counts[0]/total*100:.1f}%), '
          f'neutre={counts[1]} ({counts[1]/total*100:.1f}%), '
          f'hausse={counts[2]} ({counts[2]/total*100:.1f}%)')

    # Split chronologique
    n = len(X)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    X_train, y_train = X[:t_end], y[:t_end]
    X_val, y_val = X[t_end:v_end], y[t_end:v_end]
    X_test, y_test = X[v_end:], y[v_end:]

    print(f'Train: {len(X_train):,} — Val: {len(X_val):,} — Test: {len(X_test):,}')

    # Poids pour équilibrer les classes
    sample_weights = compute_sample_weight('balanced', y_train)

    # XGBoost
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        n_jobs=-1,
        verbosity=1,
    )

    print(f'\nTraining...')
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Évaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print(f'\n--- Résultats Test {horizon} (seuil {threshold}%) ---')
    print(classification_report(y_test, y_pred, target_names=['Baisse', 'Neutre', 'Hausse']))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # AUC one-vs-rest
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        print(f'\nAUC macro (one-vs-rest): {auc:.4f}')
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
    model_path = os.path.join(MODELS_DIR, f'xgb_{horizon}_{threshold}pct.json')
    model.save_model(model_path)
    print(f'\n✅ Modèle sauvé: {model_path}')

    # Upload GCS
    gcs_path = f'{GCS_BUCKET}/models/xgb_{horizon}_{threshold}pct.json'
    result = subprocess.run(['gsutil', 'cp', model_path, gcs_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f'✅ Uploadé sur GCS: {gcs_path}')

    # Sauvegarder les résultats
    results = {
        'horizon': horizon,
        'threshold_pct': threshold,
        'test_accuracy': float(np.mean(y_pred == y_test)),
        'test_auc_macro': float(auc),
        'distribution': {
            'baisse': int(counts[0]),
            'neutre': int(counts[1]),
            'hausse': int(counts[2]),
        },
        'top_features': [
            {'name': str(feature_cols[idx]), 'importance': float(importance[idx])}
            for idx in top_idx[:20]
        ],
        'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else 0,
    }

    results_path = os.path.join(MODELS_DIR, f'xgb_{horizon}_{threshold}pct_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Training XGBoost')
    parser.add_argument('--threshold', type=float, default=0.5, help='Seuil mouvement %% (défaut: 0.5)')
    parser.add_argument('--horizon', choices=['5m', '15m', '1h'], help='Un seul horizon')
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

    # Charger les prix bruts pour calculer les mouvements
    print('📂 Chargement des prix SOL...')
    prices = load_raw_prices()
    print(f'Prix: {len(prices)} points')

    # Vérifier alignement
    if len(prices) != len(features_np):
        print(f'⚠️ Alignement: prices={len(prices)}, features={len(features_np)}')
        min_len = min(len(prices), len(features_np))
        prices = prices[:min_len]
        features_np = features_np[:min_len]

    # Split
    n = len(features_np)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Entraîner
    horizons = [args.horizon] if args.horizon else HORIZONS
    all_results = []

    for h in horizons:
        results = train_horizon(h, features_np, feature_cols, prices, args.threshold, train_end, val_end)
        all_results.append(results)

    # Résumé
    print(f'\n{"="*60}')
    print(f'  RÉSUMÉ — Seuil {args.threshold}%')
    print(f'{"="*60}')
    for r in all_results:
        print(f'  XGBoost {r["horizon"]:3s} — Accuracy: {r["test_accuracy"]:.4f}, AUC: {r["test_auc_macro"]:.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
