#!/usr/bin/env python3
"""
Training LSTM local — 3 modèles (5m, 15m, 1h) avec reprise automatique.

Usage:
    python train_lstm_local.py                  # Lance/reprend tout
    python train_lstm_local.py --horizon 15m    # Lance/reprend un seul modèle
    python train_lstm_local.py --reset          # Repart de zéro (supprime checkpoints)

Les données sont chargées depuis GCS (prepared/*.npy).
Les checkpoints et modèles finaux sont sauvés en local ET sur GCS.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import re
import sys
import subprocess
import argparse

# --- Config ---
WINDOW_SIZE = 30
BATCH_SIZE = 512
EPOCHS = 20
HORIZONS = ['15m', '5m', '1h']
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LOCAL_DIR, 'data')
CHECKPOINT_BASE = os.path.join(LOCAL_DIR, 'checkpoints')
MODELS_DIR = os.path.join(LOCAL_DIR, 'models')
GCS_BUCKET = 'gs://trade-ml-bot-data'


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0


def download_data():
    """Télécharge les .npy depuis GCS si pas déjà en local."""
    os.makedirs(DATA_DIR, exist_ok=True)
    features_path = os.path.join(DATA_DIR, 'features.npy')

    if os.path.exists(features_path):
        print('✅ Données déjà en local')
        return

    # Vérifier si prepared existe sur GCS
    if run_cmd(f'gsutil ls {GCS_BUCKET}/prepared/features.npy'):
        print('📥 Téléchargement depuis GCS...')
        os.system(f'gsutil -m cp {GCS_BUCKET}/prepared/*.npy {DATA_DIR}/')
        print('✅ Download OK')
    else:
        print('❌ Données pas trouvées sur GCS. Lance d\'abord le notebook Colab PARTIE 1.')
        sys.exit(1)


def make_dataset(features, labs, start, end, batch_size, shuffle=False):
    feat_slice = features[start:end + WINDOW_SIZE - 1]
    lab_slice = labs[start + WINDOW_SIZE - 1:end + WINDOW_SIZE - 1]
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=feat_slice,
        targets=lab_slice,
        sequence_length=WINDOW_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(n_features, horizon):
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(32),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name=f'lstm_{horizon}')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


def get_initial_epoch(checkpoint_dir):
    """Trouve le dernier checkpoint et retourne l'epoch."""
    if not os.path.exists(checkpoint_dir):
        return 0

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')]
    if not files:
        return 0

    files.sort()
    match = re.search(r'ep(\d+)', files[-1])
    return int(match.group(1)) if match else 0


def train_horizon(horizon, features_np, n_features, train_end, val_end, n_samples):
    """Entraîne un modèle pour un horizon donné."""
    print(f'\n{"="*60}')
    print(f'  LSTM {horizon}')
    print(f'{"="*60}')

    checkpoint_dir = os.path.join(CHECKPOINT_BASE, f'lstm_{horizon}')
    gcs_checkpoint = f'{GCS_BUCKET}/checkpoints/lstm_{horizon}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Vérifier si le modèle final existe déjà
    final_model = os.path.join(MODELS_DIR, f'lstm_{horizon}.keras')
    if os.path.exists(final_model):
        print(f'✅ Modèle {horizon} déjà entraîné ({final_model}). Skip.')
        return

    # Charger labels
    labels = np.load(os.path.join(DATA_DIR, f'lab_{horizon}.npy'))

    # Datasets
    train_ds = make_dataset(features_np, labels, 0, train_end, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(features_np, labels, train_end, val_end, BATCH_SIZE)
    test_ds = make_dataset(features_np, labels, val_end, n_samples, BATCH_SIZE)

    # Build model
    model = build_model(n_features, horizon)

    # Reprendre depuis checkpoint local ou GCS
    initial_epoch = get_initial_epoch(checkpoint_dir)

    if initial_epoch == 0:
        # Vérifier GCS
        check = subprocess.run(['gsutil', 'ls', f'{gcs_checkpoint}/'], capture_output=True, text=True)
        if check.returncode == 0 and '.weights.h5' in check.stdout:
            print('🔄 Checkpoint trouvé sur GCS, téléchargement...')
            os.system(f'gsutil -m cp {gcs_checkpoint}/* {checkpoint_dir}/')
            initial_epoch = get_initial_epoch(checkpoint_dir)

    if initial_epoch > 0:
        files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')])
        model.load_weights(os.path.join(checkpoint_dir, files[-1]))
        print(f'🔄 Reprise à l\'epoch {initial_epoch + 1}/{EPOCHS}')
    else:
        print('Démarrage de zéro')

    if initial_epoch >= EPOCHS:
        print(f'Training déjà terminé ({initial_epoch} epochs).')
        # Évaluer et sauvegarder quand même
        save_model(model, horizon, test_ds)
        return

    # Callback checkpoint local + GCS
    class CheckpointCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            path = os.path.join(checkpoint_dir, f'ep{epoch + 1:02d}.weights.h5')
            model.save_weights(path)
            os.system(f'gsutil cp {path} {gcs_checkpoint}/ep{epoch + 1:02d}.weights.h5 2>/dev/null')
            val_auc = logs.get('val_auc', 0)
            print(f'  → Checkpoint epoch {epoch + 1} sauvé (val_auc: {val_auc:.4f})')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=5, mode='max', restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        CheckpointCallback()
    ]

    print(f'\nTraining LSTM {horizon} — {train_end:,} samples, epochs {initial_epoch + 1}→{EPOCHS}')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )

    save_model(model, horizon, test_ds)


def save_model(model, horizon, test_ds):
    """Évalue et sauvegarde le modèle."""
    from sklearn.metrics import classification_report

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Évaluation
    test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
    print(f'\nTest {horizon} — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}')

    y_test = np.concatenate([y for _, y in test_ds])
    y_pred = (model.predict(test_ds, verbose=0).flatten() > 0.5).astype(int)
    print(f'\nClassification Report {horizon}:')
    print(classification_report(y_test, y_pred, target_names=['Baisse', 'Hausse']))

    # Sauvegarder en local
    local_path = os.path.join(MODELS_DIR, f'lstm_{horizon}.keras')
    model.save(local_path)
    print(f'✅ Modèle sauvé: {local_path}')

    # Upload sur GCS
    gcs_path = f'{GCS_BUCKET}/models/lstm_{horizon}.keras'
    if run_cmd(f'gsutil cp {local_path} {gcs_path}'):
        print(f'✅ Uploadé sur GCS: {gcs_path}')

    # TF.js format
    try:
        import tensorflowjs as tfjs
        tfjs_dir = os.path.join(MODELS_DIR, f'lstm_{horizon}')
        os.makedirs(tfjs_dir, exist_ok=True)
        tfjs.converters.save_keras_model(model, tfjs_dir)
        os.system(f'gsutil -m cp -r {tfjs_dir}/* {GCS_BUCKET}/models/lstm_{horizon}/ 2>/dev/null')
        print(f'✅ TF.js exporté: {tfjs_dir}')
    except ImportError:
        print('⚠️ tensorflowjs pas installé, skip export TF.js (pip install tensorflowjs)')


def main():
    parser = argparse.ArgumentParser(description='Training LSTM local')
    parser.add_argument('--horizon', choices=['5m', '15m', '1h'], help='Un seul horizon')
    parser.add_argument('--reset', action='store_true', help='Supprimer checkpoints et repartir de zéro')
    args = parser.parse_args()

    print(f'TensorFlow: {tf.__version__}')
    print(f'GPU: {tf.config.list_physical_devices("GPU")}')

    if args.reset:
        import shutil
        if os.path.exists(CHECKPOINT_BASE):
            shutil.rmtree(CHECKPOINT_BASE)
            print('🗑️  Checkpoints supprimés')
        if os.path.exists(MODELS_DIR):
            shutil.rmtree(MODELS_DIR)
            print('🗑️  Modèles supprimés')

    # Télécharger données
    download_data()

    # Charger features
    print('\n📂 Chargement des données...')
    features_np = np.load(os.path.join(DATA_DIR, 'features.npy'))
    n_features = features_np.shape[1]
    n_samples = len(features_np) - WINDOW_SIZE + 1
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    print(f'Features: {features_np.shape}')
    print(f'Train: {train_end:,} — Val: {val_end-train_end:,} — Test: {n_samples-val_end:,}')

    # Entraîner
    horizons = [args.horizon] if args.horizon else HORIZONS

    for h in horizons:
        train_horizon(h, features_np, n_features, train_end, val_end, n_samples)

    print(f'\n{"="*60}')
    print('  TERMINÉ')
    print(f'{"="*60}')
    print(f'Modèles dans: {MODELS_DIR}/')


if __name__ == '__main__':
    main()
