import * as tf from '@tensorflow/tfjs';
import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://trade:trade_secret@172.17.0.1:5432/trade';
const pool = new pg.Pool({ connectionString: DATABASE_URL });

// --- Config ---
const DATASET_TABLE = process.argv[2] || 'dataset_wide_5';
const HORIZON = process.argv[3] || '15m';        // 5m, 15m, 1h
const QUICK_MODE = process.argv.includes('--quick');

const WINDOW_SIZE = 60;         // 60 minutes of history as input
const BATCH_SIZE = 256;
const EPOCHS = QUICK_MODE ? 3 : 30;
const TRAIN_RATIO = 0.7;
const VAL_RATIO = 0.15;
// TEST_RATIO = 0.15 (implicit)

const LABEL_COL = {
  '5m': 'label_5m',
  '15m': 'label_15m',
  '1h': 'label_1h',
}[HORIZON];

if (!LABEL_COL) {
  console.error(`Invalid horizon: ${HORIZON}. Use 5m, 15m, or 1h`);
  process.exit(1);
}

const MODEL_DIR = `/app/models/${DATASET_TABLE}_${HORIZON}`;

// --- Load normalization stats ---
async function loadNormStats() {
  const res = await pool.query(`SELECT feature, mean, std FROM ${DATASET_TABLE}_norm ORDER BY feature`);
  const stats = {};
  for (const row of res.rows) {
    stats[row.feature] = { mean: parseFloat(row.mean), std: parseFloat(row.std) };
  }
  return stats;
}

// --- Load feature column names ---
async function getFeatureColumns() {
  const res = await pool.query(`
    SELECT column_name FROM information_schema.columns
    WHERE table_name = $1 AND column_name != 'time'
    ORDER BY ordinal_position
  `, [DATASET_TABLE]);
  return res.rows.map(r => r.column_name);
}

// --- Load data in chunks ---
async function loadDataset(featureCols, normStats) {
  console.log('📦 Chargement des données...');

  const limit = QUICK_MODE ? 50000 : null;
  const limitClause = limit ? `LIMIT ${limit}` : '';

  // Load features + labels joined
  const query = `
    SELECT d.*, l.${LABEL_COL} as label
    FROM ${DATASET_TABLE} d
    JOIN ${DATASET_TABLE}_labels l ON l.time = d.time
    WHERE l.${LABEL_COL} IS NOT NULL
    ORDER BY d.time ASC
    ${limitClause}
  `;

  const res = await pool.query(query);
  const rows = res.rows;
  console.log(`   ${rows.length.toLocaleString()} lignes chargées`);

  if (rows.length < WINDOW_SIZE + 1) {
    throw new Error('Pas assez de données');
  }

  // Normalize and build arrays
  const numFeatures = featureCols.length;
  const numSamples = rows.length - WINDOW_SIZE;

  console.log(`   ${numSamples.toLocaleString()} fenêtres possibles (${numFeatures} features × ${WINDOW_SIZE} pas de temps)`);

  // Pre-normalize all data into a flat Float32Array for speed
  const normalizedData = new Float32Array(rows.length * numFeatures);
  const labels = new Uint8Array(rows.length);

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    labels[i] = row.label;
    for (let j = 0; j < numFeatures; j++) {
      const col = featureCols[j];
      const val = parseFloat(row[col]) || 0;
      const { mean, std } = normStats[col] || { mean: 0, std: 1 };
      normalizedData[i * numFeatures + j] = (val - mean) / std;
    }
  }

  // Split chronologically
  const trainEnd = Math.floor(numSamples * TRAIN_RATIO);
  const valEnd = Math.floor(numSamples * (TRAIN_RATIO + VAL_RATIO));

  return { normalizedData, labels, numFeatures, numSamples, trainEnd, valEnd };
}

// --- Create windowed tensors from a range ---
function createTensors(normalizedData, labels, numFeatures, startIdx, endIdx) {
  const count = endIdx - startIdx;
  const xData = new Float32Array(count * WINDOW_SIZE * numFeatures);
  const yData = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const sampleIdx = startIdx + i;
    // Copy window of WINDOW_SIZE rows
    for (let w = 0; w < WINDOW_SIZE; w++) {
      const srcOffset = (sampleIdx + w) * numFeatures;
      const dstOffset = (i * WINDOW_SIZE + w) * numFeatures;
      for (let f = 0; f < numFeatures; f++) {
        xData[dstOffset + f] = normalizedData[srcOffset + f];
      }
    }
    // Label is at the end of the window
    yData[i] = labels[sampleIdx + WINDOW_SIZE];
  }

  const xs = tf.tensor3d(xData, [count, WINDOW_SIZE, numFeatures]);
  const ys = tf.tensor2d(yData, [count, 1]);

  return { xs, ys };
}

// --- Build LSTM model ---
function buildModel(numFeatures) {
  const model = tf.sequential();

  model.add(tf.layers.lstm({
    units: 64,
    inputShape: [WINDOW_SIZE, numFeatures],
    returnSequences: true,
    dropout: 0.2,
    recurrentDropout: 0.2,
  }));

  model.add(tf.layers.lstm({
    units: 32,
    returnSequences: false,
    dropout: 0.2,
    recurrentDropout: 0.2,
  }));

  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.1 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// --- Train ---
async function main() {
  console.log(`🧠 Entraînement LSTM — ${DATASET_TABLE} — horizon ${HORIZON}`);
  console.log(`   Window: ${WINDOW_SIZE} | Batch: ${BATCH_SIZE} | Epochs: ${EPOCHS}`);
  if (QUICK_MODE) console.log('   ⚡ Mode rapide (50k lignes, 3 epochs)');
  console.log('');

  const featureCols = await getFeatureColumns();
  const normStats = await loadNormStats();

  console.log(`   ${featureCols.length} features`);

  const { normalizedData, labels, numFeatures, numSamples, trainEnd, valEnd } = await loadDataset(featureCols, normStats);

  console.log('');
  console.log(`📊 Split chronologique:`);
  console.log(`   Train: 0 → ${trainEnd.toLocaleString()} (${trainEnd.toLocaleString()} samples)`);
  console.log(`   Val:   ${trainEnd.toLocaleString()} → ${valEnd.toLocaleString()} (${(valEnd - trainEnd).toLocaleString()} samples)`);
  console.log(`   Test:  ${valEnd.toLocaleString()} → ${numSamples.toLocaleString()} (${(numSamples - valEnd).toLocaleString()} samples)`);

  // Create tensors
  console.log('');
  console.log('🔧 Création des tenseurs...');
  const train = createTensors(normalizedData, labels, numFeatures, 0, trainEnd);
  const val = createTensors(normalizedData, labels, numFeatures, trainEnd, valEnd);
  const test = createTensors(normalizedData, labels, numFeatures, valEnd, numSamples);

  console.log(`   Train: ${train.xs.shape} → ${train.ys.shape}`);
  console.log(`   Val:   ${val.xs.shape} → ${val.ys.shape}`);
  console.log(`   Test:  ${test.xs.shape} → ${test.ys.shape}`);

  // Build model
  console.log('');
  const model = buildModel(numFeatures);
  model.summary();

  // Train
  console.log('');
  console.log('🚀 Entraînement...');
  const startTime = Date.now();

  await model.fit(train.xs, train.ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [val.xs, val.ys],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
        console.log(`   Epoch ${epoch + 1}/${EPOCHS} — loss: ${logs.loss.toFixed(4)} — acc: ${logs.acc.toFixed(4)} — val_loss: ${logs.val_loss.toFixed(4)} — val_acc: ${logs.val_acc.toFixed(4)} — ${elapsed}min`);
      },
    },
  });

  // Evaluate on test set
  console.log('');
  console.log('📈 Évaluation sur test set...');
  const evalResult = model.evaluate(test.xs, test.ys);
  const testLoss = evalResult[0].dataSync()[0];
  const testAcc = evalResult[1].dataSync()[0];
  console.log(`   Test loss: ${testLoss.toFixed(4)}`);
  console.log(`   Test accuracy: ${(testAcc * 100).toFixed(2)}%`);

  // Predictions distribution
  const preds = model.predict(test.xs);
  const predValues = preds.dataSync();
  let above50 = 0, above60 = 0, above70 = 0;
  for (let i = 0; i < predValues.length; i++) {
    if (predValues[i] > 0.5) above50++;
    if (predValues[i] > 0.6) above60++;
    if (predValues[i] > 0.7) above70++;
  }
  console.log(`   Prédictions >50%: ${above50} (${(above50 / predValues.length * 100).toFixed(1)}%)`);
  console.log(`   Prédictions >60%: ${above60} (${(above60 / predValues.length * 100).toFixed(1)}%)`);
  console.log(`   Prédictions >70%: ${above70} (${(above70 / predValues.length * 100).toFixed(1)}%)`);

  // Save model
  console.log('');
  console.log(`💾 Sauvegarde du modèle → ${MODEL_DIR}`);
  await model.save(`file://${MODEL_DIR}`);
  console.log('   ✅ Modèle sauvegardé');

  // Cleanup tensors
  train.xs.dispose(); train.ys.dispose();
  val.xs.dispose(); val.ys.dispose();
  test.xs.dispose(); test.ys.dispose();
  preds.dispose();
  evalResult.forEach(t => t.dispose());

  const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  console.log('');
  console.log(`✅ Terminé en ${totalTime} minutes`);
  console.log(`   Modèle: ${MODEL_DIR}`);
  console.log(`   Test accuracy: ${(testAcc * 100).toFixed(2)}%`);

  await pool.end();
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
