import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://trade:trade_secret@172.17.0.1:5432/trade';
const pool = new pg.Pool({ connectionString: DATABASE_URL });

// Which dataset table to use
const DATASET_TABLE = process.argv[2] || 'dataset_wide_5';

// Horizons in minutes
const HORIZONS = [5, 15, 60];

async function main() {
  console.log(`🏷️  Build labels for ${DATASET_TABLE}`);
  console.log(`   Horizons: ${HORIZONS.map(h => h + 'min').join(', ')}`);
  console.log('');

  // Step 1: Create labels table
  const labelsTable = `${DATASET_TABLE}_labels`;
  await pool.query(`DROP TABLE IF EXISTS ${labelsTable}`);
  await pool.query(`
    CREATE TABLE ${labelsTable} (
      time TIMESTAMPTZ NOT NULL PRIMARY KEY,
      price_now REAL NOT NULL,
      label_5m SMALLINT,
      label_15m SMALLINT,
      label_1h SMALLINT
    )
  `);
  console.log(`✅ Table ${labelsTable} créée`);

  // Step 2: Insert labels using self-join on prices table
  // For each timestamp in the dataset, look up SOL price now and at +5m, +15m, +1h
  console.log('   Calcul des labels...');

  const startTime = Date.now();

  const res = await pool.query(`
    INSERT INTO ${labelsTable} (time, price_now, label_5m, label_15m, label_1h)
    SELECT
      d.time,
      p_now.close AS price_now,
      CASE WHEN p_5m.close > p_now.close THEN 1 ELSE 0 END AS label_5m,
      CASE WHEN p_15m.close > p_now.close THEN 1 ELSE 0 END AS label_15m,
      CASE WHEN p_1h.close > p_now.close THEN 1 ELSE 0 END AS label_1h
    FROM ${DATASET_TABLE} d
    JOIN prices p_now ON p_now.time = d.time AND p_now.symbol = 'SOLUSDT'
    LEFT JOIN prices p_5m ON p_5m.time = d.time + INTERVAL '5 minutes' AND p_5m.symbol = 'SOLUSDT'
    LEFT JOIN prices p_15m ON p_15m.time = d.time + INTERVAL '15 minutes' AND p_15m.symbol = 'SOLUSDT'
    LEFT JOIN prices p_1h ON p_1h.time = d.time + INTERVAL '60 minutes' AND p_1h.symbol = 'SOLUSDT'
    ORDER BY d.time
  `);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`   ${res.rowCount.toLocaleString()} labels insérés en ${elapsed}s`);

  // Step 3: Stats — how many nulls (timestamps near the end won't have future prices)
  const stats = await pool.query(`
    SELECT
      COUNT(*) as total,
      COUNT(label_5m) as has_5m,
      COUNT(label_15m) as has_15m,
      COUNT(label_1h) as has_1h,
      ROUND(AVG(label_5m::numeric), 3) as pct_up_5m,
      ROUND(AVG(label_15m::numeric), 3) as pct_up_15m,
      ROUND(AVG(label_1h::numeric), 3) as pct_up_1h
    FROM ${labelsTable}
  `);
  const s = stats.rows[0];
  console.log('');
  console.log('📊 Stats:');
  console.log(`   Total:  ${parseInt(s.total).toLocaleString()}`);
  console.log(`   5min:   ${parseInt(s.has_5m).toLocaleString()} labels (${(s.pct_up_5m * 100).toFixed(1)}% hausse)`);
  console.log(`   15min:  ${parseInt(s.has_15m).toLocaleString()} labels (${(s.pct_up_15m * 100).toFixed(1)}% hausse)`);
  console.log(`   1h:     ${parseInt(s.has_1h).toLocaleString()} labels (${(s.pct_up_1h * 100).toFixed(1)}% hausse)`);

  // Step 4: Compute normalization stats (mean + std per feature column)
  console.log('');
  console.log('📐 Calcul normalisation...');

  // Get column names from dataset table (exclude 'time')
  const colRes = await pool.query(`
    SELECT column_name FROM information_schema.columns
    WHERE table_name = $1 AND column_name != 'time'
    ORDER BY ordinal_position
  `, [DATASET_TABLE]);
  const featureCols = colRes.rows.map(r => r.column_name);

  const normTable = `${DATASET_TABLE}_norm`;
  await pool.query(`DROP TABLE IF EXISTS ${normTable}`);
  await pool.query(`
    CREATE TABLE ${normTable} (
      feature VARCHAR(100) PRIMARY KEY,
      mean DOUBLE PRECISION NOT NULL,
      std DOUBLE PRECISION NOT NULL
    )
  `);

  // Compute mean and std for each column (NULLIF filters NaN which poisons AVG/STDDEV)
  const avgExprs = featureCols.map(c => `AVG(NULLIF(${c}::double precision, 'NaN')) AS "${c}_mean"`).join(', ');
  const stdExprs = featureCols.map(c => `STDDEV(NULLIF(${c}::double precision, 'NaN')) AS "${c}_std"`).join(', ');

  const normRes = await pool.query(`SELECT ${avgExprs}, ${stdExprs} FROM ${DATASET_TABLE}`);
  const normRow = normRes.rows[0];

  // Insert into norm table
  const normValues = [];
  const normParams = [];
  let idx = 1;
  for (const col of featureCols) {
    const mean = normRow[`${col}_mean`] || 0;
    const std = normRow[`${col}_std`] || 1;
    normValues.push(`($${idx}, $${idx + 1}, $${idx + 2})`);
    normParams.push(col, mean, std === 0 ? 1 : std);
    idx += 3;
  }

  await pool.query(`
    INSERT INTO ${normTable} (feature, mean, std) VALUES ${normValues.join(', ')}
  `, normParams);

  console.log(`   ${featureCols.length} features normalisées → ${normTable}`);

  // Step 5: Create index
  await pool.query(`CREATE INDEX IF NOT EXISTS idx_${labelsTable}_time ON ${labelsTable} (time ASC)`);

  console.log('');
  console.log('✅ Terminé !');
  console.log(`   Labels:        ${labelsTable}`);
  console.log(`   Normalisation: ${normTable}`);

  await pool.end();
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
