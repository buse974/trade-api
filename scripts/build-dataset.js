import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://trade:trade_secret@172.17.0.1:5432/trade';
const pool = new pg.Pool({ connectionString: DATABASE_URL });

// Config: override via env or CLI args
// Usage: node build-dataset.js [table_name] [symbols_comma_separated]
const TABLE_NAME = process.argv[2] || 'dataset_wide';
const SYMBOLS = (process.argv[3] || 'SOLUSDT,BTCUSDT,ETHUSDT,DOGEUSDT,SHIBUSDT,PEPEUSDT').split(',');
const FEATURE_COLS = [
  'd1_short', 'd2_short', 'd1_mid', 'd2_mid', 'd1_long', 'd2_long',
  'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
  'atr', 'ma_deviation', 'integral_deviation'
];

// Short prefixes for column names
const PREFIXES = {
  SOLUSDT: 'sol',
  BTCUSDT: 'btc',
  ETHUSDT: 'eth',
  DOGEUSDT: 'doge',
  SHIBUSDT: 'shib',
  PEPEUSDT: 'pepe',
};

function buildColumnDefs() {
  const cols = [];
  for (const symbol of SYMBOLS) {
    const prefix = PREFIXES[symbol];
    for (const feat of FEATURE_COLS) {
      cols.push(`${prefix}_${feat} REAL`);
    }
  }
  return cols;
}

function buildSelectJoins() {
  const selects = ['sol.time'];
  const joins = [];

  for (const symbol of SYMBOLS) {
    const alias = PREFIXES[symbol];
    for (const feat of FEATURE_COLS) {
      selects.push(`${alias}.${feat} AS ${alias}_${feat}`);
    }
    if (symbol === 'SOLUSDT') {
      // SOL is the base table
      continue;
    }
    joins.push(`JOIN features ${alias} ON ${alias}.time = sol.time AND ${alias}.symbol = '${symbol}'`);
  }

  return { selects, joins };
}

async function createTable() {
  const columnDefs = buildColumnDefs();

  await pool.query(`DROP TABLE IF EXISTS ${TABLE_NAME}`);
  await pool.query(`
    CREATE TABLE ${TABLE_NAME} (
      time TIMESTAMPTZ NOT NULL,
      ${columnDefs.join(',\n      ')},
      PRIMARY KEY (time)
    )
  `);

  console.log(`✅ Table ${TABLE_NAME} créée (${SYMBOLS.length} cryptos × ${FEATURE_COLS.length} features = ${SYMBOLS.length * FEATURE_COLS.length} colonnes)`);
}

async function getMonthRanges() {
  // Find the time range where ALL selected symbols have data (intersection)
  const placeholders = SYMBOLS.map((_, i) => `$${i + 1}`).join(',');
  const res = await pool.query(`
    SELECT MAX(min_time) as start_time, MIN(max_time) as end_time
    FROM (
      SELECT symbol, MIN(time) as min_time, MAX(time) as max_time
      FROM features
      WHERE symbol IN (${placeholders})
      GROUP BY symbol
    ) sub
  `, SYMBOLS);

  const start = new Date(res.rows[0].start_time);
  const end = new Date(res.rows[0].end_time);

  console.log(`📅 Plage commune: ${start.toISOString().slice(0, 10)} → ${end.toISOString().slice(0, 10)}`);

  // Generate month boundaries
  const months = [];
  const cursor = new Date(start.getFullYear(), start.getMonth(), 1);

  while (cursor <= end) {
    const monthStart = new Date(cursor);
    cursor.setMonth(cursor.getMonth() + 1);
    const monthEnd = new Date(cursor);
    if (monthStart < end) {
      months.push({
        start: monthStart.toISOString(),
        end: monthEnd.toISOString(),
        label: monthStart.toISOString().slice(0, 7),
      });
    }
  }

  return months;
}

async function getLastInserted() {
  try {
    const res = await pool.query(`SELECT MAX(time) as last_time FROM ${TABLE_NAME}`);
    return res.rows[0]?.last_time || null;
  } catch {
    return null;
  }
}

async function insertMonth(month, { selects, joins }) {
  const selectStr = selects.join(', ');
  const joinStr = joins.join('\n    ');

  const query = `
    INSERT INTO ${TABLE_NAME}
    SELECT ${selectStr}
    FROM features sol
    ${joinStr}
    WHERE sol.symbol = 'SOLUSDT'
      AND sol.time >= $1
      AND sol.time < $2
    ORDER BY sol.time
    ON CONFLICT (time) DO NOTHING
  `;

  const res = await pool.query(query, [month.start, month.end]);
  return res.rowCount;
}

async function main() {
  console.log(`🏗️  Build ${TABLE_NAME} — pivot multi-crypto`);
  console.log(`   Cryptos: ${SYMBOLS.join(', ')}`);
  console.log(`   Features par crypto: ${FEATURE_COLS.length}`);
  console.log('');

  // Check if table exists and has data (resume support)
  const lastInserted = await getLastInserted();

  if (!lastInserted) {
    await createTable();
  } else {
    console.log(`🔄 Reprise détectée — dernier insert: ${new Date(lastInserted).toISOString().slice(0, 10)}`);
  }

  const months = await getMonthRanges();
  const { selects, joins } = buildSelectJoins();

  let totalRows = 0;
  let skipped = 0;
  const startTime = Date.now();

  for (let i = 0; i < months.length; i++) {
    const month = months[i];

    // Skip already processed months
    if (lastInserted && new Date(month.end) <= new Date(lastInserted)) {
      skipped++;
      continue;
    }

    const count = await insertMonth(month, { selects, joins });
    totalRows += count;

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
    const progress = i + 1;
    const remaining = months.length - progress;
    const avgPerMonth = (Date.now() - startTime) / (progress - skipped);
    const eta = ((remaining * avgPerMonth) / 1000 / 60).toFixed(1);

    console.log(`   ${month.label}  +${count.toLocaleString().padStart(8)} lignes | ${progress}/${months.length} mois | ${elapsed}s écoulé | ~${eta}min restant`);
  }

  // Final stats
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  const finalCount = await pool.query(`SELECT COUNT(*) as cnt FROM ${TABLE_NAME}`);

  console.log('');
  console.log(`✅ Terminé en ${totalTime}s`);
  console.log(`   ${totalRows.toLocaleString()} lignes insérées`);
  console.log(`   ${parseInt(finalCount.rows[0].cnt).toLocaleString()} lignes total dans ${TABLE_NAME}`);

  // Create index for fast sequential access
  console.log('   Création index...');
  await pool.query(`CREATE INDEX IF NOT EXISTS idx_${TABLE_NAME}_time ON ${TABLE_NAME} (time ASC)`);
  console.log('   ✅ Index créé');

  await pool.end();
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
