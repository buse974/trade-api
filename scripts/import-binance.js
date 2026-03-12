import { createWriteStream, createReadStream, unlinkSync, mkdirSync, existsSync } from 'fs';
import { pipeline } from 'stream/promises';
import { createUnzip } from 'zlib';
import { createInterface } from 'readline';
import pg from 'pg';

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://trade:trade_secret@trade-postgres:5432/trade';

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'];
const INTERVAL = '1m';
const TMP_DIR = '/tmp/binance-import';

// Binance Vision available start dates per symbol
const START_DATES = {
  BTCUSDT: { year: 2017, month: 8 },
  ETHUSDT: { year: 2017, month: 8 },
  SOLUSDT: { year: 2020, month: 8 },
  DOGEUSDT: { year: 2019, month: 7 },
  SHIBUSDT: { year: 2021, month: 5 },
  PEPEUSDT: { year: 2023, month: 5 },
};

const pool = new pg.Pool({ connectionString: DATABASE_URL });

async function getLastImportedDate(symbol) {
  const res = await pool.query(
    'SELECT MAX(time) as last_time FROM prices WHERE symbol = $1',
    [symbol]
  );
  return res.rows[0]?.last_time || null;
}

function buildMonthlyUrls(symbol) {
  const start = START_DATES[symbol];
  const now = new Date();
  const urls = [];

  let year = start.year;
  let month = start.month;

  // Stop at previous month (current month data is incomplete)
  const endYear = now.getFullYear();
  const endMonth = now.getMonth(); // 0-indexed, so this is previous month in 1-indexed

  while (year < endYear || (year === endYear && month <= endMonth)) {
    const mm = String(month).padStart(2, '0');
    const url = `https://data.binance.vision/data/spot/monthly/klines/${symbol}/${INTERVAL}/${symbol}-${INTERVAL}-${year}-${mm}.zip`;
    urls.push({ url, year, month });
    month++;
    if (month > 12) { month = 1; year++; }
  }

  return urls;
}

async function downloadFile(url, dest) {
  const res = await fetch(url);
  if (!res.ok) return false;
  const fileStream = createWriteStream(dest);
  await pipeline(res.body, fileStream);
  return true;
}

async function unzipFile(zipPath, csvPath) {
  // Binance ZIP contains a single CSV file
  // Use unzip command since Node's zlib doesn't handle ZIP archives (only gzip/deflate)
  const { execSync } = await import('child_process');
  execSync(`unzip -o -d ${TMP_DIR} ${zipPath}`, { stdio: 'pipe' });
  return true;
}

async function importCsv(csvPath, symbol) {
  const rl = createInterface({ input: createReadStream(csvPath) });
  let batch = [];
  let total = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;
    const parts = line.split(',');
    // Binance kline CSV format:
    // 0: Open time (ms), 1: Open, 2: High, 3: Low, 4: Close, 5: Volume,
    // 6: Close time, 7: Quote volume, 8: Trades count, 9: Taker buy base, 10: Taker buy quote, 11: Ignore
    let ts = parseInt(parts[0]);
    // Binance switched to microseconds (16 digits) for 2025+ data
    if (ts > 1e15) ts = Math.floor(ts / 1000);
    // Validate: must be a reasonable ms timestamp (2015-2030)
    if (ts < 1420070400000 || ts > 1893456000000) continue;
    const time = new Date(ts);
    const open = parts[1];
    const high = parts[2];
    const low = parts[3];
    const close = parts[4];
    const volume = parts[5];

    batch.push({ time, symbol, open, high, low, close, volume });

    if (batch.length >= 5000) {
      await insertBatch(batch);
      total += batch.length;
      batch = [];
    }
  }

  if (batch.length > 0) {
    await insertBatch(batch);
    total += batch.length;
  }

  return total;
}

async function insertBatch(rows) {
  if (rows.length === 0) return;

  const values = [];
  const params = [];
  let i = 1;

  for (const row of rows) {
    values.push(`($${i}, $${i + 1}, $${i + 2}, $${i + 3}, $${i + 4}, $${i + 5}, $${i + 6})`);
    params.push(row.time, row.symbol, row.open, row.high, row.low, row.close, row.volume);
    i += 7;
  }

  await pool.query(
    `INSERT INTO prices (time, symbol, open, high, low, close, volume)
     VALUES ${values.join(',')}
     ON CONFLICT (symbol, time) DO NOTHING`,
    params
  );
}

async function importSymbol(symbol) {
  const lastDate = await getLastImportedDate(symbol);
  const urls = buildMonthlyUrls(symbol);

  console.log(`\n📊 ${symbol} — ${urls.length} mois disponibles`);
  if (lastDate) {
    console.log(`   Dernière donnée en base: ${lastDate.toISOString()}`);
  }

  let totalRows = 0;

  for (const { url, year, month } of urls) {
    // Skip months already fully imported
    if (lastDate) {
      const monthEnd = new Date(year, month, 0, 23, 59, 59);
      if (monthEnd < lastDate) continue;
    }

    const mm = String(month).padStart(2, '0');
    const zipFile = `${TMP_DIR}/${symbol}-${INTERVAL}-${year}-${mm}.zip`;
    const csvFile = `${TMP_DIR}/${symbol}-${INTERVAL}-${year}-${mm}.csv`;

    process.stdout.write(`   ${year}-${mm}...`);

    const ok = await downloadFile(url, zipFile);
    if (!ok) {
      console.log(' ⏭️  non disponible');
      continue;
    }

    await unzipFile(zipFile, csvFile);
    const rows = await importCsv(csvFile, symbol);
    totalRows += rows;
    console.log(` ✅ ${rows.toLocaleString()} lignes`);

    // Cleanup
    try { unlinkSync(zipFile); } catch {}
    try { unlinkSync(csvFile); } catch {}
  }

  console.log(`   Total ${symbol}: ${totalRows.toLocaleString()} lignes importées`);
  return totalRows;
}

async function main() {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });

  console.log('🚀 Import Binance Vision → PostgreSQL');
  console.log(`   DB: ${DATABASE_URL.replace(/:[^:@]+@/, ':***@')}`);
  console.log(`   Cryptos: ${SYMBOLS.join(', ')}`);
  console.log(`   Intervalle: ${INTERVAL}`);

  let grandTotal = 0;

  for (const symbol of SYMBOLS) {
    try {
      grandTotal += await importSymbol(symbol);
    } catch (err) {
      console.error(`\n❌ Erreur ${symbol}:`, err.message);
    }
  }

  console.log(`\n✅ Import terminé — ${grandTotal.toLocaleString()} lignes au total`);

  // Stats
  const res = await pool.query(`
    SELECT symbol, COUNT(*) as rows, MIN(time) as first, MAX(time) as last
    FROM prices GROUP BY symbol ORDER BY symbol
  `);
  console.log('\n📈 État de la base:');
  for (const row of res.rows) {
    console.log(`   ${row.symbol}: ${parseInt(row.rows).toLocaleString()} lignes (${row.first.toISOString().slice(0, 10)} → ${row.last.toISOString().slice(0, 10)})`);
  }

  await pool.end();
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
