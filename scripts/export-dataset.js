import pg from 'pg';
import fs from 'fs';
import { createGzip } from 'zlib';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';
import dotenv from 'dotenv';

dotenv.config();

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://trade:trade_secret@172.17.0.1:5432/trade';
const pool = new pg.Pool({ connectionString: DATABASE_URL });

const DATASET_TABLE = process.argv[2] || 'dataset_wide_5';
const OUTPUT_DIR = process.argv[3] || '/app/exports';

async function exportTable(tableName, outputFile, query) {
  console.log(`   Exporting ${tableName}...`);
  const res = await pool.query(query || `SELECT * FROM ${tableName}`);

  if (res.rows.length === 0) {
    console.log(`   ⏭️ Empty table`);
    return;
  }

  const columns = Object.keys(res.rows[0]);
  const lines = [columns.join(',')];

  for (const row of res.rows) {
    const vals = columns.map(c => {
      const v = row[c];
      if (v instanceof Date) return v.toISOString();
      if (v === null) return '';
      return v;
    });
    lines.push(vals.join(','));
  }

  const csvContent = lines.join('\n') + '\n';
  const gzipPath = `${outputFile}.gz`;

  await pipeline(
    Readable.from([csvContent]),
    createGzip({ level: 6 }),
    fs.createWriteStream(gzipPath)
  );

  const sizeMB = (fs.statSync(gzipPath).size / 1024 / 1024).toFixed(1);
  console.log(`   ✅ ${gzipPath} (${sizeMB} MB, ${res.rows.length.toLocaleString()} rows)`);
}

async function exportLargeTable(tableName, outputFile) {
  console.log(`   Exporting ${tableName} (streaming)...`);

  // Get columns first
  const colRes = await pool.query(`
    SELECT column_name FROM information_schema.columns
    WHERE table_name = $1 ORDER BY ordinal_position
  `, [tableName]);
  const columns = colRes.rows.map(r => r.column_name);

  // Count
  const countRes = await pool.query(`SELECT COUNT(*) as c FROM ${tableName}`);
  const totalRows = parseInt(countRes.rows[0].c);

  const gzipPath = `${outputFile}.gz`;
  const gzip = createGzip({ level: 6 });
  const out = fs.createWriteStream(gzipPath);
  gzip.pipe(out);

  // Header
  gzip.write(columns.join(',') + '\n');

  // Stream by chunks
  const CHUNK = 100000;
  let offset = 0;
  let written = 0;

  while (offset < totalRows) {
    const res = await pool.query(
      `SELECT * FROM ${tableName} ORDER BY time ASC LIMIT ${CHUNK} OFFSET ${offset}`
    );

    for (const row of res.rows) {
      const vals = columns.map(c => {
        const v = row[c];
        if (v instanceof Date) return v.toISOString();
        if (v === null) return '';
        return v;
      });
      gzip.write(vals.join(',') + '\n');
    }

    written += res.rows.length;
    offset += CHUNK;
    const pct = ((written / totalRows) * 100).toFixed(0);
    process.stdout.write(`\r   ${written.toLocaleString()}/${totalRows.toLocaleString()} rows (${pct}%)`);
  }

  // Close gzip stream
  await new Promise((resolve, reject) => {
    gzip.end(() => {
      out.on('finish', resolve);
      out.on('error', reject);
    });
  });

  const sizeMB = (fs.statSync(gzipPath).size / 1024 / 1024).toFixed(1);
  console.log(`\n   ✅ ${gzipPath} (${sizeMB} MB, ${totalRows.toLocaleString()} rows)`);
}

async function main() {
  console.log(`📦 Export dataset: ${DATASET_TABLE}`);
  console.log(`   Output: ${OUTPUT_DIR}/`);
  console.log('');

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Export main dataset (large — stream it)
  await exportLargeTable(DATASET_TABLE, `${OUTPUT_DIR}/${DATASET_TABLE}`);

  // Export labels
  await exportTable(`${DATASET_TABLE}_labels`, `${OUTPUT_DIR}/${DATASET_TABLE}_labels`);

  // Export norm stats
  await exportTable(`${DATASET_TABLE}_norm`, `${OUTPUT_DIR}/${DATASET_TABLE}_norm`);

  console.log('');
  console.log('✅ Export terminé !');

  // List files
  const files = fs.readdirSync(OUTPUT_DIR).filter(f => f.startsWith(DATASET_TABLE));
  for (const f of files) {
    const size = (fs.statSync(`${OUTPUT_DIR}/${f}`).size / 1024 / 1024).toFixed(1);
    console.log(`   ${f} — ${size} MB`);
  }

  await pool.end();
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
