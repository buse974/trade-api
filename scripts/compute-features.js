import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://trade:trade_secret@trade-postgres:5432/trade';
const pool = new pg.Pool({ connectionString: DATABASE_URL });

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'];

// Windows for short/mid/long derivatives
const WIN_SHORT = 15;
const WIN_MID = 60;
const WIN_LONG = 240;

// RSI period
const RSI_PERIOD = 14;

// MACD
const MACD_FAST = 12;
const MACD_SLOW = 26;
const MACD_SIGNAL = 9;

// Bollinger
const BOLL_PERIOD = 20;
const BOLL_STD = 2;

// ATR
const ATR_PERIOD = 14;

// MA for deviation (P component)
const MA_PERIOD = 60;

// Integral window
const INTEGRAL_WINDOW = 60;

// --- Math helpers ---

function slope(values, start, len) {
  if (len < 2) return 0;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < len; i++) {
    const y = values[start + i];
    sumX += i;
    sumY += y;
    sumXY += i * y;
    sumX2 += i * i;
  }
  return (len * sumXY - sumX * sumY) / (len * sumX2 - sumX * sumX);
}

function ema(prev, value, period) {
  const k = 2 / (period + 1);
  return value * k + prev * (1 - k);
}

function computeFeatures(closes, highs, lows, idx) {
  if (idx < WIN_LONG) return null;

  // --- D1/D2 at 3 intervals ---
  const d1Short = slope(closes, idx - WIN_SHORT, WIN_SHORT);
  const d1Mid = slope(closes, idx - WIN_MID, WIN_MID);
  const d1Long = slope(closes, idx - WIN_LONG, WIN_LONG);

  // D2 = slope of recent D1 values (we approximate using half-window slopes)
  const halfShort = Math.floor(WIN_SHORT / 2);
  const d1ShortPrev = slope(closes, idx - WIN_SHORT, halfShort);
  const d2Short = (d1Short - d1ShortPrev) / halfShort;

  const halfMid = Math.floor(WIN_MID / 2);
  const d1MidPrev = slope(closes, idx - WIN_MID, halfMid);
  const d2Mid = (d1Mid - d1MidPrev) / halfMid;

  const halfLong = Math.floor(WIN_LONG / 2);
  const d1LongPrev = slope(closes, idx - WIN_LONG, halfLong);
  const d2Long = (d1Long - d1LongPrev) / halfLong;

  return { d1Short, d2Short, d1Mid, d2Mid, d1Long, d2Long };
}

async function processSymbol(symbol) {
  console.log(`\n📊 ${symbol}`);

  // Check last computed feature
  const lastRes = await pool.query(
    'SELECT MAX(time) as last_time FROM features WHERE symbol = $1',
    [symbol]
  );
  const lastTime = lastRes.rows[0]?.last_time;

  // We need WIN_LONG rows of warmup before we can compute, so fetch from earlier
  const query = lastTime
    ? `SELECT time, open, high, low, close, volume FROM prices WHERE symbol = $1 ORDER BY time`
    : `SELECT time, open, high, low, close, volume FROM prices WHERE symbol = $1 ORDER BY time`;

  console.log(`   Loading prices...`);
  const result = await pool.query(query, [symbol]);
  const rows = result.rows;
  console.log(`   ${rows.length.toLocaleString()} rows loaded`);

  if (rows.length < WIN_LONG + 1) {
    console.log(`   ⏭️ Not enough data`);
    return 0;
  }

  // Parse all values into arrays for fast access
  const times = [];
  const closes = new Float64Array(rows.length);
  const highs = new Float64Array(rows.length);
  const lows = new Float64Array(rows.length);
  const opens = new Float64Array(rows.length);

  for (let i = 0; i < rows.length; i++) {
    times.push(rows[i].time);
    closes[i] = parseFloat(rows[i].close);
    highs[i] = parseFloat(rows[i].high);
    lows[i] = parseFloat(rows[i].low);
    opens[i] = parseFloat(rows[i].open);
  }

  // Find start index (skip already computed)
  let startIdx = WIN_LONG;
  if (lastTime) {
    const lastTimeMs = new Date(lastTime).getTime();
    for (let i = 0; i < times.length; i++) {
      if (new Date(times[i]).getTime() > lastTimeMs) {
        startIdx = Math.max(WIN_LONG, i);
        break;
      }
    }
    if (startIdx === WIN_LONG && new Date(times[times.length - 1]).getTime() <= lastTimeMs) {
      console.log(`   ✅ Already up to date`);
      return 0;
    }
  }

  console.log(`   Computing from index ${startIdx.toLocaleString()} to ${(rows.length - 1).toLocaleString()}...`);

  // Running state for EMA-based indicators
  let emaFast = closes[startIdx];
  let emaSlow = closes[startIdx];
  let emaSignal = 0;
  let avgGain = 0;
  let avgLoss = 0;
  let atr = 0;

  // Initialize RSI
  for (let i = startIdx - RSI_PERIOD; i < startIdx; i++) {
    const diff = closes[i + 1] - closes[i];
    if (diff > 0) avgGain += diff;
    else avgLoss -= diff;
  }
  avgGain /= RSI_PERIOD;
  avgLoss /= RSI_PERIOD;

  // Initialize ATR
  for (let i = startIdx - ATR_PERIOD; i < startIdx; i++) {
    const tr = Math.max(highs[i] - lows[i], Math.abs(highs[i] - closes[i - 1] || 0), Math.abs(lows[i] - closes[i - 1] || 0));
    atr += tr;
  }
  atr /= ATR_PERIOD;

  // Initialize MACD EMAs
  for (let i = startIdx - MACD_SLOW; i < startIdx; i++) {
    emaFast = ema(emaFast, closes[i], MACD_FAST);
    emaSlow = ema(emaSlow, closes[i], MACD_SLOW);
  }
  emaSignal = emaFast - emaSlow;

  let batch = [];
  let total = 0;

  for (let i = startIdx; i < rows.length; i++) {
    const price = closes[i];

    // D1/D2
    const derivs = computeFeatures(closes, highs, lows, i);
    if (!derivs) continue;

    // RSI
    const diff = closes[i] - closes[i - 1];
    avgGain = (avgGain * (RSI_PERIOD - 1) + Math.max(diff, 0)) / RSI_PERIOD;
    avgLoss = (avgLoss * (RSI_PERIOD - 1) + Math.max(-diff, 0)) / RSI_PERIOD;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));

    // MACD
    emaFast = ema(emaFast, price, MACD_FAST);
    emaSlow = ema(emaSlow, price, MACD_SLOW);
    const macdLine = emaFast - emaSlow;
    emaSignal = ema(emaSignal, macdLine, MACD_SIGNAL);

    // Bollinger Bands
    let sum = 0, sumSq = 0;
    for (let j = i - BOLL_PERIOD; j < i; j++) {
      sum += closes[j];
      sumSq += closes[j] * closes[j];
    }
    const ma = sum / BOLL_PERIOD;
    const std = Math.sqrt(sumSq / BOLL_PERIOD - ma * ma);
    const bollUpper = ma + BOLL_STD * std;
    const bollLower = ma - BOLL_STD * std;

    // ATR
    const tr = Math.max(highs[i] - lows[i], Math.abs(highs[i] - closes[i - 1]), Math.abs(lows[i] - closes[i - 1]));
    atr = (atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD;

    // MA Deviation (P component) — % deviation from MA
    let maSum = 0;
    for (let j = i - MA_PERIOD; j < i; j++) maSum += closes[j];
    const maValue = maSum / MA_PERIOD;
    const maDeviation = ((price - maValue) / maValue) * 100;

    // Integral deviation (I component) — sum of deviations over window
    let integralDev = 0;
    for (let j = Math.max(0, i - INTEGRAL_WINDOW); j < i; j++) {
      let localMaSum = 0;
      const localStart = Math.max(0, j - MA_PERIOD);
      const localLen = j - localStart;
      if (localLen < 1) continue;
      for (let k = localStart; k < j; k++) localMaSum += closes[k];
      const localMa = localMaSum / localLen;
      integralDev += (closes[j] - localMa) / localMa;
    }
    integralDev *= 100;

    batch.push({
      time: times[i],
      symbol,
      ...derivs,
      rsi,
      macd: macdLine,
      macdSignal: emaSignal,
      bollUpper,
      bollLower,
      atr,
      maDeviation,
      integralDev,
    });

    if (batch.length >= 500) {
      await insertBatch(batch);
      total += batch.length;
      const pct = ((i / rows.length) * 100).toFixed(1);
      process.stdout.write(`\r   ${total.toLocaleString()} features (${pct}%)`);
      batch = [];
    }
  }

  if (batch.length > 0) {
    await insertBatch(batch);
    total += batch.length;
  }

  console.log(`\n   ✅ ${total.toLocaleString()} features computed`);
  return total;
}

async function insertBatch(rows) {
  if (rows.length === 0) return;

  const values = [];
  const params = [];
  let i = 1;

  for (const r of rows) {
    values.push(`($${i},$${i+1},$${i+2},$${i+3},$${i+4},$${i+5},$${i+6},$${i+7},$${i+8},$${i+9},$${i+10},$${i+11},$${i+12},$${i+13},$${i+14},$${i+15})`);
    params.push(
      r.time, r.symbol,
      r.d1Short, r.d2Short, r.d1Mid, r.d2Mid, r.d1Long, r.d2Long,
      r.rsi, r.macd, r.macdSignal,
      r.bollUpper, r.bollLower, r.atr,
      r.maDeviation, r.integralDev
    );
    i += 16;
  }

  await pool.query(
    `INSERT INTO features (time, symbol, d1_short, d2_short, d1_mid, d2_mid, d1_long, d2_long, rsi, macd, macd_signal, bollinger_upper, bollinger_lower, atr, ma_deviation, integral_deviation)
     VALUES ${values.join(',')}
     ON CONFLICT (symbol, time) DO NOTHING`,
    params
  );
}

async function main() {
  console.log('🧮 Computing features for all symbols');
  console.log(`   Windows: short=${WIN_SHORT} mid=${WIN_MID} long=${WIN_LONG}`);

  let grandTotal = 0;
  for (const symbol of SYMBOLS) {
    try {
      grandTotal += await processSymbol(symbol);
    } catch (err) {
      console.error(`\n❌ ${symbol}:`, err.message);
    }
  }

  console.log(`\n✅ Done — ${grandTotal.toLocaleString()} features total`);

  const res = await pool.query(`
    SELECT symbol, COUNT(*) as rows, MIN(time)::date as first, MAX(time)::date as last
    FROM features GROUP BY symbol ORDER BY symbol
  `);
  console.log('\n📈 Features table:');
  for (const row of res.rows) {
    console.log(`   ${row.symbol}: ${parseInt(row.rows).toLocaleString()} rows (${row.first} → ${row.last})`);
  }

  await pool.end();
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
