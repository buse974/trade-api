import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import swaggerUi from 'swagger-ui-express';
import FeedManager from './feeds/manager.js';
import Portfolio from './simulation/portfolio.js';
import TradingEngine from './simulation/engine.js';
import { calculateCorrelationMatrix } from './utils/correlation.js';
import db from './db.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const server = createServer(app);

// --- State ---
const feedManager = new FeedManager();
const portfolio = new Portfolio(10000);
const engine = new TradingEngine(portfolio);

// --- Swagger ---
const swaggerDocument = {
  openapi: '3.0.0',
  info: {
    title: 'trade API',
    version: '1.0.0',
    description: 'Bot de trading SOL/USDC avec simulation et prédiction ML multi-crypto'
  },
  servers: [
    { url: 'http://localhost:3000', description: 'Local' },
    { url: 'https://api.trade.51.77.223.61.nip.io', description: 'Production' }
  ],
  paths: {
    '/health': {
      get: { summary: 'Health check', responses: { 200: { description: 'OK' } } }
    },
    '/api/prices': {
      get: { summary: 'Current prices', responses: { 200: { description: 'All prices' } } }
    },
    '/api/portfolio': {
      get: { summary: 'Portfolio state', responses: { 200: { description: 'Portfolio' } } }
    },
    '/api/correlation': {
      get: { summary: 'Correlation matrix', responses: { 200: { description: 'Matrix' } } }
    },
    '/api/config': {
      get: { summary: 'Bot config', responses: { 200: { description: 'Config' } } },
      post: { summary: 'Update bot config', responses: { 200: { description: 'Updated' } } }
    }
  }
};

app.use(cors());
app.use(express.json());
app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// --- REST Routes ---
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'trade-api', type: 'api' });
});

app.get('/api/prices', (req, res) => {
  res.json(feedManager.getPrices());
});

app.get('/api/portfolio', (req, res) => {
  res.json(portfolio.getState(feedManager.getPrices()));
});

app.get('/api/correlation', (req, res) => {
  const result = calculateCorrelationMatrix(feedManager.getAllHistory());
  res.json(result);
});

app.get('/api/config', (req, res) => {
  res.json(engine.getConfig());
});

app.post('/api/config', (req, res) => {
  const allowed = ['stopLoss', 'takeProfit', 'positionSize', 'interval', 'enabled'];
  const filtered = {};
  for (const key of allowed) {
    if (req.body[key] !== undefined) {
      if (key === 'enabled') {
        filtered[key] = Boolean(req.body[key]);
      } else {
        const val = Number(req.body[key]);
        if (!isNaN(val) && val >= 0 && val <= 100) filtered[key] = val;
      }
    }
  }
  engine.updateConfig(filtered);
  res.json(engine.getConfig());
});

app.post('/api/portfolio/reset', (req, res) => {
  portfolio.reset();
  res.json(portfolio.getState());
});

// --- Historical prices from DB ---
app.get('/api/history/:symbol', async (req, res) => {
  const symbol = req.params.symbol.toUpperCase() + 'USDT';
  const range = req.query.range || '7d'; // 1d, 7d, 30d, 90d, 1y, all
  const intervals = {
    '1d': { truncTo: 'minute', since: '1 day' },
    '7d': { truncTo: 'hour', since: '7 days' },
    '30d': { truncTo: 'hour', since: '30 days' },
    '90d': { truncTo: 'hour', bucketSeconds: 14400, since: '90 days' },
    '1y': { truncTo: 'day', since: '365 days' },
    'all': { truncTo: 'day', since: '100 years' },
  };
  const cfg = intervals[range] || intervals['7d'];

  try {
    const bucketExpr = cfg.bucketSeconds
      ? `to_timestamp(floor(extract(epoch from time) / ${cfg.bucketSeconds}) * ${cfg.bucketSeconds})`
      : `date_trunc('${cfg.truncTo}', time)`;

    const result = await db.query(`
      SELECT
        ${bucketExpr} as time,
        (array_agg(open ORDER BY time))[1] as open,
        MAX(high) as high,
        MIN(low) as low,
        (array_agg(close ORDER BY time DESC))[1] as close,
        SUM(volume) as volume
      FROM prices
      WHERE symbol = $1 AND time >= NOW() - $2::interval
      GROUP BY ${bucketExpr}
      ORDER BY time
    `, [symbol, cfg.since]);

    const rows = result.rows
      .map(r => ({
        time: Math.floor(new Date(r.time).getTime() / 1000),
        open: parseFloat(r.open),
        high: parseFloat(r.high),
        low: parseFloat(r.low),
        close: parseFloat(r.close),
        volume: parseFloat(r.volume),
      }))
      .filter(r => isFinite(r.open) && isFinite(r.close) && r.time > 0);
    res.json(rows);
  } catch (err) {
    console.error('History error:', err.message);
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

// --- Derivatives calculation ---
function slopeOverWindow(prices, end, win) {
  const start = Math.max(0, end - win);
  const n = end - start;
  if (n < 2) return 0;
  // Linear regression slope using only indices
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    const y = prices[start + i];
    sumX += i;
    sumY += y;
    sumXY += i * y;
    sumX2 += i * i;
  }
  return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
}

function getMarketState(prices, idx, win, prevSlopes) {
  if (idx < win) return { d1: 0, d2: 0, profile: 'bullish' };

  const d1 = slopeOverWindow(prices, idx, win);
  prevSlopes.push(d1);

  const halfWin = Math.floor(win / 2);
  const d2 = prevSlopes.length >= halfWin
    ? slopeOverWindow(prevSlopes, prevSlopes.length, halfWin)
    : 0;

  let profile;
  if (d1 > 0 && d2 > 0) profile = 'bullish';
  else if (d1 > 0 && d2 <= 0) profile = 'exhaustion';
  else if (d1 <= 0 && d2 <= 0) profile = 'bearish';
  else profile = 'reversal';

  return { d1, d2, profile };
}

// --- Backtest ---
app.post('/api/backtest', async (req, res) => {
  const {
    symbol = 'SOL',
    month,
    capital = 100,
    window = 60,        // derivative calculation window (minutes)
    profiles = null,     // { bullish, exhaustion, bearish, reversal } each with { stopLoss, takeProfit, positionSize, enabled }
    // Fallback: simple mode with single set of params
    stopLoss = 5,
    takeProfit = 10,
    positionSize = 100,
  } = req.body;

  if (!month || !/^\d{4}-\d{2}$/.test(month)) {
    return res.status(400).json({ error: 'month required (format: 2025-01)' });
  }

  // Build profiles: either from request or single-mode fallback
  const defaultProfile = { stopLoss, takeProfit, positionSize, enabled: true };
  const p = profiles || {
    bullish: { ...defaultProfile },
    exhaustion: { ...defaultProfile },
    bearish: { stopLoss: 0, takeProfit: 0, positionSize: 0, enabled: false },
    reversal: { ...defaultProfile },
  };

  const pair = symbol.toUpperCase() + 'USDT';
  const startDate = `${month}-01`;
  const [y, m] = month.split('-').map(Number);
  const endDate = new Date(y, m, 0);

  try {
    // Fetch extra data before the month for derivative warmup
    const warmupDate = new Date(y, m - 1, 1);
    warmupDate.setDate(warmupDate.getDate() - window * 2);

    const result = await db.query(`
      SELECT time, open, high, low, close
      FROM prices
      WHERE symbol = $1 AND time >= $2 AND time < ($3::date + interval '1 day')
      ORDER BY time
    `, [pair, warmupDate.toISOString().slice(0, 10), endDate.toISOString().slice(0, 10)]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'No data for this period' });
    }

    // Find index where the actual month starts
    const monthStart = new Date(`${month}-01T00:00:00Z`).getTime();
    const monthStartIdx = result.rows.findIndex(r => new Date(r.time).getTime() >= monthStart);
    if (monthStartIdx === -1) {
      return res.status(404).json({ error: 'No data for this month' });
    }

    let cash = capital;
    let position = null;
    const trades = [];
    const capitalHistory = [];
    const profileHistory = [];
    let peakValue = capital;
    let maxDrawdown = 0;
    const closePrices = [];
    const slopeHistory = [];
    let currentState = { d1: 0, d2: 0, profile: 'bullish' };

    for (let i = 0; i < result.rows.length; i++) {
      const row = result.rows[i];
      const price = parseFloat(row.close);
      const high = parseFloat(row.high);
      const low = parseFloat(row.low);
      const time = new Date(row.time).getTime();
      closePrices.push(price);

      // Only trade during the actual month (warmup period is just for derivative calculation)
      if (i < monthStartIdx) continue;

      // Recalculate market state every 5 minutes
      const inMonth = i - monthStartIdx;
      if (inMonth % 5 === 0) {
        currentState = getMarketState(closePrices, closePrices.length, window, slopeHistory);
      }
      const state = currentState;
      const activeProfile = p[state.profile];

      if (position) {
        if (!activeProfile.enabled) {
          // Profile says sell everything
          const amount = position.qty * price;
          const pnl = amount - (position.qty * position.entryPrice);
          cash += amount;
          trades.push({ type: 'SELL', price, qty: position.qty, pnl, reason: 'profile_exit', profile: state.profile, time });
          position = null;
        } else {
          // Check stop loss on low
          const slPrice = position.entryPrice * (1 - activeProfile.stopLoss / 100);
          if (low <= slPrice) {
            const amount = position.qty * slPrice;
            const pnl = amount - (position.qty * position.entryPrice);
            cash += amount;
            trades.push({ type: 'SELL', price: slPrice, qty: position.qty, pnl, reason: 'stop_loss', profile: state.profile, time });
            position = null;
          }
          // Check take profit on high
          else {
            const tpPrice = position.entryPrice * (1 + activeProfile.takeProfit / 100);
            if (high >= tpPrice) {
              const amount = position.qty * tpPrice;
              const pnl = amount - (position.qty * position.entryPrice);
              cash += amount;
              trades.push({ type: 'SELL', price: tpPrice, qty: position.qty, pnl, reason: 'take_profit', profile: state.profile, time });
              position = null;
            }
          }
        }
      }

      // Buy if no position and profile allows it
      if (!position && cash > 1 && activeProfile.enabled && activeProfile.positionSize > 0) {
        const investAmount = cash * (activeProfile.positionSize / 100);
        const qty = investAmount / price;
        cash -= investAmount;
        position = { qty, entryPrice: price, entryTime: time };
        trades.push({ type: 'BUY', price, qty, profile: state.profile, time });
      }

      const totalValue = cash + (position ? position.qty * price : 0);
      if (totalValue > peakValue) peakValue = totalValue;
      const dd = ((peakValue - totalValue) / peakValue) * 100;
      if (dd > maxDrawdown) maxDrawdown = dd;

      // Sample every 60 rows (~1h)
      if (inMonth % 60 === 0) {
        capitalHistory.push({ time: Math.floor(time / 1000), value: totalValue, profile: state.profile });
        profileHistory.push({ time: Math.floor(time / 1000), profile: state.profile, d1: state.d1, d2: state.d2 });
      }
    }

    // Close position at end of month
    if (position) {
      const lastPrice = parseFloat(result.rows[result.rows.length - 1].close);
      const amount = position.qty * lastPrice;
      const pnl = amount - (position.qty * position.entryPrice);
      cash += amount;
      trades.push({ type: 'SELL', price: lastPrice, qty: position.qty, pnl, reason: 'end_of_period', time: new Date(result.rows[result.rows.length - 1].time).getTime() });
      position = null;
    }

    const finalValue = cash;
    const totalPnl = finalValue - capital;
    const wins = trades.filter(t => t.type === 'SELL' && t.pnl > 0).length;
    const losses = trades.filter(t => t.type === 'SELL' && t.pnl <= 0).length;

    // Add final point
    capitalHistory.push({ time: capitalHistory[capitalHistory.length - 1]?.time, value: finalValue });

    // Profile distribution stats
    const profileCounts = { bullish: 0, exhaustion: 0, bearish: 0, reversal: 0 };
    for (const ph of profileHistory) profileCounts[ph.profile]++;
    const totalSamples = profileHistory.length || 1;

    res.json({
      symbol,
      month,
      capital,
      finalValue,
      pnl: totalPnl,
      pnlPercent: (totalPnl / capital) * 100,
      trades,
      capitalHistory,
      profileHistory,
      metrics: {
        totalTrades: trades.filter(t => t.type === 'SELL').length,
        wins,
        losses,
        winRate: wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0,
        maxDrawdown,
        bestTrade: Math.max(...trades.filter(t => t.pnl !== undefined).map(t => t.pnl), 0),
        worstTrade: Math.min(...trades.filter(t => t.pnl !== undefined).map(t => t.pnl), 0),
        profileDistribution: {
          bullish: ((profileCounts.bullish / totalSamples) * 100).toFixed(0),
          exhaustion: ((profileCounts.exhaustion / totalSamples) * 100).toFixed(0),
          bearish: ((profileCounts.bearish / totalSamples) * 100).toFixed(0),
          reversal: ((profileCounts.reversal / totalSamples) * 100).toFixed(0),
        },
      },
      dataPoints: result.rows.length - monthStartIdx,
    });
  } catch (err) {
    console.error('Backtest error:', err.message);
    res.status(500).json({ error: 'Backtest failed' });
  }
});

// --- WebSocket ---
const wss = new WebSocketServer({ server, path: '/ws' });

function broadcast(type, data) {
  const message = JSON.stringify({ type, data, timestamp: Date.now() });
  for (const client of wss.clients) {
    if (client.readyState === 1) { // OPEN
      client.send(message);
    }
  }
}

wss.on('connection', (ws) => {
  console.log('Client connected');

  // Send current state on connect
  ws.send(JSON.stringify({
    type: 'init',
    data: {
      prices: feedManager.getPrices(),
      portfolio: portfolio.getState(feedManager.getPrices()),
      config: engine.getConfig(),
      correlation: calculateCorrelationMatrix(feedManager.getAllHistory())
    },
    timestamp: Date.now()
  }));

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'config') {
        engine.updateConfig(msg.data);
        broadcast('config', engine.getConfig());
      }
    } catch (err) {
      // Ignore bad messages
    }
  });

  ws.on('close', () => console.log('Client disconnected'));
});

// --- Feed updates → broadcast + engine ---
feedManager.onUpdate((type, data) => {
  if (type === 'price') {
    broadcast('price', data);
    engine.evaluate(feedManager.getPrices());
  }
});

// Trade events
engine.onTrade = (trade) => {
  broadcast('trade', trade);
  broadcast('portfolio', portfolio.getState(feedManager.getPrices()));
};

// Correlation update every 30s
setInterval(() => {
  const corr = calculateCorrelationMatrix(feedManager.getAllHistory());
  if (corr.symbols.length >= 2) {
    broadcast('correlation', corr);
  }
}, 30000);

// Portfolio state broadcast every 5s
setInterval(() => {
  broadcast('portfolio', portfolio.getState(feedManager.getPrices()));
}, 5000);

// --- Start ---
feedManager.start();

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`WebSocket: ws://localhost:${PORT}/ws`);
  console.log(`Swagger docs: http://localhost:${PORT}/docs`);
});
