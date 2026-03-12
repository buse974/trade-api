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
    '1d': { interval: '1 minute', since: '1 day' },
    '7d': { interval: '15 minutes', since: '7 days' },
    '30d': { interval: '1 hour', since: '30 days' },
    '90d': { interval: '4 hours', since: '90 days' },
    '1y': { interval: '1 day', since: '365 days' },
    'all': { interval: '1 day', since: '100 years' },
  };
  const cfg = intervals[range] || intervals['7d'];

  try {
    const result = await db.query(`
      SELECT
        date_trunc($1, time) as time,
        (array_agg(open ORDER BY time))[1] as open,
        MAX(high) as high,
        MIN(low) as low,
        (array_agg(close ORDER BY time DESC))[1] as close,
        SUM(volume) as volume
      FROM prices
      WHERE symbol = $2 AND time >= NOW() - $3::interval
      GROUP BY date_trunc($1, time)
      ORDER BY time
    `, [cfg.interval, symbol, cfg.since]);

    res.json(result.rows.map(r => ({
      time: Math.floor(new Date(r.time).getTime() / 1000),
      open: parseFloat(r.open),
      high: parseFloat(r.high),
      low: parseFloat(r.low),
      close: parseFloat(r.close),
      volume: parseFloat(r.volume),
    })));
  } catch (err) {
    console.error('History error:', err.message);
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

// --- Backtest ---
app.post('/api/backtest', async (req, res) => {
  const {
    symbol = 'SOL',
    month,          // "2025-01"
    capital = 100,
    stopLoss = 5,
    takeProfit = 10,
    positionSize = 100,  // % du capital à investir
  } = req.body;

  if (!month || !/^\d{4}-\d{2}$/.test(month)) {
    return res.status(400).json({ error: 'month required (format: 2025-01)' });
  }

  const pair = symbol.toUpperCase() + 'USDT';
  const startDate = `${month}-01`;
  const [y, m] = month.split('-').map(Number);
  const endDate = new Date(y, m, 0); // last day of month

  try {
    const result = await db.query(`
      SELECT time, open, high, low, close
      FROM prices
      WHERE symbol = $1 AND time >= $2 AND time < ($3::date + interval '1 day')
      ORDER BY time
    `, [pair, startDate, endDate.toISOString().slice(0, 10)]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'No data for this period' });
    }

    // Simulate
    let cash = capital;
    let position = null; // { qty, entryPrice, entryTime }
    const trades = [];
    const capitalHistory = [];
    let peakValue = capital;
    let maxDrawdown = 0;

    for (const row of result.rows) {
      const price = parseFloat(row.close);
      const high = parseFloat(row.high);
      const low = parseFloat(row.low);
      const time = new Date(row.time).getTime();

      if (position) {
        // Check stop loss on low
        const slPrice = position.entryPrice * (1 - stopLoss / 100);
        if (low <= slPrice) {
          const sellPrice = slPrice;
          const amount = position.qty * sellPrice;
          const pnl = amount - (position.qty * position.entryPrice);
          cash += amount;
          trades.push({ type: 'SELL', price: sellPrice, qty: position.qty, pnl, reason: 'stop_loss', time });
          position = null;
        }
        // Check take profit on high
        else {
          const tpPrice = position.entryPrice * (1 + takeProfit / 100);
          if (high >= tpPrice) {
            const sellPrice = tpPrice;
            const amount = position.qty * sellPrice;
            const pnl = amount - (position.qty * position.entryPrice);
            cash += amount;
            trades.push({ type: 'SELL', price: sellPrice, qty: position.qty, pnl, reason: 'take_profit', time });
            position = null;
          }
        }
      }

      // If no position, buy
      if (!position && cash > 1) {
        const investAmount = cash * (positionSize / 100);
        const qty = investAmount / price;
        cash -= investAmount;
        position = { qty, entryPrice: price, entryTime: time };
        trades.push({ type: 'BUY', price, qty, time });
      }

      // Track capital
      const totalValue = cash + (position ? position.qty * price : 0);
      if (totalValue > peakValue) peakValue = totalValue;
      const dd = ((peakValue - totalValue) / peakValue) * 100;
      if (dd > maxDrawdown) maxDrawdown = dd;

      // Sample capital every 60 rows (~1h) to keep response small
      if (capitalHistory.length === 0 || result.rows.indexOf(row) % 60 === 0) {
        capitalHistory.push({ time: Math.floor(time / 1000), value: totalValue });
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

    res.json({
      symbol,
      month,
      capital,
      finalValue,
      pnl: totalPnl,
      pnlPercent: (totalPnl / capital) * 100,
      trades,
      capitalHistory,
      metrics: {
        totalTrades: trades.filter(t => t.type === 'SELL').length,
        wins,
        losses,
        winRate: wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0,
        maxDrawdown,
        bestTrade: Math.max(...trades.filter(t => t.pnl !== undefined).map(t => t.pnl), 0),
        worstTrade: Math.min(...trades.filter(t => t.pnl !== undefined).map(t => t.pnl), 0),
      },
      dataPoints: result.rows.length,
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
