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
