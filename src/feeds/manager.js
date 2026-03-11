import CoinGeckoFeed from './coingecko.js';
import DexPaprikaFeed from './dexpaprika.js';

class FeedManager {
  constructor() {
    this.coingecko = new CoinGeckoFeed();
    this.dexpaprika = new DexPaprikaFeed();
    this.prices = {};
    this.priceHistory = {}; // { symbol: [{ price, timestamp }] }
    this.historyLimit = 500; // Keep last 500 data points per symbol
    this.listeners = [];
  }

  start() {
    // CoinGecko polling every 10s
    this.coingecko.start((prices) => {
      for (const [symbol, data] of Object.entries(prices)) {
        // Don't override SOL if DexPaprika has fresher data
        if (symbol === 'SOL' && this.prices.SOL?.source === 'dexpaprika') {
          const dexAge = Date.now() - this.prices.SOL.timestamp;
          if (dexAge < 15000) continue; // Skip if DexPaprika data < 15s old
        }
        this.updatePrice(symbol, data);
      }
    }, 10000);

    // DexPaprika SSE for SOL real-time
    this.dexpaprika.start((data) => {
      this.updatePrice('SOL', data);
    });

    console.log('Feed manager started');
  }

  updatePrice(symbol, data) {
    this.prices[symbol] = data;

    // Add to history
    if (!this.priceHistory[symbol]) {
      this.priceHistory[symbol] = [];
    }
    this.priceHistory[symbol].push({
      price: data.price,
      timestamp: data.timestamp
    });

    // Trim history
    if (this.priceHistory[symbol].length > this.historyLimit) {
      this.priceHistory[symbol] = this.priceHistory[symbol].slice(-this.historyLimit);
    }

    // Notify listeners
    this.emit('price', { symbol, ...data });
  }

  onUpdate(listener) {
    this.listeners.push(listener);
  }

  emit(type, data) {
    for (const listener of this.listeners) {
      listener(type, data);
    }
  }

  getPrices() {
    return { ...this.prices };
  }

  getHistory(symbol) {
    return this.priceHistory[symbol] || [];
  }

  getAllHistory() {
    return { ...this.priceHistory };
  }

  stop() {
    this.coingecko.stop();
    this.dexpaprika.stop();
  }
}

export default FeedManager;
