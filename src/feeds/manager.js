import CoinGeckoFeed from './coingecko.js';

class FeedManager {
  constructor() {
    this.coingecko = new CoinGeckoFeed();
    this.prices = {};
    this.priceHistory = {}; // { symbol: [{ price, timestamp }] }
    this.historyLimit = 500;
    this.listeners = [];
  }

  start() {
    // CoinGecko polling every 15s (to avoid rate limits on free tier)
    this.coingecko.start((prices) => {
      for (const [symbol, data] of Object.entries(prices)) {
        this.updatePrice(symbol, data);
      }
    }, 15000);

    console.log('Feed manager started (CoinGecko polling 15s)');
  }

  updatePrice(symbol, data) {
    this.prices[symbol] = data;

    if (!this.priceHistory[symbol]) {
      this.priceHistory[symbol] = [];
    }
    this.priceHistory[symbol].push({
      price: data.price,
      timestamp: data.timestamp
    });

    if (this.priceHistory[symbol].length > this.historyLimit) {
      this.priceHistory[symbol] = this.priceHistory[symbol].slice(-this.historyLimit);
    }

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
  }
}

export default FeedManager;
