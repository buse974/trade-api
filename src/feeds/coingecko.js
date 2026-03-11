const COINS = {
  bitcoin: 'BTC',
  ethereum: 'ETH',
  solana: 'SOL',
  dogecoin: 'DOGE',
  shiba_inu: 'SHIB',
  jupiter_exchange_solana: 'JUP',
  dogwifcoin: 'WIF',
  pepe: 'PEPE'
};

const API_URL = 'https://api.coingecko.com/api/v3/simple/price';

class CoinGeckoFeed {
  constructor() {
    this.prices = {};
    this.interval = null;
    this.onUpdate = null;
  }

  start(onUpdate, intervalMs = 10000) {
    this.onUpdate = onUpdate;
    this.poll();
    this.interval = setInterval(() => this.poll(), intervalMs);
    console.log(`CoinGecko feed started (polling every ${intervalMs / 1000}s)`);
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  async poll() {
    try {
      const ids = Object.keys(COINS).join(',');
      const url = `${API_URL}?ids=${ids}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true`;
      const res = await fetch(url);

      if (!res.ok) {
        console.error(`CoinGecko HTTP ${res.status}`);
        return;
      }

      const data = await res.json();
      const now = Date.now();

      for (const [id, symbol] of Object.entries(COINS)) {
        if (data[id]) {
          const prev = this.prices[symbol]?.price || data[id].usd;
          this.prices[symbol] = {
            symbol,
            price: data[id].usd,
            change24h: data[id].usd_24h_change || 0,
            volume24h: data[id].usd_24h_vol || 0,
            speed: prev ? ((data[id].usd - prev) / prev) * 100 : 0,
            timestamp: now,
            source: 'coingecko'
          };
        }
      }

      if (this.onUpdate) {
        this.onUpdate(this.prices);
      }
    } catch (err) {
      console.error('CoinGecko poll error:', err.message);
    }
  }
}

export default CoinGeckoFeed;
