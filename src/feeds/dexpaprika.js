import EventSource from 'eventsource';

// SOL/USDC pool on Raydium (most liquid)
const DEXPAPRIKA_SSE_URL = 'https://api.dexpaprika.com/sse/pools';

class DexPaprikaFeed {
  constructor() {
    this.es = null;
    this.price = null;
    this.onUpdate = null;
  }

  start(onUpdate) {
    this.onUpdate = onUpdate;
    this.connect();
    console.log('DexPaprika SSE feed started for SOL/USDC');
  }

  connect() {
    try {
      this.es = new EventSource(DEXPAPRIKA_SSE_URL);

      this.es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Filter for SOL/USDC pools
          if (data.base_token?.symbol === 'SOL' && data.quote_token?.symbol === 'USDC') {
            const now = Date.now();
            const prev = this.price?.price || data.price_usd;

            this.price = {
              symbol: 'SOL',
              price: parseFloat(data.price_usd),
              speed: prev ? ((parseFloat(data.price_usd) - prev) / prev) * 100 : 0,
              volume: data.volume_usd || 0,
              timestamp: now,
              source: 'dexpaprika'
            };

            if (this.onUpdate) {
              this.onUpdate(this.price);
            }
          }
        } catch (err) {
          // Skip unparseable messages
        }
      };

      this.es.onerror = (err) => {
        console.error('DexPaprika SSE error, reconnecting in 5s...');
        this.es.close();
        setTimeout(() => this.connect(), 5000);
      };
    } catch (err) {
      console.error('DexPaprika connection error:', err.message);
      setTimeout(() => this.connect(), 5000);
    }
  }

  stop() {
    if (this.es) {
      this.es.close();
      this.es = null;
    }
  }
}

export default DexPaprikaFeed;
