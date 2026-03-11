class Portfolio {
  constructor(initialCapital = 10000) {
    this.initialCapital = initialCapital;
    this.cash = initialCapital;
    this.positions = {}; // { symbol: { qty, entryPrice, entryTime } }
    this.trades = []; // Historique complet
    this.stats = {
      totalTrades: 0,
      wins: 0,
      losses: 0,
      totalPnl: 0,
      maxDrawdown: 0,
      peakValue: initialCapital
    };
  }

  buy(symbol, price, sizePercent) {
    const amount = this.cash * (sizePercent / 100);
    if (amount <= 0 || amount > this.cash) return null;

    const qty = amount / price;
    this.cash -= amount;

    if (this.positions[symbol]) {
      // Average up/down
      const existing = this.positions[symbol];
      const totalQty = existing.qty + qty;
      const avgPrice = (existing.qty * existing.entryPrice + qty * price) / totalQty;
      this.positions[symbol] = { qty: totalQty, entryPrice: avgPrice, entryTime: Date.now() };
    } else {
      this.positions[symbol] = { qty, entryPrice: price, entryTime: Date.now() };
    }

    const trade = {
      type: 'BUY',
      symbol,
      qty,
      price,
      amount,
      timestamp: Date.now()
    };
    this.trades.push(trade);
    this.stats.totalTrades++;

    return trade;
  }

  sell(symbol, price) {
    const position = this.positions[symbol];
    if (!position) return null;

    const amount = position.qty * price;
    const pnl = amount - (position.qty * position.entryPrice);
    const pnlPercent = (pnl / (position.qty * position.entryPrice)) * 100;

    this.cash += amount;
    delete this.positions[symbol];

    if (pnl > 0) this.stats.wins++;
    else this.stats.losses++;
    this.stats.totalPnl += pnl;

    const trade = {
      type: 'SELL',
      symbol,
      qty: position.qty,
      price,
      amount,
      pnl,
      pnlPercent,
      holdTime: Date.now() - position.entryTime,
      timestamp: Date.now()
    };
    this.trades.push(trade);
    this.stats.totalTrades++;

    // Update drawdown
    const totalValue = this.getTotalValue({ [symbol]: price });
    if (totalValue > this.stats.peakValue) {
      this.stats.peakValue = totalValue;
    }
    const drawdown = ((this.stats.peakValue - totalValue) / this.stats.peakValue) * 100;
    if (drawdown > this.stats.maxDrawdown) {
      this.stats.maxDrawdown = drawdown;
    }

    return trade;
  }

  getTotalValue(currentPrices) {
    let value = this.cash;
    for (const [symbol, pos] of Object.entries(this.positions)) {
      const price = currentPrices[symbol]?.price || currentPrices[symbol] || pos.entryPrice;
      value += pos.qty * price;
    }
    return value;
  }

  getState(currentPrices = {}) {
    const totalValue = this.getTotalValue(currentPrices);
    const pnl = totalValue - this.initialCapital;
    const pnlPercent = (pnl / this.initialCapital) * 100;

    const openPositions = {};
    for (const [symbol, pos] of Object.entries(this.positions)) {
      const currentPrice = currentPrices[symbol]?.price || currentPrices[symbol] || pos.entryPrice;
      const unrealizedPnl = (currentPrice - pos.entryPrice) * pos.qty;
      openPositions[symbol] = {
        ...pos,
        currentPrice,
        unrealizedPnl,
        unrealizedPnlPercent: ((currentPrice - pos.entryPrice) / pos.entryPrice) * 100
      };
    }

    return {
      cash: this.cash,
      totalValue,
      pnl,
      pnlPercent,
      positions: openPositions,
      stats: {
        ...this.stats,
        winRate: this.stats.totalTrades > 0
          ? (this.stats.wins / Math.max(1, this.stats.wins + this.stats.losses)) * 100
          : 0
      },
      recentTrades: this.trades.slice(-20)
    };
  }

  reset() {
    this.cash = this.initialCapital;
    this.positions = {};
    this.trades = [];
    this.stats = {
      totalTrades: 0,
      wins: 0,
      losses: 0,
      totalPnl: 0,
      maxDrawdown: 0,
      peakValue: this.initialCapital
    };
  }
}

export default Portfolio;
