class TradingEngine {
  constructor(portfolio) {
    this.portfolio = portfolio;
    this.config = {
      stopLoss: 5,         // % de perte max
      takeProfit: 10,      // % de gain cible
      positionSize: 10,    // % du capital par trade
      interval: 30,        // secondes entre décisions
      enabled: false       // bot actif ou non
    };
    this.lastDecisionTime = 0;
    this.onTrade = null;
  }

  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
  }

  evaluate(prices) {
    if (!this.config.enabled) return;

    const now = Date.now();
    if (now - this.lastDecisionTime < this.config.interval * 1000) return;
    this.lastDecisionTime = now;

    // Check stop loss / take profit on open positions
    for (const [symbol, pos] of Object.entries(this.portfolio.positions)) {
      const currentPrice = prices[symbol]?.price;
      if (!currentPrice) continue;

      const pnlPercent = ((currentPrice - pos.entryPrice) / pos.entryPrice) * 100;

      // Stop loss
      if (pnlPercent <= -this.config.stopLoss) {
        const trade = this.portfolio.sell(symbol, currentPrice);
        if (trade && this.onTrade) {
          this.onTrade({ ...trade, reason: 'stop_loss' });
        }
        continue;
      }

      // Take profit
      if (pnlPercent >= this.config.takeProfit) {
        const trade = this.portfolio.sell(symbol, currentPrice);
        if (trade && this.onTrade) {
          this.onTrade({ ...trade, reason: 'take_profit' });
        }
      }
    }

    // Simple entry logic for SOL (will be replaced by ML in phase 3)
    const sol = prices.SOL;
    if (sol && !this.portfolio.positions.SOL) {
      // Basic momentum: buy if speed is positive and not too volatile
      if (sol.speed > 0.05 && sol.speed < 2) {
        const trade = this.portfolio.buy('SOL', sol.price, this.config.positionSize);
        if (trade && this.onTrade) {
          this.onTrade({ ...trade, reason: 'momentum_entry' });
        }
      }
    }
  }

  getConfig() {
    return { ...this.config };
  }
}

export default TradingEngine;
