/**
 * Calcule la matrice de corrélation entre les séries de prix
 * @param {Object} priceHistory - { symbol: [{ price, timestamp }] }
 * @returns {Object} - { matrix: [[]], symbols: [] }
 */
export function calculateCorrelationMatrix(priceHistory) {
  const symbols = Object.keys(priceHistory).filter(
    s => priceHistory[s].length >= 10
  );

  if (symbols.length < 2) {
    return { matrix: [], symbols };
  }

  // Convert to returns
  const returns = {};
  for (const symbol of symbols) {
    const prices = priceHistory[symbol];
    returns[symbol] = [];
    for (let i = 1; i < prices.length; i++) {
      returns[symbol].push(
        (prices[i].price - prices[i - 1].price) / prices[i - 1].price
      );
    }
  }

  // Find min length
  const minLen = Math.min(...symbols.map(s => returns[s].length));
  if (minLen < 5) return { matrix: [], symbols };

  // Trim to same length
  for (const s of symbols) {
    returns[s] = returns[s].slice(-minLen);
  }

  // Build correlation matrix
  const matrix = [];
  for (let i = 0; i < symbols.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < symbols.length; j++) {
      if (i === j) {
        matrix[i][j] = 1;
      } else if (j < i) {
        matrix[i][j] = matrix[j][i]; // Symmetric
      } else {
        matrix[i][j] = pearson(returns[symbols[i]], returns[symbols[j]]);
      }
    }
  }

  return { matrix, symbols };
}

function pearson(x, y) {
  const n = x.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
    sumY2 += y[i] * y[i];
  }

  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  return den === 0 ? 0 : num / den;
}
