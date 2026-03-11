/**
 * Calcule la vitesse de variation (% par seconde)
 */
export function priceSpeed(history, windowSize = 10) {
  if (history.length < 2) return 0;

  const recent = history.slice(-windowSize);
  const first = recent[0];
  const last = recent[recent.length - 1];
  const timeDiff = (last.timestamp - first.timestamp) / 1000; // en secondes

  if (timeDiff === 0) return 0;
  return ((last.price - first.price) / first.price) * 100 / timeDiff;
}

/**
 * Calcule la volatilité (écart-type des returns)
 */
export function volatility(history, windowSize = 30) {
  if (history.length < 3) return 0;

  const recent = history.slice(-windowSize);
  const returns = [];
  for (let i = 1; i < recent.length; i++) {
    returns.push((recent[i].price - recent[i - 1].price) / recent[i - 1].price);
  }

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;

  return Math.sqrt(variance) * 100; // En pourcentage
}

/**
 * Détecte un spike de volume
 */
export function volumeSpike(currentVolume, avgVolume, threshold = 2) {
  if (!avgVolume || avgVolume === 0) return false;
  return currentVolume / avgVolume > threshold;
}
