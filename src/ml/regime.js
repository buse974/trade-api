/**
 * Module régime de marché — communique avec le process Python predict_service.py.
 *
 * Usage:
 *   import Regime from './ml/regime.js';
 *   const regime = new Regime();
 *   await regime.start();
 *   const result = await regime.predict('15m');
 *   // { regime: 'actif', proba: 0.87, ... }
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createInterface } from 'readline';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PREDICT_SCRIPT = join(__dirname, 'predict_service.py');

class Regime {
  constructor() {
    this.process = null;
    this.ready = false;
    this.pendingCallbacks = [];
    this.currentRegime = { regime: 'actif', proba: 0.5, stale: true }; // default = trade normally
    this.allRegimes = {};
    this.updateInterval = null;
    this.listeners = [];
  }

  /**
   * Start the Python prediction service.
   * @param {number} refreshMs - How often to refresh predictions (default: 60s)
   */
  async start(refreshMs = 60000) {
    const pythonCmd = process.env.PYTHON_CMD || 'python3';

    return new Promise((resolve, reject) => {
      this.process = spawn(pythonCmd, [PREDICT_SCRIPT], {
        env: {
          ...process.env,
          MODELS_DIR: process.env.MODELS_DIR || join(__dirname, '../../notebooks/models'),
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      // Read stderr for logs
      const stderrRl = createInterface({ input: this.process.stderr });
      stderrRl.on('line', (line) => {
        console.log(`[regime] ${line}`);
      });

      // Read stdout for JSON responses
      const stdoutRl = createInterface({ input: this.process.stdout });
      stdoutRl.on('line', (line) => {
        try {
          const data = JSON.parse(line);

          // First message = ready signal
          if (!this.ready && data.status === 'ready') {
            this.ready = true;
            console.log(`[regime] Service ready, models: ${data.models}`);
            resolve();
            return;
          }

          // Resolve pending callback
          if (this.pendingCallbacks.length > 0) {
            const cb = this.pendingCallbacks.shift();
            cb(data);
          }
        } catch (e) {
          console.error(`[regime] Bad JSON from Python: ${line}`);
        }
      });

      this.process.on('error', (err) => {
        console.error(`[regime] Process error: ${err.message}`);
        this.ready = false;
        if (!this.ready) reject(err);
      });

      this.process.on('exit', (code) => {
        console.log(`[regime] Process exited with code ${code}`);
        this.ready = false;
        // Auto-restart after 5s
        setTimeout(() => {
          console.log('[regime] Restarting...');
          this.start(refreshMs);
        }, 5000);
      });

      // Timeout if Python doesn't start in 10s
      setTimeout(() => {
        if (!this.ready) {
          console.error('[regime] Timeout waiting for Python service');
          resolve(); // Don't block the API, just run without ML
        }
      }, 10000);
    });

    // Start periodic refresh after ready
    this._startRefresh(refreshMs);
  }

  _startRefresh(intervalMs) {
    // Initial prediction
    setTimeout(() => this.refreshAll(), 2000);

    // Periodic refresh
    this.updateInterval = setInterval(() => {
      this.refreshAll();
    }, intervalMs);
  }

  /**
   * Refresh predictions for all horizons.
   */
  async refreshAll() {
    if (!this.ready) return;

    try {
      const result = await this._send({ action: 'predict_all' });
      if (result && !result.error) {
        this.allRegimes = result;
        // Use 15m as the primary regime
        if (result['15m']) {
          this.currentRegime = result['15m'];
        }
        // Notify listeners
        for (const listener of this.listeners) {
          listener(this.allRegimes);
        }
      }
    } catch (e) {
      console.error(`[regime] Refresh error: ${e.message}`);
    }
  }

  /**
   * Get prediction for a specific horizon.
   */
  async predict(horizon = '15m') {
    if (!this.ready) {
      return { regime: 'actif', proba: 0.5, stale: true, error: 'Service not ready' };
    }
    return this._send({ action: 'predict', horizon });
  }

  /**
   * Get the current cached regime (no DB query).
   */
  getRegime() {
    return this.currentRegime;
  }

  /**
   * Get all cached regimes.
   */
  getAllRegimes() {
    return this.allRegimes;
  }

  /**
   * Register listener for regime updates.
   */
  onUpdate(listener) {
    this.listeners.push(listener);
  }

  /**
   * Send command to Python process.
   */
  _send(cmd) {
    return new Promise((resolve, reject) => {
      if (!this.process || !this.ready) {
        resolve({ regime: 'actif', proba: 0.5, stale: true });
        return;
      }

      this.pendingCallbacks.push(resolve);

      try {
        this.process.stdin.write(JSON.stringify(cmd) + '\n');
      } catch (e) {
        this.pendingCallbacks.pop();
        reject(e);
      }

      // Timeout after 5s
      setTimeout(() => {
        const idx = this.pendingCallbacks.indexOf(resolve);
        if (idx !== -1) {
          this.pendingCallbacks.splice(idx, 1);
          resolve({ regime: 'actif', proba: 0.5, stale: true, error: 'timeout' });
        }
      }, 5000);
    });
  }

  stop() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
    this.ready = false;
  }
}

export default Regime;
