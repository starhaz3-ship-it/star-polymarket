"""
HMM Regime Filter V1.0 — Hidden Markov Model for market regime classification.

Classifies BTC into one of 3 regimes:
  0 = TRENDING    (strong directional moves — good for momentum)
  1 = MEAN_REVERT (oscillating around a level — good for contrarian/RSI)
  2 = CHOPPY      (low vol, no direction — bad for everything)

Uses Gaussian HMM with Baum-Welch (EM) on returns + volatility features.
Retrains every RETRAIN_INTERVAL minutes on rolling LOOKBACK_BARS of 1-min data.

Usage:
    from hmm_regime_filter import HMMRegimeFilter

    hmm = HMMRegimeFilter()
    await hmm.start()  # seeds data from Binance REST

    # In your signal loop:
    regime = hmm.current_regime()       # "TRENDING" / "MEAN_REVERT" / "CHOPPY"
    ok = hmm.is_favorable("momentum")   # True if regime suits strategy
    ok = hmm.is_favorable("contrarian") # True if mean-reverting
    report = hmm.get_report()           # printable status string

Standalone test:
    python hmm_regime_filter.py
"""

import json
import time
import math
import asyncio
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

try:
    import httpx
except ImportError:
    httpx = None

# ============================================================================
# CONFIG
# ============================================================================
LOOKBACK_BARS = 120         # 1-min bars for training (2 hours)
RETRAIN_INTERVAL = 300      # retrain every 5 minutes
N_REGIMES = 3               # TRENDING, MEAN_REVERT, CHOPPY
EM_MAX_ITER = 50            # Baum-Welch iterations
EM_TOL = 1e-4               # convergence tolerance
MIN_BARS_TO_FIT = 60        # need at least 60 bars before first fit
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"

# Regime labels (assigned after fitting based on feature characteristics)
REGIME_NAMES = {0: "TRENDING", 1: "MEAN_REVERT", 2: "CHOPPY"}

# Strategy compatibility
STRATEGY_REGIMES = {
    "momentum":   ["TRENDING", "MEAN_REVERT", "CHOPPY"],  # V2.4: Added CHOPPY — 15M binary options resolve too fast for regime to matter. Sniper proves 90% WR in CHOPPY.
    "contrarian": ["MEAN_REVERT"],
    "rsi":        ["MEAN_REVERT", "TRENDING"],
    "arb":        ["CHOPPY", "MEAN_REVERT", "TRENDING"],  # arb works in any regime
    "sniper":     ["TRENDING", "MEAN_REVERT", "CHOPPY"],  # V2.4: Sniper works in all regimes (90% WR live)
}


# ============================================================================
# GAUSSIAN HMM (pure numpy, no hmmlearn dependency)
# ============================================================================
class GaussianHMM:
    """
    Gaussian Hidden Markov Model with diagonal covariance.
    Implements Baum-Welch (EM) for training and Viterbi for decoding.
    """

    def __init__(self, n_states: int = 3, n_features: int = 2):
        self.n_states = n_states
        self.n_features = n_features
        self.pi = np.ones(n_states) / n_states  # initial state probs
        self.A = np.ones((n_states, n_states)) / n_states  # transition matrix
        self.means = np.zeros((n_states, n_features))
        self.vars = np.ones((n_states, n_features))  # diagonal covariance
        self.fitted = False

    def _emission_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute P(x_t | state_k) for all states. x shape: (T, D), returns (T, K)."""
        T = x.shape[0]
        K = self.n_states
        D = self.n_features
        probs = np.zeros((T, K))
        for k in range(K):
            diff = x - self.means[k]  # (T, D)
            var = self.vars[k] + 1e-10  # (D,)
            exponent = -0.5 * np.sum(diff ** 2 / var, axis=1)
            norm = np.prod(np.sqrt(2 * np.pi * var))
            probs[:, k] = np.exp(exponent) / (norm + 1e-300)
        return probs

    def _forward(self, emit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm. Returns alpha (T, K) and scaling factors (T,)."""
        T, K = emit.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        alpha[0] = self.pi * emit[0]
        scale[0] = alpha[0].sum() + 1e-300
        alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = emit[t] * (alpha[t - 1] @ self.A)
            scale[t] = alpha[t].sum() + 1e-300
            alpha[t] /= scale[t]

        return alpha, scale

    def _backward(self, emit: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward algorithm. Returns beta (T, K)."""
        T, K = emit.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (emit[t + 1] * beta[t + 1]))
            beta[t] /= scale[t + 1] + 1e-300

        return beta

    def fit(self, X: np.ndarray, max_iter: int = EM_MAX_ITER, tol: float = EM_TOL):
        """Baum-Welch EM algorithm. X shape: (T, D)."""
        T, D = X.shape
        K = self.n_states

        # Initialize with k-means-style splitting
        self._init_params(X)

        prev_ll = -np.inf
        for iteration in range(max_iter):
            emit = self._emission_prob(X)  # (T, K)

            # E-step
            alpha, scale = self._forward(emit)
            beta = self._backward(emit, scale)

            # Log-likelihood
            ll = np.sum(np.log(scale + 1e-300))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # Posterior state probabilities gamma(t, k)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Xi: joint posterior of (state_t, state_{t+1})
            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                num = np.outer(alpha[t], emit[t + 1] * beta[t + 1]) * self.A
                xi[t] = num / (num.sum() + 1e-300)

            # M-step
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)

            xi_sum = xi.sum(axis=0)
            self.A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

            for k in range(K):
                wk = gamma[:, k]
                wk_sum = wk.sum() + 1e-300
                self.means[k] = (wk[:, None] * X).sum(axis=0) / wk_sum
                diff = X - self.means[k]
                self.vars[k] = (wk[:, None] * diff ** 2).sum(axis=0) / wk_sum
                self.vars[k] = np.maximum(self.vars[k], 1e-6)  # floor variance

        self.fitted = True

    def _init_params(self, X: np.ndarray):
        """Initialize parameters via sorted percentile splits."""
        T = X.shape[0]
        K = self.n_states

        # Sort by first feature (returns) and split into K groups
        order = np.argsort(X[:, 0])
        chunk = T // K
        for k in range(K):
            start = k * chunk
            end = (k + 1) * chunk if k < K - 1 else T
            subset = X[order[start:end]]
            self.means[k] = subset.mean(axis=0)
            self.vars[k] = subset.var(axis=0) + 1e-6

        # Slight self-persistence in transitions
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)
        self.pi = np.ones(K) / K

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding. Returns most likely state sequence."""
        if not self.fitted:
            return np.zeros(X.shape[0], dtype=int)

        T = X.shape[0]
        K = self.n_states
        emit = self._emission_prob(X)

        # Log-space Viterbi
        log_pi = np.log(self.pi + 1e-300)
        log_A = np.log(self.A + 1e-300)
        log_emit = np.log(emit + 1e-300)

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        delta[0] = log_pi + log_emit[0]

        for t in range(1, T):
            for k in range(K):
                scores = delta[t - 1] + log_A[:, k]
                psi[t, k] = np.argmax(scores)
                delta[t, k] = scores[psi[t, k]] + log_emit[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probabilities for last observation. Shape: (K,)."""
        if not self.fitted:
            return np.ones(self.n_states) / self.n_states

        emit = self._emission_prob(X)
        alpha, scale = self._forward(emit)
        # Last time step posterior = normalized alpha
        proba = alpha[-1] / (alpha[-1].sum() + 1e-300)
        return proba


# ============================================================================
# HMM REGIME FILTER
# ============================================================================
class HMMRegimeFilter:
    """
    Market regime classifier using Gaussian HMM on BTC 1-min returns + volatility.

    Features:
        1. Log returns (1-min): captures direction/magnitude
        2. Rolling volatility (5-bar std of returns): captures regime stability

    After fitting, labels regimes by their characteristics:
        - Highest |mean_return| + highest vol → TRENDING
        - Low |mean_return| + moderate vol → MEAN_REVERT
        - Lowest vol overall → CHOPPY
    """

    STATE_FILE = Path(__file__).parent / "hmm_regime_state.json"

    def __init__(self):
        self.hmm = GaussianHMM(n_states=N_REGIMES, n_features=2)
        self.closes: deque = deque(maxlen=LOOKBACK_BARS + 10)
        self._regime: str = "UNKNOWN"
        self._regime_proba: dict = {}
        self._last_retrain: float = 0
        self._label_map: dict = {}  # maps HMM state index -> regime name
        self._retrain_count: int = 0
        self._history: list = []  # recent regime history for display
        self._load_state()

    def _load_state(self):
        """Load persisted regime state."""
        try:
            with open(self.STATE_FILE) as f:
                data = json.load(f)
            self._label_map = {int(k): v for k, v in data.get("label_map", {}).items()}
            self._retrain_count = data.get("retrain_count", 0)
            self._history = data.get("history", [])[-100:]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Persist regime state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "label_map": {str(k): v for k, v in self._label_map.items()},
                    "retrain_count": self._retrain_count,
                    "regime": self._regime,
                    "history": self._history[-100:],
                    "last_retrain": self._last_retrain,
                }, f, indent=2)
        except Exception:
            pass

    async def start(self, feed=None):
        """
        Initialize with Binance REST candles.
        If feed is provided (BinanceLiveFeed), use its candle buffer instead.
        """
        if feed and hasattr(feed, 'get_candles'):
            candles = feed.get_candles()
            for c in candles:
                self.closes.append(float(c["close"]))
            print(f"[HMM] Seeded {len(self.closes)} bars from live feed")
        else:
            await self._seed_from_rest()

        if len(self.closes) >= MIN_BARS_TO_FIT:
            self._retrain()

    async def _seed_from_rest(self):
        """Fetch initial candle data from Binance REST."""
        if httpx is None:
            print("[HMM] httpx not installed, cannot seed from REST")
            return
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": LOOKBACK_BARS
                })
                r.raise_for_status()
                for k in r.json():
                    self.closes.append(float(k[4]))  # close price
            print(f"[HMM] Seeded {len(self.closes)} bars from Binance REST")
        except Exception as e:
            print(f"[HMM] REST seed error: {e}")

    def update(self, close_price: float):
        """
        Feed a new 1-min close price. Automatically retrains when interval elapsed.
        Call this from your main loop whenever you get a new candle.
        """
        self.closes.append(close_price)

        now = time.time()
        if (now - self._last_retrain >= RETRAIN_INTERVAL and
                len(self.closes) >= MIN_BARS_TO_FIT):
            self._retrain()

    def _build_features(self) -> Optional[np.ndarray]:
        """Build feature matrix from close prices. Returns (T-1, 2) or None."""
        closes = list(self.closes)
        if len(closes) < MIN_BARS_TO_FIT:
            return None

        prices = np.array(closes, dtype=float)
        # Feature 1: log returns
        log_returns = np.diff(np.log(prices + 1e-10))

        # Feature 2: rolling 5-bar volatility of returns
        vol_window = 5
        T = len(log_returns)
        rolling_vol = np.zeros(T)
        for i in range(T):
            start = max(0, i - vol_window + 1)
            window = log_returns[start:i + 1]
            rolling_vol[i] = np.std(window) if len(window) > 1 else 0.0

        # Stack features
        X = np.column_stack([log_returns, rolling_vol])
        return X

    def _retrain(self):
        """Retrain HMM on current data and classify regimes."""
        X = self._build_features()
        if X is None or len(X) < MIN_BARS_TO_FIT:
            return

        try:
            self.hmm.fit(X)
        except Exception as e:
            print(f"[HMM] Fit error: {e}")
            return

        # Decode states
        states = self.hmm.predict(X)

        # Label regimes by their characteristics
        self._assign_labels()

        # Current regime = last decoded state
        last_state = int(states[-1])
        self._regime = self._label_map.get(last_state, f"STATE_{last_state}")

        # Probabilities for current observation
        proba = self.hmm.predict_proba(X)
        self._regime_proba = {
            self._label_map.get(k, f"STATE_{k}"): float(proba[k])
            for k in range(N_REGIMES)
        }

        self._last_retrain = time.time()
        self._retrain_count += 1

        # Track history
        self._history.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "regime": self._regime,
            "proba": {k: round(v, 3) for k, v in self._regime_proba.items()},
        })

        self._save_state()

    def _assign_labels(self):
        """
        Assign semantic labels to HMM states based on learned parameters.
        - TRENDING: highest absolute mean return
        - CHOPPY: lowest volatility (mean of var feature)
        - MEAN_REVERT: the remaining one
        """
        means = self.hmm.means  # (K, 2) — [return, vol]
        vars_ = self.hmm.vars   # (K, 2)

        abs_returns = np.abs(means[:, 0])
        mean_vol = means[:, 1]

        # TRENDING = state with largest |return|
        trending_idx = int(np.argmax(abs_returns))

        # Among the remaining, CHOPPY = lowest vol
        remaining = [i for i in range(N_REGIMES) if i != trending_idx]
        if len(remaining) == 2:
            choppy_idx = remaining[0] if mean_vol[remaining[0]] < mean_vol[remaining[1]] else remaining[1]
            mean_revert_idx = remaining[0] if choppy_idx == remaining[1] else remaining[1]
        else:
            choppy_idx = remaining[0]
            mean_revert_idx = remaining[0]

        self._label_map = {
            trending_idx: "TRENDING",
            choppy_idx: "CHOPPY",
            mean_revert_idx: "MEAN_REVERT",
        }

    def current_regime(self) -> str:
        """Return current regime label: TRENDING, MEAN_REVERT, CHOPPY, or UNKNOWN."""
        return self._regime

    def current_proba(self) -> dict:
        """Return probability distribution over regimes."""
        return dict(self._regime_proba)

    def is_favorable(self, strategy_type: str) -> bool:
        """
        Check if current regime is favorable for the given strategy type.
        Returns True if current regime is in the strategy's compatible list.
        If HMM hasn't been fit yet, returns True (permissive default).
        """
        if self._regime == "UNKNOWN":
            return True  # don't block trades before first fit

        compatible = STRATEGY_REGIMES.get(strategy_type, list(REGIME_NAMES.values()))
        return self._regime in compatible

    def get_regime_confidence(self) -> float:
        """Return confidence (0-1) in current regime classification."""
        if not self._regime_proba:
            return 0.0
        return max(self._regime_proba.values())

    def get_report(self) -> str:
        """Printable regime status."""
        lines = [f"[HMM] Regime: {self._regime} (retrain #{self._retrain_count})"]
        if self._regime_proba:
            probs = " | ".join(
                f"{name}={p:.0%}" for name, p in sorted(self._regime_proba.items())
            )
            lines.append(f"  Probabilities: {probs}")
        if self.hmm.fitted:
            for k in range(N_REGIMES):
                label = self._label_map.get(k, f"STATE_{k}")
                m_ret = self.hmm.means[k, 0] * 10000  # basis points
                m_vol = self.hmm.means[k, 1] * 10000
                lines.append(f"  {label:12s} | mean_ret={m_ret:+.1f}bp | vol={m_vol:.1f}bp")
            # Transition matrix
            lines.append("  Transitions:")
            for i in range(N_REGIMES):
                from_label = self._label_map.get(i, f"S{i}")
                probs = " ".join(f"{self.hmm.A[i, j]:.2f}" for j in range(N_REGIMES))
                to_labels = " ".join(f"{self._label_map.get(j, f'S{j}'):>12s}" for j in range(N_REGIMES))
                lines.append(f"    {from_label:>12s} -> [{probs}]")
        lines.append(f"  Data: {len(self.closes)} bars | Last retrain: "
                     f"{datetime.fromtimestamp(self._last_retrain).strftime('%H:%M:%S') if self._last_retrain else 'never'}")
        return "\n".join(lines)

    def get_transition_proba(self, from_regime: str = None) -> dict:
        """Get transition probabilities FROM current (or specified) regime."""
        if not self.hmm.fitted:
            return {}

        if from_regime is None:
            from_regime = self._regime

        # Find state index for this regime
        from_idx = None
        for idx, label in self._label_map.items():
            if label == from_regime:
                from_idx = idx
                break

        if from_idx is None:
            return {}

        return {
            self._label_map.get(j, f"STATE_{j}"): float(self.hmm.A[from_idx, j])
            for j in range(N_REGIMES)
        }


# ============================================================================
# STANDALONE TEST
# ============================================================================
async def _test():
    """Run standalone test: fetch BTC data, fit HMM, display regime."""
    print("=" * 60)
    print("  HMM REGIME FILTER — STANDALONE TEST")
    print("=" * 60)

    hmm_filter = HMMRegimeFilter()
    await hmm_filter.start()

    print()
    print(hmm_filter.get_report())
    print()

    regime = hmm_filter.current_regime()
    print(f"Current regime: {regime}")
    print(f"Momentum favorable: {hmm_filter.is_favorable('momentum')}")
    print(f"Contrarian favorable: {hmm_filter.is_favorable('contrarian')}")
    print(f"Confidence: {hmm_filter.get_regime_confidence():.1%}")
    print()

    # Show transition probabilities
    trans = hmm_filter.get_transition_proba()
    if trans:
        print(f"From {regime}, next regime probabilities:")
        for name, prob in sorted(trans.items(), key=lambda x: -x[1]):
            print(f"  -> {name}: {prob:.1%}")


if __name__ == "__main__":
    asyncio.run(_test())
