"""
BTC Direction Predictor — ML Model for 5m and 15m Polymarket Trading
====================================================================
Trains LightGBM on Binance BTC candle data to predict short-term direction.
Supports both 5-minute and 15-minute timeframes.

Features: RSI, MACD, BB position, momentum, volume, volatility, time-of-day
Training: 1000+ Binance candles → labeled UP/DOWN → LightGBM classifier
Prediction: Returns (direction, confidence, kelly_size)

Usage:
    predictor = BTCPredictor(timeframe='5m')
    await predictor.initialize()  # Fetch data + train
    direction, confidence, kelly = predictor.predict(candles)
"""

import numpy as np
import json
import os
import time
import asyncio
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

try:
    import lightgbm as lgb
    import requests
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False


# ============================================================
# Feature names for each timeframe
# ============================================================
FEATURE_NAMES_5M = [
    'rsi_14', 'rsi_7', 'rsi_slope',
    'macd_hist', 'macd_hist_delta', 'macd_signal',
    'bb_position', 'bb_width',
    'momentum_1bar', 'momentum_3bar', 'momentum_5bar', 'momentum_10bar',
    'volume_ratio', 'volume_trend',
    'volatility_5bar', 'volatility_20bar', 'vol_ratio',
    'atr_14', 'atr_ratio',
    'candle_body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    'consec_direction', 'close_vs_open',
    'hour_sin', 'hour_cos',
    'returns_std_10', 'returns_skew_10',
    'high_low_range', 'close_position_in_range',
]

FEATURE_NAMES_15M = FEATURE_NAMES_5M  # Same feature set, different timeframe data


@dataclass
class MLPrediction:
    direction: str          # "UP", "DOWN", or "SKIP"
    confidence: float       # 0.0 to 1.0
    up_probability: float   # P(UP)
    kelly_fraction: float   # Optimal bet fraction (quarter-Kelly)
    kelly_size_usd: float   # Suggested USD size
    vol_regime: str = "normal"       # "high", "normal", "low"
    vol_score: float = 0.5           # 0=low vol, 1=high vol (good for Polymarket)
    momentum_quality: float = 0.5    # 0=weak/choppy, 1=strong/clean
    trade_score: float = 0.5         # Combined quality score 0-1
    features: Optional[np.ndarray] = None


class BTCPredictor:
    """
    LightGBM-based BTC direction predictor.
    Trained on Binance candle data, used for Polymarket entry decisions.
    """

    def __init__(self, timeframe: str = '5m', bankroll: float = 38.0,
                 model_path: str = None):
        if not HAS_ML:
            raise ImportError("lightgbm, requests, sklearn required")

        self.timeframe = timeframe
        self.bankroll = bankroll
        self.model_path = model_path or f"ml_btc_{timeframe}_model.json"
        self.state_path = f"ml_btc_{timeframe}_state.json"

        # Models (3 models for different aspects)
        self.model: Optional[lgb.Booster] = None           # Direction/momentum continuation
        self.vol_model: Optional[lgb.Booster] = None       # Volatility prediction
        self.quality_model: Optional[lgb.Booster] = None   # Momentum quality
        self.scaler = StandardScaler()
        self.vol_scaler = StandardScaler()
        self.is_trained = False

        # Training data
        self.candles_raw: List[dict] = []
        self.live_outcomes: List[dict] = []  # From Polymarket results

        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.last_train_time = 0
        self.train_interval = 1800  # Retrain every 30 min

        # Thresholds
        self.MIN_CONFIDENCE = 0.55  # Minimum to trade
        self.KELLY_FRACTION = 0.25  # Quarter-Kelly (conservative)
        self.MIN_KELLY_BET = 1.0    # $1 minimum bet
        self.MAX_KELLY_BET = 10.0   # $10 max bet

    async def initialize(self) -> bool:
        """Fetch candle data and train model. Call on startup."""
        try:
            # Try loading saved model first
            if self._load_state():
                # Still fetch fresh candles for prediction features
                self.candles_raw = self._fetch_binance_candles(limit=1000)
                if len(self.candles_raw) >= 100:
                    print(f"[ML-{self.timeframe}] Loaded saved model, refreshed {len(self.candles_raw)} candles")
                    # Retrain if model is >1hr old
                    if time.time() - self.last_train_time > 3600:
                        print(f"[ML-{self.timeframe}] Model stale, retraining...")
                        self._train_model()
                        self._save_state()
                    return True

            # No saved model — fetch data and train fresh
            self.candles_raw = self._fetch_binance_candles_paginated(total=3000)
            if len(self.candles_raw) < 200:
                print(f"[ML-{self.timeframe}] Only {len(self.candles_raw)} candles, need 200+")
                return False

            success = self._train_model()
            if success:
                self._save_state()
            return success
        except Exception as e:
            print(f"[ML-{self.timeframe}] Init error: {e}")
            return False

    # ============================================================
    # DATA FETCHING
    # ============================================================

    def _fetch_binance_candles(self, limit: int = 1000) -> List[dict]:
        """Fetch BTC/USDT candles from Binance public API."""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': self.timeframe,
                'limit': min(limit, 1000),
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                print(f"[ML-{self.timeframe}] Binance API error: {r.status_code}")
                return []

            raw = r.json()
            candles = []
            for c in raw:
                candles.append({
                    'time': c[0],
                    'open': float(c[1]),
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4]),
                    'volume': float(c[5]),
                    'trades': int(c[8]),
                })

            return candles

        except Exception as e:
            print(f"[ML-{self.timeframe}] Fetch error: {e}")
            return []

    def _fetch_binance_candles_paginated(self, total: int = 3000) -> List[dict]:
        """Fetch multiple pages of candles for larger training sets."""
        all_candles = []
        end_time = None
        pages = 0

        while len(all_candles) < total and pages < 5:
            try:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': 'BTCUSDT',
                    'interval': self.timeframe,
                    'limit': 1000,
                }
                if end_time:
                    params['endTime'] = end_time

                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    break

                raw = r.json()
                if not raw:
                    break

                batch = []
                for c in raw:
                    batch.append({
                        'time': c[0],
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5]),
                        'trades': int(c[8]),
                    })

                # Prepend (older candles go first)
                all_candles = batch + all_candles
                end_time = batch[0]['time'] - 1  # Go further back
                pages += 1
                time.sleep(0.2)  # Rate limit

            except Exception as e:
                print(f"[ML-{self.timeframe}] Pagination error: {e}")
                break

        # Deduplicate by time, sort chronologically
        seen = set()
        unique = []
        for c in all_candles:
            if c['time'] not in seen:
                seen.add(c['time'])
                unique.append(c)
        unique.sort(key=lambda x: x['time'])

        print(f"[ML-{self.timeframe}] Fetched {len(unique)} candles ({pages} pages) from Binance")
        return unique

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    def compute_features(self, candles: List[dict], idx: int) -> Optional[np.ndarray]:
        """
        Compute feature vector for candle at position idx.
        Requires at least 30 candles of history before idx.
        """
        if idx < 30 or idx >= len(candles):
            return None

        # Extract price/volume arrays up to (and including) idx
        closes = [c['close'] for c in candles[max(0, idx - 50):idx + 1]]
        highs = [c['high'] for c in candles[max(0, idx - 50):idx + 1]]
        lows = [c['low'] for c in candles[max(0, idx - 50):idx + 1]]
        opens = [c['open'] for c in candles[max(0, idx - 50):idx + 1]]
        volumes = [c['volume'] for c in candles[max(0, idx - 50):idx + 1]]
        times = [c['time'] for c in candles[max(0, idx - 50):idx + 1]]

        c = closes[-1]
        o = opens[-1]
        h = highs[-1]
        lo = lows[-1]

        features = []

        # --- RSI(14) and RSI(7) ---
        rsi_14 = self._compute_rsi(closes, 14)
        rsi_7 = self._compute_rsi(closes, 7)
        # RSI slope (change over last 3 bars)
        rsi_prev = self._compute_rsi(closes[:-3], 14) if len(closes) > 17 else rsi_14
        rsi_slope = (rsi_14 - rsi_prev) / 3.0
        features.extend([rsi_14, rsi_7, rsi_slope])

        # --- MACD(12,26,9) ---
        macd_line, signal_line, histogram = self._compute_macd(closes)
        hist_prev = self._compute_macd(closes[:-1])[2] if len(closes) > 27 else histogram
        macd_hist_delta = histogram - hist_prev
        features.extend([histogram, macd_hist_delta, signal_line])

        # --- Bollinger Bands(20,2) ---
        bb_upper, bb_middle, bb_lower = self._compute_bb(closes, 20, 2)
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        bb_position = (c - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        features.extend([bb_position, bb_width])

        # --- Momentum (1, 3, 5, 10 bar returns) ---
        mom_1 = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
        mom_3 = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        mom_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
        mom_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0
        features.extend([mom_1, mom_3, mom_5, mom_10])

        # --- Volume ---
        vol_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / vol_avg_20 if vol_avg_20 > 0 else 1.0
        vol_trend = (np.mean(volumes[-5:]) - np.mean(volumes[-20:])) / np.mean(volumes[-20:]) \
            if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 0
        features.extend([vol_ratio, vol_trend])

        # --- Volatility ---
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(max(1, len(closes) - 20), len(closes))]
        vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
        vol_20 = np.std(returns) if len(returns) >= 10 else vol_5
        vol_rat = vol_5 / vol_20 if vol_20 > 0 else 1.0
        features.extend([vol_5, vol_20, vol_rat])

        # --- ATR ---
        atr_14 = self._compute_atr(highs, lows, closes, 14)
        atr_7 = self._compute_atr(highs, lows, closes, 7)
        atr_ratio = atr_7 / atr_14 if atr_14 > 0 else 1.0
        features.extend([atr_14 / c if c > 0 else 0, atr_ratio])

        # --- Candle shape ---
        body = abs(c - o)
        full_range = h - lo if h > lo else 0.0001
        body_ratio = body / full_range
        upper_wick = (h - max(c, o)) / full_range
        lower_wick = (min(c, o) - lo) / full_range
        features.extend([body_ratio, upper_wick, lower_wick])

        # --- Consecutive direction ---
        consec = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                if consec >= 0:
                    consec += 1
                else:
                    break
            elif closes[i] < closes[i - 1]:
                if consec <= 0:
                    consec -= 1
                else:
                    break
            else:
                break
        close_vs_open = 1.0 if c > o else (-1.0 if c < o else 0.0)
        features.extend([consec, close_vs_open])

        # --- Time of day (cyclical) ---
        ts = times[-1] / 1000 if times[-1] > 1e12 else times[-1]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour_frac = dt.hour + dt.minute / 60
        features.extend([
            np.sin(2 * np.pi * hour_frac / 24),
            np.cos(2 * np.pi * hour_frac / 24),
        ])

        # --- Returns distribution ---
        ret_10 = returns[-10:] if len(returns) >= 10 else returns
        ret_std = np.std(ret_10) if len(ret_10) >= 3 else 0
        ret_skew = float(np.mean([(r - np.mean(ret_10)) ** 3 for r in ret_10]) /
                         (ret_std ** 3 + 1e-10)) if ret_std > 0 else 0
        features.extend([ret_std, ret_skew])

        # --- Price range features ---
        high_low_20 = max(highs[-20:]) - min(lows[-20:]) if len(highs) >= 20 else h - lo
        range_norm = high_low_20 / c if c > 0 else 0
        position_in_range = (c - min(lows[-20:])) / high_low_20 \
            if high_low_20 > 0 and len(lows) >= 20 else 0.5
        features.extend([range_norm, position_in_range])

        return np.array(features, dtype=np.float64)

    def compute_features_live(self, candle_objects) -> Optional[np.ndarray]:
        """
        Compute features from live Candle objects (from ws_feeder or ta_signals).
        Adapts between dict and Candle object formats.
        """
        candles = []
        for c in candle_objects:
            if isinstance(c, dict):
                candles.append(c)
            elif hasattr(c, 'close'):
                candles.append({
                    'time': getattr(c, 'time', 0),
                    'open': getattr(c, 'open', c.close),
                    'high': getattr(c, 'high', c.close),
                    'low': getattr(c, 'low', c.close),
                    'close': c.close,
                    'volume': getattr(c, 'volume', 0),
                    'trades': getattr(c, 'trades', 0),
                })
            else:
                continue

        if len(candles) < 31:
            return None

        return self.compute_features(candles, len(candles) - 1)

    # ============================================================
    # TA INDICATOR HELPERS
    # ============================================================

    @staticmethod
    def _compute_rsi(closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(closes: List[float],
                      fast: int = 12, slow: int = 26, signal: int = 9
                      ) -> Tuple[float, float, float]:
        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0

        def ema(data, period):
            if len(data) < period:
                return data[-1] if data else 0
            k = 2 / (period + 1)
            result = data[0]
            for val in data[1:]:
                result = val * k + result * (1 - k)
            return result

        # MACD line = EMA(fast) - EMA(slow)
        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow

        # Signal line = EMA(MACD, signal)
        # Compute full MACD history for signal line
        macd_hist_values = []
        for i in range(slow, len(closes)):
            ef = ema(closes[:i + 1], fast)
            es = ema(closes[:i + 1], slow)
            macd_hist_values.append(ef - es)

        sig = ema(macd_hist_values, signal) if len(macd_hist_values) >= signal else macd_line
        histogram = macd_line - sig

        return macd_line, sig, histogram

    @staticmethod
    def _compute_bb(closes: List[float], period: int = 20, std_mult: float = 2.0
                    ) -> Tuple[float, float, float]:
        if len(closes) < period:
            c = closes[-1] if closes else 0
            return c, c, c
        window = closes[-period:]
        middle = np.mean(window)
        std = np.std(window)
        return middle + std_mult * std, middle, middle - std_mult * std

    @staticmethod
    def _compute_atr(highs, lows, closes, period=14) -> float:
        if len(highs) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            h_l = highs[i] - lows[i]
            h_pc = abs(highs[i] - closes[i - 1])
            l_pc = abs(lows[i] - closes[i - 1])
            trs.append(max(h_l, h_pc, l_pc))
        return np.mean(trs)

    # ============================================================
    # MODEL TRAINING
    # ============================================================

    def _train_model(self) -> bool:
        """
        Train LightGBM on Binance candle data with multi-strategy approach:
        1. Primary model: predict next-candle direction (threshold-filtered)
        2. Momentum-conditioned: only label candles where momentum exists
        3. Ensemble: combine primary + momentum model
        """
        candles = self.candles_raw
        if len(candles) < 200:
            return False

        # ================================================================
        # BUILD TRAINING DATA — MOMENTUM CONTINUATION labeling
        # The question: "Given recent momentum direction, will price
        # continue in that direction over the NEXT candle?"
        # This is the actual question our 5m trading system needs answered.
        #
        # Also train on ALL candles for regime detection:
        # Was this a "momentum" candle or "noise" candle?
        # ================================================================
        X_mom = []     # Features where momentum existed
        y_mom = []     # Label: did momentum continue?
        w_mom = []     # Weights
        X_all = []     # All candle features (for regime model)
        y_all = []     # Label: raw direction

        MOM_THRESHOLD = 0.0003   # 0.03% min momentum to qualify
        MOVE_THRESHOLD = 0.0002  # 0.02% min next-bar move to count

        for i in range(30, len(candles) - 2):
            features = self.compute_features(candles, i)
            if features is None:
                continue

            curr = candles[i]['close']
            prev3 = candles[i - 3]['close'] if i >= 3 else curr
            next_close = candles[i + 1]['close']

            # Current momentum (3-bar)
            mom_3bar = (curr - prev3) / prev3 if prev3 > 0 else 0
            # Next candle move
            next_move = (next_close - curr) / curr

            # --- ALL-CANDLE dataset (raw direction, threshold-filtered) ---
            if abs(next_move) >= MOVE_THRESHOLD:
                X_all.append(features)
                y_all.append(1 if next_move > 0 else 0)

            # --- MOMENTUM-CONDITIONED dataset ---
            # Only include candles where momentum EXISTS
            if abs(mom_3bar) >= MOM_THRESHOLD:
                # Label: did momentum CONTINUE?
                # If mom was UP (+), did next candle also go UP?
                # If mom was DOWN (-), did next candle also go DOWN?
                mom_up = mom_3bar > 0
                next_up = next_move > 0

                if mom_up:
                    label = 1 if next_up else 0  # 1 = continuation, 0 = reversal
                else:
                    label = 1 if not next_up else 0  # 1 = continuation (down continues)

                # Weight by momentum strength (stronger mom = clearer signal)
                weight = min(3.0, abs(mom_3bar) / MOM_THRESHOLD)

                X_mom.append(features)
                y_mom.append(label)
                w_mom.append(weight)

        # Use momentum dataset if we have enough, else fall back to all
        if len(X_mom) >= 100:
            X_list = X_mom
            y_list = y_mom
            weights = w_mom
            self._model_type = "momentum_continuation"
            print(f"[ML-{self.timeframe}] Training MOMENTUM CONTINUATION model "
                  f"({len(X_mom)} momentum samples, {len(X_all)} total)")
        elif len(X_all) >= 100:
            X_list = X_all
            y_list = y_all
            weights = [1.0] * len(X_all)
            self._model_type = "direction"
            print(f"[ML-{self.timeframe}] Training DIRECTION model "
                  f"(not enough momentum samples: {len(X_mom)})")
        else:
            print(f"[ML-{self.timeframe}] Not enough data: {len(X_mom)} mom, {len(X_all)} all")
            return False

        X = np.array(X_list)
        y = np.array(y_list)
        w = np.array(weights)

        # ================================================================
        # ADD LIVE OUTCOMES (3x weight — real Polymarket data is gold)
        # ================================================================
        if self.live_outcomes:
            live_X = []
            live_y = []
            live_w = []
            for out in self.live_outcomes:
                f = out.get('features')
                if f is not None:
                    live_X.append(f)
                    live_y.append(out['label'])
                    live_w.append(5.0)  # Live outcomes worth 5x
            if live_X:
                X = np.vstack([X, np.array(live_X)])
                y = np.concatenate([y, np.array(live_y)])
                w = np.concatenate([w, np.array(live_w)])
                print(f"[ML-{self.timeframe}] Added {len(live_X)} live outcomes (5x weight)")

        # ================================================================
        # SCALE + SPLIT (time-based split, not random — prevents future leak)
        # ================================================================
        X_scaled = self.scaler.fit_transform(X)

        n_up = int(np.sum(y == 1))
        n_down = int(np.sum(y == 0))
        scale_pos = n_down / max(n_up, 1)

        split = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]
        w_train, w_val = w[:split], w[split:]

        # ================================================================
        # TRAIN LIGHTGBM
        # ================================================================
        lgb_train = lgb.Dataset(X_train, y_train, weight=w_train,
                                feature_name=FEATURE_NAMES_5M)
        lgb_val = lgb.Dataset(X_val, y_val, weight=w_val, reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'min_child_samples': 5,
            'scale_pos_weight': scale_pos,
            'lambda_l1': 0.2,
            'lambda_l2': 1.0,
            'max_depth': 4,         # Shallow to prevent overfit
            'min_gain_to_split': 0.01,
            'verbose': -1,
        }

        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # ================================================================
        # EVALUATE
        # ================================================================
        val_pred = self.model.predict(X_val)
        val_acc = np.mean((val_pred > 0.5) == y_val)

        # Profitable accuracy: only confident predictions
        for thresh in [0.05, 0.08, 0.10, 0.15]:
            mask = np.abs(val_pred - 0.5) > thresh
            n = int(np.sum(mask))
            if n >= 5:
                acc = np.mean((val_pred[mask] > 0.5) == y_val[mask])
                print(f"[ML-{self.timeframe}]   conf>{thresh:.0%}: {acc:.1%} ({n}T)")

        # Feature importance
        importance = dict(zip(FEATURE_NAMES_5M, self.model.feature_importance()))
        top_5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ', '.join(f"{k}={v}" for k, v in top_5)

        model_label = getattr(self, '_model_type', 'direction')
        print(f"[ML-{self.timeframe}] Model: {model_label} | {len(X_train)}T train | Val: {val_acc:.1%} ({len(X_val)}T)")
        print(f"[ML-{self.timeframe}] Labels: {n_up} continue / {n_down} reverse | {len(candles)} candles")
        print(f"[ML-{self.timeframe}] Top features: {top_str}")

        self.is_trained = True
        self.last_train_time = time.time()

        # ================================================================
        # TRAIN VOLATILITY MODEL
        # Predicts: will the NEXT candle have a BIG move (>0.1%)?
        # Big moves = good for Polymarket (prices diverge from 0.50)
        # Volatility IS predictable (vol clustering/GARCH effects)
        # ================================================================
        self._train_volatility_model(candles)

        # ================================================================
        # TRAIN MOMENTUM QUALITY MODEL
        # Predicts: is current momentum "clean" (sustainable) or "spiky" (noise)?
        # Clean momentum = momentum will sustain through next 5m window
        # ================================================================
        self._train_quality_model(candles)

        return True

    def _train_volatility_model(self, candles: List[dict]):
        """Train model to predict if next candle will be a big move."""
        X_list = []
        y_list = []
        BIG_MOVE = 0.001  # 0.1% = "big move" for 5m candle

        for i in range(30, len(candles) - 1):
            features = self.compute_features(candles, i)
            if features is None:
                continue
            next_move = abs(candles[i + 1]['close'] - candles[i]['close']) / candles[i]['close']
            label = 1 if next_move >= BIG_MOVE else 0
            X_list.append(features)
            y_list.append(label)

        if len(X_list) < 100:
            return

        X = np.array(X_list)
        y = np.array(y_list)
        X_scaled = self.vol_scaler.fit_transform(X)

        n_big = int(np.sum(y == 1))
        n_small = int(np.sum(y == 0))
        scale_pos = n_small / max(n_big, 1)

        split = int(len(X) * 0.8)
        lgb_train = lgb.Dataset(X_scaled[:split], y[:split],
                                feature_name=FEATURE_NAMES_5M)
        lgb_val = lgb.Dataset(X_scaled[split:], y[split:], reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': 'auc',  # AUC better for imbalanced vol prediction
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'scale_pos_weight': scale_pos,
            'lambda_l1': 0.3,
            'lambda_l2': 1.0,
            'max_depth': 3,
            'verbose': -1,
        }

        self.vol_model = lgb.train(
            params, lgb_train, num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        val_pred = self.vol_model.predict(X_scaled[split:])
        val_y = y[split:]
        auc_approx = np.mean((val_pred > 0.5) == val_y)
        print(f"[ML-{self.timeframe}] Vol model: {n_big} big / {n_small} small | "
              f"Val acc: {auc_approx:.1%}")

    def _train_quality_model(self, candles: List[dict]):
        """
        Train model to score momentum quality.
        Label: 1 if momentum continued for 2+ consecutive candles, 0 if reversed immediately.
        """
        X_list = []
        y_list = []

        for i in range(30, len(candles) - 3):
            features = self.compute_features(candles, i)
            if features is None:
                continue

            curr = candles[i]['close']
            prev3 = candles[i - 3]['close']
            if prev3 == 0:
                continue
            mom = (curr - prev3) / prev3

            if abs(mom) < 0.0003:
                continue  # Only train on momentum candles

            # Did momentum sustain for next 2 bars?
            n1 = candles[i + 1]['close']
            n2 = candles[i + 2]['close']

            if mom > 0:
                sustained = n1 > curr and n2 > curr  # Both bars above current
            else:
                sustained = n1 < curr and n2 < curr

            X_list.append(features)
            y_list.append(1 if sustained else 0)

        if len(X_list) < 80:
            return

        X = np.array(X_list)
        y = np.array(y_list)
        X_scaled = self.scaler.transform(X)  # Reuse main scaler

        split = int(len(X) * 0.8)
        n_sus = int(np.sum(y == 1))
        n_rev = int(np.sum(y == 0))

        lgb_train = lgb.Dataset(X_scaled[:split], y[:split],
                                feature_name=FEATURE_NAMES_5M)
        lgb_val = lgb.Dataset(X_scaled[split:], y[split:], reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 8,
            'lambda_l1': 0.2,
            'lambda_l2': 0.8,
            'max_depth': 3,
            'verbose': -1,
        }

        self.quality_model = lgb.train(
            params, lgb_train, num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        val_pred = self.quality_model.predict(X_scaled[split:])
        val_acc = np.mean((val_pred > 0.5) == y[split:])
        print(f"[ML-{self.timeframe}] Quality model: {n_sus} sustained / {n_rev} reversed | "
              f"Val acc: {val_acc:.1%}")

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(self, candle_data) -> MLPrediction:
        """
        Predict BTC direction for the next candle.

        Args:
            candle_data: List of candle dicts or Candle objects

        Returns:
            MLPrediction with direction, confidence, kelly sizing
        """
        if not self.is_trained or self.model is None:
            return MLPrediction("SKIP", 0.0, 0.5, 0.0, 0.0)

        # Compute features
        if candle_data and isinstance(candle_data[0], dict):
            if len(candle_data) < 31:
                return MLPrediction("SKIP", 0.0, 0.5, 0.0, 0.0)
            features = self.compute_features(candle_data, len(candle_data) - 1)
        else:
            features = self.compute_features_live(candle_data)

        if features is None:
            return MLPrediction("SKIP", 0.0, 0.5, 0.0, 0.0)

        # Scale and predict
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        raw_prob = float(self.model.predict(X_scaled)[0])

        # Interpretation depends on model type
        model_type = getattr(self, '_model_type', 'direction')

        if model_type == 'momentum_continuation':
            # raw_prob = P(momentum continues)
            # We need to know current momentum direction to translate
            # Use momentum features from the feature vector
            # momentum_1bar is at index 8, momentum_3bar at index 9
            mom_1 = features[8] if len(features) > 8 else 0
            mom_3 = features[9] if len(features) > 9 else 0

            continuation_prob = raw_prob

            if mom_3 > 0:
                # Momentum is UP → continuation means UP
                up_prob = continuation_prob
            elif mom_3 < 0:
                # Momentum is DOWN → continuation means DOWN
                up_prob = 1 - continuation_prob
            else:
                # No momentum → model doesn't apply
                up_prob = 0.5
        else:
            # Standard direction model
            up_prob = raw_prob

        # Direction and confidence
        confidence = abs(up_prob - 0.5) * 2  # 0-1 scale
        direction = "UP" if up_prob > 0.5 else "DOWN"

        if confidence < (self.MIN_CONFIDENCE - 0.5) * 2:
            direction = "SKIP"

        # --- Volatility prediction ---
        vol_score = 0.5
        vol_regime = "normal"
        if self.vol_model is not None:
            try:
                X_vol = self.vol_scaler.transform(X)
                vol_score = float(self.vol_model.predict(X_vol)[0])
                if vol_score > 0.65:
                    vol_regime = "high"
                elif vol_score < 0.35:
                    vol_regime = "low"
            except Exception:
                pass

        # --- Momentum quality prediction ---
        momentum_quality = 0.5
        if self.quality_model is not None:
            try:
                q_pred = float(self.quality_model.predict(X_scaled)[0])
                momentum_quality = q_pred
            except Exception:
                pass

        # --- Combined trade score ---
        # Direction confidence: 0-1 (how sure are we of direction)
        # Vol score: higher = bigger expected move = better for us
        # Quality: higher = momentum more likely to sustain
        # Trade score = weighted combination
        dir_weight = 0.3   # Direction matters but is weakest signal
        vol_weight = 0.35  # Volatility is predictable and profitable
        qual_weight = 0.35 # Momentum quality determines trade outcome

        trade_score = (
            confidence * dir_weight +
            vol_score * vol_weight +
            momentum_quality * qual_weight
        )

        # --- Kelly sizing based on trade_score ---
        # Map trade_score to position size
        # Score < 0.40 → skip
        # Score 0.40-0.55 → small ($2-3)
        # Score 0.55-0.70 → medium ($5-7)
        # Score > 0.70 → full ($8-10)
        if trade_score < 0.40 or direction == "SKIP":
            kelly_size = 0.0
            kelly_fraction = 0.0
            if direction != "SKIP":
                direction = "SKIP"
        else:
            # Scale linearly from min to max based on trade_score
            score_pct = (trade_score - 0.40) / 0.60  # 0 at 0.40, 1 at 1.0
            score_pct = min(1.0, max(0.0, score_pct))
            kelly_size = self.MIN_KELLY_BET + score_pct * (self.MAX_KELLY_BET - self.MIN_KELLY_BET)
            kelly_fraction = kelly_size / self.bankroll

        self.predictions_made += 1

        return MLPrediction(
            direction=direction,
            confidence=confidence,
            up_probability=up_prob,
            kelly_fraction=round(kelly_fraction, 4),
            kelly_size_usd=round(kelly_size, 2),
            vol_regime=vol_regime,
            vol_score=round(vol_score, 3),
            momentum_quality=round(momentum_quality, 3),
            trade_score=round(trade_score, 3),
            features=features,
        )

    def predict_live(self) -> MLPrediction:
        """
        Fetch fresh Binance candles and predict current direction.
        Convenience method for integration — call this from trading loop.
        """
        if not self.is_trained:
            return MLPrediction("SKIP", 0.0, 0.5, 0.0, 0.0)

        # Fetch latest candles (small request, fast)
        fresh = self._fetch_binance_candles(limit=100)
        if len(fresh) < 31:
            return MLPrediction("SKIP", 0.0, 0.5, 0.0, 0.0)

        return self.predict(fresh)

    # ============================================================
    # ONLINE LEARNING
    # ============================================================

    def add_outcome(self, features: np.ndarray, direction_taken: str,
                    won: bool):
        """
        Record a live trade outcome for online learning.
        Features should be the array from predict() call.
        """
        # Label: 1 if UP won (BTC went up), 0 if DOWN won
        if direction_taken == "UP":
            label = 1 if won else 0
        else:
            label = 0 if won else 1

        self.live_outcomes.append({
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'label': label,
            'direction': direction_taken,
            'won': won,
            'time': datetime.now(timezone.utc).isoformat(),
        })

        if won:
            self.correct_predictions += 1

        self._save_state()

    async def maybe_retrain(self) -> bool:
        """Retrain if enough time has passed. Call periodically."""
        if time.time() - self.last_train_time < self.train_interval:
            return False

        # Fetch fresh candles
        fresh = self._fetch_binance_candles(limit=1000)
        if len(fresh) >= 100:
            self.candles_raw = fresh
            success = self._train_model()
            if success:
                self._save_state()
            return success
        return False

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def _save_state(self):
        """Save all models and state to disk."""
        try:
            if self.model is not None:
                self.model.save_model(self.model_path)
            vol_path = self.model_path.replace('_model.', '_vol_model.')
            if self.vol_model is not None:
                self.vol_model.save_model(vol_path)
            qual_path = self.model_path.replace('_model.', '_qual_model.')
            if self.quality_model is not None:
                self.quality_model.save_model(qual_path)

            def scaler_to_dict(sc):
                if hasattr(sc, 'mean_') and sc.mean_ is not None:
                    return {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist()}
                return None

            state = {
                'live_outcomes': self.live_outcomes[-500:],
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions,
                'last_train_time': self.last_train_time,
                'model_type': getattr(self, '_model_type', 'direction'),
                'scaler': scaler_to_dict(self.scaler),
                'vol_scaler': scaler_to_dict(self.vol_scaler),
            }
            with open(self.state_path, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"[ML-{self.timeframe}] Save error: {e}")

    def _load_state(self) -> bool:
        """Load saved models and state."""
        try:
            loaded_any = False

            if os.path.exists(self.model_path):
                self.model = lgb.Booster(model_file=self.model_path)
                loaded_any = True

            vol_path = self.model_path.replace('_model.', '_vol_model.')
            if os.path.exists(vol_path):
                self.vol_model = lgb.Booster(model_file=vol_path)

            qual_path = self.model_path.replace('_model.', '_qual_model.')
            if os.path.exists(qual_path):
                self.quality_model = lgb.Booster(model_file=qual_path)

            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                self.live_outcomes = state.get('live_outcomes', [])
                self.predictions_made = state.get('predictions_made', 0)
                self.correct_predictions = state.get('correct_predictions', 0)
                self.last_train_time = state.get('last_train_time', 0)
                self._model_type = state.get('model_type', 'direction')

                def restore_scaler(sc, data):
                    if data:
                        sc.mean_ = np.array(data['mean'])
                        sc.scale_ = np.array(data['scale'])
                        sc.n_features_in_ = len(data['mean'])
                        sc.var_ = np.array(data['scale']) ** 2

                restore_scaler(self.scaler, state.get('scaler'))
                restore_scaler(self.vol_scaler, state.get('vol_scaler'))

            if loaded_any:
                self.is_trained = True
                print(f"[ML-{self.timeframe}] Loaded saved models ({self.predictions_made} predictions, "
                      f"{self.correct_predictions} correct)")
                return True
        except Exception as e:
            print(f"[ML-{self.timeframe}] Load error: {e}")

        return False


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == '__main__':
    import asyncio

    async def test():
        for tf in ['5m', '15m']:
            print(f"\n{'=' * 60}")
            print(f"Testing {tf} predictor")
            print('=' * 60)
            p = BTCPredictor(timeframe=tf)
            ok = await p.initialize()
            if ok:
                pred = p.predict(p.candles_raw)
                print(f"Prediction: {pred.direction} | conf={pred.confidence:.2%} | "
                      f"P(UP)={pred.up_probability:.3f} | kelly=${pred.kelly_size_usd}")
            else:
                print("Failed to initialize")

    asyncio.run(test())
