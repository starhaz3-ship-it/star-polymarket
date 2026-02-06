"""
Advanced ML Engine V3 - Implements research findings from 2025-2026

Key algorithms implemented:
1. LightGBM for direction prediction (best for BTC/ETH/crypto)
2. XGBoost ensemble for feature boosting
3. Online learning with incremental updates
4. Strategy selector (DQN-inspired) for UP/DOWN choice

Based on research:
- LightGBM ranked #1 for ETH, BTC, LTC prediction
- Combined XGBoost + LightGBM outperforms neural networks
- Online learning adapts to changing market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from datetime import datetime, timezone

try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    print("[ML V3] Warning: ML libraries not available")


@dataclass
class MLFeatures:
    """Feature vector for ML prediction."""
    # Price features
    price_change_1m: float = 0.0
    price_change_5m: float = 0.0
    price_change_15m: float = 0.0
    price_vs_vwap: float = 0.0

    # Momentum features
    rsi: float = 50.0
    rsi_slope: float = 0.0
    momentum_5m: float = 0.0
    momentum_15m: float = 0.0

    # Volatility features
    volatility_5m: float = 0.0
    volatility_15m: float = 0.0
    atr_ratio: float = 1.0

    # Trend features
    ema_cross: float = 0.0  # Price vs EMA20
    trend_strength: float = 0.0
    heiken_streak: int = 0

    # Market features
    up_price: float = 0.5
    down_price: float = 0.5
    market_edge: float = 0.0
    kl_divergence: float = 0.0

    # Time features
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    minute_in_window: float = 0.0

    # Order flow features (from polymarket-assistant research)
    obi: float = 0.0              # Order book imbalance: -1 to +1
    cvd_5m: float = 0.0           # Cumulative volume delta (5 min)
    squeeze_on: float = 0.0       # TTM Squeeze: 1 if BB inside KC, else 0
    squeeze_momentum: float = 0.0  # Squeeze momentum direction
    ema_cross_signal: float = 0.0  # -1 DEATH, 0 NEUTRAL, +1 GOLDEN

    # V3.2 features (from research + NYU insight)
    macd_histogram: float = 0.0   # MACD histogram value (momentum)
    macd_signal_line: float = 0.0 # MACD signal line
    price_dist_from_50: float = 0.0  # NYU key: distance from 0.50 = edge quality
    rsi_macd_confluence: float = 0.0  # RSI+MACD agreement (73-77% WR per research)
    volume_ratio: float = 1.0     # Current vs average volume

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array([
            self.price_change_1m, self.price_change_5m, self.price_change_15m,
            self.price_vs_vwap, self.rsi, self.rsi_slope,
            self.momentum_5m, self.momentum_15m,
            self.volatility_5m, self.volatility_15m, self.atr_ratio,
            self.ema_cross, self.trend_strength, self.heiken_streak,
            self.up_price, self.down_price, self.market_edge, self.kl_divergence,
            self.hour_sin, self.hour_cos, self.minute_in_window,
            # Order flow features (5)
            self.obi, self.cvd_5m, self.squeeze_on, self.squeeze_momentum, self.ema_cross_signal,
            # V3.2 features (5 new)
            self.macd_histogram, self.macd_signal_line, self.price_dist_from_50,
            self.rsi_macd_confluence, self.volume_ratio
        ])

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'price_change_1m', 'price_change_5m', 'price_change_15m',
            'price_vs_vwap', 'rsi', 'rsi_slope',
            'momentum_5m', 'momentum_15m',
            'volatility_5m', 'volatility_15m', 'atr_ratio',
            'ema_cross', 'trend_strength', 'heiken_streak',
            'up_price', 'down_price', 'market_edge', 'kl_divergence',
            'hour_sin', 'hour_cos', 'minute_in_window',
            # Order flow features (5)
            'obi', 'cvd_5m', 'squeeze_on', 'squeeze_momentum', 'ema_cross_signal',
            # V3.2 features (5 new)
            'macd_histogram', 'macd_signal_line', 'price_dist_from_50',
            'rsi_macd_confluence', 'volume_ratio'
        ]


@dataclass
class MLPrediction:
    """ML model prediction output."""
    up_probability: float
    down_probability: float
    confidence: float
    recommended_side: str  # "UP", "DOWN", or "SKIP"
    model_agreement: float  # How much LightGBM and XGBoost agree
    feature_importance: Dict[str, float] = field(default_factory=dict)


class AdvancedMLEngine:
    """
    Advanced ML Engine using LightGBM + XGBoost ensemble.

    Features:
    - LightGBM for fast, accurate predictions (best for crypto per research)
    - XGBoost for ensemble agreement checking
    - Online learning: updates model weights after each trade result
    - Adaptive thresholds based on recent performance
    """

    MODEL_DIR = Path(__file__).parent.parent / "ml_models"
    TRAINING_DATA_FILE = Path(__file__).parent.parent / "ml_training_data.json"
    MIN_SAMPLES_TO_TRAIN = 30
    RETRAIN_EVERY_N_SAMPLES = 20

    # Prediction thresholds
    MIN_CONFIDENCE = 0.60
    MIN_AGREEMENT = 0.70

    def __init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True)

        self.lgb_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Training data accumulator
        self.training_samples: List[Dict] = []
        self.samples_since_last_train = 0

        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0

        # Online learning state
        self.recent_results: List[bool] = []  # Last N trade outcomes
        self.adaptive_threshold = 0.60

        self._load_training_data()
        self._load_models()

    def _load_training_data(self):
        """Load accumulated training data."""
        if self.TRAINING_DATA_FILE.exists():
            try:
                with open(self.TRAINING_DATA_FILE) as f:
                    data = json.load(f)
                    self.training_samples = data.get("samples", [])
                    self.recent_results = data.get("recent_results", [])[-50:]
                    print(f"[ML V3] Loaded {len(self.training_samples)} training samples")
            except Exception as e:
                print(f"[ML V3] Error loading training data: {e}")

    def _save_training_data(self):
        """Save training data for persistence."""
        try:
            with open(self.TRAINING_DATA_FILE, 'w') as f:
                json.dump({
                    "samples": self.training_samples[-1000:],  # Keep last 1000
                    "recent_results": self.recent_results[-50:],
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }, f)
        except Exception as e:
            print(f"[ML V3] Error saving training data: {e}")

    def _load_models(self):
        """Load pre-trained models if they exist."""
        lgb_path = self.MODEL_DIR / "lgb_model.pkl"
        xgb_path = self.MODEL_DIR / "xgb_model.pkl"
        scaler_path = self.MODEL_DIR / "scaler.pkl"

        if lgb_path.exists() and xgb_path.exists() and scaler_path.exists():
            try:
                with open(lgb_path, 'rb') as f:
                    self.lgb_model = pickle.load(f)
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("[ML V3] Loaded pre-trained models")
            except Exception as e:
                print(f"[ML V3] Error loading models: {e}")

    def _save_models(self):
        """Save trained models."""
        try:
            with open(self.MODEL_DIR / "lgb_model.pkl", 'wb') as f:
                pickle.dump(self.lgb_model, f)
            with open(self.MODEL_DIR / "xgb_model.pkl", 'wb') as f:
                pickle.dump(self.xgb_model, f)
            with open(self.MODEL_DIR / "scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            print("[ML V3] Saved trained models")
        except Exception as e:
            print(f"[ML V3] Error saving models: {e}")

    def extract_features(self, candles: List, current_price: float,
                        up_price: float, down_price: float,
                        signal_data: Dict = None,
                        order_flow: Dict = None) -> MLFeatures:
        """
        Extract features from candle data and market state.

        Args:
            candles: List of Candle objects (most recent last)
            current_price: Current BTC/asset price
            up_price: Market UP price
            down_price: Market DOWN price
            signal_data: Optional dict with RSI, VWAP, etc.
            order_flow: Optional dict with OBI, CVD, squeeze data:
                        {'obi': float, 'cvd_5m': float, 'squeeze_on': bool,
                         'squeeze_momentum': float, 'ema_cross': str}
        """
        features = MLFeatures()

        if not candles or len(candles) < 15:
            return features

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Price changes
        if len(closes) >= 2:
            features.price_change_1m = (closes[-1] - closes[-2]) / closes[-2]
        if len(closes) >= 6:
            features.price_change_5m = (closes[-1] - closes[-6]) / closes[-6]
        if len(closes) >= 16:
            features.price_change_15m = (closes[-1] - closes[-16]) / closes[-16]

        # Momentum
        if len(closes) >= 6:
            features.momentum_5m = sum(1 if closes[i] > closes[i-1] else -1
                                       for i in range(-5, 0)) / 5
        if len(closes) >= 16:
            features.momentum_15m = sum(1 if closes[i] > closes[i-1] else -1
                                        for i in range(-15, 0)) / 15

        # Volatility (standard deviation of returns)
        if len(closes) >= 6:
            returns_5m = [(closes[i] - closes[i-1]) / closes[i-1]
                         for i in range(-5, 0)]
            features.volatility_5m = np.std(returns_5m) if returns_5m else 0

        if len(closes) >= 16:
            returns_15m = [(closes[i] - closes[i-1]) / closes[i-1]
                          for i in range(-15, 0)]
            features.volatility_15m = np.std(returns_15m) if returns_15m else 0

        # EMA cross (price vs EMA20)
        if len(closes) >= 20:
            ema20 = self._calculate_ema(closes[-20:], 20)
            features.ema_cross = (current_price - ema20) / ema20

        # ATR ratio (short vs long)
        if len(highs) >= 30 and len(lows) >= 30:
            atr_short = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:])
            atr_long = self._calculate_atr(highs[-30:], lows[-30:], closes[-30:])
            if atr_long > 0:
                features.atr_ratio = atr_short / atr_long

        # Market features
        features.up_price = up_price
        features.down_price = down_price
        features.market_edge = max(
            (1.0 - up_price) - 0.5 if up_price < 0.5 else 0,
            (1.0 - down_price) - 0.5 if down_price < 0.5 else 0
        )

        # Signal data (RSI, VWAP, etc.)
        if signal_data:
            features.rsi = signal_data.get('rsi', 50.0)
            features.price_vs_vwap = signal_data.get('price_vs_vwap', 0.0)
            features.kl_divergence = signal_data.get('kl_divergence', 0.0)
            features.heiken_streak = signal_data.get('heiken_streak', 0)
            features.trend_strength = signal_data.get('trend_strength', 0.0)

        # Time features (cyclical encoding)
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60
        features.hour_sin = np.sin(2 * np.pi * hour / 24)
        features.hour_cos = np.cos(2 * np.pi * hour / 24)
        features.minute_in_window = (now.minute % 15) / 15

        # Order flow features (OBI, CVD, TTM Squeeze, EMA cross)
        if order_flow:
            features.obi = order_flow.get('obi', 0.0)
            features.cvd_5m = order_flow.get('cvd_5m', 0.0)
            features.squeeze_on = 1.0 if order_flow.get('squeeze_on', False) else 0.0
            features.squeeze_momentum = order_flow.get('squeeze_momentum', 0.0)
            # EMA cross: GOLDEN=+1, DEATH=-1, NEUTRAL=0
            ema_cross_str = order_flow.get('ema_cross', 'NEUTRAL')
            if ema_cross_str == 'GOLDEN':
                features.ema_cross_signal = 1.0
            elif ema_cross_str == 'DEATH':
                features.ema_cross_signal = -1.0
            else:
                features.ema_cross_signal = 0.0

        # V3.2 features: MACD, price distance, RSI+MACD confluence, volume ratio
        if signal_data:
            features.macd_histogram = signal_data.get('macd_histogram', 0.0)
            features.macd_signal_line = signal_data.get('macd_signal_line', 0.0)
            features.volume_ratio = signal_data.get('volume_ratio', 1.0)

        # NYU key insight: distance from 0.50 determines edge quality
        # Extreme prices = lower volatility = better edge
        avg_price = (up_price + down_price) / 2
        features.price_dist_from_50 = abs(avg_price - 0.50)

        # RSI + MACD confluence (research: 73-77% WR when both agree)
        # +1 = both bullish, -1 = both bearish, 0 = conflicting
        rsi_val = features.rsi
        macd_val = features.macd_histogram
        if rsi_val > 55 and macd_val > 0:
            features.rsi_macd_confluence = 1.0  # Both bullish
        elif rsi_val < 45 and macd_val < 0:
            features.rsi_macd_confluence = -1.0  # Both bearish
        elif (rsi_val > 55 and macd_val < 0) or (rsi_val < 45 and macd_val > 0):
            features.rsi_macd_confluence = 0.0  # Conflicting - danger
        else:
            features.rsi_macd_confluence = 0.0  # Neutral

        return features

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate exponential moving average."""
        if not prices:
            return 0.0
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calculate_atr(self, highs: List[float], lows: List[float],
                       closes: List[float]) -> float:
        """Calculate Average True Range."""
        if len(highs) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        return np.mean(true_ranges) if true_ranges else 0.0

    def add_training_sample(self, features: MLFeatures, outcome: str,
                           pnl: float, side_taken: str):
        """
        Add a completed trade as training data.

        Args:
            features: Features at time of trade entry
            outcome: "WIN" or "LOSS"
            pnl: Profit/loss amount
            side_taken: "UP" or "DOWN"
        """
        # Label: 1 if UP won (price went up), 0 if DOWN won
        if side_taken == "UP":
            label = 1 if outcome == "WIN" else 0
        else:
            label = 0 if outcome == "WIN" else 1

        sample = {
            "features": features.to_array().tolist(),
            "label": label,
            "outcome": outcome,
            "pnl": pnl,
            "side": side_taken,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.training_samples.append(sample)
        self.samples_since_last_train += 1
        self.recent_results.append(outcome == "WIN")
        self.recent_results = self.recent_results[-50:]

        # Update adaptive threshold based on recent performance
        if len(self.recent_results) >= 10:
            recent_wr = sum(self.recent_results[-10:]) / 10
            if recent_wr < 0.45:
                self.adaptive_threshold = min(0.75, self.adaptive_threshold + 0.02)
            elif recent_wr > 0.60:
                self.adaptive_threshold = max(0.55, self.adaptive_threshold - 0.01)

        self._save_training_data()

        # Retrain if we have enough new samples
        if (self.samples_since_last_train >= self.RETRAIN_EVERY_N_SAMPLES and
            len(self.training_samples) >= self.MIN_SAMPLES_TO_TRAIN):
            self.train_models()

    def train_models(self):
        """Train LightGBM and XGBoost models on accumulated data."""
        if len(self.training_samples) < self.MIN_SAMPLES_TO_TRAIN:
            print(f"[ML V3] Need {self.MIN_SAMPLES_TO_TRAIN} samples, have {len(self.training_samples)}")
            return False

        print(f"[ML V3] Training on {len(self.training_samples)} samples...")

        # Prepare data - handle old samples with fewer features
        expected_features = len(MLFeatures.feature_names())
        X_list = []
        y_list = []
        for s in self.training_samples:
            feat = s["features"]
            # Pad old samples that have fewer features with zeros
            if len(feat) < expected_features:
                feat = feat + [0.0] * (expected_features - len(feat))
            elif len(feat) > expected_features:
                feat = feat[:expected_features]
            X_list.append(feat)
            y_list.append(s["label"])

        X = np.array(X_list)
        y = np.array(y_list)

        # Calculate class imbalance ratio for scale_pos_weight
        # DOWN historically wins 60%+, so label=0 (DOWN won) is more common
        n_pos = max(1, np.sum(y == 1))
        n_neg = max(1, np.sum(y == 0))
        scale_pos_weight = n_neg / n_pos
        print(f"[ML V3] Class balance: UP={n_pos}, DOWN={n_neg}, scale={scale_pos_weight:.2f}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split for validation (stratified to maintain class balance)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train LightGBM with improved hyperparameters
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 47,           # Increased from 31 for more expressiveness
            'learning_rate': 0.03,      # Slower learning, better generalization
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'min_child_samples': 5,     # Prevent overfitting on small datasets
            'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            'lambda_l1': 0.1,           # L1 regularization
            'lambda_l2': 0.1,           # L2 regularization
            'verbose': -1
        }

        self.lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=200,        # More rounds with lower LR
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )

        # Train XGBoost with improved hyperparameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,             # Slightly deeper
            'learning_rate': 0.03,      # Match LightGBM LR
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 3,      # Prevent overfitting
            'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            'reg_alpha': 0.1,           # L1 regularization
            'reg_lambda': 0.1,          # L2 regularization
            'verbosity': 0
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=200,        # More rounds
            evals=[(dval, 'val')],
            early_stopping_rounds=15,
            verbose_eval=False
        )

        # Evaluate
        lgb_pred = self.lgb_model.predict(X_val)
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_val))

        lgb_acc = np.mean((lgb_pred > 0.5) == y_val)
        xgb_acc = np.mean((xgb_pred > 0.5) == y_val)

        # Also check ensemble accuracy
        ensemble_pred = (lgb_pred + xgb_pred) / 2
        ensemble_acc = np.mean((ensemble_pred > 0.5) == y_val)

        print(f"[ML V3.2] LightGBM accuracy: {lgb_acc:.1%}")
        print(f"[ML V3.2] XGBoost accuracy: {xgb_acc:.1%}")
        print(f"[ML V3.2] Ensemble accuracy: {ensemble_acc:.1%}")

        self.is_trained = True
        self.samples_since_last_train = 0
        self._save_models()

        return True

    def predict(self, features: MLFeatures) -> MLPrediction:
        """
        Make prediction using ensemble of LightGBM + XGBoost.

        Returns MLPrediction with UP/DOWN probabilities and recommendation.
        """
        if not self.is_trained or self.lgb_model is None:
            # Fallback to heuristic if no trained model
            return self._heuristic_prediction(features)

        # Scale features
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get predictions from both models
        lgb_prob = self.lgb_model.predict(X_scaled)[0]  # P(UP wins)
        xgb_prob = self.xgb_model.predict(xgb.DMatrix(X_scaled))[0]

        # Ensemble: average probabilities
        up_prob = (lgb_prob + xgb_prob) / 2
        down_prob = 1 - up_prob

        # Model agreement (how close are the two predictions)
        agreement = 1 - abs(lgb_prob - xgb_prob)

        # Confidence based on distance from 0.5
        confidence = abs(up_prob - 0.5) * 2

        # Determine recommendation
        if confidence < self.adaptive_threshold - 0.5:
            recommended_side = "SKIP"
        elif agreement < self.MIN_AGREEMENT:
            recommended_side = "SKIP"
        elif up_prob > 0.5:
            recommended_side = "UP"
        else:
            recommended_side = "DOWN"

        # Feature importance from LightGBM
        importance = {}
        if hasattr(self.lgb_model, 'feature_importance'):
            for name, imp in zip(MLFeatures.feature_names(),
                                self.lgb_model.feature_importance()):
                importance[name] = float(imp)

        self.predictions_made += 1

        return MLPrediction(
            up_probability=up_prob,
            down_probability=down_prob,
            confidence=confidence,
            recommended_side=recommended_side,
            model_agreement=agreement,
            feature_importance=importance
        )

    def _heuristic_prediction(self, features: MLFeatures) -> MLPrediction:
        """Fallback heuristic when no trained model available."""
        # Simple rule-based prediction
        score = 0.0

        # Momentum
        score += features.momentum_5m * 0.3
        score += features.momentum_15m * 0.2

        # RSI
        if features.rsi > 70:
            score -= 0.2
        elif features.rsi < 30:
            score += 0.2

        # Price change
        score += features.price_change_5m * 10

        # EMA cross
        score += features.ema_cross * 5

        # Order flow features (new)
        # OBI: positive = more bids = bullish
        score += features.obi * 0.3

        # CVD: positive = net buying
        if features.cvd_5m > 0:
            score += 0.1
        elif features.cvd_5m < 0:
            score -= 0.1

        # TTM Squeeze: if squeeze is on and momentum rising, expect breakout
        if features.squeeze_on > 0.5 and features.squeeze_momentum > 0:
            score += 0.2
        elif features.squeeze_on > 0.5 and features.squeeze_momentum < 0:
            score -= 0.2

        # EMA cross signal: GOLDEN (+1) is bullish, DEATH (-1) is bearish
        score += features.ema_cross_signal * 0.15

        # Convert to probability
        up_prob = 0.5 + np.tanh(score) * 0.3

        return MLPrediction(
            up_probability=up_prob,
            down_probability=1 - up_prob,
            confidence=abs(up_prob - 0.5) * 2,
            recommended_side="UP" if up_prob > 0.55 else ("DOWN" if up_prob < 0.45 else "SKIP"),
            model_agreement=0.5,
            feature_importance={}
        )

    def record_outcome(self, prediction_was_correct: bool):
        """Record whether our prediction was correct (for accuracy tracking)."""
        if prediction_was_correct:
            self.correct_predictions += 1

    def get_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made

    def get_status(self) -> Dict:
        """Get current ML engine status."""
        return {
            "is_trained": self.is_trained,
            "training_samples": len(self.training_samples),
            "predictions_made": self.predictions_made,
            "accuracy": self.get_accuracy(),
            "adaptive_threshold": self.adaptive_threshold,
            "recent_win_rate": sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0
        }


# Singleton instance
_ml_engine = None

def get_ml_engine() -> AdvancedMLEngine:
    """Get or create the ML engine singleton."""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = AdvancedMLEngine()
    return _ml_engine
