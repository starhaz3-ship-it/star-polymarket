"""
Meta-Labeling System (Lopez de Prado framework).

Instead of predicting direction (UP/DOWN), predicts WHETHER the primary
signal is correct. This filters bad signals and scales position size
by calibrated confidence.

Reference: mlfinlab labeling.py + "Advances in Financial Machine Learning" Ch. 3
"""

import json
import numpy as np
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

from .tsfresh_features import TSFreshFeatureExtractor


MODEL_DIR = Path(__file__).parent.parent / "ml_models"
TRAINING_FILE = Path(__file__).parent.parent / "meta_training_data.json"
HISTORY_FILE = Path(__file__).parent.parent / "ta_paper_results.json"
ML_DATA_FILE = Path(__file__).parent.parent / "ml_training_data.json"


@dataclass
class MetaFeatures:
    """Feature vector for the meta-labeling model (~25 features)."""

    # Category 1: Primary model confidence (5)
    primary_confidence: float = 0.5
    model_agreement: float = 0.5
    direction_score_margin: float = 0.0
    indicator_count: int = 0
    edge_magnitude: float = 0.0

    # Category 2: Market regime & context (6)
    regime_encoded: int = 0  # 0=chop, 1=range, 2=trend_up, 3=trend_down
    signal_regime_alignment: int = 0  # +1=with trend, 0=range, -1=counter
    entry_price: float = 0.5
    price_dist_from_50: float = 0.0
    nyu_edge_score: float = 0.0
    time_remaining_min: float = 15.0

    # Category 3: Momentum alignment (4)
    momentum_aligns_signal: int = 0  # +1 aligned, -1 opposing
    rsi_aligns_signal: int = 0
    macd_aligns_signal: int = 0
    squeeze_state: int = 0  # 0=none, 1=on, 2=fired

    # Category 4: Recent performance (4)
    recent_win_rate_10: float = 0.5
    recent_win_rate_by_side: float = 0.5
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Category 5: Order flow quality (3)
    obi_aligns_signal: int = 0
    cvd_aligns_signal: int = 0
    ema_cross_aligns_signal: int = 0

    # Category 6: Bregman info (3)
    kl_divergence: float = 0.0
    kelly_fraction: float = 0.0
    guaranteed_profit: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.primary_confidence, self.model_agreement,
            self.direction_score_margin, self.indicator_count, self.edge_magnitude,
            self.regime_encoded, self.signal_regime_alignment,
            self.entry_price, self.price_dist_from_50,
            self.nyu_edge_score, self.time_remaining_min,
            self.momentum_aligns_signal, self.rsi_aligns_signal,
            self.macd_aligns_signal, self.squeeze_state,
            self.recent_win_rate_10, self.recent_win_rate_by_side,
            self.consecutive_wins, self.consecutive_losses,
            self.obi_aligns_signal, self.cvd_aligns_signal,
            self.ema_cross_aligns_signal,
            self.kl_divergence, self.kelly_fraction, self.guaranteed_profit,
        ], dtype=np.float64)

    FEATURE_NAMES = [
        "primary_confidence", "model_agreement",
        "direction_score_margin", "indicator_count", "edge_magnitude",
        "regime_encoded", "signal_regime_alignment",
        "entry_price", "price_dist_from_50",
        "nyu_edge_score", "time_remaining_min",
        "momentum_aligns_signal", "rsi_aligns_signal",
        "macd_aligns_signal", "squeeze_state",
        "recent_win_rate_10", "recent_win_rate_by_side",
        "consecutive_wins", "consecutive_losses",
        "obi_aligns_signal", "cvd_aligns_signal",
        "ema_cross_aligns_signal",
        "kl_divergence", "kelly_fraction", "guaranteed_profit",
    ]


@dataclass
class MetaPrediction:
    """Output of the meta-labeling model."""
    p_correct: float          # P(primary signal is correct)
    recommended_action: str   # "TRADE" or "SKIP"
    position_size_mult: float  # Sizing multiplier (0.0 to 2.0)
    confidence_tier: str      # "HIGH", "MEDIUM", "LOW"


class MetaLabeler:
    """
    Meta-labeling model: predicts P(primary signal is correct).

    - Primary model (TA signals) determines direction (UP/DOWN)
    - Meta-model determines whether to take the trade and how to size it
    - Trained on binary labels: 1=primary was correct (WIN), 0=wrong (LOSS)
    """

    MIN_SAMPLES = 30
    RETRAIN_EVERY = 20
    TRADE_THRESHOLD = 0.55
    HIGH_CONFIDENCE = 0.75

    def __init__(self):
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_samples: List[dict] = []
        self.samples_since_train = 0
        self.predictions_made = 0
        self.avg_p_correct = 0.5
        self.trade_rate = 1.0

        # Performance tracking for recent_win_rate features
        self.recent_outcomes: List[Tuple[str, bool]] = []  # (side, won)
        self.streak_wins = 0
        self.streak_losses = 0

        # tsfresh feature extractor
        self.tsfresh = TSFreshFeatureExtractor()

        self._load()
        if not self.is_trained and len(self.training_samples) >= self.MIN_SAMPLES:
            self._train()

    def _load(self):
        """Load saved meta-model and training data."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Load training data
        if TRAINING_FILE.exists():
            try:
                data = json.load(open(TRAINING_FILE))
                self.training_samples = data.get("samples", [])
                self.recent_outcomes = [
                    (o["side"], o["won"]) for o in data.get("recent_outcomes", [])
                ]
                self.streak_wins = data.get("streak_wins", 0)
                self.streak_losses = data.get("streak_losses", 0)
                self.predictions_made = data.get("predictions_made", 0)
            except Exception as e:
                print(f"[META] Error loading training data: {e}")

        # Load model
        meta_model_path = MODEL_DIR / "meta_model.pkl"
        scaler_path = MODEL_DIR / "meta_scaler.pkl"
        if meta_model_path.exists() and scaler_path.exists():
            try:
                self.meta_model = pickle.load(open(meta_model_path, "rb"))
                self.scaler = pickle.load(open(scaler_path, "rb"))
                self.is_trained = True
                print(f"[META] Loaded meta-model ({len(self.training_samples)} samples)")
            except Exception as e:
                print(f"[META] Error loading model: {e}")

        # Bootstrap from history if no training data
        if not self.training_samples:
            self._bootstrap_from_history()

    def _save(self):
        """Save training data and model."""
        try:
            data = {
                "samples": self.training_samples[-1000:],
                "recent_outcomes": [
                    {"side": s, "won": w} for s, w in self.recent_outcomes[-100:]
                ],
                "streak_wins": self.streak_wins,
                "streak_losses": self.streak_losses,
                "predictions_made": self.predictions_made,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(TRAINING_FILE, "w") as f:
                json.dump(data, f, indent=2)

            if self.meta_model is not None:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                pickle.dump(self.meta_model, open(MODEL_DIR / "meta_model.pkl", "wb"))
                pickle.dump(self.scaler, open(MODEL_DIR / "meta_scaler.pkl", "wb"))
        except Exception as e:
            print(f"[META] Error saving: {e}")

    def _bootstrap_from_history(self):
        """Convert existing trade history to meta-training data."""
        trades = []

        # Load from ta_paper_results.json
        if HISTORY_FILE.exists():
            try:
                data = json.load(open(HISTORY_FILE))
                for tid, t in data.get("trades", {}).items():
                    if t.get("status") == "closed":
                        trades.append(t)
            except Exception:
                pass

        if not trades:
            print("[META] No historical trades for bootstrap")
            return

        # Sort chronologically
        trades.sort(key=lambda t: t.get("entry_time", ""))

        # Build meta-training samples from available data
        win_history = []
        up_history = []
        down_history = []

        for t in trades:
            won = t.get("pnl", 0) > 0
            side = t.get("side", "DOWN")
            entry_price = t.get("entry_price", 0.5)

            # Compute recent performance features chronologically
            recent_wr = sum(win_history[-10:]) / max(1, len(win_history[-10:])) if win_history else 0.5
            side_history = up_history if side == "UP" else down_history
            side_wr = sum(side_history[-20:]) / max(1, len(side_history[-20:])) if side_history else 0.5

            # Count streaks
            c_wins = 0
            c_losses = 0
            for w in reversed(win_history):
                if w:
                    c_wins += 1
                else:
                    break
            for w in reversed(win_history):
                if not w:
                    c_losses += 1
                else:
                    break

            # Build partial meta-features from available data
            features = MetaFeatures(
                primary_confidence=0.6 + t.get("edge_at_entry", 0.1),
                model_agreement=0.7,  # Estimate - not stored
                direction_score_margin=2 if won else -1,  # Rough estimate
                indicator_count=4,  # Estimate
                edge_magnitude=abs(t.get("edge_at_entry", 0.1)),
                regime_encoded=1,  # Unknown - default to range
                signal_regime_alignment=0,  # Unknown
                entry_price=entry_price,
                price_dist_from_50=abs(entry_price - 0.50),
                nyu_edge_score=abs(entry_price - 0.50) * 2,  # Approximate
                time_remaining_min=10.0,  # Estimate
                momentum_aligns_signal=1 if won else 0,  # Rough
                rsi_aligns_signal=0,  # Unknown
                macd_aligns_signal=0,  # Unknown
                squeeze_state=0,  # Unknown
                recent_win_rate_10=recent_wr,
                recent_win_rate_by_side=side_wr,
                consecutive_wins=c_wins,
                consecutive_losses=c_losses,
                obi_aligns_signal=0,  # Unknown
                cvd_aligns_signal=0,  # Unknown
                ema_cross_aligns_signal=0,  # Unknown
                kl_divergence=t.get("kl_divergence", 0.1),
                kelly_fraction=t.get("kelly_fraction", 0.2),
                guaranteed_profit=t.get("guaranteed_profit", 0.0),
            )

            sample = {
                "features": features.to_array().tolist(),
                "meta_label": 1 if won else 0,
                "pnl": t.get("pnl", 0),
                "side": side,
                "timestamp": t.get("entry_time", ""),
            }
            self.training_samples.append(sample)

            # Update tracking
            win_history.append(won)
            if side == "UP":
                up_history.append(won)
            else:
                down_history.append(won)
            self.recent_outcomes.append((side, won))

        print(f"[META] Bootstrapped {len(self.training_samples)} samples from trade history")
        self._save()

    def extract_meta_features(
        self, signal, bregman_signal, ml_prediction, nyu_result,
        candles, entry_price: float, side: str, momentum: float
    ) -> MetaFeatures:
        """Extract meta-features from all available signal data."""

        # Category 1: Primary model confidence
        primary_conf = signal.model_up if side == "UP" else signal.model_down
        model_agree = ml_prediction.model_agreement if ml_prediction else 0.5
        dir_margin = 0
        indicator_count = 0
        if hasattr(signal, 'direction') and signal.direction:
            dir_margin = signal.direction.up_score - signal.direction.down_score
            if side == "DOWN":
                dir_margin = -dir_margin
            if hasattr(signal.direction, 'scoring_breakdown'):
                indicator_count = len([v for v in signal.direction.scoring_breakdown.values() if v])

        edge = signal.edge_up if side == "UP" else signal.edge_down
        edge_mag = abs(edge) if edge else 0.0

        # Category 2: Market regime
        regime_map = {"chop": 0, "range": 1, "trend_up": 2, "trend_down": 3}
        regime_enc = regime_map.get(signal.regime.value, 0) if hasattr(signal.regime, 'value') else 0

        # Signal-regime alignment
        alignment = 0
        regime_val = signal.regime.value if hasattr(signal.regime, 'value') else ""
        if side == "UP" and regime_val == "trend_up":
            alignment = 1
        elif side == "DOWN" and regime_val == "trend_down":
            alignment = 1
        elif regime_val in ("trend_up", "trend_down"):
            alignment = -1  # Counter-trend

        price_dist = abs(entry_price - 0.50)
        nyu_score = nyu_result.edge_score if nyu_result and hasattr(nyu_result, 'edge_score') else price_dist * 2
        time_rem = signal.time_remaining_min if hasattr(signal, 'time_remaining_min') else 10.0

        # Category 3: Momentum alignment
        mom_align = 1 if (side == "UP" and momentum > 0) or (side == "DOWN" and momentum < 0) else -1
        rsi_align = 0
        if signal.rsi:
            if side == "UP" and signal.rsi > 50:
                rsi_align = 1
            elif side == "DOWN" and signal.rsi < 50:
                rsi_align = 1
            else:
                rsi_align = -1

        macd_align = 0
        if signal.macd and hasattr(signal.macd, 'histogram_delta'):
            h = signal.macd.histogram_delta
            if (side == "UP" and h > 0) or (side == "DOWN" and h < 0):
                macd_align = 1
            elif h != 0:
                macd_align = -1

        sq_state = 0
        if signal.squeeze:
            if hasattr(signal.squeeze, 'squeeze_fired') and signal.squeeze.squeeze_fired:
                sq_state = 2
            elif signal.squeeze.squeeze_on:
                sq_state = 1

        # Category 4: Recent performance
        recent_10 = [w for _, w in self.recent_outcomes[-10:]]
        wr_10 = sum(recent_10) / max(1, len(recent_10)) if recent_10 else 0.5
        side_outcomes = [w for s, w in self.recent_outcomes[-20:] if s == side]
        wr_side = sum(side_outcomes) / max(1, len(side_outcomes)) if side_outcomes else 0.5

        # Category 5: Order flow alignment
        obi_align = 0
        cvd_align = 0
        ema_align = 0
        if signal.order_flow:
            if (side == "UP" and signal.order_flow.obi > 0.05) or \
               (side == "DOWN" and signal.order_flow.obi < -0.05):
                obi_align = 1
            elif abs(signal.order_flow.obi) > 0.05:
                obi_align = -1

            if (side == "UP" and signal.order_flow.cvd_5m > 0) or \
               (side == "DOWN" and signal.order_flow.cvd_5m < 0):
                cvd_align = 1
            elif signal.order_flow.cvd_5m != 0:
                cvd_align = -1

        if signal.ema_cross:
            if (side == "UP" and signal.ema_cross == "GOLDEN") or \
               (side == "DOWN" and signal.ema_cross == "DEATH"):
                ema_align = 1
            elif signal.ema_cross != "NEUTRAL":
                ema_align = -1

        return MetaFeatures(
            primary_confidence=primary_conf,
            model_agreement=model_agree,
            direction_score_margin=dir_margin,
            indicator_count=indicator_count,
            edge_magnitude=edge_mag,
            regime_encoded=regime_enc,
            signal_regime_alignment=alignment,
            entry_price=entry_price,
            price_dist_from_50=price_dist,
            nyu_edge_score=nyu_score,
            time_remaining_min=time_rem,
            momentum_aligns_signal=mom_align,
            rsi_aligns_signal=rsi_align,
            macd_aligns_signal=macd_align,
            squeeze_state=sq_state,
            recent_win_rate_10=wr_10,
            recent_win_rate_by_side=wr_side,
            consecutive_wins=self.streak_wins,
            consecutive_losses=self.streak_losses,
            obi_aligns_signal=obi_align,
            cvd_aligns_signal=cvd_align,
            ema_cross_aligns_signal=ema_align,
            kl_divergence=bregman_signal.kl_divergence if bregman_signal else 0.0,
            kelly_fraction=bregman_signal.kelly_fraction if bregman_signal else 0.0,
            guaranteed_profit=bregman_signal.guaranteed_profit if bregman_signal else 0.0,
        )

    def predict(self, meta_features: MetaFeatures) -> MetaPrediction:
        """Predict P(primary signal is correct) and recommended sizing."""
        self.predictions_made += 1

        if not self.is_trained or not LGB_AVAILABLE:
            return self._heuristic_predict(meta_features)

        try:
            X = meta_features.to_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            p_correct = float(self.meta_model.predict(X_scaled)[0])
            p_correct = np.clip(p_correct, 0.0, 1.0)
        except Exception:
            return self._heuristic_predict(meta_features)

        # Update running stats
        self.avg_p_correct = 0.95 * self.avg_p_correct + 0.05 * p_correct

        # Determine action and sizing
        if p_correct < self.TRADE_THRESHOLD:
            action = "SKIP"
            size_mult = 0.0
            tier = "LOW"
        elif p_correct >= self.HIGH_CONFIDENCE:
            action = "TRADE"
            size_mult = 1.0 + (p_correct - self.HIGH_CONFIDENCE) / 0.25 * 1.0
            size_mult = min(size_mult, 2.0)
            tier = "HIGH"
        else:
            action = "TRADE"
            size_mult = 0.5 + (p_correct - self.TRADE_THRESHOLD) / (self.HIGH_CONFIDENCE - self.TRADE_THRESHOLD) * 0.5
            tier = "MEDIUM"

        return MetaPrediction(
            p_correct=p_correct,
            recommended_action=action,
            position_size_mult=size_mult,
            confidence_tier=tier,
        )

    def _heuristic_predict(self, mf: MetaFeatures) -> MetaPrediction:
        """Heuristic prediction when model isn't trained yet."""
        score = 0.5

        # Strong edge = likely correct
        score += mf.edge_magnitude * 0.5

        # Momentum alignment is critical
        if mf.momentum_aligns_signal > 0:
            score += 0.08
        elif mf.momentum_aligns_signal < 0:
            score -= 0.10

        # RSI + MACD alignment
        score += mf.rsi_aligns_signal * 0.03
        score += mf.macd_aligns_signal * 0.03

        # With-trend signals are better
        score += mf.signal_regime_alignment * 0.05

        # Cheap entries (far from 0.50) are better
        score += mf.price_dist_from_50 * 0.15

        # Order flow confirmation
        score += mf.obi_aligns_signal * 0.03
        score += mf.cvd_aligns_signal * 0.02

        # Squeeze fired is very bullish for signal quality
        if mf.squeeze_state == 2:
            score += 0.08

        # Recent performance matters
        score += (mf.recent_win_rate_10 - 0.5) * 0.1

        # High KL divergence = strong mispricing = better signal
        score += min(mf.kl_divergence, 1.0) * 0.05

        p_correct = np.clip(score, 0.1, 0.95)

        if p_correct < self.TRADE_THRESHOLD:
            action = "SKIP"
            size_mult = 0.0
            tier = "LOW"
        elif p_correct >= self.HIGH_CONFIDENCE:
            action = "TRADE"
            size_mult = min(2.0, 1.0 + (p_correct - self.HIGH_CONFIDENCE) / 0.25)
            tier = "HIGH"
        else:
            action = "TRADE"
            size_mult = 0.5 + (p_correct - self.TRADE_THRESHOLD) / (self.HIGH_CONFIDENCE - self.TRADE_THRESHOLD) * 0.5
            tier = "MEDIUM"

        return MetaPrediction(p_correct=p_correct, recommended_action=action,
                              position_size_mult=size_mult, confidence_tier=tier)

    def calculate_position_size(self, p_correct: float, base_size: float,
                                 kelly_fraction: float) -> float:
        """
        Meta-label guided position sizing.
        P(correct) scales the Kelly-optimal position.
        """
        if p_correct < self.TRADE_THRESHOLD:
            return 0.0

        # Scale from 0.5x at threshold to 2.0x at 0.95
        range_p = 0.95 - self.TRADE_THRESHOLD
        pct = min(1.0, (p_correct - self.TRADE_THRESHOLD) / range_p)
        multiplier = 0.5 + pct * 1.5

        # Apply Kelly fraction cap
        max_kelly_size = base_size * min(kelly_fraction * 4, 2.5)
        sized = multiplier * base_size

        return max(base_size * 0.5, min(sized, max_kelly_size, base_size * 2.5))

    def add_sample(self, meta_features: MetaFeatures, won: bool, pnl: float,
                   side: str = "DOWN"):
        """Add completed trade as training data."""
        sample = {
            "features": meta_features.to_array().tolist(),
            "meta_label": 1 if won else 0,
            "pnl": pnl,
            "side": side,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.training_samples.append(sample)
        self.samples_since_train += 1

        # Update streak tracking
        self.recent_outcomes.append((side, won))
        if won:
            self.streak_wins += 1
            self.streak_losses = 0
        else:
            self.streak_losses += 1
            self.streak_wins = 0

        # Track trade rate
        recent_preds = self.predictions_made
        recent_trades = len([s for s in self.training_samples[-50:]])
        self.trade_rate = recent_trades / max(1, recent_preds) if recent_preds > 0 else 1.0

        # Auto-retrain
        if self.samples_since_train >= self.RETRAIN_EVERY and \
           len(self.training_samples) >= self.MIN_SAMPLES:
            self._train()

        self._save()

    def _train(self):
        """Train the meta-model on accumulated data."""
        if not LGB_AVAILABLE:
            print("[META] LightGBM not available, using heuristic")
            return

        samples = self.training_samples
        if len(samples) < self.MIN_SAMPLES:
            return

        X = np.array([s["features"] for s in samples])
        y = np.array([s["meta_label"] for s in samples])

        # Sample weights: recent trades weighted more, high-PnL trades weighted more
        weights = np.ones(len(samples))
        for i, s in enumerate(samples):
            # Time decay: newer samples get higher weight
            age_pct = i / len(samples)  # 0=oldest, 1=newest
            weights[i] = 0.5 + 0.5 * age_pct

            # PnL weighting: big winners/losers are more informative
            pnl_weight = min(2.0, 1.0 + abs(s.get("pnl", 0)) / 20.0)
            weights[i] *= pnl_weight

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train/val split (time-based: last 20% for validation)
        split = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]
        w_train = weights[:split]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(1, n_pos)

        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train,
                                 feature_name=MetaFeatures.FEATURE_NAMES)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "max_depth": 4,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 3,
            "scale_pos_weight": scale,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        }

        try:
            self.meta_model = lgb.train(
                params, train_data, num_boost_round=150,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            self.is_trained = True
            self.samples_since_train = 0

            # Evaluate on val set
            val_preds = self.meta_model.predict(X_val)
            val_acc = ((val_preds > 0.5) == y_val).mean()

            # Feature importance
            importance = dict(zip(MetaFeatures.FEATURE_NAMES,
                                  self.meta_model.feature_importance("gain")))
            top_3 = sorted(importance.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v:.0f}" for k, v in top_3)

            print(f"[META] Trained on {len(X_train)} samples | Val acc: {val_acc:.1%} | Top: {top_str}")
            self._save()

        except Exception as e:
            print(f"[META] Training error: {e}")

    def get_status(self) -> dict:
        """Get meta-labeler status for reporting."""
        return {
            "is_trained": self.is_trained,
            "training_samples": len(self.training_samples),
            "predictions_made": self.predictions_made,
            "avg_p_correct": self.avg_p_correct,
            "trade_rate": self.trade_rate,
            "streak_wins": self.streak_wins,
            "streak_losses": self.streak_losses,
            "position_size_range": "$5-$20" if self.is_trained else "$10 (fixed)",
        }


# Singleton
_meta_labeler = None

def get_meta_labeler() -> MetaLabeler:
    """Get or create singleton MetaLabeler instance."""
    global _meta_labeler
    if _meta_labeler is None:
        _meta_labeler = MetaLabeler()
    return _meta_labeler
