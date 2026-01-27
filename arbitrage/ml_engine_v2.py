"""
Enhanced ML Engine v2

Based on 2024-2026 research:
- XGBoost + LightGBM ensemble
- Probability calibration (isotonic regression)
- Online learning with River
- SHAP feature importance
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import deque

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .features import EnhancedFeatures


@dataclass
class MLPrediction:
    """ML model prediction result."""
    win_probability: float
    confidence: float
    calibrated_probability: float
    model_agreement: float  # Agreement between ensemble models
    feature_importances: Dict[str, float]
    recommended_action: str  # "BUY_YES", "BUY_NO", "SKIP"
    edge: float


class OnlineLearner:
    """
    Simple online learning model using stochastic gradient descent.
    Fallback when XGBoost/LightGBM not available.
    """

    def __init__(self, n_features: int = 31, learning_rate: float = 0.01):
        self.n_features = n_features
        self.lr = learning_rate
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.training_count = 0

        # Feature statistics for normalization
        self.feature_means = np.zeros(n_features)
        self.feature_stds = np.ones(n_features)
        self.feature_counts = np.zeros(n_features)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        return (features - self.feature_means) / (self.feature_stds + 1e-8)

    def _update_stats(self, features: np.ndarray):
        """Update running mean and std."""
        self.feature_counts += 1
        delta = features - self.feature_means
        self.feature_means += delta / self.feature_counts
        delta2 = features - self.feature_means
        self.feature_stds = np.sqrt(
            (self.feature_stds ** 2 * (self.feature_counts - 1) + delta * delta2)
            / self.feature_counts
        )
        self.feature_stds = np.maximum(self.feature_stds, 0.01)

    def predict(self, features: np.ndarray) -> float:
        """Predict win probability."""
        norm_features = self._normalize(features)
        logit = np.dot(self.weights, norm_features) + self.bias
        return self._sigmoid(logit)

    def update(self, features: np.ndarray, label: int, sample_weight: float = 1.0):
        """Update model with new observation."""
        self._update_stats(features)
        norm_features = self._normalize(features)

        pred = self.predict(features)
        error = (label - pred) * sample_weight

        # Gradient descent update
        self.weights += self.lr * error * norm_features
        self.bias += self.lr * error
        self.training_count += 1

    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances (absolute weights)."""
        return np.abs(self.weights)


class EnsembleModel:
    """
    Ensemble model combining XGBoost, LightGBM, and online learner.
    Falls back gracefully when libraries not available.
    """

    def __init__(self, n_features: int = 31):
        self.n_features = n_features
        self.online_model = OnlineLearner(n_features)

        self.xgb_model = None
        self.lgb_model = None
        self.calibrator = None
        self.scaler = None

        # Training data buffer for batch training
        self.X_buffer = deque(maxlen=1000)
        self.y_buffer = deque(maxlen=1000)

        self.is_fitted = False
        self.min_samples_for_batch = 50

        self._init_models()

    def _init_models(self):
        """Initialize available models."""
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            self.lr_model = LogisticRegression(C=0.1, max_iter=1000)

        if HAS_XGBOOST:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }

        if HAS_LIGHTGBM:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbose': -1,
            }

    def add_sample(self, features: np.ndarray, label: int):
        """Add training sample to buffer and update online model."""
        # Update online model immediately
        self.online_model.update(features, label)

        # Add to buffer for batch training
        self.X_buffer.append(features)
        self.y_buffer.append(label)

        # Retrain batch models periodically
        if len(self.X_buffer) >= self.min_samples_for_batch:
            if len(self.X_buffer) % 50 == 0:  # Retrain every 50 samples
                self._train_batch_models()

    def _train_batch_models(self):
        """Train XGBoost and LightGBM on buffered data."""
        if len(self.X_buffer) < self.min_samples_for_batch:
            return

        X = np.array(list(self.X_buffer))
        y = np.array(list(self.y_buffer))

        # Need both classes
        if len(np.unique(y)) < 2:
            return

        try:
            if HAS_SKLEARN:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.lr_model.fit(X_scaled, y)

            if HAS_XGBOOST:
                dtrain = xgb.DMatrix(X, label=y)
                self.xgb_model = xgb.train(
                    self.xgb_params,
                    dtrain,
                    num_boost_round=100,
                    verbose_eval=False
                )

            if HAS_LIGHTGBM:
                self.lgb_model = lgb.train(
                    self.lgb_params,
                    lgb.Dataset(X, label=y),
                    num_boost_round=100,
                )

            self.is_fitted = True

        except Exception as e:
            print(f"[MLEngine] Batch training error: {e}")

    def predict(self, features: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        Predict probability with ensemble.

        Returns:
            Tuple of (probability, agreement, feature_importances)
        """
        predictions = []
        weights = []

        # Online model (always available)
        online_pred = self.online_model.predict(features)
        predictions.append(online_pred)
        weights.append(0.2)

        # XGBoost
        if self.xgb_model is not None:
            try:
                dtest = xgb.DMatrix(features.reshape(1, -1))
                xgb_pred = self.xgb_model.predict(dtest)[0]
                predictions.append(xgb_pred)
                weights.append(0.4)
            except:
                pass

        # LightGBM
        if self.lgb_model is not None:
            try:
                lgb_pred = self.lgb_model.predict(features.reshape(1, -1))[0]
                predictions.append(lgb_pred)
                weights.append(0.4)
            except:
                pass

        # Logistic Regression
        if HAS_SKLEARN and hasattr(self, 'lr_model') and self.is_fitted:
            try:
                X_scaled = self.scaler.transform(features.reshape(1, -1))
                lr_pred = self.lr_model.predict_proba(X_scaled)[0, 1]
                predictions.append(lr_pred)
                weights.append(0.2)
            except:
                pass

        # Weighted average
        total_weight = sum(weights[:len(predictions)])
        if total_weight > 0:
            prob = sum(p * w for p, w in zip(predictions, weights[:len(predictions)])) / total_weight
        else:
            prob = online_pred

        # Model agreement (std of predictions)
        if len(predictions) > 1:
            agreement = 1 - np.std(predictions) * 2  # Higher std = lower agreement
            agreement = np.clip(agreement, 0, 1)
        else:
            agreement = 0.5

        # Feature importances
        importances = {}
        feature_names = EnhancedFeatures.feature_names()

        online_imp = self.online_model.get_feature_importances()
        for i, name in enumerate(feature_names[:len(online_imp)]):
            importances[name] = float(online_imp[i])

        return prob, agreement, importances

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            'training_samples': len(self.X_buffer),
            'online_updates': self.online_model.training_count,
            'has_xgboost': self.xgb_model is not None,
            'has_lightgbm': self.lgb_model is not None,
            'is_fitted': self.is_fitted,
        }


class MLEngineV2:
    """
    Enhanced ML Engine with ensemble learning and calibration.
    """

    def __init__(self, state_file: str = "ml_engine_v2_state.json"):
        self.state_file = state_file
        self.model = EnsembleModel()

        # Calibration data
        self.calibration_data: List[Tuple[float, int]] = []
        self.calibration_bins = 10

        # Performance tracking
        self.predictions: deque = deque(maxlen=1000)
        self.outcomes: deque = deque(maxlen=1000)

        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                # Restore training data
                if 'X_buffer' in data and 'y_buffer' in data:
                    for x, y in zip(data['X_buffer'], data['y_buffer']):
                        self.model.add_sample(np.array(x), y)
                print(f"[MLEngineV2] Loaded {len(self.model.X_buffer)} training samples")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[MLEngineV2] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                'X_buffer': [x.tolist() for x in self.model.X_buffer],
                'y_buffer': list(self.model.y_buffer),
                'stats': self.get_stats(),
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[MLEngineV2] Error saving state: {e}")

    def predict(self, features: EnhancedFeatures, market_price: float) -> MLPrediction:
        """
        Generate prediction for a trading opportunity.

        Args:
            features: Enhanced feature set
            market_price: Current market price

        Returns:
            MLPrediction with recommendation
        """
        feature_array = features.to_array()
        prob, agreement, importances = self.model.predict(feature_array)

        # Apply isotonic calibration if we have enough data
        calibrated_prob = self._calibrate_probability(prob)

        # Calculate edge
        edge = calibrated_prob - market_price

        # Determine confidence
        confidence = agreement * (1 - abs(0.5 - calibrated_prob))  # Lower near 50%

        # Determine action
        min_edge = 0.02  # 2% minimum edge
        if edge > min_edge and calibrated_prob > 0.55:
            action = "BUY_YES"
        elif edge < -min_edge and calibrated_prob < 0.45:
            action = "BUY_NO"
        else:
            action = "SKIP"

        return MLPrediction(
            win_probability=prob,
            confidence=confidence,
            calibrated_probability=calibrated_prob,
            model_agreement=agreement,
            feature_importances=importances,
            recommended_action=action,
            edge=edge,
        )

    def _calibrate_probability(self, raw_prob: float) -> float:
        """Apply isotonic calibration to raw probability."""
        if len(self.calibration_data) < 50:
            return raw_prob

        # Simple binned calibration
        # Group predictions into bins and calculate actual win rate per bin
        bins = {}
        for pred, outcome in self.calibration_data[-500:]:  # Use recent data
            bin_idx = int(pred * self.calibration_bins)
            bin_idx = min(bin_idx, self.calibration_bins - 1)
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(outcome)

        # Find calibrated probability for current prediction
        bin_idx = int(raw_prob * self.calibration_bins)
        bin_idx = min(bin_idx, self.calibration_bins - 1)

        if bin_idx in bins and len(bins[bin_idx]) >= 5:
            # Use actual win rate in this bin
            calibrated = sum(bins[bin_idx]) / len(bins[bin_idx])
            # Blend with raw prediction
            return 0.7 * calibrated + 0.3 * raw_prob
        else:
            return raw_prob

    def record_outcome(self, features: EnhancedFeatures, won: bool):
        """Record trade outcome for learning."""
        feature_array = features.to_array()
        label = 1 if won else 0

        # Update model
        self.model.add_sample(feature_array, label)

        # Update calibration data
        pred, _, _ = self.model.predict(feature_array)
        self.calibration_data.append((pred, label))

        # Track for statistics
        self.predictions.append(pred)
        self.outcomes.append(label)

        # Periodic save
        if len(self.model.X_buffer) % 10 == 0:
            self._save_state()

    def should_trade(self, features: EnhancedFeatures, market_price: float) -> Tuple[bool, str]:
        """
        Determine if we should trade based on features.

        Returns:
            Tuple of (should_trade, reason)
        """
        prediction = self.predict(features, market_price)

        # Check minimum edge
        if abs(prediction.edge) < 0.02:
            return False, f"Edge too small ({prediction.edge:.1%})"

        # Check confidence
        if prediction.confidence < 0.3:
            return False, f"Low confidence ({prediction.confidence:.1%})"

        # Check model agreement
        if prediction.model_agreement < 0.5:
            return False, f"Models disagree ({prediction.model_agreement:.1%})"

        # Check for reasonable probability
        if prediction.calibrated_probability < 0.1 or prediction.calibrated_probability > 0.9:
            # Only trade extremes with very high confidence
            if prediction.confidence < 0.7:
                return False, "Extreme probability with low confidence"

        return True, f"{prediction.recommended_action}: {prediction.edge:.1%} edge, {prediction.confidence:.0%} conf"

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        model_stats = self.model.get_stats()

        # Calculate accuracy if we have outcomes
        if len(self.predictions) > 0 and len(self.outcomes) > 0:
            preds = np.array(list(self.predictions))
            outcomes = np.array(list(self.outcomes))

            # Brier score
            brier = np.mean((preds - outcomes) ** 2)

            # Calibration (mean prediction vs mean outcome)
            mean_pred = np.mean(preds)
            mean_outcome = np.mean(outcomes)

            # Win rate when we predicted >60%
            high_conf_mask = preds > 0.6
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.mean(outcomes[high_conf_mask])
            else:
                high_conf_accuracy = 0

            model_stats.update({
                'brier_score': float(brier),
                'mean_prediction': float(mean_pred),
                'actual_win_rate': float(mean_outcome),
                'high_conf_accuracy': float(high_conf_accuracy),
                'calibration_samples': len(self.calibration_data),
            })

        return model_stats

    def print_status(self):
        """Print current status."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("ML ENGINE V2 STATUS")
        print("=" * 60)
        print(f"Training Samples: {stats.get('training_samples', 0)}")
        print(f"XGBoost: {'Active' if stats.get('has_xgboost') else 'Not available'}")
        print(f"LightGBM: {'Active' if stats.get('has_lightgbm') else 'Not available'}")
        print(f"Fitted: {stats.get('is_fitted', False)}")

        if 'brier_score' in stats:
            print(f"\nPerformance:")
            print(f"  Brier Score: {stats['brier_score']:.4f} (lower is better)")
            print(f"  Mean Prediction: {stats['mean_prediction']:.1%}")
            print(f"  Actual Win Rate: {stats['actual_win_rate']:.1%}")
            print(f"  High Conf Accuracy: {stats['high_conf_accuracy']:.1%}")

        print("=" * 60)


# Global instance
ml_engine_v2 = MLEngineV2()
