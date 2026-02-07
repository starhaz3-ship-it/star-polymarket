"""
tsfresh auto-feature extraction from candle data.

Uses MinimalFCParameters for real-time speed (~10 extractors per series).
Extracts features from close/high/low/volume as 4 separate time series.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False


CACHE_FILE = Path(__file__).parent.parent / "ml_models" / "tsfresh_selected.json"


class TSFreshFeatureExtractor:
    """
    Extracts time-series features from candle data using tsfresh.
    Designed for real-time inference with MinimalFCParameters.
    """

    def __init__(self):
        self.selected_features: List[str] = []
        self._load_selected()

    def _load_selected(self):
        """Load cached selected feature names."""
        if CACHE_FILE.exists():
            try:
                data = json.load(open(CACHE_FILE))
                self.selected_features = data.get("features", [])
            except Exception:
                pass

    def _save_selected(self):
        """Save selected feature names to cache."""
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump({"features": self.selected_features}, f, indent=2)

    def candles_to_dataframe(self, candles) -> pd.DataFrame:
        """
        Convert candle list to tsfresh-compatible long-format DataFrame.
        Creates 4 time series: close, high, low, volume.
        """
        rows = []
        for i, c in enumerate(candles):
            rows.append({"id": "close", "time": i, "value": c.close})
            rows.append({"id": "high", "time": i, "value": c.high})
            rows.append({"id": "low", "time": i, "value": c.low})
            rows.append({"id": "volume", "time": i, "value": c.volume})
        return pd.DataFrame(rows)

    def extract(self, candles) -> Dict[str, float]:
        """
        Extract tsfresh features from recent candles.
        Returns dict of feature_name -> value.
        """
        if not TSFRESH_AVAILABLE or not candles or len(candles) < 5:
            return {}

        try:
            df = self.candles_to_dataframe(candles[-15:])

            features = extract_features(
                df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True,
                n_jobs=1,
            )

            result = {}
            for col in features.columns:
                val = features[col].values[0]
                if np.isfinite(val):
                    # Only return selected features if we have a selection
                    if not self.selected_features or col in self.selected_features:
                        result[col] = float(val)

            return result
        except Exception as e:
            return {}

    def discover_features(self, candle_sets: List[list], meta_labels: List[int]) -> List[str]:
        """
        Offline feature discovery: find which tsfresh features predict meta-label.

        Args:
            candle_sets: List of candle lists (one per trade)
            meta_labels: List of 0/1 meta-labels (1=primary was correct)

        Returns:
            List of selected feature names
        """
        if not TSFRESH_AVAILABLE or len(candle_sets) < 20:
            return []

        try:
            from tsfresh import select_features

            # Build combined DataFrame with trade_id as identifier
            all_rows = []
            for trade_idx, candles in enumerate(candle_sets):
                if not candles or len(candles) < 5:
                    continue
                for i, c in enumerate(candles[-15:]):
                    all_rows.append({"id": trade_idx, "time": i, "close": c.close,
                                     "high": c.high, "low": c.low, "volume": c.volume})

            if not all_rows:
                return []

            df = pd.DataFrame(all_rows)

            # Extract features (wide format - one row per trade)
            features = extract_features(
                df, column_id="id", column_sort="time",
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True, n_jobs=1
            )

            # Align with labels
            y = pd.Series(meta_labels[:len(features)], index=features.index)

            # Select relevant features
            selected = select_features(features, y)
            self.selected_features = list(selected.columns)
            self._save_selected()

            print(f"[tsfresh] Selected {len(self.selected_features)} features from {len(features.columns)}")
            return self.selected_features

        except Exception as e:
            print(f"[tsfresh] Discovery error: {e}")
            return []
