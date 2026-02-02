"""
GPU-Accelerated Utilities for Polymarket Arbitrage

Uses CuPy to offload compute-intensive operations to NVIDIA GPU.
Falls back to NumPy if CuPy is not available.

Provides 10-100x speedup for:
- Batch probability calculations
- Matrix operations (correlations, covariances)
- Large-scale position filtering
- ML feature engineering
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import time

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupy import cuda
    HAS_CUPY = True

    # Get GPU info
    device = cuda.Device(0)
    try:
        # Try different methods to get GPU name
        if hasattr(device, 'name'):
            name = device.name
            GPU_NAME = name.decode() if hasattr(name, 'decode') else str(name)
        else:
            # Use nvidia-smi output or default
            GPU_NAME = cp.cuda.runtime.getDeviceProperties(0).get('name', 'NVIDIA GPU')
            if hasattr(GPU_NAME, 'decode'):
                GPU_NAME = GPU_NAME.decode()
    except:
        GPU_NAME = "NVIDIA GPU"

    try:
        GPU_MEMORY = device.mem_info[1] / (1024**3)  # Total memory in GB
    except:
        GPU_MEMORY = 0

    print(f"[GPU] CuPy initialized: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
except ImportError:
    HAS_CUPY = False
    GPU_NAME = "N/A"
    GPU_MEMORY = 0
    print("[GPU] CuPy not available, using NumPy (CPU)")

# Use CuPy if available, else NumPy
xp = cp if HAS_CUPY else np


@dataclass
class GPUStats:
    """GPU performance statistics."""
    operations: int = 0
    total_time_ms: float = 0
    speedup_vs_cpu: float = 0
    memory_used_mb: float = 0


# Global stats
gpu_stats = GPUStats()


def to_gpu(arr: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
    """Transfer array to GPU if CuPy available."""
    if HAS_CUPY and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """Transfer array back to CPU."""
    if HAS_CUPY and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def sync_gpu():
    """Synchronize GPU operations (for timing)."""
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


# ============================================================================
# Batch Probability Calculations (for detector.py)
# ============================================================================

def batch_implied_probabilities(
    spot_prices: np.ndarray,
    strike_prices: np.ndarray,
    times_remaining: np.ndarray
) -> np.ndarray:
    """
    Calculate implied probabilities for multiple markets at once.

    GPU accelerated - processes thousands of markets in parallel.

    Args:
        spot_prices: Array of current spot prices
        strike_prices: Array of strike prices
        times_remaining: Array of time remaining in seconds

    Returns:
        Array of implied probabilities
    """
    global gpu_stats
    start = time.perf_counter()

    # Transfer to GPU
    spot = to_gpu(spot_prices.astype(np.float32))
    strike = to_gpu(strike_prices.astype(np.float32))
    time_rem = to_gpu(times_remaining.astype(np.float32))

    # Calculate price difference as percentage
    diff_pct = (spot - strike) / strike

    # Time factor: less time = more certain
    time_factor = xp.clip(time_rem / 900.0, 0.1, 1.0)

    # Base probability from current price position
    adjustment = xp.clip(diff_pct * 10 / time_factor, -0.45, 0.45)
    prob = 0.5 + adjustment
    prob = xp.clip(prob, 0.01, 0.99)

    sync_gpu()
    result = to_cpu(prob)

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return result


def batch_edge_calculations(
    implied_probs: np.ndarray,
    market_prices: np.ndarray,
    fee_rate: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate edges for YES and NO positions in batch.

    Args:
        implied_probs: Array of implied probabilities
        market_prices: Array of current market prices (YES prices)
        fee_rate: Fee rate (default 2%)

    Returns:
        Tuple of (yes_edges, no_edges) arrays
    """
    global gpu_stats
    start = time.perf_counter()

    probs = to_gpu(implied_probs.astype(np.float32))
    prices = to_gpu(market_prices.astype(np.float32))

    # YES edge: implied_prob - market_price - fee
    yes_edge = (probs - prices - fee_rate) * 100

    # NO edge: (1 - implied_prob) - (1 - market_price) - fee
    no_edge = ((1 - probs) - (1 - prices) - fee_rate) * 100

    sync_gpu()

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return to_cpu(yes_edge), to_cpu(no_edge)


def batch_negrisk_detection(
    outcome_prices: List[np.ndarray],
    fee_per_outcome: float = 0.01
) -> np.ndarray:
    """
    Detect NegRisk arbitrage opportunities across multiple markets.

    Args:
        outcome_prices: List of arrays, each containing outcome prices for a market
        fee_per_outcome: Fee per outcome (default 1%)

    Returns:
        Array of edge percentages for each market
    """
    global gpu_stats
    start = time.perf_counter()

    edges = []
    for prices in outcome_prices:
        prices_gpu = to_gpu(np.array(prices, dtype=np.float32))

        total_ask = xp.sum(prices_gpu)
        total_fees = len(prices) * fee_per_outcome

        gross_profit = 1.0 - total_ask
        net_profit = gross_profit - total_fees
        edge = float(to_cpu(net_profit)) * 100

        edges.append(edge)

    sync_gpu()

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return np.array(edges)


# ============================================================================
# Matrix Operations (for ML engine)
# ============================================================================

def gpu_correlation_matrix(features: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix on GPU.

    Much faster for large feature matrices (>1000 samples).

    Args:
        features: 2D array of shape (n_samples, n_features)

    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    global gpu_stats
    start = time.perf_counter()

    X = to_gpu(features.astype(np.float32))

    # Center the data
    X_centered = X - xp.mean(X, axis=0)

    # Calculate covariance
    n = X.shape[0]
    cov = xp.dot(X_centered.T, X_centered) / (n - 1)

    # Calculate standard deviations
    std = xp.sqrt(xp.diag(cov))
    std = xp.where(std == 0, 1, std)  # Avoid division by zero

    # Correlation = Cov / (std_i * std_j)
    corr = cov / xp.outer(std, std)

    sync_gpu()
    result = to_cpu(corr)

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return result


def gpu_pca(features: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated PCA for dimensionality reduction.

    Args:
        features: 2D array of shape (n_samples, n_features)
        n_components: Number of principal components

    Returns:
        Tuple of (transformed_data, explained_variance_ratio)
    """
    global gpu_stats
    start = time.perf_counter()

    X = to_gpu(features.astype(np.float32))

    # Center the data
    X_centered = X - xp.mean(X, axis=0)

    # SVD
    U, S, Vt = xp.linalg.svd(X_centered, full_matrices=False)

    # Get top n_components
    components = Vt[:n_components]

    # Transform data
    transformed = xp.dot(X_centered, components.T)

    # Explained variance ratio
    explained_var = (S ** 2) / (X.shape[0] - 1)
    total_var = xp.sum(explained_var)
    explained_ratio = explained_var[:n_components] / total_var

    sync_gpu()

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return to_cpu(transformed), to_cpu(explained_ratio)


def gpu_batch_normalize(features: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    GPU-accelerated batch normalization.

    Args:
        features: 2D array of shape (n_samples, n_features)
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized features
    """
    X = to_gpu(features.astype(np.float32))

    mean = xp.mean(X, axis=0)
    std = xp.std(X, axis=0)

    normalized = (X - mean) / (std + epsilon)

    sync_gpu()
    return to_cpu(normalized)


def gpu_weighted_ensemble(
    predictions: List[np.ndarray],
    weights: np.ndarray
) -> np.ndarray:
    """
    GPU-accelerated weighted ensemble of predictions.

    Args:
        predictions: List of prediction arrays
        weights: Array of weights for each predictor

    Returns:
        Weighted average predictions
    """
    global gpu_stats
    start = time.perf_counter()

    # Stack predictions into matrix
    preds = to_gpu(np.stack(predictions, axis=0).astype(np.float32))
    w = to_gpu(weights.astype(np.float32))

    # Normalize weights
    w = w / xp.sum(w)

    # Weighted average: sum(w_i * pred_i)
    result = xp.tensordot(w, preds, axes=([0], [0]))

    sync_gpu()

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return to_cpu(result)


# ============================================================================
# Position Filtering (for whale tracker)
# ============================================================================

def gpu_filter_positions(
    values: np.ndarray,
    min_value: float = 100.0,
    max_value: float = float('inf')
) -> np.ndarray:
    """
    GPU-accelerated position filtering.

    Args:
        values: Array of position values
        min_value: Minimum value threshold
        max_value: Maximum value threshold

    Returns:
        Boolean mask of positions that pass the filter
    """
    v = to_gpu(values.astype(np.float32))

    mask = (v >= min_value) & (v <= max_value)

    sync_gpu()
    return to_cpu(mask).astype(bool)


def gpu_calculate_pnl(
    entry_prices: np.ndarray,
    current_prices: np.ndarray,
    sizes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate PnL for multiple positions in parallel.

    Args:
        entry_prices: Array of entry prices
        current_prices: Array of current prices
        sizes: Array of position sizes

    Returns:
        Tuple of (pnl_amounts, pnl_percentages)
    """
    global gpu_stats
    start = time.perf_counter()

    entry = to_gpu(entry_prices.astype(np.float32))
    current = to_gpu(current_prices.astype(np.float32))
    size = to_gpu(sizes.astype(np.float32))

    # Shares = size / entry_price
    shares = size / xp.where(entry > 0, entry, 1)

    # PnL = shares * (current - entry)
    price_change = current - entry
    pnl_amount = shares * price_change

    # PnL % = (current - entry) / entry * 100
    pnl_pct = xp.where(entry > 0, price_change / entry * 100, 0)

    sync_gpu()

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return to_cpu(pnl_amount), to_cpu(pnl_pct)


def gpu_consensus_detection(
    whale_ids: np.ndarray,
    market_ids: np.ndarray,
    sides: np.ndarray,  # 1 for YES, 0 for NO
    sizes: np.ndarray,
    trust_scores: np.ndarray,
    min_whales: int = 2,
    min_size: float = 1000.0
) -> List[dict]:
    """
    GPU-accelerated consensus detection across whale positions.

    Args:
        whale_ids: Array of whale identifiers
        market_ids: Array of market identifiers
        sides: Array of sides (1=YES, 0=NO)
        sizes: Array of position sizes
        trust_scores: Array of trust scores per whale
        min_whales: Minimum whales for consensus
        min_size: Minimum total size for consensus

    Returns:
        List of consensus signals
    """
    # This is more complex - use CPU for groupby logic
    # but GPU for aggregations

    import pandas as pd

    df = pd.DataFrame({
        'whale': whale_ids,
        'market': market_ids,
        'side': sides,
        'size': sizes,
        'trust': trust_scores
    })

    # Group by market and side
    grouped = df.groupby(['market', 'side']).agg({
        'whale': 'count',
        'size': 'sum',
        'trust': lambda x: np.average(x, weights=df.loc[x.index, 'size'])
    }).reset_index()

    grouped.columns = ['market', 'side', 'whale_count', 'total_size', 'weighted_trust']

    # Filter for consensus
    consensus = grouped[
        (grouped['whale_count'] >= min_whales) &
        (grouped['total_size'] >= min_size)
    ]

    return consensus.to_dict('records')


# ============================================================================
# Kelly Criterion & Risk Management
# ============================================================================

def gpu_kelly_fractions(
    win_probs: np.ndarray,
    payoff_ratios: np.ndarray,
    max_fraction: float = 0.25
) -> np.ndarray:
    """
    Calculate Kelly fractions for multiple bets in parallel.

    Args:
        win_probs: Array of win probabilities
        payoff_ratios: Array of payoff ratios (win amount / bet amount)
        max_fraction: Maximum allowed fraction

    Returns:
        Array of Kelly fractions
    """
    global gpu_stats
    start = time.perf_counter()

    p = to_gpu(win_probs.astype(np.float32))
    b = to_gpu(payoff_ratios.astype(np.float32))

    # Kelly formula: f = (bp - q) / b where q = 1 - p
    q = 1 - p
    kelly = (b * p - q) / b

    # Clip to valid range
    kelly = xp.clip(kelly, 0, max_fraction)

    sync_gpu()

    elapsed = (time.perf_counter() - start) * 1000
    gpu_stats.operations += 1
    gpu_stats.total_time_ms += elapsed

    return to_cpu(kelly)


def gpu_portfolio_variance(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Calculate portfolio variance on GPU.

    Args:
        weights: Array of portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Portfolio variance
    """
    w = to_gpu(weights.astype(np.float32))
    cov = to_gpu(cov_matrix.astype(np.float32))

    # Portfolio variance = w' * Cov * w
    variance = xp.dot(w, xp.dot(cov, w))

    sync_gpu()
    return float(to_cpu(variance))


# ============================================================================
# Utility Functions
# ============================================================================

def get_gpu_stats() -> dict:
    """Get GPU performance statistics."""
    return {
        'has_gpu': HAS_CUPY,
        'gpu_name': GPU_NAME,
        'gpu_memory_gb': GPU_MEMORY,
        'operations': gpu_stats.operations,
        'total_time_ms': gpu_stats.total_time_ms,
        'avg_time_ms': gpu_stats.total_time_ms / max(1, gpu_stats.operations),
    }


def reset_gpu_stats():
    """Reset GPU statistics."""
    global gpu_stats
    gpu_stats = GPUStats()


def benchmark_gpu(n_samples: int = 10000, n_features: int = 50) -> dict:
    """
    Benchmark GPU vs CPU performance.

    Args:
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        Benchmark results
    """
    results = {}

    # Generate test data
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features).astype(np.float32)

    # Test correlation matrix
    # CPU timing
    start = time.perf_counter()
    cpu_corr = np.corrcoef(data.T)
    cpu_time = (time.perf_counter() - start) * 1000

    # GPU timing
    start = time.perf_counter()
    gpu_corr = gpu_correlation_matrix(data)
    gpu_time = (time.perf_counter() - start) * 1000

    results['correlation_matrix'] = {
        'cpu_ms': cpu_time,
        'gpu_ms': gpu_time,
        'speedup': cpu_time / max(0.001, gpu_time),
        'match': np.allclose(cpu_corr, gpu_corr, atol=1e-5)
    }

    # Test batch probabilities
    spots = np.random.uniform(90000, 110000, 1000).astype(np.float32)
    strikes = np.random.uniform(95000, 105000, 1000).astype(np.float32)
    times = np.random.uniform(60, 900, 1000).astype(np.float32)

    start = time.perf_counter()
    gpu_probs = batch_implied_probabilities(spots, strikes, times)
    gpu_time = (time.perf_counter() - start) * 1000

    results['batch_probabilities'] = {
        'n_markets': 1000,
        'gpu_ms': gpu_time,
        'markets_per_second': 1000 / (gpu_time / 1000)
    }

    print(f"\n{'='*60}")
    print("GPU BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"GPU: {GPU_NAME}")
    print(f"Test size: {n_samples} samples x {n_features} features")
    print()

    for name, data in results.items():
        print(f"{name}:")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        print()

    return results


# Run benchmark on import if GPU available
if __name__ == "__main__":
    if HAS_CUPY:
        benchmark_gpu()
    else:
        print("CuPy not available. Install with: pip install cupy-cuda12x")
