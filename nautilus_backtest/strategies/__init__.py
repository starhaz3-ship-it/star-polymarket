from nautilus_backtest.strategies.alignment import Alignment
from nautilus_backtest.strategies.tdi_squeeze import TdiSqueeze
from nautilus_backtest.strategies.fisher_cascade import FisherCascade
from nautilus_backtest.strategies.volcano_breakout import VolcanoBreakout
from nautilus_backtest.strategies.wyckoff_vortex import WyckoffVortex
from nautilus_backtest.strategies.momentum_regime import MomentumRegime
from nautilus_backtest.strategies.mean_revert_extreme import MeanRevertExtreme
from nautilus_backtest.strategies.divergence_hunter import DivergenceHunter
from nautilus_backtest.strategies.exhaustion_bar import ExhaustionBar
from nautilus_backtest.strategies.vwap_reversion import VwapReversion
from nautilus_backtest.strategies.squeeze_fail import SqueezeFail
from nautilus_backtest.strategies.multi_extreme import MultiExtreme

ALL_STRATEGIES = [
    Alignment, TdiSqueeze, FisherCascade, VolcanoBreakout, WyckoffVortex,
    MomentumRegime, MeanRevertExtreme, DivergenceHunter,
    ExhaustionBar, VwapReversion, SqueezeFail, MultiExtreme,
]
