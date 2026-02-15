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
from nautilus_backtest.strategies.roc_extreme import RocExtreme
from nautilus_backtest.strategies.trix_cross import TrixCross
from nautilus_backtest.strategies.chaikin_mf import ChaikinMf
from nautilus_backtest.strategies.aroon_cross import AroonCross
from nautilus_backtest.strategies.kama_trend import KamaTrend
from nautilus_backtest.strategies.rsi_divergence import RsiDivergence
from nautilus_backtest.strategies.williams_vwap import WilliamsVwap
from nautilus_backtest.strategies.cci_bounce import CciBounce
from nautilus_backtest.strategies.stoch_bb import StochBb
from nautilus_backtest.strategies.elder_impulse import ElderImpulse
from nautilus_backtest.strategies.triple_rsi import TripleRsi
from nautilus_backtest.strategies.dpo_reversal import DpoReversal
from nautilus_backtest.strategies.ultimate_osc import UltimateOsc
from nautilus_backtest.strategies.double_bottom_rsi import DoubleBottomRsi
from nautilus_backtest.strategies.mass_index import MassIndex
from nautilus_backtest.strategies.ppo_momentum import PpoMomentum
from nautilus_backtest.strategies.adx_di_cross import AdxDiCross
from nautilus_backtest.strategies.pivot_bounce import PivotBounce
from nautilus_backtest.strategies.ichimoku_simple import IchimokuSimple
from nautilus_backtest.strategies.obv_divergence import ObvDivergence

ALL_STRATEGIES = [
    Alignment, TdiSqueeze, FisherCascade, VolcanoBreakout, WyckoffVortex,
    MomentumRegime, MeanRevertExtreme, DivergenceHunter,
    ExhaustionBar, VwapReversion, SqueezeFail, MultiExtreme,
    RocExtreme, TrixCross, ChaikinMf, AroonCross, KamaTrend,
    RsiDivergence, WilliamsVwap, CciBounce, StochBb, ElderImpulse,
    TripleRsi, DpoReversal, UltimateOsc, DoubleBottomRsi, MassIndex,
    PpoMomentum, AdxDiCross, PivotBounce, IchimokuSimple, ObvDivergence,
]
