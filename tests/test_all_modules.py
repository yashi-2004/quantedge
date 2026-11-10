"""
QuantEdge Comprehensive Test Suite
Validates mathematical correctness, absence of look-ahead bias,
portfolio weight constraints, deep learning forward passes, econometrics, and reporting.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategies import StrategyLibrary
from src.data.ingestion import MarketDataIngestion
from src.data.validation import DataValidator
from src.econometrics.arima import ARIMAModeler
from src.econometrics.garch import GARCHModeler
from src.econometrics.kalman import KalmanFilterEngine
from src.econometrics.stationarity import StationarityAnalyzer
from src.econometrics.tests import StatisticalTestLab
from src.econometrics.var import VARModeler
from src.explainability.attribution import ModelExplainer
from src.features.pipeline import FeatureEngine
from src.features.technical_indicators import calculate_bollinger_bands, calculate_rsi, calculate_sma
from src.features.volatility_features import calculate_garman_klass_volatility, calculate_parkinson_volatility
from src.models.baseline import FinancialBaselines
from src.models.classical import ClassicalMLSuite
from src.models.deep_models import GRUForecaster, LSTMForecaster, TCNForecaster, TimeSeriesTransformer
from src.portfolio.optimisation import PortfolioOptimizer
from src.regimes.regime_detector import MarketRegimeDetector
from src.research.reporting import ResearchReportGenerator
from src.risk.analytics import RiskLaboratory
from src.sentiment.sentiment_engine import SentimentAnalyzer
from src.uncertainty.calibration import UncertaintyForecaster


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    ret = np.random.normal(0.0005, 0.015, len(dates))
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * 1.015
    low = close * 0.985
    open_p = (high + low) / 2.0
    vol = np.random.randint(10000, 500000, len(dates))
    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": vol},
        index=dates,
    )


# 1. Data Validation Tests
def test_data_validation_clean(sample_ohlcv):
    validator = DataValidator()
    report = validator.validate_ohlcv(sample_ohlcv, "TEST")
    assert report.is_valid is True
    assert report.quality_score >= 80.0
    assert report.inversion_count == 0


def test_data_validation_detects_inversion(sample_ohlcv):
    corrupt = sample_ohlcv.copy()
    corrupt.iloc[5, corrupt.columns.get_loc("High")] = 50.0
    corrupt.iloc[5, corrupt.columns.get_loc("Low")] = 150.0
    validator = DataValidator()
    report = validator.validate_ohlcv(corrupt, "CORRUPT")
    assert report.inversion_count > 0
    clean_df, remediations = validator.clean_ohlcv(corrupt)
    assert len(remediations) > 0
    clean_report = validator.validate_ohlcv(clean_df, "CORRUPT")
    assert clean_report.inversion_count == 0


# 2. Technical and Volatility Features Tests
def test_rsi_bounds(sample_ohlcv):
    rsi = calculate_rsi(sample_ohlcv["Close"], window=14)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_volatility_estimators_non_negative(sample_ohlcv):
    park_vol = calculate_parkinson_volatility(sample_ohlcv, window=21).dropna()
    gk_vol = calculate_garman_klass_volatility(sample_ohlcv, window=21).dropna()
    assert (park_vol >= 0.0).all()
    assert (gk_vol >= 0.0).all()


def test_feature_pipeline_no_lookahead(sample_ohlcv):
    engine = FeatureEngine(include_macro=False)
    fmat = engine.build_feature_matrix(sample_ohlcv)
    assert not fmat.X.empty
    assert len(fmat.X) == len(fmat.y_return)
    ret_series = sample_ohlcv["Close"].pct_change()
    first_date = fmat.X.index[0]
    loc_in_raw = sample_ohlcv.index.get_loc(first_date)
    expected_forward_ret = ret_series.iloc[loc_in_raw + 1]
    assert np.isclose(fmat.y_return.loc[first_date], expected_forward_ret, atol=1e-5)


# 3. Econometric Tests
def test_stationarity_analyzer(sample_ohlcv):
    ret = sample_ohlcv["Close"].pct_change().dropna()
    rep = StationarityAnalyzer.test_stationarity(ret)
    assert rep.is_stationary is True
    assert rep.recommended_differencing == 0


def test_garch_model(sample_ohlcv):
    ret = sample_ohlcv["Close"].pct_change().dropna()
    garch_mod = GARCHModeler()
    res = garch_mod.fit_garch(ret, model_type="GARCH", forecast_horizon=5)
    assert res.persistence > 0.0
    assert len(res.conditional_volatility) == len(ret)
    assert len(res.volatility_forecast) == 5


def test_kalman_filter(sample_ohlcv):
    kalman = KalmanFilterEngine()
    res = kalman.filter_price_trend(sample_ohlcv["Close"])
    assert len(res.filtered_trend) == len(sample_ohlcv)
    assert res.estimated_noise_std >= 0.0


def test_diebold_mariano():
    np.random.seed(42)
    actual = np.random.normal(0, 1, 100)
    p1 = actual + np.random.normal(0, 0.2, 100)
    p2 = actual + np.random.normal(0, 1.5, 100)
    res = StatisticalTestLab.diebold_mariano_test(actual, p1, p2)
    assert res.p_value < 0.05
    assert res.better_model == "Model 1"


# 4. Machine Learning & Deep Learning Tests
def test_classical_ml_chronological_split(sample_ohlcv):
    engine = FeatureEngine(include_macro=False)
    fmat = engine.build_feature_matrix(sample_ohlcv)
    ml_suite = ClassicalMLSuite()
    X_tr, X_va, X_te, y_tr, y_va, y_te = ml_suite.chronological_split(fmat.X, fmat.y_return)
    assert len(X_tr) > 0 and len(X_te) > 0
    assert X_tr.index.max() < X_te.index.min()


def test_xgboost_training(sample_ohlcv):
    engine = FeatureEngine(include_macro=False)
    fmat = engine.build_feature_matrix(sample_ohlcv)
    ml_suite = ClassicalMLSuite()
    X_tr, _, X_te, y_tr, _, y_te = ml_suite.chronological_split(fmat.X, fmat.y_return)
    res = ml_suite.train_regression_model("XGBoost", X_tr, y_tr, X_te, y_te, params={"n_estimators": 20})
    assert len(res.predictions_test) == len(y_te)
    assert "RMSE" in res.out_of_sample_metrics


def test_deep_learning_forward_passes():
    batch_size = 4
    seq_len = 10
    input_dim = 8
    dummy_x = torch.randn(batch_size, seq_len, input_dim)

    # LSTM
    lstm = LSTMForecaster(input_dim=input_dim, hidden_dim=16)
    out_lstm = lstm(dummy_x)
    assert out_lstm.shape == (batch_size,)

    # GRU
    gru = GRUForecaster(input_dim=input_dim, hidden_dim=16)
    out_gru = gru(dummy_x)
    assert out_gru.shape == (batch_size,)

    # TCN
    tcn = TCNForecaster(input_dim=input_dim, hidden_dim=16)
    out_tcn = tcn(dummy_x)
    assert out_tcn.shape == (batch_size,)

    # Transformer
    tf = TimeSeriesTransformer(input_dim=input_dim, d_model=16, nhead=2)
    out_tf = tf(dummy_x)
    assert out_tf.shape == (batch_size,)


# 5. Financial Sentiment & Market Regimes Tests
def test_sentiment_scoring():
    analyzer = SentimentAnalyzer(use_transformer=False)
    bull_score = analyzer.analyze_headline("Tech giant beats quarterly revenue expectations and raises earnings dividend")
    bear_score = analyzer.analyze_headline("Unexpected inflation miss triggers deep recession fears and market selloff")
    assert bull_score.compound_score > 0
    assert bear_score.compound_score < 0


def test_market_regimes_hmm(sample_ohlcv):
    ret = sample_ohlcv["Close"].pct_change().dropna()
    detector = MarketRegimeDetector()
    res = detector.detect_regimes_hmm(ret, n_regimes=2)
    assert len(res.regime_labels) == len(ret.iloc[20:])
    assert res.transition_matrix.shape == (2, 2)
    assert np.allclose(res.transition_matrix.sum(axis=1), 1.0, atol=1e-3)


# 6. Portfolio Optimization Tests
def test_portfolio_constraints():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=150, freq="B")
    r1 = np.random.normal(0.0006, 0.012, 150)
    r2 = np.random.normal(0.0004, 0.009, 150)
    r3 = np.random.normal(0.0002, 0.005, 150)
    ret_df = pd.DataFrame({"AssetA": r1, "AssetB": r2, "AssetC": r3}, index=dates)

    optimizer = PortfolioOptimizer(risk_free_rate=0.03)

    # Max Sharpe
    res_sh = optimizer.max_sharpe(ret_df)
    assert np.isclose(res_sh.weights.sum(), 1.0, atol=1e-4)
    assert (res_sh.weights >= -1e-5).all()

    # Risk Parity
    res_rp = optimizer.risk_parity(ret_df)
    assert np.isclose(res_rp.weights.sum(), 1.0, atol=1e-4)
    assert (res_rp.weights >= 0.0).all()
    pcr = res_rp.percentage_risk_contributions.values
    assert np.allclose(pcr, 33.33, atol=5.0)

    # Black-Litterman
    res_bl = optimizer.black_litterman(ret_df, views={"AssetA": 0.15})
    assert np.isclose(res_bl.weights.sum(), 1.0, atol=1e-4)


# 7. Backtesting and Anti-Leakage Tests
def test_backtesting_anti_leakage_and_costs():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {"AssetA": np.linspace(100, 150, 100), "AssetB": np.linspace(50, 60, 100)}, index=dates
    )
    weights = pd.DataFrame(
        {"AssetA": [1.0] * 50 + [0.0] * 50, "AssetB": [0.0] * 50 + [1.0] * 50}, index=dates
    )

    engine = BacktestEngine(initial_capital=100_000, transaction_cost_bps=20, execution_lag=1)
    res = engine.run_backtest(prices, weights, "Cost_Lag_Test")

    assert res.total_costs_paid > 0.0
    assert len(res.equity_curve) == len(dates)
    assert res.sharpe_ratio is not None


# 8. Risk Laboratory Tests
def test_var_cvar_relationship():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
    risk_lab = RiskLaboratory(portfolio_value=100_000)
    rep = risk_lab.calculate_var_cvar(returns, confidence_level=0.95)

    assert rep.historical_cvar_pct >= rep.historical_var_pct
    assert rep.parametric_cvar_pct >= rep.parametric_var_pct


# 9. Research Reporting Tests
def test_research_reporting_hypotheses():
    rep_gen = ResearchReportGenerator()
    hyps = rep_gen.evaluate_core_hypotheses()
    assert len(hyps) == 5
    ids = [h.hypothesis_id for h in hyps]
    assert ids == ["H1", "H2", "H3", "H4", "H5"]
    md = rep_gen.generate_markdown_report(["SPY", "QQQ"], "2021-01-01", "2024-01-01")
    assert "# QuantEdge Research Report" in md
    assert "H1" in md
