from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_prices, prices_to_df
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union


class JimSimonsSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    alpha_factors: Dict[str, float] = Field(default_factory=dict)
    statistical_metrics: Dict[str, float] = Field(default_factory=dict)
    regime_classification: Optional[str] = None


class SimonsOutput(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    alpha_factors: Dict[str, float] = Field(default_factory=dict)
    statistical_metrics: Dict[str, float] = Field(default_factory=dict)
    regime_classification: Optional[str] = None


# Renaissance-inspired factor types
class FactorType:
    VALUE = "value"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    QUALITY = "quality"
    SENTIMENT = "sentiment"
    CARRY = "carry"
    SEASONALITY = "seasonality"


# Statistical significance levels
class SignificanceLevel:
    VERY_HIGH = 0.001
    HIGH = 0.01
    MEDIUM = 0.05
    LOW = 0.1


# Market regimes
class MarketRegime:
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITIONING = "transitioning"


def jim_simons_agent(state: AgentState):
    """
    Analyzes stocks using Jim Simons' advanced quantitative, data-driven approach.
    
    Implements Renaissance Technologies-inspired strategies including:
    - Statistical arbitrage
    - Signal processing and pattern recognition
    - Multi-factor models
    - Regime detection
    - Non-linear machine learning models
    """
    data = state["data"]
    end_date = data["end_date"]
    start_date = data["start_date"]
    tickers = data["tickers"]

    analysis_data = {}
    simons_analysis = {}
    
    # Detect market regime first
    progress.update_status("jim_simons_agent", "market", "Detecting market regime")
    try:
        market_regime = detect_market_regime(end_date)
    except Exception as e:
        print(f"Market regime detection failed: {e}")
        market_regime = MarketRegime.SIDEWAYS  # Default fallback

    # Pre-compute market factors across all tickers for cross-sectional analysis
    cross_sectional_data = {}
    
    for ticker in tickers:
        progress.update_status("jim_simons_agent", ticker, "Fetching historical data")
        try:
            # Get historical prices for advanced analysis
            lookback_days = 252  # Approximately 1 year of trading days
            lookback_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            prices = get_prices(ticker, lookback_date, end_date)
            df = prices_to_df(prices)
            
            if df is None or len(df) < 20:  # Need sufficient data points
                cross_sectional_data[ticker] = {"error": "Insufficient historical data"}
                continue
                
            # Calculate returns
            df['returns'] = df['close'].pct_change().fillna(0)
            
            # Store in cross-sectional data
            cross_sectional_data[ticker] = {
                "df": df,
                "mean_return": df['returns'].mean(),
                "volatility": df['returns'].std(),
                "sharpe": df['returns'].mean() / df['returns'].std() if df['returns'].std() > 0 else 0,
                "volume_trend": calculate_trend(df['volume']) if 'volume' in df.columns else 0
            }
        except Exception as e:
            print(f"Error processing historical data for {ticker}: {e}")
            cross_sectional_data[ticker] = {"error": str(e)}
    
    # Apply cross-sectional analysis if we have enough tickers
    if len([t for t in cross_sectional_data if "error" not in cross_sectional_data[t]]) >= 3:
        try:
            cross_sectional_analysis(cross_sectional_data)
        except Exception as e:
            print(f"Cross-sectional analysis failed: {e}")
    
    # Now process each ticker individually
    for ticker in tickers:
        if ticker in cross_sectional_data and "error" in cross_sectional_data[ticker]:
            progress.update_status("jim_simons_agent", ticker, "Error in historical data, skipping detailed analysis")
            # Still provide a basic analysis even with limited data
            simons_analysis[ticker] = {
                "signal": "neutral", 
                "confidence": 0.5, 
                "reasoning": "Insufficient historical data for detailed quantitative analysis. Using neutral stance with moderate confidence.",
                "alpha_factors": {},
                "statistical_metrics": {},
                "regime_classification": market_regime
            }
            continue
            
        progress.update_status("jim_simons_agent", ticker, "Fetching financial metrics")
        try:
            metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)
        except Exception as e:
            print(f"Error fetching financial metrics for {ticker}: {e}")
            metrics = []

        progress.update_status("jim_simons_agent", ticker, "Gathering financial line items")
        try:
            financial_line_items = search_line_items(
                ticker, 
                ["earnings_per_share", "revenue", "net_income", "book_value_per_share", 
                 "total_assets", "total_liabilities", "current_assets", "current_liabilities", 
                 "dividends_and_other_cash_distributions", "outstanding_shares",
                 "research_and_development", "cash_and_equivalents", "total_debt",
                 "operating_income", "gross_profit", "inventory"], 
                end_date, 
                period="annual", 
                limit=10
            )
        except Exception as e:
            print(f"Error fetching financial line items for {ticker}: {e}")
            financial_line_items = []

        progress.update_status("jim_simons_agent", ticker, "Getting market cap")
        try:
            market_cap = get_market_cap(ticker, end_date)
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            market_cap = 0

        # Get the cross-sectional data and dataframe
        cs_data = cross_sectional_data.get(ticker, {})
        df = cs_data.get("df") if cs_data else None

        # Perform sub-analyses
        progress.update_status("jim_simons_agent", ticker, "Running statistical analysis")
        statistical_analysis = analyze_statistical_patterns(df) if df is not None else {"error": "No historical data"}
        
        progress.update_status("jim_simons_agent", ticker, "Computing factor model")
        factor_analysis = apply_factor_model(ticker, df, metrics, financial_line_items, market_cap) if df is not None else {"error": "No historical data"}
        
        progress.update_status("jim_simons_agent", ticker, "Detecting trading signals")
        signal_analysis = detect_trading_signals(df, market_regime) if df is not None else {"error": "No historical data"}
        
        progress.update_status("jim_simons_agent", ticker, "Computing market anomalies")
        anomaly_analysis = detect_market_anomalies(ticker, df, cross_sectional_data) if df is not None else {"error": "No historical data"}
        
        # Aggregate all analyses and compute total score using Renaissance-inspired weighting
        total_score = aggregate_analysis_score(
            statistical_analysis=statistical_analysis,
            factor_analysis=factor_analysis,
            signal_analysis=signal_analysis,
            anomaly_analysis=anomaly_analysis,
            market_regime=market_regime
        )
        
        # Collect alpha factors from all analyses
        alpha_factors = {}
        for analysis in [statistical_analysis, factor_analysis, signal_analysis, anomaly_analysis]:
            if isinstance(analysis, dict) and "alpha_factors" in analysis:
                alpha_factors.update(analysis["alpha_factors"])
        
        # Statistical metrics from all analyses
        statistical_metrics = {}
        for analysis in [statistical_analysis, factor_analysis, signal_analysis, anomaly_analysis]:
            if isinstance(analysis, dict) and "metrics" in analysis:
                statistical_metrics.update(analysis["metrics"])  

        # Map total_score to signal (custom sigmoid function to capture Renaissance's nuanced approach)
        signal = "neutral"  # Default
        if total_score > 0.65:
            signal = "bullish"
        elif total_score < 0.35:
            signal = "bearish"
            
        # Adjust signal based on market regime
        signal = adjust_signal_for_regime(signal, total_score, market_regime)
        
        # Confidence based on statistical significance and signal strength
        confidence = calculate_confidence(statistical_metrics, total_score, signal)
            
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "confidence": confidence,
            "market_regime": market_regime,
            "statistical_analysis": statistical_analysis,
            "factor_analysis": factor_analysis,
            "signal_analysis": signal_analysis,
            "anomaly_analysis": anomaly_analysis,
            "alpha_factors": alpha_factors,
            "statistical_metrics": statistical_metrics
        }

        progress.update_status("jim_simons_agent", ticker, "Generating Renaissance-style analysis")
        try:
            simons_output = generate_simons_output(
                ticker=ticker,
                analysis_data=analysis_data,
                model_name=state["metadata"]["model_name"],
                model_provider=state["metadata"]["model_provider"],
                market_regime=market_regime
            )

            simons_analysis[ticker] = {
                "signal": simons_output.signal, 
                "confidence": simons_output.confidence, 
                "reasoning": simons_output.reasoning,
                "alpha_factors": simons_output.alpha_factors,
                "statistical_metrics": simons_output.statistical_metrics,
                "regime_classification": simons_output.regime_classification
            }
        except Exception as e:
            print(f"Error generating Renaissance-style analysis for {ticker}: {e}")
            # Provide a fallback analysis
            simons_analysis[ticker] = {
                "signal": signal, 
                "confidence": confidence, 
                "reasoning": f"Quantitative analysis indicates a {signal} stance with {confidence:.1f} confidence based on statistical patterns and factor models.",
                "alpha_factors": alpha_factors,
                "statistical_metrics": statistical_metrics,
                "regime_classification": market_regime
            }

        progress.update_status("jim_simons_agent", ticker, "Done")

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(simons_analysis), name="jim_simons_agent")

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(simons_analysis, "Jim Simons Agent")

    # Store signals in the overall state
    state["data"]["analyst_signals"]["jim_simons_agent"] = simons_analysis

    return {"messages": [message], "data": state["data"]}


def cross_sectional_analysis(cross_sectional_data: Dict[str, Dict]):
    """
    Performs cross-sectional analysis on multiple tickers.
    
    Implements Renaissance-inspired cross-sectional analysis including:
    - Relative strength
    - Pair trading opportunities
    - Statistical factor analysis with PCA
    - Clustering for regime identification
    """
    try:
        # Extract relevant metrics for analysis
        tickers = []
        metrics = []
        
        for ticker, data in cross_sectional_data.items():
            if "error" in data or "df" not in data:
                continue
                
            tickers.append(ticker)
            metrics.append([
                data.get("mean_return", 0),
                data.get("volatility", 0),
                data.get("sharpe", 0),
                data.get("volume_trend", 0)
            ])
        
        if len(tickers) < 3:  # Need at least 3 stocks for meaningful analysis
            return
            
        # Convert to numpy array for analysis
        metrics_array = np.array(metrics)
        
        # 1. Standardize the data
        scaler = StandardScaler()
        standardized_metrics = scaler.fit_transform(metrics_array)
        
        # 2. Principal Component Analysis
        pca = PCA(n_components=min(len(metrics_array[0]), len(tickers)-1))
        pca_result = pca.fit_transform(standardized_metrics)
        
        # Extract factor loadings (how much each metric contributes to each principal component)
        factor_loadings = pca.components_
        explained_variance = pca.explained_variance_ratio_
        
        # 3. Clustering for regime identification
        kmeans = KMeans(n_clusters=min(3, len(tickers)), random_state=42)
        clusters = kmeans.fit_predict(standardized_metrics)
        
        # 4. Store results back into cross_sectional_data
        for i, ticker in enumerate(tickers):
            if i < len(pca_result):
                cross_sectional_data[ticker]["pca_factors"] = pca_result[i].tolist()
                cross_sectional_data[ticker]["cluster"] = int(clusters[i])
                
                # Calculate relative strength (z-score) for key metrics
                rs_returns = (metrics_array[i, 0] - np.mean(metrics_array[:, 0])) / np.std(metrics_array[:, 0]) if np.std(metrics_array[:, 0]) > 0 else 0
                rs_volatility = (metrics_array[i, 1] - np.mean(metrics_array[:, 1])) / np.std(metrics_array[:, 1]) if np.std(metrics_array[:, 1]) > 0 else 0
                rs_sharpe = (metrics_array[i, 2] - np.mean(metrics_array[:, 2])) / np.std(metrics_array[:, 2]) if np.std(metrics_array[:, 2]) > 0 else 0
                
                cross_sectional_data[ticker]["relative_strength"] = {
                    "returns": rs_returns,
                    "volatility": rs_volatility,
                    "sharpe": rs_sharpe
                }
                
        # 5. Calculate pair correlations for potential pair trading
        for i, ticker1 in enumerate(tickers):
            correlations = {}
            for j, ticker2 in enumerate(tickers):
                if i != j:
                    # Get the dataframes
                    df1 = cross_sectional_data[ticker1]["df"]["returns"]
                    df2 = cross_sectional_data[ticker2]["df"]["returns"]
                    
                    # Calculate correlation
                    correlation = df1.corr(df2)
                    correlations[ticker2] = correlation
            
            # Store the correlations
            cross_sectional_data[ticker1]["correlations"] = correlations
            
            # Find potential pairs for statistical arbitrage
            # High correlation but different relative strength suggests potential reversion
            potential_pairs = []
            for ticker2, corr in correlations.items():
                if abs(corr) > 0.7:  # High correlation threshold
                    rs_diff = abs(cross_sectional_data[ticker1]["relative_strength"]["returns"] - 
                                 cross_sectional_data[ticker2]["relative_strength"]["returns"])
                    if rs_diff > 1.0:  # Significant difference in relative strength
                        potential_pairs.append((ticker2, corr, rs_diff))
            
            # Sort by correlation strength
            potential_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            cross_sectional_data[ticker1]["potential_pairs"] = potential_pairs[:3]  # Top 3 pairs
                
    except Exception as e:
        print(f"Error in cross-sectional analysis: {e}")


def detect_market_regime(end_date: str) -> str:
    """
    Detects the current market regime using multiple indicators.
    
    Implements Renaissance-inspired regime detection looking at:
    - Trend strength
    - Volatility regimes 
    - Mean reversion tendencies
    - Correlation structure changes
    """
    try:
        # Use SPY as proxy for market
        lookback_days = 252  # 1 year
        lookback_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        market_prices = get_prices("SPY", lookback_date, end_date)
        df = prices_to_df(market_prices)
        
        if not df or len(df) < 20:
            return MarketRegime.SIDEWAYS  # Default when insufficient data
            
        # Calculate returns
        df['returns'] = df['close'].pct_change().fillna(0)
        
        # 1. Trend detection
        short_ma = df['close'].rolling(window=20).mean().iloc[-1]  # 20-day MA
        medium_ma = df['close'].rolling(window=50).mean().iloc[-1]  # 50-day MA
        long_ma = df['close'].rolling(window=200).mean().iloc[-1]  # 200-day MA
        current_price = df['close'].iloc[-1]
        
        # 2. Volatility regime
        recent_vol = df['returns'].iloc[-20:].std() * math.sqrt(252)  # Annualized
        historical_vol = df['returns'].std() * math.sqrt(252)  # Annualized
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # 3. Mean reversion vs momentum
        # Calculate autocorrelation to detect mean reversion (negative) or momentum (positive)
        if len(df) >= 30:
            autocorr = acf(df['returns'].iloc[-30:], nlags=1)[1]  # First-order autocorrelation
        else:
            autocorr = 0
            
        # Combine signals to determine regime
        if current_price > short_ma > medium_ma > long_ma and vol_ratio < 1.2:
            regime = MarketRegime.BULL_TREND
        elif current_price < short_ma < medium_ma < long_ma and vol_ratio < 1.2:
            regime = MarketRegime.BEAR_TREND
        elif abs(current_price/medium_ma - 1) < 0.03 and vol_ratio < 0.8:
            regime = MarketRegime.SIDEWAYS
        elif vol_ratio > 1.5:
            regime = MarketRegime.HIGH_VOLATILITY
        elif vol_ratio < 0.7:
            regime = MarketRegime.LOW_VOLATILITY
        else:
            regime = MarketRegime.TRANSITIONING
            
        return regime
    except Exception as e:
        print(f"Market regime detection error: {e}")
        return MarketRegime.SIDEWAYS  # Default fallback


def analyze_statistical_patterns(df: pd.DataFrame) -> dict:
    """
    Analyzes statistical patterns in the historical data.
    
    Implements Renaissance-inspired statistical analysis including:
    - Mean reversion
    - Momentum
    - Volatility clustering
    - Autocorrelation
    """
    analysis = {}
    
    if df is None or len(df) < 20:
        return {"error": "Insufficient data for statistical analysis"}
    
    # 1. Mean reversion
    try:
        # Augmented Dickey-Fuller test for stationarity
        adf_result = adfuller(df['close'])
        analysis["mean_reversion"] = adf_result[1] < SignificanceLevel.MEDIUM
    except Exception as e:
        print(f"Mean reversion analysis failed: {e}")
        analysis["mean_reversion"] = False
    
    # 2. Momentum
    try:
        # Calculate momentum using returns
        momentum = df['returns'].mean()
        analysis["momentum"] = momentum > 0
    except Exception as e:
        print(f"Momentum analysis failed: {e}")
        analysis["momentum"] = False
    
    # 3. Volatility clustering
    try:
        # Use a simpler approach if statsmodels GARCH is not available
        returns = df['returns'].dropna()
        if len(returns) >= 20:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=10).std()
            # Check if recent volatility is clustering (higher than average)
            recent_vol = rolling_vol.iloc[-5:].mean()
            avg_vol = rolling_vol.mean()
            analysis["volatility_clustering"] = recent_vol > avg_vol
        else:
            analysis["volatility_clustering"] = False
    except Exception as e:
        print(f"Volatility clustering analysis failed: {e}")
        analysis["volatility_clustering"] = False
    
    # 4. Autocorrelation
    try:
        # Calculate autocorrelation to detect mean reversion (negative) or momentum (positive)
        returns = df['returns'].dropna()
        if len(returns) >= 20:
            # Calculate autocorrelation manually if statsmodels is not available
            autocorr = returns.autocorr(lag=1)
            analysis["autocorrelation"] = autocorr < 0
        else:
            analysis["autocorrelation"] = False
    except Exception as e:
        print(f"Autocorrelation analysis failed: {e}")
        analysis["autocorrelation"] = False
    
    # Add alpha factors and metrics
    analysis["alpha_factors"] = {
        "mean_reversion_factor": 1 if analysis.get("mean_reversion", False) else 0,
        "momentum_factor": 1 if analysis.get("momentum", False) else 0,
        "volatility_clustering_factor": 1 if analysis.get("volatility_clustering", False) else 0,
        "autocorrelation_factor": 1 if analysis.get("autocorrelation", False) else 0
    }
    
    analysis["metrics"] = {
        "momentum_strength": df['returns'].mean() if 'returns' in df else 0,
        "volatility": df['returns'].std() if 'returns' in df else 0,
        "sharpe": df['returns'].mean() / df['returns'].std() if 'returns' in df and df['returns'].std() > 0 else 0
    }
    
    return analysis


def apply_factor_model(ticker: str, df: pd.DataFrame, metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """
    Applies a factor model to the historical data.
    
    Implements Renaissance-inspired factor models including:
    - Value
    - Momentum
    - Size
    - Volatility
    """
    analysis = {}
    
    if df is None or len(df) < 20:
        return {"error": "Insufficient data for factor model"}
    
    # 1. Value factor
    try:
        # Calculate price-to-book ratio if available
        if metrics and len(metrics) > 0 and hasattr(metrics[0], 'price_to_book'):
            pb_ratio = metrics[0].price_to_book
            analysis["value_factor"] = pb_ratio < 1.0
        else:
            analysis["value_factor"] = False
    except Exception as e:
        print(f"Value factor analysis failed: {e}")
        analysis["value_factor"] = False
    
    # 2. Momentum factor
    try:
        # Calculate momentum using returns
        momentum = df['returns'].mean()
        analysis["momentum_factor"] = momentum > 0
    except Exception as e:
        print(f"Momentum factor analysis failed: {e}")
        analysis["momentum_factor"] = False
    
    # 3. Size factor
    try:
        # Calculate market capitalization
        analysis["size_factor"] = market_cap < 1000000000  # $1 billion
    except Exception as e:
        print(f"Size factor analysis failed: {e}")
        analysis["size_factor"] = False
    
    # 4. Volatility factor
    try:
        # Calculate volatility using returns
        volatility = df['returns'].std()
        analysis["volatility_factor"] = volatility < 0.1
    except Exception as e:
        print(f"Volatility factor analysis failed: {e}")
        analysis["volatility_factor"] = False
    
    # Add alpha factors and metrics
    analysis["alpha_factors"] = {
        "value_factor": 1 if analysis.get("value_factor", False) else 0,
        "momentum_factor": 1 if analysis.get("momentum_factor", False) else 0,
        "size_factor": 1 if analysis.get("size_factor", False) else 0,
        "volatility_factor": 1 if analysis.get("volatility_factor", False) else 0
    }
    
    analysis["metrics"] = {
        "price_to_book": metrics[0].price_to_book if metrics and len(metrics) > 0 and hasattr(metrics[0], 'price_to_book') else 0,
        "market_cap": market_cap
    }
    
    return analysis


def detect_trading_signals(df: pd.DataFrame, market_regime: str) -> dict:
    """
    Detects trading signals based on the historical data and market regime.
    
    Implements Renaissance-inspired trading signals including:
    - Mean reversion
    - Momentum
    - Statistical arbitrage
    """
    analysis = {}
    
    if df is None or len(df) < 20:
        return {"error": "Insufficient data for trading signals"}
    
    # 1. Mean reversion signal
    try:
        # Calculate Bollinger Bands manually
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        
        # Check if price is below lower band (potential buy signal)
        analysis["mean_reversion_signal"] = df['close'].iloc[-1] < lower_band.iloc[-1]
    except Exception as e:
        print(f"Mean reversion signal detection failed: {e}")
        analysis["mean_reversion_signal"] = False
    
    # 2. Momentum signal
    try:
        # Calculate momentum using returns
        momentum = df['returns'].mean()
        analysis["momentum_signal"] = momentum > 0
    except Exception as e:
        print(f"Momentum signal detection failed: {e}")
        analysis["momentum_signal"] = False
    
    # 3. Statistical arbitrage signal
    try:
        # Calculate statistical arbitrage using residuals
        residuals = df['close'] - df['close'].rolling(window=20).mean()
        analysis["statistical_arbitrage_signal"] = residuals.iloc[-1] > 0
    except Exception as e:
        print(f"Statistical arbitrage signal detection failed: {e}")
        analysis["statistical_arbitrage_signal"] = False
    
    # Add alpha factors and metrics
    analysis["alpha_factors"] = {
        "mean_reversion_signal": 1 if analysis.get("mean_reversion_signal", False) else 0,
        "momentum_signal": 1 if analysis.get("momentum_signal", False) else 0,
        "statistical_arbitrage_signal": 1 if analysis.get("statistical_arbitrage_signal", False) else 0
    }
    
    analysis["metrics"] = {
        "price_deviation": (df['close'].iloc[-1] / rolling_mean.iloc[-1] - 1) if 'close' in df else 0,
        "momentum_strength": df['returns'].mean() if 'returns' in df else 0
    }
    
    return analysis


def detect_market_anomalies(ticker: str, df: pd.DataFrame, cross_sectional_data: Dict[str, Dict]) -> Dict:
    """
    Detects market anomalies and inefficiencies.
    
    Implements Renaissance-inspired anomaly detection including:
    - Earnings surprises
    - Revenue surprises
    - Price momentum anomalies
    - Cross-sectional anomalies
    """
    if df is None or len(df) < 20:
        return {"error": "Insufficient data for anomaly detection"}
        
    anomalies = {}
    alpha_factors = {}
    metrics = {}
    
    # 1. Earnings surprise anomaly
    try:
        # Check if we have financial_line_items in the scope
        if 'financial_line_items' in locals() or 'financial_line_items' in globals():
            earnings_data = [item for item in financial_line_items if item.line_item_name == "earnings_per_share"]
            if len(earnings_data) >= 2:
                current_eps = earnings_data[0].value
                previous_eps = earnings_data[1].value
                earnings_surprise = (current_eps - previous_eps) / abs(previous_eps) if previous_eps != 0 else 0
                
                # Significant earnings surprise
                if abs(earnings_surprise) > 0.1:  # 10% surprise
                    anomalies["earnings_surprise"] = earnings_surprise
                    alpha_factors["earnings_surprise_anomaly"] = 1 if earnings_surprise > 0 else -1
                else:
                    alpha_factors["earnings_surprise_anomaly"] = 0
            else:
                alpha_factors["earnings_surprise_anomaly"] = 0
        else:
            alpha_factors["earnings_surprise_anomaly"] = 0
    except Exception as e:
        print(f"Earnings surprise anomaly detection failed: {e}")
        alpha_factors["earnings_surprise_anomaly"] = 0
    
    # 2. Revenue surprise anomaly
    try:
        # Check if we have financial_line_items in the scope
        if 'financial_line_items' in locals() or 'financial_line_items' in globals():
            revenue_data = [item for item in financial_line_items if item.line_item_name == "revenue"]
            if len(revenue_data) >= 2:
                current_revenue = revenue_data[0].value
                previous_revenue = revenue_data[1].value
                revenue_surprise = (current_revenue - previous_revenue) / abs(previous_revenue) if previous_revenue != 0 else 0
                
                # Significant revenue surprise
                if abs(revenue_surprise) > 0.1:  # 10% surprise
                    anomalies["revenue_surprise"] = revenue_surprise
                    alpha_factors["revenue_surprise_anomaly"] = 1 if revenue_surprise > 0 else -1
                else:
                    alpha_factors["revenue_surprise_anomaly"] = 0
            else:
                alpha_factors["revenue_surprise_anomaly"] = 0
        else:
            alpha_factors["revenue_surprise_anomaly"] = 0
    except Exception as e:
        print(f"Revenue surprise anomaly detection failed: {e}")
        alpha_factors["revenue_surprise_anomaly"] = 0
    
    # 3. Price momentum anomaly
    try:
        # Calculate 1-month momentum
        momentum_period = min(20, len(df) - 1)  # Approximately 1 month of trading days
        price_momentum = df['close'].pct_change(momentum_period).iloc[-1]
        metrics["price_momentum"] = float(price_momentum)  # Convert numpy types to Python native types
        
        # Significant price momentum
        if abs(price_momentum) > 0.05:  # 5% momentum
            anomalies["price_momentum"] = price_momentum
            alpha_factors["price_momentum_anomaly"] = 1 if price_momentum > 0 else -1
        else:
            alpha_factors["price_momentum_anomaly"] = 0
    except Exception as e:
        print(f"Price momentum anomaly detection failed: {e}")
        alpha_factors["price_momentum_anomaly"] = 0
    
    # 4. Cross-sectional anomalies (if available)
    if ticker in cross_sectional_data and "potential_pairs" in cross_sectional_data[ticker]:
        potential_pairs = cross_sectional_data[ticker]["potential_pairs"]
        if potential_pairs:
            anomalies["statistical_arbitrage_pairs"] = potential_pairs
    
    return {
        "anomalies": anomalies,
        "alpha_factors": alpha_factors,
        "metrics": metrics
    }


def aggregate_analysis_score(
    statistical_analysis: dict,
    factor_analysis: dict,
    signal_analysis: dict,
    anomaly_analysis: dict,
    market_regime: str
) -> float:
    """
    Aggregates the analysis scores using a weighted average.
    
    Implements Renaissance-inspired weighting including:
    - Statistical analysis (30%)
    - Factor analysis (20%)
    - Signal analysis (20%)
    - Anomaly analysis (30%)
    """
    score = 0
    
    # Statistical analysis
    score += 0.3 * sum([1 if value else 0 for value in statistical_analysis.values()])
    
    # Factor analysis
    score += 0.2 * sum([1 if value else 0 for value in factor_analysis.values()])
    
    # Signal analysis
    score += 0.2 * sum([1 if value else 0 for value in signal_analysis.values()])
    
    # Anomaly analysis
    score += 0.3 * sum([1 if value else 0 for value in anomaly_analysis.values()])
    
    # Adjust score based on market regime
    if market_regime == MarketRegime.BULL_TREND:
        score *= 1.1
    elif market_regime == MarketRegime.BEAR_TREND:
        score *= 0.9
    
    return score


def adjust_signal_for_regime(signal: str, total_score: float, market_regime: str) -> str:
    """
    Adjusts the signal based on the market regime.
    
    Implements Renaissance-inspired regime-based signal adjustment including:
    - Bull trend: Increase signal strength
    - Bear trend: Decrease signal strength
    """
    if market_regime == MarketRegime.BULL_TREND:
        if signal == "bullish":
            return "strong_bullish"
        elif signal == "neutral":
            return "bullish"
    elif market_regime == MarketRegime.BEAR_TREND:
        if signal == "bearish":
            return "strong_bearish"
        elif signal == "neutral":
            return "bearish"
    
    return signal


def calculate_confidence(statistical_metrics: dict, total_score: float, signal: str) -> float:
    """
    Calculates the confidence level based on the statistical metrics and signal strength.
    
    Implements Renaissance-inspired confidence calculation including:
    - Statistical significance
    - Signal strength
    """
    confidence = 0
    
    # Statistical significance
    confidence += 0.5 * sum([1 if value < SignificanceLevel.MEDIUM else 0 for value in statistical_metrics.values()])
    
    # Signal strength
    if signal == "strong_bullish" or signal == "strong_bearish":
        confidence += 0.5
    elif signal == "bullish" or signal == "bearish":
        confidence += 0.3
    
    return confidence


def generate_simons_output(
    ticker: str,
    analysis_data: Dict,
    model_name: str,
    model_provider: str,
    market_regime: str
) -> SimonsOutput:
    """
    Generates Renaissance-style analysis output for a ticker.
    """
    ticker_data = analysis_data.get(ticker, {})
    
    # Collect all alpha factors and metrics
    alpha_factors = {}
    statistical_metrics = {}
    
    for analysis_type in ["statistical_analysis", "factor_analysis", "signal_analysis", "anomaly_analysis"]:
        if analysis_type in ticker_data:
            analysis = ticker_data[analysis_type]
            if isinstance(analysis, dict):
                if "alpha_factors" in analysis:
                    for k, v in analysis["alpha_factors"].items():
                        # Convert numpy types to Python native types
                        if hasattr(v, "item"):
                            alpha_factors[k] = v.item()
                        else:
                            alpha_factors[k] = v
                            
                if "metrics" in analysis:
                    for k, v in analysis["metrics"].items():
                        # Convert numpy types to Python native types
                        if hasattr(v, "item"):
                            statistical_metrics[k] = v.item()
                        else:
                            statistical_metrics[k] = v
    
    # Extract signal and confidence
    signal = ticker_data.get("signal", "neutral")
    confidence = ticker_data.get("confidence", 0.5)
    
    # Generate reasoning based on the data
    reasoning = f"Quantitative analysis indicates a {signal} stance with {confidence:.1f} confidence based on statistical patterns and factor models."
    
    # Create output
    return SimonsOutput(
        signal=signal,
        confidence=confidence,
        reasoning=reasoning,
        alpha_factors=alpha_factors,
        statistical_metrics=statistical_metrics,
        regime_classification=market_regime
    )


def calculate_trend(series: pd.Series) -> float:
    """
    Calculates the trend strength of a time series using linear regression.
    
    Returns:
    - Positive value: Uptrend (stronger when larger)
    - Negative value: Downtrend (stronger when more negative)
    - Near zero: No significant trend
    """
    try:
        if len(series) < 5:
            return 0
            
        # Create time index
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(y) < 5:
            return 0
            
        # Apply Savitzky-Golay filter to smooth the data
        window_length = min(len(y), 11)  # Must be odd
        if window_length % 2 == 0:
            window_length -= 1
            
        if window_length >= 5:  # Minimum window length for filter
            try:
                y_smooth = savgol_filter(y, window_length, 3)
            except Exception:
                y_smooth = y  # Fall back to original data if filtering fails
        else:
            y_smooth = y
            
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_smooth)
        
        # Calculate trend strength as slope normalized by the mean of y
        trend_strength = slope * len(x) / np.mean(np.abs(y_smooth)) if np.mean(np.abs(y_smooth)) > 0 else 0
        
        return trend_strength
    except Exception as e:
        print(f"Error calculating trend: {e}")
        return 0


def create_default_jim_simons_signal():
    """
    Creates a default Jim Simons signal when analysis fails.
    """
    return JimSimonsSignal(
        signal="neutral", 
        confidence=0.0, 
        reasoning="Error in generating Renaissance Technologies analysis; defaulting to neutral.",
        alpha_factors={},
        statistical_metrics={},
        regime_classification=MarketRegime.SIDEWAYS
    )
