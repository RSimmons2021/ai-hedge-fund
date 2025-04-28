"""Crypto Analyst Agent - Specialized for cryptocurrency analysis."""

import json
import pandas as pd
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List

from graph.state import AgentState, show_agent_reasoning
from tools.api import get_price_data, is_crypto_pair
from utils.progress import progress
from utils.llm import call_llm
from pydantic import BaseModel, Field
from typing import Literal


class CryptoTradingSignal(BaseModel):
    """Output schema for crypto trading signals"""
    signal: Literal["buy", "sell", "hold"] = Field(description="Trading action to take")
    confidence: float = Field(description="Confidence in the recommendation (0-100)")
    reasoning: str = Field(description="Detailed reasoning for the recommendation")


def generate_crypto_trading_signal(ticker: str, technical_data: Dict[str, Any], model_name: str, model_provider: str) -> CryptoTradingSignal:
    """Generate a trading signal for a cryptocurrency based on technical analysis.
    
    Args:
        ticker: The cryptocurrency ticker symbol
        technical_data: Dictionary of technical indicators and price data
        model_name: The name of the LLM model to use
        model_provider: The provider of the LLM model
        
    Returns:
        CryptoTradingSignal object with signal, confidence, and reasoning
    """
    # Create a prompt template for the crypto analyst
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a cryptocurrency analyst specializing in technical analysis and market trends.
                Your task is to analyze the provided cryptocurrency data and generate trading signals.
                
                Based on the technical indicators, price movements, and volume data, determine whether to:
                - BUY: Strong bullish signals indicate potential upward movement
                - SELL: Strong bearish signals indicate potential downward movement
                - HOLD: Mixed or unclear signals
                
                Provide a confidence score (0-100) and detailed reasoning for your recommendation.
                Your response must be in a valid JSON format with the following fields:
                {"signal": "buy|sell|hold", "confidence": 0-100, "reasoning": "your detailed analysis"}
                """,
            ),
            (
                "human",
                f"""Please analyze the following cryptocurrency data for {ticker}:
                
                Current Price: ${technical_data.get('price', 'N/A')}
                
                Price Changes:
                - 24h Change: {technical_data.get('day_change_pct', 'N/A')}%
                - 7d Change: {technical_data.get('week_change_pct', 'N/A')}%
                - 30d Change: {technical_data.get('month_change_pct', 'N/A')}%
                
                Technical Indicators:
                - SMA (20-day): ${technical_data.get('sma_20', 'N/A')}
                - SMA (50-day): ${technical_data.get('sma_50', 'N/A')}
                - SMA (200-day): ${technical_data.get('sma_200', 'N/A')}
                - RSI (14-day): {technical_data.get('rsi', 'N/A')}
                - MACD: {technical_data.get('macd', 'N/A')}
                - MACD Signal: {technical_data.get('macd_signal', 'N/A')}
                - Bollinger Bands: Upper=${technical_data.get('bb_upper', 'N/A')}, Middle=${technical_data.get('bb_middle', 'N/A')}, Lower=${technical_data.get('bb_lower', 'N/A')}
                - Volatility (20-day): {technical_data.get('volatility', 'N/A')}
                
                {volume_section}
                
                Based on this data, what is your trading recommendation (BUY, SELL, or HOLD)?
                Provide your confidence level (0-100) and detailed reasoning.
                
                Remember to format your response as valid JSON with the fields: signal, confidence, and reasoning.
                """.format(
                    volume_section="Volume Data:\n- Current Volume: {volume}\n- Volume SMA (20-day): {volume_sma_20}\n- Volume Ratio: {volume_ratio}".format(
                        volume=technical_data.get('volume', 'N/A'),
                        volume_sma_20=technical_data.get('volume_sma_20', 'N/A'),
                        volume_ratio=technical_data.get('volume_ratio', 'N/A')
                    ) if 'volume' in technical_data else "Volume Data: Not Available"
                ),
            ),
        ]
    )
    
    # Create message for the LLM
    message = HumanMessage(content=template.format_messages()[1].content)
    
    # Call the LLM with retry logic
    try:
        # Call the LLM
        response = call_llm(
            prompt=message.content,
            model_name=model_name,
            model_provider=model_provider,
            pydantic_model=CryptoTradingSignal,
            agent_name="crypto_analyst_agent",
            max_retries=3
        )
        
        return response
    except Exception as e:
        print(f"Error generating crypto trading signal for {ticker}: {e}")
        # Return a neutral signal in case of error
        return CryptoTradingSignal(
            signal="hold",
            confidence=0,
            reasoning=f"Error generating signal: {str(e)}"
        )


def crypto_analyst_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Crypto Analyst Agent - Specialized for cryptocurrency analysis.
    
    This agent analyzes cryptocurrency price data using technical indicators
    and market trends without relying on traditional financial metrics.
    """
    progress.update_status("crypto_analyst_agent", None, "Analyzing cryptocurrency data")
    
    # Get tickers and date range from state
    tickers = state["data"]["tickers"]
    start_date = state["start_date"]
    end_date = state["end_date"]
    
    # Initialize signals structure if not present
    if "signals" not in state["data"]:
        state["data"]["signals"] = {}
    
    # Initialize analyst_signals structure for backward compatibility
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    
    if "crypto_analyst" not in state["data"]["analyst_signals"]:
        state["data"]["analyst_signals"]["crypto_analyst"] = {}
    
    # Process each crypto ticker
    for ticker in tickers:
        progress.update_status("crypto_analyst_agent", ticker, "Analyzing")
        
        # Skip if not a crypto pair
        if not is_crypto_pair(ticker):
            progress.update_status("crypto_analyst_agent", ticker, "Skipped (not a crypto asset)")
            continue
        
        # Initialize ticker in signals structure if not present
        if ticker not in state["data"]["signals"]:
            state["data"]["signals"][ticker] = {}
        
        # Get price data
        try:
            progress.update_status("crypto_analyst_agent", ticker, "Fetching price data")
            prices_df = get_price_data(ticker, start_date, end_date)
            if prices_df.empty:
                progress.update_status("crypto_analyst_agent", ticker, "No price data available")
                # Store the error in the signals
                state["data"]["signals"][ticker]["crypto_analyst_agent"] = {
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": "No price data available for analysis",
                    "timestamp": datetime.now().isoformat(),
                    "error": "No price data available"
                }
                # Also update the legacy structure
                state["data"]["analyst_signals"]["crypto_analyst"][ticker] = {
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": "No price data available for analysis"
                }
                continue
        except Exception as e:
            progress.update_status("crypto_analyst_agent", ticker, f"Error fetching price data: {e}")
            # Store the error in the signals
            state["data"]["signals"][ticker]["crypto_analyst_agent"] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error fetching price data: {e}",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            # Also update the legacy structure
            state["data"]["analyst_signals"]["crypto_analyst"][ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error fetching price data: {e}"
            }
            continue
        
        # Calculate technical indicators
        try:
            progress.update_status("crypto_analyst_agent", ticker, "Calculating technical indicators")
            
            # Add technical indicators to the dataframe
            # 1. Moving Averages
            prices_df['SMA_20'] = prices_df['close'].rolling(window=20).mean()
            prices_df['SMA_50'] = prices_df['close'].rolling(window=50).mean()
            prices_df['SMA_200'] = prices_df['close'].rolling(window=200).mean()
            
            # 2. Relative Strength Index (RSI)
            delta = prices_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            prices_df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. MACD
            prices_df['EMA_12'] = prices_df['close'].ewm(span=12, adjust=False).mean()
            prices_df['EMA_26'] = prices_df['close'].ewm(span=26, adjust=False).mean()
            prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
            prices_df['MACD_Signal'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 4. Bollinger Bands
            prices_df['BB_Middle'] = prices_df['close'].rolling(window=20).mean()
            prices_df['BB_Std'] = prices_df['close'].rolling(window=20).std()
            prices_df['BB_Upper'] = prices_df['BB_Middle'] + 2 * prices_df['BB_Std']
            prices_df['BB_Lower'] = prices_df['BB_Middle'] - 2 * prices_df['BB_Std']
            
            # 5. Volume indicators
            if 'volume' in prices_df.columns:
                prices_df['Volume_SMA_20'] = prices_df['volume'].rolling(window=20).mean()
                prices_df['Volume_Ratio'] = prices_df['volume'] / prices_df['Volume_SMA_20']
            
            # 6. Volatility
            prices_df['Daily_Return'] = prices_df['close'].pct_change()
            prices_df['Volatility_20'] = prices_df['Daily_Return'].rolling(window=20).std()
            
            # Drop NaN values after calculating indicators
            prices_df = prices_df.dropna()
            
            if prices_df.empty:
                progress.update_status("crypto_analyst_agent", ticker, "Insufficient data for analysis")
                # Store the error in the signals
                state["data"]["signals"][ticker]["crypto_analyst_agent"] = {
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": "Insufficient data for technical analysis",
                    "timestamp": datetime.now().isoformat(),
                    "error": "Insufficient data after calculating indicators"
                }
                # Also update the legacy structure
                state["data"]["analyst_signals"]["crypto_analyst"][ticker] = {
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": "Insufficient data for technical analysis"
                }
                continue
        except Exception as e:
            progress.update_status("crypto_analyst_agent", ticker, f"Error calculating indicators: {e}")
            # Store the error in the signals
            state["data"]["signals"][ticker]["crypto_analyst_agent"] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error calculating technical indicators: {e}",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            # Also update the legacy structure
            state["data"]["analyst_signals"]["crypto_analyst"][ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error calculating technical indicators: {e}"
            }
            continue
        
        # Prepare data for analysis
        try:
            progress.update_status("crypto_analyst_agent", ticker, "Analyzing market data")
            
            # Get the latest data point
            latest_data = prices_df.iloc[-1].to_dict()
            
            # Create a summary of the technical indicators
            technical_summary = {
                "price": latest_data["close"],
                "sma_20": latest_data["SMA_20"],
                "sma_50": latest_data["SMA_50"],
                "sma_200": latest_data["SMA_200"] if not pd.isna(latest_data.get("SMA_200")) else None,
                "rsi": latest_data["RSI"],
                "macd": latest_data["MACD"],
                "macd_signal": latest_data["MACD_Signal"],
                "bb_upper": latest_data["BB_Upper"],
                "bb_middle": latest_data["BB_Middle"],
                "bb_lower": latest_data["BB_Lower"],
                "volatility": latest_data["Volatility_20"],
            }
            
            # Add volume data if available
            if "volume" in latest_data:
                technical_summary["volume"] = latest_data["volume"]
                technical_summary["volume_sma_20"] = latest_data["Volume_SMA_20"]
                technical_summary["volume_ratio"] = latest_data["Volume_Ratio"]
            
            # Calculate price changes
            if len(prices_df) > 1:
                day_change = (latest_data["close"] / prices_df.iloc[-2]["close"] - 1) * 100
                technical_summary["day_change_pct"] = day_change
            
            if len(prices_df) > 7:
                week_change = (latest_data["close"] / prices_df.iloc[-7]["close"] - 1) * 100
                technical_summary["week_change_pct"] = week_change
            
            if len(prices_df) > 30:
                month_change = (latest_data["close"] / prices_df.iloc[-30]["close"] - 1) * 100
                technical_summary["month_change_pct"] = month_change
            
            # Generate trading signal based on technical analysis
            model_name = state.get("model_name", "gpt-4o")
            model_provider = state.get("model_provider", "OpenAI")
            
            # Call the LLM to analyze the data
            progress.update_status("crypto_analyst_agent", ticker, "Generating trading signal")
            analysis_result = generate_crypto_trading_signal(
                ticker=ticker,
                technical_data=technical_summary,
                model_name=model_name,
                model_provider=model_provider
            )
            
            # Store the analysis result in the state
            state["data"]["signals"][ticker]["crypto_analyst_agent"] = {
                "signal": analysis_result.signal,
                "confidence": analysis_result.confidence,
                "reasoning": analysis_result.reasoning,
                "timestamp": datetime.now().isoformat(),
                "technical_indicators": technical_summary
            }
            
            # Also update the legacy structure
            state["data"]["analyst_signals"]["crypto_analyst"][ticker] = {
                "signal": analysis_result.signal,
                "confidence": analysis_result.confidence,
                "reasoning": analysis_result.reasoning
            }
            
            progress.update_status("crypto_analyst_agent", ticker, f"Analysis complete: {analysis_result.signal.upper()} ({analysis_result.confidence}%)")
            
        except Exception as e:
            progress.update_status("crypto_analyst_agent", ticker, f"Error analyzing data: {e}")
            # Store the error in the signals
            state["data"]["signals"][ticker]["crypto_analyst_agent"] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error analyzing data: {e}",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            # Also update the legacy structure
            state["data"]["analyst_signals"]["crypto_analyst"][ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Error analyzing data: {e}"
            }
    
    # Show agent reasoning
    show_agent_reasoning("Crypto Analyst", f"Analyzed {len(tickers)} crypto assets using technical indicators")
    
    return state
