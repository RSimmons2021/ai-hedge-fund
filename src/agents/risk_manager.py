from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df, get_price_data, is_crypto_pair
import json
from typing import Dict, Any


##### Risk Management Agent #####
def risk_management_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        try:
            # Use the appropriate price data function based on asset type
            if is_crypto_pair(ticker):
                prices_df = get_price_data(ticker, state["start_date"], state["end_date"])
                if prices_df.empty:
                    progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
                    continue
            else:
                prices = get_prices(
                    ticker=ticker,
                    start_date=state["start_date"],
                    end_date=state["end_date"],
                )

                if not prices:
                    progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
                    continue

                prices_df = prices_to_df(prices)

            progress.update_status("risk_management_agent", ticker, "Calculating position limits")

            # Calculate portfolio value
            current_price = prices_df["close"].iloc[-1]
            current_prices[ticker] = current_price  # Store the current price

            # Calculate current position value for this ticker
            current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

            # Calculate total portfolio value using stored prices
            total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

            # Base limit is 20% of portfolio for any single position
            position_limit = total_portfolio_value * 0.20

            # For existing positions, subtract current position value from limit
            remaining_position_limit = position_limit - current_position_value

            # Calculate volatility-based risk adjustment
            # Higher volatility = lower position limit
            returns = prices_df["close"].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility

            # Adjust position limit based on volatility
            # Higher volatility = lower position limit
            volatility_adjustment = 1.0 - min(0.5, volatility)  # Cap adjustment at 50%
            adjusted_position_limit = remaining_position_limit * volatility_adjustment

            # Store risk analysis
            risk_analysis[ticker] = {
                "current_price": current_price,
                "position_limit": adjusted_position_limit,
                "volatility": volatility,
                "volatility_adjustment": volatility_adjustment
            }

            progress.update_status("risk_management_agent", ticker, "Done")

        except Exception as e:
            progress.update_status("risk_management_agent", ticker, f"Error: {str(e)}")
            print(f"Error in risk analysis for {ticker}: {e}")

    # Store risk analysis in state
    if "signals" not in state["data"]:
        state["data"]["signals"] = {}

    # Add risk analysis for each ticker
    for ticker, analysis in risk_analysis.items():
        if ticker not in state["data"]["signals"]:
            state["data"]["signals"][ticker] = {}
        
        state["data"]["signals"][ticker]["risk_management_agent"] = analysis

    show_agent_reasoning(risk_analysis, "Risk Management")
    return state
