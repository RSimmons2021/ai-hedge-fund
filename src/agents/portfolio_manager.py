import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from typing import Dict, Any, List, Optional

from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from utils.llm import call_llm
from tools.api import get_current_price, is_crypto_pair
from pydantic import BaseModel, Field
from typing import Literal


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"] = Field(description="Trading action to take")
    quantity: float = Field(description="Number of shares or amount of crypto to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: Dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Makes final trading decisions and generates orders for multiple tickers"""
    progress.update_status("portfolio_management_agent", None, "Making trading decisions")
    
    try:
        # Get tickers and portfolio from state
        tickers = state["data"]["tickers"]
        portfolio = state["data"]["portfolio"]
        
        # Initialize signals structure if not present
        if "signals" not in state["data"]:
            state["data"]["signals"] = {}
        
        # Initialize analyst_signals structure for backward compatibility
        if "analyst_signals" not in state["data"]:
            state["data"]["analyst_signals"] = {}
        
        # Get all analyst signals for each ticker
        signals_by_ticker = {}
        current_prices = {}
        max_shares = {}
        
        for ticker in tickers:
            progress.update_status("portfolio_management_agent", ticker, "Analyzing signals")
            
            # Initialize ticker in signals structure if not present
            if ticker not in state["data"]["signals"]:
                state["data"]["signals"][ticker] = {}
            
            # Get signals for this ticker
            ticker_signals = {}
            
            # Look for signals from different agents
            for agent_name, signal in state["data"]["signals"].get(ticker, {}).items():
                if agent_name != "portfolio_management_agent" and agent_name != "order_executor_agent":
                    ticker_signals[agent_name] = signal
            
            # Also check legacy analyst_signals structure
            for analyst_name, analyst_signals in state["data"]["analyst_signals"].items():
                if ticker in analyst_signals:
                    ticker_signals[analyst_name] = analyst_signals[ticker]
            
            signals_by_ticker[ticker] = ticker_signals
            
            # Get current price
            try:
                current_price = get_current_price(ticker)
                current_prices[ticker] = current_price
            except Exception as e:
                print(f"Error getting price for {ticker}: {e}")
                current_prices[ticker] = 0
            
            # Get max shares/amount from risk manager
            try:
                if "risk_management_agent" in ticker_signals:
                    risk_signal = ticker_signals["risk_management_agent"]
                    max_shares[ticker] = risk_signal.get("max_position_size", 0)
                else:
                    # Default to 10% of portfolio for each asset if no risk signal
                    portfolio_value = portfolio.get("total_value", 10000)  # Default to $10k if not available
                    max_shares[ticker] = (portfolio_value * 0.1) / current_prices[ticker] if current_prices[ticker] > 0 else 0
            except Exception as e:
                print(f"Error calculating max shares for {ticker}: {e}")
                max_shares[ticker] = 0
        
        # Generate trading decisions
        model_name = state.get("model_name", "gpt-4o")
        model_provider = state.get("model_provider", "OpenAI")
        
        trading_decisions = generate_trading_decision(
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            max_shares=max_shares,
            portfolio=portfolio,
            model_name=model_name,
            model_provider=model_provider
        )
        
        # Store decisions in state
        for ticker, decision in trading_decisions.decisions.items():
            # Ensure quantity is appropriate for the asset type
            if is_crypto_pair(ticker):
                # For crypto, allow fractional amounts but ensure it's a float
                quantity = float(decision.quantity)
            else:
                # For stocks, ensure quantity is an integer
                quantity = int(float(decision.quantity))
            
            # Update the decision with the corrected quantity
            decision.quantity = quantity
            
            # Store the decision in the state
            state["data"]["signals"][ticker]["portfolio_manager_agent"] = {
                "action": decision.action,
                "quantity": quantity,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "timestamp": datetime.now().isoformat()
            }
            
            # Also update the legacy structure
            if "portfolio_manager_agent" not in state["data"]["analyst_signals"]:
                state["data"]["analyst_signals"]["portfolio_manager_agent"] = {}
            
            state["data"]["analyst_signals"]["portfolio_manager_agent"][ticker] = {
                "action": decision.action,
                "quantity": quantity,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            }
            
            progress.update_status("portfolio_management_agent", ticker, f"Decision: {decision.action.upper()} {quantity}")

        # Show reasoning
        show_agent_reasoning("Portfolio Manager", f"Generated trading decisions for {len(tickers)} assets")

    except Exception as e:
        progress.update_status("portfolio_management_agent", None, f"Error: {str(e)}")
        print(f"Error in portfolio management agent: {e}")

    return state


def generate_trading_decision(
    tickers: List[str],
    signals_by_ticker: Dict[str, Dict],
    current_prices: Dict[str, float],
    max_shares: Dict[str, float],
    portfolio: Dict[str, Any],
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""

    # Create a prompt template for the portfolio manager
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager responsible for making final trading decisions.
                Based on the signals from various analysts and risk management constraints, 
                you need to decide whether to buy, sell, or hold each ticker.
                
                For each ticker, provide:
                1. Action: buy, sell, short, cover, or hold
                2. Quantity: Number of shares or amount of crypto to trade
                3. Confidence: A score from 0-100 indicating your confidence in this decision
                4. Reasoning: A brief explanation of your decision
                
                Consider the following guidelines:
                - Never exceed the maximum shares/amount allowed for each ticker
                - Prioritize tickers with stronger analyst consensus
                - Consider diversification across different tickers
                - For crypto assets (containing '/'), you can trade fractional amounts
                - For stocks, you must trade whole shares
                
                IMPORTANT: Your response must be valid JSON that matches the expected output format.
                Format your response as: {"decisions": {"TICKER1": {"action": "...", "quantity": X, "confidence": Y, "reasoning": "..."}}}.
                """,
            ),
            (
                "human",
                """Please analyze the following data and make trading decisions:
                
                Available Cash: ${cash}
                
                Tickers to analyze: {tickers}
                
                Analyst Signals:
                {signals}
                
                Current Prices:
                {prices}
                
                Maximum Shares/Amount Allowed (based on risk limits):
                {max_shares}
                
                Current Portfolio Holdings:
                {holdings}
                
                Provide your trading decisions for each ticker in valid JSON format.
                Remember to format your response as: {"decisions": {"TICKER1": {"action": "...", "quantity": X, "confidence": Y, "reasoning": "..."}}}.
                """.format(
                    cash=portfolio.get("cash", 0),
                    tickers=", ".join(tickers),
                    signals=json.dumps(signals_by_ticker, indent=2, default=str),
                    prices="\n".join([f"{t}: ${p}" for t, p in current_prices.items()]),
                    max_shares="\n".join([f"{t}: {s}" for t, s in max_shares.items()]),
                    holdings=json.dumps(portfolio.get("holdings", {}), indent=2, default=str),
                ),
            ),
        ]
    )

    # Create message for the LLM
    message = HumanMessage(content=template.format_messages()[1].content)

    # Call the LLM with retry logic
    try:
        response = call_llm(
            prompt=message.content,
            model_name=model_name,
            model_provider=model_provider,
            pydantic_model=PortfolioManagerOutput,
            agent_name="portfolio_manager_agent",
            max_retries=3,
            default_factory=lambda: PortfolioManagerOutput(decisions={}),
        )
        return response
    except Exception as e:
        print(f"Error generating trading decisions: {e}")
        # Return empty decisions in case of error
        return PortfolioManagerOutput(decisions={})
