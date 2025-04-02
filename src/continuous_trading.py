import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import questionary
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Import LLM models
from llm.models import AVAILABLE_MODELS, ModelProvider, get_model, get_model_info

# Import analyst configurations
from utils.analysts import ANALYST_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import agent configuration
from utils.analysts import get_analyst_nodes, ANALYST_ORDER
from graph.state import AgentState
from utils.display import print_trading_output
from agents.portfolio_manager import portfolio_management_agent
from agents.risk_manager import risk_management_agent
from tools.api import get_account_info, place_order, get_position

# Import pandas for data manipulation
import pandas as pd

def get_current_date_range():
    """Get the current date range for analysis (last 30 days to today)."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    return start_date, end_date

def run_analysis(tickers, selected_analysts, model_provider="OpenAI", model_name="gpt-4o"):
    """Run analysis for the specified tickers using the selected analysts."""
    logger.info(f"Running analysis for {len(tickers)} tickers using {len(selected_analysts)} analysts")
    logger.info(f"Using model: {model_name} from provider: {model_provider}")
    
    # Filter out any analysts that are not in ANALYST_CONFIG
    valid_analysts = [a for a in selected_analysts if a in ANALYST_CONFIG]
    if len(valid_analysts) < len(selected_analysts):
        logger.warning(f"Filtered out {len(selected_analysts) - len(valid_analysts)} invalid analysts")
    
    # Get current date range for analysis
    start_date, end_date = get_current_date_range()
    logger.info(f"Analyzing data from {start_date} to {end_date}")
    
    # Initialize state
    state = {
        "messages": [],
        "data": {
            "tickers": tickers,
            "portfolio": {
                "cash": 100000.0,  # Default starting cash
                "positions": {}
            },
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": True,
            "model_name": model_name,
            "model_provider": model_provider,
        }
    }
    
    # Get account information if available
    try:
        account_info = get_account_info()
        if account_info:
            # Handle both dictionary-like and object-like access
            try:
                cash = float(account_info.get("cash", 100000.0))
            except AttributeError:
                # Try object attribute access
                cash = float(getattr(account_info, "cash", 100000.0))
                
            try:
                equity = float(account_info.get("equity", 100000.0))
            except AttributeError:
                # Try object attribute access
                equity = float(getattr(account_info, "equity", 100000.0))
                
            state["data"]["portfolio"] = {
                "cash": cash,
                "equity": equity,
                "positions": {}
            }
            logger.info(f"Initial account balance: ${cash:.2f}")
        else:
            # Use default values if account_info is None
            state["data"]["portfolio"] = {
                "cash": 100000.0,
                "equity": 100000.0,
                "positions": {}
            }
            logger.info(f"Using default initial account balance: ${state['data']['portfolio']['cash']:.2f}")
    except Exception as e:
        logger.error(f"Error getting account information: {e}")
        # Use default values
        state["data"]["portfolio"] = {
            "cash": 100000.0,
            "equity": 100000.0,
            "positions": {}
        }
        logger.info(f"Using default initial account balance: ${state['data']['portfolio']['cash']:.2f}")
    
    # Get position information for each ticker
    for ticker in tickers:
        try:
            position = get_position(ticker)
            if position:
                # Handle both dictionary-like and object-like access
                try:
                    shares = float(position.get("qty", 0))
                except AttributeError:
                    # Try object attribute access
                    shares = float(getattr(position, "qty", 0))
                    
                try:
                    avg_price = float(position.get("avg_entry_price", 0.0))
                except AttributeError:
                    # Try object attribute access
                    avg_price = float(getattr(position, "avg_entry_price", 0.0))
                    
                try:
                    current_price = float(position.get("current_price", 0.0))
                except AttributeError:
                    # Try object attribute access
                    current_price = float(getattr(position, "current_price", 0.0))
                    
                try:
                    market_value = float(position.get("market_value", 0.0))
                except AttributeError:
                    # Try object attribute access
                    market_value = float(getattr(position, "market_value", 0.0))
                    
                try:
                    unrealized_pl = float(position.get("unrealized_pl", 0.0))
                except AttributeError:
                    # Try object attribute access
                    unrealized_pl = float(getattr(position, "unrealized_pl", 0.0))
                    
                state["data"]["portfolio"]["positions"][ticker] = {
                    "shares": shares,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl
                }
        except Exception as e:
            logger.error(f"Error getting position for {ticker}: {e}")
            # Initialize with empty position
            state["data"]["portfolio"]["positions"][ticker] = {
                "shares": 0,
                "avg_price": 0.0,
                "current_price": 0.0,
                "market_value": 0.0,
                "unrealized_pl": 0.0
            }
    
    # Run analysis for each analyst
    for analyst_key in valid_analysts:
        logger.info(f"Running analysis with {ANALYST_CONFIG[analyst_key]['display_name']}")
        try:
            # Get the analyst function
            analyst_func = ANALYST_CONFIG[analyst_key]["agent_func"]
            
            # Run the analyst function
            analyst_state = analyst_func(state)
            
            # Update the state with the analyst's signals
            if analyst_state and "data" in analyst_state and "analyst_signals" in analyst_state["data"]:
                signals = analyst_state["data"]["analyst_signals"].get(f"{analyst_key}_agent")
                if signals:
                    state["data"]["analyst_signals"][f"{analyst_key}_agent"] = signals
                    logger.info(f"Received signals from {ANALYST_CONFIG[analyst_key]['display_name']}")
                else:
                    logger.warning(f"No signals received from {ANALYST_CONFIG[analyst_key]['display_name']}")
            else:
                logger.warning(f"Invalid state returned from {ANALYST_CONFIG[analyst_key]['display_name']}")
        except Exception as e:
            logger.error(f"Error running analysis with {ANALYST_CONFIG[analyst_key]['display_name']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Run risk management analysis
    try:
        logger.info("Running risk management analysis")
        risk_state = risk_management_agent(state)
        if risk_state and "data" in risk_state and "analyst_signals" in risk_state["data"]:
            signals = risk_state["data"]["analyst_signals"].get("risk_management_agent")
            if signals:
                state["data"]["analyst_signals"]["risk_management_agent"] = signals
                logger.info("Received signals from risk management")
            else:
                logger.warning("No signals received from risk management")
        else:
            logger.warning("Invalid state returned from risk management")
    except Exception as e:
        logger.error(f"Error running risk management analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Run portfolio management analysis
    try:
        logger.info("Running portfolio management analysis")
        portfolio_state = portfolio_management_agent(state)
        if portfolio_state and "data" in portfolio_state and "analyst_signals" in portfolio_state["data"]:
            signals = portfolio_state["data"]["analyst_signals"].get("portfolio_manager_agent")
            if signals:
                state["data"]["analyst_signals"]["portfolio_manager_agent"] = signals
                logger.info("Received signals from portfolio management")
            else:
                logger.warning("No signals received from portfolio management")
        else:
            logger.warning("Invalid state returned from portfolio management")
    except Exception as e:
        logger.error(f"Error running portfolio management analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return state

def execute_trades(state, api_provider="alpaca", paper_trading=True):
    """Execute trades based on the portfolio manager's recommendations."""
    # Check if we have any portfolio signals
    if "portfolio_manager_agent" not in state["data"]["analyst_signals"]:
        logger.warning("No portfolio signals available. Skipping trade execution.")
        return
    
    portfolio_signals = state["data"]["analyst_signals"]["portfolio_manager_agent"]
    
    # Debug: Print available signals
    logger.info("Before execute_trades - Available analyst signals:")
    for key, signals in state["data"]["analyst_signals"].items():
        logger.info(f"  {key}: {signals}")
    
    # Check if we're using paper trading
    if paper_trading:
        logger.info(f"Using {api_provider} API - trades will be simulated only")
        logger.info(f"Would execute trades: {portfolio_signals}")
        return
    
    # Execute real trades
    logger.info(f"Using {api_provider} API - executing REAL trades")
    
    try:
        # Initialize API client based on provider
        if api_provider.lower() == "alpaca":
            # Check if Alpaca API keys are available
            if not os.environ.get("APCA_API_KEY_ID") or not os.environ.get("APCA_API_SECRET_KEY"):
                logger.error("Alpaca API keys not found. Cannot execute real trades.")
                return
            
            # Import Alpaca API client
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce
            except ImportError:
                logger.error("Alpaca SDK not installed. Run 'pip install alpaca-py'")
                return
            
            # Initialize Alpaca client
            api_key = os.environ.get("APCA_API_KEY_ID")
            api_secret = os.environ.get("APCA_API_SECRET_KEY")
            trading_client = TradingClient(api_key, api_secret, paper=paper_trading)
            
            # Get current positions to check available quantities
            try:
                positions = {}
                all_positions = trading_client.get_all_positions()
                for position in all_positions:
                    try:
                        positions[position.symbol] = {
                            "qty": float(position.qty),
                            "side": "long" if float(position.qty) > 0 else "short"
                        }
                    except Exception as e:
                        logger.error(f"Error processing position {position.symbol}: {e}")
                logger.info(f"Current positions: {positions}")
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                positions = {}
            
            # Execute trades for each ticker
            for ticker, signal in portfolio_signals.items():
                # Skip if quantity is 0
                quantity = signal.get("quantity", 0)
                if quantity <= 0:
                    logger.info(f"Skipping {ticker} with quantity {quantity}")
                    continue
                
                # Determine order side
                signal_type = signal.get("signal", "").lower()
                action = signal.get("action", "").lower()
                
                # Check both signal and action fields for compatibility
                if action in ["buy", "bullish", "strong buy"]:
                    side = OrderSide.BUY
                elif action in ["sell", "bearish", "strong sell"]:
                    side = OrderSide.SELL
                elif signal_type in ["buy", "bullish", "strong buy"]:
                    side = OrderSide.BUY
                elif signal_type in ["sell", "bearish", "strong sell"]:
                    side = OrderSide.SELL
                else:
                    logger.info(f"Skipping {ticker} with signal {signal_type} and action {action}")
                    continue
                
                # Create market order with shorting enabled
                try:
                    market_order = MarketOrderRequest(
                        symbol=ticker,
                        qty=quantity,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    # Submit order
                    order = trading_client.submit_order(market_order)
                    logger.info(f"Submitted order: {ticker} {side} {quantity} shares")
                    logger.info(f"Order ID: {order.id}")
                except Exception as e:
                    logger.error(f"Error submitting order for {ticker}: {e}")
                    
                    # If the error is due to insufficient quantity, try to place the order with available quantity
                    if "insufficient qty available" in str(e).lower():
                        try:
                            # Extract available quantity from error message if possible
                            import re
                            available_match = re.search(r'available:\s*(\d+)', str(e))
                            available_qty = int(available_match.group(1)) if available_match else 0
                            
                            if available_qty > 0 and side == OrderSide.SELL:
                                logger.info(f"Retrying with available quantity: {available_qty} shares")
                                market_order = MarketOrderRequest(
                                    symbol=ticker,
                                    qty=available_qty,
                                    side=side,
                                    time_in_force=TimeInForce.DAY
                                )
                                order = trading_client.submit_order(market_order)
                                logger.info(f"Submitted adjusted order: {ticker} {side} {available_qty} shares")
                                logger.info(f"Order ID: {order.id}")
                        except Exception as retry_error:
                            logger.error(f"Error retrying order for {ticker}: {retry_error}")
        else:
            logger.warning(f"Real trading not implemented for {api_provider} API")
            logger.info(f"Would execute trades: {portfolio_signals}")
    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        import traceback
        logger.error(traceback.format_exc())

def continuous_trading(tickers, selected_analysts, interval_minutes=60, max_runtime=None, api_provider="alpaca", paper_trading=True, model_name="gpt-4o", model_provider="OpenAI"):
    """Run continuous trading with the specified tickers and analysts."""
    logger.info(f"Starting continuous trading with {len(tickers)} tickers and {len(selected_analysts)} analysts")
    logger.info(f"Using model: {model_name} from provider: {model_provider}")
    logger.info(f"Trading interval: {interval_minutes} minutes")
    
    if max_runtime:
        logger.info(f"Maximum runtime: {max_runtime} minutes")
        end_time = datetime.now() + timedelta(minutes=max_runtime)
    else:
        end_time = None
        logger.info("Running indefinitely (no maximum runtime specified)")
    
    # Initialize state
    state = {
        "messages": [],
        "data": {
            "tickers": tickers,
            "portfolio": {},
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": True,
            "model_name": model_name,
            "model_provider": model_provider,
        }
    }
    
    # Get initial account information
    try:
        account_info = get_account_info()
        if account_info:
            # Handle both dictionary-like and object-like access
            try:
                cash = float(account_info.get("cash", 100000.0))
            except AttributeError:
                # Try object attribute access
                cash = float(getattr(account_info, "cash", 100000.0))
                
            try:
                equity = float(account_info.get("equity", 100000.0))
            except AttributeError:
                # Try object attribute access
                equity = float(getattr(account_info, "equity", 100000.0))
                
            state["data"]["portfolio"] = {
                "cash": cash,
                "equity": equity,
                "positions": {}
            }
            logger.info(f"Initial account balance: ${cash:.2f}")
        else:
            # Use default values if account_info is None
            state["data"]["portfolio"] = {
                "cash": 100000.0,
                "equity": 100000.0,
                "positions": {}
            }
            logger.info(f"Using default initial account balance: ${state['data']['portfolio']['cash']:.2f}")
    except Exception as e:
        logger.error(f"Error getting account information: {e}")
        # Use default values
        state["data"]["portfolio"] = {
            "cash": 100000.0,
            "equity": 100000.0,
            "positions": {}
        }
        logger.info(f"Using default initial account balance: ${state['data']['portfolio']['cash']:.2f}")
    
    # Run continuous trading loop
    try:
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"\n=== Trading Iteration {iteration} ===\n")
            
            # Run analysis
            state = run_analysis(tickers, selected_analysts, model_provider, model_name)
            
            # Execute trades based on analysis
            execute_trades(state, api_provider, paper_trading)
            
            # Print summary
            logger.info("\nTrading Summary:")
            for ticker in tickers:
                for analyst_name, signals in state["data"]["analyst_signals"].items():
                    if ticker in signals:
                        signal = signals[ticker]
                        if isinstance(signal, dict):
                            if "signal" in signal:
                                logger.info(f"{ticker} - {analyst_name}: {signal['signal']}")
                            elif "action" in signal:
                                logger.info(f"{ticker} - {analyst_name}: {signal['action']} {signal.get('quantity', 0)} shares")
            
            # Check if we've reached the maximum runtime
            if max_runtime and (time.time() - start_time) > (max_runtime * 60 * 60):
                logger.info(f"Reached maximum runtime of {max_runtime} hours. Exiting.")
                break
            
            # Wait for the next iteration
            logger.info(f"Waiting {interval_minutes} minutes until next analysis...")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in continuous trading: {e}")
        raise
    finally:
        logger.info("Continuous trading ended")

def get_available_analysts():
    """Get a list of available analysts for display."""
    return [(key, config["display_name"]) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]

def main():
    parser = argparse.ArgumentParser(description="Run continuous trading with specified tickers and analysts")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to analyze")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes between trading cycles")
    parser.add_argument("--runtime", type=int, help="Maximum runtime in minutes (optional)")
    parser.add_argument("--paper", action="store_true", help="Use paper trading (default)")
    parser.add_argument("--live", action="store_true", help="Use live trading (use with caution!)")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # If no tickers provided, prompt for input
    if not args.tickers:
        tickers_input = questionary.text(
            "Enter tickers to analyze (comma-separated):",
            default="AAPL,MSFT,GOOGL,AMZN,META"
        ).ask()
        tickers = [t.strip() for t in tickers_input.split(',')]
    else:
        tickers = [t.strip() for t in args.tickers.split(',')]
    
    # Check if we have valid tickers
    if not tickers or not all(tickers):
        logger.error("No valid tickers provided. Exiting.")
        return
    
    logger.info(f"Selected tickers: {tickers}")
    
    # Get available analysts
    analyst_choices = []
    for k, v in ANALYST_CONFIG.items():
        analyst_choices.append({
            "name": v["display_name"],
            "value": k
        })
    
    # Sort analyst choices alphabetically by display name
    analyst_choices.sort(key=lambda x: x["name"])
    
    # Prompt for analysts selection
    try:
        selected_analysts = questionary.checkbox(
            "Select analysts to use:",
            choices=analyst_choices
        ).ask()
        
        if not selected_analysts:
            logger.warning("No analysts selected. Using default analysts.")
            # Default to first two analysts if available
            selected_analysts = []
            if len(analyst_choices) > 0:
                selected_analysts.append(analyst_choices[0]["value"])
            if len(analyst_choices) > 1:
                selected_analysts.append(analyst_choices[1]["value"])
    except Exception as e:
        logger.error(f"Error in analyst selection: {e}")
        # Fall back to default analysts
        selected_analysts = []
        if len(analyst_choices) > 0:
            selected_analysts.append(analyst_choices[0]["value"])
        if len(analyst_choices) > 1:
            selected_analysts.append(analyst_choices[1]["value"])
    
    logger.info(f"Selected analysts: {[ANALYST_CONFIG[a]['display_name'] for a in selected_analysts]}")
    
    # Remove risk_management from selected_analysts since it's not in ANALYST_CONFIG
    if "risk_management" in selected_analysts:
        selected_analysts.remove("risk_management")
        logger.info("Removed risk_management from selected analysts (not in ANALYST_CONFIG)")
    
    # Select model provider and model using the approach from main.py
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=(model_name, provider)) 
                 for display, model_name, provider in [model.to_choice_tuple() for model in AVAILABLE_MODELS]],
    ).ask()
    
    if not model_choice:
        logger.warning("No model selected. Using default model.")
        model_name = "gpt-4o"
        model_provider = "OpenAI"
    else:
        # model_choice is a tuple (model_name, provider)
        model_name, model_provider = model_choice
    
    logger.info(f"Selected model: {model_name} from provider {model_provider}")
    
    # Determine paper vs live trading
    if args.live:
        paper_trading = False
        # Confirm live trading with the user
        try:
            confirm_live = questionary.confirm(
                "WARNING: You are about to use REAL MONEY for trading. Continue?",
                default=False
            ).ask()
            
            if not confirm_live:
                logger.info("Live trading cancelled. Switching to paper trading.")
                paper_trading = True
        except Exception as e:
            logger.error(f"Error in live trading confirmation: {e}")
            logger.info("Defaulting to paper trading for safety.")
            paper_trading = True
    else:
        paper_trading = True
        logger.info("Using paper trading mode (no real money will be used)")
    
    # Run continuous trading
    try:
        continuous_trading(
            tickers=tickers,
            selected_analysts=selected_analysts,
            interval_minutes=args.interval,
            max_runtime=args.runtime,
            api_provider=model_provider.lower(),
            paper_trading=paper_trading,
            model_name=model_name,
            model_provider=model_provider
        )
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Error in continuous trading: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Initialize colorama for colored terminal output
    init(autoreset=True)
    
    main()
