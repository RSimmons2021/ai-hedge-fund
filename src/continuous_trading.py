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
from agents.order_executor import order_executor_agent
from tools.api import get_account_info, place_order, get_position, is_crypto_pair

# Import pandas for data manipulation
import pandas as pd

# Check for required API keys
def check_api_keys():
    """Check if all required API keys are present in environment variables."""
    required_keys = {
        "ALPHA_VANTAGE_API_KEY": "Alpha Vantage API (for stock price data)",
        "FINANCIAL_DATASETS_API_KEY": "Financial Datasets API (for financial data)",
    }
    
    # At least one of these LLM API keys is required
    llm_keys = {
        "OPENAI_API_KEY": "OpenAI API",
        "GOOGLE_API_KEY": "Google/Gemini API",
        "ANTHROPIC_API_KEY": "Anthropic API",
        "GROQ_API_KEY": "Groq API",
        "DEEPSEEK_API_KEY": "DeepSeek API"
    }
    
    # Trading API keys (optional for paper trading)
    trading_keys = {
        "APCA_API_KEY_ID": "Alpaca API Key ID",
        "APCA_API_SECRET_KEY": "Alpaca API Secret Key"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.environ.get(key):
            missing_keys.append(f"{key} ({description})")
    
    # Check if at least one LLM API key is present
    has_llm_key = False
    for key in llm_keys:
        if os.environ.get(key):
            has_llm_key = True
            break
    
    if not has_llm_key:
        missing_keys.append(f"At least one of: {', '.join(llm_keys.keys())} (for LLM-based analysis)")
    
    # Check trading API keys
    missing_trading_keys = []
    for key, description in trading_keys.items():
        if not os.environ.get(key):
            missing_trading_keys.append(f"{key} ({description})")
    
    # Print warnings
    if missing_keys:
        logger.warning("The following required API keys are missing:")
        for key in missing_keys:
            logger.warning(f"  - {key}")
        logger.warning("The application may not function correctly without these keys.")
    
    if missing_trading_keys:
        logger.warning("The following trading API keys are missing:")
        for key in missing_trading_keys:
            logger.warning(f"  - {key}")
        logger.warning("Live trading will be disabled. Only paper trading will be available.")
    
    return len(missing_keys) == 0

def get_current_date_range():
    """Get the current date range for analysis (last 30 days to today)."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    return start_date, end_date

def run_analysis(tickers, selected_analysts, model_provider="OpenAI", model_name="gpt-4o"):
    """Run analysis for the specified tickers using the selected analysts."""
    logger.info(f"Running analysis for {len(tickers)} tickers using {len(selected_analysts)} analysts")
    logger.info(f"Using model: {model_name} from provider: {model_provider}")
    
    # Convert model_provider string to ModelProvider enum if needed
    from llm.models import ModelProvider
    if isinstance(model_provider, str):
        try:
            # Try to convert to enum (case-insensitive)
            model_provider = ModelProvider(model_provider.capitalize())
        except ValueError:
            # If conversion fails, default to OpenAI
            logger.warning(f"Invalid model provider '{model_provider}', defaulting to OpenAI")
            model_provider = ModelProvider.OPENAI
    
    # Filter out any analysts that are not in ANALYST_CONFIG
    valid_analysts = [a for a in selected_analysts if a in ANALYST_CONFIG]
    if len(valid_analysts) < len(selected_analysts):
        logger.warning(f"Filtered out {len(selected_analysts) - len(valid_analysts)} invalid analysts")
        logger.warning(f"Invalid analysts: {set(selected_analysts) - set(valid_analysts)}")
    
    # Get current date range for analysis
    start_date, end_date = get_current_date_range()
    logger.info(f"Analyzing data from {start_date} to {end_date}")
    
    # Check if any of the tickers are crypto pairs
    crypto_tickers = [ticker for ticker in tickers if is_crypto_pair(ticker)]
    stock_tickers = [ticker for ticker in tickers if not is_crypto_pair(ticker)]
    
    # If we have crypto tickers but no crypto analyst, add it automatically
    if crypto_tickers and "crypto_analyst" not in valid_analysts:
        logger.info("Crypto pairs detected. Adding Crypto Analyst automatically.")
        valid_analysts.append("crypto_analyst")
    
    # Initialize state
    state = {
        "data": {
            "signals": {},  # Main structure for all signals
            "portfolio": {
                "cash": 100000.0,  # Default starting cash
                "positions": {}
            },
            "tickers": tickers,
            "analyst_signals": {}  # Legacy structure for backward compatibility
        },
        "model_name": model_name,
        "model_provider": model_provider,
        "start_date": start_date,
        "end_date": end_date,
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
    
    # Run the analysis pipeline
    try:
        # Initialize signals structure for each ticker
        for ticker in tickers:
            state["data"]["signals"][ticker] = {}
        
        # Run each analyst in order
        for analyst_name in valid_analysts:
            logger.info(f"Running {analyst_name}...")
            if analyst_name not in ANALYST_CONFIG:
                logger.warning(f"Analyst {analyst_name} not found in configuration. Skipping.")
                continue
                
            analyst_fn = ANALYST_CONFIG[analyst_name]["function"]
            
            # For crypto analyst, only pass crypto tickers
            if analyst_name == "crypto_analyst" and crypto_tickers:
                # Create a copy of state with only crypto tickers
                crypto_state = state.copy()
                crypto_state["data"] = state["data"].copy()
                crypto_state["data"]["tickers"] = crypto_tickers
                
                # Run crypto analyst
                try:
                    updated_state = analyst_fn(crypto_state)
                    # Merge the results back into the main state
                    for ticker in crypto_tickers:
                        if ticker in updated_state["data"].get("signals", {}):
                            state["data"]["signals"][ticker].update(updated_state["data"]["signals"][ticker])
                    # Also merge analyst_signals if present (for backward compatibility)
                    if "analyst_signals" in updated_state["data"]:
                        for analyst, signals in updated_state["data"]["analyst_signals"].items():
                            if analyst not in state["data"]["analyst_signals"]:
                                state["data"]["analyst_signals"][analyst] = {}
                            state["data"]["analyst_signals"][analyst].update(signals)
                except Exception as e:
                    logger.error(f"Error running {analyst_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # For stock analysts, only pass stock tickers
            elif analyst_name != "crypto_analyst" and stock_tickers:
                # Create a copy of state with only stock tickers
                stock_state = state.copy()
                stock_state["data"] = state["data"].copy()
                stock_state["data"]["tickers"] = stock_tickers
                
                # Run stock analyst
                try:
                    updated_state = analyst_fn(stock_state)
                    # Merge the results back into the main state
                    for ticker in stock_tickers:
                        if ticker in updated_state["data"].get("signals", {}):
                            state["data"]["signals"][ticker].update(updated_state["data"]["signals"][ticker])
                    # Also merge analyst_signals if present (for backward compatibility)
                    if "analyst_signals" in updated_state["data"]:
                        for analyst, signals in updated_state["data"]["analyst_signals"].items():
                            if analyst not in state["data"]["analyst_signals"]:
                                state["data"]["analyst_signals"][analyst] = {}
                            state["data"]["analyst_signals"][analyst].update(signals)
                except Exception as e:
                    logger.error(f"Error running {analyst_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Run risk management agent
        logger.info("Running risk management agent...")
        try:
            state = risk_management_agent(state)
        except Exception as e:
            logger.error(f"Error running risk management agent: {e}")
            import traceback
            traceback.print_exc()
        
        # Run portfolio management agent
        logger.info("Running portfolio management agent...")
        try:
            state = portfolio_management_agent(state)
        except Exception as e:
            logger.error(f"Error running portfolio management agent: {e}")
            import traceback
            traceback.print_exc()
        
        # Run order executor agent
        logger.info("Running order executor agent...")
        try:
            state = order_executor_agent(state)
        except Exception as e:
            logger.error(f"Error running order executor agent: {e}")
            import traceback
            traceback.print_exc()
        
        # Print trading output
        print_trading_output(state)
        
        return state
    
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        import traceback
        traceback.print_exc()
        return state

def continuous_trading(tickers, selected_analysts, interval_minutes=60, max_runtime=None, model_provider="OpenAI", model_name="gpt-4o", paper_trading=True):
    """Run continuous trading with the specified tickers and analysts.
    
    Args:
        tickers: List of ticker symbols to analyze
        selected_analysts: List of analyst names to use
        interval_minutes: Minutes between trading cycles
        max_runtime: Maximum runtime in minutes (None for unlimited)
        model_provider: LLM provider to use
        model_name: LLM model to use
        paper_trading: Whether to use paper trading (True) or live trading (False)
    """
    logger.info(f"Starting continuous trading with {len(tickers)} tickers and {len(selected_analysts)} analysts")
    logger.info(f"Trading interval: {interval_minutes} minutes")
    
    if max_runtime:
        logger.info(f"Maximum runtime: {max_runtime} minutes")
    
    # Set up the API provider based on paper_trading flag
    from tools.api import set_paper_trading
    set_paper_trading(paper_trading)
    
    # Track the start time
    start_time = time.time()
    run_count = 0
    
    try:
        while True:
            run_count += 1
            logger.info(f"\n=== Trading cycle {run_count} ===\n")
            
            # Check if any of the tickers are crypto pairs
            from tools.api import is_crypto_pair
            crypto_tickers = [ticker for ticker in tickers if is_crypto_pair(ticker)]
            
            # If we have crypto tickers but no crypto analyst, add it automatically
            analysts_to_use = selected_analysts.copy()
            if crypto_tickers and "crypto_analyst" not in analysts_to_use:
                logger.info("Crypto pairs detected. Adding Crypto Analyst automatically.")
                analysts_to_use.append("crypto_analyst")
            
            # Run the analysis
            run_analysis(tickers, analysts_to_use, model_provider, model_name)
            
            # Check if we've exceeded the maximum runtime
            if max_runtime:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= max_runtime:
                    logger.info(f"Maximum runtime of {max_runtime} minutes reached. Stopping.")
                    break
            
            # Wait for the next interval
            logger.info(f"Waiting {interval_minutes} minutes until next trading cycle...")
            time.sleep(interval_minutes * 60)
    
    except KeyboardInterrupt:
        logger.info("\nTrading stopped by user.")
    except Exception as e:
        logger.error(f"Error in continuous trading: {e}")
        import traceback
        traceback.print_exc()

def get_available_analysts():
    """Get a list of available analysts for display."""
    return [name for name in ANALYST_CONFIG.keys()]

def main():
    """Main function to run the continuous trading system."""
    parser = argparse.ArgumentParser(description="AI Hedge Fund - Continuous Trading")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to analyze")
    parser.add_argument("--analysts", type=str, help="Comma-separated list of analysts to use")
    parser.add_argument("--interval", type=int, default=60, help="Trading interval in minutes")
    parser.add_argument("--runtime", type=int, help="Maximum runtime in minutes")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--provider", type=str, help="LLM provider to use")
    parser.add_argument("--paper", action="store_true", help="Use paper trading")
    parser.add_argument("--live", action="store_true", help="Use live trading (default is paper trading)")
    
    args = parser.parse_args()
    
    # Check API keys
    check_api_keys()
    
    # Get available analysts
    available_analysts = get_available_analysts()
    
    # If tickers are provided via command line, use them
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        # Otherwise, prompt the user for tickers
        ticker_input = questionary.text(
            "Enter ticker symbols (comma-separated):",
            default="AAPL,MSFT,GOOGL"
        ).ask()
        tickers = [t.strip() for t in ticker_input.split(",")]
    
    # If analysts are provided via command line, use them
    if args.analysts:
        selected_analysts = [a.strip() for a in args.analysts.split(",")]
        # Validate the analysts
        selected_analysts = [a for a in selected_analysts if a in available_analysts]
    else:
        # Otherwise, prompt the user for analysts
        selected_analysts = questionary.checkbox(
            "Select analysts to use:",
            choices=available_analysts
        ).ask()
    
    # If no analysts were selected, use the default ones
    if not selected_analysts:
        selected_analysts = ["technical_analyst", "fundamental_analyst", "sentiment_analyst"]
        logger.info(f"No analysts selected, using defaults: {', '.join(selected_analysts)}")
    
    # Determine trading mode (paper vs live)
    paper_trading = not args.live if args.live else True
    if args.paper:
        paper_trading = True
    
    # If model and provider are provided via command line, use them
    model_name = args.model
    model_provider = args.provider
    
    # If model or provider not specified, prompt the user to select
    if not model_name or not model_provider:
        # Create choices for questionary
        model_choices = []
        for model in AVAILABLE_MODELS:
            model_choices.append({
                'name': f"{model.display_name} ({model.provider.value})",
                'value': (model.model_name, model.provider.value)
            })
        
        # Sort choices by provider and name
        model_choices.sort(key=lambda x: x['name'])
        
        # Prompt the user to select a model
        selected_model = questionary.select(
            "Select LLM model to use:",
            choices=model_choices,
            default=next((c for c in model_choices if "gpt-4o" in c['name']), model_choices[0])
        ).ask()
        
        if selected_model:
            model_name, model_provider = selected_model
        else:
            # Default to GPT-4o if no selection
            model_name = "gpt-4o"
            model_provider = "OpenAI"
    
    # Print configuration
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Analysts: {', '.join(selected_analysts)}")
    logger.info(f"Trading mode: {'Paper' if paper_trading else 'Live'}")
    logger.info(f"Model: {model_name} ({model_provider})")
    
    # Run continuous trading
    continuous_trading(
        tickers=tickers,
        selected_analysts=selected_analysts,
        interval_minutes=args.interval,
        max_runtime=args.runtime,
        model_provider=model_provider,
        model_name=model_name,
        paper_trading=paper_trading
    )

if __name__ == "__main__":
    # Initialize colorama for colored terminal output
    init(autoreset=True)
    
    # Run the main function
    main()
