import os
import json
from datetime import datetime
import sys
import argparse

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agents.jim_simons import jim_simons_agent
from graph.state import AgentState
from utils.display import print_trading_output

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Debug environment variables
print(f"Loaded .env from: {dotenv_path}")
print(f"APCA_API_KEY_ID: {os.environ.get('APCA_API_KEY_ID') is not None}")
print(f"APCA_API_SECRET_KEY: {os.environ.get('APCA_API_SECRET_KEY') is not None}")

def run_jim_simons(tickers, start_date, end_date, show_reasoning=True):
    """Run the Jim Simons agent on the specified tickers and date range."""
    print(f"Running Jim Simons agent on {tickers} from {start_date} to {end_date}")
    
    # Initialize state
    state = {
        "data": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {}
        },
        "metadata": {
            "model_name": "gpt-4o",
            "model_provider": "OpenAI",
            "show_reasoning": show_reasoning
        }
    }
    
    # Run the Jim Simons agent
    result = jim_simons_agent(AgentState(state))
    
    # Print the results
    signals = state["data"]["analyst_signals"]["jim_simons_agent"]
    print("\n==========       Jim Simons Agent       ==========")
    print(json.dumps(signals, indent=2))
    print("================================================")
    
    print("\nJim Simons Analysis Results:")
    for ticker, analysis in signals.items():
        print(f"\n{ticker}:")
        print(f"  Signal: {analysis['signal']}")
        print(f"  Confidence: {analysis['confidence']}")
        print(f"  Reasoning: {analysis['reasoning']}")
        print(f"  Market Regime: {analysis['regime_classification']}")
        
        if analysis['alpha_factors']:
            print("\n  Alpha Factors:")
            for factor, value in analysis['alpha_factors'].items():
                print(f"    {factor}: {value}")
        
        if analysis['statistical_metrics']:
            print("\n  Statistical Metrics:")
            for metric, value in analysis['statistical_metrics'].items():
                print(f"    {metric}: {value}")
    
    return signals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Jim Simons agent on specified tickers")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of tickers to analyze")
    parser.add_argument("--start-date", type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--show-reasoning", action="store_true", help="Show detailed reasoning in the output")
    
    args = parser.parse_args()
    tickers = args.tickers.split(",")
    
    run_jim_simons(tickers, args.start_date, args.end_date, args.show_reasoning)
