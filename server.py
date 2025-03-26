import os
import sys
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.main import run_hedge_fund, create_workflow
from src.utils.analysts import ANALYST_CONFIG
from src.llm.models import get_model_info, LLM_ORDER

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api/analysts', methods=['GET'])
def get_analysts():
    """Return the list of available analysts with their details"""
    analysts = {}
    for key, config in ANALYST_CONFIG.items():
        # Get the first 500 characters of the agent function's docstring
        agent_func = config["agent_func"]
        strategy = agent_func.__doc__.strip() if agent_func.__doc__ else "No strategy information available."
        
        analysts[key] = {
            "id": key,
            "name": config["display_name"],
            "strategy": strategy,
            "order": config["order"]
        }
    
    return jsonify(analysts)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return the list of available LLM models"""
    models = []
    for display, value, _ in LLM_ORDER:
        model_info = get_model_info(value)
        if model_info:
            models.append({
                "id": value,
                "name": display,
                "provider": model_info.provider.value
            })
        else:
            models.append({
                "id": value,
                "name": display,
                "provider": "Unknown"
            })
    
    return jsonify(models)

class MockArgs:
    """Mock class to simulate command line arguments"""
    def __init__(self):
        self.show_reasoning = False
        self.show_agent_graph = False

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run a hedge fund simulation with the provided parameters"""
    data = request.json
    
    # Get required parameters
    tickers = data.get('tickers', [])
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    
    # Get optional parameters with defaults
    initial_cash = data.get('initial_cash', 100000.0)
    margin_requirement = data.get('margin_requirement', 0.0)
    show_reasoning = data.get('show_reasoning', False)
    selected_analysts = data.get('selected_analysts', [])
    model_choice = data.get('model', 'gpt-4o')
    
    # Set dates
    end_date = data.get('end_date') or datetime.now().strftime("%Y-%m-%d")
    if not data.get('start_date'):
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = data.get('start_date')
    
    # Initialize portfolio
    portfolio = {
        "cash": initial_cash,
        "margin_requirement": margin_requirement,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            } for ticker in tickers
        }
    }
    
    # Get model info
    model_info = get_model_info(model_choice)
    if model_info:
        model_provider = model_info.provider.value
    else:
        model_provider = "Unknown"
    
    try:
        # Run the hedge fund simulation
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_choice,
            model_provider=model_provider,
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "parameters": {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "initial_cash": initial_cash,
                "margin_requirement": margin_requirement,
                "selected_analysts": selected_analysts,
                "model": model_choice,
                "model_provider": model_provider
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
