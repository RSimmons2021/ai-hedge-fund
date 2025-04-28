"""Constants and utilities related to analysts configuration."""

from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.cathie_wood import cathie_wood_agent
from agents.charlie_munger import charlie_munger_agent
from agents.fundamentals import fundamentals_agent
from agents.phil_fisher import phil_fisher_agent
from agents.sentiment import sentiment_agent
from agents.stanley_druckenmiller import stanley_druckenmiller_agent
from agents.technicals import technical_analyst_agent
from agents.valuation import valuation_agent
from agents.jim_simons import jim_simons_agent
from agents.warren_buffett import warren_buffett_agent
from agents.crypto_analyst import crypto_analyst_agent

# Define analyst configuration - single source of truth
ANALYST_CONFIG = {
    "ben_graham": {
        "display_name": "Ben Graham",
        "function": ben_graham_agent,
        "order": 0,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "function": bill_ackman_agent,
        "order": 1,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "function": cathie_wood_agent,
        "order": 2,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "function": charlie_munger_agent,
        "order": 3,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "function": phil_fisher_agent,
        "order": 4,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "function": stanley_druckenmiller_agent,
        "order": 5,
    },
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "function": warren_buffett_agent,
        "order": 6,
    },
    "jim_simons": {
        "display_name": "Jim Simons",
        "function": jim_simons_agent,
        "order": 7,
    },
    "crypto_analyst": {
        "display_name": "Crypto Analyst",
        "function": crypto_analyst_agent,
        "order": 8,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "function": technical_analyst_agent,
        "order": 9,
    },
    "fundamental_analyst": {
        "display_name": "Fundamental Analyst",
        "function": fundamentals_agent,
        "order": 10,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "function": sentiment_agent,
        "order": 11,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "function": valuation_agent,
        "order": 12,
    },
    
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["function"]) for key, config in ANALYST_CONFIG.items()}
