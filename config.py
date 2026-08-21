# Starter universe. Add/remove tickers as you like.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "JPM", "XOM", "COST",
]

MARKET_TICKER = "SPY"

# Prediction target:
# "Will the close be at least TARGET_RETURN higher after HORIZON trading days?"
HORIZON = 5
TARGET_RETURN = 0.01       # +1.0%

# Approximate round-trip transaction/slippage assumption used in diagnostics.
ROUND_TRIP_COST = 0.001    # 0.10%

# Only show model signals at/above this calibrated probability.
SIGNAL_THRESHOLD = 0.65

# How much history to request.
START_DATE = "2014-01-01"

RANDOM_STATE = 42
