import pandas as pd
import numpy as np
import glob
import os
from pykalman import KalmanFilter
from tqdm import tqdm
import talib
import warnings

warnings.filterwarnings("ignore")

# Date ranges for processing
CALCULATION_START_DATE = '2019-09-01'
SIGNAL_START_DATE = '2020-01-01'
SIGNAL_END_DATE = '2025-07-31'

def fetch_and_prepare_prices(path="/Users/shreyasshivapuji/Desktop/pairs-trading/data"):
    # Load and merge S&P 500 price data
    files = glob.glob(os.path.join(path, "sp500_prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No price files found in '{path}'.")
    
    dfs = [pd.read_csv(f, index_col='datetime', parse_dates=True) for f in sorted(files)]
    
    prices = pd.concat(dfs, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.loc[~prices.index.duplicated(keep='first')]
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices

def compute_kalman_signals(prices, stock1, stock2):
    # Generate z-score and hedge ratio using a Kalman filter
    pair_data = prices.loc[CALCULATION_START_DATE:SIGNAL_END_DATE, [stock1, stock2]].dropna()
    if len(pair_data) < 100: # Need enough data
        return None
    
    observation_matrix = np.vstack([
        pair_data[stock2].values, 
        np.ones(len(pair_data))
    ]).T[:, np.newaxis, :]

    kf = KalmanFilter(
        n_dim_state=2,
        n_dim_obs=1,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrix,
        observation_covariance=0.1,
        transition_covariance=1e-11 * np.eye(2)
    )
    
    state_means, _ = kf.filter(pair_data[stock1].values)
    
    spread = pair_data[stock1] - (state_means[:, 0] * pair_data[stock2])
    z_score = (spread - spread.rolling(window=60).mean()) / spread.rolling(window=60).std()
    
    signals = pd.DataFrame({
        'stock1_price': pair_data[stock1],
        'stock2_price': pair_data[stock2],
        'z_score': z_score,
        'hedge_ratio': state_means[:, 0]
    }, index=pair_data.index)
    
    return signals.loc[SIGNAL_START_DATE:SIGNAL_END_DATE]

def enrich_with_features(signals, prices, stock1, stock2):
    # Add technical analysis features for the ML model
    if signals is None or signals.empty:
        return None
    
    features = signals.copy()
    spread = features['stock1_price'] - (features['hedge_ratio'] * features['stock2_price'])
    
    # Technical indicators
    features['stock1_rsi'] = talib.RSI(features['stock1_price'], timeperiod=14)
    features['stock1_atr_norm'] = talib.ATR(features['stock1_price'], features['stock1_price'], features['stock1_price'], 14) / features['stock1_price']
    
    features['stock2_rsi'] = talib.RSI(features['stock2_price'], timeperiod=14)
    features['stock2_atr_norm'] = talib.ATR(features['stock2_price'], features['stock2_price'], features['stock2_price'], 14) / features['stock2_price']
    
    # Spread volatility
    features['spread_volatility'] = spread.diff().abs().rolling(window=30).std().shift(1)
    
    # Rolling correlation
    full_prices = prices.loc[CALCULATION_START_DATE:SIGNAL_END_DATE, [stock1, stock2]].dropna()
    features['correlation_30d'] = full_prices[stock1].rolling(window=30).corr(full_prices[stock2]).shift(1)
    
    return features

def main(price_path="/Users/shreyasshivapuji/Desktop/pairs-trading/data", pairs_file='/Users/shreyasshivapuji/Desktop/pairs-trading/final_candidate_pairs.csv'):
    try:
        prices = fetch_and_prepare_prices(price_path)
        pairs_df = pd.read_csv(pairs_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    output_dir = "kalman_with_features"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(pairs_df)} pairs from '{pairs_file}'...")
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Generating Signals & Features"):
        pair_string = row['Pair']
        stock1, stock2 = pair_string.split('-')
        
        try:
            kalman_signals = compute_kalman_signals(prices, stock1, stock2)
            if kalman_signals is None:
                continue
                
            ml_features = enrich_with_features(kalman_signals.copy(), prices, stock1, stock2)
            if ml_features is not None:
                ml_features.to_csv(os.path.join(output_dir, f"{pair_string}.csv"))
                
        except Exception as e:
            print(f"Skipping pair {pair_string} due to error: {str(e)}")
            continue

    print(f"\nSignal and feature generation complete. Files saved in '{output_dir}'.")

if __name__ == "__main__":
    main()