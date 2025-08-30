import pandas as pd
import quantstats as qs
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys 

warnings.filterwarnings("ignore")

def load_and_process_tradelog(filename):
    if not os.path.exists(filename):
        print(f"'{filename}' not found. Please run a backtest first.")
        return None

    df = pd.read_csv(filename)
    print(f"Loaded '{filename}' with {len(df)} trades.")
    
    df.dropna(subset=['entry_date', 'exit_date', 'pnlcomm'], inplace=True)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['pnlcomm'] = pd.to_numeric(df['pnlcomm'])
    return df

def calculate_daily_returns(tradelog_df, initial_capital=1_000_000):
    # Convert tradelog to daily returns for quantstats
    if tradelog_df is None or tradelog_df.empty:
        return pd.Series(dtype=float)

    start_date = tradelog_df['entry_date'].min()
    end_date = tradelog_df['exit_date'].max()
    
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.Series(dtype=float)

    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    daily_pnl = pd.Series(0.0, index=date_range)
    
    pnl_by_exit_date = tradelog_df.groupby('exit_date')['pnlcomm'].sum()
    daily_pnl.update(pnl_by_exit_date)
    
    equity_curve = initial_capital + daily_pnl.cumsum()
    daily_returns = equity_curve.pct_change().fillna(0)
    
    return daily_returns

def display_performance_results(strategy_name, daily_returns, tradelog_df, risk_free_rate=0.0, output_filename=None):
    # Calculate and print performance metrics
    if output_filename:
        report_file_path = os.path.splitext(output_filename)[0] + '.txt'
        original_stdout = sys.stdout 
        print(f"\nWriting performance report to '{report_file_path}'...")
        f = open(report_file_path, 'w')
        sys.stdout = f
    
    try:
        print("\n" + "="*60)
        print(f"PERFORMANCE ANALYSIS: {strategy_name}")
        print("="*60)

        qs.reports.metrics(daily_returns, mode='full', rf=risk_free_rate)

        print("\n--- Trade Statistics ---")
        total_trades = len(tradelog_df)
        winning_trades = tradelog_df[tradelog_df['pnlcomm'] > 0]
        losing_trades = tradelog_df[tradelog_df['pnlcomm'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        avg_win = winning_trades['pnlcomm'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnlcomm'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnlcomm'].sum() / losing_trades['pnlcomm'].sum()) if len(losing_trades) > 0 and losing_trades['pnlcomm'].sum() != 0 else np.inf

        print(f"Total Trades:       {total_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print(f"Average Win:        ${avg_win:,.2f}")
        print(f"Average Loss:       ${avg_loss:,.2f}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print("="*60)
    
    finally:
        if output_filename:
            sys.stdout = original_stdout
            f.close()
            print("Report saved.")

    print("\nGenerating performance plots...")
    
    fig = qs.plots.snapshot(daily_returns, title=f"{strategy_name} - Performance Snapshot", show=False, rf=risk_free_rate)
    
    if output_filename:
        plot_file_path = os.path.splitext(output_filename)[0] + '.png'
        print(f"Saving performance plot to '{plot_file_path}'...")
        fig.savefig(plot_file_path, dpi=300, bbox_inches='tight')
        print("Plot saved.")

    plt.show()

def main():
    STRATEGY_NAME = 'ML-Enhanced Kalman Filter Strategy'
    TRADE_LOG_FILE = 'ml_backtest_results.csv'
    INITIAL_CAPITAL = 1_000_000
    RISK_FREE_RATE = 0.01
    OUTPUT_FILENAME = 'ml_strategy_report'

    tradelog_df = load_and_process_tradelog(TRADE_LOG_FILE)
    if tradelog_df is None:
        return
        
    daily_returns = calculate_daily_returns(tradelog_df, INITIAL_CAPITAL)
    
    if daily_returns.empty:
        print("\nCould not calculate daily returns.")
        return
    
    display_performance_results(STRATEGY_NAME, daily_returns, tradelog_df, RISK_FREE_RATE, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()