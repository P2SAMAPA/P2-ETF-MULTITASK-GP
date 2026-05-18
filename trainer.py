import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from multitask_gp import train_individual_gps, predict_individual

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_dataset(returns_df, macro_df, window):
    ret_win = returns_df.iloc[-window:]
    macro_win = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret_win.index, columns=config.MACRO_COLUMNS)
    common_idx = ret_win.index.intersection(macro_win.index)
    ret_win = ret_win.loc[common_idx]
    macro_win = macro_win.loc[common_idx]
    t = np.arange(len(ret_win)).reshape(-1,1) / len(ret_win)
    macro_vals = macro_win.values
    X = np.hstack([t, macro_vals])
    Y = ret_win.shift(-1).dropna().values
    X = X[:-1]
    return X, Y

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Multi‑task GP) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            X, Y = create_dataset(returns, macro, win)
            if X.shape[0] < 20:
                continue
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
            num_tasks = Y.shape[1]
            models, likelihoods = train_individual_gps(X_t, Y_t, num_tasks,
                                                       lr=config.LR,
                                                       iterations=config.ITERATIONS)
            test_X = X[-1:].reshape(1, -1)
            test_X_t = torch.tensor(test_X, dtype=torch.float32).to(device)
            mean = predict_individual(models, likelihoods, test_X_t, num_tasks)
            scores = {tickers[i]: mean[i] for i in range(num_tasks)}
            window_results[win] = scores
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by GP mean: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/multitask_gp_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Multi‑task Gaussian Process Engine complete ===")

if __name__ == "__main__":
    main()
