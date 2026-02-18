# Batch sensitivity analysis
"""
run_sensitivity_analysis.py - Pareto frontier analysis

Runs ILP optimization across different diversity levels to construct the 
Pareto frontier between predicted engagement and topic diversity.

"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import load_data, preprocess_features
from src.models.train import xgb_train
from src.models.predict import xgb_predict
from src.models.evaluate import *
from src.optimization.lp_solver import extended_news_solver
from scripts.run_full_pipeline import *


def run_sensitivity_analysis(config, diversity_levels):
    """
    Run ILP optimization across multiple diversity levels.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    diversity_levels : list of int
        Diversity levels to test (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
    
    Returns
    -------
    results : list of dict
        Results for each diversity level
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS - PARETO FRONTIER")
    print("="*60)
    print(f"Diversity levels to test: {diversity_levels}")
    print(f"Random seed: {config['data']['random_state']}")
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, categories_features = preprocess_features(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Step 2: Train model
    print("\n[2/4] Training XGBoost model...")
    model = xgb_train(
        X_train, 
        y_train,
        **config['model']['params']
    )
    y_pred = xgb_predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred, prefix="test_", task='regression', verbose=False)
    print(f"Model trained. Test RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.4f}")
    
    # Step 3: Extract topic indicators (ONCE)
    print("\n[3/4] Extracting topic indicators...")
    topic_indicators = {}
    for col in df.columns:
        if col.startswith(' data_channel_is'):
            topic_indicators[col] = df[col].values
    print(f"Extracted {len(topic_indicators)} categories")
    
    # Step 4: Run ILP for each diversity level
    print("\n[4/4] Running ILP optimization for each diversity level...")
    print("-" * 60)
    
    results = []
    
    for D in diversity_levels:
        print(f"\n  Testing D={D}...")
        
        # Get indicators
        lifestyle_indicator = topic_indicators.get(' data_channel_is_lifestyle', np.zeros(len(y_pred)))
        entertainment_indicator = topic_indicators.get(' data_channel_is_entertainment', np.zeros(len(y_pred)))
        bus_indicator = topic_indicators.get(' data_channel_is_bus', np.zeros(len(y_pred)))
        socmed_indicator = topic_indicators.get(' data_channel_is_socmed', np.zeros(len(y_pred)))
        tech_indicator = topic_indicators.get(' data_channel_is_tech', np.zeros(len(y_pred)))
        world_indicator = topic_indicators.get(' data_channel_is_world', np.zeros(len(y_pred)))
        other_indicator = topic_indicators.get(' data_channel_is_other', np.zeros(len(y_pred)))
        
        # Solve ILP
        selected_indices, status = extended_news_solver(
            shares=y_pred,
            lifestyle_indicator=lifestyle_indicator,
            entertainment_indicator=entertainment_indicator,
            bus_indicator=bus_indicator,
            socmed_indicator=socmed_indicator,
            tech_indicator=tech_indicator,
            world_indicator=world_indicator,
            other_indicator=other_indicator,
            K=config['optimization']['K'],
            diversity_lower_bound=D,
            solver_name="PULP_CBC_CMD",
            verbose=False
        )
        
        # Calculate metrics
        lp_shares = y_pred[selected_indices].sum()
        top10_shares = np.sort(y_pred)[::-1][:10].sum()
        diversity_cost = top10_shares - lp_shares
        pct_cost = (diversity_cost / top10_shares) * 100 if top10_shares > 0 else 0
        
        print(f"    Total shares (ILP Optimization): {lp_shares:.5f}")
        print(f"    Total shares (Naive Top 10): {top10_shares:.5f}")
        print(f"    Diversity cost: {diversity_cost:.5f} shares ({pct_cost:.2f}%)")
        
        # Count actual number of categories in selection
        selected_df = df.iloc[selected_indices]
        actual_categories = sum(
            selected_df[col].sum() > 0 
            for col in topic_indicators.keys()
        )
        
        # Store results
        result = {
            'diversity_level': D,
            'lp_shares': float(lp_shares),
            'top10_shares': float(top10_shares),
            'diversity_cost': float(diversity_cost),
            'pct_cost': float(pct_cost),
            'n_selected': len(selected_indices),
            'actual_categories': int(actual_categories),
            'selected_indices': selected_indices,
            'status': status
        }
        
        results.append(result)
        
        print(f"    Status: {status}")
        print(f"    Actual categories: {actual_categories}")
    
    print("\n" + "-" * 60)
    print("Sensitivity analysis complete")
    
    return results, metrics


def compute_marginal_costs(results):
    """Compute marginal cost of increasing diversity."""
    df = pd.DataFrame(results)
    df = df.sort_values('diversity_level')
    
    # Marginal cost = reduction in shares when increasing D by 1
    df['marginal_cost'] = -df['lp_shares'].diff()
    
    # Percentage cost
    max_shares = df['lp_shares'].max()
    df['pct_cost'] = (max_shares - df['lp_shares']) / max_shares * 100
    df['max_shares'] = max_shares
    return df


def save_sensitivity_results(results, metrics, config, output_dir='results/sensitivity'):
    """Save sensitivity analysis results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    seed = config['data']['random_state']
    
    # Save full results as JSON
    results_path = f"{output_dir}/sensitivity_seed_{seed}.json"
    with open(results_path, 'w') as f:
        # Convert numpy types
        results_clean = []
        for r in results:
            r_clean = {
                k: int(v) if isinstance(v, (np.integer, np.int64)) else # type: ignore
                   float(v) if isinstance(v, (np.floating, np.float64)) else
                   v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in r.items()
            }
            results_clean.append(r_clean)
        json.dump(results_clean, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Save summary as CSV
    summary_df = compute_marginal_costs(results)
    summary_path = f"{output_dir}/pareto_frontier_seed_{seed}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Pareto frontier saved to {summary_path}")
    
    # Save model metrics
    metrics_path = f"{output_dir}/model_metrics_seed_{seed}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Model metrics saved to {metrics_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Run sensitivity analysis for Pareto frontier'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--diversity_levels',
        type=int,
        nargs='+',
        default=None,
        help='Diversity levels to test (default: 0 to 7)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/sensitivity',
        help='Output directory'
    )
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config not found, using defaults")
        config = {
            'data': {'test_size': 0.2, 'random_state': args.seed},
            'model': {'params': {'n_estimators': 100, 'max_depth': 6}},
            'optimization': {'K': 10}
        }
    
    # Override seed
    config['data']['random_state'] = args.seed
    config['model']['params']['random_state'] = args.seed
    
    # Determine diversity levels
    diversity_levels = args.diversity_levels if args.diversity_levels else list(range(8))
    
    # Run analysis
    start_time = datetime.now()
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS - PARETO FRONTIER CONSTRUCTION")
    print("="*60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results, metrics = run_sensitivity_analysis(config, diversity_levels)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    summary_df = save_sensitivity_results(results, metrics, config, args.output)
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("PARETO FRONTIER SUMMARY")
    print("="*60)
    print(summary_df[['diversity_level', 'lp_shares', 
                     'actual_categories', 'pct_cost', 'max_shares']].to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Results saved to: {args.output}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()