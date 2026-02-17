"""
run_full_pipeline.py - End-to-end reproducible research pipeline

This script executes the complete research workflow:
1. Data loading and preprocessing
2. Predictive modeling (XGBoost)
3. ILP optimization
4. Pareto frontier analysis
5. Results generation

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --config config.yaml
    python scripts/run_full_pipeline.py --seed 42
"""

# Imports
import os
import sys
import argparse
from pathlib import Path
import yaml
import json
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


# === PART 1: CONFIGURATION ===
def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration...")
        return {
            'data': {'test_size': 0.2, 'random_state': 42},
            'model': {'params': {'n_estimators': 100}},
            'optimization': {'K': 10}
        }

# === PART 2: SETUP ===
def setup_directories(config):
    """Create necessary directories for results."""
    dirs_to_create = [
        config['paths']['figures'],
        config['paths']['tables'],
        'results/logs',
        'models/saved',
        'data/processed'
    ]
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)  
    
    print(f" Created {len(dirs_to_create)} directories")  


# === PART 3: DATA PIPELINE ===
def run_data_pipeline(config):
    """Load and preprocess data.
    
    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    topic_indicators : dict of arrays
    """
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*60)

    df = load_data()
    print(f"Loaded {len(df)} articles")
    
    # Preprocess
    print("Preprocessing features...")
    X_train, X_test, y_train, y_test, categories_features = preprocess_features(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Extract topic indicators
    print("Extracting topic indicators...")
    topic_cols = [c for c in df.columns if c.startswith(' data_channel_is')]
    topic_indicators = {
        col: df[col].values for col in topic_cols
    }
    print(f"Extracted {len(topic_indicators)} topic categories")
    
    return X_train, X_test, y_train, y_test, topic_indicators


# === PART 4: MODELING PIPELINE ===
def run_modeling_pipeline(X_train, y_train, X_test, y_test, config):
    """Train and evaluate predictive model."""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    # Train model
    print("Training XGBoost model...")
    model = xgb_train(
        X_train, 
        y_train, 
        **config['model']['params']
    )
    print("Model trained")
    
    # Predict
    print("Generating predictions...")
    y_pred = xgb_predict(model, X_test)
    print("Predictions generated")
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(y_test, y_pred, y_pred_proba=None, task='regression', 
                   prefix="test__", verbose=True)
    save_metrics_to_file(metrics, filepath="results/model_metrics.json")
    return model, y_pred, metrics


# === PART 5: OPTIMIZATION PIPELINE ===
def run_optimization_pipeline(predictions, indicators, config):
    """Run ILP optimization and Pareto analysis."""
    print("\n" + "="*60)
    print("STEP 3: ILP OPTIMIZATION")
    print("="*60)
    
    # Extract indicators from the dict
    lifestyle_indicator = indicators.get(' data_channel_is_lifestyle', np.zeros(len(predictions)))
    entertainment_indicator = indicators.get(' data_channel_is_entertainment', np.zeros(len(predictions)))
    bus_indicator = indicators.get(' data_channel_is_bus', np.zeros(len(predictions)))
    socmed_indicator = indicators.get(' data_channel_is_socmed', np.zeros(len(predictions)))
    tech_indicator = indicators.get(' data_channel_is_tech', np.zeros(len(predictions)))
    world_indicator = indicators.get(' data_channel_is_world', np.zeros(len(predictions)))
    other_indicator = indicators.get(' data_channel_is_other', np.zeros(len(predictions)))
    
    print(f"Solving ILP with K={config['optimization']['K']}...")
    selected_indices, status = extended_news_solver(
        shares=predictions,
        lifestyle_indicator=lifestyle_indicator,
        entertainment_indicator=entertainment_indicator,
        bus_indicator=bus_indicator,
        socmed_indicator=socmed_indicator,
        tech_indicator=tech_indicator,
        world_indicator=world_indicator,
        other_indicator=other_indicator,
        K=config['optimization']['K'],
        diversity_lower_bound=config['optimization'].get('diversity_lower_bound', 1),
        solver_name="PULP_CBC_CMD",
        verbose=False
    )
    
    print(f"ILP solved: {status}")
    print(f"Selected {len(selected_indices)} articles")
    
    # Calculate results
    total_shares = predictions[selected_indices].sum()
    print(f"Total predicted shares: {total_shares:.0f}")
    
    return {
        'selected_indices': selected_indices,
        'status': status,
        'total_shares': total_shares,
        'n_selected': len(selected_indices)
    }


# === PART 6: RESULTS GENERATION ===
def save_results(results, config):
    """Save all results to files."""
    print("\n" + "="*60)
    print("STEP 4: SAVING RESULTS")
    print("="*60)
    
    # Create timestamp for this run
    timestamp = results['timestamp']
    
    # Save metrics as JSON
    metrics_path = f"{config['paths']['results']}/model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results['model_metrics'], f, indent=2)
    print(f"Model metrics saved to {metrics_path}")
    
    # Save optimization results
    opt_path = f"{config['paths']['results']}/optimization_results.json"
    with open(opt_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        opt_results = {
            k: int(v) if isinstance(v, (np.integer, np.int64)) else  # type: ignore
               float(v) if isinstance(v, (np.floating, np.float64)) else
               v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results['optimization'].items()
        }
        json.dump(opt_results, f, indent=2)
    print(f"Optimization results saved to {opt_path}")
    
    # Save full config used
    config_path = f"{config['paths']['results']}/config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(results['config'], f)
    print(f"Configuration saved to {config_path}")


  # === PART 7: MAIN EXECUTION ===
def main():
    """Main pipeline orchestration."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run full research pipeline'
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
        default=None,
        help='Random seed (overrides config)'
    )
    args = parser.parse_args()
    
    # Start
    start_time = datetime.now()
    print("\n" + "="*60)
    print("NEWS DIVERSITY ILP - FULL PIPELINE")
    print("="*60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['data']['random_state'] = args.seed
        config['model']['random_state'] = args.seed
    
    # Setup
    setup_directories(config)
    
    # Run pipelines
    X_train, X_test, y_train, y_test, indicators = run_data_pipeline(config)
    model, y_pred, metrics = run_modeling_pipeline(
        X_train, y_train, X_test, y_test, config
    )
    optimization_results = run_optimization_pipeline(
        y_pred, indicators, config
    )
    
    # Save everything
    all_results = {
        'config': config,
        'model_metrics': metrics,
        'optimization': optimization_results,
        'timestamp': start_time.isoformat()
    }
    save_results(all_results, config)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Results saved to: {config['paths']['results']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
