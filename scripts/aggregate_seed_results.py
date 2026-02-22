"""
aggregate_seed_results.py - Aggregate results across seeds

Combines results from multiple seed runs and performs statistical analysis.

Usage:
    python scripts/aggregate_seed_results.py
    python scripts/aggregate_seed_results.py --output results/aggregated
"""

import json
import yaml
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_seed_results(results_dir='results/seed_runs'):
    """Load all seed results into a list of dictionaries."""
    seed_dirs = sorted(Path(results_dir).glob('seed_*'))
    
    all_results = []
    
    for seed_dir in seed_dirs:
        seed_num = int(seed_dir.name.split('_')[1])
        
        # Load metrics
        metrics_path = seed_dir / 'model_metrics.json'
        if not metrics_path.exists():
            print(f"⚠ Skipping {seed_dir.name}: metrics not found")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load optimization results
        opt_path = seed_dir / 'optimization_results.json'
        with open(opt_path, 'r') as f:
            opt_results = json.load(f)
        
        # Combine
        result = {
            'seed': seed_num,
            **metrics,
            **{f"opt_{k}": v for k, v in opt_results.items()}
        }
        
        all_results.append(result)
    
    return all_results


def create_summary_dataframe(all_results):
    """Create summary DataFrame from all seed results."""
    df = pd.DataFrame(all_results)
    
    # Sort by seed
    df = df.sort_values('seed').reset_index(drop=True)
    
    return df


def compute_statistics(df, metric_columns):
    """Compute summary statistics for metrics."""
    stats = []
    
    for col in metric_columns:
        if col in df.columns:
            stats.append({
                'metric': col,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan  # Coefficient of variation
            })
    
    return pd.DataFrame(stats)


def analyze_article_overlap(results_dir='results/seed_runs'):
    """Analyze overlap in selected articles across seeds."""
    seed_dirs = sorted(Path(results_dir).glob('seed_*'))
    
    selected_sets = {}
    
    for seed_dir in seed_dirs:
        seed_num = int(seed_dir.name.split('_')[1])
        opt_path = seed_dir / 'optimization_results.json'
        
        if opt_path.exists():
            with open(opt_path, 'r') as f:
                opt_results = json.load(f)
                selected_sets[seed_num] = set(opt_results.get('selected_indices', []))
    
    # Compute pairwise overlaps
    seeds = sorted(selected_sets.keys())
    n_seeds = len(seeds)
    
    overlap_matrix = np.zeros((n_seeds, n_seeds))
    
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds):
            if i <= j:
                set1 = selected_sets[seed1]
                set2 = selected_sets[seed2]
                overlap = len(set1 & set2)  # Intersection
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
    
    # Create DataFrame
    overlap_df = pd.DataFrame(
        overlap_matrix,
        index=[f"seed_{s}" for s in seeds],
        columns=[f"seed_{s}" for s in seeds]
    )
    
    return overlap_df


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results across seeds'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/seed_runs',
        help='Directory containing seed results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/aggregated',
        help='Output directory for aggregated results'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AGGREGATING SEED RESULTS")
    print("="*60)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print(f"Loading results from {args.input}...")
    all_results = load_seed_results(args.input)
    print(f"✓ Loaded {len(all_results)} seed runs")
    
    # Create summary DataFrame
    print("\nCreating summary DataFrame...")
    summary_df = create_summary_dataframe(all_results)
    
    # Save summary
    summary_path = f"{args.output}/all_seeds_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to {summary_path}")
    
    # Compute statistics
    print("\nComputing statistics...")
    metric_columns = [
        'test_rmse', 'test_mae', 'test_r2', 'test_mse',
        'opt_total_shares', 'opt_n_selected'
    ]
    stats_df = compute_statistics(summary_df, metric_columns)
    
    # Save statistics
    stats_path = f"{args.output}/summary_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Statistics saved to {stats_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(stats_df.to_string(index=False))
    
    # Analyze article overlap
    print("\n" + "="*60)
    print("ANALYZING ARTICLE SELECTION OVERLAP")
    print("="*60)
    overlap_df = analyze_article_overlap(args.input)
    overlap_path = f"{args.output}/selected_articles_overlap.csv"
    overlap_df.to_csv(overlap_path)
    print(f"✓ Overlap matrix saved to {overlap_path}")
    
    # Print overlap summary
    # Get upper triangle (excluding diagonal)
    n = len(overlap_df)
    upper_triangle = overlap_df.values[np.triu_indices(n, k=1)]
    print(f"\nArticle overlap statistics (out of 10 articles):")
    print(f"  Mean overlap: {upper_triangle.mean():.2f}")
    print(f"  Std overlap: {upper_triangle.std():.2f}")
    print(f"  Min overlap: {upper_triangle.min():.0f}")
    print(f"  Max overlap: {upper_triangle.max():.0f}")
    
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
    