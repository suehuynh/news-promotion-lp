"""
aggregate_pareto_curves.py - Combine Pareto frontiers across seeds

Aggregates sensitivity analysis results from multiple seeds to show
robustness of the diversity-engagement trade-off.

Usage:
    python scripts/aggregate_pareto_curves.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_pareto_curves(results_dir='results/seed_runs'):
    """Load all Pareto frontier CSV files."""
    seed_dirs = sorted(Path(results_dir).glob('seed_*'))
    
    all_curves = []
    
    for seed_dir in seed_dirs:
        pareto_file = seed_dir / 'pareto_frontier.csv'
        
        if pareto_file.exists():
            seed_num = int(seed_dir.name.split('_')[1])
            df = pd.read_csv(pareto_file)
            df['seed'] = seed_num
            all_curves.append(df)
    
    if not all_curves:
        print("No Pareto frontier files found!")
        return None
    
    # Combine all
    combined = pd.concat(all_curves, ignore_index=True)
    return combined


def compute_pareto_statistics(combined_df):
    """Compute mean and std for each diversity level."""
    stats = combined_df.groupby('diversity_level').agg({
        'lp_shares': ['mean', 'std', 'min', 'max', 'count'],
        'actual_categories': ['mean', 'std'],
        'pct_cost': ['mean', 'std'],
        'diversity_cost': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    return stats


def plot_pareto_with_confidence(stats_df, output_path='results/figures/pareto_frontier_robust.pdf'):
    """Plot Pareto frontier with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    D = stats_df['diversity_level']
    mean_shares = stats_df['lp_shares_mean']
    std_shares = stats_df['lp_shares_std']
    
    # Main line
    ax.plot(D, mean_shares, 'o-', linewidth=2.5, markersize=8, 
            label='Mean', color='#2E86AB')
    
    # Confidence band (±1 SD)
    ax.fill_between(D, 
                    mean_shares - std_shares,
                    mean_shares + std_shares,
                    alpha=0.3, color='#2E86AB',
                    label='±1 SD')
    
    ax.set_xlabel('Minimum Diversity Level (D)', fontsize=13)
    ax.set_ylabel('LP-Optimized Shares', fontsize=13)
    ax.set_title('Pareto Frontier: Diversity vs. Engagement across seeds', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Robust Pareto plot saved to {output_path}")
    plt.close()


def plot_all_seeds_pareto(combined_df, output_path='results/figures/pareto_all_seeds.pdf'):
    """Plot individual Pareto curves for each seed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seeds = combined_df['seed'].unique()
    
    for seed in seeds:
        seed_data = combined_df[combined_df['seed'] == seed]
        ax.plot(seed_data['diversity_level'], 
               seed_data['lp_shares'],
               'o-', alpha=0.5, linewidth=1.5, label=f'Seed {seed}')
    
    # Add mean curve
    mean_curve = combined_df.groupby('diversity_level')['lp_shares'].mean()
    ax.plot(mean_curve.index, mean_curve.values, 
           'k-', linewidth=3, label='Mean', zorder=10)
    
    ax.set_xlabel('Minimum Diversity Level (D)', fontsize=13)
    ax.set_ylabel('LP-Optimized Shares', fontsize=13)
    ax.set_title('Pareto Frontiers Across All Seeds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ All seeds Pareto plot saved to {output_path}")
    plt.close()

def plot_marginal_cost(stats_df, output_path='results/figures/pareto_marginal_cost.pdf'):
    """Plot marginal cost of increasing diversity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    D = stats_df['diversity_level']
    marginal_diversity_cost = stats_df['diversity_cost_mean']
    
    ax.plot(D, marginal_diversity_cost, 'o-', linewidth=2.5, markersize=8, color='#E27D60')
    
    ax.set_xlabel('Minimum Diversity Level (D)', fontsize=13)
    ax.set_ylabel('Average Share Cost', fontsize=13)
    ax.set_title('Marginal Cost of Increasing Diversity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Marginal cost plot saved to {output_path}")
    plt.close()

def main():
    print("\n" + "="*60)
    print("AGGREGATING PARETO CURVES")
    print("="*60)
    
    # Load all Pareto curves
    print("\nLoading Pareto frontier files...")
    combined_df = load_pareto_curves()
    
    if combined_df is None:
        return
    
    n_seeds = combined_df['seed'].nunique()
    print(f"✓ Loaded Pareto curves from {n_seeds} seeds")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats_df = compute_pareto_statistics(combined_df)
    
    # Save aggregated results
    output_dir = 'results/aggregated'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stats_path = f"{output_dir}/pareto_all_seeds.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Aggregated Pareto saved to {stats_path}")
    
    # Save all individual curves
    all_curves_path = f"{output_dir}/pareto_all_curves_individual.csv"
    combined_df.to_csv(all_curves_path, index=False)
    print(f"✓ All individual curves saved to {all_curves_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PARETO FRONTIER SUMMARY")
    print("="*60)
    print(stats_df[['diversity_level', 
                    'lp_shares_mean', 
                    'lp_shares_std',
                    'pct_cost_mean']].to_string(index=False))
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_pareto_with_confidence(stats_df)
    plot_all_seeds_pareto(combined_df)
    plot_marginal_cost(stats_df)
    
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("Figures saved to: results/figures")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()