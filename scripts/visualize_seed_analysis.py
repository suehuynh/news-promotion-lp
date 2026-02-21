"""
visualize_seed_analysis.py - Create publication-quality visualizations

Generates figures comparing results across seeds.

Usage:
    python scripts/visualize_seed_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_metric_distributions(summary_df, output_dir='results/figures'):
    """Plot distributions of key metrics across seeds."""
    metrics = ['test_rmse', 'test_r2', 'opt_total_shares']
    labels = ['RMSE', 'R²', 'Total Predicted Shares']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, metric, label in zip(axes, metrics, labels):
        if metric in summary_df.columns:
            ax.hist(summary_df[metric], bins=10, alpha=0.7, edgecolor='black')
            ax.axvline(summary_df[metric].mean(), 
                      color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {summary_df[metric].mean():.2f}')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {label}', fontsize=13, fontweight='bold')
            ax.legend()
    
    plt.tight_layout()
    output_path = f"{output_dir}/metric_distributions.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Metric distributions saved to {output_path}")
    plt.close()


def plot_metric_boxplots(summary_df, output_dir='results/figures'):
    """Create boxplots for key metrics."""
    metrics = {
        'test_rmse': 'RMSE',
        'test_r2': 'R²',
        'opt_total_shares': 'Total Shares'
    }
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    
    for ax, (metric, label) in zip(axes, metrics.items()):
        if metric in summary_df.columns:
            bp = ax.boxplot([summary_df[metric]], 
                           labels=[label],
                           patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/seed_comparison_boxplot.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Boxplots saved to {output_path}")
    plt.close()


def plot_overlap_heatmap(overlap_df, output_dir='results/figures'):
    """Plot heatmap of article selection overlap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        overlap_df,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Overlapping Articles'},
        ax=ax,
        vmin=0,
        vmax=10
    )
    
    ax.set_title('Article Selection Overlap Across Seeds', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Seed', fontsize=12)
    
    plt.tight_layout()
    output_path = f"{output_dir}/stability_heatmap.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Overlap heatmap saved to {output_path}")
    plt.close()


def plot_seed_comparison(summary_df, output_dir='results/figures'):
    """Plot metrics across different seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('test_rmse', 'RMSE', axes[0, 0]),
        ('test_r2', 'R²', axes[0, 1]),
        ('opt_total_shares', 'Total Predicted Shares', axes[1, 0]),
        ('test_mae', 'MAE', axes[1, 1])
    ]
    
    for metric, label, ax in metrics:
        if metric in summary_df.columns:
            ax.plot(summary_df['seed'], summary_df[metric], 
                   'o-', linewidth=2, markersize=6)
            ax.axhline(summary_df[metric].mean(), 
                      color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {summary_df[metric].mean():.2f}')
            ax.fill_between(
                summary_df['seed'],
                summary_df[metric].mean() - summary_df[metric].std(),
                summary_df[metric].mean() + summary_df[metric].std(),
                alpha=0.2, color='red',
                label=f'±1 SD'
            )
            ax.set_xlabel('Random Seed', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f'{label} Across Seeds', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/seed_comparison_line.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Seed comparison plot saved to {output_path}")
    plt.close()


def main():
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Load data
    summary_path = 'results/aggregated/all_seeds_summary.csv'
    overlap_path = 'results/aggregated/selected_articles_overlap.csv'
    
    if not Path(summary_path).exists():
        print(f"✗ Summary file not found: {summary_path}")
        print("  Run aggregate_seed_results.py first")
        return
    
    summary_df = pd.read_csv(summary_path)
    print(f"✓ Loaded summary from {summary_path}")
    
    # Create output directory
    output_dir = 'results/figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_metric_distributions(summary_df, output_dir)
    plot_metric_boxplots(summary_df, output_dir)
    plot_seed_comparison(summary_df, output_dir)
    
    # Overlap heatmap (if available)
    if Path(overlap_path).exists():
        overlap_df = pd.read_csv(overlap_path, index_col=0)
        plot_overlap_heatmap(overlap_df, output_dir)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()