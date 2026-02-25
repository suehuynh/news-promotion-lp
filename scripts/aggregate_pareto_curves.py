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

    # Marginal cost = f(D-1) - f(D)
    stats['marginal_cost_mean'] = -stats['lp_shares_mean'].diff()
    stats.loc[stats['diversity_level'] == 0, 'marginal_cost_mean'] = 0
    
    
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

def plot_cost_analysis(stats_df, output_path='results/figures/cost_analysis.pdf'):
    """Plot comprehensive cost analysis: diversity cost vs. marginal cost."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    D = stats_df['diversity_level']
    shares = stats_df['lp_shares_mean']
    diversity_cost = stats_df['diversity_cost_mean']
    marginal_cost = stats_df['marginal_cost_mean']
    pct_cost = stats_df['pct_cost_mean']
    
    # ============================================
    # Plot 1: Pareto Frontier (Top Left)
    # ============================================
    ax1 = axes[0, 0]
    ax1.plot(D, shares, 'o-', linewidth=2.5, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Diversity Level (D)', fontsize=12)
    ax1.set_ylabel('Total Predicted Shares', fontsize=12)
    ax1.set_title('Pareto Frontier: Shares vs. Diversity', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(D)
    
    # Add max shares line
    max_shares = shares.max()
    ax1.axhline(y=max_shares, color='red', linestyle='--', 
                linewidth=1, alpha=0.5, label=f'Maximum: {max_shares:.0f}')
    ax1.legend()
    
    # ============================================
    # Plot 2: Diversity Cost (Top Right)
    # ============================================
    ax2 = axes[0, 1]
    ax2.plot(D, diversity_cost, 'o-', linewidth=2.5, markersize=8, color='#E27D60')
    ax2.fill_between(D, 0, diversity_cost, alpha=0.3, color='#E27D60')
    ax2.set_xlabel('Diversity Level (D)', fontsize=12)
    ax2.set_ylabel('Diversity Cost (Shares)', fontsize=12)
    ax2.set_title('Share Cost per Diversity Level\n[Top-10 Shares - f(D)]', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(D)
    
    # Add percentage annotations
    for d, cost, pct in zip(D, diversity_cost, pct_cost):
        if d > 0:  # Skip D=0
            ax2.text(d, cost + 100, f'{pct:.1f}%', 
                    ha='center', va='bottom', fontsize=9, color='#C44536')
    
    # ============================================
    # Plot 3: Marginal Cost (Bottom Left)
    # ============================================
    ax3 = axes[1, 0]
    
    # Color bars by magnitude
    valid = marginal_cost > 0  # Exclude D=0
    colors = []
    for cost in marginal_cost[valid]:
        if cost < 700:
            colors.append('#2ECC71')  # Green (cheap)
        elif cost < 1000:
            colors.append('#F39C12')  # Orange (moderate)
        else:
            colors.append('#E74C3C')  # Red (expensive)
    
    bars = ax3.bar(D[valid], marginal_cost[valid], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Diversity Level (D)', fontsize=12)
    ax3.set_ylabel('Marginal Cost (Shares)', fontsize=12)
    ax3.set_title('Marginal Cost of Each Diversity Level\n[f(D-1) - f(D)]', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(D)
    
    # Add value labels on bars
    for d, cost in zip(D[valid], marginal_cost[valid]):
        ax3.text(d, cost + 50, f'{cost:.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', label='Low (<700)'),
        Patch(facecolor='#F39C12', label='Moderate (700-1000)'),
        Patch(facecolor='#E74C3C', label='High (>1000)')
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # ============================================
    # Plot 4: Cost Comparison (Bottom Right)
    # ============================================
    ax4 = axes[1, 1]
    
    # Plot both costs on same axis (normalized)
    ax4_twin = ax4.twinx()
    
    # Diversity cost (cumulative)
    line1 = ax4.plot(D, diversity_cost, 'o-', linewidth=2.5, markersize=8, 
                     color='#E27D60', label='Total Cost (vs. Naive)')
    ax4.set_xlabel('Diversity Level (D)', fontsize=12)
    ax4.set_ylabel('Total Diversity Cost (Shares)', fontsize=12, color='#E27D60')
    ax4.tick_params(axis='y', labelcolor='#E27D60')
    
    # Marginal cost (incremental)
    line2 = ax4_twin.plot(D[valid], marginal_cost[valid], 's-', linewidth=2.5, 
                          markersize=8, color='#2E86AB', label='Marginal Cost (Per Level)')
    ax4_twin.set_ylabel('Marginal Cost (Shares)', fontsize=12, color='#2E86AB')
    ax4_twin.tick_params(axis='y', labelcolor='#2E86AB')
    
    ax4.set_title('Cost Comparison: Total vs. Marginal', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(D)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, 
                    hspace=0.35, wspace=0.25)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Cost analysis plot saved to {output_path}")
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
    plot_cost_analysis(stats_df)
    
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("Figures saved to: results/figures")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()