"""
run_robustness_analysis.py - Run pipeline across multiple seeds

This script runs the full pipeline with different random seeds to assess
model and optimization robustness.

Usage:
    python scripts/run_robustness_analysis.py
    python scripts/run_robustness_analysis.py --seeds 42 123 456
    python scripts/run_robustness_analysis.py --n_seeds 10
"""

import argparse
import subprocess
import yaml
from pathlib import Path
from datetime import datetime


def load_seeds_from_config(config_path='config.yaml'):
    """Load seeds from config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        seeds = config.get('experiments', {}).get('seeds', [42, 123, 456])
        return seeds
    except FileNotFoundError:
        print(f"Config not found, using default seeds")
        return [42, 123, 456, 789, 1011]


def run_pipeline_with_seed(seed, config_path='config.yaml'):
    """Run the full pipeline with a specific seed."""
    print("\n" + "="*70)
    print(f"RUNNING PIPELINE WITH SEED = {seed}")
    print("="*70)
    
    cmd = [
        "python", "scripts/run_full_pipeline.py",
        "--config", config_path,
        "--seed", str(seed)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"✗ Pipeline failed with seed {seed}")
        print(e.stderr)
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Run robustness analysis across multiple seeds'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='List of seeds to run (e.g., --seeds 42 123 456)'
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=None,
        help='Generate N random seeds'
    )
    args = parser.parse_args()
    
    # Determine which seeds to run
    if args.seeds:
        seeds = args.seeds
    elif args.n_seeds:
        import numpy as np
        np.random.seed(42)
        seeds = np.random.randint(1, 10000, size=args.n_seeds).tolist()
    else:
        seeds = load_seeds_from_config(args.config)
    
    # Start
    start_time = datetime.now()
    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS - MULTI-SEED PIPELINE")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds to run: {seeds}")
    print(f"Total runs: {len(seeds)}")
    print("="*70)
    
    # Run each seed
    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] Running seed {seed}...")
        success, error = run_pipeline_with_seed(seed, args.config)
        results.append({
            'seed': seed,
            'success': success,
            'error': error
        })
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("="*70)
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Successful runs: {successful}/{len(seeds)}")
    print(f"Failed runs: {failed}/{len(seeds)}")
    
    if failed > 0:
        print("\nFailed seeds:")
        for r in results:
            if not r['success']:
                print(f"  Seed {r['seed']}: {r['error']}")
    
    print("\nNext step: Run aggregation analysis")
    print("  python scripts/aggregate_seed_results.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()