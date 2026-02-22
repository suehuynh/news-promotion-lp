#!/bin/bash
# run_all_with_seeds.sh - Run both analyses with same seeds

# Configuration
SEEDS=(10 42 123 456 789 1010 1234 2026 3443 7890)
CONFIG="config.yaml"

echo "========================================================================"
echo "RUNNING COMPLETE ANALYSIS"
echo "========================================================================"
echo "Seeds: ${SEEDS[@]}"
echo ""

# Phase 1: Robustness (base pipeline)
echo "========================================================================"
echo "PHASE 1: ROBUSTNESS ANALYSIS"
echo "========================================================================"
for seed in "${SEEDS[@]}"; do
    echo "Running pipeline for seed ${seed}..."
    python scripts/run_full_pipeline.py --config ${CONFIG} --seed ${seed}
done

# Phase 2: Sensitivity
echo ""
echo "========================================================================"
echo "PHASE 2: SENSITIVITY ANALYSIS"
echo "========================================================================"
for seed in "${SEEDS[@]}"; do
    echo "Running sensitivity for seed ${seed}..."
    python scripts/run_sensitivity_analysis.py --config ${CONFIG} --seed ${seed}
done

# Phase 3: Aggregation
echo ""
echo "========================================================================"
echo "PHASE 3: AGGREGATION"
echo "========================================================================"
python scripts/aggregate_seed_results.py
python scripts/aggregate_pareto_curves.py
python scripts/visualize_seed_analysis.py

echo ""
echo "========================================================================"
echo "COMPLETE"
echo "========================================================================"
echo "Results in: results/"