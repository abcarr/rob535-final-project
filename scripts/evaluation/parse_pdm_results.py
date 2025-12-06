#!/usr/bin/env python3
"""
Parse PDM-Score evaluation results and display in a formatted table.

Usage:
    python parse_pdm_results.py <path_to_csv>
    python parse_pdm_results.py exp/eval/deformable_ablation/2025.12.05.10.13.21.csv
"""

import pandas as pd
import sys
from pathlib import Path


def parse_pdm_results(csv_path):
    """Parse PDM-Score CSV and display formatted results."""
    
    # Read CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Get the average row (last row with token='average')
    avg_row = df[df['token'] == 'average'].iloc[0]
    
    # Get statistics
    total_scenarios = len(df) - 1  # Exclude average row
    valid_scenarios = df[df['token'] != 'average']['valid'].sum()
    
    print("=" * 80)
    print(f"PDM-Score Evaluation Results")
    print("=" * 80)
    print(f"CSV File: {csv_path}")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Valid Scenarios: {valid_scenarios} ({valid_scenarios/total_scenarios*100:.1f}%)")
    print("=" * 80)
    print()
    
    # Overall Score
    print(f"{'OVERALL PDM-SCORE:':<40} {avg_row['score']:.4f} ({avg_row['score']*100:.2f}%)")
    print()
    
    # Detailed Metrics Table
    print("Detailed Metrics:")
    print("-" * 80)
    print(f"{'Metric':<45} {'Score':<10} {'Percentage'}")
    print("-" * 80)
    
    metrics = [
        ('no_at_fault_collisions', 'NC (No At-Fault Collision)'),
        ('drivable_area_compliance', 'DAC (Drivable Area Compliance)'),
        ('driving_direction_compliance', 'Driving Direction Compliance'),
        ('ego_progress', 'EP (Ego Progress)'),
        ('time_to_collision_within_bound', 'TTC (Time-to-Collision)'),
        ('comfort', 'Comf. (Comfort)'),
    ]
    
    for metric_key, metric_name in metrics:
        score = avg_row[metric_key]
        print(f"{metric_name:<45} {score:<10.4f} {score*100:>6.2f}%")
    
    print("-" * 80)
    print()
    
    # Performance Summary
    print("Performance Summary:")
    print("-" * 80)
    
    # Find strengths (>= 90%)
    strengths = []
    weaknesses = []
    
    for metric_key, metric_name in metrics:
        score = avg_row[metric_key]
        if score >= 0.90:
            strengths.append(f"  ✅ {metric_name}: {score*100:.1f}%")
        elif score < 0.70:
            weaknesses.append(f"  ⚠️  {metric_name}: {score*100:.1f}%")
    
    if strengths:
        print("Strengths:")
        for s in strengths:
            print(s)
        print()
    
    if weaknesses:
        print("Areas for Improvement:")
        for w in weaknesses:
            print(w)
        print()
    
    # Failure analysis
    failed_scenarios = df[(df['token'] != 'average') & (df['score'] == 0.0)]
    if len(failed_scenarios) > 0:
        print(f"Failed Scenarios: {len(failed_scenarios)} ({len(failed_scenarios)/total_scenarios*100:.1f}%)")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_pdm_results.py <path_to_csv>")
        print("Example: python parse_pdm_results.py exp/eval/deformable_ablation/2025.12.05.10.13.21.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    parse_pdm_results(csv_path)
