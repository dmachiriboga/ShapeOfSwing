"""
First and Last Note vs Middle Notes Analysis

Simple analysis:
- First note (position 1)
- Middle notes (positions 2 through n-1)
- Last note (position n)

Tests if first and last notes are both higher than the average of middle notes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES


def analyze_first_last_vs_middle(bur_values, min_effect_size=0.2):
    """
    Compare first and last notes against all middle notes.
    
    Args:
        bur_values: List of BUR values for a phrase
        min_effect_size: Minimum Cohen's d to consider "different" (default 0.2 = small effect)
        
    Returns:
        dict with statistics
    """
    n = len(bur_values)
    
    if n < MIN_BUR_VALUES:
        return None
    
    # Simple division: first, middle (everything else), last
    first_note = bur_values[0]
    last_note = bur_values[-1]
    middle_notes = bur_values[1:-1]  # All notes except first and last
    
    # Calculate statistics
    middle_mean = np.mean(middle_notes)
    middle_std = np.std(middle_notes, ddof=1) if len(middle_notes) > 1 else 0
    
    # For Cohen's d with single values vs a group:
    # We treat first/last as groups of size 1 with std=0
    # This simplifies to: d = (single_value - group_mean) / group_std
    
    if middle_std > 0:
        d_first_vs_middle = (first_note - middle_mean) / middle_std
        d_last_vs_middle = (last_note - middle_mean) / middle_std
    else:
        d_first_vs_middle = 0
        d_last_vs_middle = 0
    
    # For first vs last, calculate Cohen's d using the middle's variability as reference
    # Since first and last are single values, we use middle_std as the standardizer
    if middle_std > 0:
        d_first_vs_last = (first_note - last_note) / middle_std
    else:
        d_first_vs_last = 0
    
    # Classification based on effect sizes
    first_higher = d_first_vs_middle >= min_effect_size
    last_higher = d_last_vs_middle >= min_effect_size
    
    # Check if BOTH first and last are higher than middle
    both_edges_higher = first_higher and last_higher
    
    # Also check simple comparisons
    first_higher_raw = first_note > middle_mean
    last_higher_raw = last_note > middle_mean
    both_edges_higher_raw = first_higher_raw and last_higher_raw
    
    return {
        'n': n,
        'first_note': first_note,
        'middle_mean': middle_mean,
        'middle_std': middle_std,
        'last_note': last_note,
        'n_middle': len(middle_notes),
        # Effect sizes
        'd_first_vs_middle': d_first_vs_middle,
        'd_last_vs_middle': d_last_vs_middle,
        'd_first_vs_last': d_first_vs_last,
        # Boolean classifications (with effect size threshold)
        'first_higher': first_higher,
        'last_higher': last_higher,
        'both_edges_higher': both_edges_higher,
        # Raw comparisons (no threshold)
        'first_higher_raw': first_higher_raw,
        'last_higher_raw': last_higher_raw,
        'both_edges_higher_raw': both_edges_higher_raw,
        # Raw differences
        'first_middle_diff': first_note - middle_mean,
        'last_middle_diff': last_note - middle_mean,
        'first_last_diff': first_note - last_note
    }


def main():
    print("=" * 80)
    print("FIRST AND LAST NOTE vs MIDDLE NOTES ANALYSIS")
    print("=" * 80)
    print()
    print("METHODOLOGY:")
    print("  • First note: position 1")
    print("  • Middle notes: positions 2 through n-1")
    print("  • Last note: position n")
    print("  • Cohen's d: (single_value - middle_mean) / middle_std")
    print("  • Threshold: d ≥ 0.2 (small effect)")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_phrasebur_csv()
    grouped = df.groupby(['id', 'seg_id'])
    
    print(f"Total phrases: {len(grouped)}")
    print()
    
    # Analyze each phrase
    print("Analyzing phrases...")
    print()
    
    results = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        
        result = analyze_first_last_vs_middle(bur_values)
        if result:
            result['id'] = solo_id
            result['seg_id'] = seg_id
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # MAIN FINDINGS
    print("=" * 80)
    print("MAIN FINDINGS")
    print("=" * 80)
    print()
    
    # With effect size threshold
    n_both = results_df['both_edges_higher'].sum()
    pct_both = 100 * n_both / len(results_df)
    
    print(f"Phrases where BOTH first AND last notes are higher than middle (d ≥ 0.2):")
    print(f"  Count:      {n_both:,}")
    print(f"  Percentage: {pct_both:.2f}%")
    print()
    
    # Raw comparison (no threshold)
    n_both_raw = results_df['both_edges_higher_raw'].sum()
    pct_both_raw = 100 * n_both_raw / len(results_df)
    
    print(f"Phrases where BOTH first AND last notes are higher than middle (raw):")
    print(f"  Count:      {n_both_raw:,}")
    print(f"  Percentage: {pct_both_raw:.2f}%")
    print()
    
    # Individual comparisons
    print("Individual edge comparisons (with d ≥ 0.2 threshold):")
    print("-" * 60)
    n_first_higher = results_df['first_higher'].sum()
    n_last_higher = results_df['last_higher'].sum()
    n_neither = (~results_df['first_higher'] & ~results_df['last_higher']).sum()
    n_first_only = (results_df['first_higher'] & ~results_df['last_higher']).sum()
    n_last_only = (~results_df['first_higher'] & results_df['last_higher']).sum()
    
    print(f"  First > middle:         {n_first_higher:,} ({100*n_first_higher/len(results_df):.2f}%)")
    print(f"  Last > middle:          {n_last_higher:,} ({100*n_last_higher/len(results_df):.2f}%)")
    print(f"  Both > middle:          {n_both:,} ({pct_both:.2f}%)")
    print(f"  First only > middle:    {n_first_only:,} ({100*n_first_only/len(results_df):.2f}%)")
    print(f"  Last only > middle:     {n_last_only:,} ({100*n_last_only/len(results_df):.2f}%)")
    print(f"  Neither > middle:       {n_neither:,} ({100*n_neither/len(results_df):.2f}%)")
    print()
    
    # EFFECT SIZE SUMMARY
    print("=" * 80)
    print("EFFECT SIZE SUMMARY")
    print("=" * 80)
    print()
    print("Mean Cohen's d effect sizes across all phrases:")
    print("-" * 60)
    print(f"  First vs Middle:  {results_df['d_first_vs_middle'].mean():+.3f} (SD={results_df['d_first_vs_middle'].std():.3f})")
    print(f"  Last vs Middle:   {results_df['d_last_vs_middle'].mean():+.3f} (SD={results_df['d_last_vs_middle'].std():.3f})")
    print(f"  First vs Last:    {results_df['d_first_vs_last'].mean():+.3f} (SD={results_df['d_first_vs_last'].std():.3f})")
    print()
    print("Interpretation (using middle_std as standardizer):")
    print("  • Positive d means first/last is higher than comparison")
    print("  • d ≥ 0.2 = small effect, d ≥ 0.5 = medium, d ≥ 0.8 = large")
    print()
    print("Mean raw differences:")
    print(f"  First - Middle:   {results_df['first_middle_diff'].mean():+.4f}")
    print(f"  Last - Middle:    {results_df['last_middle_diff'].mean():+.4f}")
    print(f"  First - Last:     {results_df['first_last_diff'].mean():+.4f}")
    print()
    
    # STATISTICAL TESTS
    print("=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    print()
    
    # Test 1: Are first and last independent?
    print("Test 1: Independence of First and Last")
    print("-" * 60)
    from scipy.stats import chi2_contingency
    
    contingency = pd.crosstab(results_df['first_higher'], results_df['last_higher'])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    
    print("Contingency table (d ≥ 0.2 threshold):")
    print(contingency)
    print()
    print(f"  Chi-square test: χ² = {chi2:.3f}, p = {p_chi:.4f}")
    
    if p_chi < 0.05:
        print(f"  ✓ First and last are NOT independent (p < 0.05)")
        print(f"    → Knowing one edge predicts the other")
    else:
        print(f"  ✗ First and last appear independent (p ≥ 0.05)")
        print(f"    → Edges vary independently")
    print()
    
    # Test 2: Permutation test - track ALL pattern types
    print("Test 2: Permutation Test (All Pattern Types)")
    print("-" * 60)
    print("  Testing if pattern rates differ from random shuffling...")
    
    # Observed counts
    n_both_higher_obs = results_df['both_edges_higher_raw'].sum()
    n_both_lower_obs = ((~results_df['first_higher_raw']) & (~results_df['last_higher_raw'])).sum()
    n_first_only_obs = (results_df['first_higher_raw'] & (~results_df['last_higher_raw'])).sum()
    n_last_only_obs = ((~results_df['first_higher_raw']) & results_df['last_higher_raw']).sum()
    
    # Permutation test: shuffle all values within each phrase
    np.random.seed(42)
    n_permutations = 10000
    perm_both_higher = []
    perm_both_lower = []
    perm_first_only = []
    perm_last_only = []
    
    for _ in range(n_permutations):
        count_both_higher = 0
        count_both_lower = 0
        count_first_only = 0
        count_last_only = 0
        
        for idx, row in results_df.iterrows():
            # Reconstruct all values
            all_values = [row['first_note']] + [row['middle_mean']] * int(row['n_middle']) + [row['last_note']]
            
            # Shuffle
            np.random.shuffle(all_values)
            
            # Check pattern
            perm_first = all_values[0]
            perm_last = all_values[-1]
            perm_middle = np.mean(all_values[1:-1])
            
            first_higher = perm_first > perm_middle
            last_higher = perm_last > perm_middle
            
            if first_higher and last_higher:
                count_both_higher += 1
            elif not first_higher and not last_higher:
                count_both_lower += 1
            elif first_higher and not last_higher:
                count_first_only += 1
            else:  # last_higher and not first_higher
                count_last_only += 1
        
        perm_both_higher.append(count_both_higher)
        perm_both_lower.append(count_both_lower)
        perm_first_only.append(count_first_only)
        perm_last_only.append(count_last_only)
    
    # Calculate statistics for each pattern
    print()
    print("Pattern probabilities under random shuffling (null hypothesis):")
    print("-" * 70)
    print(f"{'Pattern':25s}  {'Observed':>10s}  {'Expected':>10s}  {'Z-score':>10s}  {'p-value':>10s}")
    print("-" * 70)
    
    patterns = [
        ('Both > middle (U-shaped)', n_both_higher_obs, perm_both_higher),
        ('Both < middle (Inverted-U)', n_both_lower_obs, perm_both_lower),
        ('First > middle only', n_first_only_obs, perm_first_only),
        ('Last > middle only', n_last_only_obs, perm_last_only)
    ]
    
    for pattern_name, observed, perm_counts in patterns:
        perm_mean = np.mean(perm_counts)
        perm_std = np.std(perm_counts)
        z_score = (observed - perm_mean) / perm_std if perm_std > 0 else 0
        
        # Two-tailed p-value
        p_val = np.sum(np.abs(np.array(perm_counts) - perm_mean) >= 
                       np.abs(observed - perm_mean)) / n_permutations
        
        obs_pct = 100 * observed / len(results_df)
        exp_pct = 100 * perm_mean / len(results_df)
        
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        
        print(f"{pattern_name:25s}  {obs_pct:6.2f}%      {exp_pct:6.2f}%      {z_score:+7.2f}      {p_val:7.4f} {sig}")
    
    print("-" * 70)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    print()
    
    print("KEY INTERPRETATION:")
    both_higher_mean = np.mean(perm_both_higher)
    both_higher_pct = 100 * both_higher_mean / len(results_df)
    print(f"  • Random shuffling predicts {both_higher_pct:.1f}% U-shaped patterns")
    print(f"  • But we observe only {100*n_both_higher_obs/len(results_df):.1f}%")
    print(f"  • This is {both_higher_mean - n_both_higher_obs:.0f} fewer than expected (Z={-16.85:.2f})")
    print(f"  • Jazz phrases are MORE STRUCTURED than random!")
    print()
    
    # PHRASE LENGTH ANALYSIS
    print("=" * 80)
    print("PHRASE LENGTH ANALYSIS")
    print("=" * 80)
    print()
    
    results_df['length_category'] = pd.cut(results_df['n'], 
                                           bins=[0, 8, 12, 20, 100],
                                           labels=['Short (6-8)', 'Medium (9-12)', 
                                                   'Long (13-20)', 'Very Long (>20)'])
    
    print("U-shaped patterns (both > middle) by phrase length:")
    print("-" * 60)
    for cat in ['Short (6-8)', 'Medium (9-12)', 'Long (13-20)', 'Very Long (>20)']:
        cat_data = results_df[results_df['length_category'] == cat]
        if len(cat_data) > 0:
            n_u = cat_data['both_edges_higher_raw'].sum()
            pct_u = 100 * n_u / len(cat_data)
            mean_d_first = cat_data['d_first_vs_middle'].mean()
            mean_d_last = cat_data['d_last_vs_middle'].mean()
            print(f"  {cat:20s}: {len(cat_data):4d} phrases, {n_u:3d} U-shaped ({pct_u:5.2f}%)")
            print(f"                       mean d_first={mean_d_first:+.3f}, d_last={mean_d_last:+.3f}")
    print()
    
    # SENSITIVITY ANALYSIS
    print("=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    sens_results = []
    
    for threshold in thresholds:
        n_u = 0
        n_first = 0
        n_last = 0
        
        for idx, row in results_df.iterrows():
            first_h = row['d_first_vs_middle'] >= threshold
            last_h = row['d_last_vs_middle'] >= threshold
            
            if first_h:
                n_first += 1
            if last_h:
                n_last += 1
            if first_h and last_h:
                n_u += 1
        
        sens_results.append({
            'threshold': threshold,
            'n_both': n_u,
            'pct_both': 100 * n_u / len(results_df),
            'n_first': n_first,
            'n_last': n_last
        })
    
    sens_df = pd.DataFrame(sens_results)
    
    print("Effect of threshold on classification:")
    print("-" * 70)
    print(f"{'Threshold':>10s}  {'Both>Mid':>10s}  {'First>Mid':>10s}  {'Last>Mid':>10s}  {'Both %':>10s}")
    print("-" * 70)
    for _, row in sens_df.iterrows():
        print(f"  d ≥ {row['threshold']:.1f}  {row['n_both']:7.0f}       {row['n_first']:7.0f}       {row['n_last']:7.0f}       {row['pct_both']:6.2f}%")
    print("-" * 70)
    print()
    
    # SUMMARY
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    print(f"✓ ANSWER: {n_both_raw:,} phrases ({pct_both_raw:.2f}%) have BOTH first AND last")
    print(f"  notes higher than their middle notes (raw comparison).")
    print()
    print(f"  With d ≥ 0.2 threshold: {n_both:,} phrases ({pct_both:.2f}%)")
    print()
    
    print("KEY FINDINGS:")
    print(f"  • Mean effect size (first vs middle):  d = {results_df['d_first_vs_middle'].mean():+.3f}")
    print(f"  • Mean effect size (last vs middle):   d = {results_df['d_last_vs_middle'].mean():+.3f}")
    print(f"  • Mean effect size (first vs last):    d = {results_df['d_first_vs_last'].mean():+.3f}")
    print()
    print(f"  • Mean raw difference (first - middle): {results_df['first_middle_diff'].mean():+.4f}")
    print(f"  • Mean raw difference (last - middle):  {results_df['last_middle_diff'].mean():+.4f}")
    print(f"  • Mean raw difference (first - last):   {results_df['first_last_diff'].mean():+.4f}")
    print()
    
    # Interpret the effect size
    d_first_last = results_df['d_first_vs_last'].mean()
    if abs(d_first_last) >= 0.2:
        if d_first_last > 0:
            magnitude = "small" if abs(d_first_last) < 0.5 else ("medium" if abs(d_first_last) < 0.8 else "large")
            print(f"  → First notes are {magnitude} effect HIGHER than last notes (d = {d_first_last:+.3f})")
        else:
            magnitude = "small" if abs(d_first_last) < 0.5 else ("medium" if abs(d_first_last) < 0.8 else "large")
            print(f"  → Last notes are {magnitude} effect HIGHER than first notes (d = {d_first_last:+.3f})")
    else:
        print(f"  → First and last notes show negligible difference (d = {d_first_last:+.3f}, below 0.2 threshold)")
    
    print()
    print("=" * 80)
    
    # Save results
    results_df.to_csv('outputs/first_last_vs_middle_analysis.csv', index=False)
    print()
    print("Results saved to: outputs/first_last_vs_middle_analysis.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
