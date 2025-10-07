"""
Sensitivity Analysis: Effect of Minimum BUR Threshold

Compares Mann-Kendall and FDR results across different minimum phrase lengths:
- MIN_BUR = 3 (less conservative, includes more phrases)
- MIN_BUR = 4
- MIN_BUR = 5
- MIN_BUR = 6 (current default)
- MIN_BUR = 8 (more conservative)
- MIN_BUR = 10 (very conservative)

This tests whether the "rare surge" finding depends on filtering threshold.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from utils.data_utils import load_phrasebur_csv


def mann_kendall_test(data):
    """
    Perform Mann-Kendall trend test.
    
    Returns:
        tau: Kendall's tau (effect size)
        p_value: two-tailed p-value
    """
    n = len(data)
    if n < 3:
        return None, None
    
    # Calculate S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    
    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Kendall's tau
    tau = s / (0.5 * n * (n - 1))
    
    return tau, p_value


def analyze_with_threshold(min_bur_values, df):
    """
    Run Mann-Kendall analysis with specified minimum threshold.
    
    Returns:
        dict with results
    """
    grouped = df.groupby(['id', 'seg_id'])
    
    results = []
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n < min_bur_values:
            continue
        
        # Mann-Kendall test
        tau, p_value = mann_kendall_test(bur_values)
        
        if tau is not None:
            results.append({
                'id': solo_id,
                'seg_id': seg_id,
                'n': n,
                'tau': tau,
                'p_value': p_value,
                'mean_bur': np.mean(bur_values),
                'std_bur': np.std(bur_values)
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return {
            'min_threshold': min_bur_values,
            'n_phrases': 0,
            'n_tests': 0,
            'n_significant_raw': 0,
            'pct_significant_raw': 0,
            'n_significant_fdr': 0,
            'pct_significant_fdr': 0,
            'mean_tau': 0,
            'median_tau': 0
        }
    
    # Raw significance (α = 0.05)
    n_significant_raw = (results_df['p_value'] < 0.05).sum()
    pct_significant_raw = 100 * n_significant_raw / len(results_df)
    
    # FDR correction (Benjamini-Hochberg)
    p_values = results_df['p_value'].tolist()
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    n_significant_fdr = reject.sum()
    pct_significant_fdr = 100 * n_significant_fdr / len(results_df)
    
    # Effect sizes
    mean_tau = results_df['tau'].abs().mean()
    median_tau = results_df['tau'].abs().median()
    
    return {
        'min_threshold': min_bur_values,
        'n_phrases': len(results_df),
        'n_tests': len(results_df),
        'n_significant_raw': n_significant_raw,
        'pct_significant_raw': pct_significant_raw,
        'n_significant_fdr': n_significant_fdr,
        'pct_significant_fdr': pct_significant_fdr,
        'mean_tau': mean_tau,
        'median_tau': median_tau,
        'mean_bur': results_df['mean_bur'].mean(),
        'std_bur': results_df['std_bur'].mean()
    }


def main():
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Minimum BUR Threshold")
    print("Effect on Mann-Kendall and FDR Results")
    print("=" * 80)
    print()
    
    # Load raw data (unfiltered)
    print("Loading raw data...")
    df_raw = pd.read_csv('data/phrasebur_raw.csv', delimiter=';')
    
    total_phrases_raw = len(df_raw.groupby(['id', 'seg_id']))
    print(f"Total phrases in raw data: {total_phrases_raw:,}")
    print()
    
    # Test different thresholds
    thresholds = [3, 4, 5, 6, 8, 10, 12, 15]
    
    print("Running Mann-Kendall analysis for each threshold...")
    print()
    
    all_results = []
    
    for threshold in thresholds:
        print(f"  Analyzing with MIN_BUR = {threshold}...")
        result = analyze_with_threshold(threshold, df_raw)
        all_results.append(result)
    
    print()
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_results)
    
    print("Effect of Minimum BUR Threshold on Significance Rates:")
    print("-" * 80)
    print(f"{'Threshold':>10s} | {'N Phrases':>10s} | {'% of Total':>10s} | "
          f"{'Sig (raw)':>10s} | {'Sig (FDR)':>10s} | {'% FDR':>8s}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        pct_of_total = 100 * row['n_phrases'] / total_phrases_raw
        print(f"{int(row['min_threshold']):10d} | {int(row['n_phrases']):10d} | {pct_of_total:9.1f}% | "
              f"{int(row['n_significant_raw']):10d} | {int(row['n_significant_fdr']):10d} | "
              f"{row['pct_significant_fdr']:7.2f}%")
    
    print("-" * 80)
    print()
    
    # Detailed comparison
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print()
    
    for _, row in comparison_df.iterrows():
        print(f"MIN_BUR = {int(row['min_threshold'])}:")
        print(f"  Phrases included:      {int(row['n_phrases']):,} ({100*row['n_phrases']/total_phrases_raw:.1f}% of total)")
        print(f"  Phrases excluded:      {int(total_phrases_raw - row['n_phrases']):,}")
        print(f"  Significant (raw):     {int(row['n_significant_raw']):,} ({row['pct_significant_raw']:.2f}%)")
        print(f"  Significant (FDR):     {int(row['n_significant_fdr']):,} ({row['pct_significant_fdr']:.2f}%)")
        print(f"  Mean |tau|:            {row['mean_tau']:.4f}")
        print(f"  Median |tau|:          {row['median_tau']:.4f}")
        print(f"  Mean BUR:              {row['mean_bur']:.4f}")
        print(f"  Mean within-phrase SD: {row['std_bur']:.4f}")
        print()
    
    # Analysis of trend
    print("=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)
    print()
    
    # Compare MIN=3 vs MIN=6
    min3_result = comparison_df[comparison_df['min_threshold'] == 3].iloc[0]
    min6_result = comparison_df[comparison_df['min_threshold'] == 6].iloc[0]
    
    print("Comparing MIN_BUR = 3 vs MIN_BUR = 6 (current):")
    print()
    print(f"                           MIN=3        MIN=6      Change")
    print("-" * 65)
    print(f"N phrases:              {int(min3_result['n_phrases']):7,}    {int(min6_result['n_phrases']):7,}    {int(min3_result['n_phrases'] - min6_result['n_phrases']):+7,}")
    print(f"% of total:             {100*min3_result['n_phrases']/total_phrases_raw:6.1f}%    {100*min6_result['n_phrases']/total_phrases_raw:6.1f}%")
    print(f"Significant (FDR):      {int(min3_result['n_significant_fdr']):7,}    {int(min6_result['n_significant_fdr']):7,}    {int(min3_result['n_significant_fdr'] - min6_result['n_significant_fdr']):+7,}")
    print(f"% Significant (FDR):    {min3_result['pct_significant_fdr']:6.2f}%    {min6_result['pct_significant_fdr']:6.2f}%    {min3_result['pct_significant_fdr'] - min6_result['pct_significant_fdr']:+6.2f}%")
    print(f"Mean |tau|:             {min3_result['mean_tau']:6.4f}    {min6_result['mean_tau']:6.4f}    {min3_result['mean_tau'] - min6_result['mean_tau']:+6.4f}")
    print(f"Mean BUR:               {min3_result['mean_bur']:6.4f}    {min6_result['mean_bur']:6.4f}")
    print()
    
    # Statistical power consideration
    print("=" * 80)
    print("STATISTICAL POWER CONSIDERATIONS")
    print("=" * 80)
    print()
    
    print("Mann-Kendall test power by phrase length:")
    print()
    print("  n=3:  Very low power, detects only extreme trends")
    print("  n=4:  Low power, still limited")
    print("  n=5:  Moderate power")
    print("  n=6:  Good power for moderate effects (current threshold)")
    print("  n=8:  Very good power")
    print("  n=10: Excellent power")
    print()
    
    print("Trade-off:")
    print("  Lower threshold → More phrases but lower power per phrase")
    print("  Higher threshold → Fewer phrases but higher power per phrase")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    if min3_result['pct_significant_fdr'] < 5:
        print("✓ ROBUST FINDING: Significant rate remains low (<5%) even with MIN=3")
        print()
        print("  This suggests the 'rare surge' finding is NOT an artifact of")
        print("  the filtering threshold. Even when including short phrases,")
        print("  systematic BUR trends are uncommon.")
        print()
    else:
        print("⚠ THRESHOLD-DEPENDENT: Significant rate increases substantially with MIN=3")
        print()
        print("  This suggests the filtering threshold affects results.")
        print("  Short phrases may show different patterns than long phrases.")
        print()
    
    # Effect size comparison
    if abs(min3_result['mean_tau'] - min6_result['mean_tau']) < 0.02:
        print("✓ CONSISTENT EFFECT SIZES: Mean |tau| similar across thresholds")
        print()
        print("  Phrase length doesn't substantially affect trend strength.")
        print()
    else:
        print("⚠ VARYING EFFECT SIZES: Mean |tau| differs across thresholds")
        print()
        print("  Longer phrases may have systematically different trends.")
        print()
    
    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    
    if min3_result['pct_significant_fdr'] < 3 and min6_result['pct_significant_fdr'] < 3:
        print("✓ Use MIN=6 (current threshold)")
        print()
        print("  Rationale:")
        print("  - Both thresholds give similar significance rates (~1-2%)")
        print("  - MIN=6 has better statistical power per phrase")
        print("  - Results are more reliable and less noisy")
        print("  - Standard practice in time series analysis")
        print()
        print("  Alternatively, report results for both MIN=3 and MIN=6")
        print("  as sensitivity analysis to show robustness.")
    else:
        print("⚠ Report results for multiple thresholds")
        print()
        print("  The choice of threshold affects conclusions.")
        print("  Present MIN=3, MIN=6, and MIN=10 as sensitivity analysis.")
    
    print()
    print("=" * 80)
    
    # Save results
    comparison_df.to_csv('outputs/bur_threshold_sensitivity_analysis.csv', index=False)
    print()
    print("Results saved to: outputs/bur_threshold_sensitivity_analysis.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
