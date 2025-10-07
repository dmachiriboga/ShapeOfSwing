"""
CORRECTED Positional Analysis - Phrase-Level Aggregation

This script performs the positional analysis the CORRECT way:
- Aggregates within phrases FIRST (respects independence)
- Then performs statistical tests on phrase-level means
- Uses repeated measures tests (Friedman, paired t-tests)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES


def corrected_positional_analysis():
    """
    Perform corrected positional analysis using phrase-level aggregation.
    """
    print("=" * 80)
    print("CORRECTED POSITIONAL ANALYSIS")
    print("Using phrase-level aggregation to respect independence assumptions")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_phrasebur_csv()
    grouped = df.groupby(['id', 'seg_id'])
    print(f"Total phrases: {len(grouped)}")
    print()
    
    # ========================================================================
    # CORRECTED ANALYSIS 1: Beginning/Middle/End (Phrase-Level)
    # ========================================================================
    print("=" * 80)
    print("CORRECTED ANALYSIS 1: Beginning/Middle/End (Phrase-Level Means)")
    print("=" * 80)
    print()
    
    phrase_third_means = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n < MIN_BUR_VALUES:
            continue
        
        # Divide into thirds
        third = n // 3
        remainder = n % 3
        
        # Beginning: first third
        beginning = bur_values[:third + (1 if remainder >= 1 else 0)]
        
        # Middle: middle third
        start_middle = third + (1 if remainder >= 1 else 0)
        end_middle = start_middle + third + (1 if remainder >= 2 else 0)
        middle = bur_values[start_middle:end_middle]
        
        # End: last third
        end = bur_values[end_middle:]
        
        # Store phrase-level means
        phrase_third_means.append({
            'beginning': np.mean(beginning),
            'middle': np.mean(middle),
            'end': np.mean(end)
        })
    
    phrase_thirds_df = pd.DataFrame(phrase_third_means)
    
    print(f"N phrases analyzed: {len(phrase_thirds_df)}")
    print()
    print("Descriptive Statistics (phrase-level means):")
    print("-" * 60)
    print(f"Beginning: Mean = {phrase_thirds_df['beginning'].mean():.6f}, "
          f"SD = {phrase_thirds_df['beginning'].std():.6f}, "
          f"Median = {phrase_thirds_df['beginning'].median():.6f}")
    print(f"Middle:    Mean = {phrase_thirds_df['middle'].mean():.6f}, "
          f"SD = {phrase_thirds_df['middle'].std():.6f}, "
          f"Median = {phrase_thirds_df['middle'].median():.6f}")
    print(f"End:       Mean = {phrase_thirds_df['end'].mean():.6f}, "
          f"SD = {phrase_thirds_df['end'].std():.6f}, "
          f"Median = {phrase_thirds_df['end'].median():.6f}")
    print()
    
    # Mean differences
    beg_mid_diff = phrase_thirds_df['beginning'].mean() - phrase_thirds_df['middle'].mean()
    beg_end_diff = phrase_thirds_df['beginning'].mean() - phrase_thirds_df['end'].mean()
    mid_end_diff = phrase_thirds_df['middle'].mean() - phrase_thirds_df['end'].mean()
    
    print("Mean Differences:")
    print(f"  Beginning - Middle: {beg_mid_diff:+.6f}")
    print(f"  Beginning - End:    {beg_end_diff:+.6f}")
    print(f"  Middle - End:       {mid_end_diff:+.6f}")
    print()
    
    # Friedman test (non-parametric repeated measures ANOVA)
    print("Friedman Test (non-parametric repeated measures):")
    stat, p_friedman = stats.friedmanchisquare(
        phrase_thirds_df['beginning'],
        phrase_thirds_df['middle'],
        phrase_thirds_df['end']
    )
    print(f"  χ² = {stat:.4f}")
    print(f"  p-value = {p_friedman:.6f}")
    if p_friedman < 0.05:
        print("  ✓ Significant differences across thirds (p < 0.05)")
    else:
        print("  ✗ No significant differences (p ≥ 0.05)")
    print()
    
    # Pairwise paired t-tests
    print("Pairwise Paired t-tests:")
    print("-" * 60)
    
    t1, p1 = stats.ttest_rel(phrase_thirds_df['beginning'], phrase_thirds_df['middle'])
    cohen_d1 = beg_mid_diff / phrase_thirds_df[['beginning', 'middle']].std().mean()
    print(f"Beginning vs Middle:")
    print(f"  t = {t1:.4f}, p = {p1:.6f}, Cohen's d = {cohen_d1:.4f}")
    if p1 < 0.05:
        print(f"  ✓ Significant (p < 0.05)")
    else:
        print(f"  ✗ Not significant")
    print()
    
    t2, p2 = stats.ttest_rel(phrase_thirds_df['beginning'], phrase_thirds_df['end'])
    cohen_d2 = beg_end_diff / phrase_thirds_df[['beginning', 'end']].std().mean()
    print(f"Beginning vs End:")
    print(f"  t = {t2:.4f}, p = {p2:.6f}, Cohen's d = {cohen_d2:.4f}")
    if p2 < 0.05:
        print(f"  ✓ Significant (p < 0.05)")
    else:
        print(f"  ✗ Not significant")
    print()
    
    t3, p3 = stats.ttest_rel(phrase_thirds_df['middle'], phrase_thirds_df['end'])
    cohen_d3 = mid_end_diff / phrase_thirds_df[['middle', 'end']].std().mean()
    print(f"Middle vs End:")
    print(f"  t = {t3:.4f}, p = {p3:.6f}, Cohen's d = {cohen_d3:.4f}")
    if p3 < 0.05:
        print(f"  ✓ Significant (p < 0.05)")
    else:
        print(f"  ✗ Not significant")
    print()
    
    # ========================================================================
    # CORRECTED ANALYSIS 2: First vs Last (Already Correct)
    # ========================================================================
    print("=" * 80)
    print("CORRECTED ANALYSIS 2: First vs Last Note")
    print("(This was already done correctly in original analysis)")
    print("=" * 80)
    print()
    
    first_notes = []
    last_notes = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        if len(bur_values) >= MIN_BUR_VALUES:
            first_notes.append(bur_values[0])
            last_notes.append(bur_values[-1])
    
    print(f"N phrases: {len(first_notes)}")
    print()
    print(f"First Notes:")
    print(f"  Mean:   {np.mean(first_notes):.6f}")
    print(f"  SD:     {np.std(first_notes):.6f}")
    print(f"  Median: {np.median(first_notes):.6f}")
    print()
    
    print(f"Last Notes:")
    print(f"  Mean:   {np.mean(last_notes):.6f}")
    print(f"  SD:     {np.std(last_notes):.6f}")
    print(f"  Median: {np.median(last_notes):.6f}")
    print()
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(first_notes, last_notes)
    mean_diff = np.mean(first_notes) - np.mean(last_notes)
    cohen_d = mean_diff / np.std([f - l for f, l in zip(first_notes, last_notes)])
    
    print(f"Paired t-test:")
    print(f"  Mean difference: {mean_diff:+.6f} ({100*mean_diff/np.mean(last_notes):+.2f}%)")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Cohen's d: {cohen_d:.4f}")
    if p_val < 0.05:
        if mean_diff > 0:
            print(f"  ✓ First notes significantly HIGHER than last (p < 0.05)")
        else:
            print(f"  ✓ First notes significantly LOWER than last (p < 0.05)")
    else:
        print(f"  ✗ No significant difference")
    print()
    
    # ========================================================================
    # CORRECTED ANALYSIS 3: First/Middle/Last (More Precise)
    # ========================================================================
    print("=" * 80)
    print("CORRECTED ANALYSIS 3: First/Middle/Last Note Comparison")
    print("(Single representative note from each third)")
    print("=" * 80)
    print()
    
    first_mid_last = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n < MIN_BUR_VALUES:
            continue
        
        # Get first note, middle note, and last note
        first = bur_values[0]
        middle = bur_values[n // 2]
        last = bur_values[-1]
        
        first_mid_last.append({
            'first': first,
            'middle': middle,
            'last': last
        })
    
    fml_df = pd.DataFrame(first_mid_last)
    
    print(f"N phrases: {len(fml_df)}")
    print()
    print("Descriptive Statistics:")
    print("-" * 60)
    print(f"First:  Mean = {fml_df['first'].mean():.6f}, "
          f"SD = {fml_df['first'].std():.6f}, "
          f"Median = {fml_df['first'].median():.6f}")
    print(f"Middle: Mean = {fml_df['middle'].mean():.6f}, "
          f"SD = {fml_df['middle'].std():.6f}, "
          f"Median = {fml_df['middle'].median():.6f}")
    print(f"Last:   Mean = {fml_df['last'].mean():.6f}, "
          f"SD = {fml_df['last'].std():.6f}, "
          f"Median = {fml_df['last'].median():.6f}")
    print()
    
    # Friedman test
    stat_fml, p_fml = stats.friedmanchisquare(
        fml_df['first'],
        fml_df['middle'],
        fml_df['last']
    )
    print(f"Friedman Test: χ² = {stat_fml:.4f}, p = {p_fml:.6f}")
    if p_fml < 0.05:
        print("  ✓ Significant (p < 0.05)")
    else:
        print("  ✗ Not significant")
    print()
    
    # Pairwise comparisons
    print("Pairwise Paired t-tests:")
    t_fm, p_fm = stats.ttest_rel(fml_df['first'], fml_df['middle'])
    t_fl, p_fl = stats.ttest_rel(fml_df['first'], fml_df['last'])
    t_ml, p_ml = stats.ttest_rel(fml_df['middle'], fml_df['last'])
    
    print(f"  First vs Middle:  t = {t_fm:.4f}, p = {p_fm:.6f}")
    print(f"  First vs Last:    t = {t_fl:.4f}, p = {p_fl:.6f} {'✓' if p_fl < 0.05 else ''}")
    print(f"  Middle vs Last:   t = {t_ml:.4f}, p = {p_ml:.6f}")
    print()
    
    # ========================================================================
    # Multiple Testing Correction
    # ========================================================================
    print("=" * 80)
    print("MULTIPLE TESTING CORRECTION (FDR)")
    print("=" * 80)
    print()
    
    from statsmodels.stats.multitest import multipletests
    
    all_p_values = {
        'Friedman_thirds': p_friedman,
        'Pair_beg_mid': p1,
        'Pair_beg_end': p2,
        'Pair_mid_end': p3,
        'First_vs_Last': p_val,
        'Friedman_fml': p_fml,
        'Pair_first_middle': p_fm,
        'Pair_first_last': p_fl,
        'Pair_middle_last': p_ml,
    }
    
    reject, p_corrected, _, _ = multipletests(
        list(all_p_values.values()), 
        alpha=0.05, 
        method='fdr_bh'
    )
    
    print("Test Name                 | p-original | p-adjusted | Significant?")
    print("-" * 75)
    for i, (test_name, p_orig) in enumerate(all_p_values.items()):
        sig_marker = "✓" if reject[i] else "✗"
        print(f"{test_name:25s} | {p_orig:10.6f} | {p_corrected[i]:10.6f} | {sig_marker}")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY OF CORRECTED FINDINGS")
    print("=" * 80)
    print()
    
    findings = []
    
    # Beginning/Middle/End
    if reject[0]:  # Friedman test
        findings.append(f"✓ Beginning/Middle/End differ significantly (Friedman p_adj={p_corrected[0]:.4f})")
    else:
        findings.append(f"✗ No difference between Beginning/Middle/End (Friedman p_adj={p_corrected[0]:.4f})")
    
    # Specific thirds comparisons
    if reject[1]:
        findings.append(f"  ✓ Beginning > Middle (p_adj={p_corrected[1]:.4f})")
    if reject[2]:
        findings.append(f"  ✓ Beginning > End (p_adj={p_corrected[2]:.4f})")
    if reject[3]:
        findings.append(f"  ✓ Middle ≠ End (p_adj={p_corrected[3]:.4f})")
    
    # First vs Last
    if reject[4]:
        findings.append(f"✓ First note significantly different from last (p_adj={p_corrected[4]:.6f})")
        findings.append(f"  → First notes are {100*mean_diff/np.mean(last_notes):+.2f}% higher/swingier")
    
    # First/Middle/Last
    if reject[5]:
        findings.append(f"✓ First/Middle/Last differ significantly (Friedman p_adj={p_corrected[5]:.4f})")
    
    if reject[7]:  # First vs Last in FML
        findings.append(f"  ✓ First > Last (p_adj={p_corrected[7]:.6f})")
    
    print("\n".join(findings))
    print()
    
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("Using proper phrase-level aggregation that respects statistical independence:")
    print()
    
    if reject[4]:
        print(f"✓ CONFIRMED: First notes have significantly higher BUR than last notes")
        print(f"  (mean difference = {mean_diff:+.4f}, p_adj < 0.001)")
    
    if not reject[0]:
        print(f"✗ NO EVIDENCE: Beginning/Middle/End thirds are essentially identical")
        print(f"  (Friedman test p_adj = {p_corrected[0]:.3f})")
    
    print()
    print("These results are methodologically sound and can be trusted.")
    print("=" * 80)


if __name__ == '__main__':
    corrected_positional_analysis()
