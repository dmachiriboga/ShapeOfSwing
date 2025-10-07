"""
BUR Positional Pattern Analysis

Explores whether BUR values show systematic patterns based on:
1. Absolute position within phrase (note 1, 2, 3, etc.)
2. Relative position (beginning, middle, end)
3. Normalized position (0.0 = start, 1.0 = end)

This is exploratory analysis to find any positional patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES
import matplotlib.pyplot as plt


def analyze_positional_patterns():
    """
    Analyze BUR patterns by position within phrases.
    """
    print("=" * 80)
    print("BUR POSITIONAL PATTERN ANALYSIS")
    print("Exploring systematic patterns within phrases")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_phrasebur_csv()
    
    # Group by phrase
    grouped = df.groupby(['id', 'seg_id'])
    
    print(f"Total phrases: {len(grouped)}")
    print()
    
    # ========================================================================
    # ANALYSIS 1: BUR by Absolute Position (first 20 positions)
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS 1: BUR BY ABSOLUTE POSITION")
    print("Does position 1, 2, 3, etc. have characteristic BUR values?")
    print("=" * 80)
    print()
    
    position_data = {}  # position -> list of BUR values
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        for i, bur in enumerate(bur_values):
            if i not in position_data:
                position_data[i] = []
            position_data[i].append(bur)
    
    # Calculate statistics for each position
    print("Position | N     | Mean BUR | Std Dev | Median | Min  | Max")
    print("-" * 70)
    
    for pos in sorted(position_data.keys())[:20]:  # First 20 positions
        values = position_data[pos]
        print(f"{pos+1:8d} | {len(values):5d} | {np.mean(values):8.3f} | "
              f"{np.std(values):7.3f} | {np.median(values):6.3f} | "
              f"{np.min(values):4.2f} | {np.max(values):4.2f}")
    
    print()
    
    # Test for differences across positions (ANOVA)
    # Use first 10 positions (most data)
    position_groups = [position_data[i] for i in range(10) if i in position_data]
    f_stat, p_value = stats.f_oneway(*position_groups)
    
    print(f"ANOVA Test (positions 1-10):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✓ Significant differences across positions (p < 0.05)")
    else:
        print(f"  ✗ No significant differences (p ≥ 0.05)")
    print()
    
    # Calculate trend across first 10 positions
    first_10_means = [np.mean(position_data[i]) for i in range(10) if i in position_data]
    if len(first_10_means) >= 10:
        x = np.arange(10)
        slope, intercept, r_value, p_value_trend, stderr = stats.linregress(x, first_10_means)
        print(f"Linear Trend (positions 1-10):")
        print(f"  Slope: {slope:.6f} BUR/position")
        print(f"  R²: {r_value**2:.6f}")
        print(f"  p-value: {p_value_trend:.6f}")
        if p_value_trend < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
            print(f"  ✓ Significant {direction} trend (p < 0.05)")
        else:
            print(f"  ✗ No significant trend (p ≥ 0.05)")
    print()
    
    # ========================================================================
    # ANALYSIS 2: BUR by Relative Position (Thirds)
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS 2: BUR BY RELATIVE POSITION (Beginning/Middle/End)")
    print("=" * 80)
    print()
    
    beginning_bur = []
    middle_bur = []
    end_bur = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n < MIN_BUR_VALUES:
            continue
        
        # Divide into thirds
        third = n // 3
        remainder = n % 3
        
        # Beginning: first third
        beginning_bur.extend(bur_values[:third + (1 if remainder >= 1 else 0)])
        
        # Middle: middle third
        start_middle = third + (1 if remainder >= 1 else 0)
        end_middle = start_middle + third + (1 if remainder >= 2 else 0)
        middle_bur.extend(bur_values[start_middle:end_middle])
        
        # End: last third
        end_bur.extend(bur_values[end_middle:])
    
    # Calculate statistics
    segments = {
        'Beginning': beginning_bur,
        'Middle': middle_bur,
        'End': end_bur
    }
    
    print("Segment   | N      | Mean BUR | Std Dev | Median")
    print("-" * 60)
    for segment_name, values in segments.items():
        print(f"{segment_name:9s} | {len(values):6d} | {np.mean(values):8.3f} | "
              f"{np.std(values):7.3f} | {np.median(values):6.3f}")
    print()
    
    # Statistical tests
    # ANOVA
    f_stat, p_value = stats.f_oneway(beginning_bur, middle_bur, end_bur)
    print(f"ANOVA Test (Beginning vs Middle vs End):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✓ Significant differences (p < 0.05)")
    else:
        print(f"  ✗ No significant differences (p ≥ 0.05)")
    print()
    
    # Pairwise comparisons (Welch's t-test)
    print("Pairwise Comparisons (Welch's t-test):")
    
    pairs = [
        ('Beginning', 'Middle', beginning_bur, middle_bur),
        ('Beginning', 'End', beginning_bur, end_bur),
        ('Middle', 'End', middle_bur, end_bur)
    ]
    
    for name1, name2, vals1, vals2 in pairs:
        t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
        mean_diff = np.mean(vals1) - np.mean(vals2)
        effect_size = mean_diff / np.sqrt((np.var(vals1) + np.var(vals2)) / 2)  # Cohen's d
        
        print(f"  {name1} vs {name2}:")
        print(f"    Mean difference: {mean_diff:+.6f}")
        print(f"    Cohen's d: {effect_size:+.4f}")
        print(f"    p-value: {p_val:.6f}", end="")
        if p_val < 0.05:
            print(" ✓ Significant")
        else:
            print()
    print()
    
    # ========================================================================
    # ANALYSIS 3: BUR by Normalized Position (0.0-1.0)
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS 3: BUR BY NORMALIZED POSITION")
    print("Position as fraction of phrase length (0.0 = start, 1.0 = end)")
    print("=" * 80)
    print()
    
    # Bin normalized positions into 10 deciles
    n_bins = 10
    normalized_bins = {i: [] for i in range(n_bins)}
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n < MIN_BUR_VALUES:
            continue
        
        for i, bur in enumerate(bur_values):
            normalized_pos = i / (n - 1) if n > 1 else 0
            bin_idx = min(int(normalized_pos * n_bins), n_bins - 1)
            normalized_bins[bin_idx].append(bur)
    
    # Calculate statistics for each bin
    print("Decile | Range      | N      | Mean BUR | Std Dev | Median")
    print("-" * 70)
    
    for bin_idx in range(n_bins):
        values = normalized_bins[bin_idx]
        if values:
            range_start = bin_idx * 0.1
            range_end = (bin_idx + 1) * 0.1
            print(f"{bin_idx+1:6d} | {range_start:.1f}-{range_end:.1f} | {len(values):6d} | "
                  f"{np.mean(values):8.3f} | {np.std(values):7.3f} | {np.median(values):6.3f}")
    print()
    
    # Test for trend across normalized positions
    bin_means = [np.mean(normalized_bins[i]) for i in range(n_bins) if normalized_bins[i]]
    x = np.arange(len(bin_means))
    slope, intercept, r_value, p_value_trend, stderr = stats.linregress(x, bin_means)
    
    print(f"Linear Trend (normalized position):")
    print(f"  Slope: {slope:.6f} BUR/decile")
    print(f"  R²: {r_value**2:.6f}")
    print(f"  p-value: {p_value_trend:.6f}")
    if p_value_trend < 0.05:
        direction = "increasing" if slope > 0 else "decreasing"
        print(f"  ✓ Significant {direction} trend (p < 0.05)")
    else:
        print(f"  ✗ No significant trend (p ≥ 0.05)")
    print()
    
    # ========================================================================
    # ANALYSIS 4: First vs Last Note
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS 4: FIRST VS LAST NOTE")
    print("Do phrases start/end differently?")
    print("=" * 80)
    print()
    
    first_notes = []
    last_notes = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        if len(bur_values) >= MIN_BUR_VALUES:
            first_notes.append(bur_values[0])
            last_notes.append(bur_values[-1])
    
    print(f"First Notes:")
    print(f"  N: {len(first_notes)}")
    print(f"  Mean: {np.mean(first_notes):.4f}")
    print(f"  Std Dev: {np.std(first_notes):.4f}")
    print(f"  Median: {np.median(first_notes):.4f}")
    print()
    
    print(f"Last Notes:")
    print(f"  N: {len(last_notes)}")
    print(f"  Mean: {np.mean(last_notes):.4f}")
    print(f"  Std Dev: {np.std(last_notes):.4f}")
    print(f"  Median: {np.median(last_notes):.4f}")
    print()
    
    # Paired t-test (same phrases)
    t_stat, p_val = stats.ttest_rel(first_notes, last_notes)
    mean_diff = np.mean(first_notes) - np.mean(last_notes)
    
    print(f"Paired t-test (First vs Last):")
    print(f"  Mean difference: {mean_diff:+.6f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.6f}")
    if p_val < 0.05:
        if mean_diff > 0:
            print(f"  ✓ First notes significantly HIGHER than last (p < 0.05)")
        else:
            print(f"  ✓ First notes significantly LOWER than last (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p ≥ 0.05)")
    print()
    
    # ========================================================================
    # ANALYSIS 5: Phrase Length Effect
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS 5: PHRASE LENGTH EFFECT")
    print("Do longer phrases have different BUR patterns?")
    print("=" * 80)
    print()
    
    phrase_stats = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n >= MIN_BUR_VALUES:
            phrase_stats.append({
                'length': n,
                'mean_bur': np.mean(bur_values),
                'std_bur': np.std(bur_values),
                'first_bur': bur_values[0],
                'last_bur': bur_values[-1],
                'range_bur': np.max(bur_values) - np.min(bur_values)
            })
    
    phrase_df = pd.DataFrame(phrase_stats)
    
    # Correlation between length and BUR characteristics
    print("Correlations with Phrase Length:")
    print()
    
    for metric in ['mean_bur', 'std_bur', 'first_bur', 'last_bur', 'range_bur']:
        corr, p_val = stats.pearsonr(phrase_df['length'], phrase_df[metric])
        print(f"  {metric:12s}: r = {corr:+.4f}, p = {p_val:.6f}", end="")
        if p_val < 0.05:
            print(" ✓")
        else:
            print()
    print()
    
    # Group by length categories
    phrase_df['length_category'] = pd.cut(phrase_df['length'], 
                                           bins=[0, 10, 15, 20, 100],
                                           labels=['Short (6-10)', 'Medium (11-15)', 
                                                   'Long (16-20)', 'Very Long (21+)'])
    
    print("Mean BUR by Phrase Length Category:")
    print(phrase_df.groupby('length_category')['mean_bur'].agg(['count', 'mean', 'std']))
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    print()
    
    findings = []
    
    # Position 1-10 trend
    if len(first_10_means) >= 10:
        slope, intercept, r_value, p_value_trend, stderr = stats.linregress(range(10), first_10_means)
        if p_value_trend < 0.05:
            direction = "increases" if slope > 0 else "decreases"
            findings.append(f"✓ BUR {direction} across positions 1-10 (p={p_value_trend:.4f})")
        else:
            findings.append(f"✗ No trend across positions 1-10 (p={p_value_trend:.4f})")
    
    # Beginning vs Middle vs End
    f_stat, p_value = stats.f_oneway(beginning_bur, middle_bur, end_bur)
    if p_value < 0.05:
        findings.append(f"✓ Beginning/Middle/End differ significantly (p={p_value:.4f})")
    else:
        findings.append(f"✗ Beginning/Middle/End do NOT differ (p={p_value:.4f})")
    
    # First vs Last
    t_stat, p_val = stats.ttest_rel(first_notes, last_notes)
    if p_val < 0.05:
        diff = np.mean(first_notes) - np.mean(last_notes)
        findings.append(f"✓ First vs Last differ by {diff:+.4f} (p={p_val:.4f})")
    else:
        findings.append(f"✗ First vs Last do NOT differ (p={p_val:.4f})")
    
    # Normalized position trend
    bin_means = [np.mean(normalized_bins[i]) for i in range(n_bins) if normalized_bins[i]]
    x = np.arange(len(bin_means))
    slope, intercept, r_value, p_value_trend, stderr = stats.linregress(x, bin_means)
    if p_value_trend < 0.05:
        direction = "increases" if slope > 0 else "decreases"
        findings.append(f"✓ BUR {direction} with normalized position (p={p_value_trend:.4f})")
    else:
        findings.append(f"✗ No trend with normalized position (p={p_value_trend:.4f})")
    
    # Phrase length correlations
    for metric in ['mean_bur', 'std_bur']:
        corr, p_val = stats.pearsonr(phrase_df['length'], phrase_df[metric])
        if p_val < 0.05:
            direction = "increases" if corr > 0 else "decreases"
            findings.append(f"✓ {metric} {direction} with phrase length (r={corr:+.3f}, p={p_val:.4f})")
    
    for finding in findings:
        print(finding)
    print()
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    analyze_positional_patterns()
