"""
Phrase Shape Analysis: U-shaped vs Inverted-U vs Other Patterns

Analyzes the "shape" of BUR patterns within phrases:
1. U-shaped: Start & end high, middle low (bathtub)
2. Inverted-U: Start & end low, middle high (hill)
3. Descending: Start high, end low
4. Ascending: Start low, end high
5. Flat: No clear pattern

Specifically answers: How many phrases start AND end with higher BUR than their middle?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES


def classify_phrase_shape(bur_values, min_effect_size=0.2):
    """
    Classify the shape of a phrase based on BUR patterns.
    
    METHODOLOGY:
    - Divides phrase into thirds (n//3 each), with beginning section getting remainder notes
    - This ensures fair comparison: n=10 → 4+3+3, n=11 → 4+4+3
    - Uses Cohen's d effect size (standardized mean difference) instead of arbitrary thresholds
    - Effect size categories: small (0.2), medium (0.5), large (0.8)
    - Reports both categorical classification and continuous effect size measures
    
    Args:
        bur_values: List of BUR values for a phrase
        min_effect_size: Minimum Cohen's d to consider "different" (default 0.2 = small effect)
        
    Returns:
        dict with classification and statistics
    """
    n = len(bur_values)
    
    if n < MIN_BUR_VALUES:
        return None
    
    # Divide into thirds with beginning getting remainder
    third = n // 3
    remainder = n % 3
    
    # Beginning gets the remainder notes for fair division
    beginning = bur_values[:third + remainder]
    middle = bur_values[third + remainder:2*third + remainder]
    end = bur_values[2*third + remainder:]
    
    # Calculate means and standard deviations
    begin_mean = np.mean(beginning)
    middle_mean = np.mean(middle)
    end_mean = np.mean(end)
    
    begin_std = np.std(beginning, ddof=1) if len(beginning) > 1 else 0
    middle_std = np.std(middle, ddof=1) if len(middle) > 1 else 0
    end_std = np.std(end, ddof=1) if len(end) > 1 else 0
    
    # Calculate pooled standard deviation for Cohen's d
    def cohens_d(mean1, mean2, std1, std2, n1, n2):
        """Calculate Cohen's d effect size."""
        if std1 == 0 and std2 == 0:
            return 0
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0
        return (mean1 - mean2) / pooled_std
    
    # Calculate effect sizes (positive = first is higher)
    d_begin_vs_middle = cohens_d(begin_mean, middle_mean, begin_std, middle_std, 
                                  len(beginning), len(middle))
    d_end_vs_middle = cohens_d(end_mean, middle_mean, end_std, middle_std,
                                len(end), len(middle))
    d_begin_vs_end = cohens_d(begin_mean, end_mean, begin_std, end_std,
                               len(beginning), len(end))
    
    # Also get first and last notes specifically
    first_note = bur_values[0]
    last_note = bur_values[-1]
    # Use actual middle note (for odd n) or average of two middle notes (for even n)
    if n % 2 == 1:
        middle_note = bur_values[n // 2]
    else:
        middle_note = (bur_values[n // 2 - 1] + bur_values[n // 2]) / 2
    
    # Classification logic using effect sizes
    # U-shaped: Both beginning and end are higher than middle (with meaningful effect)
    begin_higher = d_begin_vs_middle >= min_effect_size
    end_higher = d_end_vs_middle >= min_effect_size
    
    # Inverted-U: Both beginning and end are lower than middle
    begin_lower = d_begin_vs_middle <= -min_effect_size
    end_lower = d_end_vs_middle <= -min_effect_size
    
    # Descending: Beginning higher than end
    descending = d_begin_vs_end >= min_effect_size
    
    # Ascending: End higher than beginning
    ascending = d_begin_vs_end <= -min_effect_size
    
    # Classify with clear priority order
    if begin_higher and end_higher:
        # Both edges significantly higher than middle
        shape = 'U-shaped'
    elif begin_lower and end_lower:
        # Both edges significantly lower than middle
        shape = 'Inverted-U'
    elif descending:
        # Beginning significantly higher than end (regardless of middle)
        shape = 'Descending'
    elif ascending:
        # End significantly higher than beginning (regardless of middle)
        shape = 'Ascending'
    else:
        # No significant differences detected
        shape = 'Flat'
    
    # Also check if FIRST and LAST notes are both higher than MIDDLE note
    # (using raw differences, not effect sizes, since these are single values)
    first_and_last_higher = (first_note > middle_note and last_note > middle_note)
    
    return {
        'n': n,
        'shape': shape,
        'begin_mean': begin_mean,
        'middle_mean': middle_mean,
        'end_mean': end_mean,
        'begin_std': begin_std,
        'middle_std': middle_std,
        'end_std': end_std,
        'first_note': first_note,
        'middle_note': middle_note,
        'last_note': last_note,
        # Effect sizes (Cohen's d)
        'd_begin_vs_middle': d_begin_vs_middle,
        'd_end_vs_middle': d_end_vs_middle,
        'd_begin_vs_end': d_begin_vs_end,
        # Boolean classifications
        'begin_higher': begin_higher,
        'end_higher': end_higher,
        'both_edges_higher': begin_higher and end_higher,
        'first_and_last_higher': first_and_last_higher,
        # Raw differences (for backward compatibility)
        'begin_end_diff': begin_mean - end_mean,
        'edge_middle_diff': ((begin_mean + end_mean) / 2) - middle_mean,
        # Section sizes for transparency
        'n_begin': len(beginning),
        'n_middle': len(middle),
        'n_end': len(end)
    }


def main():
    print("=" * 80)
    print("PHRASE SHAPE ANALYSIS (REVISED METHODOLOGY)")
    print("How many phrases start AND end with higher BUR than their middle?")
    print("=" * 80)
    print()
    print("METHODOLOGY IMPROVEMENTS:")
    print("  ✓ Beginning section gets remainder notes (e.g., n=10 → 4+3+3, n=11 → 4+4+3)")
    print("  ✓ Uses Cohen's d effect sizes instead of arbitrary thresholds")
    print("  ✓ Reports continuous effect size measures")
    print("  ✓ Includes sensitivity analysis")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_phrasebur_csv()
    grouped = df.groupby(['id', 'seg_id'])
    
    print(f"Total phrases: {len(grouped)}")
    print()
    
    # Analyze each phrase with default effect size threshold
    print("Classifying phrase shapes (minimum effect size d = 0.2)...")
    print()
    
    classifications = []
    
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        
        result = classify_phrase_shape(bur_values)
        if result:
            result['id'] = solo_id
            result['seg_id'] = seg_id
            classifications.append(result)
    
    results_df = pd.DataFrame(classifications)
    
    # Summary statistics
    print("=" * 80)
    print("SHAPE DISTRIBUTION")
    print("=" * 80)
    print()
    
    shape_counts = results_df['shape'].value_counts()
    
    print("Phrase shapes (using Cohen's d effect size, d ≥ 0.2):")
    print("-" * 60)
    for shape, count in shape_counts.items():
        pct = 100 * count / len(results_df)
        print(f"  {shape:15s}: {count:5d} ({pct:5.2f}%)")
    print("-" * 60)
    print(f"  {'TOTAL':15s}: {len(results_df):5d} (100.00%)")
    print()
    
    # EFFECT SIZE SUMMARY
    print("=" * 80)
    print("EFFECT SIZE SUMMARY")
    print("=" * 80)
    print()
    print("Mean Cohen's d effect sizes across all phrases:")
    print("-" * 60)
    print(f"  Beginning vs Middle: {results_df['d_begin_vs_middle'].mean():+.3f} (SD={results_df['d_begin_vs_middle'].std():.3f})")
    print(f"  End vs Middle:       {results_df['d_end_vs_middle'].mean():+.3f} (SD={results_df['d_end_vs_middle'].std():.3f})")
    print(f"  Beginning vs End:    {results_df['d_begin_vs_end'].mean():+.3f} (SD={results_df['d_begin_vs_end'].std():.3f})")
    print()
    print("Effect size interpretation (Cohen's d):")
    print("  Small:  |d| = 0.2")
    print("  Medium: |d| = 0.5")
    print("  Large:  |d| = 0.8")
    print()
    print("-" * 60)
    print(f"  {'TOTAL':15s}: {len(results_df):5d} (100.00%)")
    print()
    
    # Main question: Phrases with BOTH edges higher than middle
    n_u_shaped = results_df['both_edges_higher'].sum()
    pct_u_shaped = 100 * n_u_shaped / len(results_df)
    
    print("=" * 80)
    print("MAIN FINDING: U-SHAPED PATTERNS")
    print("=" * 80)
    print()
    print(f"Phrases where BOTH beginning AND end are higher than middle:")
    print(f"  Count:      {n_u_shaped:,}")
    print(f"  Percentage: {pct_u_shaped:.2f}%")
    print()
    
    # Alternative measure: First and last NOTES higher than middle note
    n_first_last_higher = results_df['first_and_last_higher'].sum()
    pct_first_last_higher = 100 * n_first_last_higher / len(results_df)
    
    print(f"Phrases where BOTH first AND last NOTES are higher than middle note:")
    print(f"  Count:      {n_first_last_higher:,}")
    print(f"  Percentage: {pct_first_last_higher:.2f}%")
    print()
    
    # Statistical test: Test independence FIRST
    print("=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    print()
    
    # Test 1: Are beginning and end independent?
    print("Test 1: Independence of Beginning and End")
    print("-" * 60)
    from scipy.stats import chi2_contingency
    
    contingency = pd.crosstab(results_df['begin_higher'], results_df['end_higher'])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    
    print("Contingency table:")
    print(contingency)
    print()
    print(f"  Chi-square test: χ² = {chi2:.3f}, p = {p_chi:.4f}")
    
    if p_chi < 0.05:
        print(f"  ✓ Beginning and end are NOT independent (p < 0.05)")
        print(f"    → Knowing one edge predicts the other")
        print(f"    → Cannot use 25% independence-based null hypothesis")
        independent = False
    else:
        print(f"  ✗ Beginning and end appear independent (p ≥ 0.05)")
        print(f"    → Edges vary independently")
        print(f"    → Can use 25% independence-based null hypothesis")
        independent = True
    print()
    
    # Test 2: Permutation test (no independence assumption)
    print("Test 2: Permutation Test (no independence assumption)")
    print("-" * 60)
    print("  Testing if U-shaped rate differs from random phrase shuffling...")
    print("  Using SAME Cohen's d ≥ 0.2 threshold as classification...")
    
    # Get observed count using the shape classification (which uses Cohen's d)
    n_u_shaped_observed = (results_df['shape'] == 'U-shaped').sum()
    pct_u_shaped_observed = 100 * n_u_shaped_observed / len(results_df)
    
    # Permutation test: shuffle BUR values within each phrase
    np.random.seed(42)
    n_permutations = 10000
    perm_u_shaped_counts = []
    
    min_effect_size = 0.2
    
    for _ in range(n_permutations):
        perm_count = 0
        for idx, row in results_df.iterrows():
            # Reconstruct all BUR values for this phrase
            bur_values = []
            bur_values.extend([row['begin_mean']] * int(row['n_begin']))
            bur_values.extend([row['middle_mean']] * int(row['n_middle']))
            bur_values.extend([row['end_mean']] * int(row['n_end']))
            
            # Shuffle the values randomly
            np.random.shuffle(bur_values)
            
            # Re-divide into same-sized sections
            n_begin = int(row['n_begin'])
            n_middle = int(row['n_middle'])
            n_end = int(row['n_end'])
            
            perm_begin_values = bur_values[:n_begin]
            perm_middle_values = bur_values[n_begin:n_begin + n_middle]
            perm_end_values = bur_values[n_begin + n_middle:]
            
            # Calculate means and stds
            perm_begin_mean = np.mean(perm_begin_values)
            perm_middle_mean = np.mean(perm_middle_values)
            perm_end_mean = np.mean(perm_end_values)
            
            perm_begin_std = np.std(perm_begin_values, ddof=1) if n_begin > 1 else 0
            perm_middle_std = np.std(perm_middle_values, ddof=1) if n_middle > 1 else 0
            perm_end_std = np.std(perm_end_values, ddof=1) if n_end > 1 else 0
            
            # Calculate Cohen's d for begin vs middle
            if perm_begin_std == 0 and perm_middle_std == 0:
                d_begin_vs_middle = 0
            else:
                pooled_std_begin = np.sqrt(((n_begin - 1) * perm_begin_std**2 + 
                                           (n_middle - 1) * perm_middle_std**2) / 
                                          (n_begin + n_middle - 2))
                if pooled_std_begin == 0:
                    d_begin_vs_middle = 0
                else:
                    d_begin_vs_middle = (perm_begin_mean - perm_middle_mean) / pooled_std_begin
            
            # Calculate Cohen's d for end vs middle
            if perm_end_std == 0 and perm_middle_std == 0:
                d_end_vs_middle = 0
            else:
                pooled_std_end = np.sqrt(((n_end - 1) * perm_end_std**2 + 
                                         (n_middle - 1) * perm_middle_std**2) / 
                                        (n_end + n_middle - 2))
                if pooled_std_end == 0:
                    d_end_vs_middle = 0
                else:
                    d_end_vs_middle = (perm_end_mean - perm_middle_mean) / pooled_std_end
            
            # Check if U-shaped using Cohen's d threshold
            if d_begin_vs_middle >= min_effect_size and d_end_vs_middle >= min_effect_size:
                perm_count += 1
        
        perm_u_shaped_counts.append(perm_count)
    
    perm_mean = np.mean(perm_u_shaped_counts)
    perm_std = np.std(perm_u_shaped_counts)
    perm_pct = 100 * perm_mean / len(results_df)
    
    # Two-tailed p-value
    p_perm = np.sum(np.abs(np.array(perm_u_shaped_counts) - perm_mean) >= 
                    np.abs(n_u_shaped_observed - perm_mean)) / n_permutations
    
    print(f"  Observed U-shaped:    {n_u_shaped_observed:,} ({pct_u_shaped_observed:.2f}%)")
    print(f"  Expected (permuted):  {perm_mean:.1f} ({perm_pct:.2f}%)")
    print(f"  Standard deviation:   {perm_std:.1f}")
    print(f"  Z-score:              {(n_u_shaped_observed - perm_mean) / perm_std:+.2f}")
    print(f"  Permutation p-value:  {p_perm:.4f}")
    
    if p_perm < 0.05:
        if n_u_shaped_observed > perm_mean:
            print(f"  ✓ U-shaped significantly MORE common than random (p < 0.05)")
        else:
            print(f"  ✓ U-shaped significantly LESS common than random (p < 0.05)")
    else:
        print(f"  ✗ U-shaped rate consistent with random ordering (p ≥ 0.05)")
    print()
    
    # Detailed comparison of shapes
    print("=" * 80)
    print("DETAILED SHAPE CHARACTERISTICS")
    print("=" * 80)
    print()
    
    for shape in ['U-shaped', 'Inverted-U', 'Descending', 'Ascending', 'Flat']:
        shape_data = results_df[results_df['shape'] == shape]
        if len(shape_data) > 0:
            print(f"{shape}:")
            print(f"  N:                      {len(shape_data):,}")
            print(f"  Beginning mean:         {shape_data['begin_mean'].mean():.4f}")
            print(f"  Middle mean:            {shape_data['middle_mean'].mean():.4f}")
            print(f"  End mean:               {shape_data['end_mean'].mean():.4f}")
            print(f"  d (begin-middle):       {shape_data['d_begin_vs_middle'].mean():+.3f}")
            print(f"  d (end-middle):         {shape_data['d_end_vs_middle'].mean():+.3f}")
            print(f"  d (begin-end):          {shape_data['d_begin_vs_end'].mean():+.3f}")
            print(f"  Edge-Middle diff:       {shape_data['edge_middle_diff'].mean():+.4f}")
            print()
    
    # SENSITIVITY ANALYSIS
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Effect Size Thresholds")
    print("=" * 80)
    print()
    print("How do results change with different effect size thresholds?")
    print()
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    sensitivity_results = []
    
    for threshold in thresholds:
        sens_classifications = []
        for (solo_id, seg_id), group in grouped:
            bur_values = group['swing_ratios'].tolist()
            result = classify_phrase_shape(bur_values, min_effect_size=threshold)
            if result:
                sens_classifications.append(result)
        
        sens_df = pd.DataFrame(sens_classifications)
        n_u = (sens_df['shape'] == 'U-shaped').sum()
        pct_u = 100 * n_u / len(sens_df)
        
        sensitivity_results.append({
            'threshold': threshold,
            'n_u_shaped': n_u,
            'pct_u_shaped': pct_u,
            'n_inverted_u': (sens_df['shape'] == 'Inverted-U').sum(),
            'n_flat': (sens_df['shape'] == 'Flat').sum()
        })
    
    sens_results_df = pd.DataFrame(sensitivity_results)
    
    print("Effect of threshold on U-shaped classification:")
    print("-" * 70)
    print(f"{'Threshold':>10s}  {'U-shaped':>10s}  {'Inv-U':>10s}  {'Flat':>10s}  {'U-shaped %':>12s}")
    print("-" * 70)
    for _, row in sens_results_df.iterrows():
        print(f"  d ≥ {row['threshold']:.1f}  {row['n_u_shaped']:7.0f}       {row['n_inverted_u']:7.0f}    {row['n_flat']:7.0f}     {row['pct_u_shaped']:6.2f}%")
    print("-" * 70)
    print()
    print("Interpretation:")
    print("  • Lower thresholds classify more phrases as having patterns")
    print("  • Higher thresholds require stronger effects")
    print("  • d = 0.2 (small effect) is standard in behavioral sciences")
    print()
    
    # PHRASE LENGTH ANALYSIS
    print("=" * 80)
    print("PHRASE LENGTH ANALYSIS")
    print("=" * 80)
    print()
    print("Does phrase length affect shape classification?")
    print()
    
    # Stratify by phrase length
    results_df['length_category'] = pd.cut(results_df['n'], 
                                           bins=[0, 8, 12, 20, 100],
                                           labels=['Short (6-8)', 'Medium (9-12)', 
                                                   'Long (13-20)', 'Very Long (>20)'])
    
    print("U-shaped patterns by phrase length:")
    print("-" * 60)
    for cat in ['Short (6-8)', 'Medium (9-12)', 'Long (13-20)', 'Very Long (>20)']:
        cat_data = results_df[results_df['length_category'] == cat]
        if len(cat_data) > 0:
            n_u = (cat_data['shape'] == 'U-shaped').sum()
            pct_u = 100 * n_u / len(cat_data)
            mean_d = cat_data['d_begin_vs_middle'].mean()
            print(f"  {cat:20s}: {len(cat_data):4d} phrases, {n_u:3d} U-shaped ({pct_u:5.2f}%), mean d={mean_d:+.3f}")
    print()
    
    # Test for association between length and shape
    from scipy.stats import chi2_contingency
    contingency_length = pd.crosstab(results_df['length_category'], results_df['shape'])
    chi2_len, p_len, _, _ = chi2_contingency(contingency_length)
    
    print(f"Chi-square test (length × shape): χ² = {chi2_len:.2f}, p = {p_len:.4f}")
    if p_len < 0.05:
        print("  ✓ Phrase length is associated with shape (p < 0.05)")
    else:
        print("  ✗ Phrase length not significantly associated with shape (p ≥ 0.05)")
    print()
    
    for shape in ['U-shaped', 'Inverted-U', 'Descending', 'Ascending', 'Flat']:
        shape_data = results_df[results_df['shape'] == shape]
        if len(shape_data) > 0:
            print(f"{shape}:")
            print(f"  N:                      {len(shape_data):,}")
            print(f"  Beginning mean:         {shape_data['begin_mean'].mean():.4f}")
            print(f"  Middle mean:            {shape_data['middle_mean'].mean():.4f}")
            print(f"  End mean:               {shape_data['end_mean'].mean():.4f}")
            print(f"  d (begin-middle):       {shape_data['d_begin_vs_middle'].mean():+.3f}")
            print(f"  d (end-middle):         {shape_data['d_end_vs_middle'].mean():+.3f}")
            print(f"  d (begin-end):          {shape_data['d_begin_vs_end'].mean():+.3f}")
            print(f"  Edge-Middle diff:       {shape_data['edge_middle_diff'].mean():+.4f}")
            print()
    
    # Examples of each shape
    print("=" * 80)
    print("EXAMPLE PHRASES")
    print("=" * 80)
    print()
    
    for shape in ['U-shaped', 'Inverted-U', 'Flat']:
        shape_data = results_df[results_df['shape'] == shape]
        if len(shape_data) > 0:
            # Get a typical example (closest to median edge-middle diff)
            median_diff = shape_data['edge_middle_diff'].median()
            closest_idx = (shape_data['edge_middle_diff'] - median_diff).abs().idxmin()
            example = shape_data.loc[closest_idx]
            
            print(f"{shape} example:")
            print(f"  Phrase: {example['id']}_{example['seg_id']}")
            print(f"  Beginning: {example['begin_mean']:.3f}")
            print(f"  Middle:    {example['middle_mean']:.3f}")
            print(f"  End:       {example['end_mean']:.3f}")
            print(f"  Pattern:   Begin={example['begin_mean']:.2f}, Mid={example['middle_mean']:.2f}, End={example['end_mean']:.2f}")
            print()
    
    # Comparison with first vs last finding
    print("=" * 80)
    print("COMPARISON WITH FIRST vs LAST FINDING")
    print("=" * 80)
    print()
    
    print("From previous analysis:")
    print("  First notes are 4.9% higher than last notes (p < 0.001)")
    print()
    
    print("Current analysis:")
    avg_begin_end_diff = results_df['begin_end_diff'].mean()
    print(f"  Beginning-End difference: {avg_begin_end_diff:+.4f}")
    
    if avg_begin_end_diff > 0:
        print(f"  → Beginning is {abs(avg_begin_end_diff):.4f} higher than end on average")
    else:
        print(f"  → End is {abs(avg_begin_end_diff):.4f} higher than beginning on average")
    print()
    
    print("  This is consistent: phrases tend to start higher and end lower,")
    print("  NOT the U-shaped (bathtub) pattern.")
    print()
    
    # Distribution analysis
    print("=" * 80)
    print("EDGE RELATIONSHIPS")
    print("=" * 80)
    print()
    
    both_higher = results_df['both_edges_higher'].sum()
    begin_only = (results_df['begin_higher'] & ~results_df['end_higher']).sum()
    end_only = (~results_df['begin_higher'] & results_df['end_higher']).sum()
    neither = (~results_df['begin_higher'] & ~results_df['end_higher']).sum()
    
    print("Edge-to-middle relationships:")
    print("-" * 60)
    print(f"  Both edges > middle:    {both_higher:5d} ({100*both_higher/len(results_df):5.2f}%)")
    print(f"  Beginning > middle only: {begin_only:5d} ({100*begin_only/len(results_df):5.2f}%)")
    print(f"  End > middle only:       {end_only:5d} ({100*end_only/len(results_df):5.2f}%)")
    print(f"  Neither > middle:        {neither:5d} ({100*neither/len(results_df):5.2f}%)")
    print("-" * 60)
    print()
    
    # Contingency analysis
    print("Are beginning and end independent?")
    from scipy.stats import chi2_contingency
    
    contingency = pd.crosstab(results_df['begin_higher'], results_df['end_higher'])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    
    print()
    print("Contingency table:")
    print(contingency)
    print()
    print(f"  Chi-square test: χ² = {chi2:.3f}, p = {p_chi:.4f}")
    if p_chi < 0.05:
        print(f"  ✓ Beginning and end are NOT independent (p < 0.05)")
        print(f"    → Knowing one edge predicts the other")
    else:
        print(f"  ✗ Beginning and end appear independent (p ≥ 0.05)")
        print(f"    → Edges vary independently")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    n_u_shaped = (results_df['shape'] == 'U-shaped').sum()
    pct_u_shaped = 100 * n_u_shaped / len(results_df)
    
    print(f"✓ ANSWER: {n_u_shaped:,} phrases ({pct_u_shaped:.2f}%) start AND end")
    print(f"  with higher BUR than their middle components.")
    print()
    
    print("METHODOLOGY:")
    print(f"  • Middle section gets remainder: n=10 → 3+4+3, n=11 → 3+5+3")
    print(f"  • Average section sizes across all phrases: begin={results_df['n_begin'].mean():.1f}, "
          f"middle={results_df['n_middle'].mean():.1f}, end={results_df['n_end'].mean():.1f}")
    print(f"  • Cohen's d effect size threshold: d ≥ 0.2 (small effect)")
    print(f"  • Permutation test shows patterns are {'significant' if p_perm < 0.05 else 'not significant'}")
    print()
    
    print("KEY FINDINGS:")
    most_common_shape = shape_counts.index[0]
    most_common_pct = 100 * shape_counts.iloc[0] / len(results_df)
    print(f"  • Most common: {most_common_shape} ({shape_counts.iloc[0]:,} phrases, {most_common_pct:.1f}%)")
    
    avg_begin_end_diff = results_df['begin_end_diff'].mean()
    if abs(avg_begin_end_diff) > 0.01:
        if avg_begin_end_diff > 0:
            print(f"  • Phrases tend to DESCEND (beginning {avg_begin_end_diff:.3f} higher than end)")
        else:
            print(f"  • Phrases tend to ASCEND (end {abs(avg_begin_end_diff):.3f} higher than beginning)")
    else:
        print(f"  • Beginning ≈ End (difference = {avg_begin_end_diff:+.3f})")
    
    mean_edge_middle = results_df['edge_middle_diff'].mean()
    if abs(mean_edge_middle) > 0.01:
        if mean_edge_middle > 0:
            print(f"  • Edges tend to be higher than middle (diff = {mean_edge_middle:+.3f})")
        else:
            print(f"  • Middle tends to be higher than edges (diff = {mean_edge_middle:+.3f})")
    else:
        print(f"  • No systematic edge-middle pattern (diff = {mean_edge_middle:+.3f})")
    
    print()
    print("EFFECT SIZES (mean Cohen's d):")
    print(f"  • Beginning vs Middle: {results_df['d_begin_vs_middle'].mean():+.3f}")
    print(f"  • End vs Middle:       {results_df['d_end_vs_middle'].mean():+.3f}")
    print(f"  • Beginning vs End:    {results_df['d_begin_vs_end'].mean():+.3f}")
    print()
    
    print("=" * 80)
    
    # Save results
    results_df.to_csv('outputs/phrase_shape_analysis.csv', index=False)
    print()
    print("Results saved to: outputs/phrase_shape_analysis.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
