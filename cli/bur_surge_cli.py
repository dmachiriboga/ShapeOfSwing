#!/usr/bin/env python3
"""
BUR Surge Analysis CLI

Analyzes linear trends in Beat-Upbeat Ratio (BUR) values across phrase positions
using simple linear regression with False Discovery Rate (FDR) correction for
multiple testing.

Statistical Approach:
- Linear regression on BUR vs. position for each phrase
- FDR correction (Benjamini-Hochberg) for multiple comparisons
- Reports slopes, confidence intervals, and corrected p-values
"""

import pandas as pd
from collections import defaultdict

from utils.data_utils import load_phrasebur_csv, get_artist_from_id, ensure_output_dir
from analysis.bur_surge_analysis import linear_trend_analysis, fdr_correction
from utils.config import FDR_ALPHA, MIN_BUR_VALUES, DW_AUTOCORR_THRESHOLD


def main():
    # Load the CSV
    df = load_phrasebur_csv()

    results = []
    all_p_values = []

    print("\nAnalyzing BUR trends across phrases...")
    print("=" * 60)

    # First pass: collect all trend analyses
    for (solo_id, phrase_id), group in df.groupby(['id', 'seg_id']):
        bur_values = group['swing_ratios'].astype(float).tolist()
        
        trend = linear_trend_analysis(bur_values)
        if trend is None:
            continue
        
        artist = get_artist_from_id(solo_id)
        
        results.append({
            'id': solo_id,
            'seg_id': phrase_id,
            'artist': artist,
            **trend
        })
        
        all_p_values.append(trend['p_value'])

    total_phrases = len(results)
    print(f"Analyzed {total_phrases} phrases with n >= {MIN_BUR_VALUES} BUR values")
    print()

    # Apply FDR correction
    print("Applying False Discovery Rate (FDR) correction...")
    reject, p_corrected = fdr_correction(all_p_values, alpha=FDR_ALPHA)
    
    # Add corrected p-values to results
    for i, result in enumerate(results):
        result['p_value_corrected'] = p_corrected[i]
        result['significant_fdr'] = reject[i]

    # Count significant trends
    sig_increase = sum(1 for r in results if r['significant_fdr'] and r['direction'] == 'increase')
    sig_decrease = sum(1 for r in results if r['significant_fdr'] and r['direction'] == 'decrease')

    # Calculate Durbin-Watson statistics
    dw_values = [r['durbin_watson'] for r in results]
    mean_dw = sum(dw_values) / len(dw_values)
    autocorr_phrases = sum(1 for dw in dw_values if dw < DW_AUTOCORR_THRESHOLD)  # Strong positive autocorrelation

    print(f"FDR correction complete (α = {FDR_ALPHA})")
    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Significant increase: {sig_increase} / {total_phrases} ({100*sig_increase/total_phrases:.1f}%)")
    print(f"Significant decrease: {sig_decrease} / {total_phrases} ({100*sig_decrease/total_phrases:.1f}%)")
    print()
    print("Autocorrelation Analysis:")
    print(f"Mean Durbin-Watson statistic: {mean_dw:.3f}")
    print(f"  (2.0 = no autocorrelation, <2.0 = positive, >2.0 = negative)")
    print(f"Phrases with strong autocorrelation (DW < {DW_AUTOCORR_THRESHOLD}): {autocorr_phrases} ({100*autocorr_phrases/total_phrases:.1f}%)")
    print()

    # Per-artist statistics
    artist_phrase_counts = defaultdict(int)
    artist_sig_counts_increase = defaultdict(int)
    artist_sig_counts_decrease = defaultdict(int)

    for result in results:
        artist = result['artist']
        artist_phrase_counts[artist] += 1
        
        if result['significant_fdr']:
            if result['direction'] == 'increase':
                artist_sig_counts_increase[artist] += 1
            elif result['direction'] == 'decrease':
                artist_sig_counts_decrease[artist] += 1

    # Sort artists by total phrase count
    sorted_artists = sorted(
        artist_phrase_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Display top performers
    n_top = int(input(f"How many top performers to display? (default 20): ") or "20")
    
    print()
    print("=" * 60)
    print(f"Top {n_top} performers (by phrase count):")
    print("=" * 60)
    
    for artist, count in sorted_artists[:n_top]:
        inc = artist_sig_counts_increase[artist]
        dec = artist_sig_counts_decrease[artist]
        inc_pct = 100 * inc / count if count > 0 else 0
        dec_pct = 100 * dec / count if count > 0 else 0
        
        print(f"{artist}:")
        print(f"  Phrases: {count}")
        print(f"  Increase: {inc} ({inc_pct:.1f}%)")
        print(f"  Decrease: {dec} ({dec_pct:.1f}%)")
        print()

    # Save detailed results
    ensure_output_dir()
    df_results = pd.DataFrame(results)
    
    # Reorder columns for clarity
    cols = ['id', 'seg_id', 'artist', 'n_values', 'slope', 'conf_interval', 
            'r2', 'p_value', 'p_value_corrected', 'significant_fdr', 'direction', 
            'durbin_watson', 'std_err', 'intercept']
    df_results = df_results[cols]
    
    output_file = 'outputs/bur_surge_results_fdr.csv'
    df_results.to_csv(output_file, index=False)
    print("=" * 60)
    print(f"Detailed results saved to: {output_file}")
    print()
    print("Statistical Notes:")
    print("- Used linear regression to detect BUR trends")
    print(f"- Applied Benjamini-Hochberg FDR correction (α = {FDR_ALPHA})")
    print("- Slope: BUR change per position (negative = decrease)")
    print("- conf_interval: 95% confidence interval for slope")
    print(f"- significant_fdr: True if FDR-corrected p < {FDR_ALPHA}")
    print("- durbin_watson: Autocorrelation test (2 = independent, <2 = positive autocorr)")
    print()
    print("Limitations:")
    print("- Linear regression assumes independence (often violated by musical data)")
    print("- Autocorrelation can lead to underestimated standard errors")
    print("- P-values may be optimistic; true significance may be even rarer")
    print("=" * 60)


if __name__ == '__main__':
    main()
