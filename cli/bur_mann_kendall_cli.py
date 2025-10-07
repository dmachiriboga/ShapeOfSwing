#!/usr/bin/env python3
"""
Mann-Kendall BUR Trend Analysis CLI

Non-parametric trend detection using Mann-Kendall tests.
Advantages over linear regression:
- No assumption of linear trend (detects any monotonic trend)
- Robust to outliers
- Doesn't assume normal distribution
- Can account for autocorrelation
"""

import pandas as pd
from collections import defaultdict
import numpy as np

from utils.data_utils import load_phrasebur_csv, get_artist_from_id, ensure_output_dir
from analysis.bur_surge_analysis import (
    mann_kendall_trend_analysis,
    mann_kendall_modified_analysis,
    sens_slope_analysis,
    fdr_correction
)
from utils.config import FDR_ALPHA, MIN_BUR_VALUES


def main():
    # Load the CSV
    df = load_phrasebur_csv()

    results = []
    all_p_values_original = []
    all_p_values_modified = []

    print("\n" + "=" * 80)
    print("MANN-KENDALL TREND ANALYSIS")
    print("=" * 80)
    print("\nNon-parametric trend detection for BUR values")
    print("Advantages:")
    print("  • No linearity assumption (detects any monotonic trend)")
    print("  • Robust to outliers")
    print("  • Works with non-normal distributions")
    print("  • Modified test accounts for autocorrelation")
    print()

    # First pass: collect all analyses
    for (solo_id, phrase_id), group in df.groupby(['id', 'seg_id']):
        bur_values = group['swing_ratios'].astype(float).tolist()
        
        if len(bur_values) < MIN_BUR_VALUES:
            continue
        
        original_mk = mann_kendall_trend_analysis(bur_values)
        modified_mk = mann_kendall_modified_analysis(bur_values)
        sens_slope = sens_slope_analysis(bur_values)
        
        if original_mk is None:
            continue
        
        artist = get_artist_from_id(solo_id)
        
        result = {
            'id': solo_id,
            'seg_id': phrase_id,
            'artist': artist,
            'n_values': len(bur_values),
            
            # Original Mann-Kendall
            'mk_trend': original_mk['trend'],
            'mk_p': original_mk['p_value'],
            'mk_tau': original_mk['tau'],
            'mk_z': original_mk['z_score'],
            'mk_slope': original_mk['slope'],
            
            # Modified Mann-Kendall (with autocorrelation adjustment)
            'mk_mod_trend': modified_mk['trend'],
            'mk_mod_p': modified_mk['p_value'],
            'mk_mod_tau': modified_mk['tau'],
            'mk_mod_z': modified_mk['z_score'],
            'mk_mod_slope': modified_mk['slope'],
            
            # Sen's slope with CI
            'sens_slope': sens_slope['slope'] if sens_slope else None,
            'sens_ci_lower': sens_slope['conf_interval'][0] if sens_slope else None,
            'sens_ci_upper': sens_slope['conf_interval'][1] if sens_slope else None,
        }
        
        results.append(result)
        all_p_values_original.append(original_mk['p_value'])
        all_p_values_modified.append(modified_mk['p_value'])

    total_phrases = len(results)
    print(f"✓ Analyzed {total_phrases} phrases with n >= {MIN_BUR_VALUES} BUR values")
    print()

    # Apply FDR correction
    print("=" * 80)
    print("APPLYING FDR CORRECTION")
    print("=" * 80)
    
    reject_orig, p_corrected_orig = fdr_correction(all_p_values_original, alpha=FDR_ALPHA)
    reject_mod, p_corrected_mod = fdr_correction(all_p_values_modified, alpha=FDR_ALPHA)
    
    for i, result in enumerate(results):
        result['mk_p_corrected'] = p_corrected_orig[i]
        result['mk_significant'] = reject_orig[i]
        result['mk_mod_p_corrected'] = p_corrected_mod[i]
        result['mk_mod_significant'] = reject_mod[i]
    
    print(f"FDR correction applied at α = {FDR_ALPHA}")
    print()

    # Count significant trends
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    # Original MK results
    print("\nOriginal Mann-Kendall Test:")
    sig_inc_orig = sum(1 for r in results if r['mk_significant'] and r['mk_trend'] == 'increasing')
    sig_dec_orig = sum(1 for r in results if r['mk_significant'] and r['mk_trend'] == 'decreasing')
    print(f"  Significant increasing: {sig_inc_orig} / {total_phrases} ({100*sig_inc_orig/total_phrases:.2f}%)")
    print(f"  Significant decreasing: {sig_dec_orig} / {total_phrases} ({100*sig_dec_orig/total_phrases:.2f}%)")
    
    # Modified MK results
    print("\nModified Mann-Kendall Test (autocorrelation-adjusted):")
    sig_inc_mod = sum(1 for r in results if r['mk_mod_significant'] and r['mk_mod_trend'] == 'increasing')
    sig_dec_mod = sum(1 for r in results if r['mk_mod_significant'] and r['mk_mod_trend'] == 'decreasing')
    print(f"  Significant increasing: {sig_inc_mod} / {total_phrases} ({100*sig_inc_mod/total_phrases:.2f}%)")
    print(f"  Significant decreasing: {sig_dec_mod} / {total_phrases} ({100*sig_dec_mod/total_phrases:.2f}%)")
    
    # Kendall's tau statistics
    print("\nKendall's Tau Statistics:")
    tau_values = [r['mk_tau'] for r in results]
    print(f"  Mean tau: {np.mean(tau_values):.3f}")
    print(f"  Median tau: {np.median(tau_values):.3f}")
    print(f"  Tau interpretation: 0.0-0.3 = weak, 0.3-0.5 = moderate, 0.5+ = strong")
    
    # Sen's slope statistics
    print("\nSen's Slope Statistics:")
    sens_slopes = [r['sens_slope'] for r in results if r['sens_slope'] is not None]
    if sens_slopes:
        print(f"  Mean Sen's slope: {np.mean(sens_slopes):.4f}")
        print(f"  Median Sen's slope: {np.median(sens_slopes):.4f}")
        positive_slopes = sum(1 for s in sens_slopes if s > 0)
        print(f"  Positive slopes: {positive_slopes} / {len(sens_slopes)} ({100*positive_slopes/len(sens_slopes):.1f}%)")

    # Comparison with uncorrected p-values
    print("\n" + "=" * 80)
    print("COMPARISON: Before vs After FDR Correction")
    print("=" * 80)
    
    uncorr_sig_orig = sum(1 for r in results if r['mk_p'] < 0.05)
    uncorr_sig_mod = sum(1 for r in results if r['mk_mod_p'] < 0.05)
    
    print(f"\nOriginal Mann-Kendall:")
    print(f"  Before FDR: {uncorr_sig_orig} significant ({100*uncorr_sig_orig/total_phrases:.1f}%)")
    print(f"  After FDR:  {sig_inc_orig + sig_dec_orig} significant ({100*(sig_inc_orig + sig_dec_orig)/total_phrases:.2f}%)")
    print(f"  Reduction: {uncorr_sig_orig - (sig_inc_orig + sig_dec_orig)} false positives removed")
    
    print(f"\nModified Mann-Kendall:")
    print(f"  Before FDR: {uncorr_sig_mod} significant ({100*uncorr_sig_mod/total_phrases:.1f}%)")
    print(f"  After FDR:  {sig_inc_mod + sig_dec_mod} significant ({100*(sig_inc_mod + sig_dec_mod)/total_phrases:.2f}%)")
    print(f"  Reduction: {uncorr_sig_mod - (sig_inc_mod + sig_dec_mod)} false positives removed")

    # Per-artist statistics
    print("\n" + "=" * 80)
    print("PER-PERFORMER ANALYSIS")
    print("=" * 80)
    
    artist_stats = defaultdict(lambda: {
        'total': 0,
        'mk_inc': 0,
        'mk_dec': 0,
        'mk_mod_inc': 0,
        'mk_mod_dec': 0,
        'tau_sum': 0,
        'sens_slope_sum': 0
    })
    
    for result in results:
        artist = result['artist']
        stats_dict = artist_stats[artist]
        stats_dict['total'] += 1
        stats_dict['tau_sum'] += result['mk_tau']
        if result['sens_slope'] is not None:
            stats_dict['sens_slope_sum'] += result['sens_slope']
        
        if result['mk_significant']:
            if result['mk_trend'] == 'increasing':
                stats_dict['mk_inc'] += 1
            elif result['mk_trend'] == 'decreasing':
                stats_dict['mk_dec'] += 1
        
        if result['mk_mod_significant']:
            if result['mk_mod_trend'] == 'increasing':
                stats_dict['mk_mod_inc'] += 1
            elif result['mk_mod_trend'] == 'decreasing':
                stats_dict['mk_mod_dec'] += 1

    # Sort by total phrases
    sorted_artists = sorted(artist_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    n_top = int(input(f"\nHow many top performers to display? (default 20): ") or "20")
    print()
    
    for artist, stats in sorted_artists[:n_top]:
        total = stats['total']
        avg_tau = stats['tau_sum'] / total
        avg_sens = stats['sens_slope_sum'] / total
        
        print(f"{artist} ({total} phrases):")
        print(f"  Original MK - Increase: {stats['mk_inc']:3d} ({100*stats['mk_inc']/total:5.1f}%)")
        print(f"  Original MK - Decrease: {stats['mk_dec']:3d} ({100*stats['mk_dec']/total:5.1f}%)")
        print(f"  Modified MK - Increase: {stats['mk_mod_inc']:3d} ({100*stats['mk_mod_inc']/total:5.1f}%)")
        print(f"  Modified MK - Decrease: {stats['mk_mod_dec']:3d} ({100*stats['mk_mod_dec']/total:5.1f}%)")
        print(f"  Avg Kendall's tau: {avg_tau:6.3f}")
        print(f"  Avg Sen's slope:   {avg_sens:6.4f}")
        print()

    # Save detailed results
    ensure_output_dir()
    df_results = pd.DataFrame(results)
    
    output_file = 'outputs/bur_mann_kendall_results.csv'
    df_results.to_csv(output_file, index=False)
    
    print("=" * 80)
    print(f"✓ Detailed results saved to: {output_file}")
    print()
    print("Statistical Notes:")
    print("• Mann-Kendall test is non-parametric (no normality assumption)")
    print("• Detects monotonic trends (not just linear)")
    print("• Kendall's tau measures trend strength (-1 to +1)")
    print("• Sen's slope is median of all pairwise slopes (robust to outliers)")
    print("• Modified MK adjusts for autocorrelation (Hamed-Rao method)")
    print("• FDR correction controls false discovery rate")
    print()
    print("Interpretation:")
    print("• tau ≈ 0: no trend")
    print("• tau = 0.0-0.3: weak trend")
    print("• tau = 0.3-0.5: moderate trend")
    print("• tau > 0.5: strong trend")
    print("=" * 80)


if __name__ == '__main__':
    main()
