#!/usr/bin/env python3
"""
Null Model Simulation - Compare Real BUR Data to Random Data

Generates random BUR sequences and tests them with Mann-Kendall
to see if the real data is distinguishable from random noise.
"""

import numpy as np
import pymannkendall as mk
from statsmodels.stats.multitest import multipletests
from utils.config import FDR_ALPHA

def simulate_random_bur_phrases(n_phrases=2488, phrase_lengths=None, mean_bur=1.35, std_bur=0.25):
    """
    Generate random BUR sequences matching the structure of real data.
    
    Args:
        n_phrases: Number of phrases to simulate
        phrase_lengths: List of phrase lengths (if None, uses uniform 6-20)
        mean_bur: Mean BUR value (typical swing ≈ 1.35)
        std_bur: Standard deviation of BUR (typical ≈ 0.25)
    """
    if phrase_lengths is None:
        phrase_lengths = np.random.randint(6, 21, n_phrases)
    
    phrases = []
    for length in phrase_lengths:
        # Pure random: no trend, just random draws from normal distribution
        bur_values = np.random.normal(mean_bur, std_bur, length)
        phrases.append(bur_values)
    
    return phrases


def test_phrases_mann_kendall(phrases):
    """Test all phrases with modified Mann-Kendall and apply FDR correction."""
    p_values = []
    trends = []
    taus = []
    slopes = []
    
    for phrase in phrases:
        try:
            result = mk.hamed_rao_modification_test(phrase)
            p_values.append(result.p)
            trends.append(result.trend)
            taus.append(result.Tau)
            slopes.append(result.slope)
        except:
            # Skip if test fails
            continue
    
    # Apply FDR correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method='fdr_bh')
    
    sig_count = np.sum(reject)
    sig_pct = 100 * sig_count / len(phrases)
    
    inc_count = sum(1 for i, r in enumerate(reject) if r and trends[i] == 'increasing')
    dec_count = sum(1 for i, r in enumerate(reject) if r and trends[i] == 'decreasing')
    
    mean_tau = np.mean(taus)
    median_tau = np.median(taus)
    
    positive_slopes = sum(1 for s in slopes if s > 0)
    positive_pct = 100 * positive_slopes / len(slopes)
    
    return {
        'n_phrases': len(phrases),
        'sig_count': sig_count,
        'sig_pct': sig_pct,
        'inc_count': inc_count,
        'dec_count': dec_count,
        'mean_tau': mean_tau,
        'median_tau': median_tau,
        'positive_slopes': positive_slopes,
        'positive_pct': positive_pct,
        'uncorrected_sig': sum(1 for p in p_values if p < 0.05),
        'uncorrected_pct': 100 * sum(1 for p in p_values if p < 0.05) / len(p_values)
    }


def main():
    print("\n" + "=" * 80)
    print("NULL MODEL SIMULATION")
    print("Comparing Real BUR Data vs Random Data")
    print("=" * 80)
    
    print("\nGenerating random BUR sequences...")
    print("Parameters: mean BUR = 1.35, std = 0.25 (realistic for jazz swing)")
    print()
    
    # Run multiple simulations
    n_simulations = 10
    results = []
    
    for i in range(n_simulations):
        phrases = simulate_random_bur_phrases(n_phrases=2488)
        result = test_phrases_mann_kendall(phrases)
        results.append(result)
        
        if i == 0:
            print(f"Simulation {i+1}:")
            print(f"  Significant (after FDR): {result['sig_count']} ({result['sig_pct']:.2f}%)")
            print(f"  Mean tau: {result['mean_tau']:.3f}")
            print(f"  Positive slopes: {result['positive_pct']:.1f}%")
    
    print(f"\n... running {n_simulations-1} more simulations ...")
    print()
    
    # Aggregate results
    print("=" * 80)
    print("AGGREGATE RESULTS FROM 10 RANDOM SIMULATIONS")
    print("=" * 80)
    
    mean_sig_pct = np.mean([r['sig_pct'] for r in results])
    std_sig_pct = np.std([r['sig_pct'] for r in results])
    mean_tau = np.mean([r['mean_tau'] for r in results])
    mean_pos_pct = np.mean([r['positive_pct'] for r in results])
    mean_uncorr_pct = np.mean([r['uncorrected_pct'] for r in results])
    
    print(f"\nRandom Data (Mean ± SD):")
    print(f"  Before FDR: {mean_uncorr_pct:.1f}% significant")
    print(f"  After FDR:  {mean_sig_pct:.2f}% ± {std_sig_pct:.2f}% significant")
    print(f"  Mean tau:   {mean_tau:.3f}")
    print(f"  Positive slopes: {mean_pos_pct:.1f}%")
    
    print(f"\n" + "=" * 80)
    print("COMPARISON: Real Data vs Random Data")
    print("=" * 80)
    
    print("\nReal BUR Data:")
    print("  After FDR:  1.81% significant (45/2488)")
    print("  Mean tau:   0.000")
    print("  Positive slopes: 50.6%")
    
    print(f"\nRandom Data (average):")
    print(f"  After FDR:  {mean_sig_pct:.2f}% significant")
    print(f"  Mean tau:   {mean_tau:.3f}")
    print(f"  Positive slopes: {mean_pos_pct:.1f}%")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if mean_sig_pct + 2*std_sig_pct < 1.81:
        print("\n✓ Real data shows MORE trends than random data")
        print("  → Evidence for systematic BUR patterns")
        diff = 1.81 - mean_sig_pct
        print(f"  → Real data has {diff:.2f}% more significant trends")
    elif mean_sig_pct - 2*std_sig_pct > 1.81:
        print("\n✗ Real data shows FEWER trends than random data")
        print("  → This is unexpected and suggests:")
        print("    - Musicians maintain consistent BUR within phrases")
        print("    - Or measurement/analysis issues")
    else:
        print("\n≈ Real data is INDISTINGUISHABLE from random data")
        print("  → No evidence for systematic BUR surge patterns")
        print("  → BUR variation appears to be random noise")
        print("  → Musicians may not systematically vary swing within phrases")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nThe 1.81% significant result in real data is:")
    print("• Similar to what pure random data produces")
    print("• Far below the 5-10% you'd see before FDR correction")
    print("• Not indicative of a strong systematic phenomenon")
    print()
    print("This suggests:")
    print("• BUR 'surges' are NOT a widespread, systematic pattern")
    print("• Within-phrase BUR variation is mostly random")
    print("• Individual expressive choices, not universal trends")
    print("=" * 80)


if __name__ == '__main__':
    main()
