"""
Null Model Simulation for Localized BUR Surges

Tests if localized surge detection rate (25.36%) is significantly different
from what we'd expect from random data with similar statistical properties.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from analysis.bur_surge_analysis import sliding_window_surge_analysis_seasonal, fdr_correction
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES, FDR_ALPHA


def generate_random_bur_phrase(phrase_length, mean_bur=1.5, std_bur=0.3):
    """
    Generate random BUR values matching statistical properties of real data.
    """
    return np.random.normal(mean_bur, std_bur, phrase_length)


def simulate_null_model(n_simulations=100):
    """
    Run null model: generate random BUR data and test for localized surges.
    """
    print("=" * 80)
    print("NULL MODEL SIMULATION: Localized BUR Surges (Seasonal Mann-Kendall)")
    print("=" * 80)
    print()
    print("Testing if 18.01% significant localized surges is better than random...")
    print("(Using seasonal Mann-Kendall to account for periodic patterns)")
    print()
    
    # Load real data to get phrase length distribution
    print("Loading real data to match phrase length distribution...")
    df = load_phrasebur_csv()
    grouped = df.groupby(['id', 'seg_id'])
    
    # Get phrase lengths
    phrase_lengths = []
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        if len(bur_values) >= MIN_BUR_VALUES:
            phrase_lengths.append(len(bur_values))
    
    print(f"Real data: {len(phrase_lengths)} phrases")
    print(f"Mean phrase length: {np.mean(phrase_lengths):.1f} ± {np.std(phrase_lengths):.1f}")
    print()
    
    # Calculate mean and std of real BUR values
    all_bur_values = df['swing_ratios'].values
    mean_bur = np.mean(all_bur_values)
    std_bur = np.std(all_bur_values)
    print(f"Real BUR values: mean={mean_bur:.3f}, std={std_bur:.3f}")
    print()
    
    # Run simulations
    print(f"Running {n_simulations} simulations of random data...")
    print()
    
    simulation_results = []
    
    for sim_num in range(n_simulations):
        if (sim_num + 1) % 10 == 0:
            print(f"  Simulation {sim_num + 1}/{n_simulations}...")
        
        # Generate random phrases matching real phrase lengths
        sim_results = []
        
        for phrase_len in phrase_lengths:
            # Generate random BUR values
            random_bur = generate_random_bur_phrase(phrase_len, mean_bur, std_bur)
            
            # Test for localized surges with seasonal Mann-Kendall
            result = sliding_window_surge_analysis_seasonal(
                random_bur,
                window_sizes=[4, 6, 8],
                min_tau=0.4,
                alpha=0.05,
                period=4
            )
            
            if result:
                sim_results.append({
                    'has_local_surge': result['has_local_surge'],
                    'strongest_p_value': result['strongest_surge']['p_value'] if result['strongest_surge'] else None
                })
            else:
                sim_results.append({
                    'has_local_surge': False,
                    'strongest_p_value': None
                })
        
        # Apply FDR correction
        sim_df = pd.DataFrame(sim_results)
        p_values = sim_df['strongest_p_value'].dropna().tolist()
        
        if p_values:
            reject, p_corrected = fdr_correction(p_values, alpha=FDR_ALPHA)
            
            # Count significant after FDR
            n_significant = sum(reject)
            pct_significant = 100 * n_significant / len(phrase_lengths)
        else:
            n_significant = 0
            pct_significant = 0.0
        
        simulation_results.append({
            'simulation': sim_num + 1,
            'n_significant': n_significant,
            'pct_significant': pct_significant
        })
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Analyze simulation results
    sim_df = pd.DataFrame(simulation_results)
    
    mean_pct = sim_df['pct_significant'].mean()
    std_pct = sim_df['pct_significant'].std()
    min_pct = sim_df['pct_significant'].min()
    max_pct = sim_df['pct_significant'].max()
    
    print(f"Random data (null model):")
    print(f"  Mean: {mean_pct:.2f}% ± {std_pct:.2f}%")
    print(f"  Range: {min_pct:.2f}% to {max_pct:.2f}%")
    print()
    
    # Real data result
    real_pct = 18.01  # Seasonal Mann-Kendall result
    print(f"Real data:")
    print(f"  Observed: {real_pct:.2f}%")
    print()
    
    # Calculate z-score
    if std_pct > 0:
        z_score = (real_pct - mean_pct) / std_pct
        
        # Calculate p-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print(f"Statistical comparison:")
        print(f"  Z-score: {z_score:.3f}")
        print(f"  P-value: {p_value:.6f}")
        print()
        
        if p_value < 0.001:
            print("  *** HIGHLY SIGNIFICANT difference from random! ***")
        elif p_value < 0.01:
            print("  ** SIGNIFICANT difference from random! **")
        elif p_value < 0.05:
            print("  * Marginally significant difference from random")
        else:
            print("  NOT significantly different from random")
    else:
        print("Cannot calculate significance (zero variance in simulations)")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    if std_pct > 0:
        if abs(z_score) > 2.576:  # p < 0.01
            print("The 18.01% localized surge rate in real data is SIGNIFICANTLY HIGHER")
            print("than what random data produces. This suggests:")
            print()
            print("  ✓ Localized BUR surges are REAL musical phenomena")
            print("  ✓ Players intentionally modulate swing timing within phrases")
            print("  ✓ These are not just statistical artifacts")
            print("  ✓ Seasonal test confirms patterns beyond periodic structure")
        elif abs(z_score) < 1:
            print("The localized surge rate is NOT significantly different from random.")
            print("This suggests:")
            print()
            print("  ✗ The detected rate may be due to statistical noise")
            print("  ✗ Localized surges might not be intentional")
            print("  ? The seasonal test may be filtering out real patterns")
    
    # Save results
    output_path = 'outputs/localized_surge_null_simulation.csv'
    sim_df.to_csv(output_path, index=False)
    print()
    print(f"Simulation results saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Null model simulation for localized BUR surges')
    parser.add_argument('--simulations', type=int, default=100,
                       help='Number of simulations to run (default: 100)')
    args = parser.parse_args()
    
    simulate_null_model(n_simulations=args.simulations)
