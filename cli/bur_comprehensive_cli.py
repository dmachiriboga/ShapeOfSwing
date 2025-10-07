#!/usr/bin/env python3
"""
Comprehensive BUR Surge Analysis CLI

Analyzes BUR trends using multiple models:
- Linear regression
- Exponential regression
- Logarithmic regression
- Quadratic regression
- Step change detection
- End-phrase surge detection

Statistical Approach:
- Tests all models with FDR correction for multiple comparisons
- Reports best-fitting model for each phrase
- Compares detection rates across different pattern types
"""

import pandas as pd
from collections import defaultdict

from utils.data_utils import load_phrasebur_csv, get_artist_from_id, ensure_output_dir
from analysis.bur_surge_analysis import (
    comprehensive_trend_analysis,
    fdr_correction
)
from utils.config import FDR_ALPHA, MIN_BUR_VALUES


def main():
    # Load the CSV
    df = load_phrasebur_csv()

    results = []
    all_p_values = {
        'linear': [],
        'exponential': [],
        'logarithmic': [],
        'quadratic': [],
        'step_change': [],
        'end_surge': []
    }

    print("\n" + "=" * 80)
    print("COMPREHENSIVE BUR TREND ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing BUR patterns using multiple models...")
    print(f"Minimum phrase length: {MIN_BUR_VALUES} BUR values")
    print()

    # First pass: collect all trend analyses
    for (solo_id, phrase_id), group in df.groupby(['id', 'seg_id']):
        bur_values = group['swing_ratios'].astype(float).tolist()
        
        analysis = comprehensive_trend_analysis(bur_values)
        
        if analysis['linear'] is None:
            continue
        
        artist = get_artist_from_id(solo_id)
        
        # Flatten results for DataFrame
        result = {
            'id': solo_id,
            'seg_id': phrase_id,
            'artist': artist,
            'n_values': len(bur_values),
            'best_model': analysis['best_model']
        }
        
        # Add linear results
        if analysis['linear']:
            result.update({
                'linear_slope': analysis['linear']['slope'],
                'linear_r2': analysis['linear']['r2'],
                'linear_p': analysis['linear']['p_value'],
                'linear_direction': analysis['linear']['direction']
            })
            all_p_values['linear'].append(analysis['linear']['p_value'])
        
        # Add exponential results
        if analysis['exponential']:
            result.update({
                'exp_growth_rate': analysis['exponential']['growth_rate'],
                'exp_r2': analysis['exponential']['r2'],
                'exp_p': analysis['exponential']['p_value'],
                'exp_direction': analysis['exponential']['direction']
            })
            all_p_values['exponential'].append(analysis['exponential']['p_value'])
        
        # Add logarithmic results
        if analysis['logarithmic']:
            result.update({
                'log_coef': analysis['logarithmic']['log_coefficient'],
                'log_r2': analysis['logarithmic']['r2'],
                'log_p': analysis['logarithmic']['p_value'],
                'log_direction': analysis['logarithmic']['direction']
            })
            all_p_values['logarithmic'].append(analysis['logarithmic']['p_value'])
        
        # Add quadratic results
        if analysis['quadratic']:
            result.update({
                'quad_coef': analysis['quadratic']['quadratic_coef'],
                'quad_r2': analysis['quadratic']['r2'],
                'quad_p': analysis['quadratic']['p_value'],
                'quad_shape': analysis['quadratic']['shape']
            })
            all_p_values['quadratic'].append(analysis['quadratic']['p_value'])
        
        # Add step change results
        if analysis['step_change']:
            result.update({
                'step_detected': analysis['step_change']['has_step'],
                'step_position': analysis['step_change']['step_position'],
                'step_magnitude': analysis['step_change']['step_magnitude'],
                'step_p': analysis['step_change']['p_value'],
                'step_direction': analysis['step_change']['direction']
            })
            all_p_values['step_change'].append(analysis['step_change']['p_value'])
        
        # Add end surge results
        if analysis['end_surge']:
            result.update({
                'end_surge_detected': analysis['end_surge']['has_end_surge'],
                'end_surge_magnitude': analysis['end_surge']['surge_magnitude'],
                'end_surge_p': analysis['end_surge']['p_value'],
                'end_surge_effect_size': analysis['end_surge']['effect_size']
            })
            all_p_values['end_surge'].append(analysis['end_surge']['p_value'])
        
        results.append(result)

    total_phrases = len(results)
    print(f"✓ Analyzed {total_phrases} phrases")
    print()

    # Apply FDR correction for each model
    print("=" * 80)
    print("APPLYING FDR CORRECTION (Benjamini-Hochberg)")
    print("=" * 80)
    
    fdr_results = {}
    for model_name, p_vals in all_p_values.items():
        if len(p_vals) > 0:
            reject, p_corrected = fdr_correction(p_vals, alpha=FDR_ALPHA)
            fdr_results[model_name] = {
                'reject': reject,
                'p_corrected': p_corrected
            }
            
            # Add corrected p-values and significance to results
            for i, result in enumerate(results):
                result[f'{model_name}_p_corrected'] = p_corrected[i]
                result[f'{model_name}_significant'] = reject[i]
    
    print(f"FDR correction applied at α = {FDR_ALPHA}")
    print()

    # Count significant trends by model
    print("=" * 80)
    print("OVERALL RESULTS BY MODEL TYPE")
    print("=" * 80)
    
    for model_name in ['linear', 'exponential', 'logarithmic', 'quadratic', 'step_change', 'end_surge']:
        sig_key = f'{model_name}_significant'
        if sig_key in results[0]:
            sig_count = sum(1 for r in results if r.get(sig_key, False))
            sig_pct = 100 * sig_count / total_phrases
            
            # Count by direction
            if model_name in ['linear', 'exponential', 'logarithmic']:
                dir_key = f'{model_name}_direction'
                inc_count = sum(1 for r in results if r.get(sig_key, False) and r.get(dir_key) == 'increase')
                dec_count = sum(1 for r in results if r.get(sig_key, False) and r.get(dir_key) == 'decrease')
                
                print(f"\n{model_name.upper()} Model:")
                print(f"  Total significant: {sig_count} / {total_phrases} ({sig_pct:.2f}%)")
                print(f"    • Increases: {inc_count}")
                print(f"    • Decreases: {dec_count}")
            elif model_name == 'quadratic':
                u_count = sum(1 for r in results if r.get(sig_key, False) and r.get('quad_shape') == 'u_shaped')
                inv_u_count = sum(1 for r in results if r.get(sig_key, False) and r.get('quad_shape') == 'inverted_u')
                
                print(f"\n{model_name.upper()} Model:")
                print(f"  Total significant: {sig_count} / {total_phrases} ({sig_pct:.2f}%)")
                print(f"    • U-shaped: {u_count}")
                print(f"    • Inverted-U: {inv_u_count}")
            elif model_name == 'step_change':
                inc_count = sum(1 for r in results if r.get('step_detected', False) and r.get('step_direction') == 'increase')
                dec_count = sum(1 for r in results if r.get('step_detected', False) and r.get('step_direction') == 'decrease')
                
                print(f"\n{model_name.upper()} Detection:")
                print(f"  Total detected: {sig_count} / {total_phrases} ({sig_pct:.2f}%)")
                print(f"    • Upward steps: {inc_count}")
                print(f"    • Downward steps: {dec_count}")
            elif model_name == 'end_surge':
                surge_count = sum(1 for r in results if r.get('end_surge_detected', False))
                
                print(f"\n{model_name.upper()} Detection:")
                print(f"  End surges detected: {surge_count} / {total_phrases} ({sig_pct:.2f}%)")

    # Best model statistics
    print("\n" + "=" * 80)
    print("BEST-FITTING MODEL DISTRIBUTION")
    print("=" * 80)
    model_counts = defaultdict(int)
    for r in results:
        if r['best_model']:
            model_counts[r['best_model']] += 1
    
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_phrases
        print(f"  {model}: {count} ({pct:.1f}%)")

    # Per-artist statistics
    print("\n" + "=" * 80)
    print("PER-PERFORMER ANALYSIS")
    print("=" * 80)
    
    artist_stats = defaultdict(lambda: {
        'total': 0,
        'linear_inc': 0,
        'exp_inc': 0,
        'log_inc': 0,
        'quad_sig': 0,
        'step_detected': 0,
        'end_surge': 0
    })
    
    for result in results:
        artist = result['artist']
        artist_stats[artist]['total'] += 1
        
        if result.get('linear_significant', False) and result.get('linear_direction') == 'increase':
            artist_stats[artist]['linear_inc'] += 1
        if result.get('exponential_significant', False) and result.get('exp_direction') == 'increase':
            artist_stats[artist]['exp_inc'] += 1
        if result.get('logarithmic_significant', False) and result.get('log_direction') == 'increase':
            artist_stats[artist]['log_inc'] += 1
        if result.get('quadratic_significant', False):
            artist_stats[artist]['quad_sig'] += 1
        if result.get('step_detected', False):
            artist_stats[artist]['step_detected'] += 1
        if result.get('end_surge_detected', False):
            artist_stats[artist]['end_surge'] += 1

    # Sort by total phrases
    sorted_artists = sorted(artist_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    n_top = int(input(f"\nHow many top performers to display? (default 20): ") or "20")
    print()
    
    for artist, stats in sorted_artists[:n_top]:
        total = stats['total']
        print(f"{artist} ({total} phrases):")
        print(f"  Linear increase:      {stats['linear_inc']:3d} ({100*stats['linear_inc']/total:5.1f}%)")
        print(f"  Exponential increase: {stats['exp_inc']:3d} ({100*stats['exp_inc']/total:5.1f}%)")
        print(f"  Logarithmic increase: {stats['log_inc']:3d} ({100*stats['log_inc']/total:5.1f}%)")
        print(f"  Quadratic patterns:   {stats['quad_sig']:3d} ({100*stats['quad_sig']/total:5.1f}%)")
        print(f"  Step changes:         {stats['step_detected']:3d} ({100*stats['step_detected']/total:5.1f}%)")
        print(f"  End-phrase surges:    {stats['end_surge']:3d} ({100*stats['end_surge']/total:5.1f}%)")
        print()

    # Save detailed results
    ensure_output_dir()
    df_results = pd.DataFrame(results)
    
    output_file = 'outputs/bur_comprehensive_results.csv'
    df_results.to_csv(output_file, index=False)
    
    print("=" * 80)
    print(f"✓ Detailed results saved to: {output_file}")
    print()
    print("Statistical Notes:")
    print("• Multiple models tested to capture different surge patterns")
    print("• FDR correction applied separately for each model type")
    print("• Best model = highest R² among curve-fitting models")
    print("• Step changes use t-test to compare phrase segments")
    print("• End surges compare last 25% of phrase to rest")
    print("=" * 80)


if __name__ == '__main__':
    main()
