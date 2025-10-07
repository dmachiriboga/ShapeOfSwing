"""
Localized BUR Surge Analysis CLI (with Seasonal Mann-Kendall)

Detects BUR surges happening in specific parts of phrases using:
1. Sliding window analysis - finds surges in overlapping sub-segments
2. Positional segment analysis - tests beginning, middle, end separately

Uses SEASONAL Mann-Kendall to account for periodic patterns in jazz phrasing.
This answers the question: "Are BUR surges happening in only part of the phrase?"
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from analysis.bur_surge_analysis import (
    sliding_window_surge_analysis_seasonal,
    sliding_window_surge_analysis_seasonal_all_windows,
    fdr_correction
)
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES, FDR_ALPHA


def analyze_localized_surges():
    """
    Perform localized surge analysis on all phrases and save results.
    """
    print("=" * 80)
    print("LOCALIZED BUR SURGE ANALYSIS (Seasonal Mann-Kendall)")
    print("Detecting BUR surges in sub-segments with periodic pattern adjustment")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_phrasebur_csv()
    
    # Group by phrase (combination of id and seg_id)
    grouped = df.groupby(['id', 'seg_id'])
    
    print(f"Total phrases to analyze: {len(grouped)}")
    print()
    
    results = []
    all_window_p_values = []  # Collect ALL p-values from ALL windows
    phrase_to_windows = {}  # Map phrases to their window indices
    
    # Process each phrase
    phrase_idx = 0
    for (solo_id, seg_id), group in grouped:
        # Get BUR values (already in temporal order from the dataset)
        bur_values = group['swing_ratios'].tolist()
        
        if len(bur_values) < MIN_BUR_VALUES:
            continue
        
        # Get artist name from id
        from utils.data_utils import get_artist_from_id
        artist = get_artist_from_id(solo_id)
        
        # Run sliding window analysis with seasonal Mann-Kendall
        # Use the version that returns ALL windows for proper FDR correction
        sliding_result_all = sliding_window_surge_analysis_seasonal_all_windows(
            bur_values,
            window_sizes=[4, 6, 8],  # Test 4, 6, 8-note windows
            min_tau=0.4,  # Moderate to strong trend (filter applied later)
            period=4  # 4-beat periodicity (common in jazz)
        )
        
        # Also run the original for backward compatibility with has_local_surge
        sliding_result = sliding_window_surge_analysis_seasonal(
            bur_values,
            window_sizes=[4, 6, 8],
            min_tau=0.4,
            alpha=0.05,
            period=4
        )
        
        # Collect ALL p-values from ALL windows for this phrase
        window_indices_for_phrase = []
        if sliding_result_all and sliding_result_all['all_windows']:
            for window in sliding_result_all['all_windows']:
                window_indices_for_phrase.append(len(all_window_p_values))
                all_window_p_values.append(window['p_value'])
        
        phrase_to_windows[phrase_idx] = {
            'indices': window_indices_for_phrase,
            'windows': sliding_result_all['all_windows'] if sliding_result_all else []
        }
        
        # Compile results for this phrase
        result_row = {
            'id': solo_id,
            'seg_id': seg_id,
            'artist': artist,
            'n_values': len(bur_values),
            'mean_bur': np.mean(bur_values),
            'std_bur': np.std(bur_values),
            'phrase_idx': phrase_idx  # Track phrase index for later mapping
        }
        
        phrase_idx += 1
        
        # Sliding window results
        if sliding_result:
            result_row['has_local_surge'] = sliding_result['has_local_surge']
            result_row['n_surge_windows'] = sliding_result['n_significant_windows']
            result_row['windows_tested'] = sliding_result['windows_tested']
            
            if sliding_result['strongest_surge']:
                strongest = sliding_result['strongest_surge']
                result_row['strongest_window_start'] = strongest['start_pos']
                result_row['strongest_window_end'] = strongest['end_pos']
                result_row['strongest_window_size'] = strongest['window_size']
                result_row['strongest_tau'] = strongest['tau']
                result_row['strongest_p_value'] = strongest['p_value']
                result_row['strongest_slope'] = strongest['sens_slope']
                result_row['strongest_direction'] = strongest['direction']
            else:
                result_row['strongest_window_start'] = None
                result_row['strongest_window_end'] = None
                result_row['strongest_window_size'] = None
                result_row['strongest_tau'] = None
                result_row['strongest_p_value'] = None
                result_row['strongest_slope'] = None
                result_row['strongest_direction'] = None
        else:
            result_row['has_local_surge'] = False
            result_row['n_surge_windows'] = 0
            result_row['windows_tested'] = 0
            result_row['strongest_window_start'] = None
            result_row['strongest_window_end'] = None
            result_row['strongest_window_size'] = None
            result_row['strongest_tau'] = None
            result_row['strongest_p_value'] = None
            result_row['strongest_slope'] = None
            result_row['strongest_direction'] = None
        
        # Positional segment results (removed - function doesn't exist)
        result_row['has_significant_segment'] = None
        for seg_label in ['beginning', 'middle', 'end']:
            result_row[f'{seg_label}_tau'] = None
            result_row[f'{seg_label}_p_value'] = None
            result_row[f'{seg_label}_slope'] = None
            result_row[f'{seg_label}_trend'] = None
        
        results.append(result_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"Analysis complete. Processed {len(results_df)} phrases.")
    print(f"Total windows tested across all phrases: {len(all_window_p_values)}")
    print()
    
    # Apply FDR correction to ALL window p-values across ALL phrases
    print("Applying FDR correction to ALL sliding window results...")
    print(f"  Correcting {len(all_window_p_values)} p-values from all windows...")
    
    if all_window_p_values:
        reject, p_corrected = fdr_correction(all_window_p_values, alpha=FDR_ALPHA)
        
        # Determine which phrases have at least one significant window after FDR
        results_df['significant_local_surge'] = False
        results_df['n_significant_windows_fdr'] = 0
        results_df['strongest_p_corrected'] = None
        results_df['strongest_tau_fdr'] = None
        results_df['strongest_window_start_fdr'] = None
        results_df['strongest_window_end_fdr'] = None
        results_df['strongest_direction_fdr'] = None
        
        for phrase_idx, phrase_data in phrase_to_windows.items():
            window_indices = phrase_data['indices']
            windows = phrase_data['windows']
            
            if not window_indices:
                continue
            
            # Get corrected p-values for this phrase's windows
            phrase_p_corrected = [p_corrected[i] for i in window_indices]
            
            # Apply min_tau threshold AND FDR significance
            significant_indices = []
            for i, (p_corr, window) in enumerate(zip(phrase_p_corrected, windows)):
                if p_corr < FDR_ALPHA and abs(window['tau']) >= 0.4:
                    significant_indices.append(i)
            
            n_sig = len(significant_indices)
            
            # Find row in dataframe
            df_idx = results_df[results_df['phrase_idx'] == phrase_idx].index[0]
            
            results_df.at[df_idx, 'significant_local_surge'] = n_sig > 0
            results_df.at[df_idx, 'n_significant_windows_fdr'] = n_sig
            
            # If any significant windows, find the strongest one (by tau)
            if n_sig > 0:
                significant_windows = [windows[i] for i in significant_indices]
                strongest = max(significant_windows, key=lambda x: abs(x['tau']))
                strongest_idx = next(i for i, w in enumerate(windows) if w == strongest)
                
                results_df.at[df_idx, 'strongest_p_corrected'] = phrase_p_corrected[strongest_idx]
                results_df.at[df_idx, 'strongest_tau_fdr'] = strongest['tau']
                results_df.at[df_idx, 'strongest_window_start_fdr'] = strongest['start_pos']
                results_df.at[df_idx, 'strongest_window_end_fdr'] = strongest['end_pos']
                results_df.at[df_idx, 'strongest_direction_fdr'] = strongest['direction']
        
        print(f"  FDR correction complete.")
        print(f"  Significant windows after FDR: {sum(reject)} / {len(all_window_p_values)}")
        print(f"  Phrases with at least one significant window: {results_df['significant_local_surge'].sum()} / {len(results_df)}")
    else:
        results_df['significant_local_surge'] = False
        results_df['n_significant_windows_fdr'] = 0
        results_df['strongest_p_corrected'] = None
        results_df['strongest_tau_fdr'] = None
        results_df['strongest_window_start_fdr'] = None
        results_df['strongest_window_end_fdr'] = None
        results_df['strongest_direction_fdr'] = None
    
    # Remove temporary phrase_idx column
    results_df = results_df.drop(columns=['phrase_idx'])
    
    # Apply FDR correction to positional segment results (removed - function doesn't exist)
    # Keeping columns for backward compatibility but marking as None
    for seg_label in ['beginning', 'middle', 'end']:
        p_corr_col = f'{seg_label}_p_corrected'
        sig_col = f'{seg_label}_significant'
        results_df[p_corr_col] = None
        results_df[sig_col] = False
    
    # Save results
    output_path = 'outputs/bur_localized_surge_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print()
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    # Sliding window results - BEFORE vs AFTER proper FDR correction
    n_with_windows = results_df['has_local_surge'].sum()
    n_significant_after_fdr = results_df['significant_local_surge'].sum()
    pct_before = 100 * n_with_windows / len(results_df)
    pct_after = 100 * n_significant_after_fdr / len(results_df)
    
    print("SLIDING WINDOW ANALYSIS:")
    print(f"  Total phrases analyzed: {len(results_df)}")
    print(f"  Total windows tested: {len(all_window_p_values)}")
    print()
    print(f"  BEFORE FDR CORRECTION:")
    print(f"    Phrases with detected window surges: {n_with_windows} / {len(results_df)} ({pct_before:.2f}%)")
    print()
    print(f"  AFTER PROPER FDR CORRECTION (all {len(all_window_p_values)} windows):")
    print(f"    Phrases with significant window surges: {n_significant_after_fdr} / {len(results_df)} ({pct_after:.2f}%)")
    print()
    
    if pct_before > 0:
        reduction_factor = pct_before / pct_after if pct_after > 0 else float('inf')
        print(f"  CORRECTION IMPACT:")
        print(f"    Reduction factor: {reduction_factor:.1f}x")
        print(f"    Absolute reduction: {pct_before - pct_after:.2f} percentage points")
    print()
    
    if n_significant_after_fdr > 0:
        sig_phrases = results_df[results_df['significant_local_surge']]
        print(f"  CHARACTERISTICS OF SIGNIFICANT SURGES (after FDR):")
        print(f"    Mean Kendall's tau: {sig_phrases['strongest_tau_fdr'].mean():.3f}")
        print(f"    Direction breakdown:")
        direction_counts = sig_phrases['strongest_direction_fdr'].value_counts()
        for direction, count in direction_counts.items():
            print(f"      {direction}: {count} ({100*count/n_significant_after_fdr:.1f}%)")
        print()
        
        # Show position distribution
        print("    Position in phrase (where surges start):")
        avg_start = sig_phrases['strongest_window_start_fdr'].mean()
        avg_phrase_len = sig_phrases['n_values'].mean()
        print(f"      Average start position: {avg_start:.1f} / {avg_phrase_len:.1f} ({100*avg_start/avg_phrase_len:.1f}% into phrase)")
    else:
        print(f"  No significant localized surges found after proper FDR correction.")
        print(f"  This suggests BUR patterns are generally STABLE within phrases.")
    
    print()
    print("POSITIONAL SEGMENT ANALYSIS:")
    print("  (Feature not available - functions need to be implemented)")
    print()
    print("=" * 80)
    print("PERFORMER BREAKDOWN (Top 10 with most localized surges)")
    print("=" * 80)
    print()
    
    performer_counts = results_df[results_df['significant_local_surge']]['artist'].value_counts().head(10)
    
    for performer, count in performer_counts.items():
        total_phrases = len(results_df[results_df['artist'] == performer])
        pct = 100 * count / total_phrases
        print(f"  {performer}: {count} / {total_phrases} phrases ({pct:.1f}%)")
    
    print()
    print("Analysis complete!")
    

if __name__ == '__main__':
    analyze_localized_surges()
