#!/usr/bin/env python3
"""
BUR Variation Analysis CLI

Analyzes within-phrase consistency of Beat-Upbeat Ratio (BUR) values by 
calculating standard deviation for each phrase. Lower standard deviation 
indicates more consistent swing timing within a phrase.

Output:
- Overall average standard deviation across all phrases
- Per-artist average standard deviation (sorted by highest variation)
- CSV file with phrase-level variation statistics

Note: This is descriptive statistics only (no hypothesis testing).
"""

import pandas as pd
import numpy as np
from collections import defaultdict

from utils.data_utils import load_phrasebur_csv, get_artist_from_id, ensure_output_dir
from analysis.bur_variation_analysis import phrase_variation_stats
from utils.config import DEFAULT_TOP_N

def main():
    # Load the CSV
    df = load_phrasebur_csv()

    results = []
    artist_phrase_stds = defaultdict(list)
    total_phrases = 0

    for (solo_id, phrase_id), group in df.groupby(['id', 'seg_id']):
        bur_values = group['swing_ratios'].astype(float).tolist()
        stats = phrase_variation_stats(bur_values)
        if stats is None:
            continue
            
        artist = get_artist_from_id(solo_id)
        artist_phrase_stds[artist].append(stats['std_bur'])
        
        results.append({
            'id': solo_id,
            'seg_id': phrase_id,
            'artist': artist,
            **stats
        })
        total_phrases += 1

    # Convert to DataFrame for easy sorting/analysis
    variation_df = pd.DataFrame(results)

    # Compute average phrase stddev per artist
    artist_avg_std = []
    for artist, stds in artist_phrase_stds.items():
        avg_std = np.mean(stds) if stds else 0
        count = len(stds)
        artist_avg_std.append((artist, avg_std, count))

    # Print total phrase average std deviation
    if len(variation_df) > 0:
        total_avg_std = variation_df['std_bur'].mean()
        print(f"\nTotal phrase average std deviation: {total_avg_std:.3f}")
        print("(Note: Simple average across phrases; does not account for phrase length differences)")

    # Ask user for sorting preference
    print("\nSort order:")
    print("  1. Highest variation first (most variable swing)")
    print("  2. Lowest variation first (most consistent swing)")
    sort_choice = input("Enter choice (default 1): ").strip() or "1"
    
    if sort_choice == "2":
        # Sort ascending (lowest variation first)
        artist_avg_std.sort(key=lambda x: x[1], reverse=False)
        sort_label = "lowest variation (most consistent)"
    else:
        # Sort descending (highest variation first) - default
        artist_avg_std.sort(key=lambda x: x[1], reverse=True)
        sort_label = "highest variation (most variable)"

    # Ask user for number of top performers to display
    try:
        n_top = int(input(f"How many top performers to display? (default {DEFAULT_TOP_N}): ") or DEFAULT_TOP_N)
    except Exception:
        n_top = DEFAULT_TOP_N

    print()
    print("=" * 60)
    print(f"Top {n_top} performers (sorted by {sort_label}):")
    print("=" * 60)
    print("Higher std = more variable swing, Lower std = more consistent swing")
    print()
    for artist, avg_std, count in artist_avg_std[:n_top]:
        print(f"{artist}: avg phrase stddev = {avg_std:.3f} ({count} phrases)")

    # Save the phrase-level variation data
    ensure_output_dir()
    variation_df.to_csv("outputs/phrase_bur_variation.csv", index=False)
    print()
    print("=" * 60)
    print(f"Detailed results saved to: outputs/phrase_bur_variation.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()