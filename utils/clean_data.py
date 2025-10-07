"""
Data cleaning utility for PhraseBur dataset.

Filters phrases to include only those with at least 6 BUR values,
ensuring statistical validity for regression and variation analyses.
"""

import pandas as pd
from pathlib import Path
from utils.config import MIN_BUR_VALUES


def clean_phrasebur_data(input_csv='data/phrasebur_raw.csv', output_csv='data/phrasebur_filtered.csv', min_bur_values=MIN_BUR_VALUES):
    """
    Clean PhraseBur dataset by filtering phrases with insufficient data points.
    
    Args:
        input_csv: Path to input CSV file (default: 'data/phrasebur_raw.csv')
        output_csv: Path to output CSV file (default: 'data/phrasebur_filtered.csv')
        min_bur_values: Minimum number of BUR values per phrase (default: 6)
    
    Returns:
        Dictionary with cleaning statistics
    """
    # Load the data
    df = pd.read_csv(input_csv, sep=';')
    
    original_rows = len(df)
    original_phrases = df.groupby(['id', 'seg_id']).ngroups
    
    # Count BUR values per phrase
    phrase_counts = df.groupby(['id', 'seg_id']).size().reset_index(name='n_values')
    
    # Filter phrases with at least min_bur_values
    valid_phrases = phrase_counts[phrase_counts['n_values'] >= min_bur_values][['id', 'seg_id']]
    
    # Merge to keep only valid phrases
    df_cleaned = df.merge(valid_phrases, on=['id', 'seg_id'], how='inner')
    
    cleaned_rows = len(df_cleaned)
    cleaned_phrases = df_cleaned.groupby(['id', 'seg_id']).ngroups
    
    # Save cleaned data
    df_cleaned.to_csv(output_csv, sep=';', index=False)
    
    # Compute statistics
    stats = {
        'original_rows': original_rows,
        'cleaned_rows': cleaned_rows,
        'rows_removed': original_rows - cleaned_rows,
        'rows_removed_pct': 100 * (original_rows - cleaned_rows) / original_rows,
        'original_phrases': original_phrases,
        'cleaned_phrases': cleaned_phrases,
        'phrases_removed': original_phrases - cleaned_phrases,
        'phrases_removed_pct': 100 * (original_phrases - cleaned_phrases) / original_phrases,
        'min_bur_values': min_bur_values
    }
    
    return stats


def main():
    """Command-line interface for data cleaning."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean PhraseBur dataset by filtering short phrases'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/phrasebur_raw.csv',
        help='Input CSV file (default: data/phrasebur_raw.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/phrasebur_filtered.csv',
        help='Output CSV file (default: data/phrasebur_filtered.csv)'
    )
    parser.add_argument(
        '--min-values', '-n',
        type=int,
        default=MIN_BUR_VALUES,
        help=f'Minimum number of BUR values per phrase (default: {MIN_BUR_VALUES})'
    )
    
    args = parser.parse_args()
    
    print(f"Cleaning {args.input}...")
    print(f"Minimum BUR values per phrase: {args.min_values}")
    print()
    
    stats = clean_phrasebur_data(
        input_csv=args.input,
        output_csv=args.output,
        min_bur_values=args.min_values
    )
    
    print("=" * 60)
    print("DATA CLEANING SUMMARY")
    print("=" * 60)
    print(f"Original dataset:")
    print(f"  Rows (BUR values):     {stats['original_rows']:,}")
    print(f"  Phrases:               {stats['original_phrases']:,}")
    print()
    print(f"Cleaned dataset (n >= {stats['min_bur_values']}):")
    print(f"  Rows (BUR values):     {stats['cleaned_rows']:,}")
    print(f"  Phrases:               {stats['cleaned_phrases']:,}")
    print()
    print(f"Removed:")
    print(f"  Rows:                  {stats['rows_removed']:,} ({stats['rows_removed_pct']:.1f}%)")
    print(f"  Phrases:               {stats['phrases_removed']:,} ({stats['phrases_removed_pct']:.1f}%)")
    print()
    print(f"Output saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
