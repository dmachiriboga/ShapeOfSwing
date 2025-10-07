"""
Data Cleaning Script for FormBUR CSV

Removes NaN values from swing_ratios column and creates a clean dataset
for structural position analysis.
"""

import pandas as pd
import numpy as np

def clean_formbur_data():
    """Clean the formbur.csv data by removing NaN values."""
    
    print("Loading original formbur.csv...")
    df = pd.read_csv('data/formbur.csv', delimiter=';')
    print(f"Original data: {len(df)} rows")
    
    # Check for NaN values
    nan_count = df['swing_ratios'].isna().sum()
    print(f"NaN values in swing_ratios: {nan_count}")
    
    # Remove rows with NaN swing_ratios
    df_clean = df.dropna(subset=['swing_ratios'])
    print(f"After removing NaN: {len(df_clean)} rows")
    print(f"Removed: {len(df) - len(df_clean)} rows ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")
    
    # Additional data quality checks
    print(f"\nData quality checks:")
    print(f"Min swing_ratio: {df_clean['swing_ratios'].min():.4f}")
    print(f"Max swing_ratio: {df_clean['swing_ratios'].max():.4f}")
    print(f"Mean swing_ratio: {df_clean['swing_ratios'].mean():.4f}")
    
    # Check for any infinite values
    inf_count = np.isinf(df_clean['swing_ratios']).sum()
    if inf_count > 0:
        print(f"Infinite values found: {inf_count}")
        df_clean = df_clean[np.isfinite(df_clean['swing_ratios'])]
        print(f"After removing infinite values: {len(df_clean)} rows")
    
    # Save cleaned data
    output_file = 'data/formbur_clean.csv'
    df_clean.to_csv(output_file, sep=';', index=False)
    print(f"\nCleaned data saved to: {output_file}")
    
    return df_clean

if __name__ == '__main__':
    clean_df = clean_formbur_data()