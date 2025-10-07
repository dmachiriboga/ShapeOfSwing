"""
BUR Histogram Visualization

Creates histogram plots showing the distribution of Beat-Upbeat Ratio (BUR) 
values for each performer. Saves individual PNG files for each performer.

Output:
- One histogram per performer showing BUR value distribution
- Bins are optimized for typical jazz swing ratios (1.0-2.5 range)
- Auto-adjusts if data falls outside typical range
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_utils import load_phrasebur_csv, ensure_output_dir, get_artist_from_id


def create_performer_bur_histograms(output_dir="outputs/bur_histograms"):
    """
    Create BUR histograms for each performer and save as PNG files.
    
    Args:
        output_dir: Directory to save histogram PNG files (default: outputs/bur_histograms)
        
    Returns:
        str: Path to output directory containing histogram files
        
    Notes:
        - Uses 0.1 bin width for typical jazz swing ratios
        - Default range is 1.0-2.5 (covers most swing timing variations)
        - Automatically expands range if data falls outside defaults
        - Creates one PNG file per performer
    """
    # Load the CSV
    df = load_phrasebur_csv()
    
    # Create output directory for histograms
    ensure_output_dir(output_dir)
    
    # Get all unique performers
    performers = df['id'].apply(lambda x: x.split('_')[0]).unique()
    
    # For each performer, collect all BUR values and plot histogram
    for performer in performers:
        # Select all rows for this performer
        performer_rows = df[df['id'].str.startswith(performer)]
        bur_values = performer_rows['swing_ratios'].astype(float)
        if bur_values.empty:
            continue
            
        # Histogram bins: 0.1 increments, typically 1.0 to 2.5 for jazz swing
        # Auto-adjust range if data falls outside typical range
        bin_min = max(0.5, np.floor(bur_values.min() * 10) / 10)  # Round down to nearest 0.1
        bin_max = min(3.5, np.ceil(bur_values.max() * 10) / 10)   # Round up to nearest 0.1
        bin_edges = np.arange(bin_min, bin_max + 0.1, 0.1).tolist()
        
        plt.figure(figsize=(8, 5))
        plt.hist(bur_values, bins=bin_edges, edgecolor='black', alpha=0.7)
        plt.title(f"BUR Histogram for {get_artist_from_id(performer + '_')}")
        plt.xlabel("BUR Value")
        plt.ylabel("Count")
        plt.xticks(bin_edges, rotation=45)
        plt.tight_layout()
        # Save to file
        filename = os.path.join(output_dir, f"{performer}_bur_histogram.png")
        plt.savefig(filename)
        plt.close()
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_performer_bur_histograms()
    print(f"Histograms saved to {output_dir}/ (one PNG per performer)")