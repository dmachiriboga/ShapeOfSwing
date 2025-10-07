#!/usr/bin/env python3
"""
BUR Histogram Visualization CLI

Creates histogram plots showing the distribution of Beat-Upbeat Ratio (BUR) 
values for each performer. Generates one PNG file per performer showing their
swing timing distribution.

Usage:
    python bur_histogram_cli.py
    
Output:
    Creates outputs/bur_histograms/ directory with one PNG per performer
    
Note:
    This is visualization only - no statistical analysis or significance testing.
    Histograms use 0.1 bin width and auto-adjust range to fit the data.
"""

from visualization.bur_histograms import create_performer_bur_histograms


def main():
    print("\nGenerating BUR histograms for each performer...")
    print("=" * 60)
    
    output_dir = create_performer_bur_histograms()
    
    print(f"\n✓ Histograms saved to {output_dir}/")
    print("  (one PNG file per performer)")
    print("=" * 60)


if __name__ == "__main__":
    main()