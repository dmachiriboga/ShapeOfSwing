"""
BUR by Absolute Position
Shows mean BUR for each note position across all phrases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_utils import load_phrasebur_csv

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def create_bur_by_position_plot():
    """
    Create plot showing mean BUR by absolute position (1st, 2nd, 3rd note, etc.)
    """
    print("Loading data...")
    df = load_phrasebur_csv()
    
    # Group by phrase
    grouped = df.groupby(['id', 'seg_id'])
    
    print(f"Total phrases: {len(grouped):,}")
    print()
    
    # Collect BUR values by absolute position
    position_data = {}
    
    print("Collecting BUR values by position...")
    for (solo_id, seg_id), group in grouped:
        bur_values = group['swing_ratios'].tolist()
        
        # Store each position's value
        for pos, bur in enumerate(bur_values, start=1):
            if pos not in position_data:
                position_data[pos] = []
            position_data[pos].append(bur)
    
    # Calculate statistics for each position
    max_position = max(position_data.keys())
    print(f"Maximum phrase length: {max_position} notes")
    print()
    
    positions = []
    means = []
    sems = []
    counts = []
    
    for pos in range(1, max_position + 1):
        if pos in position_data:
            values = position_data[pos]
            positions.append(pos)
            means.append(np.mean(values))
            sems.append(np.std(values, ddof=1) / np.sqrt(len(values)))  # SEM
            counts.append(len(values))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot mean with error bars
    ax.plot(positions, means, marker='o', linewidth=2.5, markersize=8,
            color='#2E86AB', label='Mean BUR', markeredgecolor='black', 
            markeredgewidth=1.5, zorder=3)
    
    # Add shaded error region (mean ± SEM)
    ax.fill_between(positions, 
                     [m - s for m, s in zip(means, sems)],
                     [m + s for m, s in zip(means, sems)],
                     alpha=0.25, color='#2E86AB', label='± SEM', zorder=2)
    
    # Add horizontal line at overall mean
    overall_mean = df['swing_ratios'].mean()
    ax.axhline(y=overall_mean, color='gray', linestyle='--', 
              linewidth=2, alpha=0.6, label=f'Overall mean ({overall_mean:.3f})', zorder=1)
    
    # Formatting
    ax.set_xlabel('Note Position (1 = first note)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean BUR', fontsize=14, fontweight='bold')
    ax.set_title('BUR by Absolute Note Position Across All Phrases\n' +
                 f'(N = {len(grouped):,} phrases)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add gridlines
    ax.grid(True, alpha=0.3, which='both')
    ax.set_axisbelow(True)
    
    # Set x-axis to show all positions
    ax.set_xlim(0, max_position + 1)
    
    # Add legend
    legend = ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2)
    
    # Add sample size annotation
    ax.text(0.02, 0.98, f'Note: Sample size decreases with position\n' +
                        f'Position 1: {counts[0]:,} phrases\n' +
                        f'Position 10: {counts[9]:,} phrases (if exists)\n' +
                        f'Position {max_position}: {counts[-1]:,} phrases',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    
    output_path = 'outputs/phrase_shape_visualizations/bur_by_absolute_position.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ SUCCESS! Saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    plt.close()
    
    # Print summary statistics
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    print(f"{'Position':<10} {'Mean BUR':<12} {'SEM':<12} {'N phrases':<12}")
    print("-" * 80)
    
    # Show first 20 positions
    for i in range(min(20, len(positions))):
        pos = positions[i]
        mean = means[i]
        sem = sems[i]
        count = counts[i]
        print(f"{pos:<10} {mean:.4f}      {sem:.4f}      {count:,}")
    
    if len(positions) > 20:
        print(f"... ({len(positions) - 20} more positions)")
    
    print("-" * 80)
    print()
    
    # Check for first vs last trend
    first_mean = means[0]
    first_sem = sems[0]
    
    print(f"First note (position 1):  {first_mean:.4f} ± {first_sem:.4f}")
    print(f"Overall mean:             {overall_mean:.4f}")
    print(f"Difference:               {first_mean - overall_mean:+.4f}")
    print()
    
    if first_mean > overall_mean + 0.01:
        print("→ First notes tend to be HIGHER than average")
    elif first_mean < overall_mean - 0.01:
        print("→ First notes tend to be LOWER than average")
    else:
        print("→ First notes are similar to average")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    print("=" * 80)
    print("BUR BY ABSOLUTE POSITION")
    print("=" * 80)
    print()
    
    create_bur_by_position_plot()
    
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
