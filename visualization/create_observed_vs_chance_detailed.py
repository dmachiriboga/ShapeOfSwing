"""
Detailed Observed vs Chance Comparison
Shows all 5 shape patterns with significance testing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def create_detailed_comparison():
    """
    Create detailed figure showing observed vs expected for all 5 patterns.
    Uses corrected permutation test results with Cohen's d >= 0.2 threshold.
    """
    print("Loading data...")
    df = pd.read_csv('outputs/phrase_shape_analysis.csv')
    total = len(df)
    print(f"Loaded {total:,} phrases")
    print()
    
    # Observed percentages (using Cohen's d >= 0.2)
    shapes = ['U-shaped', 'Inverted-U', 'Descending', 'Ascending', 'Flat']
    observed_counts = [(df['shape'] == shape).sum() for shape in shapes]
    observed_pcts = [100 * count / total for count in observed_counts]
    
    # Expected from permutation test (Cohen's d >= 0.2)
    expected_pcts = [23.20, 22.78, 22.38, 22.48, 9.16]
    expected_counts = [pct * total / 100 for pct in expected_pcts]
    
    # Z-scores from permutation test
    z_scores = [+3.14, +1.41, +0.06, +0.46, -7.65]
    
    # Determine significance
    significance = []
    for z in z_scores:
        if abs(z) > 2.58:  # p < 0.01
            significance.append('***')
        elif abs(z) > 1.96:  # p < 0.05
            significance.append('**')
        else:
            significance.append('ns')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(shapes))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, observed_pcts, width, 
                   label='Observed (Jazz)',
                   color='#2E86AB', alpha=0.85, 
                   edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, expected_pcts, width, 
                   label='Expected (Random)',
                   color='gray', alpha=0.85, 
                   edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}%',
                   ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    # Add significance stars (only for positive deviations)
    max_height = max(max(observed_pcts), max(expected_pcts))
    for i, (z, sig) in enumerate(zip(z_scores, significance)):
        y_pos = max_height + 1.5
        
        # Only add significance stars for positive Z-scores (more common than random)
        if sig != 'ns' and z > 0:
            ax.text(i, y_pos, sig, ha='center', va='bottom',
                   fontsize=18, fontweight='bold', 
                   color='green')
    
    # Formatting
    ax.set_ylabel('Percentage of Phrases (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Shape Pattern', fontsize=14, fontweight='bold')
    ax.set_title('Jazz Phrase Shapes: Observed vs Random Chance\n' +
                 '(Using Cohen\'s d ≥ 0.2 effect size threshold | 10,000 permutations)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, fontsize=12, fontweight='bold')
    legend = ax.legend(fontsize=13, loc='upper left', framealpha=0.95)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2)
    ax.set_ylim(0, max_height + 5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = 'outputs/phrase_shape_visualizations/7_observed_vs_chance_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ SUCCESS! Saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    plt.close()
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    print()
    print(f"{'Shape':<15} {'Observed':<12} {'Expected':<12} {'Z-score':<10} {'Sig.'}")
    print("-" * 70)
    for i, shape in enumerate(shapes):
        print(f"{shape:<15} {observed_pcts[i]:5.2f}%      {expected_pcts[i]:5.2f}%      "
              f"{z_scores[i]:+6.2f}     {significance[i]}")
    print("-" * 70)
    print()
    print("Interpretation:")
    print("  • Positive Z: More common in jazz than random shuffling")
    print("  • Negative Z: Less common in jazz than random shuffling")
    print("  • |Z| > 1.96: Statistically significant (p < 0.05)")
    print()


if __name__ == '__main__':
    print("=" * 80)
    print("CREATING DETAILED OBSERVED VS CHANCE COMPARISON")
    print("=" * 80)
    print()
    
    create_detailed_comparison()
    
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
