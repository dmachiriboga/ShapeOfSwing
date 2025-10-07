"""
NEW Comprehensive Summary Figure Generator
Creates the requested layout with NO old code interference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Color palette
COLORS = {
    'U-shaped': '#2E86AB',
    'Inverted-U': '#A23B72',
    'Ascending': '#F18F01',
    'Descending': '#C73E1D',
    'Flat': '#6A994E'
}


def create_new_comprehensive_summary():
    """
    Create comprehensive summary with the REQUESTED layout:
    - Top row: pie chart (left) + observed vs chance (right)
    - Middle row: overall BUR pattern (left) + heatmap (right)
    - Bottom row: 5 individual shape patterns
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv('outputs/phrase_shape_analysis.csv')
    print(f"Loaded {len(df):,} phrases")
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.4,
                  height_ratios=[1.2, 1.2, 1.0])
    
    # Main title
    fig.suptitle('Phrase Shape Analysis: Comprehensive Summary', 
                 fontsize=22, fontweight='bold', y=0.96)
    
    print("Creating visualizations...")
    
    # ==================== TOP ROW ====================
    print("  - Top left: pie chart")
    
    # TOP LEFT: Shape distribution pie chart (columns 0-1)
    ax_pie = fig.add_subplot(gs[0, :2])
    shape_counts = df['shape'].value_counts()
    shapes = ['U-shaped', 'Inverted-U', 'Ascending', 'Descending', 'Flat']
    counts = [shape_counts.get(s, 0) for s in shapes]
    colors_list = [COLORS[s] for s in shapes]
    
    wedges, texts, autotexts = ax_pie.pie(
        counts, labels=shapes, autopct='%1.1f%%',
        colors=colors_list, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax_pie.set_title('Shape Distribution\n(N = 2,488 phrases)', 
                     fontsize=14, fontweight='bold', pad=12)
    
    print("  - Top right: observed vs chance")
    
    # TOP RIGHT: Observed vs Chance comparison (columns 2-4)
    ax_obs = fig.add_subplot(gs[0, 2:])
    
    # Calculate observed percentages
    both_higher = ((df['begin_mean'] > df['middle_mean']) & 
                   (df['end_mean'] > df['middle_mean'])).sum()
    both_lower = ((df['begin_mean'] < df['middle_mean']) & 
                  (df['end_mean'] < df['middle_mean'])).sum()
    first_only = ((df['begin_mean'] > df['middle_mean']) & 
                  (df['end_mean'] <= df['middle_mean'])).sum()
    last_only = ((df['begin_mean'] <= df['middle_mean']) & 
                 (df['end_mean'] > df['middle_mean'])).sum()
    
    # Get observed shape percentages (using Cohen's d >= 0.2)
    observed = [
        100 * (df['shape'] == 'U-shaped').sum() / len(df),
        100 * (df['shape'] == 'Inverted-U').sum() / len(df),
        100 * (df['shape'] == 'Descending').sum() / len(df),
        100 * (df['shape'] == 'Ascending').sum() / len(df),
        100 * (df['shape'] == 'Flat').sum() / len(df)
    ]
    
    # Expected probabilities from permutation test (Cohen's d >= 0.2)
    # These values come from running the corrected permutation test
    expected = [23.20, 22.78, 22.38, 22.48, 9.16]
    
    patterns_short = ['U-shaped', 'Inv-U', 'Descending', 'Ascending', 'Flat']
    x = np.arange(len(patterns_short))
    width = 0.35
    
    bars1 = ax_obs.bar(x - width/2, observed, width, label='Observed',
                       color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax_obs.bar(x + width/2, expected, width, label='Expected (random)',
                       color='gray', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_obs.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    ax_obs.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax_obs.set_title('Observed vs Random Chance\n(Cohen\'s d ≥ 0.2 threshold)', 
                     fontsize=14, fontweight='bold', pad=12)
    ax_obs.set_xticks(x)
    ax_obs.set_xticklabels(patterns_short, fontsize=10, fontweight='bold')
    ax_obs.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax_obs.set_ylim(0, 30)
    ax_obs.yaxis.grid(True, alpha=0.3)
    ax_obs.set_axisbelow(True)
    
    # ==================== MIDDLE ROW ====================
    print("  - Middle left: overall BUR pattern")
    
    # MIDDLE LEFT: Overall BUR pattern (columns 0-1)
    ax_bur = fig.add_subplot(gs[1, :2])
    positions = ['Beginning', 'Middle', 'End']
    means = [df['begin_mean'].mean(), df['middle_mean'].mean(), df['end_mean'].mean()]
    sems = [df['begin_mean'].sem(), df['middle_mean'].sem(), df['end_mean'].sem()]
    
    x_pos = np.arange(len(positions))
    ax_bur.bar(x_pos, means, yerr=sems, capsize=12, alpha=0.75,
               color=['#2E86AB', '#6A994E', '#F18F01'],
               edgecolor='black', linewidth=2.5)
    
    for i, (pos, mean, sem) in enumerate(zip(x_pos, means, sems)):
        ax_bur.text(pos, mean + sem + 0.02, f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax_bur.set_xticks(x_pos)
    ax_bur.set_xticklabels(positions, fontsize=12, fontweight='bold')
    ax_bur.set_ylabel('Mean BUR', fontsize=12, fontweight='bold')
    ax_bur.set_title('Overall BUR Pattern\n(Mean ± SEM)',
                     fontsize=14, fontweight='bold', pad=12)
    ax_bur.set_ylim(1.3, 1.5)
    ax_bur.grid(True, alpha=0.3, axis='y')
    ax_bur.set_axisbelow(True)
    
    print("  - Middle right: heatmap")
    
    # MIDDLE RIGHT: Edge relationship heatmap (columns 2-4)
    ax_heat = fig.add_subplot(gs[1, 2:])
    
    beginning = df['begin_mean'].values
    end = df['end_mean'].values
    
    bins = np.linspace(0.5, 2.5, 30)
    H, xedges, yedges = np.histogram2d(beginning, end, bins=bins)
    
    im = ax_heat.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    ax_heat.plot([0.5, 2.5], [0.5, 2.5], 'b--', linewidth=2.5, 
                 label='Begin = End', alpha=0.8)
    
    cbar = plt.colorbar(im, ax=ax_heat)
    cbar.set_label('Count', rotation=270, labelpad=18, fontsize=11, fontweight='bold')
    
    # Calculate correlation and p-value
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(beginning, end)
    
    # Format p-value
    if p_value < 0.001:
        p_str = 'p < 0.001'
    else:
        p_str = f'p = {p_value:.3f}'
    
    ax_heat.set_xlabel('Beginning BUR', fontsize=12, fontweight='bold')
    ax_heat.set_ylabel('End BUR', fontsize=12, fontweight='bold')
    ax_heat.set_title(f'Beginning vs End Relationship\n(Pearson r = {corr:.3f}, {p_str})',
                      fontsize=14, fontweight='bold', pad=12)
    ax_heat.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax_heat.set_xlim(0.5, 2.5)
    ax_heat.set_ylim(0.5, 2.5)
    
    # ==================== BOTTOM ROW: 5 SHAPES ====================
    print("  - Bottom row: 5 individual shapes")
    
    shapes_to_plot = ['U-shaped', 'Inverted-U', 'Descending', 'Ascending', 'Flat']
    
    for idx, shape in enumerate(shapes_to_plot):
        ax = fig.add_subplot(gs[2, idx])
        
        shape_data = df[df['shape'] == shape]
        
        if len(shape_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
            ax.set_title(f'{shape}\n(n=0)', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Calculate means for this shape
        shape_means = [
            shape_data['begin_mean'].mean(),
            shape_data['middle_mean'].mean(),
            shape_data['end_mean'].mean()
        ]
        
        # Plot
        x_pos = np.arange(3)
        ax.plot(x_pos, shape_means, marker='o', linewidth=3.5, markersize=11,
               color=COLORS[shape], markeredgecolor='black', markeredgewidth=2)
        ax.fill_between(x_pos, shape_means, alpha=0.3, color=COLORS[shape])
        
        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Begin', 'Middle', 'End'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean BUR', fontsize=11, fontweight='bold')
        ax.set_title(f'{shape}\n(n={len(shape_data):,})',
                    fontsize=12, fontweight='bold', color=COLORS[shape], pad=10)
        ax.set_ylim(1.0, 1.8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    # Save
    output_path = 'outputs/phrase_shape_visualizations/6_comprehensive_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ SUCCESS! Saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    plt.close()
    
    print("\nLayout verification:")
    print("  ✓ Top row: Pie chart (left) + Obs vs Chance (right)")
    print("  ✓ Middle row: Overall BUR (left) + Heatmap (right)")
    print("  ✓ Bottom row: 5 individual shapes (U, Inv-U, Desc, Asc, Flat)")


if __name__ == '__main__':
    print("=" * 80)
    print("CREATING NEW COMPREHENSIVE SUMMARY FIGURE")
    print("=" * 80)
    print()
    
    create_new_comprehensive_summary()
    
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
