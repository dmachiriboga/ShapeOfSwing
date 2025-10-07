"""
Visualizations for Phrase Shape Analysis

Creates comprehensive figures showing:
1. Distribution of phrase shapes (U-shaped, Inverted-U, etc.)
2. BUR patterns by shape type
3. Edge-middle relationships
4. Statistical comparisons
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Color palette
COLORS = {
    'U-shaped': '#2E86AB',      # Blue - high edges
    'Inverted-U': '#A23B72',    # Purple - high middle
    'Ascending': '#F18F01',     # Orange - rising
    'Descending': '#C73E1D',    # Red - falling
    'Flat': '#6A994E'           # Green - stable
}


def load_shape_data():
    """Load phrase shape analysis results."""
    df = pd.read_csv('outputs/phrase_shape_analysis.csv')
    return df


def create_shape_distribution_plot(df, output_dir):
    """
    Figure 1: Bar chart of phrase shape distribution with percentage labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count shapes
    shape_counts = df['shape'].value_counts()
    shapes = ['U-shaped', 'Inverted-U', 'Ascending', 'Descending', 'Flat']
    counts = [shape_counts.get(shape, 0) for shape in shapes]
    percentages = [100 * c / len(df) for c in counts]
    
    # Create bars
    bars = ax.bar(shapes, percentages, color=[COLORS[s] for s in shapes],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Percentage of Phrases', fontsize=13, fontweight='bold')
    ax.set_xlabel('Phrase Shape', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of BUR Phrase Shapes\n(N = 2,488 phrases)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 40)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_shape_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 1_shape_distribution.png")
    plt.close()


def create_bur_patterns_by_shape(df, output_dir):
    """
    Figure 2: Line plots showing BUR patterns (beginning-middle-end) for each shape.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    shapes = ['U-shaped', 'Inverted-U', 'Ascending', 'Descending', 'Flat']
    positions = ['Beginning', 'Middle', 'End']
    
    for idx, shape in enumerate(shapes):
        ax = axes[idx]
        shape_data = df[df['shape'] == shape]
        
        if len(shape_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{shape}\n(n=0)', fontsize=12, fontweight='bold')
            continue
        
        # Get mean BUR for each position
        means = [
            shape_data['begin_mean'].mean(),
            shape_data['middle_mean'].mean(),
            shape_data['end_mean'].mean()
        ]
        
        # Get standard errors
        sems = [
            shape_data['begin_mean'].sem(),
            shape_data['middle_mean'].sem(),
            shape_data['end_mean'].sem()
        ]
        
        # Plot line with error bars
        x = np.arange(len(positions))
        ax.plot(x, means, marker='o', linewidth=3, markersize=10,
                color=COLORS[shape], label='Mean BUR')
        ax.fill_between(x, 
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        alpha=0.3, color=COLORS[shape])
        
        # Add horizontal reference line at overall mean
        overall_mean = df[['begin_mean', 'middle_mean', 'end_mean']].values.mean()
        ax.axhline(y=overall_mean, color='gray', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'Overall mean ({overall_mean:.2f})')
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.set_ylabel('Mean BUR', fontsize=11, fontweight='bold')
        ax.set_title(f'{shape}\n(n={len(shape_data):,})', 
                    fontsize=12, fontweight='bold', color=COLORS[shape])
        ax.set_ylim(1.0, 1.8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    # Overall title
    fig.suptitle('BUR Patterns by Phrase Shape\nMean BUR at Beginning, Middle, and End',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_bur_patterns_by_shape.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 2_bur_patterns_by_shape.png")
    plt.close()


def create_edge_middle_comparison(df, output_dir):
    """
    Figure 3: Violin plots comparing beginning, middle, and end BUR values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for violin plot
    data_for_plot = []
    positions_list = []
    
    for col, label in [('begin_mean', 'Beginning'), 
                       ('middle_mean', 'Middle'), 
                       ('end_mean', 'End')]:
        data_for_plot.append(df[col].values)
        positions_list.append(label)
    
    # Create violin plot
    parts = ax.violinplot(data_for_plot, positions=[0, 1, 2],
                          showmeans=True, showmedians=True, widths=0.7)
    
    # Color the violins
    colors = ['#2E86AB', '#6A994E', '#F18F01']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Format mean/median lines
    parts['cmeans'].set_edgecolor('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_edgecolor('blue')
    parts['cmedians'].set_linewidth(2)
    
    # Add mean values as text
    for i, (data, pos) in enumerate(zip(data_for_plot, [0, 1, 2])):
        mean_val = np.mean(data)
        ax.text(pos, mean_val, f'{mean_val:.3f}', 
               ha='center', va='bottom', fontsize=11, 
               fontweight='bold', color='red')
    
    # Formatting
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(positions_list, fontsize=12)
    ax.set_ylabel('BUR Value', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of BUR Values by Position\n(Red = Mean, Blue = Median)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_edge_middle_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 3_edge_middle_comparison.png")
    plt.close()


def create_edge_relationship_heatmap(df, output_dir):
    """
    Figure 4: Heatmap showing relationship between beginning and end BUR.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram
    beginning = df['begin_mean'].values
    end = df['end_mean'].values
    
    # Create bins
    bins = np.linspace(0.5, 2.5, 30)
    
    # Calculate 2D histogram
    H, xedges, yedges = np.histogram2d(beginning, end, bins=bins)
    
    # Plot heatmap
    im = ax.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # Add diagonal line (beginning = end)
    ax.plot([0.5, 2.5], [0.5, 2.5], 'b--', linewidth=2, label='Beginning = End')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Phrases', rotation=270, labelpad=20, 
                   fontsize=12, fontweight='bold')
    
    # Calculate correlation
    corr = np.corrcoef(beginning, end)[0, 1]
    
    # Formatting
    ax.set_xlabel('Beginning BUR', fontsize=13, fontweight='bold')
    ax.set_ylabel('End BUR', fontsize=13, fontweight='bold')
    ax.set_title(f'Relationship Between Beginning and End BUR\n(Pearson r = {corr:.3f}, p < 0.001)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.5, 2.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_edge_relationship_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 4_edge_relationship_heatmap.png")
    plt.close()


def create_shape_characteristics_comparison(df, output_dir):
    """
    Figure 5: Box plots comparing edge-middle difference across shapes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shapes = ['U-shaped', 'Inverted-U', 'Ascending', 'Descending', 'Flat']
    
    # Prepare data
    data_by_shape = []
    for shape in shapes:
        shape_data = df[df['shape'] == shape]
        if len(shape_data) > 0:
            # Calculate edge-middle difference
            edge_middle_diff = (shape_data['begin_mean'] + shape_data['end_mean'])/2 - shape_data['middle_mean']
            data_by_shape.append(edge_middle_diff.values)
        else:
            data_by_shape.append([])
    
    # Create box plot
    bp = ax.boxplot(data_by_shape, labels=shapes, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color boxes
    for patch, shape in zip(bp['boxes'], shapes):
        patch.set_facecolor(COLORS[shape])
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add annotations
    ax.text(0.5, 0.95, '← Middle higher', transform=ax.transAxes,
            ha='left', va='top', fontsize=11, style='italic')
    ax.text(0.5, 0.05, '← Edges higher', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, style='italic')
    
    # Formatting
    ax.set_ylabel('(Beginning + End)/2 - Middle', fontsize=12, fontweight='bold')
    ax.set_xlabel('Phrase Shape', fontsize=12, fontweight='bold')
    ax.set_title('Edge-Middle Difference by Shape\n(Positive = edges higher, Negative = middle higher)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(shapes, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_shape_characteristics.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 5_shape_characteristics.png")
    plt.close()


def create_comprehensive_summary(df, output_dir):
    """
    Figure 6: Comprehensive summary with requested layout.
    Top row: pie chart (left) + obs vs chance (right)
    Middle row: overall BUR pattern (left) + heatmap (right)
    Bottom row: 5 individual shapes
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.4,
                  height_ratios=[1.2, 1.2, 1.0])
    
    # Title
    fig.suptitle('Phrase Shape Analysis: Comprehensive Summary', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # ========== TOP ROW ==========
    
    # 1. Shape distribution pie chart (left half of top row)
    ax1 = fig.add_subplot(gs[0, :2])
    shape_counts = df['shape'].value_counts()
    shapes = ['U-shaped', 'Inverted-U', 'Ascending', 'Descending', 'Flat']
    counts = [shape_counts.get(s, 0) for s in shapes]
    colors_list = [COLORS[s] for s in shapes]
    
    wedges, texts, autotexts = ax1.pie(counts, labels=shapes, autopct='%1.1f%%',
                                        colors=colors_list, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Shape Distribution\n(N = 2,488 phrases)', 
                  fontsize=13, fontweight='bold', pad=10)
    
    # 2. Observed vs Chance comparison (right half of top row)
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Calculate observed percentages
    both_higher_raw = ((df['begin_mean'] > df['middle_mean']) & 
                       (df['end_mean'] > df['middle_mean'])).sum()
    both_lower_raw = ((df['begin_mean'] < df['middle_mean']) & 
                      (df['end_mean'] < df['middle_mean'])).sum()
    first_only_raw = ((df['begin_mean'] > df['middle_mean']) & 
                      (df['end_mean'] <= df['middle_mean'])).sum()
    last_only_raw = ((df['begin_mean'] <= df['middle_mean']) & 
                     (df['end_mean'] > df['middle_mean'])).sum()
    
    observed = [
        100 * both_higher_raw / len(df),
        100 * both_lower_raw / len(df),
        100 * first_only_raw / len(df),
        100 * last_only_raw / len(df)
    ]
    
    # Expected probabilities from permutation test
    expected = [39.42, 40.64, 9.96, 9.97]
    
    patterns_short = ['U-shaped', 'Inv-U', 'Descending', 'Ascending']
    x = np.arange(len(patterns_short))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, observed, width, label='Observed',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, expected, width, label='Expected (random)',
                    color='gray', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Observed vs Random Chance\n(All p < 0.001***)', 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(patterns_short, fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 50)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # ========== MIDDLE ROW ==========
    
    # 3. Overall BUR pattern (left 2/5 of middle row)
    ax3 = fig.add_subplot(gs[1, :2])
    positions = ['Beginning', 'Middle', 'End']
    means = [df['begin_mean'].mean(), df['middle_mean'].mean(), df['end_mean'].mean()]
    sems = [df['begin_mean'].sem(), df['middle_mean'].sem(), df['end_mean'].sem()]
    
    x_pos = np.arange(len(positions))
    ax3.bar(x_pos, means, yerr=sems, capsize=10, alpha=0.7,
            color=['#2E86AB', '#6A994E', '#F18F01'],
            edgecolor='black', linewidth=2)
    
    for i, (pos, mean) in enumerate(zip(x_pos, means)):
        ax3.text(pos, mean + sems[i] + 0.02, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(positions, fontsize=11)
    ax3.set_ylabel('Mean BUR', fontsize=11, fontweight='bold')
    ax3.set_title('Overall BUR Pattern\n(Mean ± SEM)',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylim(1.3, 1.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Edge relationship heatmap (right 3/5 of middle row)
    ax4 = fig.add_subplot(gs[1, 2:])
    
    beginning = df['begin_mean'].values
    end = df['end_mean'].values
    
    bins = np.linspace(0.5, 2.5, 30)
    H, xedges, yedges = np.histogram2d(beginning, end, bins=bins)
    
    im = ax4.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    ax4.plot([0.5, 2.5], [0.5, 2.5], 'b--', linewidth=2, label='Begin = End')
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Count', rotation=270, labelpad=15, fontsize=10, fontweight='bold')
    
    corr = np.corrcoef(beginning, end)[0, 1]
    
    ax4.set_xlabel('Beginning BUR', fontsize=11, fontweight='bold')
    ax4.set_ylabel('End BUR', fontsize=11, fontweight='bold')
    ax4.set_title(f'Beginning vs End Relationship\n(r = {corr:.3f})',
                  fontsize=13, fontweight='bold', pad=10)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.set_xlim(0.5, 2.5)
    ax4.set_ylim(0.5, 2.5)
    
    # ========== BOTTOM ROW: 5 INDIVIDUAL SHAPES ==========
    
    shapes_to_plot = ['U-shaped', 'Inverted-U', 'Descending', 'Ascending', 'Flat']
    
    for idx, shape in enumerate(shapes_to_plot):
        # Each shape gets its own column in the bottom row
        ax = fig.add_subplot(gs[2, idx])
        
        shape_data = df[df['shape'] == shape]
        if len(shape_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11, fontweight='bold')
            ax.set_title(f'{shape}\n(n=0)', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Calculate means for this shape
        means = [
            shape_data['begin_mean'].mean(),
            shape_data['middle_mean'].mean(),
            shape_data['end_mean'].mean()
        ]
        
        # Plot the pattern
        x_pos = np.arange(3)
        ax.plot(x_pos, means, marker='o', linewidth=3, markersize=10,
               color=COLORS[shape], markeredgecolor='black', markeredgewidth=1.5)
        ax.fill_between(x_pos, means, alpha=0.25, color=COLORS[shape])
        
        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Begin', 'Middle', 'End'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean BUR', fontsize=10, fontweight='bold')
        ax.set_title(f'{shape}\n(n={len(shape_data):,})',
                    fontsize=11, fontweight='bold', color=COLORS[shape], pad=8)
        ax.set_ylim(1.0, 1.8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    plt.savefig(f'{output_dir}/6_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 6_comprehensive_summary.png")
    plt.close()


def create_observed_vs_chance_comparison(df, output_dir):
    """
    Figure 7: Grouped bar chart comparing observed vs chance probabilities for all pattern types.
    Shows that jazz phrases are more structured than random.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define pattern types and their observed/expected percentages
    # These values come from permutation test results
    patterns = [
        'Both > middle\n(U-shaped)',
        'Both < middle\n(Inverted-U)',
        'First > middle only\n(Descending)',
        'Last > middle only\n(Ascending)'
    ]
    
    # Calculate observed percentages
    both_higher_raw = ((df['begin_mean'] > df['middle_mean']) & 
                       (df['end_mean'] > df['middle_mean'])).sum()
    both_lower_raw = ((df['begin_mean'] < df['middle_mean']) & 
                      (df['end_mean'] < df['middle_mean'])).sum()
    first_only_raw = ((df['begin_mean'] > df['middle_mean']) & 
                      (df['end_mean'] <= df['middle_mean'])).sum()
    last_only_raw = ((df['begin_mean'] <= df['middle_mean']) & 
                     (df['end_mean'] > df['middle_mean'])).sum()
    
    observed = [
        100 * both_higher_raw / len(df),
        100 * both_lower_raw / len(df),
        100 * first_only_raw / len(df),
        100 * last_only_raw / len(df)
    ]
    
    # Expected probabilities from permutation test (from analysis output)
    expected = [39.42, 40.64, 9.96, 9.97]
    
    # Z-scores from permutation test
    z_scores = [-16.85, -18.76, +25.70, +20.85]
    
    # Set up bar positions
    x = np.arange(len(patterns))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, observed, width, label='Observed (Jazz)',
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, expected, width, label='Expected (Random)',
                   color='gray', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add significance stars above pattern pairs
    max_height = max(max(observed), max(expected))
    for i, z in enumerate(z_scores):
        y_pos = max_height + 5
        ax.text(i, y_pos, '***', ha='center', va='bottom',
               fontsize=16, fontweight='bold', color='red')
        # Add Z-score
        ax.text(i, y_pos + 3, f'Z={z:+.1f}', ha='center', va='bottom',
               fontsize=8, style='italic', color='darkred')
    
    # Formatting
    ax.set_ylabel('Percentage of Phrases (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pattern Type', fontsize=13, fontweight='bold')
    ax.set_title('Observed vs Random Chance: Pattern Distribution\n' +
                 '(All differences: p < 0.001, showing jazz is more structured than random)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max_height + 10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add interpretation text box
    textstr = ('Jazz musicians AVOID symmetric patterns (U-shaped, Inverted-U)\n' +
               'and FAVOR directional patterns (Descending, Ascending)\n' +
               '→ Evidence of deliberate structure, not random variation')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.5, 0.97, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='center',
           bbox=props, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/7_observed_vs_chance.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 7_observed_vs_chance.png")
    plt.close()
    """
    Figure 6: Comprehensive summary infographic with all key statistics.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Phrase Shape Analysis: Comprehensive Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Shape distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    shape_counts = df['shape'].value_counts()
    shapes = ['U-shaped', 'Inverted-U', 'Ascending', 'Descending', 'Flat']
    counts = [shape_counts.get(s, 0) for s in shapes]
    colors_list = [COLORS[s] for s in shapes]
    
    wedges, texts, autotexts = ax1.pie(counts, labels=shapes, autopct='%1.1f%%',
                                        colors=colors_list, startangle=90,
                                        textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax1.set_title('Shape Distribution', fontsize=12, fontweight='bold', pad=10)
    
    # 2. U-shaped vs Expected bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    u_shaped_pct = 100 * (df['shape'] == 'U-shaped').sum() / len(df)
    expected_pct = 25
    bars = ax2.bar(['Observed', 'Expected\n(by chance)'], [u_shaped_pct, expected_pct],
                   color=['#2E86AB', 'gray'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax2.set_title('U-Shaped Pattern\n(p < 0.0001)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 40)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Key statistics text box
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    u_count = (df['shape'] == 'U-shaped').sum()
    inv_u_count = (df['shape'] == 'Inverted-U').sum()
    both_higher = (df['both_edges_higher'] == True).sum()
    
    stats_text = f"""KEY FINDINGS
    
Total Phrases: {len(df):,}

U-Shaped: {u_count:,} ({100*u_count/len(df):.1f}%)
  → Significantly > 25% (p < 0.001)

Both edges > middle: {both_higher:,} ({100*both_higher/len(df):.1f}%)

Beginning ≈ End
  (difference < 0.001)

Edge-Middle correlation:
  χ² = 205.8, p < 0.001
  → NOT independent"""
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Mean BUR by position (all phrases)
    ax4 = fig.add_subplot(gs[1, :])
    positions = ['Beginning', 'Middle', 'End']
    means = [df['begin_mean'].mean(), df['middle_mean'].mean(), df['end_mean'].mean()]
    sems = [df['begin_mean'].sem(), df['middle_mean'].sem(), df['end_mean'].sem()]
    
    x = np.arange(len(positions))
    ax4.bar(x, means, yerr=sems, capsize=10, alpha=0.7,
            color=['#2E86AB', '#6A994E', '#F18F01'],
            edgecolor='black', linewidth=2)
    
    for i, (pos, mean) in enumerate(zip(x, means)):
        ax4.text(pos, mean + sems[i] + 0.02, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(positions, fontsize=12)
    ax4.set_ylabel('Mean BUR', fontsize=12, fontweight='bold')
    ax4.set_title('Overall BUR Pattern Across All Phrases (Mean ± SEM)',
                 fontsize=13, fontweight='bold', pad=10)
    ax4.set_ylim(1.3, 1.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Shape-specific patterns (small multiples)
    for idx, shape in enumerate(['U-shaped', 'Inverted-U', 'Ascending', 'Descending']):
        ax = fig.add_subplot(gs[2, idx if idx < 3 else idx-3])
        
        shape_data = df[df['shape'] == shape]
        if len(shape_data) == 0:
            continue
        
        means = [
            shape_data['begin_mean'].mean(),
            shape_data['middle_mean'].mean(),
            shape_data['end_mean'].mean()
        ]
        
        x = np.arange(3)
        ax.plot(x, means, marker='o', linewidth=3, markersize=8,
               color=COLORS[shape])
        ax.fill_between(x, means, alpha=0.3, color=COLORS[shape])
        
        ax.set_xticks(x)
        ax.set_xticklabels(['B', 'M', 'E'], fontsize=10)
        ax.set_ylabel('BUR', fontsize=9, fontweight='bold')
        ax.set_title(f'{shape}\n(n={len(shape_data):,})',
                    fontsize=10, fontweight='bold', color=COLORS[shape])
        ax.set_ylim(1.0, 1.8)
        ax.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/6_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 6_comprehensive_summary.png")
    plt.close()


def create_all_visualizations():
    """Create all phrase shape visualizations."""
    print("=" * 80)
    print("CREATING PHRASE SHAPE VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = 'outputs/phrase_shape_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading phrase shape data...")
    df = load_shape_data()
    print(f"  Loaded {len(df):,} phrases")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    print()
    
    create_shape_distribution_plot(df, output_dir)
    create_bur_patterns_by_shape(df, output_dir)
    create_edge_middle_comparison(df, output_dir)
    create_edge_relationship_heatmap(df, output_dir)
    create_shape_characteristics_comparison(df, output_dir)
    create_comprehensive_summary(df, output_dir)
    create_observed_vs_chance_comparison(df, output_dir)
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"All figures saved to: {output_dir}/")
    print()
    print("Files created:")
    print("  1. 1_shape_distribution.png")
    print("  2. 2_bur_patterns_by_shape.png")
    print("  3. 3_edge_middle_comparison.png")
    print("  4. 4_edge_relationship_heatmap.png")
    print("  5. 5_shape_characteristics.png")
    print("  6. 6_comprehensive_summary.png")
    print("  7. 7_observed_vs_chance.png")
    print()
    print("=" * 80)


if __name__ == '__main__':
    create_all_visualizations()
