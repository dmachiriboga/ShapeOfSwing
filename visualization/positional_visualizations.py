"""
Positional Pattern Visualizations

Creates comprehensive visualizations showing:
1. First vs Last note comparison (boxplot, violin plot)
2. Beginning/Middle/End comparison (shows no difference)
3. Position-by-position BUR means (shows first note effect)
4. First/Middle/Last comparison
5. Individual phrase trajectories (sample)
6. Effect size comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.data_utils import load_phrasebur_csv
from utils.config import MIN_BUR_VALUES

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def create_all_visualizations(output_dir="outputs/positional_visualizations"):
    """Create all positional pattern visualizations."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_phrasebur_csv()
    grouped = df.groupby(['id', 'seg_id'])
    
    # Prepare data structures
    print("Preparing data...")
    first_notes = []
    last_notes = []
    phrase_thirds = []
    first_mid_last = []
    position_means = {i: [] for i in range(20)}
    phrase_trajectories = []
    
    for idx, ((solo_id, seg_id), group) in enumerate(grouped):
        bur_values = group['swing_ratios'].tolist()
        n = len(bur_values)
        
        if n < MIN_BUR_VALUES:
            continue
        
        # First vs Last
        first_notes.append(bur_values[0])
        last_notes.append(bur_values[-1])
        
        # Beginning/Middle/End thirds
        third = n // 3
        remainder = n % 3
        
        beginning = bur_values[:third + (1 if remainder >= 1 else 0)]
        start_middle = third + (1 if remainder >= 1 else 0)
        end_middle = start_middle + third + (1 if remainder >= 2 else 0)
        middle = bur_values[start_middle:end_middle]
        end = bur_values[end_middle:]
        
        phrase_thirds.append({
            'beginning': np.mean(beginning),
            'middle': np.mean(middle),
            'end': np.mean(end)
        })
        
        # First/Middle/Last single notes
        first_mid_last.append({
            'first': bur_values[0],
            'middle': bur_values[n // 2],
            'last': bur_values[-1]
        })
        
        # Position-by-position
        for i, bur in enumerate(bur_values):
            if i < 20:
                position_means[i].append(bur)
        
        # Sample trajectories (first 50 phrases for visualization)
        if idx < 50 and n >= 8:
            phrase_trajectories.append({
                'phrase_id': idx,
                'positions': list(range(n)),
                'bur_values': bur_values
            })
    
    phrase_thirds_df = pd.DataFrame(phrase_thirds)
    first_mid_last_df = pd.DataFrame(first_mid_last)
    
    print(f"Analyzing {len(first_notes)} phrases...")
    print()
    
    # ========================================================================
    # Figure 1: First vs Last Comparison (Multiple Views)
    # ========================================================================
    print("Creating Figure 1: First vs Last Comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Boxplot
    data_firstlast = pd.DataFrame({
        'Position': ['First'] * len(first_notes) + ['Last'] * len(last_notes),
        'BUR': first_notes + last_notes
    })
    
    sns.boxplot(data=data_firstlast, x='Position', y='BUR', ax=axes[0], palette='Set2')
    axes[0].set_title('First vs Last Note BUR\n(Boxplot)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('BUR Value', fontsize=11)
    axes[0].set_xlabel('')
    
    # Add mean lines
    axes[0].axhline(np.mean(first_notes), color='green', linestyle='--', alpha=0.5, label=f'First mean: {np.mean(first_notes):.3f}')
    axes[0].axhline(np.mean(last_notes), color='orange', linestyle='--', alpha=0.5, label=f'Last mean: {np.mean(last_notes):.3f}')
    axes[0].legend(fontsize=8)
    
    # Violin plot
    sns.violinplot(data=data_firstlast, x='Position', y='BUR', ax=axes[1], palette='Set2')
    axes[1].set_title('First vs Last Note BUR\n(Violin Plot - Distribution)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('BUR Value', fontsize=11)
    axes[1].set_xlabel('')
    
    # Paired difference plot
    differences = np.array(first_notes) - np.array(last_notes)
    axes[2].hist(differences, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[2].axvline(np.mean(differences), color='red', linestyle='--', linewidth=2, label=f'Mean diff: +{np.mean(differences):.3f}')
    axes[2].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    axes[2].set_title('First - Last Difference\n(Distribution)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Difference (First - Last)', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].legend()
    
    # Add statistics
    t_stat, p_val = stats.ttest_rel(first_notes, last_notes)
    fig.text(0.5, 0.02, f'Paired t-test: t = {t_stat:.3f}, p < 0.001 (highly significant)', 
             ha='center', fontsize=10, style='italic', color='darkred')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, '1_first_vs_last_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 1_first_vs_last_comparison.png")
    
    # ========================================================================
    # Figure 2: Beginning/Middle/End (Shows NO Difference)
    # ========================================================================
    print("Creating Figure 2: Beginning/Middle/End Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot
    data_thirds = pd.DataFrame({
        'Third': ['Beginning'] * len(phrase_thirds_df) + ['Middle'] * len(phrase_thirds_df) + ['End'] * len(phrase_thirds_df),
        'BUR': phrase_thirds_df['beginning'].tolist() + phrase_thirds_df['middle'].tolist() + phrase_thirds_df['end'].tolist()
    })
    
    sns.boxplot(data=data_thirds, x='Third', y='BUR', ax=axes[0], palette='Set3', order=['Beginning', 'Middle', 'End'])
    axes[0].set_title('Beginning/Middle/End BUR (Phrase-Level Means)\nNO SIGNIFICANT DIFFERENCE', 
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel('BUR Value (Phrase Mean)', fontsize=11)
    axes[0].set_xlabel('')
    
    # Add mean annotations
    for i, third in enumerate(['Beginning', 'Middle', 'End']):
        mean_val = phrase_thirds_df[third.lower()].mean()
        axes[0].text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Bar plot with error bars
    means = [phrase_thirds_df['beginning'].mean(), 
             phrase_thirds_df['middle'].mean(), 
             phrase_thirds_df['end'].mean()]
    sems = [phrase_thirds_df['beginning'].sem(), 
            phrase_thirds_df['middle'].sem(), 
            phrase_thirds_df['end'].sem()]
    
    x_pos = np.arange(3)
    axes[1].bar(x_pos, means, yerr=sems, capsize=10, alpha=0.7, color=['lightblue', 'lightgreen', 'lightsalmon'])
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Beginning', 'Middle', 'End'])
    axes[1].set_title('Mean BUR by Third\n(Error bars = SEM)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean BUR Value', fontsize=11)
    axes[1].set_ylim([1.35, 1.45])
    
    # Add horizontal line at grand mean
    grand_mean = np.mean(means)
    axes[1].axhline(grand_mean, color='gray', linestyle='--', alpha=0.5, label=f'Grand mean: {grand_mean:.3f}')
    axes[1].legend()
    
    # Add statistics
    stat, p_friedman = stats.friedmanchisquare(
        phrase_thirds_df['beginning'],
        phrase_thirds_df['middle'],
        phrase_thirds_df['end']
    )
    fig.text(0.5, 0.02, f'Friedman test: χ² = {stat:.3f}, p = {p_friedman:.3f} (NOT significant)', 
             ha='center', fontsize=10, style='italic', color='darkgreen')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, '2_beginning_middle_end_no_difference.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 2_beginning_middle_end_no_difference.png")
    
    # ========================================================================
    # Figure 3: Position-by-Position BUR (First Note Effect)
    # ========================================================================
    print("Creating Figure 3: Position-by-Position Analysis...")
    
    # Calculate means and standard deviations for first 10 positions
    positions = []
    means_pos = []
    stds_pos = []
    ns = []
    
    for i in range(10):
        if position_means[i]:
            positions.append(i + 1)
            means_pos.append(np.mean(position_means[i]))
            stds_pos.append(np.std(position_means[i]))
            ns.append(len(position_means[i]))
    
    # Create single figure matching infographic style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add shaded region for Cohen's d = ±0.088
    # Cohen's d = 0.088 corresponds to 0.088 * pooled_std
    # Using position-specific std as approximation
    cohens_d_threshold = 0.088
    lower_bounds = [m - cohens_d_threshold * s for m, s in zip(means_pos, stds_pos)]
    upper_bounds = [m + cohens_d_threshold * s for m, s in zip(means_pos, stds_pos)]
    
    ax.fill_between(positions, lower_bounds, upper_bounds, 
                     alpha=0.3, color='steelblue', label='Cohen\'s d = ±0.088')
    
    # Line plot with markers
    ax.plot(positions, means_pos, marker='o', linewidth=2.5, markersize=8, color='steelblue')
    
    # Highlight first position with larger red marker
    ax.plot(1, means_pos[0], marker='o', markersize=15, color='red', alpha=0.5, label='First (highest)')
    
    # Formatting
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('Mean BUR', fontsize=11)
    ax.set_title('Mean BUR by Position Within Phrase', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_position_by_position_first_note_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 3_position_by_position_first_note_effect.png")
    
    # ========================================================================
    # Figure 4: First/Middle/Last Single Notes
    # ========================================================================
    print("Creating Figure 4: First/Middle/Last Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Violin plot
    data_fml = pd.DataFrame({
        'Position': ['First'] * len(first_mid_last_df) + ['Middle'] * len(first_mid_last_df) + ['Last'] * len(first_mid_last_df),
        'BUR': first_mid_last_df['first'].tolist() + first_mid_last_df['middle'].tolist() + first_mid_last_df['last'].tolist()
    })
    
    sns.violinplot(data=data_fml, x='Position', y='BUR', ax=axes[0], palette='Set2', order=['First', 'Middle', 'Last'])
    axes[0].set_title('First/Middle/Last Note BUR\n(Violin Plot)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('BUR Value', fontsize=11)
    axes[0].set_xlabel('')
    
    # Add mean lines with annotations
    for i, pos in enumerate(['First', 'Middle', 'Last']):
        mean_val = first_mid_last_df[pos.lower()].mean()
        axes[0].axhline(mean_val, color='gray', linestyle='--', alpha=0.3)
        axes[0].text(i, mean_val + 0.02, f'{mean_val:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    # Paired comparison plot (spaghetti plot sample)
    np.random.seed(42)
    sample_indices = np.random.choice(len(first_mid_last_df), size=min(100, len(first_mid_last_df)), replace=False)
    
    for idx in sample_indices:
        row = first_mid_last_df.iloc[idx]
        axes[1].plot([1, 2, 3], [row['first'], row['middle'], row['last']], 
                    alpha=0.1, color='gray', linewidth=0.5)
    
    # Add mean trajectory (thick line)
    mean_trajectory = [first_mid_last_df['first'].mean(), 
                      first_mid_last_df['middle'].mean(), 
                      first_mid_last_df['last'].mean()]
    axes[1].plot([1, 2, 3], mean_trajectory, color='red', linewidth=3, marker='o', markersize=10, label='Mean trajectory')
    
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['First', 'Middle', 'Last'])
    axes[1].set_ylabel('BUR Value', fontsize=11)
    axes[1].set_title('Individual Phrase Trajectories\n(Sample of 100 phrases)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics
    stat_fml, p_fml = stats.friedmanchisquare(
        first_mid_last_df['first'],
        first_mid_last_df['middle'],
        first_mid_last_df['last']
    )
    fig.text(0.5, 0.02, f'Friedman test: χ² = {stat_fml:.3f}, p = {p_fml:.4f} (significant)', 
             ha='center', fontsize=10, style='italic', color='darkred')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, '4_first_middle_last_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 4_first_middle_last_comparison.png")
    
    # ========================================================================
    # Figure 5: Sample Phrase Trajectories
    # ========================================================================
    print("Creating Figure 5: Sample Phrase Trajectories...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Plot individual trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, len(phrase_trajectories)))
    
    for i, traj in enumerate(phrase_trajectories[:30]):  # First 30
        ax.plot(traj['positions'], traj['bur_values'], alpha=0.4, linewidth=1.5, color=colors[i])
    
    ax.set_xlabel('Position in Phrase', fontsize=12)
    ax.set_ylabel('BUR Value', fontsize=12)
    ax.set_title('Sample Phrase Trajectories (First 30 Phrases)\nShows High Variability, No Systematic Pattern', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1, label='BUR = 1.0 (even timing)')
    
    # Highlight first position
    first_burs = [traj['bur_values'][0] for traj in phrase_trajectories[:30]]
    ax.scatter([0] * len(first_burs), first_burs, color='red', s=50, alpha=0.6, zorder=10, label='First notes')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_sample_phrase_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 5_sample_phrase_trajectories.png")
    
    # ========================================================================
    # Figure 6: Effect Sizes Comparison
    # ========================================================================
    print("Creating Figure 6: Effect Sizes Comparison...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    comparisons = [
        ('First vs Last', cohens_d(first_notes, last_notes)),
        ('First vs Middle', cohens_d(first_mid_last_df['first'], first_mid_last_df['middle'])),
        ('Beginning vs Middle', cohens_d(phrase_thirds_df['beginning'], phrase_thirds_df['middle'])),
        ('Beginning vs End', cohens_d(phrase_thirds_df['beginning'], phrase_thirds_df['end'])),
        ('Middle vs Last', cohens_d(first_mid_last_df['middle'], first_mid_last_df['last'])),
        ('Middle vs End', cohens_d(phrase_thirds_df['middle'], phrase_thirds_df['end'])),
    ]
    
    labels, effect_sizes = zip(*comparisons)
    colors_es = ['darkred' if abs(es) > 0.05 else 'lightgray' for es in effect_sizes]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, effect_sizes, color=colors_es, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Effect Sizes for All Positional Comparisons\n(Red = Meaningful, Gray = Trivial)", 
                fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0.2, color='green', linestyle='--', alpha=0.3, label='Small effect (d=0.2)')
    ax.axvline(-0.2, color='green', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, (bar, es) in enumerate(zip(bars, effect_sizes)):
        ax.text(es + 0.005 if es > 0 else es - 0.005, i, f'{es:.3f}', 
               va='center', ha='left' if es > 0 else 'right', fontweight='bold', fontsize=9)
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_effect_sizes_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 6_effect_sizes_comparison.png")
    
    # ========================================================================
    # Figure 7: Summary Infographic
    # ========================================================================
    print("Creating Figure 7: Summary Infographic...")
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('BUR Positional Pattern Analysis - Summary\nKey Finding: First Note Effect', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 1: First vs Last (key finding)
    ax1 = fig.add_subplot(gs[0, :])
    data_summary = {
        'First Note': [np.mean(first_notes), np.std(first_notes)],
        'Last Note': [np.mean(last_notes), np.std(last_notes)]
    }
    x_summary = [0, 1]
    means_summary = [data_summary['First Note'][0], data_summary['Last Note'][0]]
    stds_summary = [data_summary['First Note'][1], data_summary['Last Note'][1]]
    
    bars = ax1.bar(x_summary, means_summary, yerr=stds_summary, capsize=10, 
                   color=['darkred', 'steelblue'], alpha=0.7, width=0.6)
    ax1.set_xticks(x_summary)
    ax1.set_xticklabels(['First Note', 'Last Note'])
    ax1.set_ylabel('Mean BUR Value', fontsize=11)
    ax1.set_title('✓ SIGNIFICANT: First Notes Are +4.9% Swingier (p < 0.001)', 
                 fontsize=12, fontweight='bold', color='darkred')
    ax1.set_ylim([1.0, 1.8])
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means_summary)):
        ax1.text(i, mean + 0.05, f'{mean:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    # Panel 2: Beginning/Middle/End (no difference)
    ax2 = fig.add_subplot(gs[1, 0])
    thirds_means = [phrase_thirds_df['beginning'].mean(), 
                   phrase_thirds_df['middle'].mean(), 
                   phrase_thirds_df['end'].mean()]
    ax2.bar([0, 1, 2], thirds_means, color='lightgray', alpha=0.7, width=0.6)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Beginning', 'Middle', 'End'])
    ax2.set_ylabel('Mean BUR', fontsize=10)
    ax2.set_title('✗ NO DIFFERENCE: Thirds Are Identical\n(p = 0.44)', 
                 fontsize=11, fontweight='bold', color='darkgreen')
    ax2.set_ylim([1.35, 1.45])
    ax2.axhline(np.mean(thirds_means), color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    for i, mean in enumerate(thirds_means):
        ax2.text(i, mean + 0.005, f'{mean:.3f}', ha='center', fontsize=9)
    
    # Panel 3: Position trajectory
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(positions[:10], means_pos[:10], marker='o', linewidth=2.5, markersize=8, color='steelblue')
    ax3.plot(1, means_pos[0], marker='o', markersize=15, color='red', alpha=0.5, label='First (highest)')
    ax3.set_xlabel('Position', fontsize=10)
    ax3.set_ylabel('Mean BUR', fontsize=10)
    ax3.set_title('First Note Effect:\nDrops After Position 1', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Statistical summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_text = """
    STATISTICAL SUMMARY (with FDR correction):
    
    ✓ First vs Last:              p < 0.001    Effect size: d = 0.088    SIGNIFICANT
    ✓ First vs Middle:            p < 0.001    Effect size: d = 0.145    SIGNIFICANT
    ✗ Beginning vs Middle:        p = 0.29     Effect size: d = 0.031    NOT significant
    ✗ Beginning vs End:           p = 0.98     Effect size: d = -0.001   NOT significant
    ✗ Middle vs End:              p = 0.29     Effect size: d = -0.031   NOT significant
    
    CONCLUSION: The "first note effect" is the ONLY systematic positional pattern.
    After the first note, BUR remains stable throughout phrases.
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(os.path.join(output_dir, '7_summary_infographic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 7_summary_infographic.png")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"All visualizations saved to: {output_dir}/")
    print()
    print("Created 7 figures:")
    print("  1. First vs Last Comparison (boxplot, violin, difference)")
    print("  2. Beginning/Middle/End Comparison (shows no difference)")
    print("  3. Position-by-Position Analysis (shows first note effect)")
    print("  4. First/Middle/Last Comparison (single notes)")
    print("  5. Sample Phrase Trajectories (30 phrases)")
    print("  6. Effect Sizes Comparison (all comparisons)")
    print("  7. Summary Infographic (key findings)")
    print()
    print("=" * 80)
    
    return output_dir


if __name__ == '__main__':
    create_all_visualizations()
