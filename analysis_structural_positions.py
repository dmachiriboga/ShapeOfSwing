"""
Structural Position BUR Analysis (32-Bar Cycles)

Analyzes BUR values at specific structural positions:
1. Multiples of bar 31, beat 4 (bar 31, 63, 95... beat 4)
2. Multiples of bar 32, beat 1 (bar 32, 64, 96... beat 1) 
3. All other positions

Questions to address:
- Do these 32-bar cycle structural positions have different BUR characteristics?
- Statistical significance testing between categories
- Effect sizes for meaningful differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_formbur_data():
    """Load the cleaned formbur.csv data and exclude blues tunes from 32-bar analysis."""
    print("Loading formbur_clean.csv data...")
    df = pd.read_csv('data/formbur_clean.csv', delimiter=';')
    print(f"Loaded {len(df)} valid BUR measurements (NaN values removed)")
    
    # Identify and exclude blues tunes (they follow 12-bar form, not 32-bar AABA)
    blues_keywords = ['blues', 'Blues']
    blues_songs = []
    for song in df['id'].unique():
        song_title = song.replace('_FINAL.sv', '').split('_', 1)[1] if '_' in song else song
        if any(keyword in song_title for keyword in blues_keywords):
            blues_songs.append(song)
    
    print(f"Excluding {len(blues_songs)} blues tunes from 32-bar analysis (12-bar form)")
    print("Blues tunes excluded:", [s.split('_')[1].replace('FINAL.sv', '') for s in blues_songs[:5]], "..." if len(blues_songs) > 5 else "")
    
    # Filter out blues tunes for 32-bar structural analysis
    df_filtered = df[~df['id'].isin(blues_songs)]
    print(f"Remaining for 32-bar analysis: {len(df_filtered)} measurements from non-blues tunes")
    
    return df_filtered

def categorize_structural_positions(df):
    """
    Categorize each row based on structural position:
    1. 'bar_3_beat_4': Multiples of bar 3, beat 4 (bars 3, 7, 11, 15...)
    2. 'bar_4_beat_1': Multiples of bar 4, beat 1 (bars 4, 8, 12, 16...)
    3. 'other': All other positions
    """
    
    def get_category(row):
        bar = row['bar']
        beat = row['beat']
        
        # Category 1: Multiples of bar 31, beat 4
        # Pattern: bar 31, 63, 95... (i.e., bar = 32n - 1 for n >= 1)
        if beat == 4 and bar >= 31 and (bar + 1) % 32 == 0:
            return 'bar_31_beat_4'
        
        # Category 2: Multiples of bar 32, beat 1  
        # Pattern: bar 32, 64, 96... (i.e., bar = 32n for n >= 1)
        elif beat == 1 and bar >= 32 and bar % 32 == 0:
            return 'bar_32_beat_1'
        
        # Category 3: Everything else
        else:
            return 'other'
    
    df['structural_category'] = df.apply(get_category, axis=1)
    return df

def analyze_structural_positions():
    """Main analysis function."""
    
    # Load data
    df = load_formbur_data()
    
    # Categorize positions
    df = categorize_structural_positions(df)
    
    # Summary statistics
    print("\n" + "="*80)
    print("STRUCTURAL POSITION ANALYSIS")
    print("="*80)
    
    category_counts = df['structural_category'].value_counts()
    print(f"\nData Distribution:")
    print(f"Bar 31,63,95... Beat 4: {category_counts.get('bar_31_beat_4', 0):,} measurements")
    print(f"Bar 32,64,96... Beat 1: {category_counts.get('bar_32_beat_1', 0):,} measurements") 
    print(f"All other positions:   {category_counts.get('other', 0):,} measurements")
    print(f"Total:                {len(df):,} measurements")
    
    # Calculate mean BUR for each category
    category_stats = df.groupby('structural_category')['swing_ratios'].agg([
        'count', 'mean', 'std', 'median'
    ]).round(4)
    
    print(f"\nBUR Statistics by Category:")
    print(category_stats)
    
    # Extract data for statistical testing
    bar_31_beat_4_data = df[df['structural_category'] == 'bar_31_beat_4']['swing_ratios'].dropna()
    bar_32_beat_1_data = df[df['structural_category'] == 'bar_32_beat_1']['swing_ratios'].dropna()
    other_data = df[df['structural_category'] == 'other']['swing_ratios'].dropna()
    
    print(f"\nActual data sizes for statistical testing:")
    print(f"Bar 31,63,95... Beat 4: {len(bar_31_beat_4_data)} valid measurements")
    print(f"Bar 32,64,96... Beat 1: {len(bar_32_beat_1_data)} valid measurements") 
    print(f"All other positions:   {len(other_data)} valid measurements")
    
    # Statistical tests
    print(f"\n" + "-"*60)
    print("STATISTICAL TESTS")
    print("-"*60)
    
    # One-way ANOVA (assumes normal distribution)
    if len(bar_31_beat_4_data) > 0 and len(bar_32_beat_1_data) > 0:
        f_stat, p_anova = f_oneway(bar_31_beat_4_data, bar_32_beat_1_data, other_data)
        print(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_anova:.6f}")
        
        # Kruskal-Wallis (non-parametric alternative)
        h_stat, p_kruskal = kruskal(bar_31_beat_4_data, bar_32_beat_1_data, other_data)
        print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kruskal:.6f}")
        
        # Pairwise comparisons
        print(f"\nPairwise Comparisons (t-tests):")
        
        # Bar 31 Beat 4 vs Bar 32 Beat 1
        if len(bar_31_beat_4_data) > 0 and len(bar_32_beat_1_data) > 0:
            t1, p1 = stats.ttest_ind(bar_31_beat_4_data, bar_32_beat_1_data)
            d1 = cohens_d(bar_31_beat_4_data, bar_32_beat_1_data)
            print(f"  Bar31-Beat4 vs Bar32-Beat1: t = {t1:.4f}, p = {p1:.6f}, Cohen's d = {d1:.4f}")
        
        # Bar 31 Beat 4 vs Other
        if len(bar_31_beat_4_data) > 0:
            t2, p2 = stats.ttest_ind(bar_31_beat_4_data, other_data)
            d2 = cohens_d(bar_31_beat_4_data, other_data)
            print(f"  Bar31-Beat4 vs Other:       t = {t2:.4f}, p = {p2:.6f}, Cohen's d = {d2:.4f}")
        
        # Bar 32 Beat 1 vs Other
        if len(bar_32_beat_1_data) > 0:
            t3, p3 = stats.ttest_ind(bar_32_beat_1_data, other_data)
            d3 = cohens_d(bar_32_beat_1_data, other_data)
            print(f"  Bar32-Beat1 vs Other:       t = {t3:.4f}, p = {p3:.6f}, Cohen's d = {d3:.4f}")
    
    # Visualizations
    print(f"\nCreating visualizations...")
    
    # Filter out categories with no data for plotting
    plot_data = df[df['structural_category'].isin(['bar_31_beat_4', 'bar_32_beat_1', 'other'])]
    
    if len(plot_data) > 0:
        # Figure 1: Box plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Boxplot
        if len(plot_data['structural_category'].unique()) > 1:
            sns.boxplot(data=plot_data, x='structural_category', y='swing_ratios', ax=axes[0,0])
            axes[0,0].set_title('BUR by Structural Position (Boxplot)')
            axes[0,0].set_xlabel('Structural Category')
            axes[0,0].set_ylabel('BUR Value')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Violin plot
            sns.violinplot(data=plot_data, x='structural_category', y='swing_ratios', ax=axes[0,1])
            axes[0,1].set_title('BUR by Structural Position (Violin Plot)')
            axes[0,1].set_xlabel('Structural Category')
            axes[0,1].set_ylabel('BUR Value')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Histogram comparison
        categories = plot_data['structural_category'].unique()
        colors = ['red', 'blue', 'gray']
        
        for i, (cat, color) in enumerate(zip(categories, colors)):
            data = plot_data[plot_data['structural_category'] == cat]['swing_ratios']
            axes[1,0].hist(data, bins=50, alpha=0.6, label=cat, color=color, density=True)
        
        axes[1,0].set_title('BUR Distribution by Category')
        axes[1,0].set_xlabel('BUR Value')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # Bar plot of means with error bars
        means = category_stats['mean']
        stds = category_stats['std']
        
        x_pos = np.arange(len(means))
        bars = axes[1,1].bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                            color=['red', 'blue', 'gray'])
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(means.index, rotation=45)
        axes[1,1].set_title('Mean BUR by Category (±1 SD)')
        axes[1,1].set_ylabel('Mean BUR Value')
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            axes[1,1].text(i, mean_val + stds.iloc[i]/2, f'{mean_val:.3f}', 
                          ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/structural_position_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Sample examples showing the specific bars/beats
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Show first few examples of each category
        sample_size = min(20, len(plot_data) // 3)
        
        for cat, color in zip(['bar_3_beat_4', 'bar_4_beat_1', 'other'], ['red', 'blue', 'gray']):
            cat_data = plot_data[plot_data['structural_category'] == cat].head(sample_size)
            if len(cat_data) > 0:
                ax.scatter(cat_data['bar'], cat_data['swing_ratios'], 
                          alpha=0.6, label=f'{cat} (n={category_counts.get(cat, 0)})', 
                          color=color, s=30)
        
        ax.set_xlabel('Bar Number')
        ax.set_ylabel('BUR Value')
        ax.set_title('BUR Values by Bar Number (Sample Data)\nShowing Structural Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/structural_position_scatter.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if len(bar_31_beat_4_data) > 0 and len(bar_32_beat_1_data) > 0:
        print(f"✓ Analysis completed successfully")
        print(f"✓ Found structural position effects: p = {p_anova:.6f} (ANOVA)")
        
        # Interpret significance
        if p_anova < 0.001:
            print(f"✓ HIGHLY SIGNIFICANT differences between structural positions")
        elif p_anova < 0.05:
            print(f"✓ SIGNIFICANT differences between structural positions") 
        else:
            print(f"✗ NO significant differences between structural positions")
            
        print(f"✓ Visualizations saved to outputs/")
        
    else:
        print(f"⚠️  Insufficient data in one or more categories")
        if len(bar_31_beat_4_data) == 0:
            print(f"   - No data found for Bar 31,63,95... Beat 4")
        if len(bar_32_beat_1_data) == 0:
            print(f"   - No data found for Bar 32,64,96... Beat 1")
    
    return df, category_stats

def cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return 0.0
    
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

if __name__ == '__main__':
    # Create outputs directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    df, stats = analyze_structural_positions()