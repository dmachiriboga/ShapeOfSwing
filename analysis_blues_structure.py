"""
Blues Structure BUR Analysis (12-Bar Form)

Analyzes BUR values at specific structural positions in 12-bar blues:
1. End of blues form: Bar 11, beat 4 and Bar 12, beat 1-4
2. Start of new blues chorus: Bar 1, beat 1 (and following beats)
3. Turnaround positions: Bar 11-12 (traditional blues turnaround)
4. All other positions

12-bar blues structure:
- Bars 1-4: I chord (tonic)
- Bars 5-6: IV chord (subdominant) 
- Bars 7-8: I chord (return to tonic)
- Bars 9-10: V chord (dominant)
- Bars 11-12: I chord (turnaround back to beginning)
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

def load_blues_data():
    """Load formbur data and filter for blues tunes only."""
    print("Loading formbur_clean.csv data...")
    df = pd.read_csv('data/formbur_clean.csv', delimiter=';')
    print(f"Loaded {len(df)} valid BUR measurements (NaN values removed)")
    
    # Identify blues tunes
    blues_keywords = ['blues', 'Blues']
    blues_songs = []
    for song in df['id'].unique():
        song_title = song.replace('_FINAL.sv', '').split('_', 1)[1] if '_' in song else song
        if any(keyword in song_title for keyword in blues_keywords):
            blues_songs.append(song)
    
    print(f"Found {len(blues_songs)} blues tunes:")
    for song in blues_songs:
        artist_title = song.replace('_FINAL.sv', '').replace('_', ' - ', 1)
        print(f"  • {artist_title}")
    
    # Filter for blues tunes only
    df_blues = df[df['id'].isin(blues_songs)]
    print(f"\nBlues data: {len(df_blues)} measurements from {len(blues_songs)} blues tunes")
    
    return df_blues, blues_songs

def categorize_blues_positions(df):
    """
    Categorize each row based on 12-bar blues structure:
    1. 'chorus_start': Bar 1, beats 1-2 (beginning of new chorus)
    2. 'turnaround': Bars 11-12, all beats (traditional blues turnaround)
    3. 'form_end': Bar 12, beats 3-4 (end of chorus, leading to next)
    4. 'other': All other positions in the blues form
    """
    
    def get_blues_category(row):
        # Use modulo 12 to get position within 12-bar cycle
        bar_in_cycle = (row['bar']) % 12
        beat = row['beat']
        
        # Adjust for 0-indexing: bar 0 = bar 12 of previous cycle
        if bar_in_cycle == 0:
            bar_in_cycle = 12
        
        # Category 1: Start of new chorus (bar 1, beats 1-2)
        if bar_in_cycle == 1 and beat in [1, 2]:
            return 'chorus_start'
        
        # Category 2: Turnaround section (bars 11-12, all beats)
        elif bar_in_cycle in [11, 12]:
            return 'turnaround'
        
        # Category 3: End of form (bar 12, beats 3-4) - subset of turnaround
        elif bar_in_cycle == 12 and beat in [3, 4]:
            return 'form_end'
        
        # Category 4: Everything else in the blues form
        else:
            return 'other'
    
    df['blues_category'] = df.apply(get_blues_category, axis=1)
    return df

def analyze_blues_structure():
    """Main analysis function for 12-bar blues structure."""
    
    # Load blues data
    df_blues, blues_songs = load_blues_data()
    
    if len(df_blues) == 0:
        print("No blues data found!")
        return
    
    # Categorize positions
    df_blues = categorize_blues_positions(df_blues)
    
    # Summary statistics
    print("\n" + "="*80)
    print("12-BAR BLUES STRUCTURAL ANALYSIS")
    print("="*80)
    
    category_counts = df_blues['blues_category'].value_counts()
    print(f"\nData Distribution in Blues Forms:")
    print(f"Chorus start (Bar 1, beats 1-2): {category_counts.get('chorus_start', 0):,} measurements")
    print(f"Turnaround (Bars 11-12):         {category_counts.get('turnaround', 0):,} measurements")
    print(f"Form end (Bar 12, beats 3-4):    {category_counts.get('form_end', 0):,} measurements")
    print(f"Other positions:                  {category_counts.get('other', 0):,} measurements")
    print(f"Total blues measurements:         {len(df_blues):,}")
    
    # Calculate mean BUR for each category
    category_stats = df_blues.groupby('blues_category')['swing_ratios'].agg([
        'count', 'mean', 'std', 'median'
    ]).round(4)
    
    print(f"\nBUR Statistics by Blues Position:")
    print(category_stats)
    
    # Extract data for statistical testing
    chorus_start_data = df_blues[df_blues['blues_category'] == 'chorus_start']['swing_ratios'].dropna()
    turnaround_data = df_blues[df_blues['blues_category'] == 'turnaround']['swing_ratios'].dropna()
    form_end_data = df_blues[df_blues['blues_category'] == 'form_end']['swing_ratios'].dropna()
    other_data = df_blues[df_blues['blues_category'] == 'other']['swing_ratios'].dropna()
    
    print(f"\nActual data sizes for statistical testing:")
    print(f"Chorus start: {len(chorus_start_data)} valid measurements")
    print(f"Turnaround:   {len(turnaround_data)} valid measurements")
    print(f"Form end:     {len(form_end_data)} valid measurements")
    print(f"Other:        {len(other_data)} valid measurements")
    
    # Statistical tests
    print(f"\n" + "-"*60)
    print("STATISTICAL TESTS")
    print("-"*60)
    
    # Test if we have enough data for each category
    categories_with_data = [
        ('chorus_start', chorus_start_data),
        ('turnaround', turnaround_data), 
        ('form_end', form_end_data),
        ('other', other_data)
    ]
    
    # Filter categories with sufficient data (at least 10 measurements)
    valid_categories = [(name, data) for name, data in categories_with_data if len(data) >= 10]
    
    if len(valid_categories) >= 2:
        # One-way ANOVA
        valid_data = [data for _, data in valid_categories]
        f_stat, p_anova = f_oneway(*valid_data)
        print(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_anova:.6f}")
        
        # Kruskal-Wallis (non-parametric alternative)
        h_stat, p_kruskal = kruskal(*valid_data)
        print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kruskal:.6f}")
        
        # Pairwise comparisons (only between categories with sufficient data)
        print(f"\nPairwise Comparisons (t-tests):")
        
        for i, (name1, data1) in enumerate(valid_categories):
            for name2, data2 in valid_categories[i+1:]:
                if len(data1) > 5 and len(data2) > 5:
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    d = cohens_d(data1, data2)
                    print(f"  {name1:12s} vs {name2:12s}: t = {t_stat:6.3f}, p = {p_val:.6f}, Cohen's d = {d:6.4f}")
        
        # Specific blues comparisons
        print(f"\nBlues-Specific Analysis:")
        
        # Compare turnaround vs other positions
        if len(turnaround_data) > 5 and len(other_data) > 5:
            turnaround_mean = turnaround_data.mean()
            other_mean = other_data.mean()
            diff = turnaround_mean - other_mean
            print(f"Turnaround effect: {diff:+.4f} BUR units from other positions")
            
        # Compare chorus start vs other positions  
        if len(chorus_start_data) > 5 and len(other_data) > 5:
            start_mean = chorus_start_data.mean()
            other_mean = other_data.mean()
            diff = start_mean - other_mean
            print(f"Chorus start effect: {diff:+.4f} BUR units from other positions")
            
    else:
        print("Insufficient data for comprehensive statistical analysis")
        print(f"Need at least 10 measurements per category, found: {[(name, len(data)) for name, data in valid_categories]}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    if len(valid_categories) >= 2:
        # Figure 1: Box plot and statistics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Filter plot data to only categories with sufficient data
        plot_categories = [name for name, data in valid_categories]
        plot_data = df_blues[df_blues['blues_category'].isin(plot_categories)]
        
        if len(plot_data) > 0:
            # Boxplot
            sns.boxplot(data=plot_data, x='blues_category', y='swing_ratios', ax=axes[0,0])
            axes[0,0].set_title('BUR by Blues Position (Boxplot)')
            axes[0,0].set_xlabel('Blues Position')
            axes[0,0].set_ylabel('BUR Value')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Violin plot
            if len(plot_data['blues_category'].unique()) > 1:
                sns.violinplot(data=plot_data, x='blues_category', y='swing_ratios', ax=axes[0,1])
                axes[0,1].set_title('BUR by Blues Position (Violin Plot)')
                axes[0,1].set_xlabel('Blues Position')  
                axes[0,1].set_ylabel('BUR Value')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # Bar plot of means
            means = category_stats['mean'][plot_categories]
            stds = category_stats['std'][plot_categories]
            
            x_pos = np.arange(len(means))
            bars = axes[1,0].bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                                color=['red', 'blue', 'green', 'gray'][:len(means)])
            axes[1,0].set_xticks(x_pos)
            axes[1,0].set_xticklabels(means.index, rotation=45)
            axes[1,0].set_title('Mean BUR by Blues Position (±1 SD)')
            axes[1,0].set_ylabel('Mean BUR Value')
            
            # Add value labels on bars
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                axes[1,0].text(i, mean_val + stds.iloc[i]/2, f'{mean_val:.3f}', 
                              ha='center', fontweight='bold')
            
            # 12-bar blues cycle visualization
            axes[1,1].text(0.1, 0.9, "12-Bar Blues Structure:", fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.8, "Bars 1-4:  I chord (Tonic)", fontsize=10, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.7, "Bars 5-6:  IV chord (Subdominant)", fontsize=10, transform=axes[1,1].transAxes) 
            axes[1,1].text(0.1, 0.6, "Bars 7-8:  I chord (Return to Tonic)", fontsize=10, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.5, "Bars 9-10: V chord (Dominant)", fontsize=10, transform=axes[1,1].transAxes)
            axes[1,1].text(0.1, 0.4, "Bars 11-12: I chord (Turnaround)", fontsize=10, transform=axes[1,1].transAxes, color='red')
            axes[1,1].text(0.1, 0.3, "Bar 1 beats 1-2: Chorus Start", fontsize=10, transform=axes[1,1].transAxes, color='blue')
            axes[1,1].set_xlim([0, 1])
            axes[1,1].set_ylim([0, 1])
            axes[1,1].axis('off')
            axes[1,1].set_title('Blues Form Analysis')
        
        plt.suptitle('12-Bar Blues Structural Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/blues_structural_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: blues_structural_analysis.png")
    
    # Summary
    print(f"\n" + "="*80)
    print("BLUES ANALYSIS SUMMARY")
    print("="*80)
    
    if len(valid_categories) >= 2 and 'f_stat' in locals():
        print(f"✓ Analysis completed for {len(blues_songs)} blues tunes")
        print(f"✓ Structural position effects: p = {p_anova:.6f} (ANOVA)")
        
        # Interpret significance
        if p_anova < 0.001:
            print(f"✓ HIGHLY SIGNIFICANT differences between blues positions")
        elif p_anova < 0.05:
            print(f"✓ SIGNIFICANT differences between blues positions") 
        else:
            print(f"✗ NO significant differences between blues positions")
            
        # Show key findings
        print(f"\nKey Findings:")
        for name, data in valid_categories:
            if len(data) > 0:
                mean_val = data.mean()
                count = len(data)
                print(f"• {name:15s}: {mean_val:.4f} BUR (n={count:,})")
                
        print(f"✓ Visualizations saved to outputs/")
        
    else:
        print(f"⚠️  Insufficient data for robust statistical analysis")
        print(f"   Need more measurements in blues structural positions")
        print(f"   Consider collecting more 12-bar blues recordings")
    
    return df_blues, category_stats

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
    
    df_blues, stats = analyze_blues_structure()