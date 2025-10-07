#!/usr/bin/env python3
"""
Visualize BUR Patterns - Show Examples of Different Surge Types

Creates plots showing examples of different BUR pattern types detected
by the comprehensive analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.data_utils import load_phrasebur_csv, ensure_output_dir


def plot_pattern_examples():
    """Create visualization of different BUR pattern types."""
    
    # Load data
    df = load_phrasebur_csv()
    results_df = pd.read_csv('outputs/bur_comprehensive_results.csv')
    
    # Find examples of each pattern type
    examples = {
        'Linear Increase': results_df[results_df['linear_significant'] == True].head(1),
        'Exponential': results_df[results_df['exponential_significant'] == True].head(1),
        'Step Change': results_df[results_df['step_detected'] == True].head(3),
        'End Surge': results_df[results_df['end_surge_detected'] == True].head(3),
        'Quadratic U': results_df[
            (results_df['quadratic_significant'] == True) & 
            (results_df['quad_shape'] == 'u_shaped')
        ].head(1),
        'Quadratic Inv-U': results_df[
            (results_df['quadratic_significant'] == True) & 
            (results_df['quad_shape'] == 'inverted_u')
        ].head(1),
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Examples of Different BUR Pattern Types', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    plot_idx = 0
    for pattern_type, example_df in examples.items():
        if plot_idx >= 6:
            break
            
        if len(example_df) == 0:
            axes[plot_idx].text(0.5, 0.5, f'No {pattern_type}\nDetected', 
                               ha='center', va='center', fontsize=12)
            axes[plot_idx].set_title(pattern_type)
            axes[plot_idx].axis('off')
            plot_idx += 1
            continue
        
        # Get first example (or multiple for step/end surge)
        for _, row in example_df.iterrows():
            if plot_idx >= 6:
                break
                
            # Get BUR values for this phrase
            phrase_data = df[
                (df['id'] == row['id']) & 
                (df['seg_id'] == row['seg_id'])
            ]
            bur_values = phrase_data['swing_ratios'].astype(float).values
            
            # Plot
            ax = axes[plot_idx]
            x = np.arange(len(bur_values))
            ax.plot(x, bur_values, 'o-', linewidth=2, markersize=6, label='Actual BUR')
            
            # Add fitted line based on pattern type
            if 'Linear' in pattern_type:
                slope = row['linear_slope']
                intercept = bur_values[0] - slope * 0
                y_fit = slope * x + intercept
                ax.plot(x, y_fit, '--', linewidth=2, alpha=0.7, label='Linear fit')
                title = f"{pattern_type}\n{row['artist']}\nSlope: {slope:.3f}, R²: {row['linear_r2']:.3f}"
                
            elif 'Exponential' in pattern_type:
                growth = row['exp_growth_rate']
                title = f"{pattern_type}\n{row['artist']}\nGrowth: {growth:.3f}, R²: {row['exp_r2']:.3f}"
                
            elif 'Step' in pattern_type:
                if pd.notna(row['step_position']):
                    pos = int(row['step_position'])
                    ax.axvline(x=pos, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Step position')
                    title = f"{pattern_type}\n{row['artist']}\nPos: {pos}, Mag: {row['step_magnitude']:.3f}"
                else:
                    title = f"{pattern_type}\n{row['artist']}"
                    
            elif 'End Surge' in pattern_type:
                end_n = int(row['n_values'] * 0.25)
                split = len(bur_values) - end_n
                ax.axvspan(split, len(bur_values)-1, alpha=0.2, color='green', label='End portion')
                title = f"{pattern_type}\n{row['artist']}\nMag: {row['end_surge_magnitude']:.3f}, d: {row['end_surge_effect_size']:.2f}"
                
            elif 'Quadratic' in pattern_type:
                a = row['quad_coef']
                title = f"{pattern_type}\n{row['artist']}\nQuad coef: {a:.4f}, R²: {row['quad_r2']:.3f}"
            
            else:
                title = f"{pattern_type}\n{row['artist']}"
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Position in Phrase')
            ax.set_ylabel('BUR (Beat-Upbeat Ratio)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            plot_idx += 1
            
            if 'Step' not in pattern_type and 'End Surge' not in pattern_type:
                break
    
    # Hide unused subplots
    for idx in range(plot_idx, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    ensure_output_dir()
    output_file = 'outputs/bur_pattern_examples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Pattern examples saved to: {output_file}")
    plt.close()


def plot_model_comparison_histogram():
    """Create histogram comparing R² values across models."""
    
    results_df = pd.read_csv('outputs/bur_comprehensive_results.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Fit Quality (R²) Comparison', fontsize=14, fontweight='bold')
    
    models = [
        ('linear_r2', 'Linear Model'),
        ('exp_r2', 'Exponential Model'),
        ('log_r2', 'Logarithmic Model'),
        ('quad_r2', 'Quadratic Model')
    ]
    
    for idx, (col, title) in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        r2_values = results_df[col].dropna()
        
        ax.hist(r2_values, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(r2_values.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {r2_values.median():.3f}')
        ax.axvline(r2_values.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {r2_values.mean():.3f}')
        
        ax.set_title(title)
        ax.set_xlabel('R² Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'outputs/model_r2_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ R² comparison saved to: {output_file}")
    plt.close()


def main():
    print("\nGenerating BUR pattern visualizations...")
    print("=" * 60)
    
    try:
        plot_pattern_examples()
        plot_model_comparison_histogram()
        print("\n✓ All visualizations complete!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
