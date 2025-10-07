"""
BUR Variation Analysis - Within-Phrase Consistency

Measures how much BUR varies within individual phrases using standard deviation.
Lower variation indicates more consistent swing timing within a phrase.
"""

import numpy as np
from utils.config import MIN_BUR_VALUES


def phrase_variation_stats(bur_values):
    """
    Calculate standard deviation of BUR values within a phrase.
    
    Args:
        bur_values: List or array of BUR values for a single phrase
        
    Returns:
        Dictionary containing:
            - n_values: Number of BUR values in the phrase
            - std_bur: Standard deviation of BUR values (measure of variation/consistency)
        Returns None if fewer than MIN_BUR_VALUES values
        
    Note:
        - Uses sample standard deviation (ddof=1, divide by n-1)
        - This is pure descriptive statistics (no hypothesis testing)
        - Lower std_bur = more consistent swing within phrase
        - Higher std_bur = more variable swing within phrase
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    std_bur = np.std(bur_values, ddof=1)
    return {
        'n_values': n,
        'std_bur': std_bur
    }
