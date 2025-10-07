"""
BUR Surge Analysis - Linear Trend Detection

Performs simple linear regression on BUR values across phrase positions
to detect trends (increases/decreases) in swing timing within phrases.
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
import pymannkendall as mk
from utils.config import MIN_BUR_VALUES, CONFIDENCE_LEVEL, LINEAR_REGRESSION_PARAMS, FDR_ALPHA


def linear_trend_analysis(bur_values):
    """
    Perform linear regression on BUR values to detect trends.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - slope: Rate of BUR change per position
            - intercept: BUR value at position 0
            - r2: Coefficient of determination (proportion of variance explained)
            - p_value: Significance of the slope (null hypothesis: slope = 0)
            - std_err: Standard error of the slope
            - conf_interval: Confidence interval for the slope (default 95%)
            - direction: 'increase', 'decrease', or 'none'
            - n_values: Number of BUR values
            - durbin_watson: Test statistic for autocorrelation (2 = none, <2 = positive, >2 = negative)
            
    Note:
        - Uses scipy.stats.linregress for regression
        - p_value tests H0: slope = 0 (no trend) with two-tailed test
        - Confidence interval uses t-distribution with n-2 degrees of freedom
        - **ASSUMES independence of observations** (often violated in musical data)
        - Durbin-Watson test checks this assumption (values near 2 indicate independence)
        - **WARNING:** If DW < 1.5 (autocorrelation present), standard errors and p-values
          are UNRELIABLE. Should use GLS, ARIMA, or Modified Mann-Kendall instead.
        - Current analysis reports DW but doesn't act on violations (~42% of phrases)
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
        
    x = np.arange(n)
    y = np.array(bur_values, dtype=float)
    
    # Perform linear regression
    result = stats.linregress(x, y)
    
    # Calculate confidence interval for slope
    # For a two-tailed test with confidence_level (e.g., 95%), we need the
    # critical value at (1 + confidence_level) / 2 = 0.975 for 95%
    # This accounts for splitting alpha/2 into each tail
    alpha = 1 - CONFIDENCE_LEVEL
    t_percentile = 1 - alpha / 2  # 0.975 for 95% CI
    df = n - LINEAR_REGRESSION_PARAMS  # degrees of freedom
    t_critical = stats.t.ppf(t_percentile, df)
    ci_lower = result.slope - t_critical * result.stderr # type: ignore
    ci_upper = result.slope + t_critical * result.stderr # type: ignore
    
    # Determine direction
    if result.slope > 0: # type: ignore
        direction = 'increase'
    elif result.slope < 0: # type: ignore
        direction = 'decrease'
    else:
        direction = 'none'
    
    # Calculate Durbin-Watson statistic for autocorrelation
    # DW ≈ 2: no autocorrelation
    # DW < 2: positive autocorrelation (adjacent values are similar)
    # DW > 2: negative autocorrelation (adjacent values alternate)
    residuals = y - (result.slope * x + result.intercept) # type: ignore
    dw_stat = durbin_watson(residuals)
    
    return {
        'slope': result.slope, # type: ignore
        'intercept': result.intercept, # type: ignore
        'r2': result.rvalue ** 2, # type: ignore
        'p_value': result.pvalue, # type: ignore
        'std_err': result.stderr, # type: ignore
        'conf_interval': (ci_lower, ci_upper),
        'direction': direction,
        'n_values': n,
        'durbin_watson': dw_stat
    }


def fdr_correction(p_values, alpha=FDR_ALPHA):
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.
    
    Args:
        p_values: List of p-values to correct
        alpha: Desired false discovery rate (default from config: 0.05)
        
    Returns:
        Tuple of (reject, p_values_corrected)
            - reject: Boolean array indicating which tests to reject
            - p_values_corrected: FDR-corrected p-values
            
    Note:
        Uses Benjamini-Hochberg procedure to control false discovery rate.
        This is less conservative than Bonferroni correction and more
        appropriate when testing many hypotheses.
    """
    # Remove None values for correction
    valid_p_values = [p for p in p_values if p is not None]
    
    if len(valid_p_values) == 0:
        return [], []
    
    # Apply FDR correction
    reject, p_corrected, _, _ = multipletests(
        valid_p_values, 
        alpha=alpha, 
        method='fdr_bh'  # Benjamini-Hochberg
    )
    
    return reject, p_corrected


def exponential_trend_analysis(bur_values):
    """
    Perform exponential regression on BUR values to detect exponential trends.
    Model: y = a * exp(b * x) or equivalently: ln(y) = ln(a) + b*x
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - growth_rate: Exponential growth rate parameter (b)
            - amplitude: Amplitude parameter (a)
            - r2: Coefficient of determination
            - p_value: Significance of exponential fit
            - direction: 'increase', 'decrease', or 'none'
            - n_values: Number of BUR values
            - residual_std: Standard deviation of residuals
            
    Note:
        - Uses log-linear regression for exponential fitting
        - Requires positive BUR values (typically satisfied for swing ratios)
        - p_value tests H0: growth_rate = 0 (no exponential trend)
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    x = np.arange(n)
    y = np.array(bur_values, dtype=float)
    
    # Ensure positive values for log transformation
    if np.any(y <= 0):
        return None
    
    # Log-linear regression: ln(y) = ln(a) + b*x
    log_y = np.log(y)
    result = stats.linregress(x, log_y)
    
    growth_rate = result.slope  # b parameter
    ln_amplitude = result.intercept  # ln(a)
    amplitude = np.exp(ln_amplitude)
    
    # Calculate R² for exponential fit
    y_pred = amplitude * np.exp(growth_rate * x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Determine direction
    if growth_rate > 0:
        direction = 'increase'
    elif growth_rate < 0:
        direction = 'decrease'
    else:
        direction = 'none'
    
    # Residual standard deviation
    residuals = y - y_pred
    residual_std = np.std(residuals, ddof=1)
    
    return {
        'growth_rate': result.slope,
        'amplitude': amplitude,
        'r2': r2,
        'p_value': result.pvalue,
        'std_err': result.stderr,
        'direction': direction,
        'n_values': n,
        'residual_std': residual_std
    }


def logarithmic_trend_analysis(bur_values):
    """
    Perform logarithmic regression on BUR values to detect logarithmic trends.
    Model: y = a + b * ln(x + 1)
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - log_coefficient: Logarithmic coefficient (b)
            - intercept: Intercept (a)
            - r2: Coefficient of determination
            - p_value: Significance of logarithmic fit
            - direction: 'increase', 'decrease', or 'none'
            - n_values: Number of BUR values
            - residual_std: Standard deviation of residuals
            
    Note:
        - Uses x+1 to avoid log(0)
        - p_value tests H0: log_coefficient = 0 (no logarithmic trend)
        - Good for detecting rapid initial changes that level off
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    x = np.arange(n)
    y = np.array(bur_values, dtype=float)
    
    # Logarithmic transformation of x (add 1 to avoid log(0))
    log_x = np.log(x + 1)
    result = stats.linregress(log_x, y)
    
    # Calculate R²
    y_pred = result.intercept + result.slope * log_x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Determine direction
    if result.slope > 0:
        direction = 'increase'
    elif result.slope < 0:
        direction = 'decrease'
    else:
        direction = 'none'
    
    # Residual standard deviation
    residuals = y - y_pred
    residual_std = np.std(residuals, ddof=1)
    
    return {
        'log_coefficient': result.slope,
        'intercept': result.intercept,
        'r2': r2,
        'p_value': result.pvalue,
        'std_err': result.stderr,
        'direction': direction,
        'n_values': n,
        'residual_std': residual_std
    }


def quadratic_trend_analysis(bur_values):
    """
    Perform quadratic regression on BUR values to detect U-shaped or inverted-U patterns.
    Model: y = a + b*x + c*x²
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - quadratic_coef: Quadratic coefficient (c)
            - linear_coef: Linear coefficient (b)
            - intercept: Intercept (a)
            - r2: Coefficient of determination
            - p_value: Significance of quadratic term
            - shape: 'u_shaped', 'inverted_u', or 'linear'
            - n_values: Number of BUR values
            - residual_std: Standard deviation of residuals
            
    Note:
        - Uses numpy polyfit for quadratic regression
        - p_value tests significance of quadratic term via F-test
        - shape determined by sign of quadratic coefficient
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    x = np.arange(n)
    y = np.array(bur_values, dtype=float)
    
    # Fit quadratic model
    coeffs = np.polyfit(x, y, 2)
    c, b, a = coeffs  # c*x² + b*x + a
    
    # Calculate predictions and R²
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # F-test for quadratic term significance
    # Compare quadratic model vs linear model
    linear_coeffs = np.polyfit(x, y, 1)
    y_pred_linear = np.polyval(linear_coeffs, x)
    ss_res_linear = np.sum((y - y_pred_linear) ** 2)
    
    # F-statistic: (SS_res_linear - SS_res_quadratic) / (df_diff) / (SS_res_quadratic / df_residual)
    df_quad = n - 3  # 3 parameters
    df_linear = n - 2  # 2 parameters
    
    if ss_res > 0 and df_quad > 0:
        f_stat = ((ss_res_linear - ss_res) / 1) / (ss_res / df_quad)
        p_value = 1 - stats.f.cdf(f_stat, 1, df_quad)
    else:
        p_value = 1.0
    
    # Determine shape
    if abs(c) < 1e-10:
        shape = 'linear'
    elif c > 0:
        shape = 'u_shaped'
    else:
        shape = 'inverted_u'
    
    # Residual standard deviation
    residuals = y - y_pred
    residual_std = np.std(residuals, ddof=2)
    
    return {
        'quadratic_coef': c,
        'linear_coef': b,
        'intercept': a,
        'r2': r2,
        'p_value': p_value,
        'shape': shape,
        'n_values': n,
        'residual_std': residual_std
    }


def step_change_analysis(bur_values, min_step_size=0.1):
    """
    Detect sudden step changes (jumps) in BUR values.
    Uses change point detection via sliding window comparison.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        min_step_size: Minimum step size to consider significant (default 0.1)
        
    Returns:
        Dictionary containing:
            - has_step: Boolean indicating if significant step detected
            - step_position: Position of largest step (None if no step)
            - step_magnitude: Magnitude of step change
            - p_value: T-test p-value comparing segments before/after step
            - direction: 'increase', 'decrease', or 'none'
            - n_values: Number of BUR values
    
    **WARNING - METHODOLOGICAL ISSUE:**
        This function tests EVERY possible split point and selects the "best" one
        by minimum p-value. This is p-hacking within each phrase!
        
        Problems:
        1. For 15-note phrase, tests 10 different splits
        2. Selects split with lowest p-value (most "significant")
        3. Returns that p-value WITHOUT correcting for testing 10 splits
        4. This inflates Type I error dramatically
        
        **RECOMMENDED FIX:**
        - Use proper changepoint detection (PELT, Binary Segmentation, etc.)
        - These methods account for testing multiple points
        - Or apply Bonferroni correction: p_adjusted = p_min * n_splits_tested
        
        **Current 4.30% significant rate is likely overestimate**
            
    Note:
        - Tests all possible split points (need at least 3 values on each side)
        - Uses Welch's t-test for unequal variances
        - p_value tests H0: no difference between segments
        - **Does NOT correct for multiple split points tested**
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    best_p_value = 1.0
    best_position = None
    best_magnitude = 0
    best_direction = 'none'
    
    # Test each possible split point (need at least 3 values on each side)
    for split in range(3, n - 2):
        segment1 = y[:split]
        segment2 = y[split:]
        
        mean1 = np.mean(segment1)
        mean2 = np.mean(segment2)
        magnitude = mean2 - mean1
        
        # Only consider if magnitude exceeds threshold
        if abs(magnitude) >= min_step_size:
            # Welch's t-test (doesn't assume equal variances)
            t_stat, p_val = stats.ttest_ind(segment1, segment2, equal_var=False)
            
            if p_val < best_p_value:
                best_p_value = p_val
                best_position = split
                best_magnitude = magnitude
                best_direction = 'increase' if magnitude > 0 else 'decrease'
    
    has_step = best_p_value < 0.05 and best_position is not None
    
    return {
        'has_step': has_step,
        'step_position': best_position,
        'step_magnitude': best_magnitude,
        'p_value': best_p_value,
        'direction': best_direction,
        'n_values': n
    }


def end_phrase_surge_analysis(bur_values, end_proportion=0.25):
    """
    Detect BUR surges specifically at the end of phrases.
    Compares the last portion of the phrase to the rest.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        end_proportion: Proportion of phrase to consider as "end" (default 0.25 = last 25%)
        
    Returns:
        Dictionary containing:
            - has_end_surge: Boolean indicating if significant end surge detected
            - end_mean: Mean BUR in end portion
            - rest_mean: Mean BUR in rest of phrase
            - surge_magnitude: Difference (end_mean - rest_mean)
            - p_value: T-test p-value comparing end vs rest
            - effect_size: Cohen's d effect size
            - direction: 'increase', 'decrease', or 'none'
            - n_values: Number of BUR values
            - n_end: Number of values in end portion
            
    Note:
        - Uses one-tailed t-test (testing if end > rest)
        - Effect size helps interpret practical significance
        - Good for detecting "final push" timing patterns
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    # Determine split point (at least 2 values in end section)
    n_end = max(2, int(n * end_proportion))
    split = n - n_end
    
    rest_portion = y[:split]
    end_portion = y[split:]
    
    rest_mean = np.mean(rest_portion)
    end_mean = np.mean(end_portion)
    surge_magnitude = end_mean - rest_mean
    
    # One-tailed t-test: is end significantly different from rest?
    t_stat, p_val_two_tailed = stats.ttest_ind(end_portion, rest_portion, equal_var=False)
    
    # Convert to one-tailed p-value based on direction of t-statistic
    if surge_magnitude > 0:
        # Testing if end > rest (positive difference)
        p_value = p_val_two_tailed / 2 if t_stat > 0 else 1 - (p_val_two_tailed / 2)
        direction = 'increase'
    else:
        # Testing if end < rest (negative difference)
        p_value = p_val_two_tailed / 2 if t_stat < 0 else 1 - (p_val_two_tailed / 2)
        direction = 'decrease'
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt((np.var(rest_portion, ddof=1) + np.var(end_portion, ddof=1)) / 2)
    effect_size = surge_magnitude / pooled_std if pooled_std > 0 else 0
    
    has_end_surge = p_value < 0.05 and surge_magnitude > 0
    
    return {
        'has_end_surge': has_end_surge,
        'end_mean': end_mean,
        'rest_mean': rest_mean,
        'surge_magnitude': surge_magnitude,
        'p_value': p_value,
        'effect_size': effect_size,
        'direction': direction,
        'n_values': n,
        'n_end': n_end
    }


def comprehensive_trend_analysis(bur_values):
    """
    Run all trend analysis methods and return comprehensive results.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing results from all analysis methods:
            - linear: Linear trend results
            - exponential: Exponential trend results
            - logarithmic: Logarithmic trend results
            - quadratic: Quadratic trend results
            - step_change: Step change detection results
            - end_surge: End-phrase surge results
            - mann_kendall: Mann-Kendall comprehensive results
            - best_model: Name of model with highest R² (for curve fits)
    
    **WARNING - MODEL SELECTION BIAS:**
        Selecting "best model" by R² AFTER fitting all models is model selection
        after the fact (data dredging). Problems:
        
        1. No penalty for model complexity (AIC/BIC would be better)
        2. R² not comparable across different transformations:
           - Exponential R² computed on original scale
           - Log R² computed on log scale
           - These are NOT directly comparable!
        3. Should use cross-validation or information criteria
        4. Testing multiple models increases false positive rate
        
        **RECOMMENDATION:**
        - Choose model based on theory/domain knowledge BEFORE analysis
        - Or use AIC/BIC for model selection (penalizes complexity)
        - Or use cross-validation
        - Apply multiple testing correction if testing all models
            
    Note:
        - Useful for comparing multiple model fits
        - Best model selection based on R² (but see warning above)
        - Mann-Kendall tests are non-parametric alternatives to parametric tests
    """
    linear = linear_trend_analysis(bur_values)
    exponential = exponential_trend_analysis(bur_values)
    logarithmic = logarithmic_trend_analysis(bur_values)
    quadratic = quadratic_trend_analysis(bur_values)
    step_change = step_change_analysis(bur_values)
    end_surge = end_phrase_surge_analysis(bur_values)
    mann_kendall = mann_kendall_comprehensive(bur_values)
    
    # Determine best model by R²
    models = {}
    if linear and 'r2' in linear:
        models['linear'] = linear['r2']
    if exponential and 'r2' in exponential:
        models['exponential'] = exponential['r2']
    if logarithmic and 'r2' in logarithmic:
        models['logarithmic'] = logarithmic['r2']
    if quadratic and 'r2' in quadratic:
        models['quadratic'] = quadratic['r2']
    
    best_model = max(models.items(), key=lambda x: x[1])[0] if models else None
    
    return {
        'linear': linear,
        'exponential': exponential,
        'logarithmic': logarithmic,
        'quadratic': quadratic,
        'step_change': step_change,
        'end_surge': end_surge,
        'mann_kendall': mann_kendall,
        'best_model': best_model
    }


def mann_kendall_trend_analysis(bur_values):
    """
    Perform Mann-Kendall trend test on BUR values.
    Non-parametric test for monotonic trends that doesn't assume linearity.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - trend: Trend direction ('increasing', 'decreasing', 'no trend')
            - p_value: Two-tailed p-value
            - tau: Kendall's tau statistic (-1 to 1, measure of trend strength)
            - z_score: Normalized test statistic
            - slope: Sen's slope (median slope of all pairwise slopes)
            - n_values: Number of BUR values
            
    Note:
        - Non-parametric: doesn't assume normal distribution
        - Robust to outliers
        - Tests for monotonic trend (not necessarily linear)
        - tau ≈ 0.3+ indicates moderate trend, 0.5+ strong trend
        - Sen's slope is more robust than linear regression slope
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    # Perform original Mann-Kendall test
    result = mk.original_test(y)
    
    return {
        'trend': result.trend,
        'p_value': result.p,
        'tau': result.Tau,
        'z_score': result.z,
        'slope': result.slope,  # Sen's slope
        'n_values': n
    }


def mann_kendall_modified_analysis(bur_values):
    """
    Perform Modified Mann-Kendall test that accounts for autocorrelation.
    This is the Hamed and Rao (1998) modification.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - trend: Trend direction ('increasing', 'decreasing', 'no trend')
            - p_value: Two-tailed p-value (adjusted for autocorrelation)
            - tau: Kendall's tau statistic
            - z_score: Modified z-score accounting for autocorrelation
            - slope: Sen's slope
            - n_values: Number of BUR values
            
    Note:
        - Better for data with autocorrelation (common in time series)
        - Uses lag-1 autocorrelation to adjust variance
        - More conservative than original Mann-Kendall when autocorrelation present
        - Recommended for musical phrase data where adjacent notes may be correlated
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    # Perform modified Mann-Kendall test (Hamed-Rao)
    result = mk.hamed_rao_modification_test(y)
    
    return {
        'trend': result.trend,
        'p_value': result.p,
        'tau': result.Tau,
        'z_score': result.z,
        'slope': result.slope,
        'n_values': n
    }


def mann_kendall_seasonal_analysis(bur_values, period=4):
    """
    Perform Seasonal Mann-Kendall test to detect trends with periodic patterns.
    Useful if phrases have recurring structural patterns.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        period: Period of seasonality (default 4 for 4-bar phrases)
        
    Returns:
        Dictionary containing:
            - trend: Overall trend direction
            - p_value: Two-tailed p-value
            - tau: Kendall's tau statistic
            - z_score: Test statistic
            - slope: Sen's slope
            - n_values: Number of BUR values
            
    Note:
        - Accounts for periodic/cyclical patterns in data
        - Tests for trend after removing seasonal component
        - Good if phrase structure creates repeating patterns
        - Period should match phrase structure (e.g., 4 for 4-bar phrases)
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES or n < period * 2:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    try:
        # Perform seasonal Mann-Kendall test
        result = mk.seasonal_test(y, period=period)
        
        return {
            'trend': result.trend,
            'p_value': result.p,
            'tau': result.Tau,
            'z_score': result.z,
            'slope': result.slope,
            'n_values': n
        }
    except:
        # If seasonal test fails, return None
        return None


def sens_slope_analysis(bur_values):
    """
    Calculate Sen's slope with confidence interval.
    Sen's slope is the median of all pairwise slopes - more robust than OLS.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - slope: Sen's slope estimate
            - conf_interval: Confidence interval for slope
            - intercept: Intercept based on Sen's slope
            - n_values: Number of BUR values
            
    Note:
        - More robust to outliers than linear regression
        - Doesn't assume normal distribution
        - Confidence interval uses distribution-free method
        - Good companion to Mann-Kendall test
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    x = np.arange(n)
    y = np.array(bur_values, dtype=float)
    
    # Calculate all pairwise slopes
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)
    
    if len(slopes) == 0:
        return None
    
    # Sen's slope is the median
    sens_slope = np.median(slopes)
    
    # Calculate confidence interval using distribution-free quantile method
    # This is the correct method for Sen's slope CI
    alpha = 1 - CONFIDENCE_LEVEL
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    # Calculate variance of S statistic (NOT the slope variance)
    # S is the sum of signs of all pairwise differences
    n_slopes = len(slopes)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate C_alpha for quantile method
    c_alpha = z_alpha * np.sqrt(var_s)
    
    # Calculate indices for confidence interval
    # M1 = (N - C_alpha) / 2, M2 = (N + C_alpha) / 2
    # where N is the number of slopes
    sorted_slopes = np.sort(slopes)
    m1 = int(np.floor((n_slopes - c_alpha) / 2))
    m2 = int(np.ceil((n_slopes + c_alpha) / 2))
    
    # Bound indices to valid range
    m1 = max(0, min(m1, n_slopes - 1))
    m2 = max(0, min(m2, n_slopes - 1))
    
    ci_lower = sorted_slopes[m1]
    ci_upper = sorted_slopes[m2]
    
    # Calculate intercept: median(y - slope*x)
    intercepts = y - sens_slope * x
    intercept = np.median(intercepts)
    
    return {
        'slope': sens_slope,
        'conf_interval': (ci_lower, ci_upper),
        'intercept': intercept,
        'n_values': n
    }


def mann_kendall_comprehensive(bur_values):
    """
    Run all Mann-Kendall variants and return comprehensive results.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        
    Returns:
        Dictionary containing:
            - original: Original Mann-Kendall test results
            - modified: Modified MK test (accounts for autocorrelation)
            - seasonal: Seasonal MK test results
            - sens_slope: Sen's slope analysis
            - recommended_test: Which test is most appropriate
            
    Note:
        - Compare results across tests to assess robustness
        - If autocorrelation present, use modified test
        - Sen's slope more robust than linear regression slope
    """
    original = mann_kendall_trend_analysis(bur_values)
    modified = mann_kendall_modified_analysis(bur_values)
    seasonal = mann_kendall_seasonal_analysis(bur_values)
    sens = sens_slope_analysis(bur_values)
    
    # Determine recommended test based on data characteristics
    if original is None:
        recommended_test = None
    else:
        # Calculate lag-1 autocorrelation
        y = np.array(bur_values, dtype=float)
        if len(y) > 2:
            autocorr = np.corrcoef(y[:-1], y[1:])[0, 1]
            
            if abs(autocorr) > 0.3:
                recommended_test = 'modified'  # Use modified if strong autocorrelation
            elif seasonal is not None:
                recommended_test = 'seasonal'  # Use seasonal if available
            else:
                recommended_test = 'original'
        else:
            recommended_test = 'original'
    
    return {
        'original': original,
        'modified': modified,
        'seasonal': seasonal,
        'sens_slope': sens,
        'recommended_test': recommended_test
    }


def sliding_window_surge_analysis(bur_values, window_sizes=[4, 6, 8], min_tau=0.4, alpha=0.05):
    """
    Detect localized BUR surges within sub-segments of a phrase using sliding windows.
    Tests Mann-Kendall trend on overlapping windows to find where surges occur.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        window_sizes: List of window sizes to test (default [4, 6, 8])
        min_tau: Minimum Kendall's tau to consider a meaningful trend (default 0.4)
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary containing:
            - has_local_surge: Boolean indicating if any significant localized surge found
            - surges: List of detected surge windows, each containing:
                - window_size: Size of window
                - start_pos: Starting position of surge window
                - end_pos: Ending position (exclusive)
                - tau: Kendall's tau for this window
                - p_value: Mann-Kendall p-value
                - sens_slope: Sen's slope for window
                - mean_bur: Mean BUR in this window
                - direction: 'increasing' or 'decreasing'
            - strongest_surge: Details of window with highest |tau| (if any)
            - n_values: Total number of BUR values in phrase
            - windows_tested: Total number of windows tested
            
    **WARNING - CRITICAL METHODOLOGICAL ISSUE:**
        This function tests MANY windows per phrase (typically 20-50 windows) with
        heavy overlap (5/6 of data points shared between adjacent windows). This creates:
        
        1. MASSIVE multiple testing problem:
           - For 2,488 phrases with avg 40 windows each = ~100,000 tests
           - Current FDR correction only on strongest window per phrase (2,488 tests)
           - This inflates false positives by 40x!
        
        2. Pseudo-replication due to overlapping windows:
           - Adjacent windows share 5/6 of their data points
           - Violates independence assumption of FDR correction
        
        3. Selection bias:
           - Selecting strongest window BEFORE multiple testing correction
           - Should correct ALL windows FIRST, then select
        
        **RECOMMENDED FIX:**
        - Use non-overlapping windows (step size = window size)
        - Apply FDR to ALL windows across ALL phrases (not per-phrase)
        - Or use proper changepoint detection methods (PELT, BinSeg)
        
        **INTERPRETATION:**
        Current results (~25% significant) are likely MASSIVE OVERESTIMATES.
        True rate after proper correction likely closer to full-phrase analysis (~2%).
    
    Note:
        - Tests multiple window sizes to detect surges of different lengths
        - Overlapping windows allow detection anywhere in phrase
        - tau threshold filters out weak trends
        - Useful for detecting "mid-phrase acceleration" or "final phrase push"
        - **NO multiple testing correction applied within function - MUST apply FDR on ALL results**
    
    Example:
        If a phrase has BUR surge in the middle (positions 5-10), this will detect
        it even if the full phrase shows no overall trend.
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    all_surges = []
    windows_tested = 0
    
    # Test each window size
    for win_size in window_sizes:
        if win_size > n or win_size < 3:
            continue
        
        # Slide window across phrase
        for start in range(n - win_size + 1):
            end = start + win_size
            window = y[start:end]
            windows_tested += 1
            
            # Run Mann-Kendall on window (use modified for autocorrelation)
            try:
                result = mk.hamed_rao_modification_test(window)
                
                # Check if significant and strong enough
                if result.p < alpha and abs(result.Tau) >= min_tau:
                    direction = 'increasing' if result.Tau > 0 else 'decreasing'
                    
                    surge_info = {
                        'window_size': win_size,
                        'start_pos': start,
                        'end_pos': end,
                        'tau': result.Tau,
                        'p_value': result.p,
                        'sens_slope': result.slope,
                        'mean_bur': np.mean(window),
                        'direction': direction
                    }
                    all_surges.append(surge_info)
            except:
                # Skip windows that fail the test
                continue
    
    # Find strongest surge (highest |tau|)
    strongest = None
    if all_surges:
        strongest = max(all_surges, key=lambda x: abs(x['tau']))
    
    return {
        'has_local_surge': len(all_surges) > 0,
        'surges': all_surges,
        'strongest_surge': strongest,
        'n_values': n,
        'windows_tested': windows_tested,
        'n_significant_windows': len(all_surges)
    }


def sliding_window_surge_analysis_seasonal(bur_values, window_sizes=[4, 6, 8], min_tau=0.4, alpha=0.05, period=4):
    """
    Detect localized BUR surges using SEASONAL Mann-Kendall test on sliding windows.
    Better for musical data with periodic structures (e.g., 4-bar phrases).
    
    Args:
        bur_values: List or array of BUR values in temporal order
        window_sizes: List of window sizes to test (default [4, 6, 8])
        min_tau: Minimum Kendall's tau to consider a meaningful trend (default 0.4)
        alpha: Significance level (default 0.05)
        period: Period for seasonal component (default 4 for 4-beat patterns)
        
    Returns:
        Dictionary containing:
            - has_local_surge: Boolean indicating if any significant localized surge found
            - surges: List of detected surge windows, each containing:
                - window_size: Size of window
                - start_pos: Starting position of surge window
                - end_pos: Ending position (exclusive)
                - tau: Kendall's tau for this window
                - p_value: Seasonal Mann-Kendall p-value
                - sens_slope: Sen's slope for window
                - mean_bur: Mean BUR in this window
                - direction: 'increasing' or 'decreasing'
                - test_type: 'seasonal' or 'modified' (seasonal requires min length)
            - strongest_surge: Details of window with highest |tau| (if any)
            - n_values: Total number of BUR values in phrase
            - windows_tested: Total number of windows tested
    
    **WARNING - CRITICAL METHODOLOGICAL ISSUES:**
        Same multiple testing problems as sliding_window_surge_analysis(), PLUS:
        
        1. Unjustified periodicity assumption:
           - No empirical evidence that BUR has period=4 cycles
           - Musical phrases don't necessarily have 4-beat swing periodicity
           - Wrong period can INCREASE false positives
           - Should TEST for periodicity first (spectral analysis, ACF)
        
        2. Mixing test types invalidates comparison:
           - Uses seasonal MK for windows ≥8 notes
           - Falls back to modified MK for smaller windows
           - Different tests have different properties/power
           - Cannot directly compare p-values across test types
        
        **RECOMMENDED:**
        - Test for periodicity BEFORE applying seasonal adjustment
        - Use single test type consistently
        - Apply proper multiple testing correction to ALL windows
            
    Note:
        - Seasonal Mann-Kendall accounts for periodic patterns in music
        - Falls back to modified Mann-Kendall if window too small for seasonal test
        - Requires window_size >= period * 2 for seasonal test
        - **ASSUMPTION: period=4 is currently unjustified - needs validation**
    
    Example:
        For a phrase with 4-beat periodic structure, seasonal test removes
        the periodic component before testing for trends.
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    all_surges = []
    windows_tested = 0
    
    # Test each window size
    for win_size in window_sizes:
        if win_size > n or win_size < 3:
            continue
        
        # Slide window across phrase
        for start in range(n - win_size + 1):
            end = start + win_size
            window = y[start:end]
            windows_tested += 1
            
            # Try seasonal Mann-Kendall first (if window large enough)
            result = None
            test_type = 'modified'
            
            if win_size >= period * 2:
                try:
                    result = mk.seasonal_test(window, period=period)
                    test_type = 'seasonal'
                except:
                    result = None
            
            # Fall back to modified Mann-Kendall if seasonal not applicable
            if result is None:
                try:
                    result = mk.hamed_rao_modification_test(window)
                    test_type = 'modified'
                except:
                    continue
            
            # Store ALL windows (not just significant ones)
            if result:
                direction = 'increasing' if result.Tau > 0 else 'decreasing'
                
                window_info = {
                    'window_size': win_size,
                    'start_pos': start,
                    'end_pos': end,
                    'tau': result.Tau,
                    'p_value': result.p,
                    'sens_slope': result.slope,
                    'mean_bur': np.mean(window),
                    'direction': direction,
                    'test_type': test_type
                }
                all_surges.append(window_info)
    
    # Find strongest surge (highest |tau|)
    strongest = None
    if all_surges:
        strongest = max(all_surges, key=lambda x: abs(x['tau']))
    
    return {
        'has_local_surge': len(all_surges) > 0,
        'surges': all_surges,
        'strongest_surge': strongest,
        'n_values': n,
        'windows_tested': windows_tested,
        'n_significant_windows': len(all_surges)
    }


def sliding_window_surge_analysis_seasonal_all_windows(bur_values, window_sizes=[4, 6, 8], min_tau=0.4, period=4):
    """
    Detect localized BUR surges using SEASONAL Mann-Kendall test on sliding windows.
    Returns ALL windows tested with their p-values for proper FDR correction.
    
    Args:
        bur_values: List or array of BUR values in temporal order
        window_sizes: List of window sizes to test (default [4, 6, 8])
        min_tau: Minimum Kendall's tau to consider a meaningful trend (default 0.4)
        period: Period for seasonal component (default 4 for 4-beat patterns)
        
    Returns:
        Dictionary containing:
            - all_windows: List of ALL windows tested, each containing:
                - window_size: Size of window
                - start_pos: Starting position of surge window
                - end_pos: Ending position (exclusive)
                - tau: Kendall's tau for this window
                - p_value: Seasonal Mann-Kendall p-value
                - sens_slope: Sen's slope for window
                - mean_bur: Mean BUR in this window
                - direction: 'increasing' or 'decreasing'
                - test_type: 'seasonal' or 'modified'
            - n_values: Total number of BUR values in phrase
            - windows_tested: Total number of windows tested
            
    Note:
        Returns ALL windows, not just significant ones, for proper multiple testing correction.
    """
    n = len(bur_values)
    if n < MIN_BUR_VALUES:
        return None
    
    y = np.array(bur_values, dtype=float)
    
    all_windows = []
    windows_tested = 0
    
    # Test each window size
    for win_size in window_sizes:
        if win_size > n or win_size < 3:
            continue
        
        # Slide window across phrase
        for start in range(n - win_size + 1):
            end = start + win_size
            window = y[start:end]
            windows_tested += 1
            
            # Try seasonal Mann-Kendall first (if window large enough)
            result = None
            test_type = 'modified'
            
            if win_size >= period * 2:
                try:
                    result = mk.seasonal_test(window, period=period)
                    test_type = 'seasonal'
                except:
                    result = None
            
            # Fall back to modified Mann-Kendall if seasonal not applicable
            if result is None:
                try:
                    result = mk.hamed_rao_modification_test(window)
                    test_type = 'modified'
                except:
                    continue
            
            # Store ALL windows (not just significant ones)
            if result:
                direction = 'increasing' if result.Tau > 0 else 'decreasing'
                
                window_info = {
                    'window_size': win_size,
                    'start_pos': start,
                    'end_pos': end,
                    'tau': result.Tau,
                    'p_value': result.p,
                    'sens_slope': result.slope,
                    'mean_bur': np.mean(window),
                    'direction': direction,
                    'test_type': test_type
                }
                all_windows.append(window_info)
    
    return {
        'all_windows': all_windows,
        'n_values': n,
        'windows_tested': windows_tested
    }
