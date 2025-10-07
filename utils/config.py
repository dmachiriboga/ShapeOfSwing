"""
Configuration constants for PhraseBur analysis.
"""

# Minimum number of BUR values required for statistical analysis
# This ensures adequate degrees of freedom:
# - Linear regression: n-2 degrees of freedom, so n=6 gives 4 d.f.
# - Phrase structure: middle section needs multiple values for std dev
# - Variation analysis: std with ddof=1 needs adequate sample size
MIN_BUR_VALUES = 6

# Statistical confidence level for confidence intervals
# 0.95 = 95% confidence level (standard in research)
CONFIDENCE_LEVEL = 0.95

# Number of parameters in linear regression (slope + intercept)
# Used to calculate degrees of freedom: df = n - LINEAR_REGRESSION_PARAMS
LINEAR_REGRESSION_PARAMS = 2

# False Discovery Rate (FDR) alpha level for multiple testing correction
# 0.05 = 5% false discovery rate (standard threshold)
FDR_ALPHA = 0.05

# Durbin-Watson threshold for strong positive autocorrelation
# DW < 1.5 indicates strong positive autocorrelation between adjacent residuals
# DW ≈ 2.0 = no autocorrelation
# DW < 2.0 = positive autocorrelation (adjacent values similar)
# DW > 2.0 = negative autocorrelation (adjacent values alternate)
DW_AUTOCORR_THRESHOLD = 1.5

# Default number of top performers to display in CLI output
# Can be overridden by user input at runtime
DEFAULT_TOP_N = 20
