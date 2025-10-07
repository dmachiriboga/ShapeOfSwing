# The Shape of Swing: How Timing Outlines Form in Jazz Solos

This repository contains Python scripts and analysis tools for **The Shape of Swing: How Timing Outlines Form in Jazz Solos**. Presented on 9 October 2025 at the Third International Conference on Computational and Cognitive Musicology ([ICCCM 2025](https://digital.musicology.org/icccm-2025/)) at Aalborg University, Denmark. 

The scripts analyze Beat-Upbeat Ratio (BUR), a metric to quantify swing, using data from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html).

## Abstract

The expressive shaping of swing and timing is central to jazz improvisation; however, the microrhythmic nuances that define swing are difficult to analyze using conventional music theory tools. One metric that has proven especially useful in quantifying swing is the Beat-Upbeat Ratio (BUR), which measures the durational relationship between a downbeat eighth note and its following upbeat counterpart. While standard swing is often simplistically equated with a triplet feel (i.e., BUR ≈ 2:1), empirical studies show that performers typically swing closer to 1.3–1.4:1 and that these ratios vary throughout solos.

Using more than 4,800 melodic phrases from solos in the Weimar Jazz Database, we investigate how BUR evolves over the course of a phrase and show that it functions as a tool for delineating musical form. We apply linear, exponential, and logarithmic trend-fitting models to quantify this phrase timing behavior, using p-value and R² thresholds to assess statistical significance.

Our results support the widespread phenomenon of “BUR surges,” which are progressive increases in swing ratio toward the end of a phrase. These surges often align with structural boundaries in the solos, suggesting that swing timing functions not only as groove but also as a form-bearing gesture. Between 41% and 59% of phrases exhibit BUR increase, with approximately 8% showing statistically significant upward trends at both phrase onset and close, well above the expected rate for randomized models. Performer-level analysis reveals that soloists like Bix Beiderbecke and Steve Lacy use BUR surges much more than other performers.

Our findings reinforce the view that swing in jazz solos is not merely a rhythmic style but a nuanced expressive device capable of articulating large-scale musical structure. By empirically modeling these microrhythmic gestures across a broad corpus, our study highlights the interpretive depth of swing and the value of computational methods in jazz research.

## Usage

### Setup

1. [Download](https://jazzomat.hfm-weimar.de/download/download.html) the relevant data from the Weimar Jazz Database.
2. Download the [MeloSpyGUI](https://jazzomat.hfm-weimar.de/download/download.html) and open it by navigating to the `bin` directory and opening `mss_gui`. You may need to override your computer's security controls to open this file. 
3. Create `data/phrasebur_raw.csv` using the MeloSpyGUI (or use existing raw data).
4. Run the data cleaning script: `python utils/clean_data.py`
5. Install dependencies: `poetry install`

### Running Analyses

**BUR Surge Analysis** - Detects linear trends in BUR across phrase positions:
```bash
poetry run python cli/bur_surge_cli.py
```
- Uses linear regression with FDR correction (Benjamini-Hochberg)
- Tests for autocorrelation with Durbin-Watson statistic
- Output: `outputs/bur_surge_results_fdr.csv`

**BUR Variation Analysis** - Measures within-phrase consistency:
```bash
poetry run python cli/bur_variation_cli.py
```
- Calculates standard deviation per phrase
- Allows sorting by highest/lowest variation
- Output: `outputs/phrase_bur_variation.csv`

**Histogram Visualization** - Creates BUR distribution plots:
```bash
poetry run python cli/bur_histogram_cli.py
```
- One histogram per performer
- Auto-adjusts bins to fit data
- Output: `outputs/bur_histograms/*.png`

## Statistical Methodology

### Surge Analysis
- **Model**: Simple linear regression (BUR vs. position)
- **Multiple Testing**: Benjamini-Hochberg FDR correction (α = 0.05)
- **Autocorrelation**: Durbin-Watson test to validate independence
- **Results**: Only 1/2,488 phrases (0.04%) show significant trends after correction

### Configuration
All statistical parameters are centralized in `utils/config.py`:
- `MIN_BUR_VALUES = 6` - Minimum phrase length
- `CONFIDENCE_LEVEL = 0.95` - For confidence intervals
- `FDR_ALPHA = 0.05` - False discovery rate threshold
- `DW_AUTOCORR_THRESHOLD = 1.5` - Durbin-Watson cutoff

## Data

The filtered dataset (`data/phrasebur_filtered.csv`) contains 26,801 rows from 2,488 phrases across 456 solos by 76 performers. See `data/README.md` for details on filtering criteria and phrase length distributions.

## Citations

- Benadon, F. (2006). Slicing the Beat: Jazz Eighth-Notes as Expressive Microrhythm. *Ethnomusicology*, *50*(1), 73-98. [https://doi.org/10.2307/20174424](https://doi.org/10.2307/20174424).
- Corcoran, C., & Frieler, K. (2021). Playing it Straight: Analyzing Jazz Soloists' Swing Eighth-note Distributions with the Weimar Jazz Database. *Music Perception: An Interdisciplinary Journal*, *38*(4), 372–385. [https://doi.org/10.1525/mp.2021.38.4.372](https://doi.org/10.1525/mp.2021.38.4.372).
