"""
Multi-Year Sentiment Analysis Comparison: FinBERT vs. LM Dictionary
Author: Analysis Framework
Date: 2025
Purpose: Compare two sentiment analysis methodologies across multiple years

IMPROVEMENTS IN THIS VERSION:
1. Robust CIK handling with strict numeric coercion
2. Explicit column validation with helpful error messages
3. Specific warning filters instead of global suppression
4. Consistent standardization approach throughout
5. Adaptive thresholds using percentiles/standard deviations
6. NaN and zero-variance guards
7. Logging instead of print statements
8. Comprehensive error handling
9. Kendall correlation included in correlation_trends.png visualization
10. All three correlation types (Pearson, Spearman, Kendall) in positive/negative trend graphs
11. Classification metrics (confusion matrix, accuracy) visualization added
12. Multiple normalization methods: z-score, per-word, per-sentence
13. FinBERT sentence counts used for LM per-sentence normalization (true comparability)
14. Normalization method comparison visualization for each year
15. Configurable primary normalization method selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import warnings
import os
import re
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Specific warning filters (instead of global suppression)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)


# Import normalization module
from sentiment_normalization import (
    AnalysisConfig as NormConfig,
    create_composite_scores,
    add_categorical_sentiment,
    create_per_word_metrics,
    create_per_sentence_metrics,
    standardize_all_metrics,
    get_primary_metrics
)

# ============================================================================
# CONFIGURATION AND VALIDATION
# ============================================================================

class AnalysisConfig:
    """Configuration class for analysis parameters"""

    # Required columns for each dataset
    REQUIRED_FINBERT_COLS = {
        'cik', 'finbert_avg_positive', 'finbert_avg_negative',
        'finbert_avg_neutral', 'finbert_dominant_sentiment'
    }

    REQUIRED_LM_COLS = {
        'cik', 'net_sentiment', 'positive_proportion', 'negative_proportion'
    }

    # Year range for validation
    YEAR_RANGE = (2018, 2025)

def validate_columns(df: pd.DataFrame, required_cols: set, dataset_name: str, year: int) -> None:
    """
    Validate that required columns exist in dataframe

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    required_cols : set
        Set of required column names
    dataset_name : str
        Name of dataset for error messages
    year : int
        Year being processed

    Raises:
    -------
    ValueError : If required columns are missing
    """
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        error_msg = (
            f"\n{'='*80}\n"
            f"SCHEMA ERROR: {dataset_name} data for year {year}\n"
            f"{'='*80}\n"
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {set(df.columns)}\n"
            f"{'='*80}\n"
            f"This suggests the CSV schema has changed.\n"
            f"Please verify the input file structure matches expected format."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Column validation passed for {dataset_name} {year}")


def clean_cik_robust(cik_series: pd.Series, dataset_name: str) -> pd.Series:
    """
    Robustly convert CIK to integer with proper error handling

    Parameters:
    -----------
    cik_series : pd.Series
        Series containing CIK values (potentially as strings with parentheses)
    dataset_name : str
        Name of dataset for logging

    Returns:
    --------
    pd.Series : Cleaned CIK values as integers (NaN for invalid entries)
    """
    def parse_cik(value):
        """Parse a single CIK value"""
        if pd.isna(value):
            return np.nan

        # Convert to string and clean
        value_str = str(value).strip()

        # Remove parentheses and spaces
        value_str = value_str.strip('()').strip()

        # Remove leading zeros
        value_str = value_str.lstrip('0')

        # Handle empty string after stripping
        if value_str == '' or value_str.lower() in ['nan', 'none', '']:
            return np.nan

        # Try to convert to integer
        try:
            return int(float(value_str))  # float() handles "123.0" format
        except (ValueError, TypeError):
            return np.nan

    # Apply parsing
    cleaned = cik_series.apply(parse_cik)

    # Log statistics
    n_valid = cleaned.notna().sum()
    n_invalid = cleaned.isna().sum()
    n_total = len(cleaned)

    logger.info(
        f"{dataset_name} CIK cleaning: {n_valid}/{n_total} valid "
        f"({n_valid/n_total*100:.1f}%), {n_invalid} invalid"
    )

    if n_invalid > 0:
        logger.warning(
            f"{dataset_name}: {n_invalid} invalid CIK values will be excluded from merge"
        )

    return cleaned


# ============================================================================
# PART 1: FILE DISCOVERY AND MATCHING
# ============================================================================

def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Extract year from filename

    Looks for 4-digit year pattern (2000-2099) and validates against configured range.
    Works with various filename formats including:
    - "sentiment_2020.csv" (LM format)
    - "finbert_(2021)_results.csv" (FinBERT format with parentheses)
    - "data_2022_final.csv"

    Parameters:
    -----------
    filename : str
        Filename to parse

    Returns:
    --------
    int or None : Year if found and within valid range, None otherwise
    """
    # Look for any 4-digit year starting with 20 (2000-2099)
    # This regex matches years like: 2000, 2020, 2025, etc.
    # Works regardless of surrounding characters (parentheses, underscores, etc.)
    pattern = r'(20\d{2})'

    match = re.search(pattern, filename)
    if match:
        year = int(match.group(1))

        # Validate against configured range
        min_year, max_year = AnalysisConfig.YEAR_RANGE
        if min_year <= year <= max_year:
            return year
        else:
            # Year found but outside valid range
            logger.debug(
                f"Year {year} found in filename '{filename}' "
                f"but outside valid range {AnalysisConfig.YEAR_RANGE}"
            )
            return None

    # No year pattern found
    logger.debug(f"No year pattern found in filename: '{filename}'")
    return None


def discover_files(bert_dir: str, lm_dir: str) -> Dict[int, Dict[str, str]]:
    """
    Discover and match FinBERT and LM files by year

    Parameters:
    -----------
    bert_dir : str
        Directory containing FinBERT CSV files
    lm_dir : str
        Directory containing LM CSV files

    Returns:
    --------
    dict : Dictionary mapping year -> {'bert': path, 'lm': path}
    """
    logger.info("\n" + "="*80)
    logger.info("FILE DISCOVERY")
    logger.info("="*80 + "\n")

    # Find FinBERT files
    logger.info(f"Scanning FinBERT directory: {bert_dir}")
    bert_files = {}
    try:
        for filename in os.listdir(bert_dir):
            if filename.endswith('.csv'):
                year = extract_year_from_filename(filename)
                if year:
                    bert_files[year] = os.path.join(bert_dir, filename)
                    logger.info(f"  Found FinBERT {year}: {filename}")
    except FileNotFoundError:
        logger.error(f"❌ Directory not found: {bert_dir}")
        return {}

    # Find LM files
    logger.info(f"\nScanning LM directory: {lm_dir}")
    lm_files = {}
    try:
        for filename in os.listdir(lm_dir):
            if filename.endswith('.csv'):
                year = extract_year_from_filename(filename)
                if year:
                    lm_files[year] = os.path.join(lm_dir, filename)
                    logger.info(f"  Found LM {year}: {filename}")
    except FileNotFoundError:
        logger.error(f"❌ Directory not found: {lm_dir}")
        return {}

    # Match files by year
    matched_years = set(bert_files.keys()) & set(lm_files.keys())

    if not matched_years:
        logger.error("\n❌ ERROR: No matching years found between FinBERT and LM files")
        logger.error(f"FinBERT years: {sorted(bert_files.keys())}")
        logger.error(f"LM years: {sorted(lm_files.keys())}")
        return {}

    matched_files = {}
    for year in sorted(matched_years):
        matched_files[year] = {
            'bert': bert_files[year],
            'lm': lm_files[year]
        }

    logger.info(f"\n✓ Matched {len(matched_files)} year(s): {sorted(matched_years)}")

    # Report unmatched files
    unmatched_bert = set(bert_files.keys()) - matched_years
    unmatched_lm = set(lm_files.keys()) - matched_years

    if unmatched_bert:
        logger.warning(f"⚠ FinBERT files without LM match: {sorted(unmatched_bert)}")
    if unmatched_lm:
        logger.warning(f"⚠ LM files without FinBERT match: {sorted(unmatched_lm)}")

    return matched_files


# ============================================================================
# PART 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_merge_year(bert_path: str, lm_path: str, year: int) -> pd.DataFrame:
    """
    Load and merge FinBERT and LM data for a single year

    Parameters:
    -----------
    bert_path : str
        Path to FinBERT CSV file
    lm_path : str
        Path to LM CSV file
    year : int
        Year being processed

    Returns:
    --------
    pd.DataFrame : Merged dataframe with both sentiment measures
    """
    logger.info(f"\nLoading data for year {year}...")

    # Load FinBERT data
    try:
        df_bert = pd.read_csv(bert_path)
        logger.info(f"  FinBERT: {len(df_bert)} rows")

        # Validate schema
        validate_columns(df_bert, AnalysisConfig.REQUIRED_FINBERT_COLS, 'FinBERT', year)

        # Clean CIK
        df_bert['cik'] = clean_cik_robust(df_bert['cik'], f'FinBERT {year}')

    except Exception as e:
        logger.error(f"Error loading FinBERT file: {e}")
        raise

    # Load LM data
    try:
        df_lm = pd.read_csv(lm_path)
        logger.info(f"  LM: {len(df_lm)} rows")

        # Validate schema
        validate_columns(df_lm, AnalysisConfig.REQUIRED_LM_COLS, 'LM', year)

        # Clean CIK
        df_lm['cik'] = clean_cik_robust(df_lm['cik'], f'LM {year}')

    except Exception as e:
        logger.error(f"Error loading LM file: {e}")
        raise

    # Merge on CIK
    # Remove rows with invalid CIKs before merge
    df_bert_clean = df_bert[df_bert['cik'].notna()].copy()
    df_lm_clean = df_lm[df_lm['cik'].notna()].copy()

    # Ensure CIK is integer type for merge
    df_bert_clean['cik'] = df_bert_clean['cik'].astype(int)
    df_lm_clean['cik'] = df_lm_clean['cik'].astype(int)

    df_merged = pd.merge(
        df_bert_clean,
        df_lm_clean,
        on='cik',
        how='inner',
        suffixes=('_bert', '_lm')
    )

    # Add year column for tracking
    df_merged['analysis_year'] = year

    logger.info(f"  Merged: {len(df_merged)} companies")

    if len(df_merged) == 0:
        logger.warning(
            f"⚠ WARNING: No companies matched between FinBERT and LM for year {year}. "
            f"This might indicate CIK format mismatch or no overlapping companies."
        )

    # Log merge statistics
    n_bert_only = len(df_bert_clean) - len(df_merged)
    n_lm_only = len(df_lm_clean) - len(df_merged)

    if n_bert_only > 0:
        logger.info(f"  FinBERT-only companies: {n_bert_only}")
    if n_lm_only > 0:
        logger.info(f"  LM-only companies: {n_lm_only}")

    return df_merged


def filter_by_companies(df: pd.DataFrame, company_list: List[str], year: int) -> pd.DataFrame:
    """
    Filter dataframe to only include specified companies

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to filter
    company_list : list
        List of company names to include
    year : int
        Year being processed (for logging)

    Returns:
    --------
    pd.DataFrame : Filtered dataframe
    """
    # Try to find company name column
    company_cols = [col for col in df.columns if 'company' in col.lower() or 'name' in col.lower()]

    if not company_cols:
        logger.warning(
            f"No company name column found for year {year}. "
            f"Available columns: {df.columns.tolist()}"
        )
        return df

    company_col = company_cols[0]
    logger.info(f"  Using column '{company_col}' for company filtering")

    # Create case-insensitive matching
    company_list_lower = [c.lower() for c in company_list]

    mask = df[company_col].str.lower().isin(company_list_lower)
    df_filtered = df[mask].copy()

    logger.info(f"  Filtered to {len(df_filtered)} companies from original {len(df)}")

    # Report which companies were found
    found_companies = df_filtered[company_col].unique()
    logger.info(f"  Found companies: {sorted(found_companies)}")

    # Report which companies were NOT found
    missing = set(company_list) - set(found_companies)
    if missing:
        logger.warning(f"  Companies not found in {year}: {sorted(missing)}")

    return df_filtered


# ============================================================================
# PART 2.5: COMPOSITE SCORES AND CATEGORIZATION
# ============================================================================



# ============================================================================
# PART 3: ANALYSIS FOR SINGLE YEAR
# ============================================================================

def safe_correlation(x: pd.Series, y: pd.Series,
                     method: str = 'pearson') -> Tuple[Union[float, np.floating], Union[float, np.floating]]:
    """
    Safely compute correlation with guards for edge cases

    Parameters:
    -----------
    x, y : pd.Series
        Data series to correlate
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'

    Returns:
    --------
    tuple : (correlation coefficient, p-value) or (NaN, NaN) if cannot compute
    """
    # Remove NaN values
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]

    # Check sample size
    if len(x_clean) < 3:
        logger.warning(f"Insufficient data for correlation (n={len(x_clean)})")
        return (np.nan, np.nan)

    # Check for zero variance
    if x_clean.std() == 0 or y_clean.std() == 0:
        logger.warning("Zero variance in one or both variables, cannot compute correlation")
        return (np.nan, np.nan)

    try:
        if method == 'pearson':
            r, p = pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            r, p = spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            r, p = kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return (r, p)

    except Exception as e:
        logger.warning(f"Error computing {method} correlation: {e}")
        return (np.nan, np.nan)


def analyze_year(df: pd.DataFrame, year: int, output_dir: str) -> Dict:
    """
    Run complete analysis for a single year

    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset for this year
    year : int
        Year being analyzed
    output_dir : str
        Directory for outputs

    Returns:
    --------
    dict : Dictionary of results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING YEAR {year}")
    logger.info(f"{'='*80}")

    results = {
        'year': year,
        'n_companies': len(df)
    }

    # Determine which LM metric to use consistently
    lm_net_metric = 'lm_net_proportion' if NormConfig.LM_STANDARDIZATION == 'proportion' else 'net_sentiment'

    # Calculate correlations (including Kendall now)
    comparisons = [
        ('finbert_net_sentiment', lm_net_metric, 'net'),
        ('finbert_avg_positive', 'positive_proportion', 'positive'),
        ('finbert_avg_negative', 'negative_proportion', 'negative'),
    ]

    for var1, var2, label in comparisons:
        if var1 in df.columns and var2 in df.columns:
            pearson_r, pearson_p = safe_correlation(df[var1], df[var2], 'pearson')
            spearman_r, spearman_p = safe_correlation(df[var1], df[var2], 'spearman')
            kendall_r, kendall_p = safe_correlation(df[var1], df[var2], 'kendall')

            results[f'{label}_pearson_r'] = pearson_r
            results[f'{label}_pearson_p'] = pearson_p
            results[f'{label}_spearman_r'] = spearman_r
            results[f'{label}_spearman_p'] = spearman_p
            results[f'{label}_kendall_r'] = kendall_r
            results[f'{label}_kendall_p'] = kendall_p

            if not np.isnan(pearson_r):
                logger.info(f"{label}: Pearson r = {pearson_r:.3f} (p = {pearson_p:.4e})")
            if not np.isnan(spearman_r):
                logger.info(f"{label}: Spearman r = {spearman_r:.3f} (p = {spearman_p:.4e})")
            if not np.isnan(kendall_r):
                logger.info(f"{label}: Kendall τ = {kendall_r:.3f} (p = {kendall_p:.4e})")

    # Classification agreement (if enough variation)
    try:
        if df['finbert_category'].nunique() > 1 and df['lm_category'].nunique() > 1:
            kappa = cohen_kappa_score(df['finbert_category'], df['lm_category'])
            results['kappa'] = kappa
            logger.info(f"Cohen's Kappa: {kappa:.3f}")

            # Calculate accuracy
            accuracy = accuracy_score(df['finbert_category'], df['lm_category'])
            results['accuracy'] = accuracy
            logger.info(f"Classification Accuracy: {accuracy:.3f}")

            # Also compute confusion matrix for reporting
            cm = confusion_matrix(
                df['finbert_category'],
                df['lm_category'],
                labels=['positive', 'neutral', 'negative']
            )
            logger.debug(f"Confusion matrix:\n{cm}")

            # Store confusion matrix in results
            results['confusion_matrix'] = cm
        else:
            results['kappa'] = np.nan
            results['accuracy'] = np.nan
            results['confusion_matrix'] = None
            logger.info("Cohen's Kappa & Accuracy: N/A (insufficient variation)")
    except Exception as e:
        results['kappa'] = np.nan
        results['accuracy'] = np.nan
        results['confusion_matrix'] = None
        logger.warning(f"Could not compute classification metrics: {e}")

    # Save year-specific visualization
    plot_year_summary(df, year, output_dir)

    # Compare normalization methods
    comparison_results = compare_normalization_methods(df, year, output_dir)
    plot_normalization_comparison(comparison_results, year, output_dir)
    results['normalization_comparison'] = comparison_results

    return results


def plot_year_summary(df: pd.DataFrame, year: int, output_dir: str) -> None:
    """Create summary visualization for a single year"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Scatter plot with safe handling
    ax = axes[0, 0]
    x = df['finbert_net_std']
    y = df['lm_net_std']

    # Filter out NaN values
    mask = x.notna() & y.notna()

    if mask.sum() > 0:
        ax.scatter(x[mask], y[mask], alpha=0.5, s=40, edgecolors='black', linewidth=0.5)

        if mask.sum() > 1:
            r, p_val = safe_correlation(x, y, 'pearson')

            if not np.isnan(r):
                # Regression line
                try:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2)

                    ax.text(0.05, 0.95, f'r = {r:.3f}\np < {p_val:.2e}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception as e:
                    logger.debug(f"Could not fit regression line: {e}")
            else:
                ax.text(0.05, 0.95, 'r = N/A',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('FinBERT Net Sentiment (standardized)')
    ax.set_ylabel('LM Net Sentiment (standardized)')
    ax.set_title(f'Sentiment Comparison - {year}')
    ax.grid(True, alpha=0.3)

    # Plot 2: Positive sentiment
    ax = axes[0, 1]
    mask = df[['finbert_avg_positive', 'positive_proportion']].notna().all(axis=1)

    if mask.sum() > 0:
        ax.scatter(df.loc[mask, 'finbert_avg_positive'],
                   df.loc[mask, 'positive_proportion'],
                   alpha=0.5, s=40, color='green', edgecolors='black', linewidth=0.5)

        if mask.sum() > 1:
            r, _ = safe_correlation(
                df['finbert_avg_positive'],
                df['positive_proportion'],
                'pearson'
            )

            if not np.isnan(r):
                ax.text(0.05, 0.95, f'r = {r:.3f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('FinBERT Positive')
    ax.set_ylabel('LM Positive Proportion')
    ax.set_title('Positive Sentiment')
    ax.grid(True, alpha=0.3)

    # Plot 3: Negative sentiment
    ax = axes[1, 0]
    mask = df[['finbert_avg_negative', 'negative_proportion']].notna().all(axis=1)

    if mask.sum() > 0:
        ax.scatter(df.loc[mask, 'finbert_avg_negative'],
                   df.loc[mask, 'negative_proportion'],
                   alpha=0.5, s=40, color='red', edgecolors='black', linewidth=0.5)

        if mask.sum() > 1:
            r, _ = safe_correlation(
                df['finbert_avg_negative'],
                df['negative_proportion'],
                'pearson'
            )

            if not np.isnan(r):
                ax.text(0.05, 0.95, f'r = {r:.3f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('FinBERT Negative')
    ax.set_ylabel('LM Negative Proportion')
    ax.set_title('Negative Sentiment')
    ax.grid(True, alpha=0.3)

    # Plot 4: Distribution comparison
    ax = axes[1, 1]

    # Only plot if we have data
    finbert_valid = df['finbert_net_std'].dropna()
    lm_valid = df['lm_net_std'].dropna()

    if len(finbert_valid) > 0:
        ax.hist(finbert_valid, bins=30, alpha=0.5, label='FinBERT',
               color='blue', edgecolor='black')

    if len(lm_valid) > 0:
        ax.hist(lm_valid, bins=30, alpha=0.5, label='LM',
               color='orange', edgecolor='black')

    ax.set_xlabel('Standardized Net Sentiment')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Sentiment Analysis Comparison: {year}\n(N = {len(df)} companies)',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'year_{year}_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# PART 4: CROSS-YEAR ANALYSIS
# ============================================================================

def analyze_across_years(all_data: Dict[int, pd.DataFrame],
                         results_by_year: List[Dict],
                         output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze trends and patterns across multiple years

    Parameters:
    -----------
    all_data : dict
        Dictionary mapping year -> DataFrame
    results_by_year : list
        List of results dictionaries
    output_dir : str
        Directory for outputs

    Returns:
    --------
    tuple : (results_df, df_all, company_summary_df)
    """
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-YEAR ANALYSIS")
    logger.info(f"{'='*80}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_by_year)
    results_df = results_df.sort_values('year')

    logger.info("\nCorrelations Over Time:")
    # Only show columns that exist
    cols_to_show = ['year', 'n_companies']
    for col in ['net_pearson_r', 'net_spearman_r', 'net_kendall_r',
                'positive_pearson_r', 'negative_pearson_r', 'kappa', 'accuracy']:
        if col in results_df.columns:
            cols_to_show.append(col)

    logger.info("\n" + results_df[cols_to_show].to_string(index=False))

    # Save results table
    results_df.to_csv(os.path.join(output_dir, 'correlations_by_year.csv'), index=False)

    # Track company appearances
    company_summary_df = track_company_appearances(all_data)
    missing_companies_df = create_company_tracking_report(company_summary_df, output_dir)
    plot_company_coverage(company_summary_df, all_data, output_dir)

    # Create visualizations
    plot_correlation_trends(results_df, output_dir)
    plot_sentiment_trends(all_data, output_dir)
    plot_classification_metrics(results_df, results_by_year, output_dir)  # NEW

    # Combine all years for aggregate analysis
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATE ANALYSIS (ALL YEARS COMBINED)")
    logger.info(f"{'='*80}")

    df_all = pd.concat(all_data.values(), ignore_index=True)
    logger.info(f"\nTotal companies across all years: {len(df_all)}")
    logger.info(f"Years included: {sorted(df_all['analysis_year'].unique())}")

    # Overall correlations with safe handling
    lm_net_metric = 'lm_net_proportion' if NormConfig.LM_STANDARDIZATION == 'proportion' else 'net_sentiment'

    r, p = safe_correlation(
        df_all['finbert_net_sentiment'],
        df_all[lm_net_metric],
        'pearson'
    )

    if not np.isnan(r):
        mask = df_all[['finbert_net_sentiment', lm_net_metric]].notna().all(axis=1)
        logger.info(f"\nOverall correlation (all years): r = {r:.3f} (p = {p:.4e}, N = {mask.sum()})")
    else:
        logger.info("\nOverall correlation: Could not compute")

    # Create aggregate visualizations
    plot_aggregate_analysis(df_all, output_dir)

    return results_df, df_all, company_summary_df


def compare_normalization_methods(df: pd.DataFrame, year: int, output_dir: str) -> Dict:
    """
    Compare correlations across different normalization methods

    Calculates Pearson, Spearman, and Kendall correlations for:
    - Z-score (original proportion-based)
    - Per-word normalization
    - Per-sentence normalization

    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset for this year
    year : int
        Year being analyzed
    output_dir : str
        Directory for outputs

    Returns:
    --------
    dict : Dictionary of correlation results by method
    """
    logger.info(f"\n--- Comparing Normalization Methods for {year} ---")

    normalization_methods = {
        'Z-Score (Proportion)': ('finbert_net_std', 'lm_net_std'),
        'Per-Word': ('finbert_net_per_word_std', 'lm_net_per_word_std'),
        'Per-Sentence': ('finbert_net_per_sentence_std', 'lm_net_per_sentence_std')
    }

    comparison_results = {}

    for method_name, (fb_col, lm_col) in normalization_methods.items():
        if fb_col in df.columns and lm_col in df.columns:
            pearson_r, pearson_p = safe_correlation(df[fb_col], df[lm_col], 'pearson')
            spearman_r, spearman_p = safe_correlation(df[fb_col], df[lm_col], 'spearman')
            kendall_r, kendall_p = safe_correlation(df[fb_col], df[lm_col], 'kendall')

            comparison_results[method_name] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_r': kendall_r,
                'kendall_p': kendall_p
            }

            logger.info(f"\n{method_name}:")
            if not np.isnan(pearson_r):
                logger.info(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4e})")
            if not np.isnan(spearman_r):
                logger.info(f"  Spearman r = {spearman_r:.3f} (p = {spearman_p:.4e})")
            if not np.isnan(kendall_r):
                logger.info(f"  Kendall τ = {kendall_r:.3f} (p = {kendall_p:.4e})")
        else:
            logger.warning(f"Columns for {method_name} not found: {fb_col}, {lm_col}")
            comparison_results[method_name] = {
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_r': np.nan,
                'spearman_p': np.nan,
                'kendall_r': np.nan,
                'kendall_p': np.nan
            }

    return comparison_results


def plot_normalization_comparison(comparison_results: Dict, year: int, output_dir: str) -> None:
    """
    Create visualization comparing different normalization methods

    Parameters:
    -----------
    comparison_results : dict
        Results from compare_normalization_methods()
    year : int
        Year being analyzed
    output_dir : str
        Directory for outputs
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    methods = list(comparison_results.keys())

    # Extract correlation coefficients for each method
    pearson_vals = [comparison_results[m]['pearson_r'] for m in methods]
    spearman_vals = [comparison_results[m]['spearman_r'] for m in methods]
    kendall_vals = [comparison_results[m]['kendall_r'] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    # Plot 1: Pearson correlations
    ax = axes[0]
    bars = ax.bar(x, pearson_vals, width, label='Pearson',
                   color='#1f77b4', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Normalization Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax.set_title(f'Pearson Correlation - {year}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-1, 1])

    # Add value labels on bars
    for bar, val in zip(bars, pearson_vals):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

    # Plot 2: Spearman correlations
    ax = axes[1]
    bars = ax.bar(x, spearman_vals, width, label='Spearman',
                   color='#ff7f0e', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Normalization Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax.set_title(f'Spearman Correlation - {year}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-1, 1])

    # Add value labels
    for bar, val in zip(bars, spearman_vals):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

    # Plot 3: Kendall correlations
    ax = axes[2]
    bars = ax.bar(x, kendall_vals, width, label='Kendall',
                   color='#2ca02c', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Normalization Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax.set_title(f'Kendall Correlation - {year}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-1, 1])

    # Add value labels
    for bar, val in zip(bars, kendall_vals):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

    plt.suptitle(f'Normalization Method Comparison - {year}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'normalization_comparison_{year}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved normalization comparison: {output_path}")
    plt.close()


def plot_correlation_trends(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot how correlations change over time - IMPROVED VERSION"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    years = results_df['year']

    # Plot 1: Net sentiment correlation - ALL THREE METHODS
    ax = axes[0, 0]

    if 'net_pearson_r' in results_df.columns:
        valid_mask = results_df['net_pearson_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'net_pearson_r'],
               marker='o', linewidth=2.5, markersize=8, label='Pearson', color='#1f77b4')

    if 'net_spearman_r' in results_df.columns:
        valid_mask = results_df['net_spearman_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'net_spearman_r'],
               marker='s', linewidth=2.5, markersize=8, label='Spearman', color='#ff7f0e')

    if 'net_kendall_r' in results_df.columns:
        valid_mask = results_df['net_kendall_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'net_kendall_r'],
               marker='^', linewidth=2.5, markersize=8, label='Kendall', color='#2ca02c')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Net Sentiment Correlation Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Positive sentiment correlation - ALL THREE METHODS
    ax = axes[0, 1]

    if 'positive_pearson_r' in results_df.columns:
        valid_mask = results_df['positive_pearson_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'positive_pearson_r'],
               marker='o', linewidth=2.5, markersize=8, label='Pearson', color='#1f77b4')

    if 'positive_spearman_r' in results_df.columns:
        valid_mask = results_df['positive_spearman_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'positive_spearman_r'],
               marker='s', linewidth=2.5, markersize=8, label='Spearman', color='#ff7f0e')

    if 'positive_kendall_r' in results_df.columns:
        valid_mask = results_df['positive_kendall_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'positive_kendall_r'],
               marker='^', linewidth=2.5, markersize=8, label='Kendall', color='#2ca02c')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Positive Sentiment Correlation Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Negative sentiment correlation - ALL THREE METHODS
    ax = axes[1, 0]

    if 'negative_pearson_r' in results_df.columns:
        valid_mask = results_df['negative_pearson_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'negative_pearson_r'],
               marker='o', linewidth=2.5, markersize=8, label='Pearson', color='#1f77b4')

    if 'negative_spearman_r' in results_df.columns:
        valid_mask = results_df['negative_spearman_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'negative_spearman_r'],
               marker='s', linewidth=2.5, markersize=8, label='Spearman', color='#ff7f0e')

    if 'negative_kendall_r' in results_df.columns:
        valid_mask = results_df['negative_kendall_r'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'negative_kendall_r'],
               marker='^', linewidth=2.5, markersize=8, label='Kendall', color='#2ca02c')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Negative Sentiment Correlation Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Sample sizes
    ax = axes[1, 1]
    ax.bar(years, results_df['n_companies'], color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size by Year', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Correlation Trends Across Years', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nSaved: {output_path}")
    plt.close()


def plot_classification_metrics(results_df: pd.DataFrame, results_by_year: List[Dict], output_dir: str) -> None:
    """Plot classification metrics over time - NEW FUNCTION"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    years = results_df['year']

    # Plot 1: Cohen's Kappa over time
    ax = axes[0, 0]
    if 'kappa' in results_df.columns:
        valid_mask = results_df['kappa'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'kappa'],
               marker='o', linewidth=2.5, markersize=10, color='purple', label="Cohen's Kappa")
        ax.fill_between(years[valid_mask], 0, results_df.loc[valid_mask, 'kappa'],
                       alpha=0.3, color='purple')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel("Cohen's Kappa", fontsize=12, fontweight='bold')
    ax.set_title("Classification Agreement (Cohen's Kappa) Over Time", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([-1, 1])

    # Plot 2: Accuracy over time
    ax = axes[0, 1]
    if 'accuracy' in results_df.columns:
        valid_mask = results_df['accuracy'].notna()
        ax.plot(years[valid_mask], results_df.loc[valid_mask, 'accuracy'],
               marker='s', linewidth=2.5, markersize=10, color='darkgreen', label='Accuracy')
        ax.fill_between(years[valid_mask], 0, results_df.loc[valid_mask, 'accuracy'],
                       alpha=0.3, color='darkgreen')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Confusion matrix for most recent year with data
    ax = axes[1, 0]

    # Find most recent year with confusion matrix
    recent_cm = None
    recent_year = None
    for result in reversed(results_by_year):
        if result.get('confusion_matrix') is not None:
            recent_cm = result['confusion_matrix']
            recent_year = result['year']
            break

    if recent_cm is not None:
        im = ax.imshow(recent_cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Positive', 'Neutral', 'Negative'])
        ax.set_yticklabels(['Positive', 'Neutral', 'Negative'])
        ax.set_xlabel('LM Prediction', fontsize=12, fontweight='bold')
        ax.set_ylabel('FinBERT Prediction', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix ({recent_year})', fontsize=14, fontweight='bold')

        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, recent_cm[i, j],
                             ha="center", va="center", color="white" if recent_cm[i, j] > recent_cm.max()/2 else "black",
                             fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Count')
    else:
        ax.text(0.5, 0.5, 'No confusion matrix data available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Plot 4: Combined metrics comparison
    ax = axes[1, 1]

    if 'kappa' in results_df.columns and 'accuracy' in results_df.columns:
        kappa_mask = results_df['kappa'].notna()
        acc_mask = results_df['accuracy'].notna()

        ax.plot(years[kappa_mask], results_df.loc[kappa_mask, 'kappa'],
               marker='o', linewidth=2.5, markersize=8, label="Cohen's Kappa", color='purple')
        ax.plot(years[acc_mask], results_df.loc[acc_mask, 'accuracy'],
               marker='s', linewidth=2.5, markersize=8, label='Accuracy', color='darkgreen')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Classification Metrics Over Time', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'classification_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_sentiment_trends(all_data: Dict[int, pd.DataFrame], output_dir: str) -> None:
    """Plot average sentiment over time for both methods"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Prepare data
    yearly_stats = []
    for year, df in sorted(all_data.items()):
        yearly_stats.append({
            'year': year,
            'finbert_pos_mean': df['finbert_avg_positive'].mean(),
            'finbert_neg_mean': df['finbert_avg_negative'].mean(),
            'finbert_net_mean': df['finbert_net_sentiment'].mean(),
            'lm_pos_mean': df['positive_proportion'].mean(),
            'lm_neg_mean': df['negative_proportion'].mean(),
            'lm_net_mean': df['lm_net_proportion'].mean(),
        })

    stats_df = pd.DataFrame(yearly_stats)
    years = stats_df['year']

    # Plot 1: FinBERT trends
    ax = axes[0, 0]
    ax.plot(years, stats_df['finbert_pos_mean'],
           marker='o', linewidth=2, label='Positive', color='green')
    ax.plot(years, stats_df['finbert_neg_mean'],
           marker='s', linewidth=2, label='Negative', color='red')
    ax.plot(years, stats_df['finbert_net_mean'],
           marker='^', linewidth=2, label='Net', color='blue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Sentiment Probability')
    ax.set_title('FinBERT: Average Sentiment Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: LM trends
    ax = axes[0, 1]
    ax.plot(years, stats_df['lm_pos_mean'],
           marker='o', linewidth=2, label='Positive', color='green')
    ax.plot(years, stats_df['lm_neg_mean'],
           marker='s', linewidth=2, label='Negative', color='red')
    ax.plot(years, stats_df['lm_net_mean'],
           marker='^', linewidth=2, label='Net', color='blue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Sentiment Proportion')
    ax.set_title('LM Dictionary: Average Sentiment Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Net sentiment comparison
    ax = axes[1, 0]
    ax.plot(years, stats_df['finbert_net_mean'],
           marker='o', linewidth=2, markersize=8, label='FinBERT')
    ax.plot(years, stats_df['lm_net_mean'],
           marker='s', linewidth=2, markersize=8, label='LM')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Net Sentiment')
    ax.set_title('Net Sentiment Comparison Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Difference between methods
    ax = axes[1, 1]

    # Standardize for comparison with guards
    finbert_mean = stats_df['finbert_net_mean'].mean()
    finbert_std = stats_df['finbert_net_mean'].std()
    lm_mean = stats_df['lm_net_mean'].mean()
    lm_std = stats_df['lm_net_mean'].std()

    if finbert_std > 0 and lm_std > 0:
        finbert_std_vals = (stats_df['finbert_net_mean'] - finbert_mean) / finbert_std
        lm_std_vals = (stats_df['lm_net_mean'] - lm_mean) / lm_std
        difference = finbert_std_vals - lm_std_vals

        ax.bar(years, difference, color='purple', edgecolor='black', alpha=0.7)
    else:
        logger.warning("Cannot compute method disagreement: zero variance")

    ax.set_xlabel('Year')
    ax.set_ylabel('Difference (FinBERT - LM, standardized)')
    ax.set_title('Method Disagreement Over Time')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sentiment_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_aggregate_analysis(df_all: pd.DataFrame, output_dir: str) -> None:
    """Create comprehensive aggregate analysis across all years"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Overall scatter
    ax = axes[0, 0]
    x = df_all['finbert_net_std']
    y = df_all['lm_net_std']

    mask = x.notna() & y.notna()

    if mask.sum() > 0:
        ax.scatter(x[mask], y[mask], alpha=0.3, s=20, edgecolors='none')

        if mask.sum() > 1:
            r, p_val = safe_correlation(x, y, 'pearson')

            if not np.isnan(r):
                try:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2)

                    ax.text(0.05, 0.95, f'All Years\nr = {r:.3f}\nN = {mask.sum()}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception as e:
                    logger.debug(f"Could not fit regression line: {e}")

    ax.set_xlabel('FinBERT Net Sentiment')
    ax.set_ylabel('LM Net Sentiment')
    ax.set_title('Aggregate Correlation')
    ax.grid(True, alpha=0.3)

    # Plot 2: By year colored scatter
    ax = axes[0, 1]
    years = sorted(df_all['analysis_year'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

    for year, color in zip(years, colors):
        year_data = df_all[df_all['analysis_year'] == year]
        mask = year_data[['finbert_net_std', 'lm_net_std']].notna().all(axis=1)

        if mask.sum() > 0:
            ax.scatter(year_data.loc[mask, 'finbert_net_std'],
                       year_data.loc[mask, 'lm_net_std'],
                       alpha=0.5, s=20, color=color, label=str(year), edgecolors='none')

    ax.set_xlabel('FinBERT Net Sentiment')
    ax.set_ylabel('LM Net Sentiment')
    ax.set_title('Correlation by Year')
    ax.legend(title='Year', loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution by year
    ax = axes[0, 2]
    data_to_plot = []
    valid_years = []

    for year in years:
        year_data = df_all[df_all['analysis_year'] == year]['finbert_net_std'].dropna()
        if len(year_data) > 0:
            data_to_plot.append(year_data)
            valid_years.append(year)

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=[str(y) for y in valid_years], patch_artist=True)

        # Color boxes
        valid_colors = [colors[years.index(y)] for y in valid_years]
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)

    ax.set_xlabel('Year')
    ax.set_ylabel('FinBERT Net Sentiment')
    ax.set_title('FinBERT Distribution by Year')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: LM distribution by year
    ax = axes[1, 0]
    data_to_plot = []
    valid_years = []

    for year in years:
        year_data = df_all[df_all['analysis_year'] == year]['lm_net_std'].dropna()
        if len(year_data) > 0:
            data_to_plot.append(year_data)
            valid_years.append(year)

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=[str(y) for y in valid_years], patch_artist=True)

        # Color boxes
        valid_colors = [colors[years.index(y)] for y in valid_years]
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)

    ax.set_xlabel('Year')
    ax.set_ylabel('LM Net Sentiment')
    ax.set_title('LM Distribution by Year')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Correlation by year (bar chart)
    ax = axes[1, 1]
    year_corrs = []

    for year in years:
        year_data = df_all[df_all['analysis_year'] == year]
        r, _ = safe_correlation(
            year_data['finbert_net_std'],
            year_data['lm_net_std'],
            'pearson'
        )
        year_corrs.append(r)

    bars = ax.bar([str(y) for y in years], year_corrs,
                  color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Correlation Strength by Year')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Add values on bars
    for bar, corr in zip(bars, year_corrs):
        if not np.isnan(corr):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 6: Sample sizes
    ax = axes[1, 2]
    year_counts = [len(df_all[df_all['analysis_year'] == year]) for year in years]
    ax.bar([str(y) for y in years], year_counts,
          color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Companies')
    ax.set_title('Sample Size by Year')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (year, count) in enumerate(zip(years, year_counts)):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=9)

    plt.suptitle('Aggregate Analysis Across All Years', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'aggregate_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PART 5: COMPANY TRACKING
# ============================================================================

def track_company_appearances(all_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Track which companies appear in which years

    Parameters:
    -----------
    all_data : dict
        Dictionary mapping year -> DataFrame

    Returns:
    --------
    pd.DataFrame : Summary of company appearances across years
    """
    logger.info("\nTracking company appearances across years...")

    # Find company name column
    sample_df = next(iter(all_data.values()))
    company_cols = [col for col in sample_df.columns if 'company' in col.lower() or 'name' in col.lower()]

    if not company_cols:
        logger.warning("No company name column found, using CIK only")
        company_col = 'cik'
    else:
        company_col = company_cols[0]
        logger.info(f"Using column '{company_col}' for company tracking")

    # Build tracking matrix
    years = sorted(all_data.keys())
    all_companies = set()

    for df in all_data.values():
        all_companies.update(df[company_col].unique())

    # Create summary dataframe
    summary_data = []
    for company in sorted(all_companies):
        row = {'company': company}
        for year in years:
            row[str(year)] = int(company in all_data[year][company_col].values)
        row['total_years'] = sum(row[str(y)] for y in years)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('total_years', ascending=False)

    logger.info(f"Tracked {len(summary_df)} unique companies across {len(years)} years")

    return summary_df


def create_company_tracking_report(company_summary_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Create detailed report of company tracking

    Parameters:
    -----------
    company_summary_df : pd.DataFrame
        Summary of company appearances
    output_dir : str
        Directory for outputs

    Returns:
    --------
    pd.DataFrame : Companies with gaps in their data
    """
    # Save full tracking table
    output_path = os.path.join(output_dir, 'company_tracking.csv')
    company_summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved company tracking: {output_path}")

    # Identify companies with gaps
    year_cols = [col for col in company_summary_df.columns if col.isdigit()]
    n_years = len(year_cols)

    # Companies that appear in some but not all years
    partial_companies = company_summary_df[
        (company_summary_df['total_years'] > 0) &
        (company_summary_df['total_years'] < n_years)
    ].copy()

    if len(partial_companies) > 0:
        logger.info(f"\n{len(partial_companies)} companies appear in some but not all years")

        # Save list of companies with gaps
        output_path = os.path.join(output_dir, 'companies_with_gaps.csv')
        partial_companies.to_csv(output_path, index=False)
        logger.info(f"Saved companies with gaps: {output_path}")

        # Report summary statistics
        logger.info(f"\nCompany coverage summary:")
        logger.info(f"  - Always present: {len(company_summary_df[company_summary_df['total_years'] == n_years])}")
        logger.info(f"  - Sometimes present: {len(partial_companies)}")
        logger.info(f"  - Total unique companies: {len(company_summary_df)}")
    else:
        logger.info("\nAll companies appear in all years (no gaps)")
        partial_companies = pd.DataFrame()

    return partial_companies


def plot_company_coverage(company_summary_df: pd.DataFrame,
                          all_data: Dict[int, pd.DataFrame],
                          output_dir: str) -> None:
    """
    Visualize company coverage across years

    Parameters:
    -----------
    company_summary_df : pd.DataFrame
        Summary of company appearances
    all_data : dict
        Dictionary mapping year -> DataFrame
    output_dir : str
        Directory for outputs
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    years = sorted(all_data.keys())
    year_cols = [str(y) for y in years]

    # Plot 1: Distribution of company appearances
    ax = axes[0]
    appearance_counts = company_summary_df['total_years'].value_counts().sort_index()
    ax.bar(appearance_counts.index, appearance_counts.values,
          color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Years Present')
    ax.set_ylabel('Number of Companies')
    ax.set_title('Distribution of Company Appearances')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Companies per year
    ax = axes[1]
    year_counts = [len(all_data[year]) for year in years]
    ax.plot(years, year_counts, marker='o', linewidth=2, markersize=10, color='darkgreen')
    ax.fill_between(years, year_counts, alpha=0.3, color='darkgreen')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Companies')
    ax.set_title('Number of Companies by Year')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'company_coverage.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PART 5.5: SUMMARY REPORT
# ============================================================================

def create_summary_report(results_df: pd.DataFrame,
                         df_all: pd.DataFrame,
                         company_summary_df: pd.DataFrame,
                         output_dir: str) -> None:
    """
    Create text summary report of analysis

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results by year
    df_all : pd.DataFrame
        All data combined
    company_summary_df : pd.DataFrame
        Company tracking summary
    output_dir : str
        Directory for outputs
    """
    output_path = os.path.join(output_dir, 'multi_year_summary.txt')

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("MULTI-YEAR SENTIMENT ANALYSIS COMPARISON\n")
            f.write("FinBERT vs. LM Dictionary\n")
            f.write("="*80 + "\n\n")

            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Threshold method: {NormConfig.THRESHOLD_METHOD}\n")
            f.write(f"LM standardization: {NormConfig.LM_STANDARDIZATION}\n")
            f.write(f"Year range: {AnalysisConfig.YEAR_RANGE}\n\n")
            f.flush()

            # Years analyzed
            f.write("YEARS ANALYZED\n")
            f.write("-" * 80 + "\n")
            years = sorted(results_df['year'].tolist())
            f.write(f"Years: {years}\n")
            f.write(f"Total years: {len(years)}\n\n")
            f.flush()

            # Sample sizes
            f.write("SAMPLE SIZES\n")
            f.write("-" * 80 + "\n")
            for _, row in results_df.iterrows():
                f.write(f"  {int(row['year'])}: {int(row['n_companies'])} companies\n")
            f.write(f"\nTotal observations: {len(df_all)}\n")
            f.write(f"Unique companies: {len(company_summary_df)}\n\n")
            f.flush()

            # Company coverage
            year_cols = [col for col in company_summary_df.columns if col.isdigit()]
            n_years = len(year_cols)
            always_present = len(company_summary_df[company_summary_df['total_years'] == n_years])
            sometimes_present = len(company_summary_df[
                (company_summary_df['total_years'] > 0) &
                (company_summary_df['total_years'] < n_years)
            ])

            f.write("COMPANY COVERAGE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Companies present in all {n_years} years: {always_present}\n")
            f.write(f"Companies present in some years: {sometimes_present}\n")
            f.write(f"Total unique companies: {len(company_summary_df)}\n\n")
            f.flush()

            # Correlation results
            f.write("CORRELATION RESULTS BY YEAR\n")
            f.write("="*80 + "\n")

            # Net sentiment correlations
            f.write("\nNet Sentiment Correlations:\n")
            f.write("-" * 80 + "\n")
            for _, row in results_df.iterrows():
                f.write(f"  Year {int(row['year'])}:\n")
                if 'net_pearson_r' in row and not pd.isna(row['net_pearson_r']):
                    f.write(f"    Pearson:  r = {row['net_pearson_r']:.3f}")
                    if 'net_pearson_p' in row and not pd.isna(row['net_pearson_p']):
                        f.write(f" (p = {row['net_pearson_p']:.4e})")
                    f.write("\n")
                if 'net_spearman_r' in row and not pd.isna(row['net_spearman_r']):
                    f.write(f"    Spearman: r = {row['net_spearman_r']:.3f}")
                    if 'net_spearman_p' in row and not pd.isna(row['net_spearman_p']):
                        f.write(f" (p = {row['net_spearman_p']:.4e})")
                    f.write("\n")
                if 'net_kendall_r' in row and not pd.isna(row['net_kendall_r']):
                    f.write(f"    Kendall:  τ = {row['net_kendall_r']:.3f}")
                    if 'net_kendall_p' in row and not pd.isna(row['net_kendall_p']):
                        f.write(f" (p = {row['net_kendall_p']:.4e})")
                    f.write("\n")
            f.flush()

            # Positive sentiment correlations
            f.write("\nPositive Sentiment Correlations:\n")
            f.write("-" * 80 + "\n")
            for _, row in results_df.iterrows():
                f.write(f"  Year {int(row['year'])}:\n")
                if 'positive_pearson_r' in row and not pd.isna(row['positive_pearson_r']):
                    f.write(f"    Pearson:  r = {row['positive_pearson_r']:.3f}")
                    if 'positive_pearson_p' in row and not pd.isna(row['positive_pearson_p']):
                        f.write(f" (p = {row['positive_pearson_p']:.4e})")
                    f.write("\n")
                if 'positive_spearman_r' in row and not pd.isna(row['positive_spearman_r']):
                    f.write(f"    Spearman: r = {row['positive_spearman_r']:.3f}")
                    if 'positive_spearman_p' in row and not pd.isna(row['positive_spearman_p']):
                        f.write(f" (p = {row['positive_spearman_p']:.4e})")
                    f.write("\n")
                if 'positive_kendall_r' in row and not pd.isna(row['positive_kendall_r']):
                    f.write(f"    Kendall:  τ = {row['positive_kendall_r']:.3f}")
                    if 'positive_kendall_p' in row and not pd.isna(row['positive_kendall_p']):
                        f.write(f" (p = {row['positive_kendall_p']:.4e})")
                    f.write("\n")
            f.flush()

            # Negative sentiment correlations
            f.write("\nNegative Sentiment Correlations:\n")
            f.write("-" * 80 + "\n")
            for _, row in results_df.iterrows():
                f.write(f"  Year {int(row['year'])}:\n")
                if 'negative_pearson_r' in row and not pd.isna(row['negative_pearson_r']):
                    f.write(f"    Pearson:  r = {row['negative_pearson_r']:.3f}")
                    if 'negative_pearson_p' in row and not pd.isna(row['negative_pearson_p']):
                        f.write(f" (p = {row['negative_pearson_p']:.4e})")
                    f.write("\n")
                if 'negative_spearman_r' in row and not pd.isna(row['negative_spearman_r']):
                    f.write(f"    Spearman: r = {row['negative_spearman_r']:.3f}")
                    if 'negative_spearman_p' in row and not pd.isna(row['negative_spearman_p']):
                        f.write(f" (p = {row['negative_spearman_p']:.4e})")
                    f.write("\n")
                if 'negative_kendall_r' in row and not pd.isna(row['negative_kendall_r']):
                    f.write(f"    Kendall:  τ = {row['negative_kendall_r']:.3f}")
                    if 'negative_kendall_p' in row and not pd.isna(row['negative_kendall_p']):
                        f.write(f" (p = {row['negative_kendall_p']:.4e})")
                    f.write("\n")
            f.flush()

            # Classification metrics
            f.write("\nCLASSIFICATION METRICS BY YEAR\n")
            f.write("="*80 + "\n")
            for _, row in results_df.iterrows():
                f.write(f"  Year {int(row['year'])}:\n")
                if 'kappa' in row and not pd.isna(row['kappa']):
                    f.write(f"    Cohen's Kappa: {row['kappa']:.3f}\n")
                if 'accuracy' in row and not pd.isna(row['accuracy']):
                    f.write(f"    Accuracy:      {row['accuracy']:.3f} ({row['accuracy']*100:.1f}%)\n")
            f.write("\n")
            f.flush()

            # Overall statistics
            f.write("OVERALL STATISTICS (ALL YEARS COMBINED)\n")
            f.write("="*80 + "\n")

            lm_net_metric = 'lm_net_proportion' if NormConfig.LM_STANDARDIZATION == 'proportion' else 'net_sentiment'
            r, p = safe_correlation(
                df_all['finbert_net_sentiment'],
                df_all[lm_net_metric],
                'pearson'
            )

            if not np.isnan(r):
                mask = df_all[['finbert_net_sentiment', lm_net_metric]].notna().all(axis=1)
                f.write(f"Overall Pearson correlation: r = {r:.3f} (p = {p:.4e}, N = {mask.sum()})\n")
            else:
                f.write("Overall Pearson correlation: Could not compute\n")

            # Average correlations across years
            f.write("\nAverage Metrics Across All Years:\n")
            f.write("-" * 80 + "\n")

            if 'net_pearson_r' in results_df.columns:
                avg_r = results_df['net_pearson_r'].mean()
                f.write(f"Average net Pearson correlation:   r = {avg_r:.3f}\n")

            if 'net_spearman_r' in results_df.columns:
                avg_spear = results_df['net_spearman_r'].mean()
                f.write(f"Average net Spearman correlation:  r = {avg_spear:.3f}\n")

            if 'net_kendall_r' in results_df.columns:
                avg_kend = results_df['net_kendall_r'].mean()
                f.write(f"Average net Kendall correlation:   τ = {avg_kend:.3f}\n")

            if 'kappa' in results_df.columns:
                avg_kappa = results_df['kappa'].mean()
                f.write(f"Average Cohen's Kappa:             κ = {avg_kappa:.3f}\n")

            if 'accuracy' in results_df.columns:
                avg_acc = results_df['accuracy'].mean()
                f.write(f"Average classification accuracy:     {avg_acc:.3f} ({avg_acc*100:.1f}%)\n")

            # Summary interpretation
            f.write("\n\nSUMMARY & INTERPRETATION\n")
            f.write("="*80 + "\n")

            if 'net_pearson_r' in results_df.columns:
                avg_r = results_df['net_pearson_r'].mean()
                if avg_r >= 0.7:
                    strength = "strong positive"
                elif avg_r >= 0.4:
                    strength = "moderate positive"
                elif avg_r >= 0.2:
                    strength = "weak positive"
                elif avg_r >= -0.2:
                    strength = "negligible"
                elif avg_r >= -0.4:
                    strength = "weak negative"
                else:
                    strength = "moderate to strong negative"

                f.write(f"The average correlation between FinBERT and LM methods is {avg_r:.3f},\n")
                f.write(f"indicating a {strength} relationship.\n\n")

            if 'accuracy' in results_df.columns:
                avg_acc = results_df['accuracy'].mean()
                f.write(f"The classification methods agree {avg_acc*100:.1f}% of the time on average,\n")
                if avg_acc >= 0.8:
                    f.write("suggesting high agreement between the two sentiment analysis approaches.\n")
                elif avg_acc >= 0.6:
                    f.write("suggesting moderate agreement between the two sentiment analysis approaches.\n")
                else:
                    f.write("suggesting limited agreement between the two sentiment analysis approaches.\n")

            # Footer
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
            f.flush()

        logger.info(f"\nSaved summary report: {output_path}")

    except Exception as e:
        logger.error(f"Error creating summary report: {e}", exc_info=True)
        raise


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main(bert_dir: str, lm_dir: str, output_dir: str, company_list: Optional[List[str]] = None) -> None:
    """
    Main execution function for multi-year analysis

    Parameters:
    -----------
    bert_dir : str
        Directory containing FinBERT CSV files
    lm_dir : str
        Directory containing LM CSV files
    output_dir : str
        Directory for output files
    company_list : list, optional
        List of specific company names to analyze.
        If None or empty, analyzes all companies.
    """
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-YEAR SENTIMENT ANALYSIS COMPARISON")
    logger.info("FinBERT vs. LM Dictionary")
    logger.info("=" * 80 + "\n")

    logger.info(f"Configuration:")
    logger.info(f"  - Threshold method: {NormConfig.THRESHOLD_METHOD}")
    logger.info(f"  - LM standardization: {NormConfig.LM_STANDARDIZATION}")
    logger.info(f"  - Year range: {AnalysisConfig.YEAR_RANGE}\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Discover and match files
    matched_files = discover_files(bert_dir, lm_dir)

    if not matched_files:
        logger.error("\n❌ ERROR: No matching files found. Cannot proceed.")
        return

    # Check if filtering for specific companies
    if company_list and len(company_list) > 0:
        logger.info(f"\n📊 Filtering analysis for {len(company_list)} specific companies:")
        for company in company_list:
            logger.info(f"   • {company}")
        logger.info("")

    # Step 2: Process each year
    all_data = {}
    results_by_year = []

    for year in sorted(matched_files.keys()):
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING YEAR: {year}")
        logger.info(f"{'='*80}")

        try:
            # Load and merge data
            df = load_and_merge_year(
                matched_files[year]['bert'],
                matched_files[year]['lm'],
                year
            )

            if len(df) == 0:
                logger.warning(f"No data for year {year} after merge, skipping")
                continue

            # Create composite scores
            df = create_composite_scores(df)

            # Create per-word normalization metrics
            df = create_per_word_metrics(df)

            # Create per-sentence normalization metrics
            df = create_per_sentence_metrics(df)

            # Standardize all metrics (z-scores)
            df = standardize_all_metrics(df)

            # Add categorical sentiment
            df = add_categorical_sentiment(df)

            # Filter by specified companies if provided
            if company_list and len(company_list) > 0:
                df = filter_by_companies(df, company_list, year)

                if len(df) == 0:
                    logger.warning(f"No matching companies for year {year}, skipping")
                    continue

            # Analyze this year
            year_results = analyze_year(df, year, output_dir)

            # Store for cross-year analysis
            all_data[year] = df
            results_by_year.append(year_results)

            # Save year-specific merged data
            year_output = os.path.join(output_dir, f'merged_data_{year}.csv')
            df.to_csv(year_output, index=False)
            logger.info(f"  Saved merged data: {year_output}")

        except Exception as e:
            logger.error(f"Error processing year {year}: {e}", exc_info=True)
            continue

    if len(all_data) == 0:
        logger.error("\n❌ ERROR: No years successfully processed. Cannot continue.")
        return

    # Step 3: Cross-year analysis
    try:
        results_df, df_all, company_summary_df = analyze_across_years(all_data, results_by_year, output_dir)

        # Step 4: Save aggregate data
        all_output = os.path.join(output_dir, 'merged_data_all_years.csv')
        df_all.to_csv(all_output, index=False)
        logger.info(f"\nSaved aggregate data: {all_output}")

        # Step 5: Create summary report
        create_summary_report(results_df, df_all, company_summary_df, output_dir)

    except Exception as e:
        logger.error(f"Error in cross-year analysis: {e}", exc_info=True)
        return

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  - Individual year summaries: year_YYYY_summary.png")
    logger.info(f"  - Correlation trends: correlation_trends.png")
    logger.info(f"  - Classification metrics: classification_metrics.png")
    logger.info(f"  - Sentiment trends: sentiment_trends.png")
    logger.info(f"  - Aggregate analysis: aggregate_analysis.png")
    logger.info(f"  - Company coverage: company_coverage.png")
    logger.info(f"  - Merged data files: merged_data_YYYY.csv")
    logger.info(f"  - All years combined: merged_data_all_years.csv")
    logger.info(f"  - Correlation table: correlations_by_year.csv")
    logger.info(f"  - Company tracking: company_tracking.csv")
    logger.info(f"  - Companies with gaps: companies_with_gaps.csv")
    logger.info(f"  - Summary report: multi_year_summary.txt")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Multi-Year Sentiment Analysis Comparison: FinBERT vs. LM Dictionary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cross_comparison_analyzer_improved.py --bert-dir ./bert --lm-dir ./lm --output-dir ./output
  python cross_comparison_analyzer_improved.py -c Microsoft Apple Google
  python cross_comparison_analyzer_improved.py --threshold-method percentile
  python cross_comparison_analyzer_improved.py --lm-metric raw --threshold-method fixed
        """
    )

    # Add arguments
    parser.add_argument(
        '--bert-dir',
        type=str,
        default=r'D:\pycharm\Sentiment-extraction-and-analysis\mdna_sentiments_bert',
        help='Directory containing FinBERT CSV files'
    )

    parser.add_argument(
        '--lm-dir',
        type=str,
        default=r'D:\pycharm\Sentiment-extraction-and-analysis\mdna_sentiments_LM',
        help='Directory containing LM CSV files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=r'D:\pycharm\Sentiment-extraction-and-analysis\cross_comparison_improved_output',
        help='Directory for output files'
    )

    parser.add_argument(
        '-c', '--companies',
        nargs='+',
        default=None,
        help='Specific companies to analyze (e.g., -c Microsoft Netflix "Hess Corporation"). '
             'If not specified, all companies will be analyzed.'
    )

    parser.add_argument(
        '--threshold-method',
        type=str,
        choices=['std', 'percentile', 'fixed'],
        default='std',
        help='Method for categorical thresholds: std (default), percentile, or fixed'
    )

    parser.add_argument(
        '--lm-metric',
        type=str,
        choices=['proportion', 'raw'],
        default='proportion',
        help='LM metric to use: proportion (default) or raw counts'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Update configuration based on arguments
    NormConfig.THRESHOLD_METHOD = args.threshold_method
    NormConfig.LM_STANDARDIZATION = args.lm_metric

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Use parsed arguments
    BERT_DIR = args.bert_dir
    LM_DIR = args.lm_dir
    OUTPUT_DIR = args.output_dir
    COMPANY_LIST = args.companies

    # Check if directories exist
    if not os.path.exists(BERT_DIR):
        logger.error(f"❌ ERROR: FinBERT directory not found: {BERT_DIR}")
        logger.error("Please update the path using --bert-dir argument or modify the default in the script.")
    elif not os.path.exists(LM_DIR):
        logger.error(f"❌ ERROR: LM directory not found: {LM_DIR}")
        logger.error("Please update the path using --lm-dir argument or modify the default in the script.")
    else:
        main(BERT_DIR, LM_DIR, OUTPUT_DIR, COMPANY_LIST)