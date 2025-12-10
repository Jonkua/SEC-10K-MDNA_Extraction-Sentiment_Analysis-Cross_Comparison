"""
Sentiment Score Normalization Module
Author: Analysis Framework
Date: 2025
Purpose: Normalize and standardize sentiment scores from FinBERT and LM Dictionary methods

This module handles all score normalization, composite score creation, and
categorical sentiment assignment for cross-method comparison.

NORMALIZATION METHODS:
1. Z-score (proportion-based): Original method using proportions, then z-scored
2. Per-word: Normalizes by word count, then z-scored for document length control
3. Per-sentence: Normalizes by sentence count, then z-scored for sentence structure control

All methods use FinBERT sentence counts for LM per-sentence normalization to ensure
identical denominators and true comparability.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

# Get logger
logger = logging.getLogger(__name__)


class AnalysisConfig:
    """Configuration class for analysis parameters"""

    # Threshold method: 'fixed', 'percentile', or 'std'
    THRESHOLD_METHOD = 'std'  # More adaptive across years

    # For 'std' method: thresholds in standard deviations
    POSITIVE_THRESHOLD_STD = 0.5
    NEGATIVE_THRESHOLD_STD = -0.5

    # For 'percentile' method: thresholds in percentiles
    POSITIVE_THRESHOLD_PERCENTILE = 60
    NEGATIVE_THRESHOLD_PERCENTILE = 40

    # For 'fixed' method: absolute thresholds
    POSITIVE_THRESHOLD_FIXED = 10
    NEGATIVE_THRESHOLD_FIXED = -10

    # Standardization approach: 'raw' or 'proportion'
    # 'raw' = use net_sentiment (raw counts)
    # 'proportion' = use lm_net_proportion (proportions)
    LM_STANDARDIZATION = 'proportion'  # Consistent with FinBERT's probability scale

    # Normalization method for primary correlation analysis
    # 'z_score' = Original proportion-based z-score (DEFAULT)
    # 'per_word' = Per-word normalized, then z-scored
    # 'per_sentence' = Per-sentence normalized, then z-scored
    NORMALIZATION_METHOD = 'per_sentence'


def safe_standardize(series: pd.Series, name: str) -> pd.Series:
    """
    Safely standardize a series with guards for edge cases

    Parameters:
    -----------
    series : pd.Series
        Series to standardize
    name : str
        Name of variable for logging

    Returns:
    --------
    pd.Series : Standardized series (z-scores) or original if cannot standardize
    """
    clean = series.dropna()

    if len(clean) == 0:
        logger.warning(f"Cannot standardize {name}: all values are NaN")
        return pd.Series(np.nan, index=series.index)

    std_val = clean.std()

    if std_val == 0 or np.isnan(std_val):
        logger.warning(f"Cannot standardize {name}: zero variance or NaN std")
        return pd.Series(0.0, index=series.index)

    mean_val = clean.mean()
    standardized = (series - mean_val) / std_val

    logger.debug(f"Standardized {name}: mean={mean_val:.3f}, std={std_val:.3f}")

    return standardized


def create_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparable sentiment scores from both methods with consistent standardization

    Uses configuration to ensure consistent comparison approach

    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataframe containing both FinBERT and LM Dictionary results

    Returns:
    --------
    pd.DataFrame : Dataframe with added composite and standardized score columns
    """
    # FinBERT composite scores
    df['finbert_net_sentiment'] = (
            df['finbert_avg_positive'] - df['finbert_avg_negative']
    )

    df['finbert_pos_ratio'] = df['finbert_avg_positive'] / (
            df['finbert_avg_positive'] + df['finbert_avg_negative'] + 1e-10
    )

    df['finbert_weighted'] = (
            (df['finbert_avg_positive'] - df['finbert_avg_negative']) *
            (1 - df['finbert_avg_neutral'])
    )

    # LM scores
    df['lm_net_proportion'] = (
            df['positive_proportion'] - df['negative_proportion']
    )

    # CONSISTENT STANDARDIZATION: Use the same LM metric throughout
    # Based on configuration, choose either raw counts or proportions
    if AnalysisConfig.LM_STANDARDIZATION == 'proportion':
        lm_metric = 'lm_net_proportion'
        logger.info("Using LM net proportions (consistent with FinBERT probability scale)")
    else:
        lm_metric = 'net_sentiment'
        logger.info("Using LM net raw counts")

    # Standardized scores (z-scores) for direct comparison with guards
    df['finbert_net_std'] = safe_standardize(
        df['finbert_net_sentiment'],
        'FinBERT net sentiment'
    )

    df['lm_net_std'] = safe_standardize(
        df[lm_metric],
        f'LM {lm_metric}'
    )

    # Also keep raw standardization available if needed
    if lm_metric != 'net_sentiment':
        df['lm_net_raw_std'] = safe_standardize(
            df['net_sentiment'],
            'LM net_sentiment (raw)'
        )

    if lm_metric != 'lm_net_proportion':
        df['lm_net_proportion_std'] = safe_standardize(
            df['lm_net_proportion'],
            'LM net_proportion'
        )

    return df


def create_per_word_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sentiment scores normalized by word count

    Both FinBERT and LM scores are divided by their respective word counts
    to create per-word sentiment density metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataframe containing both FinBERT and LM Dictionary results

    Returns:
    --------
    pd.DataFrame : Dataframe with added per-word metric columns
    """
    # Validate required columns exist
    required_cols = ['total_words_bert', 'total_words_lm', 'finbert_positive_count',
                     'finbert_negative_count', 'positive_count', 'negative_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for per-word metrics: {missing_cols}")
        raise ValueError(f"Cannot create per-word metrics: missing columns {missing_cols}")

    # Validate word counts are positive
    if not (df['total_words_bert'] > 0).all():
        logger.warning("Found zero or negative word counts in total_words_bert")
    if not (df['total_words_lm'] > 0).all():
        logger.warning("Found zero or negative word counts in total_words_lm")

    # FinBERT per-word metrics
    # Net sentiment per word
    df['finbert_net_per_word'] = (
        (df['finbert_positive_count'] - df['finbert_negative_count']) /
        df['total_words_bert']
    )

    # Individual components per word
    df['finbert_pos_per_word'] = df['finbert_positive_count'] / df['total_words_bert']
    df['finbert_neg_per_word'] = df['finbert_negative_count'] / df['total_words_bert']

    # LM per-word metrics
    # Net sentiment per word
    df['lm_net_per_word'] = (
        (df['positive_count'] - df['negative_count']) /
        df['total_words_lm']
    )

    # Individual components per word
    df['lm_pos_per_word'] = df['positive_count'] / df['total_words_lm']
    df['lm_neg_per_word'] = df['negative_count'] / df['total_words_lm']

    logger.info("Created per-word normalization metrics")
    logger.debug(f"FinBERT net per word: mean={df['finbert_net_per_word'].mean():.6f}, "
                f"std={df['finbert_net_per_word'].std():.6f}")
    logger.debug(f"LM net per word: mean={df['lm_net_per_word'].mean():.6f}, "
                f"std={df['lm_net_per_word'].std():.6f}")

    return df


def create_per_sentence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sentiment scores normalized by sentence count

    CRITICAL: Uses FinBERT's sentence count for BOTH methods since they analyze
    the same documents. This ensures true comparability by using identical denominators.

    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataframe containing both FinBERT and LM Dictionary results

    Returns:
    --------
    pd.DataFrame : Dataframe with added per-sentence metric columns
    """
    # Validate required columns exist
    required_cols = ['finbert_total_sentences', 'finbert_positive_count',
                     'finbert_negative_count', 'positive_count', 'negative_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for per-sentence metrics: {missing_cols}")
        raise ValueError(f"Cannot create per-sentence metrics: missing columns {missing_cols}")

    # Validate sentence counts are positive
    if not (df['finbert_total_sentences'] > 0).all():
        logger.warning("Found zero or negative sentence counts in finbert_total_sentences")
        invalid_count = (df['finbert_total_sentences'] <= 0).sum()
        logger.warning(f"Number of documents with invalid sentence counts: {invalid_count}")

    # FinBERT per-sentence metrics
    # Net sentiment per sentence (count of net sentiment sentences per total sentences)
    df['finbert_net_per_sentence'] = (
        (df['finbert_positive_count'] - df['finbert_negative_count']) /
        df['finbert_total_sentences']
    )

    # Individual components per sentence
    df['finbert_pos_per_sentence'] = (
        df['finbert_positive_count'] / df['finbert_total_sentences']
    )
    df['finbert_neg_per_sentence'] = (
        df['finbert_negative_count'] / df['finbert_total_sentences']
    )

    # LM per-sentence metrics
    # CRITICAL: Use FinBERT's sentence count for LM to ensure comparability
    # This represents "sentiment words per sentence" for LM
    df['lm_net_per_sentence'] = (
        (df['positive_count'] - df['negative_count']) /
        df['finbert_total_sentences']
    )

    # Individual components per sentence
    df['lm_pos_per_sentence'] = df['positive_count'] / df['finbert_total_sentences']
    df['lm_neg_per_sentence'] = df['negative_count'] / df['finbert_total_sentences']

    logger.info("Created per-sentence normalization metrics using FinBERT sentence counts")
    logger.debug(f"FinBERT net per sentence: mean={df['finbert_net_per_sentence'].mean():.6f}, "
                f"std={df['finbert_net_per_sentence'].std():.6f}")
    logger.debug(f"LM net per sentence: mean={df['lm_net_per_sentence'].mean():.6f}, "
                f"std={df['lm_net_per_sentence'].std():.6f}")

    return df


def standardize_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-score standardization to all normalization metrics

    Creates standardized versions of per-word and per-sentence metrics
    for direct comparison across different normalization approaches.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with raw per-word and per-sentence metrics

    Returns:
    --------
    pd.DataFrame : Dataframe with added standardized metric columns
    """
    # Standardize per-word metrics
    if 'finbert_net_per_word' in df.columns:
        df['finbert_net_per_word_std'] = safe_standardize(
            df['finbert_net_per_word'],
            'FinBERT net per word'
        )

    if 'lm_net_per_word' in df.columns:
        df['lm_net_per_word_std'] = safe_standardize(
            df['lm_net_per_word'],
            'LM net per word'
        )

    # Standardize per-sentence metrics
    if 'finbert_net_per_sentence' in df.columns:
        df['finbert_net_per_sentence_std'] = safe_standardize(
            df['finbert_net_per_sentence'],
            'FinBERT net per sentence'
        )

    if 'lm_net_per_sentence' in df.columns:
        df['lm_net_per_sentence_std'] = safe_standardize(
            df['lm_net_per_sentence'],
            'LM net per sentence'
        )

    logger.info("Standardized all per-word and per-sentence metrics")

    return df


def get_primary_metrics(normalization_method: str = None) -> Tuple[str, str]:
    """
    Get the primary metric column names based on normalization method

    Parameters:
    -----------
    normalization_method : str, optional
        Normalization method to use. If None, uses AnalysisConfig.NORMALIZATION_METHOD
        Options: 'z_score', 'per_word', 'per_sentence'

    Returns:
    --------
    tuple : (finbert_column_name, lm_column_name) for primary correlation analysis
    """
    if normalization_method is None:
        normalization_method = AnalysisConfig.NORMALIZATION_METHOD

    method_map = {
        'z_score': ('finbert_net_std', 'lm_net_std'),
        'per_word': ('finbert_net_per_word_std', 'lm_net_per_word_std'),
        'per_sentence': ('finbert_net_per_sentence_std', 'lm_net_per_sentence_std')
    }

    if normalization_method not in method_map:
        logger.warning(
            f"Unknown normalization method '{normalization_method}'. "
            f"Using default 'z_score'"
        )
        normalization_method = 'z_score'

    finbert_col, lm_col = method_map[normalization_method]
    logger.info(
        f"Using normalization method '{normalization_method}': "
        f"FinBERT={finbert_col}, LM={lm_col}"
    )

    return finbert_col, lm_col


def get_adaptive_thresholds(series: pd.Series, method: str = None) -> Tuple[float, float]:
    """
    Calculate adaptive thresholds based on data distribution

    Parameters:
    -----------
    series : pd.Series
        Data to calculate thresholds for
    method : str
        Threshold method: 'std', 'percentile', or 'fixed'
        If None, uses AnalysisConfig.THRESHOLD_METHOD

    Returns:
    --------
    tuple : (positive_threshold, negative_threshold)
    """
    if method is None:
        method = AnalysisConfig.THRESHOLD_METHOD

    clean_series = series.dropna()

    if len(clean_series) == 0:
        logger.warning("Cannot calculate thresholds: all values are NaN")
        return (0.0, 0.0)

    if method == 'std':
        # Use standard deviations from mean
        mean_val = clean_series.mean()
        std_val = clean_series.std()

        if std_val == 0 or np.isnan(std_val):
            logger.warning("Zero variance, using mean as threshold")
            return (mean_val, mean_val)

        pos_threshold = mean_val + AnalysisConfig.POSITIVE_THRESHOLD_STD * std_val
        neg_threshold = mean_val + AnalysisConfig.NEGATIVE_THRESHOLD_STD * std_val

        logger.debug(
            f"Threshold method 'std': positive > {pos_threshold:.3f}, "
            f"negative < {neg_threshold:.3f}"
        )

    elif method == 'percentile':
        # Use percentiles
        pos_threshold = np.percentile(clean_series, AnalysisConfig.POSITIVE_THRESHOLD_PERCENTILE)
        neg_threshold = np.percentile(clean_series, AnalysisConfig.NEGATIVE_THRESHOLD_PERCENTILE)

        logger.debug(
            f"Threshold method 'percentile': positive > {pos_threshold:.3f}, "
            f"negative < {neg_threshold:.3f}"
        )

    else:  # 'fixed'
        pos_threshold = AnalysisConfig.POSITIVE_THRESHOLD_FIXED
        neg_threshold = AnalysisConfig.NEGATIVE_THRESHOLD_FIXED

        logger.debug(
            f"Threshold method 'fixed': positive > {pos_threshold}, "
            f"negative < {neg_threshold}"
        )

    return (pos_threshold, neg_threshold)


def add_categorical_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous sentiment to categorical with adaptive thresholds

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with composite sentiment scores

    Returns:
    --------
    pd.DataFrame : Dataframe with added categorical sentiment columns
    """
    # FinBERT categories with ADAPTIVE thresholds based on net_sentiment (same approach as LM)
    fb_pos_threshold, fb_neg_threshold = get_adaptive_thresholds(df['finbert_net_sentiment'])

    fb_conditions = [
        df['finbert_net_sentiment'] > fb_pos_threshold,
        df['finbert_net_sentiment'] < fb_neg_threshold,
    ]
    df['finbert_category'] = np.select(fb_conditions, ['positive', 'negative'], default='neutral')

    # Log FinBERT distribution
    fb_dist = df['finbert_category'].value_counts()
    logger.info(f"FinBERT category distribution: {fb_dist.to_dict()}")

    # LM categories with ADAPTIVE thresholds based on net_sentiment
    pos_threshold, neg_threshold = get_adaptive_thresholds(df['net_sentiment'])

    conditions = [
        df['net_sentiment'] > pos_threshold,
        df['net_sentiment'] < neg_threshold,
    ]
    choices = ['positive', 'negative']
    df['lm_category'] = np.select(conditions, choices, default='neutral')

    # Also create proportion-based categories for comparison
    pos_threshold_prop, neg_threshold_prop = get_adaptive_thresholds(df['lm_net_proportion'])

    conditions_prop = [
        df['lm_net_proportion'] > pos_threshold_prop,
        df['lm_net_proportion'] < neg_threshold_prop,
    ]
    df['lm_category_prop'] = np.select(conditions_prop, choices, default='neutral')

    # Log distribution
    lm_dist = df['lm_category'].value_counts()
    logger.info(f"LM category distribution: {lm_dist.to_dict()}")

    return df