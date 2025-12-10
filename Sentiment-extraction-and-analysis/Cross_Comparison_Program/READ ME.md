# Multi-Year Sentiment Analysis Cross-Comparison Tool

## Overview

A comprehensive analytical framework for comparing FinBERT transformer-based sentiment analysis with Loughran-McDonald (LM) Dictionary lexicon-based sentiment analysis across multiple years of MD&A sections from SEC 10-K filings. This tool provides rigorous statistical comparison, multiple normalization methods, correlation analysis, and publication-quality visualizations.

## Purpose

Financial sentiment analysis can be performed using different methodologies, each with distinct advantages. This tool enables researchers to:

1. **Compare methodologies systematically** across multiple years
2. **Identify agreement and divergence** between neural network and lexicon-based approaches
3. **Validate results** using multiple correlation metrics and statistical tests
4. **Control for document characteristics** using per-word and per-sentence normalization
5. **Track temporal trends** in sentiment and method agreement
6. **Generate publication-ready outputs** including visualizations and statistical tables

## Key Features

### Multi-Method Comparison
- **FinBERT**: Transformer-based contextual sentiment (ProsusAI/finbert)
- **LM Dictionary**: Lexicon-based financial word counting (Loughran-McDonald)
- **Side-by-side analysis**: Direct comparison on same documents

### Multiple Normalization Approaches
- **Z-score (proportion-based)**: Original method using sentiment proportions
- **Per-word normalization**: Controls for document length differences
- **Per-sentence normalization**: Controls for sentence structure differences
- **Consistent denominators**: Uses FinBERT sentence counts for both methods

### Comprehensive Statistical Analysis
- **Pearson correlation**: Linear relationship strength
- **Spearman correlation**: Monotonic relationship (rank-based)
- **Kendall tau**: Concordance measure (robust to ties)
- **Cohen's kappa**: Agreement beyond chance
- **Confusion matrices**: Classification agreement patterns
- **Accuracy scores**: Overall agreement rates

### Robust Data Processing
- **Intelligent file matching**: Automatically pairs FinBERT and LM files by year
- **CIK validation**: Robust company identifier cleaning and matching
- **Missing data handling**: Graceful handling of gaps in data
- **Company filtering**: Analyze specific companies or entire datasets
- **Schema validation**: Ensures required columns present

### Publication-Quality Outputs
- **8 comprehensive visualizations** per analysis
- **CSV exports** of all statistics and merged data
- **Summary report** with key findings
- **Company tracking** across years
- **Correlation trend charts** with multiple metrics

## Architecture

### Main Components

#### 1. Configuration Classes

**AnalysisConfig** (Cross_comparisonV2.py)
```python
class AnalysisConfig:
    REQUIRED_FINBERT_COLS = {
        'cik', 'finbert_avg_positive', 'finbert_avg_negative',
        'finbert_avg_neutral', 'finbert_dominant_sentiment'
    }
    
    REQUIRED_LM_COLS = {
        'cik', 'net_sentiment', 'positive_proportion', 'negative_proportion'
    }
    
    YEAR_RANGE = (2018, 2025)
```

**NormConfig** (sentiment_normalization.py)
```python
class AnalysisConfig:  # Also called NormConfig when imported
    THRESHOLD_METHOD = 'std'  # 'fixed', 'percentile', or 'std'
    LM_STANDARDIZATION = 'proportion'  # 'raw' or 'proportion'
    NORMALIZATION_METHOD = 'per_sentence'  # 'z_score', 'per_word', 'per_sentence'
    
    # Threshold parameters for categorical classification
    POSITIVE_THRESHOLD_STD = 0.5
    NEGATIVE_THRESHOLD_STD = -0.5
    POSITIVE_THRESHOLD_PERCENTILE = 60
    NEGATIVE_THRESHOLD_PERCENTILE = 40
    POSITIVE_THRESHOLD_FIXED = 10
    NEGATIVE_THRESHOLD_FIXED = -10
```

---

#### 2. File Discovery and Matching

**Function: `discover_files(bert_dir, lm_dir)`**

Automatically discovers and pairs sentiment CSV files by year.

**Process:**
```
1. Scan FinBERT directory for CSV files
   ↓
2. Extract year from each filename (regex: 20\d{2})
   ↓
3. Scan LM directory for CSV files
   ↓
4. Extract year from each filename
   ↓
5. Match files by year
   ↓
6. Report matched pairs and missing years
```

**Filename Patterns Supported:**
```python
# FinBERT formats
"sentiment_summary_bert_2020.csv"
"finbert_(2021)_results.csv"
"results_2022_bert.csv"

# LM formats
"sentiment_summary_lm_2020.csv"
"lm_dictionary_2021.csv"
"mdna_lm_2022.csv"
```

**Returns:**
```python
{
    2020: {
        'bert': '/path/to/sentiment_bert_2020.csv',
        'lm': '/path/to/sentiment_lm_2020.csv'
    },
    2021: {
        'bert': '/path/to/sentiment_bert_2021.csv',
        'lm': '/path/to/sentiment_lm_2021.csv'
    },
    ...
}
```

**Validation:**
- Warns if year found in one directory but not the other
- Checks year against valid range (2018-2025)
- Reports total matched pairs

---

#### 3. Data Loading and Merging

**Function: `load_and_merge_year(bert_file, lm_file, year)`**

Loads and merges FinBERT and LM data for a single year.

**Process:**
```
1. Load FinBERT CSV
   ↓
2. Validate required columns (raises error if missing)
   ↓
3. Clean CIK values (robust parsing with error handling)
   ↓
4. Load LM CSV
   ↓
5. Validate required columns
   ↓
6. Clean CIK values
   ↓
7. Merge on CIK (inner join - only matching companies)
   ↓
8. Add suffixes to distinguish columns (_bert, _lm)
   ↓
9. Add year column
   ↓
10. Log merge statistics (total rows, matched companies)
```

**CIK Cleaning** (`clean_cik_robust`):
```python
# Handles various CIK formats:
"0000051143" → 51143
"(51143)" → 51143
"51143.0" → 51143
"  51143  " → 51143

# Returns NaN for invalid values:
"", "nan", "None", "invalid" → NaN
```

**Column Validation:**
- Checks for required columns
- Provides helpful error messages if columns missing
- Lists available columns for debugging

**Merge Statistics:**
```
FinBERT 2020: 150 companies
LM 2020: 145 companies
Merged: 140 companies (93.3% match rate)
```

---

#### 4. Normalization Methods

The tool implements three normalization approaches, all from `sentiment_normalization.py`:

##### Method 1: Z-Score (Proportion-Based)

**Default method used in original research.**

**FinBERT:**
```python
net_sentiment = avg_positive - avg_negative
finbert_net_std = (net_sentiment - mean) / std
```

**LM Dictionary:**
```python
net_proportion = positive_proportion - negative_proportion
lm_net_std = (net_proportion - mean) / std
```

**Characteristics:**
- Based on probabilities/proportions (0-1 scale)
- Standardized within each year
- Comparable across methods
- Doesn't account for document length

---

##### Method 2: Per-Word Normalization

**Controls for document length differences.**

**FinBERT:**
```python
# Counts of sentences classified in each category
pos_count = finbert_positive_count
neg_count = finbert_negative_count

net_per_word = (pos_count - neg_count) / total_words_bert
finbert_net_per_word_std = (net_per_word - mean) / std
```

**LM Dictionary:**
```python
# Counts of words in each sentiment category
pos_count = positive_count
neg_count = negative_count

net_per_word = (pos_count - neg_count) / total_words_lm
lm_net_per_word_std = (net_per_word - mean) / std
```

**Characteristics:**
- Normalizes by word count
- Measures "sentiment density"
- Controls for document length variation
- FinBERT and LM use their own word counts

---

##### Method 3: Per-Sentence Normalization

**Controls for sentence structure differences.**

**FinBERT:**
```python
# Sentences classified in each category
pos_count = finbert_positive_count
neg_count = finbert_negative_count

net_per_sentence = (pos_count - neg_count) / finbert_total_sentences
finbert_net_per_sentence_std = (net_per_sentence - mean) / std
```

**LM Dictionary:**
```python
# CRITICAL: Uses FinBERT sentence count for comparability
pos_count = positive_count
neg_count = negative_count

net_per_sentence = (pos_count - neg_count) / finbert_total_sentences
lm_net_per_sentence_std = (net_per_sentence - mean) / std
```

**Characteristics:**
- Normalizes by sentence count
- **Uses FinBERT sentence count for both methods** (ensures identical denominator)
- Controls for sentence structure differences
- Represents "sentiment words per sentence" for LM

**Why Use FinBERT Sentence Count for LM?**
- Both methods analyze the same documents
- FinBERT explicitly tokenizes sentences
- Provides consistent sentence count
- Ensures true per-sentence comparability
- Avoids denominator mismatch

---

#### 5. Composite Score Creation

**Function: `create_composite_scores(df)`**

Creates derived sentiment metrics for analysis.

**FinBERT Scores:**
```python
# Net sentiment (main metric)
finbert_net_sentiment = avg_positive - avg_negative

# Positive ratio
finbert_pos_ratio = avg_positive / (avg_positive + avg_negative + 1e-10)

# Weighted sentiment (accounts for neutral)
finbert_weighted = (avg_positive - avg_negative) × (1 - avg_neutral)
```

**LM Scores:**
```python
# Net proportion
lm_net_proportion = positive_proportion - negative_proportion

# Can also use raw counts
net_sentiment = positive_count - negative_count  # (already in data)
```

**Standardization:**
```python
# Choose which LM metric to use based on config
if LM_STANDARDIZATION == 'proportion':
    lm_metric = 'lm_net_proportion'  # Consistent with FinBERT scale
else:
    lm_metric = 'net_sentiment'  # Raw counts

# Standardize both
finbert_net_std = (finbert_net_sentiment - mean) / std
lm_net_std = (lm_metric - mean) / std
```

---

#### 6. Categorical Sentiment Classification

**Function: `add_categorical_sentiment(df)`**

Converts continuous sentiment to positive/negative/neutral categories.

**Adaptive Thresholds:**

Three methods available (configured via `THRESHOLD_METHOD`):

**Method 1: Standard Deviation (default)**
```python
mean = sentiment.mean()
std = sentiment.std()

positive_threshold = mean + 0.5 × std
negative_threshold = mean - 0.5 × std

if sentiment > positive_threshold: category = 'positive'
elif sentiment < negative_threshold: category = 'negative'
else: category = 'neutral'
```

**Method 2: Percentile**
```python
positive_threshold = 60th percentile
negative_threshold = 40th percentile

# Top 40% → positive
# Bottom 40% → negative
# Middle 20% → neutral
```

**Method 3: Fixed**
```python
positive_threshold = +10
negative_threshold = -10

# Fixed values regardless of distribution
```

**Applied to Both Methods:**
- FinBERT: Based on `finbert_net_sentiment`
- LM: Based on `net_sentiment` (raw counts) or `lm_net_proportion`

**Why Adaptive Thresholds?**
- Account for year-to-year differences in sentiment distributions
- More robust across different time periods
- Avoid arbitrary fixed cutoffs
- Better handle distributional shifts

---

#### 7. Year-Specific Analysis

**Function: `analyze_year(df, year, output_dir)`**

Performs comprehensive analysis for a single year.

**Components:**

**A. Correlation Analysis**
```python
# Three correlation types
pearson_r, pearson_p = pearsonr(finbert_scores, lm_scores)
spearman_r, spearman_p = spearmanr(finbert_scores, lm_scores)
kendall_tau, kendall_p = kendalltau(finbert_scores, lm_scores)
```

**Pearson:**
- Measures linear relationship
- Sensitive to outliers
- Assumes normal distribution
- Range: -1 to +1

**Spearman:**
- Measures monotonic relationship
- Rank-based (robust to outliers)
- No distribution assumptions
- Range: -1 to +1

**Kendall:**
- Measures concordance
- More robust than Spearman
- Better for small samples or ties
- Range: -1 to +1

**B. Classification Metrics**
```python
# Compare categorical classifications
confusion_matrix(lm_category, finbert_category)
accuracy = accuracy_score(lm_category, finbert_category)
kappa = cohen_kappa_score(lm_category, finbert_category)
```

**Confusion Matrix:**
```
                 FinBERT
              Pos  Neu  Neg
LM   Pos  [   40   12    3  ]
     Neu  [   15   35   10  ]
     Neg  [    5   13   27  ]
```

**Accuracy:**
- Simple agreement rate
- (Diagonal sum) / (Total observations)

**Cohen's Kappa:**
- Agreement beyond chance
- Adjusts for random agreement
- < 0: Less than chance
- 0-0.2: Slight agreement
- 0.2-0.4: Fair agreement
- 0.4-0.6: Moderate agreement
- 0.6-0.8: Substantial agreement
- 0.8-1.0: Almost perfect agreement

**C. Descriptive Statistics**
```python
# Summary statistics for both methods
finbert_stats = {
    'mean': finbert_scores.mean(),
    'std': finbert_scores.std(),
    'min': finbert_scores.min(),
    'max': finbert_scores.max(),
    'median': finbert_scores.median()
}

lm_stats = { ... }  # Same for LM
```

**D. Normalization Method Comparison**
```python
# Compare all three normalization approaches
correlations = {}
for method in ['z_score', 'per_word', 'per_sentence']:
    finbert_col, lm_col = get_primary_metrics(method)
    r, p = pearsonr(df[finbert_col], df[lm_col])
    correlations[method] = {'r': r, 'p': p}
```

**E. Visualization**

Creates comprehensive summary figure with 8 subplots:

1. **Scatter plot**: FinBERT vs LM (standardized)
2. **Density hexbin**: Color-coded density
3. **Positive sentiment**: Time series trends
4. **Negative sentiment**: Time series trends
5. **Distribution comparison**: Histograms
6. **Normalization methods**: Correlation comparison
7. **Confusion matrix**: Classification agreement
8. **Residuals**: Prediction errors

**Returns:**
```python
{
    'year': 2020,
    'n_companies': 140,
    'correlations': {
        'pearson': {'r': 0.58, 'p': 0.001},
        'spearman': {'r': 0.62, 'p': 0.0001},
        'kendall': {'r': 0.45, 'p': 0.002}
    },
    'classification': {
        'accuracy': 0.67,
        'kappa': 0.52,
        'confusion_matrix': [[40, 12, 3], [15, 35, 10], [5, 13, 27]]
    },
    'normalization_comparison': {
        'z_score': 0.58,
        'per_word': 0.61,
        'per_sentence': 0.64
    }
}
```

---

#### 8. Cross-Year Analysis

**Function: `analyze_across_years(all_data, results_by_year, output_dir)`**

Analyzes trends and patterns across multiple years.

**A. Correlation Trends**

Creates visualization showing how correlations change over time:

```python
years = [2018, 2019, 2020, 2021, 2022]
pearson_rs = [0.55, 0.58, 0.62, 0.59, 0.61]
spearman_rs = [0.60, 0.63, 0.65, 0.62, 0.64]
kendall_taus = [0.43, 0.46, 0.48, 0.45, 0.47]

# Plot all three metrics with confidence bands
```

**Output:**
- Line plot with markers
- Separate lines for Pearson, Spearman, Kendall
- Shows temporal stability of agreement
- Identifies years with stronger/weaker agreement

**B. Sentiment Trends**

Tracks average sentiment levels over time:

```python
# Average positive sentiment by year
avg_finbert_pos = df.groupby('year')['finbert_avg_positive'].mean()
avg_lm_pos = df.groupby('year')['positive_proportion'].mean()

# Average negative sentiment by year
avg_finbert_neg = df.groupby('year')['finbert_avg_negative'].mean()
avg_lm_neg = df.groupby('year')['negative_proportion'].mean()

# Plot trends for both methods
```

**Output:**
- Dual-axis line plots
- FinBERT and LM on same chart
- Separate panels for positive and negative
- Shows whether sentiment trends align

**C. Classification Metrics Trends**

Tracks classification agreement over time:

```python
accuracy_by_year = [...]
kappa_by_year = [...]

# Visualize how agreement changes
```

**D. Aggregate Analysis**

Combines all years into single dataset:

```python
df_all = pd.concat(all_data.values(), ignore_index=True)

# Overall statistics across all years
overall_r = pearsonr(df_all['finbert_net_std'], df_all['lm_net_std'])
overall_accuracy = accuracy_score(df_all['lm_category'], df_all['finbert_category'])
```

**E. Company Tracking**

Identifies which companies appear in which years:

```python
# Track company coverage
company_tracking = {}
for year, df in all_data.items():
    for company in df['company_name'].unique():
        if company not in company_tracking:
            company_tracking[company] = []
        company_tracking[company].append(year)

# Find companies with gaps
companies_with_gaps = [
    company for company, years in company_tracking.items()
    if len(years) < len(all_data)
]
```

**Output CSV: company_tracking.csv**
```csv
company_name,years_present,year_list,total_years,has_gaps
IBM,5,"2018,2019,2020,2021,2022",5,False
Microsoft,4,"2018,2020,2021,2022",4,True
...
```

**Output CSV: companies_with_gaps.csv**
```csv
company_name,years_present,missing_years
Microsoft,4,2019
Apple,3,"2019,2021"
...
```

**F. Results Summary Table**

**Output CSV: correlations_by_year.csv**
```csv
year,n_companies,pearson_r,pearson_p,spearman_r,spearman_p,kendall_tau,kendall_p,accuracy,kappa
2018,128,0.547,0.001,0.603,0.0001,0.427,0.002,0.651,0.492
2019,135,0.582,0.0005,0.627,0.00001,0.458,0.001,0.673,0.518
2020,140,0.618,0.0001,0.651,0.00001,0.475,0.0005,0.687,0.539
2021,132,0.591,0.0003,0.635,0.00001,0.462,0.001,0.669,0.512
2022,138,0.609,0.0002,0.645,0.00001,0.471,0.0008,0.681,0.531
```

---

#### 9. Visualization Suite

**8 Publication-Quality Visualizations:**

**1. Year-Specific Summary (year_YYYY_summary.png)**
```
8-panel figure:
  - Scatter: FinBERT vs LM (with regression line)
  - Hexbin: Density visualization
  - Positive trends: Both methods over time
  - Negative trends: Both methods over time
  - Distributions: Histogram comparison
  - Normalization comparison: Bar chart
  - Confusion matrix: Heatmap
  - Residuals: Error distribution
```

**2. Correlation Trends (correlation_trends.png)**
```
Line plot with 3 metrics:
  - Pearson (blue line, circles)
  - Spearman (red line, squares)
  - Kendall (green line, triangles)
X-axis: Year
Y-axis: Correlation coefficient
```

**3. Classification Metrics (classification_metrics.png)**
```
Dual-panel:
  - Accuracy by year (bar chart)
  - Cohen's Kappa by year (bar chart)
Threshold lines for interpretation
```

**4. Sentiment Trends (sentiment_trends.png)**
```
4-panel figure:
  - FinBERT positive over time
  - LM positive over time
  - FinBERT negative over time
  - LM negative over time
```

**5. Aggregate Analysis (aggregate_analysis.png)**
```
Large scatter plot: All years combined
Color-coded by year
Regression line
Statistics panel with overall correlations
```

**6. Company Coverage (company_coverage.png)**
```
Heatmap:
  Companies (rows) × Years (columns)
  Present = colored cell
  Missing = white cell
Shows data completeness
```

**7. Normalization Method Comparison (normalization_comparison_YYYY.png)**
```
Grouped bar chart:
  X-axis: Year
  Bars: Z-score, Per-word, Per-sentence
  Y-axis: Correlation coefficient
Shows which normalization works best
```

**8. Residual Analysis (included in year summary)**
```
Scatter plot:
  X-axis: Predicted LM (from FinBERT)
  Y-axis: Residual (actual - predicted)
Identifies systematic biases
```

---

#### 10. Summary Report

**Function: `create_summary_report(results_df, df_all, company_summary_df, output_dir)`**

Generates comprehensive text report: `multi_year_summary.txt`

**Report Structure:**

```text
============================================================
MULTI-YEAR SENTIMENT ANALYSIS COMPARISON SUMMARY
FinBERT vs. Loughran-McDonald Dictionary
============================================================

ANALYSIS CONFIGURATION
------------------------------------------------------------
Normalization Method: per_sentence
Threshold Method: std
LM Standardization: proportion
Years Analyzed: 2018-2022

OVERALL RESULTS (All Years Combined)
------------------------------------------------------------
Total Observations: 673
Total Unique Companies: 145
Company-Year Combinations: 673

CORRELATION SUMMARY
------------------------------------------------------------
Average Pearson r: 0.589 (range: 0.547-0.618)
Average Spearman r: 0.632 (range: 0.603-0.651)
Average Kendall tau: 0.459 (range: 0.427-0.475)

CLASSIFICATION AGREEMENT
------------------------------------------------------------
Average Accuracy: 0.672 (range: 0.651-0.687)
Average Cohen's Kappa: 0.518 (range: 0.492-0.539)

YEAR-BY-YEAR BREAKDOWN
------------------------------------------------------------
Year: 2018
  Companies: 128
  Pearson r: 0.547 (p=0.001)
  Spearman r: 0.603 (p<0.001)
  Kendall tau: 0.427 (p=0.002)
  Accuracy: 0.651
  Cohen's Kappa: 0.492

[... more years ...]

NORMALIZATION METHOD COMPARISON
------------------------------------------------------------
Best performing method: per_sentence (avg r=0.623)
Z-score method: avg r=0.589
Per-word method: avg r=0.608

COMPANY COVERAGE
------------------------------------------------------------
Companies present all years: 112 (77.2%)
Companies with missing years: 33 (22.8%)
Most common gap pattern: Missing 2019

INTERPRETATION
------------------------------------------------------------
- Moderate-to-strong positive correlations indicate substantial 
  agreement between methods
- Spearman > Pearson suggests some non-linearity
- Kappa scores indicate moderate agreement on categorical classification
- Per-sentence normalization provides best comparability
- Both methods track similar temporal trends

GENERATED FILES
------------------------------------------------------------
[List of all output files]
```

---

#### 11. Company Filtering

**Function: `filter_by_companies(df, company_list, year)`**

Filters dataset to specific companies when requested.

**Features:**
- Case-insensitive partial matching
- Handles company name variations
- Reports matched companies
- Warns about missing companies

**Example:**
```python
company_list = ['Microsoft', 'Apple', 'Google']

# Matches:
# "MICROSOFT CORP"
# "Apple Inc"
# "GOOGLE LLC"
# "Alphabet Inc" (contains Google in full name)
```

**Matching Strategy:**
```python
def matches_company(df_company_name, search_name):
    # Convert both to lowercase
    df_name_lower = df_company_name.lower()
    search_lower = search_name.lower()
    
    # Check if search term appears in company name
    return search_lower in df_name_lower
```

**Output:**
```
Filtering for companies: Microsoft, Apple, Google
Matched companies for 2020:
  • MICROSOFT CORP (CIK: 789019)
  • APPLE INC (CIK: 320193)
  • GOOGLE LLC (CIK: 1652044)
Filtered to 3 companies (from 140 total)
```

## Usage

### Command-Line Interface

```bash
# Basic usage with default directories
python Cross_comparisonV2.py

# Specify directories
python Cross_comparisonV2.py \
  --bert-dir ./bert_results \
  --lm-dir ./lm_results \
  --output-dir ./comparison_output

# Analyze specific companies
python Cross_comparisonV2.py \
  -c Microsoft Apple Google

# Use different normalization and threshold methods
python Cross_comparisonV2.py \
  --threshold-method percentile \
  --lm-metric raw

# Full example with all options
python Cross_comparisonV2.py \
  --bert-dir /path/to/finbert/results \
  --lm-dir /path/to/lm/results \
  --output-dir /path/to/output \
  -c "Microsoft" "Apple Inc" "Alphabet" \
  --threshold-method std \
  --lm-metric proportion \
  --log-level DEBUG
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bert-dir` | str | (configured) | Directory with FinBERT CSV files |
| `--lm-dir` | str | (configured) | Directory with LM CSV files |
| `--output-dir` | str | (configured) | Output directory |
| `-c, --companies` | list | None | Specific companies to analyze |
| `--threshold-method` | str | 'std' | Categorical threshold method: std/percentile/fixed |
| `--lm-metric` | str | 'proportion' | LM metric: proportion/raw |
| `--log-level` | str | 'INFO' | Logging level: DEBUG/INFO/WARNING/ERROR |

### Python API Usage

```python
from Cross_comparisonV2 import main, discover_files, analyze_year
from sentiment_normalization import AnalysisConfig as NormConfig

# Configure analysis
NormConfig.NORMALIZATION_METHOD = 'per_sentence'
NormConfig.THRESHOLD_METHOD = 'std'
NormConfig.LM_STANDARDIZATION = 'proportion'

# Run complete analysis
main(
    bert_dir='./bert_results',
    lm_dir='./lm_results',
    output_dir='./comparison_output',
    company_list=['Microsoft', 'Apple', 'Google']
)

# Or step-by-step

# Step 1: Discover files
matched_files = discover_files('./bert_results', './lm_results')
print(f"Found {len(matched_files)} matched year pairs")

# Step 2: Load and analyze single year
from Cross_comparisonV2 import load_and_merge_year, analyze_year

df_2020 = load_and_merge_year(
    matched_files[2020]['bert'],
    matched_files[2020]['lm'],
    2020
)

results_2020 = analyze_year(df_2020, 2020, './output')
print(f"2020 Pearson r: {results_2020['correlations']['pearson']['r']:.3f}")
```

### Custom Normalization Configuration

```python
from sentiment_normalization import AnalysisConfig as NormConfig

# Configure before running analysis

# Use z-score normalization (original method)
NormConfig.NORMALIZATION_METHOD = 'z_score'

# Use per-word normalization
NormConfig.NORMALIZATION_METHOD = 'per_word'

# Use per-sentence normalization (recommended)
NormConfig.NORMALIZATION_METHOD = 'per_sentence'

# Change threshold method
NormConfig.THRESHOLD_METHOD = 'percentile'  # Top/bottom 40%
# or
NormConfig.THRESHOLD_METHOD = 'fixed'  # Fixed values
# or
NormConfig.THRESHOLD_METHOD = 'std'  # Standard deviations (adaptive)

# Change LM standardization
NormConfig.LM_STANDARDIZATION = 'raw'  # Raw counts
# or
NormConfig.LM_STANDARDIZATION = 'proportion'  # Proportions (recommended)
```

## Output Files

### CSV Outputs

**1. merged_data_YYYY.csv** (one per year)
```csv
cik,company_name,year,finbert_avg_positive,finbert_avg_negative,...
51143,IBM,2020,0.456,0.234,...
789019,MICROSOFT,2020,0.523,0.198,...
```
- All companies for specific year
- Both FinBERT and LM metrics
- Composite scores
- All normalization variants
- Categorical classifications

**2. merged_data_all_years.csv**
```csv
cik,company_name,year,finbert_avg_positive,finbert_avg_negative,...
51143,IBM,2018,0.441,0.252,...
51143,IBM,2019,0.449,0.241,...
51143,IBM,2020,0.456,0.234,...
```
- All companies, all years
- Panel data format
- Ready for time series / panel regression

**3. correlations_by_year.csv**
```csv
year,n_companies,pearson_r,pearson_p,spearman_r,spearman_p,kendall_tau,kendall_p,accuracy,kappa
2018,128,0.547,0.001,0.603,0.0001,0.427,0.002,0.651,0.492
2019,135,0.582,0.0005,0.627,0.00001,0.458,0.001,0.673,0.518
```
- Statistical summary by year
- All correlation types
- Classification metrics
- P-values

**4. company_tracking.csv**
```csv
company_name,years_present,year_list,total_years,has_gaps
IBM,5,"2018,2019,2020,2021,2022",5,False
Microsoft,4,"2018,2020,2021,2022",4,True
```
- Company coverage across years
- Identifies data completeness
- Flags missing years

**5. companies_with_gaps.csv**
```csv
company_name,years_present,missing_years
Microsoft,4,2019
Apple,3,"2019,2021"
```
- Only companies with missing years
- Lists specific gaps
- Useful for data quality assessment

### Visualization Outputs

All visualizations are high-resolution PNG files suitable for publication.

**Generated Files:**
1. `year_2018_summary.png` (one per year)
2. `year_2019_summary.png`
3. `year_2020_summary.png`
4. `year_2021_summary.png`
5. `year_2022_summary.png`
6. `correlation_trends.png`
7. `classification_metrics.png`
8. `sentiment_trends.png`
9. `aggregate_analysis.png`
10. `company_coverage.png`
11. `normalization_comparison_YYYY.png` (one per year)

### Text Output

**multi_year_summary.txt**
- Comprehensive summary report
- Configuration details
- Statistical summaries
- Year-by-year breakdowns
- Interpretation notes
- File inventory

## Dependencies

### Required Packages

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Package Versions

```
pandas>=1.3.0              # Data manipulation
numpy>=1.21.0              # Numerical operations
matplotlib>=3.4.0          # Plotting
seaborn>=0.11.0            # Statistical visualization
scipy>=1.7.0               # Statistical tests
scikit-learn>=0.24.0       # Machine learning metrics
```

### Module Structure

```
Cross_comparisonV2.py      # Main analysis script
sentiment_normalization.py  # Normalization functions
```

**Import Requirements:**
```python
from sentiment_normalization import (
    AnalysisConfig as NormConfig,
    create_composite_scores,
    add_categorical_sentiment,
    create_per_word_metrics,
    create_per_sentence_metrics,
    standardize_all_metrics,
    get_primary_metrics
)
```

## Interpretation Guide

### Correlation Coefficients

**Strength Guidelines:**
```
0.00-0.19: Very weak
0.20-0.39: Weak
0.40-0.59: Moderate
0.60-0.79: Strong
0.80-1.00: Very strong
```

**Typical Results for Sentiment Methods:**
- Pearson r: 0.50-0.65 (moderate-to-strong)
- Spearman r: 0.55-0.70 (stronger, accounts for non-linearity)
- Kendall tau: 0.40-0.55 (more conservative)

**Why Three Correlation Types?**
- **Pearson**: Tests linear relationship
- **Spearman**: Tests monotonic relationship (rank-based)
- **Kendall**: Tests concordance (most robust)

**If Spearman > Pearson:** Relationship is monotonic but not perfectly linear

**If Kendall << Pearson:** Many ties in data or outliers present

### Cohen's Kappa

**Interpretation Scale:**
```
< 0.00: Less than chance agreement
0.00-0.20: Slight agreement
0.21-0.40: Fair agreement
0.41-0.60: Moderate agreement
0.61-0.80: Substantial agreement
0.81-1.00: Almost perfect agreement
```

**Typical Results:**
- κ = 0.45-0.55: Moderate agreement
- Better than random (κ > 0)
- Room for methodological differences

**Why Use Kappa?**
- Adjusts for chance agreement
- More stringent than accuracy
- Better for imbalanced classes

### Accuracy

**Interpretation:**
```
0.60-0.70: Moderate agreement
0.70-0.80: Good agreement
0.80-0.90: Strong agreement
0.90-1.00: Very strong agreement
```

**Typical Results:**
- Accuracy: 0.65-0.75
- Better than random (0.33 for 3-class)
- Indicates substantial method overlap

**Limitations:**
- Doesn't account for chance
- Can be misleading with imbalanced classes
- Use with Kappa for complete picture

### Normalization Method Comparison

**Expected Patterns:**

**Z-score (proportion-based):**
- Baseline method
- Correlations: 0.50-0.60
- Best when documents similar length

**Per-word:**
- Slightly higher correlations: 0.55-0.65
- Controls for document length
- Better for varying document sizes

**Per-sentence:**
- Often highest correlations: 0.60-0.70
- Controls for sentence structure
- **Recommended for MD&A** (variable sentence complexity)
- Uses identical denominator (FinBERT sentence count)

**Why Per-Sentence Often Best:**
- MD&A sections vary in sentence structure
- Long sentences vs. short sentences affects word counts
- Sentence-level normalization controls this
- True per-sentence comparability (same denominator)

### Confusion Matrix Patterns

**Example Matrix:**
```
                 FinBERT
              Pos  Neu  Neg
LM   Pos  [   40   12    3  ]
     Neu  [   15   35   10  ]
     Neg  [    5   13   27  ]
```

**Analysis:**
- **Diagonal** (40, 35, 27): Agreement = 102/160 = 63.8%
- **Off-diagonal**: Disagreement patterns
- **LM Pos, FB Neu (12)**: LM more positive than FinBERT
- **LM Neg, FB Neu (13)**: LM more negative than FinBERT

**Common Patterns:**
1. **FinBERT more neutral**: FinBERT neutral class larger
2. **LM more extreme**: More positive/negative classifications
3. **Asymmetric disagreement**: May disagree more on positive than negative

### Temporal Trends

**Stable Correlations:**
- Correlations consistent across years (e.g., 0.55-0.65)
- Indicates robust methodological agreement
- Both methods tracking similar signals

**Increasing Correlations:**
- Correlations improving over time
- Could indicate:
  - Better data quality
  - More standardized reporting
  - Changing MD&A writing style

**Decreasing Correlations:**
- Correlations declining over time
- Could indicate:
  - Diverging sentiment
  - Methodological drift
  - Changing language patterns

**Event Impact:**
- Sharp changes in specific years
- Example: COVID-19 in 2020
- Check if sentiment levels also changed

## Performance Characteristics

### Processing Speed

**Typical Performance (moderate hardware):**
```
Single Year (150 companies):
  - File loading: 2-5 seconds
  - Merging: 1-2 seconds
  - Normalization: 3-5 seconds
  - Analysis: 5-10 seconds
  - Visualization: 10-15 seconds
  Total: ~30-40 seconds per year

5 Years (700 total observations):
  - Year-by-year: 2-3 minutes
  - Cross-year analysis: 30-60 seconds
  - Total runtime: 3-4 minutes
```

**Factors Affecting Speed:**
- Number of companies per year
- Number of years
- Complexity of visualizations
- Disk I/O speed

### Memory Usage

**Per Year:**
- DataFrame: ~50-100 KB per company
- 150 companies: ~10-15 MB
- Visualizations: ~20-30 MB peak

**Full Analysis (5 years):**
- Peak memory: ~100-200 MB
- Primarily from matplotlib figures
- Cleared after each visualization saved

### Disk Space

**Output Requirements:**
- CSV files: ~500 KB - 2 MB per year
- PNG files: ~500 KB - 1 MB each
- Total for 5 years: ~50-100 MB

## Best Practices

### 1. File Organization

```
project/
├── finbert_results/
│   ├── sentiment_summary_bert_2018.csv
│   ├── sentiment_summary_bert_2019.csv
│   ├── sentiment_summary_bert_2020.csv
│   ├── sentiment_summary_bert_2021.csv
│   └── sentiment_summary_bert_2022.csv
├── lm_results/
│   ├── sentiment_summary_lm_2018.csv
│   ├── sentiment_summary_lm_2019.csv
│   ├── sentiment_summary_lm_2020.csv
│   ├── sentiment_summary_lm_2021.csv
│   └── sentiment_summary_lm_2022.csv
└── comparison_output/
    └── [generated files]
```

### 2. Configuration Selection

**For Publication:**
```python
# Recommended settings for academic research
NormConfig.NORMALIZATION_METHOD = 'per_sentence'  # Best comparability
NormConfig.THRESHOLD_METHOD = 'std'  # Adaptive across years
NormConfig.LM_STANDARDIZATION = 'proportion'  # Consistent with FinBERT
```

**For Exploration:**
```python
# Try different configurations
for method in ['z_score', 'per_word', 'per_sentence']:
    NormConfig.NORMALIZATION_METHOD = method
    main(bert_dir, lm_dir, f'output_{method}')
```

### 3. Data Quality Checks

```python
# Before running analysis
import pandas as pd

# Check FinBERT file
df_bert = pd.read_csv('finbert_2020.csv')
print(f"FinBERT rows: {len(df_bert)}")
print(f"Unique CIKs: {df_bert['cik'].nunique()}")
print(f"Missing CIKs: {df_bert['cik'].isna().sum()}")

# Check LM file
df_lm = pd.read_csv('lm_2020.csv')
print(f"LM rows: {len(df_lm)}")
print(f"Unique CIKs: {df_lm['cik'].nunique()}")
print(f"Missing CIKs: {df_lm['cik'].isna().sum()}")

# Check overlap
common_ciks = set(df_bert['cik']) & set(df_lm['cik'])
print(f"Common CIKs: {len(common_ciks)}")
print(f"Match rate: {len(common_ciks)/min(len(df_bert), len(df_lm))*100:.1f}%")
```

### 4. Result Validation

```python
# After analysis
results = pd.read_csv('output/correlations_by_year.csv')

# Check correlation ranges
print(results[['year', 'pearson_r', 'spearman_r', 'kendall_tau']])

# Verify statistical significance
print(results[['year', 'pearson_p', 'spearman_p', 'kendall_p']])

# Check classification metrics
print(results[['year', 'accuracy', 'kappa']])

# Look for anomalies
if (results['pearson_r'] < 0).any():
    print("WARNING: Negative correlations found")
if (results['accuracy'] < 0.5).any():
    print("WARNING: Accuracy below chance")
```

### 5. Company Filtering Strategy

**Targeted Analysis:**
```bash
# Analyze specific industries
python Cross_comparisonV2.py -c "Microsoft" "Apple" "Google" "Meta"

# Analyze by size (specify large caps)
python Cross_comparisonV2.py -c "ExxonMobil" "Berkshire" "JPMorgan"
```

**Full Dataset:**
```bash
# Run without -c flag for all companies
python Cross_comparisonV2.py \
  --bert-dir ./bert \
  --lm-dir ./lm \
  --output-dir ./output_all
```

## Troubleshooting

### Issue: No Matching Files Found

**Symptom:**
```
ERROR: No matching files found. Cannot proceed.
```

**Causes:**
1. Incorrect directory paths
2. Filename patterns don't match year extraction regex
3. No overlapping years between directories

**Solutions:**
```python
# Check directory contents
import os
print("FinBERT files:", os.listdir('./bert_dir'))
print("LM files:", os.listdir('./lm_dir'))

# Test year extraction
from Cross_comparisonV2 import extract_year_from_filename
filename = "sentiment_bert_2020.csv"
year = extract_year_from_filename(filename)
print(f"Extracted year: {year}")

# Ensure filenames contain 4-digit years (20XX)
# Valid: "data_2020.csv", "sentiment_(2021).csv"
# Invalid: "data_20.csv", "sentiment_21.csv"
```

---

### Issue: Low Match Rate After Merge

**Symptom:**
```
FinBERT: 150 companies
LM: 145 companies
Merged: 50 companies (33.3% match rate)
```

**Causes:**
1. CIK format mismatch
2. Different company sets in each file
3. CIK cleaning issues

**Solutions:**
```python
# Investigate CIK formats
import pandas as pd

df_bert = pd.read_csv('bert_2020.csv')
df_lm = pd.read_csv('lm_2020.csv')

print("FinBERT CIK sample:", df_bert['cik'].head())
print("LM CIK sample:", df_lm['cik'].head())

# Check CIK types
print("FinBERT CIK dtype:", df_bert['cik'].dtype)
print("LM CIK dtype:", df_lm['cik'].dtype)

# Find non-matching CIKs
bert_ciks = set(df_bert['cik'].dropna())
lm_ciks = set(df_lm['cik'].dropna())
only_bert = bert_ciks - lm_ciks
only_lm = lm_ciks - bert_ciks

print(f"Only in FinBERT: {len(only_bert)}")
print(f"Only in LM: {len(only_lm)}")
```

---

### Issue: Column Validation Error

**Symptom:**
```
SCHEMA ERROR: FinBERT data for year 2020
Missing required columns: {'finbert_avg_positive'}
Available columns: {'cik', 'company_name', ...}
```

**Causes:**
1. CSV from different analysis version
2. Column names changed
3. Wrong file selected

**Solutions:**
```python
# Check actual column names
import pandas as pd
df = pd.read_csv('problematic_file.csv')
print("Columns:", list(df.columns))

# Compare to expected
from Cross_comparisonV2 import AnalysisConfig
print("Required FinBERT cols:", AnalysisConfig.REQUIRED_FINBERT_COLS)
print("Required LM cols:", AnalysisConfig.REQUIRED_LM_COLS)

# Update required columns if needed (in script)
AnalysisConfig.REQUIRED_FINBERT_COLS = {
    'cik', 'actual_column_name', ...
}
```

---

### Issue: Zero Variance Warning

**Symptom:**
```
WARNING: Cannot standardize FinBERT net sentiment: zero variance
```

**Causes:**
1. All companies have identical sentiment
2. Only one company in dataset
3. Data quality issue

**Solutions:**
```python
# Check data distribution
import pandas as pd
df = pd.read_csv('merged_data_2020.csv')

# Examine sentiment distribution
print(df['finbert_net_sentiment'].describe())
print(df['finbert_net_sentiment'].value_counts())

# Check for duplicates
print(f"Unique values: {df['finbert_net_sentiment'].nunique()}")

# If all identical, check source data
# May need to filter or re-run sentiment analysis
```

---

### Issue: Very Low Correlations

**Symptom:**
```
Pearson r: 0.05 (p=0.65)
```

**Causes:**
1. Methods measuring different aspects of sentiment
2. Data quality issues
3. Wrong normalization method
4. CIK mismatch (comparing different documents)

**Solutions:**
```python
# Try different normalization methods
from sentiment_normalization import AnalysisConfig as NormConfig

for method in ['z_score', 'per_word', 'per_sentence']:
    NormConfig.NORMALIZATION_METHOD = method
    # Run analysis
    print(f"{method}: r = {result['correlations']['pearson']['r']:.3f}")

# Check for outliers
import matplotlib.pyplot as plt
plt.scatter(df['finbert_net_std'], df['lm_net_std'])
plt.xlabel('FinBERT')
plt.ylabel('LM')
plt.title('Scatter plot - check for outliers')
plt.show()

# Examine extreme disagreements
df['disagreement'] = abs(df['finbert_net_std'] - df['lm_net_std'])
worst_cases = df.nlargest(10, 'disagreement')
print(worst_cases[['company_name', 'finbert_net_std', 'lm_net_std']])
```

---

### Issue: Memory Error

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Causes:**
1. Too many years at once
2. Very large datasets
3. Memory leak in visualization

**Solutions:**
```python
# Process years in batches
years = [2018, 2019, 2020, 2021, 2022]
for year_batch in [years[:3], years[3:]]:
    # Process subset
    matched_files_subset = {
        y: matched_files[y] for y in year_batch
    }
    # Run analysis on subset

# Or reduce visualization resolution
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100  # Lower DPI
plt.rcParams['savefig.dpi'] = 150  # Lower save DPI

# Clear figures after saving
plt.close('all')
```

## Advanced Usage

### Custom Threshold Configuration

```python
from sentiment_normalization import AnalysisConfig as NormConfig

# Use fixed thresholds
NormConfig.THRESHOLD_METHOD = 'fixed'
NormConfig.POSITIVE_THRESHOLD_FIXED = 15
NormConfig.NEGATIVE_THRESHOLD_FIXED = -15

# Use percentile thresholds
NormConfig.THRESHOLD_METHOD = 'percentile'
NormConfig.POSITIVE_THRESHOLD_PERCENTILE = 70  # Top 30%
NormConfig.NEGATIVE_THRESHOLD_PERCENTILE = 30  # Bottom 30%

# Use adaptive standard deviation thresholds
NormConfig.THRESHOLD_METHOD = 'std'
NormConfig.POSITIVE_THRESHOLD_STD = 0.75  # More conservative
NormConfig.NEGATIVE_THRESHOLD_STD = -0.75
```

### Programmatic Analysis

```python
from Cross_comparisonV2 import (
    discover_files,
    load_and_merge_year,
    analyze_year,
    analyze_across_years
)
from sentiment_normalization import (
    create_composite_scores,
    create_per_word_metrics,
    create_per_sentence_metrics,
    standardize_all_metrics,
    add_categorical_sentiment
)
import pandas as pd

# Discover files
matched_files = discover_files('./bert', './lm')

# Process each year
all_data = {}
results = []

for year, files in matched_files.items():
    # Load and merge
    df = load_and_merge_year(files['bert'], files['lm'], year)
    
    # Apply normalization pipeline
    df = create_composite_scores(df)
    df = create_per_word_metrics(df)
    df = create_per_sentence_metrics(df)
    df = standardize_all_metrics(df)
    df = add_categorical_sentiment(df)
    
    # Analyze
    year_results = analyze_year(df, year, './output')
    
    # Store
    all_data[year] = df
    results.append(year_results)

# Cross-year analysis
results_df, df_all, company_summary = analyze_across_years(
    all_data, results, './output'
)

# Custom analysis on combined data
print(f"Total observations: {len(df_all)}")
print(f"Unique companies: {df_all['company_name'].nunique()}")

# Year-specific statistics
by_year = df_all.groupby('year').agg({
    'finbert_avg_positive': 'mean',
    'finbert_avg_negative': 'mean',
    'positive_proportion': 'mean',
    'negative_proportion': 'mean'
})
print(by_year)
```

### Integration with Statistical Software

**Export for R:**
```python
# Save in R-friendly format
df_all = pd.read_csv('output/merged_data_all_years.csv')

# Ensure proper column names (R doesn't like spaces)
df_all.columns = df_all.columns.str.replace(' ', '_')

# Save with row names as separate column
df_all.reset_index().to_csv('output/data_for_r.csv', index=False)
```

**R Analysis:**
```r
library(tidyverse)
library(lme4)

# Load data
df <- read_csv('output/data_for_r.csv')

# Panel regression
model <- lmer(finbert_net_std ~ lm_net_std + (1|cik), data=df)
summary(model)

# Year-specific models
models_by_year <- df %>%
  group_by(year) %>%
  do(model = lm(finbert_net_std ~ lm_net_std, data=.))
```

**Export for Stata:**
```python
# Install pyreadstat for Stata format
# pip install pyreadstat

import pyreadstat

df_all = pd.read_csv('output/merged_data_all_years.csv')

# Save as Stata .dta file
pyreadstat.write_dta(
    df_all,
    'output/data_for_stata.dta',
    version=118  # Stata 14+
)
```

**Stata Analysis:**
```stata
use "output/data_for_stata.dta", clear

* Panel structure
xtset cik year

* Fixed effects regression
xtreg finbert_net_std lm_net_std, fe

* Year-by-year regressions
bysort year: regress finbert_net_std lm_net_std
```

## Research Applications

### 1. Methodology Validation

**Question**: Do neural network and lexicon approaches provide consistent sentiment signals?

**Analysis**:
```python
# Run cross-comparison
main(bert_dir, lm_dir, output_dir)

# Examine correlations
results = pd.read_csv('output/correlations_by_year.csv')
mean_pearson = results['pearson_r'].mean()
mean_kappa = results['kappa'].mean()

# Interpretation
if mean_pearson > 0.5 and mean_kappa > 0.4:
    print("Substantial methodological agreement")
else:
    print("Methods capture different sentiment aspects")
```

### 2. Temporal Trend Analysis

**Question**: How has corporate sentiment evolved over time?

**Analysis**:
```python
df_all = pd.read_csv('output/merged_data_all_years.csv')

# Average sentiment by year
trends = df_all.groupby('year').agg({
    'finbert_avg_positive': 'mean',
    'finbert_avg_negative': 'mean',
    'positive_proportion': 'mean',
    'negative_proportion': 'mean'
})

# Test for temporal pattern
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(
    trends.index,
    trends['finbert_avg_positive']
)
print(f"Positive sentiment trend: slope={slope:.4f}, p={p:.4f}")
```

### 3. Event Study

**Question**: Did COVID-19 impact corporate sentiment?

**Analysis**:
```python
df_all = pd.read_csv('output/merged_data_all_years.csv')

# Pre-COVID (2018-2019) vs COVID (2020) vs Post-COVID (2021-2022)
df_all['period'] = pd.cut(
    df_all['year'],
    bins=[2017, 2019, 2020, 2023],
    labels=['pre_covid', 'covid', 'post_covid']
)

# Compare periods
period_sentiment = df_all.groupby('period').agg({
    'finbert_avg_positive': 'mean',
    'finbert_avg_negative': 'mean'
})

# Statistical test
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(
    df_all[df_all['period']=='pre_covid']['finbert_avg_positive'],
    df_all[df_all['period']=='covid']['finbert_avg_positive'],
    df_all[df_all['period']=='post_covid']['finbert_avg_positive']
)
print(f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
```

### 4. Company-Specific Analysis

**Question**: How does sentiment vary across specific companies?

**Analysis**:
```python
# Run for specific companies
main(
    bert_dir, lm_dir, output_dir,
    company_list=['Microsoft', 'Apple', 'Google']
)

df = pd.read_csv('output/merged_data_all_years.csv')

# Plot company trajectories
import matplotlib.pyplot as plt

for company in ['Microsoft', 'Apple', 'Google']:
    company_data = df[df['company_name'].str.contains(company, case=False)]
    plt.plot(
        company_data['year'],
        company_data['finbert_avg_positive'],
        marker='o',
        label=company
    )

plt.xlabel('Year')
plt.ylabel('FinBERT Average Positive')
plt.legend()
plt.title('Sentiment Trajectories by Company')
plt.savefig('output/company_trajectories.png')
```

### 5. Cross-Sectional Analysis

**Question**: Which companies show greatest sentiment divergence between methods?

**Analysis**:
```python
df = pd.read_csv('output/merged_data_all_years.csv')

# Calculate method divergence
df['method_divergence'] = abs(
    df['finbert_net_std'] - df['lm_net_std']
)

# Identify high-divergence companies
high_divergence = df.groupby('company_name')['method_divergence'].mean()
top_divergent = high_divergence.nlargest(10)

print("Companies with highest method divergence:")
print(top_divergent)

# Investigate characteristics
for company in top_divergent.index[:5]:
    company_data = df[df['company_name'] == company]
    print(f"\n{company}:")
    print(f"  FinBERT mean: {company_data['finbert_net_std'].mean():.2f}")
    print(f"  LM mean: {company_data['lm_net_std'].mean():.2f}")
    print(f"  Years: {company_data['year'].tolist()}")
```

## Citation

If using this tool in academic research, please cite:

```bibtex
@software{sentiment_cross_comparison,
  title={Multi-Year Sentiment Analysis Cross-Comparison Tool},
  author={Joshua Burke},
  year={2025},
  publisher={GitHub},
  url={}
}
```

**Methods to Cite:**

**FinBERT:**
```bibtex
@article{araci2019finbert,
  title={FinBERT: Financial Sentiment Analysis with Pre-trained Language Models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019}
}
```

**Loughran-McDonald Dictionary:**
```bibtex
@article{loughran2011liability,
  title={When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks},
  author={Loughran, Tim and McDonald, Bill},
  journal={The Journal of Finance},
  volume={66},
  number={1},
  pages={35--65},
  year={2011}
}
```

## Version History

### Version 2.0 (Current)
- Multiple normalization methods (z-score, per-word, per-sentence)
- FinBERT sentence counts for LM normalization
- Comprehensive visualization suite
- Company filtering capability
- Robust CIK handling
- Adaptive thresholds
- Classification metrics
- All three correlation types in visualizations
- Logging infrastructure
- Comprehensive error handling

### Version 1.0
- Basic cross-comparison
- Single normalization method
- Basic visualizations
- Fixed thresholds

---

**Note**: This tool is designed for academic research in financial sentiment analysis. Results should be validated against domain knowledge and supplemented with qualitative analysis. The tool provides statistical comparison but does not determine which method is "correct" - both capture different but valid aspects of sentiment in financial text.