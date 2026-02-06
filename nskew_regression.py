import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CRASH PREDICTION REGRESSION ANALYSIS")
print("Chen, Hong, & Stein (2001) Replication - Corrected Methodology")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n[1/6] Loading data...")
# [cite: 168] Data extracted from CRSP daily and monthly stock files
crsp_daily = pd.read_csv('crsp_daily.csv')
crsp_market = pd.read_csv('crsp_market.csv')

# Standardize column names
crsp_daily = crsp_daily.rename(columns={'date': 'DATE'})
crsp_market = crsp_market.rename(columns={'DlyCalDt': 'DATE'})
crsp_daily['DATE'] = pd.to_datetime(crsp_daily['DATE'])
crsp_market['DATE'] = pd.to_datetime(crsp_market['DATE'])

print(f"   - Loaded {len(crsp_daily):,} daily stock observations")

# ============================================================================
# STEP A: UNIVERSE DEFINITION & CLEANING
# ============================================================================
print("\n[2/6] Data cleaning and filtering...")

# 1. Merge market returns
df = crsp_daily.merge(crsp_market[['DATE', 'vwretd']], on='DATE', how='left')

# 2. Type Conversion
numeric_cols = ['RET', 'PRC', 'SHROUT', 'VOL', 'vwretd', 'HEXCD', 'SHRCD']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Drop missing returns
df = df.dropna(subset=['RET', 'vwretd'])

# 4. Filter Share Codes 10, 11
# [cite: 171] "exclude ADRs, REITs... keep only stocks with CRSP share type code of 10 or 11"
if 'SHRCD' in df.columns:
    df = df[df['SHRCD'].isin([10, 11])].copy()

# 5. Filter Exchange NYSE (1) / AMEX (2)
# [cite: 170] "We do not include NASDAQ firms... turnover... not directly comparable"
if 'HEXCD' in df.columns:
    df = df[df['HEXCD'].isin([1, 2])].copy()

# 6. Calculate Market Cap (in MILLIONS)
#  Table 1 Mean LOGSIZE is 5.177. This implies MKTCAP is in Millions ($e^5.17 \approx 177$).
# CRSP PRC is absolute price; SHROUT is in THOUSANDS.
# MKTCAP ($M) = (ABS(PRC) * (SHROUT * 1000)) / 1,000,000 = ABS(PRC) * SHROUT / 1000
df['MKTCAP'] = (df['PRC'].abs() * df['SHROUT']) / 1000.0
df = df[df['MKTCAP'] > 0].copy()

print(f"   - Initial Universe: {len(df):,} observations")

# ============================================================================
# STEP B: VARIABLE CONSTRUCTION
# ============================================================================
print("\n[3/6] Constructing 6-month period variables...")

# 1. Define Periods (Jan-Jun, Jul-Dec)
# [cite: 209] "use nonoverlapping six-month observations... Jan 1-Jun 30 or Jul 1-Dec 31"
df['Year'] = df['DATE'].dt.year
df['Half'] = np.where(df['DATE'].dt.month <= 6, 1, 2)
df['Period_ID'] = df['Year'].astype(str) + '_H' + df['Half'].astype(str)

# 2. Data Completeness Filter
# [cite: 185] "drop any firm that has more than five missing observations... in a given period"
period_max_days = df.groupby('Period_ID')['DATE'].nunique()
firm_days = df.groupby(['PERMNO', 'Period_ID']).size().reset_index(name='N_Days')
firm_days = firm_days.merge(period_max_days.rename('Max_Days'), on='Period_ID')
firm_days['Missing'] = firm_days['Max_Days'] - firm_days['N_Days']
valid_firm_periods = firm_days[firm_days['Missing'] <= 5][['PERMNO', 'Period_ID']]

# Filter df to valid periods only
df = df.merge(valid_firm_periods, on=['PERMNO', 'Period_ID'], how='inner')

# 3. Log Returns & Market Adjustment
# [cite: 186] "We use log changes as opposed to simple daily percentage returns"
# [cite: 194] "market-adjusted returns... log change in stock i less log change in value-weighted index"
df['ret_log'] = np.log(1 + df['RET'])
df['mkt_log'] = np.log(1 + df['vwretd'])
df['e'] = df['ret_log'] - df['mkt_log']

# 4. Aggregation
grouped = df.groupby(['PERMNO', 'Period_ID'])


# Functions
def calc_ncskew(x):
    # [cite: 181] NCSKEW formula (Unbiased/Sample Analog) requires bias=False in scipy
    return -1 * stats.skew(x, nan_policy='omit', bias=False)


def calc_ret(x):
    # [cite: 225] Cumulative return over 6 months
    return x.sum()  # Sum of log returns = Cumulative Log Return


# Apply
period_vars = grouped.agg({
    'e': [calc_ncskew, 'std'],
    'ret_log': calc_ret,
    'HEXCD': 'last'  # Keep exchange code for size filtering
}).reset_index()

# Flatten columns
period_vars.columns = ['PERMNO', 'Period_ID', 'NCSKEW', 'SIGMA', 'RET', 'HEXCD']

# Add End-of-Period Market Cap (LOGSIZE)
# [cite: 228] "LOGSIZE... is the log of firm i's stock market capitalization at the end of period t"
last_obs = df.sort_values('DATE').groupby(['PERMNO', 'Period_ID']).tail(1)
last_obs['LOGSIZE'] = np.log(last_obs['MKTCAP'])
period_vars = period_vars.merge(last_obs[['PERMNO', 'Period_ID', 'LOGSIZE']], on=['PERMNO', 'Period_ID'])

print(f"   - Created {len(period_vars):,} firm-period observations")

# ============================================================================
# STEP C: TURNOVER DETRENDING
# ============================================================================
print("\n[4/6] Calculating detrended turnover (DTURNOVER)...")

# 1. Calculate Monthly Turnover (Corrected Units)
df['YearMonth'] = df['DATE'].dt.to_period('M')

# Group by Month
monthly_grp = df.groupby(['PERMNO', 'YearMonth'])
m_vol = monthly_grp['VOL'].sum()
m_shrout = monthly_grp['SHROUT'].last()  # SHROUT is in Thousands

#  "TURNOVER is... shares traded divided by shares outstanding"
# CRSP VOL is Shares. CRSP SHROUT is Thousands.
# We must multiply SHROUT by 1000 to match units.
monthly_data = pd.DataFrame({'TURNOVER': m_vol / (m_shrout * 1000)}).reset_index()

# 2. Calculate Detrended Turnover
# [cite: 231] "subtracting... moving average of its value over the prior 18 months"
period_list = sorted(df['Period_ID'].unique())


def get_prior_months(period_id):
    # Returns list of 18 YearMonths preceding the period
    y, h = map(int, period_id.split('_H'))
    start_m = pd.Period(f"{y}-{'01' if h == 1 else '07'}", freq='M')
    return [start_m - i for i in range(1, 19)]


def get_current_months(period_id):
    y, h = map(int, period_id.split('_H'))
    start_m = int(1) if h == 1 else 7
    return [pd.Period(f"{y}-{m:02d}", freq='M') for m in range(start_m, start_m + 6)]


# Processing loop (optimized)
dt_records = []
monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)  # String for faster lookup

for pid in period_list:
    # Get firms in this period
    current_firms = period_vars[period_vars['Period_ID'] == pid]['PERMNO'].unique()

    # Define time windows
    curr_months = [str(m) for m in get_current_months(pid)]
    prior_months = [str(m) for m in get_prior_months(pid)]

    # Filter monthly data
    # (In a full production script, this could be vectorized further, but this is safe)
    curr_data = monthly_data[monthly_data['YearMonth'].isin(curr_months)]
    prior_data = monthly_data[monthly_data['YearMonth'].isin(prior_months)]

    # Group means
    curr_means = curr_data[curr_data['PERMNO'].isin(current_firms)].groupby('PERMNO')['TURNOVER'].mean()
    prior_means = prior_data[prior_data['PERMNO'].isin(current_firms)].groupby('PERMNO')['TURNOVER'].mean()
    prior_counts = prior_data[prior_data['PERMNO'].isin(current_firms)].groupby('PERMNO')['TURNOVER'].count()

    # Calculate DTURNOVER
    # Require 12 months of prior data
    valid_priors = prior_counts[prior_counts >= 12].index
    common_firms = curr_means.index.intersection(valid_priors)

    for permno in common_firms:
        dturn = curr_means[permno] - prior_means[permno]
        dt_records.append({'PERMNO': permno, 'Period_ID': pid, 'DTURNOVER': dturn})

dturn_df = pd.DataFrame(dt_records)
period_vars = period_vars.merge(dturn_df, on=['PERMNO', 'Period_ID'], how='inner')

print(f"   - Calculated DTURNOVER for {len(period_vars):,} firm-periods")

# ============================================================================
# STEP D: REGRESSION SETUP
# ============================================================================
print("\n[5/6] Final regression setup...")

# 1. Lags
period_vars = period_vars.sort_values(['PERMNO', 'Period_ID'])
period_vars['NCSKEW_Future'] = period_vars.groupby('PERMNO')['NCSKEW'].shift(-1)
for i in range(1, 6):
    period_vars[f'RET_Lag{i}'] = period_vars.groupby('PERMNO')['RET'].shift(i)

# 2. Size Filter (NYSE 20th Percentile)
#  "eliminate... those with a market capitalization below the 20th percentile NYSE breakpoint"
# We calculate this using the End-of-Period LOGSIZE of NYSE firms only.
nyse_firms = period_vars[period_vars['HEXCD'] == 1]
size_cutoffs = nyse_firms.groupby('Period_ID')['LOGSIZE'].quantile(0.20).reset_index(name='Size_Cutoff')

period_vars = period_vars.merge(size_cutoffs, on='Period_ID', how='left')
reg_data = period_vars[period_vars['LOGSIZE'] > period_vars['Size_Cutoff']].copy()

print(f"   - Observations after Size Filter: {len(reg_data):,}")

# 3. Final Cleanup
cols = ['NCSKEW_Future', 'NCSKEW', 'DTURNOVER', 'RET', 'SIGMA', 'LOGSIZE'] + [f'RET_Lag{i}' for i in range(1, 6)]
reg_data = reg_data.dropna(subset=cols)

# ============================================================================
# STEP E: REGRESSION (Raw Values)
# ============================================================================
print("\n[6/6] Running OLS...")

#  Paper reports unstandardized coefficients. We use raw variables.
y = reg_data['NCSKEW_Future']
X_vars = ['NCSKEW', 'SIGMA', 'LOGSIZE', 'DTURNOVER', 'RET',
          'RET_Lag1', 'RET_Lag2', 'RET_Lag3', 'RET_Lag4', 'RET_Lag5']

X = reg_data[X_vars]
X = sm.add_constant(X)

# Time Fixed Effects
dummies = pd.get_dummies(reg_data['Period_ID'], prefix='Period', drop_first=True)
X = pd.concat([X, dummies], axis=1)

# Fit
model = OLS(y, X.astype(float))
results = model.fit(cov_type='cluster', cov_kwds={'groups': reg_data['PERMNO']})


# Output Formatter
def print_paper_table(res):
    print("-" * 50)
    print(f"{'Variable':<20} | {'Coeff':>10} | {'t-stat':>10}")
    print("-" * 50)
    target_vars = ['NCSKEW', 'SIGMA', 'LOGSIZE', 'DTURNOVER', 'RET'] + [f'RET_Lag{i}' for i in range(1, 6)]
    for v in target_vars:
        if v in res.params:
            print(f"{v:<20} | {res.params[v]:10.3f} | {res.tvalues[v]:10.3f}")
    print("-" * 50)
    print(f"R-squared: {res.rsquared:.3f}")
    print(f"Observations: {int(res.nobs):,}")


print_paper_table(results)

# Check condition number on MAIN variables only (excluding dummies for clarity)
cond_check = OLS(y, sm.add_constant(reg_data[X_vars])).fit()
print(f"\nCondition Number (Vars Only): {cond_check.condition_number:.2e}")