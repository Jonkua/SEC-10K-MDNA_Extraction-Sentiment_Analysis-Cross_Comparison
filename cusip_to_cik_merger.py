import pandas as pd

# Read the CSV files
cik_cusip = pd.read_csv('cik-cusip-maps.csv')
crsp_data = pd.read_csv('CRSP_ticker_cusip_list.csv')

# Convert CIK to integer to remove decimals
cik_cusip['cik'] = cik_cusip['cik'].astype(int)

# Clean and prepare data
# Remove any leading/trailing whitespace from CUSIPs
cik_cusip['cusip8'] = cik_cusip['cusip8'].astype(str).str.strip()
crsp_data['CUSIP'] = crsp_data['CUSIP'].astype(str).str.strip()

# Filter out rows without tickers in CRSP data
crsp_data = crsp_data[crsp_data['HTICK'].notna() & (crsp_data['HTICK'] != '')]

# Merge on CUSIP (using cusip8 from cik file and CUSIP from CRSP)
merged = pd.merge(
    cik_cusip[['cik', 'cusip8']],
    crsp_data[['CUSIP', 'HTICK']],
    left_on='cusip8',
    right_on='CUSIP',
    how='inner'  # inner join only keeps matches
)

# Create final output with just CIK and ticker
final_output = merged[['cik', 'HTICK']].drop_duplicates()

# Rename for clarity
final_output.columns = ['CIK', 'Ticker']

# Remove any remaining rows where either CIK or Ticker is missing
final_output = final_output.dropna()

# Save to CSV
final_output.to_csv('cik_ticker_mapping.csv', index=False)

print(f"Successfully created mapping with {len(final_output)} CIK-Ticker pairs")
print(f"\nFirst few rows:")
print(final_output.head(10))