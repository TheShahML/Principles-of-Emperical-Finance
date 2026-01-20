"""
Assignment 0: R&D Portfolio Analysis
Data Pull from WRDS (Compustat + CRSP)
"""

import wrds
import pandas as pd
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

# Output format options: 'parquet', 'csv', or 'both'
# parquet: faster read/write, smaller file size, preserves data types
# csv: human readable, can open in Excel, easier to share
OUTPUT_FORMAT = 'both'

# =============================================================================
# Step 1: Connect to WRDS
# =============================================================================

# This will prompt for your WRDS username/password on first run
# After that, it caches credentials
conn = wrds.Connection()

# =============================================================================
# Step 2: Pull Compustat Annual Fundamentals
# =============================================================================

# We need fiscal years 1979-2021 so we have t-1 data for portfolios 1980-2022
compustat_query = """
    SELECT gvkey, datadate, fyear, xrd, sich, curcd, fic, exchg
    FROM comp.funda
    WHERE fyear BETWEEN 1979 AND 2021
        AND curcd = 'USD'
        AND fic = 'USA'
        AND exchg IN (11, 14)             -- NYSE (11) and NASDAQ (14) only
        AND sich IS NOT NULL
        AND (sich < 6000 OR sich > 6999)  -- exclude financials
        AND sich != 2834                   -- exclude pharmaceuticals
        AND indfmt = 'INDL'               -- industrial format (standard)
        AND datafmt = 'STD'               -- standardized data
        AND popsrc = 'D'                  -- domestic population
        AND consol = 'C'                  -- consolidated statements
"""

print("Pulling Compustat data...")
compustat = conn.raw_sql(compustat_query)
print(f"Compustat rows: {len(compustat)}")
print(f"Unique firms (gvkey): {compustat['gvkey'].nunique()}")
print(f"Date range: {compustat['datadate'].min()} to {compustat['datadate'].max()}")

# Check R&D coverage
rd_reported = compustat['xrd'].notna() & (compustat['xrd'] > 0)
print(f"Observations with positive R&D: {rd_reported.sum()} ({100*rd_reported.mean():.1f}%)")

# =============================================================================
# Step 3: Pull CCM Link Table
# =============================================================================

ccm_query = """
    SELECT gvkey, lpermno, linkdt, linkenddt, linktype, linkprim
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LC', 'LU')
        AND linkprim IN ('P', 'C')
"""

print("\nPulling CCM link table...")
ccm = conn.raw_sql(ccm_query)
print(f"CCM link rows: {len(ccm)}")
print(f"Unique gvkeys in link table: {ccm['gvkey'].nunique()}")

# =============================================================================
# Step 4: Merge Compustat with CCM Link Table
# =============================================================================

# Merge on gvkey, then filter for valid date range
print("\nMerging Compustat with CCM...")

# Convert date columns to datetime first (WRDS sometimes returns strings)
compustat['datadate'] = pd.to_datetime(compustat['datadate'])
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])

compustat_linked = compustat.merge(ccm, on='gvkey', how='inner')

# Filter: datadate must be within link validity period
# linkenddt can be NaT (still active), so we handle that
compustat_linked['linkenddt'] = compustat_linked['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
compustat_linked = compustat_linked[
    (compustat_linked['datadate'] >= compustat_linked['linkdt']) &
    (compustat_linked['datadate'] <= compustat_linked['linkenddt'])
]

print(f"After date filtering: {len(compustat_linked)} rows")
print(f"Unique firms (lpermno): {compustat_linked['lpermno'].nunique()}")

# =============================================================================
# Step 5: Pull CRSP Daily Stock File
# =============================================================================

# Pull daily returns (as required by assignment) - we'll compound to monthly later
crsp_daily_query = """
    SELECT permno, date, ret, prc, shrout, hsiccd
    FROM crsp.dsf
    WHERE date BETWEEN '1980-01-01' AND '2022-12-31'
        AND hsiccd IS NOT NULL
        AND (hsiccd < 6000 OR hsiccd > 6999)
        AND hsiccd != 2834
"""

print("\nPulling CRSP daily data...")
crsp_daily = conn.raw_sql(crsp_daily_query)
crsp_daily['date'] = pd.to_datetime(crsp_daily['date'])
print(f"CRSP daily rows: {len(crsp_daily)}")
print(f"Unique securities (permno): {crsp_daily['permno'].nunique()}")

# =============================================================================
# Step 5b: Pull Delisting Returns (daily delisting table)
# =============================================================================

delist_query = """
    SELECT permno, dlstdt, dlret
    FROM crsp.dsedelist
    WHERE dlstdt BETWEEN '1980-01-01' AND '2022-12-31'
"""

print("\nPulling delisting returns...")
delist = conn.raw_sql(delist_query)
delist['dlstdt'] = pd.to_datetime(delist['dlstdt'])
print(f"Delisting events: {len(delist)}")

# Merge delisting returns with daily file on exact date
crsp_daily = crsp_daily.merge(
    delist[['permno', 'dlstdt', 'dlret']].rename(columns={'dlstdt': 'date'}),
    on=['permno', 'date'],
    how='left'
)
print(f"CRSP daily rows after delist merge: {len(crsp_daily)}")

# =============================================================================
# Step 6: Clean Daily Returns and Compound to Monthly
# =============================================================================

print("\nCleaning CRSP daily returns...")

# Convert ret to numeric (handles any letter codes)
crsp_daily['ret'] = pd.to_numeric(crsp_daily['ret'], errors='coerce')

# Remove if return is NA or less than -100% (as per assignment)
crsp_daily = crsp_daily[crsp_daily['ret'].notna()]
crsp_daily = crsp_daily[crsp_daily['ret'] >= -1.0]  # -100% = -1.0 in decimal

# Handle delisting returns at daily level
crsp_daily['dlret'] = pd.to_numeric(crsp_daily['dlret'], errors='coerce')

# Compound ret and dlret where both exist, otherwise use available one
crsp_daily['ret_adj'] = crsp_daily.apply(
    lambda row: (1 + row['ret']) * (1 + row['dlret']) - 1
                if pd.notna(row['dlret'])
                else row['ret'],
    axis=1
)

print(f"After cleaning: {len(crsp_daily)} daily rows")
print(f"Returns with delisting adjustment: {crsp_daily['dlret'].notna().sum()}")

# Calculate market cap (price * shares outstanding)
# Note: prc can be negative (indicates bid/ask midpoint), take absolute value
crsp_daily['mktcap'] = abs(crsp_daily['prc']) * crsp_daily['shrout']

# =============================================================================
# Step 6b: Compound Daily Returns to Monthly Returns
# =============================================================================

print("\nCompounding daily returns to monthly...")

# Create year-month identifier
crsp_daily['year'] = crsp_daily['date'].dt.year
crsp_daily['month'] = crsp_daily['date'].dt.month

# Compound daily returns within each month for each stock
# Monthly return = product of (1 + daily_ret) - 1
def compound_returns(daily_rets):
    """Compound daily returns to get monthly return"""
    return (1 + daily_rets).prod() - 1

# Get end-of-month values for price, shares, market cap, and SIC code
crsp_monthly = crsp_daily.groupby(['permno', 'year', 'month']).agg({
    'ret_adj': compound_returns,      # Compound daily returns
    'prc': 'last',                     # End of month price
    'shrout': 'last',                  # End of month shares
    'mktcap': 'last',                  # End of month market cap
    'hsiccd': 'last',                  # SIC code (should be constant)
    'date': 'max'                      # Last trading day of month
}).reset_index()

# Rename for clarity
crsp = crsp_monthly.copy()

print(f"Monthly observations: {len(crsp)}")
print(f"Unique securities: {crsp['permno'].nunique()}")

# =============================================================================
# Step 7: Close connection and save intermediate data
# =============================================================================

conn.close()
print("\nWRDS connection closed.")

# Save data based on OUTPUT_FORMAT setting
print(f"\nSaving data (format: {OUTPUT_FORMAT})...")

if OUTPUT_FORMAT in ('parquet', 'both'):
    compustat_linked.to_parquet('compustat_linked.parquet', index=False)
    crsp.to_parquet('crsp_monthly.parquet', index=False)
    print("  Saved: compustat_linked.parquet, crsp_monthly.parquet")

if OUTPUT_FORMAT in ('csv', 'both'):
    compustat_linked.to_csv('compustat_linked.csv', index=False)
    crsp.to_csv('crsp_monthly.csv', index=False)
    print("  Saved: compustat_linked.csv, crsp_monthly.csv")

print("Data save complete.")
