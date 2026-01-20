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
        AND exchg BETWEEN 11 AND 19
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
# Step 5: Pull CRSP Monthly Stock File
# =============================================================================

crsp_query = """
    SELECT permno, date, ret, prc, shrout, hsiccd
    FROM crsp.msf
    WHERE date BETWEEN '1980-01-01' AND '2022-12-31'
        AND hsiccd IS NOT NULL
        AND (hsiccd < 6000 OR hsiccd > 6999)
        AND hsiccd != 2834
"""

print("\nPulling CRSP monthly data...")
crsp = conn.raw_sql(crsp_query)
crsp['date'] = pd.to_datetime(crsp['date'])
print(f"CRSP rows: {len(crsp)}")
print(f"Unique securities (permno): {crsp['permno'].nunique()}")

# =============================================================================
# Step 5b: Pull Delisting Returns (separate table)
# =============================================================================

delist_query = """
    SELECT permno, dlstdt, dlret
    FROM crsp.msedelist
    WHERE dlstdt BETWEEN '1980-01-01' AND '2022-12-31'
"""

print("\nPulling delisting returns...")
delist = conn.raw_sql(delist_query)
delist['dlstdt'] = pd.to_datetime(delist['dlstdt'])
print(f"Delisting events: {len(delist)}")

# Merge delisting returns with monthly file
# Match on permno and year-month (normalize both to end of month)
delist['year'] = delist['dlstdt'].dt.year
delist['month'] = delist['dlstdt'].dt.month
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month

crsp = crsp.merge(delist[['permno', 'year', 'month', 'dlret']],
                  on=['permno', 'year', 'month'], how='left')
crsp = crsp.drop(columns=['year', 'month'])
print(f"CRSP rows after delist merge: {len(crsp)}")

# =============================================================================
# Step 6: Clean CRSP Returns
# =============================================================================

print("\nCleaning CRSP returns...")

# Convert ret to numeric (handles any letter codes)
crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')

# Remove if return is NA or less than -100% (as per assignment)
crsp = crsp[crsp['ret'].notna()]
crsp = crsp[crsp['ret'] >= -1.0]  # -100% = -1.0 in decimal

# Handle delisting returns
crsp['dlret'] = pd.to_numeric(crsp['dlret'], errors='coerce')

# Compound ret and dlret where both exist, otherwise use available one
crsp['ret_adj'] = crsp.apply(
    lambda row: (1 + row['ret']) * (1 + row['dlret']) - 1
                if pd.notna(row['dlret'])
                else row['ret'],
    axis=1
)

print(f"After cleaning: {len(crsp)} rows")
print(f"Returns with delisting adjustment: {crsp['dlret'].notna().sum()}")

# Calculate market cap (price * shares outstanding)
# Note: prc can be negative (indicates bid/ask midpoint), take absolute value
crsp['mktcap'] = abs(crsp['prc']) * crsp['shrout']

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
