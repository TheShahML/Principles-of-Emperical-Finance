"""
Assignment 0: R&D Portfolio Analysis
Complete Pipeline: Data Pull + Portfolio Construction + CAPM Alpha Analysis

This script performs a comprehensive analysis of R&D reporting and stock returns:
1. Pulls data from WRDS (Compustat + CRSP)
2. Constructs R&D and No-R&D portfolios (equal and value weighted)
3. Calculates CAPM alphas using Fama-French method with excess returns

Sample Period: 1980-2022
Data Sources: Compustat, CRSP, CCM Link Table
"""

import wrds
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Save intermediate CSV files (Compustat, CRSP, Market data)?
# Set to True if you want to inspect the raw data
# Set to False to only save final results
SAVE_INTERMEDIATE_CSV = False

print("="*80)
print("ASSIGNMENT 0: R&D PORTFOLIO ANALYSIS")
print("="*80)
print("Sample Period: 1980-2022")
print("Methodology: CAPM Alpha (Fama-French Method)")
print("="*80)

# =============================================================================
# PART 1: DATA COLLECTION FROM WRDS
# =============================================================================

print("\n" + "="*80)
print("PART 1: DATA COLLECTION FROM WRDS")
print("="*80)

# =============================================================================
# Step 1: Connect to WRDS
# =============================================================================

print("\nStep 1: Connecting to WRDS...")
conn = wrds.Connection()
print("✓ Connected to WRDS")

# =============================================================================
# Step 2: Pull Compustat Annual Fundamentals
# =============================================================================

print("\nStep 2: Pulling Compustat Annual Fundamentals...")

# We need fiscal years 1979-2021 so we have t-1 data for portfolios 1980-2022
compustat_query = """
    SELECT gvkey, datadate, fyear, xrd, sich, curcd, fic, exchg
    FROM comp.funda
    WHERE fyear BETWEEN 1979 AND 2021
        AND curcd = 'USD'
        AND fic = 'USA'
        AND exchg BETWEEN 11 AND 19       -- Major US exchanges (NYSE, AMEX, NASDAQ)
        AND sich IS NOT NULL
        AND (sich < 6000 OR sich > 6999)  -- exclude financials
        AND sich != 2834                   -- exclude pharmaceuticals
        AND indfmt = 'INDL'               -- industrial format (standard)
        AND datafmt = 'STD'               -- standardized data
        AND popsrc = 'D'                  -- domestic population
        AND consol = 'C'                  -- consolidated statements
"""

compustat = conn.raw_sql(compustat_query)
print(f"✓ Retrieved {len(compustat):,} firm-year observations")
print(f"  - Unique firms (gvkey): {compustat['gvkey'].nunique():,}")
print(f"  - Date range: {compustat['datadate'].min()} to {compustat['datadate'].max()}")

# Check R&D coverage
rd_reported = compustat['xrd'].notna() & (compustat['xrd'] > 0)
print(f"  - Firms with positive R&D: {rd_reported.sum():,} ({100*rd_reported.mean():.1f}%)")

# =============================================================================
# Step 3: Pull CCM Link Table (CRSP-Compustat Merged)
# =============================================================================

print("\nStep 3: Pulling CCM Link Table...")

ccm_query = """
    SELECT gvkey, lpermno, linkdt, linkenddt, linktype, linkprim
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LC', 'LU')      -- confirmed or usable links
        AND linkprim IN ('P', 'C')      -- primary security only
"""

ccm = conn.raw_sql(ccm_query)
print(f"✓ Retrieved {len(ccm):,} GVKEY-PERMNO links")
print(f"  - Unique gvkeys: {ccm['gvkey'].nunique():,}")

# =============================================================================
# Step 4: Merge Compustat with CCM Link Table
# =============================================================================

print("\nStep 4: Merging Compustat with CCM...")

# Convert date columns to datetime
compustat['datadate'] = pd.to_datetime(compustat['datadate'])
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])

# Merge on gvkey
compustat_linked = compustat.merge(ccm, on='gvkey', how='inner')

# Filter: datadate must be within link validity period
# linkenddt can be NaT (still active), so we handle that
compustat_linked['linkenddt'] = compustat_linked['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
compustat_linked = compustat_linked[
    (compustat_linked['datadate'] >= compustat_linked['linkdt']) &
    (compustat_linked['datadate'] <= compustat_linked['linkenddt'])
]

print(f"✓ After CCM date filtering: {len(compustat_linked):,} observations")
print(f"  - Unique firms (lpermno): {compustat_linked['lpermno'].nunique():,}")

# =============================================================================
# Step 5: Pull CRSP Monthly Stock File
# =============================================================================

print("\nStep 5: Pulling CRSP Monthly Stock Returns...")

crsp_monthly_query = """
    SELECT permno, date, ret, prc, shrout, hsiccd
    FROM crsp.msf
    WHERE date BETWEEN '1980-01-01' AND '2022-12-31'
        AND hsiccd IS NOT NULL
        AND (hsiccd < 6000 OR hsiccd > 6999)  -- exclude financials
        AND hsiccd != 2834                     -- exclude pharmaceuticals
"""

crsp = conn.raw_sql(crsp_monthly_query)
crsp['date'] = pd.to_datetime(crsp['date'])
print(f"✓ Retrieved {len(crsp):,} stock-month observations")
print(f"  - Unique securities (permno): {crsp['permno'].nunique():,}")

# =============================================================================
# Step 6: Pull Delisting Returns (Survivorship Bias Correction)
# =============================================================================

print("\nStep 6: Pulling Delisting Returns...")

delist_query = """
    SELECT permno, dlstdt, dlret
    FROM crsp.msedelist
    WHERE dlstdt BETWEEN '1980-01-01' AND '2022-12-31'
"""

delist = conn.raw_sql(delist_query)
delist['dlstdt'] = pd.to_datetime(delist['dlstdt'])
print(f"✓ Retrieved {len(delist):,} delisting events")

# Merge delisting returns with monthly file
crsp = crsp.merge(
    delist[['permno', 'dlstdt', 'dlret']].rename(columns={'dlstdt': 'date'}),
    on=['permno', 'date'],
    how='left'
)

# =============================================================================
# Step 7: Clean Monthly Returns
# =============================================================================

print("\nStep 7: Cleaning CRSP Returns...")

# Convert to numeric
crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')
crsp['dlret'] = pd.to_numeric(crsp['dlret'], errors='coerce')

# Remove NA returns and returns < -100%
initial_count = len(crsp)
crsp = crsp[crsp['ret'].notna()]
crsp = crsp[crsp['ret'] >= -1.0]
print(f"  - Removed {initial_count - len(crsp):,} rows (NA or ret < -100%)")

# Compound regular returns with delisting returns
# Formula: (1 + ret) * (1 + dlret) - 1
crsp['ret_adj'] = crsp.apply(
    lambda row: (1 + row['ret']) * (1 + row['dlret']) - 1
                if pd.notna(row['dlret'])
                else row['ret'],
    axis=1
)

delist_adj_count = crsp['dlret'].notna().sum()
print(f"✓ Returns cleaned")
print(f"  - Delisting adjustments applied: {delist_adj_count:,}")

# Calculate market cap (price * shares outstanding)
crsp['mktcap'] = abs(crsp['prc']) * crsp['shrout']

# Add year and month columns
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month

# =============================================================================
# Step 8: Pull CRSP Value-Weighted Market Index
# =============================================================================

print("\nStep 8: Pulling CRSP Value-Weighted Market Index...")

vwretd_query = """
    SELECT date, vwretd
    FROM crsp.msi
    WHERE date BETWEEN '1980-01-01' AND '2022-12-31'
"""

market_index = conn.raw_sql(vwretd_query)
market_index['date'] = pd.to_datetime(market_index['date'])
market_index['vwretd'] = pd.to_numeric(market_index['vwretd'], errors='coerce')
print(f"✓ Retrieved {len(market_index):,} monthly market returns")
print(f"  - Average monthly return: {market_index['vwretd'].mean()*100:.3f}%")

# =============================================================================
# Step 9: Pull Risk-Free Rate (1-Month T-Bill)
# =============================================================================

print("\nStep 9: Pulling Risk-Free Rate (1-Month T-Bill)...")

rf_query = """
    SELECT caldt, t30ret
    FROM crsp.mcti
    WHERE caldt BETWEEN '1980-01-01' AND '2022-12-31'
"""

rf_rate = conn.raw_sql(rf_query)
rf_rate = rf_rate.rename(columns={'caldt': 'date', 't30ret': 'rf'})
rf_rate['date'] = pd.to_datetime(rf_rate['date'])
rf_rate['rf'] = pd.to_numeric(rf_rate['rf'], errors='coerce')

# Convert from percentage to decimal if needed
if rf_rate['rf'].mean() > 1:
    rf_rate['rf'] = rf_rate['rf'] / 100

print(f"✓ Retrieved {len(rf_rate):,} monthly RF rates")
print(f"  - Average monthly RF: {rf_rate['rf'].mean()*100:.3f}%")
print(f"  - Average annual RF: {rf_rate['rf'].mean()*12*100:.2f}%")

# Merge risk-free rate with market index
market_index = market_index.merge(rf_rate[['date', 'rf']], on='date', how='left')

# Forward fill any missing RF values (rare case)
if market_index['rf'].isna().any():
    print(f"  - Filling {market_index['rf'].isna().sum()} missing RF values...")
    market_index['rf'] = market_index['rf'].fillna(method='ffill')

# =============================================================================
# Step 10: Close WRDS Connection and Save Data
# =============================================================================

print("\nStep 10: Closing WRDS Connection...")

conn.close()
print("✓ WRDS connection closed")

# Optionally save intermediate data for inspection
if SAVE_INTERMEDIATE_CSV:
    print("\n  Saving intermediate CSV files...")
    compustat_linked.to_csv('compustat_linked.csv', index=False)
    crsp.to_csv('crsp_monthly.csv', index=False)
    market_index.to_csv('market_index.csv', index=False)
    print("  ✓ Saved: compustat_linked.csv, crsp_monthly.csv, market_index.csv")
else:
    print("  (Intermediate CSV files not saved - set SAVE_INTERMEDIATE_CSV=True to enable)")

# =============================================================================
# PART 2: PORTFOLIO CONSTRUCTION
# =============================================================================

print("\n" + "="*80)
print("PART 2: PORTFOLIO CONSTRUCTION")
print("="*80)

# =============================================================================
# Step 11: Create Portfolio Formation Dates
# =============================================================================

print("\nStep 11: Creating Portfolio Formation Dates...")

# Portfolio formation = fiscal year end + 6 months (accounting for 10-K filing delay)
compustat_linked['portfolio_start'] = compustat_linked['datadate'] + pd.DateOffset(months=6)

# Portfolio year = the July-June period when this R&D data will be used
# Example: Dec 1999 fiscal year -> +6mo = June 2000 -> July 2000 portfolio -> portfolio_year = 2000
compustat_linked['portfolio_year'] = np.where(
    compustat_linked['portfolio_start'].dt.month >= 7,
    compustat_linked['portfolio_start'].dt.year + 1,  # July-Dec → next July
    compustat_linked['portfolio_start'].dt.year        # Jan-June → this July
)

print(f"✓ Portfolio years: {compustat_linked['portfolio_year'].min()} to {compustat_linked['portfolio_year'].max()}")

# =============================================================================
# Step 12: Classify Firms as R&D or No-R&D
# =============================================================================

print("\nStep 12: Classifying Firms by R&D Status...")

# Dynamic classification based on t-1 R&D
compustat_linked['has_rd'] = (compustat_linked['xrd'].notna()) & (compustat_linked['xrd'] > 0)

rd_counts = compustat_linked.groupby('portfolio_year')['has_rd'].agg(['sum', 'count'])
rd_counts['pct_rd'] = 100 * rd_counts['sum'] / rd_counts['count']
print(f"✓ Average % of firms with R&D: {rd_counts['pct_rd'].mean():.1f}%")

# Prepare Compustat data for merging
compustat_slim = compustat_linked[['lpermno', 'portfolio_year', 'has_rd', 'datadate', 'linkdt', 'linkenddt']].copy()
compustat_slim = compustat_slim.rename(columns={'lpermno': 'permno'})

# =============================================================================
# Step 13: Assign CRSP Returns to Portfolio Years
# =============================================================================

print("\nStep 13: Assigning CRSP Returns to Portfolio Years...")

# Fama-French convention: July year Y to June year Y+1 -> portfolio_year = Y
crsp['portfolio_year'] = np.where(
    crsp['month'] >= 7,
    crsp['year'],      # July-Dec → current year
    crsp['year'] - 1   # Jan-June → previous year
)

print(f"✓ CRSP portfolio years: {crsp['portfolio_year'].min()} to {crsp['portfolio_year'].max()}")

# =============================================================================
# Step 14: Merge CRSP with R&D Classification
# =============================================================================

print("\nStep 14: Merging CRSP with R&D Classification...")

# Merge on permno and portfolio_year
merged = crsp.merge(
    compustat_slim[['permno', 'portfolio_year', 'has_rd', 'linkdt', 'linkenddt']],
    on=['permno', 'portfolio_year'],
    how='inner'
)

print(f"  - Initial merge: {len(merged):,} firm-month observations")

# =============================================================================
# Step 15: Apply CCM Link Date Filtering
# =============================================================================

print("\nStep 15: Applying CCM Link Date Filtering...")

# Filter: CRSP date must be within link validity period
initial_count = len(merged)
merged = merged[
    (merged['date'] >= merged['linkdt']) &
    (merged['date'] <= merged['linkenddt'])
]

print(f"✓ After link filtering: {len(merged):,} observations ({initial_count - len(merged):,} removed)")
print(f"  - Unique firms: {merged['permno'].nunique():,}")

# Drop link date columns
merged = merged.drop(['linkdt', 'linkenddt'], axis=1)

# Check R&D split
rd_obs = merged['has_rd'].sum()
print(f"  - R&D portfolio: {rd_obs:,} ({100*rd_obs/len(merged):.1f}%)")
print(f"  - No-R&D portfolio: {len(merged) - rd_obs:,} ({100*(1-rd_obs/len(merged)):.1f}%)")

# =============================================================================
# Step 16: Get Market Cap at Portfolio Formation (for Value Weights)
# =============================================================================

print("\nStep 16: Getting Market Cap for Value Weights...")

# Use June market cap as weights (end of formation period)
june_mktcap = crsp[crsp['month'] == 6][['permno', 'year', 'mktcap']].copy()
june_mktcap = june_mktcap.rename(columns={'year': 'portfolio_year', 'mktcap': 'weight_mktcap'})

merged = merged.merge(june_mktcap, on=['permno', 'portfolio_year'], how='left')

merged_vw = merged[merged['weight_mktcap'].notna()].copy()
print(f"✓ Rows with valid market cap for VW: {len(merged_vw):,}")

# =============================================================================
# Step 17: Calculate Portfolio Returns
# =============================================================================

print("\nStep 17: Calculating Portfolio Returns...")

def calc_equal_weighted_return(group):
    """Simple average of returns"""
    return group['ret_adj'].mean()

def calc_value_weighted_return(group):
    """Market-cap weighted average of returns"""
    weights = group['weight_mktcap'] / group['weight_mktcap'].sum()
    return (weights * group['ret_adj']).sum()

# Equal-weighted portfolios
ew_returns = merged.groupby(['date', 'has_rd']).apply(
    calc_equal_weighted_return
).unstack()
ew_returns.columns = ['EW_NoRD', 'EW_RD']

# Value-weighted portfolios
vw_returns = merged_vw.groupby(['date', 'has_rd']).apply(
    calc_value_weighted_return
).unstack()
vw_returns.columns = ['VW_NoRD', 'VW_RD']

# Combine into single dataframe
portfolio_returns = ew_returns.join(vw_returns)
portfolio_returns = portfolio_returns.reset_index()

print(f"✓ Portfolio returns calculated for {len(portfolio_returns):,} months")
print(f"  - Date range: {portfolio_returns['date'].min()} to {portfolio_returns['date'].max()}")

# =============================================================================
# Step 18: Summary Statistics
# =============================================================================

print("\n" + "-"*60)
print("PORTFOLIO SUMMARY STATISTICS")
print("-"*60)

print("\nAnnualized Returns (mean * 12):")
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    ann_ret = portfolio_returns[col].mean() * 12 * 100
    print(f"  {col}: {ann_ret:.2f}%")

# =============================================================================
# Step 19: Save Portfolio Returns
# =============================================================================

print("\nStep 19: Saving Portfolio Returns...")

# Always save portfolio returns CSV (final output)
portfolio_returns.to_csv('portfolio_returns.csv', index=False)
print("✓ Saved: portfolio_returns.csv")

# Optionally save firm-level data for further analysis
if SAVE_INTERMEDIATE_CSV:
    merged.to_csv('firm_monthly_returns.csv', index=False)
    print("✓ Saved: firm_monthly_returns.csv")

# =============================================================================
# PART 3: CAPM ALPHA ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("PART 3: CAPM ALPHA ANALYSIS (FAMA-FRENCH METHOD)")
print("="*80)

# =============================================================================
# Step 20: Merge Portfolio Returns with Market Data
# =============================================================================

print("\nStep 20: Merging Portfolio Returns with Market Data...")

# Normalize dates to end of month for proper matching
portfolio_returns['date'] = pd.to_datetime(portfolio_returns['date'])
portfolio_returns['date'] = portfolio_returns['date'] + pd.offsets.MonthEnd(0)
market_index['date'] = market_index['date'] + pd.offsets.MonthEnd(0)

portfolio_with_market = portfolio_returns.merge(market_index, on='date', how='inner')
print(f"✓ Merged {len(portfolio_with_market):,} months with market data")

# =============================================================================
# Step 21: Calculate Excess Returns (Fama-French Method)
# =============================================================================

print("\nStep 21: Calculating Excess Returns...")

# Ensure numeric types
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD', 'vwretd', 'rf']:
    portfolio_with_market[col] = pd.to_numeric(portfolio_with_market[col], errors='coerce')

# Drop rows with NaN
portfolio_with_market = portfolio_with_market.dropna(
    subset=['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD', 'vwretd', 'rf']
)
print(f"  - Observations after dropping NaN: {len(portfolio_with_market):,}")

# Calculate excess returns: R - R_f (Fama-French approach)
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    portfolio_with_market[f'{col}_excess'] = portfolio_with_market[col] - portfolio_with_market['rf']

# Market excess return (market risk premium)
portfolio_with_market['market_excess'] = portfolio_with_market['vwretd'] - portfolio_with_market['rf']

print(f"✓ Excess returns calculated")
print(f"  - Average monthly market premium: {portfolio_with_market['market_excess'].mean()*100:.3f}%")

# =============================================================================
# Step 22: Run CAPM Regressions (Alpha Estimation)
# =============================================================================

print("\n" + "="*60)
print("CAPM ALPHA REGRESSIONS")
print("="*60)
print("Model: (R_p - R_f) = α + β(R_m - R_f) + ε")
print("="*60)

alpha_results = {}
summary_data = []

for portfolio in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    print(f"\n{'-'*60}")
    print(f"Portfolio: {portfolio}")
    print(f"{'-'*60}")

    excess_col = f'{portfolio}_excess'

    # Prepare data for CAPM regression
    # y = portfolio excess return (R_p - R_f)
    # X = market excess return (R_m - R_f)
    y = np.array(portfolio_with_market[excess_col].values, dtype=float)
    X = np.array(portfolio_with_market['market_excess'].values, dtype=float)
    X = sm.add_constant(X)  # Add intercept

    # Run OLS regression
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract results
    alpha_monthly = results.params[0]  # Intercept = alpha
    beta = results.params[1]           # Slope = market beta
    alpha_se = results.bse[0]          # Standard error of alpha
    alpha_tstat = results.tvalues[0]   # t-statistic for alpha
    alpha_pval = results.pvalues[0]    # p-value for alpha
    r_squared = results.rsquared       # R-squared
    n_obs = results.nobs               # Number of observations

    # Annualize alpha
    alpha_annual = alpha_monthly * 12

    # Significance indicator
    if alpha_pval < 0.01:
        sig = "***"
    elif alpha_pval < 0.05:
        sig = "**"
    elif alpha_pval < 0.10:
        sig = "*"
    else:
        sig = ""

    # Store results
    alpha_results[portfolio] = {
        'alpha_monthly': alpha_monthly,
        'alpha_annual': alpha_annual,
        'beta': beta,
        'alpha_se': alpha_se,
        't_stat': alpha_tstat,
        'p_value': alpha_pval,
        'r_squared': r_squared,
        'n_obs': n_obs,
        'significance': sig
    }

    # Print results
    print(f"\nRegression Results:")
    print(f"  Alpha (monthly):     {alpha_monthly*100:>8.3f}%")
    print(f"  Alpha (annual):      {alpha_annual*100:>8.3f}%")
    print(f"  Beta:                {beta:>8.3f}")
    print(f"  Alpha t-stat:        {alpha_tstat:>8.3f} {sig}")
    print(f"  Alpha p-value:       {alpha_pval:>8.4f}")
    print(f"  R-squared:           {r_squared:>8.3f}")
    print(f"  Observations:        {n_obs:>8.0f}")

    summary_data.append({
        'Portfolio': portfolio,
        'Alpha (Monthly)': f"{alpha_monthly:.4f}",
        'Alpha (Annual)': f"{alpha_annual:.4f}",
        'Alpha (Annual %)': f"{alpha_annual*100:.2f}%",
        'Beta': f"{beta:.3f}",
        't-stat': f"{alpha_tstat:.2f}",
        'p-value': f"{alpha_pval:.4f}",
        'R²': f"{r_squared:.3f}",
        'Significance': sig
    })

print("\n" + "="*60)
print("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")
print("="*60)

# =============================================================================
# Step 23: Create and Save Summary Table
# =============================================================================

print("\n" + "="*60)
print("CAPM ALPHA SUMMARY TABLE")
print("="*60)

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('alpha_summary.csv', index=False)
print("\n✓ Summary saved to alpha_summary.csv")

# =============================================================================
# ANALYSIS COMPLETE
# =============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nFinal Output Files:")
print("  ✓ portfolio_returns.csv - Monthly portfolio returns")
print("  ✓ alpha_summary.csv - CAPM alpha results")

if SAVE_INTERMEDIATE_CSV:
    print("\nIntermediate Files (SAVE_INTERMEDIATE_CSV=True):")
    print("  ✓ compustat_linked.csv")
    print("  ✓ crsp_monthly.csv")
    print("  ✓ market_index.csv")
    print("  ✓ firm_monthly_returns.csv")
else:
    print("\n(Intermediate CSVs not saved - set SAVE_INTERMEDIATE_CSV=True to enable)")

print("="*80)
