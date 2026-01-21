"""
Assignment 0: R&D Portfolio Analysis
Portfolio Construction - Build R&D and No-R&D portfolios
"""

import pandas as pd
import numpy as np

# =============================================================================
# Step 1: Load the data from data pull
# =============================================================================

print("Loading data...")
compustat = pd.read_parquet('compustat_linked.parquet')
crsp = pd.read_parquet('crsp_monthly.parquet')

print(f"Compustat rows: {len(compustat)}")
print(f"CRSP rows: {len(crsp)}")

# =============================================================================
# Step 2: Create portfolio formation dates
# =============================================================================

# Portfolio formation = fiscal year end + 6 months (to account for 10-K filing delay)
# This ensures we only use publicly available information

print("\nCreating portfolio formation dates...")

compustat['datadate'] = pd.to_datetime(compustat['datadate'])
compustat['portfolio_start'] = compustat['datadate'] + pd.DateOffset(months=6)

# Portfolio year = the July-June period this R&D data will be used in
# e.g., Dec 1999 fiscal year -> filed by June 2000 -> used in July 2000-June 2001 period
# Since the July-June period is identified by the year of July, we need:
# - If portfolio_start is Jan-June: portfolio_year = year of portfolio_start
# - If portfolio_start is July-Dec: portfolio_year = year of portfolio_start
# Actually simpler: Just use the year when July starts for this data
# Dec 1999 FYE -> +6mo = June 2000 -> July 2000 is the start -> portfolio_year = 2000

# For Fama-French alignment: the portfolio_year is when July falls
# If datadate + 6 months lands in Jan-June of year Y, July is in year Y
# If datadate + 6 months lands in July-Dec of year Y, July is in year Y+1
compustat['portfolio_year'] = np.where(
    compustat['portfolio_start'].dt.month >= 7,
    compustat['portfolio_start'].dt.year + 1,  # July-Dec → next July
    compustat['portfolio_start'].dt.year        # Jan-June → this July
)

print(f"Portfolio years range: {compustat['portfolio_year'].min()} to {compustat['portfolio_year'].max()}")

# =============================================================================
# Step 3: Classify firms as R&D or No-R&D
# =============================================================================

# Dynamic classification based on t-1 R&D:
# - If xrd > 0 -> R&D portfolio
# - If xrd is missing or 0 -> No R&D portfolio

print("\nClassifying firms into R&D vs No-R&D portfolios...")

compustat['has_rd'] = (compustat['xrd'].notna()) & (compustat['xrd'] > 0)

rd_counts = compustat.groupby('portfolio_year')['has_rd'].agg(['sum', 'count'])
rd_counts['pct_rd'] = 100 * rd_counts['sum'] / rd_counts['count']
print(f"Average % of firms with R&D across years: {rd_counts['pct_rd'].mean():.1f}%")

# Debug: Check portfolio year range in Compustat
print(f"\nCompustat portfolio years available:")
print(f"  Min portfolio_year: {compustat['portfolio_year'].min()}")
print(f"  Max portfolio_year: {compustat['portfolio_year'].max()}")
print(f"  Years with data: {sorted(compustat['portfolio_year'].unique())[:10]}")  # Show first 10 years

# Keep only the columns we need for merging (include link dates for filtering)
compustat_slim = compustat[['lpermno', 'portfolio_year', 'has_rd', 'datadate', 'linkdt', 'linkenddt']].copy()
compustat_slim = compustat_slim.rename(columns={'lpermno': 'permno'})

# =============================================================================
# Step 4: Assign CRSP returns to portfolio years
# =============================================================================

# Each monthly return belongs to a portfolio year based on:
# - July year Y to June year Y+1 -> portfolio_year = Y
# This matches the Fama-French convention

print("\nAssigning CRSP returns to portfolio years...")

crsp['date'] = pd.to_datetime(crsp['date'])
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month

# Determine portfolio year for each return
# July-Dec of year Y -> portfolio_year = Y
# Jan-June of year Y -> portfolio_year = Y-1
crsp['portfolio_year'] = np.where(
    crsp['month'] >= 7,
    crsp['year'],
    crsp['year'] - 1
)

print(f"CRSP portfolio years range: {crsp['portfolio_year'].min()} to {crsp['portfolio_year'].max()}")

# Debug: Show overlap
print(f"\nChecking Compustat-CRSP portfolio year overlap:")
compustat_years = set(compustat_slim['portfolio_year'].unique())
crsp_years = set(crsp['portfolio_year'].unique())
overlap_years = sorted(compustat_years & crsp_years)
print(f"  Overlapping years: {overlap_years[:5]} ... {overlap_years[-5:]}")
print(f"  Total overlapping years: {len(overlap_years)}")

# =============================================================================
# Step 5: Merge CRSP returns with R&D classification
# =============================================================================

print("\nMerging CRSP with R&D classification...")

# Merge on permno and portfolio_year
# This assigns each firm-month to R&D or No-R&D based on prior year's R&D
merged = crsp.merge(
    compustat_slim[['permno', 'portfolio_year', 'has_rd', 'linkdt', 'linkenddt']],
    on=['permno', 'portfolio_year'],
    how='inner'
)

print(f"Initial merge rows: {len(merged)}")

# =============================================================================
# Step 5b: Apply CCM Link Date Filtering
# =============================================================================

print("\nApplying CCM link date filtering...")

# Filter: CRSP date must be within link validity period
# This ensures we only use valid GVKEY-PERMNO links
initial_count = len(merged)
merged = merged[
    (merged['date'] >= merged['linkdt']) &
    (merged['date'] <= merged['linkenddt'])
]

print(f"After CCM link date filtering: {len(merged)} rows ({initial_count - len(merged)} removed)")
print(f"Unique firms in merged data: {merged['permno'].nunique()}")

# Drop link date columns - no longer needed
merged = merged.drop(['linkdt', 'linkenddt'], axis=1)

# Check the split
rd_obs = merged['has_rd'].sum()
no_rd_obs = len(merged) - rd_obs
print(f"Observations in R&D portfolio: {rd_obs} ({100*rd_obs/len(merged):.1f}%)")
print(f"Observations in No-R&D portfolio: {no_rd_obs} ({100*no_rd_obs/len(merged):.1f}%)")

# =============================================================================
# Step 6: Get market cap at portfolio formation for value weights
# =============================================================================

print("\nGetting market cap for value weights...")

# We need market cap at the start of the holding period (June of portfolio_year)
# Create a lookup for June market caps
june_mktcap = crsp[crsp['month'] == 6][['permno', 'year', 'mktcap']].copy()
june_mktcap = june_mktcap.rename(columns={'year': 'portfolio_year', 'mktcap': 'weight_mktcap'})

# Merge market cap weights
merged = merged.merge(
    june_mktcap,
    on=['permno', 'portfolio_year'],
    how='left'
)

# Drop rows without market cap (can't include in value-weighted portfolio)
merged_vw = merged[merged['weight_mktcap'].notna()].copy()
print(f"Rows with valid market cap for VW: {len(merged_vw)}")

# =============================================================================
# Step 7: Calculate portfolio returns
# =============================================================================

print("\nCalculating portfolio returns...")

# Use ret_adj (return adjusted for delisting) as the return measure
# Group by date and R&D status

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

print(f"Portfolio returns calculated for {len(portfolio_returns)} months")
print(f"Date range: {portfolio_returns['date'].min()} to {portfolio_returns['date'].max()}")

# =============================================================================
# Step 8: Summary statistics
# =============================================================================

print("\n" + "="*60)
print("PORTFOLIO SUMMARY STATISTICS (Monthly Returns)")
print("="*60)

stats = portfolio_returns[['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']].describe()
print(stats)

print("\nAnnualized Returns (mean * 12):")
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    ann_ret = portfolio_returns[col].mean() * 12 * 100
    print(f"  {col}: {ann_ret:.2f}%")

# =============================================================================
# Step 9: Save portfolio returns
# =============================================================================

portfolio_returns.to_csv('portfolio_returns.csv', index=False)
portfolio_returns.to_parquet('portfolio_returns.parquet', index=False)
print("\nPortfolio returns saved to portfolio_returns.csv and portfolio_returns.parquet")

# Also save the merged firm-level data for potential further analysis
merged.to_parquet('firm_monthly_returns.parquet', index=False)
print("Firm-level monthly returns saved to firm_monthly_returns.parquet")

# =============================================================================
# Step 10: Load CRSP Value-Weighted Market Index and Risk-Free Rate
# =============================================================================

print("\n" + "="*60)
print("LOADING CRSP VALUE-WEIGHTED MARKET INDEX & RISK-FREE RATE")
print("="*60)

# Load market index and risk-free rate from saved data
market_data = pd.read_parquet('market_index.parquet')
market_data['date'] = pd.to_datetime(market_data['date'])

print(f"Market data rows: {len(market_data)}")
print(f"Date range: {market_data['date'].min()} to {market_data['date'].max()}")
print(f"Average monthly market return: {market_data['vwretd'].mean()*100:.3f}%")
print(f"Average monthly RF rate: {market_data['rf'].mean()*100:.3f}%")

# Normalize dates to end of month for proper matching
portfolio_returns['date'] = pd.to_datetime(portfolio_returns['date'])
portfolio_returns['date'] = portfolio_returns['date'] + pd.offsets.MonthEnd(0)
market_data['date'] = market_data['date'] + pd.offsets.MonthEnd(0)

# Merge portfolio returns with market data
portfolio_with_market = portfolio_returns.merge(market_data, on='date', how='inner')
print(f"Portfolio returns with market data: {len(portfolio_with_market)} months")

# =============================================================================
# Step 11: Calculate Excess Returns (Portfolio - RF and Market - RF)
# =============================================================================

print("\nCalculating excess returns for CAPM regression...")

# Ensure numeric types
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD', 'vwretd', 'rf']:
    portfolio_with_market[col] = pd.to_numeric(portfolio_with_market[col], errors='coerce')

# Drop rows with any NaN values in key columns
portfolio_with_market = portfolio_with_market.dropna(subset=['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD', 'vwretd', 'rf'])
print(f"Rows after dropping NaN: {len(portfolio_with_market)}")

# Calculate excess returns: Return - Risk-free rate
# This is the theoretically correct approach for CAPM
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    portfolio_with_market[f'{col}_excess'] = portfolio_with_market[col] - portfolio_with_market['rf']

# Market excess return (market risk premium)
portfolio_with_market['market_excess'] = portfolio_with_market['vwretd'] - portfolio_with_market['rf']

# Display sample
print("\nSample of excess returns (first 5 months):")
print(portfolio_with_market[['date', 'EW_NoRD_excess', 'EW_RD_excess', 'VW_NoRD_excess', 'VW_RD_excess', 'market_excess']].head())
print(f"\nAverage monthly market premium: {portfolio_with_market['market_excess'].mean()*100:.3f}%")

# =============================================================================
# Step 12: Run CAPM Regressions (Jensen's Alpha)
# =============================================================================

print("\n" + "="*60)
print("CAPM ALPHA REGRESSIONS (JENSEN'S ALPHA)")
print("="*60)
print("Model: (R_p - R_f) = alpha + beta * (R_m - R_f) + epsilon")
print("="*60)

import statsmodels.api as sm

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
    alpha_monthly = results.params[0]  # Intercept = Jensen's alpha
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
# Step 13: Summary Table
# =============================================================================

print("\n" + "="*60)
print("CAPM ALPHA SUMMARY TABLE")
print("="*60)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('alpha_summary.csv', index=False)
print("\n" + "="*60)
print("Summary saved to alpha_summary.csv")
print("="*60)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
