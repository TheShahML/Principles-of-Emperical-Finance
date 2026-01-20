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

# Portfolio year = the year in which we form the portfolio
# e.g., Dec 2000 fiscal year -> portfolio starts July 2001 -> portfolio_year = 2001
compustat['portfolio_year'] = compustat['portfolio_start'].dt.year

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

# Keep only the columns we need for merging
compustat_slim = compustat[['lpermno', 'portfolio_year', 'has_rd', 'datadate']].copy()
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

# =============================================================================
# Step 5: Merge CRSP returns with R&D classification
# =============================================================================

print("\nMerging CRSP with R&D classification...")

# Merge on permno and portfolio_year
# This assigns each firm-month to R&D or No-R&D based on prior year's R&D
merged = crsp.merge(
    compustat_slim[['permno', 'portfolio_year', 'has_rd']],
    on=['permno', 'portfolio_year'],
    how='inner'
)

print(f"Merged rows: {len(merged)}")
print(f"Unique firms in merged data: {merged['permno'].nunique()}")

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
# Step 10: Pull Fama-French Factors and Risk-Free Rate
# =============================================================================

print("\n" + "="*60)
print("PULLING FAMA-FRENCH FACTORS")
print("="*60)

import wrds
conn = wrds.Connection()

# Pull Fama-French 3 factors (includes market excess return and risk-free rate)
ff_query = """
    SELECT date, mktrf, smb, hml, rf
    FROM ff.factors_monthly
    WHERE date BETWEEN '1980-01-01' AND '2022-12-31'
"""

print("Pulling Fama-French factors...")
ff_factors = conn.raw_sql(ff_query)
ff_factors['date'] = pd.to_datetime(ff_factors['date'])

# Check FF factors scale - print sample values
print(f"DEBUG - FF factor sample values BEFORE scaling:")
print(ff_factors[['mktrf', 'smb', 'hml', 'rf']].head())
print(f"DEBUG - mktrf mean: {ff_factors['mktrf'].mean()}")

# If mktrf mean is > 0.5, factors are in percentage terms (e.g., 1.5 = 1.5%)
# If mktrf mean is < 0.05, factors are already in decimal form (e.g., 0.015 = 1.5%)
if ff_factors['mktrf'].mean() > 0.5:
    print("Factors appear to be in percentage terms, converting to decimal...")
    for col in ['mktrf', 'smb', 'hml', 'rf']:
        ff_factors[col] = ff_factors[col] / 100
else:
    print("Factors appear to already be in decimal form, no conversion needed.")

conn.close()
print(f"Fama-French factors: {len(ff_factors)} months")

# Merge factors with portfolio returns
portfolio_returns['date'] = pd.to_datetime(portfolio_returns['date'])

# Debug: check date formats
print(f"\nDEBUG - Portfolio returns date sample: {portfolio_returns['date'].head()}")
print(f"DEBUG - FF factors date sample: {ff_factors['date'].head()}")

# Normalize both to end of month for proper matching
portfolio_returns['date'] = portfolio_returns['date'] + pd.offsets.MonthEnd(0)
ff_factors['date'] = ff_factors['date'] + pd.offsets.MonthEnd(0)

print(f"DEBUG - After normalization:")
print(f"DEBUG - Portfolio returns date sample: {portfolio_returns['date'].head()}")
print(f"DEBUG - FF factors date sample: {ff_factors['date'].head()}")

portfolio_with_factors = portfolio_returns.merge(ff_factors, on='date', how='inner')
print(f"Portfolio returns with factors: {len(portfolio_with_factors)} months")

# =============================================================================
# Step 11: Calculate Excess Returns
# =============================================================================

print("\nCalculating excess returns...")

# Ensure numeric types
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD', 'mktrf', 'smb', 'hml', 'rf']:
    portfolio_with_factors[col] = pd.to_numeric(portfolio_with_factors[col], errors='coerce')

# Drop rows with any NaN values in key columns
portfolio_with_factors = portfolio_with_factors.dropna(subset=['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD', 'mktrf', 'smb', 'hml', 'rf'])
print(f"Rows after dropping NaN: {len(portfolio_with_factors)}")

# Excess return = portfolio return - risk-free rate
for col in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    portfolio_with_factors[f'{col}_excess'] = portfolio_with_factors[col] - portfolio_with_factors['rf']

# =============================================================================
# Step 12: Run Alpha Regressions (CAPM)
# =============================================================================

print("\n" + "="*60)
print("CAPM ALPHA REGRESSIONS")
print("="*60)
print("Model: R_portfolio - Rf = alpha + beta * (R_market - Rf) + epsilon")
print("="*60)

import statsmodels.api as sm

def run_capm_regression(excess_returns, mktrf):
    """Run CAPM regression and return results"""
    # Convert to numpy arrays to avoid dtype issues
    y = np.asarray(excess_returns, dtype=float)
    x = np.asarray(mktrf, dtype=float)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model

capm_results = {}
for portfolio in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    excess_col = f'{portfolio}_excess'
    model = run_capm_regression(
        portfolio_with_factors[excess_col],
        portfolio_with_factors['mktrf']
    )
    capm_results[portfolio] = model

    # params[0] is const/alpha, params[1] is beta
    alpha = model.params[0]
    alpha_tstat = model.tvalues[0]
    alpha_pval = model.pvalues[0]
    beta = model.params[1]
    r_squared = model.rsquared

    print(f"\n{portfolio}:")
    print(f"  Alpha (monthly): {alpha:.4f} ({alpha*12:.4f} annualized)")
    print(f"  Alpha t-stat:    {alpha_tstat:.2f}")
    print(f"  Alpha p-value:   {alpha_pval:.4f}")
    print(f"  Beta:            {beta:.3f}")
    print(f"  R-squared:       {r_squared:.3f}")

    # Significance indicator
    if alpha_pval < 0.01:
        sig = "***"
    elif alpha_pval < 0.05:
        sig = "**"
    elif alpha_pval < 0.10:
        sig = "*"
    else:
        sig = ""
    print(f"  Significance:    {sig} (*** p<0.01, ** p<0.05, * p<0.10)")

# =============================================================================
# Step 13: Run Alpha Regressions (Fama-French 3-Factor)
# =============================================================================

print("\n" + "="*60)
print("FAMA-FRENCH 3-FACTOR ALPHA REGRESSIONS")
print("="*60)
print("Model: R_p - Rf = alpha + b1*MktRf + b2*SMB + b3*HML + epsilon")
print("="*60)

def run_ff3_regression(excess_returns, mktrf, smb, hml):
    """Run Fama-French 3-factor regression and return results"""
    # Convert to numpy arrays to avoid dtype issues
    y = np.asarray(excess_returns, dtype=float)
    X = np.column_stack([
        np.asarray(mktrf, dtype=float),
        np.asarray(smb, dtype=float),
        np.asarray(hml, dtype=float)
    ])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

ff3_results = {}
for portfolio in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    excess_col = f'{portfolio}_excess'
    model = run_ff3_regression(
        portfolio_with_factors[excess_col],
        portfolio_with_factors['mktrf'],
        portfolio_with_factors['smb'],
        portfolio_with_factors['hml']
    )
    ff3_results[portfolio] = model

    # params[0] is const/alpha, params[1-3] are factor betas
    alpha = model.params[0]
    alpha_tstat = model.tvalues[0]
    alpha_pval = model.pvalues[0]

    print(f"\n{portfolio}:")
    print(f"  Alpha (monthly): {alpha:.4f} ({alpha*12:.4f} annualized)")
    print(f"  Alpha t-stat:    {alpha_tstat:.2f}")
    print(f"  Alpha p-value:   {alpha_pval:.4f}")
    print(f"  MktRf beta:      {model.params[1]:.3f}")
    print(f"  SMB beta:        {model.params[2]:.3f}")
    print(f"  HML beta:        {model.params[3]:.3f}")
    print(f"  R-squared:       {model.rsquared:.3f}")

    # Significance indicator
    if alpha_pval < 0.01:
        sig = "***"
    elif alpha_pval < 0.05:
        sig = "**"
    elif alpha_pval < 0.10:
        sig = "*"
    else:
        sig = ""
    print(f"  Significance:    {sig}")

# =============================================================================
# Step 14: Summary Table
# =============================================================================

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

summary_data = []
for portfolio in ['EW_NoRD', 'EW_RD', 'VW_NoRD', 'VW_RD']:
    capm = capm_results[portfolio]
    ff3 = ff3_results[portfolio]

    # Use index 0 for const/alpha (numpy array indexing)
    summary_data.append({
        'Portfolio': portfolio,
        'CAPM Alpha (ann.)': f"{capm.params[0]*12:.4f}",
        'CAPM t-stat': f"{capm.tvalues[0]:.2f}",
        'FF3 Alpha (ann.)': f"{ff3.params[0]*12:.4f}",
        'FF3 t-stat': f"{ff3.tvalues[0]:.2f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('alpha_summary.csv', index=False)
print("\nSummary saved to alpha_summary.csv")
