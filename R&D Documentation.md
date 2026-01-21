### Assignment 0: R&D Portfolio Analysis

Create two portfolios with and without reported R&D Expenditure in year t-1. Construct value and equally weighted portfolios and calculate their alphas with respect to the CRSP value-weighted index as well as their signifigance.

#### Overview

Pull from Compustat and CRSP

- Apply the appropriate filters
  -- Stock exchanges: major US exchanges (NYSE, AMEX, NASDAQ)
  -- Exclude financials and pharmaceuticals
  -- US only companies
  -- Remove unusual returns
- Pull monthly returns from CRSP
- Pull t-1 R&D from Compustat
- From 1980 to 2022 (effective sample: 1985-2022 due to early data availability)

CRSP uses PERMNO and Compustat uses GVKEY as their unique identifier
Merge them using CCM link table with date filtering

#### R&D Classification Logic (Dynamic Approach)

- Originally thought about keeping same tickers throughout but this causes look-ahead bias
- Example: firm reports 0 R&D 1980-1999 then positive R&D 2000+, can't classify as R&D firm in 1985 using info from 2000
- Solution: dynamic rebalancing each year based on t-1 R&D
  -- If t-1 R&D > 0 → R&D portfolio
  -- If t-1 R&D is missing or 0 → No R&D portfolio
  -- Firms can move between portfolios over time, this is fine

Missing R&D Handling

- If xrd is missing (NaN) → treat as No R&D portfolio
- If xrd = 0 → treat as No R&D portfolio
- If xrd > 0 → treat as R&D portfolio
- No special handling for "always 0" vs "sometimes 0" - both go to No R&D
- Dynamic classification each year means firms can switch portfolios

#### Accounting for Look-Ahead Bias

- t-1 for R&D
- +6 months to account for 10K disclosure timing
- Fiscal year ends (e.g., Dec 31, 2000) but investors don't know R&D yet
- SEC gives companies up to 6 months to file 10-K
- So Dec 2000 fiscal year → 10-K filed by June 2001 → portfolio formed July 1, 2001
- Holding period: July 2001 - June 2002
- Then rebalance July 2002 using fiscal year 2001 data

#### Portfolio Construction

From there, create 4 portfolios

- 2 equal weighted, 1 R&D 1 no R&D
- 2 value weighted, 1 R&D, 1 no R&D

##### Portfolio Rebalancing

- Portfolios rebalanced annually on July 1st (Fama-French convention)
- Use most recent fiscal year data that's at least 6 months old
- Weights fixed for entire 12-month holding period (July year Y to June year Y+1)
- Equal-weighted: 1/N weight for each firm, recomputed each July
- Value-weighted: market cap weights from June, held constant for 12 months

##### Non-December Fiscal Year-End Firms

- Not all firms have Dec fiscal year-end, some have June, March, Sept, etc.
- Standard approach (Fama-French style): form portfolios once per year in July using most recent fiscal year data that's at least 6 months old
- Example for July 1, 2001 portfolio formation:
  -- Dec FYE firm: use Dec 2000 data (filed by June 2001) → 6 months old
  -- June FYE firm: use June 2000 data (filed by Dec 2000) → 12 months old
  -- March FYE firm: March 2001 not available yet (filed by Sept 2001), use March 2000 → 15 months old
- Yes firms get treated slightly differently in terms of data "freshness" but this is the accepted approach
- Avoids look-ahead bias for ALL firms, simple (one rebalance per year), and comparable to other research

##### Portfolio Year Assignment Logic

Our approach (different from Nathan's but avoids look-ahead bias):

Compustat Side (when R&D data becomes available for portfolios):

- datadate = fiscal year end date
- portfolio_start = datadate + 6 months (when 10-K is filed)
- If portfolio_start month is July-Dec → portfolio_year = portfolio_start.year + 1
- If portfolio_start month is Jan-June → portfolio_year = portfolio_start.year
- Example: Dec 2000 FYE → +6mo = June 2001 → month 6 (Jan-June) → portfolio_year = 2001
- This means the R&D data is used in July 2001 - June 2002 holding period

CRSP Side (assigning monthly returns to portfolio years):

- If return month is July-Dec → portfolio_year = return.year
- If return month is Jan-June → portfolio_year = return.year - 1
- Example: August 2001 return → month 8 (July-Dec) → portfolio_year = 2001
- Example: March 2002 return → month 3 (Jan-June) → portfolio_year = 2001
- Both August 2001 and March 2002 use the same R&D data (from Dec 2000 fiscal year)

Why this works:

- July 2001 - June 2002 returns all get portfolio_year = 2001
- Dec 2000 fiscal year data also gets portfolio_year = 2001
- They match on portfolio_year → correct association with 6-month lag

#### Merging CRSP and Compustat (Link Table Logic)

- CRSP uses PERMNO, Compustat uses GVKEY
- CCM link table bridges them: GVKEY ↔ LPERMNO
- Not a simple join, need date conditions:
  -- LINKDT <= datadate <= LINKENDDT (or LINKENDDT is missing meaning still active)
  -- If LINKENDDT is missing, treat as still active through 2099-12-31
  -- INNER JOIN (only keep firms with valid CRSP links)
- Filter for link quality:
  -- LINKTYPE in ('LC', 'LU') - confirmed or usable links
  -- LINKPRIM in ('P', 'C') - primary security only

##### CCM Link Date Filtering in Portfolio Construction

- After merging CRSP returns with R&D classification, filter again:
  -- Ensure CRSP date falls within link validity period: LINKDT <= date <= LINKENDDT
  -- This handles cases where link changes over time (mergers, restatements, etc.)
  -- Prevents using wrong GVKEY-PERMNO mapping for a given date
- Example: if PERMNO 12345 linked to GVKEY 001234 from 1990-2000, then to GVKEY 005678 from 2001-2010
  -- Returns from 1995 should only use R&D from GVKEY 001234
  -- Returns from 2005 should only use R&D from GVKEY 005678

#### Delisted Returns (Survivorship Bias)

- When firms delist (bankruptcy, merger, etc.) regular RET often missing
- CRSP provides DLRET (delisting return) to capture final return
- If you ignore DLRET you drop firms right before they fail → survivorship bias
- Fix: if RET missing but DLRET exists → use DLRET
- If both exist → compound them: (1 + RET) \* (1 + DLRET) - 1
- Create ret_adj = delisting-adjusted return
- Use ret_adj (not ret) for all portfolio calculations

#### Market Cap for Value Weights

- Need market cap to calculate value-weighted returns
- Use market cap at portfolio formation date (end of June) to avoid look-ahead bias
- Market cap = |PRC| \* SHROUT from CRSP
  -- PRC can be negative (indicates bid/ask midpoint), take absolute value
  -- SHROUT = shares outstanding in thousands
- Weights stay fixed for the 12-month holding period
- Drop observations without valid June market cap for VW portfolios
- Why June? Portfolio forms July 1st, June is the last complete month of data

#### Portfolio Return Calculations

Equal weighted:
R_portfolio = (1/N) × Σ R_i

- Just the simple average of all firm returns

Value weighted:
R_portfolio = Σ (w_i × R_i)
where w_i = MarketCap_i / Σ MarketCap

- Bigger firms (by market cap) have more influence on portfolio returns

#### Data Cleaning and Filters

Return Filters:

- Remove returns where ret is missing (NaN)
- Remove returns < -100% (ret < -1.0 in decimal form)
- Convert ret and dlret to numeric with pd.to_numeric(errors='coerce')
- Missing values after conversion treated as NaN

Compustat Filters:

- curcd = 'USD' (only US dollar reporting)
- fic = 'USA' (only US incorporated)
- exchg between 11 and 19 (major US exchanges)
  -- 11 = NYSE
  -- 12 = NYSE MKT (formerly AMEX)
  -- 13 = NASDAQ (National Market System)
  -- 14 = NASDAQ (Small Cap)
  -- 19 = Other NASDAQ
- sich not in 6000-6999 (exclude financials)
- sich != 2834 (exclude pharmaceuticals)
- indfmt = 'INDL' (industrial format, standard for most firms)
- datafmt = 'STD' (standardized data format)
- popsrc = 'D' (domestic population)
- consol = 'C' (consolidated financial statements)
- Fiscal years 1979-2021 (need 1979 for t-1 data in 1980)

CRSP Filters:

- hsiccd not in 6000-6999 (exclude financials)
- hsiccd != 2834 (exclude pharmaceuticals)
- Date range 1980-01-01 to 2022-12-31
- Note: filtering on both sich (Compustat) and hsiccd (CRSP) for consistency

Date Normalization:

- Normalize all dates to month-end using pd.offsets.MonthEnd(0)
- Ensures proper matching between portfolio returns and market data
- Example: 2001-06-29 and 2001-06-30 both become 2001-06-30

WRDS Data Pull (Python + wrds library)

Compustat Annual Fundamentals: comp.funda

- gvkey: firm identifier
- datadate: fiscal year end date
- fyear: fiscal year
- xrd: R&D expense (this is what we care about)
- sich: SIC code (for filtering out financials 6000-6999 and pharma 2834)
- curcd: currency (filter for USD)
- fic: country of incorporation (filter for USA)
- exchg: stock exchange code (filter for 11-19 = major US exchanges)

CRSP Monthly Stock File: crsp.msf

- permno: security identifier
- date: month-end trading date
- ret: monthly holding period return
- prc: price (need absolute value, sometimes negative to indicate bid/ask avg)
- shrout: shares outstanding
- hsiccd: historical SIC code (filter for financials and pharma)

Delisting Returns: crsp.msedelist

- permno: security identifier
- dlstdt: delisting date
- dlret: delisting return

CRSP Market Index & Risk-Free Rate:

- CRSP value-weighted market index from crsp.msi (vwretd)
- Risk-free rate from crsp.mcti (t30ret = 30-day T-bill rate)
- Used for CAPM alpha calculations

CCM Link Table: crsp.ccmxpf_lnkhist

- gvkey: Compustat identifier
- lpermno: linked PERMNO (CRSP identifier)
- linkdt: link start date
- linkenddt: link end date (missing = still active)
- linktype: filter for LC, LU
- linkprim: filter for P, C

Note: verify exact table names against WRDS documentation, sometimes they change or have alternatives

Merge Strategy

Data Pull Stage (assignment_0_data_pull.py):

- Compustat + CCM: INNER JOIN (only firms with valid CRSP links)
- CRSP + Delisting Returns: LEFT JOIN (keep all returns, add dlret when available)
- Market Index + Risk-Free Rate: LEFT JOIN (keep all market dates, add RF)

Portfolio Construction Stage (portfolio_construction.py):

- CRSP + R&D Classification: INNER JOIN (only firm-months with both returns AND R&D data)
- Add Market Cap: LEFT JOIN (keep all observations, add June mktcap when available)
- Portfolio Returns + Market Data: INNER JOIN (only months with both portfolio and benchmark)

#### CAPM Alpha Regression (Fama-French Method)

What is Alpha?

- Alpha measures risk-adjusted abnormal returns
- Alpha = intercept from CAPM regression using excess returns
- Tells us if portfolio beats market after adjusting for risk (beta)

CAPM Regression Model:
(R_p - R_f) = α + β(R_m - R_f) + ε

Where:

- R_p = portfolio return
- R_f = risk-free rate (1-month T-bill)
- R_m = market return (CRSP value-weighted index)
- α = alpha (abnormal return / risk-adjusted performance)
- β = market beta (systematic risk)
- ε = error term

Step-by-Step Implementation:

1. Calculate Excess Returns:

   - Portfolio excess return = R_p - R_f
   - Market excess return (market premium) = R_m - R_f
   - This adjusts for the time value of money

2. Run OLS Regression:

   - Dependent variable (y): Portfolio excess return
   - Independent variable (X): Market excess return
   - Add constant term (intercept)

3. Extract Results:

   - Alpha = regression intercept (what we care about!)
   - Beta = regression slope (market sensitivity)
   - t-statistic = alpha / standard error of alpha
   - p-value = statistical significance of alpha
   - R² = proportion of variance explained by market

4. Annualize Alpha:
   - Monthly alpha × 12 = annual alpha
   - t-statistic stays the same (already standardized)

Why This Methodology?

- Proper CAPM framework adjusts for market risk
- Simple mean difference test ignores beta differences
- OLS regression gives us proper t-statistics and p-values
- Excess returns account for risk-free rate (theoretically correct)
- Matches academic standard (Fama-French methodology)

Interpretation:

- Positive significant alpha → portfolio beats market on risk-adjusted basis
- Beta near 1.0 → portfolio moves with market
- Beta > 1.0 → portfolio is riskier than market (amplifies market moves)
- High R² → most portfolio variance explained by market movements

Sample Period Explanation

- Requested period: 1980-2022
- Actual portfolio period: July 1985 - December 2022 (450 months)
- Why the difference?
  -- Early 1980s data is sparse in Compustat after all filters applied
  -- Only 4 firm-year observations in 1982-1985 period
  -- First meaningful portfolio formation: July 1985
  -- This is a data availability limitation, not a code error
- We still pull Compustat from 1979 and CRSP from 1980 to capture all available data
- Decision: keep all observations even if early years have low firm counts

#### Results

**Sample Period:** July 1985 - December 2022 (450 months)

**Raw Returns (Annualized):**

| Portfolio | Annualized Return |
| --------- | ----------------- |
| EW_NoRD   | 16.20%            |
| EW_RD     | 19.23%            |
| VW_NoRD   | 10.38%            |
| VW_RD     | 12.63%            |

_R&D firms outperform in raw returns_

**CAPM Alpha Results (vs CRSP VW Index):**

| Portfolio | Alpha (Annual) | Beta  | t-stat | p-value | R²    | Significant? |
| --------- | -------------- | ----- | ------ | ------- | ----- | ------------ |
| EW_NoRD   | 1.11%          | 1.111 | 0.57   | 0.5674  | 0.697 | No           |
| EW_RD     | 2.28%          | 1.275 | 0.92   | 0.3573  | 0.652 | No           |
| VW_NoRD   | -0.10%         | 0.964 | -0.10  | 0.9235  | 0.851 | No           |
| VW_RD     | 0.14%          | 1.098 | 0.14   | 0.8920  | 0.890 | No           |

\*Note: Significance levels: **\* p<0.01, ** p<0.05, _ p<0.10_

Key Findings:

**Equal-Weighted Portfolios:**

- Neither EW portfolio generates significant alpha (both p > 0.35)
- EW_NoRD: 1.11% annual alpha (t = 0.57, not significant)
- EW_RD: 2.28% annual alpha (t = 0.92, not significant)
- R&D premium: 1.17% (2.28% - 1.11%)
- R&D firms have higher beta (1.275 vs 1.111) → higher systematic risk
- Lower R² for R&D portfolio (0.652 vs 0.697) → more idiosyncratic risk

**Value-Weighted Portfolios:**

- Neither VW portfolio generates significant alpha
- VW_NoRD: -0.10% alpha (t = -0.10, not significant)
- VW_RD: 0.14% alpha (t = 0.14, not significant)
- Alphas essentially zero → large-cap firms are efficiently priced
- Betas near 1.0 → track market closely
- Very high R² (0.85+) → market explains most variance

Conclusion:

- **No significant alpha in any portfolio**: After controlling for market risk, no abnormal returns
- **Market efficiency holds**: Broader exchange coverage (including AMEX) shows efficient pricing
- **R&D firms have higher beta**: R&D portfolios show higher systematic risk (1.275 vs 1.111 for EW; 1.098 vs 0.964 for VW)
- **R&D premium exists but not risk-adjusted**: 1.17% higher alpha for R&D firms, but not statistically significant
- **Size effect diminished**: Including AMEX and broader NASDAQ coverage eliminates small-cap alpha

#### Assignment Checklist

- Pull from Compustat and CRSP → done
- Apply filters (major US exchanges via exchg 11-19: NYSE, AMEX, NASDAQ) → done
- Exclude financials (SIC 6000-6999) and pharma (SIC 2834) → done
- US only companies (fic='USA', curcd='USD') → done
- Pull monthly returns from CRSP (crsp.msf) → done
- Handle delisting returns to avoid survivorship bias → done
- Remove unusual returns (ret < -100%) → done
- Create 4 portfolios (2 EW, 2 VW with and without R&D) → done
- t-1 R&D with look-ahead bias handling (+6 month lag) → done
- CCM link date filtering for valid GVKEY-PERMNO mappings → done
- Alpha vs CRSP value-weighted index (CAPM with excess returns) → done
- CAPM regression with risk-free rate adjustment (Fama-French method) → done
- Calculate significance (t-stats and p-values from OLS regression) → done
