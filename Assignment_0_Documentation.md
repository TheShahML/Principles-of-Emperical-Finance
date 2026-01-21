### Assignment 0: R&D Portfolio Analysis

**Complete Documentation - Aligned with assignment_0.py**

Create portfolios with and without reported R&D Expenditure in year t-1. Construct value and equally weighted portfolios and calculate their alphas with respect to the CRSP value-weighted index using the Fama-French method.

---

## Overview

**Sample Period:** 1980-2022 (effective sample: 1985-2022 due to early data availability)

**Data Sources:**

- Compustat Annual Fundamentals (comp.funda)
- CRSP Monthly Stock File (crsp.msf)
- CRSP Delisting Returns (crsp.msedelist)
- CRSP Market Index (crsp.msi)
- CRSP Risk-Free Rate (crsp.mcti)
- CCM Link Table (crsp.ccmxpf_lnkhist)

**Methodology:** CAPM Alpha using Fama-French method with excess returns

**Output Files:**

- `portfolio_returns.csv` - Monthly portfolio returns (EW and VW, R&D and No-R&D)
- `alpha_summary.csv` - CAPM alpha regression results

---

## PART 1: DATA COLLECTION FROM WRDS

### Step 1: Connect to WRDS

- Uses `wrds.Connection()` to establish connection
- Prompts for credentials on first run, then caches them
- Connection used throughout data pull, then closed after Step 10

### Step 2: Pull Compustat Annual Fundamentals

**Query Parameters:**

- `fyear BETWEEN 1979 AND 2021` - Need 1979 for t-1 data in 1980
- `curcd = 'USD'` - Only US dollar reporting
- `fic = 'USA'` - Only US incorporated companies
- `exchg BETWEEN 11 AND 19` - Major US exchanges:
  - 11 = NYSE
  - 12 = NYSE MKT (formerly AMEX)
  - 13 = NASDAQ (National Market System)
  - 14 = NASDAQ (Small Cap)
  - 19 = Other NASDAQ
- `sich NOT IN (6000-6999)` - Exclude financials
- `sich != 2834` - Exclude pharmaceuticals
- `indfmt = 'INDL'` - Industrial format (standard)
- `datafmt = 'STD'` - Standardized data format
- `popsrc = 'D'` - Domestic population
- `consol = 'C'` - Consolidated financial statements

**Key Variables:**

- `gvkey` - Compustat firm identifier
- `datadate` - Fiscal year end date
- `fyear` - Fiscal year
- `xrd` - R&D expense (our key variable)
- `sich` - SIC code for industry filtering

### Step 3: Pull CCM Link Table

**Purpose:** Links Compustat (gvkey) to CRSP (permno)

**Query Parameters:**

- `linktype IN ('LC', 'LU')` - Confirmed or usable links only
- `linkprim IN ('P', 'C')` - Primary security only

**Key Variables:**

- `gvkey` - Compustat identifier
- `lpermno` - Linked PERMNO (CRSP identifier)
- `linkdt` - Link start date
- `linkenddt` - Link end date (missing = still active)

**Why CCM Links Matter:**

- Not a simple one-to-one mapping
- Links can change over time (mergers, restatements)
- Need to filter by date validity to ensure correct mapping

### Step 4: Merge Compustat with CCM Link Table

**Merge Logic:**

- INNER JOIN on `gvkey`
- Filter: `linkdt <= datadate <= linkenddt`
- Handle missing `linkenddt` by filling with 2099-12-31 (still active)

**Result:** Compustat data now has PERMNO for merging with CRSP

### Step 5: Pull CRSP Monthly Stock File

**Query Parameters:**

- `date BETWEEN '1980-01-01' AND '2022-12-31'`
- `hsiccd IS NOT NULL` - Must have SIC code
- `hsiccd NOT IN (6000-6999)` - Exclude financials
- `hsiccd != 2834` - Exclude pharmaceuticals

**Key Variables:**

- `permno` - Security identifier
- `date` - Month-end trading date
- `ret` - Monthly holding period return
- `prc` - Price (can be negative = bid/ask midpoint)
- `shrout` - Shares outstanding (in thousands)
- `hsiccd` - Historical SIC code

**Why Monthly Returns?**

- Much faster than daily data
- CRSP monthly file already has month-end prices and returns
- Sufficient granularity for portfolio analysis

### Step 6: Pull Delisting Returns

**Purpose:** Avoid survivorship bias

**The Problem:**

- When firms delist (bankruptcy, merger, etc.), regular RET is often missing
- Ignoring delisting returns drops firms right before they fail
- This creates survivorship bias (only successful firms remain)

**The Solution:**

- CRSP provides `dlret` (delisting return) in separate table
- Merge on `permno` and `date` (dlstdt = delisting date)
- LEFT JOIN to keep all CRSP returns, add dlret when available

### Step 7: Clean Monthly Returns

**Return Filters:**

- Remove rows where `ret` is missing (NaN)
- Remove returns < -100% (`ret < -1.0` in decimal form)
- Convert `ret` and `dlret` to numeric with `pd.to_numeric(errors='coerce')`

**Delisting Adjustment:**

- If both `ret` and `dlret` exist: `ret_adj = (1 + ret) * (1 + dlret) - 1`
- If only `ret` exists: `ret_adj = ret`
- Use `ret_adj` for all portfolio calculations

**Market Cap Calculation:**

- `mktcap = |prc| * shrout`
- Take absolute value of `prc` (can be negative for bid/ask average)
- `shrout` is in thousands, already accounts for units

**Date Variables:**

- Extract `year` and `month` from date for portfolio assignment

### Step 8: Pull CRSP Value-Weighted Market Index

**Source:** `crsp.msi` table

**Key Variable:**

- `vwretd` - Value-weighted return (including dividends)

**Why Value-Weighted?**

- Market-cap weighted, represents overall market performance
- Standard benchmark for CAPM
- Matches what Fama-French use

### Step 9: Pull Risk-Free Rate (1-Month T-Bill)

**Source:** `crsp.mcti` table

**Key Variable:**

- `t30ret` - 30-day T-bill return (closest to 1-month)

**Data Cleaning:**

- Check if stored as percentage or decimal
- Convert to decimal if needed (`rf / 100`)
- Forward fill any missing values (rare)

**Why Risk-Free Rate?**

- Theoretically correct CAPM specification uses excess returns
- R_p - R_f removes time value of money
- Standard in Fama-French methodology

**Merge with Market Index:**

- LEFT JOIN on date to combine market return and RF rate
- Creates single dataset with both benchmark components

### Step 10: Close WRDS Connection

**Actions:**

- Close WRDS connection with `conn.close()`
- Optionally save intermediate CSV files if `SAVE_INTERMEDIATE_CSV=True`
- Intermediate files: compustat_linked.csv, crsp_monthly.csv, market_index.csv

**Why Optional Intermediate Files?**

- Raw data is large and rarely needed after processing
- Can enable if debugging or inspecting data quality
- Saves disk space when disabled

---

## PART 2: PORTFOLIO CONSTRUCTION

### Step 11: Create Portfolio Formation Dates

**Look-Ahead Bias Prevention:**

- Fiscal year ends (e.g., Dec 31, 2000) but investors don't know R&D yet
- SEC gives companies up to 6 months to file 10-K
- Solution: `portfolio_start = datadate + 6 months`

**Portfolio Year Logic:**

- Fama-French convention: July-June holding periods
- If `portfolio_start` is July-Dec → `portfolio_year = year + 1` (next July)
- If `portfolio_start` is Jan-June → `portfolio_year = year` (this July)

**Example:**

- Dec 2000 fiscal year → +6mo = June 2001 → portfolio_year = 2001
- R&D data used in July 2001 - June 2002 holding period

### Step 12: Classify Firms as R&D or No-R&D

**Classification Rule:**

- `has_rd = True` if `xrd` is not null AND `xrd > 0`
- `has_rd = False` if `xrd` is null OR `xrd = 0`

**Dynamic Classification:**

- Firms can move between portfolios over time
- Based on t-1 R&D each year
- Avoids look-ahead bias (can't use future R&D reporting to classify past periods)

**Example:**

- Firm reports 0 R&D in 1980-1999, then positive R&D in 2000+
- 1985 returns → use 1984 R&D → No-R&D portfolio
- 2005 returns → use 2004 R&D → R&D portfolio

**Data Preparation:**

- Keep only columns needed for merge: permno, portfolio_year, has_rd, link dates
- Rename `lpermno` to `permno` for consistency with CRSP

### Step 13: Assign CRSP Returns to Portfolio Years

**Fama-French Convention:**

- July year Y to June year Y+1 → portfolio_year = Y

**Assignment Logic:**

- If return month is July-Dec → `portfolio_year = year`
- If return month is Jan-June → `portfolio_year = year - 1`

**Example:**

- August 2001 return → month 8 (July-Dec) → portfolio_year = 2001
- March 2002 return → month 3 (Jan-June) → portfolio_year = 2001
- Both use the same R&D data (from Dec 2000 fiscal year via portfolio_year = 2001)

**Why This Works:**

- July 2001 - June 2002 returns all get portfolio_year = 2001
- Dec 2000 fiscal year also gets portfolio_year = 2001 (via Step 11 logic)
- They match on portfolio_year → correct t-1 R&D association

### Step 14: Merge CRSP with R&D Classification

**Merge Logic:**

- INNER JOIN on `permno` and `portfolio_year`
- Only keeps firm-months where we have BOTH return data AND R&D classification

**Result:**

- Each firm-month observation now has:
  - Return data from CRSP
  - R&D classification from Compustat (t-1)
  - Link validity dates for filtering

**Why INNER JOIN?**

- Only analyze firms with complete data
- Can't calculate portfolio returns without knowing R&D status
- Can't use firms without valid CRSP-Compustat link

### Step 15: Apply CCM Link Date Filtering

**Second Date Filter (After Merge):**

- Filter: `linkdt <= date <= linkenddt`
- Ensures CRSP date falls within link validity period

**Why Filter Again?**

- Step 4 filtered Compustat datadate vs link dates
- This filters CRSP monthly dates vs link dates
- Handles cases where link changes over time

**Example:**

- PERMNO 12345 linked to GVKEY 001234 from 1990-2000
- PERMNO 12345 linked to GVKEY 005678 from 2001-2010
- Returns from 1995 only use R&D from GVKEY 001234
- Returns from 2005 only use R&D from GVKEY 005678

**Cleanup:**

- Drop `linkdt` and `linkenddt` columns (no longer needed)

### Step 16: Get Market Cap at Portfolio Formation

**Purpose:** Calculate value-weighted portfolio returns

**Timing:**

- Use market cap at END of June (portfolio formation date)
- Weights stay fixed for entire 12-month holding period (July-June)
- Avoids look-ahead bias

**Implementation:**

- Filter CRSP for June observations (`month == 6`)
- Extract `permno`, `year`, `mktcap`
- Rename `year` to `portfolio_year` for merging
- LEFT JOIN with merged data

**Why June?**

- Portfolio forms July 1st
- June is the last complete month of data before formation
- Standard Fama-French timing

**Handling Missing Market Cap:**

- Create `merged_vw` dataset with only valid market caps
- Drop observations without June market cap from VW portfolios
- EW portfolios don't need market cap (equal weights)

### Step 17: Calculate Portfolio Returns

**Equal-Weighted Returns:**

- Simple average: `R_portfolio = (1/N) × Σ R_i`
- Each firm gets equal weight (1/N)
- Emphasizes small firms (they get same weight as large firms)

**Value-Weighted Returns:**

- Market-cap weighted: `R_portfolio = Σ (w_i × R_i)`
- Where `w_i = MarketCap_i / Σ MarketCap`
- Emphasizes large firms (more weight)

**Implementation:**

- Group by `date` and `has_rd`
- Apply aggregation functions
- Unstack to create columns: EW_NoRD, EW_RD, VW_NoRD, VW_RD

**Result:**

- Time series of monthly portfolio returns
- Four portfolios: 2 weighting schemes × 2 R&D classifications

### Step 18: Summary Statistics

**Annualized Returns:**

- Monthly mean × 12 = annual return
- Simple approximation (exact would be geometric)
- Sufficient for summary statistics

**Display:**

- Show annualized returns for all 4 portfolios
- Provides preliminary view of raw performance
- Not risk-adjusted yet (that comes in Part 3)

### Step 19: Save Portfolio Returns

**Always Saved:**

- `portfolio_returns.csv` - Final output with monthly returns

**Optionally Saved (if SAVE_INTERMEDIATE_CSV=True):**

- `firm_monthly_returns.csv` - Firm-level data for further analysis

**Why Save Firm-Level Data?**

- Allows additional analysis (cross-sectional tests, robustness checks)
- Large file size, only save if needed
- Portfolio-level is sufficient for this assignment

---

## PART 3: CAPM ALPHA ANALYSIS

### Step 20: Merge Portfolio Returns with Market Data

**Date Normalization:**

- Normalize both datasets to month-end using `pd.offsets.MonthEnd(0)`
- Example: 2001-06-29 and 2001-06-30 both become 2001-06-30
- Ensures proper matching across datasets

**Merge Logic:**

- INNER JOIN on `date`
- Only keeps months where we have BOTH portfolio returns AND market data

**Result:**

- Dataset with portfolio returns, market return (vwretd), and risk-free rate (rf)
- Ready for excess return calculation

### Step 21: Calculate Excess Returns (Fama-French Method)

**Fama-French CAPM Specification:**

- Uses excess returns to remove risk-free component
- Theoretically correct approach

**Calculations:**

- Portfolio excess return: `R_p - R_f`
- Market excess return: `R_m - R_f` (market risk premium)

**Implementation:**

- Create `{portfolio}_excess` columns for all 4 portfolios
- Create `market_excess` column for market risk premium
- Drop any rows with NaN values (need complete data for regression)

**Why Excess Returns?**

- Removes time value of money
- Focuses on risk premium
- Standard in academic finance (Fama-French, Carhart, etc.)

### Step 22: Run CAPM Regressions

**CAPM Model:**

```
(R_p - R_f) = α + β(R_m - R_f) + ε
```

**Where:**

- α (alpha) = abnormal return / risk-adjusted performance
- β (beta) = market sensitivity / systematic risk
- ε (epsilon) = error term / idiosyncratic risk

**Regression Setup:**

- Dependent variable (y): Portfolio excess return
- Independent variable (X): Market excess return
- Add constant term (intercept = alpha)
- Use OLS (Ordinary Least Squares)

**Results Extracted:**

- `alpha_monthly` - Intercept (monthly alpha)
- `alpha_annual` - Monthly alpha × 12 (annualized)
- `beta` - Slope (market sensitivity)
- `alpha_se` - Standard error of alpha
- `alpha_tstat` - t-statistic for alpha (alpha / SE)
- `alpha_pval` - p-value for alpha (statistical significance)
- `r_squared` - R² (proportion of variance explained by market)
- `n_obs` - Number of observations

**Significance Levels:**

- \*\*\* p < 0.01 (highly significant)
- \*\* p < 0.05 (significant)
- - p < 0.10 (marginally significant)
- (blank) p >= 0.10 (not significant)

**Interpretation:**

- **Alpha > 0 and significant:** Portfolio beats market on risk-adjusted basis
- **Alpha ≈ 0 or not significant:** No abnormal returns, market is efficient
- **Beta near 1.0:** Portfolio moves with market
- **Beta > 1.0:** Portfolio amplifies market moves (higher systematic risk)
- **High R²:** Market explains most variance (low idiosyncratic risk)

### Step 23: Create and Save Summary Table

**Summary Table Contents:**

- Portfolio name
- Alpha (monthly, annual, percentage)
- Beta
- t-statistic
- p-value
- R²
- Significance indicator

**Output:**

- Printed to console for immediate review
- Saved to `alpha_summary.csv` for records

---

## Results

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

---

## Key Findings

### Equal-Weighted Portfolios:

- Neither EW portfolio generates significant alpha (both p > 0.35)
- EW_NoRD: 1.11% annual alpha (t = 0.57, not significant)
- EW_RD: 2.28% annual alpha (t = 0.92, not significant)
- R&D premium: 1.17% (2.28% - 1.11%)
- R&D firms have higher beta (1.275 vs 1.111) → higher systematic risk
- Lower R² for R&D portfolio (0.652 vs 0.697) → more idiosyncratic risk

### Value-Weighted Portfolios:

- Neither VW portfolio generates significant alpha
- VW_NoRD: -0.10% alpha (t = -0.10, not significant)
- VW_RD: 0.14% alpha (t = 0.14, not significant)
- Alphas essentially zero → large-cap firms are efficiently priced
- Betas near 1.0 → track market closely
- Very high R² (0.85+) → market explains most variance

---

## Conclusion

- **No significant alpha in any portfolio:** After controlling for market risk, no abnormal returns
- **Market efficiency holds:** Broader exchange coverage (including AMEX) shows efficient pricing
- **R&D firms have higher beta:** R&D portfolios show higher systematic risk (1.275 vs 1.111 for EW; 1.098 vs 0.964 for VW)
- **R&D premium exists but not risk-adjusted:** 1.17% higher alpha for R&D firms, but not statistically significant
- **Size effect diminished:** Including AMEX and broader NASDAQ coverage eliminates small-cap alpha

---

## Technical Implementation Notes

### Data Quality Checks

**Compustat:**

- Remove financials (SIC 6000-6999) and pharmaceuticals (SIC 2834)
- Require US dollar reporting (curcd = 'USD')
- Require US incorporation (fic = 'USA')
- Use only consolidated statements (consol = 'C')
- Use standardized format (datafmt = 'STD')

**CRSP:**

- Remove returns < -100% (likely data errors)
- Remove missing returns
- Apply same SIC exclusions as Compustat
- Compound delisting returns with regular returns

**CCM Links:**

- Use only confirmed or usable links (LC, LU)
- Use only primary securities (P, C)
- Filter by date validity at both Compustat and CRSP levels

### Why Our Approach Differs from Simple Methods

**Look-Ahead Bias Prevention:**

- Simple approach: use R&D from same calendar year as returns
- Our approach: use R&D from prior fiscal year + 6 month lag
- Result: Realistic trading strategy (investors couldn't trade on unknown R&D)

**Portfolio Year Mapping:**

- Simple approach: merge on calendar year - 1
- Our approach: complex portfolio_year logic with July rebalancing
- Result: Fama-French standard methodology, proper non-overlapping periods

**Link Date Filtering:**

- Simple approach: merge once, assume links are stable
- Our approach: filter at both Compustat and CRSP merge points
- Result: Correct handling of link changes over time

### Common Pitfalls Avoided

1. **Survivorship Bias:** Included delisting returns
2. **Look-Ahead Bias:** Used t-1 R&D with 6-month filing lag
3. **Link Mismatches:** Filtered by link validity dates
4. **Return Outliers:** Removed returns < -100%
5. **Missing Data:** Used INNER JOIN to ensure complete data
6. **Incorrect Timing:** Used June market cap for July portfolio formation

---

## Assignment Checklist

- [x] Pull from Compustat and CRSP
- [x] Apply filters (major US exchanges via exchg 11-19: NYSE, AMEX, NASDAQ)
- [x] Exclude financials (SIC 6000-6999) and pharma (SIC 2834)
- [x] US only companies (fic='USA', curcd='USD')
- [x] Pull monthly returns from CRSP (crsp.msf)
- [x] Handle delisting returns to avoid survivorship bias
- [x] Remove unusual returns (ret < -100%)
- [x] Create 4 portfolios (2 EW, 2 VW with and without R&D)
- [x] t-1 R&D with look-ahead bias handling (+6 month lag)
- [x] CCM link date filtering for valid GVKEY-PERMNO mappings
- [x] Alpha vs CRSP value-weighted index (CAPM with excess returns)
- [x] CAPM regression with risk-free rate adjustment (Fama-French method)
- [x] Calculate significance (t-stats and p-values from OLS regression)

---

## File Structure

```
assignment_0.py          # Main script (all steps in one file)
portfolio_returns.csv    # Output: Monthly portfolio returns
alpha_summary.csv        # Output: CAPM alpha regression results

# Optional intermediate files (if SAVE_INTERMEDIATE_CSV=True):
compustat_linked.csv     # Compustat + CCM merged
crsp_monthly.csv         # CRSP monthly returns (cleaned)
market_index.csv         # Market index + risk-free rate
firm_monthly_returns.csv # Firm-level merged data
```

---

## Configuration Options

**SAVE_INTERMEDIATE_CSV:**

- `False` (default): Only save final results (portfolio_returns.csv, alpha_summary.csv)
- `True`: Save all intermediate CSV files for inspection/debugging

**To Enable Intermediate Files:**

```python
# At top of assignment_0.py
SAVE_INTERMEDIATE_CSV = True
```

---

## References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. _Journal of Financial Economics_, 33(1), 3-56.
- Jensen, M. C. (1968). The performance of mutual funds in the period 1945–1964. _The Journal of Finance_, 23(2), 389-416.
- WRDS Documentation: https://wrds-www.wharton.upenn.edu/
