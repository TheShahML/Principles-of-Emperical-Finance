### Assignment 0:

Create two portfolios with and without reported R&D Expenditure in year t-1. Construct value and equally weighted portfolios and calculate their alphas with respect to the CRSP value-weighted index as well as their signifigance.

Pull from Compustat and CRSP

- Apply the appropriate filters
  -- Stock exchanges NYSE and NASDAQ
  -- Exclude financials and pharmaceuticals
  -- US only companies
  -- Remove unusual returns
- Pull daily returns from CRSP
- Pull t-1 R&D from Compustat
- From 1980 to 2022

CRSP uses PERMNO and Compustat uses TVKEY as their unique identifier
Merge them on linkeddt

Policy decision to decide what no R&D looks like

- Does not report?
- Discloses but sometimes 0?
- Always 0 = same as not disclosing

Accounting for look ahead bias

- t-1 for R&D
- +6 months to account for 10K disclosure timing

From there, create 4 portfolios

- 2 equal weighted, 1 R&D 1 no R&D
- 2 value weighted, 1 R&D, 1 no R&D

Perhaps had error handling to check for constant 0 R&D to count that as no R&D

R&D Classification Logic (Dynamic Approach)

- Originally thought about keeping same tickers throughout but this causes look-ahead bias
- Example: firm reports 0 R&D 1980-1999 then positive R&D 2000+, can't classify as R&D firm in 1985 using info from 2000
- Solution: dynamic rebalancing each year based on t-1 R&D
  -- If t-1 R&D > 0 → R&D portfolio
  -- If t-1 R&D is missing or 0 → No R&D portfolio
  -- Firms can move between portfolios over time, this is fine

Why +6 Months for Look-Ahead Bias

- Fiscal year ends (e.g., Dec 31, 2000) but investors don't know R&D yet
- SEC gives companies up to 6 months to file 10-K
- So Dec 2000 fiscal year → 10-K filed by June 2001 → portfolio formed July 1, 2001
- Holding period: July 2001 - June 2002
- Then rebalance July 2002 using fiscal year 2001 data

Merging CRSP and Compustat (Link Table Logic)

- CRSP uses PERMNO, Compustat uses GVKEY
- CCM link table bridges them: GVKEY ↔ LPERMNO
- Not a simple join, need date conditions:
  -- LINKDT <= datadate <= LINKENDDT (or LINKENDDT is missing meaning still active)
- Filter for link quality:
  -- LINKTYPE in ('LC', 'LU') - confirmed or usable links
  -- LINKPRIM in ('P', 'C') - primary security only

Delisted Returns (Survivorship Bias)

- When firms delist (bankruptcy, merger, etc.) regular RET often missing
- CRSP provides DLRET (delisting return) to capture final return
- If you ignore DLRET you drop firms right before they fail → survivorship bias
- Fix: if RET missing but DLRET exists → use DLRET
- If both exist → compound them: (1 + RET) \* (1 + DLRET) - 1

Market Cap for Value Weights

- Need market cap to calculate value-weighted returns
- Use market cap at portfolio formation date (end of June) to avoid look-ahead bias
- Market cap = PRC \* SHROUT from CRSP
- Weights stay fixed for the 12-month holding period

Equal weighted:
R_portfolio = (1/N) × Σ R_i

- Just the simple average of all firm returns

Value weighted:
R_portfolio = Σ (w_i × R_i)
where w_i = MarketCap_i / Σ MarketCap

- Bigger firms (by market cap) have more influence on portfolio returns

Non-December Fiscal Year-End Firms

- Not all firms have Dec fiscal year-end, some have June, March, Sept, etc.
- Standard approach (Fama-French style): form portfolios once per year in July using most recent fiscal year data that's at least 6 months old
- Example for July 1, 2001 portfolio formation:
  -- Dec FYE firm: use Dec 2000 data (filed by June 2001) → 6 months old
  -- June FYE firm: use June 2000 data (filed by Dec 2000) → 12 months old
  -- March FYE firm: March 2001 not available yet (filed by Sept 2001), use March 2000 → 15 months old
- Yes firms get treated slightly differently in terms of data "freshness" but this is the accepted approach
- Avoids look-ahead bias for ALL firms, simple (one rebalance per year), and comparable to other research

WRDS Data Pull (Python + wrds library)

Compustat Annual Fundamentals: comp.funda

- gvkey: firm identifier
- datadate: fiscal year end date
- fyear: fiscal year
- xrd: R&D expense (this is what we care about)
- sich: SIC code (for filtering out financials 6000-6999 and pharma 2834)
- curcd: currency (filter for USD)
- fic: country of incorporation (filter for USA)
- exchg: stock exchange code (filter for 11-19)

CRSP Monthly Stock File: crsp.msf

- permno: security identifier
- date: month end date
- ret: holding period return
- dlret: delisting return
- prc: price (need absolute value, sometimes negative to indicate bid/ask avg)
- shrout: shares outstanding
- siccd: SIC code (also filter here for financials and pharma)

CCM Link Table: crsp.ccmxpf_lnkhist

- gvkey: Compustat identifier
- lpermno: linked PERMNO (CRSP identifier)
- linkdt: link start date
- linkenddt: link end date (missing = still active)
- linktype: filter for LC, LU
- linkprim: filter for P, C

Note: verify exact table names against WRDS documentation, sometimes they change or have alternatives

Results

Raw Returns (Annualized):

- EW_NoRD: 11.50%
- EW_RD: 15.25%
- VW_NoRD: 9.69%
- VW_RD: 12.31%
- R&D firms outperform in raw returns

CAPM Alpha Results:

- EW_NoRD: 0.16% annual alpha, t-stat 0.08 (not significant)
- EW_RD: 1.19% annual alpha, t-stat 0.48 (not significant)
- VW_NoRD: -0.62% annual alpha, t-stat -0.68 (not significant)
- VW_RD: -0.50% annual alpha, t-stat -0.52 (not significant)

Fama-French 3-Factor Alpha Results:

- EW_NoRD: -0.83% annual alpha, t-stat -0.63 (not significant)
- EW_RD: 2.12% annual alpha, t-stat 1.37 (not significant)
- VW_NoRD: -1.17% annual alpha, t-stat -1.36 (not significant)
- VW_RD: 0.25% annual alpha, t-stat 0.29 (not significant)

Key Finding: No Significant Alpha

- None of the portfolios have statistically significant alphas (all t-stats < 1.96)
- After adjusting for risk, neither R&D nor non-R&D firms generate abnormal returns
- The raw return difference is explained by risk exposures

What the Factor Loadings Tell Us:

- R&D firms have higher market betas (1.30 vs 1.09 for EW) → more systematic risk
- EW portfolios have high SMB exposure (0.75 to 1.08) → equal weighting emphasizes small stocks
- R&D firms have negative HML betas (-0.10 to -0.24) → R&D firms are growth stocks
- Non-R&D firms have positive HML betas (0.17 to 0.46) → non-R&D firms are more value-like

Conclusion:

- R&D firms don't generate alpha
- Their higher raw returns are compensation for higher risk (higher beta, growth characteristics)
- After controlling for market, size, and value factors, no significant outperformance

Assignment Checklist

- Pull from Compustat and CRSP → done
- Apply filters (exchanges, exclude financials/pharma, US only) → done
- Remove unusual returns (ret < -100%) → done
- Create 4 portfolios (2 EW, 2 VW with and without R&D) → done
- t-1 R&D with look-ahead bias handling (+6 month lag) → done
- Alpha vs CRSP value-weighted index (using mktrf) → done
- Calculate significance (t-stats and p-values) → done
- CAPM alpha → done
- Fama-French 3-factor alpha → done
