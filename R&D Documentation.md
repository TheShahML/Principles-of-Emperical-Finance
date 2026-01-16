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
