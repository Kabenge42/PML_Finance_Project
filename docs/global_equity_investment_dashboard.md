Theme Details

Colors:

- Accent: Gold (#D4AF37)
- Accent Positive: Green (#10B981)
- Accent Negative: Red (#EF4444)
- Background Page: Navy (#0F172A)
- Background Content: Dark Slate (#1E293B)
- Body Text: Light Slate (#E2E8F0)
- Border: Slate (#334155)
- Text: Dark Slate (#1E293B)
- Heading Text: Dark Slate (#0F172A)

Chart Colors - Colorway:

- Dark Blue (#1E3A8A)
- Gold (#D4AF37)
- Green (#10B981)
- Sky Blue (#0EA5E9)
- Purple (#8B5CF6)
- Pink (#EC4899)
- Amber (#F59E0B)
- Cyan (#06B6D4)
- Indigo (#6366F1)

Chart Colors - Colorscale:

- Navy (#0F172A)
- Dark Blue (#1E3A8A)
- Blue (#1E40AF)
- Bright Blue (#2563EB)
- Medium Blue (#3B82F6)
- Light Blue (#60A5FA)
- Lighter Blue (#93C5FD)
- Very Light Blue (#DBEAFE)
- Almost White Blue (#F0F9FF)
- Gold (#D4AF37)

Chart Colors - Grid:

- Graph Grid Color: Slate (#334155)

Typography:

- Font Family: Segoe UI, Roboto, Helvetica Neue, sans-serif
- Font Family Header: Segoe UI, Roboto, Helvetica Neue, sans-serif
- Font Family Headings: Segoe UI, Roboto, Helvetica Neue, sans-serif
- Font Size: 14px
- Font Size Smaller Screen: 13px
- Font Size Header: 28px
- Section Title Font Size: 22px

Buttons:

- Background Color: Blue (#1E40AF)
- Text Color: White (#FFFFFF)
- Border: 0px 2px 2px 0px with White (#FFFFFF)
- Border Radius: 6px
- Text Capitalization: none

Cards:

- Background Color: Dark Slate (#1E293B)
- Margin: 20px
- Padding: 8px
- Border: 0px solid Slate (#334155), 8px radius
- Box Shadow: 0px 4px 12px rgba(0,0,0,0.3)
- Outline: 1px solid Slate (#334155)
- Accent: Gold (#D4AF37)

Card Header:

- Background Color: Dark Slate (#1E293B)
- Margin: 0px 0px 16px 0px
- Padding: 16px
- Border: 0px 0px 2px 0px solid Gold (#D4AF37), 0px radius
- Box Shadow: 0px 0px 0px rgba(0,0,0,0)
- Accent: Gold (#D4AF37)

Card Title:

- Text Color: White (#FFFFFF)
- Font Size: 18px

Card Description:

- Background Color: Dark Slate (#1E293B)
- Text Color: Light Slate (#CBD5E1)
- Font Size: 14px

Card Menu:

- Background Color: Dark Slate (#1E293B)
- Text Color: Light Slate (#E2E8F0)

Controls:

- Background Color: Dark Slate (#1E293B)
- Text Color: Light Slate (#E2E8F0)
- Border: 1px solid Slate (#475569)
- Border Radius: 6px

Border Style:

- Border Width: 0px 0px 1px 0px
- Border Style: solid
- Border Radius: 0px

Hero:

- Background Color: Navy (#0F172A)
- Title Text: White (#FFFFFF)
- Title Font Size: 42px
- Subtitle Text: Light Slate (#CBD5E1)
- Subtitle Font Size: 16px
- Controls Background Color: rgba(15,23,42,0.95)
- Controls Label Text: Dark Slate (#1E293B)
- Controls Label Font Size: 13px
- Controls Grid Columns: 4
- Controls Accent: Gold (#D4AF37)
- Border: 0px solid transparent
- Padding: 32px
- Gap: 20px

Header:

- Background Color: Navy (#0F172A)
- Text Color: White (#FFFFFF)
- Content Alignment: spread
- Margin: 0px 0px 32px 0px
- Padding: 20px 24px 20px 24px
- Border: 0px 0px 2px 0px solid Gold (#D4AF37)
- Box Shadow: 0px 2px 8px rgba(0,0,0,0.3)
- Controls Background Color: Navy (#0F172A)

Footer:

- Background Color: Navy (#0F172A)
- Title Text: Gold (#D4AF37)
- Title Font Size: 16px

Tags:

- Background Color: Blue (#1E40AF)
- Text Color: Light Blue (#93C5FD)
- Font Size: 12px
- Border: 0px solid Bright Blue (#3B82F6), 6px radius

Tooltip:

- Background Color: Navy (#0F172A)
- Text Color: White (#FFFFFF)
- Font Size: 13px

Tables:

- Striped Even: Dark Slate (#1E293B)
- Striped Odd: Navy (#0F172A)
- Border: Slate (#334155)

Top Control Panel:

- Border: 1px solid Slate (#334155)

Layout:

- Section Padding: 24px
- Section Gap: 24px
- Breakpoint Font: 700px
- Breakpoint Stack Blocks: 700px

DBC Colors:

- Primary: Dark Blue (#1E3A8A)
- Secondary: Slate (#64748B)
- Info: Sky Blue (#0EA5E9)
- Gray: Light Slate (#94A3B8)
- Success: Green (#10B981)
- Warning: Amber (#F59E0B)
- Danger: Red (#EF4444)

Report:

- Background: Navy (#0F172A)
- Background Content: Dark Slate (#1E293B)
- Background Page: Navy (#0F172A)
- Text: Light Slate (#E2E8F0)
- Font Family: Segoe UI, Roboto, Helvetica Neue, sans-serif
- Font Size: 12px
- Border: Slate (#334155)

Color Scheme:

- Dark

Filter Component

Sector:

- Multi-select dropdown
- Filters on: `sector`
- Options: Dynamically populated from unique values in `sector` column
- Default: All values selected
- Searchable: False

Country:

- Multi-select dropdown
- Filters on: `country`
- Options: Dynamically populated from unique values in `country` column
- Default: All values selected
- Searchable: True

Exchange:

- Multi-select dropdown
- Filters on: `exchange`
- Options: Dynamically populated from unique values in `exchange` column
- Default: All values selected
- Searchable: True

Market Cap Range:

- Multi-select dropdown
- Filters on: `market_cap`
- Options: 0-1000M, 1000M-10B, 10B-100B, 100B+
- Default: All values selected (0-1000M, 1000M-10B, 10B-100B, 100B+)
- Searchable: False
- Filter behavior: Maps selected ranges to `market_cap` column conditions (0-1000M: 0 to <1000, 1000M-10B: 1000 to <
  10000, 10B-100B: 10000 to <100000, 100B+: ≥100000)

Results Display:

- Shows filtered row count and total row count in format "X,XXX / X,XXX rows"
- Updates when any filter changes or when refresh trigger is activated

Data Cards Component

Card 1: "Total Securities"

- Value: Count of rows in filtered dataset
- Format: Integer with thousands separator (1,234)
- Background color: Default card styling

Card 2: "Avg Market Cap (M)"

- Value: Average of `market_cap`
- Format: Integer with thousands separator and no decimal places (1,234)
- Background color: Default card styling

Card 3: "Avg Expected Return"

- Value: Average of `expected_return_kalman`
- Format: Decimal with 4 places (0.1234)
- Background color: Default card styling

Card 4: "Avg Signal Strength"

- Value: Average of `signal_strength`
- Format: Decimal with 2 places (12.34)
- Background color: Default card styling

Card 5: "Avg Reward/CVaR"

- Value: Average of `reward_to_cvar`
- Format: Decimal with 2 places (12.34)
- Background color: Default card styling

Data Filters:

- All cards filtered by global filter inputs passed through callback
- Cards update automatically when filters change or refresh trigger is activated
- If no data remains after filtering, all cards display "No Data"
- If an error occurs during calculation, all cards display "Error"

Layout:

- 5 cards arranged in single row
- Each card has equal width (20% of container)
- Cards responsive to screen size

Expected Return vs. Risk-Adjusted Return

Chart:

- Type: Scatter
- X: Expected return (Kalman-filtered) (`expected_return_kalman`)
- Y: Risk-adjusted return (`risk_adj_return`)
- Size: Signal strength (`signal_strength`) or market cap (`market_cap`) depending on control selection, or None
- Color: Sector (`sector`), industry (`industry`), country (`country`), or None depending on control selection

Load data:

- Load data from source

Filter data:

- Filter by selected sectors using multi-select dropdown on `sector`
- Apply global filter inputs

Dropdown with label "Color By:":

- Options: None, Sector, Industry, Country
- Default: Sector

Dropdown with label "Size By:":

- Options: None, Signal Strength, Market Cap
- Default: Signal Strength

Multi-select dropdown with label "Filter by Sector:":

- Options: Industrials, Information Technology, Consumer Discretionary, Health Care, Materials, Communication Services,
  Consumer Staples, Energy, Utilities
- Default: All sectors selected

Labels:

- Card title: "Expected Return vs. Risk-Adjusted Return"
- Card description: "Scatter plot showing the relationship between expected return (Kalman-filtered) and risk-adjusted
  return, with optional coloring by sector/industry/country and sizing by signal strength or market cap."
- X-axis label: "Expected Return (Kalman)"
- Y-axis label: "Risk-Adjusted Return"
- Legend labels: Dynamically set based on color-by selection (Sector, Industry, or Country)

Styles:

- Layout: Minimum height of 550px, responsive height calculated as viewport height minus 600px
- Legend: Title text dynamically updated based on selected color dimension
- Marker sizing: Minimum marker size of 6 when size dimension is applied
- Error display: Red text for error messages

Efficient Frontier Optimization

Chart:

- Type: Scatter with overlaid line and markers
- X: Portfolio Volatility (Annualized %) (`volatility`)
- Y: Expected Return (Annualized %) (`return`)
- Color: Sharpe Ratio (`sharpe_ratio`) with Viridis colorscale for all portfolios trace
- Size: 5 for all portfolios, 12 for special portfolios

Load data:

- Load from data source using data loading function
- Filter using global filter inputs

Handle data types:

- String types for `ticker`, `sector`
- Float types for `market_cap`, `mc_prob_pos`, `expected_return_kalman`, `kalman_variance`

Compute additional columns:

- Calculate `annualized_return` as `expected_return_kalman` multiplied by 252
- Calculate `annualized_volatility` as square root of `kalman_variance` multiplied by square root of 252
- Generate covariance matrix from simulated returns using random normal distribution with seed 42
- Generate 500, 1,000, 2,000, or 5,000 random portfolio combinations using Dirichlet distribution with seed 42
- Calculate portfolio return as weighted sum of asset returns
- Calculate portfolio volatility as square root of weighted covariance
- Calculate Sharpe ratio as (portfolio return - risk-free rate) / portfolio volatility for each portfolio
- Identify efficient frontier by binning portfolios into 50 volatility bins and selecting maximum return portfolio in
  each bin
- Identify minimum variance portfolio as portfolio with lowest volatility
- Identify maximum Sharpe ratio portfolio as portfolio with highest Sharpe ratio

Filter data:

- Filter by `mc_prob_pos` greater than 0.5
- Filter by `market_cap` greater than or equal to minimum market cap threshold (1,000, 5,000, 10,000, or 50,000)
- Filter by `sector` when sectors are selected in multi-select dropdown

Dropdown with label "Risk-Free Rate:":

- Options: 0%, 2%, 3%, 4%, 5%
- Default: 3% (0.03)

Multi-select dropdown with label "Sectors:":

- Options: Dynamically populated from unique values in `sector` column
- Default: Empty selection

Dropdown with label "Min Market Cap ($M):":

- Options: 1,000, 5,000, 10,000, 50,000
- Default: 1,000

Dropdown with label "Number of Portfolios:":

- Options: 500, 1,000, 2,000, 5,000
- Default: 1,000

Labels:

- Card title: "Efficient Frontier Optimization"
- Card description: "Visualizes optimal risk-return tradeoff for portfolio combinations using mean-variance
  optimization. Adjust parameters to find portfolios that maximize return for each level of risk."
- X-axis label: "Portfolio Volatility (Annualized %)"
- Y-axis label: "Expected Return (Annualized %)"
- Trace labels: "All Portfolios", "Efficient Frontier", "Min Variance", "Max Sharpe Ratio"
- Hover labels: "Return: {return}%<br>Volatility: {volatility}%<br>Sharpe: {sharpe_ratio}" for all portfolios; "
  Volatility: {volatility}%<br>Return: {return}%" for efficient frontier and special portfolios
- Colorbar label: "Sharpe Ratio"
- Table columns: "Portfolio", "Expected Return (%)", "Volatility (%)", "Sharpe Ratio", "Top 5 Holdings"
- Table data: Two rows for "Minimum Variance" and "Maximum Sharpe Ratio" portfolios with their metrics and top 5
  holdings by weight

Styles:

- Colors: Viridis colorscale for all portfolios scatter points; red dashed line for efficient frontier; green star
  marker for minimum variance portfolio; gold star marker for maximum Sharpe ratio portfolio
- Layout: Chart height 600px, responsive height calculation (100vh - 600px); table width 100% with border collapse and
  1px solid #ddd borders
- Legend: Positioned automatically with closest hover mode
- Other styling: Scatter point opacity 0.6 for all portfolios; colorbar displayed for Sharpe ratio

Sharpe Ratio & Risk-Adjusted Return Comparison

Chart 1:

- Type: Bar
- X: Sharpe Ratio (`sharpe_ratio`)
- Y: Stock name (`name`)
- Color: `sector`

Chart 2:

- Type: Scatter
- X: Annualized Volatility (%) (`annualized_volatility`)
- Y: Annualized Return (%) (`annualized_return`)
- Size: `market_cap`
- Color: Sharpe Ratio (`sharpe_ratio`)

Load data:

- Load data from source and apply global filters

Handle data types:

- String types for `name`, `sector`
- Float types for `market_cap`, `expected_return_kalman`, `kalman_variance`, `mc_prob_pos`

Compute additional columns:

- Calculate annualized return by multiplying `expected_return_kalman` by 252 and converting to percentage
- Calculate annualized volatility by taking square root of `kalman_variance`, multiplying by square root of 252, and
  converting to percentage
- Calculate Sharpe ratio as (annualized return - risk-free rate × 100) / annualized volatility

Filter data:

- Filter by `mc_prob_pos` greater than or equal to minimum probability of positive return threshold
- Filter by `market_cap` greater than or equal to minimum market cap threshold
- Filter by `sector` using multi-select sector filter (if any sectors selected)
- Sort by `sharpe_ratio` in descending order
- Limit results to top N stocks (or all stocks if "All" selected)

Dropdown with label "Risk-Free Rate:":

- Options: 0%, 2%, 3%, 4%, 5%
- Default: 3%

Dropdown with label "Min Prob Positive:":

- Options: 0.5, 0.7, 0.8, 0.9, 0.95
- Default: 0.7

Dropdown with label "Min Market Cap:":

- Options: 1,000, 5,000, 10,000, 50,000
- Default: 5,000

Dropdown with label "Top Stocks:":

- Options: 10, 20, 50, 100, All
- Default: 50

Multi-select dropdown with label "Sector:":

- Options: Dynamically populated from unique values in `sector` column
- Default: Empty selection

Labels:

- Card title: "Sharpe Ratio & Risk-Adjusted Return Comparison"
- Card description: "Compares stocks by their risk-adjusted performance using the Sharpe ratio, which measures excess
  return per unit of risk. Higher values indicate better risk-adjusted returns—use this to identify stocks that deliver
  strong returns relative to their volatility."
- Chart 1 title: "Sharpe Ratio by Stock"
- Chart 1 X-axis label: "Sharpe Ratio"
- Chart 1 Y-axis label: "Stock"
- Chart 2 title: "Risk-Return Profile"
- Chart 2 X-axis label: "Annualized Volatility (%)"
- Chart 2 Y-axis label: "Annualized Return (%)"
- Hover labels: Stock name on Chart 2

Styles:

- Chart 1 height: Minimum 400px, dynamically scaled to 20px per stock
- Chart 2 color scale: Viridis
- Chart 2 marker minimum size: 6px
- Layout: Two charts displayed side-by-side in flex row with 10px gap
- Controls displayed in flex row with 15px right margin and 10px row gap
- Error messages displayed in red pre-formatted text

Value at Risk (VaR): Downside Risk Assessment

Chart 1:

- Type: Bar
- X: Stock name (`name`)
- Y: Conditional Value at Risk at selected confidence level (`cvar_value`)
- Color: `sector`

Chart 2:

- Type: Scatter
- X: Expected return (`expected_return_kalman`)
- Y: Conditional Value at Risk at selected confidence level (`cvar_value`)
- Size: `market_cap`
- Color: Reward-to-CVaR ratio (`reward_to_cvar`)

Load data:

- Load data from source and filter using global filter inputs

Handle data types:

- String types for `name`, `sector`
- Float types for `market_cap`, `mc_prob_pos`, `cvar_5pct_kalman`, `expected_return_kalman`, `reward_to_cvar`, `er_p05`,
  `er_p50`, `er_p95`

Compute additional columns:

- Calculate `cvar_value` based on selected confidence level:
    - At 5% confidence: use `er_p05`
    - At 10% confidence: use average of `er_p05` and `er_p50`
    - At 25% confidence: use `er_p50`
    - At 50% confidence: use `er_p95`
    - Otherwise: use `cvar_5pct_kalman`

Filter data:

- Filter by `market_cap` greater than or equal to minimum market cap threshold
- Filter by `mc_prob_pos` greater than or equal to minimum probability of positive return threshold
- Filter by `sector` using multi-select sector filter (if sectors selected)
- Limit results to top N stocks based on selected sort metric

Aggregate data:

- Sort by selected metric:
    - "Highest Reward-to-CVaR": sort `reward_to_cvar` descending
    - "Lowest CVaR": sort `cvar_value` ascending
    - "Highest Expected Return": sort `expected_return_kalman` descending

Dropdown with label "Confidence Level":

- Options: 5%, 10%, 25%, 50%
- Default: 5%

Dropdown with label "Sort By":

- Options: Highest Reward-to-CVaR, Lowest CVaR, Highest Expected Return
- Default: Highest Reward-to-CVaR

Dropdown with label "Min Market Cap (M)":

- Options: 1,000M, 5,000M, 10,000M, 50,000M
- Default: 5,000M

Dropdown with label "Min Prob Positive":

- Options: 50%, 70%, 80%, 90%
- Default: 70%

Dropdown with label "Number of Stocks":

- Options: 20, 50, 100, All
- Default: 50

Multi-select dropdown with label "Sectors":

- Options: Dynamically populated from unique values in `sector` column
- Default: (empty)

Labels:

- Card title: "Value at Risk (VaR): Downside Risk Assessment"
- Card description: "Calculates the maximum expected loss at different confidence levels using Conditional Value at
  Risk (CVaR). Lower CVaR values indicate less downside risk."
- Chart 1 title: "CVaR at [confidence_level]% Confidence Level"
- Chart 2 title: "Risk-Return Profile"
- Chart 1 X-axis label: "Stock"
- Chart 1 Y-axis label: "CVaR ([confidence_level]%)"
- Chart 1 color legend label: "Sector"
- Chart 2 X-axis label: "Expected Return (Kalman)"
- Chart 2 Y-axis label: "CVaR ([confidence_level]%)"
- Chart 2 color legend label: "Reward-to-CVaR"
- Chart 2 size legend label: "Market Cap (M)"
- Hover data: Stock name, sector, CVaR value, market cap, expected return, reward-to-CVaR ratio

Styles:

- Layout: Two side-by-side charts with flex layout, minimum height 550px each, responsive to viewport height
- Legend: Hover mode set to closest
- Other styling: X-axis labels rotated -45 degrees on bar chart, minimum marker size of 6 on scatter plot

Kelly Criterion Position Sizing

Chart:

- Type: Bar
- X: Stock name (`name`)
- Y: Allocation percentage (`allocation_pct`)
- Color: `sector`

Load data:

- Load from data source using data loading function

Handle data types:

- String types for `ticker`, `name`, `sector`
- Float types for `market_cap`, `mc_prob_pos`, `expected_return_kalman`, `cvar_5pct_kalman`

Compute additional columns:

- Calculate Kelly fraction for each stock using win probability (`mc_prob_pos`), expected return (
  `expected_return_kalman`), and conditional value at risk (`cvar_5pct_kalman`)
- Apply Kelly multiplier to Kelly fraction
- Cap position sizes at maximum position size threshold
- Normalize capped allocations to sum to 100%

Filter data:

- Filter by `sector` using multi-select dropdown
- Filter by minimum market cap threshold in millions
- Filter by minimum win probability threshold
- Select top N stocks by Kelly fraction value

Dropdown with label "Kelly Multiplier:":

- Options: Quarter-Kelly (0.25), Half-Kelly (0.5), Three-Quarter-Kelly (0.75), Full-Kelly (1.0)
- Default: 0.5

Dropdown with label "Max Position Size:":

- Options: 5%, 10%, 15%, 20%, 25%
- Default: 10%

Dropdown with label "Min Win Probability:":

- Options: 60%, 70%, 80%, 90%
- Default: 70%

Dropdown with label "Top N Stocks:":

- Options: Top 10, Top 20, Top 30, Top 50
- Default: Top 20

Multi-select dropdown with label "Sector:":

- Options: All unique sectors from data
- Default: All sectors

Dropdown with label "Min Market Cap ($M):":

- Options: $1B (1000), $5B (5000), $10B (10000), $50B (50000)
- Default: $5B (5000)

Labels:

- Card title: "Kelly Criterion Position Sizing"
- Card description: "Optimal portfolio allocation using Kelly criterion to maximize long-term growth while managing
  risk. Kelly fraction shows the percentage of portfolio to allocate to each stock."
- X-axis label: "Stock"
- Y-axis label: "Allocation (%)"
- Legend label: "Sector"
- Table title: "Top Stocks by Kelly Fraction"
- Table columns: Ticker, Name, Allocation (%), Expected Return (%), Win Probability
- Hover data: Allocation (%), Sector, Ticker

Styles:

- Layout: Flexbox row layout with wrapping for control section, minimum height 550px for chart with responsive height
  calculation
- Legend: Unified hover mode for X-axis
- Other styling: X-axis labels rotated -45 degrees, table cells with padding and bottom borders, table header with bold
  bottom border

Monte Carlo Return Distribution Simulation

Chart:

- Type: Heatmap (Probability Density)
    - X: Stock name (`name`)
    - Y: Return range (binned returns from -50% to +200% in 10% increments)
    - Color: Probability of return falling within each bin
- Type: Line (Cumulative Distribution Function)
    - X: Return level (%) (`return_level`)
    - Y: Cumulative probability (`cumulative_probability`)
    - Color: Stock name (`name`)

Load data:

- Load data using data loading function
- Select columns: `name`, `ticker`, `market_cap`, `er_mean`, `kalman_variance`, `mc_prob_pos`

Filter data:

- Apply global filter inputs to data

Compute additional columns:

- Simulate future returns using Monte Carlo method with normal distribution based on expected return (`er_mean`) and
  volatility (`kalman_variance`)
    - Generate number of simulations specified by user (default 10,000)
    - Scale volatility by time horizon in years (time horizon in days / 252 trading days)
    - Handle zero or negative variance by setting minimum value of 0.01
    - Convert simulated returns to percentage format
- Calculate probability density by binning simulated returns into 10% increments from -50% to +200%
    - Count occurrences of returns within each bin
    - Calculate probability as count divided by total simulations
- Calculate cumulative distribution function by sorting returns and computing cumulative probability at each return
  level (5% increments from -50% to +200%)
- Calculate confidence interval bounds using lower and upper percentiles based on selected confidence level (90%, 95%,
  or 99%)

Dropdown with label "Time Horizon":

- Options: 1 Month (30 days), 3 Months (90 days), 6 Months (180 days), 1 Year (252 days)
- Default: 90 days

Dropdown with label "Simulations":

- Options: 1,000, 5,000, 10,000, 50,000
- Default: 10,000

Dropdown with label "Stock Selection":

- Options: Top 10 by Market Cap, Top 20 by Market Cap, Top 10 by Expected Return, Custom Selection
- Default: Top 10 by Market Cap

Dropdown with label "Confidence Interval":

- Options: 90%, 95%, 99%
- Default: 95%

Labels:

- Card title: "Monte Carlo Return Distribution Simulation"
- Card description: "Simulates thousands of possible future return scenarios using Monte Carlo methods based on each
  stock's expected return and volatility. View the probability distribution of outcomes to understand the range of
  potential returns and downside risks."
- X-axis label (Heatmap): "Stock Name"
- Y-axis label (Heatmap): "Return Range"
- X-axis label (CDF): "Return Level (%)"
- Y-axis label (CDF): "Cumulative Probability"
- Heatmap label: "Probability Density Heatmap"
- CDF label: "Cumulative Distribution Function"
- Colorbar label (Heatmap): "Probability"
- Confidence interval annotations: Lower and upper percentile bounds displayed as vertical dashed lines on CDF chart

Styles:

- Heatmap colorscale: Viridis
- CDF line colors: Automatic color assignment per stock
- Confidence interval lines: Gray dashed lines with 50% opacity
- Chart heights: Minimum 500px, responsive to viewport height (calculated as 100vh - 700px)
- Control layout: Horizontal flex layout with 10px row gap, controls aligned to center
- Chart layout: Two-column flex layout with equal width (flex: 1)
- Error messages: Red text color