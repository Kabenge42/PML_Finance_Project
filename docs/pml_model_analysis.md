# **Probabilistic Machine-Learning (PML) Analysis**

## Unified materialized view containing all calculated stock features

### Covers 26 feature categories from 63 calc_* functions:

1. Valuation Ratios (4 functions)
2. Momentum (2 functions)
3. Technical Analysis (1 function)
4. Profitability (4 functions)
5. Earnings (6 functions)
6. Growth (5 functions)
7. Quality & Risk (5 functions)
8. Leverage & Liquidity (6 functions)
9. Analyst Sentiment (2 functions)
10. Dividends (3 functions)
11. Employment (2 functions)
12. Cash Flow (4 functions)
13. Temporal (2 functions)
14. Balance Sheet (3 functions)
15. Cost Structure (3 functions)
16. Composite Scores (2 functions)
17. Unusual Items (1 function)
18. Volatility Surface (1 function) - Enhancement 2+3
19. Tax Rate Features (1 function) - Enhancement 4
20. OpEx Temporal (1 function) - Enhancement 5
21. Asset Sale Features (1 function) - Enhancement 8
22. FCF Estimate Curve (1 function) - Enhancement 9
23. Dividend History (1 function) - Enhancement 10
24. Investment Income Temporal (1 function) - Enhancement 11
25. Share Dilution Tracking (1 function) - Enhancement 12
26. Forward Consensus (1 function) - Enhancement 7

Direct reference columns include: Enhancement 1 (17 cols), Enhancement 6 (6 cols),

## 1. Setup & Data Loading

```python
# ── Configure PyTensor to use Python-only mode (must run before any PyMC import) ──
# On Windows with MSVC-built CPython 3.14, MinGW g++ cannot reliably link against
# python314.dll due to ABI incompatibility.  Setting cxx="" disables C compilation
# entirely and uses PyTensor's pure-Python VM.
import os

os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="
```

```python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from probabilistic_ml_model.statistical_functions.screening import (
    create_enhanced_screener,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_dividend_quality,
    screen_financial_health,
    rank_stocks_by_composite_score,
)
from probabilistic_ml_model.statistical_functions.statistical_models import (
    bayesian_category_analysis,
    fit_distributions_by_category,
)

# Core PML Models (Bayesian / PyMC)
from probabilistic_ml_model.pymc_models.AccountingAnomalyModel import AccountingAnomalyBayesian
from probabilistic_ml_model.pymc_models.CreditRiskModel import CreditRiskBayesian
from probabilistic_ml_model.pymc_models.DividendSafetyModel import DividendSafetyBayesian
from probabilistic_ml_model.pymc_models.EarningsBeatModel import EarningsBeatBayesian
from probabilistic_ml_model.pymc_models.KalmanFilterModel import KalmanFilterPriceTarget
from probabilistic_ml_model.pymc_models.PriceTargetModel import PriceTargetAchievement

import arviz as az
from probabilistic_ml_model.visualizations.probability_viz import (
    create_beat_probability_posterior,
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:.4f}".format)

COLORS = {
    "primary": "#264653",
    "secondary": "#2A9D8F",
    "accent": "#E9C46A",
    "danger": "#E76F51",
    "info": "#457B9D",
    "light": "#F4A261",
}
```

## 2. Data Overview & Quality Assessment

```sql
%%sql
select *
from public.mv_exp_returns_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isin</th>
      <th>ticker</th>
      <th>name</th>
      <th>description</th>
      <th>region</th>
      <th>country</th>
      <th>trading_country</th>
      <th>exchange</th>
      <th>sector</th>
      <th>industry</th>
      <th>dividend_record_frequency</th>
      <th>earnings_report_frequency</th>
      <th>fy_end</th>
      <th>next_earnings_report</th>
      <th>next_earnings_status</th>
      <th>next_earnings_when</th>
      <th>next_fiscal_quarter</th>
      <th>reporting_interval</th>
      <th>size_class</th>
      <th>style_class</th>
      <th>unit</th>
      <th>dividend_record_announce_date</th>
      <th>dividend_record_ex_date</th>
      <th>dividend_record_payable_date</th>
      <th>dividend_record_record_date</th>
      <th>fy_end_date</th>
      <th>income_statement_report_date</th>
      <th>last_updated</th>
      <th>next_earnings</th>
      <th>next_fy_end_date</th>
      <th>...</th>
      <th>pt_accuracy_1y</th>
      <th>pt_optimism_bias</th>
      <th>pt_range_hit_rate</th>
      <th>pt_median_vs_mean_spread</th>
      <th>pt_high_low_convergence_1y</th>
      <th>analyst_count_stability</th>
      <th>log_market_cap</th>
      <th>daily_turnover_ratio</th>
      <th>liquidity_score</th>
      <th>fcf_est_fy1</th>
      <th>fcf_est_fy2</th>
      <th>fcf_est_fy3</th>
      <th>fcf_est_fy4</th>
      <th>fcf_est_fy5</th>
      <th>fcf_est_growth_fy1_vs_ltm</th>
      <th>fcf_est_growth_fy2_vs_fy1</th>
      <th>fcf_est_growth_fy3_vs_fy2</th>
      <th>fcf_est_growth_fy4_vs_fy3</th>
      <th>fcf_est_growth_fy5_vs_fy4</th>
      <th>fcf_est_cagr_3y</th>
      <th>fcf_est_cagr_5y_fwd</th>
      <th>fcf_est_margin_fy1</th>
      <th>fcf_est_yield_fy1</th>
      <th>fcf_est_growth_acceleration</th>
      <th>fcf_est_growth_deceleration</th>
      <th>fcf_est_trajectory_score_fwd</th>
      <th>fcf_est_always_positive_fwd</th>
      <th>fcf_est_vs_historical</th>
      <th>fcf_est_capex_implied_ratio</th>
      <th>feature_calculated_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US0231351067</td>
      <td>AMZN</td>
      <td>Amazon.com Inc.</td>
      <td>Amazon.com Inc. engages in the retail sale of ...</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>Consumer Discretionary</td>
      <td>Broadline Retail</td>
      <td>NaN</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Estimated</td>
      <td>After-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Large Cap</td>
      <td>Core</td>
      <td>USD</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-04-29</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>0.0241</td>
      <td>-0.0241</td>
      <td>1</td>
      <td>-0.0083</td>
      <td>0.2811</td>
      <td>0.9948</td>
      <td>14.8245</td>
      <td>0.0027</td>
      <td>1371939.4460</td>
      <td>-20161.6400</td>
      <td>48246.9100</td>
      <td>79426.8000</td>
      <td>99249.0000</td>
      <td>141282.0000</td>
      <td>-362.0096</td>
      <td>339.3005</td>
      <td>64.6257</td>
      <td>24.9566</td>
      <td>42.3511</td>
      <td>117.7306</td>
      <td>78.9681</td>
      <td>-2.8122</td>
      <td>-0.7351</td>
      <td>701.3101</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>-285.4143</td>
      <td>-2.6201</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US67066G1040</td>
      <td>NVDA</td>
      <td>NVIDIA Corporation</td>
      <td>NVIDIA Corporation operates as a data center s...</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>Information Technology</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>Quarterly</td>
      <td>Annually</td>
      <td>Jan 2026</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>After-Market</td>
      <td>Q4 2028</td>
      <td>12.0000</td>
      <td>Large Cap</td>
      <td>Growth</td>
      <td>USD</td>
      <td>2026-02-25</td>
      <td>2026-03-11</td>
      <td>2026-04-01</td>
      <td>2026-03-11</td>
      <td>2026-01-31</td>
      <td>2026-01-25</td>
      <td>2026-04-23</td>
      <td>2026-05-20</td>
      <td>2027-01-31</td>
      <td>...</td>
      <td>0.2118</td>
      <td>-0.2118</td>
      <td>1</td>
      <td>0.0136</td>
      <td>0.2844</td>
      <td>1.0000</td>
      <td>15.3945</td>
      <td>0.0039</td>
      <td>5528625.0355</td>
      <td>181667.4400</td>
      <td>234892.4600</td>
      <td>288666.8400</td>
      <td>376909.5000</td>
      <td>410181.5000</td>
      <td>87.9137</td>
      <td>29.2981</td>
      <td>22.8932</td>
      <td>30.5690</td>
      <td>8.8276</td>
      <td>43.9990</td>
      <td>33.5154</td>
      <td>84.1294</td>
      <td>3.7457</td>
      <td>-58.6156</td>
      <td>1</td>
      <td>100</td>
      <td>1</td>
      <td>29.0456</td>
      <td>1.8791</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US92892B1035</td>
      <td>VOYG</td>
      <td>Voyager Technologies Inc.</td>
      <td>Voyager Technologies Inc. operates as a defens...</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NYSE</td>
      <td>Industrials</td>
      <td>Aerospace and Defense</td>
      <td>NaN</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>After-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Small Cap</td>
      <td>Core</td>
      <td>USD</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-05-04</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>-0.0256</td>
      <td>NaN</td>
      <td>2.5000</td>
      <td>7.4852</td>
      <td>0.0177</td>
      <td>98111.0027</td>
      <td>-505.4700</td>
      <td>-319.9000</td>
      <td>-28.0500</td>
      <td>12.7500</td>
      <td>326.7500</td>
      <td>-145.8273</td>
      <td>36.7124</td>
      <td>91.2316</td>
      <td>145.4545</td>
      <td>2462.7451</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-303.7315</td>
      <td>-28.3744</td>
      <td>182.5396</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>-55.7903</td>
      <td>6.0369</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US02079K3059</td>
      <td>GOOGL</td>
      <td>Alphabet Inc.</td>
      <td>Alphabet Inc. offers various products and plat...</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>Communication Services</td>
      <td>Interactive Media and Services</td>
      <td>Quarterly</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>After-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Large Cap</td>
      <td>Core</td>
      <td>USD</td>
      <td>2026-02-04</td>
      <td>2026-03-09</td>
      <td>2026-03-16</td>
      <td>2026-03-09</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-04-29</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>0.6690</td>
      <td>-0.6690</td>
      <td>0</td>
      <td>-0.0224</td>
      <td>0.2684</td>
      <td>1.0633</td>
      <td>15.2243</td>
      <td>0.0011</td>
      <td>613182.9255</td>
      <td>22587.8300</td>
      <td>49966.3600</td>
      <td>96136.0300</td>
      <td>151310.7600</td>
      <td>195811.9800</td>
      <td>-69.1701</td>
      <td>121.2092</td>
      <td>92.4015</td>
      <td>57.3924</td>
      <td>29.4105</td>
      <td>9.4783</td>
      <td>21.7271</td>
      <td>5.6072</td>
      <td>0.5521</td>
      <td>190.3793</td>
      <td>0</td>
      <td>100</td>
      <td>1</td>
      <td>-69.8600</td>
      <td>0.3083</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NL0015002IE0</td>
      <td>AVTX</td>
      <td>Avantium N.V.</td>
      <td>Avantium N.V. a chemical technology company de...</td>
      <td>Europe</td>
      <td>NL</td>
      <td>NL</td>
      <td>ENXTAM</td>
      <td>Industrials</td>
      <td>Professional Services</td>
      <td>NaN</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>Pre-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Small Cap</td>
      <td>Core</td>
      <td>EUR</td>
      <td>2025-09-04</td>
      <td>2025-09-05</td>
      <td>2025-09-09</td>
      <td>2025-09-08</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-08-19</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>0.8486</td>
      <td>0.8486</td>
      <td>0</td>
      <td>1.8260</td>
      <td>7.3278</td>
      <td>0.9375</td>
      <td>5.3792</td>
      <td>0.0061</td>
      <td>17249.1252</td>
      <td>-25.4000</td>
      <td>-5.8500</td>
      <td>1.1700</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>55.2896</td>
      <td>76.9685</td>
      <td>120.0000</td>
      <td>-100.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-148.1914</td>
      <td>-11.7137</td>
      <td>21.6789</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>13.4251</td>
      <td>-3.0639</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6671</th>
      <td>JP3297000006</td>
      <td>7984</td>
      <td>Kokuyo Co. Ltd.</td>
      <td>Kokuyo Co. Ltd. together with its subsidiaries...</td>
      <td>Asia / Pacific</td>
      <td>JP</td>
      <td>JP</td>
      <td>TSE</td>
      <td>Industrials</td>
      <td>Commercial Services and Supplies</td>
      <td>Interim Payment</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>Pre-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Small Cap</td>
      <td>Core</td>
      <td>JPY</td>
      <td>2026-04-09</td>
      <td>2026-06-29</td>
      <td>2026-09-04</td>
      <td>2026-06-30</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-04-28</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>0.1310</td>
      <td>-0.1310</td>
      <td>0</td>
      <td>0.0351</td>
      <td>0.0543</td>
      <td>1.0000</td>
      <td>7.7466</td>
      <td>0.0017</td>
      <td>94712.9549</td>
      <td>234.5700</td>
      <td>135.4400</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>364.4950</td>
      <td>-42.2603</td>
      <td>-100.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.2092</td>
      <td>10.1387</td>
      <td>-406.7554</td>
      <td>1</td>
      <td>40</td>
      <td>0</td>
      <td>398.6369</td>
      <td>4.6450</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>6672</th>
      <td>CNE100006053</td>
      <td>688484</td>
      <td>Southchip Semiconductor Technology(Shanghai) C...</td>
      <td>Southchip Semiconductor Technology(Shanghai) C...</td>
      <td>Asia / Pacific</td>
      <td>CN</td>
      <td>CN</td>
      <td>SHSE</td>
      <td>Information Technology</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>Final Payment</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>During-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Small Cap</td>
      <td>Core</td>
      <td>CNY</td>
      <td>2025-04-29</td>
      <td>2025-07-09</td>
      <td>2025-07-09</td>
      <td>2025-07-08</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-04-30</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>0.1543</td>
      <td>0.1543</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.0717</td>
      <td>0.6667</td>
      <td>7.7465</td>
      <td>0.0168</td>
      <td>1294878.7259</td>
      <td>41.0200</td>
      <td>65.3400</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>59.2882</td>
      <td>-100.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.7960</td>
      <td>1.7730</td>
      <td>NaN</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>6673</th>
      <td>SE0015987946</td>
      <td>SYNT</td>
      <td>SyntheticMR AB (publ)</td>
      <td>SyntheticMR AB (publ) engages in the developme...</td>
      <td>Europe</td>
      <td>SE</td>
      <td>SE</td>
      <td>NGM</td>
      <td>Health Care</td>
      <td>Health Care Technology</td>
      <td>NaN</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>NaN</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Small Cap</td>
      <td>Core</td>
      <td>SEK</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2024-01-31</td>
      <td>2026-05-12</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>4.7708</td>
      <td>-4.7708</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.1138</td>
      <td>1.5000</td>
      <td>5.0112</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.2000</td>
      <td>-0.8700</td>
      <td>0.2400</td>
      <td>1.0000</td>
      <td>2.0900</td>
      <td>-266.6667</td>
      <td>60.4545</td>
      <td>127.5862</td>
      <td>316.6667</td>
      <td>109.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-31.6092</td>
      <td>-1.4658</td>
      <td>327.1212</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>8.3333</td>
      <td>-3.7288</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>6674</th>
      <td>US11135F1012</td>
      <td>AVGO</td>
      <td>Broadcom Inc.</td>
      <td>Broadcom Inc. designs develops and supplies va...</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>Information Technology</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>Quarterly</td>
      <td>Quarterly</td>
      <td>Nov 2025</td>
      <td>Interim</td>
      <td>Estimated</td>
      <td>During-Market</td>
      <td>Q1 2026</td>
      <td>3.0000</td>
      <td>Large Cap</td>
      <td>Growth</td>
      <td>USD</td>
      <td>2026-03-04</td>
      <td>2026-03-23</td>
      <td>2026-03-31</td>
      <td>2026-03-23</td>
      <td>2025-11-30</td>
      <td>2026-02-01</td>
      <td>2026-04-23</td>
      <td>2026-06-04</td>
      <td>2026-11-30</td>
      <td>...</td>
      <td>0.7525</td>
      <td>-0.7525</td>
      <td>0</td>
      <td>-0.0042</td>
      <td>0.1491</td>
      <td>1.0413</td>
      <td>14.5031</td>
      <td>0.0031</td>
      <td>711723.3630</td>
      <td>50056.4400</td>
      <td>81391.7300</td>
      <td>101222.1500</td>
      <td>110281.0000</td>
      <td>127208.0000</td>
      <td>73.1398</td>
      <td>62.5999</td>
      <td>24.3642</td>
      <td>8.9495</td>
      <td>15.3490</td>
      <td>51.8463</td>
      <td>34.4901</td>
      <td>73.3084</td>
      <td>2.5168</td>
      <td>-10.5399</td>
      <td>1</td>
      <td>100</td>
      <td>1</td>
      <td>34.5079</td>
      <td>1.7314</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
    <tr>
      <th>6675</th>
      <td>IT0004585243</td>
      <td>TES</td>
      <td>Tesmec S.p.A.</td>
      <td>Tesmec S.p.A. designs manufactures and sells p...</td>
      <td>Europe</td>
      <td>IT</td>
      <td>IT</td>
      <td>BIT</td>
      <td>Industrials</td>
      <td>Machinery</td>
      <td>NaN</td>
      <td>Annually</td>
      <td>Dec 2025</td>
      <td>Full Year</td>
      <td>Confirmed</td>
      <td>Pre-Market</td>
      <td>Q4 2027</td>
      <td>12.0000</td>
      <td>Small Cap</td>
      <td>Value</td>
      <td>EUR</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-12-31</td>
      <td>2025-12-31</td>
      <td>2026-04-23</td>
      <td>2026-05-08</td>
      <td>2026-12-31</td>
      <td>...</td>
      <td>0.8125</td>
      <td>-0.8125</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>4.6261</td>
      <td>0.0028</td>
      <td>183871.3096</td>
      <td>4.4500</td>
      <td>13.1100</td>
      <td>16.3900</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-88.9496</td>
      <td>194.6067</td>
      <td>25.0191</td>
      <td>-100.0000</td>
      <td>NaN</td>
      <td>-25.8919</td>
      <td>NaN</td>
      <td>1.4704</td>
      <td>4.3576</td>
      <td>283.5563</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>-356.7121</td>
      <td>0.1105</td>
      <td>2026-04-24 02:02:56.282513+02</td>
    </tr>
  </tbody>
</table>
<p>6676 rows × 831 columns</p>
</div>

```sql
%%sql
select cfr.category, cfr.feature_alias, cfr.calculation_type
from public.calculated_features_registry cfr
where category <> 'Identifier'
  and category <> 'Market Data';
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>feature_alias</th>
      <th>calculation_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Interest Income</td>
      <td>interest_income_yoy_growth</td>
      <td>growth</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Accounting Quality</td>
      <td>asset_sale_gain_loss_ltm</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Quality &amp; Risk</td>
      <td>net_buyback_flag</td>
      <td>flag</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Analyst Sentiment</td>
      <td>ebitda_est_fy1e</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Analyst Sentiment</td>
      <td>analyst_count_stability</td>
      <td>ratio</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>802</th>
      <td>Valuation Ratios</td>
      <td>tangible_asset_quality</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>803</th>
      <td>Valuation Ratios</td>
      <td>tangible_book_value_fy</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>804</th>
      <td>Valuation Ratios</td>
      <td>tangible_book_value_ltm</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>805</th>
      <td>Valuation Ratios</td>
      <td>tbv_vs_calculated</td>
      <td>direct</td>
    </tr>
    <tr>
      <th>806</th>
      <td>Valuation Ratios</td>
      <td>tbv_yoy_growth</td>
      <td>growth</td>
    </tr>
  </tbody>
</table>
<p>807 rows × 3 columns</p>
</div>

```python
# Build a proper dictionary from the DataFrame: {category: [feature_alias, ...]}
FEATURE_CATEGORIES = feature_cat.groupby("category")["feature_alias"].apply(list).to_dict()

print(f"Dataset shape: {df.shape[0]} stocks × {df.shape[1]} features")
print(f"Feature categories: {len(FEATURE_CATEGORIES)}")
print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
```

    Dataset shape: 6676 stocks × 831 features
    Feature categories: 27
    
    Column dtypes:
    float64    694
    int64      105
    str         32
    Name: count, dtype: int64

```python
# Identifier columns overview
id_cols = ["ticker", "name", "industry", "sector", "next_earnings"]
available_ids = [c for c in id_cols if c in df.columns]
print("Identifier columns available:", available_ids)
df[available_ids].head(10)
```

    Identifier columns available: ['ticker', 'name', 'industry', 'sector', 'next_earnings']

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>industry</th>
      <th>sector</th>
      <th>next_earnings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AMZN</td>
      <td>Amazon.com Inc.</td>
      <td>Broadline Retail</td>
      <td>Consumer Discretionary</td>
      <td>2026-04-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NVDA</td>
      <td>NVIDIA Corporation</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>Information Technology</td>
      <td>2026-05-20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VOYG</td>
      <td>Voyager Technologies Inc.</td>
      <td>Aerospace and Defense</td>
      <td>Industrials</td>
      <td>2026-05-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GOOGL</td>
      <td>Alphabet Inc.</td>
      <td>Interactive Media and Services</td>
      <td>Communication Services</td>
      <td>2026-04-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AVTX</td>
      <td>Avantium N.V.</td>
      <td>Professional Services</td>
      <td>Industrials</td>
      <td>2026-08-19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>META</td>
      <td>Meta Platforms Inc.</td>
      <td>Interactive Media and Services</td>
      <td>Communication Services</td>
      <td>2026-04-29</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CANTA</td>
      <td>Cantargia AB (publ)</td>
      <td>Biotechnology</td>
      <td>Health Care</td>
      <td>2026-05-19</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LLY</td>
      <td>Eli Lilly and Company</td>
      <td>Pharmaceuticals</td>
      <td>Health Care</td>
      <td>2026-04-30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>XLS</td>
      <td>Xlife Sciences AG</td>
      <td>Life Sciences Tools and Services</td>
      <td>Health Care</td>
      <td>2026-04-28</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JNJ</td>
      <td>Johnson &amp; Johnson</td>
      <td>Pharmaceuticals</td>
      <td>Health Care</td>
      <td>2026-07-14</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Missing data analysis per feature category
missing_report = {}
for cat, cols in FEATURE_CATEGORIES.items():
    present = [c for c in cols if c in df.columns]
    if present:
        pct_missing = df[present].isnull().mean().mean() * 100
        missing_report[cat] = {
            "defined": len(cols),
            "present": len(present),
            "avg_missing_pct": round(pct_missing, 1),
        }

missing_df = pd.DataFrame(missing_report).T.sort_values("avg_missing_pct", ascending=False)
missing_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>defined</th>
      <th>present</th>
      <th>avg_missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Employee Productivity</th>
      <td>7.0000</td>
      <td>7.0000</td>
      <td>26.9000</td>
    </tr>
    <tr>
      <th>Dividend Reliability</th>
      <td>34.0000</td>
      <td>34.0000</td>
      <td>23.3000</td>
    </tr>
    <tr>
      <th>Valuation Timeseries</th>
      <td>23.0000</td>
      <td>22.0000</td>
      <td>16.0000</td>
    </tr>
    <tr>
      <th>EPS Trajectory</th>
      <td>11.0000</td>
      <td>11.0000</td>
      <td>13.5000</td>
    </tr>
    <tr>
      <th>GAAP vs Adjusted</th>
      <td>50.0000</td>
      <td>50.0000</td>
      <td>13.2000</td>
    </tr>
    <tr>
      <th>Employment Dynamics</th>
      <td>10.0000</td>
      <td>10.0000</td>
      <td>12.7000</td>
    </tr>
    <tr>
      <th>Efficiency Ratios</th>
      <td>45.0000</td>
      <td>45.0000</td>
      <td>11.5000</td>
    </tr>
    <tr>
      <th>Cash Flow</th>
      <td>59.0000</td>
      <td>57.0000</td>
      <td>9.7000</td>
    </tr>
    <tr>
      <th>Cash flow</th>
      <td>28.0000</td>
      <td>28.0000</td>
      <td>9.4000</td>
    </tr>
    <tr>
      <th>Valuation Ratios</th>
      <td>17.0000</td>
      <td>16.0000</td>
      <td>8.7000</td>
    </tr>
    <tr>
      <th>Efficiency</th>
      <td>4.0000</td>
      <td>4.0000</td>
      <td>7.9000</td>
    </tr>
    <tr>
      <th>Analyst Sentiment</th>
      <td>30.0000</td>
      <td>30.0000</td>
      <td>7.4000</td>
    </tr>
    <tr>
      <th>Earnings Quality</th>
      <td>76.0000</td>
      <td>76.0000</td>
      <td>6.1000</td>
    </tr>
    <tr>
      <th>Interest Income</th>
      <td>16.0000</td>
      <td>12.0000</td>
      <td>4.8000</td>
    </tr>
    <tr>
      <th>Growth Metrics</th>
      <td>14.0000</td>
      <td>14.0000</td>
      <td>4.5000</td>
    </tr>
    <tr>
      <th>Accounting Quality</th>
      <td>46.0000</td>
      <td>46.0000</td>
      <td>4.2000</td>
    </tr>
    <tr>
      <th>Quality &amp; Risk</th>
      <td>18.0000</td>
      <td>17.0000</td>
      <td>4.2000</td>
    </tr>
    <tr>
      <th>Balance Sheet</th>
      <td>51.0000</td>
      <td>51.0000</td>
      <td>3.9000</td>
    </tr>
    <tr>
      <th>Revenue Forecasting</th>
      <td>49.0000</td>
      <td>49.0000</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>Momentum &amp; Technical</th>
      <td>43.0000</td>
      <td>39.0000</td>
      <td>1.8000</td>
    </tr>
    <tr>
      <th>Profitability</th>
      <td>75.0000</td>
      <td>75.0000</td>
      <td>1.6000</td>
    </tr>
    <tr>
      <th>Price Target Dynamics</th>
      <td>15.0000</td>
      <td>15.0000</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>Leverage &amp; Liquidity</th>
      <td>47.0000</td>
      <td>47.0000</td>
      <td>1.3000</td>
    </tr>
    <tr>
      <th>Financial Distress</th>
      <td>9.0000</td>
      <td>9.0000</td>
      <td>0.7000</td>
    </tr>
    <tr>
      <th>Temporal Patterns</th>
      <td>16.0000</td>
      <td>16.0000</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>Technical Analysis</th>
      <td>11.0000</td>
      <td>11.0000</td>
      <td>0.4000</td>
    </tr>
    <tr>
      <th>Composite Scores</th>
      <td>3.0000</td>
      <td>3.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Visualize missing data by category
fig, ax = plt.subplots(figsize=(14, 8))
colors_bar = [
    COLORS["danger"] if v > 50 else COLORS["accent"] if v > 20 else COLORS["secondary"]
    for v in missing_df["avg_missing_pct"]
]
bars = ax.barh(missing_df.index, missing_df["avg_missing_pct"], color=colors_bar, edgecolor="white")
ax.set_xlabel("Average Missing %", fontsize=12)
ax.set_title("Data Completeness by Feature Category", fontsize=14, fontweight="bold")
ax.axvline(20, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\2861767602.py:15: UserWarning: The figure layout has changed to tight
      plt.tight_layout()

![png](pml_model_analysis_files/pml_model_analysis_10_1.png)

## 3. Descriptive Statistics by Feature Category

```python
# Summary statistics for each category
for cat_name in FEATURE_CATEGORIES:
    cols = [c for c in FEATURE_CATEGORIES.get(cat_name, []) if c in df.columns]
    if cols:
        print(f"\n{'=' * 60}")
        print(f"  {cat_name} ({len(cols)} features)")
        print(f"{'=' * 60}")
        display(df[cols].describe().round(3))
```

    ============================================================
      Accounting Quality (46 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>asset_sale_gain_loss_ltm</th>
      <th>asset_sale_frequency</th>
      <th>asset_sale_trend</th>
      <th>effective_tax_rate_fy</th>
      <th>tax_rate_yoy_change</th>
      <th>tax_rate_qoq_change</th>
      <th>tax_rate_stability</th>
      <th>low_tax_flag</th>
      <th>tax_rate_trend_4q</th>
      <th>effective_tax_rate_ltm</th>
      <th>goodwill_vs_5y_avg</th>
      <th>goodwill_change_rate</th>
      <th>restructuring_intensity</th>
      <th>exceptional_items_frequency</th>
      <th>merger_impact_ratio</th>
      <th>non_operating_income_share</th>
      <th>asset_sale_boost</th>
      <th>accounting_quality_score</th>
      <th>goodwill_1fy</th>
      <th>goodwill_2fq</th>
      <th>goodwill_2fy</th>
      <th>goodwill_3fq</th>
      <th>goodwill_3fy</th>
      <th>goodwill_3y_growth</th>
      <th>goodwill_4fq</th>
      <th>goodwill_4fy</th>
      <th>goodwill_accumulation_rate</th>
      <th>goodwill_concentration</th>
      <th>goodwill_fy</th>
      <th>goodwill_ltm</th>
      <th>goodwill_qoq_change</th>
      <th>goodwill_to_assets_trend</th>
      <th>goodwill_yoy_change</th>
      <th>impairment_risk_score</th>
      <th>recent_acquisition_flag</th>
      <th>asset_writedown_frequency</th>
      <th>asset_writedown_ltm</th>
      <th>exceptional_items_total_ltm</th>
      <th>goodwill_impairment_frequency</th>
      <th>goodwill_impairment_ltm</th>
      <th>has_goodwill_impairment_ltm</th>
      <th>quality_issues_count_5y</th>
      <th>restructuring_frequency</th>
      <th>restructuring_ltm</th>
      <th>goodwill_1fq</th>
      <th>goodwill_fq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4817.0000</td>
      <td>4720.0000</td>
      <td>6339.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6668.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4591.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4591.0000</td>
      <td>6643.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4093.0000</td>
      <td>6575.0000</td>
      <td>4720.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.2110</td>
      <td>4.1390</td>
      <td>-3.6380</td>
      <td>0.2320</td>
      <td>-0.0810</td>
      <td>-0.0210</td>
      <td>0.4280</td>
      <td>0.3280</td>
      <td>-0.0210</td>
      <td>0.2530</td>
      <td>1.1260</td>
      <td>19.8640</td>
      <td>-0.0020</td>
      <td>0.5800</td>
      <td>-0.0040</td>
      <td>0.2780</td>
      <td>0.3010</td>
      <td>85.7380</td>
      <td>1071.1730</td>
      <td>1026.5840</td>
      <td>1057.7360</td>
      <td>966.6760</td>
      <td>1020.7080</td>
      <td>785.0120</td>
      <td>1034.8710</td>
      <td>994.8650</td>
      <td>6.7540</td>
      <td>24.5940</td>
      <td>1132.4910</td>
      <td>1101.3910</td>
      <td>4.8340</td>
      <td>-0.0750</td>
      <td>1993.4600</td>
      <td>16.9000</td>
      <td>0.0180</td>
      <td>1.9890</td>
      <td>-31.0260</td>
      <td>72.2490</td>
      <td>0.4050</td>
      <td>-15.3530</td>
      <td>0.0810</td>
      <td>3.2780</td>
      <td>0.8840</td>
      <td>-21.7530</td>
      <td>1032.6620</td>
      <td>1101.2280</td>
    </tr>
    <tr>
      <th>std</th>
      <td>225.6490</td>
      <td>3.8600</td>
      <td>188.2570</td>
      <td>1.1280</td>
      <td>4.7540</td>
      <td>2.3040</td>
      <td>3.7310</td>
      <td>0.4690</td>
      <td>1.6410</td>
      <td>1.9600</td>
      <td>9.0180</td>
      <td>1257.0660</td>
      <td>0.0070</td>
      <td>0.7000</td>
      <td>0.0340</td>
      <td>2.6210</td>
      <td>0.4590</td>
      <td>16.9730</td>
      <td>4593.5590</td>
      <td>4651.7170</td>
      <td>4333.9290</td>
      <td>4314.9790</td>
      <td>4214.8870</td>
      <td>21660.9160</td>
      <td>4588.4490</td>
      <td>4208.9330</td>
      <td>76.9070</td>
      <td>218.9360</td>
      <td>4774.8380</td>
      <td>4751.5350</td>
      <td>219.2580</td>
      <td>3.6750</td>
      <td>125707.1080</td>
      <td>29.4090</td>
      <td>0.1330</td>
      <td>2.0350</td>
      <td>306.1030</td>
      <td>488.5840</td>
      <td>0.9340</td>
      <td>188.7640</td>
      <td>0.2720</td>
      <td>3.0960</td>
      <td>1.6600</td>
      <td>290.7270</td>
      <td>4464.6060</td>
      <td>4751.5610</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-11049.1100</td>
      <td>0.0000</td>
      <td>-13005.0720</td>
      <td>0.0000</td>
      <td>-307.5180</td>
      <td>-142.4750</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-61.2130</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-3.2210</td>
      <td>-0.1830</td>
      <td>0.0000</td>
      <td>-1.3190</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>20.0000</td>
      <td>0.0000</td>
      <td>-1.0100</td>
      <td>0.0000</td>
      <td>-1.0700</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-0.0400</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-11438.3840</td>
      <td>-1.0100</td>
      <td>-1.9100</td>
      <td>-100.0000</td>
      <td>-53.0280</td>
      <td>-217.4420</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-16440.3400</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-6734.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-20197.3700</td>
      <td>-1.7800</td>
      <td>-1.9100</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0220</td>
      <td>0.0000</td>
      <td>-0.0350</td>
      <td>-0.0300</td>
      <td>0.0220</td>
      <td>0.0000</td>
      <td>-0.0540</td>
      <td>0.0070</td>
      <td>0.5100</td>
      <td>-0.0110</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>0.0140</td>
      <td>0.0000</td>
      <td>75.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-9.8340</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-3.3920</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.5280</td>
      <td>-0.2860</td>
      <td>-0.2390</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-2.4700</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>0.0000</td>
      <td>0.1630</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0870</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1940</td>
      <td>1.0000</td>
      <td>0.0390</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0550</td>
      <td>0.0000</td>
      <td>90.0000</td>
      <td>38.5950</td>
      <td>13.7800</td>
      <td>36.0450</td>
      <td>14.3600</td>
      <td>30.4350</td>
      <td>2.8640</td>
      <td>23.1000</td>
      <td>25.3850</td>
      <td>0.9460</td>
      <td>2.5780</td>
      <td>40.7300</td>
      <td>27.2700</td>
      <td>0.0530</td>
      <td>0.0000</td>
      <td>3.4580</td>
      <td>2.5210</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.2100</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>3.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>17.8100</td>
      <td>27.0500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.1300</td>
      <td>8.0000</td>
      <td>0.0050</td>
      <td>0.2610</td>
      <td>0.0180</td>
      <td>0.0180</td>
      <td>0.2510</td>
      <td>1.0000</td>
      <td>0.0260</td>
      <td>0.2630</td>
      <td>1.1570</td>
      <td>0.1300</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.1500</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>419.5680</td>
      <td>328.0350</td>
      <td>411.2880</td>
      <td>317.0120</td>
      <td>374.0000</td>
      <td>27.4090</td>
      <td>358.6900</td>
      <td>346.6500</td>
      <td>8.4090</td>
      <td>32.2700</td>
      <td>457.2780</td>
      <td>406.1420</td>
      <td>0.8340</td>
      <td>0.0400</td>
      <td>12.8970</td>
      <td>17.7560</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>0.0000</td>
      <td>14.9850</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>5.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>355.0020</td>
      <td>406.0080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9687.0000</td>
      <td>10.0000</td>
      <td>1925.7500</td>
      <td>62.0000</td>
      <td>61.8160</td>
      <td>76.3640</td>
      <td>183.6380</td>
      <td>1.0000</td>
      <td>75.7830</td>
      <td>142.9370</td>
      <td>498.0000</td>
      <td>86253.0000</td>
      <td>0.0300</td>
      <td>3.0000</td>
      <td>0.0760</td>
      <td>178.5450</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>119220.0000</td>
      <td>119509.0000</td>
      <td>117043.0000</td>
      <td>119329.0000</td>
      <td>113010.0000</td>
      <td>1293710.0000</td>
      <td>119191.0000</td>
      <td>115796.0000</td>
      <td>2247.5970</td>
      <td>9991.1070</td>
      <td>119509.0000</td>
      <td>119622.0000</td>
      <td>8000.0000</td>
      <td>62.2370</td>
      <td>8625300.0000</td>
      <td>100.0000</td>
      <td>1.0000</td>
      <td>5.0000</td>
      <td>1574.6700</td>
      <td>20197.3700</td>
      <td>5.0000</td>
      <td>498.3400</td>
      <td>1.0000</td>
      <td>15.0000</td>
      <td>5.0000</td>
      <td>95.1900</td>
      <td>119497.0000</td>
      <td>119622.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Analyst Sentiment (30 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ebitda_est_fy1e</th>
      <th>analyst_count_stability</th>
      <th>pt_accuracy_1y</th>
      <th>pt_optimism_bias</th>
      <th>pt_high_low_convergence_1y</th>
      <th>eps_gaap_vs_norm_ntm</th>
      <th>eps_gaap_vs_norm_fy1e</th>
      <th>forward_adjustment_trend</th>
      <th>ev_ebitda_est_fy1</th>
      <th>ebitda_forward_growth</th>
      <th>earnings_revision_divergence</th>
      <th>forward_pe_vs_sector_proxy</th>
      <th>pt_achievement_1y</th>
      <th>pt_range_hit_rate</th>
      <th>pt_median_vs_mean_spread</th>
      <th>pe_forward_discount</th>
      <th>pe_est_fy1</th>
      <th>ebitda_est_ntm</th>
      <th>pe_ntm</th>
      <th>analyst_bullish_pct</th>
      <th>analyst_bearish_pct</th>
      <th>analyst_neutral_pct</th>
      <th>upside_potential</th>
      <th>price_target_spread_pct</th>
      <th>price_target_revision_1m</th>
      <th>price_target_revision_3m</th>
      <th>eps_revision_momentum</th>
      <th>analyst_rating_normalized</th>
      <th>analyst_coverage_quality</th>
      <th>analyst_conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6605.0000</td>
      <td>6350.0000</td>
      <td>6350.0000</td>
      <td>6350.0000</td>
      <td>5557.0000</td>
      <td>5491.0000</td>
      <td>3451.0000</td>
      <td>6056.0000</td>
      <td>6621.0000</td>
      <td>4589.0000</td>
      <td>5217.0000</td>
      <td>6350.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5003.0000</td>
      <td>5775.0000</td>
      <td>6676.0000</td>
      <td>5811.0000</td>
      <td>6590.0000</td>
      <td>6673.0000</td>
      <td>6673.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6633.0000</td>
      <td>6583.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6673.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1650.0900</td>
      <td>1.0890</td>
      <td>0.4520</td>
      <td>-0.1430</td>
      <td>-0.0060</td>
      <td>-0.2780</td>
      <td>-0.2450</td>
      <td>-0.9110</td>
      <td>15.1330</td>
      <td>0.0470</td>
      <td>0.7160</td>
      <td>-0.0650</td>
      <td>0.8450</td>
      <td>0.3170</td>
      <td>0.0060</td>
      <td>-0.1480</td>
      <td>26.2710</td>
      <td>1697.2040</td>
      <td>26.2620</td>
      <td>67.3140</td>
      <td>6.1790</td>
      <td>25.4080</td>
      <td>23.8790</td>
      <td>40.5840</td>
      <td>0.0110</td>
      <td>0.0600</td>
      <td>0.1390</td>
      <td>75.7710</td>
      <td>1.0040</td>
      <td>62.5430</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8869.6000</td>
      <td>0.5180</td>
      <td>0.7590</td>
      <td>0.8720</td>
      <td>0.4040</td>
      <td>3.7900</td>
      <td>1.4540</td>
      <td>4.1520</td>
      <td>22.1740</td>
      <td>22.4150</td>
      <td>51.0270</td>
      <td>1.4470</td>
      <td>0.2010</td>
      <td>0.4650</td>
      <td>0.0640</td>
      <td>1.1110</td>
      <td>33.8520</td>
      <td>9373.5300</td>
      <td>35.4080</td>
      <td>31.0230</td>
      <td>14.1080</td>
      <td>26.2840</td>
      <td>37.5900</td>
      <td>39.1610</td>
      <td>0.0980</td>
      <td>0.2400</td>
      <td>8.5390</td>
      <td>21.3940</td>
      <td>0.7380</td>
      <td>32.5020</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2837.1700</td>
      <td>0.2140</td>
      <td>0.0000</td>
      <td>-16.5150</td>
      <td>-10.4160</td>
      <td>-262.5700</td>
      <td>-62.3600</td>
      <td>-116.6800</td>
      <td>0.0000</td>
      <td>-1190.0000</td>
      <td>-59.0170</td>
      <td>-1.0000</td>
      <td>0.0070</td>
      <td>0.0000</td>
      <td>-0.2330</td>
      <td>-1.0000</td>
      <td>0.8000</td>
      <td>-2837.1700</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-95.5600</td>
      <td>0.0000</td>
      <td>-0.9950</td>
      <td>-0.9910</td>
      <td>-0.9100</td>
      <td>-25.0000</td>
      <td>0.0920</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>86.1300</td>
      <td>0.9470</td>
      <td>0.1320</td>
      <td>-0.2860</td>
      <td>-0.1270</td>
      <td>-0.1300</td>
      <td>-0.1400</td>
      <td>-0.9400</td>
      <td>6.6000</td>
      <td>0.0320</td>
      <td>-0.0120</td>
      <td>-0.4770</td>
      <td>0.7180</td>
      <td>0.0000</td>
      <td>-0.0140</td>
      <td>-0.4160</td>
      <td>12.0500</td>
      <td>87.4980</td>
      <td>11.9000</td>
      <td>47.3680</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>4.7280</td>
      <td>15.9290</td>
      <td>-0.0120</td>
      <td>-0.0330</td>
      <td>-0.0280</td>
      <td>65.7500</td>
      <td>0.4120</td>
      <td>38.4620</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>336.7100</td>
      <td>1.0000</td>
      <td>0.2840</td>
      <td>0.0380</td>
      <td>0.0000</td>
      <td>-0.0100</td>
      <td>-0.0100</td>
      <td>-0.0400</td>
      <td>9.7000</td>
      <td>0.1890</td>
      <td>0.0000</td>
      <td>-0.2120</td>
      <td>0.9620</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.2130</td>
      <td>17.3000</td>
      <td>344.2500</td>
      <td>17.0000</td>
      <td>75.0000</td>
      <td>0.0000</td>
      <td>20.0000</td>
      <td>17.4850</td>
      <td>35.5860</td>
      <td>0.0000</td>
      <td>0.0140</td>
      <td>0.0000</td>
      <td>79.6250</td>
      <td>0.8450</td>
      <td>66.6670</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1054.9230</td>
      <td>1.1180</td>
      <td>0.4920</td>
      <td>0.2820</td>
      <td>0.1230</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0300</td>
      <td>15.5000</td>
      <td>0.4570</td>
      <td>0.0200</td>
      <td>0.0510</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0150</td>
      <td>-0.0680</td>
      <td>27.8000</td>
      <td>1071.2750</td>
      <td>27.4000</td>
      <td>100.0000</td>
      <td>7.1430</td>
      <td>42.8570</td>
      <td>34.7830</td>
      <td>55.3940</td>
      <td>0.0170</td>
      <td>0.1080</td>
      <td>0.0170</td>
      <td>90.7500</td>
      <td>1.4550</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>266264.7800</td>
      <td>24.0000</td>
      <td>16.5150</td>
      <td>0.9930</td>
      <td>7.3280</td>
      <td>35.4900</td>
      <td>35.2300</td>
      <td>42.5000</td>
      <td>410.3000</td>
      <td>1254.0000</td>
      <td>3455.8320</td>
      <td>62.2270</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>2.3440</td>
      <td>41.6560</td>
      <td>500.0000</td>
      <td>296960.7200</td>
      <td>482.5000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>488.2350</td>
      <td>893.0000</td>
      <td>2.1040</td>
      <td>6.8420</td>
      <td>694.0510</td>
      <td>100.0000</td>
      <td>4.5230</td>
      <td>100.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Balance Sheet (51 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inventory_2fq</th>
      <th>cash_to_assets_pct</th>
      <th>cash_change_qoq</th>
      <th>cash_vs_5y_avg</th>
      <th>inventory_change_yoy</th>
      <th>inventory_vs_5y_avg</th>
      <th>working_capital_vs_5y_avg</th>
      <th>retained_earnings_vs_5y</th>
      <th>intangibles_growth_flag</th>
      <th>asset_quality_score</th>
      <th>balance_sheet_strength</th>
      <th>debt_maturity_risk</th>
      <th>receivables_change_yoy</th>
      <th>receivables_vs_5y_avg</th>
      <th>inventory_1fq</th>
      <th>inventory_1fy</th>
      <th>inventory_2fy</th>
      <th>inventory_3fq</th>
      <th>inventory_3fy</th>
      <th>inventory_4fq</th>
      <th>inventory_4fy</th>
      <th>inventory_4q_trend</th>
      <th>inventory_buildup_flag</th>
      <th>inventory_days</th>
      <th>inventory_fq</th>
      <th>inventory_ltm</th>
      <th>inventory_qoq_change</th>
      <th>inventory_reduction_flag</th>
      <th>inventory_to_assets</th>
      <th>inventory_to_revenue</th>
      <th>inventory_turnover_itf</th>
      <th>inventory_volatility</th>
      <th>inventory_vs_5y_avg_itf</th>
      <th>inventory_yoy_change</th>
      <th>inventory_fy</th>
      <th>asset_base_stable</th>
      <th>asset_growth_accel</th>
      <th>assets_1fq</th>
      <th>assets_1fy</th>
      <th>assets_2fq</th>
      <th>assets_2fy</th>
      <th>assets_3fq</th>
      <th>assets_3fy</th>
      <th>assets_3y_cagr</th>
      <th>assets_4fq</th>
      <th>assets_4fy</th>
      <th>assets_fq</th>
      <th>assets_fy</th>
      <th>assets_ltm</th>
      <th>assets_qoq_growth</th>
      <th>assets_yoy_growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6339.0000</td>
      <td>6556.0000</td>
      <td>6066.0000</td>
      <td>5390.0000</td>
      <td>5433.0000</td>
      <td>6356.0000</td>
      <td>6001.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6621.0000</td>
      <td>6426.0000</td>
      <td>5925.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5414.0000</td>
      <td>6676.0000</td>
      <td>6446.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5594.0000</td>
      <td>6676.0000</td>
      <td>6339.0000</td>
      <td>6532.0000</td>
      <td>5392.0000</td>
      <td>5781.0000</td>
      <td>5433.0000</td>
      <td>5722.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6642.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6578.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6584.0000</td>
      <td>6663.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>840.3040</td>
      <td>13.7740</td>
      <td>0.0980</td>
      <td>1.1670</td>
      <td>0.0410</td>
      <td>1.0860</td>
      <td>1.3470</td>
      <td>0.8740</td>
      <td>0.1070</td>
      <td>56.2750</td>
      <td>58.2230</td>
      <td>1.9590</td>
      <td>0.6810</td>
      <td>1.1530</td>
      <td>879.4440</td>
      <td>801.9840</td>
      <td>807.9220</td>
      <td>822.1080</td>
      <td>803.8290</td>
      <td>786.8180</td>
      <td>702.8210</td>
      <td>23.4270</td>
      <td>0.2040</td>
      <td>105.4700</td>
      <td>843.8510</td>
      <td>845.1190</td>
      <td>-1.9580</td>
      <td>0.1490</td>
      <td>9.8970</td>
      <td>16.1910</td>
      <td>44.9320</td>
      <td>1.0280</td>
      <td>1.0860</td>
      <td>21.1440</td>
      <td>856.2330</td>
      <td>0.3650</td>
      <td>32.3560</td>
      <td>11338.4300</td>
      <td>10585.2100</td>
      <td>10965.8500</td>
      <td>10349.8470</td>
      <td>10664.6770</td>
      <td>10049.8370</td>
      <td>9.2580</td>
      <td>10329.1070</td>
      <td>9706.7840</td>
      <td>11294.7890</td>
      <td>11466.5470</td>
      <td>11303.8370</td>
      <td>0.0860</td>
      <td>51.9490</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3094.7800</td>
      <td>13.8330</td>
      <td>3.9120</td>
      <td>0.9630</td>
      <td>3.7850</td>
      <td>0.6030</td>
      <td>9.3930</td>
      <td>19.7440</td>
      <td>0.3090</td>
      <td>21.9070</td>
      <td>34.8640</td>
      <td>93.1580</td>
      <td>25.0960</td>
      <td>0.6170</td>
      <td>3251.4540</td>
      <td>2967.4680</td>
      <td>2984.9100</td>
      <td>3053.0650</td>
      <td>3024.3840</td>
      <td>2981.4050</td>
      <td>2774.2500</td>
      <td>369.3950</td>
      <td>0.4030</td>
      <td>362.1580</td>
      <td>3163.0040</td>
      <td>3163.9880</td>
      <td>93.3850</td>
      <td>0.3560</td>
      <td>11.3600</td>
      <td>104.6620</td>
      <td>499.1180</td>
      <td>2.0720</td>
      <td>0.6030</td>
      <td>204.6380</td>
      <td>3129.1070</td>
      <td>0.4820</td>
      <td>1920.6430</td>
      <td>34590.3180</td>
      <td>31920.0590</td>
      <td>33696.0160</td>
      <td>30833.2920</td>
      <td>32438.8720</td>
      <td>30435.9020</td>
      <td>33.4160</td>
      <td>31891.2400</td>
      <td>29767.2560</td>
      <td>35490.7890</td>
      <td>35150.9520</td>
      <td>35496.3820</td>
      <td>61.7880</td>
      <td>1881.3220</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>0.0000</td>
      <td>-134.1820</td>
      <td>-1434.6800</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-5493.0000</td>
      <td>-11.4690</td>
      <td>-8.5900</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>-3878.4350</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-2.7260</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-24558.1370</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.0280</td>
      <td>4.8000</td>
      <td>0.0000</td>
      <td>0.7050</td>
      <td>0.0000</td>
      <td>0.8800</td>
      <td>0.7090</td>
      <td>0.8990</td>
      <td>0.0000</td>
      <td>46.0820</td>
      <td>25.0000</td>
      <td>0.4150</td>
      <td>-0.0290</td>
      <td>0.8860</td>
      <td>11.1920</td>
      <td>12.0480</td>
      <td>11.4600</td>
      <td>9.0820</td>
      <td>9.2300</td>
      <td>5.0080</td>
      <td>6.2350</td>
      <td>-2.9410</td>
      <td>0.0000</td>
      <td>8.9480</td>
      <td>4.9180</td>
      <td>4.9620</td>
      <td>-6.2670</td>
      <td>0.0000</td>
      <td>0.7400</td>
      <td>1.1770</td>
      <td>2.6340</td>
      <td>0.1980</td>
      <td>0.8800</td>
      <td>-3.1510</td>
      <td>11.9000</td>
      <td>0.0000</td>
      <td>-3.6100</td>
      <td>743.1650</td>
      <td>673.4320</td>
      <td>626.6200</td>
      <td>622.8680</td>
      <td>667.9950</td>
      <td>555.4100</td>
      <td>-0.1350</td>
      <td>551.4700</td>
      <td>464.0520</td>
      <td>638.7080</td>
      <td>751.5850</td>
      <td>640.0800</td>
      <td>-1.3150</td>
      <td>1.4530</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>115.9200</td>
      <td>9.7800</td>
      <td>0.0000</td>
      <td>1.0300</td>
      <td>0.0000</td>
      <td>1.0950</td>
      <td>1.1210</td>
      <td>1.2050</td>
      <td>0.0000</td>
      <td>55.9310</td>
      <td>75.0000</td>
      <td>1.9480</td>
      <td>0.0940</td>
      <td>1.0890</td>
      <td>129.0500</td>
      <td>119.4850</td>
      <td>118.6300</td>
      <td>120.1400</td>
      <td>111.5750</td>
      <td>103.3000</td>
      <td>91.1550</td>
      <td>8.7000</td>
      <td>0.0000</td>
      <td>57.9810</td>
      <td>112.3200</td>
      <td>112.6550</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>6.7440</td>
      <td>9.8590</td>
      <td>4.8080</td>
      <td>0.3390</td>
      <td>1.0950</td>
      <td>8.1270</td>
      <td>126.3450</td>
      <td>0.0000</td>
      <td>6.6440</td>
      <td>2485.3500</td>
      <td>2275.3950</td>
      <td>2304.8800</td>
      <td>2234.0200</td>
      <td>2296.2300</td>
      <td>2083.9400</td>
      <td>5.5080</td>
      <td>2118.7650</td>
      <td>1866.9300</td>
      <td>2412.7650</td>
      <td>2502.1400</td>
      <td>2410.1000</td>
      <td>0.5800</td>
      <td>8.9270</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>538.6300</td>
      <td>17.7640</td>
      <td>0.0000</td>
      <td>1.3950</td>
      <td>0.0000</td>
      <td>1.3090</td>
      <td>1.5170</td>
      <td>1.4870</td>
      <td>0.0000</td>
      <td>66.5080</td>
      <td>100.0000</td>
      <td>3.9570</td>
      <td>0.2510</td>
      <td>1.3290</td>
      <td>571.1850</td>
      <td>517.8620</td>
      <td>518.3200</td>
      <td>533.6620</td>
      <td>515.2420</td>
      <td>501.4800</td>
      <td>432.2780</td>
      <td>23.0000</td>
      <td>0.0000</td>
      <td>119.6110</td>
      <td>546.4280</td>
      <td>547.8520</td>
      <td>4.5180</td>
      <td>0.0000</td>
      <td>15.3690</td>
      <td>19.3990</td>
      <td>11.5350</td>
      <td>0.6790</td>
      <td>1.3090</td>
      <td>22.3680</td>
      <td>564.1620</td>
      <td>1.0000</td>
      <td>18.5500</td>
      <td>8050.3950</td>
      <td>7451.3400</td>
      <td>7674.5280</td>
      <td>7312.2450</td>
      <td>7566.8650</td>
      <td>6990.9800</td>
      <td>13.3470</td>
      <td>7168.3500</td>
      <td>6644.9650</td>
      <td>7932.8480</td>
      <td>8175.0950</td>
      <td>7932.8480</td>
      <td>4.2450</td>
      <td>19.2060</td>
    </tr>
    <tr>
      <th>max</th>
      <td>108155.7900</td>
      <td>100.0000</td>
      <td>294.3590</td>
      <td>17.3200</td>
      <td>276.5710</td>
      <td>9.3990</td>
      <td>572.7500</td>
      <td>299.2800</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>4325.0000</td>
      <td>1958.0000</td>
      <td>10.2050</td>
      <td>116313.6600</td>
      <td>105790.9700</td>
      <td>112195.7400</td>
      <td>111078.5500</td>
      <td>111867.2800</td>
      <td>105788.0500</td>
      <td>110692.7000</td>
      <td>24083.3330</td>
      <td>1.0000</td>
      <td>14041.0940</td>
      <td>116667.3000</td>
      <td>116667.3000</td>
      <td>5519.8380</td>
      <td>1.0000</td>
      <td>94.4180</td>
      <td>8092.3080</td>
      <td>31396.5000</td>
      <td>10.0000</td>
      <td>9.3990</td>
      <td>9816.6670</td>
      <td>116667.3000</td>
      <td>1.0000</td>
      <td>120722.3460</td>
      <td>727921.0000</td>
      <td>645561.0900</td>
      <td>682170.0000</td>
      <td>661118.9700</td>
      <td>651912.4300</td>
      <td>663258.6600</td>
      <td>933.3800</td>
      <td>645561.0900</td>
      <td>576056.7900</td>
      <td>818042.0000</td>
      <td>818042.0000</td>
      <td>818042.0000</td>
      <td>3662.1610</td>
      <td>120711.1260</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Cash Flow (57 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fcf_fy</th>
      <th>fcf_est_avg_fy1e</th>
      <th>fcf_est_avg_fy2e</th>
      <th>fcf_est_avg_fy3e</th>
      <th>fcf_est_avg_fy4e</th>
      <th>fcf_est_avg_fy5e</th>
      <th>fcf_est_cagr_5y</th>
      <th>fcf_est_trend</th>
      <th>fcf_est_fy2</th>
      <th>cfo_to_net_income</th>
      <th>fcf_to_net_income</th>
      <th>fcf_margin</th>
      <th>cfo_growth_yoy</th>
      <th>fcf_positive_ratio</th>
      <th>acquisition_intensity</th>
      <th>fcf_est_fy3</th>
      <th>fcf_est_growth_fy1_vs_ltm</th>
      <th>fcf_est_growth_fy2_vs_fy1</th>
      <th>fcf_est_cagr_3y</th>
      <th>fcf_est_cagr_5y</th>
      <th>fcf_est_margin_fy1</th>
      <th>fcf_est_yield_fy1</th>
      <th>fcf_est_growth_acceleration</th>
      <th>self_funding_ratio</th>
      <th>fcf_est_fy1</th>
      <th>cff_pattern_score</th>
      <th>cff_quarterly_trend</th>
      <th>cfi_negative_quarters</th>
      <th>cfi_quarterly_trend</th>
      <th>cfo_positive_quarters</th>
      <th>cfo_yoy_quarterly</th>
      <th>fcf_fq</th>
      <th>cfo_quarterly_trend</th>
      <th>fcf_quarterly_trend</th>
      <th>financing_dependency</th>
      <th>operating_cf_momentum</th>
      <th>cfo_fq</th>
      <th>cfo_fy</th>
      <th>cfo_ltm</th>
      <th>cfo_positive_years</th>
      <th>fcf_growth_yoy</th>
      <th>fcf_ltm</th>
      <th>fcf_positive_years_comp</th>
      <th>fcf_yield</th>
      <th>cash_burn_rate</th>
      <th>cf_volatility_score</th>
      <th>fcf_est_always_positive_fwd</th>
      <th>fcf_est_cagr_5y_fwd</th>
      <th>fcf_est_capex_implied_ratio</th>
      <th>fcf_est_growth_deceleration</th>
      <th>fcf_est_growth_fy3_vs_fy2</th>
      <th>fcf_est_growth_fy4_vs_fy3</th>
      <th>fcf_est_growth_fy5_vs_fy4</th>
      <th>fcf_est_trajectory_score_fwd</th>
      <th>fcf_est_fy4</th>
      <th>fcf_est_fy5</th>
      <th>fcf_est_vs_historical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>1604.0000</td>
      <td>5761.0000</td>
      <td>6676.0000</td>
      <td>6668.0000</td>
      <td>6668.0000</td>
      <td>6532.0000</td>
      <td>6638.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6090.0000</td>
      <td>5761.0000</td>
      <td>3418.0000</td>
      <td>1604.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>5284.0000</td>
      <td>6071.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5896.0000</td>
      <td>6676.0000</td>
      <td>5892.0000</td>
      <td>6676.0000</td>
      <td>5928.0000</td>
      <td>6676.0000</td>
      <td>5928.0000</td>
      <td>5968.0000</td>
      <td>6090.0000</td>
      <td>6150.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6655.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6665.0000</td>
      <td>6183.0000</td>
      <td>6676.0000</td>
      <td>1451.0000</td>
      <td>6089.0000</td>
      <td>6676.0000</td>
      <td>5737.0000</td>
      <td>4842.0000</td>
      <td>2510.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6082.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>508.3510</td>
      <td>611.2520</td>
      <td>774.1130</td>
      <td>862.2630</td>
      <td>716.1820</td>
      <td>713.9480</td>
      <td>16.8830</td>
      <td>-0.6220</td>
      <td>774.1130</td>
      <td>2.4510</td>
      <td>1.1550</td>
      <td>-1.4270</td>
      <td>2.1000</td>
      <td>0.6200</td>
      <td>113.8000</td>
      <td>862.2630</td>
      <td>93.8310</td>
      <td>98.8690</td>
      <td>19.3320</td>
      <td>16.8830</td>
      <td>-114.8960</td>
      <td>3.7580</td>
      <td>-24.8500</td>
      <td>3.8820</td>
      <td>611.2520</td>
      <td>0.3630</td>
      <td>945.3640</td>
      <td>3.8320</td>
      <td>-582.5100</td>
      <td>3.6400</td>
      <td>98.8790</td>
      <td>153.1390</td>
      <td>98.8790</td>
      <td>50.1750</td>
      <td>2.1210</td>
      <td>292.9200</td>
      <td>308.4940</td>
      <td>1074.3870</td>
      <td>1044.2780</td>
      <td>4.2260</td>
      <td>234.5270</td>
      <td>500.5350</td>
      <td>3.5640</td>
      <td>4.2810</td>
      <td>0.2930</td>
      <td>15.4340</td>
      <td>0.2350</td>
      <td>14.5370</td>
      <td>0.8200</td>
      <td>0.1610</td>
      <td>44.7040</td>
      <td>-22.2560</td>
      <td>21.5670</td>
      <td>55.3860</td>
      <td>716.1820</td>
      <td>713.9480</td>
      <td>-112.5220</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3019.8630</td>
      <td>4354.2900</td>
      <td>5206.1770</td>
      <td>6131.0920</td>
      <td>6648.7400</td>
      <td>7417.7700</td>
      <td>23.0860</td>
      <td>5.3720</td>
      <td>5206.1770</td>
      <td>34.4980</td>
      <td>18.3020</td>
      <td>44.4230</td>
      <td>134.0080</td>
      <td>0.3580</td>
      <td>681.7380</td>
      <td>6131.0920</td>
      <td>2386.8060</td>
      <td>1107.5780</td>
      <td>56.0810</td>
      <td>23.0860</td>
      <td>4107.7770</td>
      <td>13.9350</td>
      <td>2673.8530</td>
      <td>121.2810</td>
      <td>4354.2900</td>
      <td>0.8900</td>
      <td>52151.5150</td>
      <td>1.5860</td>
      <td>11724.4350</td>
      <td>1.7370</td>
      <td>3214.0090</td>
      <td>1142.8040</td>
      <td>3214.0090</td>
      <td>2174.7330</td>
      <td>13.2410</td>
      <td>9520.8370</td>
      <td>1724.8070</td>
      <td>5164.3820</td>
      <td>5347.8610</td>
      <td>1.4160</td>
      <td>10345.6940</td>
      <td>3179.6140</td>
      <td>1.6780</td>
      <td>17.5540</td>
      <td>19.4190</td>
      <td>290.4350</td>
      <td>0.4240</td>
      <td>23.8710</td>
      <td>25.0230</td>
      <td>0.3670</td>
      <td>627.0760</td>
      <td>195.8550</td>
      <td>431.5660</td>
      <td>34.1370</td>
      <td>6648.7400</td>
      <td>7417.7700</td>
      <td>10582.3370</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-14846.1000</td>
      <td>-28372.0300</td>
      <td>-28652.1200</td>
      <td>-18082.0000</td>
      <td>-37183.0000</td>
      <td>-18644.0000</td>
      <td>-49.6620</td>
      <td>-241.4650</td>
      <td>-28652.1200</td>
      <td>-513.1320</td>
      <td>-385.4380</td>
      <td>-2659.0000</td>
      <td>-334.8130</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-18082.0000</td>
      <td>-114800.0000</td>
      <td>-23608.3330</td>
      <td>-89.4670</td>
      <td>-49.6620</td>
      <td>-298916.6670</td>
      <td>-429.3200</td>
      <td>-93366.4140</td>
      <td>-4326.3330</td>
      <td>-28372.0300</td>
      <td>-1.0000</td>
      <td>-1869250.0000</td>
      <td>0.0000</td>
      <td>-749977.7780</td>
      <td>0.0000</td>
      <td>-120420.0000</td>
      <td>-15066.0900</td>
      <td>-120420.0000</td>
      <td>-63100.0000</td>
      <td>0.0000</td>
      <td>-30800.0000</td>
      <td>-13111.7200</td>
      <td>-5462.8800</td>
      <td>-5462.8800</td>
      <td>0.0000</td>
      <td>-42291.0210</td>
      <td>-24736.0000</td>
      <td>0.0000</td>
      <td>-254.6270</td>
      <td>0.0000</td>
      <td>0.0690</td>
      <td>0.0000</td>
      <td>-34.6730</td>
      <td>-1147.0000</td>
      <td>0.0000</td>
      <td>-6245.0660</td>
      <td>-2069.9540</td>
      <td>-11264.0000</td>
      <td>0.0000</td>
      <td>-37183.0000</td>
      <td>-18644.0000</td>
      <td>-798080.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>3.2450</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>5.1150</td>
      <td>-1.0000</td>
      <td>3.2450</td>
      <td>0.2260</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.3620</td>
      <td>0.4000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-45.1680</td>
      <td>1.9990</td>
      <td>-1.7970</td>
      <td>5.1150</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-44.9550</td>
      <td>0.6680</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>-105.7640</td>
      <td>3.0000</td>
      <td>-105.5520</td>
      <td>3.0000</td>
      <td>-34.2180</td>
      <td>0.0000</td>
      <td>-34.2180</td>
      <td>-52.7010</td>
      <td>0.3240</td>
      <td>-24.1660</td>
      <td>0.0000</td>
      <td>37.2720</td>
      <td>17.2200</td>
      <td>4.0000</td>
      <td>-38.5280</td>
      <td>0.0000</td>
      <td>3.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.5270</td>
      <td>0.0000</td>
      <td>2.1400</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-5.1650</td>
      <td>-100.0000</td>
      <td>-6.3400</td>
      <td>40.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-98.3280</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>81.0500</td>
      <td>82.5900</td>
      <td>119.8550</td>
      <td>90.3250</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>11.7300</td>
      <td>-1.0000</td>
      <td>119.8550</td>
      <td>1.3130</td>
      <td>0.8180</td>
      <td>0.0590</td>
      <td>0.0480</td>
      <td>0.8000</td>
      <td>0.0000</td>
      <td>90.3250</td>
      <td>1.3660</td>
      <td>17.6660</td>
      <td>9.1310</td>
      <td>11.7300</td>
      <td>5.8070</td>
      <td>3.7410</td>
      <td>13.4580</td>
      <td>1.6410</td>
      <td>82.5900</td>
      <td>1.0000</td>
      <td>-5.5650</td>
      <td>5.0000</td>
      <td>-8.5420</td>
      <td>4.0000</td>
      <td>7.8720</td>
      <td>16.6750</td>
      <td>7.8720</td>
      <td>7.2010</td>
      <td>0.6180</td>
      <td>22.2300</td>
      <td>41.7650</td>
      <td>198.9350</td>
      <td>165.5950</td>
      <td>5.0000</td>
      <td>8.1340</td>
      <td>65.1000</td>
      <td>4.0000</td>
      <td>3.2910</td>
      <td>0.0000</td>
      <td>2.8590</td>
      <td>0.0000</td>
      <td>9.1580</td>
      <td>0.7750</td>
      <td>0.0000</td>
      <td>11.4590</td>
      <td>-16.6090</td>
      <td>7.5680</td>
      <td>60.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-15.2960</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>361.5600</td>
      <td>385.6000</td>
      <td>456.5970</td>
      <td>463.6980</td>
      <td>175.6250</td>
      <td>84.4780</td>
      <td>21.7850</td>
      <td>0.0260</td>
      <td>456.5970</td>
      <td>2.2580</td>
      <td>1.4720</td>
      <td>0.1330</td>
      <td>0.3650</td>
      <td>1.0000</td>
      <td>10.7050</td>
      <td>463.6980</td>
      <td>73.0820</td>
      <td>54.8250</td>
      <td>24.0280</td>
      <td>21.7850</td>
      <td>13.3910</td>
      <td>7.4070</td>
      <td>68.1500</td>
      <td>3.2680</td>
      <td>385.6000</td>
      <td>1.0000</td>
      <td>87.9600</td>
      <td>5.0000</td>
      <td>54.9830</td>
      <td>5.0000</td>
      <td>56.6430</td>
      <td>104.5750</td>
      <td>56.6430</td>
      <td>74.1220</td>
      <td>1.0540</td>
      <td>92.2380</td>
      <td>186.0700</td>
      <td>691.4500</td>
      <td>651.0200</td>
      <td>5.0000</td>
      <td>68.3120</td>
      <td>338.5600</td>
      <td>5.0000</td>
      <td>7.7650</td>
      <td>0.0000</td>
      <td>5.7360</td>
      <td>0.0000</td>
      <td>20.0150</td>
      <td>1.2110</td>
      <td>0.0000</td>
      <td>29.9700</td>
      <td>15.4100</td>
      <td>21.1180</td>
      <td>80.0000</td>
      <td>175.6250</td>
      <td>84.4780</td>
      <td>77.9760</td>
    </tr>
    <tr>
      <th>max</th>
      <td>98767.0000</td>
      <td>181667.4400</td>
      <td>234892.4600</td>
      <td>288666.8400</td>
      <td>376909.5000</td>
      <td>410181.5000</td>
      <td>201.0880</td>
      <td>81.1820</td>
      <td>234892.4600</td>
      <td>1702.0000</td>
      <td>755.0000</td>
      <td>31.8970</td>
      <td>10680.3330</td>
      <td>1.0000</td>
      <td>17541.0000</td>
      <td>288666.8400</td>
      <td>93322.9510</td>
      <td>62415.3850</td>
      <td>1109.3560</td>
      <td>201.0880</td>
      <td>38461.5380</td>
      <td>255.9800</td>
      <td>115048.4740</td>
      <td>4797.0000</td>
      <td>181667.4400</td>
      <td>1.0000</td>
      <td>2400590.0000</td>
      <td>5.0000</td>
      <td>24200.0000</td>
      <td>5.0000</td>
      <td>140680.0000</td>
      <td>51552.0000</td>
      <td>140680.0000</td>
      <td>66816.6670</td>
      <td>509.7620</td>
      <td>680800.0000</td>
      <td>54459.0000</td>
      <td>164713.0000</td>
      <td>164713.0000</td>
      <td>5.0000</td>
      <td>797980.0000</td>
      <td>123324.0000</td>
      <td>5.0000</td>
      <td>753.8270</td>
      <td>1583.5830</td>
      <td>15791.9230</td>
      <td>1.0000</td>
      <td>314.8320</td>
      <td>919.1610</td>
      <td>1.0000</td>
      <td>27177.7780</td>
      <td>7310.1270</td>
      <td>7253.7570</td>
      <td>100.0000</td>
      <td>376909.5000</td>
      <td>410181.5000</td>
      <td>93209.2870</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Composite Scores (3 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dilution_score</th>
      <th>quality_momentum_score</th>
      <th>piotroski_f_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48.7980</td>
      <td>53.0880</td>
      <td>3.4710</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.0550</td>
      <td>13.8550</td>
      <td>1.5620</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>7.5000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>49.0330</td>
      <td>43.6720</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.0000</td>
      <td>53.2510</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>50.8390</td>
      <td>62.6050</td>
      <td>4.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.0000</td>
      <td>90.0000</td>
      <td>9.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Dividend Reliability (34 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>div_yield_3fyind</th>
      <th>div_yield_5y_trend</th>
      <th>div_yield_stability</th>
      <th>div_yield_2fyind</th>
      <th>div_yield_4fyind</th>
      <th>div_yield_declining_flag</th>
      <th>div_yield_mean_5y</th>
      <th>div_yield_vs_5y_mean</th>
      <th>div_yield_5fyind</th>
      <th>dividend_streak</th>
      <th>dividend_yield_ltm</th>
      <th>dividend_yield_ntm</th>
      <th>dividend_payout_ratio</th>
      <th>fcf_dividend_coverage</th>
      <th>buyback_yield</th>
      <th>total_shareholder_yield</th>
      <th>dividend_growth_expectation</th>
      <th>div_yield_1fy_ind</th>
      <th>div_yield_vs_5y_avg</th>
      <th>high_yield_flag</th>
      <th>sustainable_dividend_flag</th>
      <th>days_since_ex_date</th>
      <th>days_to_payment</th>
      <th>dividend_announced_flag</th>
      <th>dividend_consistency</th>
      <th>dividend_frequency_score</th>
      <th>dividend_yield_vs_5y_avg</th>
      <th>ex_date_approaching_flag</th>
      <th>recent_dividend_change</th>
      <th>div_yield_5y_avg</th>
      <th>div_yield_growth_expected</th>
      <th>div_yield_ind</th>
      <th>div_yield_ltm</th>
      <th>div_yield_ntm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4762.0000</td>
      <td>4471.0000</td>
      <td>5215.0000</td>
      <td>4906.0000</td>
      <td>4579.0000</td>
      <td>6676.0000</td>
      <td>5215.0000</td>
      <td>4905.0000</td>
      <td>4410.0000</td>
      <td>6676.0000</td>
      <td>4220.0000</td>
      <td>5821.0000</td>
      <td>3861.0000</td>
      <td>4089.0000</td>
      <td>4274.0000</td>
      <td>6676.0000</td>
      <td>3955.0000</td>
      <td>5043.0000</td>
      <td>3669.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4621.0000</td>
      <td>4568.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>3669.0000</td>
      <td>6676.0000</td>
      <td>4398.0000</td>
      <td>4508.0000</td>
      <td>3934.0000</td>
      <td>4921.0000</td>
      <td>4220.0000</td>
      <td>5821.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0300</td>
      <td>-0.0010</td>
      <td>0.0330</td>
      <td>0.0290</td>
      <td>0.0210</td>
      <td>0.0650</td>
      <td>0.0270</td>
      <td>0.0900</td>
      <td>0.0220</td>
      <td>2.2350</td>
      <td>0.0320</td>
      <td>0.6770</td>
      <td>0.4360</td>
      <td>14.9120</td>
      <td>-0.0060</td>
      <td>0.0160</td>
      <td>0.0010</td>
      <td>0.0280</td>
      <td>1.0540</td>
      <td>0.1040</td>
      <td>0.0780</td>
      <td>94.3210</td>
      <td>-74.1160</td>
      <td>0.0680</td>
      <td>0.1690</td>
      <td>0.8260</td>
      <td>1.0540</td>
      <td>0.0500</td>
      <td>10.3090</td>
      <td>0.0320</td>
      <td>60.1640</td>
      <td>0.0280</td>
      <td>0.0320</td>
      <td>0.6770</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0450</td>
      <td>0.0220</td>
      <td>0.0680</td>
      <td>0.0390</td>
      <td>0.0320</td>
      <td>0.2470</td>
      <td>0.0300</td>
      <td>0.8200</td>
      <td>0.0290</td>
      <td>5.3500</td>
      <td>0.0520</td>
      <td>49.5360</td>
      <td>7.3670</td>
      <td>406.0630</td>
      <td>0.0900</td>
      <td>0.0860</td>
      <td>0.2200</td>
      <td>0.0350</td>
      <td>0.7620</td>
      <td>0.3050</td>
      <td>0.2690</td>
      <td>132.4760</td>
      <td>140.5640</td>
      <td>0.2520</td>
      <td>0.2410</td>
      <td>1.4300</td>
      <td>0.7620</td>
      <td>0.2180</td>
      <td>155.9320</td>
      <td>0.0360</td>
      <td>678.4170</td>
      <td>0.0530</td>
      <td>0.0520</td>
      <td>49.5360</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>-0.3530</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-300.6000</td>
      <td>-6229.0000</td>
      <td>-2.1120</td>
      <td>-2.1120</td>
      <td>-1.8520</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-586.0000</td>
      <td>-365.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0070</td>
      <td>-0.0040</td>
      <td>0.0070</td>
      <td>0.0070</td>
      <td>0.0040</td>
      <td>0.0000</td>
      <td>0.0090</td>
      <td>-0.3400</td>
      <td>0.0020</td>
      <td>0.0000</td>
      <td>0.0100</td>
      <td>0.0020</td>
      <td>0.0000</td>
      <td>0.7220</td>
      <td>-0.0020</td>
      <td>0.0000</td>
      <td>-0.0020</td>
      <td>0.0080</td>
      <td>0.6550</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>7.0000</td>
      <td>-190.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.6550</td>
      <td>0.0000</td>
      <td>-37.2980</td>
      <td>0.0120</td>
      <td>-7.2070</td>
      <td>0.0080</td>
      <td>0.0100</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0200</td>
      <td>-0.0000</td>
      <td>0.0180</td>
      <td>0.0210</td>
      <td>0.0140</td>
      <td>0.0000</td>
      <td>0.0200</td>
      <td>-0.0010</td>
      <td>0.0150</td>
      <td>1.0000</td>
      <td>0.0230</td>
      <td>0.0180</td>
      <td>0.1120</td>
      <td>1.9740</td>
      <td>0.0020</td>
      <td>0.0150</td>
      <td>0.0010</td>
      <td>0.0200</td>
      <td>0.9550</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>49.0000</td>
      <td>-28.0000</td>
      <td>0.0000</td>
      <td>0.1000</td>
      <td>0.0000</td>
      <td>0.9550</td>
      <td>0.0000</td>
      <td>-7.3950</td>
      <td>0.0240</td>
      <td>5.9790</td>
      <td>0.0210</td>
      <td>0.0230</td>
      <td>0.0180</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0400</td>
      <td>0.0030</td>
      <td>0.0370</td>
      <td>0.0390</td>
      <td>0.0280</td>
      <td>0.0000</td>
      <td>0.0360</td>
      <td>0.3350</td>
      <td>0.0320</td>
      <td>2.0000</td>
      <td>0.0400</td>
      <td>0.0360</td>
      <td>0.5380</td>
      <td>3.9350</td>
      <td>0.0170</td>
      <td>0.0410</td>
      <td>0.0050</td>
      <td>0.0390</td>
      <td>1.3100</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>199.0000</td>
      <td>20.0000</td>
      <td>0.0000</td>
      <td>0.2000</td>
      <td>1.0000</td>
      <td>1.3100</td>
      <td>0.0000</td>
      <td>28.3050</td>
      <td>0.0410</td>
      <td>26.6360</td>
      <td>0.0370</td>
      <td>0.0400</td>
      <td>0.0360</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.1340</td>
      <td>0.6620</td>
      <td>2.2000</td>
      <td>0.8210</td>
      <td>0.8220</td>
      <td>1.0000</td>
      <td>0.6190</td>
      <td>4.0000</td>
      <td>0.4880</td>
      <td>54.0000</td>
      <td>1.8520</td>
      <td>3779.3930</td>
      <td>250.0000</td>
      <td>15033.0000</td>
      <td>0.9390</td>
      <td>1.8530</td>
      <td>13.4100</td>
      <td>0.6680</td>
      <td>22.4440</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>366.0000</td>
      <td>593.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>12.0000</td>
      <td>22.4440</td>
      <td>1.0000</td>
      <td>6250.0000</td>
      <td>1.3470</td>
      <td>29669.2480</td>
      <td>2.2000</td>
      <td>1.8520</td>
      <td>3779.3930</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      EPS Trajectory (11 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eps_qoq_growth</th>
      <th>eps_yoy_quarterly</th>
      <th>eps_positive_streak</th>
      <th>eps_cagr_3y</th>
      <th>eps_cagr_5y</th>
      <th>eps_improvement_count</th>
      <th>eps_trajectory_score</th>
      <th>composite_eps_trajectory_score</th>
      <th>eps_growth_accel</th>
      <th>eps_stability</th>
      <th>eps_vs_5y_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6034.0000</td>
      <td>5986.0000</td>
      <td>6676.0000</td>
      <td>4520.0000</td>
      <td>3974.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>3752.0000</td>
      <td>6267.0000</td>
      <td>6267.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-4.8130</td>
      <td>45.8250</td>
      <td>3.6180</td>
      <td>8.5700</td>
      <td>11.1340</td>
      <td>2.7470</td>
      <td>54.9310</td>
      <td>54.9310</td>
      <td>-3.8180</td>
      <td>0.4480</td>
      <td>4.3880</td>
    </tr>
    <tr>
      <th>std</th>
      <td>979.3410</td>
      <td>1359.1130</td>
      <td>1.7880</td>
      <td>43.5310</td>
      <td>25.2340</td>
      <td>1.0960</td>
      <td>21.9300</td>
      <td>21.9300</td>
      <td>36.0950</td>
      <td>0.3400</td>
      <td>1323.8730</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-34200.0000</td>
      <td>-13533.3330</td>
      <td>0.0000</td>
      <td>-93.6150</td>
      <td>-80.5010</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-280.8250</td>
      <td>0.0000</td>
      <td>-49457.1430</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-33.3330</td>
      <td>-28.5710</td>
      <td>2.0000</td>
      <td>-11.2390</td>
      <td>-1.2760</td>
      <td>2.0000</td>
      <td>40.0000</td>
      <td>40.0000</td>
      <td>-18.3070</td>
      <td>0.0000</td>
      <td>-30.4900</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>9.0910</td>
      <td>5.0000</td>
      <td>4.0400</td>
      <td>8.5410</td>
      <td>3.0000</td>
      <td>60.0000</td>
      <td>60.0000</td>
      <td>-5.1550</td>
      <td>0.5300</td>
      <td>12.1320</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27.1220</td>
      <td>66.4020</td>
      <td>5.0000</td>
      <td>20.1230</td>
      <td>20.1120</td>
      <td>3.0000</td>
      <td>60.0000</td>
      <td>60.0000</td>
      <td>5.5160</td>
      <td>0.7560</td>
      <td>54.4890</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22300.0000</td>
      <td>71900.0000</td>
      <td>5.0000</td>
      <td>805.3180</td>
      <td>259.2030</td>
      <td>5.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>588.4050</td>
      <td>1.0000</td>
      <td>46100.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Earnings Quality (76 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eps_surprise_pct</th>
      <th>revenue_surprise_pct</th>
      <th>gaap_adj_eps_gap_pct</th>
      <th>ebitda_adjustment_ratio</th>
      <th>eps_quarterly_trend</th>
      <th>eps_adjustment_ratio</th>
      <th>eps_yoy_growth</th>
      <th>eps_adj_ltm</th>
      <th>eps_basic_fq</th>
      <th>eps_basic_fy</th>
      <th>eps_cont_1fy</th>
      <th>eps_cont_2fqfq</th>
      <th>eps_cont_2fy</th>
      <th>eps_cont_3fy</th>
      <th>eps_cont_4fy</th>
      <th>eps_cont_3fqfq</th>
      <th>eps_cont_cagr_3y</th>
      <th>eps_cont_fq</th>
      <th>eps_cont_fy</th>
      <th>eps_cont_ltm</th>
      <th>eps_cont_positive_streak</th>
      <th>eps_cont_qoq_growth</th>
      <th>eps_cont_trajectory_score</th>
      <th>eps_cont_vs_total_eps</th>
      <th>eps_cont_yoy_growth</th>
      <th>eps_basic_ltm</th>
      <th>eps_norm_est_fy1e</th>
      <th>eps_positive_years</th>
      <th>core_earnings_stability</th>
      <th>discontinued_ops_impact</th>
      <th>...</th>
      <th>net_income_is_3fy</th>
      <th>net_income_is_4fqfq</th>
      <th>net_income_is_4fy</th>
      <th>net_income_is_5yavgfq</th>
      <th>net_income_is_5yavgltm</th>
      <th>net_income_is_fq</th>
      <th>net_income_is_ltm</th>
      <th>gaap_vs_norm_revision_spread</th>
      <th>net_income_margin_ltm</th>
      <th>net_income_positive_years</th>
      <th>net_income_qoq_growth</th>
      <th>net_income_vs_5y_avg</th>
      <th>net_income_yoy_quarterly</th>
      <th>ni_adjustment_ratio</th>
      <th>normalized_ni_5yavgfq</th>
      <th>normalized_ni_5yavgltm</th>
      <th>normalized_ni_ltm</th>
      <th>normalized_ni_vs_5y_avg</th>
      <th>net_income_is_2fqfq</th>
      <th>earnings_quality_composite</th>
      <th>net_income_is_fy</th>
      <th>earnings_quality_impact</th>
      <th>has_unusual_items_flag</th>
      <th>impairment_goodwill_ltm</th>
      <th>other_unusual_items_ltm</th>
      <th>restructuring_charges_ltm</th>
      <th>total_unusual_items</th>
      <th>unusual_asset_writedown_ltm</th>
      <th>unusual_items_to_ebitda</th>
      <th>unusual_items_to_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3506.0000</td>
      <td>6432.0000</td>
      <td>5448.0000</td>
      <td>6621.0000</td>
      <td>5986.0000</td>
      <td>3535.0000</td>
      <td>6443.0000</td>
      <td>3572.0000</td>
      <td>6496.0000</td>
      <td>6586.0000</td>
      <td>6604.0000</td>
      <td>6519.0000</td>
      <td>6586.0000</td>
      <td>6521.0000</td>
      <td>6395.0000</td>
      <td>6467.0000</td>
      <td>4533.0000</td>
      <td>6496.0000</td>
      <td>6587.0000</td>
      <td>6575.0000</td>
      <td>6676.0000</td>
      <td>6033.0000</td>
      <td>6676.0000</td>
      <td>6478.0000</td>
      <td>6438.0000</td>
      <td>6575.0000</td>
      <td>5697.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6478.0000</td>
      <td>...</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4621.0000</td>
      <td>6339.0000</td>
      <td>6676.0000</td>
      <td>6615.0000</td>
      <td>6201.0000</td>
      <td>6476.0000</td>
      <td>6668.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6199.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6621.0000</td>
      <td>6532.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-27.0690</td>
      <td>8.5200</td>
      <td>-13.5370</td>
      <td>0.4160</td>
      <td>0.4580</td>
      <td>1.1580</td>
      <td>7.6380</td>
      <td>2.5590</td>
      <td>0.7370</td>
      <td>2.0290</td>
      <td>1.8010</td>
      <td>0.4330</td>
      <td>-49.5980</td>
      <td>-24.3240</td>
      <td>-48.2610</td>
      <td>0.4030</td>
      <td>8.7000</td>
      <td>0.7390</td>
      <td>2.0250</td>
      <td>2.2460</td>
      <td>3.6210</td>
      <td>8.0430</td>
      <td>52.9880</td>
      <td>0.9970</td>
      <td>-12.4980</td>
      <td>2.2520</td>
      <td>3.5890</td>
      <td>3.8620</td>
      <td>73.8330</td>
      <td>-2.1130</td>
      <td>...</td>
      <td>552.7110</td>
      <td>126.1520</td>
      <td>511.4130</td>
      <td>129.8010</td>
      <td>493.6150</td>
      <td>138.0390</td>
      <td>571.9310</td>
      <td>0.7190</td>
      <td>0.0660</td>
      <td>3.9540</td>
      <td>-21.6970</td>
      <td>0.8970</td>
      <td>99.8890</td>
      <td>0.7290</td>
      <td>110.3040</td>
      <td>421.0680</td>
      <td>492.0140</td>
      <td>1.6740</td>
      <td>146.0650</td>
      <td>58.1090</td>
      <td>563.1050</td>
      <td>81.2350</td>
      <td>0.6950</td>
      <td>-15.3530</td>
      <td>-38.6920</td>
      <td>-21.7530</td>
      <td>-106.8230</td>
      <td>-31.0260</td>
      <td>30.2360</td>
      <td>6.9250</td>
    </tr>
    <tr>
      <th>std</th>
      <td>734.5430</td>
      <td>1194.6510</td>
      <td>179.0110</td>
      <td>9.8840</td>
      <td>13.5910</td>
      <td>7.3470</td>
      <td>1427.9900</td>
      <td>10.1420</td>
      <td>20.9070</td>
      <td>53.1410</td>
      <td>45.7650</td>
      <td>10.2570</td>
      <td>4222.8790</td>
      <td>1471.6110</td>
      <td>3849.6900</td>
      <td>7.6650</td>
      <td>45.1090</td>
      <td>20.9050</td>
      <td>53.1380</td>
      <td>50.4230</td>
      <td>1.7890</td>
      <td>914.8330</td>
      <td>25.3230</td>
      <td>1.2530</td>
      <td>1450.1220</td>
      <td>50.4230</td>
      <td>55.7060</td>
      <td>1.6550</td>
      <td>33.8520</td>
      <td>125.2550</td>
      <td>...</td>
      <td>3358.2730</td>
      <td>992.7270</td>
      <td>2902.3130</td>
      <td>753.2130</td>
      <td>2840.2990</td>
      <td>1230.1500</td>
      <td>3928.9460</td>
      <td>50.8670</td>
      <td>0.6500</td>
      <td>1.6040</td>
      <td>2296.8120</td>
      <td>19.8210</td>
      <td>5096.5700</td>
      <td>9.2720</td>
      <td>671.4280</td>
      <td>2550.6120</td>
      <td>3130.0310</td>
      <td>24.4920</td>
      <td>939.7410</td>
      <td>22.7030</td>
      <td>3804.4600</td>
      <td>31.4080</td>
      <td>0.4600</td>
      <td>188.7640</td>
      <td>417.5100</td>
      <td>290.7270</td>
      <td>803.5630</td>
      <td>306.1030</td>
      <td>281.0510</td>
      <td>170.4670</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-21000.0000</td>
      <td>-157.2730</td>
      <td>-6200.0000</td>
      <td>-665.1240</td>
      <td>-135.3330</td>
      <td>-67.0000</td>
      <td>-98350.0000</td>
      <td>-70.5000</td>
      <td>-135.6000</td>
      <td>-1363.3800</td>
      <td>-1189.7300</td>
      <td>-514.9100</td>
      <td>-342671.1000</td>
      <td>-97941.7700</td>
      <td>-307654.0000</td>
      <td>-222.1200</td>
      <td>-93.6150</td>
      <td>-135.6000</td>
      <td>-1363.3800</td>
      <td>-191.7700</td>
      <td>0.0000</td>
      <td>-22300.0000</td>
      <td>0.0000</td>
      <td>-10.0000</td>
      <td>-98350.0000</td>
      <td>-183.4800</td>
      <td>-21.6200</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-9750.0000</td>
      <td>...</td>
      <td>-20343.7200</td>
      <td>-23333.6800</td>
      <td>-12285.3900</td>
      <td>-1219.9000</td>
      <td>-5876.4300</td>
      <td>-12436.6500</td>
      <td>-26278.2000</td>
      <td>-59.0180</td>
      <td>-2.9710</td>
      <td>0.0000</td>
      <td>-145130.0000</td>
      <td>-1206.7500</td>
      <td>-37700.0000</td>
      <td>-82.2660</td>
      <td>-907.4400</td>
      <td>-3472.4800</td>
      <td>-7283.0000</td>
      <td>-243.0000</td>
      <td>-7824.0000</td>
      <td>5.0000</td>
      <td>-26278.2000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-6734.0000</td>
      <td>-16834.1600</td>
      <td>-20197.3700</td>
      <td>-34060.8900</td>
      <td>-16440.3400</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-43.6600</td>
      <td>-15.0730</td>
      <td>-11.7300</td>
      <td>0.0000</td>
      <td>-0.2860</td>
      <td>0.6790</td>
      <td>-22.0660</td>
      <td>0.0800</td>
      <td>0.0000</td>
      <td>0.0300</td>
      <td>0.0300</td>
      <td>0.0100</td>
      <td>0.0300</td>
      <td>0.0300</td>
      <td>0.0200</td>
      <td>0.0000</td>
      <td>-10.8590</td>
      <td>0.0000</td>
      <td>0.0300</td>
      <td>0.0300</td>
      <td>2.0000</td>
      <td>-40.0000</td>
      <td>25.0000</td>
      <td>1.0000</td>
      <td>-39.0860</td>
      <td>0.0300</td>
      <td>0.1300</td>
      <td>3.0000</td>
      <td>60.3860</td>
      <td>0.0000</td>
      <td>...</td>
      <td>5.8020</td>
      <td>0.0000</td>
      <td>3.2180</td>
      <td>0.3000</td>
      <td>2.4950</td>
      <td>0.7100</td>
      <td>8.0180</td>
      <td>-0.0160</td>
      <td>0.0180</td>
      <td>3.0000</td>
      <td>-40.4140</td>
      <td>0.5840</td>
      <td>-56.0340</td>
      <td>0.0000</td>
      <td>0.9700</td>
      <td>4.3320</td>
      <td>10.4500</td>
      <td>0.7210</td>
      <td>1.2300</td>
      <td>45.0000</td>
      <td>7.9000</td>
      <td>80.8080</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-6.6230</td>
      <td>0.0000</td>
      <td>-23.9400</td>
      <td>-2.4700</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-13.9720</td>
      <td>-6.2780</td>
      <td>-0.2890</td>
      <td>0.5350</td>
      <td>0.0910</td>
      <td>1.0000</td>
      <td>11.1110</td>
      <td>0.7100</td>
      <td>0.0800</td>
      <td>0.3200</td>
      <td>0.2800</td>
      <td>0.0800</td>
      <td>0.2900</td>
      <td>0.3100</td>
      <td>0.2800</td>
      <td>0.0600</td>
      <td>4.1280</td>
      <td>0.0800</td>
      <td>0.3200</td>
      <td>0.3200</td>
      <td>5.0000</td>
      <td>0.0000</td>
      <td>50.0000</td>
      <td>1.0000</td>
      <td>3.5390</td>
      <td>0.3200</td>
      <td>0.6600</td>
      <td>5.0000</td>
      <td>89.6010</td>
      <td>0.0000</td>
      <td>...</td>
      <td>79.5100</td>
      <td>18.4100</td>
      <td>68.1800</td>
      <td>18.8050</td>
      <td>73.7150</td>
      <td>23.0400</td>
      <td>97.8350</td>
      <td>0.0000</td>
      <td>0.0640</td>
      <td>5.0000</td>
      <td>0.0000</td>
      <td>1.1170</td>
      <td>1.7090</td>
      <td>0.2400</td>
      <td>15.9350</td>
      <td>60.8700</td>
      <td>81.9750</td>
      <td>1.1440</td>
      <td>23.3250</td>
      <td>60.0000</td>
      <td>95.9300</td>
      <td>97.0040</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.1850</td>
      <td>0.0000</td>
      <td>1.6250</td>
      <td>0.2520</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-2.5400</td>
      <td>-1.3750</td>
      <td>0.0000</td>
      <td>1.0570</td>
      <td>0.6640</td>
      <td>1.1200</td>
      <td>50.0000</td>
      <td>3.0120</td>
      <td>0.4700</td>
      <td>1.6400</td>
      <td>1.5300</td>
      <td>0.4200</td>
      <td>1.5600</td>
      <td>1.6500</td>
      <td>1.5150</td>
      <td>0.3700</td>
      <td>20.1230</td>
      <td>0.4600</td>
      <td>1.6400</td>
      <td>1.6700</td>
      <td>5.0000</td>
      <td>18.7500</td>
      <td>75.0000</td>
      <td>1.0000</td>
      <td>34.2110</td>
      <td>1.6800</td>
      <td>2.7100</td>
      <td>5.0000</td>
      <td>100.0000</td>
      <td>0.0000</td>
      <td>...</td>
      <td>345.3470</td>
      <td>81.0000</td>
      <td>315.0350</td>
      <td>77.5320</td>
      <td>298.6420</td>
      <td>92.7200</td>
      <td>350.8700</td>
      <td>0.0260</td>
      <td>0.1310</td>
      <td>5.0000</td>
      <td>21.5520</td>
      <td>1.5960</td>
      <td>45.3220</td>
      <td>1.0000</td>
      <td>63.1250</td>
      <td>241.7520</td>
      <td>296.9600</td>
      <td>1.5800</td>
      <td>88.9680</td>
      <td>70.0000</td>
      <td>348.7020</td>
      <td>100.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>8.6880</td>
      <td>1.3080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>31800.0000</td>
      <td>95404.4690</td>
      <td>4000.0000</td>
      <td>252.6340</td>
      <td>719.0000</td>
      <td>258.6670</td>
      <td>30700.0000</td>
      <td>410.1700</td>
      <td>1495.1300</td>
      <td>3993.9800</td>
      <td>3214.7500</td>
      <td>512.1600</td>
      <td>3434.6300</td>
      <td>3213.2800</td>
      <td>2245.7200</td>
      <td>459.4100</td>
      <td>1040.6300</td>
      <td>1495.1300</td>
      <td>3993.9800</td>
      <td>3993.9800</td>
      <td>5.0000</td>
      <td>34200.0000</td>
      <td>100.0000</td>
      <td>98.5000</td>
      <td>19666.6670</td>
      <td>3993.9800</td>
      <td>4114.2000</td>
      <td>5.0000</td>
      <td>100.0000</td>
      <td>650.0000</td>
      <td>...</td>
      <td>158892.9400</td>
      <td>36330.0000</td>
      <td>105266.7600</td>
      <td>28422.2600</td>
      <td>110623.6600</td>
      <td>42960.0000</td>
      <td>132170.0000</td>
      <td>3457.1320</td>
      <td>36.6380</td>
      <td>5.0000</td>
      <td>45455.5560</td>
      <td>318.3460</td>
      <td>329073.2760</td>
      <td>580.5370</td>
      <td>34723.2900</td>
      <td>134430.8500</td>
      <td>121936.8500</td>
      <td>1712.6670</td>
      <td>28196.0000</td>
      <td>100.0000</td>
      <td>132170.0000</td>
      <td>100.0000</td>
      <td>1.0000</td>
      <td>498.3400</td>
      <td>2770.6900</td>
      <td>95.1900</td>
      <td>3149.4200</td>
      <td>1574.6700</td>
      <td>10828.5110</td>
      <td>12901.4670</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 76 columns</p>
</div>

    ============================================================
      Efficiency (4 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>asset_turnover</th>
      <th>inventory_turnover</th>
      <th>receivables_days</th>
      <th>working_capital_turns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6339.0000</td>
      <td>5392.0000</td>
      <td>6531.0000</td>
      <td>6338.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.7580</td>
      <td>44.9320</td>
      <td>67.1420</td>
      <td>5.4820</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.5940</td>
      <td>499.1180</td>
      <td>135.0850</td>
      <td>238.0840</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.0610</td>
      <td>-2.7260</td>
      <td>-7642.1880</td>
      <td>-2304.8670</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.3960</td>
      <td>2.6340</td>
      <td>29.2570</td>
      <td>0.4600</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.6400</td>
      <td>4.8080</td>
      <td>53.5620</td>
      <td>2.6450</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.9650</td>
      <td>11.5350</td>
      <td>82.2740</td>
      <td>6.2470</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.5730</td>
      <td>31396.5000</td>
      <td>3701.5760</td>
      <td>16638.2590</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Efficiency Ratios (45 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>opex_fq</th>
      <th>opex_vs_revenue_trend</th>
      <th>opex_ltm</th>
      <th>opex_fy</th>
      <th>opex_qoq_growth</th>
      <th>opex_yoy_growth</th>
      <th>sga_qoq_growth</th>
      <th>sga_yoy_growth</th>
      <th>operating_leverage_score</th>
      <th>cogs_to_revenue</th>
      <th>opex_to_revenue</th>
      <th>sga_to_revenue</th>
      <th>rnd_to_revenue</th>
      <th>interest_to_revenue</th>
      <th>cost_efficiency_score</th>
      <th>marketing_to_revenue</th>
      <th>marketing_trend_yoy</th>
      <th>marketing_vs_5y_avg</th>
      <th>operating_leverage_proxy</th>
      <th>sga_efficiency_trend</th>
      <th>sga_trend_yoy</th>
      <th>sga_vs_5y_avg</th>
      <th>high_rnd_intensity_flag</th>
      <th>rnd_1fy</th>
      <th>rnd_2fqfq</th>
      <th>rnd_2fy</th>
      <th>rnd_3fqfq</th>
      <th>rnd_3fy</th>
      <th>rnd_4fqfq</th>
      <th>rnd_4fy</th>
      <th>rnd_cagr_3y</th>
      <th>rnd_cut_flag</th>
      <th>rnd_1fqfq</th>
      <th>rnd_fq</th>
      <th>rnd_fy</th>
      <th>rnd_ltm</th>
      <th>rnd_per_employee</th>
      <th>rnd_increasing_flag</th>
      <th>rnd_intensity_fy</th>
      <th>rnd_intensity_ltm</th>
      <th>rnd_intensity_trend</th>
      <th>rnd_qoq_growth</th>
      <th>rnd_roi_proxy</th>
      <th>rnd_to_gross_profit</th>
      <th>rnd_yoy_growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6488.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6618.0000</td>
      <td>6642.0000</td>
      <td>6517.0000</td>
      <td>6517.0000</td>
      <td>6482.0000</td>
      <td>6532.0000</td>
      <td>6532.0000</td>
      <td>6531.0000</td>
      <td>6532.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6531.0000</td>
      <td>832.0000</td>
      <td>1254.0000</td>
      <td>5166.0000</td>
      <td>6488.0000</td>
      <td>6488.0000</td>
      <td>6086.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>2353.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5259.0000</td>
      <td>6676.0000</td>
      <td>6531.0000</td>
      <td>6532.0000</td>
      <td>6488.0000</td>
      <td>2343.0000</td>
      <td>2403.0000</td>
      <td>6550.0000</td>
      <td>2485.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1756.9670</td>
      <td>-125.9200</td>
      <td>6774.7270</td>
      <td>6733.7420</td>
      <td>0.1200</td>
      <td>0.2710</td>
      <td>-0.6510</td>
      <td>0.2200</td>
      <td>0.3110</td>
      <td>84.3050</td>
      <td>199.3370</td>
      <td>55.8890</td>
      <td>44.7500</td>
      <td>-6.5940</td>
      <td>43.3590</td>
      <td>0.6070</td>
      <td>34.6380</td>
      <td>0.6430</td>
      <td>14.2010</td>
      <td>44.0960</td>
      <td>-44.0960</td>
      <td>1.0380</td>
      <td>0.1100</td>
      <td>174.1960</td>
      <td>46.7380</td>
      <td>166.8730</td>
      <td>45.7100</td>
      <td>152.3000</td>
      <td>48.1410</td>
      <td>135.5710</td>
      <td>10.3640</td>
      <td>0.0420</td>
      <td>49.1380</td>
      <td>54.6880</td>
      <td>195.3660</td>
      <td>196.0080</td>
      <td>0.0380</td>
      <td>0.0800</td>
      <td>44.7510</td>
      <td>44.7500</td>
      <td>-62.3040</td>
      <td>11.0750</td>
      <td>14.0040</td>
      <td>30.1400</td>
      <td>25.7800</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6382.1780</td>
      <td>5776.1970</td>
      <td>23837.2700</td>
      <td>23646.6320</td>
      <td>3.0450</td>
      <td>5.5550</td>
      <td>1.8370</td>
      <td>4.2620</td>
      <td>8.1190</td>
      <td>701.0220</td>
      <td>3191.7500</td>
      <td>1054.6610</td>
      <td>2072.3490</td>
      <td>171.0680</td>
      <td>17.5020</td>
      <td>4.3950</td>
      <td>495.8950</td>
      <td>0.8430</td>
      <td>836.9170</td>
      <td>1615.2980</td>
      <td>1615.2980</td>
      <td>9.4120</td>
      <td>0.3130</td>
      <td>1638.5180</td>
      <td>450.0650</td>
      <td>1557.1290</td>
      <td>406.8050</td>
      <td>1369.6420</td>
      <td>530.5070</td>
      <td>1128.4860</td>
      <td>27.1230</td>
      <td>0.2010</td>
      <td>479.0150</td>
      <td>662.8320</td>
      <td>1969.1460</td>
      <td>1982.5110</td>
      <td>0.2060</td>
      <td>0.2720</td>
      <td>2072.4610</td>
      <td>2072.3490</td>
      <td>4248.5880</td>
      <td>297.3080</td>
      <td>382.1200</td>
      <td>1063.4120</td>
      <td>470.0110</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4554.2300</td>
      <td>-424902.9390</td>
      <td>-169.8400</td>
      <td>-169.8400</td>
      <td>-21.9490</td>
      <td>-5.7780</td>
      <td>-22.9180</td>
      <td>-6.6790</td>
      <td>-82.7240</td>
      <td>-41.9960</td>
      <td>-40.8190</td>
      <td>-13.5950</td>
      <td>-17.3440</td>
      <td>-11323.0770</td>
      <td>0.0000</td>
      <td>-2.4690</td>
      <td>-209.3750</td>
      <td>-3.5000</td>
      <td>-1973.5990</td>
      <td>-12667.4900</td>
      <td>-85174.2100</td>
      <td>-708.4020</td>
      <td>0.0000</td>
      <td>-10.0600</td>
      <td>-142.6800</td>
      <td>-47.7400</td>
      <td>-150.3200</td>
      <td>-62.7300</td>
      <td>-600.2400</td>
      <td>-47.9000</td>
      <td>-75.7380</td>
      <td>0.0000</td>
      <td>-146.9900</td>
      <td>-577.1500</td>
      <td>-17.2800</td>
      <td>-15.3500</td>
      <td>-0.0340</td>
      <td>0.0000</td>
      <td>-19.8040</td>
      <td>-17.3440</td>
      <td>-339740.7770</td>
      <td>-957.9710</td>
      <td>-6405.6000</td>
      <td>-23536.8420</td>
      <td>-114.2860</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>106.7650</td>
      <td>-2.3040</td>
      <td>426.0250</td>
      <td>413.3300</td>
      <td>-0.0200</td>
      <td>0.0200</td>
      <td>-0.7480</td>
      <td>0.0140</td>
      <td>-0.0190</td>
      <td>45.4660</td>
      <td>81.6420</td>
      <td>6.1550</td>
      <td>0.0000</td>
      <td>-3.2000</td>
      <td>32.3990</td>
      <td>0.0000</td>
      <td>-11.9170</td>
      <td>0.0000</td>
      <td>-0.2780</td>
      <td>-0.6680</td>
      <td>-0.8160</td>
      <td>1.0130</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.4130</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-5.8300</td>
      <td>0.2310</td>
      <td>0.0000</td>
      <td>-1.3060</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>361.4300</td>
      <td>-0.2320</td>
      <td>1408.5350</td>
      <td>1394.7450</td>
      <td>0.0190</td>
      <td>0.0990</td>
      <td>-0.7080</td>
      <td>0.1000</td>
      <td>0.0030</td>
      <td>64.0760</td>
      <td>90.1720</td>
      <td>14.0090</td>
      <td>0.0000</td>
      <td>-1.2750</td>
      <td>41.3820</td>
      <td>0.0000</td>
      <td>7.8060</td>
      <td>0.0000</td>
      <td>1.0280</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.1920</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>6.7370</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>2.0870</td>
      <td>1.4420</td>
      <td>0.0000</td>
      <td>9.4240</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1219.4330</td>
      <td>1.5520</td>
      <td>4758.7580</td>
      <td>4718.8380</td>
      <td>0.1130</td>
      <td>0.2070</td>
      <td>-0.6540</td>
      <td>0.2130</td>
      <td>0.0290</td>
      <td>78.5830</td>
      <td>95.7790</td>
      <td>26.9050</td>
      <td>2.6190</td>
      <td>-0.3750</td>
      <td>52.4490</td>
      <td>0.0000</td>
      <td>26.2910</td>
      <td>1.2010</td>
      <td>2.3100</td>
      <td>0.8160</td>
      <td>0.6680</td>
      <td>1.4170</td>
      <td>0.0000</td>
      <td>37.5800</td>
      <td>8.1220</td>
      <td>34.3250</td>
      <td>9.0920</td>
      <td>30.7750</td>
      <td>8.2820</td>
      <td>24.8000</td>
      <td>15.9650</td>
      <td>0.0000</td>
      <td>8.6400</td>
      <td>9.0480</td>
      <td>38.7780</td>
      <td>36.2800</td>
      <td>0.0100</td>
      <td>0.0000</td>
      <td>2.6640</td>
      <td>2.6190</td>
      <td>0.0000</td>
      <td>15.0350</td>
      <td>4.8260</td>
      <td>8.5560</td>
      <td>21.9670</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190909.0000</td>
      <td>27677.2730</td>
      <td>683338.0000</td>
      <td>683338.0000</td>
      <td>226.1500</td>
      <td>432.1740</td>
      <td>138.0780</td>
      <td>336.2920</td>
      <td>380.2760</td>
      <td>43933.3330</td>
      <td>243200.0000</td>
      <td>78466.6670</td>
      <td>164633.3330</td>
      <td>2400.0000</td>
      <td>100.0000</td>
      <td>155.9000</td>
      <td>13400.0000</td>
      <td>11.2020</td>
      <td>59285.6810</td>
      <td>85174.2100</td>
      <td>12667.4900</td>
      <td>36.4490</td>
      <td>1.0000</td>
      <td>88544.0000</td>
      <td>23081.0000</td>
      <td>85622.0000</td>
      <td>19058.0000</td>
      <td>73213.0000</td>
      <td>34210.0000</td>
      <td>56052.0000</td>
      <td>570.3240</td>
      <td>1.0000</td>
      <td>24732.0000</td>
      <td>41650.0000</td>
      <td>108521.0000</td>
      <td>108521.0000</td>
      <td>6.5080</td>
      <td>1.0000</td>
      <td>164633.3330</td>
      <td>164633.3330</td>
      <td>12296.8120</td>
      <td>11120.0000</td>
      <td>16008.0000</td>
      <td>73112.5000</td>
      <td>22500.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Employee Productivity (7 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue_per_employee</th>
      <th>profit_per_employee</th>
      <th>ebitda_per_employee</th>
      <th>assets_per_employee</th>
      <th>fte_growth_1y_pct</th>
      <th>fte_growth_3y_pct</th>
      <th>workforce_stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5259.0000</td>
      <td>5259.0000</td>
      <td>5259.0000</td>
      <td>5259.0000</td>
      <td>5741.0000</td>
      <td>5519.0000</td>
      <td>1862.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.7900</td>
      <td>0.0400</td>
      <td>0.1880</td>
      <td>3.6600</td>
      <td>6.8250</td>
      <td>25.7930</td>
      <td>1.6530</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.6700</td>
      <td>2.9140</td>
      <td>3.1940</td>
      <td>52.3230</td>
      <td>796.0620</td>
      <td>627.6680</td>
      <td>18.6750</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-228.0550</td>
      <td>-146.6350</td>
      <td>-79.7300</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.2150</td>
      <td>0.0040</td>
      <td>0.0190</td>
      <td>0.3080</td>
      <td>-6.2500</td>
      <td>-13.3100</td>
      <td>0.7860</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.3640</td>
      <td>0.0160</td>
      <td>0.0470</td>
      <td>0.5840</td>
      <td>0.1470</td>
      <td>2.9410</td>
      <td>1.0140</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.6530</td>
      <td>0.0420</td>
      <td>0.1080</td>
      <td>1.2470</td>
      <td>6.8680</td>
      <td>22.3100</td>
      <td>1.2050</td>
    </tr>
    <tr>
      <th>max</th>
      <td>260.6150</td>
      <td>92.8900</td>
      <td>165.2420</td>
      <td>2931.6830</td>
      <td>59900.0000</td>
      <td>34185.7140</td>
      <td>778.5000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Employment Dynamics (10 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fte_acceleration</th>
      <th>fte_growth_2y_pct</th>
      <th>headcount_vs_revenue</th>
      <th>hiring_intensity</th>
      <th>layoff_risk_flag</th>
      <th>productivity_trend</th>
      <th>rapid_hiring_flag</th>
      <th>sustainable_growth_flag</th>
      <th>workforce_efficiency_gain</th>
      <th>workforce_volatility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5386.0000</td>
      <td>5612.0000</td>
      <td>5604.0000</td>
      <td>4464.0000</td>
      <td>6676.0000</td>
      <td>4975.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5515.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.3020</td>
      <td>13.0870</td>
      <td>-57.2860</td>
      <td>-1.0370</td>
      <td>0.1110</td>
      <td>33.1470</td>
      <td>0.0820</td>
      <td>0.2910</td>
      <td>51.7080</td>
      <td>40.6550</td>
    </tr>
    <tr>
      <th>std</th>
      <td>811.5380</td>
      <td>579.6430</td>
      <td>863.9490</td>
      <td>14.1990</td>
      <td>0.3140</td>
      <td>612.2540</td>
      <td>0.2740</td>
      <td>0.4540</td>
      <td>791.1390</td>
      <td>826.7870</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-480.3460</td>
      <td>-100.0000</td>
      <td>-38031.0220</td>
      <td>-630.8130</td>
      <td>0.0000</td>
      <td>-162.0540</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-4.4770</td>
      <td>-10.7150</td>
      <td>-23.8070</td>
      <td>-0.5120</td>
      <td>0.0000</td>
      <td>0.1060</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>3.0440</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>1.1180</td>
      <td>-9.5680</td>
      <td>0.0700</td>
      <td>0.0000</td>
      <td>7.5180</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>6.2380</td>
      <td>8.0450</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.7610</td>
      <td>14.4130</td>
      <td>-1.2000</td>
      <td>0.5030</td>
      <td>0.0000</td>
      <td>17.1390</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>19.3780</td>
      <td>22.6210</td>
    </tr>
    <tr>
      <th>max</th>
      <td>59300.0970</td>
      <td>39900.0000</td>
      <td>1054.7140</td>
      <td>296.5440</td>
      <td>1.0000</td>
      <td>30154.3350</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>38031.0220</td>
      <td>59933.3330</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Cash flow (28 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>self_funding_flag</th>
      <th>capex_qoq_growth</th>
      <th>capex_volatility</th>
      <th>capex_yoy_growth</th>
      <th>cash_flow_quality_score</th>
      <th>cff_share_of_cf</th>
      <th>cfi_share_of_cf</th>
      <th>cfo_share_of_cf</th>
      <th>fcf_4q_improvement</th>
      <th>fcf_always_positive</th>
      <th>fcf_positive_years</th>
      <th>investment_efficiency</th>
      <th>organic_vs_inorganic</th>
      <th>serial_acquirer_flag</th>
      <th>sustainable_ma_flag</th>
      <th>total_investment_to_cfo</th>
      <th>underinvestment_flag</th>
      <th>acquisition_pause_flag</th>
      <th>acquisition_to_fcf</th>
      <th>acquisitions_ltm_total</th>
      <th>acquisitions_vs_5y_avg</th>
      <th>acquisitions_yoy_growth</th>
      <th>capex_3y_trend</th>
      <th>capex_acceleration</th>
      <th>capex_cut_flag</th>
      <th>capex_vs_5y_avg</th>
      <th>overinvestment_flag</th>
      <th>ma_intensity_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>5783.0000</td>
      <td>6116.0000</td>
      <td>6543.0000</td>
      <td>6676.0000</td>
      <td>6090.0000</td>
      <td>6090.0000</td>
      <td>6090.0000</td>
      <td>5968.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6565.0000</td>
      <td>2446.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6090.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6090.0000</td>
      <td>6676.0000</td>
      <td>4312.0000</td>
      <td>2580.0000</td>
      <td>6297.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5911.0000</td>
      <td>6676.0000</td>
      <td>6339.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.6040</td>
      <td>107.8530</td>
      <td>1.9560</td>
      <td>141.3270</td>
      <td>58.5980</td>
      <td>0.2940</td>
      <td>0.2760</td>
      <td>0.4300</td>
      <td>0.5020</td>
      <td>0.4340</td>
      <td>3.5640</td>
      <td>29.6660</td>
      <td>230.2010</td>
      <td>0.3030</td>
      <td>0.8060</td>
      <td>1.3570</td>
      <td>0.2710</td>
      <td>0.1900</td>
      <td>0.6050</td>
      <td>108.2280</td>
      <td>0.9610</td>
      <td>1722.9660</td>
      <td>338.9510</td>
      <td>0.2870</td>
      <td>0.2080</td>
      <td>1.4790</td>
      <td>0.2400</td>
      <td>1.2590</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.4890</td>
      <td>2941.9780</td>
      <td>1.5160</td>
      <td>3454.7810</td>
      <td>37.3820</td>
      <td>0.1760</td>
      <td>0.1740</td>
      <td>0.1680</td>
      <td>21.7470</td>
      <td>0.4960</td>
      <td>1.6780</td>
      <td>920.3760</td>
      <td>2568.2600</td>
      <td>0.4600</td>
      <td>0.3950</td>
      <td>10.4450</td>
      <td>0.4440</td>
      <td>0.3920</td>
      <td>7.9880</td>
      <td>641.1530</td>
      <td>14.8050</td>
      <td>25599.3030</td>
      <td>6520.9020</td>
      <td>0.4530</td>
      <td>0.4060</td>
      <td>14.8830</td>
      <td>0.4270</td>
      <td>4.2370</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0400</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0010</td>
      <td>-631.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-3077.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0000</td>
      <td>-13.7310</td>
      <td>0.8920</td>
      <td>-19.7240</td>
      <td>25.0000</td>
      <td>0.1560</td>
      <td>0.1340</td>
      <td>0.3390</td>
      <td>-0.5270</td>
      <td>0.0000</td>
      <td>3.0000</td>
      <td>0.1280</td>
      <td>0.5440</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.1860</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-28.8950</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5710</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0000</td>
      <td>1.3160</td>
      <td>1.5440</td>
      <td>7.1900</td>
      <td>75.0000</td>
      <td>0.2840</td>
      <td>0.2550</td>
      <td>0.4730</td>
      <td>0.0720</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>1.2860</td>
      <td>3.3120</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.4310</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-71.8420</td>
      <td>15.9160</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0530</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0000</td>
      <td>41.7390</td>
      <td>2.6040</td>
      <td>40.0770</td>
      <td>100.0000</td>
      <td>0.4150</td>
      <td>0.4110</td>
      <td>0.5270</td>
      <td>0.7410</td>
      <td>1.0000</td>
      <td>5.0000</td>
      <td>4.3310</td>
      <td>19.6290</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.8730</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0740</td>
      <td>9.6420</td>
      <td>0.0600</td>
      <td>89.2880</td>
      <td>81.2550</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.5640</td>
      <td>0.0000</td>
      <td>0.3110</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0000</td>
      <td>200839.4740</td>
      <td>10.0000</td>
      <td>178749.3510</td>
      <td>100.0000</td>
      <td>0.9810</td>
      <td>0.9870</td>
      <td>0.9970</td>
      <td>668.1670</td>
      <td>1.0000</td>
      <td>5.0000</td>
      <td>67853.3000</td>
      <td>89429.5000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>407.7460</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>488.0000</td>
      <td>16681.2600</td>
      <td>830.3640</td>
      <td>814471.4290</td>
      <td>352575.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1139.0000</td>
      <td>1.0000</td>
      <td>73.7340</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Financial Distress (9 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>distress_risk_score</th>
      <th>liquidity_stress_score</th>
      <th>working_capital_trend</th>
      <th>cash_runway_months</th>
      <th>accumulated_deficit_flag</th>
      <th>adequate_cash_buffer</th>
      <th>combined_distress_score</th>
      <th>wc_deteriorating_flag</th>
      <th>retained_earnings_growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6582.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6342.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66.0350</td>
      <td>9.1310</td>
      <td>-0.0250</td>
      <td>141.6370</td>
      <td>0.1980</td>
      <td>0.9730</td>
      <td>69.9520</td>
      <td>0.0750</td>
      <td>-0.0770</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43.2330</td>
      <td>13.1440</td>
      <td>6.3260</td>
      <td>1126.7810</td>
      <td>0.3990</td>
      <td>0.1610</td>
      <td>39.6530</td>
      <td>0.2630</td>
      <td>2.5270</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-352.6880</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-139.5000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10.8330</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>120.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>33.5000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>120.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>100.0000</td>
      <td>20.0000</td>
      <td>0.0000</td>
      <td>120.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.0000</td>
      <td>40.0000</td>
      <td>236.6780</td>
      <td>73249.3850</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>1.0000</td>
      <td>80.6100</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      GAAP vs Adjusted (50 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>earnings_quality_score</th>
      <th>earnings_quality_warning</th>
      <th>ebit_adjustment_pct_1fqfq</th>
      <th>ebit_adjustment_pct_1fy</th>
      <th>ebit_adjustment_pct_2fqfq</th>
      <th>ebit_adjustment_pct_3fqfq</th>
      <th>ebit_adjustment_pct_3fy</th>
      <th>ebit_adjustment_pct_4fqfq</th>
      <th>ebit_adjustment_pct_4fy</th>
      <th>ebit_adjustment_pct_fq</th>
      <th>ebit_adjustment_pct_fy</th>
      <th>ebit_adjustment_pct_ltm</th>
      <th>ebitda_adjustment_pct_1fqfq</th>
      <th>ebitda_adjustment_pct_1fy</th>
      <th>ebitda_adjustment_pct_2fqfq</th>
      <th>ebitda_adjustment_pct_2fy</th>
      <th>ebitda_adjustment_pct_3fqfq</th>
      <th>ebitda_adjustment_pct_3fy</th>
      <th>ebitda_adjustment_pct_4fy</th>
      <th>ebitda_adjustment_pct_fq</th>
      <th>ebitda_adjustment_pct_fy</th>
      <th>ebitda_adjustment_pct_ltm</th>
      <th>eps_adjustment_pct</th>
      <th>eps_adjustment_spread_1fqfq</th>
      <th>eps_adjustment_spread_1fy</th>
      <th>eps_adjustment_spread_2fqfq</th>
      <th>eps_adjustment_spread_2fy</th>
      <th>eps_adjustment_spread_3fqfq</th>
      <th>eps_adjustment_spread_3fy</th>
      <th>eps_adjustment_spread_4fqfq</th>
      <th>eps_adjustment_spread_4fy</th>
      <th>eps_adjustment_spread_fy</th>
      <th>eps_adjustment_spread_ltm</th>
      <th>forward_eps_gaap_adj_spread</th>
      <th>net_income_adjustment_pct</th>
      <th>net_income_adjustment_ratio_1fqfq</th>
      <th>net_income_adjustment_ratio_1fy</th>
      <th>net_income_adjustment_ratio_2fqfq</th>
      <th>net_income_adjustment_ratio_2fy</th>
      <th>net_income_adjustment_ratio_3fqfq</th>
      <th>net_income_adjustment_ratio_3fy</th>
      <th>net_income_adjustment_ratio_4fqfq</th>
      <th>net_income_adjustment_ratio_4fy</th>
      <th>net_income_adjustment_ratio_fq</th>
      <th>net_income_adjustment_ratio_fy</th>
      <th>net_income_adjustment_ratio_ltm</th>
      <th>ebit_adjustment_pct_2fy</th>
      <th>ebitda_adjustment_pct_4fqfq</th>
      <th>eps_adjustment_spread_fq</th>
      <th>net_income_adjustment_ratio_5yavgfq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6615.0000</td>
      <td>6638.0000</td>
      <td>6594.0000</td>
      <td>6615.0000</td>
      <td>6589.0000</td>
      <td>6471.0000</td>
      <td>6483.0000</td>
      <td>6593.0000</td>
      <td>6669.0000</td>
      <td>6669.0000</td>
      <td>6586.0000</td>
      <td>6609.0000</td>
      <td>6531.0000</td>
      <td>6615.0000</td>
      <td>6578.0000</td>
      <td>6561.0000</td>
      <td>6455.0000</td>
      <td>6534.0000</td>
      <td>6646.0000</td>
      <td>6621.0000</td>
      <td>3535.0000</td>
      <td>3346.0000</td>
      <td>3154.0000</td>
      <td>3263.0000</td>
      <td>3051.0000</td>
      <td>3226.0000</td>
      <td>2925.0000</td>
      <td>3245.0000</td>
      <td>2767.0000</td>
      <td>5389.0000</td>
      <td>3567.0000</td>
      <td>5491.0000</td>
      <td>6668.0000</td>
      <td>6615.0000</td>
      <td>6649.0000</td>
      <td>6590.0000</td>
      <td>6642.0000</td>
      <td>6613.0000</td>
      <td>6590.0000</td>
      <td>6476.0000</td>
      <td>6479.0000</td>
      <td>6592.0000</td>
      <td>6667.0000</td>
      <td>6668.0000</td>
      <td>6648.0000</td>
      <td>6440.0000</td>
      <td>3203.0000</td>
      <td>6116.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>80.7170</td>
      <td>0.2910</td>
      <td>5870.3330</td>
      <td>-8.6020</td>
      <td>8.6600</td>
      <td>-0.1870</td>
      <td>-15.1930</td>
      <td>16.0110</td>
      <td>2.7560</td>
      <td>4.4470</td>
      <td>-82.5230</td>
      <td>-13.7660</td>
      <td>2.9410</td>
      <td>-9.6630</td>
      <td>38.9030</td>
      <td>-9.4830</td>
      <td>12.3540</td>
      <td>-11.7170</td>
      <td>289.1860</td>
      <td>13.9970</td>
      <td>55.1180</td>
      <td>3.0110</td>
      <td>88.2020</td>
      <td>0.3250</td>
      <td>0.5240</td>
      <td>0.1400</td>
      <td>0.4550</td>
      <td>0.1300</td>
      <td>0.5180</td>
      <td>0.1790</td>
      <td>0.5200</td>
      <td>0.5070</td>
      <td>0.5670</td>
      <td>0.2450</td>
      <td>34.7090</td>
      <td>0.5000</td>
      <td>0.7430</td>
      <td>0.5780</td>
      <td>0.7600</td>
      <td>0.4330</td>
      <td>0.5200</td>
      <td>0.5280</td>
      <td>0.4390</td>
      <td>0.4370</td>
      <td>0.8830</td>
      <td>0.7290</td>
      <td>-10.3700</td>
      <td>24.8260</td>
      <td>0.0610</td>
      <td>0.5180</td>
    </tr>
    <tr>
      <th>std</th>
      <td>32.8880</td>
      <td>0.4540</td>
      <td>478387.4110</td>
      <td>401.3670</td>
      <td>940.6030</td>
      <td>552.5200</td>
      <td>683.6640</td>
      <td>868.0660</td>
      <td>1880.1090</td>
      <td>1115.3100</td>
      <td>5293.7010</td>
      <td>567.3920</td>
      <td>976.7200</td>
      <td>418.7490</td>
      <td>3280.5470</td>
      <td>460.5610</td>
      <td>868.5510</td>
      <td>655.1770</td>
      <td>23421.0720</td>
      <td>1042.5100</td>
      <td>1627.0290</td>
      <td>990.0830</td>
      <td>729.5660</td>
      <td>8.6400</td>
      <td>3.5650</td>
      <td>0.9740</td>
      <td>3.0240</td>
      <td>1.1050</td>
      <td>10.9360</td>
      <td>2.4420</td>
      <td>22.7850</td>
      <td>3.0620</td>
      <td>3.0830</td>
      <td>1.4540</td>
      <td>926.9460</td>
      <td>5.5380</td>
      <td>17.6410</td>
      <td>5.6640</td>
      <td>17.6650</td>
      <td>4.7460</td>
      <td>10.3920</td>
      <td>4.5780</td>
      <td>20.8390</td>
      <td>10.2270</td>
      <td>20.0030</td>
      <td>9.2720</td>
      <td>446.2640</td>
      <td>1464.3410</td>
      <td>13.4200</td>
      <td>6.8460</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-4450.0000</td>
      <td>-2634.4830</td>
      <td>-33900.0000</td>
      <td>-14000.0000</td>
      <td>-39920.0000</td>
      <td>-5100.0000</td>
      <td>-83925.0000</td>
      <td>-49325.0000</td>
      <td>-326500.0000</td>
      <td>-25042.8570</td>
      <td>-1136.6100</td>
      <td>-1797.1100</td>
      <td>-424.7400</td>
      <td>-3700.7930</td>
      <td>-1401.4290</td>
      <td>-745.7980</td>
      <td>-6053.2710</td>
      <td>-4025.0000</td>
      <td>-34804.9500</td>
      <td>-25163.3660</td>
      <td>-1387.5000</td>
      <td>-20.8600</td>
      <td>-100.8500</td>
      <td>-14.4800</td>
      <td>-61.3800</td>
      <td>-34.9000</td>
      <td>-278.6600</td>
      <td>-104.0400</td>
      <td>-126.9400</td>
      <td>-34.5500</td>
      <td>-42.3000</td>
      <td>-35.2300</td>
      <td>-1474.6270</td>
      <td>-230.0000</td>
      <td>-146.5000</td>
      <td>-157.8260</td>
      <td>-141.1690</td>
      <td>-218.6740</td>
      <td>-280.0000</td>
      <td>-174.7270</td>
      <td>-1236.0000</td>
      <td>-617.0000</td>
      <td>-1417.6000</td>
      <td>-82.2660</td>
      <td>-10573.3330</td>
      <td>-7852.1740</td>
      <td>-720.7200</td>
      <td>-361.3430</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>75.7120</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-2.1390</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-3.3330</td>
      <td>0.0000</td>
      <td>-0.0100</td>
      <td>0.0000</td>
      <td>-0.0100</td>
      <td>0.0000</td>
      <td>-0.0200</td>
      <td>0.0000</td>
      <td>-0.0200</td>
      <td>0.0000</td>
      <td>-0.0200</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5220</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.0000</td>
      <td>0.0000</td>
      <td>-0.0110</td>
      <td>-1.4970</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-2.3310</td>
      <td>-0.1270</td>
      <td>-2.7200</td>
      <td>-0.1210</td>
      <td>0.0000</td>
      <td>-2.3000</td>
      <td>-0.8070</td>
      <td>-3.4320</td>
      <td>-1.0380</td>
      <td>-11.9600</td>
      <td>-1.4360</td>
      <td>-14.7350</td>
      <td>-16.3660</td>
      <td>-1.2080</td>
      <td>6.6290</td>
      <td>-4.8340</td>
      <td>1.3330</td>
      <td>0.0000</td>
      <td>0.0200</td>
      <td>0.0000</td>
      <td>0.0100</td>
      <td>0.0000</td>
      <td>0.0100</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0100</td>
      <td>0.0100</td>
      <td>-1.3180</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.2400</td>
      <td>-2.0090</td>
      <td>-1.6530</td>
      <td>0.0100</td>
      <td>0.6070</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>100.0000</td>
      <td>1.0000</td>
      <td>2.8540</td>
      <td>6.0390</td>
      <td>5.1230</td>
      <td>6.1850</td>
      <td>6.2030</td>
      <td>10.7720</td>
      <td>6.7570</td>
      <td>7.2610</td>
      <td>9.8150</td>
      <td>4.2460</td>
      <td>9.2940</td>
      <td>15.6070</td>
      <td>10.3260</td>
      <td>14.5780</td>
      <td>11.6410</td>
      <td>14.0020</td>
      <td>14.2880</td>
      <td>16.0190</td>
      <td>24.4570</td>
      <td>13.8520</td>
      <td>45.4920</td>
      <td>0.1380</td>
      <td>0.5500</td>
      <td>0.1300</td>
      <td>0.4900</td>
      <td>0.1200</td>
      <td>0.5000</td>
      <td>0.1500</td>
      <td>0.3500</td>
      <td>0.3200</td>
      <td>0.5800</td>
      <td>0.1400</td>
      <td>29.0050</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0560</td>
      <td>1.0000</td>
      <td>6.6670</td>
      <td>18.8970</td>
      <td>0.1900</td>
      <td>1.0370</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.0000</td>
      <td>1.0000</td>
      <td>38908500.0000</td>
      <td>19684.7060</td>
      <td>41488.3500</td>
      <td>31375.0000</td>
      <td>32816.9810</td>
      <td>48075.0000</td>
      <td>116968.7500</td>
      <td>52836.6670</td>
      <td>16003.9600</td>
      <td>16003.9600</td>
      <td>60350.0000</td>
      <td>24150.0000</td>
      <td>259600.0000</td>
      <td>21800.0000</td>
      <td>40566.6670</td>
      <td>49145.2550</td>
      <td>1880100.0000</td>
      <td>70475.0000</td>
      <td>123300.0000</td>
      <td>66612.3710</td>
      <td>25766.6670</td>
      <td>492.8400</td>
      <td>75.6700</td>
      <td>18.7600</td>
      <td>59.8400</td>
      <td>22.3800</td>
      <td>433.0100</td>
      <td>50.6500</td>
      <td>1126.2900</td>
      <td>106.3700</td>
      <td>54.3200</td>
      <td>62.3600</td>
      <td>57953.6590</td>
      <td>154.3880</td>
      <td>1137.5000</td>
      <td>329.5000</td>
      <td>1251.8460</td>
      <td>81.6210</td>
      <td>716.5000</td>
      <td>195.7500</td>
      <td>1111.4290</td>
      <td>271.6220</td>
      <td>580.5370</td>
      <td>580.5370</td>
      <td>18654.0820</td>
      <td>96250.0000</td>
      <td>222.6400</td>
      <td>139.7250</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Growth Metrics (14 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue_growth_yoy</th>
      <th>ebitda_growth_yoy</th>
      <th>operating_income_growth</th>
      <th>fcf_growth</th>
      <th>revenue_cagr_5y</th>
      <th>forward_revenue_growth</th>
      <th>revenue_vs_5y_avg</th>
      <th>growth_ebitda_growth_yoy</th>
      <th>revenue_5yavgfq</th>
      <th>revenue_5yavgltm</th>
      <th>revenue_fq_vs_avg</th>
      <th>revenue_momentum</th>
      <th>revenue_vs_5y_avg_fq</th>
      <th>revenue_vs_5y_avg_ltm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6501.0000</td>
      <td>6609.0000</td>
      <td>6669.0000</td>
      <td>6507.0000</td>
      <td>6106.0000</td>
      <td>6305.0000</td>
      <td>6078.0000</td>
      <td>6609.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5985.0000</td>
      <td>6501.0000</td>
      <td>5985.0000</td>
      <td>6078.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48.0060</td>
      <td>67.9210</td>
      <td>5.1570</td>
      <td>-3.2550</td>
      <td>0.1540</td>
      <td>0.7830</td>
      <td>1.2550</td>
      <td>67.9210</td>
      <td>1718.4730</td>
      <td>6684.4040</td>
      <td>25.3540</td>
      <td>51.1030</td>
      <td>1.2540</td>
      <td>1.2550</td>
    </tr>
    <tr>
      <th>std</th>
      <td>821.8260</td>
      <td>1390.8700</td>
      <td>291.4250</td>
      <td>642.0940</td>
      <td>0.2560</td>
      <td>33.7910</td>
      <td>0.5980</td>
      <td>1390.8700</td>
      <td>6134.4060</td>
      <td>23780.6440</td>
      <td>71.3330</td>
      <td>858.2250</td>
      <td>0.7130</td>
      <td>0.5980</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-162.0540</td>
      <td>-15318.2770</td>
      <td>-9552.6320</td>
      <td>-42086.3640</td>
      <td>-0.5700</td>
      <td>-1.0000</td>
      <td>-15.2570</td>
      <td>-15318.2770</td>
      <td>-1968.8600</td>
      <td>-302.2500</td>
      <td>-950.0000</td>
      <td>-613.9600</td>
      <td>-8.5000</td>
      <td>-15.2570</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.8560</td>
      <td>-4.6120</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0490</td>
      <td>0.0230</td>
      <td>1.0320</td>
      <td>-4.6120</td>
      <td>71.5280</td>
      <td>291.8570</td>
      <td>0.7950</td>
      <td>2.5240</td>
      <td>1.0080</td>
      <td>1.0320</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.2680</td>
      <td>11.5160</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1000</td>
      <td>0.0810</td>
      <td>1.1710</td>
      <td>11.5160</td>
      <td>321.2150</td>
      <td>1251.1650</td>
      <td>16.9880</td>
      <td>11.8200</td>
      <td>1.1700</td>
      <td>1.1710</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>21.6540</td>
      <td>32.1090</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1850</td>
      <td>0.1940</td>
      <td>1.3660</td>
      <td>32.1090</td>
      <td>1153.5300</td>
      <td>4504.7480</td>
      <td>37.7640</td>
      <td>24.0310</td>
      <td>1.3780</td>
      <td>1.3660</td>
    </tr>
    <tr>
      <th>max</th>
      <td>38057.1430</td>
      <td>67885.1850</td>
      <td>19500.0000</td>
      <td>12070.6670</td>
      <td>5.6380</td>
      <td>2638.2540</td>
      <td>15.0530</td>
      <td>67885.1850</td>
      <td>160875.9500</td>
      <td>629497.5200</td>
      <td>2093.8610</td>
      <td>38057.1430</td>
      <td>21.9390</td>
      <td>15.0530</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Interest Income (12 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>interest_income_yoy_growth</th>
      <th>interest_income_to_revenue_trend</th>
      <th>interest_income_fq</th>
      <th>interest_income_fy</th>
      <th>interest_income_qoq_growth</th>
      <th>interest_coverage_ratio</th>
      <th>interest_expense_ltm</th>
      <th>interest_expense_to_revenue</th>
      <th>interest_income_ltm</th>
      <th>interest_income_to_revenue</th>
      <th>net_interest_income</th>
      <th>net_interest_margin_proxy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5819.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>4995.0000</td>
      <td>6142.0000</td>
      <td>6676.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6339.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.5640</td>
      <td>0.0850</td>
      <td>18.6670</td>
      <td>54.4270</td>
      <td>1.6670</td>
      <td>-82.2460</td>
      <td>-132.2260</td>
      <td>-6.5940</td>
      <td>55.3410</td>
      <td>8.4910</td>
      <td>187.5660</td>
      <td>2.0620</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.5880</td>
      <td>2.4770</td>
      <td>113.6080</td>
      <td>205.8570</td>
      <td>36.8760</td>
      <td>1181.5890</td>
      <td>375.2240</td>
      <td>171.0680</td>
      <td>219.6850</td>
      <td>247.7110</td>
      <td>506.2540</td>
      <td>2.5030</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.0000</td>
      <td>-0.0010</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>-47154.0000</td>
      <td>-7662.1200</td>
      <td>-11323.0770</td>
      <td>0.0000</td>
      <td>-0.1270</td>
      <td>-11.2400</td>
      <td>-0.8100</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.2850</td>
      <td>0.0010</td>
      <td>0.0400</td>
      <td>0.7880</td>
      <td>-0.1810</td>
      <td>-16.8630</td>
      <td>-96.6500</td>
      <td>-3.2000</td>
      <td>0.7700</td>
      <td>0.1100</td>
      <td>9.6700</td>
      <td>0.8580</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.0310</td>
      <td>0.0040</td>
      <td>1.7600</td>
      <td>7.5250</td>
      <td>0.0000</td>
      <td>-5.6550</td>
      <td>-19.9300</td>
      <td>-1.2750</td>
      <td>7.5850</td>
      <td>0.4460</td>
      <td>37.3600</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.2890</td>
      <td>0.0130</td>
      <td>8.4420</td>
      <td>31.3300</td>
      <td>0.2690</td>
      <td>-2.0430</td>
      <td>-2.6600</td>
      <td>-0.3750</td>
      <td>31.5880</td>
      <td>1.3430</td>
      <td>142.4900</td>
      <td>2.4280</td>
    </tr>
    <tr>
      <th>max</th>
      <td>340.6250</td>
      <td>190.0000</td>
      <td>6715.2300</td>
      <td>4381.0000</td>
      <td>2469.0000</td>
      <td>8114.3330</td>
      <td>28.0000</td>
      <td>2400.0000</td>
      <td>6602.6500</td>
      <td>19000.0000</td>
      <td>8382.2800</td>
      <td>83.0630</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Leverage & Liquidity (47 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>debt_to_equity</th>
      <th>debt_to_assets</th>
      <th>equity_ratio</th>
      <th>interest_coverage</th>
      <th>cash_ratio</th>
      <th>working_capital_ratio</th>
      <th>debt_1fy</th>
      <th>debt_2fq</th>
      <th>debt_2fy</th>
      <th>debt_3fq</th>
      <th>debt_3fy</th>
      <th>debt_4fy</th>
      <th>days_working_capital</th>
      <th>debt_1fq</th>
      <th>debt_3y_cagr</th>
      <th>debt_4fq</th>
      <th>debt_4q_trend</th>
      <th>debt_deleveraging</th>
      <th>debt_fq</th>
      <th>debt_fy</th>
      <th>debt_ltm</th>
      <th>debt_qoq_change</th>
      <th>debt_to_equity_trend</th>
      <th>debt_yoy_change</th>
      <th>negative_wc_flag</th>
      <th>wc_efficiency_score</th>
      <th>wc_to_assets</th>
      <th>wc_to_revenue</th>
      <th>wc_1fq</th>
      <th>wc_1fy</th>
      <th>wc_2fq</th>
      <th>wc_2fy</th>
      <th>wc_3fq</th>
      <th>wc_3fy</th>
      <th>wc_4fq</th>
      <th>wc_4fy</th>
      <th>wc_4q_trend</th>
      <th>wc_5yavgfy</th>
      <th>wc_fq</th>
      <th>wc_fy</th>
      <th>wc_improving_flag</th>
      <th>wc_ltm</th>
      <th>wc_positive_quarters</th>
      <th>wc_qoq_change</th>
      <th>wc_volatility</th>
      <th>wc_vs_5y_avg</th>
      <th>wc_yoy_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6643.0000</td>
      <td>6339.0000</td>
      <td>6339.0000</td>
      <td>6142.0000</td>
      <td>6338.0000</td>
      <td>6339.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6461.0000</td>
      <td>6676.0000</td>
      <td>6433.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6516.0000</td>
      <td>6649.0000</td>
      <td>6548.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6339.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6328.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6581.0000</td>
      <td>6668.0000</td>
      <td>6356.0000</td>
      <td>6661.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.2380</td>
      <td>0.2620</td>
      <td>0.4710</td>
      <td>-82.2460</td>
      <td>6.8240</td>
      <td>0.1770</td>
      <td>3186.1790</td>
      <td>3407.8430</td>
      <td>3125.2830</td>
      <td>3275.5000</td>
      <td>3021.7670</td>
      <td>2957.8610</td>
      <td>1043.8460</td>
      <td>3447.6250</td>
      <td>15.0900</td>
      <td>3170.6320</td>
      <td>207.3820</td>
      <td>0.2480</td>
      <td>3514.7130</td>
      <td>3469.1070</td>
      <td>3519.0970</td>
      <td>35.7310</td>
      <td>0.0700</td>
      <td>201.3950</td>
      <td>0.1820</td>
      <td>63.7900</td>
      <td>17.7230</td>
      <td>285.9850</td>
      <td>849.7380</td>
      <td>742.0440</td>
      <td>760.2380</td>
      <td>778.4140</td>
      <td>766.9440</td>
      <td>783.1050</td>
      <td>733.3170</td>
      <td>771.7860</td>
      <td>6.0190</td>
      <td>741.9390</td>
      <td>819.6460</td>
      <td>819.6310</td>
      <td>0.3050</td>
      <td>820.9620</td>
      <td>3.8900</td>
      <td>7.9430</td>
      <td>3.2560</td>
      <td>1.3470</td>
      <td>8.6660</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.8040</td>
      <td>0.3220</td>
      <td>0.4440</td>
      <td>1181.5890</td>
      <td>454.5400</td>
      <td>0.4230</td>
      <td>10182.5140</td>
      <td>10839.4800</td>
      <td>9941.8650</td>
      <td>10446.1390</td>
      <td>9805.0000</td>
      <td>9867.7760</td>
      <td>32083.2300</td>
      <td>10987.9490</td>
      <td>81.3670</td>
      <td>10131.3520</td>
      <td>8544.0880</td>
      <td>0.4320</td>
      <td>11405.0280</td>
      <td>11027.0840</td>
      <td>11406.6200</td>
      <td>1728.8480</td>
      <td>1.8860</td>
      <td>8244.5100</td>
      <td>0.3860</td>
      <td>27.3600</td>
      <td>42.2880</td>
      <td>8789.9260</td>
      <td>4118.9500</td>
      <td>3711.6070</td>
      <td>3844.7910</td>
      <td>3829.6670</td>
      <td>3854.8280</td>
      <td>3924.2670</td>
      <td>3661.9540</td>
      <td>3786.2700</td>
      <td>1854.2810</td>
      <td>3540.3600</td>
      <td>4211.6840</td>
      <td>4140.6110</td>
      <td>0.4610</td>
      <td>4212.5240</td>
      <td>1.8440</td>
      <td>1192.0370</td>
      <td>21.5040</td>
      <td>9.3930</td>
      <td>1707.0440</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-191.9890</td>
      <td>0.0000</td>
      <td>-28.0000</td>
      <td>-47154.0000</td>
      <td>0.0000</td>
      <td>-28.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-175610.6250</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-63.4760</td>
      <td>-100.0000</td>
      <td>0.0000</td>
      <td>15.0000</td>
      <td>-2800.0000</td>
      <td>-48112.5000</td>
      <td>-28417.5900</td>
      <td>-24248.0000</td>
      <td>-25249.4300</td>
      <td>-27084.1200</td>
      <td>-25897.0000</td>
      <td>-26781.3100</td>
      <td>-24248.0000</td>
      <td>-23829.2300</td>
      <td>-70815.7000</td>
      <td>-20910.0500</td>
      <td>-56440.3600</td>
      <td>-25214.2200</td>
      <td>0.0000</td>
      <td>-56440.3600</td>
      <td>0.0000</td>
      <td>-24243.8300</td>
      <td>0.0100</td>
      <td>-134.1820</td>
      <td>-70815.7000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.1240</td>
      <td>0.0910</td>
      <td>0.3330</td>
      <td>-16.8630</td>
      <td>0.1930</td>
      <td>0.0280</td>
      <td>64.4520</td>
      <td>69.0420</td>
      <td>57.8950</td>
      <td>64.8350</td>
      <td>49.9620</td>
      <td>36.3180</td>
      <td>3.9090</td>
      <td>73.1350</td>
      <td>-6.3950</td>
      <td>57.4200</td>
      <td>-6.2600</td>
      <td>0.0000</td>
      <td>74.1720</td>
      <td>73.0480</td>
      <td>74.6200</td>
      <td>-3.3830</td>
      <td>-0.0150</td>
      <td>-6.7480</td>
      <td>0.0000</td>
      <td>65.0000</td>
      <td>2.7970</td>
      <td>1.0710</td>
      <td>19.4720</td>
      <td>18.1400</td>
      <td>7.5450</td>
      <td>17.5300</td>
      <td>16.4900</td>
      <td>16.0980</td>
      <td>5.0750</td>
      <td>14.8980</td>
      <td>-20.7700</td>
      <td>13.8580</td>
      <td>7.1320</td>
      <td>18.8020</td>
      <td>0.0000</td>
      <td>7.3420</td>
      <td>3.0000</td>
      <td>-12.2460</td>
      <td>0.3250</td>
      <td>0.7090</td>
      <td>-21.2370</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.4470</td>
      <td>0.2360</td>
      <td>0.4790</td>
      <td>-5.6550</td>
      <td>0.3840</td>
      <td>0.1440</td>
      <td>420.4800</td>
      <td>471.2050</td>
      <td>410.8750</td>
      <td>437.2800</td>
      <td>389.7050</td>
      <td>339.4750</td>
      <td>63.0940</td>
      <td>476.0900</td>
      <td>4.2340</td>
      <td>413.6800</td>
      <td>6.7910</td>
      <td>0.0000</td>
      <td>479.8000</td>
      <td>474.4400</td>
      <td>481.3800</td>
      <td>0.0630</td>
      <td>0.0140</td>
      <td>5.9820</td>
      <td>0.0000</td>
      <td>80.0000</td>
      <td>14.4410</td>
      <td>17.2860</td>
      <td>241.3950</td>
      <td>211.7400</td>
      <td>202.3500</td>
      <td>205.2350</td>
      <td>212.9450</td>
      <td>193.5100</td>
      <td>190.9150</td>
      <td>176.7300</td>
      <td>6.6090</td>
      <td>190.6300</td>
      <td>214.7300</td>
      <td>232.6650</td>
      <td>0.0000</td>
      <td>215.3200</td>
      <td>5.0000</td>
      <td>0.0710</td>
      <td>0.7210</td>
      <td>1.1210</td>
      <td>5.9570</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.9930</td>
      <td>0.3780</td>
      <td>0.6420</td>
      <td>-2.0430</td>
      <td>0.8360</td>
      <td>0.3010</td>
      <td>2172.1200</td>
      <td>2347.5750</td>
      <td>2065.9980</td>
      <td>2246.5750</td>
      <td>1975.0080</td>
      <td>1816.3920</td>
      <td>155.9750</td>
      <td>2356.9780</td>
      <td>18.2950</td>
      <td>2175.8220</td>
      <td>26.2000</td>
      <td>0.0000</td>
      <td>2396.8180</td>
      <td>2375.8350</td>
      <td>2404.4320</td>
      <td>3.7780</td>
      <td>0.1150</td>
      <td>25.2960</td>
      <td>0.0000</td>
      <td>80.0000</td>
      <td>30.1040</td>
      <td>42.7330</td>
      <td>799.0550</td>
      <td>732.9550</td>
      <td>740.8580</td>
      <td>746.2650</td>
      <td>736.2120</td>
      <td>750.8180</td>
      <td>707.4420</td>
      <td>727.1880</td>
      <td>34.7340</td>
      <td>684.9880</td>
      <td>792.6100</td>
      <td>804.9650</td>
      <td>1.0000</td>
      <td>793.3900</td>
      <td>5.0000</td>
      <td>8.4570</td>
      <td>1.8730</td>
      <td>1.5170</td>
      <td>34.7170</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2096.7860</td>
      <td>19.0000</td>
      <td>0.9990</td>
      <td>8114.3330</td>
      <td>36181.5000</td>
      <td>1.0000</td>
      <td>247121.6700</td>
      <td>266915.1100</td>
      <td>224532.4600</td>
      <td>264276.8600</td>
      <td>221075.8800</td>
      <td>234588.5000</td>
      <td>2398901.6670</td>
      <td>269597.3200</td>
      <td>3451.7640</td>
      <td>251537.5600</td>
      <td>674813.7930</td>
      <td>1.0000</td>
      <td>268964.7500</td>
      <td>264276.8600</td>
      <td>268964.7500</td>
      <td>138086.6670</td>
      <td>80.5650</td>
      <td>652316.6670</td>
      <td>1.0000</td>
      <td>100.0000</td>
      <td>99.9990</td>
      <td>657233.3330</td>
      <td>100932.2900</td>
      <td>90580.6400</td>
      <td>94477.4100</td>
      <td>110501.0500</td>
      <td>89869.7500</td>
      <td>140064.1200</td>
      <td>90580.6400</td>
      <td>123889.0000</td>
      <td>47071.6140</td>
      <td>100740.6700</td>
      <td>103293.0000</td>
      <td>103293.0000</td>
      <td>1.0000</td>
      <td>103293.0000</td>
      <td>5.0000</td>
      <td>90050.0000</td>
      <td>899.8970</td>
      <td>572.7500</td>
      <td>55839.5520</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Momentum & Technical (39 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market_cap</th>
      <th>daily_turnover_ratio</th>
      <th>liquidity_score</th>
      <th>price_momentum_1m</th>
      <th>price_momentum_3m</th>
      <th>price_momentum_6m</th>
      <th>price_momentum_1y</th>
      <th>price_momentum_5d</th>
      <th>ema_crossover_20_50</th>
      <th>ema_crossover_50_250</th>
      <th>price_vs_ema_20d</th>
      <th>price_vs_ema_250d</th>
      <th>pct_off_52w_high</th>
      <th>pct_above_52w_low</th>
      <th>range_52w_position</th>
      <th>beta_momentum</th>
      <th>volatility_regime</th>
      <th>volatility_1m</th>
      <th>volatility_3m</th>
      <th>volatility_6m</th>
      <th>volatility_1y</th>
      <th>volatility_trend_short</th>
      <th>volatility_trend_long</th>
      <th>vol_ratio_3m_1y</th>
      <th>vol_hump</th>
      <th>beta_1y</th>
      <th>beta_2y</th>
      <th>beta_5y</th>
      <th>beta_term_structure</th>
      <th>beta_convexity</th>
      <th>realized_vs_implied_proxy</th>
      <th>price_momentum_5y</th>
      <th>long_term_trend_score</th>
      <th>price_momentum_3y</th>
      <th>multi_year_high_flag</th>
      <th>secular_trend_flag</th>
      <th>log_market_cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6668.0000</td>
      <td>6668.0000</td>
      <td>6447.0000</td>
      <td>6654.0000</td>
      <td>6613.0000</td>
      <td>6535.0000</td>
      <td>6668.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6674.0000</td>
      <td>6547.0000</td>
      <td>6674.0000</td>
      <td>6674.0000</td>
      <td>6676.0000</td>
      <td>6189.0000</td>
      <td>6670.0000</td>
      <td>6670.0000</td>
      <td>6673.0000</td>
      <td>6673.0000</td>
      <td>6674.0000</td>
      <td>6670.0000</td>
      <td>6673.0000</td>
      <td>6673.0000</td>
      <td>6673.0000</td>
      <td>6350.0000</td>
      <td>6350.0000</td>
      <td>6189.0000</td>
      <td>6172.0000</td>
      <td>6189.0000</td>
      <td>6670.0000</td>
      <td>5770.0000</td>
      <td>6676.0000</td>
      <td>6237.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17028.7460</td>
      <td>0.0090</td>
      <td>1447705.6980</td>
      <td>9.0970</td>
      <td>4.0180</td>
      <td>15.2110</td>
      <td>55.1380</td>
      <td>0.3740</td>
      <td>0.0760</td>
      <td>0.1960</td>
      <td>0.0190</td>
      <td>0.0940</td>
      <td>-0.8020</td>
      <td>-0.1930</td>
      <td>0.5580</td>
      <td>-0.0010</td>
      <td>1.0360</td>
      <td>43.0570</td>
      <td>46.6590</td>
      <td>44.2570</td>
      <td>42.9520</td>
      <td>3.6130</td>
      <td>-1.2980</td>
      <td>1.1190</td>
      <td>-0.5520</td>
      <td>0.7680</td>
      <td>0.7870</td>
      <td>0.7670</td>
      <td>-0.0160</td>
      <td>0.0200</td>
      <td>1.0360</td>
      <td>124.3550</td>
      <td>0.7580</td>
      <td>97.4950</td>
      <td>0.3640</td>
      <td>0.3900</td>
      <td>8.0490</td>
    </tr>
    <tr>
      <th>std</th>
      <td>121510.7720</td>
      <td>0.0190</td>
      <td>8878675.9720</td>
      <td>15.3590</td>
      <td>31.4730</td>
      <td>55.2400</td>
      <td>149.3730</td>
      <td>6.4220</td>
      <td>0.9960</td>
      <td>0.9710</td>
      <td>0.0620</td>
      <td>0.2840</td>
      <td>0.1540</td>
      <td>1.6120</td>
      <td>0.2930</td>
      <td>0.6930</td>
      <td>0.3100</td>
      <td>23.6080</td>
      <td>22.1490</td>
      <td>22.3650</td>
      <td>24.9990</td>
      <td>13.2520</td>
      <td>14.1440</td>
      <td>0.2210</td>
      <td>9.1520</td>
      <td>0.7710</td>
      <td>0.6450</td>
      <td>0.6750</td>
      <td>4.7920</td>
      <td>0.4400</td>
      <td>0.3100</td>
      <td>508.1790</td>
      <td>2.0090</td>
      <td>294.2520</td>
      <td>0.4810</td>
      <td>0.4880</td>
      <td>1.7490</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.9200</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-51.0260</td>
      <td>-84.5480</td>
      <td>-95.3830</td>
      <td>-97.3070</td>
      <td>-51.8150</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-0.4740</td>
      <td>-0.9720</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>0.0000</td>
      <td>-16.8100</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4600</td>
      <td>2.6000</td>
      <td>0.0000</td>
      <td>-191.2200</td>
      <td>-150.3700</td>
      <td>0.0070</td>
      <td>-317.8250</td>
      <td>-3.5300</td>
      <td>-1.4500</td>
      <td>-5.4900</td>
      <td>-79.0000</td>
      <td>-8.4950</td>
      <td>0.0000</td>
      <td>-99.9950</td>
      <td>-0.9860</td>
      <td>-99.9770</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>2.4780</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1110.0580</td>
      <td>0.0010</td>
      <td>15104.5890</td>
      <td>0.5530</td>
      <td>-11.5490</td>
      <td>-11.0690</td>
      <td>-5.6610</td>
      <td>-2.9410</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-0.0160</td>
      <td>-0.0680</td>
      <td>-0.9210</td>
      <td>-0.8200</td>
      <td>0.2990</td>
      <td>-0.3200</td>
      <td>0.8360</td>
      <td>28.2020</td>
      <td>31.8900</td>
      <td>29.9700</td>
      <td>28.8900</td>
      <td>-2.4400</td>
      <td>-4.6700</td>
      <td>0.9860</td>
      <td>-2.5950</td>
      <td>0.3000</td>
      <td>0.3800</td>
      <td>0.3700</td>
      <td>-0.5070</td>
      <td>-0.1550</td>
      <td>0.8360</td>
      <td>-33.1880</td>
      <td>-0.0610</td>
      <td>-17.4060</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>7.0120</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3422.9600</td>
      <td>0.0040</td>
      <td>79542.3390</td>
      <td>6.3360</td>
      <td>-0.9330</td>
      <td>4.5450</td>
      <td>21.9680</td>
      <td>-0.1980</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0100</td>
      <td>0.0490</td>
      <td>-0.8360</td>
      <td>-0.6060</td>
      <td>0.5930</td>
      <td>-0.0200</td>
      <td>1.0140</td>
      <td>38.1950</td>
      <td>41.4900</td>
      <td>38.7200</td>
      <td>37.5650</td>
      <td>2.6400</td>
      <td>-1.7600</td>
      <td>1.1220</td>
      <td>-0.9450</td>
      <td>0.6800</td>
      <td>0.7100</td>
      <td>0.7000</td>
      <td>-0.0710</td>
      <td>0.0100</td>
      <td>1.0140</td>
      <td>19.8710</td>
      <td>0.2430</td>
      <td>25.3810</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>8.1380</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9218.3380</td>
      <td>0.0090</td>
      <td>490597.0810</td>
      <td>14.5550</td>
      <td>12.6840</td>
      <td>27.2130</td>
      <td>66.9880</td>
      <td>2.7480</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0420</td>
      <td>0.2000</td>
      <td>-0.7180</td>
      <td>-0.1420</td>
      <td>0.8220</td>
      <td>0.3000</td>
      <td>1.2030</td>
      <td>52.2900</td>
      <td>56.6300</td>
      <td>53.4200</td>
      <td>51.2700</td>
      <td>8.4600</td>
      <td>0.8500</td>
      <td>1.2610</td>
      <td>0.9050</td>
      <td>1.1300</td>
      <td>1.0800</td>
      <td>1.0800</td>
      <td>0.4120</td>
      <td>0.1850</td>
      <td>1.2030</td>
      <td>116.4160</td>
      <td>0.8300</td>
      <td>101.4140</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>9.1290</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4850089.0000</td>
      <td>0.4650</td>
      <td>505489213.4770</td>
      <td>157.1010</td>
      <td>716.8200</td>
      <td>1352.2780</td>
      <td>5599.0910</td>
      <td>63.5040</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.5310</td>
      <td>3.6590</td>
      <td>-0.0340</td>
      <td>59.1610</td>
      <td>1.0000</td>
      <td>20.4700</td>
      <td>3.7440</td>
      <td>469.0100</td>
      <td>277.7900</td>
      <td>540.4200</td>
      <td>747.1600</td>
      <td>163.6700</td>
      <td>644.5100</td>
      <td>1.9040</td>
      <td>307.7050</td>
      <td>21.9000</td>
      <td>21.9400</td>
      <td>19.8900</td>
      <td>96.0000</td>
      <td>21.0400</td>
      <td>3.7440</td>
      <td>16552.0790</td>
      <td>37.8790</td>
      <td>6332.2430</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>15.3950</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Price Target Dynamics (15 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pt_momentum_1w</th>
      <th>pt_momentum_1m</th>
      <th>pt_momentum_3m</th>
      <th>pt_momentum_6m</th>
      <th>pt_momentum_1y</th>
      <th>analyst_coverage_change_1m</th>
      <th>analyst_coverage_change_3m</th>
      <th>analyst_coverage_change_1y</th>
      <th>pt_acceleration_long</th>
      <th>pt_acceleration_short</th>
      <th>pt_consensus_convergence</th>
      <th>pt_median_momentum_1m</th>
      <th>pt_median_momentum_3m</th>
      <th>pt_vs_price_momentum</th>
      <th>analyst_coverage_trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6662.0000</td>
      <td>6633.0000</td>
      <td>6583.0000</td>
      <td>6465.0000</td>
      <td>6350.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6328.0000</td>
      <td>6574.0000</td>
      <td>6583.0000</td>
      <td>6633.0000</td>
      <td>6583.0000</td>
      <td>6577.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0040</td>
      <td>0.0110</td>
      <td>0.0600</td>
      <td>0.1450</td>
      <td>0.2950</td>
      <td>0.0220</td>
      <td>0.0020</td>
      <td>0.3580</td>
      <td>-0.2350</td>
      <td>-0.0470</td>
      <td>-0.0090</td>
      <td>0.0110</td>
      <td>0.0610</td>
      <td>0.0540</td>
      <td>-0.0010</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0440</td>
      <td>0.0980</td>
      <td>0.2400</td>
      <td>0.5330</td>
      <td>0.7980</td>
      <td>0.6460</td>
      <td>1.0810</td>
      <td>2.2740</td>
      <td>0.6740</td>
      <td>0.1770</td>
      <td>0.2470</td>
      <td>0.1070</td>
      <td>0.2570</td>
      <td>0.2210</td>
      <td>0.2120</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.8140</td>
      <td>-0.9950</td>
      <td>-0.9910</td>
      <td>-0.9910</td>
      <td>-0.9920</td>
      <td>-8.0000</td>
      <td>-9.0000</td>
      <td>-12.0000</td>
      <td>-15.8300</td>
      <td>-3.3620</td>
      <td>-5.6210</td>
      <td>-0.9950</td>
      <td>-0.9910</td>
      <td>-0.9440</td>
      <td>-4.1750</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0000</td>
      <td>-0.0120</td>
      <td>-0.0330</td>
      <td>-0.0510</td>
      <td>-0.0720</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>-0.3110</td>
      <td>-0.0790</td>
      <td>-0.0790</td>
      <td>-0.0060</td>
      <td>-0.0330</td>
      <td>-0.0640</td>
      <td>-0.0250</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0140</td>
      <td>0.0510</td>
      <td>0.1110</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0790</td>
      <td>-0.0070</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0090</td>
      <td>0.0340</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0000</td>
      <td>0.0170</td>
      <td>0.1080</td>
      <td>0.2220</td>
      <td>0.4130</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0530</td>
      <td>0.0130</td>
      <td>0.0590</td>
      <td>0.0130</td>
      <td>0.1110</td>
      <td>0.1440</td>
      <td>0.0500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0740</td>
      <td>2.1040</td>
      <td>6.8420</td>
      <td>17.6670</td>
      <td>17.4230</td>
      <td>11.0000</td>
      <td>13.0000</td>
      <td>30.0000</td>
      <td>1.8710</td>
      <td>0.9330</td>
      <td>4.1250</td>
      <td>2.6010</td>
      <td>6.8420</td>
      <td>4.0300</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Profitability (75 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ebit_1fqfq</th>
      <th>roe</th>
      <th>roa</th>
      <th>gross_margin_pct</th>
      <th>operating_margin_pct</th>
      <th>net_margin_pct</th>
      <th>ebitda_margin_pct</th>
      <th>roic</th>
      <th>rnd_intensity</th>
      <th>equity_multiplier</th>
      <th>gross_margin_trend_yoy</th>
      <th>operating_margin_trend</th>
      <th>net_margin_trend_yoy</th>
      <th>ebitda_margin_trend</th>
      <th>margin_expansion_flag</th>
      <th>ebit_1fy</th>
      <th>ebit_2fy</th>
      <th>ebit_3fqfq</th>
      <th>ebit_3fy</th>
      <th>ebit_4fqfq</th>
      <th>ebit_4fy</th>
      <th>ebit_5yavgfq</th>
      <th>ebit_5yavgltm</th>
      <th>ebit_adj_fq</th>
      <th>ebit_adj_fy</th>
      <th>ebit_adj_ltm</th>
      <th>ebit_cagr_3y</th>
      <th>ebit_fq</th>
      <th>ebit_fy</th>
      <th>ebit_growth_yoy</th>
      <th>...</th>
      <th>ebitda_adj_ltm</th>
      <th>ebitda_fq</th>
      <th>ebitda_fy</th>
      <th>ebitda_ltm</th>
      <th>ebitda_margin_ltm</th>
      <th>ebitda_positive_years</th>
      <th>ebitda_qoq_growth</th>
      <th>ebitda_vs_5y_avg</th>
      <th>ebit_2fqfq</th>
      <th>ebitda_2fqfq</th>
      <th>gp_1fqfq</th>
      <th>gp_2fqfq</th>
      <th>gp_2fy</th>
      <th>gp_3fqfq</th>
      <th>gp_3fy</th>
      <th>gp_4fqfq</th>
      <th>gp_4fy</th>
      <th>gp_fq</th>
      <th>gp_ltm</th>
      <th>gp_margin_expansion</th>
      <th>gp_margin_fq</th>
      <th>gp_margin_trend</th>
      <th>gp_positive_quarters</th>
      <th>gp_qoq_growth</th>
      <th>gp_yoy_growth</th>
      <th>ebitda_cagr_3y</th>
      <th>margin_stability_score</th>
      <th>gp_1fy</th>
      <th>gp_fy</th>
      <th>ebitda_adj_fq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6180.0000</td>
      <td>6249.0000</td>
      <td>6424.0000</td>
      <td>6532.0000</td>
      <td>6339.0000</td>
      <td>6532.0000</td>
      <td>6643.0000</td>
      <td>6532.0000</td>
      <td>6643.0000</td>
      <td>6421.0000</td>
      <td>6531.0000</td>
      <td>6336.0000</td>
      <td>6531.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>5234.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6638.0000</td>
      <td>...</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6532.0000</td>
      <td>6676.0000</td>
      <td>6586.0000</td>
      <td>6180.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6448.0000</td>
      <td>5887.0000</td>
      <td>6676.0000</td>
      <td>6492.0000</td>
      <td>6518.0000</td>
      <td>5604.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>231.6240</td>
      <td>0.0460</td>
      <td>0.0370</td>
      <td>0.3930</td>
      <td>-99.3350</td>
      <td>0.0660</td>
      <td>-87.6810</td>
      <td>8.7470</td>
      <td>0.4480</td>
      <td>3.2280</td>
      <td>0.0000</td>
      <td>-0.3400</td>
      <td>0.0020</td>
      <td>-0.3390</td>
      <td>0.0850</td>
      <td>801.8620</td>
      <td>766.8780</td>
      <td>204.7050</td>
      <td>836.6860</td>
      <td>200.4190</td>
      <td>701.4360</td>
      <td>197.3520</td>
      <td>755.6630</td>
      <td>205.5100</td>
      <td>914.1820</td>
      <td>785.2850</td>
      <td>11.9580</td>
      <td>227.7740</td>
      <td>878.2960</td>
      <td>70.1570</td>
      <td>...</td>
      <td>1045.9050</td>
      <td>326.3730</td>
      <td>1268.0860</td>
      <td>1278.0900</td>
      <td>-87.6810</td>
      <td>4.3620</td>
      <td>12.4360</td>
      <td>1.4050</td>
      <td>218.0400</td>
      <td>318.8120</td>
      <td>603.5240</td>
      <td>577.9880</td>
      <td>2097.9500</td>
      <td>552.1610</td>
      <td>2085.0830</td>
      <td>542.8700</td>
      <td>1868.9840</td>
      <td>618.3290</td>
      <td>2373.0540</td>
      <td>0.1440</td>
      <td>6.2680</td>
      <td>-29.6260</td>
      <td>4.6790</td>
      <td>28.1160</td>
      <td>47.6160</td>
      <td>10.7730</td>
      <td>99.9920</td>
      <td>2145.5090</td>
      <td>2346.7140</td>
      <td>273.2360</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1289.3660</td>
      <td>4.0860</td>
      <td>0.0970</td>
      <td>0.2640</td>
      <td>3191.7480</td>
      <td>0.6500</td>
      <td>3098.8890</td>
      <td>458.2030</td>
      <td>20.7230</td>
      <td>58.4600</td>
      <td>0.0290</td>
      <td>92.0230</td>
      <td>0.2140</td>
      <td>91.3770</td>
      <td>0.2780</td>
      <td>4439.3230</td>
      <td>4235.6930</td>
      <td>1160.5720</td>
      <td>5113.9760</td>
      <td>1211.7890</td>
      <td>3856.6100</td>
      <td>1082.8560</td>
      <td>4126.7520</td>
      <td>1528.4150</td>
      <td>4920.3790</td>
      <td>5053.1300</td>
      <td>46.7490</td>
      <td>1438.8430</td>
      <td>4862.1190</td>
      <td>1331.1070</td>
      <td>...</td>
      <td>6404.6590</td>
      <td>1742.8500</td>
      <td>6050.0490</td>
      <td>6182.8040</td>
      <td>3098.8890</td>
      <td>1.4450</td>
      <td>2044.0100</td>
      <td>11.6470</td>
      <td>1172.0490</td>
      <td>1470.2860</td>
      <td>2551.4250</td>
      <td>2385.0370</td>
      <td>8449.2950</td>
      <td>2296.6260</td>
      <td>8735.9830</td>
      <td>2435.6840</td>
      <td>7427.0820</td>
      <td>2820.9750</td>
      <td>9856.4360</td>
      <td>0.3510</td>
      <td>1242.1350</td>
      <td>1219.2110</td>
      <td>1.0080</td>
      <td>2773.2250</td>
      <td>790.5790</td>
      <td>38.1260</td>
      <td>0.1040</td>
      <td>8876.5560</td>
      <td>9747.2330</td>
      <td>1800.7310</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3315.9200</td>
      <td>-311.7780</td>
      <td>-1.1530</td>
      <td>-2.9200</td>
      <td>-243100.0000</td>
      <td>-2.9710</td>
      <td>-236433.3330</td>
      <td>-25471.8750</td>
      <td>-0.1730</td>
      <td>-850.4040</td>
      <td>-0.6890</td>
      <td>-5840.4940</td>
      <td>-4.4180</td>
      <td>-5807.1850</td>
      <td>0.0000</td>
      <td>-10019.0000</td>
      <td>-13476.0300</td>
      <td>-5921.4100</td>
      <td>-25932.8400</td>
      <td>-2211.7500</td>
      <td>-7183.7000</td>
      <td>-1208.7100</td>
      <td>-4648.6400</td>
      <td>-17445.8100</td>
      <td>-17723.1500</td>
      <td>-9667.4900</td>
      <td>-92.0000</td>
      <td>-5413.4500</td>
      <td>-7232.1400</td>
      <td>-15318.4870</td>
      <td>...</td>
      <td>-2859.0000</td>
      <td>-5413.2200</td>
      <td>-5443.4700</td>
      <td>-5443.4700</td>
      <td>-236433.3330</td>
      <td>0.0000</td>
      <td>-129293.7950</td>
      <td>-156.8570</td>
      <td>-4747.0000</td>
      <td>-4256.0000</td>
      <td>-3263.0500</td>
      <td>-4990.5500</td>
      <td>-2690.0000</td>
      <td>-8043.3600</td>
      <td>-23672.5300</td>
      <td>-28708.0000</td>
      <td>-2612.5300</td>
      <td>-31120.0000</td>
      <td>-2056.0000</td>
      <td>0.0000</td>
      <td>-71215.3850</td>
      <td>-69234.9440</td>
      <td>0.0000</td>
      <td>-131785.7140</td>
      <td>-14138.7100</td>
      <td>-90.9090</td>
      <td>92.2030</td>
      <td>-2771.0000</td>
      <td>-2056.0000</td>
      <td>-2005.5400</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.7380</td>
      <td>0.0250</td>
      <td>0.0180</td>
      <td>0.2170</td>
      <td>4.2210</td>
      <td>0.0180</td>
      <td>7.6010</td>
      <td>3.5250</td>
      <td>0.0000</td>
      <td>1.4450</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>23.4120</td>
      <td>19.8980</td>
      <td>3.6700</td>
      <td>16.2350</td>
      <td>2.9500</td>
      <td>9.3320</td>
      <td>3.4400</td>
      <td>14.0000</td>
      <td>0.0000</td>
      <td>24.4050</td>
      <td>0.0000</td>
      <td>-6.4200</td>
      <td>4.6850</td>
      <td>28.8880</td>
      <td>-9.8200</td>
      <td>...</td>
      <td>0.0000</td>
      <td>11.4950</td>
      <td>53.8000</td>
      <td>53.4820</td>
      <td>7.6010</td>
      <td>5.0000</td>
      <td>-19.2670</td>
      <td>0.9140</td>
      <td>5.1980</td>
      <td>11.5780</td>
      <td>38.2080</td>
      <td>35.9680</td>
      <td>122.7980</td>
      <td>33.3200</td>
      <td>109.0500</td>
      <td>29.2180</td>
      <td>83.2270</td>
      <td>38.2980</td>
      <td>165.7120</td>
      <td>0.0000</td>
      <td>20.8220</td>
      <td>-8.6410</td>
      <td>5.0000</td>
      <td>-5.8490</td>
      <td>0.5320</td>
      <td>-4.2520</td>
      <td>99.9990</td>
      <td>135.3480</td>
      <td>162.4570</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41.3500</td>
      <td>0.0970</td>
      <td>0.0420</td>
      <td>0.3620</td>
      <td>9.8280</td>
      <td>0.0640</td>
      <td>14.7350</td>
      <td>7.9390</td>
      <td>0.0000</td>
      <td>1.9670</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>138.1250</td>
      <td>128.1050</td>
      <td>33.6750</td>
      <td>124.1800</td>
      <td>31.8850</td>
      <td>99.0450</td>
      <td>31.5700</td>
      <td>119.8800</td>
      <td>5.6100</td>
      <td>164.4400</td>
      <td>40.0750</td>
      <td>6.3650</td>
      <td>38.8700</td>
      <td>159.7100</td>
      <td>11.8150</td>
      <td>...</td>
      <td>19.0600</td>
      <td>60.4500</td>
      <td>249.8400</td>
      <td>251.0900</td>
      <td>14.7350</td>
      <td>5.0000</td>
      <td>0.0000</td>
      <td>1.1780</td>
      <td>38.3600</td>
      <td>59.6950</td>
      <td>131.9950</td>
      <td>125.6300</td>
      <td>441.9900</td>
      <td>117.0300</td>
      <td>407.7250</td>
      <td>113.1450</td>
      <td>356.8250</td>
      <td>134.3500</td>
      <td>535.2700</td>
      <td>0.0000</td>
      <td>35.6060</td>
      <td>-1.6980</td>
      <td>5.0000</td>
      <td>0.8540</td>
      <td>11.0530</td>
      <td>6.3550</td>
      <td>100.0000</td>
      <td>463.0950</td>
      <td>527.6100</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>149.2720</td>
      <td>0.1770</td>
      <td>0.0700</td>
      <td>0.5480</td>
      <td>18.3580</td>
      <td>0.1310</td>
      <td>25.0480</td>
      <td>14.1200</td>
      <td>0.0260</td>
      <td>2.8370</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>499.3700</td>
      <td>480.1280</td>
      <td>123.6720</td>
      <td>489.0520</td>
      <td>124.5450</td>
      <td>436.8380</td>
      <td>120.2050</td>
      <td>454.8800</td>
      <td>98.2480</td>
      <td>577.6020</td>
      <td>384.6700</td>
      <td>21.5810</td>
      <td>145.6620</td>
      <td>560.7100</td>
      <td>37.2990</td>
      <td>...</td>
      <td>475.1650</td>
      <td>210.2220</td>
      <td>841.0600</td>
      <td>837.4650</td>
      <td>25.0480</td>
      <td>5.0000</td>
      <td>15.9670</td>
      <td>1.4910</td>
      <td>137.6850</td>
      <td>207.9400</td>
      <td>412.0750</td>
      <td>395.1550</td>
      <td>1419.6620</td>
      <td>375.3250</td>
      <td>1403.0180</td>
      <td>370.2450</td>
      <td>1271.9950</td>
      <td>418.3750</td>
      <td>1645.1480</td>
      <td>0.0000</td>
      <td>55.0450</td>
      <td>2.9920</td>
      <td>5.0000</td>
      <td>12.4040</td>
      <td>24.8720</td>
      <td>18.8910</td>
      <td>100.0000</td>
      <td>1460.3550</td>
      <td>1615.6500</td>
      <td>118.6150</td>
    </tr>
    <tr>
      <th>max</th>
      <td>53027.8400</td>
      <td>62.2500</td>
      <td>1.4590</td>
      <td>1.4200</td>
      <td>140.8190</td>
      <td>36.6380</td>
      <td>456.4100</td>
      <td>22680.0000</td>
      <td>1646.3330</td>
      <td>4538.1430</td>
      <td>0.7800</td>
      <td>2650.0000</td>
      <td>15.5950</td>
      <td>2618.7500</td>
      <td>1.0000</td>
      <td>209196.6500</td>
      <td>232638.9400</td>
      <td>51026.8800</td>
      <td>304295.7500</td>
      <td>48035.3200</td>
      <td>205489.1700</td>
      <td>55961.2800</td>
      <td>216842.8700</td>
      <td>50852.0000</td>
      <td>188435.6200</td>
      <td>188434.8200</td>
      <td>1134.5950</td>
      <td>50852.0000</td>
      <td>196872.0300</td>
      <td>61700.0000</td>
      <td>...</td>
      <td>213253.2700</td>
      <td>54066.0000</td>
      <td>219964.8100</td>
      <td>219964.8100</td>
      <td>456.4100</td>
      <td>5.0000</td>
      <td>93032.8700</td>
      <td>816.5000</td>
      <td>44550.7400</td>
      <td>52059.6700</td>
      <td>91499.0000</td>
      <td>86893.0000</td>
      <td>281964.2000</td>
      <td>78691.0000</td>
      <td>354456.7100</td>
      <td>88899.0000</td>
      <td>247208.0000</td>
      <td>103427.0000</td>
      <td>360510.0000</td>
      <td>1.0000</td>
      <td>7376.4710</td>
      <td>6226.4710</td>
      <td>5.0000</td>
      <td>141800.0000</td>
      <td>34328.5710</td>
      <td>780.1920</td>
      <td>100.0000</td>
      <td>311671.0000</td>
      <td>360510.0000</td>
      <td>54066.0000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 75 columns</p>
</div>

    ============================================================
      Quality & Risk (17 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>net_buyback_flag</th>
      <th>beta_trend</th>
      <th>shares_yoy_change_pct</th>
      <th>low_beta_flag</th>
      <th>has_goodwill_impairment</th>
      <th>has_asset_writedown</th>
      <th>has_restructuring</th>
      <th>goodwill_to_assets_pct</th>
      <th>intangible_intensity</th>
      <th>exceptional_items_to_ebitda</th>
      <th>altman_z_score</th>
      <th>altman_z_trend</th>
      <th>current_ratio</th>
      <th>quick_ratio</th>
      <th>beta_spread</th>
      <th>high_beta_flag</th>
      <th>beta_stability_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6172.0000</td>
      <td>6483.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6339.0000</td>
      <td>6339.0000</td>
      <td>6621.0000</td>
      <td>5602.0000</td>
      <td>5586.0000</td>
      <td>6339.0000</td>
      <td>6338.0000</td>
      <td>6189.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.3460</td>
      <td>50.7260</td>
      <td>0.2860</td>
      <td>0.3650</td>
      <td>0.0810</td>
      <td>0.4150</td>
      <td>0.2090</td>
      <td>9.4130</td>
      <td>0.0800</td>
      <td>0.1800</td>
      <td>4.8090</td>
      <td>-0.0300</td>
      <td>23.1650</td>
      <td>20.1400</td>
      <td>-0.0010</td>
      <td>0.1230</td>
      <td>80.8730</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.4760</td>
      <td>476.5540</td>
      <td>9.8700</td>
      <td>0.4810</td>
      <td>0.2720</td>
      <td>0.4930</td>
      <td>0.4070</td>
      <td>14.0200</td>
      <td>0.1600</td>
      <td>2.1610</td>
      <td>9.7540</td>
      <td>2.7340</td>
      <td>1626.8080</td>
      <td>1421.1640</td>
      <td>0.6930</td>
      <td>0.3280</td>
      <td>18.7490</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>-6300.0000</td>
      <td>-0.6760</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.9430</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-33.4200</td>
      <td>-112.4600</td>
      <td>0.0000</td>
      <td>-2.6510</td>
      <td>-16.8100</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0000</td>
      <td>-43.5960</td>
      <td>-0.0060</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.7200</td>
      <td>0.0000</td>
      <td>1.1000</td>
      <td>0.8070</td>
      <td>-0.3200</td>
      <td>0.0000</td>
      <td>72.8750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>-2.2860</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.9210</td>
      <td>0.0090</td>
      <td>0.0010</td>
      <td>2.8400</td>
      <td>0.0000</td>
      <td>1.6000</td>
      <td>1.1840</td>
      <td>-0.0200</td>
      <td>0.0000</td>
      <td>86.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0000</td>
      <td>51.9050</td>
      <td>0.0110</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>14.1440</td>
      <td>0.0920</td>
      <td>0.0370</td>
      <td>4.7970</td>
      <td>0.0000</td>
      <td>2.5000</td>
      <td>1.9270</td>
      <td>0.3000</td>
      <td>0.0000</td>
      <td>94.5000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0000</td>
      <td>9600.0000</td>
      <td>658.9090</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>85.3880</td>
      <td>2.3540</td>
      <td>100.0850</td>
      <td>225.3800</td>
      <td>103.8000</td>
      <td>129523.6000</td>
      <td>113141.0000</td>
      <td>20.4700</td>
      <td>1.0000</td>
      <td>100.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Revenue Forecasting (49 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue_avg_med_diff_pct</th>
      <th>revenue_consensus_strength</th>
      <th>revenue_est_avg_ntm</th>
      <th>revenue_est_med_fy1e</th>
      <th>revenue_est_med_ntm</th>
      <th>revenue_revision_trend</th>
      <th>revenue_vs_current</th>
      <th>ebit_estimate_spread</th>
      <th>ebitda_est_vs_actual</th>
      <th>estimate_confidence_score</th>
      <th>forward_ebitda_margin</th>
      <th>consensus_revenue_growth</th>
      <th>forward_revenue_multiple</th>
      <th>revenue_acceleration</th>
      <th>revenue_est_revision_trend</th>
      <th>revenue_est_spread</th>
      <th>revenue_estimate_count</th>
      <th>revenue_guidance_gap</th>
      <th>revenue_1fqfq</th>
      <th>revenue_1fy</th>
      <th>revenue_2fqfq</th>
      <th>revenue_2fy</th>
      <th>revenue_2y_growth</th>
      <th>revenue_3fqfq</th>
      <th>revenue_3fy</th>
      <th>revenue_3y_growth</th>
      <th>revenue_4fqfq</th>
      <th>revenue_4fy</th>
      <th>revenue_4q_avg</th>
      <th>revenue_4q_trend</th>
      <th>revenue_qoq_4q</th>
      <th>revenue_qoq_growth</th>
      <th>revenue_stability_score</th>
      <th>revenue_yoy_quarterly</th>
      <th>revenue_5y_avg</th>
      <th>revenue_accelerating_flag</th>
      <th>revenue_cagr_3y</th>
      <th>revenue_cagr_4y</th>
      <th>revenue_fq</th>
      <th>revenue_fq_vs_4q_avg</th>
      <th>revenue_fy</th>
      <th>revenue_growth_flag</th>
      <th>revenue_ltm</th>
      <th>revenue_positive_qoq_streak</th>
      <th>revenue_qoq_2q</th>
      <th>revenue_qoq_3q</th>
      <th>revenue_est_avg_fy1e</th>
      <th>revenue_beat_potential</th>
      <th>revenue_4y_growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6408.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6305.0000</td>
      <td>6532.0000</td>
      <td>6458.0000</td>
      <td>6454.0000</td>
      <td>6676.0000</td>
      <td>6432.0000</td>
      <td>6531.0000</td>
      <td>6425.0000</td>
      <td>5909.0000</td>
      <td>6305.0000</td>
      <td>6408.0000</td>
      <td>6676.0000</td>
      <td>6432.0000</td>
      <td>6468.0000</td>
      <td>6676.0000</td>
      <td>6449.0000</td>
      <td>6425.0000</td>
      <td>6425.0000</td>
      <td>6465.0000</td>
      <td>6345.0000</td>
      <td>6345.0000</td>
      <td>6340.0000</td>
      <td>6315.0000</td>
      <td>6374.0000</td>
      <td>6326.0000</td>
      <td>6289.0000</td>
      <td>6468.0000</td>
      <td>6676.0000</td>
      <td>6326.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6326.0000</td>
      <td>6293.0000</td>
      <td>6676.0000</td>
      <td>6374.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6391.0000</td>
      <td>6422.0000</td>
      <td>6676.0000</td>
      <td>6432.0000</td>
      <td>6314.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.7890</td>
      <td>98.3990</td>
      <td>8416.4950</td>
      <td>8288.1120</td>
      <td>8397.4470</td>
      <td>0.7830</td>
      <td>1.7890</td>
      <td>-7.2460</td>
      <td>12.4660</td>
      <td>99.0980</td>
      <td>-443.9060</td>
      <td>80.4640</td>
      <td>53.0100</td>
      <td>0.0700</td>
      <td>0.7830</td>
      <td>1.7890</td>
      <td>6.2010</td>
      <td>4.2420</td>
      <td>1982.9400</td>
      <td>7063.6680</td>
      <td>1938.5440</td>
      <td>7369.7050</td>
      <td>153.4230</td>
      <td>1861.2070</td>
      <td>7393.8150</td>
      <td>196.0350</td>
      <td>1916.2380</td>
      <td>6537.5570</td>
      <td>1960.6820</td>
      <td>42.0920</td>
      <td>6.0120</td>
      <td>28.5420</td>
      <td>73.5450</td>
      <td>42.0920</td>
      <td>6684.4040</td>
      <td>0.6480</td>
      <td>10.8010</td>
      <td>12.1290</td>
      <td>1984.7410</td>
      <td>1.0840</td>
      <td>7612.0380</td>
      <td>0.7830</td>
      <td>7664.1770</td>
      <td>2.3960</td>
      <td>66.9810</td>
      <td>37.5460</td>
      <td>8303.7150</td>
      <td>8.5200</td>
      <td>750.0210</td>
    </tr>
    <tr>
      <th>std</th>
      <td>78.2340</td>
      <td>6.6400</td>
      <td>30099.4440</td>
      <td>29622.8630</td>
      <td>30060.9980</td>
      <td>33.7910</td>
      <td>34.0030</td>
      <td>174.2170</td>
      <td>3381.2120</td>
      <td>5.0690</td>
      <td>24924.1630</td>
      <td>3400.1250</td>
      <td>1782.4740</td>
      <td>4.0860</td>
      <td>33.7910</td>
      <td>78.2340</td>
      <td>7.1130</td>
      <td>212.8180</td>
      <td>6828.5290</td>
      <td>25060.9770</td>
      <td>6654.4320</td>
      <td>25268.6170</td>
      <td>4293.0190</td>
      <td>6409.0410</td>
      <td>26099.4230</td>
      <td>5721.3290</td>
      <td>6796.9500</td>
      <td>22439.7930</td>
      <td>6809.7390</td>
      <td>883.4430</td>
      <td>408.1010</td>
      <td>1027.2810</td>
      <td>27.0410</td>
      <td>883.4430</td>
      <td>23780.6440</td>
      <td>0.4770</td>
      <td>40.0050</td>
      <td>31.9010</td>
      <td>7210.3320</td>
      <td>1.8670</td>
      <td>26546.3840</td>
      <td>0.4120</td>
      <td>26779.7130</td>
      <td>0.9290</td>
      <td>3121.2920</td>
      <td>1555.2320</td>
      <td>29662.9710</td>
      <td>1194.6510</td>
      <td>24712.0990</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-36.0750</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
      <td>-1.7460</td>
      <td>-13140.0000</td>
      <td>-30737.5000</td>
      <td>0.0000</td>
      <td>-1951300.0000</td>
      <td>-100.0000</td>
      <td>-7.9220</td>
      <td>-5.3170</td>
      <td>-1.0000</td>
      <td>-36.0750</td>
      <td>0.0000</td>
      <td>-100.0000</td>
      <td>-4050.3000</td>
      <td>-42.3300</td>
      <td>-187.5500</td>
      <td>-96.9000</td>
      <td>-282.9840</td>
      <td>-187.5500</td>
      <td>-6.1900</td>
      <td>-2563.4290</td>
      <td>-3865.0200</td>
      <td>0.0000</td>
      <td>-35.4180</td>
      <td>-10800.0000</td>
      <td>-15600.0000</td>
      <td>-1800.0000</td>
      <td>0.0000</td>
      <td>-10800.0000</td>
      <td>-302.2500</td>
      <td>0.0000</td>
      <td>-67.2570</td>
      <td>-75.1130</td>
      <td>-4494.4900</td>
      <td>-22.6670</td>
      <td>-456.1100</td>
      <td>0.0000</td>
      <td>-15.7500</td>
      <td>0.0000</td>
      <td>-2659.5160</td>
      <td>-288.0600</td>
      <td>0.0000</td>
      <td>-157.2730</td>
      <td>-1444.6640</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.1030</td>
      <td>98.9060</td>
      <td>532.1220</td>
      <td>520.6850</td>
      <td>523.4930</td>
      <td>0.0230</td>
      <td>1.0100</td>
      <td>0.0000</td>
      <td>-38.0530</td>
      <td>99.4530</td>
      <td>10.4170</td>
      <td>1.8560</td>
      <td>1.0170</td>
      <td>-0.0970</td>
      <td>0.0230</td>
      <td>-0.1030</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>134.5820</td>
      <td>393.1850</td>
      <td>130.8000</td>
      <td>429.7100</td>
      <td>-2.0650</td>
      <td>121.6100</td>
      <td>412.4700</td>
      <td>-2.5260</td>
      <td>122.5400</td>
      <td>351.1900</td>
      <td>135.2780</td>
      <td>1.5260</td>
      <td>-8.9810</td>
      <td>-2.0650</td>
      <td>63.4190</td>
      <td>1.5260</td>
      <td>291.8570</td>
      <td>0.0000</td>
      <td>-0.8010</td>
      <td>1.5760</td>
      <td>121.0320</td>
      <td>0.9910</td>
      <td>467.6050</td>
      <td>1.0000</td>
      <td>478.5500</td>
      <td>2.0000</td>
      <td>-3.4670</td>
      <td>-0.0500</td>
      <td>522.6230</td>
      <td>-15.0730</td>
      <td>6.2540</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>99.7280</td>
      <td>1791.1600</td>
      <td>1759.5900</td>
      <td>1784.5000</td>
      <td>0.0810</td>
      <td>1.0630</td>
      <td>0.0000</td>
      <td>-19.7150</td>
      <td>99.8640</td>
      <td>17.7600</td>
      <td>7.8970</td>
      <td>2.0890</td>
      <td>-0.0200</td>
      <td>0.0810</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>0.0000</td>
      <td>434.7100</td>
      <td>1413.7150</td>
      <td>419.6900</td>
      <td>1497.5500</td>
      <td>11.2870</td>
      <td>402.4700</td>
      <td>1440.7800</td>
      <td>18.1600</td>
      <td>406.8000</td>
      <td>1251.2900</td>
      <td>433.3410</td>
      <td>10.7160</td>
      <td>0.1520</td>
      <td>1.7340</td>
      <td>82.2430</td>
      <td>10.7160</td>
      <td>1251.1650</td>
      <td>1.0000</td>
      <td>5.7570</td>
      <td>7.4620</td>
      <td>412.9550</td>
      <td>1.0450</td>
      <td>1610.5650</td>
      <td>1.0000</td>
      <td>1636.7000</td>
      <td>2.0000</td>
      <td>2.5520</td>
      <td>6.2290</td>
      <td>1763.6400</td>
      <td>-6.2780</td>
      <td>33.1800</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.2170</td>
      <td>100.0000</td>
      <td>5875.3880</td>
      <td>5786.4280</td>
      <td>5849.0100</td>
      <td>0.1940</td>
      <td>1.1700</td>
      <td>0.0000</td>
      <td>-7.4400</td>
      <td>100.0000</td>
      <td>28.8530</td>
      <td>19.6190</td>
      <td>4.2030</td>
      <td>0.0580</td>
      <td>0.1940</td>
      <td>0.2170</td>
      <td>9.0000</td>
      <td>0.0000</td>
      <td>1413.0280</td>
      <td>4898.4680</td>
      <td>1371.5900</td>
      <td>5085.8600</td>
      <td>31.2500</td>
      <td>1311.0900</td>
      <td>4958.4900</td>
      <td>50.9580</td>
      <td>1345.3550</td>
      <td>4466.9550</td>
      <td>1380.7510</td>
      <td>22.5720</td>
      <td>8.8790</td>
      <td>10.3040</td>
      <td>93.4080</td>
      <td>22.5720</td>
      <td>4504.7480</td>
      <td>1.0000</td>
      <td>14.7580</td>
      <td>15.9100</td>
      <td>1374.6550</td>
      <td>1.1230</td>
      <td>5316.3950</td>
      <td>1.0000</td>
      <td>5363.1480</td>
      <td>3.0000</td>
      <td>10.1540</td>
      <td>12.8750</td>
      <td>5808.6520</td>
      <td>-1.3750</td>
      <td>80.3670</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6080.0000</td>
      <td>100.0000</td>
      <td>808042.7200</td>
      <td>806170.0000</td>
      <td>806170.0000</td>
      <td>2638.2540</td>
      <td>2679.0770</td>
      <td>830.4790</td>
      <td>250320.6900</td>
      <td>100.0000</td>
      <td>1275.3720</td>
      <td>267807.6920</td>
      <td>100284.4170</td>
      <td>270.3300</td>
      <td>2638.2540</td>
      <td>6080.0000</td>
      <td>59.0000</td>
      <td>17028.5710</td>
      <td>180169.0000</td>
      <td>680985.0000</td>
      <td>177402.0000</td>
      <td>648125.0000</td>
      <td>280833.3330</td>
      <td>165609.0000</td>
      <td>611289.0000</td>
      <td>391733.3330</td>
      <td>187792.0000</td>
      <td>572754.0000</td>
      <td>179231.0000</td>
      <td>51300.0000</td>
      <td>21608.4000</td>
      <td>72600.0000</td>
      <td>100.0000</td>
      <td>51300.0000</td>
      <td>629497.5200</td>
      <td>1.0000</td>
      <td>1476.5240</td>
      <td>995.7870</td>
      <td>213386.0000</td>
      <td>140.4130</td>
      <td>716924.0000</td>
      <td>1.0000</td>
      <td>716924.0000</td>
      <td>4.0000</td>
      <td>225600.0000</td>
      <td>122544.4440</td>
      <td>808042.7200</td>
      <td>95404.4690</td>
      <td>1441700.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Technical Analysis (11 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>high_volume_flag</th>
      <th>ema_slope_20d</th>
      <th>price_vs_ema_100d</th>
      <th>near_52w_low_flag</th>
      <th>volume_momentum_score</th>
      <th>breakout_signal</th>
      <th>volatility_compression</th>
      <th>volatility_term_structure</th>
      <th>near_52w_high_flag</th>
      <th>low_volume_flag</th>
      <th>ema_trend_consistency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6676.0000</td>
      <td>6666.0000</td>
      <td>6637.0000</td>
      <td>6676.0000</td>
      <td>6438.0000</td>
      <td>6676.0000</td>
      <td>6670.0000</td>
      <td>6673.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.1010</td>
      <td>0.0090</td>
      <td>4.2430</td>
      <td>0.0470</td>
      <td>0.0890</td>
      <td>0.1580</td>
      <td>-0.1040</td>
      <td>2.4030</td>
      <td>0.1600</td>
      <td>0.1200</td>
      <td>0.1500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.3010</td>
      <td>0.0510</td>
      <td>16.0420</td>
      <td>0.2110</td>
      <td>0.2140</td>
      <td>0.3640</td>
      <td>21.9310</td>
      <td>11.4360</td>
      <td>0.3670</td>
      <td>0.3250</td>
      <td>0.7830</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>-0.5880</td>
      <td>-86.5160</td>
      <td>0.0000</td>
      <td>-7.0260</td>
      <td>0.0000</td>
      <td>-319.3900</td>
      <td>-465.0400</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-1.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0000</td>
      <td>-0.0200</td>
      <td>-4.8370</td>
      <td>0.0000</td>
      <td>0.0020</td>
      <td>0.0000</td>
      <td>-7.2100</td>
      <td>-0.0900</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>0.0030</td>
      <td>1.7740</td>
      <td>0.0000</td>
      <td>0.0470</td>
      <td>0.0000</td>
      <td>-0.4900</td>
      <td>3.0500</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0000</td>
      <td>0.0310</td>
      <td>10.0940</td>
      <td>0.0000</td>
      <td>0.1250</td>
      <td>0.0000</td>
      <td>6.0300</td>
      <td>6.3600</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0000</td>
      <td>0.5340</td>
      <td>218.1730</td>
      <td>1.0000</td>
      <td>3.2760</td>
      <td>1.0000</td>
      <td>681.9900</td>
      <td>72.2700</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Temporal Patterns (16 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fiscal_quarter</th>
      <th>fiscal_month</th>
      <th>fiscal_year</th>
      <th>days_to_earnings</th>
      <th>earnings_report_recency</th>
      <th>reporting_lag</th>
      <th>fiscal_year_progress</th>
      <th>days_since_last_report</th>
      <th>earnings_season_flag</th>
      <th>fiscal_quarter_progress</th>
      <th>is_fy_end_month</th>
      <th>is_quarter_end_month</th>
      <th>post_earnings_window</th>
      <th>pre_earnings_window</th>
      <th>days_to_fy_end</th>
      <th>reporting_freshness_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6585.0000</td>
      <td>6585.0000</td>
      <td>6585.0000</td>
      <td>6676.0000</td>
      <td>6668.0000</td>
      <td>6668.0000</td>
      <td>6585.0000</td>
      <td>6668.0000</td>
      <td>6676.0000</td>
      <td>6585.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6587.0000</td>
      <td>6676.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.5210</td>
      <td>10.4580</td>
      <td>2025.9470</td>
      <td>20.8580</td>
      <td>124.7130</td>
      <td>68.0840</td>
      <td>0.8710</td>
      <td>124.7130</td>
      <td>1.0000</td>
      <td>0.9650</td>
      <td>0.0060</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5690</td>
      <td>-181.4760</td>
      <td>2.9890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.8660</td>
      <td>2.7470</td>
      <td>0.4670</td>
      <td>28.3370</td>
      <td>142.2860</td>
      <td>147.9000</td>
      <td>0.2290</td>
      <td>142.2860</td>
      <td>0.0000</td>
      <td>0.1030</td>
      <td>0.0780</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4950</td>
      <td>136.3400</td>
      <td>13.7040</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0000</td>
      <td>2.0000</td>
      <td>2000.0000</td>
      <td>-2.0000</td>
      <td>24.0000</td>
      <td>-67.0000</td>
      <td>0.1670</td>
      <td>24.0000</td>
      <td>1.0000</td>
      <td>0.3330</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-4862.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.0000</td>
      <td>9.0000</td>
      <td>2026.0000</td>
      <td>5.0000</td>
      <td>114.0000</td>
      <td>30.0000</td>
      <td>0.7500</td>
      <td>114.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-206.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.0000</td>
      <td>12.0000</td>
      <td>2026.0000</td>
      <td>12.0000</td>
      <td>114.0000</td>
      <td>38.0000</td>
      <td>1.0000</td>
      <td>114.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>-114.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.0000</td>
      <td>12.0000</td>
      <td>2026.0000</td>
      <td>20.0000</td>
      <td>114.0000</td>
      <td>83.0000</td>
      <td>1.0000</td>
      <td>114.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>-114.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.0000</td>
      <td>12.0000</td>
      <td>2027.0000</td>
      <td>193.0000</td>
      <td>9795.0000</td>
      <td>9738.0000</td>
      <td>1.0000</td>
      <td>9795.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>-24.0000</td>
      <td>100.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Valuation Ratios (16 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_e_ratio</th>
      <th>p_b_ratio</th>
      <th>ev_ebitda_ratio</th>
      <th>ev_sales_ratio</th>
      <th>peg_ratio</th>
      <th>valuation_dividend_yield</th>
      <th>tangible_book_per_share</th>
      <th>price_to_tangible_book</th>
      <th>tangible_equity_ratio</th>
      <th>intangibles_to_equity</th>
      <th>goodwill_to_equity</th>
      <th>tangible_asset_quality</th>
      <th>tangible_book_value_fy</th>
      <th>tangible_book_value_ltm</th>
      <th>tbv_vs_calculated</th>
      <th>tbv_yoy_growth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5199.0000</td>
      <td>6394.0000</td>
      <td>5962.0000</td>
      <td>6482.0000</td>
      <td>4341.0000</td>
      <td>4220.0000</td>
      <td>6676.0000</td>
      <td>5357.0000</td>
      <td>6339.0000</td>
      <td>6643.0000</td>
      <td>6643.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6676.0000</td>
      <td>6643.0000</td>
      <td>6649.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.6180</td>
      <td>5.2300</td>
      <td>19.5540</td>
      <td>5.8800</td>
      <td>0.1150</td>
      <td>0.0320</td>
      <td>0.0000</td>
      <td>9.2860</td>
      <td>28.3820</td>
      <td>44.5370</td>
      <td>24.5940</td>
      <td>83.6650</td>
      <td>2356.1810</td>
      <td>2395.3570</td>
      <td>0.2480</td>
      <td>4.2930</td>
    </tr>
    <tr>
      <th>std</th>
      <td>53.6670</td>
      <td>11.6550</td>
      <td>31.6890</td>
      <td>20.6620</td>
      <td>19.6640</td>
      <td>0.0520</td>
      <td>0.0000</td>
      <td>23.7300</td>
      <td>50.5440</td>
      <td>1527.8150</td>
      <td>218.9360</td>
      <td>23.2780</td>
      <td>13486.8890</td>
      <td>13697.9990</td>
      <td>51.9820</td>
      <td>183.6880</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>0.1000</td>
      <td>0.1000</td>
      <td>0.0000</td>
      <td>-934.4980</td>
      <td>0.0000</td>
      <td>-0.0010</td>
      <td>0.2000</td>
      <td>-2800.0000</td>
      <td>-6234.3430</td>
      <td>-11438.3840</td>
      <td>0.0000</td>
      <td>-91551.0000</td>
      <td>-91551.0000</td>
      <td>-3674.5000</td>
      <td>-1704.4740</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.0000</td>
      <td>1.5000</td>
      <td>7.4000</td>
      <td>1.1000</td>
      <td>-1.4970</td>
      <td>0.0100</td>
      <td>0.0000</td>
      <td>1.9000</td>
      <td>8.0840</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>75.7730</td>
      <td>56.2500</td>
      <td>54.6720</td>
      <td>0.7740</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.7000</td>
      <td>2.6000</td>
      <td>11.1000</td>
      <td>2.2000</td>
      <td>0.3480</td>
      <td>0.0230</td>
      <td>0.0000</td>
      <td>3.7000</td>
      <td>30.5430</td>
      <td>0.7490</td>
      <td>2.5780</td>
      <td>95.4450</td>
      <td>504.8750</td>
      <td>510.3450</td>
      <td>0.9660</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>40.7000</td>
      <td>5.2000</td>
      <td>18.7000</td>
      <td>4.7000</td>
      <td>1.6910</td>
      <td>0.0400</td>
      <td>0.0000</td>
      <td>8.1000</td>
      <td>52.9460</td>
      <td>19.0950</td>
      <td>32.2700</td>
      <td>100.0000</td>
      <td>1904.9150</td>
      <td>1923.7920</td>
      <td>1.0040</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>497.9000</td>
      <td>447.5000</td>
      <td>493.7000</td>
      <td>469.6000</td>
      <td>342.3780</td>
      <td>1.8520</td>
      <td>0.0210</td>
      <td>486.2000</td>
      <td>99.9320</td>
      <td>120671.4290</td>
      <td>9991.1070</td>
      <td>100.0000</td>
      <td>381885.0000</td>
      <td>381885.0000</td>
      <td>1482.1670</td>
      <td>14000.0000</td>
    </tr>
  </tbody>
</table>
</div>

    ============================================================
      Valuation Timeseries (22 features)
    ============================================================

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ev_sales_trend_1y</th>
      <th>ev_ebitda_momentum</th>
      <th>p_e_momentum_yoy</th>
      <th>p_e_momentum_qoq</th>
      <th>ev_sales_vs_3y_avg</th>
      <th>ev_ebitda_vs_3y_avg</th>
      <th>p_e_vs_3y_avg</th>
      <th>ev_sales_forward_discount</th>
      <th>ev_ebitda_forward_discount</th>
      <th>p_e_forward_discount</th>
      <th>p_b_vs_5y_avg</th>
      <th>ev_sales_qoq_1q</th>
      <th>p_b_momentum_yoy</th>
      <th>forward_pe_premium</th>
      <th>ev_ebitda_qoq_trend</th>
      <th>ev_sales_qoq_2q</th>
      <th>ev_sales_qoq_4q</th>
      <th>p_e_percentile_proxy</th>
      <th>p_e_vs_5y_avg</th>
      <th>valuation_compression</th>
      <th>valuation_mean_reversion</th>
      <th>ev_sales_qoq_3q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6426.0000</td>
      <td>5926.0000</td>
      <td>5162.0000</td>
      <td>5165.0000</td>
      <td>6049.0000</td>
      <td>5533.0000</td>
      <td>4792.0000</td>
      <td>6331.0000</td>
      <td>5773.0000</td>
      <td>4989.0000</td>
      <td>5469.0000</td>
      <td>6447.0000</td>
      <td>6353.0000</td>
      <td>4989.0000</td>
      <td>5933.0000</td>
      <td>6483.0000</td>
      <td>6448.0000</td>
      <td>4792.0000</td>
      <td>4462.0000</td>
      <td>4729.0000</td>
      <td>4724.0000</td>
      <td>6450.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.1310</td>
      <td>0.3420</td>
      <td>0.1800</td>
      <td>0.3030</td>
      <td>0.1620</td>
      <td>0.1260</td>
      <td>0.2410</td>
      <td>-0.0740</td>
      <td>-0.1180</td>
      <td>-0.1620</td>
      <td>1.1660</td>
      <td>0.1280</td>
      <td>0.1390</td>
      <td>-16.1910</td>
      <td>0.4100</td>
      <td>0.3170</td>
      <td>0.0650</td>
      <td>0.4820</td>
      <td>0.2560</td>
      <td>0.1720</td>
      <td>0.1680</td>
      <td>0.5320</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.9540</td>
      <td>13.2740</td>
      <td>1.1270</td>
      <td>1.6410</td>
      <td>0.5920</td>
      <td>0.6210</td>
      <td>0.9880</td>
      <td>0.9500</td>
      <td>0.6750</td>
      <td>0.8810</td>
      <td>0.9910</td>
      <td>1.1050</td>
      <td>0.7210</td>
      <td>88.0770</td>
      <td>17.2650</td>
      <td>18.7480</td>
      <td>3.6450</td>
      <td>1.9760</td>
      <td>1.1730</td>
      <td>0.6480</td>
      <td>0.5500</td>
      <td>28.3550</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.0000</td>
      <td>-0.9590</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-0.9840</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-0.9860</td>
      <td>0.0370</td>
      <td>-1.0000</td>
      <td>-0.9360</td>
      <td>-98.5630</td>
      <td>-0.9590</td>
      <td>-1.0000</td>
      <td>-1.0000</td>
      <td>-2.0000</td>
      <td>-1.0000</td>
      <td>-0.8990</td>
      <td>-0.8400</td>
      <td>-0.9980</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.0500</td>
      <td>-0.0580</td>
      <td>-0.0610</td>
      <td>-0.1160</td>
      <td>-0.1430</td>
      <td>-0.1850</td>
      <td>-0.2260</td>
      <td>-0.1670</td>
      <td>-0.2790</td>
      <td>-0.4060</td>
      <td>0.7060</td>
      <td>-0.0960</td>
      <td>-0.0620</td>
      <td>-40.6190</td>
      <td>-0.1060</td>
      <td>-0.0780</td>
      <td>-0.1900</td>
      <td>-0.4520</td>
      <td>-0.3050</td>
      <td>-0.1890</td>
      <td>-0.1530</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0000</td>
      <td>0.0220</td>
      <td>0.0330</td>
      <td>0.0410</td>
      <td>0.0340</td>
      <td>0.0160</td>
      <td>0.0440</td>
      <td>-0.0790</td>
      <td>-0.1320</td>
      <td>-0.2010</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0290</td>
      <td>-20.1260</td>
      <td>0.0190</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0880</td>
      <td>0.0090</td>
      <td>0.0320</td>
      <td>0.0500</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.1540</td>
      <td>0.1330</td>
      <td>0.1600</td>
      <td>0.2700</td>
      <td>0.3200</td>
      <td>0.2760</td>
      <td>0.3990</td>
      <td>0.0000</td>
      <td>-0.0320</td>
      <td>-0.0570</td>
      <td>1.3530</td>
      <td>0.1740</td>
      <td>0.1670</td>
      <td>-5.6960</td>
      <td>0.1700</td>
      <td>0.1180</td>
      <td>0.1250</td>
      <td>0.7980</td>
      <td>0.4230</td>
      <td>0.3330</td>
      <td>0.3310</td>
      <td>0.1210</td>
    </tr>
    <tr>
      <th>max</th>
      <td>56.5740</td>
      <td>963.0000</td>
      <td>27.6820</td>
      <td>36.3290</td>
      <td>10.1040</td>
      <td>13.5330</td>
      <td>18.7200</td>
      <td>70.0950</td>
      <td>31.9230</td>
      <td>41.6560</td>
      <td>43.8890</td>
      <td>69.0000</td>
      <td>21.6920</td>
      <td>4165.6250</td>
      <td>1271.0000</td>
      <td>1507.0000</td>
      <td>288.4620</td>
      <td>37.4400</td>
      <td>20.2420</td>
      <td>9.2600</td>
      <td>6.2060</td>
      <td>2249.0000</td>
    </tr>
  </tbody>
</table>
</div>

## 4. Valuation Analysis

```python
val_cols = [c for c in FEATURE_CATEGORIES.get("Valuation Ratios", []) if c in df.columns]
if len(val_cols) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("Valuation Ratios Overview", fontsize=16, fontweight="bold")

    # P/E Distribution
    ax = axes[0, 0]
    pe_col = next((c for c in ["p_e_ratio"] if c in df.columns), None)
    if pe_col:
        pe_data = df[pe_col].dropna().clip(-50, 200)
        ax.hist(pe_data, bins=60, color=COLORS["primary"], alpha=0.85, edgecolor="white")
        ax.axvline(pe_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
                   label=f"Median: {pe_data.median():.1f}")
        ax.set_title("P/E Ratio Distribution", fontweight="bold")
        ax.set_xlabel("P/E Ratio")
        ax.legend()

    # P/B Distribution
    ax = axes[0, 1]
    pb_col = next((c for c in ["p_b_ratio"] if c in df.columns), None)
    if pb_col:
        pb_data = df[pb_col].dropna().clip(0, 30)
        ax.hist(pb_data, bins=50, color=COLORS["info"], alpha=0.85, edgecolor="white")
        ax.axvline(pb_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
                   label=f"Median: {pb_data.median():.1f}")
        ax.set_title("P/B Ratio Distribution", fontweight="bold")
        ax.set_xlabel("P/B Ratio")
        ax.legend()

    # EV/EBITDA Distribution
    ax = axes[1, 0]
    ev_col = next((c for c in ["ev_ebitda_ratio"] if c in df.columns), None)
    if ev_col:
        ev_data = df[ev_col].dropna().clip(-10, 80)
        ax.hist(ev_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
        ax.axvline(ev_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
                   label=f"Median: {ev_data.median():.1f}")
        ax.set_title("EV/EBITDA Distribution", fontweight="bold")
        ax.set_xlabel("EV/EBITDA")
        ax.legend()

    # Valuation correlation heatmap
    ax = axes[1, 1]
    val_data = df[val_cols].dropna(thresh=len(val_cols) // 2)
    if len(val_data) > 10:
        corr = val_data.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    ax=ax, cbar_kws={"shrink": 0.8}, square=True,
                    xticklabels=[c.replace("_", "\n") for c in corr.columns],
                    yticklabels=[c.replace("_", "\n") for c in corr.columns])
        ax.set_title("Valuation Metrics Correlation", fontweight="bold")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_14_0.png)

## 5. Profitability & Margins Analysis

```python
prof_cols = [c for c in FEATURE_CATEGORIES.get("Profitability", []) if c in df.columns]
margin_cols = [c for c in prof_cols if "margin" in c.lower()]
if margin_cols:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Profitability & Margin Analysis", fontsize=16, fontweight="bold")

    # Margin distributions (violin)
    ax = axes[0, 0]
    margin_data = df[margin_cols].melt(var_name="Margin", value_name="Value")
    margin_data["Value"] = margin_data["Value"].clip(-100, 100)
    sns.violinplot(data=margin_data, x="Margin", y="Value", ax=ax, inner="box",
                   palette="Set2", cut=0)
    ax.set_xticklabels([c.replace("_", "\n") for c in margin_cols], rotation=45, ha="right", fontsize=8)
    ax.set_title("Margin Distributions", fontweight="bold")
    ax.axhline(0, color="grey", linestyle=":", lw=0.8)

    # ROE vs ROA scatter
    ax = axes[0, 1]
    roe_col = next((c for c in ["roe"] if c in df.columns), None)
    roa_col = next((c for c in ["roa"] if c in df.columns), None)
    if roe_col and roa_col:
        valid = df[[roe_col, roa_col]].dropna()
        valid = valid[(valid[roe_col].between(-100, 200)) & (valid[roa_col].between(-50, 50))]
        ax.scatter(valid[roa_col], valid[roe_col], alpha=0.3, s=10, c=COLORS["primary"])
        ax.set_xlabel("ROA (%)", fontsize=11)
        ax.set_ylabel("ROE (%)", fontsize=11)
        ax.set_title("ROE vs ROA", fontweight="bold")
        ax.axhline(0, color="grey", linestyle=":", lw=0.8)
        ax.axvline(0, color="grey", linestyle=":", lw=0.8)

    # Profitability correlation
    ax = axes[1, 0]
    if len(prof_cols) >= 3:
        prof_data = df[prof_cols].dropna(thresh=len(prof_cols) // 2)
        if len(prof_data) > 10:
            corr = prof_data.corr()
            sns.heatmap(corr, annot=False, fmt=".1f", cmap="coolwarm", center=0, ax=ax,
                        xticklabels=[c[:12] for c in corr.columns],
                        yticklabels=[c[:12] for c in corr.columns], square=False)
            ax.set_title("Profitability Correlations", fontweight="bold")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\504397710.py:13: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax.set_xticklabels([c.replace("_", "\n") for c in margin_cols], rotation=45, ha="right", fontsize=8)

![png](pml_model_analysis_files/pml_model_analysis_16_1.png)

## 6. Momentum & Technical Analysis

```python
mom_cols = [c for c in FEATURE_CATEGORIES.get("Momentum & Technical", []) if c in df.columns]
price_mom_cols = [c for c in mom_cols if c.startswith("price_momentum")]
if price_mom_cols:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig.suptitle("Momentum & Technical Analysis", fontsize=16, fontweight="bold")

    # Momentum ribbon (box plots across timeframes)
    ax = axes[0]
    mom_data = df[price_mom_cols].melt(var_name="Timeframe", value_name="Return")
    mom_data["Return"] = mom_data["Return"].clip(-100, 300)
    sns.boxplot(data=mom_data, x="Timeframe", y="Return", ax=ax, palette="viridis",
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_xticklabels([c.replace("price_momentum_", "") for c in price_mom_cols],
                       rotation=45, ha="right")
    ax.axhline(0, color="red", linestyle="--", lw=1)
    ax.set_title("Momentum Across Timeframes", fontweight="bold")
    ax.set_ylabel("Return (%)")

    # 52-week range position histogram
    ax = axes[1]
    range_col = next((c for c in ["range_52w_position"] if c in df.columns), None)
    if range_col:
        range_data = df[range_col].dropna()
        ax.hist(range_data, bins=50, color=COLORS["info"], alpha=0.85, edgecolor="white")
        ax.axvline(range_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
                   label=f"Median: {range_data.median():.1%}")
        ax.set_title("52-Week Range Position", fontweight="bold")
        ax.set_xlabel("Position (0=Low, 1=High)")
        ax.legend()

    # Short vs long-term momentum scatter
    ax = axes[2]
    short_col = next((c for c in ["price_momentum_1m"] if c in df.columns), None)
    long_col = next((c for c in ["price_momentum_1y"] if c in df.columns), None)
    if short_col and long_col:
        valid = df[[short_col, long_col]].dropna()
        valid = valid[valid[short_col].between(-50, 50) & valid[long_col].between(-80, 200)]
        ax.hexbin(valid[long_col], valid[short_col], gridsize=30, cmap="YlGnBu", mincnt=1)
        ax.set_xlabel("1Y Momentum (%)")
        ax.set_ylabel("1M Momentum (%)")
        ax.set_title("Short vs Long-Term Momentum", fontweight="bold")
        ax.axhline(0, color="grey", linestyle=":", lw=0.8)
        ax.axvline(0, color="grey", linestyle=":", lw=0.8)

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\828036699.py:13: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax.set_xticklabels([c.replace("price_momentum_", "") for c in price_mom_cols],

![png](pml_model_analysis_files/pml_model_analysis_18_1.png)

## 7. Earnings Quality & EPS Trajectory

```python
eq_cols = [c for c in FEATURE_CATEGORIES.get("Earnings Quality", []) if c in df.columns]
eps_cols = [c for c in FEATURE_CATEGORIES.get("EPS Trajectory", []) if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle("Earnings Quality & EPS Trajectory", fontsize=16, fontweight="bold")

# EPS surprise distribution
ax = axes[0, 0]
surprise_col = next((c for c in ["eps_surprise_pct"] if c in df.columns), None)
if surprise_col:
    surprise_data = df[surprise_col].dropna().clip(-50, 100)
    ax.hist(surprise_data, bins=60, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero")
    ax.axvline(surprise_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
               label=f"Median: {surprise_data.median():.1f}%")
    ax.set_title("EPS Surprise Distribution", fontweight="bold")
    ax.set_xlabel("EPS Surprise (%)")
    ax.legend()

# Earnings quality score distribution
ax = axes[0, 1]
eq_score = next((c for c in ["earnings_quality_score", "earnings_quality_composite"] if c in df.columns), None)
if eq_score:
    score_data = df[eq_score].dropna()
    ax.hist(score_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(score_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: {score_data.median():.2f}")
    ax.set_title(f"{eq_score} Distribution", fontweight="bold")
    ax.legend()

# EPS trajectory score by sector
ax = axes[1, 0]
traj_col = next((c for c in ["eps_trajectory_score"] if c in df.columns), None)
sector_col = next((c for c in ["industry", "sector"] if c in df.columns), None)
if traj_col and sector_col:
    sectors = df[sector_col].value_counts().head(12).index
    sector_data = [df.loc[df[sector_col] == s, traj_col].dropna().values for s in sectors]
    bp = ax.boxplot(sector_data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color=COLORS["danger"], linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    box_colors = plt.cm.Set2(np.linspace(0, 1, len(sectors)))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticklabels([s[:16] for s in sectors], rotation=55, ha="right", fontsize=8)
    ax.set_title("EPS Trajectory Score by Sector", fontweight="bold")
    ax.set_ylabel("Trajectory Score")

# EPS stability vs growth scatter
ax = axes[1, 1]
stab_col = next((c for c in ["eps_stability"] if c in df.columns), None)
growth_col = next((c for c in ["eps_cagr_5y", "eps_yoy_growth"] if c in df.columns), None)
if stab_col and growth_col:
    valid = df[[stab_col, growth_col]].dropna()
    valid = valid[valid[growth_col].between(-100, 200)]
    ax.scatter(valid[growth_col], valid[stab_col], alpha=0.3, s=10, c=COLORS["info"])
    ax.set_xlabel("EPS Growth (%)")
    ax.set_ylabel("EPS Stability")
    ax.set_title("EPS Stability vs Growth", fontweight="bold")

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_20_0.png)

## 8. Growth Metrics Analysis

```python
growth_cols = [c for c in FEATURE_CATEGORIES.get("Growth Metrics", []) if c in df.columns]
if growth_cols:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig.suptitle("Growth Metrics Analysis", fontsize=16, fontweight="bold")

    # Revenue growth distribution
    ax = axes[0]
    rev_col = next((c for c in ["revenue_growth_yoy"] if c in df.columns), None)
    if rev_col:
        rev_data = df[rev_col].dropna().clip(-80, 200)
        ax.hist(rev_data, bins=60, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", lw=1)
        ax.axvline(rev_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
                   label=f"Median: {rev_data.median():.1f}%")
        ax.set_title("Revenue Growth YoY", fontweight="bold")
        ax.set_xlabel("Growth (%)")
        ax.legend()

    # Growth correlation heatmap
    ax = axes[1]
    g_data = df[growth_cols].dropna(thresh=len(growth_cols) // 2)
    if len(g_data) > 10:
        corr = g_data.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax,
                    xticklabels=[c[:14] for c in corr.columns],
                    yticklabels=[c[:14] for c in corr.columns], square=True)
        ax.set_title("Growth Metrics Correlations", fontweight="bold")

    # Revenue vs EPS growth scatter
    ax = axes[2]
    eps_g = next((c for c in ["eps_yoy_growth", "eps_cagr_5y"] if c in df.columns), None)
    if rev_col and eps_g:
        valid = df[[rev_col, eps_g]].dropna()
        valid = valid[valid[rev_col].between(-80, 200) & valid[eps_g].between(-100, 300)]
        ax.hexbin(valid[rev_col], valid[eps_g], gridsize=30, cmap="YlOrRd", mincnt=1)
        ax.set_xlabel("Revenue Growth YoY (%)")
        ax.set_ylabel("EPS Growth (%)")
        ax.set_title("Revenue vs EPS Growth", fontweight="bold")
        ax.axhline(0, color="grey", linestyle=":", lw=0.8)
        ax.axvline(0, color="grey", linestyle=":", lw=0.8)

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_22_0.png)

## 9. Quality & Risk Assessment

```python
qr_cols = [c for c in FEATURE_CATEGORIES.get("Quality & Risk", []) if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle("Quality & Risk Assessment", fontsize=16, fontweight="bold")

# Piotroski F-Score distribution
ax = axes[0, 0]
fscore_col = next((c for c in ["piotroski_f_score"] if c in df.columns), None)
if fscore_col:
    fscore_data = df[fscore_col].dropna()
    counts = fscore_data.value_counts().sort_index()
    bar_colors = [COLORS["danger"] if v <= 3 else COLORS["accent"] if v <= 6
    else COLORS["secondary"] for v in counts.index]
    ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
    ax.set_title("Piotroski F-Score Distribution", fontweight="bold")
    ax.set_xlabel("F-Score")
    ax.set_ylabel("Count")

# Altman Z-Score distribution
ax = axes[0, 1]
zscore_col = next((c for c in ["altman_z_score", "altman_z_score_fy"] if c in df.columns), None)
if zscore_col:
    zscore_data = df[zscore_col].dropna().clip(-5, 20)
    ax.hist(zscore_data, bins=50, color=COLORS["info"], alpha=0.85, edgecolor="white")
    ax.axvline(1.81, color=COLORS["danger"], linestyle="--", lw=2, label="Distress (<1.81)")
    ax.axvline(2.99, color=COLORS["secondary"], linestyle="--", lw=2, label="Safe (>2.99)")
    ax.set_title("Altman Z-Score Distribution", fontweight="bold")
    ax.set_xlabel("Z-Score")
    ax.legend()

# Quality score vs Risk scatter
ax = axes[1, 0]
aq_col = next((c for c in ["accounting_quality_score"] if c in df.columns), None)
distress_col = next((c for c in ["combined_distress_score"] if c in df.columns), None)
if aq_col and distress_col:
    valid = df[[aq_col, distress_col]].dropna()
    ax.scatter(valid[aq_col], valid[distress_col], alpha=0.3, s=10, c=COLORS["primary"])
    ax.set_xlabel("Accounting Quality Score")
    ax.set_ylabel("Distress Score")
    ax.set_title("Quality vs Distress", fontweight="bold")

# F-Score vs Z-Score joint distribution
ax = axes[1, 1]
if fscore_col and zscore_col:
    valid = df[[fscore_col, zscore_col]].dropna()
    valid = valid[valid[zscore_col].between(-5, 20)]
    ax.hexbin(valid[fscore_col], valid[zscore_col], gridsize=20, cmap="YlGnBu", mincnt=1)
    ax.set_xlabel("F-Score")
    ax.set_ylabel("Z-Score")
    ax.set_title("F-Score vs Z-Score", fontweight="bold")
    ax.axhline(1.81, color=COLORS["danger"], linestyle="--", lw=1)
    ax.axhline(2.99, color=COLORS["secondary"], linestyle="--", lw=1)

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\2282882155.py:58: UserWarning: The figure layout has changed to tight
      plt.tight_layout()

![png](pml_model_analysis_files/pml_model_analysis_24_1.png)

## 10. Leverage & Liquidity Analysis

```python
lev_cols = [c for c in FEATURE_CATEGORIES.get("Leverage & Liquidity", []) if c in df.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Leverage & Liquidity Analysis", fontsize=16, fontweight="bold")

# Debt-to-Equity distribution
ax = axes[0]
dte_col = next((c for c in ["debt_to_equity"] if c in df.columns), None)
if dte_col:
    dte_data = df[dte_col].dropna().clip(-1, 10)
    ax.hist(dte_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(dte_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: {dte_data.median():.2f}")
    ax.set_title("Debt-to-Equity Distribution", fontweight="bold")
    ax.set_xlabel("D/E Ratio")
    ax.legend()

# Current Ratio vs Quick Ratio
ax = axes[1]
cr_col = next((c for c in ["current_ratio"] if c in df.columns), None)
qr_col = next((c for c in ["quick_ratio"] if c in df.columns), None)
if cr_col and qr_col:
    valid = df[[cr_col, qr_col]].dropna()
    valid = valid[(valid[cr_col].between(0, 10)) & (valid[qr_col].between(0, 10))]
    ax.scatter(valid[cr_col], valid[qr_col], alpha=0.3, s=10, c=COLORS["info"])
    ax.plot([0, 10], [0, 10], "r--", lw=1, alpha=0.5, label="1:1 Line")
    ax.set_xlabel("Current Ratio")
    ax.set_ylabel("Quick Ratio")
    ax.set_title("Current vs Quick Ratio", fontweight="bold")
    ax.legend()

# Interest coverage distribution
ax = axes[2]
ic_col = next((c for c in ["interest_coverage"] if c in df.columns), None)
if ic_col:
    ic_data = df[ic_col].dropna().clip(-10, 100)
    ax.hist(ic_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(1.5, color=COLORS["danger"], linestyle="--", lw=2, label="Threshold (1.5x)")
    ax.set_title("Interest Coverage Distribution", fontweight="bold")
    ax.set_xlabel("Interest Coverage Ratio")
    ax.legend()

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_26_0.png)

## 11. Analyst Sentiment & Price Targets

```python
sent_cols = [c for c in FEATURE_CATEGORIES.get("Analyst Sentiment", []) if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle("Analyst Sentiment & Price Targets", fontsize=16, fontweight="bold")

# Analyst rating distribution
ax = axes[0, 0]
rating_col = next((c for c in ["analyst_rating"] if c in df.columns), None)
if rating_col:
    rating_data = df[rating_col].dropna()
    ax.hist(rating_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(rating_data.mean(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Mean: {rating_data.mean():.1f}")
    ax.set_title("Analyst Rating Distribution", fontweight="bold")
    ax.set_xlabel("Rating")
    ax.legend()

# Upside potential distribution
ax = axes[0, 1]
upside_col = next((c for c in ["upside_potential"] if c in df.columns), None)
if upside_col:
    upside_data = df[upside_col].dropna().clip(-80, 200)
    ax.hist(upside_data, bins=60, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Upside")
    ax.axvline(upside_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
               label=f"Median: {upside_data.median():.1f}%")
    ax.set_title("Upside Potential Distribution", fontweight="bold")
    ax.set_xlabel("Upside (%)")
    ax.legend()

# Bullish vs Bearish sentiment
ax = axes[1, 0]
bull_col = next((c for c in ["analyst_bullish_pct"] if c in df.columns), None)
bear_col = next((c for c in ["analyst_bearish_pct"] if c in df.columns), None)
if bull_col and bear_col:
    valid = df[[bull_col, bear_col]].dropna()
    ax.scatter(valid[bull_col], valid[bear_col], alpha=0.3, s=10, c=COLORS["info"])
    ax.set_xlabel("Bullish %")
    ax.set_ylabel("Bearish %")
    ax.set_title("Bullish vs Bearish Sentiment", fontweight="bold")
    ax.plot([0, 100], [100, 0], "r--", lw=1, alpha=0.3)

# EPS revision momentum by sector
ax = axes[1, 1]
rev_mom_col = next((c for c in ["eps_revision_momentum"] if c in df.columns), None)
sector_col = next((c for c in ["industry", "sector"] if c in df.columns), None)
if rev_mom_col and sector_col:
    sectors = df[sector_col].value_counts().head(10).index
    rev_data = [df.loc[df[sector_col] == s, rev_mom_col].dropna().clip(-2, 2).values for s in sectors]
    bp = ax.boxplot(rev_data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color=COLORS["danger"], linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    box_colors = plt.cm.Set2(np.linspace(0, 1, len(sectors)))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticklabels([s[:16] for s in sectors], rotation=55, ha="right", fontsize=8)
    ax.axhline(0, color="grey", linestyle=":", lw=0.8)
    ax.set_title("EPS Revision Momentum by Sector", fontweight="bold")

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_28_0.png)

## 12. Cash Flow Analysis

```python
cf_cols = [c for c in FEATURE_CATEGORIES.get("Cash Flow", []) if c in df.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Cash Flow Analysis", fontsize=16, fontweight="bold")

# FCF margin vs FCF yield scatter
ax = axes[0]
fcf_m = next((c for c in ["fcf_margin"] if c in df.columns), None)
fcf_y = next((c for c in ["fcf_yield"] if c in df.columns), None)
if fcf_m and fcf_y:
    valid = df[[fcf_m, fcf_y]].dropna()
    valid = valid[valid[fcf_m].between(-50, 60) & valid[fcf_y].between(-20, 30)]
    ax.scatter(valid[fcf_m], valid[fcf_y], alpha=0.3, s=10, c=COLORS["primary"])
    ax.set_xlabel("FCF Margin (%)")
    ax.set_ylabel("FCF Yield (%)")
    ax.set_title("FCF Margin vs Yield", fontweight="bold")
    ax.axhline(0, color="grey", linestyle=":", lw=0.8)
    ax.axvline(0, color="grey", linestyle=":", lw=0.8)

# Cash flow quality score
ax = axes[1]
cfq_col = next((c for c in ["cash_flow_quality_score"] if c in df.columns), None)
if cfq_col:
    cfq_data = df[cfq_col].dropna()
    ax.hist(cfq_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(cfq_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: {cfq_data.median():.2f}")
    ax.set_title("Cash Flow Quality Score", fontweight="bold")
    ax.legend()

# FCF positive years distribution
ax = axes[2]
fcf_pos = next((c for c in ["fcf_positive_years"] if c in df.columns), None)
if fcf_pos:
    pos_data = df[fcf_pos].dropna()
    counts = pos_data.value_counts().sort_index()
    bar_colors = [COLORS["danger"] if v <= 2 else COLORS["accent"] if v <= 4
    else COLORS["secondary"] for v in counts.index]
    ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
    ax.set_title("FCF Positive Years Distribution", fontweight="bold")
    ax.set_xlabel("Positive Years")
    ax.set_ylabel("Count")

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\3988772777.py:48: UserWarning: The figure layout has changed to tight
      plt.tight_layout()

![png](pml_model_analysis_files/pml_model_analysis_30_1.png)

## 13. Dividend Reliability Analysis

```python
div_cols = [c for c in FEATURE_CATEGORIES.get("Dividend Reliability", []) if c in df.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Dividend Reliability Analysis", fontsize=16, fontweight="bold")

# Dividend yield distribution
ax = axes[0]
dy_col = next((c for c in ["dividend_yield_ltm"] if c in df.columns), None)
if dy_col:
    dy_data = df[dy_col].dropna()
    ax.hist(dy_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(dy_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: {dy_data.median():.4f}%")
    ax.set_title("Dividend Yield Distribution (LTM)", fontweight="bold")
    ax.set_xlabel("Yield (%)")
    ax.legend()

# Dividend yield vs payout ratio
ax = axes[1]
payout_col = next((c for c in ["dividend_payout_ratio"] if c in df.columns), None)
if dy_col and payout_col:
    valid = df[[dy_col, payout_col]].dropna()
    valid = valid[(valid[dy_col] > 0) & valid[payout_col].between(0, 200)]
    ax.scatter(valid[dy_col], valid[payout_col], alpha=0.3, s=10, c=COLORS["info"])
    ax.axhline(100, color=COLORS["danger"], linestyle="--", lw=1, label="100% Payout")
    ax.set_xlabel("Dividend Yield (%)")
    ax.set_ylabel("Payout Ratio (%)")
    ax.set_title("Yield vs Payout Ratio", fontweight="bold")
    ax.legend()

# Dividend streak distribution
ax = axes[2]
streak_col = next((c for c in ["dividend_streak"] if c in df.columns), None)
if streak_col:
    streak_data = df[streak_col].dropna()
    streak_data = streak_data[streak_data > 0]
    ax.hist(streak_data, bins=30, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(streak_data.median(), color=COLORS["accent"], linestyle="--", lw=2,
               label=f"Median: {streak_data.median():.0f} years")
    ax.set_title("Dividend Streak Distribution", fontweight="bold")
    ax.set_xlabel("Consecutive Years")
    ax.legend()

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\758086935.py:48: UserWarning: The figure layout has changed to tight
      plt.tight_layout()

![png](pml_model_analysis_files/pml_model_analysis_32_1.png)

## 14. Volatility Surface Analysis (Enhancement 2+3)

```python
vol_cols = [c for c in FEATURE_CATEGORIES.get("Volatility Surface", []) if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle("Volatility Surface Analysis", fontsize=16, fontweight="bold")

# Volatility term structure (box plots)
ax = axes[0, 0]
vol_term_cols = [c for c in ["volatility_1m", "volatility_3m", "volatility_6m", "volatility_1y"]
                 if c in df.columns]
if vol_term_cols:
    vol_data = df[vol_term_cols].melt(var_name="Period", value_name="Volatility")
    vol_data["Volatility"] = vol_data["Volatility"].clip(0, 150)
    sns.boxplot(data=vol_data, x="Period", y="Volatility", ax=ax, palette="magma",
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_xticklabels([c.replace("volatility_", "") for c in vol_term_cols])
    ax.set_title("Volatility Term Structure", fontweight="bold")

# Vol ratio 3M/1Y distribution
ax = axes[0, 1]
vr_col = next((c for c in ["vol_ratio_3m_1y"] if c in df.columns), None)
if vr_col:
    vr_data = df[vr_col].dropna().clip(0, 5)
    ax.hist(vr_data, bins=50, color=COLORS["info"], alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color=COLORS["danger"], linestyle="--", lw=2, label="Ratio = 1.0")
    ax.set_title("Vol Ratio (3M/1Y)", fontweight="bold")
    ax.set_xlabel("Ratio")
    ax.legend()

# Beta term structure
ax = axes[1, 0]
beta_cols = [c for c in ["beta_2y", "beta_term_structure"] if c in df.columns]
if len(beta_cols) == 2:
    valid = df[beta_cols].dropna()
    valid = valid[valid[beta_cols[0]].between(-1, 4)]
    ax.scatter(valid[beta_cols[0]], valid[beta_cols[1]], alpha=0.3, s=10, c=COLORS["primary"])
    ax.set_xlabel("Beta (2Y)")
    ax.set_ylabel("Beta Term Structure")
    ax.set_title("Beta vs Term Structure", fontweight="bold")
    ax.axvline(1.0, color="grey", linestyle=":", lw=0.8)

# Volatility trend comparison
ax = axes[1, 1]
vt_short = next((c for c in ["volatility_trend_short"] if c in df.columns), None)
vt_long = next((c for c in ["volatility_trend_long"] if c in df.columns), None)
if vt_short and vt_long:
    valid = df[[vt_short, vt_long]].dropna()
    ax.hexbin(valid[vt_short], valid[vt_long], gridsize=30, cmap="YlOrRd", mincnt=1)
    ax.set_xlabel("Short-Term Vol Trend")
    ax.set_ylabel("Long-Term Vol Trend")
    ax.set_title("Vol Trend: Short vs Long", fontweight="bold")
    ax.axhline(0, color="grey", linestyle=":", lw=0.8)
    ax.axvline(0, color="grey", linestyle=":", lw=0.8)

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\1481445140.py:15: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax.set_xticklabels([c.replace("volatility_", "") for c in vol_term_cols])

![png](pml_model_analysis_files/pml_model_analysis_34_1.png)

## 15. Forward Consensus & Estimates (Enhancement 7)

```python
fwd_cols = [c for c in FEATURE_CATEGORIES.get("Forward Consensus", []) if c in df.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
fig.suptitle("Forward Consensus & Estimates", fontsize=16, fontweight="bold")

# Forward P/E discount distribution
ax = axes[0]
fpe_col = next((c for c in ["pe_forward_discount"] if c in df.columns), None)
if fpe_col:
    fpe_data = df[fpe_col].dropna().clip(-100, 100)
    ax.hist(fpe_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Discount")
    ax.axvline(fpe_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
               label=f"Median: {fpe_data.median():.1f}%")
    ax.set_title("Forward P/E Discount", fontweight="bold")
    ax.set_xlabel("Discount (%)")
    ax.legend()

# EBITDA forward growth
ax = axes[1]
ebitda_fwd = next((c for c in ["ebitda_forward_growth"] if c in df.columns), None)
if ebitda_fwd:
    ebitda_data = df[ebitda_fwd].dropna().clip(-100, 300)
    ax.hist(ebitda_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.axvline(ebitda_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
               label=f"Median: {ebitda_data.median():.1f}%")
    ax.set_title("EBITDA Forward Growth", fontweight="bold")
    ax.set_xlabel("Growth (%)")
    ax.legend()

# Forward consensus correlation
ax = axes[2]
if len(fwd_cols) >= 3:
    fwd_data = df[fwd_cols].dropna(thresh=len(fwd_cols) // 2)
    if len(fwd_data) > 10:
        corr = fwd_data.corr()
        sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax,
                    xticklabels=[c[:14] for c in corr.columns],
                    yticklabels=[c[:14] for c in corr.columns], square=True)
        ax.set_title("Forward Consensus Correlations", fontweight="bold")

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_36_0.png)

## 16. Enhancement Features: Tax, OpEx, FCF Estimates, Share Dilution

```python
# Tax Rate Features (Enhancement 4)
tax_cols = [c for c in FEATURE_CATEGORIES.get("Tax Rate", []) if c in df.columns]
# OpEx Temporal (Enhancement 5)
opex_cols = [c for c in FEATURE_CATEGORIES.get("OpEx Temporal", []) if c in df.columns]
# FCF Estimates (Enhancement 9)
fcf_est_cols = [c for c in FEATURE_CATEGORIES.get("FCF Estimates", []) if c in df.columns]
# Share Dilution (Enhancement 12)
dilution_cols = [c for c in FEATURE_CATEGORIES.get("Share Dilution", []) if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle("Enhancement Features Overview", fontsize=16, fontweight="bold")

# Tax rate distribution
ax = axes[0, 0]
tax_col = next((c for c in ["effective_tax_rate_fy"] if c in df.columns), None)
if tax_col:
    tax_data = df[tax_col].dropna().clip(-20, 60) * 100
    ax.hist(tax_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(21, color=COLORS["danger"], linestyle="--", lw=2, label="US Corp Rate (21%)")
    ax.axvline(tax_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
               label=f"Median: {tax_data.median():.1f}%")
    ax.set_title("Effective Tax Rate (FY)", fontweight="bold")
    ax.set_xlabel("Tax Rate (%)")
    ax.legend()

# OpEx operating leverage
ax = axes[0, 1]
olev_col = next((c for c in ["operating_leverage_score"] if c in df.columns), None)
if olev_col:
    olev_data = df[olev_col].dropna()
    ax.hist(olev_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(olev_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: {olev_data.median():.2f}")
    ax.set_title("Operating Leverage Score", fontweight="bold")
    ax.legend()

# FCF estimate curve
ax = axes[1, 0]
fcf_fy_cols = [c for c in ["fcf_est_avg_fy1e", "fcf_est_avg_fy2e", "fcf_est_avg_fy3e",
                           "fcf_est_avg_fy4e", "fcf_est_avg_fy5e"] if c in df.columns]
if fcf_fy_cols:
    medians = [df[c].dropna().median() for c in fcf_fy_cols]
    q25 = [df[c].dropna().quantile(0.25) for c in fcf_fy_cols]
    q75 = [df[c].dropna().quantile(0.75) for c in fcf_fy_cols]
    x = range(len(fcf_fy_cols))
    ax.fill_between(x, q25, q75, alpha=0.3, color=COLORS["info"])
    ax.plot(x, medians, "o-", color=COLORS["info"], lw=2, label="Median")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["FY1E", "FY2E", "FY3E", "FY4E", "FY5E"])
    ax.set_title("FCF Estimate Curve (Median ± IQR)", fontweight="bold")
    ax.set_ylabel("FCF Estimate ($M)")
    ax.legend()

# Share dilution tracking
ax = axes[1, 1]
shares_chg = next((c for c in ["shares_yoy_change_pct"] if c in df.columns), None)
if shares_chg:
    chg_data = df[shares_chg].dropna().clip(-30, 30)
    colors_hist = [COLORS["secondary"] if v < 0 else COLORS["danger"] for v in
                   np.histogram_bin_edges(chg_data, bins=50)[:-1]]
    ax.hist(chg_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="No Change")
    ax.axvline(chg_data.median(), color=COLORS["accent"], linestyle="-", lw=2,
               label=f"Median: {chg_data.median():.2f}%")
    ax.set_title("Share Count YoY Change", fontweight="bold")
    ax.set_xlabel("Change (%)")
    ax.legend()

for a in axes.flat:
    a.spines["top"].set_visible(True)
    a.spines["right"].set_visible(True)
    a.grid(axis="y", alpha=0.2)
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_38_0.png)

## 17. Dividend History & Asset Sales (Enhancements 8, 10)

```python
div_hist_cols = [c for c in FEATURE_CATEGORIES.get("Dividend History", []) if c in df.columns]
asset_cols = [c for c in FEATURE_CATEGORIES.get("Asset Sales", []) if c in df.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Dividend History & Asset Sales", fontsize=16, fontweight="bold")

# Dividend yield history trend
ax = axes[0]
dy_hist_cols = [c for c in ["div_yield_2fyind", "div_yield_3fyind", "div_yield_4fyind",
                            "div_yield_5fyind"] if c in df.columns]
if dy_hist_cols:
    medians = [df[c].dropna().median() for c in dy_hist_cols]
    ax.plot(range(len(dy_hist_cols)), medians, "o-", color=COLORS["secondary"], lw=2, markersize=8)
    ax.set_xticks(range(len(dy_hist_cols)))
    ax.set_xticklabels(["2FY", "3FY", "4FY", "5FY"])
    ax.set_title("Historical Dividend Yield (Median)", fontweight="bold")
    ax.set_ylabel("Yield (%)")
    ax.set_xlabel("Fiscal Year (lookback)")

# Dividend yield stability
ax = axes[1]
stab_col = next((c for c in ["div_yield_stability"] if c in df.columns), None)
if stab_col:
    stab_data = df[stab_col].dropna()
    ax.hist(stab_data, bins=50, color=COLORS["info"], alpha=0.85, edgecolor="white")
    ax.axvline(stab_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: {stab_data.median():.2f}")
    ax.set_title("Dividend Yield Stability", fontweight="bold")
    ax.legend()

# Asset sale frequency
ax = axes[2]
asf_col = next((c for c in ["asset_sale_frequency"] if c in df.columns), None)
if asf_col:
    asf_data = df[asf_col].dropna()
    ax.hist(asf_data, bins=30, color=COLORS["light"], alpha=0.85, edgecolor="white")
    ax.set_title("Asset Sale Frequency", fontweight="bold")
    ax.set_xlabel("Frequency")

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\236371961.py:44: UserWarning: The figure layout has changed to tight
      plt.tight_layout()

![png](pml_model_analysis_files/pml_model_analysis_40_1.png)

## 18. Interest Income & Employee Productivity (Enhancements 11)

```python
int_cols = [c for c in FEATURE_CATEGORIES.get("Interest Income Temporal", []) if c in df.columns]
emp_cols = [c for c in FEATURE_CATEGORIES.get("Employee Productivity", []) if c in df.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
fig.suptitle("Interest Income & Employee Productivity", fontsize=16, fontweight="bold")
fig.suptitle("Interest Income & Employee Productivity", fontsize=16, fontweight="bold")

# Interest income to revenue trend
ax = axes[0]
ii_col = next((c for c in ["interest_income_to_revenue_trend"] if c in df.columns), None)
if ii_col:
    ii_data = df[ii_col].dropna().clip(-5, 5)
    ax.hist(ii_data, bins=50, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title("Interest Income / Revenue Trend", fontweight="bold")
    ax.set_xlabel("Trend")

# Revenue per employee
ax = axes[1]
rpe_col = next((c for c in ["revenue_per_employee"] if c in df.columns), None)
if rpe_col:
    rpe_data = df[rpe_col].dropna().clip(0, df[rpe_col].dropna().quantile(0.95))
    ax.hist(rpe_data, bins=50, color=COLORS["secondary"], alpha=0.85, edgecolor="white")
    ax.axvline(rpe_data.median(), color=COLORS["danger"], linestyle="--", lw=2,
               label=f"Median: ${rpe_data.median():,.0f}")
    ax.set_title("Revenue per Employee", fontweight="bold")
    ax.set_xlabel("Revenue ($)")
    ax.legend()

# Productivity trend by sector
ax = axes[2]
prod_col = next((c for c in ["productivity_trend"] if c in df.columns), None)
sector_col = next((c for c in ["industry", "sector"] if c in df.columns), None)
if prod_col and sector_col:
    sectors = df[sector_col].value_counts().head(10).index
    prod_data = [df.loc[df[sector_col] == s, prod_col].dropna().values for s in sectors]
    bp = ax.boxplot(prod_data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color=COLORS["danger"], linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    box_colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticklabels([s[:16] for s in sectors], rotation=55, ha="right", fontsize=8)
    ax.set_title("Productivity Trend by Sector", fontweight="bold")

for a in axes.flat:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_42_0.png)

## 19. Cross-Category Correlation Analysis

```python
# Select one representative feature per category for the master correlation matrix
representative_features = {
    "P/E Ratio": "p_e_ratio",
    "ROE": "roe",
    "Revenue Growth": "revenue_growth_yoy",
    "F-Score": "piotroski_f_score",
    "D/E Ratio": "debt_to_equity",
    "FCF Margin": "fcf_margin",
    "Div Yield": "dividend_yield_ltm",
    "Analyst Rating": "analyst_rating_normalized",
    "EPS Surprise": "eps_surprise_pct",
    "Momentum 1Y": "price_momentum_1y",
    "Volatility 1Y": "volatility_1y",
    "Z-Score": "altman_z_score",
    "Tax Rate": "effective_tax_rate_fy",
    "Upside %": "upside_potential",
    "Beta": "beta_2y",
}

available_rep = {k: v for k, v in representative_features.items() if v in df.columns}
rep_df = df[list(available_rep.values())].dropna(thresh=len(available_rep) // 2)

fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=False)
corr = rep_df.corr()
corr.columns = list(available_rep.keys())
corr.index = list(available_rep.keys())
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            ax=ax, square=False, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
ax.set_title("Cross-Category Feature Correlations", fontsize=8, fontweight="bold")
plt.show()
```

![png](pml_model_analysis_files/pml_model_analysis_44_0.png)

## 20. Stock Screening Results

```python
# Enhanced screener: high-quality stocks
quality_stocks = create_enhanced_screener(df, min_fscore=7)
print(f"Enhanced Screener (F-Score ≥ 7): {len(quality_stocks)} stocks")
if len(quality_stocks) > 0:
    display_cols = [c for c in
                    ["ticker", "name", "industry", "piotroski_f_score", "last_price", "price_target", 'p_e_ratio',
                     'peg_ratio', "pe_forward_discount", "forward_pe_premium", "p_e_momentum_yoy", "altman_z_score",
                     "upside_potential", "eps_trajectory_score"] if c in quality_stocks.columns]
    display(quality_stocks[display_cols].head(50))
```

    Enhanced Screener (F-Score ≥ 7): 29 stocks

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>industry</th>
      <th>piotroski_f_score</th>
      <th>last_price</th>
      <th>price_target</th>
      <th>p_e_ratio</th>
      <th>peg_ratio</th>
      <th>pe_forward_discount</th>
      <th>forward_pe_premium</th>
      <th>p_e_momentum_yoy</th>
      <th>altman_z_score</th>
      <th>upside_potential</th>
      <th>eps_trajectory_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>LRCX</td>
      <td>Lam Research Corporation</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>9</td>
      <td>258.7380</td>
      <td>301.5806</td>
      <td>62.3000</td>
      <td>7.5777</td>
      <td>-0.4398</td>
      <td>-26.8058</td>
      <td>1.5325</td>
      <td>NaN</td>
      <td>19.8123</td>
      <td>80</td>
    </tr>
    <tr>
      <th>670</th>
      <td>RL</td>
      <td>Ralph Lauren Corporation</td>
      <td>Textiles Apparel and Luxury Goods</td>
      <td>9</td>
      <td>371.2480</td>
      <td>411.2625</td>
      <td>32.0000</td>
      <td>2.4559</td>
      <td>-0.3406</td>
      <td>-28.7500</td>
      <td>0.3389</td>
      <td>5.4000</td>
      <td>14.4787</td>
      <td>60</td>
    </tr>
    <tr>
      <th>5310</th>
      <td>7747</td>
      <td>Asahi Intecc Co. Ltd.</td>
      <td>Health Care Equipment and Supplies</td>
      <td>9</td>
      <td>3317.0000</td>
      <td>3829.3333</td>
      <td>63.8000</td>
      <td>14.4962</td>
      <td>-0.5533</td>
      <td>-53.2915</td>
      <td>0.4084</td>
      <td>13.2200</td>
      <td>20.5909</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>4062</td>
      <td>Ibiden Co.Ltd.</td>
      <td>Electronic Equipment Instruments and Components</td>
      <td>9</td>
      <td>11135.0000</td>
      <td>8864.1177</td>
      <td>92.5000</td>
      <td>-7.1799</td>
      <td>-0.4541</td>
      <td>-20.1081</td>
      <td>2.2918</td>
      <td>2.4100</td>
      <td>-17.3776</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4074</th>
      <td>6857</td>
      <td>Advantest Corporation</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>9</td>
      <td>27900.0000</td>
      <td>29210.0000</td>
      <td>120.3000</td>
      <td>7.2287</td>
      <td>-0.6633</td>
      <td>-49.7091</td>
      <td>0.6279</td>
      <td>18.7200</td>
      <td>7.5269</td>
      <td>80</td>
    </tr>
    <tr>
      <th>830</th>
      <td>NDSN</td>
      <td>Nordson Corporation</td>
      <td>Machinery</td>
      <td>8</td>
      <td>283.7510</td>
      <td>308.4286</td>
      <td>33.3000</td>
      <td>-25.8144</td>
      <td>-0.2613</td>
      <td>-25.5255</td>
      <td>0.1893</td>
      <td>4.7800</td>
      <td>14.5370</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2610</th>
      <td>RYA</td>
      <td>Ryanair Holdings plc</td>
      <td>Passenger Airlines</td>
      <td>8</td>
      <td>23.4800</td>
      <td>31.8210</td>
      <td>17.5000</td>
      <td>NaN</td>
      <td>-0.4057</td>
      <td>-35.4286</td>
      <td>0.0870</td>
      <td>4.0200</td>
      <td>36.2862</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4548</th>
      <td>9766</td>
      <td>Konami Group Corporation</td>
      <td>Entertainment</td>
      <td>8</td>
      <td>20180.0000</td>
      <td>25410.5882</td>
      <td>34.4000</td>
      <td>11.9303</td>
      <td>-0.2703</td>
      <td>-16.2791</td>
      <td>0.3080</td>
      <td>12.0300</td>
      <td>28.3449</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4413</th>
      <td>002545</td>
      <td>Qingdao East Steel Tower Stock Co.Ltd</td>
      <td>Metals and Mining</td>
      <td>8</td>
      <td>21.8500</td>
      <td>20.0000</td>
      <td>51.5000</td>
      <td>8.2191</td>
      <td>-0.5495</td>
      <td>-54.9515</td>
      <td>2.1402</td>
      <td>2.5400</td>
      <td>-8.4668</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2112</th>
      <td>NSSC</td>
      <td>Napco Security Technologies Inc.</td>
      <td>Electronic Equipment Instruments and Components</td>
      <td>8</td>
      <td>44.8750</td>
      <td>49.6667</td>
      <td>37.7000</td>
      <td>1.2040</td>
      <td>-0.2175</td>
      <td>-19.0981</td>
      <td>0.3322</td>
      <td>29.2600</td>
      <td>10.3064</td>
      <td>80</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MU</td>
      <td>Micron Technology Inc.</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>8</td>
      <td>482.0450</td>
      <td>533.7250</td>
      <td>63.5000</td>
      <td>-92.3499</td>
      <td>-0.9181</td>
      <td>-86.9291</td>
      <td>1.5709</td>
      <td>6.9400</td>
      <td>13.3712</td>
      <td>80</td>
    </tr>
    <tr>
      <th>706</th>
      <td>CRS</td>
      <td>Carpenter Technology Corporation</td>
      <td>Aerospace and Defense</td>
      <td>8</td>
      <td>426.3440</td>
      <td>439.3333</td>
      <td>57.5000</td>
      <td>NaN</td>
      <td>-0.3287</td>
      <td>-28.5217</td>
      <td>0.6571</td>
      <td>7.4000</td>
      <td>6.2522</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>CRUS</td>
      <td>Cirrus Logic Inc.</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>8</td>
      <td>172.8410</td>
      <td>160.0000</td>
      <td>28.8000</td>
      <td>9.4022</td>
      <td>-0.3507</td>
      <td>-33.6806</td>
      <td>0.7349</td>
      <td>11.7200</td>
      <td>-3.0901</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1508</th>
      <td>PSMT</td>
      <td>PriceSmart Inc.</td>
      <td>Consumer Staples Distribution and Retail</td>
      <td>7</td>
      <td>162.1540</td>
      <td>153.3333</td>
      <td>33.6000</td>
      <td>2.6756</td>
      <td>-0.2292</td>
      <td>-11.9048</td>
      <td>0.3176</td>
      <td>5.2000</td>
      <td>-1.3284</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>PSK</td>
      <td>PrairieSky Royalty Ltd.</td>
      <td>Oil Gas and Consumable Fuels</td>
      <td>7</td>
      <td>32.7700</td>
      <td>35.0227</td>
      <td>37.7000</td>
      <td>-2.8474</td>
      <td>-0.2334</td>
      <td>-26.2599</td>
      <td>0.0833</td>
      <td>5.8600</td>
      <td>8.3308</td>
      <td>60</td>
    </tr>
    <tr>
      <th>300</th>
      <td>UI</td>
      <td>Ubiquiti Inc.</td>
      <td>Communications Equipment</td>
      <td>7</td>
      <td>1038.8780</td>
      <td>753.5000</td>
      <td>88.3000</td>
      <td>3.6453</td>
      <td>-0.3330</td>
      <td>-26.9536</td>
      <td>1.0346</td>
      <td>33.9400</td>
      <td>-27.4698</td>
      <td>60</td>
    </tr>
    <tr>
      <th>803</th>
      <td>IGV</td>
      <td>I Grandi Viaggi S.p.A.</td>
      <td>Hotels Restaurants and Leisure</td>
      <td>7</td>
      <td>2.3600</td>
      <td>2.3000</td>
      <td>34.2000</td>
      <td>NaN</td>
      <td>-0.0117</td>
      <td>-1.1696</td>
      <td>0.8791</td>
      <td>3.1900</td>
      <td>-2.5424</td>
      <td>100</td>
    </tr>
    <tr>
      <th>165</th>
      <td>MCK</td>
      <td>McKesson Corporation</td>
      <td>Health Care Providers and Services</td>
      <td>7</td>
      <td>836.0540</td>
      <td>990.8667</td>
      <td>32.5000</td>
      <td>0.6206</td>
      <td>-0.4062</td>
      <td>-34.1538</td>
      <td>0.2037</td>
      <td>5.7900</td>
      <td>19.6095</td>
      <td>60</td>
    </tr>
    <tr>
      <th>92</th>
      <td>DE</td>
      <td>Deere &amp; Company</td>
      <td>Machinery</td>
      <td>7</td>
      <td>592.2040</td>
      <td>665.0996</td>
      <td>32.0000</td>
      <td>-4.2897</td>
      <td>-0.0156</td>
      <td>3.4375</td>
      <td>0.2451</td>
      <td>2.9700</td>
      <td>11.4481</td>
      <td>60</td>
    </tr>
    <tr>
      <th>199</th>
      <td>JCI</td>
      <td>Johnson Controls International plc</td>
      <td>Building Products</td>
      <td>7</td>
      <td>141.7520</td>
      <td>144.0952</td>
      <td>53.9000</td>
      <td>1.6889</td>
      <td>-0.4675</td>
      <td>-44.7124</td>
      <td>0.1951</td>
      <td>2.4900</td>
      <td>2.9968</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1610</th>
      <td>GFF</td>
      <td>Griffon Corporation</td>
      <td>Building Products</td>
      <td>7</td>
      <td>92.3470</td>
      <td>114.1429</td>
      <td>84.7000</td>
      <td>NaN</td>
      <td>-0.7922</td>
      <td>-79.2208</td>
      <td>0.3339</td>
      <td>3.7100</td>
      <td>24.5303</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4066</th>
      <td>9983</td>
      <td>Fast Retailing Co. Ltd.</td>
      <td>Specialty Retail</td>
      <td>7</td>
      <td>69290.0000</td>
      <td>71264.2857</td>
      <td>45.3000</td>
      <td>3.1563</td>
      <td>-0.0795</td>
      <td>-3.3113</td>
      <td>0.1185</td>
      <td>8.6000</td>
      <td>3.1895</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3297</th>
      <td>CFX</td>
      <td>Colefax Group PLC</td>
      <td>Household Durables</td>
      <td>7</td>
      <td>11.8000</td>
      <td>10.5000</td>
      <td>11.0000</td>
      <td>2.9453</td>
      <td>0.0091</td>
      <td>0.9091</td>
      <td>0.5714</td>
      <td>3.3900</td>
      <td>-11.0169</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>VAR</td>
      <td>Vår Energi ASA</td>
      <td>Oil Gas and Consumable Fuels</td>
      <td>7</td>
      <td>45.0600</td>
      <td>46.9648</td>
      <td>15.5000</td>
      <td>-2.3623</td>
      <td>-0.4000</td>
      <td>-43.8710</td>
      <td>0.0197</td>
      <td>NaN</td>
      <td>8.6261</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2648</th>
      <td>ALFA</td>
      <td>Alfa Laval AB (publ)</td>
      <td>Machinery</td>
      <td>7</td>
      <td>545.8000</td>
      <td>542.8823</td>
      <td>27.2000</td>
      <td>0.9790</td>
      <td>-0.0919</td>
      <td>-9.1912</td>
      <td>0.0462</td>
      <td>NaN</td>
      <td>0.7695</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4492</th>
      <td>002318</td>
      <td>Zhejiang JIULI Hi-tech Metals Co. Ltd</td>
      <td>Metals and Mining</td>
      <td>7</td>
      <td>28.7500</td>
      <td>33.6500</td>
      <td>19.8000</td>
      <td>1.1422</td>
      <td>-0.1717</td>
      <td>-17.1717</td>
      <td>0.2532</td>
      <td>5.0600</td>
      <td>17.0435</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>300502</td>
      <td>Eoptolink Technology Inc. Ltd.</td>
      <td>Electronic Equipment Instruments and Components</td>
      <td>7</td>
      <td>608.2800</td>
      <td>472.9586</td>
      <td>227.6000</td>
      <td>4.3366</td>
      <td>-0.8216</td>
      <td>-72.1002</td>
      <td>10.2118</td>
      <td>14.7900</td>
      <td>-20.9246</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4834</th>
      <td>INDUSTOWER</td>
      <td>Indus Towers Limited</td>
      <td>Diversified Telecommunication Services</td>
      <td>7</td>
      <td>404.7000</td>
      <td>454.9583</td>
      <td>9.9000</td>
      <td>0.7995</td>
      <td>0.4747</td>
      <td>50.5051</td>
      <td>0.0879</td>
      <td>NaN</td>
      <td>14.6528</td>
      <td>60</td>
    </tr>
    <tr>
      <th>5754</th>
      <td>4626</td>
      <td>Taiyo Holdings Co. Ltd.</td>
      <td>Chemicals</td>
      <td>7</td>
      <td>4745.0000</td>
      <td>5000.0000</td>
      <td>46.2000</td>
      <td>-4.9257</td>
      <td>-0.5000</td>
      <td>-46.1039</td>
      <td>1.2319</td>
      <td>4.6300</td>
      <td>5.3741</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Value opportunities
value_stocks = screen_value_opportunities(df)
print(f"\nValue Opportunities: {len(value_stocks)} stocks")
if len(value_stocks) > 0:
    display_cols = [c for c in
                    ["ticker", "name", "p_e_ratio", "pe_forward_discount", "forward_pe_premium", "p_e_momentum_yoy",
                     "upside_potential",
                     "price_to_tangible_book"] if c in value_stocks.columns]
    display(value_stocks[display_cols].head(50))
```

    Value Opportunities: 1427 stocks

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>p_e_ratio</th>
      <th>pe_forward_discount</th>
      <th>forward_pe_premium</th>
      <th>p_e_momentum_yoy</th>
      <th>upside_potential</th>
      <th>price_to_tangible_book</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>CANTA</td>
      <td>Cantargia AB (publ)</td>
      <td>6.5000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.0714</td>
      <td>197.9343</td>
      <td>3.6000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CVX</td>
      <td>Chevron Corporation</td>
      <td>28.3000</td>
      <td>-0.4311</td>
      <td>-43.1095</td>
      <td>0.0107</td>
      <td>15.1410</td>
      <td>2.1000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>BERGER</td>
      <td>Berger Paints Nigeria Plc</td>
      <td>13.6000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.0286</td>
      <td>-76.8116</td>
      <td>4.4000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>BABA</td>
      <td>Alibaba Group Holding Limited</td>
      <td>17.8000</td>
      <td>0.1292</td>
      <td>48.3146</td>
      <td>0.2109</td>
      <td>41.7480</td>
      <td>2.7000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SOTET</td>
      <td>Société Tunisienne d'Entreprises de Télécommun...</td>
      <td>10.3000</td>
      <td>3.0194</td>
      <td>301.9417</td>
      <td>0.0729</td>
      <td>-2.4155</td>
      <td>1.3000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TSM1T</td>
      <td>AS Tallinna Sadam</td>
      <td>16.8000</td>
      <td>-0.0476</td>
      <td>-4.7619</td>
      <td>0.0244</td>
      <td>6.3019</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>60</th>
      <td>FAA</td>
      <td>Fabasoft AG</td>
      <td>16.1000</td>
      <td>-0.0248</td>
      <td>-2.4845</td>
      <td>-0.2370</td>
      <td>109.2050</td>
      <td>3.7000</td>
    </tr>
    <tr>
      <th>73</th>
      <td>PQ</td>
      <td>Piquadro S.p.A.</td>
      <td>11.6000</td>
      <td>-0.0517</td>
      <td>-5.1724</td>
      <td>0.0642</td>
      <td>38.6364</td>
      <td>2.1000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>SNN</td>
      <td>S.N. Nuclearelectrica S.A.</td>
      <td>14.4000</td>
      <td>-0.3472</td>
      <td>-34.7222</td>
      <td>0.9459</td>
      <td>-32.8767</td>
      <td>1.7000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>MAB</td>
      <td>Mitchells &amp; Butlers plc</td>
      <td>8.9000</td>
      <td>-0.0674</td>
      <td>-6.7416</td>
      <td>-0.0111</td>
      <td>35.0575</td>
      <td>0.6000</td>
    </tr>
    <tr>
      <th>98</th>
      <td>COP</td>
      <td>ConocoPhillips</td>
      <td>19.6000</td>
      <td>-0.1684</td>
      <td>-16.8367</td>
      <td>0.1462</td>
      <td>12.5375</td>
      <td>2.4000</td>
    </tr>
    <tr>
      <th>105</th>
      <td>TUNE</td>
      <td>Focusrite plc</td>
      <td>20.5000</td>
      <td>NaN</td>
      <td>-49.7561</td>
      <td>-0.6595</td>
      <td>87.8378</td>
      <td>2.8000</td>
    </tr>
    <tr>
      <th>106</th>
      <td>CALN</td>
      <td>CALIDA Holding AG</td>
      <td>16.1000</td>
      <td>-0.2236</td>
      <td>-22.3602</td>
      <td>0.2677</td>
      <td>17.1393</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>115</th>
      <td>PDD</td>
      <td>PDD Holdings Inc.</td>
      <td>10.2000</td>
      <td>-0.2059</td>
      <td>-20.5882</td>
      <td>0.0303</td>
      <td>49.2628</td>
      <td>2.3000</td>
    </tr>
    <tr>
      <th>117</th>
      <td>MATX</td>
      <td>Matson Inc.</td>
      <td>12.3000</td>
      <td>0.0407</td>
      <td>4.0650</td>
      <td>0.0250</td>
      <td>26.3012</td>
      <td>2.3000</td>
    </tr>
    <tr>
      <th>119</th>
      <td>UNI</td>
      <td>UNIBEP S.A.</td>
      <td>12.5000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1161</td>
      <td>-1.1407</td>
      <td>2.8000</td>
    </tr>
    <tr>
      <th>134</th>
      <td>NEM</td>
      <td>Newmont Corporation</td>
      <td>17.4000</td>
      <td>-0.3161</td>
      <td>-31.6092</td>
      <td>-0.1122</td>
      <td>30.5612</td>
      <td>3.8000</td>
    </tr>
    <tr>
      <th>146</th>
      <td>DRIVE</td>
      <td>Emirates Driving Company P.J.S.C.</td>
      <td>9.9000</td>
      <td>-0.0606</td>
      <td>-6.0606</td>
      <td>-0.0388</td>
      <td>56.8627</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>149</th>
      <td>VTOL</td>
      <td>Bristow Group Inc.</td>
      <td>11.2000</td>
      <td>-0.2054</td>
      <td>-20.5357</td>
      <td>0.0090</td>
      <td>24.4839</td>
      <td>1.3000</td>
    </tr>
    <tr>
      <th>162</th>
      <td>MERS</td>
      <td>Al Meera Consumer Goods Company Q.P.S.C.</td>
      <td>20.4000</td>
      <td>0.0392</td>
      <td>3.9216</td>
      <td>-0.0192</td>
      <td>15.5743</td>
      <td>2.5000</td>
    </tr>
    <tr>
      <th>164</th>
      <td>UZU</td>
      <td>Uzin Utz SE</td>
      <td>14.3000</td>
      <td>0.0559</td>
      <td>5.5944</td>
      <td>0.0916</td>
      <td>38.9628</td>
      <td>1.5000</td>
    </tr>
    <tr>
      <th>167</th>
      <td>AEM</td>
      <td>Agnico Eagle Mines Limited</td>
      <td>22.5000</td>
      <td>-0.3422</td>
      <td>-34.2222</td>
      <td>-0.0302</td>
      <td>27.6709</td>
      <td>4.8000</td>
    </tr>
    <tr>
      <th>168</th>
      <td>SIM0</td>
      <td>SIMONA Aktiengesellschaft</td>
      <td>16.8000</td>
      <td>-0.3393</td>
      <td>-33.9286</td>
      <td>0.1915</td>
      <td>16.8224</td>
      <td>1.3000</td>
    </tr>
    <tr>
      <th>171</th>
      <td>HAKIB</td>
      <td>HAKI Safety AB (publ)</td>
      <td>17.2000</td>
      <td>-0.3953</td>
      <td>-44.1860</td>
      <td>0.0424</td>
      <td>45.0000</td>
      <td>2.6000</td>
    </tr>
    <tr>
      <th>175</th>
      <td>ARAMI</td>
      <td>Aramis Group SAS</td>
      <td>16.4000</td>
      <td>-0.0915</td>
      <td>-9.1463</td>
      <td>-0.1135</td>
      <td>51.5152</td>
      <td>3.7000</td>
    </tr>
    <tr>
      <th>181</th>
      <td>CNQ</td>
      <td>Canadian Natural Resources Limited</td>
      <td>12.1000</td>
      <td>-0.0496</td>
      <td>-4.9587</td>
      <td>-0.3632</td>
      <td>12.5040</td>
      <td>2.9000</td>
    </tr>
    <tr>
      <th>182</th>
      <td>FDX</td>
      <td>FedEx Corporation</td>
      <td>23.4000</td>
      <td>-0.2051</td>
      <td>-14.5299</td>
      <td>0.7463</td>
      <td>8.2113</td>
      <td>4.1000</td>
    </tr>
    <tr>
      <th>184</th>
      <td>CAPD</td>
      <td>Capital Limited</td>
      <td>4.9000</td>
      <td>1.4286</td>
      <td>140.8163</td>
      <td>0.0208</td>
      <td>45.4355</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>185</th>
      <td>IVU</td>
      <td>IVU Traffic Technologies AG</td>
      <td>24.2000</td>
      <td>-0.0950</td>
      <td>-9.5041</td>
      <td>-0.0242</td>
      <td>44.7721</td>
      <td>4.6000</td>
    </tr>
    <tr>
      <th>191</th>
      <td>PARR</td>
      <td>Par Pacific Holdings Inc.</td>
      <td>8.9000</td>
      <td>-0.3034</td>
      <td>-30.3371</td>
      <td>0.6481</td>
      <td>20.7105</td>
      <td>2.3000</td>
    </tr>
    <tr>
      <th>202</th>
      <td>MER</td>
      <td>Mears Group plc</td>
      <td>7.0000</td>
      <td>0.2000</td>
      <td>20.0000</td>
      <td>0.0000</td>
      <td>34.6103</td>
      <td>4.2000</td>
    </tr>
    <tr>
      <th>211</th>
      <td>EPD</td>
      <td>Enterprise Products Partners L.P.</td>
      <td>14.2000</td>
      <td>-0.0704</td>
      <td>-7.0423</td>
      <td>0.0441</td>
      <td>5.6608</td>
      <td>4.1000</td>
    </tr>
    <tr>
      <th>212</th>
      <td>BOI</td>
      <td>Boiron SA</td>
      <td>13.6000</td>
      <td>-0.0809</td>
      <td>-8.0882</td>
      <td>-0.4906</td>
      <td>-2.7237</td>
      <td>1.6000</td>
    </tr>
    <tr>
      <th>219</th>
      <td>DR0</td>
      <td>Deutsche Rohstoff AG</td>
      <td>10.1000</td>
      <td>-0.7426</td>
      <td>-74.2574</td>
      <td>2.4828</td>
      <td>40.9140</td>
      <td>2.4000</td>
    </tr>
    <tr>
      <th>220</th>
      <td>PMN</td>
      <td>Phoenix Mecano AG</td>
      <td>14.7000</td>
      <td>-0.1973</td>
      <td>-20.4082</td>
      <td>0.1308</td>
      <td>11.3564</td>
      <td>1.8000</td>
    </tr>
    <tr>
      <th>221</th>
      <td>REGN</td>
      <td>Regeneron Pharmaceuticals Inc.</td>
      <td>18.5000</td>
      <td>-0.0811</td>
      <td>-8.1081</td>
      <td>-0.0107</td>
      <td>14.5280</td>
      <td>2.6000</td>
    </tr>
    <tr>
      <th>224</th>
      <td>GEA</td>
      <td>Grenevia S.A.</td>
      <td>5.2000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1304</td>
      <td>6.2691</td>
      <td>0.7000</td>
    </tr>
    <tr>
      <th>227</th>
      <td>FORN</td>
      <td>Forbo Holding AG</td>
      <td>14.9000</td>
      <td>-0.1007</td>
      <td>-10.0671</td>
      <td>-0.1078</td>
      <td>20.8333</td>
      <td>1.8000</td>
    </tr>
    <tr>
      <th>231</th>
      <td>CP</td>
      <td>Canadian Pacific Kansas City Limited</td>
      <td>26.3000</td>
      <td>-0.1255</td>
      <td>-12.5475</td>
      <td>-0.0038</td>
      <td>5.3745</td>
      <td>4.3000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>SU</td>
      <td>Suncor Energy Inc.</td>
      <td>18.3000</td>
      <td>-0.4372</td>
      <td>-43.7158</td>
      <td>0.1656</td>
      <td>13.1350</td>
      <td>2.5000</td>
    </tr>
    <tr>
      <th>249</th>
      <td>ALGIL</td>
      <td>Groupe Guillin S.A.</td>
      <td>8.2000</td>
      <td>0.0244</td>
      <td>2.4390</td>
      <td>-0.0787</td>
      <td>28.2051</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>250</th>
      <td>SFL</td>
      <td>Safilo Group S.p.A.</td>
      <td>15.1000</td>
      <td>0.0331</td>
      <td>3.3113</td>
      <td>0.0342</td>
      <td>13.4430</td>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>252</th>
      <td>NSC</td>
      <td>Norfolk Southern Corporation</td>
      <td>25.2000</td>
      <td>0.0635</td>
      <td>6.3492</td>
      <td>0.0500</td>
      <td>-2.6654</td>
      <td>4.6000</td>
    </tr>
    <tr>
      <th>254</th>
      <td>FERGR</td>
      <td>Ferrari Group PLC</td>
      <td>14.4000</td>
      <td>-0.1111</td>
      <td>-11.1111</td>
      <td>0.1077</td>
      <td>43.7500</td>
      <td>3.6000</td>
    </tr>
    <tr>
      <th>260</th>
      <td>EOG</td>
      <td>EOG Resources Inc.</td>
      <td>14.7000</td>
      <td>-0.3810</td>
      <td>-38.0952</td>
      <td>0.0809</td>
      <td>15.8021</td>
      <td>2.4000</td>
    </tr>
    <tr>
      <th>263</th>
      <td>HAW</td>
      <td>Hawesko Holding SE</td>
      <td>15.5000</td>
      <td>0.1097</td>
      <td>20.6452</td>
      <td>-0.0936</td>
      <td>36.0294</td>
      <td>2.8000</td>
    </tr>
    <tr>
      <th>264</th>
      <td>GM</td>
      <td>General Motors Company</td>
      <td>24.0000</td>
      <td>-0.7333</td>
      <td>-73.3333</td>
      <td>-0.0909</td>
      <td>22.2930</td>
      <td>1.2000</td>
    </tr>
    <tr>
      <th>269</th>
      <td>ICP1V</td>
      <td>Incap Oyj</td>
      <td>22.6000</td>
      <td>-0.4204</td>
      <td>-42.0354</td>
      <td>-0.0088</td>
      <td>22.1805</td>
      <td>2.6000</td>
    </tr>
    <tr>
      <th>271</th>
      <td>CNR</td>
      <td>Canadian National Railway Company</td>
      <td>20.7000</td>
      <td>-0.0386</td>
      <td>-3.8647</td>
      <td>0.1564</td>
      <td>0.5443</td>
      <td>4.9000</td>
    </tr>
    <tr>
      <th>272</th>
      <td>ABX</td>
      <td>Barrick Mining Corporation</td>
      <td>13.8000</td>
      <td>-0.2174</td>
      <td>-21.7391</td>
      <td>-0.0800</td>
      <td>36.6541</td>
      <td>2.9000</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Growth momentum stocks
growth_stocks = screen_growth_momentum(df)
print(f"\nGrowth Momentum: {len(growth_stocks)} stocks")
if len(growth_stocks) > 5:
    display_cols = [c for c in ["ticker", "name", "revenue_growth_yoy",
                                "eps_yoy_growth", "price_momentum_1y", "eps_momentum_yoy"] if
                    c in growth_stocks.columns]
    display(growth_stocks[display_cols].head(50))
```

    Growth Momentum: 2831 stocks

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>revenue_growth_yoy</th>
      <th>eps_yoy_growth</th>
      <th>price_momentum_1y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6447</th>
      <td>TRAN</td>
      <td>Compañía de Transporte de Energía Eléctrica en...</td>
      <td>19.6933</td>
      <td>100.0000</td>
      <td>70.6278</td>
    </tr>
    <tr>
      <th>4121</th>
      <td>5803</td>
      <td>Fujikura Ltd.</td>
      <td>23.5968</td>
      <td>85.0000</td>
      <td>652.9386</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>AAOI</td>
      <td>Applied Optoelectronics Inc.</td>
      <td>82.7445</td>
      <td>85.7778</td>
      <td>1080.5317</td>
    </tr>
    <tr>
      <th>5422</th>
      <td>6442</td>
      <td>EZconn Corporation</td>
      <td>71.8979</td>
      <td>70.4545</td>
      <td>493.2452</td>
    </tr>
    <tr>
      <th>4457</th>
      <td>A298040</td>
      <td>Hyosung Heavy Industries Corporation</td>
      <td>24.7290</td>
      <td>138.9129</td>
      <td>641.8842</td>
    </tr>
    <tr>
      <th>121</th>
      <td>CLS</td>
      <td>Celestica Inc.</td>
      <td>28.4564</td>
      <td>99.4475</td>
      <td>338.5746</td>
    </tr>
    <tr>
      <th>4649</th>
      <td>6223</td>
      <td>MPI Corporation</td>
      <td>37.5969</td>
      <td>44.5946</td>
      <td>681.4029</td>
    </tr>
    <tr>
      <th>4340</th>
      <td>A267260</td>
      <td>HD Hyundai Electric Co. Ltd.</td>
      <td>25.6073</td>
      <td>49.3644</td>
      <td>279.4958</td>
    </tr>
    <tr>
      <th>4795</th>
      <td>522275</td>
      <td>GE Vernova T&amp;D India Limited</td>
      <td>32.1497</td>
      <td>250.0000</td>
      <td>195.6640</td>
    </tr>
    <tr>
      <th>953</th>
      <td>TNZ</td>
      <td>Tenaz Energy Corp.</td>
      <td>415.5915</td>
      <td>4170.0000</td>
      <td>297.8111</td>
    </tr>
    <tr>
      <th>4061</th>
      <td>300308</td>
      <td>Zhongji Innolight Co. Ltd.</td>
      <td>67.2633</td>
      <td>115.3846</td>
      <td>973.9943</td>
    </tr>
    <tr>
      <th>4252</th>
      <td>6869</td>
      <td>Yangtze Optical Fibre And Cable Joint Stock Li...</td>
      <td>21.9561</td>
      <td>25.0000</td>
      <td>1651.8797</td>
    </tr>
    <tr>
      <th>2492</th>
      <td>3363</td>
      <td>FOCI Fiber Optic Communications Inc.</td>
      <td>45.2180</td>
      <td>100.0000</td>
      <td>471.1599</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>300502</td>
      <td>Eoptolink Technology Inc. Ltd.</td>
      <td>171.4495</td>
      <td>290.0000</td>
      <td>832.4342</td>
    </tr>
    <tr>
      <th>6101</th>
      <td>ASELS</td>
      <td>ASELSAN Elektronik Sanayi ve Ticaret Anonim Si...</td>
      <td>23.7036</td>
      <td>66.6667</td>
      <td>201.0630</td>
    </tr>
    <tr>
      <th>6258</th>
      <td>CEPU</td>
      <td>Central Puerto S.A.</td>
      <td>5.5841</td>
      <td>300.0000</td>
      <td>58.7454</td>
    </tr>
    <tr>
      <th>4794</th>
      <td>6515</td>
      <td>WinWay Technology Co. Ltd.</td>
      <td>41.8404</td>
      <td>44.2308</td>
      <td>1203.7667</td>
    </tr>
    <tr>
      <th>4259</th>
      <td>3017</td>
      <td>Asia Vital Components Co. Ltd.</td>
      <td>103.6833</td>
      <td>141.5385</td>
      <td>526.1682</td>
    </tr>
    <tr>
      <th>152</th>
      <td>CVNA</td>
      <td>Carvana Co.</td>
      <td>48.6287</td>
      <td>494.1860</td>
      <td>82.4052</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>3081</td>
      <td>LandMark Optoelectronics Corporation</td>
      <td>90.8102</td>
      <td>850.0000</td>
      <td>1108.0000</td>
    </tr>
    <tr>
      <th>4174</th>
      <td>2383</td>
      <td>Elite Material Co. Ltd.</td>
      <td>53.2637</td>
      <td>56.4706</td>
      <td>685.4406</td>
    </tr>
    <tr>
      <th>607</th>
      <td>FTC</td>
      <td>Filtronic plc</td>
      <td>134.1253</td>
      <td>350.0000</td>
      <td>197.8947</td>
    </tr>
    <tr>
      <th>319</th>
      <td>LITE</td>
      <td>Lumentum Holdings Inc.</td>
      <td>21.0271</td>
      <td>104.6798</td>
      <td>1434.6746</td>
    </tr>
    <tr>
      <th>5121</th>
      <td>A007660</td>
      <td>ISU Petasys Co. Ltd.</td>
      <td>32.9946</td>
      <td>100.0000</td>
      <td>334.1040</td>
    </tr>
    <tr>
      <th>4522</th>
      <td>5801</td>
      <td>Furukawa Electric Co. Ltd.</td>
      <td>14.8038</td>
      <td>418.0328</td>
      <td>915.5205</td>
    </tr>
    <tr>
      <th>5150</th>
      <td>3939</td>
      <td>Wanguo Gold Group Limited</td>
      <td>75.9223</td>
      <td>100.0000</td>
      <td>121.2414</td>
    </tr>
    <tr>
      <th>4445</th>
      <td>300857</td>
      <td>Sharetronic Data Technology Co. Ltd.</td>
      <td>72.3480</td>
      <td>75.0000</td>
      <td>393.3295</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>POWL</td>
      <td>Powell Industries Inc.</td>
      <td>9.0837</td>
      <td>19.6643</td>
      <td>333.1685</td>
    </tr>
    <tr>
      <th>4651</th>
      <td>A000150</td>
      <td>Doosan Corporation</td>
      <td>11.6102</td>
      <td>134.1463</td>
      <td>384.7059</td>
    </tr>
    <tr>
      <th>4178</th>
      <td>300394</td>
      <td>Suzhou TFC Optical Communication Co. Ltd.</td>
      <td>65.7374</td>
      <td>54.1667</td>
      <td>576.0360</td>
    </tr>
    <tr>
      <th>2050</th>
      <td>PSIX</td>
      <td>Power Solutions International Inc.</td>
      <td>51.7743</td>
      <td>64.4518</td>
      <td>225.7630</td>
    </tr>
    <tr>
      <th>4448</th>
      <td>A010120</td>
      <td>LS ELECTRIC Co. Ltd.</td>
      <td>11.5978</td>
      <td>22.9358</td>
      <td>501.1080</td>
    </tr>
    <tr>
      <th>6457</th>
      <td>BOPP</td>
      <td>Benso Oil Palm Plantation PLC</td>
      <td>49.8521</td>
      <td>21.0526</td>
      <td>185.7143</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>TNGX</td>
      <td>Tango Therapeutics Inc.</td>
      <td>48.2767</td>
      <td>26.8908</td>
      <td>1657.7551</td>
    </tr>
    <tr>
      <th>4470</th>
      <td>2368</td>
      <td>Gold Circuit Electronics Ltd.</td>
      <td>61.2456</td>
      <td>77.1429</td>
      <td>553.7468</td>
    </tr>
    <tr>
      <th>303</th>
      <td>FIX</td>
      <td>Comfort Systems USA Inc.</td>
      <td>29.5150</td>
      <td>97.6093</td>
      <td>399.0377</td>
    </tr>
    <tr>
      <th>94</th>
      <td>APP</td>
      <td>AppLovin Corporation</td>
      <td>16.3820</td>
      <td>110.2564</td>
      <td>80.0686</td>
    </tr>
    <tr>
      <th>6389</th>
      <td>PRESCO</td>
      <td>Presco Plc</td>
      <td>70.6774</td>
      <td>80.0000</td>
      <td>152.2293</td>
    </tr>
    <tr>
      <th>4371</th>
      <td>3653</td>
      <td>Jentech Precision Industrial Co. Ltd</td>
      <td>48.6424</td>
      <td>58.1081</td>
      <td>474.0541</td>
    </tr>
    <tr>
      <th>843</th>
      <td>STRL</td>
      <td>Sterling Infrastructure Inc.</td>
      <td>17.6906</td>
      <td>13.7725</td>
      <td>253.9813</td>
    </tr>
    <tr>
      <th>4726</th>
      <td>5706</td>
      <td>Mitsui Kinzoku Company Limited</td>
      <td>11.1750</td>
      <td>151.3333</td>
      <td>885.2862</td>
    </tr>
    <tr>
      <th>5646</th>
      <td>6187</td>
      <td>All Ring Tech Co. Ltd.</td>
      <td>1.4840</td>
      <td>11.3636</td>
      <td>434.0426</td>
    </tr>
    <tr>
      <th>5976</th>
      <td>4979</td>
      <td>LuxNet Corporation</td>
      <td>32.9970</td>
      <td>41.6667</td>
      <td>296.5035</td>
    </tr>
    <tr>
      <th>4452</th>
      <td>A103590</td>
      <td>Iljin Electric Co.Ltd</td>
      <td>32.6029</td>
      <td>125.3731</td>
      <td>289.8089</td>
    </tr>
    <tr>
      <th>705</th>
      <td>ONDS</td>
      <td>Ondas Inc.</td>
      <td>605.5633</td>
      <td>-1.6393</td>
      <td>1155.7311</td>
    </tr>
    <tr>
      <th>380</th>
      <td>RKLB</td>
      <td>Rocket Lab Corporation</td>
      <td>37.9611</td>
      <td>2.6316</td>
      <td>317.1513</td>
    </tr>
    <tr>
      <th>6557</th>
      <td>VOES</td>
      <td>Voltamp Energy SAOG</td>
      <td>75.5212</td>
      <td>128.5714</td>
      <td>181.1245</td>
    </tr>
    <tr>
      <th>4690</th>
      <td>A079550</td>
      <td>LIG Defense&amp;Aerospace Co. Ltd.</td>
      <td>34.4721</td>
      <td>17.8886</td>
      <td>209.6154</td>
    </tr>
    <tr>
      <th>1184</th>
      <td>AGX</td>
      <td>Argan Inc.</td>
      <td>8.0567</td>
      <td>57.4803</td>
      <td>365.0340</td>
    </tr>
    <tr>
      <th>4572</th>
      <td>6274</td>
      <td>Taiwan Union Technology Corporation</td>
      <td>37.6579</td>
      <td>34.4828</td>
      <td>653.9683</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Dividend quality stocks
div_stocks = screen_dividend_quality(df)
print(f"\nDividend Quality: {len(div_stocks)} stocks")
if len(div_stocks) > 0:
    display_cols = [c for c in ["ticker", "name", "dividend_yield_ltm",
                                "dividend_streak", "fcf_dividend_coverage"] if c in div_stocks.columns]
    display(div_stocks[display_cols].head(50))
```

    Dividend Quality: 591 stocks

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>dividend_yield_ltm</th>
      <th>dividend_streak</th>
      <th>fcf_dividend_coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6609</th>
      <td>EVEN3</td>
      <td>Even Construtora e Incorporadora S.A.</td>
      <td>0.1050</td>
      <td>2</td>
      <td>0.5924</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>CAG</td>
      <td>Conagra Brands Inc.</td>
      <td>0.0971</td>
      <td>1</td>
      <td>1.2582</td>
    </tr>
    <tr>
      <th>5681</th>
      <td>UNVR</td>
      <td>PT Unilever Indonesia Tbk</td>
      <td>0.0823</td>
      <td>2</td>
      <td>0.9682</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>NOMD</td>
      <td>Nomad Foods Limited</td>
      <td>0.0792</td>
      <td>1</td>
      <td>2.7624</td>
    </tr>
    <tr>
      <th>2282</th>
      <td>UPBD</td>
      <td>Upbound Group Inc.</td>
      <td>0.0774</td>
      <td>8</td>
      <td>2.7169</td>
    </tr>
    <tr>
      <th>3350</th>
      <td>DNO</td>
      <td>DNO ASA</td>
      <td>0.0770</td>
      <td>6</td>
      <td>0.6806</td>
    </tr>
    <tr>
      <th>3490</th>
      <td>AKSO</td>
      <td>Aker Solutions ASA</td>
      <td>0.0752</td>
      <td>3</td>
      <td>1.3709</td>
    </tr>
    <tr>
      <th>6566</th>
      <td>CSED3</td>
      <td>Cruzeiro do Sul Educacional S.A.</td>
      <td>0.0739</td>
      <td>3</td>
      <td>4.3807</td>
    </tr>
    <tr>
      <th>587</th>
      <td>KHC</td>
      <td>The Kraft Heinz Company</td>
      <td>0.0728</td>
      <td>1</td>
      <td>1.9289</td>
    </tr>
    <tr>
      <th>3543</th>
      <td>SOMA</td>
      <td>Solstad Maritime ASA</td>
      <td>0.0726</td>
      <td>0</td>
      <td>1.5525</td>
    </tr>
    <tr>
      <th>746</th>
      <td>GIS</td>
      <td>General Mills Inc.</td>
      <td>0.0696</td>
      <td>7</td>
      <td>1.2524</td>
    </tr>
    <tr>
      <th>3672</th>
      <td>CASH</td>
      <td>Prosegur Cash S.A.</td>
      <td>0.0661</td>
      <td>4</td>
      <td>2.2744</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>PAGP</td>
      <td>Plains GP Holdings L.P.</td>
      <td>0.0658</td>
      <td>1</td>
      <td>7.5116</td>
    </tr>
    <tr>
      <th>6442</th>
      <td>ENTEL</td>
      <td>Empresa Nacional de Telecomunicaciones S.A.</td>
      <td>0.0657</td>
      <td>1</td>
      <td>5.0854</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>DOFG</td>
      <td>DOF Group ASA</td>
      <td>0.0646</td>
      <td>0</td>
      <td>1.0983</td>
    </tr>
    <tr>
      <th>99</th>
      <td>PFE</td>
      <td>Pfizer Inc.</td>
      <td>0.0644</td>
      <td>17</td>
      <td>0.9288</td>
    </tr>
    <tr>
      <th>2332</th>
      <td>TRMDA</td>
      <td>TORM plc</td>
      <td>0.0644</td>
      <td>1</td>
      <td>0.9534</td>
    </tr>
    <tr>
      <th>2710</th>
      <td>CONSTI</td>
      <td>Consti Oyj</td>
      <td>0.0634</td>
      <td>2</td>
      <td>2.5376</td>
    </tr>
    <tr>
      <th>334</th>
      <td>MORLD</td>
      <td>Moreld ASA</td>
      <td>0.0628</td>
      <td>0</td>
      <td>6.2241</td>
    </tr>
    <tr>
      <th>935</th>
      <td>BBY</td>
      <td>Best Buy Co. Inc.</td>
      <td>0.0623</td>
      <td>21</td>
      <td>1.5705</td>
    </tr>
    <tr>
      <th>3284</th>
      <td>NOS</td>
      <td>NOS S.G.P.S. S.A.</td>
      <td>0.0621</td>
      <td>1</td>
      <td>3.5503</td>
    </tr>
    <tr>
      <th>145</th>
      <td>MO</td>
      <td>Altria Group Inc.</td>
      <td>0.0620</td>
      <td>18</td>
      <td>1.3037</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>NOG</td>
      <td>Northern Oil and Gas Inc.</td>
      <td>0.0612</td>
      <td>4</td>
      <td>1.4581</td>
    </tr>
    <tr>
      <th>2287</th>
      <td>SBGI</td>
      <td>Sinclair Inc.</td>
      <td>0.0611</td>
      <td>1</td>
      <td>1.6667</td>
    </tr>
    <tr>
      <th>4530</th>
      <td>PTTEP</td>
      <td>PTT Exploration and Production Public Company ...</td>
      <td>0.0610</td>
      <td>1</td>
      <td>0.8025</td>
    </tr>
    <tr>
      <th>760</th>
      <td>HPQ</td>
      <td>HP Inc.</td>
      <td>0.0592</td>
      <td>11</td>
      <td>2.6355</td>
    </tr>
    <tr>
      <th>3642</th>
      <td>TTALO</td>
      <td>Terveystalo Oyj</td>
      <td>0.0591</td>
      <td>3</td>
      <td>2.8979</td>
    </tr>
    <tr>
      <th>762</th>
      <td>AMCR</td>
      <td>Amcor plc</td>
      <td>0.0585</td>
      <td>6</td>
      <td>0.7509</td>
    </tr>
    <tr>
      <th>63</th>
      <td>VZ</td>
      <td>Verizon Communications Inc.</td>
      <td>0.0581</td>
      <td>22</td>
      <td>1.7530</td>
    </tr>
    <tr>
      <th>211</th>
      <td>EPD</td>
      <td>Enterprise Products Partners L.P.</td>
      <td>0.0572</td>
      <td>10</td>
      <td>0.6338</td>
    </tr>
    <tr>
      <th>2534</th>
      <td>EOLUB</td>
      <td>Eolus Aktiebolag (publ)</td>
      <td>0.0572</td>
      <td>1</td>
      <td>31.7993</td>
    </tr>
    <tr>
      <th>4699</th>
      <td>2357</td>
      <td>ASUSTeK Computer Inc.</td>
      <td>0.0571</td>
      <td>3</td>
      <td>1.1121</td>
    </tr>
    <tr>
      <th>4171</th>
      <td>576</td>
      <td>Zhejiang Expressway Co. Ltd.</td>
      <td>0.0563</td>
      <td>2</td>
      <td>1.1514</td>
    </tr>
    <tr>
      <th>3502</th>
      <td>DOM</td>
      <td>Dom Development S.A.</td>
      <td>0.0560</td>
      <td>2</td>
      <td>0.5222</td>
    </tr>
    <tr>
      <th>5668</th>
      <td>DTC</td>
      <td>Dubai Taxi Company P.J.S.C.</td>
      <td>0.0553</td>
      <td>0</td>
      <td>1.1201</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>VALMT</td>
      <td>Valmet Oyj</td>
      <td>0.0539</td>
      <td>1</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>6560</th>
      <td>3003</td>
      <td>City Cement Company</td>
      <td>0.0538</td>
      <td>0</td>
      <td>1.7675</td>
    </tr>
    <tr>
      <th>3234</th>
      <td>ADEN</td>
      <td>Adecco Group AG</td>
      <td>0.0533</td>
      <td>1</td>
      <td>2.7443</td>
    </tr>
    <tr>
      <th>1713</th>
      <td>PEY</td>
      <td>Peyto Exploration &amp; Development Corp.</td>
      <td>0.0526</td>
      <td>1</td>
      <td>1.4535</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>HRB</td>
      <td>H&amp;R Block Inc.</td>
      <td>0.0521</td>
      <td>11</td>
      <td>2.5575</td>
    </tr>
    <tr>
      <th>2784</th>
      <td>ADTR</td>
      <td>Adtraction Group AB</td>
      <td>0.0519</td>
      <td>1</td>
      <td>2.0924</td>
    </tr>
    <tr>
      <th>6176</th>
      <td>CPFE3</td>
      <td>CPFL Energia S.A.</td>
      <td>0.0518</td>
      <td>1</td>
      <td>1.9299</td>
    </tr>
    <tr>
      <th>2626</th>
      <td>REP</td>
      <td>Repsol S.A.</td>
      <td>0.0517</td>
      <td>0</td>
      <td>1.3525</td>
    </tr>
    <tr>
      <th>6179</th>
      <td>4163</td>
      <td>Al-Dawaa Medical Services Company</td>
      <td>0.0516</td>
      <td>2</td>
      <td>2.2334</td>
    </tr>
    <tr>
      <th>509</th>
      <td>KMB</td>
      <td>Kimberly-Clark Corporation</td>
      <td>0.0511</td>
      <td>54</td>
      <td>0.9873</td>
    </tr>
    <tr>
      <th>991</th>
      <td>CLX</td>
      <td>The Clorox Company</td>
      <td>0.0510</td>
      <td>50</td>
      <td>1.2924</td>
    </tr>
    <tr>
      <th>4283</th>
      <td>AADI</td>
      <td>PT Adaro Andalan Indonesia Tbk</td>
      <td>0.0505</td>
      <td>0</td>
      <td>2.1415</td>
    </tr>
    <tr>
      <th>6283</th>
      <td>QEWS</td>
      <td>Nebras Energy Q.P.S.C.</td>
      <td>0.0503</td>
      <td>1</td>
      <td>0.4932</td>
    </tr>
    <tr>
      <th>6186</th>
      <td>ORDS</td>
      <td>Ooredoo Q.P.S.C.</td>
      <td>0.0492</td>
      <td>5</td>
      <td>2.0121</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>OTEX</td>
      <td>Open Text Corporation</td>
      <td>0.0488</td>
      <td>0</td>
      <td>3.2430</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Financially healthy stocks
healthy_stocks = screen_financial_health(df)
print(f"\nFinancial Health: {len(healthy_stocks)} stocks")
if len(healthy_stocks) > 0:
    display_cols = [c for c in ["ticker", "name", "altman_z_score", "piotroski_f_score",
                                "current_ratio", "interest_coverage"] if c in healthy_stocks.columns]
    display(healthy_stocks[display_cols].head(50))
```

    Financial Health: 1037 stocks

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>altman_z_score</th>
      <th>piotroski_f_score</th>
      <th>current_ratio</th>
      <th>interest_coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110</th>
      <td>PCELL</td>
      <td>PowerCell Sweden AB (publ)</td>
      <td>4.9800</td>
      <td>1</td>
      <td>2.2000</td>
      <td>4.9444</td>
    </tr>
    <tr>
      <th>6649</th>
      <td>UMED</td>
      <td>Unité de Fabrication des Médicaments S.A</td>
      <td>3.7900</td>
      <td>9</td>
      <td>2.2000</td>
      <td>-5.6170</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GUI</td>
      <td>Guillemot Corporation S.A.</td>
      <td>NaN</td>
      <td>4</td>
      <td>2.1000</td>
      <td>-2.8421</td>
    </tr>
    <tr>
      <th>112</th>
      <td>VGO</td>
      <td>VIGO Photonics S.A.</td>
      <td>5.1800</td>
      <td>1</td>
      <td>1.7000</td>
      <td>2.0568</td>
    </tr>
    <tr>
      <th>28</th>
      <td>GARO</td>
      <td>Garo Aktiebolag (publ)</td>
      <td>3.3500</td>
      <td>4</td>
      <td>1.5000</td>
      <td>-3.3973</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CEG</td>
      <td>Constellation Energy Corporation</td>
      <td>NaN</td>
      <td>3</td>
      <td>1.5000</td>
      <td>-5.9315</td>
    </tr>
    <tr>
      <th>41</th>
      <td>LEMON</td>
      <td>Lemonsoft Oyj</td>
      <td>5.7000</td>
      <td>4</td>
      <td>1.0000</td>
      <td>-3.6789</td>
    </tr>
    <tr>
      <th>118</th>
      <td>LOW</td>
      <td>Lowe's Companies Inc.</td>
      <td>3.2300</td>
      <td>4</td>
      <td>1.1000</td>
      <td>-6.6837</td>
    </tr>
    <tr>
      <th>6586</th>
      <td>SKPC</td>
      <td>Sidi Kerir Petrochemicals Co.</td>
      <td>3.1700</td>
      <td>3</td>
      <td>1.4000</td>
      <td>-2.8511</td>
    </tr>
    <tr>
      <th>6587</th>
      <td>KLKIM</td>
      <td>Kalekim Kimyevi Maddeler Sanayi Ve Ticaret Ano...</td>
      <td>5.4900</td>
      <td>4</td>
      <td>2.1000</td>
      <td>-4.7334</td>
    </tr>
    <tr>
      <th>6597</th>
      <td>6001</td>
      <td>Halwani Bros. Co. Ltd.</td>
      <td>4.6000</td>
      <td>4</td>
      <td>1.0000</td>
      <td>-2.6882</td>
    </tr>
    <tr>
      <th>6598</th>
      <td>6014</td>
      <td>Alamar Foods Company</td>
      <td>4.3500</td>
      <td>4</td>
      <td>1.2000</td>
      <td>-5.8737</td>
    </tr>
    <tr>
      <th>6602</th>
      <td>KEPL3</td>
      <td>Kepler Weber S.A.</td>
      <td>3.4000</td>
      <td>3</td>
      <td>2.0000</td>
      <td>-3.5917</td>
    </tr>
    <tr>
      <th>6618</th>
      <td>TPR</td>
      <td>Tunisie Profilés Aluminium Société Anonyme</td>
      <td>4.0300</td>
      <td>6</td>
      <td>1.8000</td>
      <td>-5.2614</td>
    </tr>
    <tr>
      <th>107</th>
      <td>MRVL</td>
      <td>Marvell Technology Inc.</td>
      <td>5.8300</td>
      <td>2</td>
      <td>2.0000</td>
      <td>-6.6086</td>
    </tr>
    <tr>
      <th>5263</th>
      <td>300748</td>
      <td>JL Mag Rare-Earth Co. Ltd.</td>
      <td>3.9400</td>
      <td>2</td>
      <td>1.8000</td>
      <td>-8.7010</td>
    </tr>
    <tr>
      <th>5268</th>
      <td>SFR</td>
      <td>Sandfire Resources Limited</td>
      <td>3.2900</td>
      <td>6</td>
      <td>1.6000</td>
      <td>-7.8459</td>
    </tr>
    <tr>
      <th>5273</th>
      <td>2018</td>
      <td>AAC Technologies Holdings Inc.</td>
      <td>NaN</td>
      <td>2</td>
      <td>1.5000</td>
      <td>-6.5507</td>
    </tr>
    <tr>
      <th>409</th>
      <td>STP</td>
      <td>Stalprodukt S.A.</td>
      <td>3.1100</td>
      <td>3</td>
      <td>4.1000</td>
      <td>4.4513</td>
    </tr>
    <tr>
      <th>412</th>
      <td>DDOG</td>
      <td>Datadog Inc.</td>
      <td>10.7300</td>
      <td>2</td>
      <td>3.4000</td>
      <td>3.8698</td>
    </tr>
    <tr>
      <th>193</th>
      <td>SNPS</td>
      <td>Synopsys Inc.</td>
      <td>3.6300</td>
      <td>7</td>
      <td>1.4000</td>
      <td>-1.6454</td>
    </tr>
    <tr>
      <th>194</th>
      <td>HDF</td>
      <td>Hydrogène de France Société anonyme</td>
      <td>NaN</td>
      <td>2</td>
      <td>8.7000</td>
      <td>49.6667</td>
    </tr>
    <tr>
      <th>206</th>
      <td>PINE</td>
      <td>Pinewood Technologies Group PLC</td>
      <td>3.7700</td>
      <td>1</td>
      <td>2.4000</td>
      <td>31.6750</td>
    </tr>
    <tr>
      <th>216</th>
      <td>VLO</td>
      <td>Valero Energy Corporation</td>
      <td>4.4700</td>
      <td>4</td>
      <td>1.6000</td>
      <td>-8.5270</td>
    </tr>
    <tr>
      <th>230</th>
      <td>PE</td>
      <td>Premier Energy PLC</td>
      <td>NaN</td>
      <td>7</td>
      <td>1.6000</td>
      <td>-6.7724</td>
    </tr>
    <tr>
      <th>238</th>
      <td>AFC</td>
      <td>AFC Energy plc</td>
      <td>3.2600</td>
      <td>1</td>
      <td>5.1000</td>
      <td>836.0000</td>
    </tr>
    <tr>
      <th>243</th>
      <td>ELV</td>
      <td>Elevance Health Inc.</td>
      <td>NaN</td>
      <td>6</td>
      <td>1.5000</td>
      <td>-5.5205</td>
    </tr>
    <tr>
      <th>255</th>
      <td>CIEN</td>
      <td>Ciena Corporation</td>
      <td>4.2700</td>
      <td>9</td>
      <td>2.8000</td>
      <td>-4.7846</td>
    </tr>
    <tr>
      <th>265</th>
      <td>DHH</td>
      <td>Dominion Hosting Holding S.p.A.</td>
      <td>3.0600</td>
      <td>4</td>
      <td>1.6000</td>
      <td>-8.1727</td>
    </tr>
    <tr>
      <th>270</th>
      <td>KOMN</td>
      <td>Komax Holding AG</td>
      <td>3.2200</td>
      <td>3</td>
      <td>2.8000</td>
      <td>-1.1910</td>
    </tr>
    <tr>
      <th>420</th>
      <td>WGO</td>
      <td>Winnebago Industries Inc.</td>
      <td>3.6900</td>
      <td>7</td>
      <td>2.3000</td>
      <td>-3.0854</td>
    </tr>
    <tr>
      <th>445</th>
      <td>XAR</td>
      <td>Xaar plc</td>
      <td>3.6600</td>
      <td>3</td>
      <td>2.6000</td>
      <td>1.4516</td>
    </tr>
    <tr>
      <th>454</th>
      <td>VMC</td>
      <td>Vulcan Materials Company</td>
      <td>4.0200</td>
      <td>4</td>
      <td>2.7000</td>
      <td>-6.5198</td>
    </tr>
    <tr>
      <th>459</th>
      <td>MLM</td>
      <td>Martin Marietta Materials Inc.</td>
      <td>3.5800</td>
      <td>4</td>
      <td>3.6000</td>
      <td>-6.4000</td>
    </tr>
    <tr>
      <th>476</th>
      <td>NXR</td>
      <td>Norcros plc</td>
      <td>3.0500</td>
      <td>5</td>
      <td>1.9000</td>
      <td>-6.0985</td>
    </tr>
    <tr>
      <th>483</th>
      <td>DYN</td>
      <td>Dyne Therapeutics Inc.</td>
      <td>NaN</td>
      <td>1</td>
      <td>22.3000</td>
      <td>75.6349</td>
    </tr>
    <tr>
      <th>496</th>
      <td>IR</td>
      <td>Ingersoll Rand Inc.</td>
      <td>3.4800</td>
      <td>4</td>
      <td>2.1000</td>
      <td>-5.9472</td>
    </tr>
    <tr>
      <th>291</th>
      <td>EAPI</td>
      <td>Euroapi S.A.</td>
      <td>3.4200</td>
      <td>4</td>
      <td>2.6000</td>
      <td>-1.0301</td>
    </tr>
    <tr>
      <th>294</th>
      <td>PSX</td>
      <td>Phillips 66</td>
      <td>3.3900</td>
      <td>4</td>
      <td>1.3000</td>
      <td>-4.2204</td>
    </tr>
    <tr>
      <th>321</th>
      <td>MSTR</td>
      <td>Strategy Inc</td>
      <td>4.8800</td>
      <td>1</td>
      <td>5.6000</td>
      <td>83.7985</td>
    </tr>
    <tr>
      <th>359</th>
      <td>LUCE</td>
      <td>Luceco plc</td>
      <td>3.2200</td>
      <td>4</td>
      <td>1.9000</td>
      <td>-4.5806</td>
    </tr>
    <tr>
      <th>362</th>
      <td>AKAST</td>
      <td>Akastor ASA</td>
      <td>3.8400</td>
      <td>3</td>
      <td>2.3000</td>
      <td>0.3206</td>
    </tr>
    <tr>
      <th>380</th>
      <td>RKLB</td>
      <td>Rocket Lab Corporation</td>
      <td>18.4200</td>
      <td>1</td>
      <td>4.1000</td>
      <td>8.6387</td>
    </tr>
    <tr>
      <th>6405</th>
      <td>TABGD</td>
      <td>Tab Gida Sanayi ve Ticaret A.S.</td>
      <td>4.2700</td>
      <td>4</td>
      <td>1.5000</td>
      <td>-3.2736</td>
    </tr>
    <tr>
      <th>6406</th>
      <td>MDIA3</td>
      <td>M. Dias Branco S.A. Indústria e Comércio de Al...</td>
      <td>3.1800</td>
      <td>3</td>
      <td>2.6000</td>
      <td>-4.2007</td>
    </tr>
    <tr>
      <th>6417</th>
      <td>KUOB</td>
      <td>Kuo S.A.B. de C.V.</td>
      <td>NaN</td>
      <td>3</td>
      <td>1.0000</td>
      <td>-3.1737</td>
    </tr>
    <tr>
      <th>6431</th>
      <td>PMR</td>
      <td>Premier Group Limited</td>
      <td>4.3900</td>
      <td>8</td>
      <td>1.4000</td>
      <td>-7.4382</td>
    </tr>
    <tr>
      <th>6438</th>
      <td>VIVA3</td>
      <td>Vivara Participações S.A.</td>
      <td>3.8800</td>
      <td>2</td>
      <td>3.6000</td>
      <td>-5.0948</td>
    </tr>
    <tr>
      <th>6443</th>
      <td>HTTBT</td>
      <td>Hitit Bilgisayar Hizmetleri A.S.</td>
      <td>12.2900</td>
      <td>4</td>
      <td>2.2000</td>
      <td>-5.0663</td>
    </tr>
    <tr>
      <th>6453</th>
      <td>TRU</td>
      <td>Truworths International Limited</td>
      <td>4.4600</td>
      <td>6</td>
      <td>1.9000</td>
      <td>-6.0079</td>
    </tr>
  </tbody>
</table>
</div>

## 21. Composite Rankings

```python
# Rank stocks by composite score
ranked_df = rank_stocks_by_composite_score(df)
print(f"Ranked {len(ranked_df)} stocks by composite score")
if "composite_score" in ranked_df.columns:
    display_cols = [c for c in ["ticker", "name", "industry", "composite_score"]
                    if c in ranked_df.columns]
    print("\nTop 50 by Composite Score:")
    display(ranked_df[display_cols].head(50))

    # Composite score distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(ranked_df["composite_score"].dropna(), bins=50, color=COLORS["primary"],
            alpha=0.85, edgecolor="white")
    ax.axvline(ranked_df["composite_score"].median(), color=COLORS["danger"],
               linestyle="--", lw=2, label=f"Median: {ranked_df['composite_score'].median():.2f}")
    ax.set_title("Composite Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
```

    Ranked 6676 stocks by composite score
    
    Top 50 by Composite Score:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>name</th>
      <th>industry</th>
      <th>composite_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>670</th>
      <td>RL</td>
      <td>Ralph Lauren Corporation</td>
      <td>Textiles Apparel and Luxury Goods</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>394</th>
      <td>AAPL</td>
      <td>Apple Inc.</td>
      <td>Technology Hardware Storage and Peripherals</td>
      <td>97.5000</td>
    </tr>
    <tr>
      <th>1748</th>
      <td>KFY</td>
      <td>Korn Ferry</td>
      <td>Professional Services</td>
      <td>97.5000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>LRCX</td>
      <td>Lam Research Corporation</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>97.5000</td>
    </tr>
    <tr>
      <th>764</th>
      <td>BR</td>
      <td>Broadridge Financial Solutions Inc.</td>
      <td>Professional Services</td>
      <td>97.2222</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>AIT</td>
      <td>Applied Industrial Technologies Inc.</td>
      <td>Trading Companies and Distributors</td>
      <td>97.2222</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>PLEJD</td>
      <td>Plejd AB (publ)</td>
      <td>Electrical Equipment</td>
      <td>97.2222</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>EAT</td>
      <td>Brinker International Inc.</td>
      <td>Hotels Restaurants and Leisure</td>
      <td>97.2222</td>
    </tr>
    <tr>
      <th>554</th>
      <td>CASY</td>
      <td>Casey's General Stores Inc.</td>
      <td>Consumer Staples Distribution and Retail</td>
      <td>97.2222</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TXN</td>
      <td>Texas Instruments Incorporated</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>94.7222</td>
    </tr>
    <tr>
      <th>1560</th>
      <td>SKY</td>
      <td>Champion Homes Inc.</td>
      <td>Household Durables</td>
      <td>94.7222</td>
    </tr>
    <tr>
      <th>51</th>
      <td>KLAC</td>
      <td>KLA Corporation</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>94.7222</td>
    </tr>
    <tr>
      <th>1508</th>
      <td>PSMT</td>
      <td>PriceSmart Inc.</td>
      <td>Consumer Staples Distribution and Retail</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>516</th>
      <td>RMD</td>
      <td>ResMed Inc.</td>
      <td>Health Care Equipment and Supplies</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>4834</th>
      <td>INDUSTOWER</td>
      <td>Indus Towers Limited</td>
      <td>Diversified Telecommunication Services</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>129</th>
      <td>PH</td>
      <td>Parker-Hannifin Corporation</td>
      <td>Machinery</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>FIZZ</td>
      <td>National Beverage Corp.</td>
      <td>Beverages</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>6322</th>
      <td>VFQS</td>
      <td>Vodafone Qatar P.Q.S.C.</td>
      <td>Wireless Telecommunication Services</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>4032</th>
      <td>2330</td>
      <td>Taiwan Semiconductor Manufacturing Company Lim...</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>2918</th>
      <td>ADDTB</td>
      <td>Addtech AB (publ.)</td>
      <td>Trading Companies and Distributors</td>
      <td>94.4444</td>
    </tr>
    <tr>
      <th>3527</th>
      <td>RUSTA</td>
      <td>Rusta AB (publ)</td>
      <td>Broadline Retail</td>
      <td>93.9861</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>ALV</td>
      <td>Autoliv Inc.</td>
      <td>Automobile Components</td>
      <td>93.8194</td>
    </tr>
    <tr>
      <th>680</th>
      <td>STE</td>
      <td>STERIS plc</td>
      <td>Health Care Equipment and Supplies</td>
      <td>92.5000</td>
    </tr>
    <tr>
      <th>634</th>
      <td>FICO</td>
      <td>Fair Isaac Corporation</td>
      <td>Software</td>
      <td>92.5000</td>
    </tr>
    <tr>
      <th>2372</th>
      <td>ASML</td>
      <td>ASML Holding N.V.</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>91.9444</td>
    </tr>
    <tr>
      <th>1636</th>
      <td>CVCO</td>
      <td>Cavco Industries Inc.</td>
      <td>Household Durables</td>
      <td>91.9444</td>
    </tr>
    <tr>
      <th>165</th>
      <td>MCK</td>
      <td>McKesson Corporation</td>
      <td>Health Care Providers and Services</td>
      <td>91.9444</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>VAR</td>
      <td>Vår Energi ASA</td>
      <td>Oil Gas and Consumable Fuels</td>
      <td>91.9444</td>
    </tr>
    <tr>
      <th>3840</th>
      <td>COOR</td>
      <td>Coor Service Management Holding AB</td>
      <td>Commercial Services and Supplies</td>
      <td>91.9444</td>
    </tr>
    <tr>
      <th>15</th>
      <td>COST</td>
      <td>Costco Wholesale Corporation</td>
      <td>Consumer Staples Distribution and Retail</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>2907</th>
      <td>SECTB</td>
      <td>Sectra AB (publ)</td>
      <td>Health Care Technology</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>261</th>
      <td>CTAS</td>
      <td>Cintas Corporation</td>
      <td>Commercial Services and Supplies</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>2130</th>
      <td>KARO</td>
      <td>Karooooo Ltd.</td>
      <td>Software</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>3096</th>
      <td>TEMN</td>
      <td>Temenos AG</td>
      <td>Software</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>502</th>
      <td>CPRT</td>
      <td>Copart Inc.</td>
      <td>Commercial Services and Supplies</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>684</th>
      <td>WWD</td>
      <td>Woodward Inc.</td>
      <td>Aerospace and Defense</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>396</th>
      <td>MSFT</td>
      <td>Microsoft Corporation</td>
      <td>Software</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>2594</th>
      <td>HMB</td>
      <td>H &amp; M Hennes &amp; Mauritz AB (publ)</td>
      <td>Specialty Retail</td>
      <td>91.6667</td>
    </tr>
    <tr>
      <th>6179</th>
      <td>4163</td>
      <td>Al-Dawaa Medical Services Company</td>
      <td>Consumer Staples Distribution and Retail</td>
      <td>91.0625</td>
    </tr>
    <tr>
      <th>2934</th>
      <td>SPM</td>
      <td>Saipem SpA</td>
      <td>Energy Equipment and Services</td>
      <td>90.9722</td>
    </tr>
    <tr>
      <th>1833</th>
      <td>HUBG</td>
      <td>Hub Group Inc.</td>
      <td>Air Freight and Logistics</td>
      <td>90.9722</td>
    </tr>
    <tr>
      <th>706</th>
      <td>CRS</td>
      <td>Carpenter Technology Corporation</td>
      <td>Aerospace and Defense</td>
      <td>90.9722</td>
    </tr>
    <tr>
      <th>40</th>
      <td>AMAT</td>
      <td>Applied Materials Inc.</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>90.6944</td>
    </tr>
    <tr>
      <th>536</th>
      <td>ET</td>
      <td>Evertz Technologies Limited</td>
      <td>Communications Equipment</td>
      <td>90.6944</td>
    </tr>
    <tr>
      <th>4074</th>
      <td>6857</td>
      <td>Advantest Corporation</td>
      <td>Semiconductors and Semiconductor Equipment</td>
      <td>90.0000</td>
    </tr>
    <tr>
      <th>4478</th>
      <td>4307</td>
      <td>Nomura Research Institute Ltd.</td>
      <td>IT Services</td>
      <td>90.0000</td>
    </tr>
    <tr>
      <th>5529</th>
      <td>603939</td>
      <td>Yifeng Pharmacy Chain Co. Ltd.</td>
      <td>Consumer Staples Distribution and Retail</td>
      <td>89.7222</td>
    </tr>
    <tr>
      <th>818</th>
      <td>PTC</td>
      <td>PTC Inc.</td>
      <td>Software</td>
      <td>89.7222</td>
    </tr>
    <tr>
      <th>5861</th>
      <td>8056</td>
      <td>BIPROGY Inc.</td>
      <td>IT Services</td>
      <td>89.7222</td>
    </tr>
    <tr>
      <th>4125</th>
      <td>6098</td>
      <td>Recruit Holdings Co. Ltd.</td>
      <td>Professional Services</td>
      <td>89.7222</td>
    </tr>
  </tbody>
</table>
</div>

    C:\Users\markm\AppData\Local\Temp\ipykernel_57952\1857897955.py:22: UserWarning: The figure layout has changed to tight
      plt.tight_layout()

![png](pml_model_analysis_files/pml_model_analysis_52_3.png)

## 22. Statistical Analysis: Bayesian Category Analysis

```python
# Bayesian analysis on ALL feature categories (excluding "direct" calculation_type features)

# Build a set of features with "direct" calculation_type to exclude
direct_features = set(
    feature_cat.loc[feature_cat["calculation_type"] == "direct", "feature_alias"].tolist()
)
print(f"Excluding {len(direct_features)} features with calculation_type='direct'")

for cat_name, features in FEATURE_CATEGORIES.items():
    # Filter out "direct" features and keep only those present in df
    available = [f for f in features if f in df.columns and f not in direct_features]
    if available:
        print(f"\n{'=' * 60}")
        print(f"  Bayesian Analysis: {cat_name} ({len(available)} features)")
        print(f"{'=' * 60}")
        result = bayesian_category_analysis(df, cat_name, available)
        if isinstance(result, dict):
            for feat, stats in result.items():
                if isinstance(stats, dict):
                    print(f"\n  {feat}:")
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            print(f"    {k}: {v:.4f}")
    else:
        skipped_reason = "all features are 'direct' or not in df"
        print(f"\n  [Skipped] {cat_name}: {skipped_reason}")
```

    Excluding 405 features with calculation_type='direct'
    
    ============================================================
      Bayesian Analysis: Accounting Quality (20 features)
    ============================================================
    
      asset_sale_frequency:
        n_obs: 6676.0000
        sample_mean: 4.1390
        sample_std: 3.8598
        posterior_mean: 4.1389
        posterior_std: 0.0472
        ci_95_low: 4.0463
        ci_95_high: 4.2315
        prob_positive: 1.0000
    
      asset_sale_trend:
        n_obs: 6676.0000
        sample_mean: -3.6384
        sample_std: 188.2569
        posterior_mean: -3.4550
        posterior_std: 2.2452
        ci_95_low: -7.8556
        ci_95_high: 0.9457
        prob_positive: 0.0619
    
      tax_rate_yoy_change:
        n_obs: 6676.0000
        sample_mean: -0.0813
        sample_std: 4.7544
        posterior_mean: -0.0813
        posterior_std: 0.0582
        ci_95_low: -0.1953
        ci_95_high: 0.0328
        prob_positive: 0.0812
    
      tax_rate_qoq_change:
        n_obs: 6676.0000
        sample_mean: -0.0206
        sample_std: 2.3036
        posterior_mean: -0.0206
        posterior_std: 0.0282
        ci_95_low: -0.0758
        ci_95_high: 0.0347
        prob_positive: 0.2329
    
      tax_rate_stability:
        n_obs: 6676.0000
        sample_mean: 0.4284
        sample_std: 3.7309
        posterior_mean: 0.4284
        posterior_std: 0.0457
        ci_95_low: 0.3389
        ci_95_high: 0.5179
        prob_positive: 1.0000
    
      low_tax_flag:
        n_obs: 6676.0000
        sample_mean: 0.3276
        sample_std: 0.4694
        posterior_mean: 0.3276
        posterior_std: 0.0057
        ci_95_low: 0.3163
        ci_95_high: 0.3389
        prob_positive: 1.0000
    
      tax_rate_trend_4q:
        n_obs: 6676.0000
        sample_mean: -0.0209
        sample_std: 1.6414
        posterior_mean: -0.0209
        posterior_std: 0.0201
        ci_95_low: -0.0603
        ci_95_high: 0.0184
        prob_positive: 0.1487
    
      goodwill_change_rate:
        n_obs: 4720.0000
        sample_mean: 19.8644
        sample_std: 1257.0662
        posterior_mean: 4.5687
        posterior_std: 8.7750
        ci_95_low: -12.6303
        ci_95_high: 21.7677
        prob_positive: 0.6987
    
      restructuring_intensity:
        n_obs: 6339.0000
        sample_mean: -0.0016
        sample_std: 0.0066
        posterior_mean: -0.0016
        posterior_std: 0.0001
        ci_95_low: -0.0017
        ci_95_high: -0.0014
        prob_positive: 0.0000
    
      exceptional_items_frequency:
        n_obs: 6676.0000
        sample_mean: 0.5804
        sample_std: 0.7001
        posterior_mean: 0.5804
        posterior_std: 0.0086
        ci_95_low: 0.5636
        ci_95_high: 0.5972
        prob_positive: 1.0000
    
      merger_impact_ratio:
        n_obs: 6676.0000
        sample_mean: -0.0041
        sample_std: 0.0336
        posterior_mean: -0.0041
        posterior_std: 0.0004
        ci_95_low: -0.0050
        ci_95_high: -0.0033
        prob_positive: 0.0000
    
      non_operating_income_share:
        n_obs: 6668.0000
        sample_mean: 0.2782
        sample_std: 2.6206
        posterior_mean: 0.2782
        posterior_std: 0.0321
        ci_95_low: 0.2153
        ci_95_high: 0.3411
        prob_positive: 1.0000
    
      asset_sale_boost:
        n_obs: 6676.0000
        sample_mean: 0.3012
        sample_std: 0.4588
        posterior_mean: 0.3012
        posterior_std: 0.0056
        ci_95_low: 0.2902
        ci_95_high: 0.3122
        prob_positive: 1.0000
    
      accounting_quality_score:
        n_obs: 6676.0000
        sample_mean: 85.7385
        sample_std: 16.9735
        posterior_mean: 85.7015
        posterior_std: 0.2077
        ci_95_low: 85.2944
        ci_95_high: 86.1086
        prob_positive: 1.0000
    
      goodwill_3y_growth:
        n_obs: 4591.0000
        sample_mean: 785.0122
        sample_std: 21660.9157
        posterior_mean: 0.7674
        posterior_std: 9.9951
        ci_95_low: -18.8230
        ci_95_high: 20.3578
        prob_positive: 0.5306
    
      goodwill_qoq_change:
        n_obs: 4093.0000
        sample_mean: 4.8344
        sample_std: 219.2584
        posterior_mean: 4.3262
        posterior_std: 3.2421
        ci_95_low: -2.0282
        ci_95_high: 10.6807
        prob_positive: 0.9090
    
      goodwill_to_assets_trend:
        n_obs: 6575.0000
        sample_mean: -0.0754
        sample_std: 3.6752
        posterior_mean: -0.0754
        posterior_std: 0.0453
        ci_95_low: -0.1642
        ci_95_high: 0.0135
        prob_positive: 0.0482
    
      goodwill_yoy_change:
        n_obs: 4720.0000
        sample_mean: 1993.4599
        sample_std: 125707.1076
        posterior_mean: 0.0595
        posterior_std: 9.9999
        ci_95_low: -19.5402
        ci_95_high: 19.6592
        prob_positive: 0.5024
    
      impairment_risk_score:
        n_obs: 6676.0000
        sample_mean: 16.9004
        sample_std: 29.4085
        posterior_mean: 16.8786
        posterior_std: 0.3597
        ci_95_low: 16.1736
        ci_95_high: 17.5836
        prob_positive: 1.0000
    
      recent_acquisition_flag:
        n_obs: 6676.0000
        sample_mean: 0.0180
        sample_std: 0.1329
        posterior_mean: 0.0180
        posterior_std: 0.0016
        ci_95_low: 0.0148
        ci_95_high: 0.0212
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Analyst Sentiment (24 features)
    ============================================================
    
      analyst_count_stability:
        n_obs: 6605.0000
        sample_mean: 1.0890
        sample_std: 0.5182
        posterior_mean: 1.0890
        posterior_std: 0.0064
        ci_95_low: 1.0765
        ci_95_high: 1.1015
        prob_positive: 1.0000
    
      pt_accuracy_1y:
        n_obs: 6350.0000
        sample_mean: 0.4523
        sample_std: 0.7592
        posterior_mean: 0.4523
        posterior_std: 0.0095
        ci_95_low: 0.4337
        ci_95_high: 0.4710
        prob_positive: 1.0000
    
      pt_optimism_bias:
        n_obs: 6350.0000
        sample_mean: -0.1431
        sample_std: 0.8721
        posterior_mean: -0.1431
        posterior_std: 0.0109
        ci_95_low: -0.1646
        ci_95_high: -0.1217
        prob_positive: 0.0000
    
      pt_high_low_convergence_1y:
        n_obs: 6350.0000
        sample_mean: -0.0057
        sample_std: 0.4043
        posterior_mean: -0.0057
        posterior_std: 0.0051
        ci_95_low: -0.0156
        ci_95_high: 0.0043
        prob_positive: 0.1310
    
      eps_gaap_vs_norm_ntm:
        n_obs: 5557.0000
        sample_mean: -0.2780
        sample_std: 3.7904
        posterior_mean: -0.2780
        posterior_std: 0.0508
        ci_95_low: -0.3776
        ci_95_high: -0.1783
        prob_positive: 0.0000
    
      eps_gaap_vs_norm_fy1e:
        n_obs: 5491.0000
        sample_mean: -0.2450
        sample_std: 1.4545
        posterior_mean: -0.2450
        posterior_std: 0.0196
        ci_95_low: -0.2835
        ci_95_high: -0.2066
        prob_positive: 0.0000
    
      forward_adjustment_trend:
        n_obs: 3451.0000
        sample_mean: -0.9114
        sample_std: 4.1516
        posterior_mean: -0.9114
        posterior_std: 0.0707
        ci_95_low: -1.0499
        ci_95_high: -0.7729
        prob_positive: 0.0000
    
      ebitda_forward_growth:
        n_obs: 6621.0000
        sample_mean: 0.0471
        sample_std: 22.4145
        posterior_mean: 0.0471
        posterior_std: 0.2754
        ci_95_low: -0.4926
        ci_95_high: 0.5868
        prob_positive: 0.5679
    
      earnings_revision_divergence:
        n_obs: 4589.0000
        sample_mean: 0.7156
        sample_std: 51.0265
        posterior_mean: 0.7115
        posterior_std: 0.7511
        ci_95_low: -0.7607
        ci_95_high: 2.1837
        prob_positive: 0.8283
    
      forward_pe_vs_sector_proxy:
        n_obs: 5217.0000
        sample_mean: -0.0648
        sample_std: 1.4468
        posterior_mean: -0.0648
        posterior_std: 0.0200
        ci_95_low: -0.1041
        ci_95_high: -0.0256
        prob_positive: 0.0006
    
      pt_achievement_1y:
        n_obs: 6350.0000
        sample_mean: 0.8454
        sample_std: 0.2006
        posterior_mean: 0.8454
        posterior_std: 0.0025
        ci_95_low: 0.8405
        ci_95_high: 0.8503
        prob_positive: 1.0000
    
      pt_range_hit_rate:
        n_obs: 6676.0000
        sample_mean: 0.3167
        sample_std: 0.4652
        posterior_mean: 0.3167
        posterior_std: 0.0057
        ci_95_low: 0.3055
        ci_95_high: 0.3278
        prob_positive: 1.0000
    
      pt_median_vs_mean_spread:
        n_obs: 6676.0000
        sample_mean: 0.0059
        sample_std: 0.0635
        posterior_mean: 0.0059
        posterior_std: 0.0008
        ci_95_low: 0.0043
        ci_95_high: 0.0074
        prob_positive: 1.0000
    
      pe_forward_discount:
        n_obs: 5003.0000
        sample_mean: -0.1476
        sample_std: 1.1111
        posterior_mean: -0.1476
        posterior_std: 0.0157
        ci_95_low: -0.1784
        ci_95_high: -0.1168
        prob_positive: 0.0000
    
      analyst_bullish_pct:
        n_obs: 6590.0000
        sample_mean: 67.3140
        sample_std: 31.0230
        posterior_mean: 67.2159
        posterior_std: 0.3819
        ci_95_low: 66.4674
        ci_95_high: 67.9644
        prob_positive: 1.0000
    
      analyst_bearish_pct:
        n_obs: 6673.0000
        sample_mean: 6.1789
        sample_std: 14.1081
        posterior_mean: 6.1770
        posterior_std: 0.1727
        ci_95_low: 5.8386
        ci_95_high: 6.5155
        prob_positive: 1.0000
    
      analyst_neutral_pct:
        n_obs: 6673.0000
        sample_mean: 25.4083
        sample_std: 26.2842
        posterior_mean: 25.3820
        posterior_std: 0.3216
        ci_95_low: 24.7516
        ci_95_high: 26.0123
        prob_positive: 1.0000
    
      upside_potential:
        n_obs: 6676.0000
        sample_mean: 23.8791
        sample_std: 37.5904
        posterior_mean: 23.8287
        posterior_std: 0.4596
        ci_95_low: 22.9279
        ci_95_high: 24.7294
        prob_positive: 1.0000
    
      price_target_spread_pct:
        n_obs: 6676.0000
        sample_mean: 40.5841
        sample_std: 39.1608
        posterior_mean: 40.4911
        posterior_std: 0.4787
        ci_95_low: 39.5528
        ci_95_high: 41.4295
        prob_positive: 1.0000
    
      price_target_revision_1m:
        n_obs: 6633.0000
        sample_mean: 0.0111
        sample_std: 0.0975
        posterior_mean: 0.0111
        posterior_std: 0.0012
        ci_95_low: 0.0088
        ci_95_high: 0.0135
        prob_positive: 1.0000
    
      price_target_revision_3m:
        n_obs: 6583.0000
        sample_mean: 0.0595
        sample_std: 0.2401
        posterior_mean: 0.0595
        posterior_std: 0.0030
        ci_95_low: 0.0537
        ci_95_high: 0.0653
        prob_positive: 1.0000
    
      eps_revision_momentum:
        n_obs: 6676.0000
        sample_mean: 0.1394
        sample_std: 8.5391
        posterior_mean: 0.1394
        posterior_std: 0.1045
        ci_95_low: -0.0654
        ci_95_high: 0.3442
        prob_positive: 0.9089
    
      analyst_rating_normalized:
        n_obs: 6676.0000
        sample_mean: 75.7709
        sample_std: 21.3940
        posterior_mean: 75.7189
        posterior_std: 0.2617
        ci_95_low: 75.2059
        ci_95_high: 76.2320
        prob_positive: 1.0000
    
      analyst_coverage_quality:
        n_obs: 6676.0000
        sample_mean: 1.0038
        sample_std: 0.7379
        posterior_mean: 1.0038
        posterior_std: 0.0090
        ci_95_low: 0.9861
        ci_95_high: 1.0215
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Balance Sheet (24 features)
    ============================================================
    
      cash_to_assets_pct:
        n_obs: 6339.0000
        sample_mean: 13.7744
        sample_std: 13.8325
        posterior_mean: 13.7703
        posterior_std: 0.1737
        ci_95_low: 13.4298
        ci_95_high: 14.1108
        prob_positive: 1.0000
    
      cash_change_qoq:
        n_obs: 6556.0000
        sample_mean: 0.0984
        sample_std: 3.9123
        posterior_mean: 0.0984
        posterior_std: 0.0483
        ci_95_low: 0.0037
        ci_95_high: 0.1931
        prob_positive: 0.9792
    
      cash_vs_5y_avg:
        n_obs: 6066.0000
        sample_mean: 1.1673
        sample_std: 0.9626
        posterior_mean: 1.1673
        posterior_std: 0.0124
        ci_95_low: 1.1431
        ci_95_high: 1.1915
        prob_positive: 1.0000
    
      inventory_change_yoy:
        n_obs: 5390.0000
        sample_mean: 0.0406
        sample_std: 3.7851
        posterior_mean: 0.0406
        posterior_std: 0.0516
        ci_95_low: -0.0604
        ci_95_high: 0.1417
        prob_positive: 0.7846
    
      inventory_vs_5y_avg:
        n_obs: 5433.0000
        sample_mean: 1.0863
        sample_std: 0.6033
        posterior_mean: 1.0863
        posterior_std: 0.0082
        ci_95_low: 1.0702
        ci_95_high: 1.1023
        prob_positive: 1.0000
    
      working_capital_vs_5y_avg:
        n_obs: 6356.0000
        sample_mean: 1.3465
        sample_std: 9.3933
        posterior_mean: 1.3463
        posterior_std: 0.1178
        ci_95_low: 1.1154
        ci_95_high: 1.5772
        prob_positive: 1.0000
    
      retained_earnings_vs_5y:
        n_obs: 6001.0000
        sample_mean: 0.8741
        sample_std: 19.7442
        posterior_mean: 0.8735
        posterior_std: 0.2548
        ci_95_low: 0.3741
        ci_95_high: 1.3729
        prob_positive: 0.9997
    
      intangibles_growth_flag:
        n_obs: 6676.0000
        sample_mean: 0.1072
        sample_std: 0.3095
        posterior_mean: 0.1072
        posterior_std: 0.0038
        ci_95_low: 0.0998
        ci_95_high: 0.1147
        prob_positive: 1.0000
    
      asset_quality_score:
        n_obs: 6676.0000
        sample_mean: 56.2747
        sample_std: 21.9068
        posterior_mean: 56.2343
        posterior_std: 0.2680
        ci_95_low: 55.7090
        ci_95_high: 56.7596
        prob_positive: 1.0000
    
      balance_sheet_strength:
        n_obs: 6676.0000
        sample_mean: 58.2235
        sample_std: 34.8642
        posterior_mean: 58.1177
        posterior_std: 0.4263
        ci_95_low: 57.2821
        ci_95_high: 58.9532
        prob_positive: 1.0000
    
      debt_maturity_risk:
        n_obs: 6621.0000
        sample_mean: 1.9595
        sample_std: 93.1580
        posterior_mean: 1.9341
        posterior_std: 1.1374
        ci_95_low: -0.2953
        ci_95_high: 4.1635
        prob_positive: 0.9555
    
      receivables_change_yoy:
        n_obs: 6426.0000
        sample_mean: 0.6813
        sample_std: 25.0960
        posterior_mean: 0.6807
        posterior_std: 0.3129
        ci_95_low: 0.0674
        ci_95_high: 1.2940
        prob_positive: 0.9852
    
      inventory_4q_trend:
        n_obs: 5414.0000
        sample_mean: 23.4268
        sample_std: 369.3948
        posterior_mean: 18.7110
        posterior_std: 4.4867
        ci_95_low: 9.9171
        ci_95_high: 27.5048
        prob_positive: 1.0000
    
      inventory_buildup_flag:
        n_obs: 6676.0000
        sample_mean: 0.2042
        sample_std: 0.4031
        posterior_mean: 0.2042
        posterior_std: 0.0049
        ci_95_low: 0.1945
        ci_95_high: 0.2138
        prob_positive: 1.0000
    
      inventory_days:
        n_obs: 6446.0000
        sample_mean: 105.4705
        sample_std: 362.1575
        posterior_mean: 87.6385
        posterior_std: 4.1118
        ci_95_low: 79.5793
        ci_95_high: 95.6977
        prob_positive: 1.0000
    
      inventory_qoq_change:
        n_obs: 5594.0000
        sample_mean: -1.9580
        sample_std: 93.3852
        posterior_mean: -1.9279
        posterior_std: 1.2390
        ci_95_low: -4.3563
        ci_95_high: 0.5004
        prob_positive: 0.0598
    
      inventory_reduction_flag:
        n_obs: 6676.0000
        sample_mean: 0.1486
        sample_std: 0.3557
        posterior_mean: 0.1486
        posterior_std: 0.0044
        ci_95_low: 0.1401
        ci_95_high: 0.1571
        prob_positive: 1.0000
    
      inventory_to_assets:
        n_obs: 6339.0000
        sample_mean: 9.8968
        sample_std: 11.3604
        posterior_mean: 9.8948
        posterior_std: 0.1427
        ci_95_low: 9.6152
        ci_95_high: 10.1745
        prob_positive: 1.0000
    
      inventory_to_revenue:
        n_obs: 6532.0000
        sample_mean: 16.1907
        sample_std: 104.6623
        posterior_mean: 15.9236
        posterior_std: 1.2843
        ci_95_low: 13.4065
        ci_95_high: 18.4408
        prob_positive: 1.0000
    
      inventory_yoy_change:
        n_obs: 5722.0000
        sample_mean: 21.1440
        sample_std: 204.6377
        posterior_mean: 19.7021
        posterior_std: 2.6114
        ci_95_low: 14.5838
        ci_95_high: 24.8205
        prob_positive: 1.0000
    
      asset_growth_accel:
        n_obs: 6642.0000
        sample_mean: 32.3563
        sample_std: 1920.6434
        posterior_mean: 4.9370
        posterior_std: 9.2055
        ci_95_low: -13.1059
        ci_95_high: 22.9798
        prob_positive: 0.7041
    
      assets_3y_cagr:
        n_obs: 6578.0000
        sample_mean: 9.2582
        sample_std: 33.4158
        posterior_mean: 9.2425
        posterior_std: 0.4117
        ci_95_low: 8.4357
        ci_95_high: 10.0494
        prob_positive: 1.0000
    
      assets_qoq_growth:
        n_obs: 6584.0000
        sample_mean: 0.0857
        sample_std: 61.7881
        posterior_mean: 0.0852
        posterior_std: 0.7593
        ci_95_low: -1.4030
        ci_95_high: 1.5734
        prob_positive: 0.5447
    
      assets_yoy_growth:
        n_obs: 6663.0000
        sample_mean: 51.9492
        sample_std: 1881.3222
        posterior_mean: 8.2303
        posterior_std: 9.1737
        ci_95_low: -9.7502
        ci_95_high: 26.2107
        prob_positive: 0.8152
    
    ============================================================
      Bayesian Analysis: Cash Flow (32 features)
    ============================================================
    
      fcf_est_cagr_5y:
        n_obs: 1604.0000
        sample_mean: 16.8827
        sample_std: 23.0861
        posterior_mean: 16.8268
        posterior_std: 0.5755
        ci_95_low: 15.6988
        ci_95_high: 17.9547
        prob_positive: 1.0000
    
      fcf_est_trend:
        n_obs: 5761.0000
        sample_mean: -0.6222
        sample_std: 5.3721
        posterior_mean: -0.6222
        posterior_std: 0.0708
        ci_95_low: -0.7609
        ci_95_high: -0.4835
        prob_positive: 0.0000
    
      cfo_to_net_income:
        n_obs: 6668.0000
        sample_mean: 2.4509
        sample_std: 34.4984
        posterior_mean: 2.4466
        posterior_std: 0.4221
        ci_95_low: 1.6192
        ci_95_high: 3.2739
        prob_positive: 1.0000
    
      fcf_to_net_income:
        n_obs: 6668.0000
        sample_mean: 1.1551
        sample_std: 18.3018
        posterior_mean: 1.1545
        posterior_std: 0.2241
        ci_95_low: 0.7153
        ci_95_high: 1.5937
        prob_positive: 1.0000
    
      fcf_margin:
        n_obs: 6532.0000
        sample_mean: -1.4271
        sample_std: 44.4228
        posterior_mean: -1.4228
        posterior_std: 0.5488
        ci_95_low: -2.4984
        ci_95_high: -0.3471
        prob_positive: 0.0048
    
      cfo_growth_yoy:
        n_obs: 6638.0000
        sample_mean: 2.0996
        sample_std: 134.0078
        posterior_mean: 2.0443
        posterior_std: 1.6230
        ci_95_low: -1.1368
        ci_95_high: 5.2253
        prob_positive: 0.8961
    
      fcf_positive_ratio:
        n_obs: 6676.0000
        sample_mean: 0.6196
        sample_std: 0.3577
        posterior_mean: 0.6196
        posterior_std: 0.0044
        ci_95_low: 0.6110
        ci_95_high: 0.6282
        prob_positive: 1.0000
    
      acquisition_intensity:
        n_obs: 6676.0000
        sample_mean: 113.8004
        sample_std: 681.7376
        posterior_mean: 67.0924
        posterior_std: 6.4065
        ci_95_low: 54.5356
        ci_95_high: 79.6492
        prob_positive: 1.0000
    
      fcf_est_growth_fy1_vs_ltm:
        n_obs: 6090.0000
        sample_mean: 93.8314
        sample_std: 2386.8065
        posterior_mean: 9.0620
        posterior_std: 9.5049
        ci_95_low: -9.5676
        ci_95_high: 27.6915
        prob_positive: 0.8298
    
      fcf_est_growth_fy2_vs_fy1:
        n_obs: 5761.0000
        sample_mean: 98.8689
        sample_std: 1107.5777
        posterior_mean: 31.5939
        posterior_std: 8.2489
        ci_95_low: 15.4260
        ci_95_high: 47.7618
        prob_positive: 0.9999
    
      fcf_est_cagr_3y:
        n_obs: 3418.0000
        sample_mean: 19.3324
        sample_std: 56.0806
        posterior_mean: 19.1561
        posterior_std: 0.9549
        ci_95_low: 17.2846
        ci_95_high: 21.0277
        prob_positive: 1.0000
    
      fcf_est_margin_fy1:
        n_obs: 6532.0000
        sample_mean: -114.8963
        sample_std: 4107.7775
        posterior_mean: -4.2820
        posterior_std: 9.8119
        ci_95_low: -23.5133
        ci_95_high: 14.9493
        prob_positive: 0.3313
    
      fcf_est_yield_fy1:
        n_obs: 6676.0000
        sample_mean: 3.7577
        sample_std: 13.9355
        posterior_mean: 3.7566
        posterior_std: 0.1705
        ci_95_low: 3.4224
        ci_95_high: 4.0909
        prob_positive: 1.0000
    
      fcf_est_growth_acceleration:
        n_obs: 5284.0000
        sample_mean: -24.8497
        sample_std: 2673.8534
        posterior_mean: -1.7102
        posterior_std: 9.6498
        ci_95_low: -20.6237
        ci_95_high: 17.2034
        prob_positive: 0.4297
    
      self_funding_ratio:
        n_obs: 6071.0000
        sample_mean: 3.8817
        sample_std: 121.2808
        posterior_mean: 3.7898
        posterior_std: 1.5380
        ci_95_low: 0.7753
        ci_95_high: 6.8044
        prob_positive: 0.9931
    
      cff_pattern_score:
        n_obs: 6676.0000
        sample_mean: 0.3631
        sample_std: 0.8901
        posterior_mean: 0.3631
        posterior_std: 0.0109
        ci_95_low: 0.3417
        ci_95_high: 0.3844
        prob_positive: 1.0000
    
      cff_quarterly_trend:
        n_obs: 5896.0000
        sample_mean: 945.3636
        sample_std: 52151.5149
        posterior_mean: 0.2049
        posterior_std: 9.9989
        ci_95_low: -19.3930
        ci_95_high: 19.8028
        prob_positive: 0.5082
    
      cfi_quarterly_trend:
        n_obs: 5892.0000
        sample_mean: -582.5097
        sample_std: 11724.4345
        posterior_mean: -2.4861
        posterior_std: 9.9786
        ci_95_low: -22.0443
        ci_95_high: 17.0720
        prob_positive: 0.4016
    
      cfo_quarterly_trend:
        n_obs: 5928.0000
        sample_mean: 98.8787
        sample_std: 3214.0091
        posterior_mean: 5.3664
        posterior_std: 9.7249
        ci_95_low: -13.6943
        ci_95_high: 24.4271
        prob_positive: 0.7095
    
      fcf_quarterly_trend:
        n_obs: 5968.0000
        sample_mean: 50.1755
        sample_std: 2174.7327
        posterior_mean: 5.6221
        posterior_std: 9.4231
        ci_95_low: -12.8472
        ci_95_high: 24.0914
        prob_positive: 0.7246
    
      operating_cf_momentum:
        n_obs: 6150.0000
        sample_mean: 292.9200
        sample_std: 9520.8367
        posterior_mean: 1.9740
        posterior_std: 9.9662
        ci_95_low: -17.5599
        ci_95_high: 21.5078
        prob_positive: 0.5785
    
      fcf_growth_yoy:
        n_obs: 6655.0000
        sample_mean: 234.5271
        sample_std: 10345.6944
        posterior_mean: 1.4492
        posterior_std: 9.9691
        ci_95_low: -18.0901
        ci_95_high: 20.9886
        prob_positive: 0.5578
    
      fcf_yield:
        n_obs: 6676.0000
        sample_mean: 4.2813
        sample_std: 17.5541
        posterior_mean: 4.2794
        posterior_std: 0.2148
        ci_95_low: 3.8584
        ci_95_high: 4.7004
        prob_positive: 1.0000
    
      cf_volatility_score:
        n_obs: 6183.0000
        sample_mean: 15.4338
        sample_std: 290.4346
        posterior_mean: 13.5810
        posterior_std: 3.4648
        ci_95_low: 6.7900
        ci_95_high: 20.3720
        prob_positive: 1.0000
    
      fcf_est_cagr_5y_fwd:
        n_obs: 1451.0000
        sample_mean: 14.5374
        sample_std: 23.8713
        posterior_mean: 14.4805
        posterior_std: 0.6254
        ci_95_low: 13.2546
        ci_95_high: 15.7064
        prob_positive: 1.0000
    
      fcf_est_capex_implied_ratio:
        n_obs: 6089.0000
        sample_mean: 0.8197
        sample_std: 25.0226
        posterior_mean: 0.8188
        posterior_std: 0.3205
        ci_95_low: 0.1906
        ci_95_high: 1.4470
        prob_positive: 0.9947
    
      fcf_est_growth_deceleration:
        n_obs: 6676.0000
        sample_mean: 0.1609
        sample_std: 0.3674
        posterior_mean: 0.1609
        posterior_std: 0.0045
        ci_95_low: 0.1521
        ci_95_high: 0.1697
        prob_positive: 1.0000
    
      fcf_est_growth_fy3_vs_fy2:
        n_obs: 5737.0000
        sample_mean: 44.7041
        sample_std: 627.0758
        posterior_mean: 26.5241
        posterior_std: 6.3771
        ci_95_low: 14.0249
        ci_95_high: 39.0232
        prob_positive: 1.0000
    
      fcf_est_growth_fy4_vs_fy3:
        n_obs: 4842.0000
        sample_mean: -22.2562
        sample_std: 195.8549
        posterior_mean: -20.6224
        posterior_std: 2.7094
        ci_95_low: -25.9328
        ci_95_high: -15.3121
        prob_positive: 0.0000
    
      fcf_est_growth_fy5_vs_fy4:
        n_obs: 2510.0000
        sample_mean: 21.5666
        sample_std: 431.5660
        posterior_mean: 12.3802
        posterior_std: 6.5265
        ci_95_low: -0.4118
        ci_95_high: 25.1722
        prob_positive: 0.9711
    
      fcf_est_vs_historical:
        n_obs: 6082.0000
        sample_mean: -112.5223
        sample_std: 10582.3366
        posterior_mean: -0.6078
        posterior_std: 9.9730
        ci_95_low: -20.1548
        ci_95_high: 18.9392
        prob_positive: 0.4757
    
    ============================================================
      Bayesian Analysis: Composite Scores (3 features)
    ============================================================
    
      dilution_score:
        n_obs: 6676.0000
        sample_mean: 48.7984
        sample_std: 14.0549
        posterior_mean: 48.7840
        posterior_std: 0.1720
        ci_95_low: 48.4469
        ci_95_high: 49.1211
        prob_positive: 1.0000
    
      quality_momentum_score:
        n_obs: 6676.0000
        sample_mean: 53.0879
        sample_std: 13.8547
        posterior_mean: 53.0727
        posterior_std: 0.1695
        ci_95_low: 52.7404
        ci_95_high: 53.4050
        prob_positive: 1.0000
    
      piotroski_f_score:
        n_obs: 6676.0000
        sample_mean: 3.4714
        sample_std: 1.5617
        posterior_mean: 3.4714
        posterior_std: 0.0191
        ci_95_low: 3.4339
        ci_95_high: 3.5088
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Dividend Reliability (18 features)
    ============================================================
    
      div_yield_5y_trend:
        n_obs: 4471.0000
        sample_mean: -0.0010
        sample_std: 0.0217
        posterior_mean: -0.0010
        posterior_std: 0.0003
        ci_95_low: -0.0017
        ci_95_high: -0.0004
        prob_positive: 0.0007
    
      div_yield_stability:
        n_obs: 5215.0000
        sample_mean: 0.0325
        sample_std: 0.0683
        posterior_mean: 0.0325
        posterior_std: 0.0009
        ci_95_low: 0.0307
        ci_95_high: 0.0344
        prob_positive: 1.0000
    
      div_yield_declining_flag:
        n_obs: 6676.0000
        sample_mean: 0.0653
        sample_std: 0.2471
        posterior_mean: 0.0653
        posterior_std: 0.0030
        ci_95_low: 0.0594
        ci_95_high: 0.0712
        prob_positive: 1.0000
    
      div_yield_mean_5y:
        n_obs: 5215.0000
        sample_mean: 0.0270
        sample_std: 0.0296
        posterior_mean: 0.0270
        posterior_std: 0.0004
        ci_95_low: 0.0262
        ci_95_high: 0.0278
        prob_positive: 1.0000
    
      div_yield_vs_5y_mean:
        n_obs: 4905.0000
        sample_mean: 0.0903
        sample_std: 0.8198
        posterior_mean: 0.0903
        posterior_std: 0.0117
        ci_95_low: 0.0673
        ci_95_high: 0.1132
        prob_positive: 1.0000
    
      dividend_payout_ratio:
        n_obs: 3861.0000
        sample_mean: 0.4359
        sample_std: 7.3668
        posterior_mean: 0.4359
        posterior_std: 0.1185
        ci_95_low: 0.2035
        ci_95_high: 0.6682
        prob_positive: 0.9999
    
      fcf_dividend_coverage:
        n_obs: 4089.0000
        sample_mean: 14.9119
        sample_std: 406.0631
        posterior_mean: 10.6267
        posterior_std: 5.3607
        ci_95_low: 0.1198
        ci_95_high: 21.1336
        prob_positive: 0.9763
    
      total_shareholder_yield:
        n_obs: 6676.0000
        sample_mean: 0.0162
        sample_std: 0.0864
        posterior_mean: 0.0162
        posterior_std: 0.0011
        ci_95_low: 0.0141
        ci_95_high: 0.0183
        prob_positive: 1.0000
    
      dividend_growth_expectation:
        n_obs: 3955.0000
        sample_mean: 0.0009
        sample_std: 0.2197
        posterior_mean: 0.0009
        posterior_std: 0.0035
        ci_95_low: -0.0060
        ci_95_high: 0.0077
        prob_positive: 0.5962
    
      high_yield_flag:
        n_obs: 6676.0000
        sample_mean: 0.1035
        sample_std: 0.3046
        posterior_mean: 0.1035
        posterior_std: 0.0037
        ci_95_low: 0.0962
        ci_95_high: 0.1108
        prob_positive: 1.0000
    
      sustainable_dividend_flag:
        n_obs: 6676.0000
        sample_mean: 0.0783
        sample_std: 0.2687
        posterior_mean: 0.0783
        posterior_std: 0.0033
        ci_95_low: 0.0719
        ci_95_high: 0.0848
        prob_positive: 1.0000
    
      days_since_ex_date:
        n_obs: 4621.0000
        sample_mean: 94.3211
        sample_std: 132.4758
        posterior_mean: 90.8700
        posterior_std: 1.9128
        ci_95_low: 87.1209
        ci_95_high: 94.6192
        prob_positive: 1.0000
    
      days_to_payment:
        n_obs: 4568.0000
        sample_mean: -74.1156
        sample_std: 140.5642
        posterior_mean: -71.0427
        posterior_std: 2.0362
        ci_95_low: -75.0336
        ci_95_high: -67.0518
        prob_positive: 0.0000
    
      dividend_announced_flag:
        n_obs: 6676.0000
        sample_mean: 0.0682
        sample_std: 0.2520
        posterior_mean: 0.0682
        posterior_std: 0.0031
        ci_95_low: 0.0621
        ci_95_high: 0.0742
        prob_positive: 1.0000
    
      dividend_frequency_score:
        n_obs: 6676.0000
        sample_mean: 0.8259
        sample_std: 1.4297
        posterior_mean: 0.8259
        posterior_std: 0.0175
        ci_95_low: 0.7916
        ci_95_high: 0.8602
        prob_positive: 1.0000
    
      ex_date_approaching_flag:
        n_obs: 6676.0000
        sample_mean: 0.0502
        sample_std: 0.2183
        posterior_mean: 0.0502
        posterior_std: 0.0027
        ci_95_low: 0.0449
        ci_95_high: 0.0554
        prob_positive: 1.0000
    
      recent_dividend_change:
        n_obs: 4398.0000
        sample_mean: 10.3095
        sample_std: 155.9322
        posterior_mean: 9.7694
        posterior_std: 2.2889
        ci_95_low: 5.2832
        ci_95_high: 14.2556
        prob_positive: 1.0000
    
      div_yield_growth_expected:
        n_obs: 3934.0000
        sample_mean: 60.1636
        sample_std: 678.4170
        posterior_mean: 27.7261
        posterior_std: 7.3427
        ci_95_low: 13.3344
        ci_95_high: 42.1178
        prob_positive: 0.9999
    
    ============================================================
      Bayesian Analysis: EPS Trajectory (9 features)
    ============================================================
    
      eps_qoq_growth:
        n_obs: 6034.0000
        sample_mean: -4.8134
        sample_std: 979.3407
        posterior_mean: -1.8588
        posterior_std: 7.8347
        ci_95_low: -17.2148
        ci_95_high: 13.4972
        prob_positive: 0.4062
    
      eps_yoy_quarterly:
        n_obs: 5986.0000
        sample_mean: 45.8253
        sample_std: 1359.1130
        posterior_mean: 11.2156
        posterior_std: 8.6905
        ci_95_low: -5.8178
        ci_95_high: 28.2491
        prob_positive: 0.9016
    
      eps_positive_streak:
        n_obs: 6676.0000
        sample_mean: 3.6182
        sample_std: 1.7876
        posterior_mean: 3.6182
        posterior_std: 0.0219
        ci_95_low: 3.5753
        ci_95_high: 3.6610
        prob_positive: 1.0000
    
      eps_cagr_3y:
        n_obs: 4520.0000
        sample_mean: 8.5702
        sample_std: 43.5305
        posterior_mean: 8.5344
        posterior_std: 0.6461
        ci_95_low: 7.2680
        ci_95_high: 9.8008
        prob_positive: 1.0000
    
      eps_cagr_5y:
        n_obs: 3974.0000
        sample_mean: 11.1342
        sample_std: 25.2340
        posterior_mean: 11.1164
        posterior_std: 0.4000
        ci_95_low: 10.3324
        ci_95_high: 11.9003
        prob_positive: 1.0000
    
      eps_improvement_count:
        n_obs: 6676.0000
        sample_mean: 2.7466
        sample_std: 1.0965
        posterior_mean: 2.7465
        posterior_std: 0.0134
        ci_95_low: 2.7202
        ci_95_high: 2.7729
        prob_positive: 1.0000
    
      eps_trajectory_score:
        n_obs: 6676.0000
        sample_mean: 54.9311
        sample_std: 21.9296
        posterior_mean: 54.8916
        posterior_std: 0.2683
        ci_95_low: 54.3657
        ci_95_high: 55.4174
        prob_positive: 1.0000
    
      composite_eps_trajectory_score:
        n_obs: 6676.0000
        sample_mean: 54.9311
        sample_std: 21.9296
        posterior_mean: 54.8916
        posterior_std: 0.2683
        ci_95_low: 54.3657
        ci_95_high: 55.4174
        prob_positive: 1.0000
    
      eps_growth_accel:
        n_obs: 3752.0000
        sample_mean: -3.8181
        sample_std: 36.0950
        posterior_mean: -3.8049
        posterior_std: 0.5883
        ci_95_low: -4.9578
        ci_95_high: -2.6519
        prob_positive: 0.0000
    
    ============================================================
      Bayesian Analysis: Earnings Quality (26 features)
    ============================================================
    
      eps_surprise_pct:
        n_obs: 3506.0000
        sample_mean: -27.0687
        sample_std: 734.5433
        posterior_mean: -10.6614
        posterior_std: 7.7855
        ci_95_low: -25.9209
        ci_95_high: 4.5981
        prob_positive: 0.0854
    
      revenue_surprise_pct:
        n_obs: 6432.0000
        sample_mean: 8.5197
        sample_std: 1194.6505
        posterior_mean: 2.6468
        posterior_std: 8.3026
        ci_95_low: -13.6263
        ci_95_high: 18.9199
        prob_positive: 0.6251
    
      gaap_adj_eps_gap_pct:
        n_obs: 5448.0000
        sample_mean: -13.5372
        sample_std: 179.0106
        posterior_mean: -12.7852
        posterior_std: 2.3569
        ci_95_low: -17.4048
        ci_95_high: -8.1656
        prob_positive: 0.0000
    
      ebitda_adjustment_ratio:
        n_obs: 6621.0000
        sample_mean: 0.4162
        sample_std: 9.8836
        posterior_mean: 0.4161
        posterior_std: 0.1215
        ci_95_low: 0.1781
        ci_95_high: 0.6542
        prob_positive: 0.9997
    
      eps_quarterly_trend:
        n_obs: 5986.0000
        sample_mean: 0.4583
        sample_std: 13.5911
        posterior_mean: 0.4581
        posterior_std: 0.1756
        ci_95_low: 0.1139
        ci_95_high: 0.8024
        prob_positive: 0.9954
    
      eps_adjustment_ratio:
        n_obs: 3535.0000
        sample_mean: 1.1579
        sample_std: 7.3471
        posterior_mean: 1.1577
        posterior_std: 0.1236
        ci_95_low: 0.9155
        ci_95_high: 1.3999
        prob_positive: 1.0000
    
      eps_yoy_growth:
        n_obs: 6443.0000
        sample_mean: 7.6377
        sample_std: 1427.9903
        posterior_mean: 1.8338
        posterior_std: 8.7172
        ci_95_low: -15.2519
        ci_95_high: 18.9196
        prob_positive: 0.5833
    
      eps_cont_cagr_3y:
        n_obs: 4533.0000
        sample_mean: 8.7004
        sample_std: 45.1095
        posterior_mean: 8.6615
        posterior_std: 0.6685
        ci_95_low: 7.3513
        ci_95_high: 9.9718
        prob_positive: 1.0000
    
      eps_cont_qoq_growth:
        n_obs: 6033.0000
        sample_mean: 8.0431
        sample_std: 914.8327
        posterior_mean: 3.3692
        posterior_std: 7.6230
        ci_95_low: -11.5719
        ci_95_high: 18.3104
        prob_positive: 0.6707
    
      eps_cont_trajectory_score:
        n_obs: 6676.0000
        sample_mean: 52.9883
        sample_std: 25.3230
        posterior_mean: 52.9375
        posterior_std: 0.3098
        ci_95_low: 52.3303
        ci_95_high: 53.5446
        prob_positive: 1.0000
    
      eps_cont_yoy_growth:
        n_obs: 6438.0000
        sample_mean: -12.4984
        sample_std: 1450.1219
        posterior_mean: -2.9296
        posterior_std: 8.7499
        ci_95_low: -20.0793
        ci_95_high: 14.2202
        prob_positive: 0.3689
    
      gaap_positive_revision_flag:
        n_obs: 6676.0000
        sample_mean: 0.2082
        sample_std: 0.4061
        posterior_mean: 0.2082
        posterior_std: 0.0050
        ci_95_low: 0.1985
        ci_95_high: 0.2179
        prob_positive: 1.0000
    
      gaap_revision_1y:
        n_obs: 5177.0000
        sample_mean: 0.0761
        sample_std: 1.7800
        posterior_mean: 0.0761
        posterior_std: 0.0247
        ci_95_low: 0.0276
        ci_95_high: 0.1246
        prob_positive: 0.9989
    
      gaap_revision_3m:
        n_obs: 5408.0000
        sample_mean: 0.0612
        sample_std: 1.0601
        posterior_mean: 0.0612
        posterior_std: 0.0144
        ci_95_low: 0.0329
        ci_95_high: 0.0894
        prob_positive: 1.0000
    
      gaap_revision_6m:
        n_obs: 5304.0000
        sample_mean: 0.0826
        sample_std: 1.6122
        posterior_mean: 0.0826
        posterior_std: 0.0221
        ci_95_low: 0.0392
        ci_95_high: 0.1260
        prob_positive: 0.9999
    
      gaap_revision_acceleration:
        n_obs: 5270.0000
        sample_mean: -0.0405
        sample_std: 1.5521
        posterior_mean: -0.0405
        posterior_std: 0.0214
        ci_95_low: -0.0824
        ci_95_high: 0.0014
        prob_positive: 0.0290
    
      gaap_revision_1m:
        n_obs: 5468.0000
        sample_mean: 0.0328
        sample_std: 0.6232
        posterior_mean: 0.0328
        posterior_std: 0.0084
        ci_95_low: 0.0162
        ci_95_high: 0.0493
        prob_positive: 0.9999
    
      gaap_revision_momentum:
        n_obs: 6676.0000
        sample_mean: 0.0462
        sample_std: 0.6557
        posterior_mean: 0.0462
        posterior_std: 0.0080
        ci_95_low: 0.0305
        ci_95_high: 0.0620
        prob_positive: 1.0000
    
      revision_quality_divergence:
        n_obs: 4589.0000
        sample_mean: 0.8663
        sample_std: 51.0242
        posterior_mean: 0.8615
        posterior_std: 0.7511
        ci_95_low: -0.6107
        ci_95_high: 2.3336
        prob_positive: 0.8743
    
      net_income_growth_yoy:
        n_obs: 6649.0000
        sample_mean: 14.6632
        sample_std: 3936.6513
        posterior_mean: 0.6032
        posterior_std: 9.7921
        ci_95_low: -18.5894
        ci_95_high: 19.7958
        prob_positive: 0.5246
    
      gaap_vs_norm_revision_spread:
        n_obs: 4621.0000
        sample_mean: 0.7190
        sample_std: 50.8665
        posterior_mean: 0.7150
        posterior_std: 0.7462
        ci_95_low: -0.7476
        ci_95_high: 2.1775
        prob_positive: 0.8310
    
      net_income_qoq_growth:
        n_obs: 6615.0000
        sample_mean: -21.6966
        sample_std: 2296.8119
        posterior_mean: -2.4175
        posterior_std: 9.4264
        ci_95_low: -20.8933
        ci_95_high: 16.0583
        prob_positive: 0.3988
    
      ni_adjustment_ratio:
        n_obs: 6668.0000
        sample_mean: 0.7287
        sample_std: 9.2720
        posterior_mean: 0.7286
        posterior_std: 0.1135
        ci_95_low: 0.5061
        ci_95_high: 0.9511
        prob_positive: 1.0000
    
      has_unusual_items_flag:
        n_obs: 6676.0000
        sample_mean: 0.6955
        sample_std: 0.4602
        posterior_mean: 0.6955
        posterior_std: 0.0056
        ci_95_low: 0.6844
        ci_95_high: 0.7065
        prob_positive: 1.0000
    
      unusual_items_to_ebitda:
        n_obs: 6621.0000
        sample_mean: 30.2364
        sample_std: 281.0509
        posterior_mean: 27.0137
        posterior_std: 3.2647
        ci_95_low: 20.6148
        ci_95_high: 33.4126
        prob_positive: 1.0000
    
      unusual_items_to_revenue:
        n_obs: 6532.0000
        sample_mean: 6.9250
        sample_std: 170.4675
        posterior_mean: 6.6300
        posterior_std: 2.0638
        ci_95_low: 2.5850
        ci_95_high: 10.6751
        prob_positive: 0.9993
    
    ============================================================
      Bayesian Analysis: Efficiency (4 features)
    ============================================================
    
      asset_turnover:
        n_obs: 6339.0000
        sample_mean: 0.7581
        sample_std: 0.5944
        posterior_mean: 0.7581
        posterior_std: 0.0075
        ci_95_low: 0.7434
        ci_95_high: 0.7727
        prob_positive: 1.0000
    
      inventory_turnover:
        n_obs: 5392.0000
        sample_mean: 44.9316
        sample_std: 499.1184
        posterior_mean: 30.7326
        posterior_std: 5.6215
        ci_95_low: 19.7145
        ci_95_high: 41.7507
        prob_positive: 1.0000
    
      receivables_days:
        n_obs: 6531.0000
        sample_mean: 67.1417
        sample_std: 135.0851
        posterior_mean: 65.3167
        posterior_std: 1.6487
        ci_95_low: 62.0853
        ci_95_high: 68.5481
        prob_positive: 1.0000
    
      working_capital_turns:
        n_obs: 6338.0000
        sample_mean: 5.4823
        sample_std: 238.0838
        posterior_mean: 5.0323
        posterior_std: 2.8652
        ci_95_low: -0.5835
        ci_95_high: 10.6480
        prob_positive: 0.9605
    
    ============================================================
      Bayesian Analysis: Efficiency Ratios (25 features)
    ============================================================
    
      opex_vs_revenue_trend:
        n_obs: 6488.0000
        sample_mean: -125.9205
        sample_std: 5776.1972
        posterior_mean: -2.4019
        posterior_std: 9.9042
        ci_95_low: -21.8141
        ci_95_high: 17.0102
        prob_positive: 0.4042
    
      opex_qoq_growth:
        n_obs: 6618.0000
        sample_mean: 0.1198
        sample_std: 3.0455
        posterior_mean: 0.1198
        posterior_std: 0.0374
        ci_95_low: 0.0464
        ci_95_high: 0.1932
        prob_positive: 0.9993
    
      opex_yoy_growth:
        n_obs: 6642.0000
        sample_mean: 0.2706
        sample_std: 5.5552
        posterior_mean: 0.2706
        posterior_std: 0.0682
        ci_95_low: 0.1370
        ci_95_high: 0.4042
        prob_positive: 1.0000
    
      sga_qoq_growth:
        n_obs: 6517.0000
        sample_mean: -0.6508
        sample_std: 1.8366
        posterior_mean: -0.6508
        posterior_std: 0.0228
        ci_95_low: -0.6954
        ci_95_high: -0.6062
        prob_positive: 0.0000
    
      sga_yoy_growth:
        n_obs: 6517.0000
        sample_mean: 0.2197
        sample_std: 4.2619
        posterior_mean: 0.2196
        posterior_std: 0.0528
        ci_95_low: 0.1162
        ci_95_high: 0.3231
        prob_positive: 1.0000
    
      operating_leverage_score:
        n_obs: 6482.0000
        sample_mean: 0.3106
        sample_std: 8.1186
        posterior_mean: 0.3106
        posterior_std: 0.1008
        ci_95_low: 0.1130
        ci_95_high: 0.5082
        prob_positive: 0.9990
    
      cogs_to_revenue:
        n_obs: 6532.0000
        sample_mean: 84.3046
        sample_std: 701.0220
        posterior_mean: 48.1096
        posterior_std: 6.5524
        ci_95_low: 35.2670
        ci_95_high: 60.9523
        prob_positive: 1.0000
    
      opex_to_revenue:
        n_obs: 6532.0000
        sample_mean: 199.3369
        sample_std: 3191.7503
        posterior_mean: 12.0112
        posterior_std: 9.6940
        ci_95_low: -6.9891
        ci_95_high: 31.0115
        prob_positive: 0.8923
    
      sga_to_revenue:
        n_obs: 6531.0000
        sample_mean: 55.8893
        sample_std: 1054.6613
        posterior_mean: 20.6758
        posterior_std: 7.9376
        ci_95_low: 5.1181
        ci_95_high: 36.2336
        prob_positive: 0.9954
    
      rnd_to_revenue:
        n_obs: 6532.0000
        sample_mean: 44.7503
        sample_std: 2072.3494
        posterior_mean: 5.9078
        posterior_std: 9.3166
        ci_95_low: -12.3526
        ci_95_high: 24.1683
        prob_positive: 0.7370
    
      interest_to_revenue:
        n_obs: 6532.0000
        sample_mean: -6.5944
        sample_std: 171.0678
        posterior_mean: -6.3116
        posterior_std: 2.0708
        ci_95_low: -10.3703
        ci_95_high: -2.2529
        prob_positive: 0.0012
    
      cost_efficiency_score:
        n_obs: 6676.0000
        sample_mean: 43.3587
        sample_std: 17.5017
        posterior_mean: 43.3388
        posterior_std: 0.2142
        ci_95_low: 42.9191
        ci_95_high: 43.7585
        prob_positive: 1.0000
    
      marketing_to_revenue:
        n_obs: 6531.0000
        sample_mean: 0.6073
        sample_std: 4.3950
        posterior_mean: 0.6073
        posterior_std: 0.0544
        ci_95_low: 0.5007
        ci_95_high: 0.7139
        prob_positive: 1.0000
    
      marketing_trend_yoy:
        n_obs: 832.0000
        sample_mean: 34.6375
        sample_std: 495.8951
        posterior_mean: 8.7564
        posterior_std: 8.6441
        ci_95_low: -8.1859
        ci_95_high: 25.6988
        prob_positive: 0.8445
    
      sga_efficiency_trend:
        n_obs: 6488.0000
        sample_mean: 44.0960
        sample_std: 1615.2977
        posterior_mean: 8.7813
        posterior_std: 8.9491
        ci_95_low: -8.7588
        ci_95_high: 26.3215
        prob_positive: 0.8368
    
      sga_trend_yoy:
        n_obs: 6488.0000
        sample_mean: -44.0960
        sample_std: 1615.2977
        posterior_mean: -8.7813
        posterior_std: 8.9491
        ci_95_low: -26.3215
        ci_95_high: 8.7588
        prob_positive: 0.1632
    
      high_rnd_intensity_flag:
        n_obs: 6676.0000
        sample_mean: 0.1098
        sample_std: 0.3127
        posterior_mean: 0.1098
        posterior_std: 0.0038
        ci_95_low: 0.1023
        ci_95_high: 0.1173
        prob_positive: 1.0000
    
      rnd_cagr_3y:
        n_obs: 2353.0000
        sample_mean: 10.3636
        sample_std: 27.1231
        posterior_mean: 10.3313
        posterior_std: 0.5583
        ci_95_low: 9.2370
        ci_95_high: 11.4255
        prob_positive: 1.0000
    
      rnd_cut_flag:
        n_obs: 6676.0000
        sample_mean: 0.0422
        sample_std: 0.2012
        posterior_mean: 0.0422
        posterior_std: 0.0025
        ci_95_low: 0.0374
        ci_95_high: 0.0471
        prob_positive: 1.0000
    
      rnd_per_employee:
        n_obs: 5259.0000
        sample_mean: 0.0380
        sample_std: 0.2063
        posterior_mean: 0.0380
        posterior_std: 0.0028
        ci_95_low: 0.0324
        ci_95_high: 0.0436
        prob_positive: 1.0000
    
      rnd_increasing_flag:
        n_obs: 6676.0000
        sample_mean: 0.0801
        sample_std: 0.2715
        posterior_mean: 0.0801
        posterior_std: 0.0033
        ci_95_low: 0.0736
        ci_95_high: 0.0867
        prob_positive: 1.0000
    
      rnd_intensity_trend:
        n_obs: 6488.0000
        sample_mean: -62.3044
        sample_std: 4248.5883
        posterior_mean: -2.1617
        posterior_std: 9.8250
        ci_95_low: -21.4187
        ci_95_high: 17.0952
        prob_positive: 0.4129
    
      rnd_qoq_growth:
        n_obs: 2343.0000
        sample_mean: 11.0754
        sample_std: 297.3083
        posterior_mean: 8.0416
        posterior_std: 5.2337
        ci_95_low: -2.2165
        ci_95_high: 18.2998
        prob_positive: 0.9378
    
      rnd_to_gross_profit:
        n_obs: 6550.0000
        sample_mean: 30.1401
        sample_std: 1063.4123
        posterior_mean: 11.0546
        posterior_std: 7.9576
        ci_95_low: -4.5422
        ci_95_high: 26.6514
        prob_positive: 0.9176
    
      rnd_yoy_growth:
        n_obs: 2485.0000
        sample_mean: 25.7798
        sample_std: 470.0114
        posterior_mean: 13.6475
        posterior_std: 6.8601
        ci_95_low: 0.2016
        ci_95_high: 27.0933
        prob_positive: 0.9767
    
    ============================================================
      Bayesian Analysis: Employee Productivity (7 features)
    ============================================================
    
      revenue_per_employee:
        n_obs: 5259.0000
        sample_mean: 0.7898
        sample_std: 5.6696
        posterior_mean: 0.7898
        posterior_std: 0.0782
        ci_95_low: 0.6366
        ci_95_high: 0.9430
        prob_positive: 1.0000
    
      profit_per_employee:
        n_obs: 5259.0000
        sample_mean: 0.0404
        sample_std: 2.9140
        posterior_mean: 0.0404
        posterior_std: 0.0402
        ci_95_low: -0.0384
        ci_95_high: 0.1191
        prob_positive: 0.8426
    
      ebitda_per_employee:
        n_obs: 5259.0000
        sample_mean: 0.1877
        sample_std: 3.1941
        posterior_mean: 0.1877
        posterior_std: 0.0440
        ci_95_low: 0.1014
        ci_95_high: 0.2741
        prob_positive: 1.0000
    
      assets_per_employee:
        n_obs: 5259.0000
        sample_mean: 3.6601
        sample_std: 52.3229
        posterior_mean: 3.6412
        posterior_std: 0.7196
        ci_95_low: 2.2307
        ci_95_high: 5.0517
        prob_positive: 1.0000
    
      fte_growth_1y_pct:
        n_obs: 5741.0000
        sample_mean: 6.8247
        sample_std: 796.0620
        posterior_mean: 3.2439
        posterior_std: 7.2435
        ci_95_low: -10.9533
        ci_95_high: 17.4411
        prob_positive: 0.6729
    
      fte_growth_3y_pct:
        n_obs: 5519.0000
        sample_mean: 25.7934
        sample_std: 627.6685
        posterior_mean: 15.0500
        posterior_std: 6.4538
        ci_95_low: 2.4006
        ci_95_high: 27.6995
        prob_positive: 0.9901
    
      workforce_stability:
        n_obs: 1862.0000
        sample_mean: 1.6533
        sample_std: 18.6752
        posterior_mean: 1.6502
        posterior_std: 0.4324
        ci_95_low: 0.8027
        ci_95_high: 2.4977
        prob_positive: 0.9999
    
    ============================================================
      Bayesian Analysis: Employment Dynamics (7 features)
    ============================================================
    
      fte_acceleration:
        n_obs: 5386.0000
        sample_mean: 13.3018
        sample_std: 811.5384
        posterior_mean: 5.9843
        posterior_std: 7.4170
        ci_95_low: -8.5530
        ci_95_high: 20.5215
        prob_positive: 0.7901
    
      fte_growth_2y_pct:
        n_obs: 5612.0000
        sample_mean: 13.0865
        sample_std: 579.6432
        posterior_mean: 8.1858
        posterior_std: 6.1195
        ci_95_low: -3.8085
        ci_95_high: 20.1801
        prob_positive: 0.9095
    
      hiring_intensity:
        n_obs: 4464.0000
        sample_mean: -1.0371
        sample_std: 14.1985
        posterior_mean: -1.0367
        posterior_std: 0.2125
        ci_95_low: -1.4531
        ci_95_high: -0.6202
        prob_positive: 0.0000
    
      layoff_risk_flag:
        n_obs: 6676.0000
        sample_mean: 0.1107
        sample_std: 0.3138
        posterior_mean: 0.1107
        posterior_std: 0.0038
        ci_95_low: 0.1032
        ci_95_high: 0.1182
        prob_positive: 1.0000
    
      productivity_trend:
        n_obs: 4975.0000
        sample_mean: 33.1470
        sample_std: 612.2542
        posterior_mean: 18.9036
        posterior_std: 6.5552
        ci_95_low: 6.0554
        ci_95_high: 31.7517
        prob_positive: 0.9980
    
      rapid_hiring_flag:
        n_obs: 6676.0000
        sample_mean: 0.0816
        sample_std: 0.2738
        posterior_mean: 0.0816
        posterior_std: 0.0034
        ci_95_low: 0.0751
        ci_95_high: 0.0882
        prob_positive: 1.0000
    
      sustainable_growth_flag:
        n_obs: 6676.0000
        sample_mean: 0.2912
        sample_std: 0.4543
        posterior_mean: 0.2912
        posterior_std: 0.0056
        ci_95_low: 0.2803
        ci_95_high: 0.3021
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Cash flow (19 features)
    ============================================================
    
      self_funding_flag:
        n_obs: 6676.0000
        sample_mean: 0.6043
        sample_std: 0.4890
        posterior_mean: 0.6043
        posterior_std: 0.0060
        ci_95_low: 0.5925
        ci_95_high: 0.6160
        prob_positive: 1.0000
    
      capex_qoq_growth:
        n_obs: 5783.0000
        sample_mean: 107.8525
        sample_std: 2941.9785
        posterior_mean: 6.7548
        posterior_std: 9.6818
        ci_95_low: -12.2215
        ci_95_high: 25.7311
        prob_positive: 0.7573
    
      capex_yoy_growth:
        n_obs: 6543.0000
        sample_mean: 141.3275
        sample_std: 3454.7813
        posterior_mean: 7.3449
        posterior_std: 9.7367
        ci_95_low: -11.7390
        ci_95_high: 26.4288
        prob_positive: 0.7747
    
      cash_flow_quality_score:
        n_obs: 6676.0000
        sample_mean: 58.5980
        sample_std: 37.3820
        posterior_mean: 58.4756
        posterior_std: 0.4570
        ci_95_low: 57.5798
        ci_95_high: 59.3714
        prob_positive: 1.0000
    
      cff_share_of_cf:
        n_obs: 6090.0000
        sample_mean: 0.2941
        sample_std: 0.1756
        posterior_mean: 0.2941
        posterior_std: 0.0022
        ci_95_low: 0.2897
        ci_95_high: 0.2985
        prob_positive: 1.0000
    
      cfi_share_of_cf:
        n_obs: 6090.0000
        sample_mean: 0.2760
        sample_std: 0.1741
        posterior_mean: 0.2760
        posterior_std: 0.0022
        ci_95_low: 0.2717
        ci_95_high: 0.2804
        prob_positive: 1.0000
    
      cfo_share_of_cf:
        n_obs: 6090.0000
        sample_mean: 0.4299
        sample_std: 0.1681
        posterior_mean: 0.4299
        posterior_std: 0.0022
        ci_95_low: 0.4257
        ci_95_high: 0.4341
        prob_positive: 1.0000
    
      serial_acquirer_flag:
        n_obs: 6676.0000
        sample_mean: 0.3032
        sample_std: 0.4597
        posterior_mean: 0.3032
        posterior_std: 0.0056
        ci_95_low: 0.2921
        ci_95_high: 0.3142
        prob_positive: 1.0000
    
      sustainable_ma_flag:
        n_obs: 6676.0000
        sample_mean: 0.8065
        sample_std: 0.3951
        posterior_mean: 0.8065
        posterior_std: 0.0048
        ci_95_low: 0.7970
        ci_95_high: 0.8159
        prob_positive: 1.0000
    
      total_investment_to_cfo:
        n_obs: 6090.0000
        sample_mean: 1.3574
        sample_std: 10.4450
        posterior_mean: 1.3571
        posterior_std: 0.1338
        ci_95_low: 1.0948
        ci_95_high: 1.6194
        prob_positive: 1.0000
    
      underinvestment_flag:
        n_obs: 6676.0000
        sample_mean: 0.2710
        sample_std: 0.4445
        posterior_mean: 0.2710
        posterior_std: 0.0054
        ci_95_low: 0.2603
        ci_95_high: 0.2816
        prob_positive: 1.0000
    
      acquisition_pause_flag:
        n_obs: 6676.0000
        sample_mean: 0.1896
        sample_std: 0.3920
        posterior_mean: 0.1896
        posterior_std: 0.0048
        ci_95_low: 0.1802
        ci_95_high: 0.1990
        prob_positive: 1.0000
    
      acquisition_to_fcf:
        n_obs: 6090.0000
        sample_mean: 0.6046
        sample_std: 7.9881
        posterior_mean: 0.6045
        posterior_std: 0.1024
        ci_95_low: 0.4039
        ci_95_high: 0.8051
        prob_positive: 1.0000
    
      acquisitions_yoy_growth:
        n_obs: 2580.0000
        sample_mean: 1722.9658
        sample_std: 25599.3029
        posterior_mean: 0.6781
        posterior_std: 9.9980
        ci_95_low: -18.9181
        ci_95_high: 20.2742
        prob_positive: 0.5270
    
      capex_3y_trend:
        n_obs: 6297.0000
        sample_mean: 338.9514
        sample_std: 6520.9020
        posterior_mean: 4.9462
        posterior_std: 9.9268
        ci_95_low: -14.5103
        ci_95_high: 24.4027
        prob_positive: 0.6909
    
      capex_acceleration:
        n_obs: 6676.0000
        sample_mean: 0.2874
        sample_std: 0.4526
        posterior_mean: 0.2874
        posterior_std: 0.0055
        ci_95_low: 0.2766
        ci_95_high: 0.2983
        prob_positive: 1.0000
    
      capex_cut_flag:
        n_obs: 6676.0000
        sample_mean: 0.2076
        sample_std: 0.4056
        posterior_mean: 0.2076
        posterior_std: 0.0050
        ci_95_low: 0.1979
        ci_95_high: 0.2173
        prob_positive: 1.0000
    
      overinvestment_flag:
        n_obs: 6676.0000
        sample_mean: 0.2404
        sample_std: 0.4274
        posterior_mean: 0.2404
        posterior_std: 0.0052
        ci_95_low: 0.2302
        ci_95_high: 0.2507
        prob_positive: 1.0000
    
      ma_intensity_score:
        n_obs: 6339.0000
        sample_mean: 1.2591
        sample_std: 4.2370
        posterior_mean: 1.2591
        posterior_std: 0.0532
        ci_95_low: 1.1548
        ci_95_high: 1.3634
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Financial Distress (9 features)
    ============================================================
    
      distress_risk_score:
        n_obs: 6676.0000
        sample_mean: 66.0352
        sample_std: 43.2334
        posterior_mean: 65.8508
        posterior_std: 0.5284
        ci_95_low: 64.8152
        ci_95_high: 66.8865
        prob_positive: 1.0000
    
      liquidity_stress_score:
        n_obs: 6676.0000
        sample_mean: 9.1312
        sample_std: 13.1438
        posterior_mean: 9.1289
        posterior_std: 0.1608
        ci_95_low: 8.8136
        ci_95_high: 9.4441
        prob_positive: 1.0000
    
      working_capital_trend:
        n_obs: 6582.0000
        sample_mean: -0.0247
        sample_std: 6.3259
        posterior_mean: -0.0247
        posterior_std: 0.0780
        ci_95_low: -0.1775
        ci_95_high: 0.1281
        prob_positive: 0.3756
    
      cash_runway_months:
        n_obs: 6676.0000
        sample_mean: 141.6367
        sample_std: 1126.7805
        posterior_mean: 48.8101
        posterior_std: 8.0956
        ci_95_low: 32.9428
        ci_95_high: 64.6775
        prob_positive: 1.0000
    
      accumulated_deficit_flag:
        n_obs: 6676.0000
        sample_mean: 0.1982
        sample_std: 0.3987
        posterior_mean: 0.1982
        posterior_std: 0.0049
        ci_95_low: 0.1886
        ci_95_high: 0.2077
        prob_positive: 1.0000
    
      adequate_cash_buffer:
        n_obs: 6676.0000
        sample_mean: 0.9733
        sample_std: 0.1611
        posterior_mean: 0.9733
        posterior_std: 0.0020
        ci_95_low: 0.9695
        ci_95_high: 0.9772
        prob_positive: 1.0000
    
      combined_distress_score:
        n_obs: 6676.0000
        sample_mean: 69.9516
        sample_std: 39.6528
        posterior_mean: 69.7872
        posterior_std: 0.4847
        ci_95_low: 68.8371
        ci_95_high: 70.7373
        prob_positive: 1.0000
    
      wc_deteriorating_flag:
        n_obs: 6676.0000
        sample_mean: 0.0750
        sample_std: 0.2635
        posterior_mean: 0.0750
        posterior_std: 0.0032
        ci_95_low: 0.0687
        ci_95_high: 0.0814
        prob_positive: 1.0000
    
      retained_earnings_growth:
        n_obs: 6342.0000
        sample_mean: -0.0767
        sample_std: 2.5273
        posterior_mean: -0.0767
        posterior_std: 0.0317
        ci_95_low: -0.1389
        ci_95_high: -0.0145
        prob_positive: 0.0078
    
    ============================================================
      Bayesian Analysis: GAAP vs Adjusted (3 features)
    ============================================================
    
      earnings_quality_score:
        n_obs: 6676.0000
        sample_mean: 80.7174
        sample_std: 32.8876
        posterior_mean: 80.5868
        posterior_std: 0.4022
        ci_95_low: 79.7986
        ci_95_high: 81.3751
        prob_positive: 1.0000
    
      eps_adjustment_pct:
        n_obs: 3535.0000
        sample_mean: 88.2023
        sample_std: 729.5660
        posterior_mean: 35.2006
        posterior_std: 7.7518
        ci_95_low: 20.0070
        ci_95_high: 50.3942
        prob_positive: 1.0000
    
      net_income_adjustment_pct:
        n_obs: 6668.0000
        sample_mean: 34.7091
        sample_std: 926.9457
        posterior_mean: 15.1662
        posterior_std: 7.5037
        ci_95_low: 0.4590
        ci_95_high: 29.8734
        prob_positive: 0.9784
    
    ============================================================
      Bayesian Analysis: Growth Metrics (7 features)
    ============================================================
    
      revenue_growth_yoy:
        n_obs: 6501.0000
        sample_mean: 48.0060
        sample_std: 821.8262
        posterior_mean: 23.5449
        posterior_std: 7.1382
        ci_95_low: 9.5540
        ci_95_high: 37.5358
        prob_positive: 0.9995
    
      ebitda_growth_yoy:
        n_obs: 6609.0000
        sample_mean: 67.9206
        sample_std: 1390.8700
        posterior_mean: 17.2954
        posterior_std: 8.6334
        ci_95_low: 0.3739
        ci_95_high: 34.2169
        prob_positive: 0.9774
    
      operating_income_growth:
        n_obs: 6669.0000
        sample_mean: 5.1566
        sample_std: 291.4253
        posterior_mean: 4.5741
        posterior_std: 3.3610
        ci_95_low: -2.0135
        ci_95_high: 11.1616
        prob_positive: 0.9132
    
      fcf_growth:
        n_obs: 6507.0000
        sample_mean: -3.2551
        sample_std: 642.0939
        posterior_mean: -1.9926
        posterior_std: 6.2278
        ci_95_low: -14.1991
        ci_95_high: 10.2139
        prob_positive: 0.3745
    
      revenue_vs_5y_avg:
        n_obs: 6078.0000
        sample_mean: 1.2550
        sample_std: 0.5979
        posterior_mean: 1.2550
        posterior_std: 0.0077
        ci_95_low: 1.2400
        ci_95_high: 1.2700
        prob_positive: 1.0000
    
      growth_ebitda_growth_yoy:
        n_obs: 6609.0000
        sample_mean: 67.9206
        sample_std: 1390.8700
        posterior_mean: 17.2954
        posterior_std: 8.6334
        ci_95_low: 0.3739
        ci_95_high: 34.2169
        prob_positive: 0.9774
    
      revenue_momentum:
        n_obs: 6501.0000
        sample_mean: 51.1031
        sample_std: 858.2254
        posterior_mean: 23.9585
        posterior_std: 7.2882
        ci_95_low: 9.6737
        ci_95_high: 38.2433
        prob_positive: 0.9995
    
    ============================================================
      Bayesian Analysis: Interest Income (6 features)
    ============================================================
    
      interest_income_yoy_growth:
        n_obs: 5819.0000
        sample_mean: 0.5640
        sample_std: 7.5875
        posterior_mean: 0.5640
        posterior_std: 0.0995
        ci_95_low: 0.3690
        ci_95_high: 0.7589
        prob_positive: 1.0000
    
      interest_income_to_revenue_trend:
        n_obs: 6532.0000
        sample_mean: 0.0849
        sample_std: 2.4771
        posterior_mean: 0.0849
        posterior_std: 0.0306
        ci_95_low: 0.0248
        ci_95_high: 0.1450
        prob_positive: 0.9972
    
      interest_income_qoq_growth:
        n_obs: 4995.0000
        sample_mean: 1.6668
        sample_std: 36.8763
        posterior_mean: 1.6622
        posterior_std: 0.5211
        ci_95_low: 0.6409
        ci_95_high: 2.6835
        prob_positive: 0.9993
    
      interest_coverage_ratio:
        n_obs: 6142.0000
        sample_mean: -82.2464
        sample_std: 1181.5889
        posterior_mean: -25.1278
        posterior_std: 8.3336
        ci_95_low: -41.4616
        ci_95_high: -8.7940
        prob_positive: 0.0013
    
      interest_expense_to_revenue:
        n_obs: 6532.0000
        sample_mean: -6.5944
        sample_std: 171.0678
        posterior_mean: -6.3116
        posterior_std: 2.0708
        ci_95_low: -10.3703
        ci_95_high: -2.2529
        prob_positive: 0.0012
    
      interest_income_to_revenue:
        n_obs: 6532.0000
        sample_mean: 8.4910
        sample_std: 247.7108
        posterior_mean: 7.7619
        posterior_std: 2.9304
        ci_95_low: 2.0183
        ci_95_high: 13.5055
        prob_positive: 0.9960
    
    ============================================================
      Bayesian Analysis: Leverage & Liquidity (20 features)
    ============================================================
    
      debt_to_equity:
        n_obs: 6643.0000
        sample_mean: 1.2383
        sample_std: 27.8043
        posterior_mean: 1.2369
        posterior_std: 0.3409
        ci_95_low: 0.5686
        ci_95_high: 1.9051
        prob_positive: 0.9999
    
      debt_to_assets:
        n_obs: 6339.0000
        sample_mean: 0.2620
        sample_std: 0.3217
        posterior_mean: 0.2620
        posterior_std: 0.0040
        ci_95_low: 0.2541
        ci_95_high: 0.2700
        prob_positive: 1.0000
    
      equity_ratio:
        n_obs: 6339.0000
        sample_mean: 0.4706
        sample_std: 0.4439
        posterior_mean: 0.4706
        posterior_std: 0.0056
        ci_95_low: 0.4597
        ci_95_high: 0.4815
        prob_positive: 1.0000
    
      interest_coverage:
        n_obs: 6142.0000
        sample_mean: -82.2464
        sample_std: 1181.5889
        posterior_mean: -25.1278
        posterior_std: 8.3336
        ci_95_low: -41.4616
        ci_95_high: -8.7940
        prob_positive: 0.0013
    
      cash_ratio:
        n_obs: 6338.0000
        sample_mean: 6.8245
        sample_std: 454.5395
        posterior_mean: 5.1468
        posterior_std: 4.9582
        ci_95_low: -4.5714
        ci_95_high: 14.8649
        prob_positive: 0.8504
    
      working_capital_ratio:
        n_obs: 6339.0000
        sample_mean: 0.1772
        sample_std: 0.4229
        posterior_mean: 0.1772
        posterior_std: 0.0053
        ci_95_low: 0.1668
        ci_95_high: 0.1876
        prob_positive: 1.0000
    
      days_working_capital:
        n_obs: 6532.0000
        sample_mean: 1043.8465
        sample_std: 32083.2297
        posterior_mean: 0.6620
        posterior_std: 9.9968
        ci_95_low: -18.9318
        ci_95_high: 20.2558
        prob_positive: 0.5264
    
      debt_3y_cagr:
        n_obs: 6461.0000
        sample_mean: 15.0905
        sample_std: 81.3674
        posterior_mean: 14.9374
        posterior_std: 1.0071
        ci_95_low: 12.9635
        ci_95_high: 16.9114
        prob_positive: 1.0000
    
      debt_4q_trend:
        n_obs: 6433.0000
        sample_mean: 207.3816
        sample_std: 8544.0884
        posterior_mean: 1.8115
        posterior_std: 9.9562
        ci_95_low: -17.7027
        ci_95_high: 21.3257
        prob_positive: 0.5722
    
      debt_qoq_change:
        n_obs: 6516.0000
        sample_mean: 35.7309
        sample_std: 1728.8476
        posterior_mean: 6.3953
        posterior_std: 9.0610
        ci_95_low: -11.3642
        ci_95_high: 24.1549
        prob_positive: 0.7598
    
      debt_to_equity_trend:
        n_obs: 6649.0000
        sample_mean: 0.0702
        sample_std: 1.8855
        posterior_mean: 0.0702
        posterior_std: 0.0231
        ci_95_low: 0.0249
        ci_95_high: 0.1156
        prob_positive: 0.9988
    
      debt_yoy_change:
        n_obs: 6548.0000
        sample_mean: 201.3948
        sample_std: 8244.5101
        posterior_mean: 1.9216
        posterior_std: 9.9522
        ci_95_low: -17.5847
        ci_95_high: 21.4279
        prob_positive: 0.5766
    
      negative_wc_flag:
        n_obs: 6676.0000
        sample_mean: 0.1821
        sample_std: 0.3860
        posterior_mean: 0.1821
        posterior_std: 0.0047
        ci_95_low: 0.1729
        ci_95_high: 0.1914
        prob_positive: 1.0000
    
      wc_efficiency_score:
        n_obs: 6676.0000
        sample_mean: 63.7897
        sample_std: 27.3601
        posterior_mean: 63.7182
        posterior_std: 0.3347
        ci_95_low: 63.0623
        ci_95_high: 64.3742
        prob_positive: 1.0000
    
      wc_to_assets:
        n_obs: 6339.0000
        sample_mean: 17.7232
        sample_std: 42.2877
        posterior_mean: 17.6734
        posterior_std: 0.5304
        ci_95_low: 16.6338
        ci_95_high: 18.7129
        prob_positive: 1.0000
    
      wc_to_revenue:
        n_obs: 6532.0000
        sample_mean: 285.9853
        sample_std: 8789.9260
        posterior_mean: 2.3975
        posterior_std: 9.9580
        ci_95_low: -17.1201
        ci_95_high: 21.9152
        prob_positive: 0.5951
    
      wc_4q_trend:
        n_obs: 6328.0000
        sample_mean: 6.0192
        sample_std: 1854.2814
        posterior_mean: 0.9356
        posterior_std: 9.1900
        ci_95_low: -17.0769
        ci_95_high: 18.9480
        prob_positive: 0.5405
    
      wc_improving_flag:
        n_obs: 6676.0000
        sample_mean: 0.3054
        sample_std: 0.4606
        posterior_mean: 0.3054
        posterior_std: 0.0056
        ci_95_low: 0.2944
        ci_95_high: 0.3165
        prob_positive: 1.0000
    
      wc_qoq_change:
        n_obs: 6581.0000
        sample_mean: 7.9433
        sample_std: 1192.0374
        posterior_mean: 2.5144
        posterior_std: 8.2672
        ci_95_low: -13.6893
        ci_95_high: 18.7180
        prob_positive: 0.6195
    
      wc_yoy_change:
        n_obs: 6661.0000
        sample_mean: 8.6662
        sample_std: 1707.0438
        posterior_mean: 1.6124
        posterior_std: 9.0219
        ci_95_low: -16.0705
        ci_95_high: 19.2953
        prob_positive: 0.5709
    
    ============================================================
      Bayesian Analysis: Momentum & Technical (28 features)
    ============================================================
    
      daily_turnover_ratio:
        n_obs: 6668.0000
        sample_mean: 0.0093
        sample_std: 0.0194
        posterior_mean: 0.0093
        posterior_std: 0.0002
        ci_95_low: 0.0089
        ci_95_high: 0.0098
        prob_positive: 1.0000
    
      liquidity_score:
        n_obs: 6668.0000
        sample_mean: 1447705.6985
        sample_std: 8878675.9719
        posterior_mean: 0.0122
        posterior_std: 10.0000
        ci_95_low: -19.5878
        ci_95_high: 19.6122
        prob_positive: 0.5005
    
      price_momentum_1m:
        n_obs: 6447.0000
        sample_mean: 9.0973
        sample_std: 15.3585
        posterior_mean: 9.0940
        posterior_std: 0.1912
        ci_95_low: 8.7192
        ci_95_high: 9.4688
        prob_positive: 1.0000
    
      price_momentum_3m:
        n_obs: 6654.0000
        sample_mean: 4.0185
        sample_std: 31.4725
        posterior_mean: 4.0125
        posterior_std: 0.3855
        ci_95_low: 3.2568
        ci_95_high: 4.7681
        prob_positive: 1.0000
    
      price_momentum_6m:
        n_obs: 6613.0000
        sample_mean: 15.2108
        sample_std: 55.2401
        posterior_mean: 15.1410
        posterior_std: 0.6777
        ci_95_low: 13.8126
        ci_95_high: 16.4693
        prob_positive: 1.0000
    
      price_momentum_1y:
        n_obs: 6535.0000
        sample_mean: 55.1382
        sample_std: 149.3729
        posterior_mean: 53.3177
        posterior_std: 1.8170
        ci_95_low: 49.7564
        ci_95_high: 56.8791
        prob_positive: 1.0000
    
      price_momentum_5d:
        n_obs: 6668.0000
        sample_mean: 0.3738
        sample_std: 6.4224
        posterior_mean: 0.3738
        posterior_std: 0.0786
        ci_95_low: 0.2196
        ci_95_high: 0.5279
        prob_positive: 1.0000
    
      ema_crossover_20_50:
        n_obs: 6676.0000
        sample_mean: 0.0761
        sample_std: 0.9964
        posterior_mean: 0.0761
        posterior_std: 0.0122
        ci_95_low: 0.0522
        ci_95_high: 0.1000
        prob_positive: 1.0000
    
      ema_crossover_50_250:
        n_obs: 6676.0000
        sample_mean: 0.1956
        sample_std: 0.9708
        posterior_mean: 0.1956
        posterior_std: 0.0119
        ci_95_low: 0.1723
        ci_95_high: 0.2189
        prob_positive: 1.0000
    
      price_vs_ema_20d:
        n_obs: 6674.0000
        sample_mean: 0.0189
        sample_std: 0.0621
        posterior_mean: 0.0189
        posterior_std: 0.0008
        ci_95_low: 0.0175
        ci_95_high: 0.0204
        prob_positive: 1.0000
    
      price_vs_ema_250d:
        n_obs: 6547.0000
        sample_mean: 0.0941
        sample_std: 0.2837
        posterior_mean: 0.0941
        posterior_std: 0.0035
        ci_95_low: 0.0873
        ci_95_high: 0.1010
        prob_positive: 1.0000
    
      pct_off_52w_high:
        n_obs: 6674.0000
        sample_mean: -0.8016
        sample_std: 0.1543
        posterior_mean: -0.8016
        posterior_std: 0.0019
        ci_95_low: -0.8053
        ci_95_high: -0.7979
        prob_positive: 0.0000
    
      pct_above_52w_low:
        n_obs: 6674.0000
        sample_mean: -0.1929
        sample_std: 1.6123
        posterior_mean: -0.1929
        posterior_std: 0.0197
        ci_95_low: -0.2316
        ci_95_high: -0.1542
        prob_positive: 0.0000
    
      range_52w_position:
        n_obs: 6676.0000
        sample_mean: 0.5577
        sample_std: 0.2927
        posterior_mean: 0.5577
        posterior_std: 0.0036
        ci_95_low: 0.5506
        ci_95_high: 0.5647
        prob_positive: 1.0000
    
      beta_momentum:
        n_obs: 6189.0000
        sample_mean: -0.0008
        sample_std: 0.6933
        posterior_mean: -0.0008
        posterior_std: 0.0088
        ci_95_low: -0.0180
        ci_95_high: 0.0165
        prob_positive: 0.4655
    
      volatility_regime:
        n_obs: 6670.0000
        sample_mean: 1.0356
        sample_std: 0.3105
        posterior_mean: 1.0356
        posterior_std: 0.0038
        ci_95_low: 1.0282
        ci_95_high: 1.0431
        prob_positive: 1.0000
    
      volatility_trend_short:
        n_obs: 6670.0000
        sample_mean: 3.6131
        sample_std: 13.2515
        posterior_mean: 3.6121
        posterior_std: 0.1622
        ci_95_low: 3.2942
        ci_95_high: 3.9301
        prob_positive: 1.0000
    
      volatility_trend_long:
        n_obs: 6673.0000
        sample_mean: -1.2984
        sample_std: 14.1444
        posterior_mean: -1.2980
        posterior_std: 0.1731
        ci_95_low: -1.6373
        ci_95_high: -0.9587
        prob_positive: 0.0000
    
      vol_ratio_3m_1y:
        n_obs: 6673.0000
        sample_mean: 1.1187
        sample_std: 0.2213
        posterior_mean: 1.1187
        posterior_std: 0.0027
        ci_95_low: 1.1134
        ci_95_high: 1.1241
        prob_positive: 1.0000
    
      vol_hump:
        n_obs: 6673.0000
        sample_mean: -0.5521
        sample_std: 9.1522
        posterior_mean: -0.5520
        posterior_std: 0.1120
        ci_95_low: -0.7716
        ci_95_high: -0.3325
        prob_positive: 0.0000
    
      beta_term_structure:
        n_obs: 6172.0000
        sample_mean: -0.0162
        sample_std: 4.7924
        posterior_mean: -0.0162
        posterior_std: 0.0610
        ci_95_low: -0.1357
        ci_95_high: 0.1034
        prob_positive: 0.3954
    
      beta_convexity:
        n_obs: 6189.0000
        sample_mean: 0.0200
        sample_std: 0.4402
        posterior_mean: 0.0200
        posterior_std: 0.0056
        ci_95_low: 0.0091
        ci_95_high: 0.0310
        prob_positive: 0.9998
    
      realized_vs_implied_proxy:
        n_obs: 6670.0000
        sample_mean: 1.0356
        sample_std: 0.3105
        posterior_mean: 1.0356
        posterior_std: 0.0038
        ci_95_low: 1.0282
        ci_95_high: 1.0431
        prob_positive: 1.0000
    
      price_momentum_5y:
        n_obs: 5770.0000
        sample_mean: 124.3554
        sample_std: 508.1791
        posterior_mean: 85.9065
        posterior_std: 5.5604
        ci_95_low: 75.0081
        ci_95_high: 96.8050
        prob_positive: 1.0000
    
      long_term_trend_score:
        n_obs: 6676.0000
        sample_mean: 0.7581
        sample_std: 2.0089
        posterior_mean: 0.7581
        posterior_std: 0.0246
        ci_95_low: 0.7099
        ci_95_high: 0.8063
        prob_positive: 1.0000
    
      price_momentum_3y:
        n_obs: 6237.0000
        sample_mean: 97.4951
        sample_std: 294.2516
        posterior_mean: 85.6104
        posterior_std: 3.4914
        ci_95_low: 78.7672
        ci_95_high: 92.4536
        prob_positive: 1.0000
    
      multi_year_high_flag:
        n_obs: 6676.0000
        sample_mean: 0.3641
        sample_std: 0.4812
        posterior_mean: 0.3641
        posterior_std: 0.0059
        ci_95_low: 0.3526
        ci_95_high: 0.3757
        prob_positive: 1.0000
    
      secular_trend_flag:
        n_obs: 6676.0000
        sample_mean: 0.3902
        sample_std: 0.4878
        posterior_mean: 0.3902
        posterior_std: 0.0060
        ci_95_low: 0.3785
        ci_95_high: 0.4019
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Price Target Dynamics (14 features)
    ============================================================
    
      pt_momentum_1w:
        n_obs: 6662.0000
        sample_mean: 0.0040
        sample_std: 0.0443
        posterior_mean: 0.0040
        posterior_std: 0.0005
        ci_95_low: 0.0030
        ci_95_high: 0.0051
        prob_positive: 1.0000
    
      pt_momentum_1m:
        n_obs: 6633.0000
        sample_mean: 0.0111
        sample_std: 0.0975
        posterior_mean: 0.0111
        posterior_std: 0.0012
        ci_95_low: 0.0088
        ci_95_high: 0.0135
        prob_positive: 1.0000
    
      pt_momentum_3m:
        n_obs: 6583.0000
        sample_mean: 0.0595
        sample_std: 0.2401
        posterior_mean: 0.0595
        posterior_std: 0.0030
        ci_95_low: 0.0537
        ci_95_high: 0.0653
        prob_positive: 1.0000
    
      pt_momentum_6m:
        n_obs: 6465.0000
        sample_mean: 0.1454
        sample_std: 0.5325
        posterior_mean: 0.1454
        posterior_std: 0.0066
        ci_95_low: 0.1325
        ci_95_high: 0.1584
        prob_positive: 1.0000
    
      pt_momentum_1y:
        n_obs: 6350.0000
        sample_mean: 0.2950
        sample_std: 0.7980
        posterior_mean: 0.2950
        posterior_std: 0.0100
        ci_95_low: 0.2754
        ci_95_high: 0.3146
        prob_positive: 1.0000
    
      analyst_coverage_change_1m:
        n_obs: 6676.0000
        sample_mean: 0.0217
        sample_std: 0.6465
        posterior_mean: 0.0217
        posterior_std: 0.0079
        ci_95_low: 0.0062
        ci_95_high: 0.0372
        prob_positive: 0.9970
    
      analyst_coverage_change_3m:
        n_obs: 6676.0000
        sample_mean: 0.0016
        sample_std: 1.0815
        posterior_mean: 0.0016
        posterior_std: 0.0132
        ci_95_low: -0.0243
        ci_95_high: 0.0276
        prob_positive: 0.5495
    
      analyst_coverage_change_1y:
        n_obs: 6676.0000
        sample_mean: 0.3578
        sample_std: 2.2738
        posterior_mean: 0.3578
        posterior_std: 0.0278
        ci_95_low: 0.3033
        ci_95_high: 0.4124
        prob_positive: 1.0000
    
      pt_acceleration_long:
        n_obs: 6328.0000
        sample_mean: -0.2350
        sample_std: 0.6743
        posterior_mean: -0.2350
        posterior_std: 0.0085
        ci_95_low: -0.2516
        ci_95_high: -0.2184
        prob_positive: 0.0000
    
      pt_acceleration_short:
        n_obs: 6574.0000
        sample_mean: -0.0470
        sample_std: 0.1773
        posterior_mean: -0.0470
        posterior_std: 0.0022
        ci_95_low: -0.0513
        ci_95_high: -0.0427
        prob_positive: 0.0000
    
      pt_median_momentum_1m:
        n_obs: 6633.0000
        sample_mean: 0.0108
        sample_std: 0.1068
        posterior_mean: 0.0108
        posterior_std: 0.0013
        ci_95_low: 0.0082
        ci_95_high: 0.0134
        prob_positive: 1.0000
    
      pt_median_momentum_3m:
        n_obs: 6583.0000
        sample_mean: 0.0607
        sample_std: 0.2572
        posterior_mean: 0.0607
        posterior_std: 0.0032
        ci_95_low: 0.0545
        ci_95_high: 0.0669
        prob_positive: 1.0000
    
      pt_vs_price_momentum:
        n_obs: 6577.0000
        sample_mean: 0.0537
        sample_std: 0.2214
        posterior_mean: 0.0537
        posterior_std: 0.0027
        ci_95_low: 0.0483
        ci_95_high: 0.0590
        prob_positive: 1.0000
    
      analyst_coverage_trend:
        n_obs: 6676.0000
        sample_mean: -0.0015
        sample_std: 0.2120
        posterior_mean: -0.0015
        posterior_std: 0.0026
        ci_95_low: -0.0066
        ci_95_high: 0.0036
        prob_positive: 0.2818
    
    ============================================================
      Bayesian Analysis: Profitability (19 features)
    ============================================================
    
      operating_margin_pct:
        n_obs: 6532.0000
        sample_mean: -99.3349
        sample_std: 3191.7485
        posterior_mean: -5.9855
        posterior_std: 9.6940
        ci_95_low: -24.9858
        ci_95_high: 13.0148
        prob_positive: 0.2685
    
      ebitda_margin_pct:
        n_obs: 6532.0000
        sample_mean: -87.6807
        sample_std: 3098.8888
        posterior_mean: -5.5842
        posterior_std: 9.6763
        ci_95_low: -24.5498
        ci_95_high: 13.3814
        prob_positive: 0.2819
    
      roic:
        n_obs: 6643.0000
        sample_mean: 8.7468
        sample_std: 458.2033
        posterior_mean: 6.6463
        posterior_std: 4.9005
        ci_95_low: -2.9587
        ci_95_high: 16.2513
        prob_positive: 0.9125
    
      rnd_intensity:
        n_obs: 6532.0000
        sample_mean: 0.4475
        sample_std: 20.7235
        posterior_mean: 0.4472
        posterior_std: 0.2563
        ci_95_low: -0.0552
        ci_95_high: 0.9496
        prob_positive: 0.9595
    
      equity_multiplier:
        n_obs: 6643.0000
        sample_mean: 3.2285
        sample_std: 58.4603
        posterior_mean: 3.2120
        posterior_std: 0.7154
        ci_95_low: 1.8097
        ci_95_high: 4.6142
        prob_positive: 1.0000
    
      gross_margin_trend_yoy:
        n_obs: 6421.0000
        sample_mean: 0.0004
        sample_std: 0.0292
        posterior_mean: 0.0004
        posterior_std: 0.0004
        ci_95_low: -0.0003
        ci_95_high: 0.0011
        prob_positive: 0.8462
    
      operating_margin_trend:
        n_obs: 6531.0000
        sample_mean: -0.3399
        sample_std: 92.0226
        posterior_mean: -0.3356
        posterior_std: 1.1314
        ci_95_low: -2.5531
        ci_95_high: 1.8819
        prob_positive: 0.3834
    
      net_margin_trend_yoy:
        n_obs: 6336.0000
        sample_mean: 0.0020
        sample_std: 0.2137
        posterior_mean: 0.0020
        posterior_std: 0.0027
        ci_95_low: -0.0033
        ci_95_high: 0.0073
        prob_positive: 0.7716
    
      ebitda_margin_trend:
        n_obs: 6531.0000
        sample_mean: -0.3391
        sample_std: 91.3766
        posterior_mean: -0.3349
        posterior_std: 1.1235
        ci_95_low: -2.5370
        ci_95_high: 1.8673
        prob_positive: 0.3828
    
      margin_expansion_flag:
        n_obs: 6676.0000
        sample_mean: 0.0846
        sample_std: 0.2784
        posterior_mean: 0.0846
        posterior_std: 0.0034
        ci_95_low: 0.0780
        ci_95_high: 0.0913
        prob_positive: 1.0000
    
      ebit_cagr_3y:
        n_obs: 5234.0000
        sample_mean: 11.9584
        sample_std: 46.7493
        posterior_mean: 11.9087
        posterior_std: 0.6448
        ci_95_low: 10.6448
        ci_95_high: 13.1726
        prob_positive: 1.0000
    
      ebit_growth_yoy:
        n_obs: 6638.0000
        sample_mean: 70.1574
        sample_std: 1331.1072
        posterior_mean: 19.1204
        posterior_std: 8.5292
        ci_95_low: 2.4032
        ci_95_high: 35.8375
        prob_positive: 0.9875
    
      ebit_qoq_growth:
        n_obs: 6615.0000
        sample_mean: -8074.1707
        sample_std: 665617.8128
        posterior_mean: -0.0121
        posterior_std: 10.0000
        ci_95_low: -19.6120
        ci_95_high: 19.5879
        prob_positive: 0.4995
    
      ebitda_qoq_growth:
        n_obs: 6586.0000
        sample_mean: 12.4358
        sample_std: 2044.0101
        posterior_mean: 1.6934
        posterior_std: 9.2942
        ci_95_low: -16.5233
        ci_95_high: 19.9101
        prob_positive: 0.5723
    
      gp_margin_trend:
        n_obs: 5887.0000
        sample_mean: -29.6265
        sample_std: 1219.2114
        posterior_mean: -8.4046
        posterior_std: 8.4635
        ci_95_low: -24.9932
        ci_95_high: 8.1839
        prob_positive: 0.1603
    
      gp_qoq_growth:
        n_obs: 6492.0000
        sample_mean: 28.1160
        sample_std: 2773.2254
        posterior_mean: 2.1886
        posterior_std: 9.6029
        ci_95_low: -16.6331
        ci_95_high: 21.0103
        prob_positive: 0.5901
    
      gp_yoy_growth:
        n_obs: 6518.0000
        sample_mean: 47.6157
        sample_std: 790.5792
        posterior_mean: 24.3073
        posterior_std: 6.9965
        ci_95_low: 10.5941
        ci_95_high: 38.0205
        prob_positive: 0.9997
    
      ebitda_cagr_3y:
        n_obs: 5604.0000
        sample_mean: 10.7728
        sample_std: 38.1258
        posterior_mean: 10.7450
        posterior_std: 0.5086
        ci_95_low: 9.7480
        ci_95_high: 11.7419
        prob_positive: 1.0000
    
      margin_stability_score:
        n_obs: 6676.0000
        sample_mean: 99.9918
        sample_std: 0.1038
        posterior_mean: 99.9918
        posterior_std: 0.0013
        ci_95_low: 99.9893
        ci_95_high: 99.9943
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Quality & Risk (15 features)
    ============================================================
    
      net_buyback_flag:
        n_obs: 6676.0000
        sample_mean: 0.3460
        sample_std: 0.4757
        posterior_mean: 0.3460
        posterior_std: 0.0058
        ci_95_low: 0.3346
        ci_95_high: 0.3574
        prob_positive: 1.0000
    
      beta_trend:
        n_obs: 6172.0000
        sample_mean: 50.7265
        sample_std: 476.5544
        posterior_mean: 37.0819
        posterior_std: 5.1864
        ci_95_low: 26.9166
        ci_95_high: 47.2472
        prob_positive: 1.0000
    
      shares_yoy_change_pct:
        n_obs: 6483.0000
        sample_mean: 0.2863
        sample_std: 9.8699
        posterior_mean: 0.2862
        posterior_std: 0.1226
        ci_95_low: 0.0460
        ci_95_high: 0.5265
        prob_positive: 0.9902
    
      low_beta_flag:
        n_obs: 6676.0000
        sample_mean: 0.3646
        sample_std: 0.4814
        posterior_mean: 0.3646
        posterior_std: 0.0059
        ci_95_low: 0.3530
        ci_95_high: 0.3761
        prob_positive: 1.0000
    
      has_goodwill_impairment:
        n_obs: 6676.0000
        sample_mean: 0.0807
        sample_std: 0.2725
        posterior_mean: 0.0807
        posterior_std: 0.0033
        ci_95_low: 0.0742
        ci_95_high: 0.0873
        prob_positive: 1.0000
    
      has_asset_writedown:
        n_obs: 6676.0000
        sample_mean: 0.4148
        sample_std: 0.4927
        posterior_mean: 0.4148
        posterior_std: 0.0060
        ci_95_low: 0.4029
        ci_95_high: 0.4266
        prob_positive: 1.0000
    
      has_restructuring:
        n_obs: 6676.0000
        sample_mean: 0.2090
        sample_std: 0.4066
        posterior_mean: 0.2090
        posterior_std: 0.0050
        ci_95_low: 0.1992
        ci_95_high: 0.2187
        prob_positive: 1.0000
    
      goodwill_to_assets_pct:
        n_obs: 6339.0000
        sample_mean: 9.4134
        sample_std: 14.0200
        posterior_mean: 9.4105
        posterior_std: 0.1761
        ci_95_low: 9.0654
        ci_95_high: 9.7556
        prob_positive: 1.0000
    
      intangible_intensity:
        n_obs: 6339.0000
        sample_mean: 0.0804
        sample_std: 0.1603
        posterior_mean: 0.0804
        posterior_std: 0.0020
        ci_95_low: 0.0765
        ci_95_high: 0.0844
        prob_positive: 1.0000
    
      exceptional_items_to_ebitda:
        n_obs: 6621.0000
        sample_mean: 0.1799
        sample_std: 2.1615
        posterior_mean: 0.1799
        posterior_std: 0.0266
        ci_95_low: 0.1278
        ci_95_high: 0.2319
        prob_positive: 1.0000
    
      altman_z_trend:
        n_obs: 5586.0000
        sample_mean: -0.0303
        sample_std: 2.7337
        posterior_mean: -0.0303
        posterior_std: 0.0366
        ci_95_low: -0.1020
        ci_95_high: 0.0414
        prob_positive: 0.2036
    
      quick_ratio:
        n_obs: 6338.0000
        sample_mean: 20.1401
        sample_std: 1421.1640
        posterior_mean: 4.8105
        posterior_std: 8.7244
        ci_95_low: -12.2892
        ci_95_high: 21.9103
        prob_positive: 0.7093
    
      beta_spread:
        n_obs: 6189.0000
        sample_mean: -0.0008
        sample_std: 0.6933
        posterior_mean: -0.0008
        posterior_std: 0.0088
        ci_95_low: -0.0180
        ci_95_high: 0.0165
        prob_positive: 0.4655
    
      high_beta_flag:
        n_obs: 6676.0000
        sample_mean: 0.1228
        sample_std: 0.3283
        posterior_mean: 0.1228
        posterior_std: 0.0040
        ci_95_low: 0.1150
        ci_95_high: 0.1307
        prob_positive: 1.0000
    
      beta_stability_score:
        n_obs: 6676.0000
        sample_mean: 80.8728
        sample_std: 18.7495
        posterior_mean: 80.8303
        posterior_std: 0.2294
        ci_95_low: 80.3806
        ci_95_high: 81.2799
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Revenue Forecasting (16 features)
    ============================================================
    
      revenue_avg_med_diff_pct:
        n_obs: 6408.0000
        sample_mean: 1.7888
        sample_std: 78.2341
        posterior_mean: 1.7719
        posterior_std: 0.9727
        ci_95_low: -0.1346
        ci_95_high: 3.6783
        prob_positive: 0.9657
    
      revenue_revision_trend:
        n_obs: 6305.0000
        sample_mean: 0.7827
        sample_std: 33.7910
        posterior_mean: 0.7813
        posterior_std: 0.4252
        ci_95_low: -0.0520
        ci_95_high: 1.6146
        prob_positive: 0.9669
    
      estimate_confidence_score:
        n_obs: 6676.0000
        sample_mean: 99.0976
        sample_std: 5.0690
        posterior_mean: 99.0937
        posterior_std: 0.0620
        ci_95_low: 98.9722
        ci_95_high: 99.2153
        prob_positive: 1.0000
    
      consensus_revenue_growth:
        n_obs: 6531.0000
        sample_mean: 80.4638
        sample_std: 3400.1247
        posterior_mean: 4.3025
        posterior_std: 9.7290
        ci_95_low: -14.7662
        ci_95_high: 23.3713
        prob_positive: 0.6708
    
      revenue_acceleration:
        n_obs: 5909.0000
        sample_mean: 0.0702
        sample_std: 4.0857
        posterior_mean: 0.0702
        posterior_std: 0.0531
        ci_95_low: -0.0339
        ci_95_high: 0.1744
        prob_positive: 0.9068
    
      revenue_est_revision_trend:
        n_obs: 6305.0000
        sample_mean: 0.7827
        sample_std: 33.7910
        posterior_mean: 0.7813
        posterior_std: 0.4252
        ci_95_low: -0.0520
        ci_95_high: 1.6146
        prob_positive: 0.9669
    
      revenue_2y_growth:
        n_obs: 6425.0000
        sample_mean: 153.4225
        sample_std: 4293.0190
        posterior_mean: 5.1684
        posterior_std: 9.8301
        ci_95_low: -14.0987
        ci_95_high: 24.4354
        prob_positive: 0.7005
    
      revenue_3y_growth:
        n_obs: 6345.0000
        sample_mean: 196.0351
        sample_std: 5721.3288
        posterior_mean: 3.7276
        posterior_std: 9.9045
        ci_95_low: -15.6851
        ci_95_high: 23.1404
        prob_positive: 0.6467
    
      revenue_4q_trend:
        n_obs: 6326.0000
        sample_mean: 42.0922
        sample_std: 883.4431
        posterior_mean: 18.8437
        posterior_std: 7.4318
        ci_95_low: 4.2773
        ci_95_high: 33.4101
        prob_positive: 0.9944
    
      revenue_qoq_growth:
        n_obs: 6468.0000
        sample_mean: 28.5422
        sample_std: 1027.2806
        posterior_mean: 10.8461
        posterior_std: 7.8740
        ci_95_low: -4.5870
        ci_95_high: 26.2791
        prob_positive: 0.9158
    
      revenue_stability_score:
        n_obs: 6676.0000
        sample_mean: 73.5450
        sample_std: 27.0406
        posterior_mean: 73.4645
        posterior_std: 0.3308
        ci_95_low: 72.8162
        ci_95_high: 74.1128
        prob_positive: 1.0000
    
      revenue_accelerating_flag:
        n_obs: 6676.0000
        sample_mean: 0.6484
        sample_std: 0.4775
        posterior_mean: 0.6484
        posterior_std: 0.0058
        ci_95_low: 0.6370
        ci_95_high: 0.6599
        prob_positive: 1.0000
    
      revenue_cagr_3y:
        n_obs: 6326.0000
        sample_mean: 10.8010
        sample_std: 40.0047
        posterior_mean: 10.7737
        posterior_std: 0.5023
        ci_95_low: 9.7891
        ci_95_high: 11.7583
        prob_positive: 1.0000
    
      revenue_cagr_4y:
        n_obs: 6293.0000
        sample_mean: 12.1292
        sample_std: 31.9010
        posterior_mean: 12.1096
        posterior_std: 0.4018
        ci_95_low: 11.3221
        ci_95_high: 12.8972
        prob_positive: 1.0000
    
      revenue_growth_flag:
        n_obs: 6676.0000
        sample_mean: 0.7830
        sample_std: 0.4123
        posterior_mean: 0.7830
        posterior_std: 0.0050
        ci_95_low: 0.7731
        ci_95_high: 0.7928
        prob_positive: 1.0000
    
      revenue_4y_growth:
        n_obs: 6314.0000
        sample_mean: 750.0206
        sample_std: 24712.0986
        posterior_mean: 0.7747
        posterior_std: 9.9948
        ci_95_low: -18.8152
        ci_95_high: 20.3645
        prob_positive: 0.5309
    
    ============================================================
      Bayesian Analysis: Technical Analysis (11 features)
    ============================================================
    
      high_volume_flag:
        n_obs: 6676.0000
        sample_mean: 0.1010
        sample_std: 0.3013
        posterior_mean: 0.1010
        posterior_std: 0.0037
        ci_95_low: 0.0937
        ci_95_high: 0.1082
        prob_positive: 1.0000
    
      ema_slope_20d:
        n_obs: 6666.0000
        sample_mean: 0.0087
        sample_std: 0.0513
        posterior_mean: 0.0087
        posterior_std: 0.0006
        ci_95_low: 0.0075
        ci_95_high: 0.0099
        prob_positive: 1.0000
    
      price_vs_ema_100d:
        n_obs: 6637.0000
        sample_mean: 4.2425
        sample_std: 16.0420
        posterior_mean: 4.2409
        posterior_std: 0.1969
        ci_95_low: 3.8550
        ci_95_high: 4.6267
        prob_positive: 1.0000
    
      near_52w_low_flag:
        n_obs: 6676.0000
        sample_mean: 0.0467
        sample_std: 0.2111
        posterior_mean: 0.0467
        posterior_std: 0.0026
        ci_95_low: 0.0417
        ci_95_high: 0.0518
        prob_positive: 1.0000
    
      volume_momentum_score:
        n_obs: 6438.0000
        sample_mean: 0.0892
        sample_std: 0.2142
        posterior_mean: 0.0892
        posterior_std: 0.0027
        ci_95_low: 0.0840
        ci_95_high: 0.0944
        prob_positive: 1.0000
    
      breakout_signal:
        n_obs: 6676.0000
        sample_mean: 0.1576
        sample_std: 0.3644
        posterior_mean: 0.1576
        posterior_std: 0.0045
        ci_95_low: 0.1488
        ci_95_high: 0.1663
        prob_positive: 1.0000
    
      volatility_compression:
        n_obs: 6670.0000
        sample_mean: -0.1039
        sample_std: 21.9312
        posterior_mean: -0.1038
        posterior_std: 0.2684
        ci_95_low: -0.6300
        ci_95_high: 0.4223
        prob_positive: 0.3494
    
      volatility_term_structure:
        n_obs: 6673.0000
        sample_mean: 2.4026
        sample_std: 11.4356
        posterior_mean: 2.4021
        posterior_std: 0.1400
        ci_95_low: 2.1278
        ci_95_high: 2.6765
        prob_positive: 1.0000
    
      near_52w_high_flag:
        n_obs: 6676.0000
        sample_mean: 0.1601
        sample_std: 0.3668
        posterior_mean: 0.1601
        posterior_std: 0.0045
        ci_95_low: 0.1513
        ci_95_high: 0.1689
        prob_positive: 1.0000
    
      low_volume_flag:
        n_obs: 6676.0000
        sample_mean: 0.1204
        sample_std: 0.3255
        posterior_mean: 0.1204
        posterior_std: 0.0040
        ci_95_low: 0.1126
        ci_95_high: 0.1282
        prob_positive: 1.0000
    
      ema_trend_consistency:
        n_obs: 6676.0000
        sample_mean: 0.1502
        sample_std: 0.7830
        posterior_mean: 0.1502
        posterior_std: 0.0096
        ci_95_low: 0.1315
        ci_95_high: 0.1690
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Temporal Patterns (7 features)
    ============================================================
    
      days_to_earnings:
        n_obs: 6676.0000
        sample_mean: 20.8578
        sample_std: 28.3372
        posterior_mean: 20.8328
        posterior_std: 0.3466
        ci_95_low: 20.1534
        ci_95_high: 21.5121
        prob_positive: 1.0000
    
      earnings_report_recency:
        n_obs: 6668.0000
        sample_mean: 124.7133
        sample_std: 142.2865
        posterior_mean: 121.0383
        posterior_std: 1.7166
        ci_95_low: 117.6737
        ci_95_high: 124.4028
        prob_positive: 1.0000
    
      fiscal_year_progress:
        n_obs: 6585.0000
        sample_mean: 0.8715
        sample_std: 0.2289
        posterior_mean: 0.8715
        posterior_std: 0.0028
        ci_95_low: 0.8660
        ci_95_high: 0.8770
        prob_positive: 1.0000
    
      days_since_last_report:
        n_obs: 6668.0000
        sample_mean: 124.7133
        sample_std: 142.2865
        posterior_mean: 121.0383
        posterior_std: 1.7166
        ci_95_low: 117.6737
        ci_95_high: 124.4028
        prob_positive: 1.0000
    
      days_to_fy_end:
        n_obs: 6587.0000
        sample_mean: -181.4756
        sample_std: 136.3398
        posterior_mean: -176.4949
        posterior_std: 1.6567
        ci_95_low: -179.7420
        ci_95_high: -173.2479
        prob_positive: 0.0000
    
      reporting_freshness_score:
        n_obs: 6676.0000
        sample_mean: 2.9888
        sample_std: 13.7045
        posterior_mean: 2.9880
        posterior_std: 0.1677
        ci_95_low: 2.6593
        ci_95_high: 3.3167
        prob_positive: 1.0000
    
    ============================================================
      Bayesian Analysis: Valuation Ratios (7 features)
    ============================================================
    
      peg_ratio:
        n_obs: 4341.0000
        sample_mean: 0.1155
        sample_std: 19.6636
        posterior_mean: 0.1154
        posterior_std: 0.2983
        ci_95_low: -0.4693
        ci_95_high: 0.7001
        prob_positive: 0.6505
    
      tangible_book_per_share:
        n_obs: 6676.0000
        sample_mean: 0.0000
        sample_std: 0.0003
        posterior_mean: 0.0000
        posterior_std: 0.0000
        ci_95_low: 0.0000
        ci_95_high: 0.0000
        prob_positive: 0.9999
    
      price_to_tangible_book:
        n_obs: 5357.0000
        sample_mean: 9.2859
        sample_std: 23.7300
        posterior_mean: 9.2762
        posterior_std: 0.3240
        ci_95_low: 8.6411
        ci_95_high: 9.9113
        prob_positive: 1.0000
    
      tangible_equity_ratio:
        n_obs: 6339.0000
        sample_mean: 28.3818
        sample_std: 50.5440
        posterior_mean: 28.2679
        posterior_std: 0.6336
        ci_95_low: 27.0261
        ci_95_high: 29.5096
        prob_positive: 1.0000
    
      intangibles_to_equity:
        n_obs: 6643.0000
        sample_mean: 44.5366
        sample_std: 1527.8150
        posterior_mean: 9.8668
        posterior_std: 8.8230
        ci_95_low: -7.4264
        ci_95_high: 27.1599
        prob_positive: 0.8683
    
      goodwill_to_equity:
        n_obs: 6643.0000
        sample_mean: 24.5937
        sample_std: 218.9356
        posterior_mean: 22.9385
        posterior_std: 2.5942
        ci_95_low: 17.8539
        ci_95_high: 28.0232
        prob_positive: 1.0000
    
      tbv_yoy_growth:
        n_obs: 6649.0000
        sample_mean: 4.2925
        sample_std: 183.6881
        posterior_mean: 4.0852
        posterior_std: 2.1976
        ci_95_low: -0.2221
        ci_95_high: 8.3926
        prob_positive: 0.9685
    
    ============================================================
      Bayesian Analysis: Valuation Timeseries (15 features)
    ============================================================
    
      ev_sales_trend_1y:
        n_obs: 6426.0000
        sample_mean: 0.1308
        sample_std: 0.9544
        posterior_mean: 0.1308
        posterior_std: 0.0119
        ci_95_low: 0.1075
        ci_95_high: 0.1542
        prob_positive: 1.0000
    
      ev_ebitda_momentum:
        n_obs: 5926.0000
        sample_mean: 0.3417
        sample_std: 13.2744
        posterior_mean: 0.3416
        posterior_std: 0.1724
        ci_95_low: 0.0037
        ci_95_high: 0.6795
        prob_positive: 0.9762
    
      p_e_momentum_yoy:
        n_obs: 5162.0000
        sample_mean: 0.1803
        sample_std: 1.1268
        posterior_mean: 0.1803
        posterior_std: 0.0157
        ci_95_low: 0.1496
        ci_95_high: 0.2111
        prob_positive: 1.0000
    
      p_e_momentum_qoq:
        n_obs: 5165.0000
        sample_mean: 0.3033
        sample_std: 1.6411
        posterior_mean: 0.3033
        posterior_std: 0.0228
        ci_95_low: 0.2585
        ci_95_high: 0.3480
        prob_positive: 1.0000
    
      ev_sales_vs_3y_avg:
        n_obs: 6049.0000
        sample_mean: 0.1623
        sample_std: 0.5919
        posterior_mean: 0.1623
        posterior_std: 0.0076
        ci_95_low: 0.1474
        ci_95_high: 0.1772
        prob_positive: 1.0000
    
      ev_ebitda_vs_3y_avg:
        n_obs: 5533.0000
        sample_mean: 0.1262
        sample_std: 0.6209
        posterior_mean: 0.1262
        posterior_std: 0.0083
        ci_95_low: 0.1098
        ci_95_high: 0.1426
        prob_positive: 1.0000
    
      p_e_vs_3y_avg:
        n_obs: 4792.0000
        sample_mean: 0.2408
        sample_std: 0.9879
        posterior_mean: 0.2408
        posterior_std: 0.0143
        ci_95_low: 0.2128
        ci_95_high: 0.2688
        prob_positive: 1.0000
    
      ev_sales_forward_discount:
        n_obs: 6331.0000
        sample_mean: -0.0742
        sample_std: 0.9501
        posterior_mean: -0.0742
        posterior_std: 0.0119
        ci_95_low: -0.0976
        ci_95_high: -0.0508
        prob_positive: 0.0000
    
      ev_ebitda_forward_discount:
        n_obs: 5773.0000
        sample_mean: -0.1181
        sample_std: 0.6751
        posterior_mean: -0.1181
        posterior_std: 0.0089
        ci_95_low: -0.1355
        ci_95_high: -0.1007
        prob_positive: 0.0000
    
      p_e_forward_discount:
        n_obs: 4989.0000
        sample_mean: -0.1619
        sample_std: 0.8808
        posterior_mean: -0.1619
        posterior_std: 0.0125
        ci_95_low: -0.1863
        ci_95_high: -0.1375
        prob_positive: 0.0000
    
      p_b_vs_5y_avg:
        n_obs: 5469.0000
        sample_mean: 1.1656
        sample_std: 0.9911
        posterior_mean: 1.1656
        posterior_std: 0.0134
        ci_95_low: 1.1394
        ci_95_high: 1.1919
        prob_positive: 1.0000
    
      ev_sales_qoq_1q:
        n_obs: 6447.0000
        sample_mean: 0.1275
        sample_std: 1.1052
        posterior_mean: 0.1275
        posterior_std: 0.0138
        ci_95_low: 0.1006
        ci_95_high: 0.1545
        prob_positive: 1.0000
    
      p_b_momentum_yoy:
        n_obs: 6353.0000
        sample_mean: 0.1386
        sample_std: 0.7209
        posterior_mean: 0.1386
        posterior_std: 0.0090
        ci_95_low: 0.1209
        ci_95_high: 0.1563
        prob_positive: 1.0000
    
      forward_pe_premium:
        n_obs: 4989.0000
        sample_mean: -16.1907
        sample_std: 88.0769
        posterior_mean: -15.9428
        posterior_std: 1.2374
        ci_95_low: -18.3681
        ci_95_high: -13.5175
        prob_positive: 0.0000
    
      ev_ebitda_qoq_trend:
        n_obs: 5933.0000
        sample_mean: 0.4104
        sample_std: 17.2651
        posterior_mean: 0.4102
        posterior_std: 0.2241
        ci_95_low: -0.0290
        ci_95_high: 0.8494
        prob_positive: 0.9664

## 23. Distribution Fitting by Category

```python
# Fit statistical distributions to ALL feature categories
# (reuses the same direct_features exclusion set built in cell 22)

for cat_name, features in FEATURE_CATEGORIES.items():
    available = [f for f in features if f in df.columns and f not in direct_features]
    if available:
        print(f"\n{'=' * 60}")
        print(f"  Distribution Fitting: {cat_name} ({len(available)} features)")
        print(f"{'=' * 60}")
        result = fit_distributions_by_category(df, cat_name, available, n_simulations=5000)
        if isinstance(result, dict):
            for feat, info in result.items():
                if isinstance(info, dict) and "best_distribution" in info:
                    print(
                        f"  {feat}: best_distribution={info['best_distribution']}, "
                        f"aic={info.get('aic', 'N/A'):.4f}, "
                        f"params={info.get('params', 'N/A')}, "
                        f"simulated_mean={info.get('simulated_mean', 'N/A'):.4f}, "
                        f"simulated_std={info.get('simulated_std', 'N/A'):.4f}, "
                        f"cvar_5_pct={info.get('cvar_5_pct', 'N/A'):.4f}"
                    )
    else:
        skipped_reason = "all features are 'direct' or not in df"
        print(f"\n  [Skipped] {cat_name}: {skipped_reason}")
```

    ============================================================
      Distribution Fitting: Accounting Quality (20 features)
    ============================================================
      asset_sale_frequency: best_distribution=skew_normal, aic=32839.6510, params=(np.float64(5885911.233500209), np.float64(-4.945915728539893e-06), np.float64(5.659979394158814)), simulated_mean=4.4239, simulated_std=3.3424, cvar_5_pct=0.1805
      asset_sale_trend: best_distribution=skew_normal, aic=47698.2230, params=(np.float64(-1.0596809667609488), np.float64(6.123760687412325), np.float64(11.42359501126373)), simulated_mean=-0.6756, simulated_std=9.3911, cvar_5_pct=-20.6967
      tax_rate_yoy_change: best_distribution=skew_normal, aic=-4591.9732, params=(np.float64(0.9581548744636401), np.float64(-0.1208854408904512), np.float64(0.2046081141041881)), simulated_mean=-0.0101, simulated_std=0.1698, cvar_5_pct=-0.3501
      tax_rate_qoq_change: best_distribution=student_t, aic=-14464.1487, params=(np.float64(0.49021395544654167), np.float64(-0.00021243414211402317), np.float64(0.008087292312664127)), simulated_mean=9.4017, simulated_std=621.9172, cvar_5_pct=-53.3022
      tax_rate_stability: best_distribution=student_t, aic=-2636.0906, params=(np.float64(1.1142305447579246), np.float64(0.06305251637215939), np.float64(0.07327803310818143)), simulated_mean=0.0805, simulated_std=1.8175, cvar_5_pct=-1.6430
      low_tax_flag: best_distribution=skew_normal, aic=2246.4429, params=(np.float64(3015718.3555330914), np.float64(-9.722960475892933e-07), np.float64(0.5723927325434768)), simulated_mean=0.4681, simulated_std=0.3517, cvar_5_pct=0.0186
      tax_rate_trend_4q: best_distribution=student_t, aic=-9906.4622, params=(np.float64(0.8068869944396324), np.float64(-0.001677213425967685), np.float64(0.028370792643650965)), simulated_mean=-0.3734, simulated_std=26.6463, cvar_5_pct=-10.5189
      goodwill_change_rate: best_distribution=student_t, aic=2300.7623, params=(np.float64(0.7292097238914379), np.float64(0.0405298124183992), np.float64(0.06656188842888566)), simulated_mean=-0.0020, simulated_std=47.6727, cvar_5_pct=-23.7421
      restructuring_intensity: best_distribution=skew_normal, aic=-61290.5066, params=(np.float64(-4699104.73000681), np.float64(3.924389419877353e-09), np.float64(0.003503976141878355)), simulated_mean=-0.0028, simulated_std=0.0021, cvar_5_pct=-0.0081
      exceptional_items_frequency: best_distribution=skew_normal, aic=8428.5956, params=(np.float64(20923472.76484959), np.float64(-2.369040045510137e-07), np.float64(0.9093198137180418)), simulated_mean=0.7124, simulated_std=0.5433, cvar_5_pct=0.0280
      merger_impact_ratio: best_distribution=skew_normal, aic=-55036.8391, params=(np.float64(-9267404.669373838), np.float64(4.2539446822300165e-09), np.float64(0.007211944921245104)), simulated_mean=-0.0057, simulated_std=0.0043, cvar_5_pct=-0.0170
      non_operating_income_share: best_distribution=student_t, aic=-8371.9195, params=(np.float64(1.0012363991981617), np.float64(0.03893387153323662), np.float64(0.04224861655893213)), simulated_mean=0.0797, simulated_std=5.5055, cvar_5_pct=-1.8624
      asset_sale_boost: best_distribution=skew_normal, aic=1686.3232, params=(np.float64(22371878.89682837), np.float64(-1.4499979928348488e-07), np.float64(0.5488560820696771)), simulated_mean=0.4375, simulated_std=0.3272, cvar_5_pct=0.0173
      accounting_quality_score: best_distribution=skew_normal, aic=50446.6358, params=(np.float64(-8572730.888942175), np.float64(100.00001379210876), np.float64(21.568248671533134)), simulated_mean=82.9395, simulated_std=13.2127, cvar_5_pct=48.9628
      goodwill_3y_growth: best_distribution=student_t, aic=50190.4025, params=(np.float64(0.7727526906914292), np.float64(2.261930205123017), np.float64(14.205469774778383)), simulated_mean=31.2557, simulated_std=2314.6956, cvar_5_pct=-1818.5657
      goodwill_qoq_change: best_distribution=student_t, aic=21781.7665, params=(np.float64(0.3257792668953061), np.float64(0.02932311898539111), np.float64(0.11428099669013203)), simulated_mean=-39556653.3226, simulated_std=3078563189.0189, cvar_5_pct=-973941146.8075
      goodwill_to_assets_trend: best_distribution=skew_normal, aic=25686.4524, params=(np.float64(0.8735864513139842), np.float64(-1.1461996844675588), np.float64(2.0868383030101256)), simulated_mean=-0.0371, simulated_std=1.7813, cvar_5_pct=-3.6391
      goodwill_yoy_change: best_distribution=student_t, aic=41708.6737, params=(np.float64(0.8805819748706405), np.float64(3.0380594920779167), np.float64(5.916823492847165)), simulated_mean=8.3048, simulated_std=565.3573, cvar_5_pct=-327.8548
      impairment_risk_score: best_distribution=skew_normal, aic=56748.0672, params=(np.float64(18421125.681128316), np.float64(-9.789430851483316e-06), np.float64(33.920613855509245)), simulated_mean=26.9839, simulated_std=20.7238, cvar_5_pct=1.0278
      recent_acquisition_flag: best_distribution=skew_normal, aic=-17132.6246, params=(np.float64(7164236.245738521), np.float64(-1.0777893546961347e-07), np.float64(0.13407260882805339)), simulated_mean=0.1058, simulated_std=0.0798, cvar_5_pct=0.0044
    
    ============================================================
      Distribution Fitting: Analyst Sentiment (24 features)
    ============================================================
      analyst_count_stability: best_distribution=student_t, aic=-1936.1175, params=(np.float64(1.235093023317022), np.float64(1.0088055908164648), np.float64(0.08527301142033492)), simulated_mean=1.0936, simulated_std=3.1402, cvar_5_pct=-0.1037
      pt_accuracy_1y: best_distribution=skew_normal, aic=2699.0817, params=(np.float64(1187930450.8954368), np.float64(0.004301831863842923), np.float64(0.6008711348518752)), simulated_mean=0.4891, simulated_std=0.3702, cvar_5_pct=0.0239
      pt_optimism_bias: best_distribution=skew_normal, aic=8638.1127, params=(np.float64(-7.415031054991534), np.float64(0.562843413459582), np.float64(0.8844475826796583)), simulated_mean=-0.1395, simulated_std=0.5385, cvar_5_pct=-1.5201
      pt_high_low_convergence_1y: best_distribution=student_t, aic=-317.2980, params=(np.float64(2.983542034946716), np.float64(-0.002963141856263615), np.float64(0.16505056040292665)), simulated_mean=-0.0063, simulated_std=0.2995, cvar_5_pct=-0.6703
      eps_gaap_vs_norm_ntm: best_distribution=skew_normal, aic=5539.1937, params=(np.float64(-4.873587793404674), np.float64(0.25088743938563973), np.float64(0.7076628270928174)), simulated_mean=-0.3007, simulated_std=0.4416, cvar_5_pct=-1.4123
      eps_gaap_vs_norm_fy1e: best_distribution=skew_normal, aic=5661.4657, params=(np.float64(-5.018060968258823), np.float64(0.24930624792348222), np.float64(0.7229976840363567)), simulated_mean=-0.3252, simulated_std=0.4513, cvar_5_pct=-1.4507
      forward_adjustment_trend: best_distribution=student_t, aic=9328.9891, params=(np.float64(0.4588919529792942), np.float64(-0.0023832263415243206), np.float64(0.08297213100289405)), simulated_mean=2280.8939, simulated_std=97689.1341, cvar_5_pct=-3905.2045
      ebitda_forward_growth: best_distribution=student_t, aic=10773.7979, params=(np.float64(1.1517559603855365), np.float64(0.18843653806865895), np.float64(0.21219588484962676)), simulated_mean=0.5919, simulated_std=23.2380, cvar_5_pct=-4.0970
      earnings_revision_divergence: best_distribution=student_t, aic=-13663.0660, params=(np.float64(0.6522876848196578), np.float64(0.0004146979188245216), np.float64(0.009474722604193013)), simulated_mean=-0.0604, simulated_std=8.8824, cvar_5_pct=-5.6550
      forward_pe_vs_sector_proxy: best_distribution=skew_normal, aic=5899.4586, params=(np.float64(5.593525639790871), np.float64(-0.7397679228681554), np.float64(0.7619231007213075)), simulated_mean=-0.1499, simulated_std=0.4685, cvar_5_pct=-0.8390
      pt_achievement_1y: best_distribution=skew_normal, aic=-8757.4646, params=(np.float64(-13022872.988543287), np.float64(1.0000001063121946), np.float64(0.24104725725422416)), simulated_mean=0.8080, simulated_std=0.1423, cvar_5_pct=0.4527
      pt_range_hit_rate: best_distribution=skew_normal, aic=2019.7924, params=(np.float64(6595233.153713169), np.float64(-4.616585731951974e-07), np.float64(0.5627064379810742)), simulated_mean=0.4480, simulated_std=0.3362, cvar_5_pct=0.0194
      pt_median_vs_mean_spread: best_distribution=student_t, aic=-26746.8956, params=(np.float64(1.4663069869789345), np.float64(-0.0006935223670945215), np.float64(0.014823596107648194)), simulated_mean=0.0005, simulated_std=0.1043, cvar_5_pct=-0.1465
      pe_forward_discount: best_distribution=student_t, aic=2448.4904, params=(np.float64(3.3519999820444246), np.float64(-0.23287153655320927), np.float64(0.22636684563281237)), simulated_mean=-0.2304, simulated_std=0.3322, cvar_5_pct=-1.0040
      analyst_bullish_pct: best_distribution=skew_normal, aic=59762.1137, params=(np.float64(-43566680.94848601), np.float64(100.00000563126355), np.float64(45.06421002919808)), simulated_mean=63.4241, simulated_std=27.3942, cvar_5_pct=-5.5093
      analyst_bearish_pct: best_distribution=skew_normal, aic=42294.4565, params=(np.float64(24669679.594315737), np.float64(-2.732827306215143e-06), np.float64(11.881789535848242)), simulated_mean=9.4801, simulated_std=7.0746, cvar_5_pct=0.3949
      analyst_neutral_pct: best_distribution=skew_normal, aic=57722.5680, params=(np.float64(17510438.15547431), np.float64(-1.1297446958330817e-05), np.float64(36.55585967460176)), simulated_mean=29.6459, simulated_std=22.6104, cvar_5_pct=1.1384
      upside_potential: best_distribution=skew_normal, aic=61208.8718, params=(np.float64(3.1316829160977937), np.float64(-8.535301166965416), np.float64(42.017107323050325)), simulated_mean=23.3832, simulated_std=27.3566, cvar_5_pct=-21.5376
      price_target_spread_pct: best_distribution=skew_normal, aic=60987.2883, params=(np.float64(117779071.51251838), np.float64(-2.230369574521002e-06), np.float64(48.79783294989346)), simulated_mean=39.1156, simulated_std=29.0780, cvar_5_pct=1.6087
      price_target_revision_1m: best_distribution=skew_normal, aic=-19172.2112, params=(np.float64(1.8826430887702497), np.float64(-0.04617140916300629), np.float64(0.07948442604640077)), simulated_mean=0.0102, simulated_std=0.0569, cvar_5_pct=-0.0931
      price_target_revision_3m: best_distribution=student_t, aic=-6733.6401, params=(np.float64(2.1023293319358656), np.float64(0.020279768283105657), np.float64(0.08584898059960752)), simulated_mean=0.0219, simulated_std=0.2370, cvar_5_pct=-0.4588
      eps_revision_momentum: best_distribution=skew_normal, aic=-11188.3939, params=(np.float64(2.0425206225642665), np.float64(-0.09671261570916484), np.float64(0.1517339027084552)), simulated_mean=0.0126, simulated_std=0.1037, cvar_5_pct=-0.1720
      analyst_rating_normalized: best_distribution=skew_normal, aic=56104.7977, params=(np.float64(-27892392.353792414), np.float64(100.00000633272049), np.float64(32.3206062903124)), simulated_mean=73.8779, simulated_std=19.3810, cvar_5_pct=25.2424
      analyst_coverage_quality: best_distribution=skew_normal, aic=10821.1831, params=(np.float64(489705660.2197225), np.float64(0.11367219657450073), np.float64(1.1060608077764265)), simulated_mean=1.0002, simulated_std=0.6682, cvar_5_pct=0.1463
    
    ============================================================
      Distribution Fitting: Balance Sheet (24 features)
    ============================================================
      cash_to_assets_pct: best_distribution=skew_normal, aic=44716.1756, params=(np.float64(1220645433.1579084), np.float64(0.1037862309058124), np.float64(17.69867794946842)), simulated_mean=14.1585, simulated_std=10.5482, cvar_5_pct=0.6183
      cash_change_qoq: best_distribution=skew_normal, aic=1890.1690, params=(np.float64(-1.1828145117184228), np.float64(0.1805669932236944), np.float64(0.35335421332893957)), simulated_mean=-0.0405, simulated_std=0.2844, cvar_5_pct=-0.6605
      cash_vs_5y_avg: best_distribution=skew_normal, aic=11445.6818, params=(np.float64(2.821383286345268), np.float64(0.3764871886952932), np.float64(0.9916272502503964)), simulated_mean=1.1207, simulated_std=0.6544, cvar_5_pct=0.0310
      inventory_change_yoy: best_distribution=skew_normal, aic=-14011.2915, params=(np.float64(-2.3708258410804834), np.float64(0.04537167098942376), np.float64(0.0975390607825731)), simulated_mean=-0.0263, simulated_std=0.0661, cvar_5_pct=-0.1825
      inventory_vs_5y_avg: best_distribution=student_t, aic=7824.6260, params=(np.float64(11.035151043088597), np.float64(1.0613162451012528), np.float64(0.4960394279623883)), simulated_mean=1.0780, simulated_std=0.5402, cvar_5_pct=-0.0840
      working_capital_vs_5y_avg: best_distribution=student_t, aic=17208.8733, params=(np.float64(1.4916350865056685), np.float64(1.110110226308667), np.float64(0.4616917760593586)), simulated_mean=1.0051, simulated_std=4.3824, cvar_5_pct=-6.0112
      retained_earnings_vs_5y: best_distribution=student_t, aic=13078.8064, params=(np.float64(1.486969129294671), np.float64(1.2128744466340637), np.float64(0.35173614266000996)), simulated_mean=1.1869, simulated_std=2.0218, cvar_5_pct=-2.3145
      intangibles_growth_flag: best_distribution=skew_normal, aic=-5208.0096, params=(np.float64(1090431.8371753814), np.float64(-1.5577398707543137e-06), np.float64(0.3274290630016896)), simulated_mean=0.2637, simulated_std=0.1971, cvar_5_pct=0.0107
      asset_quality_score: best_distribution=student_t, aic=60051.1089, params=(np.float64(4.250710613776004), np.float64(56.16823401616029), np.float64(16.969138923514848)), simulated_mean=56.3420, simulated_std=23.0897, cvar_5_pct=3.5789
      balance_sheet_strength: best_distribution=skew_normal, aic=63059.0359, params=(np.float64(-7926475.076229122), np.float64(100.00003756138773), np.float64(54.40632830567185)), simulated_mean=56.2518, simulated_std=33.1157, cvar_5_pct=-27.9134
      debt_maturity_risk: best_distribution=student_t, aic=32694.1241, params=(np.float64(2.377152701598251), np.float64(1.9765885893352837), np.float64(1.9128178757006895)), simulated_mean=2.0491, simulated_std=4.7040, cvar_5_pct=-7.6829
      receivables_change_yoy: best_distribution=student_t, aic=3368.3185, params=(np.float64(1.6124589162081961), np.float64(0.08999051340454708), np.float64(0.16004279316826142)), simulated_mean=0.0840, simulated_std=1.1021, cvar_5_pct=-1.4655
      inventory_4q_trend: best_distribution=student_t, aic=51419.9087, params=(np.float64(1.643819019093037), np.float64(8.328766041061066), np.float64(15.086124601564494)), simulated_mean=7.8329, simulated_std=73.7942, cvar_5_pct=-130.6926
      inventory_buildup_flag: best_distribution=skew_normal, aic=-910.2620, params=(np.float64(5262138.151012789), np.float64(-4.800379535224813e-07), np.float64(0.4518936114211589)), simulated_mean=0.3589, simulated_std=0.2748, cvar_5_pct=0.0140
      inventory_days: best_distribution=skew_normal, aic=71737.7809, params=(np.float64(7876033.046491452), np.float64(-8.906952245972061e-05), np.float64(133.93219674821137)), simulated_mean=105.6566, simulated_std=79.1427, cvar_5_pct=4.2137
      inventory_qoq_change: best_distribution=student_t, aic=45937.6632, params=(np.float64(0.9799910743742271), np.float64(-0.06434760915876762), np.float64(4.916278315777406)), simulated_mean=-5.2713, simulated_std=190.8682, cvar_5_pct=-248.3570
      inventory_reduction_flag: best_distribution=skew_normal, aic=-3031.3677, params=(np.float64(16994910.110091295), np.float64(-1.2930707855010737e-07), np.float64(0.38548460559056796)), simulated_mean=0.3118, simulated_std=0.2387, cvar_5_pct=0.0132
      inventory_to_assets: best_distribution=skew_normal, aic=41809.0012, params=(np.float64(11634450.425629243), np.float64(-6.52669101590148e-06), np.float64(13.532387108758321)), simulated_mean=10.7387, simulated_std=8.1287, cvar_5_pct=0.4207
      inventory_to_revenue: best_distribution=skew_normal, aic=47781.7562, params=(np.float64(47837796.65130736), np.float64(-2.196435684360205e-06), np.float64(19.463994221910355)), simulated_mean=15.3613, simulated_std=11.5904, cvar_5_pct=0.6096
      inventory_yoy_change: best_distribution=student_t, aic=53762.7525, params=(np.float64(1.7337374511694585), np.float64(7.696291803479161), np.float64(14.852261632578603)), simulated_mean=7.1906, simulated_std=49.7748, cvar_5_pct=-101.1482
      asset_growth_accel: best_distribution=student_t, aic=59878.1291, params=(np.float64(1.7794354368635736), np.float64(6.85999499562781), np.float64(13.05791161629551)), simulated_mean=6.8989, simulated_std=87.7177, cvar_5_pct=-106.4140
      assets_3y_cagr: best_distribution=student_t, aic=52792.9591, params=(np.float64(1.7621738523170962), np.float64(5.32759748491792), np.float64(7.510539186967569)), simulated_mean=4.6987, simulated_std=37.9268, cvar_5_pct=-60.6444
      assets_qoq_growth: best_distribution=student_t, aic=45824.4540, params=(np.float64(0.9306151754346883), np.float64(0.7113202750207215), np.float64(2.4557558845719667)), simulated_mean=8.4366, simulated_std=640.0820, cvar_5_pct=-146.3017
      assets_yoy_growth: best_distribution=student_t, aic=58046.4811, params=(np.float64(1.5422979798117538), np.float64(8.398064434980487), np.float64(9.687758730412657)), simulated_mean=6.4084, simulated_std=123.8700, cvar_5_pct=-124.7293
    
    ============================================================
      Distribution Fitting: Cash Flow (32 features)
    ============================================================
      fcf_est_cagr_5y: best_distribution=student_t, aic=13037.5922, params=(np.float64(2.203673447678471), np.float64(11.689114944778627), np.float64(9.412589772460018)), simulated_mean=11.9782, simulated_std=20.8409, cvar_5_pct=-34.8310
      fcf_est_trend: best_distribution=skew_normal, aic=18154.4441, params=(np.float64(1.3963365265949013), np.float64(-1.5776451484993266), np.float64(1.5949557984318026)), simulated_mean=-0.5226, simulated_std=1.2158, cvar_5_pct=-2.8661
      cfo_to_net_income: best_distribution=student_t, aic=28841.1918, params=(np.float64(1.4192866296197186), np.float64(1.2744282206114015), np.float64(1.0136790993108793)), simulated_mean=1.2896, simulated_std=9.1160, cvar_5_pct=-11.0488
      fcf_to_net_income: best_distribution=student_t, aic=25578.0835, params=(np.float64(1.436796890097249), np.float64(0.7743473802989083), np.float64(0.797492252957344)), simulated_mean=0.6942, simulated_std=5.4100, cvar_5_pct=-8.8288
      fcf_margin: best_distribution=student_t, aic=-5930.7880, params=(np.float64(1.4631562916229233), np.float64(0.06090298540757289), np.float64(0.07189031492629346)), simulated_mean=0.0475, simulated_std=0.4994, cvar_5_pct=-0.8532
      cfo_growth_yoy: best_distribution=student_t, aic=16647.0291, params=(np.float64(1.5220012419841888), np.float64(0.03838608697644097), np.float64(0.4234292180531859)), simulated_mean=-0.0236, simulated_std=4.4219, cvar_5_pct=-5.4096
      fcf_positive_ratio: best_distribution=skew_normal, aic=1019.3573, params=(np.float64(-10743291.251648106), np.float64(1.000000257190643), np.float64(0.522037640113793)), simulated_mean=0.5834, simulated_std=0.3163, cvar_5_pct=-0.2145
      acquisition_intensity: best_distribution=skew_normal, aic=81785.1899, params=(np.float64(7457978.924880296), np.float64(-0.00017635162445100105), np.float64(235.14045600800546)), simulated_mean=185.4007, simulated_std=141.2983, cvar_5_pct=7.5551
      fcf_est_growth_fy1_vs_ltm: best_distribution=student_t, aic=74522.7630, params=(np.float64(1.4635107883079232), np.float64(-2.3217483301476225), np.float64(58.81520399021376)), simulated_mean=-3.9300, simulated_std=362.2291, cvar_5_pct=-646.1063
      fcf_est_growth_fy2_vs_fy1: best_distribution=student_t, aic=63474.7691, params=(np.float64(0.9884558409191155), np.float64(15.227303605530437), np.float64(21.699364918422972)), simulated_mean=19.8040, simulated_std=652.6165, cvar_5_pct=-723.7489
      fcf_est_cagr_3y: best_distribution=student_t, aic=31433.7534, params=(np.float64(1.8901928899480323), np.float64(8.542397345883252), np.float64(14.885072282666933)), simulated_mean=8.4295, simulated_std=42.3866, cvar_5_pct=-81.1178
      fcf_est_margin_fy1: best_distribution=student_t, aic=51915.8030, params=(np.float64(1.4452049149943467), np.float64(5.805743877436089), np.float64(6.533484331344172)), simulated_mean=5.8528, simulated_std=74.6644, cvar_5_pct=-82.8035
      fcf_est_yield_fy1: best_distribution=student_t, aic=42888.3438, params=(np.float64(2.916082856865784), np.float64(3.772084562193527), np.float64(4.4519369656813055)), simulated_mean=3.6218, simulated_std=7.6899, cvar_5_pct=-13.9667
      fcf_est_growth_acceleration: best_distribution=student_t, aic=68344.0208, params=(np.float64(0.9057374316815374), np.float64(15.4798136665205), np.float64(51.87039276419148)), simulated_mean=-176.4654, simulated_std=12017.0017, cvar_5_pct=-6682.1339
      self_funding_ratio: best_distribution=student_t, aic=29749.9270, params=(np.float64(1.223980363523886), np.float64(1.5023822538755358), np.float64(1.1953463069149475)), simulated_mean=0.5248, simulated_std=66.9075, cvar_5_pct=-36.5296
      cff_pattern_score: best_distribution=skew_normal, aic=10901.2681, params=(np.float64(-26624433.942108378), np.float64(1.0000002228413303), np.float64(1.0943305521000188)), simulated_mean=0.1307, simulated_std=0.6626, cvar_5_pct=-1.5995
      cff_quarterly_trend: best_distribution=student_t, aic=81864.0316, params=(np.float64(0.9746210065197413), np.float64(-4.148937658039049), np.float64(92.08939398431971)), simulated_mean=-194.7172, simulated_std=12798.1524, cvar_5_pct=-8411.6723
      cfi_quarterly_trend: best_distribution=student_t, aic=77680.4811, params=(np.float64(1.0864800510509522), np.float64(2.2004528263207552), np.float64(72.70063473907993)), simulated_mean=-23.3519, simulated_std=2173.1161, cvar_5_pct=-2110.9806
      cfo_quarterly_trend: best_distribution=student_t, aic=70978.1156, params=(np.float64(1.3399788160045358), np.float64(5.252556237805047), np.float64(47.90746104094315)), simulated_mean=42.7423, simulated_std=2575.4516, cvar_5_pct=-725.0943
      fcf_quarterly_trend: best_distribution=student_t, aic=75384.9989, params=(np.float64(1.3372683408229846), np.float64(4.570114692596961), np.float64(66.8555833386549)), simulated_mean=14.5256, simulated_std=865.1640, cvar_5_pct=-949.5626
      operating_cf_momentum: best_distribution=student_t, aic=76237.5308, params=(np.float64(1.290149935218236), np.float64(17.15138605575363), np.float64(57.49333349849892)), simulated_mean=14.8210, simulated_std=574.5136, cvar_5_pct=-762.4404
      fcf_growth_yoy: best_distribution=student_t, aic=82221.2369, params=(np.float64(1.250525734785827), np.float64(5.200403207489266), np.float64(54.72028812291253)), simulated_mean=13.0040, simulated_std=543.4949, cvar_5_pct=-682.2072
      fcf_yield: best_distribution=student_t, aic=45404.8342, params=(np.float64(2.3513607829586562), np.float64(3.3636698493049866), np.float64(4.921874905637075)), simulated_mean=3.4985, simulated_std=12.4389, cvar_5_pct=-21.1766
      cf_volatility_score: best_distribution=student_t, aic=34330.1456, params=(np.float64(1.1293356939171768), np.float64(2.490046383381429), np.float64(1.5412202940503668)), simulated_mean=2.1694, simulated_std=46.9463, cvar_5_pct=-47.3625
      fcf_est_cagr_5y_fwd: best_distribution=skew_normal, aic=11941.6584, params=(np.float64(4.189543817453204), np.float64(-6.790770968533387), np.float64(27.463644869268734)), simulated_mean=14.8300, simulated_std=17.4309, cvar_5_pct=-12.2245
      fcf_est_capex_implied_ratio: best_distribution=student_t, aic=20706.6403, params=(np.float64(1.391340560263011), np.float64(0.7455111503813567), np.float64(0.6221726762503637)), simulated_mean=0.6962, simulated_std=22.0363, cvar_5_pct=-12.8055
      fcf_est_growth_deceleration: best_distribution=skew_normal, aic=-2501.1448, params=(np.float64(15205796.886312637), np.float64(-1.4918157542733068e-07), np.float64(0.40103646514646674)), simulated_mean=0.3203, simulated_std=0.2411, cvar_5_pct=0.0137
      fcf_est_growth_fy3_vs_fy2: best_distribution=student_t, aic=61464.9506, params=(np.float64(0.9061953855332103), np.float64(12.188375405553693), np.float64(16.733444496651288)), simulated_mean=-213.9972, simulated_std=22650.6260, cvar_5_pct=-7637.4130
      fcf_est_growth_fy4_vs_fy3: best_distribution=skew_normal, aic=50788.7673, params=(np.float64(63014820.50847234), np.float64(-100.0000090805836), np.float64(100.79258371747221)), simulated_mean=-20.4026, simulated_std=60.5017, cvar_5_pct=-96.8768
      fcf_est_growth_fy5_vs_fy4: best_distribution=student_t, aic=26222.6442, params=(np.float64(0.7678472988151477), np.float64(8.697577836473357), np.float64(11.338871142901983)), simulated_mean=101.4683, simulated_std=4421.7208, cvar_5_pct=-1343.9038
      fcf_est_vs_historical: best_distribution=student_t, aic=82855.4204, params=(np.float64(1.0594655606050551), np.float64(-17.66672532906707), np.float64(88.56185129477002)), simulated_mean=2.7292, simulated_std=2653.0216, cvar_5_pct=-2450.7951
    
    ============================================================
      Distribution Fitting: Composite Scores (3 features)
    ============================================================
      dilution_score: best_distribution=student_t, aic=52590.7865, params=(np.float64(4.044992815567543), np.float64(49.17360534350837), np.float64(13.670262428380866)), simulated_mean=49.2666, simulated_std=19.3973, cvar_5_pct=4.5568
      quality_momentum_score: best_distribution=normal, aic=52108.2877, params=(np.float64(53.11094304356478), np.float64(12.979249317493293)), simulated_mean=52.5409, simulated_std=13.1109, cvar_5_pct=25.1302
      piotroski_f_score: best_distribution=student_t, aic=24187.0458, params=(np.float64(3.766290695977412), np.float64(3.4081306355921903), np.float64(1.1265149767921177)), simulated_mean=3.3892, simulated_std=1.9346, cvar_5_pct=-0.5774
    
    ============================================================
      Distribution Fitting: Dividend Reliability (18 features)
    ============================================================
      div_yield_5y_trend: best_distribution=student_t, aic=-30636.0674, params=(np.float64(1.9447535355885384), np.float64(-0.00014841393475545952), np.float64(0.00419656078666512)), simulated_mean=-0.0002, simulated_std=0.0273, cvar_5_pct=-0.0299
      div_yield_stability: best_distribution=skew_normal, aic=-25030.9929, params=(np.float64(13147287.58949102), np.float64(-1.6546191834826183e-08), np.float64(0.04281608301142381)), simulated_mean=0.0340, simulated_std=0.0259, cvar_5_pct=0.0013
      div_yield_declining_flag: best_distribution=skew_normal, aic=-8519.5752, params=(np.float64(3222549.5802329364), np.float64(-4.466784795455397e-07), np.float64(0.2555364263528901)), simulated_mean=0.2042, simulated_std=0.1573, cvar_5_pct=0.0078
      div_yield_mean_5y: best_distribution=skew_normal, aic=-27678.7061, params=(np.float64(42038777.08415009), np.float64(-4.1041072332603e-09), np.float64(0.03312748245910051)), simulated_mean=0.0264, simulated_std=0.0203, cvar_5_pct=0.0010
      div_yield_vs_5y_mean: best_distribution=student_t, aic=9904.4384, params=(np.float64(2.411643277840111), np.float64(-0.03487940167541426), np.float64(0.4252128960840889)), simulated_mean=-0.0227, simulated_std=0.9014, cvar_5_pct=-2.0209
      dividend_payout_ratio: best_distribution=skew_normal, aic=5543.0929, params=(np.float64(6.253388574106836), np.float64(-0.20680446218402102), np.float64(0.903630793492264)), simulated_mean=0.5016, simulated_std=0.5577, cvar_5_pct=-0.2980
      fcf_dividend_coverage: best_distribution=student_t, aic=22245.5877, params=(np.float64(1.3307923261996601), np.float64(1.9070315599304146), np.float64(1.6965689806998268)), simulated_mean=1.9276, simulated_std=15.3381, cvar_5_pct=-20.7273
      total_shareholder_yield: best_distribution=student_t, aic=-23666.2421, params=(np.float64(1.9651868715210323), np.float64(0.01672262066853086), np.float64(0.02283300754911352)), simulated_mean=0.0146, simulated_std=0.0778, cvar_5_pct=-0.1289
      dividend_growth_expectation: best_distribution=student_t, aic=-25094.8974, params=(np.float64(0.9401865959935898), np.float64(0.0013072645070395295), np.float64(0.00290109506800967)), simulated_mean=0.0036, simulated_std=0.1317, cvar_5_pct=-0.1051
      high_yield_flag: best_distribution=skew_normal, aic=-5445.2961, params=(np.float64(7918013.498343041), np.float64(-2.535830795647565e-07), np.float64(0.321704371076884)), simulated_mean=0.2597, simulated_std=0.1957, cvar_5_pct=0.0112
      sustainable_dividend_flag: best_distribution=skew_normal, aic=-7304.9506, params=(np.float64(4884914.489586754), np.float64(-3.181071486812077e-07), np.float64(0.2798669028160563)), simulated_mean=0.2223, simulated_std=0.1691, cvar_5_pct=0.0084
      days_since_ex_date: best_distribution=skew_normal, aic=56001.9345, params=(np.float64(3.7510771706889994), np.float64(-51.98869507854672), np.float64(193.87390129366224)), simulated_mean=98.1152, simulated_std=123.9921, cvar_5_pct=-94.8628
      days_to_payment: best_distribution=skew_normal, aic=55952.5687, params=(np.float64(-3.7674456227799213), np.float64(83.13271736441337), np.float64(207.41746318384958)), simulated_mean=-76.0303, simulated_std=130.8432, cvar_5_pct=-404.1580
      dividend_announced_flag: best_distribution=skew_normal, aic=-8234.8090, params=(np.float64(3257068.0459414767), np.float64(-4.243172598715077e-07), np.float64(0.261059799643931)), simulated_mean=0.2076, simulated_std=0.1539, cvar_5_pct=0.0086
      dividend_frequency_score: best_distribution=skew_normal, aic=15527.9652, params=(np.float64(3953081.914699829), np.float64(-2.0557429464471565e-06), np.float64(1.551963738968663)), simulated_mean=1.2317, simulated_std=0.9400, cvar_5_pct=0.0508
      ex_date_approaching_flag: best_distribution=skew_normal, aic=-10278.7810, params=(np.float64(3847676.0406455994), np.float64(-3.271888076794303e-07), np.float64(0.22402370592039944)), simulated_mean=0.1775, simulated_std=0.1348, cvar_5_pct=0.0073
      recent_dividend_change: best_distribution=student_t, aic=47767.6356, params=(np.float64(3.4406154298530724), np.float64(-7.462034900242559), np.float64(42.88339398318392)), simulated_mean=-8.8716, simulated_std=69.2912, cvar_5_pct=-168.6713
      div_yield_growth_expected: best_distribution=student_t, aic=41304.0337, params=(np.float64(1.0601686736323055), np.float64(5.784479353587301), np.float64(17.060103973578563)), simulated_mean=129.0157, simulated_std=5614.2990, cvar_5_pct=-461.6413
    
    ============================================================
      Distribution Fitting: EPS Trajectory (9 features)
    ============================================================
      eps_qoq_growth: best_distribution=skew_normal, aic=76805.8264, params=(np.float64(0.9931965754733969), np.float64(-101.91690230245496), np.float64(193.30953496377623)), simulated_mean=5.8631, simulated_std=161.2348, cvar_5_pct=-312.5831
      eps_yoy_quarterly: best_distribution=student_t, aic=72493.5380, params=(np.float64(1.236508147719937), np.float64(10.098709353811307), np.float64(47.779463028572394)), simulated_mean=2.4787, simulated_std=776.0969, cvar_5_pct=-1013.0604
      eps_positive_streak: best_distribution=skew_normal, aic=20579.3432, params=(np.float64(-9384424.227974214), np.float64(5.000001285419165), np.float64(2.2594298899956735)), simulated_mean=3.2209, simulated_std=1.3512, cvar_5_pct=-0.2525
      eps_cagr_3y: best_distribution=student_t, aic=42532.2359, params=(np.float64(3.1347693127744476), np.float64(3.8204267091420725), np.float64(20.973874340673675)), simulated_mean=3.4003, simulated_std=33.8101, cvar_5_pct=-74.5058
      eps_cagr_5y: best_distribution=student_t, aic=34257.6181, params=(np.float64(3.4826766684589328), np.float64(8.90710471627672), np.float64(14.52027240969033)), simulated_mean=8.5839, simulated_std=22.1956, cvar_5_pct=-43.2150
      eps_improvement_count: best_distribution=skew_normal, aic=20158.9112, params=(np.float64(-1.0340187711362172), np.float64(3.514654691771998), np.float64(1.3386766805627346)), simulated_mean=2.7513, simulated_std=1.0971, cvar_5_pct=0.3716
      eps_trajectory_score: best_distribution=skew_normal, aic=60157.9285, params=(np.float64(-1.0340400922316855), np.float64(70.29337719748649), np.float64(26.77379111331809)), simulated_mean=55.5413, simulated_std=21.8274, cvar_5_pct=7.7324
      composite_eps_trajectory_score: best_distribution=skew_normal, aic=60157.9285, params=(np.float64(-1.0340400922316855), np.float64(70.29337719748649), np.float64(26.77379111331809)), simulated_mean=55.0702, simulated_std=21.8952, cvar_5_pct=7.8161
      eps_growth_accel: best_distribution=student_t, aic=33577.6603, params=(np.float64(2.5848582271516634), np.float64(-5.86186894576137), np.float64(15.382892713758281)), simulated_mean=-5.3529, simulated_std=30.5382, cvar_5_pct=-72.3110
    
    ============================================================
      Distribution Fitting: Earnings Quality (26 features)
    ============================================================
      eps_surprise_pct: best_distribution=student_t, aic=35532.1688, params=(np.float64(1.5265169298019985), np.float64(-14.681223790963234), np.float64(20.827155428240893)), simulated_mean=-11.9256, simulated_std=103.1526, cvar_5_pct=-192.3971
      revenue_surprise_pct: best_distribution=student_t, aic=49757.8989, params=(np.float64(2.0212417202004884), np.float64(-6.45451524269828), np.float64(7.337328748929807)), simulated_mean=-6.4734, simulated_std=22.0848, cvar_5_pct=-52.9705
      gaap_adj_eps_gap_pct: best_distribution=skew_normal, aic=50723.4179, params=(np.float64(-2.718924017636777), np.float64(18.993938737860653), np.float64(44.13958525736012)), simulated_mean=-14.7457, simulated_std=29.6571, cvar_5_pct=-85.3027
      ebitda_adjustment_ratio: best_distribution=skew_normal, aic=10118.5616, params=(np.float64(8.15508927640706), np.float64(-0.1628762288769156), np.float64(0.9555808025744146)), simulated_mean=0.5915, simulated_std=0.5646, cvar_5_pct=-0.2184
      eps_quarterly_trend: best_distribution=student_t, aic=18465.6814, params=(np.float64(1.236485208272077), np.float64(0.10098803698575777), np.float64(0.47780079044961676)), simulated_mean=0.2390, simulated_std=10.0346, cvar_5_pct=-6.6325
      eps_adjustment_ratio: best_distribution=student_t, aic=6328.2622, params=(np.float64(0.5559885299681846), np.float64(1.001126066003328), np.float64(0.07989813175547072)), simulated_mean=58.6129, simulated_std=4625.5913, cvar_5_pct=-331.7983
      eps_yoy_growth: best_distribution=student_t, aic=75398.2770, params=(np.float64(1.1838974237686726), np.float64(10.213020591119847), np.float64(37.26424182733124)), simulated_mean=-17.0545, simulated_std=2117.1323, cvar_5_pct=-1633.5796
      eps_cont_cagr_3y: best_distribution=student_t, aic=42543.0320, params=(np.float64(3.0674046302440026), np.float64(3.8722111185645502), np.float64(20.568590235295176)), simulated_mean=3.8100, simulated_std=32.1955, cvar_5_pct=-72.2370
      eps_cont_qoq_growth: best_distribution=skew_normal, aic=76150.8228, params=(np.float64(1.1433244977890635), np.float64(-114.92917452646714), np.float64(190.54296937849315)), simulated_mean=-2.2487, simulated_std=155.6090, cvar_5_pct=-303.8757
      eps_cont_trajectory_score: best_distribution=normal, aic=62098.5165, params=(np.float64(52.98831635710006), np.float64(25.321134246436152)), simulated_mean=52.9963, simulated_std=25.3194, cvar_5_pct=0.9682
      eps_cont_yoy_growth: best_distribution=student_t, aic=75214.6908, params=(np.float64(1.2018929448647566), np.float64(4.3530110055173035), np.float64(37.43827390558212)), simulated_mean=8.8472, simulated_std=857.1632, cvar_5_pct=-767.8605
      gaap_positive_revision_flag: best_distribution=skew_normal, aic=-779.3125, params=(np.float64(15666253.02097702), np.float64(-1.5807859905532964e-07), np.float64(0.4562672612568372)), simulated_mean=0.3655, simulated_std=0.2750, cvar_5_pct=0.0134
      gaap_revision_1y: best_distribution=student_t, aic=3368.5397, params=(np.float64(2.099979123340197), np.float64(-0.05649079401481584), np.float64(0.20149447379861118)), simulated_mean=-0.0714, simulated_std=0.4596, cvar_5_pct=-1.2415
      gaap_revision_3m: best_distribution=student_t, aic=-4140.3312, params=(np.float64(1.2622037369666432), np.float64(-0.011131972630237682), np.float64(0.06825154202821491)), simulated_mean=-0.0319, simulated_std=1.3088, cvar_5_pct=-1.5122
      gaap_revision_6m: best_distribution=student_t, aic=-343.6395, params=(np.float64(1.6131503892669143), np.float64(-0.019809665834457715), np.float64(0.11883629480618113)), simulated_mean=-0.0066, simulated_std=0.8829, cvar_5_pct=-1.0878
      gaap_revision_acceleration: best_distribution=student_t, aic=-2542.9853, params=(np.float64(1.5535950325753527), np.float64(0.015033280241392315), np.float64(0.09345896094032458)), simulated_mean=-0.0046, simulated_std=0.7981, cvar_5_pct=-1.1097
      gaap_revision_1m: best_distribution=student_t, aic=-16561.6950, params=(np.float64(0.8001438608682001), np.float64(0.003839716579992973), np.float64(0.012743899746472559)), simulated_mean=-0.3024, simulated_std=13.7383, cvar_5_pct=-6.8485
      gaap_revision_momentum: best_distribution=student_t, aic=-9003.1212, params=(np.float64(0.9527010305007138), np.float64(-0.004269105790311877), np.float64(0.03776426476834883)), simulated_mean=-0.1853, simulated_std=11.8735, cvar_5_pct=-5.2452
      revision_quality_divergence: best_distribution=student_t, aic=-14608.4833, params=(np.float64(0.7326191073351314), np.float64(0.007656660038799758), np.float64(0.010497391587463562)), simulated_mean=-0.0686, simulated_std=5.5199, cvar_5_pct=-3.2483
      net_income_growth_yoy: best_distribution=student_t, aic=79055.0285, params=(np.float64(1.0947981588526234), np.float64(5.877688461372838), np.float64(37.895204284130656)), simulated_mean=14.6990, simulated_std=3160.0304, cvar_5_pct=-1635.5008
      gaap_vs_norm_revision_spread: best_distribution=student_t, aic=-11590.5654, params=(np.float64(0.7340701992711669), np.float64(0.0012369868854238044), np.float64(0.014614814524196766)), simulated_mean=-0.0769, simulated_std=10.6057, cvar_5_pct=-3.9811
      net_income_qoq_growth: best_distribution=student_t, aic=77167.9622, params=(np.float64(0.7767065016691725), np.float64(-1.2171533552741551), np.float64(22.048520253144602)), simulated_mean=-171.5327, simulated_std=9813.9511, cvar_5_pct=-5963.2252
      ni_adjustment_ratio: best_distribution=skew_normal, aic=15392.1681, params=(np.float64(1.4083825744631386), np.float64(-0.1415852159474828), np.float64(1.0409609283387318)), simulated_mean=0.5368, simulated_std=0.7806, cvar_5_pct=-0.9624
      has_unusual_items_flag: best_distribution=skew_normal, aic=1758.9758, params=(np.float64(-2421779.5406866446), np.float64(1.0000011963366182), np.float64(0.5519017693387616)), simulated_mean=0.5621, simulated_std=0.3331, cvar_5_pct=-0.2919
      unusual_items_to_ebitda: best_distribution=skew_normal, aic=57723.2746, params=(np.float64(11277502.19566115), np.float64(-2.0006984696514945e-05), np.float64(39.543886536814)), simulated_mean=31.8159, simulated_std=23.8482, cvar_5_pct=1.1732
      unusual_items_to_revenue: best_distribution=skew_normal, aic=32056.1002, params=(np.float64(3888527.5809471225), np.float64(-9.004352773546402e-06), np.float64(5.770883946427034)), simulated_mean=4.5844, simulated_std=3.4707, cvar_5_pct=0.1677
    
    ============================================================
      Distribution Fitting: Efficiency (4 features)
    ============================================================
      asset_turnover: best_distribution=skew_normal, aic=7414.1446, params=(np.float64(5.160929254368858), np.float64(0.1376288415501244), np.float64(0.7639170181720714)), simulated_mean=0.7495, simulated_std=0.4749, cvar_5_pct=0.0399
      inventory_turnover: best_distribution=student_t, aic=39059.3712, params=(np.float64(0.7087276856314073), np.float64(3.643223230415174), np.float64(2.00373408261132)), simulated_mean=-3.3036, simulated_std=1107.8551, cvar_5_pct=-595.1693
      receivables_days: best_distribution=skew_normal, aic=66055.7780, params=(np.float64(18159819.88873396), np.float64(-2.2403244323005887e-05), np.float64(80.23695811448232)), simulated_mean=63.6073, simulated_std=47.1320, cvar_5_pct=2.4500
      working_capital_turns: best_distribution=student_t, aic=45231.2195, params=(np.float64(0.9271528534820419), np.float64(2.664022141834412), np.float64(2.7742585193530616)), simulated_mean=1.5799, simulated_std=253.9720, cvar_5_pct=-195.2591
    
    ============================================================
      Distribution Fitting: Efficiency Ratios (25 features)
    ============================================================
      opex_vs_revenue_trend: best_distribution=student_t, aic=40280.6875, params=(np.float64(0.9589841023288921), np.float64(-0.20688808795313418), np.float64(1.7990668991468985)), simulated_mean=1.5664, simulated_std=108.6576, cvar_5_pct=-69.6490
      opex_qoq_growth: best_distribution=student_t, aic=-6194.1038, params=(np.float64(1.4724285622638955), np.float64(0.023078011062314593), np.float64(0.07121882862499143)), simulated_mean=0.0236, simulated_std=0.3775, cvar_5_pct=-0.6650
      opex_yoy_growth: best_distribution=student_t, aic=-4239.8870, params=(np.float64(2.4173395104535116), np.float64(0.0995471739001236), np.float64(0.11199645578260761)), simulated_mean=0.1001, simulated_std=0.2269, cvar_5_pct=-0.3870
      sga_qoq_growth: best_distribution=student_t, aic=-9364.9925, params=(np.float64(1.2655301459137345), np.float64(-0.7097340601210052), np.float64(0.04857646239809569)), simulated_mean=-0.7687, simulated_std=2.9418, cvar_5_pct=-2.5989
      sga_yoy_growth: best_distribution=student_t, aic=-2097.6530, params=(np.float64(1.7999315622367367), np.float64(0.099080012475103), np.float64(0.1123117255131101)), simulated_mean=0.0919, simulated_std=0.6922, cvar_5_pct=-0.8276
      operating_leverage_score: best_distribution=student_t, aic=-15412.3706, params=(np.float64(0.9933631962494854), np.float64(0.0026678103215321478), np.float64(0.023441508865259086)), simulated_mean=-0.0298, simulated_std=1.9862, cvar_5_pct=-1.4258
      cogs_to_revenue: best_distribution=skew_normal, aic=58610.8615, params=(np.float64(-3.2206614326663887), np.float64(87.95049643592398), np.float64(36.383746756002694)), simulated_mean=60.7124, simulated_std=23.5154, cvar_5_pct=2.3156
      opex_to_revenue: best_distribution=student_t, aic=53058.1648, params=(np.float64(1.526297985746928), np.float64(89.76839170948675), np.float64(7.449489227302536)), simulated_mean=88.7460, simulated_std=47.4447, cvar_5_pct=8.4447
      sga_to_revenue: best_distribution=student_t, aic=54733.5782, params=(np.float64(2.2285705971545995), np.float64(13.818446872189616), np.float64(10.505370758296582)), simulated_mean=14.0072, simulated_std=26.4522, cvar_5_pct=-39.5273
      rnd_to_revenue: best_distribution=skew_normal, aic=39267.4170, params=(np.float64(22472618.81772495), np.float64(-2.588153404530368e-06), np.float64(10.104369043478766)), simulated_mean=8.0485, simulated_std=6.0052, cvar_5_pct=0.3382
      interest_to_revenue: best_distribution=skew_normal, aic=30035.3678, params=(np.float64(-24231831.873209223), np.float64(1.077602740408573e-06), np.float64(4.948994270554955)), simulated_mean=-3.9239, simulated_std=2.9793, cvar_5_pct=-11.5177
      cost_efficiency_score: best_distribution=student_t, aic=56884.9888, params=(np.float64(8.189722997345678), np.float64(41.37544795974185), np.float64(16.504043485690538)), simulated_mean=41.6222, simulated_std=19.5926, cvar_5_pct=-1.3048
      marketing_to_revenue: best_distribution=skew_normal, aic=13890.0573, params=(np.float64(23767340.319682077), np.float64(-3.942106054815289e-07), np.float64(1.4165211433573717)), simulated_mean=1.1260, simulated_std=0.8605, cvar_5_pct=0.0449
      marketing_trend_yoy: best_distribution=student_t, aic=8649.1873, params=(np.float64(1.3980566647287134), np.float64(7.22339796975861), np.float64(21.16300206877296)), simulated_mean=6.8719, simulated_std=210.4454, cvar_5_pct=-297.2204
      sga_efficiency_trend: best_distribution=student_t, aic=27976.1439, params=(np.float64(0.9442133938354651), np.float64(-0.044628171288783255), np.float64(0.6709590209804965)), simulated_mean=-1.9987, simulated_std=167.2807, cvar_5_pct=-83.6871
      sga_trend_yoy: best_distribution=student_t, aic=27976.1439, params=(np.float64(0.9441654273726325), np.float64(0.044589871378200335), np.float64(0.6709401476703702)), simulated_mean=5.0828, simulated_std=340.4926, cvar_5_pct=-30.3527
      high_rnd_intensity_flag: best_distribution=skew_normal, aic=-5051.3709, params=(np.float64(4953898.526258461), np.float64(-3.9317318163012424e-07), np.float64(0.33132776437118916)), simulated_mean=0.2638, simulated_std=0.2009, cvar_5_pct=0.0107
      rnd_cagr_3y: best_distribution=student_t, aic=18981.7156, params=(np.float64(2.9888605361590375), np.float64(6.996801227303166), np.float64(10.395248614982354)), simulated_mean=6.9174, simulated_std=17.5703, cvar_5_pct=-34.0792
      rnd_cut_flag: best_distribution=skew_normal, aic=-11428.5474, params=(np.float64(8118360.900791302), np.float64(-1.4291495987028812e-07), np.float64(0.20552170304517692)), simulated_mean=0.1597, simulated_std=0.1199, cvar_5_pct=0.0062
      rnd_per_employee: best_distribution=skew_normal, aic=-19011.3055, params=(np.float64(1977275.402971143), np.float64(-2.0751058229694665e-07), np.float64(0.07777789811666944)), simulated_mean=0.0623, simulated_std=0.0472, cvar_5_pct=0.0026
      rnd_increasing_flag: best_distribution=skew_normal, aic=-7153.5009, params=(np.float64(3076201.462748912), np.float64(-4.891618753235605e-07), np.float64(0.28306172297445387)), simulated_mean=0.2242, simulated_std=0.1731, cvar_5_pct=0.0082
      rnd_intensity_trend: best_distribution=skew_normal, aic=26428.9194, params=(np.float64(-4.996283970652197), np.float64(1.3877583531120083), np.float64(3.4758276453155155)), simulated_mean=-1.3597, simulated_std=2.1870, cvar_5_pct=-6.7824
      rnd_qoq_growth: best_distribution=student_t, aic=22109.3953, params=(np.float64(0.9456172605458325), np.float64(2.4309376847333697), np.float64(9.195277649972423)), simulated_mean=-20.1592, simulated_std=980.8602, cvar_5_pct=-873.4890
      rnd_to_gross_profit: best_distribution=skew_normal, aic=46741.6014, params=(np.float64(13674968.677189313), np.float64(-7.182735841593516e-06), np.float64(18.226072789619955)), simulated_mean=14.5721, simulated_std=10.9535, cvar_5_pct=0.6405
      rnd_yoy_growth: best_distribution=student_t, aic=23044.4088, params=(np.float64(1.701642421926835), np.float64(9.316620293993193), np.float64(13.802113748226748)), simulated_mean=9.3691, simulated_std=45.2705, cvar_5_pct=-83.7904
    
    ============================================================
      Distribution Fitting: Employee Productivity (7 features)
    ============================================================
      revenue_per_employee: best_distribution=student_t, aic=5849.7800, params=(np.float64(1.4090711548988228), np.float64(0.334647945362639), np.float64(0.1944935528691374)), simulated_mean=0.3573, simulated_std=1.9591, cvar_5_pct=-1.8400
      profit_per_employee: best_distribution=student_t, aic=-15190.2144, params=(np.float64(0.8471256216590768), np.float64(0.014147714010413218), np.float64(0.014817667341468279)), simulated_mean=0.3356, simulated_std=32.1976, cvar_5_pct=-4.3980
      ebitda_per_employee: best_distribution=student_t, aic=-7698.7575, params=(np.float64(0.8956486572537051), np.float64(0.041104554152174774), np.float64(0.03297728676848151)), simulated_mean=0.1923, simulated_std=13.6264, cvar_5_pct=-2.9370
      assets_per_employee: best_distribution=student_t, aic=13935.3693, params=(np.float64(0.9623589459755164), np.float64(0.4791482690220238), np.float64(0.2940003765791155)), simulated_mean=-2.3988, simulated_std=161.2460, cvar_5_pct=-67.3692
      fte_growth_1y_pct: best_distribution=student_t, aic=51024.8724, params=(np.float64(0.8735836919585616), np.float64(0.8762095403373258), np.float64(6.002213032019931)), simulated_mean=-22.0551, simulated_std=2830.1803, cvar_5_pct=-1275.3637
      fte_growth_3y_pct: best_distribution=student_t, aic=57510.8570, params=(np.float64(1.3931528849755574), np.float64(4.116734112622977), np.float64(21.22300724326307)), simulated_mean=4.6324, simulated_std=226.2788, cvar_5_pct=-274.8367
      workforce_stability: best_distribution=student_t, aic=2930.2503, params=(np.float64(1.3486410823104598), np.float64(1.0225657438093045), np.float64(0.23650863700260105)), simulated_mean=0.9240, simulated_std=7.4917, cvar_5_pct=-4.1915
    
    ============================================================
      Distribution Fitting: Employment Dynamics (7 features)
    ============================================================
      fte_acceleration: best_distribution=student_t, aic=37549.8782, params=(np.float64(1.4596112434797592), np.float64(-0.5265117684522153), np.float64(3.9985120645148364)), simulated_mean=-0.6680, simulated_std=22.7096, cvar_5_pct=-44.9677
      fte_growth_2y_pct: best_distribution=student_t, aic=55303.9182, params=(np.float64(1.136186484252169), np.float64(2.191341626178534), np.float64(13.243979833417225)), simulated_mean=10.6458, simulated_std=670.4842, cvar_5_pct=-320.5296
      hiring_intensity: best_distribution=student_t, aic=16038.8764, params=(np.float64(0.8615760957139629), np.float64(0.1431167081112928), np.float64(0.41388974372474985)), simulated_mean=-23.5365, simulated_std=1702.8805, cvar_5_pct=-515.0516
      layoff_risk_flag: best_distribution=skew_normal, aic=-4996.9352, params=(np.float64(1482988.7690382209), np.float64(-1.1618431236668443e-06), np.float64(0.33272034915765697)), simulated_mean=0.2654, simulated_std=0.2023, cvar_5_pct=0.0100
      productivity_trend: best_distribution=student_t, aic=41735.8645, params=(np.float64(1.9664020563001752), np.float64(7.423683267508515), np.float64(10.074885230203684)), simulated_mean=7.2617, simulated_std=26.5039, cvar_5_pct=-53.4275
      rapid_hiring_flag: best_distribution=skew_normal, aic=-7029.8722, params=(np.float64(8284797.474109357), np.float64(-1.8764152684005094e-07), np.float64(0.28571454346283226)), simulated_mean=0.2284, simulated_std=0.1728, cvar_5_pct=0.0092
      sustainable_growth_flag: best_distribution=skew_normal, aic=1460.1120, params=(np.float64(14319612.031740222), np.float64(-2.0199272218025872e-07), np.float64(0.5396286461119082)), simulated_mean=0.4390, simulated_std=0.3217, cvar_5_pct=0.0187
    
    ============================================================
      Distribution Fitting: Cash flow (19 features)
    ============================================================
      self_funding_flag: best_distribution=skew_normal, aic=3508.2340, params=(np.float64(-10227146.208805233), np.float64(1.0000003280281877), np.float64(0.6291632366255606)), simulated_mean=0.4932, simulated_std=0.3855, cvar_5_pct=-0.4929
      capex_qoq_growth: best_distribution=student_t, aic=63972.4253, params=(np.float64(1.2691391791824307), np.float64(3.3163874145147276), np.float64(27.06817362884069)), simulated_mean=-0.2031, simulated_std=311.0252, cvar_5_pct=-464.2989
      capex_yoy_growth: best_distribution=student_t, aic=71750.4114, params=(np.float64(1.9568179361775622), np.float64(5.703171777577486), np.float64(35.36211622882494)), simulated_mean=6.5186, simulated_std=95.3234, cvar_5_pct=-202.8414
      cash_flow_quality_score: best_distribution=skew_normal, aic=63390.5435, params=(np.float64(-14785629.923477631), np.float64(100.00001951786356), np.float64(55.77950276882615)), simulated_mean=55.5239, simulated_std=33.7902, cvar_5_pct=-30.6967
      cff_share_of_cf: best_distribution=skew_normal, aic=-4669.6697, params=(np.float64(3.1123321819535494), np.float64(0.08837507374653089), np.float64(0.2620699867250952)), simulated_mean=0.2881, simulated_std=0.1730, cvar_5_pct=0.0024
      cfi_share_of_cf: best_distribution=skew_normal, aic=-5072.7634, params=(np.float64(3610.159647522285), np.float64(0.003750563139428992), np.float64(0.3161617883219614)), simulated_mean=0.2559, simulated_std=0.1902, cvar_5_pct=0.0134
      cfo_share_of_cf: best_distribution=skew_normal, aic=-5651.9006, params=(np.float64(-2.7905195065081507), np.float64(0.6053144130916821), np.float64(0.23637164830970592)), simulated_mean=0.4329, simulated_std=0.1541, cvar_5_pct=0.0623
      serial_acquirer_flag: best_distribution=skew_normal, aic=1729.3482, params=(np.float64(4482632.23205452), np.float64(-6.5365625882775e-07), np.float64(0.5506202727513103)), simulated_mean=0.4337, simulated_std=0.3265, cvar_5_pct=0.0172
      sustainable_ma_flag: best_distribution=skew_normal, aic=-1267.4013, params=(np.float64(-2883703.6656467626), np.float64(1.0000007961470092), np.float64(0.43990601221322134)), simulated_mean=0.6448, simulated_std=0.2731, cvar_5_pct=-0.0571
      total_investment_to_cfo: best_distribution=student_t, aic=12103.4620, params=(np.float64(1.3613349047100534), np.float64(0.3749955428164744), np.float64(0.2940727540045367)), simulated_mean=0.4931, simulated_std=5.0666, cvar_5_pct=-2.8517
      underinvestment_flag: best_distribution=skew_normal, aic=979.6175, params=(np.float64(11100590.715883497), np.float64(-2.6280489296919776e-07), np.float64(0.5205843368669683)), simulated_mean=0.4214, simulated_std=0.3191, cvar_5_pct=0.0186
      acquisition_pause_flag: best_distribution=skew_normal, aic=-1403.1266, params=(np.float64(10935749.144203603), np.float64(-2.2445938420580294e-07), np.float64(0.43542652762787326)), simulated_mean=0.3423, simulated_std=0.2632, cvar_5_pct=0.0114
      acquisition_to_fcf: best_distribution=skew_normal, aic=5307.6358, params=(np.float64(8447953.63037945), np.float64(-4.855916669083562e-07), np.float64(0.7512515479972568)), simulated_mean=0.5981, simulated_std=0.4431, cvar_5_pct=0.0286
      acquisitions_yoy_growth: best_distribution=student_t, aic=16933.6972, params=(np.float64(0.22118087576659917), np.float64(-100.00000000638335), np.float64(6.335521381588178e-09)), simulated_mean=-1255.0816, simulated_std=76412.5565, cvar_5_pct=-23575.1736
      capex_3y_trend: best_distribution=student_t, aic=76402.0039, params=(np.float64(1.617668190257061), np.float64(9.924576978949512), np.float64(56.428582944802976)), simulated_mean=3.5654, simulated_std=285.5411, cvar_5_pct=-527.9262
      capex_acceleration: best_distribution=skew_normal, aic=1373.7051, params=(np.float64(6216679.255159868), np.float64(-4.967843473580198e-07), np.float64(0.5360732722808221)), simulated_mean=0.4246, simulated_std=0.3230, cvar_5_pct=0.0184
      capex_cut_flag: best_distribution=skew_normal, aic=-798.5516, params=(np.float64(14240242.232143484), np.float64(-1.8209065323152205e-07), np.float64(0.4556434687672348)), simulated_mean=0.3581, simulated_std=0.2748, cvar_5_pct=0.0143
      overinvestment_flag: best_distribution=skew_normal, aic=180.8600, params=(np.float64(1112701.4923648117), np.float64(-2.2748769069361576e-06), np.float64(0.49018755157097327)), simulated_mean=0.3965, simulated_std=0.2991, cvar_5_pct=0.0164
      ma_intensity_score: best_distribution=skew_normal, aic=22234.2388, params=(np.float64(8702863.324475732), np.float64(-1.7868389184598012e-06), np.float64(2.844687484704293)), simulated_mean=2.2342, simulated_std=1.6598, cvar_5_pct=0.0875
    
    ============================================================
      Distribution Fitting: Financial Distress (9 features)
    ============================================================
      distress_risk_score: best_distribution=skew_normal, aic=63197.0751, params=(np.float64(-11795662.544976253), np.float64(100.00002517855313), np.float64(54.97778135917248)), simulated_mean=56.3600, simulated_std=32.9146, cvar_5_pct=-27.2646
      liquidity_stress_score: best_distribution=skew_normal, aic=46719.2786, params=(np.float64(20189388.5774965), np.float64(-4.377254264385136e-06), np.float64(16.00271185204373)), simulated_mean=12.8101, simulated_std=9.5557, cvar_5_pct=0.5126
      working_capital_trend: best_distribution=skew_normal, aic=2105.2216, params=(np.float64(0.8350284906737218), np.float64(-0.17145327063799085), np.float64(0.3315885138405027)), simulated_mean=-0.0033, simulated_std=0.2818, cvar_5_pct=-0.5656
      cash_runway_months: best_distribution=student_t, aic=-77304.9368, params=(np.float64(0.5079208711445409), np.float64(119.9999811810266), np.float64(1.2080839481199107e-05)), simulated_mean=119.5086, simulated_std=36.7943, cvar_5_pct=109.3147
      accumulated_deficit_flag: best_distribution=skew_normal, aic=-1109.1000, params=(np.float64(1511724.3026764337), np.float64(-1.5506356600328498e-06), np.float64(0.4450974363359729)), simulated_mean=0.3558, simulated_std=0.2687, cvar_5_pct=0.0139
      adequate_cash_buffer: best_distribution=skew_normal, aic=-14500.3266, params=(np.float64(-1689158.7472856357), np.float64(1.0000005520262967), np.float64(0.16328396853339228)), simulated_mean=0.8678, simulated_std=0.0996, cvar_5_pct=0.6177
      combined_distress_score: best_distribution=skew_normal, aic=61863.0613, params=(np.float64(-5759071.866390498), np.float64(100.00004506944208), np.float64(49.755356317085145)), simulated_mean=59.8443, simulated_std=30.0212, cvar_5_pct=-17.3065
      wc_deteriorating_flag: best_distribution=skew_normal, aic=-7591.8552, params=(np.float64(6242972.164726704), np.float64(-2.53651291816929e-07), np.float64(0.2739268010715332)), simulated_mean=0.2195, simulated_std=0.1653, cvar_5_pct=0.0082
      retained_earnings_growth: best_distribution=skew_normal, aic=-2369.5660, params=(np.float64(-2.545818402640185), np.float64(0.16519605252865469), np.float64(0.31148789748817896)), simulated_mean=-0.0676, simulated_std=0.2070, cvar_5_pct=-0.5622
    
    ============================================================
      Distribution Fitting: GAAP vs Adjusted (3 features)
    ============================================================
      earnings_quality_score: best_distribution=skew_normal, aic=58308.4577, params=(np.float64(-6081052.224065529), np.float64(100.000033411434), np.float64(38.12785293546865)), simulated_mean=69.0481, simulated_std=23.3915, cvar_5_pct=9.8323
      eps_adjustment_pct: best_distribution=student_t, aic=38368.8203, params=(np.float64(0.5519038964203797), np.float64(0.6537106239414068), np.float64(8.037141349285193)), simulated_mean=-863020.6725, simulated_std=60998572.7297, cvar_5_pct=-17356345.3652
      net_income_adjustment_pct: best_distribution=skew_normal, aic=74662.0759, params=(np.float64(8370060.739836825), np.float64(-100.00009135802745), np.float64(145.23172203065263)), simulated_mean=14.2825, simulated_std=87.0280, cvar_5_pct=-95.6133
    
    ============================================================
      Distribution Fitting: Growth Metrics (7 features)
    ============================================================
      revenue_growth_yoy: best_distribution=student_t, aic=55897.4760, params=(np.float64(2.0685066184799785), np.float64(10.163861869476683), np.float64(11.528569484499961)), simulated_mean=10.1382, simulated_std=29.2669, cvar_5_pct=-55.5839
      ebitda_growth_yoy: best_distribution=student_t, aic=67701.1602, params=(np.float64(1.302771661197804), np.float64(10.532258632226462), np.float64(19.34565909102735)), simulated_mean=11.5765, simulated_std=306.1405, cvar_5_pct=-317.6831
      operating_income_growth: best_distribution=skew_normal, aic=51980.2288, params=(np.float64(1.680865158106449), np.float64(-9.773233164934055), np.float64(18.042718507266052)), simulated_mean=2.4262, simulated_std=13.0496, cvar_5_pct=-21.8905
      fcf_growth: best_distribution=skew_normal, aic=65065.1372, params=(np.float64(1.107095118032626), np.float64(-29.856344473667797), np.float64(49.59068522427245)), simulated_mean=-0.8774, simulated_std=40.1212, cvar_5_pct=-80.3945
      revenue_vs_5y_avg: best_distribution=student_t, aic=2857.2905, params=(np.float64(2.639131359281929), np.float64(1.1737802761643128), np.float64(0.20498353045024797)), simulated_mean=1.1778, simulated_std=0.4451, cvar_5_pct=0.3173
      growth_ebitda_growth_yoy: best_distribution=student_t, aic=67701.1602, params=(np.float64(1.302771661197804), np.float64(10.532258632226462), np.float64(19.34565909102735)), simulated_mean=8.4106, simulated_std=386.4942, cvar_5_pct=-356.9740
      revenue_momentum: best_distribution=student_t, aic=56969.3085, params=(np.float64(2.065415564815871), np.float64(11.532216254745995), np.float64(12.52996719285068)), simulated_mean=7.3384, simulated_std=171.6805, cvar_5_pct=-145.5006
    
    ============================================================
      Distribution Fitting: Interest Income (6 features)
    ============================================================
      interest_income_yoy_growth: best_distribution=student_t, aic=10847.9254, params=(np.float64(1.745135016913371), np.float64(-0.053819159927384404), np.float64(0.3321794887532854)), simulated_mean=-0.2441, simulated_std=12.2798, cvar_5_pct=-6.2710
      interest_income_to_revenue_trend: best_distribution=student_t, aic=-39802.0101, params=(np.float64(0.9015016996863559), np.float64(0.002816139546021482), np.float64(0.003247816245825423)), simulated_mean=0.0046, simulated_std=0.4079, cvar_5_pct=-0.2268
      interest_income_qoq_growth: best_distribution=student_t, aic=10156.6617, params=(np.float64(0.6958781363234641), np.float64(-0.0067150931605605875), np.float64(0.1348236788549156)), simulated_mean=2.2371, simulated_std=93.8545, cvar_5_pct=-21.7301
      interest_coverage_ratio: best_distribution=student_t, aic=53053.4303, params=(np.float64(0.6763009483430953), np.float64(-4.083263221289696), np.float64(3.781038440935607)), simulated_mean=-991.0038, simulated_std=63802.7773, cvar_5_pct=-21445.3113
      interest_expense_to_revenue: best_distribution=skew_normal, aic=30035.3678, params=(np.float64(-24231831.873209223), np.float64(1.077602740408573e-06), np.float64(4.948994270554955)), simulated_mean=-4.0003, simulated_std=2.9812, cvar_5_pct=-11.5127
      interest_income_to_revenue: best_distribution=student_t, aic=19742.2767, params=(np.float64(0.9131917620219421), np.float64(0.28654381409200924), np.float64(0.3284315071025997)), simulated_mean=1.0300, simulated_std=35.3048, cvar_5_pct=-13.9206
    
    ============================================================
      Distribution Fitting: Leverage & Liquidity (20 features)
    ============================================================
      debt_to_equity: best_distribution=student_t, aic=14860.2044, params=(np.float64(1.680342880117693), np.float64(0.42072753782895755), np.float64(0.39569710185853113)), simulated_mean=0.4264, simulated_std=4.3697, cvar_5_pct=-3.7065
      debt_to_assets: best_distribution=skew_normal, aic=-5604.1395, params=(np.float64(67618036.916186), np.float64(-2.2795200402883093e-08), np.float64(0.3094911607630225)), simulated_mean=0.2433, simulated_std=0.1861, cvar_5_pct=0.0080
      equity_ratio: best_distribution=skew_normal, aic=-1386.4072, params=(np.float64(-0.7496571525546818), np.float64(0.6007812119426714), np.float64(0.24647716917727078)), simulated_mean=0.4821, simulated_std=0.2176, cvar_5_pct=0.0204
      interest_coverage: best_distribution=student_t, aic=53053.4303, params=(np.float64(0.6763009483430953), np.float64(-4.083263221289696), np.float64(3.781038440935607)), simulated_mean=-2027.6962, simulated_std=138057.0620, cvar_5_pct=-41557.3595
      cash_ratio: best_distribution=student_t, aic=11370.1958, params=(np.float64(1.1570515287645213), np.float64(0.33069382337734965), np.float64(0.23210164632245353)), simulated_mean=0.2450, simulated_std=13.1616, cvar_5_pct=-6.5569
      working_capital_ratio: best_distribution=skew_normal, aic=-2567.3734, params=(np.float64(3.4899270871425676), np.float64(-0.06548344609448886), np.float64(0.32287309637543105)), simulated_mean=0.1793, simulated_std=0.2053, cvar_5_pct=-0.1555
      days_working_capital: best_distribution=student_t, aic=82170.7926, params=(np.float64(1.4646033340067746), np.float64(56.02961471866986), np.float64(70.18117782945491)), simulated_mean=72.1766, simulated_std=1022.9218, cvar_5_pct=-749.0020
      debt_3y_cagr: best_distribution=student_t, aic=60769.6459, params=(np.float64(1.5841694919258194), np.float64(3.648710749668073), np.float64(14.02518576612373)), simulated_mean=0.8965, simulated_std=76.9990, cvar_5_pct=-162.6778
      debt_4q_trend: best_distribution=student_t, aic=64715.5597, params=(np.float64(1.167822376470538), np.float64(5.228637408980072), np.float64(15.907085729835497)), simulated_mean=0.3528, simulated_std=459.3734, cvar_5_pct=-472.7241
      debt_qoq_change: best_distribution=student_t, aic=48009.0168, params=(np.float64(0.8711544658227512), np.float64(-0.08875738304697253), np.float64(2.8845026638354145)), simulated_mean=0.0074, simulated_std=252.2585, cvar_5_pct=-205.6474
      debt_to_equity_trend: best_distribution=student_t, aic=-5773.3288, params=(np.float64(1.0811828668619248), np.float64(0.014090623563945267), np.float64(0.05565573454955225)), simulated_mean=0.0436, simulated_std=2.4039, cvar_5_pct=-1.4165
      debt_yoy_change: best_distribution=student_t, aic=65526.2299, params=(np.float64(1.2042992921446896), np.float64(4.668591839642033), np.float64(15.94629247908387)), simulated_mean=-1.0653, simulated_std=394.9461, cvar_5_pct=-399.1961
      negative_wc_flag: best_distribution=skew_normal, aic=-1672.1358, params=(np.float64(5074768.38619372), np.float64(-4.380422746865159e-07), np.float64(0.4268337482400225)), simulated_mean=0.3396, simulated_std=0.2642, cvar_5_pct=0.0134
      wc_efficiency_score: best_distribution=skew_normal, aic=61208.5705, params=(np.float64(-5.074317286253395), np.float64(94.86344870238239), np.float64(41.400914548401985)), simulated_mean=62.7802, simulated_std=25.4962, cvar_5_pct=0.7792
      wc_to_assets: best_distribution=skew_normal, aic=54638.0507, params=(np.float64(3.489955875149213), np.float64(-6.548386258328771), np.float64(32.28741970407707)), simulated_mean=17.9700, simulated_std=20.1671, cvar_5_pct=-14.9920
      wc_to_revenue: best_distribution=student_t, aic=65598.2849, params=(np.float64(1.464601189641277), np.float64(15.350619343387699), np.float64(19.227699241923702)), simulated_mean=29.7703, simulated_std=1150.3922, cvar_5_pct=-233.1075
      wc_4q_trend: best_distribution=student_t, aic=71827.1945, params=(np.float64(1.0890958612816286), np.float64(6.422114687099157), np.float64(28.63652538196621)), simulated_mean=30.0143, simulated_std=1801.2139, cvar_5_pct=-872.4453
      wc_improving_flag: best_distribution=skew_normal, aic=1778.6413, params=(np.float64(5193719.303779978), np.float64(-5.546352712372244e-07), np.float64(0.552721471757351)), simulated_mean=0.4467, simulated_std=0.3319, cvar_5_pct=0.0158
      wc_qoq_change: best_distribution=student_t, aic=63306.7926, params=(np.float64(0.7194554361968093), np.float64(0.5207479579103761), np.float64(6.898731975656377)), simulated_mean=-181.9679, simulated_std=33016.8272, cvar_5_pct=-9467.1358
      wc_yoy_change: best_distribution=student_t, aic=75651.0354, params=(np.float64(1.0772901911810773), np.float64(5.42561163810185), np.float64(28.386359160115546)), simulated_mean=2.0778, simulated_std=511.8792, cvar_5_pct=-670.3836
    
    ============================================================
      Distribution Fitting: Momentum & Technical (28 features)
    ============================================================
      daily_turnover_ratio: best_distribution=student_t, aic=-45879.8119, params=(np.float64(1.000828074651102), np.float64(0.0025981309376734616), np.float64(0.0024698328868712127)), simulated_mean=0.0049, simulated_std=0.0904, cvar_5_pct=-0.0649
      liquidity_score: best_distribution=skew_normal, aic=180320.4720, params=(np.float64(116586507.74651915), np.float64(1.0152471863641521), np.float64(475629.496065777)), simulated_mean=371717.6315, simulated_std=284683.3436, cvar_5_pct=13285.9987
      price_momentum_1m: best_distribution=skew_normal, aic=48786.8079, params=(np.float64(3.381188860339832), np.float64(-5.415272193390994), np.float64(18.820254900560514)), simulated_mean=8.8340, simulated_std=12.1265, cvar_5_pct=-10.8611
      price_momentum_3m: best_distribution=skew_normal, aic=57711.4823, params=(np.float64(3.307012448763337), np.float64(-21.936113196193894), np.float64(32.9571903173009)), simulated_mean=2.9768, simulated_std=21.4966, cvar_5_pct=-31.6504
      price_momentum_6m: best_distribution=skew_normal, aic=63383.2248, params=(np.float64(4.587556375308937), np.float64(-29.13173595939373), np.float64(55.53966943048583)), simulated_mean=14.3180, simulated_std=34.5877, cvar_5_pct=-37.8109
      price_momentum_1y: best_distribution=skew_normal, aic=71547.1913, params=(np.float64(11.615277774558267), np.float64(-40.62929743213173), np.float64(122.16539912784154)), simulated_mean=57.8412, simulated_std=74.2325, cvar_5_pct=-43.3020
      price_momentum_5d: best_distribution=skew_normal, aic=39474.1447, params=(np.float64(2.1008435794184566), np.float64(-4.938159830472012), np.float64(7.305794245539436)), simulated_mean=0.4574, simulated_std=5.1088, cvar_5_pct=-8.6217
      ema_crossover_20_50: best_distribution=skew_normal, aic=13790.4466, params=(np.float64(-4284142.444909643), np.float64(1.0000016833735024), np.float64(1.3588101172954925)), simulated_mean=-0.0975, simulated_std=0.8296, cvar_5_pct=-2.2246
      ema_crossover_50_250: best_distribution=skew_normal, aic=12789.6158, params=(np.float64(-8079670.50189862), np.float64(1.000000867671674), np.float64(1.2605609344333164)), simulated_mean=0.0080, simulated_std=0.7477, cvar_5_pct=-1.8929
      price_vs_ema_20d: best_distribution=skew_normal, aic=-21125.8334, params=(np.float64(3.0432096492350516), np.float64(-0.03951025097308923), np.float64(0.07709354852617165)), simulated_mean=0.0186, simulated_std=0.0505, cvar_5_pct=-0.0653
      price_vs_ema_250d: best_distribution=skew_normal, aic=-1283.6781, params=(np.float64(3.2151526362352456), np.float64(-0.17873535659467946), np.float64(0.3546806022943918)), simulated_mean=0.0838, simulated_std=0.2315, cvar_5_pct=-0.2918
      pct_off_52w_high: best_distribution=skew_normal, aic=-9199.2886, params=(np.float64(318.3734659709179), np.float64(-0.9957911626990384), np.float64(0.23887684917773422)), simulated_mean=-0.8036, simulated_std=0.1430, cvar_5_pct=-0.9873
      pct_above_52w_low: best_distribution=skew_normal, aic=11040.4035, params=(np.float64(131404261.01998824), np.float64(-0.989858194235925), np.float64(1.125049009149751)), simulated_mean=-0.0847, simulated_std=0.6956, cvar_5_pct=-0.9558
      range_52w_position: best_distribution=skew_normal, aic=956.1991, params=(np.float64(-44800375.7588937), np.float64(0.992915849927122), np.float64(0.5204020788941208)), simulated_mean=0.5771, simulated_std=0.3146, cvar_5_pct=-0.2309
      beta_momentum: best_distribution=student_t, aic=8635.3104, params=(np.float64(11.892463445338475), np.float64(-0.008910476712397697), np.float64(0.4523556076832408)), simulated_mean=-0.0059, simulated_std=0.4957, cvar_5_pct=-1.0648
      volatility_regime: best_distribution=skew_normal, aic=1277.2811, params=(np.float64(1.9525411512256308), np.float64(0.7586735627264206), np.float64(0.38506043894583286)), simulated_mean=1.0334, simulated_std=0.2705, cvar_5_pct=0.5413
      volatility_trend_short: best_distribution=student_t, aic=48028.3501, params=(np.float64(4.318318778589094), np.float64(2.85634191544393), np.float64(7.478181557276194)), simulated_mean=2.8308, simulated_std=10.1483, cvar_5_pct=-20.2890
      volatility_trend_long: best_distribution=student_t, aic=39812.6907, params=(np.float64(3.4659644338062057), np.float64(-1.8601612651979338), np.float64(3.7427190578177654)), simulated_mean=-1.7590, simulated_std=5.6177, cvar_5_pct=-13.9541
      vol_ratio_3m_1y: best_distribution=skew_normal, aic=-2532.9749, params=(np.float64(-0.8047780406419685), np.float64(1.235895881870976), np.float64(0.23027631711841928)), simulated_mean=1.1206, simulated_std=0.2008, cvar_5_pct=0.6966
      vol_hump: best_distribution=student_t, aic=34419.0447, params=(np.float64(2.632503444926697), np.float64(-0.9654414878907775), np.float64(2.239153172822755)), simulated_mean=-0.8576, simulated_std=4.2527, cvar_5_pct=-10.8292
      beta_term_structure: best_distribution=student_t, aic=17520.7246, params=(np.float64(1.6131880356437969), np.float64(-0.07873744235587024), np.float64(0.5225277381861528)), simulated_mean=-0.1342, simulated_std=2.4064, cvar_5_pct=-5.3353
      beta_convexity: best_distribution=student_t, aic=1152.5485, params=(np.float64(13.1875999253006), np.float64(0.015793061970146585), np.float64(0.24619707480728886)), simulated_mean=0.0170, simulated_std=0.2627, cvar_5_pct=-0.5489
      realized_vs_implied_proxy: best_distribution=skew_normal, aic=1277.2811, params=(np.float64(1.9525411512256308), np.float64(0.7586735627264206), np.float64(0.38506043894583286)), simulated_mean=1.0310, simulated_std=0.2661, cvar_5_pct=0.5417
      price_momentum_5y: best_distribution=student_t, aic=72023.6387, params=(np.float64(1.4524620419872611), np.float64(10.95506353928344), np.float64(66.29889079113136)), simulated_mean=16.2286, simulated_std=828.9202, cvar_5_pct=-750.1316
      long_term_trend_score: best_distribution=student_t, aic=16413.0949, params=(np.float64(1.5564075565135294), np.float64(0.2043736282711735), np.float64(0.4197406296306154)), simulated_mean=0.2216, simulated_std=1.8784, cvar_5_pct=-3.5591
      price_momentum_3y: best_distribution=student_t, aic=75096.0447, params=(np.float64(1.511506407053853), np.float64(18.92904014418911), np.float64(54.59657718727655)), simulated_mean=18.9297, simulated_std=333.1227, cvar_5_pct=-532.1749
      multi_year_high_flag: best_distribution=skew_normal, aic=2952.5750, params=(np.float64(3469299.7992635025), np.float64(-9.251744703719385e-07), np.float64(0.6035010291510221)), simulated_mean=0.4829, simulated_std=0.3657, cvar_5_pct=0.0190
      secular_trend_flag: best_distribution=skew_normal, aic=3414.0779, params=(np.float64(12026462.606994398), np.float64(-2.794118805798756e-07), np.float64(0.6246985226572321)), simulated_mean=0.5019, simulated_std=0.3805, cvar_5_pct=0.0191
    
    ============================================================
      Distribution Fitting: Price Target Dynamics (14 features)
    ============================================================
      pt_momentum_1w: best_distribution=skew_normal, aic=-34217.0287, params=(np.float64(1.9953016382148896), np.float64(-0.01429026159099532), np.float64(0.025748010029451183)), simulated_mean=0.0044, simulated_std=0.0177, cvar_5_pct=-0.0282
      pt_momentum_1m: best_distribution=skew_normal, aic=-19172.2112, params=(np.float64(1.8826430887702497), np.float64(-0.04617140916300629), np.float64(0.07948442604640077)), simulated_mean=0.0102, simulated_std=0.0566, cvar_5_pct=-0.0934
      pt_momentum_3m: best_distribution=student_t, aic=-6733.6401, params=(np.float64(2.1023293319358656), np.float64(0.020279768283105657), np.float64(0.08584898059960752)), simulated_mean=0.0175, simulated_std=0.2220, cvar_5_pct=-0.4897
      pt_momentum_6m: best_distribution=student_t, aic=881.5382, params=(np.float64(2.25142663100622), np.float64(0.05324842272723496), np.float64(0.16070124152769216)), simulated_mean=0.0536, simulated_std=0.3752, cvar_5_pct=-0.7453
      pt_momentum_1y: best_distribution=skew_normal, aic=7523.0379, params=(np.float64(5.9851588012202575), np.float64(-0.32865032417954954), np.float64(0.7922157550528268)), simulated_mean=0.2966, simulated_std=0.4888, cvar_5_pct=-0.4100
      analyst_coverage_change_1m: best_distribution=skew_normal, aic=11316.5732, params=(np.float64(1.0042186216048195), np.float64(-0.366293864154276), np.float64(0.6882181608033964)), simulated_mean=0.0297, simulated_std=0.5686, cvar_5_pct=-1.0940
      analyst_coverage_change_3m: best_distribution=skew_normal, aic=17328.1080, params=(np.float64(0.8218238827082517), np.float64(-0.5457967383140702), np.float64(1.042508549889008)), simulated_mean=-0.0174, simulated_std=0.9087, cvar_5_pct=-1.8327
      analyst_coverage_change_1y: best_distribution=student_t, aic=25831.8057, params=(np.float64(3.12395967674639), np.float64(0.15354287036142084), np.float64(1.2240039331988934)), simulated_mean=0.1353, simulated_std=2.2373, cvar_5_pct=-4.7710
      pt_acceleration_long: best_distribution=student_t, aic=4691.6405, params=(np.float64(1.8046225776094211), np.float64(-0.07190092909496576), np.float64(0.19326025101740246)), simulated_mean=-0.0872, simulated_std=1.3264, cvar_5_pct=-1.6565
      pt_acceleration_short: best_distribution=student_t, aic=-11082.6856, params=(np.float64(1.4506420929278419), np.float64(-0.011837512729115693), np.float64(0.0480163083161534)), simulated_mean=-0.0039, simulated_std=0.4492, cvar_5_pct=-0.5093
      pt_median_momentum_1m: best_distribution=skew_normal, aic=-17724.5382, params=(np.float64(1.7801752923208944), np.float64(-0.05157346660664086), np.float64(0.08758947066310332)), simulated_mean=0.0092, simulated_std=0.0642, cvar_5_pct=-0.1093
      pt_median_momentum_3m: best_distribution=student_t, aic=-6085.2440, params=(np.float64(1.9711059993937357), np.float64(0.019362231224841826), np.float64(0.08710128142468945)), simulated_mean=0.0235, simulated_std=0.3949, cvar_5_pct=-0.4971
      pt_vs_price_momentum: best_distribution=skew_normal, aic=-4563.3010, params=(np.float64(1.8876747554434248), np.float64(-0.12225244737997382), np.float64(0.24320851651842948)), simulated_mean=0.0531, simulated_std=0.1746, cvar_5_pct=-0.2708
      analyst_coverage_trend: best_distribution=skew_normal, aic=-6221.8791, params=(np.float64(-1.2297287407803688), np.float64(0.12033735707761006), np.float64(0.19245876024938738)), simulated_mean=0.0016, simulated_std=0.1506, cvar_5_pct=-0.3328
    
    ============================================================
      Distribution Fitting: Profitability (19 features)
    ============================================================
      operating_margin_pct: best_distribution=student_t, aic=53058.2380, params=(np.float64(1.5263334653633271), np.float64(10.231588562204632), np.float64(7.449629054246779)), simulated_mean=9.9469, simulated_std=39.0522, cvar_5_pct=-62.7929
      ebitda_margin_pct: best_distribution=student_t, aic=55160.1471, params=(np.float64(1.6567532004783645), np.float64(14.914172925731737), np.float64(9.310309068331408)), simulated_mean=15.5736, simulated_std=52.3938, cvar_5_pct=-59.7829
      roic: best_distribution=student_t, aic=51490.2485, params=(np.float64(1.3793501635891405), np.float64(7.912126409769353), np.float64(5.689455000946369)), simulated_mean=8.0207, simulated_std=74.3541, cvar_5_pct=-66.7956
      rnd_intensity: best_distribution=skew_normal, aic=-20231.3820, params=(np.float64(27125178.003901206), np.float64(-2.064184870380369e-08), np.float64(0.10105278219804556)), simulated_mean=0.0809, simulated_std=0.0609, cvar_5_pct=0.0032
      equity_multiplier: best_distribution=student_t, aic=22663.5660, params=(np.float64(1.6477823559568874), np.float64(1.9516838263072853), np.float64(0.7110242249345557)), simulated_mean=1.9559, simulated_std=3.9001, cvar_5_pct=-4.2411
      gross_margin_trend_yoy: best_distribution=skew_normal, aic=-43507.9566, params=(np.float64(1.2316641067542013), np.float64(-0.005279612112041507), np.float64(0.009752535177499492)), simulated_mean=0.0007, simulated_std=0.0076, cvar_5_pct=-0.0141
      operating_margin_trend: best_distribution=skew_normal, aic=19470.1061, params=(np.float64(1.1152117582313648), np.float64(-0.757764281067939), np.float64(1.3821465346923056)), simulated_mean=0.0213, simulated_std=1.0793, cvar_5_pct=-2.1301
      net_margin_trend_yoy: best_distribution=skew_normal, aic=-37098.4542, params=(np.float64(1.0964123596464397), np.float64(-0.008282826547232525), np.float64(0.01514581761704532)), simulated_mean=0.0008, simulated_std=0.0122, cvar_5_pct=-0.0232
      ebitda_margin_trend: best_distribution=skew_normal, aic=19859.9511, params=(np.float64(-0.9254349887012319), np.float64(0.7224275566698639), np.float64(1.360466386985367)), simulated_mean=-0.0531, simulated_std=1.1457, cvar_5_pct=-2.4897
      margin_expansion_flag: best_distribution=skew_normal, aic=-6789.2701, params=(np.float64(10463457.353002239), np.float64(-1.5427782653448761e-07), np.float64(0.29093468919664467)), simulated_mean=0.2319, simulated_std=0.1771, cvar_5_pct=0.0092
      ebit_cagr_3y: best_distribution=student_t, aic=48293.0547, params=(np.float64(2.7708161740250175), np.float64(6.390367274995489), np.float64(18.248135472053825)), simulated_mean=6.2123, simulated_std=43.2720, cvar_5_pct=-80.0811
      ebit_growth_yoy: best_distribution=student_t, aic=71832.1367, params=(np.float64(1.2138039211407445), np.float64(10.705017351437668), np.float64(24.345641276213826)), simulated_mean=11.8694, simulated_std=201.5838, cvar_5_pct=-359.3582
      ebit_qoq_growth: best_distribution=student_t, aic=72142.0655, params=(np.float64(0.9009877950381397), np.float64(-0.5539491323311461), np.float64(18.327536695959694)), simulated_mean=61.9962, simulated_std=3526.2438, cvar_5_pct=-918.2626
      ebitda_qoq_growth: best_distribution=student_t, aic=67988.3256, params=(np.float64(1.059517500461403), np.float64(-1.1154655829428624), np.float64(16.444324485783508)), simulated_mean=-6.3821, simulated_std=1308.7813, cvar_5_pct=-678.0710
      gp_margin_trend: best_distribution=student_t, aic=46227.6475, params=(np.float64(1.5028692822919671), np.float64(-1.761168089536151), np.float64(6.409815579385635)), simulated_mean=0.2464, simulated_std=124.2501, cvar_5_pct=-76.6439
      gp_qoq_growth: best_distribution=student_t, aic=59779.4460, params=(np.float64(0.9483463766257444), np.float64(1.5541901876900348), np.float64(8.201469581374312)), simulated_mean=17.6699, simulated_std=1108.8044, cvar_5_pct=-431.8743
      gp_yoy_growth: best_distribution=student_t, aic=60555.2143, params=(np.float64(1.5362473509223435), np.float64(10.67193813212086), np.float64(13.566089030541708)), simulated_mean=9.4917, simulated_std=75.6541, cvar_5_pct=-135.0442
      ebitda_cagr_3y: best_distribution=student_t, aic=49332.0764, params=(np.float64(2.9832124576207026), np.float64(6.502193468593859), np.float64(15.131791766289007)), simulated_mean=7.0672, simulated_std=26.0364, cvar_5_pct=-53.3811
      margin_stability_score: best_distribution=skew_normal, aic=-47318.6612, params=(np.float64(-2098228.609447079), np.float64(100.0000000340525), np.float64(0.013485231355891157)), simulated_mean=99.9893, simulated_std=0.0081, cvar_5_pct=99.9688
    
    ============================================================
      Distribution Fitting: Quality & Risk (15 features)
    ============================================================
      net_buyback_flag: best_distribution=skew_normal, aic=2611.7344, params=(np.float64(2666263.7599638226), np.float64(-1.1268001188957028e-06), np.float64(0.5881612467586215)), simulated_mean=0.4697, simulated_std=0.3522, cvar_5_pct=0.0189
      beta_trend: best_distribution=student_t, aic=73355.9867, params=(np.float64(1.5723432289572203), np.float64(-5.195530428928258), np.float64(51.89894968185687)), simulated_mean=-17.4068, simulated_std=339.4070, cvar_5_pct=-657.6085
      shares_yoy_change_pct: best_distribution=skew_normal, aic=-13051.6808, params=(np.float64(5.802608260506135), np.float64(-0.06734978586631085), np.float64(0.1569464698072764)), simulated_mean=0.0542, simulated_std=0.0957, cvar_5_pct=-0.0843
      low_beta_flag: best_distribution=skew_normal, aic=2960.8187, params=(np.float64(1898968.3180174525), np.float64(-1.6212713063987445e-06), np.float64(0.6038541182239188)), simulated_mean=0.4888, simulated_std=0.3655, cvar_5_pct=0.0205
      has_goodwill_impairment: best_distribution=skew_normal, aic=-7103.7762, params=(np.float64(6510512.755396532), np.float64(-2.6087467244142946e-07), np.float64(0.28411771147958786)), simulated_mean=0.2268, simulated_std=0.1695, cvar_5_pct=0.0089
      has_asset_writedown: best_distribution=skew_normal, aic=3821.6751, params=(np.float64(6391141.68587975), np.float64(-5.602534260131974e-07), np.float64(0.6440759512046017)), simulated_mean=0.5088, simulated_std=0.3862, cvar_5_pct=0.0216
      has_restructuring: best_distribution=skew_normal, aic=-755.3398, params=(np.float64(10371190.363428708), np.float64(-2.5586400026613597e-07), np.float64(0.4570444694711596)), simulated_mean=0.3714, simulated_std=0.2821, cvar_5_pct=0.0150
      goodwill_to_assets_pct: best_distribution=skew_normal, aic=43715.1801, params=(np.float64(6069158.429693156), np.float64(-1.3181652967074516e-05), np.float64(15.764267021936163)), simulated_mean=12.5713, simulated_std=9.3996, cvar_5_pct=0.4787
      intangible_intensity: best_distribution=skew_normal, aic=-15472.2995, params=(np.float64(6365547.431077346), np.float64(-1.2041284946885186e-07), np.float64(0.1409670966462585)), simulated_mean=0.1116, simulated_std=0.0831, cvar_5_pct=0.0040
      exceptional_items_to_ebitda: best_distribution=skew_normal, aic=-10813.0441, params=(np.float64(3115428.484336477), np.float64(-3.545552418064041e-07), np.float64(0.21197940779293867)), simulated_mean=0.1690, simulated_std=0.1274, cvar_5_pct=0.0064
      altman_z_trend: best_distribution=skew_normal, aic=1389.1984, params=(np.float64(-1.3640849557968886), np.float64(0.20580281418302176), np.float64(0.3621913840878562)), simulated_mean=-0.0255, simulated_std=0.2748, cvar_5_pct=-0.6268
      quick_ratio: best_distribution=student_t, aic=19042.1462, params=(np.float64(1.2462292545157152), np.float64(1.0995021208496736), np.float64(0.4620687498912468)), simulated_mean=1.0576, simulated_std=5.0422, cvar_5_pct=-6.9449
      beta_spread: best_distribution=student_t, aic=8635.3104, params=(np.float64(11.892463445338475), np.float64(-0.008910476712397697), np.float64(0.4523556076832408)), simulated_mean=-0.0165, simulated_std=0.4884, cvar_5_pct=-1.0729
      high_beta_flag: best_distribution=skew_normal, aic=-4302.6020, params=(np.float64(9229748.888384573), np.float64(-2.0431718165207758e-07), np.float64(0.3504627990400615)), simulated_mean=0.2823, simulated_std=0.2142, cvar_5_pct=0.0111
      beta_stability_score: best_distribution=skew_normal, aic=52137.6107, params=(np.float64(-122531981.91775829), np.float64(100.00000109853511), np.float64(24.982374302846665)), simulated_mean=79.9639, simulated_std=15.2299, cvar_5_pct=40.8673
    
    ============================================================
      Distribution Fitting: Revenue Forecasting (16 features)
    ============================================================
      revenue_avg_med_diff_pct: best_distribution=skew_normal, aic=17071.2540, params=(np.float64(1.8458017374269204), np.float64(-0.7613698336325825), np.float64(1.3499853079155084)), simulated_mean=0.1664, simulated_std=0.9675, cvar_5_pct=-1.5998
      revenue_revision_trend: best_distribution=student_t, aic=-4382.8322, params=(np.float64(1.5918696855262493), np.float64(0.07733696321816899), np.float64(0.085343282526234)), simulated_mean=0.0835, simulated_std=0.4911, cvar_5_pct=-0.6333
      estimate_confidence_score: best_distribution=skew_normal, aic=11926.0793, params=(np.float64(-2158793.6755283396), np.float64(100.00000285403135), np.float64(1.1926071723947484)), simulated_mean=99.0347, simulated_std=0.7334, cvar_5_pct=97.1705
      consensus_revenue_growth: best_distribution=student_t, aic=57338.4005, params=(np.float64(1.2699248707766948), np.float64(7.505809404805312), np.float64(8.553959227474618)), simulated_mean=7.9238, simulated_std=88.5908, cvar_5_pct=-121.2698
      revenue_acceleration: best_distribution=student_t, aic=-4212.1409, params=(np.float64(1.8298005660390955), np.float64(-0.022415848415703626), np.float64(0.09280561594644343)), simulated_mean=-0.0245, simulated_std=0.2927, cvar_5_pct=-0.6673
      revenue_est_revision_trend: best_distribution=student_t, aic=-4382.8322, params=(np.float64(1.5918696855262493), np.float64(0.07733696321816899), np.float64(0.085343282526234)), simulated_mean=0.0860, simulated_std=1.1822, cvar_5_pct=-0.7328
      revenue_2y_growth: best_distribution=student_t, aic=61936.4101, params=(np.float64(1.8577685451616675), np.float64(10.828187722851148), np.float64(18.456290016912583)), simulated_mean=12.5409, simulated_std=149.2556, cvar_5_pct=-115.5447
      revenue_3y_growth: best_distribution=student_t, aic=66716.8446, params=(np.float64(1.8071163660803289), np.float64(17.066207365648268), np.float64(28.34558750547899)), simulated_mean=20.1897, simulated_std=96.3586, cvar_5_pct=-150.3406
      revenue_4q_trend: best_distribution=student_t, aic=57177.9833, params=(np.float64(1.6151222102268261), np.float64(10.339078270932182), np.float64(11.981761763529764)), simulated_mean=10.6892, simulated_std=67.4349, cvar_5_pct=-105.1068
      revenue_qoq_growth: best_distribution=student_t, aic=52441.8508, params=(np.float64(1.3301283552745877), np.float64(1.8761625724710167), np.float64(6.617501138200117)), simulated_mean=1.0493, simulated_std=41.2771, cvar_5_pct=-83.9775
      revenue_stability_score: best_distribution=skew_normal, aic=58205.2174, params=(np.float64(-21535298.506561384), np.float64(100.0000090706709), np.float64(37.83217482296524)), simulated_mean=69.7436, simulated_std=23.0667, cvar_5_pct=9.9745
      revenue_accelerating_flag: best_distribution=skew_normal, aic=2717.8070, params=(np.float64(-9343251.533982359), np.float64(1.000000336716787), np.float64(0.592913107290105)), simulated_mean=0.5328, simulated_std=0.3499, cvar_5_pct=-0.3406
      revenue_cagr_3y: best_distribution=student_t, aic=50429.4924, params=(np.float64(2.6970404625571245), np.float64(5.947373406177544), np.float64(9.512525480833844)), simulated_mean=6.1587, simulated_std=18.9218, cvar_5_pct=-36.2348
      revenue_cagr_4y: best_distribution=student_t, aic=49029.9230, params=(np.float64(2.5495642501562266), np.float64(7.484502656137672), np.float64(8.462488983332252)), simulated_mean=7.5563, simulated_std=17.7080, cvar_5_pct=-29.2662
      revenue_growth_flag: best_distribution=skew_normal, aic=-501.7892, params=(np.float64(-7205252.744260957), np.float64(1.0000003602366876), np.float64(0.4658819984953977)), simulated_mean=0.6310, simulated_std=0.2765, cvar_5_pct=-0.0737
      revenue_4y_growth: best_distribution=student_t, aic=70617.8916, params=(np.float64(1.4787149965313628), np.float64(29.1784024894679), np.float64(34.70164393487603)), simulated_mean=22.0393, simulated_std=517.7693, cvar_5_pct=-504.2802
    
    ============================================================
      Distribution Fitting: Technical Analysis (11 features)
    ============================================================
      high_volume_flag: best_distribution=skew_normal, aic=-5611.5937, params=(np.float64(7971918.608174043), np.float64(-2.2239564424891742e-07), np.float64(0.31774059388314324)), simulated_mean=0.2559, simulated_std=0.1946, cvar_5_pct=0.0094
      ema_slope_20d: best_distribution=skew_normal, aic=-22982.9449, params=(np.float64(2.0076221181247265), np.float64(-0.03488874297930722), np.float64(0.060604320297019065)), simulated_mean=0.0080, simulated_std=0.0423, cvar_5_pct=-0.0685
      price_vs_ema_100d: best_distribution=skew_normal, aic=51243.6350, params=(np.float64(2.9044842413620726), np.float64(-10.747462641727044), np.float64(19.744508827627133)), simulated_mean=4.2383, simulated_std=13.0024, cvar_5_pct=-17.3984
      near_52w_low_flag: best_distribution=skew_normal, aic=-10753.6303, params=(np.float64(12574120.314742478), np.float64(-9.886411817213632e-08), np.float64(0.21617364455277502)), simulated_mean=0.1751, simulated_std=0.1298, cvar_5_pct=0.0075
      volume_momentum_score: best_distribution=student_t, aic=-9296.9666, params=(np.float64(1.9615197821892218), np.float64(0.04594583920789583), np.float64(0.06661882065119372)), simulated_mean=0.0453, simulated_std=0.1724, cvar_5_pct=-0.3505
      breakout_signal: best_distribution=skew_normal, aic=-2639.3143, params=(np.float64(5827464.372618888), np.float64(-3.8347378682533175e-07), np.float64(0.39694653859596274)), simulated_mean=0.3188, simulated_std=0.2365, cvar_5_pct=0.0132
      volatility_compression: best_distribution=student_t, aic=50799.5334, params=(np.float64(3.9134674944253254), np.float64(-0.5021393839183463), np.float64(9.005849859439383)), simulated_mean=-0.2739, simulated_std=12.5784, cvar_5_pct=-29.1952
      volatility_term_structure: best_distribution=student_t, aic=41880.8417, params=(np.float64(3.501496390382695), np.float64(3.13622206567124), np.float64(4.395949920258214)), simulated_mean=3.1280, simulated_std=6.8809, cvar_5_pct=-12.8167
      near_52w_high_flag: best_distribution=skew_normal, aic=-2532.2957, params=(np.float64(7627409.9427103475), np.float64(-2.8983157726652606e-07), np.float64(0.4001297927855745)), simulated_mean=0.3204, simulated_std=0.2424, cvar_5_pct=0.0123
      low_volume_flag: best_distribution=skew_normal, aic=-4434.1536, params=(np.float64(10849813.301317915), np.float64(-1.733314739681858e-07), np.float64(0.3470450739812614)), simulated_mean=0.2723, simulated_std=0.2044, cvar_5_pct=0.0116
      ema_trend_consistency: best_distribution=skew_normal, aic=11626.0848, params=(np.float64(-6272436.966237215), np.float64(1.0000009785234738), np.float64(1.1556098559799999)), simulated_mean=0.0800, simulated_std=0.7044, cvar_5_pct=-1.7407
    
    ============================================================
      Distribution Fitting: Temporal Patterns (7 features)
    ============================================================
      days_to_earnings: best_distribution=skew_normal, aic=55378.0851, params=(np.float64(27987503.14838542), np.float64(-1.0000059748823826), np.float64(32.70830080371502)), simulated_mean=25.0114, simulated_std=19.7278, cvar_5_pct=0.0435
      earnings_report_recency: best_distribution=skew_normal, aic=66901.3325, params=(np.float64(2.1736614262227647), np.float64(84.17211263707068), np.float64(55.07308827487145)), simulated_mean=123.9773, simulated_std=38.2388, cvar_5_pct=56.0215
      fiscal_year_progress: best_distribution=skew_normal, aic=-8458.2951, params=(np.float64(-12124025.139983311), np.float64(1.0000001166432853), np.float64(0.2532561505478317)), simulated_mean=0.7959, simulated_std=0.1554, cvar_5_pct=0.3962
      days_since_last_report: best_distribution=skew_normal, aic=66901.3325, params=(np.float64(2.1736614262227647), np.float64(84.17211263707068), np.float64(55.07308827487145)), simulated_mean=124.6183, simulated_std=37.9551, cvar_5_pct=56.5981
      days_to_fy_end: best_distribution=skew_normal, aic=75723.8152, params=(np.float64(-17.599051807603438), np.float64(-91.44261170046657), np.float64(151.1389898164407)), simulated_mean=-211.8974, simulated_std=90.5907, cvar_5_pct=-438.6832
      reporting_freshness_score: best_distribution=skew_normal, aic=44492.9043, params=(np.float64(3135057.0491082435), np.float64(-2.526024967610743e-05), np.float64(13.599654030214356)), simulated_mean=10.6597, simulated_std=7.9069, cvar_5_pct=0.4288
    
    ============================================================
      Distribution Fitting: Valuation Ratios (7 features)
    ============================================================
      peg_ratio: best_distribution=student_t, aic=23578.3442, params=(np.float64(1.4331750454198937), np.float64(0.2798422008996233), np.float64(1.7970308615112547)), simulated_mean=0.3553, simulated_std=13.7030, cvar_5_pct=-21.0467
      tangible_book_per_share: best_distribution=student_t, aic=-132745.7431, params=(np.float64(1.9874835012625365), np.float64(2.153881354739988e-06), np.float64(4.224691374110449e-06)), simulated_mean=0.0000, simulated_std=0.0000, cvar_5_pct=-0.0000
      price_to_tangible_book: best_distribution=student_t, aic=33337.6075, params=(np.float64(1.081471053636943), np.float64(3.106308713461168), np.float64(2.0557356467808034)), simulated_mean=3.5575, simulated_std=59.6097, cvar_5_pct=-58.8764
      tangible_equity_ratio: best_distribution=skew_normal, aic=60254.5171, params=(np.float64(-1.365090811246247), np.float64(55.83113255868044), np.float64(40.60699118549641)), simulated_mean=29.3867, simulated_std=30.6436, cvar_5_pct=-37.2488
      intangibles_to_equity: best_distribution=skew_normal, aic=58435.5134, params=(np.float64(46.73675158785749), np.float64(-2.3791659775309637), np.float64(42.531308136152276)), simulated_mean=31.3912, simulated_std=25.7242, cvar_5_pct=-1.1471
      goodwill_to_equity: best_distribution=skew_normal, aic=59883.2719, params=(np.float64(18.701498403909735), np.float64(-5.085849210527281), np.float64(46.42582046795563)), simulated_mean=31.9373, simulated_std=28.1371, cvar_5_pct=-4.6867
      tbv_yoy_growth: best_distribution=skew_normal, aic=47738.2923, params=(np.float64(1.2888031956369872), np.float64(-6.291067666438858), np.float64(12.23453767537595)), simulated_mean=1.4731, simulated_std=9.3746, cvar_5_pct=-16.7174
    
    ============================================================
      Distribution Fitting: Valuation Timeseries (15 features)
    ============================================================
      ev_sales_trend_1y: best_distribution=student_t, aic=-1377.0828, params=(np.float64(1.5561327100348), np.float64(0.020477103390342408), np.float64(0.10730301265993802)), simulated_mean=0.0253, simulated_std=0.4278, cvar_5_pct=-0.8422
      ev_ebitda_momentum: best_distribution=student_t, aic=-1540.5055, params=(np.float64(1.5551641382696755), np.float64(0.0183267474377857), np.float64(0.10478256756062328)), simulated_mean=0.0224, simulated_std=0.6960, cvar_5_pct=-0.9614
      p_e_momentum_yoy: best_distribution=student_t, aic=151.9806, params=(np.float64(1.5487604906075472), np.float64(0.024823619624219764), np.float64(0.12108011285034023)), simulated_mean=0.0372, simulated_std=0.7787, cvar_5_pct=-1.1309
      p_e_momentum_qoq: best_distribution=student_t, aic=5424.1309, params=(np.float64(1.5856635690137115), np.float64(0.029530923911477443), np.float64(0.20735149683279963)), simulated_mean=0.0313, simulated_std=1.5645, cvar_5_pct=-2.0369
      ev_sales_vs_3y_avg: best_distribution=skew_normal, aic=6054.5666, params=(np.float64(3.968911583834573), np.float64(-0.3706113140644629), np.float64(0.6797846159902423)), simulated_mean=0.1539, simulated_std=0.4268, cvar_5_pct=-0.5217
      ev_ebitda_vs_3y_avg: best_distribution=skew_normal, aic=5851.3899, params=(np.float64(3.599476687828214), np.float64(-0.41138315122573993), np.float64(0.6883732186302711)), simulated_mean=0.1250, simulated_std=0.4535, cvar_5_pct=-0.5932
      p_e_vs_3y_avg: best_distribution=skew_normal, aic=7983.9868, params=(np.float64(5.910755190681336), np.float64(-0.5656787530533787), np.float64(1.0108251460844044)), simulated_mean=0.2265, simulated_std=0.6235, cvar_5_pct=-0.6752
      ev_sales_forward_discount: best_distribution=student_t, aic=-7694.3210, params=(np.float64(2.932291115705703), np.float64(-0.08237214138186794), np.float64(0.09047209074338933)), simulated_mean=-0.0856, simulated_std=0.1439, cvar_5_pct=-0.4224
      ev_ebitda_forward_discount: best_distribution=student_t, aic=-747.5857, params=(np.float64(3.3763765041985465), np.float64(-0.14473699999641315), np.float64(0.16546174307060998)), simulated_mean=-0.1472, simulated_std=0.2729, cvar_5_pct=-0.7719
      p_e_forward_discount: best_distribution=student_t, aic=2282.2747, params=(np.float64(3.6034663233343003), np.float64(-0.21988783459272543), np.float64(0.2278464103962588)), simulated_mean=-0.2285, simulated_std=0.3412, cvar_5_pct=-1.0236
      p_b_vs_5y_avg: best_distribution=skew_normal, aic=8241.0422, params=(np.float64(6.952434315567908), np.float64(0.4005922363798493), np.float64(0.9456091255830705)), simulated_mean=1.1436, simulated_std=0.5762, cvar_5_pct=0.3263
      ev_sales_qoq_1q: best_distribution=student_t, aic=1356.8728, params=(np.float64(2.085172204159309), np.float64(0.01823593249362983), np.float64(0.16036890353992236)), simulated_mean=0.0195, simulated_std=0.3609, cvar_5_pct=-0.8388
      p_b_momentum_yoy: best_distribution=student_t, aic=-107.4080, params=(np.float64(1.7815598995365112), np.float64(0.028155932679765593), np.float64(0.13017970366370277)), simulated_mean=0.0323, simulated_std=0.4857, cvar_5_pct=-0.7855
      forward_pe_premium: best_distribution=student_t, aic=47632.2699, params=(np.float64(7.277923361197889), np.float64(-19.18340563048214), np.float64(30.557038948740626)), simulated_mean=-19.1379, simulated_std=35.4648, cvar_5_pct=-96.0607
      ev_ebitda_qoq_trend: best_distribution=student_t, aic=1564.7321, params=(np.float64(2.131578683398654), np.float64(0.014208630497406281), np.float64(0.16673331336301567)), simulated_mean=0.0141, simulated_std=0.4400, cvar_5_pct=-0.9198

## 24. Sector-Level Summary Dashboard

```python
sector_col = next((c for c in ["industry", "sector"] if c in df.columns), None)
if sector_col:
    sector_metrics = df.groupby(sector_col).agg(
        count=("ticker", "count"),
        median_pe=("p_e_ratio", "median") if "p_e_ratio" in df.columns else ("ticker", "count"),
        median_roe=("roe", "median") if "roe" in df.columns else ("ticker", "count"),
        median_momentum=("price_momentum_1y", "median") if "price_momentum_1y" in df.columns else ("ticker", "count"),
        median_upside=("upside_potential", "median") if "upside_potential" in df.columns else ("ticker", "count"),
    ).sort_values("count", ascending=False)

    print("Sector-Level Summary:")
    display(sector_metrics.head(50))

    # Sector comparison radar-style bar chart
    fig, axes = plt.subplots(2, 2, figsize=(25, 15), constrained_layout=True)
    fig.suptitle("Sector Comparison Dashboard", fontsize=16, fontweight="bold")

    metrics_to_plot = [
        ("median_pe", "Median P/E Ratio", COLORS["primary"]),
        ("median_roe", "Median ROE (%)", COLORS["secondary"]),
        ("median_momentum", "Median 1Y Momentum (%)", COLORS["info"]),
        ("median_upside", "Median Upside (%)", COLORS["light"]),
    ]

    top_sectors = sector_metrics.head(50)
    for idx, (col, title, color) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        if col in top_sectors.columns:
            vals = top_sectors[col].fillna(0)
            bar_colors = [COLORS["danger"] if v < 0 else color for v in vals]
            ax.barh(top_sectors.index, vals, color=bar_colors, edgecolor="white", alpha=0.85)
            ax.set_title(title, fontweight="bold")
            ax.axvline(0, color="grey", linestyle=":", lw=0.8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(axis="y", labelsize=8)

    plt.show()
```

    Sector-Level Summary:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>median_pe</th>
      <th>median_roe</th>
      <th>median_momentum</th>
      <th>median_upside</th>
    </tr>
    <tr>
      <th>industry</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Metals and Mining</th>
      <td>350</td>
      <td>24.5000</td>
      <td>0.0774</td>
      <td>90.2662</td>
      <td>18.7427</td>
    </tr>
    <tr>
      <th>Machinery</th>
      <td>324</td>
      <td>28.6000</td>
      <td>0.1124</td>
      <td>32.5645</td>
      <td>13.3197</td>
    </tr>
    <tr>
      <th>Oil Gas and Consumable Fuels</th>
      <td>285</td>
      <td>16.4000</td>
      <td>0.0977</td>
      <td>44.3563</td>
      <td>11.1973</td>
    </tr>
    <tr>
      <th>Chemicals</th>
      <td>283</td>
      <td>28.0000</td>
      <td>0.0578</td>
      <td>33.6522</td>
      <td>9.9476</td>
    </tr>
    <tr>
      <th>Semiconductors and Semiconductor Equipment</th>
      <td>256</td>
      <td>61.8500</td>
      <td>0.0798</td>
      <td>115.9244</td>
      <td>-0.5414</td>
    </tr>
    <tr>
      <th>Software</th>
      <td>253</td>
      <td>31.1000</td>
      <td>0.0794</td>
      <td>-8.5947</td>
      <td>41.9445</td>
    </tr>
    <tr>
      <th>Electronic Equipment Instruments and Components</th>
      <td>248</td>
      <td>40.5000</td>
      <td>0.0999</td>
      <td>61.2690</td>
      <td>8.5410</td>
    </tr>
    <tr>
      <th>Biotechnology</th>
      <td>243</td>
      <td>27.8500</td>
      <td>-0.2925</td>
      <td>45.9656</td>
      <td>46.3544</td>
    </tr>
    <tr>
      <th>Food Products</th>
      <td>241</td>
      <td>17.8000</td>
      <td>0.1170</td>
      <td>5.3165</td>
      <td>16.4557</td>
    </tr>
    <tr>
      <th>Electrical Equipment</th>
      <td>218</td>
      <td>39.8000</td>
      <td>0.0895</td>
      <td>81.4523</td>
      <td>6.0255</td>
    </tr>
    <tr>
      <th>Pharmaceuticals</th>
      <td>200</td>
      <td>22.9000</td>
      <td>0.0810</td>
      <td>27.8777</td>
      <td>26.3590</td>
    </tr>
    <tr>
      <th>Specialty Retail</th>
      <td>187</td>
      <td>17.7000</td>
      <td>0.1394</td>
      <td>5.2134</td>
      <td>24.1196</td>
    </tr>
    <tr>
      <th>Hotels Restaurants and Leisure</th>
      <td>183</td>
      <td>21.5000</td>
      <td>0.1244</td>
      <td>1.2141</td>
      <td>28.1250</td>
    </tr>
    <tr>
      <th>Construction and Engineering</th>
      <td>173</td>
      <td>18.8000</td>
      <td>0.1310</td>
      <td>44.0168</td>
      <td>13.9241</td>
    </tr>
    <tr>
      <th>Health Care Providers and Services</th>
      <td>153</td>
      <td>21.7000</td>
      <td>0.0932</td>
      <td>5.6128</td>
      <td>24.5263</td>
    </tr>
    <tr>
      <th>Health Care Equipment and Supplies</th>
      <td>146</td>
      <td>28.6000</td>
      <td>0.0696</td>
      <td>-5.7643</td>
      <td>29.4544</td>
    </tr>
    <tr>
      <th>Automobile Components</th>
      <td>136</td>
      <td>21.7000</td>
      <td>0.0860</td>
      <td>22.3410</td>
      <td>19.1576</td>
    </tr>
    <tr>
      <th>IT Services</th>
      <td>134</td>
      <td>22.8000</td>
      <td>0.1242</td>
      <td>-1.6529</td>
      <td>27.8595</td>
    </tr>
    <tr>
      <th>Household Durables</th>
      <td>131</td>
      <td>14.2500</td>
      <td>0.1043</td>
      <td>2.4527</td>
      <td>23.9560</td>
    </tr>
    <tr>
      <th>Consumer Staples Distribution and Retail</th>
      <td>121</td>
      <td>20.3000</td>
      <td>0.1220</td>
      <td>-0.8170</td>
      <td>18.5089</td>
    </tr>
    <tr>
      <th>Aerospace and Defense</th>
      <td>111</td>
      <td>51.2000</td>
      <td>0.0917</td>
      <td>40.9145</td>
      <td>17.9875</td>
    </tr>
    <tr>
      <th>Commercial Services and Supplies</th>
      <td>110</td>
      <td>23.9500</td>
      <td>0.0989</td>
      <td>14.0378</td>
      <td>19.9523</td>
    </tr>
    <tr>
      <th>Professional Services</th>
      <td>100</td>
      <td>20.5000</td>
      <td>0.1344</td>
      <td>-17.8312</td>
      <td>33.8820</td>
    </tr>
    <tr>
      <th>Media</th>
      <td>95</td>
      <td>18.5000</td>
      <td>0.0626</td>
      <td>0.6793</td>
      <td>24.0524</td>
    </tr>
    <tr>
      <th>Independent Power and Renewable Electricity Producers</th>
      <td>95</td>
      <td>21.3500</td>
      <td>0.0736</td>
      <td>23.4475</td>
      <td>11.1330</td>
    </tr>
    <tr>
      <th>Electric Utilities</th>
      <td>91</td>
      <td>18.9000</td>
      <td>0.1003</td>
      <td>20.7539</td>
      <td>5.5519</td>
    </tr>
    <tr>
      <th>Diversified Telecommunication Services</th>
      <td>89</td>
      <td>15.7000</td>
      <td>0.1041</td>
      <td>15.6534</td>
      <td>10.2577</td>
    </tr>
    <tr>
      <th>Entertainment</th>
      <td>88</td>
      <td>23.9000</td>
      <td>0.1033</td>
      <td>1.9761</td>
      <td>27.5785</td>
    </tr>
    <tr>
      <th>Energy Equipment and Services</th>
      <td>85</td>
      <td>19.6500</td>
      <td>0.1071</td>
      <td>70.1056</td>
      <td>7.3025</td>
    </tr>
    <tr>
      <th>Beverages</th>
      <td>85</td>
      <td>18.7000</td>
      <td>0.1237</td>
      <td>-8.7428</td>
      <td>19.4737</td>
    </tr>
    <tr>
      <th>Building Products</th>
      <td>83</td>
      <td>23.2000</td>
      <td>0.1094</td>
      <td>1.1572</td>
      <td>16.7820</td>
    </tr>
    <tr>
      <th>Trading Companies and Distributors</th>
      <td>83</td>
      <td>20.7000</td>
      <td>0.1078</td>
      <td>16.6513</td>
      <td>15.4048</td>
    </tr>
    <tr>
      <th>Textiles Apparel and Luxury Goods</th>
      <td>82</td>
      <td>18.5000</td>
      <td>0.1209</td>
      <td>17.9434</td>
      <td>21.6598</td>
    </tr>
    <tr>
      <th>Construction Materials</th>
      <td>81</td>
      <td>18.8000</td>
      <td>0.0946</td>
      <td>-1.5869</td>
      <td>16.5021</td>
    </tr>
    <tr>
      <th>Ground Transportation</th>
      <td>75</td>
      <td>16.8500</td>
      <td>0.0902</td>
      <td>10.3452</td>
      <td>11.9205</td>
    </tr>
    <tr>
      <th>Technology Hardware Storage and Peripherals</th>
      <td>67</td>
      <td>24.8500</td>
      <td>0.1209</td>
      <td>36.0976</td>
      <td>13.1222</td>
    </tr>
    <tr>
      <th>Automobiles</th>
      <td>67</td>
      <td>16.3000</td>
      <td>0.0493</td>
      <td>7.1337</td>
      <td>22.2930</td>
    </tr>
    <tr>
      <th>Communications Equipment</th>
      <td>66</td>
      <td>59.8000</td>
      <td>0.0956</td>
      <td>84.7057</td>
      <td>4.7030</td>
    </tr>
    <tr>
      <th>Life Sciences Tools and Services</th>
      <td>64</td>
      <td>34.4000</td>
      <td>0.0610</td>
      <td>5.6942</td>
      <td>30.8844</td>
    </tr>
    <tr>
      <th>Industrial Conglomerates</th>
      <td>63</td>
      <td>19.8000</td>
      <td>0.0917</td>
      <td>14.4977</td>
      <td>17.0984</td>
    </tr>
    <tr>
      <th>Transportation Infrastructure</th>
      <td>61</td>
      <td>17.2000</td>
      <td>0.0986</td>
      <td>15.3086</td>
      <td>12.1324</td>
    </tr>
    <tr>
      <th>Broadline Retail</th>
      <td>59</td>
      <td>20.9000</td>
      <td>0.1113</td>
      <td>6.2753</td>
      <td>16.1572</td>
    </tr>
    <tr>
      <th>Interactive Media and Services</th>
      <td>55</td>
      <td>24.3000</td>
      <td>0.0803</td>
      <td>-11.6832</td>
      <td>30.1766</td>
    </tr>
    <tr>
      <th>Diversified Consumer Services</th>
      <td>48</td>
      <td>17.6500</td>
      <td>0.1431</td>
      <td>20.7152</td>
      <td>23.0607</td>
    </tr>
    <tr>
      <th>Personal Care Products</th>
      <td>46</td>
      <td>23.3000</td>
      <td>0.1116</td>
      <td>-1.7757</td>
      <td>14.5985</td>
    </tr>
    <tr>
      <th>Marine Transportation</th>
      <td>45</td>
      <td>9.8500</td>
      <td>0.1062</td>
      <td>35.5418</td>
      <td>5.0776</td>
    </tr>
    <tr>
      <th>Passenger Airlines</th>
      <td>45</td>
      <td>8.9000</td>
      <td>0.1434</td>
      <td>15.3242</td>
      <td>21.7019</td>
    </tr>
    <tr>
      <th>Wireless Telecommunication Services</th>
      <td>44</td>
      <td>18.9000</td>
      <td>0.1533</td>
      <td>22.3826</td>
      <td>8.2329</td>
    </tr>
    <tr>
      <th>Air Freight and Logistics</th>
      <td>43</td>
      <td>16.8000</td>
      <td>0.1117</td>
      <td>29.3049</td>
      <td>14.3552</td>
    </tr>
    <tr>
      <th>Containers and Packaging</th>
      <td>42</td>
      <td>19.0500</td>
      <td>0.0908</td>
      <td>-4.3478</td>
      <td>23.6551</td>
    </tr>
  </tbody>
</table>
</div>




![png](pml_model_analysis_files/pml_model_analysis_58_2.png)

## 25. Core PML Bayesian Modeling (PyMC & ArviZ)

This section implements the next-generation probabilistic models using PyMC for full Bayesian inference. Unlike the
baseline models above, these models produce full posterior distributions, allowing for advanced risk analysis and
uncertainty quantification.

### 25.1 Monte Carlo Return Simulation

Generates probabilistic return distributions with learnable mean/variance priors.

```python
from probabilistic_ml_model.pymc_models.MonteCarloSimulation import fit as mc_fit

pml_df = df.copy()
mu_col = "total_return_ytd"
std_col = "volatility_1y"

valid_mask = (
        pml_df[mu_col].notna() &
        pml_df[std_col].notna() &
        (pml_df[std_col] > 0)
)

pml_df_valid = pml_df[valid_mask].copy()

if len(pml_df_valid) > 0:
    print(f"Running Monte Carlo Simulation for {len(pml_df_valid)} stocks...")
    mc_idata = mc_fit(
        historical_means=pml_df_valid[mu_col].values / 100,
        historical_stds=pml_df_valid[std_col].clip(lower=1.0).values / 100,
        tickers=pml_df_valid["ticker"].values,
        n_sims=1_000,
        samples=500,
        tune=500,
        chains=2,
        target_accept=0.90,
    )
    print("Monte Carlo simulation complete.")
    print(mc_idata)

    # Posterior diagnostics
    az.plot_posterior(mc_idata, var_names=["mu_return"], coords={"ticker": pml_df_valid["ticker"].values[:5]})
    plt.suptitle("Monte Carlo: mu_return Posterior (first 5 tickers)")
    plt.show()
else:
    print("No valid stocks to simulate.")
```

### 25.2 Kalman Filter Price Targets

Noise-reduced price target signals using state-space modeling.

```python
kf_model = KalmanFilterPriceTarget()
pt_col = "price_target"  # DDL: numeric

# Cross-sectional: use all tickers' consensus price targets as observed series
pt_values = pml_df[pt_col].dropna().values

if len(pt_values) >= 5:
    example_ticker = pml_df.loc[pml_df[pt_col].notna(), "ticker"].iloc[0]
    print(f"Running Kalman Filter for {example_ticker} ({len(pt_values)} observations)...")
    kf_idata = kf_model.fit(),

        # Plotting filtered state
    az.plot_posterior(kf_idata, var_names=["state"], coords={"time": [len(pt_values) - 1]})
    plt.title(f"Kalman Filtered Price Target Posterior (T={len(pt_values)}) for {example_ticker}")
    plt.show()
    else:
    print(f"Insufficient price target data ({len(pt_values)} obs) — skipping Kalman Filter.")

```

### 25.3 Accounting Anomaly Detection (Bayesian)

Multi-layered statistical anomaly detection via Mahalanobis distance.

```python
anomaly_bayesian = AccountingAnomalyBayesian(threshold=2.5)

# Multi-category anomaly features verified against DDL
anomaly_feature_cols = [
    "eps_surprise_pct", "eps_adjustment_ratio", "gaap_adj_eps_gap_pct",
    "ebitda_adjustment_ratio", "earnings_quality_score",
    "ni_adjustment_ratio", "cfo_to_net_income", "accounting_quality_score",
    "debt_to_equity", "current_ratio",
]
anomaly_features = [f for f in anomaly_feature_cols if f in pml_df.columns][:10]

if len(anomaly_features) >= 3:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    feat_vals = scaler.fit_transform(pml_df[anomaly_features].fillna(0).values)

    print(f"Running Bayesian Anomaly Detection using: {anomaly_features}")
    anomaly_idata = anomaly_bayesian.fit(feature_values=feat_vals, isins=pml_df["ticker"].values,
                                         feature_names=anomaly_features, samples=2000, tune=1000)

    # Visualization
    az.plot_forest(anomaly_idata, var_names=["anomaly_prob"], combined=True)
    plt.title("Bayesian Anomaly Probability Posterior")
    plt.show()

```

### 25.4 Credit Risk / Distress Estimation

Bayesian distress estimation with Altman Z-score and Debt-to-Equity signals.

```python
credit_bayesian = CreditRiskBayesian()
z_col = "altman_z_score"  # DDL: numeric, Quality & Risk category
de_col = "debt_to_equity"  # DDL: numeric, Leverage & Liquidity category

valid_mask = pml_df[z_col].notna() & pml_df[de_col].notna()
cr_df = pml_df[valid_mask]

if len(cr_df) > 0:
    print(f"Running Bayesian Credit Risk Model for {len(cr_df)} stocks...")
    # Pass combined_distress_score as observed distress signal when available
    distress_obs = (cr_df["combined_distress_score"].values / 100
                    if "combined_distress_score" in cr_df.columns else None)

    credit_idata = credit_bayesian.fit(
        z_scores=cr_df[z_col].values,
        debt_to_equity=cr_df[de_col].values,
        tickers=cr_df["ticker"].values,
        sectors=cr_df["sector"].values if "sector" in cr_df.columns else None,
        distress_observed=distress_obs,
        samples=2000,
        tune=1000,
    )

    az.plot_forest(credit_idata, var_names=["distress_prob"], combined=True)
    plt.title("Credit Distress Probability Posterior")
    plt.show()

```

### 25.5 Dividend Safety Analysis

Dividend cut probability with FCF coverage and payout ratios.

```python
div_bayesian = DividendSafetyBayesian()
payout_col = "dividend_payout_ratio"  # DDL: numeric, Dividends category
fcf_cov_col = "fcf_dividend_coverage"  # DDL: numeric, Dividends category

# Filter to dividend-paying stocks only
div_mask = (pml_df[payout_col].notna() & (pml_df[payout_col] > 0) &
            pml_df[fcf_cov_col].notna())
div_df = pml_df[div_mask]

if len(div_df) > 0:
    print(f"Running Bayesian Dividend Safety Model for {len(div_df)} dividend-paying stocks...")
    div_idata = div_bayesian.fit(
        payout_ratios=div_df[payout_col].values,
        fcf_coverage=div_df[fcf_cov_col].clip(lower=0.01).values,
        tickers=div_df["ticker"].values,
        samples=2000,
        tune=1000,
    )

    az.plot_forest(div_idata, var_names=["cut_prob"], combined=True)
    plt.title("Dividend Cut Probability Posterior")
    plt.show()

```

### 25.6 Earnings Beat Probability

Hierarchical Beta-Binomial model for earnings beat prediction.

```python
earnings_bayesian = EarningsBeatBayesian()
beats_col = "eps_positive_years"  # DDL: integer — years with positive EPS (proxy for beats)
total_periods = 10  # assume 10-year lookback window for eps_positive_years

if beats_col in pml_df.columns:
    valid_mask = pml_df[beats_col].notna()
    eb_df = pml_df[valid_mask]

    if len(eb_df) > 0:
        print(f"Running Bayesian Earnings Beat Model for {len(eb_df)} stocks...")
        n_total = np.full(len(eb_df), total_periods, dtype="int32")
        n_beats = np.clip(eb_df[beats_col].astype(int).values, 0, total_periods)

        earnings_idata = earnings_bayesian.fit(
            n_beats=n_beats,
            n_total=n_total,
            tickers=eb_df["ticker"].values,
            sectors=eb_df["sector"].values if "sector" in eb_df.columns else None,
            samples=2000,
            tune=1000,
        )

        fig = create_beat_probability_posterior(earnings_idata, title="Earnings Beat Probability Posterior")
        fig.show()

```

### 25.7 Price Target Achievement

Probability-weighted expected returns with risk adjustment.

```python
pt_bayesian = PriceTargetAchievement()
upside_col = "upside_potential"  # DDL: numeric, Analyst Sentiment category
disp_col = "price_target_spread_pct"  # DDL: numeric, Analyst Sentiment category

valid_mask = pml_df[upside_col].notna()
pt_df = pml_df[valid_mask]

if len(pt_df) > 0:
    print(f"Running Bayesian Price Target Achievement Model for {len(pt_df)} stocks...")
    disp_values = pt_df[disp_col].fillna(20).values / 100 if disp_col in pt_df.columns
        else np.full(len(pt_df), 0.2)

    pt_idata = pt_bayesian.fit(
        consensus_upside=pt_df[upside_col].values / 100,
        analyst_dispersion=np.clip(disp_values, 0.01, None),
        tickers=pt_df["ticker"].values,
        samples=2000,
        tune=1000,
    )

    az.plot_forest(pt_idata, var_names=["achieve_prob"], combined=True)
    plt.title("Price Target Achievement Probability Posterior")
    plt.show()

```

## 32. Feature Category Coverage Summary

```python
# Final summary of feature coverage
print("=" * 70)
print("  FEATURE CATEGORY COVERAGE SUMMARY")
print("=" * 70)

total_defined = 0
total_present = 0
for cat, cols in sorted(FEATURE_CATEGORIES.items()):
    present = [c for c in cols if c in df.columns]
    total_defined += len(cols)
    total_present += len(present)
    coverage = len(present) / len(cols) * 100 if cols else 0
    status = "✓" if coverage == 100 else "◐" if coverage >= 50 else "✗"
    print(f"  {status} {cat:30s}  {len(present):3d}/{len(cols):3d} features  ({coverage:5.1f}%)")

print(f"\n  {'TOTAL':30s}  {total_present:3d}/{total_defined:3d} features  "
      f"({total_present / total_defined * 100:.1f}%)")
print(f"\n  Dataset: {df.shape[0]} stocks × {df.shape[1]} columns")
print("=" * 70)
```

```python

```