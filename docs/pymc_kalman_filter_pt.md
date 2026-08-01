# **PyMC Kalman Filter Price Target Model — GaussianRandomWalk state-space**

## Uses PyMC GaussianRandomWalk for latent price state with an observation model for noisy price targets.

### Schema-aligned with:
- MV: `pml.mv_pymc_kalman_pt`
- Catalogue: `SELECT * FROM pml.vw_pymc_feature_catalogue WHERE model_target = 'kalman_pt' ORDER BY pymc_role, feature_role, feature_alias`





```sql
%%sql
SELECT * FROM pml.mv_pymc_kalman_pt mpkp WHERE observed_pt IS NOT NULL AND next_earnings >= '2026-01-01'
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
      <th>region</th>
      <th>country</th>
      <th>trading_country</th>
      <th>exchange</th>
      <th>unit</th>
      <th>style_class</th>
      <th>size_class</th>
      <th>sector</th>
      <th>...</th>
      <th>feat_pt_median_drift</th>
      <th>feat_coverage_drift</th>
      <th>feat_pt_noise_drift</th>
      <th>feat_pt_noise_sigma</th>
      <th>feat_pt_range_norm</th>
      <th>feat_vol_1m</th>
      <th>feat_vol_3m</th>
      <th>feat_vol_6m</th>
      <th>feat_vol_1y</th>
      <th>feat_total_return_ytd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US67066G1040</td>
      <td>NVDA</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>USD</td>
      <td>Growth</td>
      <td>Large Cap</td>
      <td>Information Technology</td>
      <td>...</td>
      <td>0.112724</td>
      <td>0.014851</td>
      <td>0.181373</td>
      <td>53.5310</td>
      <td>1.078130</td>
      <td>41.12</td>
      <td>36.91</td>
      <td>36.12</td>
      <td>34.06</td>
      <td>0.1948</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA44955L1067</td>
      <td>IAU</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>CA</td>
      <td>TSX</td>
      <td>CAD</td>
      <td>Core</td>
      <td>Small Cap</td>
      <td>Materials</td>
      <td>...</td>
      <td>0.209955</td>
      <td>0.006667</td>
      <td>0.170614</td>
      <td>1.3518</td>
      <td>1.020624</td>
      <td>53.82</td>
      <td>65.04</td>
      <td>64.36</td>
      <td>65.75</td>
      <td>0.1089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AU0000185993</td>
      <td>IREN</td>
      <td>Asia / Pacific</td>
      <td>AU</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>USD</td>
      <td>Core</td>
      <td>Mid Cap</td>
      <td>Information Technology</td>
      <td>...</td>
      <td>0.606542</td>
      <td>0.053800</td>
      <td>1.355982</td>
      <td>23.6776</td>
      <td>1.064572</td>
      <td>116.59</td>
      <td>101.37</td>
      <td>104.79</td>
      <td>101.73</td>
      <td>0.7633</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA96467A2002</td>
      <td>WCP</td>
      <td>United States and Canada</td>
      <td>CA</td>
      <td>CA</td>
      <td>TSX</td>
      <td>CAD</td>
      <td>Value</td>
      <td>Mid Cap</td>
      <td>Energy</td>
      <td>...</td>
      <td>0.081987</td>
      <td>0.033654</td>
      <td>0.254055</td>
      <td>1.8637</td>
      <td>0.463918</td>
      <td>30.34</td>
      <td>33.43</td>
      <td>30.75</td>
      <td>27.35</td>
      <td>0.4676</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US0378331005</td>
      <td>AAPL</td>
      <td>United States and Canada</td>
      <td>US</td>
      <td>US</td>
      <td>NasdaqGS</td>
      <td>USD</td>
      <td>Core</td>
      <td>Large Cap</td>
      <td>Information Technology</td>
      <td>...</td>
      <td>0.064431</td>
      <td>0.009640</td>
      <td>0.082376</td>
      <td>39.4462</td>
      <td>0.595800</td>
      <td>16.48</td>
      <td>20.91</td>
      <td>22.15</td>
      <td>22.13</td>
      <td>0.1616</td>
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
    </tr>
    <tr>
      <th>6272</th>
      <td>BRCVCBACNOR1</td>
      <td>CVCB3</td>
      <td>Latin America and Caribbean</td>
      <td>BR</td>
      <td>BR</td>
      <td>BOVESPA</td>
      <td>BRL</td>
      <td>Core</td>
      <td>Small Cap</td>
      <td>Consumer Discretionary</td>
      <td>...</td>
      <td>-0.016667</td>
      <td>-0.040000</td>
      <td>-0.125456</td>
      <td>0.2773</td>
      <td>0.220183</td>
      <td>63.53</td>
      <td>77.62</td>
      <td>76.73</td>
      <td>64.52</td>
      <td>-0.2870</td>
    </tr>
    <tr>
      <th>6273</th>
      <td>BRVVEOACNOR0</td>
      <td>VVEO3</td>
      <td>Latin America and Caribbean</td>
      <td>BR</td>
      <td>BR</td>
      <td>BOVESPA</td>
      <td>BRL</td>
      <td>Value</td>
      <td>Small Cap</td>
      <td>Health Care</td>
      <td>...</td>
      <td>-0.084829</td>
      <td>-0.075794</td>
      <td>-0.047866</td>
      <td>0.3956</td>
      <td>0.631579</td>
      <td>67.83</td>
      <td>84.94</td>
      <td>75.63</td>
      <td>76.34</td>
      <td>0.0423</td>
    </tr>
    <tr>
      <th>6274</th>
      <td>OM0000002168</td>
      <td>AACT</td>
      <td>Africa / Middle East</td>
      <td>OM</td>
      <td>OM</td>
      <td>MSM</td>
      <td>OMR</td>
      <td>Value</td>
      <td>Small Cap</td>
      <td>Industrials</td>
      <td>...</td>
      <td>0.010319</td>
      <td>0.033333</td>
      <td>0.402448</td>
      <td>0.0210</td>
      <td>0.194444</td>
      <td>31.72</td>
      <td>37.78</td>
      <td>36.69</td>
      <td>35.80</td>
      <td>-0.1088</td>
    </tr>
    <tr>
      <th>6275</th>
      <td>BRLJQQACNOR5</td>
      <td>LJQQ3</td>
      <td>Latin America and Caribbean</td>
      <td>BR</td>
      <td>BR</td>
      <td>BOVESPA</td>
      <td>BRL</td>
      <td>Value</td>
      <td>Small Cap</td>
      <td>Consumer Discretionary</td>
      <td>...</td>
      <td>-0.020000</td>
      <td>-0.080000</td>
      <td>-0.020612</td>
      <td>0.5436</td>
      <td>0.305087</td>
      <td>87.92</td>
      <td>75.92</td>
      <td>67.80</td>
      <td>60.84</td>
      <td>-0.3575</td>
    </tr>
    <tr>
      <th>6276</th>
      <td>BRENJUACNOR9</td>
      <td>ENJU3</td>
      <td>Latin America and Caribbean</td>
      <td>BR</td>
      <td>BR</td>
      <td>BOVESPA</td>
      <td>BRL</td>
      <td>Core</td>
      <td>Small Cap</td>
      <td>Communication Services</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.3500</td>
      <td>0.424242</td>
      <td>43.13</td>
      <td>51.52</td>
      <td>49.60</td>
      <td>44.93</td>
      <td>-0.0187</td>
    </tr>
  </tbody>
</table>
<p>6277 rows × 43 columns</p>
</div>



## 1. Notebook Setup & Imports


```sql
%%sql
SELECT *
FROM pml.vw_pymc_feature_catalogue
WHERE model_target = 'kalman_pt'
ORDER BY pymc_role, feature_role, feature_alias
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
      <th>model_target</th>
      <th>pymc_role</th>
      <th>column_name</th>
      <th>category</th>
      <th>feature_role</th>
      <th>feature_alias</th>
      <th>data_type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kalman_pt</td>
      <td>constant_data</td>
      <td>price_target_num</td>
      <td>analyst_targets</td>
      <td>count</td>
      <td>n_analysts</td>
      <td>double precision</td>
      <td>Number of analyst price targets</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kalman_pt</td>
      <td>coord</td>
      <td>country</td>
      <td>classification</td>
      <td>categorical</td>
      <td>country</td>
      <td>text</td>
      <td>Country of incorporation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kalman_pt</td>
      <td>coord</td>
      <td>exchange</td>
      <td>classification</td>
      <td>categorical</td>
      <td>exchange</td>
      <td>text</td>
      <td>Stock exchange</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kalman_pt</td>
      <td>coord</td>
      <td>industry</td>
      <td>classification</td>
      <td>categorical</td>
      <td>industry</td>
      <td>text</td>
      <td>GICS industry classification</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kalman_pt</td>
      <td>coord</td>
      <td>region</td>
      <td>classification</td>
      <td>categorical</td>
      <td>region</td>
      <td>text</td>
      <td>Geographic region</td>
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
    </tr>
    <tr>
      <th>88</th>
      <td>kalman_pt</td>
      <td>observed</td>
      <td>total_return_ytd</td>
      <td>total_return</td>
      <td>target</td>
      <td>feat_total_return_ytd</td>
      <td>double precision</td>
      <td>Total return year-to-date</td>
    </tr>
    <tr>
      <th>89</th>
      <td>kalman_pt</td>
      <td>observed</td>
      <td>price_target</td>
      <td>analyst_targets</td>
      <td>target</td>
      <td>observed_pt</td>
      <td>double precision</td>
      <td>Analyst consensus price target</td>
    </tr>
    <tr>
      <th>90</th>
      <td>kalman_pt</td>
      <td>observed</td>
      <td>price_target_high</td>
      <td>analyst_targets</td>
      <td>target</td>
      <td>price_target_high</td>
      <td>double precision</td>
      <td>High analyst price target</td>
    </tr>
    <tr>
      <th>91</th>
      <td>kalman_pt</td>
      <td>observed</td>
      <td>price_target_low</td>
      <td>analyst_targets</td>
      <td>target</td>
      <td>price_target_low</td>
      <td>double precision</td>
      <td>Low analyst price target</td>
    </tr>
    <tr>
      <th>92</th>
      <td>kalman_pt</td>
      <td>observed</td>
      <td>price_target_median</td>
      <td>analyst_targets</td>
      <td>target</td>
      <td>price_target_median</td>
      <td>double precision</td>
      <td>Median analyst price target</td>
    </tr>
  </tbody>
</table>
<p>93 rows × 8 columns</p>
</div>



## 1. Notebook Setup & Imports


```python
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# ArviZ 1.0 split-package imports: arviz-plots owns `style` + plotting,
# arviz-stats owns `summary` / `rhat` / `ess`. Address each submodule directly.
import arviz_plots as azp
import arviz_stats as azs
from arviz_plots import visuals as azv  # low-level primitives for custom composition

import pymc as pm
import pytensor.tensor as pt

warnings.filterwarnings('ignore', category=FutureWarning)
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# --- Plotting backend + theme ------------------------------------------------
# Pin the arviz-plots backend so PlotCollection / plot_* render through
# matplotlib (the notebook's dark-theme target). `arviz-vibrant` is the bright
# arviz 1.x palette that reads well on a dark background — the old 0.x
# `arviz-darkgrid` style does not exist in arviz-plots 1.x (its silent failure
# in the previous revision is why the arviz plots were rendering un-themed).
azp.backend = 'matplotlib'
plt.style.use('dark_background')
try:
    azp.style.use('arviz-vibrant')
except (OSError, ValueError, AttributeError):
    pass
sns.set_theme(style='darkgrid', context='notebook',
              rc={
                  'figure.facecolor': '#1e1e1e',
                  'axes.facecolor': '#2a2a2a',
                  'savefig.facecolor': '#1e1e1e',
                  'axes.edgecolor': '#cccccc',
                  'axes.labelcolor': '#e6e6e6',
                  'xtick.color': '#e6e6e6',
                  'ytick.color': '#e6e6e6',
                  'text.color': '#e6e6e6',
                  'grid.color': '#555555',
              })

# arviz_plots builds its per-chain colour aesthetic by reshaping the *active*
# matplotlib colour cycle. seaborn.set_theme() installs that cycle as RGB
# tuples (e.g. (0.29, 0.44, 0.69)); arviz_plots then does
# `np.array(colours[:n_chains]).reshape((n_chains,))`, which for 4 chains turns
# 4 RGB triples into an array of size 12 and raises
# 'cannot reshape array of size 12 into shape (4,)'. Re-express the cycle as
# hex strings so plot_trace / PlotCollection work under the seaborn theme.
from cycler import cycler as _cycler
import matplotlib.colors as _mcolors
_cycle_cols = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
if _cycle_cols and not all(isinstance(_c, str) for _c in _cycle_cols):
    plt.rcParams['axes.prop_cycle'] = _cycler(
        color=[_mcolors.to_hex(_c) for _c in _cycle_cols]
    )
plt.rcParams['figure.dpi'] = 110

print(f'kalman_df         : {kalman_df.shape}')
print(f'feature_catalogue : {feature_catalogue.shape}')
if 'model_target' in feature_catalogue.columns:
    _mt = feature_catalogue['model_target']
    if not (_mt == 'kalman_pt').all():
        warnings.warn("feature_catalogue is not fully filtered to model_target='kalman_pt'.")
```

    kalman_df         : (6277, 43)
    feature_catalogue : (93, 8)
    

    C:\Users\markm\PML_Finance_Project\.venv\Lib\site-packages\pytensor\configparser.py:309: UserWarning: PyTensor does not recognise this flag: device
      warnings.warn(f"PyTensor does not recognise this flag: {key}")
    

### 1.1 Custom visualization — Kalman price-target path (`arviz_plots` composition)

The headline view for this model is the **Kalman-smoothed price target over time**:
the latent state is a noisy walk through the observed analyst targets, and the value
of a state-space filter is that it returns a *credible band*, not just a point.

`plot_price_target_path()` composes that view with the `arviz_plots` low-level API
(`PlotCollection.grid` + `arviz_plots.visuals`) rather than a one-shot `plot_*`
helper — see the [compose-your-own-plot tutorial](https://python.arviz.org/projects/plots/en/latest/tutorials/compose_own_plot.html).
It layers, on a single time axis:

- **nested HDI bands** (94% + 50%) of the posterior latent `state`, darkening inward;
- the **posterior-median** smoothed path;
- the **observed** analyst price targets as points;
- a dashed **last-price** reference line.

It reads the `time` coordinate straight off the posterior (a `DatetimeIndex` when the
model was fit with `dates=`), so it works for both the single-ISIN time-series filter
(Section 11) and any future per-ISIN posterior carrying a `state`/`time` pair.


```python
from typing import Optional, Sequence


def plot_price_target_path(
    idata,
    *,
    state_var: str = "state",
    observed: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    last_price: Optional[float] = None,
    ticker: Optional[str] = None,
    hdi_probs: Sequence[float] = (0.94, 0.5),
    figsize: tuple[float, float] = (11, 5),
    color: str = "#56b4e9",
    observed_color: str = "#ffb000",
):
    """Compose a Kalman-smoothed price-target trajectory with ``arviz_plots``.

    Builds the plot from the low-level composition API
    (:meth:`arviz_plots.PlotCollection.grid` + :mod:`arviz_plots.visuals`) so the
    posterior latent state, its credible bands, and the raw observations share a
    single time axis.

    Parameters
    ----------
    idata
        Inference object whose ``posterior`` holds ``state_var`` over a ``time``
        dim (e.g. the output of :meth:`KalmanFilterPriceTarget.fit`).
    state_var
        Posterior variable carrying the price-space latent state. Default ``"state"``.
    observed
        Observed analyst price targets aligned to the ``time`` axis. Plotted as
        points when supplied.
    dates
        Time index aligned to the ``time`` dim. When omitted, the ``time`` coord on
        the posterior is used (and treated as datetime if it parses as such).
    last_price
        Reference last price; drawn as a dashed horizontal line when finite.
    ticker
        Optional label for the title.
    hdi_probs
        Credible-interval masses to shade, widest first.
    figsize, color, observed_color
        Cosmetic controls.

    Returns
    -------
    arviz_plots.PlotCollection
        The composed collection (already drawn); call ``.show()`` to display.
    """
    post = idata.posterior[state_var]
    if "time" not in post.dims:
        raise ValueError(f"{state_var!r} has no 'time' dim; dims={post.dims}.")
    n_time = post.sizes["time"]

    # Resolve the x-axis: prefer explicit `dates`, else the posterior time coord.
    if dates is None and "time" in post.coords:
        coord_vals = pd.to_datetime(post["time"].values, errors="coerce")
        if not pd.isna(coord_vals).all():
            dates = pd.DatetimeIndex(coord_vals)
    use_dates = dates is not None and len(dates) == n_time and not pd.isna(dates).all()
    x = xr.DataArray(
        mdates.date2num(np.asarray(dates)) if use_dates else np.arange(n_time),
        dims="time",
    )

    median = post.median(("chain", "draw"))
    ds = post.to_dataset()

    pc = azp.PlotCollection.grid(
        ds, backend="matplotlib", figure_kwargs={"figsize": figsize}
    )
    target = pc.get_target(state_var, {})  # raw matplotlib Axes

    # Nested HDI bands: widest first with the lightest alpha so inner masses darken.
    band_alphas = (0.16, 0.28, 0.40, 0.50)
    for prob, alpha in zip(sorted(hdi_probs, reverse=True), band_alphas):
        band = post.azstats.hdi(prob=prob)
        azv.fill_between_y(
            median, target, x=x,
            y_bottom=band.sel(ci_bound="lower"),
            y_top=band.sel(ci_bound="upper"),
            facecolor=color, alpha=alpha, edgecolor="none",
        )

    azv.line_xy(median, target, x=x, y=median, color=color, linewidth=2.2, zorder=4)

    if observed is not None:
        obs = xr.DataArray(np.asarray(observed, dtype="float64"), dims="time")
        azv.scatter_xy(
            median, target, x=x, y=obs,
            color=observed_color, s=34, zorder=6,
            edgecolor="#1e1e1e", linewidth=0.6,
        )

    if last_price is not None and np.isfinite(last_price):
        target.axhline(float(last_price), ls="--", color="#bbbbbb", lw=1.2, zorder=2)

    if use_dates:
        target.xaxis_date()
        target.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    azv.labelled_x(median, target, text="as-of date" if use_dates else "time step")
    azv.labelled_y(median, target, text="price target")
    title = "Kalman-smoothed price-target path"
    if ticker:
        title += f" — {ticker}"
    target.set_title(title)

    # Hand-built legend (composition primitives don't auto-register labels).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([0], [0], color=color, lw=2.2, label="posterior median state"),
        Patch(facecolor=color, alpha=0.40,
              label=f"{int(max(hdi_probs) * 100)}% / {int(min(hdi_probs) * 100)}% HDI"),
    ]
    if observed is not None:
        handles.append(Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor=observed_color, markeredgecolor="#1e1e1e",
                              label="observed price target"))
    if last_price is not None and np.isfinite(last_price):
        handles.append(Line2D([0], [0], ls="--", color="#bbbbbb", label="last price"))
    target.legend(handles=handles, fontsize=8, loc="best", framealpha=0.25)

    return pc


print("Defined plot_price_target_path() — Kalman price-target path via arviz_plots composition.")
```

    Defined plot_price_target_path() — Kalman price-target path via arviz_plots composition.
    

## 2. Exploratory Data Analysis (EDA) — `kalman_df`

Source: `pml.mv_pymc_kalman_pt` (one row per ISIN, filtered to `observed_pt IS NOT NULL`).
We resolve column roles from `pml.vw_pymc_feature_catalogue` and fall back to the
known MV schema where the catalogue has no `kalman_pt` rows yet.


```python
# Map feature_catalogue -> columns actually present in kalman_df
catalogue = feature_catalogue.copy()
catalogue['present'] = catalogue['feature_alias'].isin(kalman_df.columns)

role_summary = (
    catalogue.groupby(['pymc_role', 'feature_role'])['present']
    .agg(n_columns='size', n_present='sum')
    .reset_index()
)
role_summary
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
      <th>pymc_role</th>
      <th>feature_role</th>
      <th>n_columns</th>
      <th>n_present</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>constant_data</td>
      <td>count</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>coord</td>
      <td>categorical</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>coord</td>
      <td>date</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>derived_input</td>
      <td>historical</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mutable_predictor</td>
      <td>predictor</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>observed</td>
      <td>target</td>
      <td>6</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Resolve column groups by pymc_role, with canonical fallbacks for the Kalman MV.
present = catalogue.loc[catalogue['present']]
PREDICTOR_COLS = present.loc[present['pymc_role'] == 'mutable_predictor', 'feature_alias'].tolist()
COORD_COLS = present.loc[present['pymc_role'] == 'coord', 'feature_alias'].tolist()
RESPONSE_COLS = present.loc[present['pymc_role'].isin(['response', 'observed']), 'feature_alias'].tolist()

# Canonical schema of pml.mv_pymc_kalman_pt (single source of truth in SQL).
# The extended MV emits a per-trail drift feature for every price_* / price_target_*
# family (mean / high / low / median / raw price / analyst-count / dispersion).
KNOWN_FEATURES = ['feat_pt_drift', 'feat_price_drift',
                  'feat_pt_high_drift', 'feat_pt_low_drift', 'feat_pt_median_drift',
                  'feat_coverage_drift', 'feat_pt_noise_drift',
                  'feat_pt_noise_sigma', 'feat_pt_range_norm',
                  'feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y',
                  'feat_total_return_ytd']
for col in KNOWN_FEATURES:
    if col in kalman_df.columns and col not in PREDICTOR_COLS:
        PREDICTOR_COLS.append(col)
if 'observed_pt' in kalman_df.columns and 'observed_pt' not in RESPONSE_COLS:
    RESPONSE_COLS.append('observed_pt')

# Hierarchical classification coords (categorical group effects) are distinct
# from the fiscal-calendar DATE anchors. Both carry pymc_role='coord', but the
# date anchors define the single-security *time axis* (used in section 11) and
# must NOT be treated as categorical effects in the cross-sectional model.
CLASSIFICATION_COORDS = [c for c in (
    'isin', 'ticker', 'region', 'country', 'trading_country',
    'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry'
) if c in kalman_df.columns]
FISCAL_CALENDAR_COLS = [c for c in (
    'income_statement_report_date', 'next_earnings', 'fy_end_date',
    'next_income_statement_report_date', 'next_fy_end_date', 'expected_report_date'
) if c in kalman_df.columns]
DAY_COUNT_COLS = [c for c in (
    'days_to_next_earnings', 'days_since_last_report', 'days_to_next_fy_end',
    'days_to_next_report', 'days_to_expected_report', 'days_to_fy_end'
) if c in kalman_df.columns]
# Keep only non-date coords as categorical-effect candidates; fall back to the
# curated classification list when the catalogue exposes no plain coords.
COORD_COLS = [c for c in COORD_COLS if c not in FISCAL_CALENDAR_COLS] or CLASSIFICATION_COORDS.copy()

print(f'#predictors     : {len(PREDICTOR_COLS)} -> {PREDICTOR_COLS}')
print(f'#response       : {len(RESPONSE_COLS)} -> {RESPONSE_COLS}')
print(f'#coords         : {len(COORD_COLS)} -> {COORD_COLS}')
print(f'#classification : {len(CLASSIFICATION_COORDS)} -> {CLASSIFICATION_COORDS}')
print(f'#fiscal-calendar: {len(FISCAL_CALENDAR_COLS)} -> {FISCAL_CALENDAR_COLS}')
print(f'#day-count      : {len(DAY_COUNT_COLS)} -> {DAY_COUNT_COLS}')
```

    #predictors     : 21 -> ['days_since_last_report', 'days_to_expected_report', 'days_to_fy_end', 'days_to_next_earnings', 'days_to_next_fy_end', 'days_to_next_report', 'feat_vol_1m', 'feat_vol_1y', 'feat_vol_3m', 'feat_vol_6m', 'last_price', 'feat_pt_drift', 'feat_price_drift', 'feat_pt_high_drift', 'feat_pt_low_drift', 'feat_pt_median_drift', 'feat_coverage_drift', 'feat_pt_noise_drift', 'feat_pt_noise_sigma', 'feat_pt_range_norm', 'feat_total_return_ytd']
    #response       : 6 -> ['feat_pt_noise_sigma', 'feat_total_return_ytd', 'observed_pt', 'price_target_high', 'price_target_low', 'price_target_median']
    #coords         : 9 -> ['country', 'exchange', 'industry', 'region', 'sector', 'size_class', 'style_class', 'trading_country', 'unit']
    #classification : 11 -> ['isin', 'ticker', 'region', 'country', 'trading_country', 'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry']
    #fiscal-calendar: 6 -> ['income_statement_report_date', 'next_earnings', 'fy_end_date', 'next_income_statement_report_date', 'next_fy_end_date', 'expected_report_date']
    #day-count      : 6 -> ['days_to_next_earnings', 'days_since_last_report', 'days_to_next_fy_end', 'days_to_next_report', 'days_to_expected_report', 'days_to_fy_end']
    


```python
# 2.1 Shape, dtype, missingness overview.
eda_overview = pd.DataFrame({
    'dtype': kalman_df.dtypes.astype(str),
    'n_missing': kalman_df.isna().sum(),
    'pct_missing': (kalman_df.isna().mean() * 100).round(1),
    'n_unique': kalman_df.nunique(),
})
print(f'kalman_df shape: {kalman_df.shape}')
eda_overview.sort_values('pct_missing', ascending=False).head(30)
```

    kalman_df shape: (6277, 43)
    




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
      <th>dtype</th>
      <th>n_missing</th>
      <th>pct_missing</th>
      <th>n_unique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>days_to_fy_end</th>
      <td>float64</td>
      <td>48</td>
      <td>0.8</td>
      <td>20</td>
    </tr>
    <tr>
      <th>next_fy_end_date</th>
      <td>str</td>
      <td>48</td>
      <td>0.8</td>
      <td>20</td>
    </tr>
    <tr>
      <th>fy_end_date</th>
      <td>str</td>
      <td>48</td>
      <td>0.8</td>
      <td>20</td>
    </tr>
    <tr>
      <th>days_to_next_fy_end</th>
      <td>float64</td>
      <td>48</td>
      <td>0.8</td>
      <td>20</td>
    </tr>
    <tr>
      <th>feat_total_return_ytd</th>
      <td>float64</td>
      <td>46</td>
      <td>0.7</td>
      <td>4741</td>
    </tr>
    <tr>
      <th>feat_pt_noise_drift</th>
      <td>float64</td>
      <td>17</td>
      <td>0.3</td>
      <td>6090</td>
    </tr>
    <tr>
      <th>days_since_last_report</th>
      <td>float64</td>
      <td>11</td>
      <td>0.2</td>
      <td>67</td>
    </tr>
    <tr>
      <th>expected_report_date</th>
      <td>str</td>
      <td>11</td>
      <td>0.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>next_income_statement_report_date</th>
      <td>str</td>
      <td>11</td>
      <td>0.2</td>
      <td>85</td>
    </tr>
    <tr>
      <th>days_to_expected_report</th>
      <td>float64</td>
      <td>11</td>
      <td>0.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>days_to_next_report</th>
      <td>float64</td>
      <td>11</td>
      <td>0.2</td>
      <td>85</td>
    </tr>
    <tr>
      <th>income_statement_report_date</th>
      <td>str</td>
      <td>11</td>
      <td>0.2</td>
      <td>67</td>
    </tr>
    <tr>
      <th>feat_vol_1m</th>
      <td>float64</td>
      <td>9</td>
      <td>0.1</td>
      <td>4191</td>
    </tr>
    <tr>
      <th>size_class</th>
      <td>str</td>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>sector</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>industry</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>59</td>
    </tr>
    <tr>
      <th>exchange</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>82</td>
    </tr>
    <tr>
      <th>trading_country</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>65</td>
    </tr>
    <tr>
      <th>country</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>83</td>
    </tr>
    <tr>
      <th>region</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ticker</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>6024</td>
    </tr>
    <tr>
      <th>isin</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>6277</td>
    </tr>
    <tr>
      <th>style_class</th>
      <td>str</td>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>unit</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>observed_pt</th>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5560</td>
    </tr>
    <tr>
      <th>days_to_next_earnings</th>
      <td>int64</td>
      <td>0</td>
      <td>0.0</td>
      <td>114</td>
    </tr>
    <tr>
      <th>next_earnings</th>
      <td>str</td>
      <td>0</td>
      <td>0.0</td>
      <td>114</td>
    </tr>
    <tr>
      <th>price_target_high</th>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>2283</td>
    </tr>
    <tr>
      <th>price_target_median</th>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>2832</td>
    </tr>
    <tr>
      <th>last_price</th>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5128</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 2.2 Expected upside by industry - arviz_plots ridge.
# Replaces the previous 2x2 panel histogram. Distributional view of the raw
# implied upside ((observed_pt / last_price) - 1) per industry, drawn as a
# stacked ridge plot so cross-industry shape differences are immediately visible.
import arviz_plots as azp


_d = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
_d['upside_pct'] = (_d['observed_pt'] / _d['last_price'] - 1.0) * 100.0
_d['upside_pct'] = _d['upside_pct'].clip(-100, 200)
_d['industry'] = (_d['industry'] if 'industry' in _d.columns
                  else pd.Series('Unknown', index=_d.index))
_d['industry'] = _d['industry'].fillna('Unknown').astype(str)

# Keep industries with enough names to form a meaningful density.
_counts = _d['industry'].value_counts()
_keep = _counts[_counts >= 5].index.tolist()
_d = _d[_d['industry'].isin(_keep)]

# Pack as (industry, sample) into a Dataset so arviz_plots treats `industry`
# as the ridge facet dimension.
_industries = sorted(_d['industry'].unique())
_max_n = int(_d['industry'].value_counts().max())
_arr = np.full((len(_industries), _max_n), np.nan)
for i, ind in enumerate(_industries):
    vals = _d.loc[_d['industry'] == ind, 'upside_pct'].to_numpy()
    _arr[i, :len(vals)] = vals
_ds_ridge = xr.Dataset(
    {'implied_upside_pct': (('industry', 'sample'), _arr)},
    coords={'industry': _industries},
)

azp.plot_ridge(_ds_ridge, var_names=['implied_upside_pct'], sample_dims=['sample'], combined=True)
plt.suptitle('Implied upside (%) by industry - consensus observed_pt vs last_price',
             y=1.02)
plt.tight_layout()
plt.show()

_d['upside_pct'].describe()

```

    C:\Users\markm\AppData\Local\Temp\ipykernel_12952\2722643373.py:36: UserWarning: The figure layout has changed to tight
      plt.tight_layout()
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_12_1.png)
    





    count    6277.000000
    mean       25.693220
    std        35.243531
    min       -96.209386
    25%         5.816752
    50%        19.673913
    75%        37.521569
    max       200.000000
    Name: upside_pct, dtype: float64




```python
# 2.3 Classification-coord cardinality.
card = {c: kalman_df[c].nunique() for c in CLASSIFICATION_COORDS}
pd.Series(card).sort_values(ascending=False)
```




    isin               6277
    ticker             6024
    country              83
    exchange             82
    trading_country      65
    industry             59
    unit                 50
    sector                9
    region                5
    size_class            3
    style_class           3
    dtype: int64



## 3. State-Space Feature Mapping (Kalman semantics)

`KalmanFilterPriceTarget` (in `probabilistic_ml_model/pymc_models/KalmanFilterModel.py`)
is a **single-security time-series** `GaussianRandomWalk` filter: a latent log-price
state with process (`sigma_state`) and observation (`sigma_obs`) noise. `pml.mv_pymc_kalman_pt`
is a **cross-sectional** snapshot - the time axis has been collapsed into per-ISIN
`feat_*` drift / noise columns - so here we reuse the same generative ideas as a
one-step Kalman *measurement update* across the whole panel.

The extended MV computes `target_drift()` for **every** `price_* / price_target_*`
trail, so the drift signal is no longer limited to the consensus mean:

| Role                                                     | MV columns                                                                                                                                             | Where it enters the model                                                                 |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Drift / state-transition mean**                        | `feat_pt_drift`, `feat_price_drift`, `feat_pt_high_drift`, `feat_pt_low_drift`, `feat_pt_median_drift`, `feat_coverage_drift`, `feat_total_return_ytd` | regression slopes `beta` on the latent log-uplift                                         |
| **Observation-noise wideners** (non-negative)            | `feat_pt_range_norm`, `feat_pt_noise_sigma`, `feat_pt_noise_drift`, `feat_vol_{1m,3m,6m,1y}`                                                           | scale the measurement noise `sigma_obs`                                                   |
| **Fiscal-calendar time axis** (DATE coords / day-counts) | `income_statement_report_date`, `next_earnings`, `fy_end_date`, ..., `days_to_*`                                                                       | reconstruct the irregular elapsed-time spacing for the **marginalized** GRW in section 11 |

The fiscal-calendar columns are `pymc_role='coord'` but are **not** categorical
group effects - section 11 uses them to scale the random-walk process variance by
real elapsed time.


```python
# Map mv_pymc_kalman_pt feat_* columns onto Kalman state-space roles.
# Drift / state-transition mean now spans every price_* / price_target_* trail.
DRIFT_FEATURES = [c for c in ('feat_pt_drift', 'feat_price_drift',
                              'feat_pt_high_drift', 'feat_pt_low_drift',
                              'feat_pt_median_drift', 'feat_coverage_drift',
                              'feat_total_return_ytd')
                  if c in kalman_df.columns]
NOISE_RANGE_COL = 'feat_pt_range_norm' if 'feat_pt_range_norm' in kalman_df.columns else None
NOISE_SIGMA_COL = 'feat_pt_noise_sigma' if 'feat_pt_noise_sigma' in kalman_df.columns else None
VOL_COLS = [c for c in ('feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y')
            if c in kalman_df.columns]

mapping_rows = [(c, 'drift / state-transition mean (beta)') for c in DRIFT_FEATURES]
if NOISE_RANGE_COL:
    mapping_rows.append((NOISE_RANGE_COL, 'observation-noise widener (range)'))
if NOISE_SIGMA_COL:
    mapping_rows.append((NOISE_SIGMA_COL, 'observation-noise widener (consensus sigma)'))
mapping_rows += [(c, 'observation-noise widener (volatility)') for c in VOL_COLS]
mapping = pd.DataFrame(mapping_rows, columns=['mv_column', 'state_space_role'])

print(f'Drift features : {DRIFT_FEATURES}')
print(f'Noise drivers  : range={NOISE_RANGE_COL}, sigma={NOISE_SIGMA_COL}, vol={VOL_COLS}')
mapping
```

    Drift features : ['feat_pt_drift', 'feat_price_drift', 'feat_pt_high_drift', 'feat_pt_low_drift', 'feat_pt_median_drift', 'feat_coverage_drift', 'feat_total_return_ytd']
    Noise drivers  : range=feat_pt_range_norm, sigma=feat_pt_noise_sigma, vol=['feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y']
    




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
      <th>mv_column</th>
      <th>state_space_role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>feat_pt_drift</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>feat_price_drift</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>feat_pt_high_drift</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>feat_pt_low_drift</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>feat_pt_median_drift</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>feat_coverage_drift</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>feat_total_return_ytd</td>
      <td>drift / state-transition mean (beta)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>feat_pt_range_norm</td>
      <td>observation-noise widener (range)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>feat_pt_noise_sigma</td>
      <td>observation-noise widener (consensus sigma)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>feat_vol_1m</td>
      <td>observation-noise widener (volatility)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>feat_vol_3m</td>
      <td>observation-noise widener (volatility)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>feat_vol_6m</td>
      <td>observation-noise widener (volatility)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>feat_vol_1y</td>
      <td>observation-noise widener (volatility)</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Build PyMC-Aligned Data Containers


```python
# 4.0 Filter to rows usable for a log-space state-space model: strictly positive
# observed_pt and last_price, and >=1 contributing analyst. Build categorical coords.
model_df = kalman_df.loc[
    (kalman_df['observed_pt'] > 0)
    & (kalman_df['last_price'] > 0)
    & kalman_df['observed_pt'].notna()
    & kalman_df['last_price'].notna()
].copy().reset_index(drop=True)

if 'n_analysts' in model_df.columns:
    model_df['n_analysts'] = model_df['n_analysts'].fillna(1).clip(lower=1)
else:
    model_df['n_analysts'] = 1.0
print(f'Modelling rows (observed_pt>0 & last_price>0): {len(model_df)}')

isin_labels = model_df['isin'].astype(str).values

# Categorical group effects use ONLY the hierarchical classification coords -
# the fiscal-calendar date coords (FISCAL_CALENDAR_COLS) define the section 11
# time axis and must never be expanded into thousands of categorical levels here.
CATEGORICAL_COORDS = [c for c in CLASSIFICATION_COORDS
                      if c in model_df.columns and c not in ('isin', 'ticker')]
coord_uniques, coord_idx = {}, {}
for col in CATEGORICAL_COORDS:
    labels = model_df[col].fillna('Unknown').astype(str).values
    uniques, idx = np.unique(labels, return_inverse=True)
    coord_uniques[col] = uniques
    coord_idx[col] = idx.astype('int64')
print(f'Categorical coords ({len(CATEGORICAL_COORDS)}): {CATEGORICAL_COORDS}')

# Log-space transition anchor + observation. KalmanFilterModel operates in log
# space to keep strictly-positive prices / targets numerically stable.
log_last = np.log(model_df['last_price'].astype('float64').to_numpy())
log_obs = np.log(model_df['observed_pt'].astype('float64').to_numpy())
print(f'log_last: {log_last.shape}, log_obs: {log_obs.shape}')
```

    Modelling rows (observed_pt>0 & last_price>0): 6277
    Categorical coords (9): ['region', 'country', 'trading_country', 'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry']
    log_last: (6277,), log_obs: (6277,)
    


```python
# 4.1 Standardised drift-feature matrix (state-transition mean inputs).
X_drift_raw = model_df[DRIFT_FEATURES].astype(float)
X_drift_std = (X_drift_raw - X_drift_raw.mean()) / X_drift_raw.std(ddof=0).replace(0, 1.0)
X_drift = X_drift_std.fillna(0.0).to_numpy()
print(f'Drift matrix X_drift: {X_drift.shape}  ({DRIFT_FEATURES})')

# 4.2 Non-negative observation-noise drivers (measurement-variance wideners).
n_obs = len(model_df)

def _nonneg(col):
    if col and col in model_df.columns:
        return model_df[col].astype('float64').fillna(0.0).clip(lower=0.0).to_numpy()
    return np.zeros(n_obs)

range_norm_xs = _nonneg(NOISE_RANGE_COL)

# Consensus stddev -> relative to last_price (NOT observed_pt, to avoid leakage
# into its own measurement-noise term).
if NOISE_SIGMA_COL and NOISE_SIGMA_COL in model_df.columns:
    _sig = model_df[NOISE_SIGMA_COL].astype('float64').fillna(0.0).to_numpy()
    _lp = np.maximum(model_df['last_price'].astype('float64').to_numpy(), 1e-9)
    noise_cv_xs = np.clip(_sig / _lp, 0.0, None)
else:
    noise_cv_xs = np.zeros(n_obs)

vol_xs = (model_df[VOL_COLS].astype('float64').fillna(0.0).clip(lower=0.0).mean(axis=1).to_numpy()
          if VOL_COLS else np.zeros(n_obs))

n_analysts_xs = model_df['n_analysts'].astype('float64').clip(lower=1).to_numpy()
sqrt_n_xs = np.sqrt(n_analysts_xs)
print(f'noise drivers (mean) — range:{range_norm_xs.mean():.3f}  '
      f'cv:{noise_cv_xs.mean():.3f}  vol:{vol_xs.mean():.3f}')
```

    Drift matrix X_drift: (6277, 7)  (['feat_pt_drift', 'feat_price_drift', 'feat_pt_high_drift', 'feat_pt_low_drift', 'feat_pt_median_drift', 'feat_coverage_drift', 'feat_total_return_ytd'])
    noise drivers (mean) — range:0.448  cv:0.189  vol:45.866
    

## 5. Cross-Sectional State-Space Model (log-space Kalman update)

> **Superseded (0.9.9.10):** this single-observation cross-sectional builder
> (`build_kalman_pt_model` / `build_model_data` / `ModelData`) was removed from
> `pymc_kalman_filter_pt.py`. The current path is the §5b fused MvGRW panel
> (`prepare_kalman_panel_inputs` + `build_fused_kalman_pt_model`). This section
> is retained as a historical record of the earlier notebook run.

Generative form, per ISIN $i$:

$$\eta_i = \mu_0 + X^{\text{drift}}_i\,\beta + \sum_g u^{(g)}_{[i]} \qquad\text{(hierarchical drift mean)}$$
$$\text{uplift}_i = \eta_i + \sigma_{\text{state}}\,z_i \qquad\text{(latent state innovation)}$$
$$\log s_i = \log(\text{last\_price}_i) + \text{uplift}_i \qquad\text{(latent log fair price-target)}$$
$$\log(\text{observed\_pt}_i) \sim \text{StudentT}\!\left(\nu,\ \log s_i,\ \sigma^{\text{obs}}_i\right)$$
$$\sigma^{\text{obs}}_i = \sigma_{\text{base}}\,\frac{1 + \text{range}_i + \text{cv}_i + \tfrac12\text{vol}_i}{\sqrt{n^{\text{analysts}}_i}}$$

The posterior `expected_pt` $= e^{\log s_i}$ is the **Kalman-smoothed** price target — a
shrinkage between the drift-implied prior ($\text{last\_price}\cdot e^{\eta}$) and the
noisy consensus `observed_pt`, with more analysts / tighter dispersion pulling it
toward the raw consensus. `HalfNormal` state / observation noise and the non-centred
innovation mirror `KalmanFilterPriceTarget`.


```python
# 5.0 Cross-sectional log-space state-space model builder.

_CANDIDATE_GROUPS = ('region', 'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry')
GROUP_EFFECTS = [c for c in _CANDIDATE_GROUPS if c in coord_idx]


def build_kalman_pt_model(*, robust: bool = True) -> pm.Model:
    """Cross-sectional log-space state-space price-target model.

    One observation per ISIN: ``log(observed_pt)`` is a noisy measurement of a
    latent log fair-value state anchored at ``log(last_price)`` and shifted by a
    hierarchical, drift-feature-driven log-uplift.

    Parameters
    ----------
    robust : bool
        ``True`` -> Student-t measurement likelihood (default; absorbs analyst
        outliers). ``False`` -> Normal-likelihood twin.
    """
    coords = {'isin': isin_labels, 'drift_feature': DRIFT_FEATURES}
    for col in GROUP_EFFECTS:
        coords[col] = coord_uniques[col]

    with pm.Model(coords=coords) as model:
        log_last_d = pm.Data('log_last_price', log_last, dims='isin')
        Xd = pm.Data('drift_features', X_drift, dims=('isin', 'drift_feature'))
        rng_d = pm.Data('feat_pt_range_norm', range_norm_xs, dims='isin')
        cv_d = pm.Data('feat_pt_noise_cv', noise_cv_xs, dims='isin')
        vol_d = pm.Data('feat_vol_mean', vol_xs, dims='isin')
        sqn = pm.Data('sqrt_n_analysts', sqrt_n_xs, dims='isin')
        log_obs_d = pm.Data('log_observed_pt', log_obs, dims='isin')
        idx_data = {col: pm.Data(f'{col}_idx', coord_idx[col], dims='isin')
                    for col in GROUP_EFFECTS}

        # --- State-transition mean: hierarchical drift regression on log-uplift.
        mu_global = pm.Normal('mu_global', 0.0, 0.25)
        beta = pm.Normal('beta', 0.0, 0.25, dims='drift_feature')
        eta = mu_global + pt.dot(Xd, beta)
        for col in GROUP_EFFECTS:
            sigma_g = pm.HalfNormal(f'sigma_{col}', 0.10)
            z_g = pm.Normal(f'z_{col}', 0.0, 1.0, dims=col)
            ge = pm.Deterministic(f'{col}_effect', sigma_g * z_g, dims=col)
            eta = eta + ge[idx_data[col]]

        # --- Latent log state (non-centred GRW-style innovation; HalfNormal sigma).
        sigma_state = pm.HalfNormal('sigma_state', 0.15)
        z_state = pm.Normal('z_state', 0.0, 1.0, dims='isin')
        log_uplift = pm.Deterministic('log_uplift', eta + sigma_state * z_state, dims='isin')
        log_state = pm.Deterministic('log_state', log_last_d + log_uplift, dims='isin')

        # --- Observation noise (Kalman measurement variance), HalfNormal base.
        sigma_obs_base = pm.HalfNormal('sigma_obs_base', 0.10)
        sigma_obs = pm.Deterministic(
            'sigma_obs',
            sigma_obs_base * (1.0 + rng_d + cv_d + 0.5 * vol_d) / sqn,
            dims='isin')

        # --- Measurement likelihood in log space.
        if robust:
            nu = pm.Gamma('nu', alpha=2.0, beta=0.1)
            pm.StudentT('log_pt_obs', nu=nu, mu=log_state, sigma=sigma_obs,
                        observed=log_obs_d, dims='isin')
        else:
            pm.Normal('log_pt_obs', mu=log_state, sigma=sigma_obs,
                      observed=log_obs_d, dims='isin')

        # --- Screening outputs on the price scale.
        pm.Deterministic('expected_pt', pt.exp(log_state), dims='isin')
        pm.Deterministic('expected_upside', pt.exp(log_uplift) - 1.0, dims='isin')
    return model


kalman_pt_model = build_kalman_pt_model(robust=True)
print(f'sec.5 - cross-sectional state-space model on {len(isin_labels)} ISINs; '
      f'{len(DRIFT_FEATURES)} drift features; group effects: {GROUP_EFFECTS}')
pm.model_to_graphviz(kalman_pt_model)
```

    sec.5 - cross-sectional state-space model on 6277 ISINs; 7 drift features; group effects: ['region', 'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry']
    




    
![svg](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_20_1.svg)
    



## 6. Prior Predictive Checks


```python
_prior_var_names = [
    'mu_global', 'beta', 'sigma_state', 'sigma_obs_base', 'nu',
    *[f'sigma_{g}' for g in GROUP_EFFECTS],
    *[f'{g}_effect' for g in GROUP_EFFECTS],
    'log_uplift', 'log_state', 'sigma_obs',
    'expected_pt', 'expected_upside', 'log_pt_obs',
]
with kalman_pt_model:
    prior_idata = pm.sample_prior_predictive(
        draws=1500, var_names=_prior_var_names,
        random_seed=RANDOM_SEED, return_inferencedata=True,
    )

# Prior implied upside vs the empirical distribution (sanity scale check).
prior_up = prior_idata.prior['expected_upside'].values.reshape(-1)
emp_up = (model_df['observed_pt'] / model_df['last_price'] - 1.0).to_numpy()
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(np.clip(prior_up, -1, 2), bins=80, density=True, alpha=0.6,
        label='prior expected_upside')
ax.hist(np.clip(emp_up, -1, 2), bins=80, density=True, histtype='step',
        linewidth=1.5, label='empirical observed_pt/last_price - 1')
ax.set_title('Prior predictive implied upside vs empirical')
ax.legend()
plt.tight_layout()
plt.show()
prior_idata
```

    Sampling: [beta, log_pt_obs, mu_global, nu, sigma_exchange, sigma_industry, sigma_obs_base, sigma_region, sigma_sector, sigma_size_class, sigma_state, sigma_style_class, sigma_unit, z_exchange, z_industry, z_region, z_sector, z_size_class, z_state, z_style_class, z_unit]
    C:\Users\markm\AppData\Local\Temp\ipykernel_12952\1238257479.py:24: UserWarning: The figure layout has changed to tight
      plt.tight_layout()
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_22_1.png)
    





<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in notebooks */

:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
  line-height: 1.6;
  padding-bottom: 4px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
}

.xr-header {
  border-bottom: solid 1px var(--xr-border-color);
  margin-bottom: 4px;
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-obj-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type,
.xr-group-box-contents > label {
  color: var(--xr-font-color2);
  display: block;
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
  margin-block-start: 0;
  margin-block-end: 0;
}

.xr-section-item {
  display: contents;
}

.xr-section-item > input,
.xr-group-box-contents > input,
.xr-array-wrap > input {
  display: block;
  opacity: 0;
  height: 0;
  margin: 0;
}

.xr-section-item > input + label,
.xr-var-item > input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item > input:enabled + label,
.xr-var-item > input:enabled + label,
.xr-array-wrap > input:enabled + label,
.xr-group-box-contents > input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item > input:focus-visible + label,
.xr-var-item > input:focus-visible + label,
.xr-array-wrap > input:focus-visible + label,
.xr-group-box-contents > input:focus-visible + label {
  outline: auto;
}

.xr-section-item > input:enabled + label:hover,
.xr-var-item > input:enabled + label:hover,
.xr-array-wrap > input:enabled + label:hover,
.xr-group-box-contents > input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
  white-space: nowrap;
}

.xr-section-summary > em {
  font-weight: normal;
}

.xr-span-grid {
  grid-column-end: -1;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.3em;
}

.xr-group-box-contents > input:checked + label > span {
  display: inline-block;
  padding-left: 0.6em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details,
.xr-group-box-contents > label {
  padding-top: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  grid-column: 1 / -1;
  margin-top: 4px;
  margin-bottom: 5px;
}

.xr-section-summary-in ~ .xr-section-details {
  display: none;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-children {
  display: inline-grid;
  grid-template-columns: 100%;
  grid-column: 1 / -1;
  padding-top: 4px;
}

.xr-group-box {
  display: inline-grid;
  grid-template-columns: 0px 30px auto;
}

.xr-group-box-vline {
  grid-column-start: 1;
  border-right: 0.2em solid;
  border-color: var(--xr-border-color);
  width: 0px;
}

.xr-group-box-hline {
  grid-column-start: 2;
  grid-row-start: 1;
  height: 1em;
  width: 26px;
  border-bottom: 0.2em solid;
  border-color: var(--xr-border-color);
}

.xr-group-box-contents {
  grid-column-start: 3;
  padding-bottom: 4px;
}

.xr-group-box-contents > label::before {
  content: "📂";
  padding-right: 0.3em;
}

.xr-group-box-contents > input:checked + label::before {
  content: "📁";
}

.xr-group-box-contents > input:checked + label {
  padding-bottom: 0px;
}

.xr-group-box-contents > input:checked ~ .xr-sections {
  display: none;
}

.xr-group-box-contents > input + label > span {
  display: none;
}

.xr-group-box-ellipsis {
  font-size: 1.4em;
  font-weight: 900;
  color: var(--xr-font-color2);
  letter-spacing: 0.15em;
  cursor: default;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}

.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}

.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataTree&gt;
Group: /
├── Group: /prior
│       Dimensions:             (chain: 1, draw: 1500, isin: 6277, sector: 9, unit: 50,
│                                style_class: 4, drift_feature: 7, exchange: 82,
│                                size_class: 4, industry: 59, region: 5)
│       Coordinates:
│         * chain               (chain) int64 8B 0
│         * draw                (draw) int64 12kB 0 1 2 3 4 ... 1495 1496 1497 1498 1499
│         * isin                (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│         * sector              (sector) &lt;U22 792B &#x27;Communication Services&#x27; ... &#x27;Util...
│         * unit                (unit) &lt;U3 600B &#x27;AED&#x27; &#x27;ARS&#x27; &#x27;AUD&#x27; ... &#x27;USD&#x27; &#x27;VND&#x27; &#x27;ZAR&#x27;
│         * style_class         (style_class) &lt;U7 112B &#x27;Core&#x27; &#x27;Growth&#x27; &#x27;Unknown&#x27; &#x27;Value&#x27;
│         * drift_feature       (drift_feature) &lt;U21 588B &#x27;feat_pt_drift&#x27; ... &#x27;feat_t...
│         * exchange            (exchange) &lt;U8 3kB &#x27;ADX&#x27; &#x27;AIM&#x27; &#x27;ASX&#x27; ... &#x27;XTRA&#x27; &#x27;ZGSE&#x27;
│         * size_class          (size_class) &lt;U9 144B &#x27;Large Cap&#x27; ... &#x27;Unknown&#x27;
│         * industry            (industry) &lt;U53 13kB &#x27;Aerospace and Defense&#x27; ... &#x27;Wir...
│         * region              (region) &lt;U27 540B &#x27;Africa / Middle East&#x27; ... &#x27;United...
│       Data variables: (12/24)
│           mu_global           (chain, draw) float64 12kB -0.4414 -0.3835 ... -0.1683
│           sigma_state         (chain, draw) float64 12kB 0.01626 0.06206 ... 0.01211
│           log_state           (chain, draw, isin) float64 75MB 4.827 ... -0.2693
│           sector_effect       (chain, draw, sector) float64 108kB 0.00126 ... 0.02049
│           sigma_region        (chain, draw) float64 12kB 0.1799 0.1776 ... 0.0553
│           sigma_unit          (chain, draw) float64 12kB 0.1254 0.06063 ... 0.2454
│           ...                  ...
│           log_uplift          (chain, draw, isin) float64 75MB -0.5798 ... -0.3181
│           nu                  (chain, draw) float64 12kB 36.22 20.02 ... 9.037 4.163
│           industry_effect     (chain, draw, industry) float64 708kB 0.1301 ... -0.0...
│           sigma_exchange      (chain, draw) float64 12kB 0.1919 0.02467 ... 0.1059
│           region_effect       (chain, draw, region) float64 60kB 0.06427 ... -0.03712
│           sigma_industry      (chain, draw) float64 12kB 0.1386 0.05796 ... 0.07764
│       Attributes:
│           created_at:                 2026-06-03T12:17:12.376301+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           inference_library:          pymc
│           inference_library_version:  6.0.1
│           sample_dims:                [&#x27;chain&#x27;, &#x27;draw&#x27;]
├── Group: /prior_predictive
│       Dimensions:     (chain: 1, draw: 1500, isin: 6277)
│       Coordinates:
│         * chain       (chain) int64 8B 0
│         * draw        (draw) int64 12kB 0 1 2 3 4 5 ... 1494 1495 1496 1497 1498 1499
│         * isin        (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│       Data variables:
│           log_pt_obs  (chain, draw, isin) float64 75MB 5.067 -0.5607 ... 0.2002 1.486
│       Attributes:
│           created_at:                 2026-06-03T12:17:12.382459+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           inference_library:          pymc
│           inference_library_version:  6.0.1
│           sample_dims:                [&#x27;chain&#x27;, &#x27;draw&#x27;]
├── Group: /observed_data
│       Dimensions:     (isin: 6277)
│       Coordinates:
│         * isin        (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│       Data variables:
│           log_pt_obs  (isin) float64 50kB 5.693 1.481 4.38 ... -1.532 1.369 0.5008
│       Attributes:
│           created_at:                 2026-06-03T12:17:12.384284+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           inference_library:          pymc
│           inference_library_version:  6.0.1
│           sample_dims:                []
└── Group: /constant_data
        Dimensions:             (isin: 6277, drift_feature: 7)
        Coordinates:
          * isin                (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
          * drift_feature       (drift_feature) &lt;U21 588B &#x27;feat_pt_drift&#x27; ... &#x27;feat_t...
        Data variables: (12/13)
            log_last_price      (isin) float64 50kB 5.406 0.8065 ... 0.3507 0.04879
            drift_features      (isin, drift_feature) float64 352kB 0.6513 ... -0.3173
            feat_pt_range_norm  (isin) float64 50kB 1.078 1.021 1.065 ... 0.3051 0.4242
            feat_pt_noise_cv    (isin) float64 50kB 0.2402 0.6035 ... 0.3828 0.3333
            feat_vol_mean       (isin) float64 50kB 37.05 62.24 106.1 ... 73.12 47.3
            sqrt_n_analysts     (isin) float64 50kB 7.616 2.449 3.742 ... 1.732 1.414
            ...                  ...
            exchange_idx        (isin) int32 25kB 56 74 56 74 56 80 ... 10 10 44 10 10
            unit_idx            (isin) int32 25kB 47 6 47 6 47 14 47 ... 33 5 5 5 33 5 5
            style_class_idx     (isin) int32 25kB 1 0 0 3 0 0 0 0 0 ... 0 3 3 0 3 3 3 0
            size_class_idx      (isin) int32 25kB 0 2 1 1 0 1 0 1 2 ... 2 2 2 2 2 2 2 2
            sector_idx          (isin) int32 25kB 6 7 6 3 6 6 0 0 7 ... 1 5 2 1 4 5 1 0
            industry_idx        (isin) int32 25kB 49 41 50 43 52 32 ... 23 29 27 7 51 35
        Attributes:
            created_at:                 2026-06-03T12:17:12.396870+00:00
            creation_library:           ArviZ
            creation_library_version:   1.1.0
            creation_library_language:  Python
            inference_library:          pymc
            inference_library_version:  6.0.1
            sample_dims:                []</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataTree</div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-children'><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-2e670f83-9b01-4216-b3c8-9abc9d8e28b1' type='checkbox' checked /><label for='group-2e670f83-9b01-4216-b3c8-9abc9d8e28b1' title='Expand/collapse group'>/prior<span>(42)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-b90d50df-c3c4-4bce-bec2-36a437deee5a' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-b90d50df-c3c4-4bce-bec2-36a437deee5a' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1500</li><li><span class='xr-has-index'>isin</span>: 6277</li><li><span class='xr-has-index'>sector</span>: 9</li><li><span class='xr-has-index'>unit</span>: 50</li><li><span class='xr-has-index'>style_class</span>: 4</li><li><span class='xr-has-index'>drift_feature</span>: 7</li><li><span class='xr-has-index'>exchange</span>: 82</li><li><span class='xr-has-index'>size_class</span>: 4</li><li><span class='xr-has-index'>industry</span>: 59</li><li><span class='xr-has-index'>region</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-b7546c02-1d98-4c5b-bfcd-3293d399b581' class='xr-section-summary-in' type='checkbox' checked /><label for='section-b7546c02-1d98-4c5b-bfcd-3293d399b581' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-3c70c1bf-27c9-43f5-b48a-64459ea74886' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3c70c1bf-27c9-43f5-b48a-64459ea74886' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d9fd2b8d-66ec-42c2-b36e-826fb76c82b1' class='xr-var-data-in' type='checkbox'><label for='data-d9fd2b8d-66ec-42c2-b36e-826fb76c82b1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1496 1497 1498 1499</div><input id='attrs-34ac7b06-5d5f-40eb-a390-95ff9ecea33c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-34ac7b06-5d5f-40eb-a390-95ff9ecea33c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-96c8a094-75e0-40f6-bf65-0757a6cd2e01' class='xr-var-data-in' type='checkbox'><label for='data-96c8a094-75e0-40f6-bf65-0757a6cd2e01' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1497, 1498, 1499], shape=(1500,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-0ff3e3ba-d3ef-4fc7-8041-ea644c42472e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0ff3e3ba-d3ef-4fc7-8041-ea644c42472e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1061bcde-2b05-412d-a985-56ed32b0cc31' class='xr-var-data-in' type='checkbox'><label for='data-1061bcde-2b05-412d-a985-56ed32b0cc31' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sector</span></div><div class='xr-var-dims'>(sector)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;Communication Services&#x27; ... &#x27;Ut...</div><input id='attrs-2ee05e57-ff30-4314-af82-ceb2c2255d44' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2ee05e57-ff30-4314-af82-ceb2c2255d44' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-289467eb-0673-4370-a8fb-43275bd88c12' class='xr-var-data-in' type='checkbox'><label for='data-289467eb-0673-4370-a8fb-43275bd88c12' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Communication Services&#x27;, &#x27;Consumer Discretionary&#x27;, &#x27;Consumer Staples&#x27;,&#x27;Energy&#x27;, &#x27;Health Care&#x27;, &#x27;Industrials&#x27;, &#x27;Information Technology&#x27;,&#x27;Materials&#x27;, &#x27;Utilities&#x27;], dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>unit</span></div><div class='xr-var-dims'>(unit)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;AED&#x27; &#x27;ARS&#x27; &#x27;AUD&#x27; ... &#x27;VND&#x27; &#x27;ZAR&#x27;</div><input id='attrs-aca16cc8-be7f-4346-8679-3b106984a9b2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-aca16cc8-be7f-4346-8679-3b106984a9b2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-887ee3a3-93f1-4f01-9d3d-944fad6a0fcf' class='xr-var-data-in' type='checkbox'><label for='data-887ee3a3-93f1-4f01-9d3d-944fad6a0fcf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;AED&#x27;, &#x27;ARS&#x27;, &#x27;AUD&#x27;, &#x27;BDT&#x27;, &#x27;BHD&#x27;, &#x27;BRL&#x27;, &#x27;CAD&#x27;, &#x27;CHF&#x27;, &#x27;CLP&#x27;, &#x27;CNY&#x27;,&#x27;COP&#x27;, &#x27;CZK&#x27;, &#x27;DKK&#x27;, &#x27;EGP&#x27;, &#x27;EUR&#x27;, &#x27;GBP&#x27;, &#x27;GHS&#x27;, &#x27;HKD&#x27;, &#x27;HUF&#x27;, &#x27;IDR&#x27;,&#x27;ILS&#x27;, &#x27;INR&#x27;, &#x27;JPY&#x27;, &#x27;KES&#x27;, &#x27;KRW&#x27;, &#x27;KWD&#x27;, &#x27;KZT&#x27;, &#x27;MAD&#x27;, &#x27;MXN&#x27;, &#x27;MYR&#x27;,&#x27;NGN&#x27;, &#x27;NOK&#x27;, &#x27;NZD&#x27;, &#x27;OMR&#x27;, &#x27;PEN&#x27;, &#x27;PHP&#x27;, &#x27;PKR&#x27;, &#x27;PLN&#x27;, &#x27;QAR&#x27;, &#x27;RON&#x27;,&#x27;RSD&#x27;, &#x27;SAR&#x27;, &#x27;SEK&#x27;, &#x27;SGD&#x27;, &#x27;THB&#x27;, &#x27;TRY&#x27;, &#x27;TWD&#x27;, &#x27;USD&#x27;, &#x27;VND&#x27;, &#x27;ZAR&#x27;],dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>style_class</span></div><div class='xr-var-dims'>(style_class)</div><div class='xr-var-dtype'>&lt;U7</div><div class='xr-var-preview xr-preview'>&#x27;Core&#x27; &#x27;Growth&#x27; &#x27;Unknown&#x27; &#x27;Value&#x27;</div><input id='attrs-a285b86b-d943-4629-8d40-2d1040699cce' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a285b86b-d943-4629-8d40-2d1040699cce' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-15bc5b6e-f1e4-4ab2-8b73-10022cef6c59' class='xr-var-data-in' type='checkbox'><label for='data-15bc5b6e-f1e4-4ab2-8b73-10022cef6c59' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Core&#x27;, &#x27;Growth&#x27;, &#x27;Unknown&#x27;, &#x27;Value&#x27;], dtype=&#x27;&lt;U7&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>drift_feature</span></div><div class='xr-var-dims'>(drift_feature)</div><div class='xr-var-dtype'>&lt;U21</div><div class='xr-var-preview xr-preview'>&#x27;feat_pt_drift&#x27; ... &#x27;feat_total_...</div><input id='attrs-659336ba-691e-483a-adce-05c22bc768b4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-659336ba-691e-483a-adce-05c22bc768b4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-56f34e3b-8ac4-4edd-90a1-77cacdb5efda' class='xr-var-data-in' type='checkbox'><label for='data-56f34e3b-8ac4-4edd-90a1-77cacdb5efda' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;feat_pt_drift&#x27;, &#x27;feat_price_drift&#x27;, &#x27;feat_pt_high_drift&#x27;,&#x27;feat_pt_low_drift&#x27;, &#x27;feat_pt_median_drift&#x27;, &#x27;feat_coverage_drift&#x27;,&#x27;feat_total_return_ytd&#x27;], dtype=&#x27;&lt;U21&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>exchange</span></div><div class='xr-var-dims'>(exchange)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;ADX&#x27; &#x27;AIM&#x27; &#x27;ASX&#x27; ... &#x27;XTRA&#x27; &#x27;ZGSE&#x27;</div><input id='attrs-79862441-ed3d-47a2-953a-ab5d3acbde0a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-79862441-ed3d-47a2-953a-ab5d3acbde0a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-362da32b-3cea-44a3-84fe-0d54a0a3b892' class='xr-var-data-in' type='checkbox'><label for='data-362da32b-3cea-44a3-84fe-0d54a0a3b892' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;ADX&#x27;, &#x27;AIM&#x27;, &#x27;ASX&#x27;, &#x27;ATSE&#x27;, &#x27;BASE&#x27;, &#x27;BAX&#x27;, &#x27;BELEX&#x27;, &#x27;BIT&#x27;, &#x27;BME&#x27;,&#x27;BMV&#x27;, &#x27;BOVESPA&#x27;, &#x27;BSE&#x27;, &#x27;BUL&#x27;, &#x27;BUSE&#x27;, &#x27;BVB&#x27;, &#x27;BVC&#x27;, &#x27;BVL&#x27;, &#x27;CASE&#x27;,&#x27;CBSE&#x27;, &#x27;CNSX&#x27;, &#x27;CPSE&#x27;, &#x27;DB&#x27;, &#x27;DFM&#x27;, &#x27;DSE&#x27;, &#x27;DSM&#x27;, &#x27;ENXTAM&#x27;, &#x27;ENXTBR&#x27;,&#x27;ENXTLS&#x27;, &#x27;ENXTPA&#x27;, &#x27;GHSE&#x27;, &#x27;HLSE&#x27;, &#x27;HOSE&#x27;, &#x27;IBSE&#x27;, &#x27;IDX&#x27;, &#x27;ISE&#x27;, &#x27;JSE&#x27;,&#x27;KAS&#x27;, &#x27;KASE&#x27;, &#x27;KLSE&#x27;, &#x27;KOSDAQ&#x27;, &#x27;KOSE&#x27;, &#x27;KWSE&#x27;, &#x27;LJSE&#x27;, &#x27;LSE&#x27;, &#x27;MSM&#x27;,&#x27;MUN&#x27;, &#x27;NASE&#x27;, &#x27;NGM&#x27;, &#x27;NGSE&#x27;, &#x27;NSEI&#x27;, &#x27;NSEL&#x27;, &#x27;NYSE&#x27;, &#x27;NYSEAM&#x27;, &#x27;NZSE&#x27;,&#x27;NasdaqCM&#x27;, &#x27;NasdaqGM&#x27;, &#x27;NasdaqGS&#x27;, &#x27;OB&#x27;, &#x27;OM&#x27;, &#x27;OTCPK&#x27;, &#x27;PSE&#x27;, &#x27;SASE&#x27;,&#x27;SEHK&#x27;, &#x27;SEP&#x27;, &#x27;SET&#x27;, &#x27;SGX&#x27;, &#x27;SHSE&#x27;, &#x27;SNSE&#x27;, &#x27;SWX&#x27;, &#x27;SZSE&#x27;, &#x27;TASE&#x27;,&#x27;TLSE&#x27;, &#x27;TPEX&#x27;, &#x27;TSE&#x27;, &#x27;TSX&#x27;, &#x27;TSXV&#x27;, &#x27;TWSE&#x27;, &#x27;WBAG&#x27;, &#x27;WSE&#x27;, &#x27;XSAT&#x27;,&#x27;XTRA&#x27;, &#x27;ZGSE&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>size_class</span></div><div class='xr-var-dims'>(size_class)</div><div class='xr-var-dtype'>&lt;U9</div><div class='xr-var-preview xr-preview'>&#x27;Large Cap&#x27; &#x27;Mid Cap&#x27; ... &#x27;Unknown&#x27;</div><input id='attrs-bcae56a9-9194-4999-93a2-5ae705e6a3c4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bcae56a9-9194-4999-93a2-5ae705e6a3c4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-444136be-6be0-4df8-8c6e-bb0d9e6b1570' class='xr-var-data-in' type='checkbox'><label for='data-444136be-6be0-4df8-8c6e-bb0d9e6b1570' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Large Cap&#x27;, &#x27;Mid Cap&#x27;, &#x27;Small Cap&#x27;, &#x27;Unknown&#x27;], dtype=&#x27;&lt;U9&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>industry</span></div><div class='xr-var-dims'>(industry)</div><div class='xr-var-dtype'>&lt;U53</div><div class='xr-var-preview xr-preview'>&#x27;Aerospace and Defense&#x27; ... &#x27;Wir...</div><input id='attrs-f791d7db-8e9a-4caf-ad5c-30ceee891e15' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f791d7db-8e9a-4caf-ad5c-30ceee891e15' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3b7bb357-ec58-40e8-bc46-c8f3331810c6' class='xr-var-data-in' type='checkbox'><label for='data-3b7bb357-ec58-40e8-bc46-c8f3331810c6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Aerospace and Defense&#x27;, &#x27;Air Freight and Logistics&#x27;,&#x27;Automobile Components&#x27;, &#x27;Automobiles&#x27;, &#x27;Beverages&#x27;, &#x27;Biotechnology&#x27;,&#x27;Broadline Retail&#x27;, &#x27;Building Products&#x27;, &#x27;Chemicals&#x27;,&#x27;Commercial Services and Supplies&#x27;, &#x27;Communications Equipment&#x27;,&#x27;Construction Materials&#x27;, &#x27;Construction and Engineering&#x27;,&#x27;Consumer Staples Distribution and Retail&#x27;, &#x27;Containers and Packaging&#x27;,&#x27;Distributors&#x27;, &#x27;Diversified Consumer Services&#x27;,&#x27;Diversified Telecommunication Services&#x27;, &#x27;Electric Utilities&#x27;,&#x27;Electrical Equipment&#x27;,&#x27;Electronic Equipment Instruments and Components&#x27;,&#x27;Energy Equipment and Services&#x27;, &#x27;Entertainment&#x27;, &#x27;Food Products&#x27;,&#x27;Gas Utilities&#x27;, &#x27;Ground Transportation&#x27;,&#x27;Health Care Equipment and Supplies&#x27;,&#x27;Health Care Providers and Services&#x27;, &#x27;Health Care Technology&#x27;,&#x27;Hotels Restaurants and Leisure&#x27;, &#x27;Household Durables&#x27;,&#x27;Household Products&#x27;, &#x27;IT Services&#x27;,&#x27;Independent Power and Renewable Electricity Producers&#x27;,&#x27;Industrial Conglomerates&#x27;, &#x27;Interactive Media and Services&#x27;,&#x27;Leisure Products&#x27;, &#x27;Life Sciences Tools and Services&#x27;, &#x27;Machinery&#x27;,&#x27;Marine Transportation&#x27;, &#x27;Media&#x27;, &#x27;Metals and Mining&#x27;,&#x27;Multi-Utilities&#x27;, &#x27;Oil Gas and Consumable Fuels&#x27;,&#x27;Paper and Forest Products&#x27;, &#x27;Passenger Airlines&#x27;,&#x27;Personal Care Products&#x27;, &#x27;Pharmaceuticals&#x27;, &#x27;Professional Services&#x27;,&#x27;Semiconductors and Semiconductor Equipment&#x27;, &#x27;Software&#x27;,&#x27;Specialty Retail&#x27;, &#x27;Technology Hardware Storage and Peripherals&#x27;,&#x27;Textiles Apparel and Luxury Goods&#x27;, &#x27;Tobacco&#x27;,&#x27;Trading Companies and Distributors&#x27;, &#x27;Transportation Infrastructure&#x27;,&#x27;Water Utilities&#x27;, &#x27;Wireless Telecommunication Services&#x27;], dtype=&#x27;&lt;U53&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>region</span></div><div class='xr-var-dims'>(region)</div><div class='xr-var-dtype'>&lt;U27</div><div class='xr-var-preview xr-preview'>&#x27;Africa / Middle East&#x27; ... &#x27;Unit...</div><input id='attrs-7f18bec5-803e-4289-974c-ceb580043ab7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7f18bec5-803e-4289-974c-ceb580043ab7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9f7a7eab-4f6f-4514-a7fe-168e1cdbb689' class='xr-var-data-in' type='checkbox'><label for='data-9f7a7eab-4f6f-4514-a7fe-168e1cdbb689' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Africa / Middle East&#x27;, &#x27;Asia / Pacific&#x27;, &#x27;Europe&#x27;,&#x27;Latin America and Caribbean&#x27;, &#x27;United States and Canada&#x27;], dtype=&#x27;&lt;U27&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ed824953-a61a-4431-8c8f-6448f9c60447' class='xr-section-summary-in' type='checkbox' /><label for='section-ed824953-a61a-4431-8c8f-6448f9c60447' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(24)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>mu_global</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.4414 -0.3835 ... -0.1683</div><input id='attrs-50583a01-9480-46f3-b58b-4ec510bf0252' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-50583a01-9480-46f3-b58b-4ec510bf0252' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8f317d67-7a8c-4872-a249-30ad3d6b9f9a' class='xr-var-data-in' type='checkbox'><label for='data-8f317d67-7a8c-4872-a249-30ad3d6b9f9a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.44139838, -0.38352167, -0.05186509, ...,  0.28637775,-0.03101896, -0.16827452]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_state</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01626 0.06206 ... 0.1402 0.01211</div><input id='attrs-41182f6d-e99b-45f4-b4aa-4250a870e474' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-41182f6d-e99b-45f4-b4aa-4250a870e474' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c62c0723-298c-4b53-b13a-31cdc31d83ed' class='xr-var-data-in' type='checkbox'><label for='data-c62c0723-298c-4b53-b13a-31cdc31d83ed' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.016257  , 0.06206468, 0.02081688, ..., 0.31189399, 0.14018164,0.01211298]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>log_state</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.827 0.03068 ... 0.5382 -0.2693</div><input id='attrs-562dbb71-2ce0-4129-991a-1812a9d0c79b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-562dbb71-2ce0-4129-991a-1812a9d0c79b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9fc3b14f-06a1-410c-98f4-2cb58a44bcb2' class='xr-var-data-in' type='checkbox'><label for='data-9fc3b14f-06a1-410c-98f4-2cb58a44bcb2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 4.82654744,  0.03067879,  0.61858701, ..., -1.92416361,0.49053299, -0.24491719],[ 4.38648651,  0.42204694,  3.70580587, ..., -2.30315549,-0.39585681, -0.17398054],[ 5.38620934,  1.44688725,  6.00871443, ..., -2.10774696,-0.12739526, -0.24072439],...,[ 5.66045172,  1.63552688,  8.18038075, ..., -1.8707992 ,0.68686653, -0.16645846],[ 6.41967154,  2.76013035, 12.72100136, ..., -2.8758674 ,-0.47122446, -0.65327001],[ 5.63005786,  0.54761619,  4.5581315 , ..., -2.12035119,0.53823105, -0.26932453]]], shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sector_effect</span></div><div class='xr-var-dims'>(chain, draw, sector)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.00126 -0.006054 ... 0.02049</div><input id='attrs-3f8e4465-55e4-49ca-8e30-85d4ebe22fd3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3f8e4465-55e4-49ca-8e30-85d4ebe22fd3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb842c03-b24d-4028-877a-c9c17bfef5bc' class='xr-var-data-in' type='checkbox'><label for='data-eb842c03-b24d-4028-877a-c9c17bfef5bc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.00125968, -0.00605355,  0.00259047, ..., -0.00479616,-0.00073402,  0.00394246],[ 0.15392323, -0.04563323, -0.1117736 , ..., -0.15814968,0.01150599,  0.27295372],[ 0.0677385 ,  0.00919636,  0.00153917, ...,  0.03837203,0.01328903, -0.00943675],...,[ 0.0625874 , -0.09199525,  0.05662662, ...,  0.09001267,0.14669085,  0.14425024],[ 0.00988178, -0.03331046, -0.08775964, ...,  0.05642218,-0.19560978, -0.00047782],[-0.17526482,  0.24461852, -0.04510214, ...,  0.04214796,-0.12522805,  0.02048864]]], shape=(1, 1500, 9))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_region</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1799 0.1776 ... 0.03868 0.0553</div><input id='attrs-9ef52d24-0f4b-4ed2-97f5-c9ea8e295e08' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9ef52d24-0f4b-4ed2-97f5-c9ea8e295e08' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-421f25ff-357c-42d1-97a9-c92bb0c9750a' class='xr-var-data-in' type='checkbox'><label for='data-421f25ff-357c-42d1-97a9-c92bb0c9750a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.17989763, 0.1775819 , 0.09371284, ..., 0.05291322, 0.03868458,0.05530334]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_unit</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1254 0.06063 ... 0.1205 0.2454</div><input id='attrs-18dd26c6-9910-49df-a8d6-165ae81df693' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-18dd26c6-9910-49df-a8d6-165ae81df693' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-76e653b7-81c9-495c-8bd2-45df812ff572' class='xr-var-data-in' type='checkbox'><label for='data-76e653b7-81c9-495c-8bd2-45df812ff572' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.12544944, 0.06062894, 0.13401776, ..., 0.01166517, 0.12052824,0.24543335]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>unit_effect</span></div><div class='xr-var-dims'>(chain, draw, unit)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05248 0.07597 ... 0.4127 0.03195</div><input id='attrs-ba36be44-d9f4-478f-92a5-fe40dffd51f7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ba36be44-d9f4-478f-92a5-fe40dffd51f7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8ad81b5e-5fc7-4727-b67d-f83b4ce964d6' class='xr-var-data-in' type='checkbox'><label for='data-8ad81b5e-5fc7-4727-b67d-f83b4ce964d6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.05247926,  0.07596919,  0.00361142, ..., -0.01249912,-0.03232716, -0.19940422],[-0.11003622,  0.03250761,  0.07708302, ..., -0.04513167,0.07916604,  0.05274733],[ 0.1070126 ,  0.11179468, -0.04971972, ..., -0.0700918 ,0.24596961, -0.19209791],...,[-0.00118017, -0.00320749, -0.01157246, ..., -0.00667415,0.00592526, -0.02132042],[-0.0729396 , -0.18662049, -0.11141227, ...,  0.21808834,-0.07843459,  0.13918112],[ 0.3981236 , -0.17541456,  0.02132575, ...,  0.07227097,0.4126904 ,  0.03195034]]], shape=(1, 1500, 50))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>style_class_effect</span></div><div class='xr-var-dims'>(chain, draw, style_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.01656 0.01412 ... 0.1503 -0.2253</div><input id='attrs-38b98079-6bfe-4207-8806-2bda2b2bf057' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-38b98079-6bfe-4207-8806-2bda2b2bf057' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-04a0118d-59e8-4de0-a947-6ed207ff9dfe' class='xr-var-data-in' type='checkbox'><label for='data-04a0118d-59e8-4de0-a947-6ed207ff9dfe' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-1.65641471e-02,  1.41197938e-02,  7.26548454e-03,-8.32592108e-03],[ 7.06504265e-02, -1.74870973e-01, -4.10731917e-02,-1.20639557e-01],[-5.19347106e-02,  2.92288799e-02,  7.81167659e-02,-2.71915418e-01],...,[-3.36597013e-05, -3.68094217e-03, -5.18727590e-02,7.77647863e-03],[-3.93696443e-02,  2.82102106e-02,  6.26906886e-03,-3.66974306e-02],[ 1.95103615e-04,  4.09881636e-02,  1.50303579e-01,-2.25290761e-01]]], shape=(1, 1500, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_style_class</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01004 0.146 ... 0.06792 0.1538</div><input id='attrs-ae1e471c-2f8d-4dc1-a48b-282994141c5f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ae1e471c-2f8d-4dc1-a48b-282994141c5f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-418fe5c0-e390-48aa-9aae-4cd27c8e48ea' class='xr-var-data-in' type='checkbox'><label for='data-418fe5c0-e390-48aa-9aae-4cd27c8e48ea' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.01004253, 0.14601151, 0.15816199, ..., 0.01614866, 0.06792142,0.15380362]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_size_class</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.00408 0.1292 ... 0.04307 0.07004</div><input id='attrs-62cd85b9-9f1e-4aa5-8d35-80925a358b00' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-62cd85b9-9f1e-4aa5-8d35-80925a358b00' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cc5b9403-c538-460f-be91-b3915bfeebc4' class='xr-var-data-in' type='checkbox'><label for='data-cc5b9403-c538-460f-be91-b3915bfeebc4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00407954, 0.12922916, 0.11060499, ..., 0.02912281, 0.04306771,0.07003899]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta</span></div><div class='xr-var-dims'>(chain, draw, drift_feature)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.2863 -0.1277 ... -0.06361</div><input id='attrs-31f6755b-f6cd-44a5-bdec-d7ca462cbfd5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-31f6755b-f6cd-44a5-bdec-d7ca462cbfd5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2395a351-032e-4db0-9c93-0e0528fc1f21' class='xr-var-data-in' type='checkbox'><label for='data-2395a351-032e-4db0-9c93-0e0528fc1f21' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.28633977, -0.12772186, -0.26067095, ...,  0.27570941,-0.52859973, -0.21709405],[-0.14692776,  0.06730456, -0.0352118 , ...,  0.45728052,0.04257582, -0.28578557],[-0.00435123,  0.0149144 ,  0.21851569, ...,  0.01944651,0.30619205, -0.28948858],...,[-0.04463661,  0.05364938,  0.43685085, ...,  0.14729928,-0.17153808, -0.05723801],[ 0.35677264,  0.28801532,  0.41417638, ...,  0.2469996 ,-0.19856741,  0.04392637],[-0.14249218, -0.45485842,  0.29137746, ...,  0.18105849,-0.12219156, -0.06360512]]], shape=(1, 1500, 7))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>expected_upside</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.44 -0.5397 ... 0.2063 -0.2725</div><input id='attrs-f0a322cc-0f6a-4b80-88a3-37934c1a46fe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f0a322cc-0f6a-4b80-88a3-37934c1a46fe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dfae60ba-a24e-4313-8263-c6786973deaf' class='xr-var-data-in' type='checkbox'><label for='data-dfae60ba-a24e-4313-8263-c6786973deaf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-4.39999070e-01, -5.39663286e-01, -9.72127579e-01, ...,-1.04307872e-01,  1.50131311e-01, -2.54505376e-01],[-6.39360977e-01, -3.19160665e-01, -3.89146858e-01, ...,-3.86852904e-01, -5.25983751e-01, -1.99701664e-01],[-1.99531783e-02,  8.97261228e-01,  5.11050763e+00, ...,-2.54531113e-01, -3.80010122e-01, -2.51373105e-01],...,[ 2.89284548e-01,  1.29114344e+00,  5.26068133e+01, ...,-5.52114361e-02,  3.99632433e-01, -1.93659057e-01],[ 1.75469648e+00,  6.05442077e+00,  5.02458318e+03, ...,-6.54188802e-01, -5.60396164e-01, -5.04436698e-01],[ 2.50687731e-01, -2.28068664e-01,  4.32508241e-01, ...,-2.63868210e-01,  2.06319734e-01, -2.72480660e-01]]],shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_obs</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1653 0.8318 ... 1.619 1.317</div><input id='attrs-2ca97d33-c7f3-4e50-93de-cc5cef268ac0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2ca97d33-c7f3-4e50-93de-cc5cef268ac0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dc946bd2-b2ca-4b5d-ab15-1a9ace8c5c73' class='xr-var-data-in' type='checkbox'><label for='data-dc946bd2-b2ca-4b5d-ab15-1a9ace8c5c73' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.16526528, 0.83184032, 0.89531265, ..., 0.81429819,1.33336358, 1.08469358],[0.25067599, 1.26174352, 1.35801898, ..., 1.23513544,2.0224589 , 1.64527381],[0.27325173, 1.37537546, 1.48032142, ..., 1.34637107,2.20460044, 1.79344627],...,[0.20028538, 1.00810925, 1.08503152, ..., 0.9868499 ,1.61590647, 1.31454271],[0.06590696, 0.33173373, 0.35704618, ..., 0.32473802,0.53173868, 0.43257033],[0.20063236, 1.00985573, 1.08691127, ..., 0.98855955,1.61870592, 1.31682007]]], shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_sector</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.003107 0.1531 ... 0.1355 0.1423</div><input id='attrs-662a239a-d634-44a3-a8d7-569a6b77723f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-662a239a-d634-44a3-a8d7-569a6b77723f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0ef517dd-e78f-4f1d-97ff-1f1bd57dff79' class='xr-var-data-in' type='checkbox'><label for='data-0ef517dd-e78f-4f1d-97ff-1f1bd57dff79' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00310676, 0.15314449, 0.03178781, ..., 0.13144191, 0.13552164,0.1423259 ]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_obs_base</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06038 0.09159 ... 0.02408 0.0733</div><input id='attrs-d9122df7-949f-472e-98e0-7863379fae69' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d9122df7-949f-472e-98e0-7863379fae69' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1ccaf7d0-5015-4a69-bb5b-1566e43a0595' class='xr-var-data-in' type='checkbox'><label for='data-1ccaf7d0-5015-4a69-bb5b-1566e43a0595' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.06038118, 0.09158676, 0.09983501, ..., 0.07317609, 0.02407971,0.07330286]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>exchange_effect</span></div><div class='xr-var-dims'>(chain, draw, exchange)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.2779 -0.01252 ... -0.1403</div><input id='attrs-4ea34135-d2ec-4f83-978c-a0a2ce52e486' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4ea34135-d2ec-4f83-978c-a0a2ce52e486' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9206f3e4-c356-4b1b-9015-3d1d92e741ab' class='xr-var-data-in' type='checkbox'><label for='data-9206f3e4-c356-4b1b-9015-3d1d92e741ab' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.27788533, -0.01251816,  0.2638478 , ...,  0.05503112,0.0820753 ,  0.00080582],[-0.00523331,  0.04071827, -0.01091674, ..., -0.00270655,0.00995012,  0.01353106],[ 0.00268005,  0.01575376, -0.03018436, ...,  0.00100361,-0.00450001,  0.00823011],...,[-0.03970775,  0.13143627, -0.01688807, ..., -0.06713058,-0.01603182,  0.02507021],[ 0.02805582,  0.05284884,  0.01173012, ...,  0.00092914,-0.04900476, -0.04629647],[-0.1752962 , -0.12346944,  0.12007152, ...,  0.12148574,0.00086379, -0.14030409]]], shape=(1, 1500, 82))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>size_class_effect</span></div><div class='xr-var-dims'>(chain, draw, size_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.001401 -0.006774 ... -0.03278</div><input id='attrs-17329017-4434-4618-adba-f782a55f4d25' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-17329017-4434-4618-adba-f782a55f4d25' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-895554de-1106-403a-96eb-14a3acab07a3' class='xr-var-data-in' type='checkbox'><label for='data-895554de-1106-403a-96eb-14a3acab07a3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.00140051, -0.0067741 , -0.00157271,  0.00534046],[ 0.02269266, -0.15438131, -0.04151551,  0.16312906],[-0.13765149, -0.01107429,  0.1795758 , -0.13194805],...,[ 0.00592603, -0.02290676,  0.02751711,  0.02959185],[ 0.00724211,  0.01252838,  0.04103109, -0.00667568],[-0.13214169,  0.0785207 , -0.00914956, -0.03277782]]],shape=(1, 1500, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>expected_pt</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>124.8 1.031 1.856 ... 1.713 0.7639</div><input id='attrs-2b58fec6-3794-4d76-91b0-f6c5f9129980' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2b58fec6-3794-4d76-91b0-f6c5f9129980' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c20aadbe-5d5c-4d6f-a810-581eec0f25dd' class='xr-var-data-in' type='checkbox'><label for='data-c20aadbe-5d5c-4d6f-a810-581eec0f25dd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[1.24779407e+02, 1.03115424e+00, 1.85630324e+00, ...,1.45997817e-01, 1.63318646e+00, 7.82769356e-01],[8.03575871e+01, 1.52508011e+00, 4.06828193e+01, ...,9.99429767e-02, 6.73103074e-01, 8.40313252e-01],[2.18374033e+02, 4.24986515e+00, 4.06959808e+02, ...,1.21511429e-01, 8.80385626e-01, 7.86058240e-01],...,[2.87278383e+02, 5.13216131e+00, 3.57021377e+03, ...,1.54000536e-01, 1.98747806e+00, 8.46657990e-01],[6.13801471e+02, 1.58019025e+01, 3.34703840e+05, ...,5.63672253e-02, 6.24237447e-01, 5.20341468e-01],[2.78678240e+02, 1.72912619e+00, 9.54050489e+01, ...,1.19989482e-01, 1.71297402e+00, 7.63895307e-01]]],shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>log_uplift</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.5798 -0.7758 ... 0.1876 -0.3181</div><input id='attrs-caf9e154-2960-40ab-9070-fd877631f9bf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-caf9e154-2960-40ab-9070-fd877631f9bf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-36a719fc-d199-4fde-8f07-e92bf8ee7200' class='xr-var-data-in' type='checkbox'><label for='data-36a719fc-d199-4fde-8f07-e92bf8ee7200' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.57981683, -0.77579707, -3.58011757, ..., -0.11015853,0.13987612, -0.29370736],[-1.01987776, -0.38442893, -0.49289871, ..., -0.48915041,-0.74651368, -0.2227707 ],[-0.02015493,  0.64041139,  1.81000985, ..., -0.29374188,-0.47805213, -0.28951456],...,[ 0.25408745,  0.82905101,  3.98167617, ..., -0.05679412,0.33620965, -0.21524862],[ 1.01330727,  1.95365448,  8.52229678, ..., -1.06186232,-0.82188133, -0.70206018],[ 0.22369358, -0.25885968,  0.35942692, ..., -0.30634611,0.18757418, -0.3181147 ]]], shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>nu</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>36.22 20.02 9.606 ... 9.037 4.163</div><input id='attrs-dfddf45b-a229-4a14-93a2-8e6aa827fc72' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dfddf45b-a229-4a14-93a2-8e6aa827fc72' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-877d8b3c-ac79-4d56-821f-eeee7ea7f600' class='xr-var-data-in' type='checkbox'><label for='data-877d8b3c-ac79-4d56-821f-eeee7ea7f600' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[36.22277864, 20.0192584 ,  9.60568985, ..., 16.98088256,9.03656591,  4.16316687]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>industry_effect</span></div><div class='xr-var-dims'>(chain, draw, industry)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1301 0.06239 ... -0.007945</div><input id='attrs-9d11308e-ee46-4a92-918f-1a85dcfd18af' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9d11308e-ee46-4a92-918f-1a85dcfd18af' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dcd91f5b-6948-4c3d-8d42-35f5fc7e19e6' class='xr-var-data-in' type='checkbox'><label for='data-dcd91f5b-6948-4c3d-8d42-35f5fc7e19e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 1.30055345e-01,  6.23917667e-02,  2.34435091e-01, ...,-6.68829598e-02, -3.89289109e-01, -3.93957314e-02],[ 1.37834808e-01,  4.84188475e-02,  4.80425673e-02, ...,-5.27088874e-03,  3.06490710e-02,  2.93458857e-02],[-6.27467600e-02, -7.63162237e-02, -1.83194942e-01, ...,2.13054322e-01,  1.80996865e-02,  1.00817479e-01],...,[ 5.90430351e-04, -3.61391062e-03, -1.33223993e-03, ...,1.01645755e-03,  1.01409447e-03,  2.46889389e-04],[-1.10204162e-02, -1.93582261e-01,  1.12906197e-01, ...,-4.78675852e-02,  2.90176807e-02, -7.98962616e-02],[-4.54257305e-03, -2.44990633e-03,  7.64082801e-03, ...,3.84776728e-03,  7.09755464e-02, -7.94528343e-03]]],shape=(1, 1500, 59))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_exchange</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1919 0.02467 ... 0.03263 0.1059</div><input id='attrs-adf7d4d4-2606-4973-8632-89fe1d8b653e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-adf7d4d4-2606-4973-8632-89fe1d8b653e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d35bf3cb-3326-4dfe-9fb5-ec487221b4ee' class='xr-var-data-in' type='checkbox'><label for='data-d35bf3cb-3326-4dfe-9fb5-ec487221b4ee' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.19189022, 0.02467022, 0.01149717, ..., 0.05692886, 0.03263098,0.10586986]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>region_effect</span></div><div class='xr-var-dims'>(chain, draw, region)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06427 0.2888 ... 0.104 -0.03712</div><input id='attrs-c9f33d79-cf0b-4ae8-8a8f-b9b7ac5e99d0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c9f33d79-cf0b-4ae8-8a8f-b9b7ac5e99d0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1123c8df-dcd1-4d96-8535-109d36498df3' class='xr-var-data-in' type='checkbox'><label for='data-1123c8df-dcd1-4d96-8535-109d36498df3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.06426998,  0.28882507, -0.10830278, -0.06709542,0.29087117],[ 0.04071519, -0.09597525,  0.18594158, -0.05201069,0.00470652],[-0.01624447,  0.10874697,  0.06180368, -0.01908726,-0.07538356],...,[ 0.01212273,  0.03703203,  0.02278174, -0.00245837,-0.1244127 ],[ 0.01937367, -0.02019158,  0.07647142,  0.03932423,-0.00287219],[ 0.00306565, -0.00401059,  0.08037764,  0.1040281 ,-0.03712096]]], shape=(1, 1500, 5))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_industry</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1386 0.05796 ... 0.06928 0.07764</div><input id='attrs-c874605d-ed89-488f-98c1-5aa6a191f812' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c874605d-ed89-488f-98c1-5aa6a191f812' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3d1e2ff7-ade5-41cf-9e10-621fa7c30c39' class='xr-var-data-in' type='checkbox'><label for='data-3d1e2ff7-ade5-41cf-9e10-621fa7c30c39' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.13855186, 0.05796484, 0.13190118, ..., 0.00170642, 0.06927729,0.0776376 ]], shape=(1, 1500))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-fd6e7856-c2d2-45fb-a456-268c090b4cff' class='xr-section-summary-in' type='checkbox' checked /><label for='section-fd6e7856-c2d2-45fb-a456-268c090b4cff' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.376301+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[&#x27;chain&#x27;, &#x27;draw&#x27;]</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-d2471381-e373-4521-aa62-5bfb3546775b' type='checkbox' checked /><label for='group-d2471381-e373-4521-aa62-5bfb3546775b' title='Expand/collapse group'>/prior_predictive<span>(11)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-f4915a4c-0f42-4131-8191-fbbcb21b2328' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-f4915a4c-0f42-4131-8191-fbbcb21b2328' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1500</li><li><span class='xr-has-index'>isin</span>: 6277</li></ul></div></li><li class='xr-section-item'><input id='section-d140e374-8bd0-4f2b-945d-69483b0ff409' class='xr-section-summary-in' type='checkbox' checked /><label for='section-d140e374-8bd0-4f2b-945d-69483b0ff409' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-7aef7887-a298-4b73-b104-f22c9cb75faf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7aef7887-a298-4b73-b104-f22c9cb75faf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f332dc94-8220-479b-9f77-4de6fbac0f2c' class='xr-var-data-in' type='checkbox'><label for='data-f332dc94-8220-479b-9f77-4de6fbac0f2c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1496 1497 1498 1499</div><input id='attrs-09e96354-90d6-4ba0-bc37-58630b5d9b59' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-09e96354-90d6-4ba0-bc37-58630b5d9b59' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e34adbbc-bbb1-46a5-9632-40392d8548b7' class='xr-var-data-in' type='checkbox'><label for='data-e34adbbc-bbb1-46a5-9632-40392d8548b7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1497, 1498, 1499], shape=(1500,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-5ad9ba03-023a-4f89-bd25-5b5f492f3737' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5ad9ba03-023a-4f89-bd25-5b5f492f3737' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-503ab602-4211-42ee-b5d4-2bf394e36f18' class='xr-var-data-in' type='checkbox'><label for='data-503ab602-4211-42ee-b5d4-2bf394e36f18' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-5cc9caea-7f83-42f9-9893-6e949cbff5a4' class='xr-section-summary-in' type='checkbox' checked /><label for='section-5cc9caea-7f83-42f9-9893-6e949cbff5a4' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>log_pt_obs</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.067 -0.5607 ... 0.2002 1.486</div><input id='attrs-b37496af-ae44-4ca3-a097-384039c766e4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b37496af-ae44-4ca3-a097-384039c766e4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-081608a1-5bf0-4cdf-a85c-7a1369d3c5da' class='xr-var-data-in' type='checkbox'><label for='data-081608a1-5bf0-4cdf-a85c-7a1369d3c5da' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 5.06734454, -0.56068546, -0.04740253, ..., -2.22850073,1.98605573, -1.28864302],[ 3.89250203,  0.60248521,  0.88996491, ..., -5.73717132,-0.18319716,  3.21042313],[ 4.99575714,  1.4900476 ,  6.7636942 , ..., -0.89771605,0.83761525, -1.14712391],...,[ 5.74314423,  1.94852195,  7.6408595 , ..., -3.46205783,-0.70500567, -0.18354805],[ 6.45156177,  2.87144397, 13.66273868, ..., -2.10221018,-0.60370203, -1.15757288],[ 5.61742055,  1.78070719,  5.68124411, ..., -4.6252466 ,0.20019369,  1.48596324]]], shape=(1, 1500, 6277))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-af641f4c-3bbb-4095-a5e9-d81830a86a6e' class='xr-section-summary-in' type='checkbox' checked /><label for='section-af641f4c-3bbb-4095-a5e9-d81830a86a6e' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.382459+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[&#x27;chain&#x27;, &#x27;draw&#x27;]</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-9a6de9e9-2206-4eea-b672-110c6ecc93f1' type='checkbox' checked /><label for='group-9a6de9e9-2206-4eea-b672-110c6ecc93f1' title='Expand/collapse group'>/observed_data<span>(9)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-d8bbbb92-9b00-4e38-a2ef-dd8713e20c3d' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-d8bbbb92-9b00-4e38-a2ef-dd8713e20c3d' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>isin</span>: 6277</li></ul></div></li><li class='xr-section-item'><input id='section-dbf84232-a9dc-4358-9042-f30a1591d539' class='xr-section-summary-in' type='checkbox' checked /><label for='section-dbf84232-a9dc-4358-9042-f30a1591d539' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-67d5345d-80db-4bc4-8e50-19aea14aa852' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-67d5345d-80db-4bc4-8e50-19aea14aa852' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b0816d9e-d464-4d2b-9a40-1aa8c02526c3' class='xr-var-data-in' type='checkbox'><label for='data-b0816d9e-d464-4d2b-9a40-1aa8c02526c3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7a62d645-1c50-4bb6-8b28-da967dfc2beb' class='xr-section-summary-in' type='checkbox' checked /><label for='section-7a62d645-1c50-4bb6-8b28-da967dfc2beb' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>log_pt_obs</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.693 1.481 4.38 ... 1.369 0.5008</div><input id='attrs-be39c554-4b0c-4dc8-b033-a8ef5d69aa27' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-be39c554-4b0c-4dc8-b033-a8ef5d69aa27' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1ec00169-455e-477f-9600-3bc07389544c' class='xr-var-data-in' type='checkbox'><label for='data-1ec00169-455e-477f-9600-3bc07389544c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 5.69309321,  1.48108168,  4.38007849, ..., -1.53247687,1.36947877,  0.50077529], shape=(6277,))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-38cbbcf5-2cc2-491e-ab6d-871aa34b4a80' class='xr-section-summary-in' type='checkbox' checked /><label for='section-38cbbcf5-2cc2-491e-ab6d-871aa34b4a80' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.384284+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[]</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 1.2em'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-8b6d1a6c-d01d-4a05-94ef-7a38bd12a446' type='checkbox' checked /><label for='group-8b6d1a6c-d01d-4a05-94ef-7a38bd12a446' title='Expand/collapse group'>/constant_data<span>(22)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-18d4c5e7-e5c6-4e3f-93cc-3d52b1eb2fe0' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-18d4c5e7-e5c6-4e3f-93cc-3d52b1eb2fe0' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>isin</span>: 6277</li><li><span class='xr-has-index'>drift_feature</span>: 7</li></ul></div></li><li class='xr-section-item'><input id='section-221c809c-4ce0-4300-b059-34c233d25c47' class='xr-section-summary-in' type='checkbox' checked /><label for='section-221c809c-4ce0-4300-b059-34c233d25c47' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-eb5469f2-8c9d-4d03-bf47-ed0663f44fbe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eb5469f2-8c9d-4d03-bf47-ed0663f44fbe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-47e2c017-3571-4264-b76d-9d46e7c205c2' class='xr-var-data-in' type='checkbox'><label for='data-47e2c017-3571-4264-b76d-9d46e7c205c2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>drift_feature</span></div><div class='xr-var-dims'>(drift_feature)</div><div class='xr-var-dtype'>&lt;U21</div><div class='xr-var-preview xr-preview'>&#x27;feat_pt_drift&#x27; ... &#x27;feat_total_...</div><input id='attrs-2bd0a780-2652-45dd-aba6-982067f3c98a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2bd0a780-2652-45dd-aba6-982067f3c98a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f14f875d-1df2-4a43-904b-97ffbf9e9343' class='xr-var-data-in' type='checkbox'><label for='data-f14f875d-1df2-4a43-904b-97ffbf9e9343' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;feat_pt_drift&#x27;, &#x27;feat_price_drift&#x27;, &#x27;feat_pt_high_drift&#x27;,&#x27;feat_pt_low_drift&#x27;, &#x27;feat_pt_median_drift&#x27;, &#x27;feat_coverage_drift&#x27;,&#x27;feat_total_return_ytd&#x27;], dtype=&#x27;&lt;U21&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e616b702-fa27-4db0-8be5-f716fdc0f6e0' class='xr-section-summary-in' type='checkbox' checked /><label for='section-e616b702-fa27-4db0-8be5-f716fdc0f6e0' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(13)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>log_last_price</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.406 0.8065 ... 0.3507 0.04879</div><input id='attrs-6996816e-24bd-4df6-ae7e-0c4ce6b3d67d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6996816e-24bd-4df6-ae7e-0c4ce6b3d67d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-01349165-b243-47cb-8a00-a154b7bada01' class='xr-var-data-in' type='checkbox'><label for='data-01349165-b243-47cb-8a00-a154b7bada01' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 5.40636427,  0.80647587,  4.19870458, ..., -1.81400508,0.35065687,  0.04879016], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>drift_features</span></div><div class='xr-var-dims'>(isin, drift_feature)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6513 0.1998 ... -0.2455 -0.3173</div><input id='attrs-453dc963-40fc-46fe-9ebb-0cd4014e029b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-453dc963-40fc-46fe-9ebb-0cd4014e029b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ff90134e-137b-4c1a-9681-f3e665e9ee8a' class='xr-var-data-in' type='checkbox'><label for='data-ff90134e-137b-4c1a-9681-f3e665e9ee8a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 6.51344363e-01,  1.99814418e-01,  1.05509829e+00, ...,4.58783001e-01, -1.33537723e-01,  3.37384527e-02],[ 1.08070822e+00,  1.59459656e+00,  6.89505404e-01, ...,1.23487726e+00, -1.95211955e-01, -1.07495794e-01],[ 5.24431914e+00,  4.56115373e+00,  6.80158439e+00, ...,4.40043714e+00,  1.59973114e-01,  9.68449387e-01],...,[-3.52752580e-01, -6.29244007e-01, -2.73738507e-01, ...,-3.58614587e-01,  5.74339579e-03, -4.65431714e-01],[-5.51121746e-01, -1.12324695e+00, -7.78865083e-01, ...,-6.00618031e-01, -8.48316846e-01, -8.74336920e-01],[-4.37020636e-01, -5.62621018e-01, -4.20252438e-01, ...,-4.40978181e-01, -2.45450793e-01, -3.17291951e-01]],shape=(6277, 7))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>feat_pt_range_norm</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.078 1.021 1.065 ... 0.3051 0.4242</div><input id='attrs-bf66cf76-177f-4b87-a9c8-9a34a7ad16ba' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bf66cf76-177f-4b87-a9c8-9a34a7ad16ba' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c8f720bd-0c0e-4059-ad40-29026354a26d' class='xr-var-data-in' type='checkbox'><label for='data-c8f720bd-0c0e-4059-ad40-29026354a26d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1.0781297 , 1.02062442, 1.06457192, ..., 0.19444444, 0.30508733,0.42424242], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>feat_pt_noise_cv</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.2402 0.6035 ... 0.3828 0.3333</div><input id='attrs-cbe3390e-44ac-499c-b2b9-b303707fdfb4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cbe3390e-44ac-499c-b2b9-b303707fdfb4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ea4c8ee9-790d-43d0-b150-3f71309afb39' class='xr-var-data-in' type='checkbox'><label for='data-ea4c8ee9-790d-43d0-b150-3f71309afb39' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.24024325, 0.60348214, 0.35551952, ..., 0.12883436, 0.3828169 ,0.33333333], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>feat_vol_mean</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>37.05 62.24 106.1 ... 73.12 47.3</div><input id='attrs-198e25dc-b589-499d-a539-bf5c53ee8557' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-198e25dc-b589-499d-a539-bf5c53ee8557' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c4800f1a-f098-4c7f-b241-f332be4fe25d' class='xr-var-data-in' type='checkbox'><label for='data-c4800f1a-f098-4c7f-b241-f332be4fe25d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 37.0525,  62.2425, 106.12  , ...,  35.4975,  73.12  ,  47.295 ],shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sqrt_n_analysts</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>7.616 2.449 3.742 ... 1.732 1.414</div><input id='attrs-b7826c5a-4182-4e7e-96e5-59275f72759f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b7826c5a-4182-4e7e-96e5-59275f72759f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c8240f43-3f20-4b5e-9f0e-ed7fbf1adb53' class='xr-var-data-in' type='checkbox'><label for='data-c8240f43-3f20-4b5e-9f0e-ed7fbf1adb53' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([7.61577311, 2.44948974, 3.74165739, ..., 1.41421356, 1.73205081,1.41421356], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>region_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>4 4 1 4 4 2 4 2 ... 3 0 3 3 3 0 3 3</div><input id='attrs-cc1b4f44-b02f-4295-b969-509ffd6515ec' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cc1b4f44-b02f-4295-b969-509ffd6515ec' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-89733e20-0967-44f8-995a-c7a8860b2d0d' class='xr-var-data-in' type='checkbox'><label for='data-89733e20-0967-44f8-995a-c7a8860b2d0d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([4, 4, 1, ..., 0, 3, 3], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>exchange_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>56 74 56 74 56 ... 10 10 44 10 10</div><input id='attrs-fc95f6b7-7187-47cd-b95a-7e73315c331b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fc95f6b7-7187-47cd-b95a-7e73315c331b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-713d14f8-7204-4051-a166-d711d4a441dd' class='xr-var-data-in' type='checkbox'><label for='data-713d14f8-7204-4051-a166-d711d4a441dd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([56, 74, 56, ..., 44, 10, 10], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>unit_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>47 6 47 6 47 14 47 ... 5 5 5 33 5 5</div><input id='attrs-b9ffa1fa-ef86-4856-81f3-721fab24fa7b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b9ffa1fa-ef86-4856-81f3-721fab24fa7b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-958aa7f0-7962-4dd6-8f90-a1ad97e001d8' class='xr-var-data-in' type='checkbox'><label for='data-958aa7f0-7962-4dd6-8f90-a1ad97e001d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([47,  6, 47, ..., 33,  5,  5], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>style_class_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>1 0 0 3 0 0 0 0 ... 0 3 3 0 3 3 3 0</div><input id='attrs-b7eb7e08-f495-4916-b5bf-23e2911002a2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b7eb7e08-f495-4916-b5bf-23e2911002a2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ae76b29b-fa1a-4adc-b19a-1e2fec09a917' class='xr-var-data-in' type='checkbox'><label for='data-ae76b29b-fa1a-4adc-b19a-1e2fec09a917' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1, 0, 0, ..., 3, 3, 0], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>size_class_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0 2 1 1 0 1 0 1 ... 2 2 2 2 2 2 2 2</div><input id='attrs-e6769cc9-d367-4965-9fac-c7506471f7c6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e6769cc9-d367-4965-9fac-c7506471f7c6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3546f0de-efdb-422e-b01f-7365b1a28ec7' class='xr-var-data-in' type='checkbox'><label for='data-3546f0de-efdb-422e-b01f-7365b1a28ec7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 2, 1, ..., 2, 2, 2], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sector_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>6 7 6 3 6 6 0 0 ... 1 5 2 1 4 5 1 0</div><input id='attrs-3da1531c-355c-4062-8f4d-494a8fc1bc7b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3da1531c-355c-4062-8f4d-494a8fc1bc7b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a983d1c1-2c6a-4e87-9e65-d2f377305bb1' class='xr-var-data-in' type='checkbox'><label for='data-a983d1c1-2c6a-4e87-9e65-d2f377305bb1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([6, 7, 6, ..., 5, 1, 0], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>industry_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>49 41 50 43 52 32 ... 29 27 7 51 35</div><input id='attrs-51699199-27fa-47aa-a509-1c01c56184bb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-51699199-27fa-47aa-a509-1c01c56184bb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2972347f-1d0d-4815-b6b2-d3c25b3b8e87' class='xr-var-data-in' type='checkbox'><label for='data-2972347f-1d0d-4815-b6b2-d3c25b3b8e87' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([49, 41, 50, ...,  7, 51, 35], shape=(6277,), dtype=int32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a05a7f76-4f25-4f62-8412-cb0d643b8cf9' class='xr-section-summary-in' type='checkbox' checked /><label for='section-a05a7f76-4f25-4f62-8412-cb0d643b8cf9' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.396870+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[]</dd></dl></div></li></ul></div></div></div></li></ul></div></div>



## 7. Posterior Inference (NUTS)


```python
# Sampler dispatch - nutpie -> numpyro -> pymc fallback. The cross-sectional
# log-space state-space graph is supported by all three NUTS backends; we try
# them in priority order and fall back on the first installed one that succeeds.
sample_kwargs = dict(
    draws=1000, tune=1000, chains=4, cores=2,
    target_accept=0.95, random_seed=RANDOM_SEED,
    progressbar=True, return_inferencedata=True,
    idata_kwargs={"log_likelihood": False},
)

import importlib.util as _ilu

_candidate_samplers = []
if _ilu.find_spec("nutpie") is not None:
    _candidate_samplers.append("nutpie")
if _ilu.find_spec("numpyro") is not None:
    _candidate_samplers.append("numpyro")
_candidate_samplers.append("pymc")  # always available - pure-Python NUTS
print(f"Available NUTS samplers (in priority order): {_candidate_samplers}")

sampling_errors = []
idata = None
for _sampler in _candidate_samplers:
    try:
        with kalman_pt_model:
            idata = pm.sample(nuts_sampler=_sampler, **sample_kwargs)
        print(f"Sampled successfully with nuts_sampler={_sampler!r}.")
        break
    except Exception as e:  # pragma: no cover - environment-dependent fallback
        sampling_errors.append((_sampler, repr(e)))
        print(f"nuts_sampler={_sampler!r} failed: {e!r}")

if idata is None:
    raise RuntimeError(
        "All NUTS samplers failed:\n"
        + "\n".join(f"  - {s}: {err}" for s, err in sampling_errors)
    )

# Merge prior groups into the posterior idata for one-object downstream access.
from probabilistic_ml_model._pymc_arviz_compat import extend_datatree

idata = extend_datatree(idata, prior_idata)
idata
```

    Available NUTS samplers (in priority order): ['nutpie', 'numpyro', 'pymc']
    

    NUTS[nutpie]: [mu_global, beta, sigma_region, z_region, sigma_exchange, z_exchange, sigma_unit, z_unit, sigma_style_class, z_style_class, sigma_size_class, z_size_class, sigma_sector, z_sector, sigma_industry, z_industry, sigma_state, z_state, sigma_obs_base, nu]
    


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampled successfully with nuts_sampler='nutpie'.
    




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in notebooks */

:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
  line-height: 1.6;
  padding-bottom: 4px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
}

.xr-header {
  border-bottom: solid 1px var(--xr-border-color);
  margin-bottom: 4px;
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-obj-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type,
.xr-group-box-contents > label {
  color: var(--xr-font-color2);
  display: block;
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
  margin-block-start: 0;
  margin-block-end: 0;
}

.xr-section-item {
  display: contents;
}

.xr-section-item > input,
.xr-group-box-contents > input,
.xr-array-wrap > input {
  display: block;
  opacity: 0;
  height: 0;
  margin: 0;
}

.xr-section-item > input + label,
.xr-var-item > input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item > input:enabled + label,
.xr-var-item > input:enabled + label,
.xr-array-wrap > input:enabled + label,
.xr-group-box-contents > input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item > input:focus-visible + label,
.xr-var-item > input:focus-visible + label,
.xr-array-wrap > input:focus-visible + label,
.xr-group-box-contents > input:focus-visible + label {
  outline: auto;
}

.xr-section-item > input:enabled + label:hover,
.xr-var-item > input:enabled + label:hover,
.xr-array-wrap > input:enabled + label:hover,
.xr-group-box-contents > input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
  white-space: nowrap;
}

.xr-section-summary > em {
  font-weight: normal;
}

.xr-span-grid {
  grid-column-end: -1;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.3em;
}

.xr-group-box-contents > input:checked + label > span {
  display: inline-block;
  padding-left: 0.6em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details,
.xr-group-box-contents > label {
  padding-top: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  grid-column: 1 / -1;
  margin-top: 4px;
  margin-bottom: 5px;
}

.xr-section-summary-in ~ .xr-section-details {
  display: none;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-children {
  display: inline-grid;
  grid-template-columns: 100%;
  grid-column: 1 / -1;
  padding-top: 4px;
}

.xr-group-box {
  display: inline-grid;
  grid-template-columns: 0px 30px auto;
}

.xr-group-box-vline {
  grid-column-start: 1;
  border-right: 0.2em solid;
  border-color: var(--xr-border-color);
  width: 0px;
}

.xr-group-box-hline {
  grid-column-start: 2;
  grid-row-start: 1;
  height: 1em;
  width: 26px;
  border-bottom: 0.2em solid;
  border-color: var(--xr-border-color);
}

.xr-group-box-contents {
  grid-column-start: 3;
  padding-bottom: 4px;
}

.xr-group-box-contents > label::before {
  content: "📂";
  padding-right: 0.3em;
}

.xr-group-box-contents > input:checked + label::before {
  content: "📁";
}

.xr-group-box-contents > input:checked + label {
  padding-bottom: 0px;
}

.xr-group-box-contents > input:checked ~ .xr-sections {
  display: none;
}

.xr-group-box-contents > input + label > span {
  display: none;
}

.xr-group-box-ellipsis {
  font-size: 1.4em;
  font-weight: 900;
  color: var(--xr-font-color2);
  letter-spacing: 0.15em;
  cursor: default;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}

.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}

.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataTree&gt;
Group: /
├── Group: /posterior
│       Dimensions:             (chain: 4, draw: 1000, drift_feature: 7, region: 5,
│                                exchange: 82, unit: 50, style_class: 4, size_class: 4,
│                                sector: 9, industry: 59, isin: 6277)
│       Coordinates:
│         * chain               (chain) int64 32B 0 1 2 3
│         * draw                (draw) int64 8kB 0 1 2 3 4 5 ... 994 995 996 997 998 999
│         * drift_feature       (drift_feature) object 56B &#x27;feat_pt_drift&#x27; ... &#x27;feat_...
│         * region              (region) object 40B &#x27;Africa / Middle East&#x27; ... &#x27;Unite...
│         * exchange            (exchange) object 656B &#x27;ADX&#x27; &#x27;AIM&#x27; ... &#x27;XTRA&#x27; &#x27;ZGSE&#x27;
│         * unit                (unit) object 400B &#x27;AED&#x27; &#x27;ARS&#x27; &#x27;AUD&#x27; ... &#x27;VND&#x27; &#x27;ZAR&#x27;
│         * style_class         (style_class) object 32B &#x27;Core&#x27; &#x27;Growth&#x27; ... &#x27;Value&#x27;
│         * size_class          (size_class) object 32B &#x27;Large Cap&#x27; ... &#x27;Unknown&#x27;
│         * sector              (sector) object 72B &#x27;Communication Services&#x27; ... &#x27;Uti...
│         * industry            (industry) object 472B &#x27;Aerospace and Defense&#x27; ... &#x27;W...
│         * isin                (isin) object 50kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│       Data variables: (12/32)
│           mu_global           (chain, draw) float64 32kB 0.06948 0.072 ... 0.08794
│           beta                (chain, draw, drift_feature) float64 224kB 0.02033 .....
│           z_region            (chain, draw, region) float64 160kB 0.06742 ... -0.2336
│           z_exchange          (chain, draw, exchange) float64 3MB 1.455 ... -1.236
│           z_unit              (chain, draw, unit) float64 2MB 0.6614 ... -0.2735
│           z_style_class       (chain, draw, style_class) float64 128kB 0.1153 ... -...
│           ...                  ...
│           industry_effect     (chain, draw, industry) float64 2MB 0.02365 ... -0.03537
│           log_uplift          (chain, draw, isin) float64 201MB 0.3011 ... 0.3069
│           log_state           (chain, draw, isin) float64 201MB 5.707 ... 0.3557
│           sigma_obs           (chain, draw, isin) float64 201MB 0.03009 ... 0.2062
│           expected_pt         (chain, draw, isin) float64 201MB 301.1 2.701 ... 1.427
│           expected_upside     (chain, draw, isin) float64 201MB 0.3513 ... 0.3592
│       Attributes:
│           created_at:                 2026-06-03T12:19:20.948093+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           sample_dims:                [&#x27;chain&#x27;, &#x27;draw&#x27;]
│           inference_library:          nutpie
│           inference_library_version:  0.16.10
│           sampling_time:              104.98427700996399
│           tuning_steps:               1000
├── Group: /sample_stats
│       Dimensions:                   (chain: 4, draw: 1000)
│       Coordinates:
│         * chain                     (chain) int64 32B 0 1 2 3
│         * draw                      (draw) int64 8kB 0 1 2 3 4 ... 995 996 997 998 999
│       Data variables: (12/20)
│           depth                     (chain, draw) uint64 32kB 6 6 6 6 6 ... 6 6 6 7 6
│           maxdepth_reached          (chain, draw) bool 4kB False False ... False False
│           step_size                 (chain, draw) float64 32kB 0.05878 ... 0.04879
│           transformation_update_id  (chain, draw) int64 32kB 0 0 0 0 0 0 ... 0 0 0 0 0
│           step_size_bar             (chain, draw) float64 32kB 0.05685 ... 0.05307
│           mean_tree_accept          (chain, draw) float64 32kB 0.998 0.9957 ... 0.955
│           ...                        ...
│           fisher_distance           (chain, draw) float64 32kB 1.834e+03 ... 1.672e+03
│           transformation_index      (chain, draw) int64 32kB 848 848 848 ... 848 848
│           diverging                 (chain, draw) bool 4kB False False ... False False
│           divergence_draw           (chain, draw) uint64 32kB 0 0 0 0 0 ... 0 0 0 0 0
│           divergence_message        (chain, draw) object 32kB None None ... None None
│           divergence_energy_error   (chain, draw) float64 32kB nan nan nan ... nan nan
│       Attributes:
│           created_at:                  2026-06-03T12:19:20.645879+00:00
│           creation_library:            ArviZ
│           creation_library_version:    1.1.0
│           creation_library_language:   Python
│           sample_dims:                 [&#x27;chain&#x27;, &#x27;draw&#x27;]
│           inference_library:           nutpie
│           inference_library_version:   0.16.10
│           inference_library_settings:  {&quot;sampler&quot;: &quot;nuts&quot;, &quot;adaptation&quot;: &quot;diag&quot;, &quot;s...
├── Group: /constant_data
│       Dimensions:             (isin: 6277, drift_feature: 7)
│       Coordinates:
│         * isin                (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│         * drift_feature       (drift_feature) &lt;U21 588B &#x27;feat_pt_drift&#x27; ... &#x27;feat_t...
│       Data variables: (12/13)
│           log_last_price      (isin) float64 50kB 5.406 0.8065 ... 0.3507 0.04879
│           drift_features      (isin, drift_feature) float64 352kB 0.6513 ... -0.3173
│           feat_pt_range_norm  (isin) float64 50kB 1.078 1.021 1.065 ... 0.3051 0.4242
│           feat_pt_noise_cv    (isin) float64 50kB 0.2402 0.6035 ... 0.3828 0.3333
│           feat_vol_mean       (isin) float64 50kB 37.05 62.24 106.1 ... 73.12 47.3
│           sqrt_n_analysts     (isin) float64 50kB 7.616 2.449 3.742 ... 1.732 1.414
│           ...                  ...
│           exchange_idx        (isin) int32 25kB 56 74 56 74 56 80 ... 10 10 44 10 10
│           unit_idx            (isin) int32 25kB 47 6 47 6 47 14 47 ... 33 5 5 5 33 5 5
│           style_class_idx     (isin) int32 25kB 1 0 0 3 0 0 0 0 0 ... 0 3 3 0 3 3 3 0
│           size_class_idx      (isin) int32 25kB 0 2 1 1 0 1 0 1 2 ... 2 2 2 2 2 2 2 2
│           sector_idx          (isin) int32 25kB 6 7 6 3 6 6 0 0 7 ... 1 5 2 1 4 5 1 0
│           industry_idx        (isin) int32 25kB 49 41 50 43 52 32 ... 23 29 27 7 51 35
│       Attributes:
│           created_at:                 2026-06-03T12:17:12.396870+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           inference_library:          pymc
│           inference_library_version:  6.0.1
│           sample_dims:                []
├── Group: /observed_data
│       Dimensions:     (isin: 6277)
│       Coordinates:
│         * isin        (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│       Data variables:
│           log_pt_obs  (isin) float64 50kB 5.693 1.481 4.38 ... -1.532 1.369 0.5008
│       Attributes:
│           created_at:                 2026-06-03T12:17:12.384284+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           inference_library:          pymc
│           inference_library_version:  6.0.1
│           sample_dims:                []
├── Group: /prior
│       Dimensions:             (chain: 1, draw: 1500, isin: 6277, sector: 9, unit: 50,
│                                style_class: 4, drift_feature: 7, exchange: 82,
│                                size_class: 4, industry: 59, region: 5)
│       Coordinates:
│         * chain               (chain) int64 8B 0
│         * draw                (draw) int64 12kB 0 1 2 3 4 ... 1495 1496 1497 1498 1499
│         * isin                (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
│         * sector              (sector) &lt;U22 792B &#x27;Communication Services&#x27; ... &#x27;Util...
│         * unit                (unit) &lt;U3 600B &#x27;AED&#x27; &#x27;ARS&#x27; &#x27;AUD&#x27; ... &#x27;USD&#x27; &#x27;VND&#x27; &#x27;ZAR&#x27;
│         * style_class         (style_class) &lt;U7 112B &#x27;Core&#x27; &#x27;Growth&#x27; &#x27;Unknown&#x27; &#x27;Value&#x27;
│         * drift_feature       (drift_feature) &lt;U21 588B &#x27;feat_pt_drift&#x27; ... &#x27;feat_t...
│         * exchange            (exchange) &lt;U8 3kB &#x27;ADX&#x27; &#x27;AIM&#x27; &#x27;ASX&#x27; ... &#x27;XTRA&#x27; &#x27;ZGSE&#x27;
│         * size_class          (size_class) &lt;U9 144B &#x27;Large Cap&#x27; ... &#x27;Unknown&#x27;
│         * industry            (industry) &lt;U53 13kB &#x27;Aerospace and Defense&#x27; ... &#x27;Wir...
│         * region              (region) &lt;U27 540B &#x27;Africa / Middle East&#x27; ... &#x27;United...
│       Data variables: (12/24)
│           mu_global           (chain, draw) float64 12kB -0.4414 -0.3835 ... -0.1683
│           sigma_state         (chain, draw) float64 12kB 0.01626 0.06206 ... 0.01211
│           log_state           (chain, draw, isin) float64 75MB 4.827 ... -0.2693
│           sector_effect       (chain, draw, sector) float64 108kB 0.00126 ... 0.02049
│           sigma_region        (chain, draw) float64 12kB 0.1799 0.1776 ... 0.0553
│           sigma_unit          (chain, draw) float64 12kB 0.1254 0.06063 ... 0.2454
│           ...                  ...
│           log_uplift          (chain, draw, isin) float64 75MB -0.5798 ... -0.3181
│           nu                  (chain, draw) float64 12kB 36.22 20.02 ... 9.037 4.163
│           industry_effect     (chain, draw, industry) float64 708kB 0.1301 ... -0.0...
│           sigma_exchange      (chain, draw) float64 12kB 0.1919 0.02467 ... 0.1059
│           region_effect       (chain, draw, region) float64 60kB 0.06427 ... -0.03712
│           sigma_industry      (chain, draw) float64 12kB 0.1386 0.05796 ... 0.07764
│       Attributes:
│           created_at:                 2026-06-03T12:17:12.376301+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.1.0
│           creation_library_language:  Python
│           inference_library:          pymc
│           inference_library_version:  6.0.1
│           sample_dims:                [&#x27;chain&#x27;, &#x27;draw&#x27;]
└── Group: /prior_predictive
        Dimensions:     (chain: 1, draw: 1500, isin: 6277)
        Coordinates:
          * chain       (chain) int64 8B 0
          * draw        (draw) int64 12kB 0 1 2 3 4 5 ... 1494 1495 1496 1497 1498 1499
          * isin        (isin) &lt;U12 301kB &#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;
        Data variables:
            log_pt_obs  (chain, draw, isin) float64 75MB 5.067 -0.5607 ... 0.2002 1.486
        Attributes:
            created_at:                 2026-06-03T12:17:12.382459+00:00
            creation_library:           ArviZ
            creation_library_version:   1.1.0
            creation_library_language:  Python
            inference_library:          pymc
            inference_library_version:  6.0.1
            sample_dims:                [&#x27;chain&#x27;, &#x27;draw&#x27;]</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataTree</div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-children'><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-73264b01-7a2b-4876-a613-f52487397ae6' type='checkbox' checked /><label for='group-73264b01-7a2b-4876-a613-f52487397ae6' title='Expand/collapse group'>/posterior<span>(52)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-985e4425-140b-490c-a658-3ff4db6e8e1e' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-985e4425-140b-490c-a658-3ff4db6e8e1e' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>drift_feature</span>: 7</li><li><span class='xr-has-index'>region</span>: 5</li><li><span class='xr-has-index'>exchange</span>: 82</li><li><span class='xr-has-index'>unit</span>: 50</li><li><span class='xr-has-index'>style_class</span>: 4</li><li><span class='xr-has-index'>size_class</span>: 4</li><li><span class='xr-has-index'>sector</span>: 9</li><li><span class='xr-has-index'>industry</span>: 59</li><li><span class='xr-has-index'>isin</span>: 6277</li></ul></div></li><li class='xr-section-item'><input id='section-29f20dba-0b5d-42f9-a112-66aa6cbf7d1e' class='xr-section-summary-in' type='checkbox' checked /><label for='section-29f20dba-0b5d-42f9-a112-66aa6cbf7d1e' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-90dbd80b-3049-4c66-8d89-5f3dc69633f3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-90dbd80b-3049-4c66-8d89-5f3dc69633f3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e65cb902-5ea6-4b17-aa31-2ba9d63d6d11' class='xr-var-data-in' type='checkbox'><label for='data-e65cb902-5ea6-4b17-aa31-2ba9d63d6d11' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-183c29cf-80c2-4ab3-a7cf-d8427683280f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-183c29cf-80c2-4ab3-a7cf-d8427683280f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cdd9e07b-d3a4-4ef3-be21-254e15a8cd58' class='xr-var-data-in' type='checkbox'><label for='data-cdd9e07b-d3a4-4ef3-be21-254e15a8cd58' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>drift_feature</span></div><div class='xr-var-dims'>(drift_feature)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;feat_pt_drift&#x27; ... &#x27;feat_total_...</div><input id='attrs-64955ec3-c518-43c1-a31a-5ecdbdab0aaf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-64955ec3-c518-43c1-a31a-5ecdbdab0aaf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d30c0371-3ec9-4eb5-9389-b7b976212f4a' class='xr-var-data-in' type='checkbox'><label for='data-d30c0371-3ec9-4eb5-9389-b7b976212f4a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;feat_pt_drift&#x27;, &#x27;feat_price_drift&#x27;, &#x27;feat_pt_high_drift&#x27;,&#x27;feat_pt_low_drift&#x27;, &#x27;feat_pt_median_drift&#x27;, &#x27;feat_coverage_drift&#x27;,&#x27;feat_total_return_ytd&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>region</span></div><div class='xr-var-dims'>(region)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Africa / Middle East&#x27; ... &#x27;Unit...</div><input id='attrs-5caf5f91-4321-4269-9ae2-732bad6a592c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5caf5f91-4321-4269-9ae2-732bad6a592c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e4e219f8-cf68-4930-9747-5da5f2107d77' class='xr-var-data-in' type='checkbox'><label for='data-e4e219f8-cf68-4930-9747-5da5f2107d77' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Africa / Middle East&#x27;, &#x27;Asia / Pacific&#x27;, &#x27;Europe&#x27;,&#x27;Latin America and Caribbean&#x27;, &#x27;United States and Canada&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>exchange</span></div><div class='xr-var-dims'>(exchange)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;ADX&#x27; &#x27;AIM&#x27; &#x27;ASX&#x27; ... &#x27;XTRA&#x27; &#x27;ZGSE&#x27;</div><input id='attrs-5e29ee17-3caa-4a7c-8d0a-21a80329f5a8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5e29ee17-3caa-4a7c-8d0a-21a80329f5a8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b9221ae9-ce7c-44a0-92a8-e7d36a2727ee' class='xr-var-data-in' type='checkbox'><label for='data-b9221ae9-ce7c-44a0-92a8-e7d36a2727ee' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;ADX&#x27;, &#x27;AIM&#x27;, &#x27;ASX&#x27;, &#x27;ATSE&#x27;, &#x27;BASE&#x27;, &#x27;BAX&#x27;, &#x27;BELEX&#x27;, &#x27;BIT&#x27;, &#x27;BME&#x27;,&#x27;BMV&#x27;, &#x27;BOVESPA&#x27;, &#x27;BSE&#x27;, &#x27;BUL&#x27;, &#x27;BUSE&#x27;, &#x27;BVB&#x27;, &#x27;BVC&#x27;, &#x27;BVL&#x27;, &#x27;CASE&#x27;,&#x27;CBSE&#x27;, &#x27;CNSX&#x27;, &#x27;CPSE&#x27;, &#x27;DB&#x27;, &#x27;DFM&#x27;, &#x27;DSE&#x27;, &#x27;DSM&#x27;, &#x27;ENXTAM&#x27;, &#x27;ENXTBR&#x27;,&#x27;ENXTLS&#x27;, &#x27;ENXTPA&#x27;, &#x27;GHSE&#x27;, &#x27;HLSE&#x27;, &#x27;HOSE&#x27;, &#x27;IBSE&#x27;, &#x27;IDX&#x27;, &#x27;ISE&#x27;, &#x27;JSE&#x27;,&#x27;KAS&#x27;, &#x27;KASE&#x27;, &#x27;KLSE&#x27;, &#x27;KOSDAQ&#x27;, &#x27;KOSE&#x27;, &#x27;KWSE&#x27;, &#x27;LJSE&#x27;, &#x27;LSE&#x27;, &#x27;MSM&#x27;,&#x27;MUN&#x27;, &#x27;NASE&#x27;, &#x27;NGM&#x27;, &#x27;NGSE&#x27;, &#x27;NSEI&#x27;, &#x27;NSEL&#x27;, &#x27;NYSE&#x27;, &#x27;NYSEAM&#x27;, &#x27;NZSE&#x27;,&#x27;NasdaqCM&#x27;, &#x27;NasdaqGM&#x27;, &#x27;NasdaqGS&#x27;, &#x27;OB&#x27;, &#x27;OM&#x27;, &#x27;OTCPK&#x27;, &#x27;PSE&#x27;, &#x27;SASE&#x27;,&#x27;SEHK&#x27;, &#x27;SEP&#x27;, &#x27;SET&#x27;, &#x27;SGX&#x27;, &#x27;SHSE&#x27;, &#x27;SNSE&#x27;, &#x27;SWX&#x27;, &#x27;SZSE&#x27;, &#x27;TASE&#x27;,&#x27;TLSE&#x27;, &#x27;TPEX&#x27;, &#x27;TSE&#x27;, &#x27;TSX&#x27;, &#x27;TSXV&#x27;, &#x27;TWSE&#x27;, &#x27;WBAG&#x27;, &#x27;WSE&#x27;, &#x27;XSAT&#x27;,&#x27;XTRA&#x27;, &#x27;ZGSE&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>unit</span></div><div class='xr-var-dims'>(unit)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;AED&#x27; &#x27;ARS&#x27; &#x27;AUD&#x27; ... &#x27;VND&#x27; &#x27;ZAR&#x27;</div><input id='attrs-a9069acf-239b-4a12-8039-3abff3994fd2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a9069acf-239b-4a12-8039-3abff3994fd2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5c831e3d-6033-4ac2-9133-8a268193a400' class='xr-var-data-in' type='checkbox'><label for='data-5c831e3d-6033-4ac2-9133-8a268193a400' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;AED&#x27;, &#x27;ARS&#x27;, &#x27;AUD&#x27;, &#x27;BDT&#x27;, &#x27;BHD&#x27;, &#x27;BRL&#x27;, &#x27;CAD&#x27;, &#x27;CHF&#x27;, &#x27;CLP&#x27;, &#x27;CNY&#x27;,&#x27;COP&#x27;, &#x27;CZK&#x27;, &#x27;DKK&#x27;, &#x27;EGP&#x27;, &#x27;EUR&#x27;, &#x27;GBP&#x27;, &#x27;GHS&#x27;, &#x27;HKD&#x27;, &#x27;HUF&#x27;, &#x27;IDR&#x27;,&#x27;ILS&#x27;, &#x27;INR&#x27;, &#x27;JPY&#x27;, &#x27;KES&#x27;, &#x27;KRW&#x27;, &#x27;KWD&#x27;, &#x27;KZT&#x27;, &#x27;MAD&#x27;, &#x27;MXN&#x27;, &#x27;MYR&#x27;,&#x27;NGN&#x27;, &#x27;NOK&#x27;, &#x27;NZD&#x27;, &#x27;OMR&#x27;, &#x27;PEN&#x27;, &#x27;PHP&#x27;, &#x27;PKR&#x27;, &#x27;PLN&#x27;, &#x27;QAR&#x27;, &#x27;RON&#x27;,&#x27;RSD&#x27;, &#x27;SAR&#x27;, &#x27;SEK&#x27;, &#x27;SGD&#x27;, &#x27;THB&#x27;, &#x27;TRY&#x27;, &#x27;TWD&#x27;, &#x27;USD&#x27;, &#x27;VND&#x27;, &#x27;ZAR&#x27;],dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>style_class</span></div><div class='xr-var-dims'>(style_class)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Core&#x27; &#x27;Growth&#x27; &#x27;Unknown&#x27; &#x27;Value&#x27;</div><input id='attrs-a29636fa-3877-455d-ba33-9c0937582920' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a29636fa-3877-455d-ba33-9c0937582920' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-40ee81e3-2186-441b-a209-27eedacbf723' class='xr-var-data-in' type='checkbox'><label for='data-40ee81e3-2186-441b-a209-27eedacbf723' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Core&#x27;, &#x27;Growth&#x27;, &#x27;Unknown&#x27;, &#x27;Value&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>size_class</span></div><div class='xr-var-dims'>(size_class)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Large Cap&#x27; &#x27;Mid Cap&#x27; ... &#x27;Unknown&#x27;</div><input id='attrs-d3865440-8d2d-41ba-8895-4b0a59b552d4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d3865440-8d2d-41ba-8895-4b0a59b552d4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-956ab594-816e-4f76-b997-963eb0174387' class='xr-var-data-in' type='checkbox'><label for='data-956ab594-816e-4f76-b997-963eb0174387' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Large Cap&#x27;, &#x27;Mid Cap&#x27;, &#x27;Small Cap&#x27;, &#x27;Unknown&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sector</span></div><div class='xr-var-dims'>(sector)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Communication Services&#x27; ... &#x27;Ut...</div><input id='attrs-5eaf08f2-0c69-4105-9939-4ceb89ed5634' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5eaf08f2-0c69-4105-9939-4ceb89ed5634' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ecd0a57d-b4f7-45b2-9c1d-ba0f6ef8d9c1' class='xr-var-data-in' type='checkbox'><label for='data-ecd0a57d-b4f7-45b2-9c1d-ba0f6ef8d9c1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Communication Services&#x27;, &#x27;Consumer Discretionary&#x27;, &#x27;Consumer Staples&#x27;,&#x27;Energy&#x27;, &#x27;Health Care&#x27;, &#x27;Industrials&#x27;, &#x27;Information Technology&#x27;,&#x27;Materials&#x27;, &#x27;Utilities&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>industry</span></div><div class='xr-var-dims'>(industry)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Aerospace and Defense&#x27; ... &#x27;Wir...</div><input id='attrs-9a74fd0b-9ad2-438e-8e0d-d5d3431db0bc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9a74fd0b-9ad2-438e-8e0d-d5d3431db0bc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fb8a0be0-6a93-4752-932e-7a5ec1e96331' class='xr-var-data-in' type='checkbox'><label for='data-fb8a0be0-6a93-4752-932e-7a5ec1e96331' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Aerospace and Defense&#x27;, &#x27;Air Freight and Logistics&#x27;,&#x27;Automobile Components&#x27;, &#x27;Automobiles&#x27;, &#x27;Beverages&#x27;, &#x27;Biotechnology&#x27;,&#x27;Broadline Retail&#x27;, &#x27;Building Products&#x27;, &#x27;Chemicals&#x27;,&#x27;Commercial Services and Supplies&#x27;, &#x27;Communications Equipment&#x27;,&#x27;Construction Materials&#x27;, &#x27;Construction and Engineering&#x27;,&#x27;Consumer Staples Distribution and Retail&#x27;, &#x27;Containers and Packaging&#x27;,&#x27;Distributors&#x27;, &#x27;Diversified Consumer Services&#x27;,&#x27;Diversified Telecommunication Services&#x27;, &#x27;Electric Utilities&#x27;,&#x27;Electrical Equipment&#x27;,&#x27;Electronic Equipment Instruments and Components&#x27;,&#x27;Energy Equipment and Services&#x27;, &#x27;Entertainment&#x27;, &#x27;Food Products&#x27;,&#x27;Gas Utilities&#x27;, &#x27;Ground Transportation&#x27;,&#x27;Health Care Equipment and Supplies&#x27;,&#x27;Health Care Providers and Services&#x27;, &#x27;Health Care Technology&#x27;,&#x27;Hotels Restaurants and Leisure&#x27;, &#x27;Household Durables&#x27;,&#x27;Household Products&#x27;, &#x27;IT Services&#x27;,&#x27;Independent Power and Renewable Electricity Producers&#x27;,&#x27;Industrial Conglomerates&#x27;, &#x27;Interactive Media and Services&#x27;,&#x27;Leisure Products&#x27;, &#x27;Life Sciences Tools and Services&#x27;, &#x27;Machinery&#x27;,&#x27;Marine Transportation&#x27;, &#x27;Media&#x27;, &#x27;Metals and Mining&#x27;,&#x27;Multi-Utilities&#x27;, &#x27;Oil Gas and Consumable Fuels&#x27;,&#x27;Paper and Forest Products&#x27;, &#x27;Passenger Airlines&#x27;,&#x27;Personal Care Products&#x27;, &#x27;Pharmaceuticals&#x27;, &#x27;Professional Services&#x27;,&#x27;Semiconductors and Semiconductor Equipment&#x27;, &#x27;Software&#x27;,&#x27;Specialty Retail&#x27;, &#x27;Technology Hardware Storage and Peripherals&#x27;,&#x27;Textiles Apparel and Luxury Goods&#x27;, &#x27;Tobacco&#x27;,&#x27;Trading Companies and Distributors&#x27;, &#x27;Transportation Infrastructure&#x27;,&#x27;Water Utilities&#x27;, &#x27;Wireless Telecommunication Services&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-f5bb04a3-1c4d-43c4-8643-f4f7f6bb3dbd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f5bb04a3-1c4d-43c4-8643-f4f7f6bb3dbd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-565cd9c0-ff11-4e6c-a0ec-b03c25d7f876' class='xr-var-data-in' type='checkbox'><label for='data-565cd9c0-ff11-4e6c-a0ec-b03c25d7f876' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=object)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a5dc04d6-163e-45f0-982e-332e089d5d24' class='xr-section-summary-in' type='checkbox' /><label for='section-a5dc04d6-163e-45f0-982e-332e089d5d24' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(32)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>mu_global</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06948 0.072 ... 0.09039 0.08794</div><input id='attrs-020f2809-2474-4f75-9e8d-d0b4f8221942' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-020f2809-2474-4f75-9e8d-d0b4f8221942' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-76c254aa-a6d2-4a20-b258-0e2fb4ded66f' class='xr-var-data-in' type='checkbox'><label for='data-76c254aa-a6d2-4a20-b258-0e2fb4ded66f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.069479  , 0.07199778, 0.05465719, ..., 0.1689037 , 0.16822121,0.19200641],[0.14444657, 0.15123398, 0.14087782, ..., 0.21996553, 0.18563292,0.19478554],[0.19031414, 0.17148346, 0.1698319 , ..., 0.12529967, 0.14009127,0.13151006],[0.20563131, 0.23524281, 0.25929374, ..., 0.09701678, 0.09039147,0.08794331]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta</span></div><div class='xr-var-dims'>(chain, draw, drift_feature)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.02033 -0.168 ... 0.01252 -0.1646</div><input id='attrs-da8339d5-cf22-41f8-8a8b-c5efecbd5132' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-da8339d5-cf22-41f8-8a8b-c5efecbd5132' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e650f347-fb29-44eb-be54-59315559d484' class='xr-var-data-in' type='checkbox'><label for='data-e650f347-fb29-44eb-be54-59315559d484' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.02033006, -0.16801254,  0.0657467 , ...,  0.07030612,0.00863164, -0.16941564],[ 0.01781173, -0.16641427,  0.06324866, ...,  0.07302178,0.00942221, -0.16642444],[ 0.02155742, -0.16909146,  0.06545388, ...,  0.06759712,0.00990841, -0.17179634],...,[ 0.07112942, -0.17544582,  0.05278185, ...,  0.03214017,0.01211882, -0.15875819],[ 0.07185344, -0.17172472,  0.04966293, ...,  0.02531646,0.01235011, -0.15545517],[ 0.08057486, -0.16378195,  0.04140584, ...,  0.04122876,0.01223223, -0.15970962]],[[ 0.02870367, -0.18506842,  0.06061198, ...,  0.07607793,0.0110215 , -0.15499831],[ 0.01176797, -0.17537549,  0.07127419, ...,  0.07528297,0.0091447 , -0.15480841],[ 0.01236674, -0.17656188,  0.05183434, ...,  0.0962706 ,0.01050377, -0.15159116],...[-0.00087972, -0.15789996,  0.06106226, ...,  0.0936424 ,0.00727216, -0.15961049],[-0.00173338, -0.16165612,  0.06585741, ...,  0.08732557,0.00967102, -0.15860568],[ 0.00716302, -0.17472095,  0.0610387 , ...,  0.08771417,0.00694673, -0.15983822]],[[ 0.03060594, -0.19095235,  0.05326381, ...,  0.07515061,0.00935169, -0.14794577],[ 0.01810172, -0.16558839,  0.05774433, ...,  0.06930219,0.01763641, -0.15629142],[ 0.01544928, -0.17165485,  0.05294343, ...,  0.07542186,0.013292  , -0.16145594],...,[ 0.04217449, -0.17300238,  0.05116072, ...,  0.06608549,0.01058275, -0.16561024],[ 0.01278641, -0.15707658,  0.04973059, ...,  0.07676573,0.01163411, -0.16617407],[ 0.0114218 , -0.15504063,  0.04980228, ...,  0.08492556,0.01252499, -0.16458792]]], shape=(4, 1000, 7))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_region</span></div><div class='xr-var-dims'>(chain, draw, region)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06742 1.514 ... -0.427 -0.2336</div><input id='attrs-f66718a1-a692-4d27-b003-14ed124f714e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f66718a1-a692-4d27-b003-14ed124f714e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8bcf7b15-e393-48f7-8bd0-62a40ff8a276' class='xr-var-data-in' type='checkbox'><label for='data-8bcf7b15-e393-48f7-8bd0-62a40ff8a276' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.06742159,  1.51418523,  0.16676276,  0.77742987,0.21548684],[ 0.21418841,  1.55235526,  0.11896594,  0.68699803,0.19970799],[ 0.37187751,  1.58783742,  0.26649863,  0.69531971,0.22439063],...,[ 0.3109601 ,  2.08368657, -0.75394386, -0.50750188,-0.84579928],[ 0.30926631,  2.13173555, -0.73108401, -0.56637622,-0.80265794],[-0.0405119 ,  1.83448229, -0.69596295, -0.61202292,-0.95925896]],[[-0.63759625,  0.5212765 , -0.60549845, -0.26184906,-1.0603654 ],[-0.35184337,  1.02883132, -0.94871846, -0.61928722,-1.24053335],[-0.3278554 ,  1.02546955, -1.05718084, -0.37901743,-1.56416596],...[-0.61288646,  0.74983012, -0.76482692, -0.34884301,-1.03394984],[ 0.00441625,  0.81812537, -0.80651458, -0.18015741,-0.9435906 ],[-0.33355756,  0.76581536, -0.69282396, -0.5803708 ,-1.2013434 ]],[[-0.23414659,  2.0366034 , -0.24375846,  0.29226561,-1.33330734],[-0.15597947,  1.30176647, -0.59932118, -0.05582816,-1.06644047],[-0.2504562 ,  1.26903289, -0.44739469, -0.06224899,-1.42734035],...,[ 0.16685846,  0.94364463,  0.26184152,  0.23413976,-0.52908046],[ 0.33546956,  0.97328681,  0.35781693,  0.20080628,-0.36605163],[ 0.52927478,  1.11355201,  0.2659462 , -0.42703384,-0.2335994 ]]], shape=(4, 1000, 5))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_exchange</span></div><div class='xr-var-dims'>(chain, draw, exchange)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.455 1.177 ... 1.018 -1.236</div><input id='attrs-74033a8d-bda3-4008-96d1-b34fe168a987' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-74033a8d-bda3-4008-96d1-b34fe168a987' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cd56f81f-ab08-46a5-b107-d4ee80ed6ff6' class='xr-var-data-in' type='checkbox'><label for='data-cd56f81f-ab08-46a5-b107-d4ee80ed6ff6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 1.45517981,  1.17749075, -0.81763325, ..., -0.08712751,1.11910717, -0.59917536],[ 1.00933485,  2.01740555, -0.91165291, ..., -1.15116322,1.25086033, -0.5727178 ],[ 0.76160598,  1.78287925, -0.66164349, ...,  0.74844429,1.02054541, -1.25045512],...,[ 0.22957115,  1.3989645 ,  0.06917127, ..., -0.20774469,1.38484547, -1.68284855],[ 0.26112562,  1.20163063,  0.16415686, ...,  0.31026985,1.67881934, -0.97569472],[ 1.08902271,  1.43579119, -0.76888288, ..., -0.49312713,1.30379071, -1.18402613]],[[ 0.25222482,  0.79141778, -1.07458146, ...,  1.08528649,0.59868394, -1.61125379],[-0.32940767,  0.71886161, -0.72650419, ..., -0.74733548,0.67810992, -0.68510024],[ 1.03326296,  1.22384273, -0.63509909, ...,  1.68900148,0.50563852, -1.34357557],...[ 1.26480291,  1.154009  , -0.1845262 , ..., -0.95603069,1.01413063, -0.90763154],[ 0.39978082,  0.88190499, -1.03613033, ..., -0.05179031,1.12620185, -1.14603299],[ 0.15046822,  1.26722502, -1.37462412, ..., -1.16597486,1.47585529, -1.32543775]],[[ 0.4408822 ,  1.71112307, -2.96910444, ...,  1.32768228,1.5986271 , -0.53949598],[ 0.4021665 ,  1.76924411, -0.99103089, ..., -1.55196337,1.55881806, -1.01048502],[-0.35435849,  1.4125569 , -1.16346896, ...,  0.52155427,1.64905655, -0.42318523],...,[ 1.19068088,  1.57297508, -0.30159688, ..., -0.29501432,1.54542558, -0.97269305],[ 0.77067747,  0.31590801, -0.46033313, ...,  0.4472245 ,1.05854632, -0.60015933],[ 0.76246372,  0.86307007, -0.53340874, ..., -0.11760379,1.0178818 , -1.23637139]]], shape=(4, 1000, 82))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_unit</span></div><div class='xr-var-dims'>(chain, draw, unit)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6614 0.9119 ... 0.6462 -0.2735</div><input id='attrs-f2f62133-2ad6-4f6a-91b1-104e049e0a9b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f2f62133-2ad6-4f6a-91b1-104e049e0a9b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e5bbbfe4-170f-475c-9723-ab0a8aaff3ae' class='xr-var-data-in' type='checkbox'><label for='data-e5bbbfe4-170f-475c-9723-ab0a8aaff3ae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.66141382,  0.91187414, -0.69720728, ...,  1.18817883,-0.11718827,  1.22700041],[ 0.46036565,  0.90947162, -0.44787056, ...,  1.1425163 ,0.16418126,  0.97033131],[ 0.36321136,  0.41239021, -0.52685144, ...,  1.12711019,0.20721978,  0.87012866],...,[ 0.18822349, -1.49205798, -1.4851659 , ...,  1.13676146,-1.00051607, -0.08833499],[ 0.18940638, -1.45748979, -1.70881257, ...,  1.08007741,-0.65384127, -0.70935805],[ 0.18547127,  1.08491336, -0.76557088, ...,  1.1094289 ,-0.3635093 ,  0.29377173]],[[ 1.34439491,  0.87557154, -0.12978106, ...,  1.97486614,0.31912374,  1.04753918],[ 1.50382524,  0.25617183, -0.67366841, ...,  1.79391715,0.46009037, -0.01232055],[ 0.52798321, -0.70380248, -0.23045785, ...,  1.91001619,1.6690534 , -0.24283607],...[ 0.69873159, -0.4609834 , -0.98276921, ...,  1.26378197,0.54070327,  1.43895497],[ 0.37227206,  0.34166406, -0.93859383, ...,  1.34201966,0.13988095,  0.77793639],[ 1.01767693,  0.85685715, -0.81045384, ...,  1.46467722,-0.25680097,  0.8624094 ]],[[ 1.09843189, -1.16346129,  1.40233758, ...,  1.26302421,0.41345439, -2.23551277],[ 0.22221384,  0.7698831 , -0.72750439, ...,  1.32155772,-0.44067804, -0.93373859],[ 0.25798027,  0.50473519, -0.58495782, ...,  1.30079248,-1.06897922, -0.80622789],...,[ 0.14789494,  1.47379351, -0.7353204 , ...,  1.44755256,0.04285426, -0.03820394],[ 0.85656924, -0.57301599, -0.83029335, ...,  0.99415265,-0.32788645,  0.37189027],[ 0.49734561, -0.22259817, -0.93990097, ...,  1.06984741,0.64618764, -0.27354334]]], shape=(4, 1000, 50))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_style_class</span></div><div class='xr-var-dims'>(chain, draw, style_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1153 -0.9688 ... -0.9966 -0.5666</div><input id='attrs-0f211bed-244b-40ad-abd8-c38dc2d38521' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0f211bed-244b-40ad-abd8-c38dc2d38521' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0c3fcb0e-0425-4e85-b343-bda9425374e3' class='xr-var-data-in' type='checkbox'><label for='data-0c3fcb0e-0425-4e85-b343-bda9425374e3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.11527081, -0.96875343,  3.0908108 ,  0.57844028],[-0.03476928, -0.79065773,  1.74534264,  0.66202225],[ 0.28941651, -1.05129287,  1.0625331 ,  0.61528392],...,[-0.46393273, -0.98379984,  1.05289008, -0.34935028],[-0.2206667 , -0.71520859,  0.5867883 , -0.30047788],[-0.5413899 , -0.77937736,  0.38636852, -0.43173637]],[[-0.59208019, -1.42298495, -0.43825838, -0.22689128],[-1.04700116, -1.30922645,  2.00201345, -0.94383778],[-0.95180742, -1.19878496, -2.46962123, -1.07517214],...,[-0.23686266, -1.10838032,  1.28929135,  0.01211562],[ 0.19175591,  0.04637787, -0.87608028,  0.21612562],[ 0.49671687,  0.33456009,  1.1144499 ,  0.54601494]],[[-0.56079126, -0.61784434,  0.25496324, -0.49549554],[-0.20802327, -0.35428667,  0.93017768, -0.21328646],[-0.33622234, -0.55836751,  0.45601282, -0.33545804],...,[ 0.42204148, -0.45574478, -0.06784592,  0.36198628],[ 0.16984713, -0.93932086,  0.35012306,  0.22680057],[ 0.10438883, -0.55937365,  0.27921354,  0.18150156]],[[-0.58059687, -0.75511388,  2.33345322, -0.52123906],[-0.96037694, -1.19218479, -0.96088151, -1.0093959 ],[-0.90550239, -1.01077526,  1.34017665, -0.88386677],...,[-0.21502124, -2.21318731, -1.97828617,  0.87367878],[-0.46516328, -1.70411892,  2.13298693, -0.34695083],[-0.61440939, -1.40624736, -0.9965582 , -0.56657537]]],shape=(4, 1000, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_size_class</span></div><div class='xr-var-dims'>(chain, draw, size_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.07 -0.7723 2.564 ... 0.633 0.521</div><input id='attrs-fc248168-ec6c-4f1f-b535-5d0cc3741033' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fc248168-ec6c-4f1f-b535-5d0cc3741033' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78fc52a2-be00-4c2d-add9-4fbd56ee5259' class='xr-var-data-in' type='checkbox'><label for='data-78fc52a2-be00-4c2d-add9-4fbd56ee5259' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-2.07003773e+00, -7.72278100e-01,  2.56379125e+00,3.81520016e-01],[-1.98984190e+00, -7.65785968e-01,  2.54288272e+00,7.18266320e-01],[-1.83426500e+00, -6.11750838e-01,  2.63196581e+00,1.89335550e+00],...,[-3.73787092e-01, -1.39845322e-01,  7.13580819e-01,-1.36333251e-01],[-4.24909423e-01, -2.46644200e-01,  7.59406378e-01,-2.74285135e-01],[-4.83904162e-01, -2.16259618e-01,  7.01503226e-01,1.62416814e+00]],[[-7.08674720e-01, -2.13823732e-01,  1.61901146e+00,-9.81037225e-01],[-4.16806570e-01, -2.38039511e-01,  1.52874207e+00,5.36665495e-01],[-3.03153126e-01,  1.50241029e-02,  1.31179700e+00,3.14226962e-01],...[-1.82362316e-01, -8.32976466e-02,  4.99028429e-01,3.38346066e-01],[-1.97515234e-01, -3.04864071e-02,  7.71007075e-01,1.17689653e+00],[-1.21897963e-01, -2.19225496e-03,  6.33127398e-01,5.64031876e-01]],[[-7.48071547e-01, -6.09059190e-01, -1.75942357e-01,4.60347235e-01],[-5.87645498e-01, -4.74526058e-01,  3.82572909e-01,4.77012947e-01],[-3.39387829e-01, -1.91064551e-01,  5.40925076e-01,-1.04634138e+00],...,[-4.32024821e-01, -2.21628759e-01,  5.13767662e-01,-9.61532215e-01],[-6.02934791e-01, -3.98978369e-01,  5.37548475e-01,2.66122740e+00],[-6.02028341e-01, -4.51366914e-01,  6.33024861e-01,5.21025281e-01]]], shape=(4, 1000, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_sector</span></div><div class='xr-var-dims'>(chain, draw, sector)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.09606 0.8533 ... 0.6455 -0.3714</div><input id='attrs-5228974b-e979-49f8-aa69-958ce52a5f1a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5228974b-e979-49f8-aa69-958ce52a5f1a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e6e5f927-6c58-4889-8790-c9ce84bbbd03' class='xr-var-data-in' type='checkbox'><label for='data-e6e5f927-6c58-4889-8790-c9ce84bbbd03' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 9.60615075e-02,  8.53347402e-01,  8.96509958e-01, ...,-2.66988054e-01,  6.33941020e-01, -8.02044314e-01],[ 2.34682628e-01,  8.88220444e-01,  8.13866310e-01, ...,-3.77161131e-03,  7.61332702e-01, -8.69526359e-01],[ 1.23440492e-01,  7.14483626e-01,  1.11390146e+00, ...,8.39973212e-05,  9.86397101e-01, -6.12054819e-01],...,[-6.70145871e-01, -5.67651086e-01, -7.92883814e-01, ...,-8.46716320e-01, -3.08619520e-01, -1.55469602e+00],[-7.28874298e-01, -6.08229790e-01, -6.54759068e-01, ...,-9.09554679e-01, -2.65726816e-01, -1.40303901e+00],[-6.56441364e-01, -9.23380986e-01, -5.85749185e-01, ...,-8.39078187e-01, -6.36252351e-01, -1.65579103e+00]],[[-2.19194137e-01,  7.09356553e-02, -1.35199916e-01, ...,-5.80564614e-02, -8.65090440e-02, -1.57068896e+00],[ 2.46003632e-01,  4.34663850e-01,  7.49839083e-02, ...,-1.01332405e-01, -9.36229414e-03, -7.65897954e-01],[ 7.02492901e-02,  1.99004356e-01,  2.24811876e-01, ...,-2.12993065e-01, -7.60855933e-03, -9.74295546e-01],...-4.24086878e-01,  1.20816231e-01, -5.16205822e-01],[-7.40742914e-01,  4.32657551e-01, -2.25935328e-01, ...,-1.64638186e-01,  9.24126293e-02, -8.18050085e-01],[-6.81435166e-01,  2.52639590e-01, -3.30376110e-01, ...,-1.22797420e-01,  2.75026464e-01, -9.35590060e-01]],[[ 1.13394669e+00,  3.58535589e-01,  1.72685804e-01, ...,-1.70713299e-01,  1.14135629e+00, -1.04962221e+00],[ 1.58444734e-01,  3.88989249e-01,  3.58814727e-01, ...,-5.45835111e-01,  6.58107475e-01, -7.77811530e-01],[ 7.90236393e-02,  4.78466171e-01,  3.17638232e-02, ...,-2.18332553e-01,  4.08537900e-01, -9.94056991e-01],...,[ 3.00089941e-01, -5.79475067e-02,  9.16464726e-02, ...,3.36348090e-02,  2.35863430e-01, -1.01216286e+00],[ 4.06524989e-01,  3.19944428e-01,  2.91270819e-01, ...,9.84421908e-02,  4.67001396e-01, -4.11139172e-01],[ 4.71614611e-01,  5.03929838e-01,  4.07424298e-01, ...,2.18278951e-02,  6.45463430e-01, -3.71378577e-01]]],shape=(4, 1000, 9))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_industry</span></div><div class='xr-var-dims'>(chain, draw, industry)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.7362 1.304 ... -0.5576 -1.126</div><input id='attrs-3819cac1-93b0-4366-ba01-2b4c0ef6c9d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3819cac1-93b0-4366-ba01-2b4c0ef6c9d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c169745d-5780-4a75-be38-29d76856b718' class='xr-var-data-in' type='checkbox'><label for='data-c169745d-5780-4a75-be38-29d76856b718' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.73619249,  1.30378203, -1.2114346 , ..., -0.93721843,-0.00518298, -0.56943553],[ 0.89670288,  1.24043913, -1.35827743, ..., -1.16990875,0.2727373 , -0.469581  ],[ 0.51927483,  0.46362591, -1.56398829, ..., -1.93711488,-0.35993745, -0.62741232],...,[ 0.83437202, -0.04115003, -1.60822126, ..., -1.32329072,-0.20028928, -0.96087441],[ 0.6679323 , -0.6648334 , -1.31740194, ..., -1.46126211,-0.89688872, -0.7487555 ],[ 0.63812406,  2.23099006, -0.3822417 , ..., -1.72859473,1.22643499, -1.87265597]],[[ 0.71109566,  0.32965241, -1.33889992, ..., -1.87157965,0.93584946, -0.88505419],[ 0.47953994, -0.75906362, -1.0843221 , ..., -1.49646419,-0.23694077, -0.92522522],[ 0.38597269, -0.17608424, -1.22801031, ..., -1.43046611,-1.18253488, -1.00413131],...[ 0.78184943,  0.17961413, -1.40241529, ..., -0.35288212,-0.49967219,  0.71631308],[ 0.51845495,  0.65926088, -1.53689988, ..., -0.68213956,-0.27305719, -0.06962224],[ 0.22110881, -0.03212184, -1.02918457, ..., -1.25678115,-0.60065005,  0.54540305]],[[ 0.0043497 ,  0.1996324 , -1.03267396, ..., -1.201619  ,-1.53981753, -1.72450934],[ 0.90222563,  0.87468656, -1.55138955, ..., -1.78863599,0.45780251, -1.40174282],[ 0.60102759,  0.09541315, -1.38933272, ..., -1.19368316,-0.42746431, -1.06474204],...,[ 0.52969197,  0.65204288, -0.69299161, ..., -1.81439348,-1.25699699, -1.23130392],[ 0.3931816 ,  0.01736582, -0.81670426, ..., -1.64801192,0.14621032, -0.72317953],[ 0.33749061,  0.59937225, -0.35905622, ..., -0.66271297,-0.55762196, -1.12640889]]], shape=(4, 1000, 59))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>z_state</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.203 1.561 ... 0.712 -0.8965</div><input id='attrs-52e508b2-bd34-45ef-879f-0b19d72a7ab5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-52e508b2-bd34-45ef-879f-0b19d72a7ab5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca45673e-af95-49ca-a9c0-7176236cc8e9' class='xr-var-data-in' type='checkbox'><label for='data-ca45673e-af95-49ca-a9c0-7176236cc8e9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 1.20264939,  1.56141905, -0.85180993, ..., -1.2664969 ,1.49306333,  0.62883085],[ 0.81538326,  0.14256931, -0.56969165, ..., -0.21805668,0.57846685,  1.18354421],[ 1.21479131,  0.03522442,  0.16218588, ...,  2.24885089,0.9060747 ,  0.385062  ],...,[ 1.67077475,  0.43542399,  0.61551675, ..., -0.80262199,-0.12481004,  0.17928583],[ 1.47119096,  0.11848916,  0.88363159, ...,  0.41720893,1.07551147,  0.38984023],[ 1.00168887,  0.80080452, -0.73145793, ..., -0.28028662,-0.94433607, -0.01630829]],[[ 0.69780948, -0.9133292 , -1.86887291, ...,  0.28210844,0.81372059,  2.71126392],[ 1.44339896, -0.18163425,  1.24662131, ..., -0.23523559,-0.26984329, -0.76882463],[ 0.78597267,  0.38780528, -1.13220956, ...,  0.14493004,0.73030837,  1.66413186],...[ 1.43772997, -1.48998802, -0.23647666, ...,  0.43359057,1.09353283, -1.38249824],[ 1.04336189,  0.30235066, -0.25923789, ...,  1.1278507 ,0.09230382, -0.22145346],[ 0.71917359, -1.30449471,  2.04867417, ...,  0.82692232,0.95693441,  1.55642536]],[[ 1.18893672, -0.54983139, -0.32202653, ..., -0.11706251,0.38601251,  0.70470185],[ 1.09724487,  1.55946638,  0.49828621, ...,  0.38253419,0.30334661, -0.55061466],[ 1.76937052, -0.06145989, -0.69958523, ...,  1.8108088 ,1.25841426, -0.21265841],...,[ 1.97213712,  1.4642311 ,  0.87371412, ..., -0.5627868 ,0.32253589, -1.19307858],[ 0.1080221 , -0.367905  , -0.39978168, ...,  0.37565748,0.25618751,  0.74730636],[ 0.02992151, -0.56757882,  1.70090703, ...,  1.58187551,0.71197346, -0.89649008]]], shape=(4, 1000, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_region</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0513 0.04951 ... 0.04486 0.04265</div><input id='attrs-a6a34729-7e15-46fd-8616-f781377103b3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a6a34729-7e15-46fd-8616-f781377103b3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-24cec9e4-06b9-42d5-ad14-37fb434cfb9a' class='xr-var-data-in' type='checkbox'><label for='data-24cec9e4-06b9-42d5-ad14-37fb434cfb9a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.05129598, 0.04951411, 0.05706696, ..., 0.03658529, 0.03620215,0.03647514],[0.04718668, 0.03661388, 0.03258564, ..., 0.05590357, 0.05210286,0.04400904],[0.03817351, 0.03218647, 0.02752303, ..., 0.05155265, 0.0558778 ,0.05424608],[0.03165398, 0.03700169, 0.04288733, ..., 0.04981997, 0.04486227,0.04264978]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_exchange</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0516 0.05088 ... 0.06497 0.06161</div><input id='attrs-6ad0b4e5-fd53-4abd-a7be-81fbc6a00225' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6ad0b4e5-fd53-4abd-a7be-81fbc6a00225' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1c1d4b14-41bc-404c-b1d7-bf2b53b96e98' class='xr-var-data-in' type='checkbox'><label for='data-1c1d4b14-41bc-404c-b1d7-bf2b53b96e98' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.0516007 , 0.05087964, 0.04734097, ..., 0.06348079, 0.06360077,0.05674865],[0.07289057, 0.06929839, 0.07206821, ..., 0.07641786, 0.07507522,0.06967748],[0.05476717, 0.05925096, 0.05999737, ..., 0.041703  , 0.04735431,0.0480573 ],[0.06400143, 0.06335118, 0.06263573, ..., 0.06477999, 0.0649666 ,0.06160888]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_unit</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.09486 0.09641 ... 0.06822 0.05819</div><input id='attrs-7f4a10d5-06ff-4984-b016-1c92f9aef3e7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7f4a10d5-06ff-4984-b016-1c92f9aef3e7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-901e0209-c234-4ff5-ae2e-7bfbb3560177' class='xr-var-data-in' type='checkbox'><label for='data-901e0209-c234-4ff5-ae2e-7bfbb3560177' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.09485544, 0.09640692, 0.09877804, ..., 0.08083724, 0.08424138,0.07893009],[0.057146  , 0.06187734, 0.05928987, ..., 0.039758  , 0.04423448,0.03967661],[0.0715352 , 0.05931719, 0.06439282, ..., 0.09189492, 0.08240835,0.08267776],[0.05841063, 0.06477789, 0.07417184, ..., 0.06216042, 0.06822433,0.0581937 ]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_style_class</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.007013 0.006806 ... 0.006512</div><input id='attrs-6abe7796-d037-4c9d-998e-e39dc6ec508f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6abe7796-d037-4c9d-998e-e39dc6ec508f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-42a89031-af6f-4591-9cdb-5f44da661417' class='xr-var-data-in' type='checkbox'><label for='data-42a89031-af6f-4591-9cdb-5f44da661417' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00701254, 0.00680649, 0.00732127, ..., 0.03681222, 0.03532509,0.08010747],[0.01956226, 0.04290286, 0.03528603, ..., 0.01985428, 0.08025423,0.12375125],[0.10265716, 0.1148794 , 0.07897645, ..., 0.0260764 , 0.02165861,0.02117581],[0.05351938, 0.08625578, 0.1347094 , ..., 0.00468133, 0.00823865,0.00651227]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_size_class</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01475 0.0127 ... 0.05285 0.04807</div><input id='attrs-4a2969db-3b1e-4521-95e2-1f0780a9225b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4a2969db-3b1e-4521-95e2-1f0780a9225b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5226189a-da88-4243-ab17-20db2f127006' class='xr-var-data-in' type='checkbox'><label for='data-5226189a-da88-4243-ab17-20db2f127006' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.01475496, 0.01270084, 0.01578068, ..., 0.05498035, 0.05359554,0.05262821],[0.02657942, 0.02948727, 0.03817722, ..., 0.13532314, 0.12445614,0.17632501],[0.04035459, 0.02569954, 0.02776246, ..., 0.0886675 , 0.06391849,0.07579914],[0.10734705, 0.06191246, 0.06528572, ..., 0.06166991, 0.05284871,0.04807061]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_sector</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.03424 0.03977 ... 0.07202 0.07425</div><input id='attrs-67266c9d-f32e-45fe-bf2f-cf336ceb17da' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-67266c9d-f32e-45fe-bf2f-cf336ceb17da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d380e038-2b70-48e7-9823-2ba0446d110f' class='xr-var-data-in' type='checkbox'><label for='data-d380e038-2b70-48e7-9823-2ba0446d110f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.0342409 , 0.03977218, 0.04397588, ..., 0.06627428, 0.07240662,0.0554203 ],[0.04779256, 0.05225091, 0.07376716, ..., 0.03389543, 0.02509322,0.0270738 ],[0.03061409, 0.02823676, 0.0293845 , ..., 0.0624173 , 0.05680745,0.0520544 ],[0.03435551, 0.04213269, 0.03863962, ..., 0.0627765 , 0.07201611,0.07424622]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_industry</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.03213 0.03149 ... 0.02997 0.0314</div><input id='attrs-6c4f9bbc-1e9a-4a89-9e32-31703a9eaf5a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6c4f9bbc-1e9a-4a89-9e32-31703a9eaf5a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9d8d9c48-1ce9-4c4d-b127-472fb656d9b2' class='xr-var-data-in' type='checkbox'><label for='data-9d8d9c48-1ce9-4c4d-b127-472fb656d9b2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.03213117, 0.03149241, 0.03029718, ..., 0.02929995, 0.03087192,0.02579442],[0.03299617, 0.03360716, 0.03928074, ..., 0.03133435, 0.03582717,0.04046809],[0.02727266, 0.03027149, 0.02762546, ..., 0.0381097 , 0.0356629 ,0.03392034],[0.03215092, 0.03016527, 0.03691747, ..., 0.02932436, 0.02997293,0.0314023 ]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_state</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05905 0.05877 ... 0.05718 0.0572</div><input id='attrs-d232d131-5277-4823-8047-c454ae10d3da' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d232d131-5277-4823-8047-c454ae10d3da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6cd1b933-1fb9-46bf-b268-9a56898fe8eb' class='xr-var-data-in' type='checkbox'><label for='data-6cd1b933-1fb9-46bf-b268-9a56898fe8eb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.05904595, 0.05877408, 0.05851212, ..., 0.04953434, 0.04899633,0.05250779],[0.06075583, 0.05908576, 0.06087076, ..., 0.05701421, 0.05895303,0.05740906],[0.05900241, 0.05531064, 0.05891085, ..., 0.05686601, 0.05730786,0.05526633],[0.05862611, 0.05824165, 0.05621402, ..., 0.05630967, 0.05718269,0.0572015 ]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_obs_base</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01099 0.01114 ... 0.01108 0.01148</div><input id='attrs-a64c0d7d-cae8-48ed-986d-c23280a94767' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a64c0d7d-cae8-48ed-986d-c23280a94767' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-81fdc90a-5b83-4510-a994-69d078dba173' class='xr-var-data-in' type='checkbox'><label for='data-81fdc90a-5b83-4510-a994-69d078dba173' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.01099454, 0.0111433 , 0.01120926, ..., 0.01243303, 0.01232614,0.01167781],[0.01062064, 0.01035167, 0.01044917, ..., 0.011298  , 0.01080547,0.01123805],[0.01119658, 0.01065178, 0.01094806, ..., 0.01131402, 0.01117115,0.01132723],[0.01123974, 0.01102108, 0.01080853, ..., 0.01157913, 0.01108227,0.01147919]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>nu</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.646 3.295 3.66 ... 3.103 3.696</div><input id='attrs-7a40af41-2b44-4677-be74-122561002553' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7a40af41-2b44-4677-be74-122561002553' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7246d21d-0e92-4494-96f8-3f6e198222e6' class='xr-var-data-in' type='checkbox'><label for='data-7246d21d-0e92-4494-96f8-3f6e198222e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[3.64572111, 3.29541905, 3.66036769, ..., 4.1067695 , 3.9776426 ,3.49331109],[3.17183278, 2.94217169, 2.9784084 , ..., 3.33050763, 3.07275088,3.29260117],[3.47617541, 2.94472501, 3.13940106, ..., 3.2696154 , 3.60453277,3.39128159],[3.43360318, 3.53541662, 2.93967747, ..., 3.81672212, 3.1028364 ,3.69624276]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>region_effect</span></div><div class='xr-var-dims'>(chain, draw, region)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.003458 0.07767 ... -0.009963</div><input id='attrs-94f55ea2-0941-48f8-b5fc-10f59825cd51' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-94f55ea2-0941-48f8-b5fc-10f59825cd51' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-36986a97-af1f-49f2-9ec9-547161047a67' class='xr-var-data-in' type='checkbox'><label for='data-36986a97-af1f-49f2-9ec9-547161047a67' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.00345846,  0.07767162,  0.00855426,  0.03987903,0.01105361],[ 0.01060535,  0.07686349,  0.00589049,  0.0340161 ,0.00988836],[ 0.02122192,  0.09061305,  0.01520827,  0.03967978,0.01280529],...,[ 0.01137656,  0.07623227, -0.02758325, -0.0185671 ,-0.03094381],[ 0.01119611,  0.07717341, -0.02646681, -0.02050404,-0.02905794],[-0.00147768,  0.06691301, -0.02538535, -0.02232362,-0.03498911]],[[-0.03008605,  0.02459731, -0.02857146, -0.01235579,-0.05003512],[-0.01288235,  0.03766951, -0.03473627, -0.02267451,-0.04542074],[-0.01068338,  0.03341558, -0.03444891, -0.01235053,-0.05096935],...[-0.03159592,  0.03865573, -0.03942886, -0.01798378,-0.05330286],[ 0.00024677,  0.04571505, -0.04506626, -0.0100668 ,-0.05272577],[-0.01809419,  0.04154248, -0.03758299, -0.03148284,-0.06516817]],[[-0.00741167,  0.0644666 , -0.00771592,  0.00925137,-0.04220448],[-0.0057715 ,  0.04816755, -0.02217589, -0.00206574,-0.03946009],[-0.0107414 ,  0.05442543, -0.01918756, -0.00266969,-0.06121481],...,[ 0.00831288,  0.04701235,  0.01304494,  0.01166484,-0.02635877],[ 0.01504992,  0.04366385,  0.01605248,  0.00900862,-0.01642191],[ 0.02257345,  0.04749275,  0.01134255, -0.0182129 ,-0.00996296]]], shape=(4, 1000, 5))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>exchange_effect</span></div><div class='xr-var-dims'>(chain, draw, exchange)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.07509 0.06076 ... -0.07617</div><input id='attrs-58b0cafd-7c5a-467c-9e99-693b732626b8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-58b0cafd-7c5a-467c-9e99-693b732626b8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b096651e-5ff3-4aec-836a-613c140a4807' class='xr-var-data-in' type='checkbox'><label for='data-b096651e-5ff3-4aec-836a-613c140a4807' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.0750883 ,  0.06075935, -0.04219045, ..., -0.00449584,0.05774672, -0.03091787],[ 0.05135459,  0.10264487, -0.04638457, ..., -0.05857077,0.06364332, -0.02913968],[ 0.03605516,  0.08440323, -0.03132284, ...,  0.03543208,0.04831361, -0.05919776],...,[ 0.01457336,  0.08880737,  0.00439105, ..., -0.0131878 ,0.08791109, -0.10682856],[ 0.01660779,  0.07642464,  0.0104405 , ...,  0.0197334 ,0.10677421, -0.06205494],[ 0.06180057,  0.08147921, -0.04363306, ..., -0.0279843 ,0.07398836, -0.06719188]],[[ 0.01838481,  0.05768689, -0.07832686, ...,  0.07910715,0.04363841, -0.11744521],[-0.02282742,  0.04981595, -0.05034557, ..., -0.05178915,0.04699193, -0.04747634],[ 0.07446541,  0.08820016, -0.04577046, ...,  0.12172332,0.03644046, -0.09682909],...[ 0.05274608,  0.04812564, -0.0076953 , ..., -0.03986935,0.04229229, -0.03785096],[ 0.01893135,  0.04176201, -0.04906524, ..., -0.00245249,0.05333052, -0.05426961],[ 0.0072311 ,  0.06089941, -0.06606072, ..., -0.0560336 ,0.07092562, -0.06369696]],[[ 0.02821709,  0.10951433, -0.19002694, ...,  0.08497357,0.10231442, -0.03452852],[ 0.02547772,  0.1120837 , -0.06278298, ..., -0.09831871,0.09875296, -0.06401542],[-0.0221955 ,  0.08847654, -0.07287473, ...,  0.03266793,0.10328987, -0.02650652],...,[ 0.0771323 ,  0.10189731, -0.01953744, ..., -0.01911103,0.10011266, -0.06301105],[ 0.0500683 ,  0.02052347, -0.02990628, ...,  0.02905466,0.06877016, -0.03899031],[ 0.04697454,  0.05317278, -0.03286272, ..., -0.00724544,0.06271056, -0.07617146]]], shape=(4, 1000, 82))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>unit_effect</span></div><div class='xr-var-dims'>(chain, draw, unit)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06274 0.0865 ... 0.0376 -0.01592</div><input id='attrs-d6c2f53f-f043-4c29-95b1-e64e5441d1ec' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d6c2f53f-f043-4c29-95b1-e64e5441d1ec' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fb49c90d-5dad-42c9-a5de-41c2d45d70c6' class='xr-var-data-in' type='checkbox'><label for='data-fb49c90d-5dad-42c9-a5de-41c2d45d70c6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.0627387 ,  0.08649622, -0.0661339 , ...,  0.11270522,-0.01111594,  0.11638766],[ 0.04438244,  0.08767936, -0.04317782, ...,  0.11014648,0.01582821,  0.09354666],[ 0.03587731,  0.0407351 , -0.05204135, ...,  0.11133374,0.02046876,  0.08594961],...,[ 0.01521547, -0.12061385, -0.12005671, ...,  0.09189266,-0.08087896, -0.00714076],[ 0.01595585, -0.12278095, -0.14395272, ...,  0.09098721,-0.05508049, -0.0597573 ],[ 0.01463926,  0.08563231, -0.06042658, ...,  0.08756733,-0.02869182,  0.02318743]],[[ 0.0768268 ,  0.05003542, -0.00741647, ...,  0.11285571,0.01823665,  0.05986268],[ 0.09305271,  0.01585123, -0.04168481, ...,  0.11100282,0.02846917, -0.00076236],[ 0.03130406, -0.04172836, -0.01366382, ...,  0.11324462,0.09895796, -0.01439772],...[ 0.06420989, -0.04236203, -0.0903115 , ...,  0.11613515,0.04968789,  0.13223266],[ 0.03067833,  0.02815597, -0.07734797, ...,  0.11059363,0.01152736,  0.06410846],[ 0.08413925,  0.07084303, -0.06700651, ...,  0.12109623,-0.02123173,  0.07130208]],[[ 0.06416009, -0.0679585 ,  0.08191141, ...,  0.07377403,0.02415013, -0.1305777 ],[ 0.01439454,  0.0498714 , -0.0471262 , ...,  0.08560772,-0.02854619, -0.06048562],[ 0.01913487,  0.03743714, -0.0433874 , ...,  0.09648217,-0.07928816, -0.05979941],...,[ 0.00919321,  0.09161163, -0.04570783, ...,  0.08998048,0.00266384, -0.00237477],[ 0.05843886, -0.03909363, -0.05664621, ...,  0.0678254 ,-0.02236983,  0.02537197],[ 0.02894238, -0.01295381, -0.05469631, ...,  0.06225838,0.03760405, -0.0159185 ]]], shape=(4, 1000, 50))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>style_class_effect</span></div><div class='xr-var-dims'>(chain, draw, style_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0008083 -0.006793 ... -0.00369</div><input id='attrs-c921e1f8-1c47-4779-b607-780f52b9b096' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c921e1f8-1c47-4779-b607-780f52b9b096' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-98adcc8a-1158-4d0a-b81b-0fdd49d0eb3f' class='xr-var-data-in' type='checkbox'><label for='data-98adcc8a-1158-4d0a-b81b-0fdd49d0eb3f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.00080834, -0.00679342,  0.02167442,  0.00405633],[-0.00023666, -0.0053816 ,  0.01187966,  0.00450605],[ 0.0021189 , -0.0076968 ,  0.00777909,  0.00450466],...,[-0.01707839, -0.03621586,  0.03875922, -0.01286036],[-0.00779507, -0.02526481,  0.02072835, -0.01061441],[-0.04336937, -0.06243395,  0.030951  , -0.03458531]],[[-0.01158243, -0.0278368 , -0.00857332, -0.00443851],[-0.04491934, -0.05616956,  0.0858921 , -0.04049334],[-0.0335855 , -0.04230036, -0.08714312, -0.03793855],...,[-0.00470274, -0.0220061 ,  0.02559795,  0.00024055],[ 0.01538922,  0.00372202, -0.07030915,  0.017345  ],[ 0.06146933,  0.04140223,  0.13791457,  0.06757003]],[[-0.05756924, -0.06342615,  0.0261738 , -0.05086617],[-0.02389759, -0.04070024,  0.10685826, -0.02450222],[-0.02655365, -0.04409789,  0.03601428, -0.02649329],...,[ 0.01100532, -0.01188418, -0.00176918,  0.0094393 ],[ 0.00367865, -0.02034438,  0.00758318,  0.00491218],[ 0.00221052, -0.01184519,  0.00591257,  0.00384344]],[[-0.03107318, -0.04041322,  0.12488496, -0.02789639],[-0.08283806, -0.10283282, -0.08288158, -0.08706623],[-0.12197969, -0.13616093,  0.1805344 , -0.11906516],...,[-0.00100659, -0.01036066, -0.00926101,  0.00408998],[-0.00383232, -0.01403963,  0.01757293, -0.00285841],[-0.0040012 , -0.00915786, -0.00648986, -0.00368969]]],shape=(4, 1000, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>size_class_effect</span></div><div class='xr-var-dims'>(chain, draw, size_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.03054 -0.01139 ... 0.02505</div><input id='attrs-642e65a9-5787-47b3-9b5c-9ed66614bb7c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-642e65a9-5787-47b3-9b5c-9ed66614bb7c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ec92d351-a9e1-4563-989f-0cfe1bde849c' class='xr-var-data-in' type='checkbox'><label for='data-ec92d351-a9e1-4563-989f-0cfe1bde849c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.03054332, -0.01139493,  0.03782863,  0.00562931],[-0.02527266, -0.00972612,  0.03229674,  0.00912258],[-0.02894595, -0.00965385,  0.04153422,  0.02987844],...,[-0.02055095, -0.00768875,  0.03923293, -0.00749565],[-0.02277325, -0.01321903,  0.0407008 , -0.01470046],[-0.02546701, -0.01138136,  0.03691886,  0.08547707]],[[-0.01883616, -0.00568331,  0.04303239, -0.0260754 ],[-0.01229049, -0.00701913,  0.04507843,  0.0158248 ],[-0.01157354,  0.00057358,  0.05008077,  0.01199631],...,[-0.09264683, -0.07683159, -0.03197252, -0.06455275],[-0.11060644, -0.10150597, -0.05253221,  0.13255102],[-0.12319384, -0.11615765, -0.06576013, -0.0848755 ]],[[-0.04533377, -0.04075064,  0.00883486,  0.0363392 ],[-0.03534584, -0.02865248,  0.02401177, -0.02368909],[-0.03785143, -0.02954845,  0.0192086 ,  0.02875908],...,[-0.01616961, -0.00738579,  0.0442476 ,  0.0300003 ],[-0.01262488, -0.00194865,  0.04928161,  0.07522545],[-0.00923976, -0.00016617,  0.04799051,  0.04275313]],[[-0.08030327, -0.06538071, -0.01888689,  0.04941692],[-0.03638258, -0.02937907,  0.02368603,  0.02953304],[-0.02215718, -0.01247379,  0.03531468, -0.06831115],...,[-0.02664293, -0.01366783,  0.031684  , -0.0592976 ],[-0.03186432, -0.02108549,  0.02840874,  0.14064243],[-0.02893987, -0.02169748,  0.03042989,  0.025046  ]]],shape=(4, 1000, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sector_effect</span></div><div class='xr-var-dims'>(chain, draw, sector)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.003289 0.02922 ... -0.02757</div><input id='attrs-50e0c514-5690-413a-9713-f99983380159' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-50e0c514-5690-413a-9713-f99983380159' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e3c4c0cb-a4b8-4236-9c53-b784f50308cc' class='xr-var-data-in' type='checkbox'><label for='data-e3c4c0cb-a4b8-4236-9c53-b784f50308cc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 3.28923255e-03,  2.92193838e-02,  3.06973086e-02, ...,-9.14191148e-03,  2.17067116e-02, -2.74627198e-02],[ 9.33383925e-03,  3.53264616e-02,  3.23692358e-02, ...,-1.50005196e-04,  3.02798597e-02, -3.45829572e-02],[ 5.42840443e-03,  3.14200471e-02,  4.89847984e-02, ...,3.69385623e-06,  4.33776818e-02, -2.69156501e-02],...,[-4.44134328e-02, -3.76206650e-02, -5.25478011e-02, ...,-5.61155115e-02, -2.04535354e-02, -1.03036354e-01],[-5.27753267e-02, -4.40398653e-02, -4.74088932e-02, ...,-6.58577830e-02, -1.92403815e-02, -1.01589317e-01],[-3.63801751e-02, -5.11740481e-02, -3.24623936e-02, ...,-4.65019620e-02, -3.52612940e-02, -9.17644302e-02]],[[-1.04758496e-02,  3.39019677e-03, -6.46155051e-03, ...,-2.77466709e-03, -4.13448893e-03, -7.50672512e-02],[ 1.28539144e-02,  2.27115831e-02,  3.91797767e-03, ...,-5.29471070e-03, -4.89188418e-04, -4.00188675e-02],[ 5.18209028e-03,  1.46799852e-02,  1.65837325e-02, ...,-1.57118925e-02, -5.61261777e-04, -7.18710107e-02],...-2.64703567e-02,  7.54102261e-03, -3.22201723e-02],[-4.20797188e-02,  2.45781738e-02, -1.28348107e-02, ...,-9.35267614e-03,  5.24972616e-03, -4.64713423e-02],[-3.54716963e-02,  1.31510014e-02, -1.71975290e-02, ...,-6.39214558e-03,  1.43163366e-02, -4.87015759e-02]],[[ 3.89573123e-02,  1.23176715e-02,  5.93270815e-03, ...,-5.86494175e-03,  3.92118727e-02, -3.60603020e-02],[ 6.67570319e-03,  1.63891642e-02,  1.51178304e-02, ...,-2.29975027e-02,  2.77278396e-02, -3.27712937e-02],[ 3.05344369e-03,  1.84877528e-02,  1.22734218e-03, ...,-8.43628770e-03,  1.57857507e-02, -3.84099881e-02],...,[ 1.88385964e-02, -3.63774168e-03,  5.75324483e-03, ...,2.11147560e-03,  1.48066807e-02, -6.35400422e-02],[ 2.92763496e-02,  2.30411542e-02,  2.09761923e-02, ...,7.08942395e-03,  3.36316254e-02, -2.96086452e-02],[ 3.50156037e-02,  3.74148872e-02,  3.02497154e-02, ...,1.62063877e-03,  4.79232218e-02, -2.75734567e-02]]],shape=(4, 1000, 9))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>industry_effect</span></div><div class='xr-var-dims'>(chain, draw, industry)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.02365 0.04189 ... -0.03537</div><input id='attrs-b9de783a-4085-4a07-9564-332ac171b9f8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b9de783a-4085-4a07-9564-332ac171b9f8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ffecec87-d1c4-428e-9131-b2e9912e5410' class='xr-var-data-in' type='checkbox'><label for='data-ffecec87-d1c4-428e-9131-b2e9912e5410' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.02365473,  0.04189204, -0.03892481, ..., -0.03011392,-0.00016654, -0.01829663],[ 0.02823934,  0.03906442, -0.04277544, ..., -0.03684325,0.00858916, -0.01478824],[ 0.01573256,  0.01404656, -0.04738444, ..., -0.05868912,-0.01090509, -0.01900882],...,[ 0.02444706, -0.00120569, -0.0471208 , ..., -0.03877235,-0.00586847, -0.02815357],[ 0.02062035, -0.02052468, -0.04067073, ..., -0.04511197,-0.02768868, -0.02311552],[ 0.01646004,  0.05754709, -0.0098597 , ..., -0.0445881 ,0.03163518, -0.04830407]],[[ 0.02346343,  0.01087727, -0.04417856, ..., -0.06175495,0.03087944, -0.02920339],[ 0.01611597, -0.02550997, -0.03644098, ..., -0.05029191,-0.00796291, -0.03109419],[ 0.01516129, -0.00691672, -0.04823716, ..., -0.05618977,-0.04645085, -0.03944302],...[ 0.02979604,  0.00684504, -0.05344562, ..., -0.01344823,-0.01904235,  0.02729847],[ 0.01848961,  0.02351115, -0.05481031, ..., -0.02432707,-0.00973801, -0.00248293],[ 0.00750009, -0.00108958, -0.03491029, ..., -0.04263044,-0.02037425,  0.01850026]],[[ 0.00013985,  0.00641837, -0.03320142, ..., -0.03863316,-0.04950655, -0.05544456],[ 0.02721588,  0.02638515, -0.04679808, ..., -0.05395468,0.01380973, -0.04228394],[ 0.02218842,  0.00352241, -0.05129065, ..., -0.04406777,-0.0157809 , -0.03930759],...,[ 0.01553288,  0.01912074, -0.02032154, ..., -0.05320593,-0.03686063, -0.0361072 ],[ 0.0117848 ,  0.0005205 , -0.02447902, ..., -0.04939574,0.00438235, -0.02167581],[ 0.01059798,  0.01882167, -0.01127519, ..., -0.02081071,-0.01751061, -0.03537183]]], shape=(4, 1000, 59))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>log_uplift</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3011 0.1872 ... 0.4926 0.3069</div><input id='attrs-704828e9-0f6b-45e2-b9ef-697a73aa05bd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-704828e9-0f6b-45e2-b9ef-697a73aa05bd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f1047001-684e-4e58-8b89-9ecbdc5efd96' class='xr-var-data-in' type='checkbox'><label for='data-f1047001-684e-4e58-8b89-9ecbdc5efd96' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.301074  ,  0.18724237,  0.09496671, ...,  0.15507847,0.54038333,  0.38425763],[ 0.26374792,  0.10747885,  0.10508371, ...,  0.25405038,0.51317421,  0.4511275 ],[ 0.29207379,  0.10935514,  0.17063959, ...,  0.36325587,0.51910924,  0.35574941],...,[ 0.29533309,  0.10069385,  0.21538951, ...,  0.21172008,0.46016101,  0.3751204 ],[ 0.26900662,  0.10878439,  0.20233136, ...,  0.1904961 ,0.53216235,  0.39486887],[ 0.25212998,  0.11964854,  0.20148367, ...,  0.24241466,0.42658218,  0.3426603 ]],[[ 0.25286952,  0.01804502,  0.04727177, ...,  0.30531969,0.50785074,  0.5080156 ],[ 0.2837087 ,  0.07109781,  0.20968286, ...,  0.18057831,0.44066755,  0.3046814 ],[ 0.25093898,  0.1215426 ,  0.05558949, ...,  0.29860887,0.49663683,  0.43481916],...[ 0.2714913 ,  0.0071937 ,  0.17071404, ...,  0.21674593,0.53393364,  0.29825509],[ 0.24521288,  0.12215706,  0.18327585, ...,  0.28858252,0.48062656,  0.33844823],[ 0.23209587, -0.0073598 ,  0.26276968, ...,  0.23005166,0.52426397,  0.41382237]],[[ 0.26827074,  0.03251713,  0.08224647, ...,  0.30442411,0.52188658,  0.40432675],[ 0.26908525,  0.18420388,  0.19687624, ...,  0.24599817,0.44669976,  0.31248511],[ 0.30533892,  0.1059248 ,  0.09284101, ...,  0.30075854,0.5289898 ,  0.31019348],...,[ 0.32161296,  0.166329  ,  0.18336166, ...,  0.2311107 ,0.48265756,  0.29864965],[ 0.20758539,  0.09110529,  0.05803997, ...,  0.3114698 ,0.45609364,  0.3914085 ],[ 0.22231227,  0.08463657,  0.20492236, ...,  0.26830526,0.49263361,  0.3068615 ]]], shape=(4, 1000, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>log_state</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.707 0.9937 ... 0.8433 0.3557</div><input id='attrs-313c1cbd-6076-4ce0-b5de-856820514f51' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-313c1cbd-6076-4ce0-b5de-856820514f51' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-24a2b4ea-eea3-4953-b5e7-9b8ff2d3e3ea' class='xr-var-data-in' type='checkbox'><label for='data-24a2b4ea-eea3-4953-b5e7-9b8ff2d3e3ea' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 5.70743828,  0.99371824,  4.29367128, ..., -1.65892661,0.8910402 ,  0.4330478 ],[ 5.67011219,  0.91395471,  4.30378829, ..., -1.5599547 ,0.86383109,  0.49991766],[ 5.69843806,  0.915831  ,  4.36934417, ..., -1.4507492 ,0.86976611,  0.40453957],...,[ 5.70169737,  0.90716971,  4.41409408, ..., -1.602285  ,0.81081788,  0.42391056],[ 5.6753709 ,  0.91526026,  4.40103593, ..., -1.62350897,0.88281922,  0.44365903],[ 5.65849425,  0.92612441,  4.40018824, ..., -1.57159042,0.77723906,  0.39145046]],[[ 5.65923379,  0.82452088,  4.24597635, ..., -1.50868539,0.85850761,  0.55680576],[ 5.69007297,  0.87757367,  4.40838743, ..., -1.63342677,0.79132442,  0.35347157],[ 5.65730325,  0.92801846,  4.25429406, ..., -1.51539621,0.8472937 ,  0.48360933],...[ 5.67785557,  0.81366957,  4.36941862, ..., -1.59725915,0.88459051,  0.34704526],[ 5.65157715,  0.92863293,  4.38198043, ..., -1.52542256,0.83128343,  0.3872384 ],[ 5.63846014,  0.79911606,  4.46147426, ..., -1.58395342,0.87492084,  0.46261254]],[[ 5.67463502,  0.838993  ,  4.28095105, ..., -1.50958097,0.87254345,  0.45311692],[ 5.67544952,  0.99067974,  4.39558082, ..., -1.5680069 ,0.79735663,  0.36127527],[ 5.71170319,  0.91240067,  4.29154559, ..., -1.51324654,0.87964667,  0.35898364],...,[ 5.72797723,  0.97280487,  4.38206624, ..., -1.58289438,0.83331443,  0.34743981],[ 5.61394966,  0.89758116,  4.25674455, ..., -1.50253528,0.80675051,  0.44019866],[ 5.62867654,  0.89111243,  4.40362694, ..., -1.54569982,0.84329048,  0.35565167]]], shape=(4, 1000, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_obs</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.03009 0.1515 ... 0.2535 0.2062</div><input id='attrs-3d802b76-59d7-4493-adee-63479415d51b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3d802b76-59d7-4493-adee-63479415d51b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-57374120-1f06-4af0-9cc2-5911d47fc066' class='xr-var-data-in' type='checkbox'><label for='data-57374120-1f06-4af0-9cc2-5911d47fc066' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.03009243, 0.15146617, 0.16302357, ..., 0.148272  ,0.24278635, 0.19750712],[0.03049958, 0.15351549, 0.16522926, ..., 0.1502781 ,0.24607122, 0.20017936],[0.03068011, 0.15442415, 0.16620725, ..., 0.1511676 ,0.24752771, 0.20136422],...,[0.03402961, 0.17128343, 0.18435295, ..., 0.16767134,0.27455159, 0.2233482 ],[0.03373706, 0.16981091, 0.18276807, ..., 0.16622988,0.27219127, 0.22142807],[0.03196255, 0.16087917, 0.17315481, ..., 0.15748649,0.25787451, 0.20978136]],[[0.02906904, 0.14631506, 0.15747942, ..., 0.14322952,0.2345296 , 0.19079023],[0.02833285, 0.14260955, 0.15349116, ..., 0.13960216,0.22859001, 0.18595837],[0.02859973, 0.14395283, 0.15493694, ..., 0.14091711,0.23074316, 0.18770996],...[0.03096685, 0.15586742, 0.16776065, ..., 0.15258043,0.24984114, 0.2032462 ],[0.03057581, 0.1538992 , 0.16564225, ..., 0.15065372,0.24668627, 0.20067971],[0.03100301, 0.15604942, 0.16795653, ..., 0.15275859,0.25013287, 0.20348352]],[[0.03076355, 0.15484415, 0.1666593 , ..., 0.15157874,0.24820094, 0.20191189],[0.03016507, 0.1518318 , 0.1634171 , ..., 0.14862992,0.24337242, 0.19798388],[0.02958332, 0.1489036 , 0.16026547, ..., 0.14576347,0.23867879, 0.19416561],...,[0.03169247, 0.15951974, 0.17169165, ..., 0.15615573,0.25569548, 0.20800871],[0.03033253, 0.15267466, 0.16432427, ..., 0.14945501,0.24472345, 0.19908295],[0.03141891, 0.15814281, 0.17020966, ..., 0.15480784,0.25348839, 0.20621324]]], shape=(4, 1000, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>expected_pt</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>301.1 2.701 73.23 ... 2.324 1.427</div><input id='attrs-0b7e1333-d3e3-4768-826b-76d984a17527' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0b7e1333-d3e3-4768-826b-76d984a17527' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1574b2be-5023-4865-aa02-cc56c1d62a43' class='xr-var-data-in' type='checkbox'><label for='data-1574b2be-5023-4865-aa02-cc56c1d62a43' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[3.01098747e+02, 2.70125976e+00, 7.32348414e+01, ...,1.90343183e-01, 2.43766399e+00, 1.54194992e+00],[2.90067075e+02, 2.49416677e+00, 7.39795191e+01, ...,2.10145591e-01, 2.37223153e+00, 1.64858552e+00],[2.98400953e+02, 2.49885094e+00, 7.89918095e+01, ...,2.34394613e-01, 2.38635264e+00, 1.49861234e+00],...,[2.99375119e+02, 2.47730113e+00, 8.26069721e+01, ...,2.01435711e-01, 2.24974726e+00, 1.52792493e+00],[2.91596470e+02, 2.49742514e+00, 8.15352901e+01, ...,1.97205495e-01, 2.41770616e+00, 1.55839903e+00],[2.86716593e+02, 2.52470547e+00, 8.14662028e+01, ...,2.07714567e-01, 2.17545765e+00, 1.47912466e+00]],[[2.86928710e+02, 2.28078774e+00, 6.98238991e+01, ...,2.21200579e-01, 2.35963657e+00, 1.74508936e+00],[2.95915214e+02, 2.40505717e+00, 8.21369055e+01, ...,1.95259317e-01, 2.20631659e+00, 1.42400250e+00],[2.86375317e+02, 2.52949193e+00, 7.04070968e+01, ...,2.19721113e-01, 2.33332363e+00, 1.62191788e+00],...2.02450645e-01, 2.42199241e+00, 1.41488076e+00],[2.84740189e+02, 2.53104669e+00, 7.99963037e+01, ...,2.17529119e-01, 2.29626395e+00, 1.47290759e+00],[2.81029639e+02, 2.22357456e+00, 8.66151082e+01, ...,2.05162400e-01, 2.39868540e+00, 1.58821785e+00]],[[2.91381969e+02, 2.31403557e+00, 7.23091768e+01, ...,2.21002566e-01, 2.39298957e+00, 1.57320811e+00],[2.91619398e+02, 2.69306444e+00, 8.10917166e+01, ...,2.08460250e-01, 2.21966577e+00, 1.43515847e+00],[3.02385651e+02, 2.49029373e+00, 7.30793316e+01, ...,2.20193948e-01, 2.41004801e+00, 1.43187338e+00],...,[3.07346947e+02, 2.64535393e+00, 8.00031684e+01, ...,2.05379790e-01, 2.30093241e+00, 1.41543911e+00],[2.74225197e+02, 2.45366090e+00, 7.05798398e+01, ...,2.22565180e-01, 2.24061528e+00, 1.55301571e+00],[2.78293563e+02, 2.43784007e+00, 8.17468221e+01, ...,2.13162644e-01, 2.32400149e+00, 1.42711035e+00]]],shape=(4, 1000, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>expected_upside</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3513 0.2059 ... 0.6366 0.3592</div><input id='attrs-a88dd46e-8d45-40ae-8e5f-03745953200c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a88dd46e-8d45-40ae-8e5f-03745953200c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c00d3489-7e53-4d72-875e-781293c96685' class='xr-var-data-in' type='checkbox'><label for='data-c00d3489-7e53-4d72-875e-781293c96685' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.35130934,  0.20591953,  0.09962224, ...,  0.16774959,0.71666478,  0.46852373],[ 0.3018    ,  0.11346731,  0.11080359, ...,  0.28923676,0.67058558,  0.57008145],[ 0.33920184,  0.11555846,  0.18606321, ...,  0.43800376,0.68053003,  0.42724985],...,[ 0.34357382,  0.105938  ,  0.24034493, ...,  0.23580191,0.58432906,  0.4551666 ],[ 0.30866381,  0.11492194,  0.2242536 , ...,  0.20984966,0.70260997,  0.48418955],[ 0.28676327,  0.12710065,  0.22321626, ...,  0.27432249,0.53201243,  0.40869015]],[[ 0.28771524,  0.01820881,  0.04840689, ...,  0.35705877,0.6617159 ,  0.66198986],[ 0.32804602,  0.07368624,  0.23328687, ...,  0.19790992,0.55374408,  0.35619286],[ 0.28523166,  0.12923747,  0.05716362, ...,  0.34798229,0.64318565,  0.54468369],...[ 0.31191946,  0.00721964,  0.18615151, ...,  0.2420285 ,0.70562846,  0.34750549],[ 0.27789332,  0.12993156,  0.2011457 , ...,  0.33453447,0.61708729,  0.40276913],[ 0.26124064, -0.00733279,  0.30052715, ...,  0.25866503,0.68921507,  0.51258843]],[[ 0.30770114,  0.03305159,  0.08572337, ...,  0.35584396,0.68520392,  0.49829344],[ 0.30876671,  0.20226091,  0.21759334, ...,  0.27889724,0.56314491,  0.36681759],[ 0.35708487,  0.11173827,  0.09728726, ...,  0.35088312,0.69721691,  0.36368894],...,[ 0.37935081,  0.18096158,  0.20124878, ...,  0.25999871,0.62037493,  0.34803725],[ 0.2307028 ,  0.09538433,  0.05975735, ...,  0.36543055,0.57789809,  0.47906258],[ 0.24896133,  0.08832146,  0.22742976, ...,  0.30774628,0.63662077,  0.35915272]]], shape=(4, 1000, 6277))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a42b9c09-7258-48b8-bbff-f6c57fa271b1' class='xr-section-summary-in' type='checkbox' checked /><label for='section-a42b9c09-7258-48b8-bbff-f6c57fa271b1' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:19:20.948093+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>sample_dims :</span></dt><dd>[&#x27;chain&#x27;, &#x27;draw&#x27;]</dd><dt><span>inference_library :</span></dt><dd>nutpie</dd><dt><span>inference_library_version :</span></dt><dd>0.16.10</dd><dt><span>sampling_time :</span></dt><dd>104.98427700996399</dd><dt><span>tuning_steps :</span></dt><dd>1000</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-bcbcfcf9-6918-43a1-b7ae-a8f355850a5c' type='checkbox' checked /><label for='group-bcbcfcf9-6918-43a1-b7ae-a8f355850a5c' title='Expand/collapse group'>/sample_stats<span>(30)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-ff66814a-1510-4c5b-a120-6c4180dd7e84' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-ff66814a-1510-4c5b-a120-6c4180dd7e84' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div></li><li class='xr-section-item'><input id='section-9c088344-4db2-4071-aeb0-d692afabc269' class='xr-section-summary-in' type='checkbox' checked /><label for='section-9c088344-4db2-4071-aeb0-d692afabc269' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-29948087-b903-4f81-8ebc-d977ed56a4df' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-29948087-b903-4f81-8ebc-d977ed56a4df' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f5c4c956-fc07-4a3a-9379-8cd79b3c081a' class='xr-var-data-in' type='checkbox'><label for='data-f5c4c956-fc07-4a3a-9379-8cd79b3c081a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-ca91e007-addf-480e-8b5e-6978f0d8d7f7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ca91e007-addf-480e-8b5e-6978f0d8d7f7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0c3ede2d-451f-4550-886f-4a4564cca003' class='xr-var-data-in' type='checkbox'><label for='data-0c3ede2d-451f-4550-886f-4a4564cca003' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c489cfcf-3228-4cbb-a1d0-1ce204fb88c3' class='xr-section-summary-in' type='checkbox' /><label for='section-c489cfcf-3228-4cbb-a1d0-1ce204fb88c3' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(20)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>depth</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>uint64</div><div class='xr-var-preview xr-preview'>6 6 6 6 6 6 6 6 ... 6 7 6 6 6 6 7 6</div><input id='attrs-8906910c-9cdc-4e6b-b46a-f8301ee2849f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8906910c-9cdc-4e6b-b46a-f8301ee2849f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3fc7844f-1e36-4984-9109-ac0881cda3f9' class='xr-var-data-in' type='checkbox'><label for='data-3fc7844f-1e36-4984-9109-ac0881cda3f9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[6, 6, 6, ..., 6, 6, 6],[6, 6, 6, ..., 6, 7, 6],[6, 6, 6, ..., 6, 6, 6],[6, 6, 6, ..., 6, 7, 6]], shape=(4, 1000), dtype=uint64)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>maxdepth_reached</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-346dde52-a142-47e9-a953-ea0a1cb3bc4d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-346dde52-a142-47e9-a953-ea0a1cb3bc4d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a87a88d9-6179-401e-b5af-f756b9f951f0' class='xr-var-data-in' type='checkbox'><label for='data-a87a88d9-6179-401e-b5af-f756b9f951f0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05878 0.05656 ... 0.05241 0.04879</div><input id='attrs-5bea109b-8e10-45e6-a275-88912c5d370b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5bea109b-8e10-45e6-a275-88912c5d370b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-00e0183e-4cd6-4a67-aa49-008abcfb29fb' class='xr-var-data-in' type='checkbox'><label for='data-00e0183e-4cd6-4a67-aa49-008abcfb29fb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.05877857, 0.05655526, 0.05863391, ..., 0.06208061, 0.05686855,0.06114591],[0.05820219, 0.05897767, 0.05727149, ..., 0.04962718, 0.05271614,0.05898478],[0.05418852, 0.05124771, 0.04966058, ..., 0.05218806, 0.05353007,0.05235783],[0.05677794, 0.04914642, 0.04976723, ..., 0.04848016, 0.05240548,0.04878991]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>transformation_update_id</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0</div><input id='attrs-3e1c26fe-4971-41fc-a969-11a1e1e60c68' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3e1c26fe-4971-41fc-a969-11a1e1e60c68' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b698a9a0-2e6d-4744-b835-80131b58f5d8' class='xr-var-data-in' type='checkbox'><label for='data-b698a9a0-2e6d-4744-b835-80131b58f5d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size_bar</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05685 0.05685 ... 0.05307 0.05307</div><input id='attrs-638e3038-e87f-4789-b346-9ac01375d653' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-638e3038-e87f-4789-b346-9ac01375d653' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0b431f6f-912f-4034-b24c-58161a270004' class='xr-var-data-in' type='checkbox'><label for='data-0b431f6f-912f-4034-b24c-58161a270004' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.05684509, 0.05684509, 0.05684509, ..., 0.05684509, 0.05684509,0.05684509],[0.05390134, 0.05390134, 0.05390134, ..., 0.05390134, 0.05390134,0.05390134],[0.05343729, 0.05343729, 0.05343729, ..., 0.05343729, 0.05343729,0.05343729],[0.05307215, 0.05307215, 0.05307215, ..., 0.05307215, 0.05307215,0.05307215]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>mean_tree_accept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.998 0.9957 ... 0.9999 0.955</div><input id='attrs-de6a875a-4839-4dfc-ba71-2a1e64cf8d46' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de6a875a-4839-4dfc-ba71-2a1e64cf8d46' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-67b9f806-09d9-415d-a5d0-48960f975e27' class='xr-var-data-in' type='checkbox'><label for='data-67b9f806-09d9-415d-a5d0-48960f975e27' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.99804789, 0.99566593, 0.93224505, ..., 0.97859814, 0.95842765,0.96182686],[0.96801482, 0.9647593 , 0.99993972, ..., 0.87973529, 0.99059639,0.98242441],[0.88123203, 0.93262176, 0.99930226, ..., 0.98124511, 0.95213028,0.95362684],[0.990433  , 0.97966927, 0.96806121, ..., 0.96543135, 0.99986591,0.95504737]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>mean_tree_accept_sym</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9798 0.945 ... 0.9576 0.9712</div><input id='attrs-425afe30-2726-4d4e-a7e4-bec9bda251a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-425afe30-2726-4d4e-a7e4-bec9bda251a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-516b0f42-f7f9-4c85-ad92-7b2d208930b5' class='xr-var-data-in' type='checkbox'><label for='data-516b0f42-f7f9-4c85-ad92-7b2d208930b5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.97979213, 0.94502111, 0.96396427, ..., 0.93873196, 0.94545621,0.91901606],[0.97549232, 0.93194847, 0.92279628, ..., 0.91954202, 0.95588313,0.89752155],[0.92452969, 0.95520468, 0.95510541, ..., 0.97782816, 0.97173447,0.96999616],[0.93932468, 0.94524826, 0.95731775, ..., 0.96752079, 0.95764956,0.97117339]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>n_steps</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>uint64</div><div class='xr-var-preview xr-preview'>63 63 63 63 63 ... 63 63 63 127 63</div><input id='attrs-b29037d2-af26-4936-b489-c141cf4f40da' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b29037d2-af26-4936-b489-c141cf4f40da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9b010b95-55a5-4d94-8bbc-15d689112072' class='xr-var-data-in' type='checkbox'><label for='data-9b010b95-55a5-4d94-8bbc-15d689112072' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 63,  63,  63, ...,  63,  63,  63],[ 63,  63,  63, ...,  63, 127,  63],[ 63,  63,  63, ...,  63,  63,  63],[ 63,  63, 127, ...,  63, 127,  63]], shape=(4, 1000), dtype=uint64)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>max_energy_error</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0999 0.2417 ... 0.1931 -0.1618</div><input id='attrs-fa52b5b8-86aa-4e74-ac1c-1f793687dd2e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fa52b5b8-86aa-4e74-ac1c-1f793687dd2e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-52f9658b-26b7-41bc-8e12-457185d6ecb1' class='xr-var-data-in' type='checkbox'><label for='data-52f9658b-26b7-41bc-8e12-457185d6ecb1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.09989733,  0.24169998, -0.17731198, ...,  0.50993265,0.30321111,  0.46681275],[ 0.12276124,  0.46360091,  0.31880901, ..., -0.45700626,0.36727925,  0.58166083],[-0.36643383, -0.26400399,  0.16108273, ...,  0.13728916,-0.18888876, -0.16266756],[ 0.31835335,  0.36390087, -0.56172081, ...,  0.20034757,0.19310595, -0.16175031]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tuning</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-1fb5886b-8aa1-4f26-8547-c0a091138a6f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1fb5886b-8aa1-4f26-8547-c0a091138a6f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bd6a7188-59f7-460b-955f-8669efdc9e34' class='xr-var-data-in' type='checkbox'><label for='data-bd6a7188-59f7-460b-955f-8669efdc9e34' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>index_in_trajectory</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>-31 -12 30 42 3 ... -11 25 53 29</div><input id='attrs-35264a49-8d39-43c2-a1e7-06ad6a351c97' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-35264a49-8d39-43c2-a1e7-06ad6a351c97' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-51d88106-aad8-4a84-9a30-a8c98ba1e6b7' class='xr-var-data-in' type='checkbox'><label for='data-51d88106-aad8-4a84-9a30-a8c98ba1e6b7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-31, -12,  30, ...,  35, -16,  61],[ 24, -37,  47, ..., -46,  71,  55],[ 23, -47, -26, ..., -38,  26, -26],[-30, -52, -39, ...,  25,  53,  29]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-5.199e+03 ... -5.264e+03</div><input id='attrs-6e84c370-4d47-4016-a345-9f79eeff9b9d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6e84c370-4d47-4016-a345-9f79eeff9b9d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c43ec63f-a1d0-447b-9f58-c247aa44284c' class='xr-var-data-in' type='checkbox'><label for='data-c43ec63f-a1d0-447b-9f58-c247aa44284c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-5199.43406279, -5041.63635168, -5155.36845959, ...,-5502.61474463, -5555.40486783, -5553.35125948],[-5103.79550726, -5163.07459768, -5044.35360462, ...,-5365.3674762 , -5282.35587122, -5319.20305154],[-5225.4790853 , -5326.23893749, -5277.95223962, ...,-5210.09166705, -5236.52891958, -5278.11535149],[-5312.31262761, -5200.881231  , -5138.35118481, ...,-5306.2795897 , -5278.15167873, -5264.47831238]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>7.259e+03 7.072e+03 ... 7.28e+03</div><input id='attrs-1e1637d9-e524-4067-95ee-8964522ef5a3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1e1637d9-e524-4067-95ee-8964522ef5a3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-57eb82a1-bd0e-4b52-a9a4-316e2f3bc7d5' class='xr-var-data-in' type='checkbox'><label for='data-57eb82a1-bd0e-4b52-a9a4-316e2f3bc7d5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[7258.92434433, 7072.23742797, 7029.53434387, ..., 7530.53440263,7509.48112764, 7623.68488398],[7125.5291244 , 7125.45732257, 7170.40571382, ..., 7375.20254633,7371.21493424, 7343.81694331],[7265.7710665 , 7251.17197691, 7365.97864737, ..., 7191.11519278,7208.09347637, 7244.77547977],[7430.40168708, 7300.05275956, 7150.24612846, ..., 7415.99064122,7336.22419314, 7279.79585177]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy_error</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.02755 -0.2317 ... 0.03314</div><input id='attrs-ebd0e40d-4d36-4e54-988f-d436e1408e38' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ebd0e40d-4d36-4e54-988f-d436e1408e38' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f73d9948-23ab-437f-b44e-198eccf5c0f5' class='xr-var-data-in' type='checkbox'><label for='data-f73d9948-23ab-437f-b44e-198eccf5c0f5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.02754595, -0.2317194 ,  0.17731198, ...,  0.05378641,0.05376152, -0.27398258],[ 0.04365474, -0.2315725 , -0.2526687 , ...,  0.10268232,-0.17808104, -0.42445925],[ 0.10740746,  0.08138375, -0.14618611, ...,  0.02297692,0.09906807,  0.05545767],[-0.03307493, -0.36390087, -0.47592515, ...,  0.00401974,-0.11224504,  0.03313846]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>fisher_distance</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.834e+03 1.621e+03 ... 1.672e+03</div><input id='attrs-6a314b50-a85f-4ee4-badc-47cae6c2183c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6a314b50-a85f-4ee4-badc-47cae6c2183c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b72c98a5-3e62-490e-b976-3941bbe505f9' class='xr-var-data-in' type='checkbox'><label for='data-b72c98a5-3e62-490e-b976-3941bbe505f9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1833.93523798, 1621.02795564, 1943.9476947 , ..., 2346.58234477,2467.26108287, 2730.33867752],[1818.28565617, 1987.57009203, 1962.75661386, ..., 2130.52048526,3140.31742576, 2292.57008458],[1763.23185206, 1976.04322676, 1533.11801411, ..., 1755.331052  ,1678.26740423, 1642.94271972],[2066.86998493, 1976.90640953, 1945.44198158, ..., 1773.72011502,1814.29798224, 1671.80974291]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>transformation_index</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>848 848 848 848 ... 848 848 848 848</div><input id='attrs-a5917d67-e0fa-41b7-a47a-879262d3bcf0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a5917d67-e0fa-41b7-a47a-879262d3bcf0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0d7d57ff-0927-4661-bfe1-8bdc5d50a598' class='xr-var-data-in' type='checkbox'><label for='data-0d7d57ff-0927-4661-bfe1-8bdc5d50a598' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[848, 848, 848, ..., 848, 848, 848],[848, 848, 848, ..., 848, 848, 848],[847, 847, 847, ..., 847, 847, 847],[848, 848, 848, ..., 848, 848, 848]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-69decd20-3141-45ca-ac70-a317736119a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-69decd20-3141-45ca-ac70-a317736119a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-44c1dc49-998e-428e-ad2a-3d9cd72c0e25' class='xr-var-data-in' type='checkbox'><label for='data-44c1dc49-998e-428e-ad2a-3d9cd72c0e25' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False],[False, False, False, ..., False, False, False]], shape=(4, 1000))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>divergence_draw</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>uint64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0</div><input id='attrs-0cf02a04-44ad-4b3d-a5e3-4f1915025dfa' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0cf02a04-44ad-4b3d-a5e3-4f1915025dfa' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-87ef523e-b3f3-448e-910b-0521f477ff82' class='xr-var-data-in' type='checkbox'><label for='data-87ef523e-b3f3-448e-910b-0521f477ff82' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0]], shape=(4, 1000), dtype=uint64)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>divergence_message</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>None None None ... None None None</div><input id='attrs-c56400a1-1171-41c6-8846-88b8e4ccba00' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c56400a1-1171-41c6-8846-88b8e4ccba00' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8acbcd91-5ebc-43fb-a9b5-278af0f921be' class='xr-var-data-in' type='checkbox'><label for='data-8acbcd91-5ebc-43fb-a9b5-278af0f921be' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[None, None, None, ..., None, None, None],[None, None, None, ..., None, None, None],[None, None, None, ..., None, None, None],[None, None, None, ..., None, None, None]],shape=(4, 1000), dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>divergence_energy_error</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>nan nan nan nan ... nan nan nan nan</div><input id='attrs-c713905d-1a00-463a-bf55-a49e38d5a339' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c713905d-1a00-463a-bf55-a49e38d5a339' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-71d805c7-07e7-46ec-a189-39cf7ae1b4ae' class='xr-var-data-in' type='checkbox'><label for='data-71d805c7-07e7-46ec-a189-39cf7ae1b4ae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[nan, nan, nan, ..., nan, nan, nan],[nan, nan, nan, ..., nan, nan, nan],[nan, nan, nan, ..., nan, nan, nan],[nan, nan, nan, ..., nan, nan, nan]], shape=(4, 1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-3af81080-af71-4caf-9b2a-a15d8facbcc7' class='xr-section-summary-in' type='checkbox' checked /><label for='section-3af81080-af71-4caf-9b2a-a15d8facbcc7' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:19:20.645879+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>sample_dims :</span></dt><dd>[&#x27;chain&#x27;, &#x27;draw&#x27;]</dd><dt><span>inference_library :</span></dt><dd>nutpie</dd><dt><span>inference_library_version :</span></dt><dd>0.16.10</dd><dt><span>inference_library_settings :</span></dt><dd>{&quot;sampler&quot;: &quot;nuts&quot;, &quot;adaptation&quot;: &quot;diag&quot;, &quot;settings&quot;: {&quot;num_tune&quot;: 1000, &quot;num_draws&quot;: 1000, &quot;maxdepth&quot;: 10, &quot;mindepth&quot;: 0, &quot;store_gradient&quot;: false, &quot;store_unconstrained&quot;: false, &quot;store_transformed&quot;: false, &quot;max_energy_error&quot;: 1000.0, &quot;store_divergences&quot;: false, &quot;adapt_options&quot;: {&quot;step_size_settings&quot;: {&quot;target_accept&quot;: 0.95, &quot;initial_step&quot;: 0.1, &quot;jitter&quot;: 0.1, &quot;adapt_options&quot;: {&quot;method&quot;: &quot;DualAverage&quot;, &quot;dual_average&quot;: {&quot;k&quot;: 0.75, &quot;t0&quot;: 10.0, &quot;gamma&quot;: 0.05, &quot;max_step_size&quot;: 3.141592653589793}, &quot;adam&quot;: {&quot;beta1&quot;: 0.9, &quot;beta2&quot;: 0.999, &quot;epsilon&quot;: 1e-08, &quot;learning_rate&quot;: 0.05}}}, &quot;mass_matrix_options&quot;: {&quot;store_mass_matrix&quot;: false, &quot;use_grad_based_estimate&quot;: true}, &quot;early_window&quot;: 0.3, &quot;step_size_window&quot;: 0.15, &quot;mass_matrix_switch_freq&quot;: 80, &quot;early_mass_matrix_switch_freq&quot;: 10, &quot;mass_matrix_update_freq&quot;: 1, &quot;mass_matrix_window_growth&quot;: 1.5}, &quot;check_turning&quot;: true, &quot;target_integration_time&quot;: null, &quot;trajectory_kind&quot;: &quot;Euclidean&quot;, &quot;num_chains&quot;: 4, &quot;seed&quot;: 534082709, &quot;extra_doublings&quot;: 0}}</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-574febd0-89bf-4ef3-8ea5-2f2c78d21a9c' type='checkbox' checked /><label for='group-574febd0-89bf-4ef3-8ea5-2f2c78d21a9c' title='Expand/collapse group'>/constant_data<span>(22)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-1d8fff9f-678e-4f34-a881-2e1dff2ae680' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-1d8fff9f-678e-4f34-a881-2e1dff2ae680' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>isin</span>: 6277</li><li><span class='xr-has-index'>drift_feature</span>: 7</li></ul></div></li><li class='xr-section-item'><input id='section-784a5568-3a22-480a-a230-46ec88ccdd8d' class='xr-section-summary-in' type='checkbox' checked /><label for='section-784a5568-3a22-480a-a230-46ec88ccdd8d' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-7b550bf8-8da4-44d1-9563-92d2eeaecc03' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7b550bf8-8da4-44d1-9563-92d2eeaecc03' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a826ad6e-245f-4b85-a78d-e387f1d4d455' class='xr-var-data-in' type='checkbox'><label for='data-a826ad6e-245f-4b85-a78d-e387f1d4d455' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>drift_feature</span></div><div class='xr-var-dims'>(drift_feature)</div><div class='xr-var-dtype'>&lt;U21</div><div class='xr-var-preview xr-preview'>&#x27;feat_pt_drift&#x27; ... &#x27;feat_total_...</div><input id='attrs-516f6dff-3ef4-47d4-887f-ffdc05205a42' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-516f6dff-3ef4-47d4-887f-ffdc05205a42' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cccac524-6f6e-42d8-a689-26e0d2f13b59' class='xr-var-data-in' type='checkbox'><label for='data-cccac524-6f6e-42d8-a689-26e0d2f13b59' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;feat_pt_drift&#x27;, &#x27;feat_price_drift&#x27;, &#x27;feat_pt_high_drift&#x27;,&#x27;feat_pt_low_drift&#x27;, &#x27;feat_pt_median_drift&#x27;, &#x27;feat_coverage_drift&#x27;,&#x27;feat_total_return_ytd&#x27;], dtype=&#x27;&lt;U21&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8e1cfaba-c1f7-4a21-99c0-6d7a264c8a5b' class='xr-section-summary-in' type='checkbox' checked /><label for='section-8e1cfaba-c1f7-4a21-99c0-6d7a264c8a5b' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(13)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>log_last_price</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.406 0.8065 ... 0.3507 0.04879</div><input id='attrs-3c237913-5ec5-4166-8282-1064963daebb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3c237913-5ec5-4166-8282-1064963daebb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-015b096a-730b-4347-b864-422e8d4b2f82' class='xr-var-data-in' type='checkbox'><label for='data-015b096a-730b-4347-b864-422e8d4b2f82' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 5.40636427,  0.80647587,  4.19870458, ..., -1.81400508,0.35065687,  0.04879016], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>drift_features</span></div><div class='xr-var-dims'>(isin, drift_feature)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6513 0.1998 ... -0.2455 -0.3173</div><input id='attrs-09cedaad-fbe5-4690-99e6-1a4a9cf4dde1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-09cedaad-fbe5-4690-99e6-1a4a9cf4dde1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b50946a9-505c-4399-b5ff-9cbcf38f07ed' class='xr-var-data-in' type='checkbox'><label for='data-b50946a9-505c-4399-b5ff-9cbcf38f07ed' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 6.51344363e-01,  1.99814418e-01,  1.05509829e+00, ...,4.58783001e-01, -1.33537723e-01,  3.37384527e-02],[ 1.08070822e+00,  1.59459656e+00,  6.89505404e-01, ...,1.23487726e+00, -1.95211955e-01, -1.07495794e-01],[ 5.24431914e+00,  4.56115373e+00,  6.80158439e+00, ...,4.40043714e+00,  1.59973114e-01,  9.68449387e-01],...,[-3.52752580e-01, -6.29244007e-01, -2.73738507e-01, ...,-3.58614587e-01,  5.74339579e-03, -4.65431714e-01],[-5.51121746e-01, -1.12324695e+00, -7.78865083e-01, ...,-6.00618031e-01, -8.48316846e-01, -8.74336920e-01],[-4.37020636e-01, -5.62621018e-01, -4.20252438e-01, ...,-4.40978181e-01, -2.45450793e-01, -3.17291951e-01]],shape=(6277, 7))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>feat_pt_range_norm</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.078 1.021 1.065 ... 0.3051 0.4242</div><input id='attrs-73782596-fdf2-4de5-b31d-df4f6211e34b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-73782596-fdf2-4de5-b31d-df4f6211e34b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7725bf9b-8336-4cd5-abd3-433adf83e147' class='xr-var-data-in' type='checkbox'><label for='data-7725bf9b-8336-4cd5-abd3-433adf83e147' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1.0781297 , 1.02062442, 1.06457192, ..., 0.19444444, 0.30508733,0.42424242], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>feat_pt_noise_cv</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.2402 0.6035 ... 0.3828 0.3333</div><input id='attrs-9221b265-7047-4a68-8539-70ebcf1abf2f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9221b265-7047-4a68-8539-70ebcf1abf2f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b92b03ee-95f7-4a0b-a299-8ed6f10a5771' class='xr-var-data-in' type='checkbox'><label for='data-b92b03ee-95f7-4a0b-a299-8ed6f10a5771' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.24024325, 0.60348214, 0.35551952, ..., 0.12883436, 0.3828169 ,0.33333333], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>feat_vol_mean</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>37.05 62.24 106.1 ... 73.12 47.3</div><input id='attrs-75019152-1edc-4b28-8952-33d68870b0f7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-75019152-1edc-4b28-8952-33d68870b0f7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1828c797-1548-4b1a-acfa-e6536702290c' class='xr-var-data-in' type='checkbox'><label for='data-1828c797-1548-4b1a-acfa-e6536702290c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 37.0525,  62.2425, 106.12  , ...,  35.4975,  73.12  ,  47.295 ],shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sqrt_n_analysts</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>7.616 2.449 3.742 ... 1.732 1.414</div><input id='attrs-d923e975-35df-43da-b7be-ae55632dd1f9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d923e975-35df-43da-b7be-ae55632dd1f9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ce4da694-057c-4de1-9b04-5825b4efcc64' class='xr-var-data-in' type='checkbox'><label for='data-ce4da694-057c-4de1-9b04-5825b4efcc64' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([7.61577311, 2.44948974, 3.74165739, ..., 1.41421356, 1.73205081,1.41421356], shape=(6277,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>region_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>4 4 1 4 4 2 4 2 ... 3 0 3 3 3 0 3 3</div><input id='attrs-372a4a17-fab1-4613-9344-7642594d8c43' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-372a4a17-fab1-4613-9344-7642594d8c43' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0f7e4748-c4af-4b55-a31f-b50f338ccaa1' class='xr-var-data-in' type='checkbox'><label for='data-0f7e4748-c4af-4b55-a31f-b50f338ccaa1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([4, 4, 1, ..., 0, 3, 3], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>exchange_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>56 74 56 74 56 ... 10 10 44 10 10</div><input id='attrs-d6f2c726-5950-41e4-b64c-74334dead3fa' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d6f2c726-5950-41e4-b64c-74334dead3fa' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-404dee30-94ad-4ba5-a313-0247d6239159' class='xr-var-data-in' type='checkbox'><label for='data-404dee30-94ad-4ba5-a313-0247d6239159' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([56, 74, 56, ..., 44, 10, 10], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>unit_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>47 6 47 6 47 14 47 ... 5 5 5 33 5 5</div><input id='attrs-86ca446b-9bb8-4f4b-b8f9-753a00163bfd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-86ca446b-9bb8-4f4b-b8f9-753a00163bfd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9911f44f-45a5-4ba3-8cc2-04c41276051d' class='xr-var-data-in' type='checkbox'><label for='data-9911f44f-45a5-4ba3-8cc2-04c41276051d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([47,  6, 47, ..., 33,  5,  5], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>style_class_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>1 0 0 3 0 0 0 0 ... 0 3 3 0 3 3 3 0</div><input id='attrs-fd4abb3c-dcdf-41d2-bf4e-47247b05bf8d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fd4abb3c-dcdf-41d2-bf4e-47247b05bf8d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2e8e733b-0bd2-4a38-a1e6-48131ec6e045' class='xr-var-data-in' type='checkbox'><label for='data-2e8e733b-0bd2-4a38-a1e6-48131ec6e045' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1, 0, 0, ..., 3, 3, 0], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>size_class_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0 2 1 1 0 1 0 1 ... 2 2 2 2 2 2 2 2</div><input id='attrs-b032cb15-e381-4a82-8637-2c3d002bfc3a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b032cb15-e381-4a82-8637-2c3d002bfc3a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3795791c-a973-40f3-b038-211d24a87a9b' class='xr-var-data-in' type='checkbox'><label for='data-3795791c-a973-40f3-b038-211d24a87a9b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 2, 1, ..., 2, 2, 2], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sector_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>6 7 6 3 6 6 0 0 ... 1 5 2 1 4 5 1 0</div><input id='attrs-8ed7d02b-08f3-4b48-bad9-be1b1d0967ad' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8ed7d02b-08f3-4b48-bad9-be1b1d0967ad' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-469f06cf-d14c-4a7f-bc9c-1739fda4eb45' class='xr-var-data-in' type='checkbox'><label for='data-469f06cf-d14c-4a7f-bc9c-1739fda4eb45' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([6, 7, 6, ..., 5, 1, 0], shape=(6277,), dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>industry_idx</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>49 41 50 43 52 32 ... 29 27 7 51 35</div><input id='attrs-2e102578-1e33-4ddc-875a-afdfdeb0932c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2e102578-1e33-4ddc-875a-afdfdeb0932c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e710290c-6b89-489b-88fa-512a3680df61' class='xr-var-data-in' type='checkbox'><label for='data-e710290c-6b89-489b-88fa-512a3680df61' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([49, 41, 50, ...,  7, 51, 35], shape=(6277,), dtype=int32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9603bb47-3904-400c-bf40-f2c5ae38c99a' class='xr-section-summary-in' type='checkbox' checked /><label for='section-9603bb47-3904-400c-bf40-f2c5ae38c99a' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.396870+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[]</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-a0321d7f-9502-4c6c-8d97-bd1bd355cc55' type='checkbox' checked /><label for='group-a0321d7f-9502-4c6c-8d97-bd1bd355cc55' title='Expand/collapse group'>/observed_data<span>(9)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-b6c638f1-4496-4212-9d6e-9ee60fb45fe4' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-b6c638f1-4496-4212-9d6e-9ee60fb45fe4' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>isin</span>: 6277</li></ul></div></li><li class='xr-section-item'><input id='section-de51950b-5d0b-4e21-9673-0bd4f7c05230' class='xr-section-summary-in' type='checkbox' checked /><label for='section-de51950b-5d0b-4e21-9673-0bd4f7c05230' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-c5905919-4fbc-4e45-adbf-c90cc3728e1b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c5905919-4fbc-4e45-adbf-c90cc3728e1b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0919c214-d5ee-46c8-94f3-bc29a3e91861' class='xr-var-data-in' type='checkbox'><label for='data-0919c214-d5ee-46c8-94f3-bc29a3e91861' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e0b08682-04be-4125-8cbd-216568a1c376' class='xr-section-summary-in' type='checkbox' checked /><label for='section-e0b08682-04be-4125-8cbd-216568a1c376' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>log_pt_obs</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.693 1.481 4.38 ... 1.369 0.5008</div><input id='attrs-9acfe8b8-7df1-4d4f-ad56-cdc78b11d295' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9acfe8b8-7df1-4d4f-ad56-cdc78b11d295' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f2d30697-96ec-4ac7-bef6-ab87efcc027f' class='xr-var-data-in' type='checkbox'><label for='data-f2d30697-96ec-4ac7-bef6-ab87efcc027f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 5.69309321,  1.48108168,  4.38007849, ..., -1.53247687,1.36947877,  0.50077529], shape=(6277,))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-15e84e40-2e3c-400b-8c5a-5071fb808b7a' class='xr-section-summary-in' type='checkbox' checked /><label for='section-15e84e40-2e3c-400b-8c5a-5071fb808b7a' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.384284+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[]</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 100%'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-deb68846-eb0a-4b68-8f93-dbf765c29129' type='checkbox' checked /><label for='group-deb68846-eb0a-4b68-8f93-dbf765c29129' title='Expand/collapse group'>/prior<span>(42)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-0362578f-12ee-4ff6-b8a3-c04927340c48' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-0362578f-12ee-4ff6-b8a3-c04927340c48' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1500</li><li><span class='xr-has-index'>isin</span>: 6277</li><li><span class='xr-has-index'>sector</span>: 9</li><li><span class='xr-has-index'>unit</span>: 50</li><li><span class='xr-has-index'>style_class</span>: 4</li><li><span class='xr-has-index'>drift_feature</span>: 7</li><li><span class='xr-has-index'>exchange</span>: 82</li><li><span class='xr-has-index'>size_class</span>: 4</li><li><span class='xr-has-index'>industry</span>: 59</li><li><span class='xr-has-index'>region</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-c35b4c06-a620-478f-8906-343f5f757944' class='xr-section-summary-in' type='checkbox' checked /><label for='section-c35b4c06-a620-478f-8906-343f5f757944' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-e525fb85-f163-4198-aac5-491d2fa8ab6f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e525fb85-f163-4198-aac5-491d2fa8ab6f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ef2c92e1-2a8c-4ba4-a75f-a45d4d973cfb' class='xr-var-data-in' type='checkbox'><label for='data-ef2c92e1-2a8c-4ba4-a75f-a45d4d973cfb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1496 1497 1498 1499</div><input id='attrs-2f318dff-acbf-4a34-b040-cae128122df0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2f318dff-acbf-4a34-b040-cae128122df0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c7dc045c-6e69-42aa-b852-e6fe6f7637de' class='xr-var-data-in' type='checkbox'><label for='data-c7dc045c-6e69-42aa-b852-e6fe6f7637de' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1497, 1498, 1499], shape=(1500,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-e4b8691e-1b00-4d17-ab15-e75dad277c73' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e4b8691e-1b00-4d17-ab15-e75dad277c73' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-44463b82-1ce5-4432-a368-c4603df891b9' class='xr-var-data-in' type='checkbox'><label for='data-44463b82-1ce5-4432-a368-c4603df891b9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sector</span></div><div class='xr-var-dims'>(sector)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;Communication Services&#x27; ... &#x27;Ut...</div><input id='attrs-c4a6aea3-e818-4fea-b6e8-85f9102955fd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c4a6aea3-e818-4fea-b6e8-85f9102955fd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b4a4c8e6-efd4-4448-a2ae-55ad593d8393' class='xr-var-data-in' type='checkbox'><label for='data-b4a4c8e6-efd4-4448-a2ae-55ad593d8393' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Communication Services&#x27;, &#x27;Consumer Discretionary&#x27;, &#x27;Consumer Staples&#x27;,&#x27;Energy&#x27;, &#x27;Health Care&#x27;, &#x27;Industrials&#x27;, &#x27;Information Technology&#x27;,&#x27;Materials&#x27;, &#x27;Utilities&#x27;], dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>unit</span></div><div class='xr-var-dims'>(unit)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;AED&#x27; &#x27;ARS&#x27; &#x27;AUD&#x27; ... &#x27;VND&#x27; &#x27;ZAR&#x27;</div><input id='attrs-bf193290-6c65-4395-871f-ef76dcce8ac8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bf193290-6c65-4395-871f-ef76dcce8ac8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-727ec125-851d-4a8c-8426-57b71a492b7f' class='xr-var-data-in' type='checkbox'><label for='data-727ec125-851d-4a8c-8426-57b71a492b7f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;AED&#x27;, &#x27;ARS&#x27;, &#x27;AUD&#x27;, &#x27;BDT&#x27;, &#x27;BHD&#x27;, &#x27;BRL&#x27;, &#x27;CAD&#x27;, &#x27;CHF&#x27;, &#x27;CLP&#x27;, &#x27;CNY&#x27;,&#x27;COP&#x27;, &#x27;CZK&#x27;, &#x27;DKK&#x27;, &#x27;EGP&#x27;, &#x27;EUR&#x27;, &#x27;GBP&#x27;, &#x27;GHS&#x27;, &#x27;HKD&#x27;, &#x27;HUF&#x27;, &#x27;IDR&#x27;,&#x27;ILS&#x27;, &#x27;INR&#x27;, &#x27;JPY&#x27;, &#x27;KES&#x27;, &#x27;KRW&#x27;, &#x27;KWD&#x27;, &#x27;KZT&#x27;, &#x27;MAD&#x27;, &#x27;MXN&#x27;, &#x27;MYR&#x27;,&#x27;NGN&#x27;, &#x27;NOK&#x27;, &#x27;NZD&#x27;, &#x27;OMR&#x27;, &#x27;PEN&#x27;, &#x27;PHP&#x27;, &#x27;PKR&#x27;, &#x27;PLN&#x27;, &#x27;QAR&#x27;, &#x27;RON&#x27;,&#x27;RSD&#x27;, &#x27;SAR&#x27;, &#x27;SEK&#x27;, &#x27;SGD&#x27;, &#x27;THB&#x27;, &#x27;TRY&#x27;, &#x27;TWD&#x27;, &#x27;USD&#x27;, &#x27;VND&#x27;, &#x27;ZAR&#x27;],dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>style_class</span></div><div class='xr-var-dims'>(style_class)</div><div class='xr-var-dtype'>&lt;U7</div><div class='xr-var-preview xr-preview'>&#x27;Core&#x27; &#x27;Growth&#x27; &#x27;Unknown&#x27; &#x27;Value&#x27;</div><input id='attrs-86eb2a5d-ac69-4413-a959-d24d823d1805' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-86eb2a5d-ac69-4413-a959-d24d823d1805' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7af925de-0281-4d73-af70-c48ae26f77a5' class='xr-var-data-in' type='checkbox'><label for='data-7af925de-0281-4d73-af70-c48ae26f77a5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Core&#x27;, &#x27;Growth&#x27;, &#x27;Unknown&#x27;, &#x27;Value&#x27;], dtype=&#x27;&lt;U7&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>drift_feature</span></div><div class='xr-var-dims'>(drift_feature)</div><div class='xr-var-dtype'>&lt;U21</div><div class='xr-var-preview xr-preview'>&#x27;feat_pt_drift&#x27; ... &#x27;feat_total_...</div><input id='attrs-79049a9a-c837-474b-88f6-d09520a28f14' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-79049a9a-c837-474b-88f6-d09520a28f14' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c75dc549-d103-4a22-a28a-6b988b425e10' class='xr-var-data-in' type='checkbox'><label for='data-c75dc549-d103-4a22-a28a-6b988b425e10' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;feat_pt_drift&#x27;, &#x27;feat_price_drift&#x27;, &#x27;feat_pt_high_drift&#x27;,&#x27;feat_pt_low_drift&#x27;, &#x27;feat_pt_median_drift&#x27;, &#x27;feat_coverage_drift&#x27;,&#x27;feat_total_return_ytd&#x27;], dtype=&#x27;&lt;U21&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>exchange</span></div><div class='xr-var-dims'>(exchange)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;ADX&#x27; &#x27;AIM&#x27; &#x27;ASX&#x27; ... &#x27;XTRA&#x27; &#x27;ZGSE&#x27;</div><input id='attrs-442a0b80-5d3e-48d2-99b2-e9b897240fbb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-442a0b80-5d3e-48d2-99b2-e9b897240fbb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a7b8e3d2-ff4f-493e-bce5-30065f147770' class='xr-var-data-in' type='checkbox'><label for='data-a7b8e3d2-ff4f-493e-bce5-30065f147770' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;ADX&#x27;, &#x27;AIM&#x27;, &#x27;ASX&#x27;, &#x27;ATSE&#x27;, &#x27;BASE&#x27;, &#x27;BAX&#x27;, &#x27;BELEX&#x27;, &#x27;BIT&#x27;, &#x27;BME&#x27;,&#x27;BMV&#x27;, &#x27;BOVESPA&#x27;, &#x27;BSE&#x27;, &#x27;BUL&#x27;, &#x27;BUSE&#x27;, &#x27;BVB&#x27;, &#x27;BVC&#x27;, &#x27;BVL&#x27;, &#x27;CASE&#x27;,&#x27;CBSE&#x27;, &#x27;CNSX&#x27;, &#x27;CPSE&#x27;, &#x27;DB&#x27;, &#x27;DFM&#x27;, &#x27;DSE&#x27;, &#x27;DSM&#x27;, &#x27;ENXTAM&#x27;, &#x27;ENXTBR&#x27;,&#x27;ENXTLS&#x27;, &#x27;ENXTPA&#x27;, &#x27;GHSE&#x27;, &#x27;HLSE&#x27;, &#x27;HOSE&#x27;, &#x27;IBSE&#x27;, &#x27;IDX&#x27;, &#x27;ISE&#x27;, &#x27;JSE&#x27;,&#x27;KAS&#x27;, &#x27;KASE&#x27;, &#x27;KLSE&#x27;, &#x27;KOSDAQ&#x27;, &#x27;KOSE&#x27;, &#x27;KWSE&#x27;, &#x27;LJSE&#x27;, &#x27;LSE&#x27;, &#x27;MSM&#x27;,&#x27;MUN&#x27;, &#x27;NASE&#x27;, &#x27;NGM&#x27;, &#x27;NGSE&#x27;, &#x27;NSEI&#x27;, &#x27;NSEL&#x27;, &#x27;NYSE&#x27;, &#x27;NYSEAM&#x27;, &#x27;NZSE&#x27;,&#x27;NasdaqCM&#x27;, &#x27;NasdaqGM&#x27;, &#x27;NasdaqGS&#x27;, &#x27;OB&#x27;, &#x27;OM&#x27;, &#x27;OTCPK&#x27;, &#x27;PSE&#x27;, &#x27;SASE&#x27;,&#x27;SEHK&#x27;, &#x27;SEP&#x27;, &#x27;SET&#x27;, &#x27;SGX&#x27;, &#x27;SHSE&#x27;, &#x27;SNSE&#x27;, &#x27;SWX&#x27;, &#x27;SZSE&#x27;, &#x27;TASE&#x27;,&#x27;TLSE&#x27;, &#x27;TPEX&#x27;, &#x27;TSE&#x27;, &#x27;TSX&#x27;, &#x27;TSXV&#x27;, &#x27;TWSE&#x27;, &#x27;WBAG&#x27;, &#x27;WSE&#x27;, &#x27;XSAT&#x27;,&#x27;XTRA&#x27;, &#x27;ZGSE&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>size_class</span></div><div class='xr-var-dims'>(size_class)</div><div class='xr-var-dtype'>&lt;U9</div><div class='xr-var-preview xr-preview'>&#x27;Large Cap&#x27; &#x27;Mid Cap&#x27; ... &#x27;Unknown&#x27;</div><input id='attrs-e9ff07ea-1904-49a7-b1a9-f9d8451e119b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9ff07ea-1904-49a7-b1a9-f9d8451e119b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-887bc947-136a-4a90-9512-31b9de8526d0' class='xr-var-data-in' type='checkbox'><label for='data-887bc947-136a-4a90-9512-31b9de8526d0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Large Cap&#x27;, &#x27;Mid Cap&#x27;, &#x27;Small Cap&#x27;, &#x27;Unknown&#x27;], dtype=&#x27;&lt;U9&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>industry</span></div><div class='xr-var-dims'>(industry)</div><div class='xr-var-dtype'>&lt;U53</div><div class='xr-var-preview xr-preview'>&#x27;Aerospace and Defense&#x27; ... &#x27;Wir...</div><input id='attrs-2c6733f6-c253-42cc-9fed-9c39ad8f6693' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2c6733f6-c253-42cc-9fed-9c39ad8f6693' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-501b87c6-6f58-4c12-915b-dcfa6a40b5cf' class='xr-var-data-in' type='checkbox'><label for='data-501b87c6-6f58-4c12-915b-dcfa6a40b5cf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Aerospace and Defense&#x27;, &#x27;Air Freight and Logistics&#x27;,&#x27;Automobile Components&#x27;, &#x27;Automobiles&#x27;, &#x27;Beverages&#x27;, &#x27;Biotechnology&#x27;,&#x27;Broadline Retail&#x27;, &#x27;Building Products&#x27;, &#x27;Chemicals&#x27;,&#x27;Commercial Services and Supplies&#x27;, &#x27;Communications Equipment&#x27;,&#x27;Construction Materials&#x27;, &#x27;Construction and Engineering&#x27;,&#x27;Consumer Staples Distribution and Retail&#x27;, &#x27;Containers and Packaging&#x27;,&#x27;Distributors&#x27;, &#x27;Diversified Consumer Services&#x27;,&#x27;Diversified Telecommunication Services&#x27;, &#x27;Electric Utilities&#x27;,&#x27;Electrical Equipment&#x27;,&#x27;Electronic Equipment Instruments and Components&#x27;,&#x27;Energy Equipment and Services&#x27;, &#x27;Entertainment&#x27;, &#x27;Food Products&#x27;,&#x27;Gas Utilities&#x27;, &#x27;Ground Transportation&#x27;,&#x27;Health Care Equipment and Supplies&#x27;,&#x27;Health Care Providers and Services&#x27;, &#x27;Health Care Technology&#x27;,&#x27;Hotels Restaurants and Leisure&#x27;, &#x27;Household Durables&#x27;,&#x27;Household Products&#x27;, &#x27;IT Services&#x27;,&#x27;Independent Power and Renewable Electricity Producers&#x27;,&#x27;Industrial Conglomerates&#x27;, &#x27;Interactive Media and Services&#x27;,&#x27;Leisure Products&#x27;, &#x27;Life Sciences Tools and Services&#x27;, &#x27;Machinery&#x27;,&#x27;Marine Transportation&#x27;, &#x27;Media&#x27;, &#x27;Metals and Mining&#x27;,&#x27;Multi-Utilities&#x27;, &#x27;Oil Gas and Consumable Fuels&#x27;,&#x27;Paper and Forest Products&#x27;, &#x27;Passenger Airlines&#x27;,&#x27;Personal Care Products&#x27;, &#x27;Pharmaceuticals&#x27;, &#x27;Professional Services&#x27;,&#x27;Semiconductors and Semiconductor Equipment&#x27;, &#x27;Software&#x27;,&#x27;Specialty Retail&#x27;, &#x27;Technology Hardware Storage and Peripherals&#x27;,&#x27;Textiles Apparel and Luxury Goods&#x27;, &#x27;Tobacco&#x27;,&#x27;Trading Companies and Distributors&#x27;, &#x27;Transportation Infrastructure&#x27;,&#x27;Water Utilities&#x27;, &#x27;Wireless Telecommunication Services&#x27;], dtype=&#x27;&lt;U53&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>region</span></div><div class='xr-var-dims'>(region)</div><div class='xr-var-dtype'>&lt;U27</div><div class='xr-var-preview xr-preview'>&#x27;Africa / Middle East&#x27; ... &#x27;Unit...</div><input id='attrs-29dfdcbe-13d9-4e73-a081-1b2cb88af6e4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-29dfdcbe-13d9-4e73-a081-1b2cb88af6e4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bacf9a7a-6bc5-4767-a14a-f8781488765f' class='xr-var-data-in' type='checkbox'><label for='data-bacf9a7a-6bc5-4767-a14a-f8781488765f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Africa / Middle East&#x27;, &#x27;Asia / Pacific&#x27;, &#x27;Europe&#x27;,&#x27;Latin America and Caribbean&#x27;, &#x27;United States and Canada&#x27;], dtype=&#x27;&lt;U27&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4396d058-b30a-42c8-a969-1244fe097ff1' class='xr-section-summary-in' type='checkbox' /><label for='section-4396d058-b30a-42c8-a969-1244fe097ff1' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(24)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>mu_global</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.4414 -0.3835 ... -0.1683</div><input id='attrs-3c86d99b-015f-43dd-92fc-0345d9b39db5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3c86d99b-015f-43dd-92fc-0345d9b39db5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-586d3f7c-bbb2-48ad-afbd-b9a2f882bcf2' class='xr-var-data-in' type='checkbox'><label for='data-586d3f7c-bbb2-48ad-afbd-b9a2f882bcf2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.44139838, -0.38352167, -0.05186509, ...,  0.28637775,-0.03101896, -0.16827452]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_state</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01626 0.06206 ... 0.1402 0.01211</div><input id='attrs-fb532fee-1dc9-4c15-8dae-2e978095400e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fb532fee-1dc9-4c15-8dae-2e978095400e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b101777a-92e3-4125-be09-8d602f1c6e56' class='xr-var-data-in' type='checkbox'><label for='data-b101777a-92e3-4125-be09-8d602f1c6e56' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.016257  , 0.06206468, 0.02081688, ..., 0.31189399, 0.14018164,0.01211298]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>log_state</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.827 0.03068 ... 0.5382 -0.2693</div><input id='attrs-76940e89-5d76-4efc-99e9-5251d37c851b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-76940e89-5d76-4efc-99e9-5251d37c851b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-797970f0-4273-476d-9520-1b6e30bd671b' class='xr-var-data-in' type='checkbox'><label for='data-797970f0-4273-476d-9520-1b6e30bd671b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 4.82654744,  0.03067879,  0.61858701, ..., -1.92416361,0.49053299, -0.24491719],[ 4.38648651,  0.42204694,  3.70580587, ..., -2.30315549,-0.39585681, -0.17398054],[ 5.38620934,  1.44688725,  6.00871443, ..., -2.10774696,-0.12739526, -0.24072439],...,[ 5.66045172,  1.63552688,  8.18038075, ..., -1.8707992 ,0.68686653, -0.16645846],[ 6.41967154,  2.76013035, 12.72100136, ..., -2.8758674 ,-0.47122446, -0.65327001],[ 5.63005786,  0.54761619,  4.5581315 , ..., -2.12035119,0.53823105, -0.26932453]]], shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sector_effect</span></div><div class='xr-var-dims'>(chain, draw, sector)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.00126 -0.006054 ... 0.02049</div><input id='attrs-1c19cf0f-0d6e-47e5-8c01-ef60659cb854' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c19cf0f-0d6e-47e5-8c01-ef60659cb854' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ad872f8e-6dab-4703-971c-e162512fc08a' class='xr-var-data-in' type='checkbox'><label for='data-ad872f8e-6dab-4703-971c-e162512fc08a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.00125968, -0.00605355,  0.00259047, ..., -0.00479616,-0.00073402,  0.00394246],[ 0.15392323, -0.04563323, -0.1117736 , ..., -0.15814968,0.01150599,  0.27295372],[ 0.0677385 ,  0.00919636,  0.00153917, ...,  0.03837203,0.01328903, -0.00943675],...,[ 0.0625874 , -0.09199525,  0.05662662, ...,  0.09001267,0.14669085,  0.14425024],[ 0.00988178, -0.03331046, -0.08775964, ...,  0.05642218,-0.19560978, -0.00047782],[-0.17526482,  0.24461852, -0.04510214, ...,  0.04214796,-0.12522805,  0.02048864]]], shape=(1, 1500, 9))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_region</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1799 0.1776 ... 0.03868 0.0553</div><input id='attrs-e637e3c8-b436-4e62-baa3-7813e3c1d01e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e637e3c8-b436-4e62-baa3-7813e3c1d01e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6900a09d-4ed4-4e2a-b707-56e12d29a101' class='xr-var-data-in' type='checkbox'><label for='data-6900a09d-4ed4-4e2a-b707-56e12d29a101' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.17989763, 0.1775819 , 0.09371284, ..., 0.05291322, 0.03868458,0.05530334]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_unit</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1254 0.06063 ... 0.1205 0.2454</div><input id='attrs-84692edb-2482-4177-ba0b-b6187c4ab07e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-84692edb-2482-4177-ba0b-b6187c4ab07e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-943f2dbf-6aba-4721-9495-dc3ae6e34438' class='xr-var-data-in' type='checkbox'><label for='data-943f2dbf-6aba-4721-9495-dc3ae6e34438' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.12544944, 0.06062894, 0.13401776, ..., 0.01166517, 0.12052824,0.24543335]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>unit_effect</span></div><div class='xr-var-dims'>(chain, draw, unit)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05248 0.07597 ... 0.4127 0.03195</div><input id='attrs-ea0059d0-804c-43db-97bd-a5e939fe6151' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ea0059d0-804c-43db-97bd-a5e939fe6151' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-04ea5648-52eb-4aab-9704-5e6f6c5042dc' class='xr-var-data-in' type='checkbox'><label for='data-04ea5648-52eb-4aab-9704-5e6f6c5042dc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.05247926,  0.07596919,  0.00361142, ..., -0.01249912,-0.03232716, -0.19940422],[-0.11003622,  0.03250761,  0.07708302, ..., -0.04513167,0.07916604,  0.05274733],[ 0.1070126 ,  0.11179468, -0.04971972, ..., -0.0700918 ,0.24596961, -0.19209791],...,[-0.00118017, -0.00320749, -0.01157246, ..., -0.00667415,0.00592526, -0.02132042],[-0.0729396 , -0.18662049, -0.11141227, ...,  0.21808834,-0.07843459,  0.13918112],[ 0.3981236 , -0.17541456,  0.02132575, ...,  0.07227097,0.4126904 ,  0.03195034]]], shape=(1, 1500, 50))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>style_class_effect</span></div><div class='xr-var-dims'>(chain, draw, style_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.01656 0.01412 ... 0.1503 -0.2253</div><input id='attrs-e352c8c2-740e-4d1b-aba2-7f482bc8bd67' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e352c8c2-740e-4d1b-aba2-7f482bc8bd67' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9695456c-b931-4293-ab63-eaaabe6392ca' class='xr-var-data-in' type='checkbox'><label for='data-9695456c-b931-4293-ab63-eaaabe6392ca' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-1.65641471e-02,  1.41197938e-02,  7.26548454e-03,-8.32592108e-03],[ 7.06504265e-02, -1.74870973e-01, -4.10731917e-02,-1.20639557e-01],[-5.19347106e-02,  2.92288799e-02,  7.81167659e-02,-2.71915418e-01],...,[-3.36597013e-05, -3.68094217e-03, -5.18727590e-02,7.77647863e-03],[-3.93696443e-02,  2.82102106e-02,  6.26906886e-03,-3.66974306e-02],[ 1.95103615e-04,  4.09881636e-02,  1.50303579e-01,-2.25290761e-01]]], shape=(1, 1500, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_style_class</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01004 0.146 ... 0.06792 0.1538</div><input id='attrs-ddacd92f-e3a5-4e64-a04e-8ef086fffda7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ddacd92f-e3a5-4e64-a04e-8ef086fffda7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-337892a2-3406-4482-b066-703a616c4fdc' class='xr-var-data-in' type='checkbox'><label for='data-337892a2-3406-4482-b066-703a616c4fdc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.01004253, 0.14601151, 0.15816199, ..., 0.01614866, 0.06792142,0.15380362]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_size_class</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.00408 0.1292 ... 0.04307 0.07004</div><input id='attrs-8e3c92df-cd52-4a05-8570-fd9daf89ab4a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8e3c92df-cd52-4a05-8570-fd9daf89ab4a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b65d010f-45ac-4a32-9869-9c8b232fc3e6' class='xr-var-data-in' type='checkbox'><label for='data-b65d010f-45ac-4a32-9869-9c8b232fc3e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00407954, 0.12922916, 0.11060499, ..., 0.02912281, 0.04306771,0.07003899]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta</span></div><div class='xr-var-dims'>(chain, draw, drift_feature)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.2863 -0.1277 ... -0.06361</div><input id='attrs-f5181bc1-6bb7-44d7-9136-5f7b42fb3ae2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f5181bc1-6bb7-44d7-9136-5f7b42fb3ae2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a04fdbd3-d502-4919-8846-78ab9c23524a' class='xr-var-data-in' type='checkbox'><label for='data-a04fdbd3-d502-4919-8846-78ab9c23524a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.28633977, -0.12772186, -0.26067095, ...,  0.27570941,-0.52859973, -0.21709405],[-0.14692776,  0.06730456, -0.0352118 , ...,  0.45728052,0.04257582, -0.28578557],[-0.00435123,  0.0149144 ,  0.21851569, ...,  0.01944651,0.30619205, -0.28948858],...,[-0.04463661,  0.05364938,  0.43685085, ...,  0.14729928,-0.17153808, -0.05723801],[ 0.35677264,  0.28801532,  0.41417638, ...,  0.2469996 ,-0.19856741,  0.04392637],[-0.14249218, -0.45485842,  0.29137746, ...,  0.18105849,-0.12219156, -0.06360512]]], shape=(1, 1500, 7))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>expected_upside</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.44 -0.5397 ... 0.2063 -0.2725</div><input id='attrs-08666337-852b-4bd8-bbbc-0d15e9016c9b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-08666337-852b-4bd8-bbbc-0d15e9016c9b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fb6193ec-0c69-485b-bc77-ec6baa9c4c00' class='xr-var-data-in' type='checkbox'><label for='data-fb6193ec-0c69-485b-bc77-ec6baa9c4c00' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-4.39999070e-01, -5.39663286e-01, -9.72127579e-01, ...,-1.04307872e-01,  1.50131311e-01, -2.54505376e-01],[-6.39360977e-01, -3.19160665e-01, -3.89146858e-01, ...,-3.86852904e-01, -5.25983751e-01, -1.99701664e-01],[-1.99531783e-02,  8.97261228e-01,  5.11050763e+00, ...,-2.54531113e-01, -3.80010122e-01, -2.51373105e-01],...,[ 2.89284548e-01,  1.29114344e+00,  5.26068133e+01, ...,-5.52114361e-02,  3.99632433e-01, -1.93659057e-01],[ 1.75469648e+00,  6.05442077e+00,  5.02458318e+03, ...,-6.54188802e-01, -5.60396164e-01, -5.04436698e-01],[ 2.50687731e-01, -2.28068664e-01,  4.32508241e-01, ...,-2.63868210e-01,  2.06319734e-01, -2.72480660e-01]]],shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_obs</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1653 0.8318 ... 1.619 1.317</div><input id='attrs-21ab13f5-4446-48e0-9692-8b4c542b3ff8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-21ab13f5-4446-48e0-9692-8b4c542b3ff8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dd1ffb4a-0bdd-4da2-b985-42d62f50d086' class='xr-var-data-in' type='checkbox'><label for='data-dd1ffb4a-0bdd-4da2-b985-42d62f50d086' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.16526528, 0.83184032, 0.89531265, ..., 0.81429819,1.33336358, 1.08469358],[0.25067599, 1.26174352, 1.35801898, ..., 1.23513544,2.0224589 , 1.64527381],[0.27325173, 1.37537546, 1.48032142, ..., 1.34637107,2.20460044, 1.79344627],...,[0.20028538, 1.00810925, 1.08503152, ..., 0.9868499 ,1.61590647, 1.31454271],[0.06590696, 0.33173373, 0.35704618, ..., 0.32473802,0.53173868, 0.43257033],[0.20063236, 1.00985573, 1.08691127, ..., 0.98855955,1.61870592, 1.31682007]]], shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_sector</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.003107 0.1531 ... 0.1355 0.1423</div><input id='attrs-a6e6a01c-9bf6-4379-abc2-3395b285352e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a6e6a01c-9bf6-4379-abc2-3395b285352e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7ae9d762-07c6-4e85-a70f-3a84ed0b7eb9' class='xr-var-data-in' type='checkbox'><label for='data-7ae9d762-07c6-4e85-a70f-3a84ed0b7eb9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00310676, 0.15314449, 0.03178781, ..., 0.13144191, 0.13552164,0.1423259 ]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_obs_base</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06038 0.09159 ... 0.02408 0.0733</div><input id='attrs-8f6a8c25-65d2-4f7d-b976-85f4aa6c34d3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8f6a8c25-65d2-4f7d-b976-85f4aa6c34d3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cf7813ac-a491-47de-8346-ba2234fdcf8c' class='xr-var-data-in' type='checkbox'><label for='data-cf7813ac-a491-47de-8346-ba2234fdcf8c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.06038118, 0.09158676, 0.09983501, ..., 0.07317609, 0.02407971,0.07330286]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>exchange_effect</span></div><div class='xr-var-dims'>(chain, draw, exchange)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.2779 -0.01252 ... -0.1403</div><input id='attrs-c839d68c-9eae-4b06-ae3b-be818559727e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c839d68c-9eae-4b06-ae3b-be818559727e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1ca0d7f8-d7c0-49ed-994e-bc584e78fc34' class='xr-var-data-in' type='checkbox'><label for='data-1ca0d7f8-d7c0-49ed-994e-bc584e78fc34' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.27788533, -0.01251816,  0.2638478 , ...,  0.05503112,0.0820753 ,  0.00080582],[-0.00523331,  0.04071827, -0.01091674, ..., -0.00270655,0.00995012,  0.01353106],[ 0.00268005,  0.01575376, -0.03018436, ...,  0.00100361,-0.00450001,  0.00823011],...,[-0.03970775,  0.13143627, -0.01688807, ..., -0.06713058,-0.01603182,  0.02507021],[ 0.02805582,  0.05284884,  0.01173012, ...,  0.00092914,-0.04900476, -0.04629647],[-0.1752962 , -0.12346944,  0.12007152, ...,  0.12148574,0.00086379, -0.14030409]]], shape=(1, 1500, 82))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>size_class_effect</span></div><div class='xr-var-dims'>(chain, draw, size_class)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.001401 -0.006774 ... -0.03278</div><input id='attrs-bab1e61f-71f5-4246-915e-adae0307a4a4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bab1e61f-71f5-4246-915e-adae0307a4a4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ebfe25f5-a1df-4ae8-8057-5180a5ddc454' class='xr-var-data-in' type='checkbox'><label for='data-ebfe25f5-a1df-4ae8-8057-5180a5ddc454' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.00140051, -0.0067741 , -0.00157271,  0.00534046],[ 0.02269266, -0.15438131, -0.04151551,  0.16312906],[-0.13765149, -0.01107429,  0.1795758 , -0.13194805],...,[ 0.00592603, -0.02290676,  0.02751711,  0.02959185],[ 0.00724211,  0.01252838,  0.04103109, -0.00667568],[-0.13214169,  0.0785207 , -0.00914956, -0.03277782]]],shape=(1, 1500, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>expected_pt</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>124.8 1.031 1.856 ... 1.713 0.7639</div><input id='attrs-6f18de6f-3461-4d73-8092-b63f82301b87' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6f18de6f-3461-4d73-8092-b63f82301b87' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4fc1010f-5a9c-402f-9329-2da7615794cd' class='xr-var-data-in' type='checkbox'><label for='data-4fc1010f-5a9c-402f-9329-2da7615794cd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[1.24779407e+02, 1.03115424e+00, 1.85630324e+00, ...,1.45997817e-01, 1.63318646e+00, 7.82769356e-01],[8.03575871e+01, 1.52508011e+00, 4.06828193e+01, ...,9.99429767e-02, 6.73103074e-01, 8.40313252e-01],[2.18374033e+02, 4.24986515e+00, 4.06959808e+02, ...,1.21511429e-01, 8.80385626e-01, 7.86058240e-01],...,[2.87278383e+02, 5.13216131e+00, 3.57021377e+03, ...,1.54000536e-01, 1.98747806e+00, 8.46657990e-01],[6.13801471e+02, 1.58019025e+01, 3.34703840e+05, ...,5.63672253e-02, 6.24237447e-01, 5.20341468e-01],[2.78678240e+02, 1.72912619e+00, 9.54050489e+01, ...,1.19989482e-01, 1.71297402e+00, 7.63895307e-01]]],shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>log_uplift</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.5798 -0.7758 ... 0.1876 -0.3181</div><input id='attrs-a1366cf1-89f6-4247-8c3a-fc351a30371e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a1366cf1-89f6-4247-8c3a-fc351a30371e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a5691a51-00e7-4743-89cb-d9d330534aa2' class='xr-var-data-in' type='checkbox'><label for='data-a5691a51-00e7-4743-89cb-d9d330534aa2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.57981683, -0.77579707, -3.58011757, ..., -0.11015853,0.13987612, -0.29370736],[-1.01987776, -0.38442893, -0.49289871, ..., -0.48915041,-0.74651368, -0.2227707 ],[-0.02015493,  0.64041139,  1.81000985, ..., -0.29374188,-0.47805213, -0.28951456],...,[ 0.25408745,  0.82905101,  3.98167617, ..., -0.05679412,0.33620965, -0.21524862],[ 1.01330727,  1.95365448,  8.52229678, ..., -1.06186232,-0.82188133, -0.70206018],[ 0.22369358, -0.25885968,  0.35942692, ..., -0.30634611,0.18757418, -0.3181147 ]]], shape=(1, 1500, 6277))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>nu</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>36.22 20.02 9.606 ... 9.037 4.163</div><input id='attrs-419cbec3-29fe-42fe-adc2-7a7b7fd78192' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-419cbec3-29fe-42fe-adc2-7a7b7fd78192' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8b7552f0-077a-4d0e-8ee7-e8339270ab2c' class='xr-var-data-in' type='checkbox'><label for='data-8b7552f0-077a-4d0e-8ee7-e8339270ab2c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[36.22277864, 20.0192584 ,  9.60568985, ..., 16.98088256,9.03656591,  4.16316687]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>industry_effect</span></div><div class='xr-var-dims'>(chain, draw, industry)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1301 0.06239 ... -0.007945</div><input id='attrs-7e243dc4-6cf4-4137-b8dc-bd27823dfba6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7e243dc4-6cf4-4137-b8dc-bd27823dfba6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ce81230f-4d51-436d-ad49-24e6d0261183' class='xr-var-data-in' type='checkbox'><label for='data-ce81230f-4d51-436d-ad49-24e6d0261183' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 1.30055345e-01,  6.23917667e-02,  2.34435091e-01, ...,-6.68829598e-02, -3.89289109e-01, -3.93957314e-02],[ 1.37834808e-01,  4.84188475e-02,  4.80425673e-02, ...,-5.27088874e-03,  3.06490710e-02,  2.93458857e-02],[-6.27467600e-02, -7.63162237e-02, -1.83194942e-01, ...,2.13054322e-01,  1.80996865e-02,  1.00817479e-01],...,[ 5.90430351e-04, -3.61391062e-03, -1.33223993e-03, ...,1.01645755e-03,  1.01409447e-03,  2.46889389e-04],[-1.10204162e-02, -1.93582261e-01,  1.12906197e-01, ...,-4.78675852e-02,  2.90176807e-02, -7.98962616e-02],[-4.54257305e-03, -2.44990633e-03,  7.64082801e-03, ...,3.84776728e-03,  7.09755464e-02, -7.94528343e-03]]],shape=(1, 1500, 59))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_exchange</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1919 0.02467 ... 0.03263 0.1059</div><input id='attrs-e4c1d24e-9f43-4e6c-9b84-b71237d1b2b4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e4c1d24e-9f43-4e6c-9b84-b71237d1b2b4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c357c44d-cdd8-4330-94ad-4185bf50b99b' class='xr-var-data-in' type='checkbox'><label for='data-c357c44d-cdd8-4330-94ad-4185bf50b99b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.19189022, 0.02467022, 0.01149717, ..., 0.05692886, 0.03263098,0.10586986]], shape=(1, 1500))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>region_effect</span></div><div class='xr-var-dims'>(chain, draw, region)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.06427 0.2888 ... 0.104 -0.03712</div><input id='attrs-c05a6ad1-0092-447a-ac46-e6671b2ada22' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c05a6ad1-0092-447a-ac46-e6671b2ada22' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-053e6bc7-0180-4b97-bce5-932ad809fe07' class='xr-var-data-in' type='checkbox'><label for='data-053e6bc7-0180-4b97-bce5-932ad809fe07' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.06426998,  0.28882507, -0.10830278, -0.06709542,0.29087117],[ 0.04071519, -0.09597525,  0.18594158, -0.05201069,0.00470652],[-0.01624447,  0.10874697,  0.06180368, -0.01908726,-0.07538356],...,[ 0.01212273,  0.03703203,  0.02278174, -0.00245837,-0.1244127 ],[ 0.01937367, -0.02019158,  0.07647142,  0.03932423,-0.00287219],[ 0.00306565, -0.00401059,  0.08037764,  0.1040281 ,-0.03712096]]], shape=(1, 1500, 5))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_industry</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1386 0.05796 ... 0.06928 0.07764</div><input id='attrs-026ae0cd-e07d-4a97-b928-ce4efa8a8753' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-026ae0cd-e07d-4a97-b928-ce4efa8a8753' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f94a0bb1-603d-4787-8831-3ca57e12fbb1' class='xr-var-data-in' type='checkbox'><label for='data-f94a0bb1-603d-4787-8831-3ca57e12fbb1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.13855186, 0.05796484, 0.13190118, ..., 0.00170642, 0.06927729,0.0776376 ]], shape=(1, 1500))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4bd905ce-0db7-496b-bb59-ca63cf1e37b8' class='xr-section-summary-in' type='checkbox' checked /><label for='section-4bd905ce-0db7-496b-bb59-ca63cf1e37b8' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.376301+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[&#x27;chain&#x27;, &#x27;draw&#x27;]</dd></dl></div></li></ul></div></div><div class='xr-group-box'><div class='xr-group-box-vline' style='height: 1.2em'></div><div class='xr-group-box-hline'></div><div class='xr-group-box-contents'><input id='group-b96a0e35-1ad0-4464-bf64-74dae8d2a4d8' type='checkbox' checked /><label for='group-b96a0e35-1ad0-4464-bf64-74dae8d2a4d8' title='Expand/collapse group'>/prior_predictive<span>(11)</span></label><ul class='xr-sections'><li class='xr-section-item'><input id='section-1b6b7a46-c11f-4292-8637-9f1c9cbbb747' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-1b6b7a46-c11f-4292-8637-9f1c9cbbb747' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 1500</li><li><span class='xr-has-index'>isin</span>: 6277</li></ul></div></li><li class='xr-section-item'><input id='section-d8157729-9aa5-4e0f-9aef-2a06cc16223f' class='xr-section-summary-in' type='checkbox' checked /><label for='section-d8157729-9aa5-4e0f-9aef-2a06cc16223f' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-44eeb050-0d2d-432e-b15c-6842f2252670' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-44eeb050-0d2d-432e-b15c-6842f2252670' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2b488a5a-c35e-4bab-8693-cc6b4fea1cc5' class='xr-var-data-in' type='checkbox'><label for='data-2b488a5a-c35e-4bab-8693-cc6b4fea1cc5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1496 1497 1498 1499</div><input id='attrs-af335470-d48a-4413-adac-5bec7837a813' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-af335470-d48a-4413-adac-5bec7837a813' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a4f2bb58-6245-4fb7-9519-a47175d86dd9' class='xr-var-data-in' type='checkbox'><label for='data-a4f2bb58-6245-4fb7-9519-a47175d86dd9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1497, 1498, 1499], shape=(1500,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>isin</span></div><div class='xr-var-dims'>(isin)</div><div class='xr-var-dtype'>&lt;U12</div><div class='xr-var-preview xr-preview'>&#x27;US67066G1040&#x27; ... &#x27;BRENJUACNOR9&#x27;</div><input id='attrs-37742a13-9662-42f8-9279-d40d21784b91' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-37742a13-9662-42f8-9279-d40d21784b91' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ece92150-9a67-4dc0-b4ca-7f5dd800949e' class='xr-var-data-in' type='checkbox'><label for='data-ece92150-9a67-4dc0-b4ca-7f5dd800949e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;US67066G1040&#x27;, &#x27;CA44955L1067&#x27;, &#x27;AU0000185993&#x27;, ..., &#x27;OM0000002168&#x27;,&#x27;BRLJQQACNOR5&#x27;, &#x27;BRENJUACNOR9&#x27;], shape=(6277,), dtype=&#x27;&lt;U12&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-63bbd060-37bf-4f47-ace3-a2281f8df16f' class='xr-section-summary-in' type='checkbox' checked /><label for='section-63bbd060-37bf-4f47-ace3-a2281f8df16f' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>log_pt_obs</span></div><div class='xr-var-dims'>(chain, draw, isin)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.067 -0.5607 ... 0.2002 1.486</div><input id='attrs-87e9a1ab-de1c-400b-8c9f-4ae8832f61e7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-87e9a1ab-de1c-400b-8c9f-4ae8832f61e7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e8c87a72-8bdb-42d8-beac-5bffd0fd3df1' class='xr-var-data-in' type='checkbox'><label for='data-e8c87a72-8bdb-42d8-beac-5bffd0fd3df1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 5.06734454, -0.56068546, -0.04740253, ..., -2.22850073,1.98605573, -1.28864302],[ 3.89250203,  0.60248521,  0.88996491, ..., -5.73717132,-0.18319716,  3.21042313],[ 4.99575714,  1.4900476 ,  6.7636942 , ..., -0.89771605,0.83761525, -1.14712391],...,[ 5.74314423,  1.94852195,  7.6408595 , ..., -3.46205783,-0.70500567, -0.18354805],[ 6.45156177,  2.87144397, 13.66273868, ..., -2.10221018,-0.60370203, -1.15757288],[ 5.61742055,  1.78070719,  5.68124411, ..., -4.6252466 ,0.20019369,  1.48596324]]], shape=(1, 1500, 6277))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-86fccefb-2ee9-4f37-b879-6a7acaab928f' class='xr-section-summary-in' type='checkbox' checked /><label for='section-86fccefb-2ee9-4f37-b879-6a7acaab928f' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2026-06-03T12:17:12.382459+00:00</dd><dt><span>creation_library :</span></dt><dd>ArviZ</dd><dt><span>creation_library_version :</span></dt><dd>1.1.0</dd><dt><span>creation_library_language :</span></dt><dd>Python</dd><dt><span>inference_library :</span></dt><dd>pymc</dd><dt><span>inference_library_version :</span></dt><dd>6.0.1</dd><dt><span>sample_dims :</span></dt><dd>[&#x27;chain&#x27;, &#x27;draw&#x27;]</dd></dl></div></li></ul></div></div></div></li></ul></div></div>



## 8. Posterior Predictive Checks


```python
with kalman_pt_model:
    pm.sample_posterior_predictive(
        idata, extend_inferencedata=True,
        random_seed=RANDOM_SEED, progressbar=True,
    )

# (a) Distributional overlay — replicated log(observed_pt) draws vs the observed
#     ECDF. A good fit hugs the observed step function.
pc_ppc = azp.plot_ppc_dist(
    idata,
    group="posterior_predictive",
    var_names=["log_pt_obs"],
    kind="ecdf",
    num_samples=500,
    backend="matplotlib",
)
pc_ppc.show()

# (b) Calibration — posterior-predictive PIT ECDF. If the model is calibrated the
#     PIT values are ~Uniform(0,1), so the curve tracks the diagonal and stays
#     inside the simultaneous confidence envelope. Systematic excursions flag
#     over-/under-dispersion of the measurement-noise model.
try:
    pc_pit = azp.plot_ppc_pit(
        idata,
        var_names=["log_pt_obs"],
        backend="matplotlib",
    )
    pc_pit.show()
except Exception as e:  # pragma: no cover - diagnostic is best-effort
    print(f"PPC PIT calibration plot skipped: {e!r}")
```

    Sampling: [log_pt_obs]
    


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_26_3.png)
    



    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_26_4.png)
    


## 9. MCMC Diagnostics


```python
# 9.1 R-hat / ESS summary across the cross-sectional parameter set.
posterior = idata.posterior
requested = ['mu_global', 'beta', 'sigma_state', 'sigma_obs_base', 'nu']
for _grp in GROUP_EFFECTS:
    requested.extend([f'sigma_{_grp}', f'{_grp}_effect'])

available, skipped = [], []
for v in requested:
    if v not in posterior.data_vars:
        skipped.append((v, 'not in posterior'))
        continue
    da = posterior[v]
    non_sample_sizes = [da.sizes[d] for d in da.dims if d not in ('chain', 'draw')]
    if any(s == 0 for s in non_sample_sizes):
        skipped.append((v, f'empty dim(s): {dict(da.sizes)}'))
        continue
    available.append(v)

if skipped:
    print('Skipping variables:')
    for name, reason in skipped:
        print(f'  - {name}: {reason}')
if not available:
    raise RuntimeError('No non-empty variables to summarise.')

summary = azs.summary(idata, var_names=available, round_to=4)
summary.sort_values('r_hat', ascending=False).head(50)
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
      <th>mean</th>
      <th>sd</th>
      <th>eti89_lb</th>
      <th>eti89_ub</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_global</th>
      <td>0.1236</td>
      <td>0.0488</td>
      <td>0.0429</td>
      <td>0.1954</td>
      <td>16.2624</td>
      <td>38.1322</td>
      <td>1.1819</td>
      <td>0.0123</td>
      <td>0.0086</td>
    </tr>
    <tr>
      <th>size_class_effect[Small Cap]</th>
      <td>0.0317</td>
      <td>0.0349</td>
      <td>-0.0278</td>
      <td>0.0819</td>
      <td>33.2218</td>
      <td>103.8579</td>
      <td>1.1244</td>
      <td>0.0056</td>
      <td>0.0056</td>
    </tr>
    <tr>
      <th>size_class_effect[Large Cap]</th>
      <td>-0.0285</td>
      <td>0.0350</td>
      <td>-0.0885</td>
      <td>0.0216</td>
      <td>33.2618</td>
      <td>96.7945</td>
      <td>1.1239</td>
      <td>0.0056</td>
      <td>0.0056</td>
    </tr>
    <tr>
      <th>size_class_effect[Mid Cap]</th>
      <td>-0.0178</td>
      <td>0.0349</td>
      <td>-0.0783</td>
      <td>0.0324</td>
      <td>33.5448</td>
      <td>106.1957</td>
      <td>1.1231</td>
      <td>0.0056</td>
      <td>0.0056</td>
    </tr>
    <tr>
      <th>exchange_effect[NYSE]</th>
      <td>-0.0487</td>
      <td>0.0272</td>
      <td>-0.0907</td>
      <td>-0.0016</td>
      <td>103.0000</td>
      <td>63.5649</td>
      <td>1.0649</td>
      <td>0.0028</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>exchange_effect[NasdaqGS]</th>
      <td>-0.0404</td>
      <td>0.0274</td>
      <td>-0.0830</td>
      <td>0.0067</td>
      <td>105.6681</td>
      <td>65.2861</td>
      <td>1.0615</td>
      <td>0.0028</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>region_effect[Asia / Pacific]</th>
      <td>0.0564</td>
      <td>0.0271</td>
      <td>0.0155</td>
      <td>0.0987</td>
      <td>61.2296</td>
      <td>201.9614</td>
      <td>1.0583</td>
      <td>0.0031</td>
      <td>0.0030</td>
    </tr>
    <tr>
      <th>unit_effect[USD]</th>
      <td>0.0973</td>
      <td>0.0309</td>
      <td>0.0453</td>
      <td>0.1432</td>
      <td>100.0256</td>
      <td>54.5761</td>
      <td>1.0549</td>
      <td>0.0032</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>sector_effect[Consumer Discretionary]</th>
      <td>0.0041</td>
      <td>0.0176</td>
      <td>-0.0248</td>
      <td>0.0306</td>
      <td>98.5100</td>
      <td>208.1119</td>
      <td>1.0521</td>
      <td>0.0018</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>region_effect[United States and Canada]</th>
      <td>-0.0328</td>
      <td>0.0255</td>
      <td>-0.0740</td>
      <td>0.0045</td>
      <td>76.4949</td>
      <td>161.9344</td>
      <td>1.0513</td>
      <td>0.0027</td>
      <td>0.0027</td>
    </tr>
    <tr>
      <th>exchange_effect[NasdaqGM]</th>
      <td>0.0828</td>
      <td>0.0314</td>
      <td>0.0338</td>
      <td>0.1359</td>
      <td>139.2288</td>
      <td>127.9744</td>
      <td>1.0496</td>
      <td>0.0028</td>
      <td>0.0021</td>
    </tr>
    <tr>
      <th>sigma_region</th>
      <td>0.0500</td>
      <td>0.0243</td>
      <td>0.0242</td>
      <td>0.0930</td>
      <td>123.7857</td>
      <td>215.4736</td>
      <td>1.0467</td>
      <td>0.0020</td>
      <td>0.0027</td>
    </tr>
    <tr>
      <th>exchange_effect[NasdaqCM]</th>
      <td>0.1125</td>
      <td>0.0346</td>
      <td>0.0585</td>
      <td>0.1689</td>
      <td>149.7796</td>
      <td>114.7961</td>
      <td>1.0462</td>
      <td>0.0029</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>sector_effect[Consumer Staples]</th>
      <td>-0.0056</td>
      <td>0.0198</td>
      <td>-0.0398</td>
      <td>0.0242</td>
      <td>95.8099</td>
      <td>447.6904</td>
      <td>1.0442</td>
      <td>0.0020</td>
      <td>0.0015</td>
    </tr>
    <tr>
      <th>unit_effect[INR]</th>
      <td>-0.0656</td>
      <td>0.0441</td>
      <td>-0.1362</td>
      <td>0.0044</td>
      <td>123.1605</td>
      <td>253.8679</td>
      <td>1.0439</td>
      <td>0.0040</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>unit_effect[CAD]</th>
      <td>0.1034</td>
      <td>0.0484</td>
      <td>0.0274</td>
      <td>0.1837</td>
      <td>179.3193</td>
      <td>243.1106</td>
      <td>1.0419</td>
      <td>0.0036</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>sector_effect[Information Technology]</th>
      <td>-0.0117</td>
      <td>0.0193</td>
      <td>-0.0429</td>
      <td>0.0179</td>
      <td>158.9363</td>
      <td>367.4995</td>
      <td>1.0409</td>
      <td>0.0015</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>sector_effect[Communication Services]</th>
      <td>-0.0052</td>
      <td>0.0198</td>
      <td>-0.0377</td>
      <td>0.0248</td>
      <td>116.7544</td>
      <td>429.4140</td>
      <td>1.0407</td>
      <td>0.0018</td>
      <td>0.0014</td>
    </tr>
    <tr>
      <th>sector_effect[Materials]</th>
      <td>0.0036</td>
      <td>0.0197</td>
      <td>-0.0279</td>
      <td>0.0340</td>
      <td>165.5016</td>
      <td>249.4154</td>
      <td>1.0393</td>
      <td>0.0015</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>exchange_effect[TSX]</th>
      <td>-0.0559</td>
      <td>0.0460</td>
      <td>-0.1305</td>
      <td>0.0167</td>
      <td>249.4124</td>
      <td>357.8650</td>
      <td>1.0388</td>
      <td>0.0029</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>region_effect[Latin America and Caribbean]</th>
      <td>0.0010</td>
      <td>0.0283</td>
      <td>-0.0433</td>
      <td>0.0444</td>
      <td>120.6435</td>
      <td>196.2619</td>
      <td>1.0385</td>
      <td>0.0026</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>region_effect[Europe]</th>
      <td>-0.0157</td>
      <td>0.0260</td>
      <td>-0.0587</td>
      <td>0.0221</td>
      <td>97.6033</td>
      <td>184.0611</td>
      <td>1.0385</td>
      <td>0.0026</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>exchange_effect[BSE]</th>
      <td>-0.0422</td>
      <td>0.0425</td>
      <td>-0.1135</td>
      <td>0.0223</td>
      <td>121.4929</td>
      <td>149.5488</td>
      <td>1.0357</td>
      <td>0.0039</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>exchange_effect[NSEI]</th>
      <td>-0.0266</td>
      <td>0.0425</td>
      <td>-0.0985</td>
      <td>0.0386</td>
      <td>120.7204</td>
      <td>145.1500</td>
      <td>1.0355</td>
      <td>0.0039</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>sector_effect[Industrials]</th>
      <td>-0.0138</td>
      <td>0.0167</td>
      <td>-0.0405</td>
      <td>0.0122</td>
      <td>133.5317</td>
      <td>276.7565</td>
      <td>1.0346</td>
      <td>0.0015</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>region_effect[Africa / Middle East]</th>
      <td>0.0030</td>
      <td>0.0293</td>
      <td>-0.0423</td>
      <td>0.0470</td>
      <td>135.0398</td>
      <td>274.5045</td>
      <td>1.0340</td>
      <td>0.0025</td>
      <td>0.0024</td>
    </tr>
    <tr>
      <th>sigma_size_class</th>
      <td>0.0639</td>
      <td>0.0375</td>
      <td>0.0218</td>
      <td>0.1358</td>
      <td>105.3586</td>
      <td>134.9015</td>
      <td>1.0338</td>
      <td>0.0031</td>
      <td>0.0030</td>
    </tr>
    <tr>
      <th>sigma_unit</th>
      <td>0.0704</td>
      <td>0.0158</td>
      <td>0.0475</td>
      <td>0.0943</td>
      <td>95.3151</td>
      <td>110.8374</td>
      <td>1.0338</td>
      <td>0.0017</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>style_class_effect[Growth]</th>
      <td>-0.0125</td>
      <td>0.0194</td>
      <td>-0.0458</td>
      <td>0.0090</td>
      <td>131.3854</td>
      <td>179.2882</td>
      <td>1.0335</td>
      <td>0.0017</td>
      <td>0.0027</td>
    </tr>
    <tr>
      <th>sigma_exchange</th>
      <td>0.0668</td>
      <td>0.0108</td>
      <td>0.0509</td>
      <td>0.0849</td>
      <td>116.4662</td>
      <td>152.3937</td>
      <td>1.0312</td>
      <td>0.0010</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>beta[feat_pt_drift]</th>
      <td>0.0328</td>
      <td>0.0276</td>
      <td>-0.0097</td>
      <td>0.0794</td>
      <td>139.8828</td>
      <td>252.2921</td>
      <td>1.0310</td>
      <td>0.0023</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>sigma_style_class</th>
      <td>0.0259</td>
      <td>0.0262</td>
      <td>0.0048</td>
      <td>0.0784</td>
      <td>144.3259</td>
      <td>293.9190</td>
      <td>1.0304</td>
      <td>0.0019</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <th>sector_effect[Utilities]</th>
      <td>-0.0509</td>
      <td>0.0207</td>
      <td>-0.0858</td>
      <td>-0.0204</td>
      <td>182.5278</td>
      <td>313.3544</td>
      <td>1.0297</td>
      <td>0.0015</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>style_class_effect[Core]</th>
      <td>0.0007</td>
      <td>0.0190</td>
      <td>-0.0307</td>
      <td>0.0237</td>
      <td>137.7752</td>
      <td>176.0908</td>
      <td>1.0287</td>
      <td>0.0016</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>unit_effect[EUR]</th>
      <td>0.0016</td>
      <td>0.0250</td>
      <td>-0.0376</td>
      <td>0.0406</td>
      <td>149.1033</td>
      <td>296.8532</td>
      <td>1.0286</td>
      <td>0.0020</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>exchange_effect[NGSE]</th>
      <td>-0.1283</td>
      <td>0.0671</td>
      <td>-0.2385</td>
      <td>-0.0237</td>
      <td>173.3876</td>
      <td>328.0127</td>
      <td>1.0267</td>
      <td>0.0051</td>
      <td>0.0039</td>
    </tr>
    <tr>
      <th>style_class_effect[Value]</th>
      <td>0.0032</td>
      <td>0.0190</td>
      <td>-0.0283</td>
      <td>0.0272</td>
      <td>152.1789</td>
      <td>171.0819</td>
      <td>1.0249</td>
      <td>0.0016</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>unit_effect[JPY]</th>
      <td>-0.0390</td>
      <td>0.0478</td>
      <td>-0.1123</td>
      <td>0.0385</td>
      <td>153.9112</td>
      <td>230.9431</td>
      <td>1.0242</td>
      <td>0.0039</td>
      <td>0.0030</td>
    </tr>
    <tr>
      <th>beta[feat_pt_median_drift]</th>
      <td>0.0700</td>
      <td>0.0197</td>
      <td>0.0363</td>
      <td>0.0994</td>
      <td>168.7851</td>
      <td>301.0341</td>
      <td>1.0242</td>
      <td>0.0015</td>
      <td>0.0010</td>
    </tr>
    <tr>
      <th>unit_effect[NGN]</th>
      <td>-0.1413</td>
      <td>0.0695</td>
      <td>-0.2548</td>
      <td>-0.0302</td>
      <td>182.9056</td>
      <td>238.1610</td>
      <td>1.0236</td>
      <td>0.0051</td>
      <td>0.0035</td>
    </tr>
    <tr>
      <th>unit_effect[TRY]</th>
      <td>0.1073</td>
      <td>0.0594</td>
      <td>0.0124</td>
      <td>0.2044</td>
      <td>189.4055</td>
      <td>235.8213</td>
      <td>1.0229</td>
      <td>0.0043</td>
      <td>0.0029</td>
    </tr>
    <tr>
      <th>unit_effect[GBP]</th>
      <td>0.1194</td>
      <td>0.0435</td>
      <td>0.0496</td>
      <td>0.1871</td>
      <td>157.0282</td>
      <td>141.0818</td>
      <td>1.0228</td>
      <td>0.0036</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>beta[feat_pt_high_drift]</th>
      <td>0.0531</td>
      <td>0.0097</td>
      <td>0.0379</td>
      <td>0.0686</td>
      <td>222.6740</td>
      <td>508.0343</td>
      <td>1.0227</td>
      <td>0.0007</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>exchange_effect[ENXTPA]</th>
      <td>0.0204</td>
      <td>0.0196</td>
      <td>-0.0118</td>
      <td>0.0523</td>
      <td>304.8759</td>
      <td>583.4254</td>
      <td>1.0226</td>
      <td>0.0011</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>sector_effect[Health Care]</th>
      <td>0.0546</td>
      <td>0.0186</td>
      <td>0.0262</td>
      <td>0.0851</td>
      <td>311.5080</td>
      <td>488.0740</td>
      <td>1.0224</td>
      <td>0.0011</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>exchange_effect[TSE]</th>
      <td>-0.0386</td>
      <td>0.0492</td>
      <td>-0.1151</td>
      <td>0.0370</td>
      <td>148.8162</td>
      <td>282.4942</td>
      <td>1.0220</td>
      <td>0.0040</td>
      <td>0.0030</td>
    </tr>
    <tr>
      <th>exchange_effect[XTRA]</th>
      <td>0.0569</td>
      <td>0.0201</td>
      <td>0.0257</td>
      <td>0.0892</td>
      <td>294.1035</td>
      <td>564.4692</td>
      <td>1.0207</td>
      <td>0.0012</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>exchange_effect[TSXV]</th>
      <td>0.1174</td>
      <td>0.0539</td>
      <td>0.0349</td>
      <td>0.2091</td>
      <td>310.7398</td>
      <td>428.0628</td>
      <td>1.0207</td>
      <td>0.0031</td>
      <td>0.0023</td>
    </tr>
    <tr>
      <th>exchange_effect[IBSE]</th>
      <td>0.0962</td>
      <td>0.0562</td>
      <td>0.0074</td>
      <td>0.1865</td>
      <td>225.3645</td>
      <td>321.9352</td>
      <td>1.0195</td>
      <td>0.0037</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>exchange_effect[SZSE]</th>
      <td>-0.0056</td>
      <td>0.0395</td>
      <td>-0.0704</td>
      <td>0.0544</td>
      <td>214.2632</td>
      <td>275.3884</td>
      <td>1.0183</td>
      <td>0.0027</td>
      <td>0.0018</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 9.2 Divergences and aggregated R-hat / ESS.
n_div = int(idata.sample_stats['diverging'].sum())


def _non_empty_vars(ds):
    keep = []
    for name, da in ds.data_vars.items():
        sizes = [da.sizes[d] for d in da.dims if d not in ('chain', 'draw')]
        if all(s > 0 for s in sizes):
            keep.append(name)
    return keep


posterior_tree = idata.posterior
posterior = posterior_tree.dataset if hasattr(posterior_tree, "dataset") else posterior_tree.to_dataset()
keep_vars = _non_empty_vars(posterior)
rhat_ds = azs.rhat(posterior[keep_vars])
ess_ds = azs.ess(posterior[keep_vars], method='bulk')

max_rhat = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
min_ess = float(min(float(ess_ds[v].min()) for v in ess_ds.data_vars))

_grp_keys = [f'sigma_{g}' for g in GROUP_EFFECTS if f'sigma_{g}' in rhat_ds.data_vars]
_grp_report = {v: (float(rhat_ds[v].max()), float(ess_ds[v].min())) for v in _grp_keys}

print(f'Divergences: {n_div}')
print(f'Max R-hat:   {max_rhat:.4f}')
print(f'Min ESS:     {min_ess:.1f}')
if _grp_report:
    print('Group-effect scale diagnostics (max R-hat, min ESS):')
    for v, (r, e) in _grp_report.items():
        print(f'  - {v:>20s}: r_hat={r:.3f}, ess_bulk={e:.1f}')
```

    Divergences: 0
    Max R-hat:   1.1819
    Min ESS:     16.3
    Group-effect scale diagnostics (max R-hat, min ESS):
      -         sigma_region: r_hat=1.047, ess_bulk=123.8
      -       sigma_exchange: r_hat=1.031, ess_bulk=116.5
      -           sigma_unit: r_hat=1.034, ess_bulk=95.3
      -    sigma_style_class: r_hat=1.030, ess_bulk=144.3
      -     sigma_size_class: r_hat=1.034, ess_bulk=105.4
      -         sigma_sector: r_hat=1.015, ess_bulk=233.4
      -       sigma_industry: r_hat=1.008, ess_bulk=613.0
    


```python
# 9.3 Trace + marginal densities for the key scalar / vector parameters.
# Uses arviz_plots.plot_trace (rank-normalised marginals + per-chain traces).
#
# arviz_plots.plot_trace (v1.1.0) can crash when a single call mixes variables
# whose non-sample dimensions differ: the per-chain aesthetic gets broadcast
# across the extra dim and reshape fails with e.g.
# `cannot reshape array of size 12 into shape (4,)` (4 chains x 3 vector coords).
# We therefore (a) classify each variable from its *actual* posterior shape
# rather than a hard-coded list, plotting any variable that carries an extra
# (vector) dim on its own, and (b) keep a defensive fallback that re-plots the
# scalars one figure per variable if the combined call still raises.
post_trace = idata.posterior
_requested = ['mu_global', 'beta', 'sigma_state', 'sigma_obs_base', 'nu',
              *(f'sigma_{g}' for g in GROUP_EFFECTS)]
_trace_vars = [v for v in _requested if v in post_trace.data_vars]

def _extra_dims(_v):
    return [d for d in post_trace[_v].dims if d not in ('chain', 'draw')]

scalar_vars = [v for v in _trace_vars if not _extra_dims(v)]
vector_vars = [v for v in _trace_vars if _extra_dims(v)]

def _show_trace(_vars):
    pc = azp.plot_trace(idata, var_names=_vars, backend='matplotlib')
    pc.show()

if scalar_vars:
    try:
        _show_trace(scalar_vars)
    except ValueError as exc:
        print(f'Combined scalar trace failed ({exc}); plotting per variable.')
        for _sv in scalar_vars:
            _show_trace([_sv])
for _vv in vector_vars:
    _show_trace([_vv])
if not scalar_vars and not vector_vars:
    print('No trace-eligible variables available.')
```


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_30_0.png)
    



    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_30_1.png)
    



```python
# 9.4 Forest plot of the hierarchical group-effect scales and drift slopes.
import arviz_plots as azp

_forest_vars = [f'sigma_{g}' for g in GROUP_EFFECTS if f'sigma_{g}' in idata.posterior.data_vars]
_forest_vars += [v for v in ('beta',) if v in idata.posterior.data_vars]

if _forest_vars:
    azp.plot_forest(idata, var_names=_forest_vars, combined=True)
    plt.title('Group-effect scales (sigma_<coord>) and drift slopes (beta)')
    plt.tight_layout()
    plt.show()
else:
    print('No group-effect / beta variables in posterior - skipped.')
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_12952\905307574.py:10: UserWarning: The figure layout has changed to tight
      plt.tight_layout()
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_31_1.png)
    


## 10. Expected Price Targets — Posterior Summary

`expected_pt` is the posterior-mean Kalman-smoothed price target (price units);
`expected_upside_pct` is the implied move vs `last_price`. The 94% HDI gives the
credible band around each smoothed target.


```python
# Posterior expected price target (smoothed) + implied upside per ISIN.
post = idata.posterior


def _post_mean(v):
    return post[v].mean(('chain', 'draw')).values


def _post_hdi(v, p=0.94):
    da = post[v].stack(s=('chain', 'draw'))
    lo = da.quantile((1 - p) / 2, dim='s').values
    hi = da.quantile(1 - (1 - p) / 2, dim='s').values
    return lo, hi


exp_pt = _post_mean('expected_pt')
exp_up = _post_mean('expected_upside')
pt_lo, pt_hi = _post_hdi('expected_pt')

results = pd.DataFrame({
    'isin': isin_labels,
    'ticker': model_df.get('ticker'),
    'sector': model_df.get('sector'),
    'last_price': model_df['last_price'].to_numpy(),
    'observed_pt': model_df['observed_pt'].to_numpy(),
    'expected_pt': exp_pt,
    'expected_pt_hdi_lo': pt_lo,
    'expected_pt_hdi_hi': pt_hi,
    'expected_upside_pct': exp_up * 100,
    'n_analysts': model_df['n_analysts'].to_numpy(),
})
results = results.sort_values('expected_upside_pct', ascending=False).reset_index(drop=True)
print(f'Expected price targets for {len(results)} ISINs.')
results.head(50)
```

    Expected price targets for 6277 ISINs.
    




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
      <th>sector</th>
      <th>last_price</th>
      <th>observed_pt</th>
      <th>expected_pt</th>
      <th>expected_pt_hdi_lo</th>
      <th>expected_pt_hdi_hi</th>
      <th>expected_upside_pct</th>
      <th>n_analysts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US9290332074</td>
      <td>VOR</td>
      <td>Health Care</td>
      <td>13.660</td>
      <td>37.8750</td>
      <td>99.392935</td>
      <td>83.904922</td>
      <td>117.180828</td>
      <td>627.620317</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US75955J4022</td>
      <td>RLMD</td>
      <td>Health Care</td>
      <td>6.420</td>
      <td>12.8000</td>
      <td>21.235235</td>
      <td>18.026537</td>
      <td>25.155795</td>
      <td>230.766905</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LU2458332611</td>
      <td>ALVO</td>
      <td>Health Care</td>
      <td>3.000</td>
      <td>13.8333</td>
      <td>8.614146</td>
      <td>7.493189</td>
      <td>9.865607</td>
      <td>187.138209</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CH0363463438</td>
      <td>IDIA</td>
      <td>Health Care</td>
      <td>4.236</td>
      <td>4.5000</td>
      <td>11.868453</td>
      <td>10.469841</td>
      <td>13.440788</td>
      <td>180.180674</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KR7000720003</td>
      <td>A000720</td>
      <td>Industrials</td>
      <td>134200.000</td>
      <td>286050.0000</td>
      <td>356467.232253</td>
      <td>288673.937851</td>
      <td>448879.100474</td>
      <td>165.623869</td>
      <td>20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SE0009581051</td>
      <td>ISOFOL</td>
      <td>Health Care</td>
      <td>0.760</td>
      <td>3.6000</td>
      <td>1.900456</td>
      <td>1.676154</td>
      <td>2.154640</td>
      <td>150.060023</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CNE100006QF7</td>
      <td>2432</td>
      <td>Industrials</td>
      <td>32.220</td>
      <td>59.6841</td>
      <td>77.121865</td>
      <td>68.480255</td>
      <td>86.899275</td>
      <td>139.360228</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NL0015002J37</td>
      <td>JBS</td>
      <td>Consumer Staples</td>
      <td>12.200</td>
      <td>19.4746</td>
      <td>28.760310</td>
      <td>24.413543</td>
      <td>33.523115</td>
      <td>135.740246</td>
      <td>15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>US09077V1008</td>
      <td>BIOA</td>
      <td>Health Care</td>
      <td>16.540</td>
      <td>49.5000</td>
      <td>38.790300</td>
      <td>34.289957</td>
      <td>43.566467</td>
      <td>134.524184</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>KR7298380007</td>
      <td>A298380</td>
      <td>Health Care</td>
      <td>107100.000</td>
      <td>223200.0000</td>
      <td>242088.857502</td>
      <td>211967.779561</td>
      <td>274153.273329</td>
      <td>126.040016</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>US92259N3026</td>
      <td>VELO</td>
      <td>Industrials</td>
      <td>22.130</td>
      <td>22.5000</td>
      <td>48.929891</td>
      <td>43.272614</td>
      <td>55.181454</td>
      <td>121.102085</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>US1079241022</td>
      <td>BBOT</td>
      <td>Health Care</td>
      <td>8.270</td>
      <td>25.1111</td>
      <td>17.563212</td>
      <td>15.654771</td>
      <td>19.733442</td>
      <td>112.372571</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12</th>
      <td>KYG622681008</td>
      <td>1860</td>
      <td>Communication Services</td>
      <td>14.920</td>
      <td>17.9331</td>
      <td>31.378873</td>
      <td>27.523866</td>
      <td>35.532493</td>
      <td>110.314161</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GB00BMCLYF79</td>
      <td>4BB</td>
      <td>Health Care</td>
      <td>5.500</td>
      <td>17.0000</td>
      <td>11.494296</td>
      <td>10.217638</td>
      <td>12.913435</td>
      <td>108.987193</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LU2356314745</td>
      <td>NVM</td>
      <td>Consumer Discretionary</td>
      <td>2.600</td>
      <td>14.3000</td>
      <td>5.413632</td>
      <td>4.310857</td>
      <td>6.647396</td>
      <td>108.216609</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>KYG8875G1029</td>
      <td>1530</td>
      <td>Health Care</td>
      <td>16.870</td>
      <td>37.5950</td>
      <td>35.040886</td>
      <td>31.399172</td>
      <td>38.752635</td>
      <td>107.711238</td>
      <td>11</td>
    </tr>
    <tr>
      <th>16</th>
      <td>JP3891600003</td>
      <td>7003</td>
      <td>Industrials</td>
      <td>4366.000</td>
      <td>7920.0000</td>
      <td>9030.019881</td>
      <td>8096.060843</td>
      <td>10035.149717</td>
      <td>106.825925</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>US0008991046</td>
      <td>ADMA</td>
      <td>Health Care</td>
      <td>7.600</td>
      <td>16.7500</td>
      <td>15.618115</td>
      <td>13.988473</td>
      <td>17.349914</td>
      <td>105.501512</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>US92337R1014</td>
      <td>VERA</td>
      <td>Health Care</td>
      <td>31.260</td>
      <td>78.0000</td>
      <td>64.161466</td>
      <td>57.041356</td>
      <td>71.855511</td>
      <td>105.251009</td>
      <td>14</td>
    </tr>
    <tr>
      <th>19</th>
      <td>AU000000EOS8</td>
      <td>EOS</td>
      <td>Industrials</td>
      <td>12.260</td>
      <td>12.9375</td>
      <td>24.817274</td>
      <td>21.666058</td>
      <td>28.116910</td>
      <td>102.424749</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>US71742Q1067</td>
      <td>PAHC</td>
      <td>Health Care</td>
      <td>28.520</td>
      <td>45.6000</td>
      <td>57.555935</td>
      <td>51.215435</td>
      <td>64.198589</td>
      <td>101.809029</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KR7214450009</td>
      <td>A214450</td>
      <td>Health Care</td>
      <td>290500.000</td>
      <td>516214.2857</td>
      <td>583348.120683</td>
      <td>516073.363314</td>
      <td>660685.580521</td>
      <td>100.808303</td>
      <td>14</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CNE100005MX1</td>
      <td>2145</td>
      <td>Consumer Staples</td>
      <td>38.020</td>
      <td>90.0707</td>
      <td>76.333799</td>
      <td>67.978870</td>
      <td>85.261820</td>
      <td>100.772749</td>
      <td>9</td>
    </tr>
    <tr>
      <th>23</th>
      <td>US82835W1080</td>
      <td>SPRY</td>
      <td>Health Care</td>
      <td>8.900</td>
      <td>28.8000</td>
      <td>17.809681</td>
      <td>15.863057</td>
      <td>19.913671</td>
      <td>100.108780</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CNE1000048G6</td>
      <td>9995</td>
      <td>Health Care</td>
      <td>76.700</td>
      <td>116.5480</td>
      <td>152.194106</td>
      <td>134.869859</td>
      <td>171.280245</td>
      <td>98.427779</td>
      <td>15</td>
    </tr>
    <tr>
      <th>25</th>
      <td>US3847471014</td>
      <td>GRAL</td>
      <td>Health Care</td>
      <td>59.900</td>
      <td>66.8571</td>
      <td>118.751905</td>
      <td>105.245805</td>
      <td>133.136870</td>
      <td>98.250258</td>
      <td>7</td>
    </tr>
    <tr>
      <th>26</th>
      <td>US23284F1057</td>
      <td>CTMX</td>
      <td>Health Care</td>
      <td>3.160</td>
      <td>12.4444</td>
      <td>6.250107</td>
      <td>5.578547</td>
      <td>6.991135</td>
      <td>97.788192</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27</th>
      <td>FR0010417345</td>
      <td>DBV</td>
      <td>Health Care</td>
      <td>3.050</td>
      <td>3.8000</td>
      <td>6.020276</td>
      <td>5.313455</td>
      <td>6.799354</td>
      <td>97.386089</td>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>SE0008241491</td>
      <td>SYNACT</td>
      <td>Health Care</td>
      <td>13.920</td>
      <td>34.6500</td>
      <td>27.409848</td>
      <td>24.553544</td>
      <td>30.427281</td>
      <td>96.909826</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>US20337X1090</td>
      <td>VISN</td>
      <td>Information Technology</td>
      <td>12.440</td>
      <td>17.3333</td>
      <td>24.302787</td>
      <td>21.213584</td>
      <td>27.708738</td>
      <td>95.360027</td>
      <td>3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>US0080642061</td>
      <td>JBIO</td>
      <td>Health Care</td>
      <td>17.750</td>
      <td>49.0000</td>
      <td>34.602719</td>
      <td>30.642535</td>
      <td>39.031715</td>
      <td>94.944896</td>
      <td>7</td>
    </tr>
    <tr>
      <th>31</th>
      <td>US48138M1053</td>
      <td>JMIA</td>
      <td>Consumer Discretionary</td>
      <td>6.880</td>
      <td>14.7270</td>
      <td>13.392405</td>
      <td>11.965012</td>
      <td>14.910515</td>
      <td>94.657056</td>
      <td>5</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CNE100003N76</td>
      <td>2696</td>
      <td>Health Care</td>
      <td>58.650</td>
      <td>103.1987</td>
      <td>113.727323</td>
      <td>102.336364</td>
      <td>126.464813</td>
      <td>93.908480</td>
      <td>6</td>
    </tr>
    <tr>
      <th>33</th>
      <td>BMG864081044</td>
      <td>SLP</td>
      <td>Materials</td>
      <td>0.950</td>
      <td>1.6982</td>
      <td>1.832886</td>
      <td>1.641603</td>
      <td>2.047727</td>
      <td>92.935350</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>SG9999014716</td>
      <td>WVE</td>
      <td>Health Care</td>
      <td>5.690</td>
      <td>21.9375</td>
      <td>10.945790</td>
      <td>9.724057</td>
      <td>12.261069</td>
      <td>92.368895</td>
      <td>16</td>
    </tr>
    <tr>
      <th>35</th>
      <td>CNE100003FF7</td>
      <td>1877</td>
      <td>Health Care</td>
      <td>18.690</td>
      <td>29.9005</td>
      <td>35.904894</td>
      <td>32.313108</td>
      <td>40.032717</td>
      <td>92.107513</td>
      <td>4</td>
    </tr>
    <tr>
      <th>36</th>
      <td>GB00B0394F60</td>
      <td>MTL</td>
      <td>Materials</td>
      <td>0.137</td>
      <td>0.3300</td>
      <td>0.263101</td>
      <td>0.235290</td>
      <td>0.294389</td>
      <td>92.044374</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37</th>
      <td>GB00BKPH9R58</td>
      <td>LBG</td>
      <td>Communication Services</td>
      <td>0.362</td>
      <td>1.1500</td>
      <td>0.695104</td>
      <td>0.619948</td>
      <td>0.778368</td>
      <td>92.017745</td>
      <td>4</td>
    </tr>
    <tr>
      <th>38</th>
      <td>BMG210A71016</td>
      <td>512</td>
      <td>Health Care</td>
      <td>5.280</td>
      <td>11.8950</td>
      <td>10.035735</td>
      <td>8.971695</td>
      <td>11.120028</td>
      <td>90.070741</td>
      <td>4</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CNE1000055W8</td>
      <td>688192</td>
      <td>Health Care</td>
      <td>43.150</td>
      <td>91.3750</td>
      <td>81.877972</td>
      <td>73.204472</td>
      <td>91.214224</td>
      <td>89.751962</td>
      <td>2</td>
    </tr>
    <tr>
      <th>40</th>
      <td>BRCSEDACNOR9</td>
      <td>CSED3</td>
      <td>Consumer Discretionary</td>
      <td>3.960</td>
      <td>8.5000</td>
      <td>7.499996</td>
      <td>6.702347</td>
      <td>8.324900</td>
      <td>89.393830</td>
      <td>5</td>
    </tr>
    <tr>
      <th>41</th>
      <td>CH1242303498</td>
      <td>OCS</td>
      <td>Health Care</td>
      <td>12.290</td>
      <td>40.1845</td>
      <td>23.235767</td>
      <td>20.714896</td>
      <td>25.951599</td>
      <td>89.062385</td>
      <td>11</td>
    </tr>
    <tr>
      <th>42</th>
      <td>US98887Q1040</td>
      <td>ZLAB</td>
      <td>Health Care</td>
      <td>17.150</td>
      <td>33.7775</td>
      <td>32.348558</td>
      <td>29.166167</td>
      <td>35.684324</td>
      <td>88.621331</td>
      <td>12</td>
    </tr>
    <tr>
      <th>43</th>
      <td>KYG4783B1032</td>
      <td>9969</td>
      <td>Health Care</td>
      <td>11.260</td>
      <td>19.6198</td>
      <td>21.234606</td>
      <td>19.201575</td>
      <td>23.452540</td>
      <td>88.584425</td>
      <td>9</td>
    </tr>
    <tr>
      <th>44</th>
      <td>US8200144058</td>
      <td>SBET</td>
      <td>Consumer Discretionary</td>
      <td>5.810</td>
      <td>17.8125</td>
      <td>10.909378</td>
      <td>9.650232</td>
      <td>12.304879</td>
      <td>87.768979</td>
      <td>8</td>
    </tr>
    <tr>
      <th>45</th>
      <td>US76243J1051</td>
      <td>RYTM</td>
      <td>Health Care</td>
      <td>84.350</td>
      <td>138.2000</td>
      <td>157.257855</td>
      <td>141.654037</td>
      <td>175.293177</td>
      <td>86.434920</td>
      <td>15</td>
    </tr>
    <tr>
      <th>46</th>
      <td>ID1000125107</td>
      <td>KLBF</td>
      <td>Health Care</td>
      <td>745.000</td>
      <td>1438.7500</td>
      <td>1387.782897</td>
      <td>1261.143817</td>
      <td>1506.262707</td>
      <td>86.279584</td>
      <td>16</td>
    </tr>
    <tr>
      <th>47</th>
      <td>US6979471090</td>
      <td>PVLA</td>
      <td>Health Care</td>
      <td>104.670</td>
      <td>229.5625</td>
      <td>193.466525</td>
      <td>172.316212</td>
      <td>216.219816</td>
      <td>84.834742</td>
      <td>16</td>
    </tr>
    <tr>
      <th>48</th>
      <td>GB00BXB07J71</td>
      <td>GTLY</td>
      <td>Industrials</td>
      <td>0.675</td>
      <td>1.9000</td>
      <td>1.245122</td>
      <td>1.107074</td>
      <td>1.398536</td>
      <td>84.462465</td>
      <td>4</td>
    </tr>
    <tr>
      <th>49</th>
      <td>CNE000000Y37</td>
      <td>600201</td>
      <td>Health Care</td>
      <td>11.480</td>
      <td>17.3500</td>
      <td>21.164457</td>
      <td>18.872205</td>
      <td>23.708312</td>
      <td>84.359380</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Shrinkage view: Kalman-smoothed expected_pt vs raw consensus observed_pt.
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(results['observed_pt'], results['expected_pt'], s=8, alpha=0.4)
hi = float(np.nanquantile(results['observed_pt'], 0.99))
ax.plot([0, hi], [0, hi], '--', color='#888888', linewidth=1)
ax.set_xlim(0, hi)
ax.set_ylim(0, hi)
ax.set_xlabel('consensus observed_pt')
ax.set_ylabel('smoothed expected_pt')
ax.set_title('Kalman-smoothed expected target vs raw consensus')
plt.tight_layout()
plt.show()
```

    C:\Users\markm\AppData\Local\Temp\ipykernel_12952\830401102.py:11: UserWarning: The figure layout has changed to tight
      plt.tight_layout()
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_34_1.png)
    



```python
# Per-industry expected_upside posterior - arviz_plots forest with HDIs.
# Aggregates the ISIN-level `expected_upside` posterior samples by industry
# (mean across ISINs within each industry, per draw) so the forest plot
# shows the posterior mean and 94% HDI of the *industry-level* expected upside.
import arviz_plots as azp
import xarray as xr

eu = idata.posterior['expected_upside'] * 100.0  # to percent
_industry_per_isin = model_df['industry'].fillna('Unknown').astype(str).to_numpy()
_industry_da = xr.DataArray(
    _industry_per_isin, dims='isin', coords={'isin': eu.coords['isin']},
)
expected_upside_by_industry = (
    eu.groupby(_industry_da.rename('industry')).mean('isin')
)

_ds_forest = xr.Dataset({'expected_upside_pct': expected_upside_by_industry})

azp.plot_forest(_ds_forest, var_names=['expected_upside_pct'], combined=True)
plt.title('Per-industry expected upside (%) - posterior mean and 94% HDI')
plt.tight_layout()
plt.show()

```

    C:\Users\markm\AppData\Local\Temp\ipykernel_12952\2512235477.py:21: UserWarning: The figure layout has changed to tight
      plt.tight_layout()
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_35_1.png)
    


## 11. Estimated Price Targets Over Time - Single-ISIN Kalman Filter

Sections 5-10 are the **cross-sectional** panel adaptation: one row per ISIN, no time
axis. This section runs the **literal** single-security `GaussianRandomWalk` filter
from `probabilistic_ml_model/pymc_models/KalmanFilterModel.py` to recover the quantity
this model exists for - a **price target evolving over time** with a credible band.

The time axis is reconstructed from the embedded `*_ago` price-target cohort
(`price_target_1w_ago`, `price_target_high_3m_ago`, `price_target_median_1y_ago`, ...),
unpivoted into a `(isin, asof_date, price_target)` panel via
`build_price_target_history()`. The cohort is short (~ 6-16 points) and **irregularly
spaced** (1w, 1m, 3m, 6m, 1y), which is exactly where an explicit latent random walk
funnels.

We therefore fit the **marginalized** parameterization. This is the corrected
integrated-out GRW: the latent path is collapsed analytically into a single
`MvNormal` likelihood whose covariance carries **both** the random-walk (process)
and observation (measurement) variances,

$$\Sigma_{st} = P_0 + \sigma_{\text{state}}^2\,\min(\tau_s, \tau_t) + \sigma_{\text{obs}}^2\,\delta_{st},$$

with $\tau$ the **real elapsed time** between observations (`_resolve_time_deltas`).
The observed log-targets are used *only* as the data - never as the mean - so
`sigma_state` and `sigma_obs` are genuinely identified (the previous revision pinned
the state to the observations, a no-op). The smoothed path is recovered as the
analytic Kalman-smoother mean and rendered with the custom `plot_price_target_path()`
from section 1.1.

The cell is guarded - it degrades cleanly to an informative message if `DB_URL` is
unset or no ISIN carries >= 2 `*_ago` observations.


```python
# Single-ISIN time-series Kalman filter on the *_ago price-target history.
import os
from pathlib import Path


def _resolve_db_url(env_file: str = 'environment_variables.txt') -> str:
    """Return DB_URL from the environment, falling back to environment_variables.txt.

    The kernel may have been started without sourcing set_env.ps1, in which case
    os.environ has no 'DB_URL'. We then parse the KEY=VALUE lines of the project's
    environment_variables.txt as a fallback so the section still runs.
    """
    url = os.environ.get('DB_URL')
    if url:
        return url

    here = Path.cwd()
    for base in (here, *here.parents):
        candidate = base / env_file
        if candidate.is_file():
            for raw in candidate.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                if key.strip() == 'DB_URL':
                    return value.strip().strip('"').strip("'")
            break
    raise KeyError(
        "DB_URL not set in os.environ and not found in environment_variables.txt. "
        "Run `. .\\set_env.ps1` before launching the kernel, or add a DB_URL line."
    )


try:
    from sqlalchemy import create_engine, text
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import KalmanFilterPriceTarget

    engine = create_engine(_resolve_db_url())
    cohort = model_df['isin'].astype(str).tolist()

    # Discover the *_ago price-target history columns that actually exist in
    # pml.pml_df, then pull only those (+ identifiers) for the modelled cohort.
    _hist_re = (r"^(price_target(_high|_low|_median)?|price)"
                r"_(5d|1w|1m|3m|6m|1y|3y|5y|mtd|qtd|ytd)_ago$")
    with engine.connect() as conn:
        hist_cols = pd.read_sql(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'pml' AND table_name = 'pml_df'
                  AND (column_name ~ :pat
                       OR column_name IN ('isin', 'ticker', 'last_price', 'price_target'))
                ORDER BY column_name
            """),
            conn, params={'pat': _hist_re},
        )['column_name'].tolist()

        col_sql = ', '.join(f'"{c}"' for c in hist_cols)
        snap = pd.read_sql(
            text(f'SELECT {col_sql} FROM pml.pml_df WHERE isin = ANY(:isins)'),
            conn, params={'isins': cohort},
        )

    n_ago = sum(c.endswith('_ago') for c in hist_cols)
    print(f'Pulled pml.pml_df history frame: {snap.shape}  ({n_ago} *_ago columns).')

    # Unpivot *_ago -> long (isin, asof_date, price_target); pick richest ISIN in
    # the cohort. now_cols omits last_price so the target series is not polluted
    # by the spot price (last_price is shown separately as the reference line).
    long_df, eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
        snap, now_cols=('price_target',),
    )
    chosen = KalmanFilterPriceTarget.select_target_isin(eligible, cohort=cohort)

    if chosen is None or date_col is None:
        print('No ISIN has >= 2 *_ago price-target observations; section skipped.')
    else:
        ts = (long_df.loc[long_df['isin'] == chosen, ['asof_date', 'price_target']]
              .dropna().sort_values('asof_date').reset_index(drop=True))
        dates = pd.DatetimeIndex(ts['asof_date'])
        observed = ts['price_target'].to_numpy()

        _row = model_df.loc[model_df['isin'] == chosen]
        ticker = str(_row['ticker'].iloc[0]) if len(_row) else str(chosen)
        last_price = float(_row['last_price'].iloc[0]) if len(_row) else None

        print(f'Fitting single-ISIN Kalman filter for {chosen} ({ticker}) '
              f'on {len(observed)} observations spanning '
              f'{dates.min():%Y-%m-%d} … {dates.max():%Y-%m-%d}.')

        kf = KalmanFilterPriceTarget()
        # Short, irregular *_ago cohort -> use the corrected *marginalized* GRW
        # (the latent path is integrated out into the MvNormal covariance, scaled
        # by real elapsed time). This is what parameterization='auto' selects for
        # series below the short-series threshold; we set it explicitly so the
        # section deterministically exercises the funnel-free path.
        kf_idata, kf_model = kf.fit(
            price_targets=observed, isin=str(chosen), dates=dates,
            samples=2500, tune=2000, chains=8,
            random_seed=RANDOM_SEED, parameterization='marginalized',
            target_accept=0.95, nuts_sampler='nutpie',
        )

        n_div = int(kf_idata.sample_stats['diverging'].sum())
        print(f'Marginalized GRW fit: {len(observed)} obs, '
              f'{dates.max().year - dates.min().year}y span, divergences={n_div}.')
        display(azs.summary(kf_idata,
                            var_names=['sigma_state', 'sigma_obs', 'log_state_init'],
                            round_to=4))

        # Headline custom plot — Kalman-smoothed price-target path over time.
        pc_path = plot_price_target_path(
            kf_idata, observed=observed, dates=dates,
            last_price=last_price, ticker=ticker,
        )
        pc_path.show()
except Exception as e:  # pragma: no cover - optional / environment-dependent
    print(f'Section 11 (single-ISIN time-series Kalman) skipped: {e!r}')
```

    Pulled pml.pml_df history frame: (6277, 47)  (43 *_ago columns).
    Fitting single-ISIN Kalman filter for AEA001901015 (AGTHIA) on 11 observations spanning 2021-06-03 … 2027-01-01.
    

    NUTS[nutpie]: [sigma_state, sigma_obs, log_state_init]
    


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Marginalized GRW fit: 11 obs, 6y span, divergences=0.
    


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
      <th>mean</th>
      <th>sd</th>
      <th>eti89_lb</th>
      <th>eti89_ub</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sigma_state</th>
      <td>0.1017</td>
      <td>0.0444</td>
      <td>0.0414</td>
      <td>0.1810</td>
      <td>13888.5953</td>
      <td>9625.6605</td>
      <td>1.0002</td>
      <td>0.0003</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>sigma_obs</th>
      <td>0.1592</td>
      <td>0.0433</td>
      <td>0.1045</td>
      <td>0.2356</td>
      <td>15831.5105</td>
      <td>12993.6877</td>
      <td>1.0002</td>
      <td>0.0004</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>log_state_init</th>
      <td>1.8269</td>
      <td>0.1298</td>
      <td>1.6199</td>
      <td>2.0342</td>
      <td>17023.5511</td>
      <td>14009.8991</td>
      <td>1.0002</td>
      <td>0.0010</td>
      <td>0.0007</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_37_6.png)
    


## 12. Estimated Price Targets over upcoming/recent Earnings Period - Mingle-ISIN Kalman Filter

Section 11 fits the literal single-security `GaussianRandomWalk` filter on the one ISIN
with the richest `*_ago` history. This section keeps the same state-space machinery but
re-frames **what the time axis represents**: instead of one security's history, it builds
a **mingled cross-sectional consensus** over every ISIN whose `next_earnings` lands in the
**recent earnings window**

```sql
SELECT *

WHERE next_earnings >= current_date - INTERVAL '1 week'
  AND next_earnings <= current_date + INTERVAL '1 week'
```

i.e. names reporting in the ±1-week band around today. For each such ISIN we unpivot the
embedded `*_ago` price-target cohort (`price_target_1w_ago`, `price_target_high_3m_ago`,
`price_target_median_1y_ago`, …) into a `(isin, asof_date, price_target)` panel via
`KalmanFilterPriceTarget.build_price_target_history()`, then **mingle the ISINs** by taking
the cross-sectional **median** price target at each shared `asof_date`. The result is a
single, time-ordered series — the earnings-cohort consensus price target as it evolved over
the recent earnings period — which is exactly the quantity the Kalman filter smooths.

The cohort series is short (~6-16 points) and **irregularly spaced** (1w, 1m, 3m, 6m, 1y),
so we fit the **marginalized** parameterization (the integrated-out GRW whose `MvNormal`
covariance carries both the random-walk and observation variances, scaled by real elapsed
time via `_resolve_time_deltas`). This is funnel-free for sparse, irregular cohorts.

**Visual comparison of the three quantities** (`last_price`, `observed_price_target`,
`expected_pt`):

- `expected_pt` — the posterior-mean Kalman-smoothed latent state (`state` in price space),
  with 94 % / 50 % HDI bands.
- `observed_price_target` — the mingled cohort-median consensus target at each `asof_date`.
- `last_price` — the cohort-median spot price, drawn as a dashed reference line.

Rendered as (a) the headline `plot_price_target_path()` time-series composition, (b) an
ArviZ `plot_forest` of the per-`asof_date` `expected_pt` posterior HDIs, and (c) a tidy
comparison table. The cell is guarded — it degrades cleanly to an informative message if
`DB_URL` is unset or the earnings window yields fewer than 2 mingled observations.



```python
# Mingle-ISIN time-series Kalman filter over the recent-earnings-window cohort.
# Reuses _resolve_db_url(), KalmanFilterPriceTarget, plot_price_target_path() and
# RANDOM_SEED defined earlier in the notebook.
try:
    from sqlalchemy import create_engine, text
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import KalmanFilterPriceTarget

    engine = create_engine(_resolve_db_url())

    # Discover the *_ago price-target history columns present in pml.pml_df, plus
    # the identifiers / reference columns and the next_earnings timing column used
    # to scope the recent-earnings window.
    _hist_re = (r"^(price_target(_high|_low|_median)?|price)"
                r"_(5d|1w|1m|3m|6m|1y|3y|5y|mtd|qtd|ytd)_ago$")
    with engine.connect() as conn:
        hist_cols = pd.read_sql(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'pml' AND table_name = 'pml_df'
                  AND (column_name ~ :pat
                       OR column_name IN ('isin', 'ticker', 'last_price',
                                          'price_target', 'next_earnings'))
                ORDER BY column_name
            """),
            conn, params={'pat': _hist_re},
        )['column_name'].tolist()

        col_sql = ', '.join(f'"{c}"' for c in hist_cols)
        # Time axis = the recent earnings period: names reporting within +/- 5 days
        # of today. next_earnings is the pml.pml_df earnings-timing column.
        snap = pd.read_sql(
            text(f"""
                SELECT {col_sql}
                FROM pml.pml_df
                WHERE next_earnings >= current_date - INTERVAL '5 days'
                  AND next_earnings <= current_date + INTERVAL '5 days'
            """),
            conn,
        )

    n_ago = sum(c.endswith('_ago') for c in hist_cols)
    n_cohort = snap['isin'].nunique() if 'isin' in snap.columns else 0
    print(f'Recent-earnings window cohort: {snap.shape[0]} rows / {n_cohort} ISINs '
          f'({n_ago} *_ago columns).')

    # Unpivot *_ago -> long (isin, asof_date, price_target) for the whole cohort.
    # now_cols omits last_price so the target series is not polluted by spot price.
    long_df, _eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
        snap, now_cols=('price_target',),
    )

    if date_col is None or long_df.empty:
        print('No *_ago price-target history in the earnings window; section skipped.')
    else:
        # MINGLE the ISINs: cross-sectional median price target at each shared
        # asof_date -> a single earnings-cohort consensus series over time.
        mingled = (
            long_df.groupby('asof_date', as_index=False)
            .agg(price_target=('price_target', 'median'),
                 n_isin=('isin', 'nunique'))
            .sort_values('asof_date')
            .reset_index(drop=True)
        )

        if len(mingled) < 2:
            print(f'Mingled cohort has only {len(mingled)} distinct as-of date(s); '
                  'need >= 2 for a Kalman fit. Section skipped.')
        else:
            dates = pd.DatetimeIndex(mingled['asof_date'])
            observed = mingled['price_target'].to_numpy()
            # Cohort-median spot price as the reference last_price.
            last_price = (float(np.nanmedian(snap['last_price']))
                          if 'last_price' in snap.columns
                          and np.isfinite(np.nanmedian(snap['last_price'])) else None)
            label = f'EARNINGS-COHORT (n={n_cohort})'

            print(f'Fitting mingled-cohort Kalman filter on {len(observed)} '
                  f'consensus observations spanning '
                  f'{dates.min():%Y-%m-%d} ... {dates.max():%Y-%m-%d}.')

            kf = KalmanFilterPriceTarget()
            # Short, irregular cohort -> corrected marginalized (integrated-out) GRW.
            kf_idata, kf_model = kf.fit(
                price_targets=observed, isin=label, dates=dates,
                samples=2500, tune=2000, chains=8,
                random_seed=RANDOM_SEED, parameterization='marginalized',
                target_accept=0.95, nuts_sampler='nutpie',
            )

            n_div = int(kf_idata.sample_stats['diverging'].sum())
            print(f'Marginalized GRW fit: {len(observed)} obs, '
                  f'{(dates.max() - dates.min()).days}d span, divergences={n_div}.')
            display(azs.summary(
                kf_idata,
                var_names=['sigma_state', 'sigma_obs', 'log_state_init'],
                round_to=4))

            # (a) Headline composition: expected_pt smoothed path + HDI bands,
            #     observed mingled consensus targets, and last_price reference.
            pc_path = plot_price_target_path(
                kf_idata, observed=observed, dates=dates,
                last_price=last_price, ticker=label,
            )
            pc_path.show()

            # (b) ArviZ forest of the per-as-of-date expected_pt (latent `state`)
            #     posterior HDIs, with the cohort last_price as a reference line.
            _state = kf_idata.posterior['state']
            _state = _state.assign_coords(
                time=[d.strftime('%Y-%m-%d') for d in dates]
            )
            azp.plot_forest(_state.to_dataset(), var_names=['state'], combined=True)
            if last_price is not None:
                plt.axvline(last_price, ls='--', color='#bbbbbb', lw=1.2,
                            label='cohort last_price')
                plt.legend(fontsize=8, framealpha=0.25)
            plt.title(f'Expected price target (Kalman state) per as-of date - {label}')
            plt.xlabel('expected_pt (price)')
            plt.tight_layout()
            plt.show()

            # (c) Tidy comparison table: last_price vs observed_pt vs expected_pt.
            _post = kf_idata.posterior['state']
            _mean = _post.mean(('chain', 'draw')).values
            _stk = _post.stack(s=('chain', 'draw'))
            _lo = _stk.quantile(0.03, dim='s').values
            _hi = _stk.quantile(0.97, dim='s').values
            comparison = pd.DataFrame({
                'asof_date': dates.strftime('%Y-%m-%d'),
                'n_isin': mingled['n_isin'].to_numpy(),
                'observed_pt': observed,
                'expected_pt': _mean,
                'expected_pt_hdi_lo': _lo,
                'expected_pt_hdi_hi': _hi,
                'last_price': last_price,
            })
            comparison['expected_vs_observed_pct'] = (
                (comparison['expected_pt'] / comparison['observed_pt'] - 1.0) * 100
            )
            print('Mingled-cohort comparison (last_price vs observed_pt vs expected_pt):')
            display(comparison.round(3))
except Exception as e:  # pragma: no cover - optional / environment-dependent
    print(f'Section 12 (mingle-ISIN earnings-window Kalman) skipped: {e!r}')
```

    Recent-earnings window cohort: 220 rows / 220 ISINs (43 *_ago columns).
    Fitting mingled-cohort Kalman filter on 11 consensus observations spanning 2021-06-03 ... 2027-01-01.
    

    NUTS[nutpie]: [sigma_state, sigma_obs, log_state_init]
    


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Marginalized GRW fit: 11 obs, 2038d span, divergences=0.
    


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
      <th>mean</th>
      <th>sd</th>
      <th>eti89_lb</th>
      <th>eti89_ub</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sigma_state</th>
      <td>0.1166</td>
      <td>0.0422</td>
      <td>0.0570</td>
      <td>0.1890</td>
      <td>15055.6980</td>
      <td>10500.8763</td>
      <td>1.0000</td>
      <td>0.0003</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>sigma_obs</th>
      <td>0.1221</td>
      <td>0.0386</td>
      <td>0.0744</td>
      <td>0.1920</td>
      <td>14882.1663</td>
      <td>12759.1950</td>
      <td>1.0002</td>
      <td>0.0003</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>log_state_init</th>
      <td>2.9281</td>
      <td>0.1260</td>
      <td>2.7254</td>
      <td>3.1282</td>
      <td>18102.2196</td>
      <td>13635.9403</td>
      <td>1.0003</td>
      <td>0.0009</td>
      <td>0.0007</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_39_6.png)
    


    C:\Users\markm\AppData\Local\Temp\ipykernel_12952\2996999407.py:120: UserWarning: The figure layout has changed to tight
      plt.tight_layout()
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_39_8.png)
    


    Mingled-cohort comparison (last_price vs observed_pt vs expected_pt):
    


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
      <th>asof_date</th>
      <th>n_isin</th>
      <th>observed_pt</th>
      <th>expected_pt</th>
      <th>expected_pt_hdi_lo</th>
      <th>expected_pt_hdi_hi</th>
      <th>last_price</th>
      <th>expected_vs_observed_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-06-03</td>
      <td>174</td>
      <td>18.870</td>
      <td>18.529</td>
      <td>17.231</td>
      <td>20.282</td>
      <td>18.735</td>
      <td>-1.808</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-03</td>
      <td>192</td>
      <td>14.440</td>
      <td>17.097</td>
      <td>15.150</td>
      <td>20.126</td>
      <td>18.735</td>
      <td>18.403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-06-03</td>
      <td>210</td>
      <td>20.384</td>
      <td>21.061</td>
      <td>20.677</td>
      <td>21.606</td>
      <td>18.735</td>
      <td>3.320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-12-03</td>
      <td>215</td>
      <td>23.849</td>
      <td>22.577</td>
      <td>21.943</td>
      <td>23.220</td>
      <td>18.735</td>
      <td>-5.334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2026-03-03</td>
      <td>217</td>
      <td>24.168</td>
      <td>23.022</td>
      <td>22.134</td>
      <td>23.736</td>
      <td>18.735</td>
      <td>-4.743</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2026-05-03</td>
      <td>220</td>
      <td>25.249</td>
      <td>23.142</td>
      <td>22.239</td>
      <td>23.706</td>
      <td>18.735</td>
      <td>-8.345</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2026-05-27</td>
      <td>220</td>
      <td>24.376</td>
      <td>23.026</td>
      <td>22.260</td>
      <td>23.235</td>
      <td>18.735</td>
      <td>-5.536</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2026-05-29</td>
      <td>219</td>
      <td>18.880</td>
      <td>23.007</td>
      <td>22.261</td>
      <td>23.204</td>
      <td>18.735</td>
      <td>21.861</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2026-06-03</td>
      <td>220</td>
      <td>24.889</td>
      <td>23.053</td>
      <td>22.266</td>
      <td>23.272</td>
      <td>18.735</td>
      <td>-7.378</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2026-07-01</td>
      <td>220</td>
      <td>23.748</td>
      <td>23.128</td>
      <td>22.284</td>
      <td>23.424</td>
      <td>18.735</td>
      <td>-2.609</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2027-01-01</td>
      <td>215</td>
      <td>23.851</td>
      <td>23.323</td>
      <td>22.347</td>
      <td>23.734</td>
      <td>18.735</td>
      <td>-2.213</td>
    </tr>
  </tbody>
</table>
</div>


## 13. Granular Earnings-Cohort Expected-Price Simulation - Posterior-Predictive Forest

Section 12 **mingles** the recent-earnings cohort into a single cross-sectional median series
and refits the literal single-series Kalman filter. This section keeps the cohort definition
identical - **names reporting within +/- 5 days of today** (`next_earnings`) - but goes the
other way: it stays **per-ISIN granular** and re-uses the already-fitted cross-sectional
state-space posterior from sections 5-10 instead of refitting.

For every cohort ISIN the model's log-space measurement likelihood (`log_pt_obs`) is
exponentiated back to **price units**, giving a posterior-predictive distribution of the
*expected stock price* (the simulated analyst price target) for that name. These per-ISIN
predictive distributions are rendered as an `arviz_plots` **posterior-predictive forest**,
with the realised analyst targets overlaid as observation points, following the
`plot_forest(group="posterior_predictive") + visuals.scatter_x` composition pattern.

Two reference layers make the forest readable as a screen:

- **Reference HDI bands** (`add_bands`) - the cohort's central expected-price region, taken
  directly from the **posterior HDIs** of the pooled `expected_pt` latent state (94 % band
  lightest, 50 % band darker). Names whose predictive interval sits entirely outside the
  94 % band are the cohort's relative outliers going into earnings.
- **Cohort `last_price` reference line** (`add_lines`) - the median spot price across the
  cohort, drawn as a dashed vertical line so the simulated targets can be read as implied
  upside / downside.

The cell is guarded - it degrades cleanly to an informative message if `DB_URL` is unset or
no earnings-window ISIN overlaps the fitted cross-sectional posterior. When the cohort is
large the forest is capped to the most extreme names by expected upside (top/bottom 20) to
keep one row per ISIN legible; the truncation is announced.



```python
# Granular per-ISIN posterior-predictive forest of simulated expected prices for the
# recent-earnings cohort (next_earnings within +/-5 days). Re-uses the fitted
# cross-sectional `idata`, `results`, `model_df`, `_resolve_db_url()` and RANDOM_SEED.
try:
    from sqlalchemy import create_engine, text

    # 1. Recent-earnings cohort: names reporting within +/- 5 days of today.
    engine = create_engine(_resolve_db_url())
    with engine.connect() as conn:
        cohort_meta = pd.read_sql(
            text("""
                SELECT isin, ticker, last_price, next_earnings
                FROM pml.pml_df
                WHERE next_earnings >= current_date - INTERVAL '5 days'
                  AND next_earnings <= current_date + INTERVAL '5 days'
            """),
            conn,
        )
    cohort_isins_all = cohort_meta['isin'].astype(str).unique().tolist()
    print(f'Recent-earnings cohort (next_earnings +/-5d): {len(cohort_isins_all)} ISINs.')

    # 2. Keep only cohort ISINs that are present in the fitted cross-sectional posterior.
    pp = idata.posterior_predictive          # DataTree node: var `log_pt_obs` over `isin`
    obsd = idata.observed_data
    modelled = set(pp['log_pt_obs'].coords['isin'].astype(str).values.tolist())
    cohort_isins = [i for i in cohort_isins_all if i in modelled]
    if not cohort_isins:
        raise RuntimeError(
            'No earnings-window ISIN overlaps the fitted cross-sectional posterior '
            f'({len(cohort_isins_all)} cohort ISINs, {len(modelled)} modelled).'
        )
    print(f'Cohort ISINs overlapping the fitted posterior: {len(cohort_isins)}.')

    # Cap the forest so one-row-per-ISIN stays legible; rank by expected upside and
    # keep the most extreme names (top/bottom) when the cohort is large.
    MAX_FOREST = 40
    cohort_results = (results[results['isin'].isin(cohort_isins)]
                      .sort_values('expected_upside_pct', ascending=False))
    if len(cohort_results) > MAX_FOREST:
        half = MAX_FOREST // 2
        keep = pd.concat([cohort_results.head(half), cohort_results.tail(half)])
        print(f'Cohort has {len(cohort_results)} ISINs; showing the {MAX_FOREST} most '
              f'extreme by expected upside (top/bottom {half}).')
    else:
        keep = cohort_results
    forest_isins = keep['isin'].astype(str).tolist()

    # 3. Simulate expected stock prices: exp() the log-space posterior-predictive draws
    #    and the observed analyst targets back into price units, subset to the cohort.
    pp_price = np.exp(pp['log_pt_obs'].sel(isin=forest_isins)).rename('expected_price')
    obs_price = np.exp(obsd['log_pt_obs'].sel(isin=forest_isins)).rename('expected_price')
    ppc_tree = xr.DataTree.from_dict({
        'posterior_predictive': pp_price.to_dataset(),
        'observed_data': obs_price.to_dataset(),
    })

    # 4. Reference bands from the POSTERIOR HDIs: pool the cohort's `expected_pt` latent
    #    state draws and take the 94% / 50% HDI as the central expected-price region.
    exp_pt_pool = (idata.posterior['expected_pt']
                   .sel(isin=forest_isins)
                   .stack(s=('chain', 'draw', 'isin')))
    _q = lambda p: float(exp_pt_pool.quantile(p).values)
    band94 = (_q(0.03), _q(0.97))
    band50 = (_q(0.25), _q(0.75))
    # Cohort last_price reference: median spot price across the shown cohort.
    cohort_last_price = float(np.nanmedian(
        cohort_meta.loc[cohort_meta['isin'].isin(forest_isins), 'last_price']
    ))

    # 5. Posterior-predictive forest + observed analyst targets (scatter_x overlay).
    pc = azp.plot_forest(
        ppc_tree, group='posterior_predictive', combined=True,
        labels=['isin'], backend='matplotlib',
    )
    pc.map(
        azv.scatter_x, 'observations',
        data=ppc_tree.observed_data.ds, coords={'column': 'forest'},
        color='#ffb000',
    )
    pc.map(
        azv.labelled_x, 'xlabel', coords={'column': 'forest'},
        text='expected price (simulated)  -  points = observed analyst target',
        ignore_aes='y',
    )

    # 6. Reference HDI bands (94% lightest, 50% darker) + cohort last_price line.
    pc.coords = {'column': 'forest'}
    pc = azp.add_bands(pc, values=[band94],
                       visuals={'ref_band': {'color': '#56b4e9', 'alpha': 0.12}})
    pc = azp.add_bands(pc, values=[band50],
                       visuals={'ref_band': {'color': '#56b4e9', 'alpha': 0.24}})
    pc = azp.add_lines(pc, values=cohort_last_price,
                       visuals={'ref_line': {'color': '#bbbbbb',
                                             'linestyle': '--', 'linewidth': 1.3}})
    pc.show()

    print(f'Cohort expected_pt 94% HDI band: ({band94[0]:.2f}, {band94[1]:.2f});  '
          f'50% HDI band: ({band50[0]:.2f}, {band50[1]:.2f});  '
          f'cohort last_price ref = {cohort_last_price:.2f}.')

    # 7. Tidy per-ISIN summary for the names shown in the forest.
    _cols = ['isin', 'ticker', 'sector', 'last_price', 'observed_pt',
             'expected_pt', 'expected_pt_hdi_lo', 'expected_pt_hdi_hi',
             'expected_upside_pct']
    display(keep[[c for c in _cols if c in keep.columns]]
            .round(3).reset_index(drop=True))
except Exception as e:  # pragma: no cover - optional / environment-dependent
    print(f'Section 13 (granular earnings-cohort posterior-predictive forest) skipped: {e!r}')
```

    Recent-earnings cohort (next_earnings +/-5d): 220 ISINs.
    Cohort ISINs overlapping the fitted posterior: 220.
    Cohort has 220 ISINs; showing the 40 most extreme by expected upside (top/bottom 20).
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_41_1.png)
    


    Cohort expected_pt 94% HDI band: (0.26, 793.27);  50% HDI band: (6.87, 81.30);  cohort last_price ref = 15.83.
    


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
      <th>sector</th>
      <th>last_price</th>
      <th>observed_pt</th>
      <th>expected_pt</th>
      <th>expected_pt_hdi_lo</th>
      <th>expected_pt_hdi_hi</th>
      <th>expected_upside_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CNE100006QF7</td>
      <td>2432</td>
      <td>Industrials</td>
      <td>32.220</td>
      <td>59.684</td>
      <td>77.122</td>
      <td>68.480</td>
      <td>86.899</td>
      <td>139.360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AU000000EOS8</td>
      <td>EOS</td>
      <td>Industrials</td>
      <td>12.260</td>
      <td>12.938</td>
      <td>24.817</td>
      <td>21.666</td>
      <td>28.117</td>
      <td>102.425</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNE100003N76</td>
      <td>2696</td>
      <td>Health Care</td>
      <td>58.650</td>
      <td>103.199</td>
      <td>113.727</td>
      <td>102.336</td>
      <td>126.465</td>
      <td>93.908</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMG210A71016</td>
      <td>512</td>
      <td>Health Care</td>
      <td>5.280</td>
      <td>11.895</td>
      <td>10.036</td>
      <td>8.972</td>
      <td>11.120</td>
      <td>90.071</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HK0000503208</td>
      <td>6055</td>
      <td>Consumer Discretionary</td>
      <td>25.000</td>
      <td>46.600</td>
      <td>43.967</td>
      <td>39.646</td>
      <td>48.650</td>
      <td>75.868</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GB00B3W40C23</td>
      <td>DOTD</td>
      <td>Information Technology</td>
      <td>0.503</td>
      <td>1.264</td>
      <td>0.881</td>
      <td>0.779</td>
      <td>0.992</td>
      <td>75.087</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NL0015073TS8</td>
      <td>CSG</td>
      <td>Industrials</td>
      <td>15.948</td>
      <td>32.050</td>
      <td>27.830</td>
      <td>24.059</td>
      <td>31.718</td>
      <td>74.503</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HK0000658531</td>
      <td>2096</td>
      <td>Health Care</td>
      <td>9.990</td>
      <td>17.163</td>
      <td>17.365</td>
      <td>15.741</td>
      <td>19.151</td>
      <td>73.823</td>
    </tr>
    <tr>
      <th>8</th>
      <td>KYG8167W1380</td>
      <td>1177</td>
      <td>Health Care</td>
      <td>4.880</td>
      <td>8.786</td>
      <td>8.407</td>
      <td>7.641</td>
      <td>9.124</td>
      <td>72.281</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CNE1000070L0</td>
      <td>2590</td>
      <td>Industrials</td>
      <td>15.720</td>
      <td>32.939</td>
      <td>26.908</td>
      <td>24.124</td>
      <td>29.825</td>
      <td>71.173</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GB00BNG73286</td>
      <td>AURR</td>
      <td>Consumer Discretionary</td>
      <td>0.740</td>
      <td>1.750</td>
      <td>1.262</td>
      <td>1.128</td>
      <td>1.411</td>
      <td>70.582</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GB00B0FVQX23</td>
      <td>RKH</td>
      <td>Energy</td>
      <td>0.769</td>
      <td>1.080</td>
      <td>1.294</td>
      <td>1.156</td>
      <td>1.449</td>
      <td>68.282</td>
    </tr>
    <tr>
      <th>12</th>
      <td>KYG5966D1051</td>
      <td>1357</td>
      <td>Communication Services</td>
      <td>5.590</td>
      <td>9.317</td>
      <td>9.285</td>
      <td>8.367</td>
      <td>10.304</td>
      <td>66.093</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KYG8813K1085</td>
      <td>3933</td>
      <td>Health Care</td>
      <td>9.070</td>
      <td>15.640</td>
      <td>14.849</td>
      <td>13.478</td>
      <td>16.193</td>
      <td>63.719</td>
    </tr>
    <tr>
      <th>14</th>
      <td>KYG608371046</td>
      <td>853</td>
      <td>Health Care</td>
      <td>7.600</td>
      <td>14.106</td>
      <td>12.441</td>
      <td>11.187</td>
      <td>13.744</td>
      <td>63.697</td>
    </tr>
    <tr>
      <th>15</th>
      <td>US6811161099</td>
      <td>OLLI</td>
      <td>Consumer Discretionary</td>
      <td>79.250</td>
      <td>129.733</td>
      <td>128.235</td>
      <td>117.459</td>
      <td>139.223</td>
      <td>61.810</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ID1000188303</td>
      <td>MBMA</td>
      <td>Materials</td>
      <td>474.000</td>
      <td>844.435</td>
      <td>753.677</td>
      <td>672.985</td>
      <td>841.221</td>
      <td>59.004</td>
    </tr>
    <tr>
      <th>17</th>
      <td>JE00B6Y3DV84</td>
      <td>CRTA</td>
      <td>Information Technology</td>
      <td>0.175</td>
      <td>0.437</td>
      <td>0.278</td>
      <td>0.248</td>
      <td>0.312</td>
      <td>58.987</td>
    </tr>
    <tr>
      <th>18</th>
      <td>KYG6382M1096</td>
      <td>3918</td>
      <td>Consumer Discretionary</td>
      <td>4.190</td>
      <td>6.438</td>
      <td>6.648</td>
      <td>6.071</td>
      <td>7.285</td>
      <td>58.662</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GB00BLGYDT21</td>
      <td>MTEC</td>
      <td>Information Technology</td>
      <td>0.380</td>
      <td>0.635</td>
      <td>0.602</td>
      <td>0.538</td>
      <td>0.669</td>
      <td>58.425</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KYG8586D1097</td>
      <td>2382</td>
      <td>Information Technology</td>
      <td>83.000</td>
      <td>81.135</td>
      <td>82.269</td>
      <td>75.700</td>
      <td>89.410</td>
      <td>-0.881</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GB00BYV81293</td>
      <td>STX</td>
      <td>Health Care</td>
      <td>0.058</td>
      <td>0.173</td>
      <td>0.058</td>
      <td>0.050</td>
      <td>0.066</td>
      <td>-1.398</td>
    </tr>
    <tr>
      <th>22</th>
      <td>KYG525621408</td>
      <td>148</td>
      <td>Information Technology</td>
      <td>60.350</td>
      <td>64.000</td>
      <td>58.617</td>
      <td>52.791</td>
      <td>65.037</td>
      <td>-2.871</td>
    </tr>
    <tr>
      <th>23</th>
      <td>NZPOTE0003S0</td>
      <td>POT</td>
      <td>Industrials</td>
      <td>8.140</td>
      <td>7.528</td>
      <td>7.888</td>
      <td>7.241</td>
      <td>8.712</td>
      <td>-3.100</td>
    </tr>
    <tr>
      <th>24</th>
      <td>EGS38191C010</td>
      <td>ABUK</td>
      <td>Materials</td>
      <td>81.810</td>
      <td>76.215</td>
      <td>77.677</td>
      <td>69.569</td>
      <td>86.461</td>
      <td>-5.052</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PLAB00000019</td>
      <td>ABE</td>
      <td>Information Technology</td>
      <td>141.000</td>
      <td>135.233</td>
      <td>133.328</td>
      <td>120.719</td>
      <td>146.485</td>
      <td>-5.441</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CNE100006V73</td>
      <td>2589</td>
      <td>Consumer Discretionary</td>
      <td>145.800</td>
      <td>88.350</td>
      <td>137.659</td>
      <td>123.059</td>
      <td>153.351</td>
      <td>-5.583</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AU000000NWH5</td>
      <td>NWH</td>
      <td>Industrials</td>
      <td>7.550</td>
      <td>6.770</td>
      <td>7.124</td>
      <td>6.512</td>
      <td>7.835</td>
      <td>-5.638</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ZAE000171963</td>
      <td>KAP</td>
      <td>Industrials</td>
      <td>2.800</td>
      <td>2.050</td>
      <td>2.575</td>
      <td>2.287</td>
      <td>2.899</td>
      <td>-8.043</td>
    </tr>
    <tr>
      <th>29</th>
      <td>KZ1C00001122</td>
      <td>KMGZ</td>
      <td>Energy</td>
      <td>31400.010</td>
      <td>22750.000</td>
      <td>28305.054</td>
      <td>23727.712</td>
      <td>34174.208</td>
      <td>-9.857</td>
    </tr>
    <tr>
      <th>30</th>
      <td>US22788C1053</td>
      <td>CRWD</td>
      <td>Information Technology</td>
      <td>768.950</td>
      <td>571.338</td>
      <td>666.898</td>
      <td>592.829</td>
      <td>754.638</td>
      <td>-13.272</td>
    </tr>
    <tr>
      <th>31</th>
      <td>GB00BS9F9D74</td>
      <td>IES</td>
      <td>Industrials</td>
      <td>0.363</td>
      <td>0.525</td>
      <td>0.309</td>
      <td>0.275</td>
      <td>0.347</td>
      <td>-14.856</td>
    </tr>
    <tr>
      <th>32</th>
      <td>US6974351057</td>
      <td>PANW</td>
      <td>Information Technology</td>
      <td>297.180</td>
      <td>234.480</td>
      <td>250.670</td>
      <td>231.530</td>
      <td>277.394</td>
      <td>-15.650</td>
    </tr>
    <tr>
      <th>33</th>
      <td>GB00BFZZM640</td>
      <td>SFOR</td>
      <td>Communication Services</td>
      <td>0.434</td>
      <td>0.407</td>
      <td>0.364</td>
      <td>0.326</td>
      <td>0.405</td>
      <td>-16.158</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CNE100001ZT0</td>
      <td>3396</td>
      <td>Information Technology</td>
      <td>18.230</td>
      <td>13.069</td>
      <td>15.011</td>
      <td>13.356</td>
      <td>16.768</td>
      <td>-17.660</td>
    </tr>
    <tr>
      <th>35</th>
      <td>FR0011341205</td>
      <td>NANO</td>
      <td>Health Care</td>
      <td>31.260</td>
      <td>46.000</td>
      <td>25.557</td>
      <td>22.554</td>
      <td>28.942</td>
      <td>-18.243</td>
    </tr>
    <tr>
      <th>36</th>
      <td>DE000A0MSN11</td>
      <td>M7U</td>
      <td>Information Technology</td>
      <td>20.200</td>
      <td>26.000</td>
      <td>16.433</td>
      <td>14.733</td>
      <td>18.239</td>
      <td>-18.648</td>
    </tr>
    <tr>
      <th>37</th>
      <td>US1717793095</td>
      <td>CIEN</td>
      <td>Information Technology</td>
      <td>627.000</td>
      <td>477.011</td>
      <td>487.002</td>
      <td>444.439</td>
      <td>535.188</td>
      <td>-22.328</td>
    </tr>
    <tr>
      <th>38</th>
      <td>US72703X1063</td>
      <td>PL</td>
      <td>Industrials</td>
      <td>48.090</td>
      <td>35.500</td>
      <td>35.261</td>
      <td>31.557</td>
      <td>39.313</td>
      <td>-26.677</td>
    </tr>
    <tr>
      <th>39</th>
      <td>US35952H7008</td>
      <td>FCEL</td>
      <td>Industrials</td>
      <td>24.640</td>
      <td>8.242</td>
      <td>11.952</td>
      <td>10.588</td>
      <td>13.400</td>
      <td>-51.492</td>
    </tr>
  </tbody>
</table>
</div>


### 13.1 Further views - results-dataframe HDI bands and a cohort distribution KDE

Two complementary `arviz_plots` views building on the posterior-predictive forest above
(both reuse `ppc_tree`, `keep`, `forest_isins` and `cohort_last_price` from the previous
cell):

- **(a) Forest with `results`-keyed reference bands** - the same per-ISIN posterior-predictive
  forest of simulated expected prices, but the reference band is now keyed off the **stored
  `results` dataframe** columns rather than re-pooled posterior draws: the band spans the
  **cohort median** of `expected_pt_hdi_lo` ... `expected_pt_hdi_hi`, with the cohort-median
  `expected_pt` and the cohort `last_price` drawn as reference lines. This reads the cohort's
  consensus-smoothed 94 % credible region straight from the section-10 screening table.
- **(b) Cohort distribution KDE** (`plot_dist`, `kind="kde"`, `sample_dims=["draw"]`) - the
  posterior distribution of the **cohort-average expected upside (%)** for the earnings-window
  names, one KDE per chain (the chain-overlay doubles as a soft convergence check). A dashed
  line at 0 % marks break-even versus `last_price`: mass to the right is net implied upside
  across the cohort going into earnings.



```python
# 13a. Posterior-predictive forest with reference bands keyed off the stored `results`
# dataframe (cohort median of expected_pt_hdi_lo/hi) instead of re-pooled posterior draws.
try:
    ppc_tree, keep, forest_isins, cohort_last_price  # defined by the Section 13 forest cell
except NameError:
    print('Run the Section 13 posterior-predictive forest cell first '
          '(ppc_tree / keep / cohort_last_price not in scope).')
else:
    # Reference band straight from the section-10 screening table: cohort-median of the
    # per-ISIN stored 94% HDI bounds, plus the cohort-median smoothed expected_pt.
    band_lo = float(np.nanmedian(keep['expected_pt_hdi_lo']))
    band_hi = float(np.nanmedian(keep['expected_pt_hdi_hi']))
    band_med = float(np.nanmedian(keep['expected_pt']))

    pc2 = azp.plot_forest(
        ppc_tree, group='posterior_predictive', combined=True,
        labels=['isin'], backend='matplotlib',
    )
    pc2.map(
        azv.scatter_x, 'observations',
        data=ppc_tree.observed_data.ds, coords={'column': 'forest'},
        color='#ffb000',
    )
    pc2.map(
        azv.labelled_x, 'xlabel', coords={'column': 'forest'},
        text='expected price (simulated)  -  band = cohort-median results HDI '
             '[expected_pt_hdi_lo, expected_pt_hdi_hi]',
        ignore_aes='y',
    )
    pc2.coords = {'column': 'forest'}
    # Band from results df; reference lines = cohort-median expected_pt and last_price.
    pc2 = azp.add_bands(pc2, values=[(band_lo, band_hi)],
                        visuals={'ref_band': {'color': '#9b59b6', 'alpha': 0.15}})
    pc2 = azp.add_lines(pc2, values=band_med,
                        visuals={'ref_line': {'color': '#9b59b6', 'linewidth': 1.4}})
    pc2 = azp.add_lines(pc2, values=cohort_last_price,
                        visuals={'ref_line': {'color': '#bbbbbb',
                                              'linestyle': '--', 'linewidth': 1.3}})
    pc2.show()
    print(f'results-df cohort-median 94% HDI band: ({band_lo:.2f}, {band_hi:.2f});  '
          f'median expected_pt = {band_med:.2f};  cohort last_price = {cohort_last_price:.2f}.')

```


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_43_0.png)
    


    results-df cohort-median 94% HDI band: (15.24, 18.69);  median expected_pt = 16.90;  cohort last_price = 15.83.
    


```python
# 13b. Cohort distribution KDE (arviz_plots.plot_dist): posterior distribution of the
# cohort-average expected upside (%) for the earnings-window names, one KDE per chain.
try:
    idata, forest_isins  # defined by the Section 13 forest cell
except NameError:
    print('Run the Section 13 posterior-predictive forest cell first '
          '(forest_isins not in scope).')
else:
    # Cohort-average expected upside per (chain, draw): mean across the forest ISINs.
    cohort_upside = (idata.posterior['expected_upside']
                     .sel(isin=forest_isins).mean('isin') * 100.0
                     ).rename('cohort_expected_upside_pct')
    ds_dist = cohort_upside.to_dataset()

    pc3 = azp.plot_dist(
        ds_dist, kind='kde',
        var_names=['cohort_expected_upside_pct'],
        sample_dims=['draw'], backend='matplotlib',
    )
    pc3.add_title('Cohort expected upside (%) - KDE by chain (earnings window +/-5d)')
    # Break-even reference: 0% = simulated expected price equals last_price.
    pc3 = azp.add_lines(pc3, values=0.0,
                        visuals={'ref_line': {'color': '#bbbbbb',
                                              'linestyle': '--', 'linewidth': 1.3}})
    pc3.show()

    _mean = float(cohort_upside.mean().values)
    _p_pos = float((cohort_upside > 0).mean().values) * 100.0
    print(f'Cohort-average expected upside: posterior mean = {_mean:.2f}%;  '
          f'P(cohort upside > 0) = {_p_pos:.1f}%.')
```


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_44_0.png)
    


    Cohort-average expected upside: posterior mean = 30.87%;  P(cohort upside > 0) = 100.0%.
    

## 14. Comprehensive Summary - Recent Earnings Period vs Historical Data

This closing section consolidates the notebook's outputs into a single decision-oriented read
on the **expected price targets for names reporting within +/- 5 days** (`next_earnings`),
benchmarked against the broader **historical / baseline** data:

- **Cross-sectional benchmark** - the earnings cohort vs the rest of the modelled universe
  (the names *not* reporting this week), on expected upside, share positive, credible-band
  width (uncertainty) and Kalman shrinkage vs raw consensus - read from the section-10
  `results` table.
- **Time-series benchmark** - the mingled cohort's analyst-target trail reconstructed from the
  embedded `*_ago` history (section 12): the oldest historical consensus vs the most recent,
  and the latest Kalman-smoothed target's implied upside vs `last_price`.
- **Distributional view** - an `arviz_plots` KDE overlay of the posterior cohort-average vs
  universe-average expected upside, with a 0 % break-even reference line.

All blocks are guarded: the summary degrades to whatever upstream artifacts (`results`,
`cohort_meta`, `comparison`) are present in the kernel, so it still produces a partial read if
the DB-dependent sections were skipped.



```python
# Section 14: comprehensive earnings-cohort vs historical-data summary. Reuses `results`
# (section 10), `cohort_meta` (section 13), `comparison` (section 12) and `idata` when present.
def _fmt(x, nd=1, suf=''):
    """Format a possibly-NaN/None scalar for narrative output."""
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return 'n/a'
        return f'{x:.{nd}f}{suf}'
    except Exception:
        return 'n/a'


def _label(row):
    t = row.get('ticker')
    return t if isinstance(t, str) and t.strip() else str(row['isin'])


def _band_width_pct(df):
    denom = df['expected_pt'].replace(0, np.nan)
    return (df['expected_pt_hdi_hi'] - df['expected_pt_hdi_lo']) / denom * 100.0


def _shrink_pct(df):
    denom = df['observed_pt'].replace(0, np.nan)
    return (df['expected_pt'] / denom - 1.0) * 100.0


try:
    results
except NameError:
    print('Section 10 `results` not in scope - run sections 5-10 first.')
else:
    universe = results.copy()

    # ---- A. Cross-sectional: earnings cohort vs the rest of the modelled universe ----
    try:
        _cohort_ids = set(cohort_meta['isin'].astype(str))
    except NameError:
        _cohort_ids = None

    if _cohort_ids:
        _in = universe['isin'].astype(str).isin(_cohort_ids)
        cohort, rest = universe[_in].copy(), universe[~_in].copy()
        groups = [('Earnings cohort (+/-5d)', cohort),
                  ('Historical baseline (not reporting)', rest),
                  ('Full universe', universe)]
    else:
        cohort = rest = None
        groups = [('Full universe', universe)]
        print('Section 13 `cohort_meta` not in scope - showing the universe only. '
              'Run Section 13 to populate the earnings cohort comparison.')

    _rows = []
    for label, df in groups:
        if df is None or len(df) == 0:
            continue
        _rows.append({
            'group': label,
            'n_names': int(len(df)),
            'median_upside_%': df['expected_upside_pct'].median(),
            'mean_upside_%': df['expected_upside_pct'].mean(),
            'positive_upside_%': (df['expected_upside_pct'] > 0).mean() * 100.0,
            'median_band_width_%': _band_width_pct(df).median(),
            'median_shrink_vs_consensus_%': _shrink_pct(df).median(),
            'median_n_analysts': (df['n_analysts'].median()
                                  if 'n_analysts' in df else np.nan),
        })
    summary_tbl = pd.DataFrame(_rows).set_index('group').round(2)
    print('Cross-sectional summary - expected price targets by group:')
    display(summary_tbl)

    # Cohort sector tilt (composition of the names reporting this week).
    if cohort is not None and len(cohort) and 'sector' in cohort.columns:
        sector_mix = (cohort.assign(sector=cohort['sector'].fillna('Unknown'))
                      .groupby('sector')
                      .agg(n=('isin', 'size'),
                           median_upside_pct=('expected_upside_pct', 'median'))
                      .sort_values('n', ascending=False).round(2))
        print('\nEarnings-cohort sector tilt:')
        display(sector_mix.head(10))

    # ---- B. Time-series: recent vs historical mingled cohort price-target trail ----
    try:
        _cmp = comparison
    except NameError:
        _cmp = None
    hist_drift = implied_now = first = last = None
    if _cmp is not None and len(_cmp) >= 2:
        first, last = _cmp.iloc[0], _cmp.iloc[-1]
        hist_drift = ((last['observed_pt'] / first['observed_pt'] - 1.0) * 100.0
                      if first['observed_pt'] else np.nan)
        implied_now = ((last['expected_pt'] / last['last_price'] - 1.0) * 100.0
                       if last['last_price'] else np.nan)

    # ---- C. Headline narrative ----
    print('\n' + '=' * 74)
    print('KEY INSIGHTS - recent earnings period vs historical data')
    print('=' * 74)
    if cohort is not None and len(cohort):
        cu = cohort['expected_upside_pct'].median()
        ru = rest['expected_upside_pct'].median() if rest is not None and len(rest) else np.nan
        print(f'- {len(cohort)} names report within +/-5d. Median expected upside '
              f'{_fmt(cu, 1, "%")} vs {_fmt(ru, 1, "%")} for non-reporting names '
              f'(delta {_fmt(cu - ru, 1, " pp")}).')
        print(f'- {_fmt((cohort["expected_upside_pct"] > 0).mean() * 100, 0, "%")} of the cohort '
              f'has positive expected upside; median credible band width '
              f'{_fmt(_band_width_pct(cohort).median(), 1, "%")} '
              f'(universe {_fmt(_band_width_pct(universe).median(), 1, "%")}).')
        sh = _shrink_pct(cohort).median()
        print(f'- Kalman-smoothed targets sit {_fmt(abs(sh), 1, "%")} '
              f'{"above" if sh >= 0 else "below"} raw consensus (median) - shrinkage toward '
              f'the hierarchical group mean.')
        _top = cohort.sort_values('expected_upside_pct', ascending=False).head(3)
        _bot = cohort.sort_values('expected_upside_pct').head(3)
        _names = lambda d: ', '.join(f'{_label(r)} ({_fmt(r["expected_upside_pct"], 0, "%")})'
                                     for _, r in d.iterrows())
        print(f'- Highest expected upside: {_names(_top)}.')
        print(f'- Lowest / most downside : {_names(_bot)}.')
    else:
        print('- Earnings cohort not available (Section 13 was skipped); '
              'cross-sectional cohort insights omitted.')
    if first is not None and last is not None:
        print(f'- Historical target trail ({first["asof_date"]} -> {last["asof_date"]}): mingled '
              f'cohort consensus {"rose" if (hist_drift or 0) >= 0 else "fell"} '
              f'{_fmt(abs(hist_drift), 1, "%")}; latest Kalman-smoothed target implies '
              f'{_fmt(implied_now, 1, "%")} upside vs cohort last price.')
    else:
        print('- Historical mingled trail not available (Section 12 was skipped); '
              'time-series drift insight omitted.')

    # ---- D. Distributional view: cohort vs universe expected upside (arviz_plots KDE) ----
    try:
        if _cohort_ids:
            _eu = idata.posterior['expected_upside'] * 100.0
            _modelled = set(_eu.coords['isin'].values.astype(str).tolist())
            _cohort_post = [i for i in _cohort_ids if i in _modelled]
            if _cohort_post:
                _cohort_avg = _eu.sel(isin=_cohort_post).mean('isin')
                _univ_avg = _eu.mean('isin')
                _stacked = xr.concat(
                    [_cohort_avg, _univ_avg],
                    dim=pd.Index(['earnings_cohort', 'universe'], name='group'),
                ).rename('avg_expected_upside_pct')
                pc_sum = azp.plot_dist(
                    _stacked.to_dataset(), kind='kde',
                    var_names=['avg_expected_upside_pct'],
                    sample_dims=['chain', 'draw'], backend='matplotlib',
                )
                pc_sum.add_title('Expected upside (%): earnings cohort vs universe '
                                 '(posterior cross-sectional average)')
                pc_sum = azp.add_lines(
                    pc_sum, values=0.0,
                    visuals={'ref_line': {'color': '#bbbbbb',
                                          'linestyle': '--', 'linewidth': 1.3}},
                )
                pc_sum.show()
    except Exception as _e:  # pragma: no cover - plot is best-effort
        print(f'Summary KDE overlay skipped: {_e!r}')
```

    Cross-sectional summary - expected price targets by group:
    


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
      <th>n_names</th>
      <th>median_upside_%</th>
      <th>mean_upside_%</th>
      <th>positive_upside_%</th>
      <th>median_band_width_%</th>
      <th>median_shrink_vs_consensus_%</th>
      <th>median_n_analysts</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Earnings cohort (+/-5d)</th>
      <td>220</td>
      <td>25.04</td>
      <td>26.43</td>
      <td>90.00</td>
      <td>20.32</td>
      <td>-0.70</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Historical baseline (not reporting)</th>
      <td>6057</td>
      <td>19.68</td>
      <td>20.15</td>
      <td>86.92</td>
      <td>19.89</td>
      <td>-0.24</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Full universe</th>
      <td>6277</td>
      <td>19.83</td>
      <td>20.37</td>
      <td>87.03</td>
      <td>19.90</td>
      <td>-0.25</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Earnings-cohort sector tilt:
    


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
      <th>n</th>
      <th>median_upside_pct</th>
    </tr>
    <tr>
      <th>sector</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Industrials</th>
      <td>53</td>
      <td>18.49</td>
    </tr>
    <tr>
      <th>Consumer Discretionary</th>
      <td>43</td>
      <td>29.68</td>
    </tr>
    <tr>
      <th>Information Technology</th>
      <td>36</td>
      <td>17.55</td>
    </tr>
    <tr>
      <th>Health Care</th>
      <td>29</td>
      <td>33.72</td>
    </tr>
    <tr>
      <th>Consumer Staples</th>
      <td>22</td>
      <td>26.16</td>
    </tr>
    <tr>
      <th>Materials</th>
      <td>20</td>
      <td>32.39</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>8</td>
      <td>6.20</td>
    </tr>
    <tr>
      <th>Communication Services</th>
      <td>5</td>
      <td>43.81</td>
    </tr>
    <tr>
      <th>Energy</th>
      <td>4</td>
      <td>24.95</td>
    </tr>
  </tbody>
</table>
</div>


    
    ==========================================================================
    KEY INSIGHTS - recent earnings period vs historical data
    ==========================================================================
    - 220 names report within +/-5d. Median expected upside 25.0% vs 19.7% for non-reporting names (delta 5.4 pp).
    - 90% of the cohort has positive expected upside; median credible band width 20.3% (universe 19.9%).
    - Kalman-smoothed targets sit 0.7% below raw consensus (median) - shrinkage toward the hierarchical group mean.
    - Highest expected upside: 2432 (139%), EOS (102%), 2696 (94%).
    - Lowest / most downside : FCEL (-51%), PL (-27%), CIEN (-22%).
    - Historical target trail (2021-06-03 -> 2027-01-01): mingled cohort consensus rose 26.4%; latest Kalman-smoothed target implies 24.5% upside vs cohort last price.
    


    
![png](pymc_kalman_filter_pt_files/pymc_kalman_filter_pt_46_5.png)
    

