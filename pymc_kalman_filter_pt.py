"""PyMC Kalman-filter price-target model — script form of ``pymc_kalman_filter_pt.ipynb``.

The cross-sectional spine is the **fused MvGRW panel model** (Model A + Model B,
:func:`probabilistic_ml_model.pymc_models.KalmanFilterModel.build_fused_kalman_pt_model`):

* **Model B spine** — a diagonal (NUTS-safe) Multivariate Gaussian Random Walk over
  the ``(isin, time, y_series)`` response tensor with a cross-sectional baseline
  ``mu_isin``.
* **Model A refinement** — the volatility-aware ``expected_return`` →
  ``risk_adj_return`` latent (with a non-centred logit-normal ``achieve_prob``)
  *is* the GRW baseline ``mu_isin``, and the heteroscedastic scale
  ``sigma_isin = sigma_base * (1 + cv) / sqrt(n)`` replaces the cv-free form.

The Kalman-specific change vs. the price-target panel: the risk adjustment is keyed on
**expected volatility** (the ``feat_vol_*`` term-structure mean) rather than analyst
conviction, so the latent target reads ``risk_adj_return = expected_return`` *given
expected_volatility*. Per-ISIN screening signals are drawn from the posterior
``risk_adj_return`` / ``sigma_isin`` / ``nu`` via the canonical structural-TS Monte-Carlo
helpers (:func:`simulate_lagged_risk_adjusted_returns` / :func:`summarize_mc_returns`).

The workflow has two halves:

* **Fused cross-sectional panel** (sections 4–10): one row per ISIN from
  ``pml.mv_pymc_kalman_pt``, fit with the fused MvGRW + volatility-conditioned model.
* **Single-security / cohort time-series** (sections 11–14): the literal
  ``KalmanFilterPriceTarget`` GRW filter on the embedded ``*_ago`` price-target
  history, plus the mingled earnings-cohort consensus and decision-oriented summaries.

Schema-aligned with (single source of truth = the ``pml`` schema):

* MV: ``pml.mv_pymc_kalman_pt``
* Catalogue: ``pml.vw_pymc_feature_catalogue WHERE model_target = 'kalman_pt'``
* Coords: ``pml.vw_pml_df_coords``

Usage::

    python pymc_kalman_filter_pt.py
"""

from __future__ import annotations

import importlib.util as _ilu
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# ArviZ 1.0 split-package imports: arviz-plots owns ``style`` + plotting, arviz-stats
# owns ``summary`` / ``rhat`` / ``ess``. Address each submodule directly.
import arviz_plots as azp
import arviz_stats as azs
import matplotlib.colors as _mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
from arviz_plots import visuals as azv  # low-level primitives for custom composition
from cycler import cycler as _cycler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sqlalchemy import create_engine, text

from probabilistic_ml_model._pymc_arviz_compat import extend_datatree
from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
    KalmanFilterPriceTarget,
    KalmanPanelInputs,
    build_fused_kalman_pt_model,
)
from probabilistic_ml_model.pymc_models._price_target_mc import (
    simulate_lagged_risk_adjusted_returns,
    summarize_mc_returns,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# --- Data-source queries (single source of truth = the pml schema) -----------
# Cross-sectional snapshot: one row per ISIN with a usable analyst target, scoped
# to names whose next earnings land in the current modelling horizon.
KALMAN_DF_QUERY = """
    SELECT *
    FROM pml.mv_pymc_kalman_pt mpkp
    WHERE observed_pt IS NOT NULL
      AND next_earnings >= '2026-01-01'
"""

# Per-model feature catalogue (pymc_role / feature_role / alias) for kalman_pt.
FEATURE_CATALOGUE_QUERY = """
    SELECT *
    FROM pml.vw_pymc_feature_catalogue
    WHERE model_target = 'kalman_pt'
    ORDER BY pymc_role, feature_role, feature_alias
"""

# Canonical schema of pml.mv_pymc_kalman_pt: per-trail drift feature for every
# price_* / price_target_* family plus the noise wideners. Resilience fallback for
# MV columns absent from the catalogue snapshot.
KNOWN_FEATURES = ['feat_pt_drift', 'feat_price_drift',
                  'feat_pt_high_drift', 'feat_pt_low_drift', 'feat_pt_median_drift',
                  'feat_coverage_drift', 'feat_pt_noise_drift',
                  'feat_pt_noise_sigma', 'feat_pt_range_norm',
                  'feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y',
                  'feat_total_return_ytd']

# Hierarchical classification coords (categorical group effects), distinct from the
# fiscal-calendar DATE anchors. Both carry pymc_role='coord', but the date anchors
# define the single-security time axis (sections 11–13) and must NOT be treated as
# categorical effects in the cross-sectional model.
CLASSIFICATION_COORDS_ALL = (
    'isin', 'ticker', 'name', 'region', 'country', 'trading_country',
    'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry',
    'next_earnings_when', 'next_earnings_status',
)
FISCAL_CALENDAR_COLS_ALL = (
    'income_statement_report_date', 'next_earnings', 'fy_end_date',
    'next_income_statement_report_date', 'next_fy_end_date', 'expected_report_date',
)
DAY_COUNT_COLS_ALL = (
    'days_to_next_earnings', 'days_since_last_report', 'days_to_next_fy_end',
    'days_to_next_report', 'days_to_expected_report', 'days_to_fy_end',
)

# Candidate categorical group-effect coords for the hierarchical drift mean (section 5).
_CANDIDATE_GROUPS = ('region', 'unit', 'style_class', 'size_class', 'sector', 'industry')

# *_ago price-target history column pattern shared by sections 11–13.
HIST_COL_PATTERN = (r"^(price_target(_high|_low|_median)?|price)"
                    r"_(5d|1w|1m|3m|6m|1y|3y|5y|mtd|qtd|ytd)_ago$")


# ``display`` exists only in IPython; fall back to ``print`` for plain-script runs.
try:  # pragma: no cover - depends on runtime
    from IPython.display import display
except ImportError:  # pragma: no cover
    def display(obj: object) -> None:
        print(obj)


# =============================================================================
# 1. Plotting setup + reusable helpers
# =============================================================================
def setup_plotting() -> None:
    """Pin the arviz-plots/matplotlib backend and install the dark notebook theme.

    Notes
    -----
    arviz_plots builds its per-chain colour aesthetic by reshaping the *active*
    matplotlib colour cycle. seaborn installs that cycle as RGB tuples; arviz_plots
    then reshapes ``colours[:n_chains]``, which for 4 chains turns 4 RGB triples into
    an array of size 12 and raises ``cannot reshape array of size 12 into shape (4,)``.
    We re-express the cycle as hex strings so ``plot_trace`` works under the theme.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)

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

    _cycle_cols = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if _cycle_cols and not all(isinstance(_c, str) for _c in _cycle_cols):
        plt.rcParams['axes.prop_cycle'] = _cycler(
            color=[_mcolors.to_hex(_c) for _c in _cycle_cols]
        )
    plt.rcParams['figure.dpi'] = 110


def _present_vars(idata, candidates: Sequence[str]) -> list[str]:
    """Return the subset of ``candidates`` actually in ``idata.posterior``.

    Which posterior variables exist depends on how ``KalmanFilterPriceTarget.fit`` was
    parameterized (marginalized → ``log_state_init``; non-centred → ``z_innov``;
    stochastic-volatility → ``vol_step_size`` / ``nu_obs`` / ``vol_anchor_offset``), so
    summaries filter to what is present rather than hard-coding names.
    """
    post = getattr(idata, 'posterior', idata)
    have = set(post.data_vars)
    return [v for v in candidates if v in have]


def _kf_fit_kind(idata) -> str:
    """Human label for the parameterization actually fitted (from posterior vars)."""
    post = getattr(idata, 'posterior', idata)
    dv = set(post.data_vars)
    if 'log_vol' in dv:
        return 'Stochastic-volatility GRW (+trend)'
    if 'log_state_init' in dv:
        return 'Marginalized GRW (+trend)'
    return 'Non-centred GRW (+trend)'


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
        Inference object whose ``posterior`` holds ``state_var`` over a ``time`` dim.
    state_var
        Posterior variable carrying the price-space latent state.
    observed
        Observed analyst price targets aligned to the ``time`` axis (plotted as points).
    dates
        Time index aligned to the ``time`` dim. When omitted, the posterior ``time``
        coord is used (treated as datetime if it parses as such).
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

    # Resolve the x-axis: prefer explicit ``dates``, else the posterior ``time`` coord.
    # Only treat the coord as datetime when its dtype actually is one -- otherwise
    # pd.to_datetime would silently coerce a plain integer time index into 1970-epoch
    # timestamps.
    if dates is None and "time" in post.coords:
        coord_vals = np.asarray(post["time"].values)
        if np.issubdtype(coord_vals.dtype, np.datetime64):
            dates = pd.DatetimeIndex(coord_vals)
        elif coord_vals.dtype == object:
            parsed = pd.to_datetime(coord_vals, errors="coerce")
            if not bool(np.asarray(pd.isna(parsed)).all()):
                dates = pd.DatetimeIndex(parsed)
    use_dates = (
            dates is not None
            and len(dates) == n_time
            and not bool(np.asarray(pd.isna(dates)).all())
    )
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


def plot_kalman_forecast(idata_fit, pred, *, observed=None, dates=None,
                         last_price=None, ticker=None, state_var='state',
                         figsize=(11, 5), hist_color='#56b4e9',
                         fc_color='#cc79a7', observed_color='#ffb000'):
    """Overlay the fitted smoothed state with the structural forecast bands.

    Mirrors the reference notebook's "Posterior Predictions Plotted": the fitted
    Kalman-smoothed state + HDI up to the last observation, a vertical boundary at
    "now", then ``KalmanFilterPriceTarget.forecast()`` predictive bands extending to
    the future fiscal-calendar events.

    Parameters
    ----------
    idata_fit
        InferenceData from :meth:`KalmanFilterPriceTarget.fit` (``state`` over ``time``).
    pred
        Output of :meth:`KalmanFilterPriceTarget.forecast` (``predictions`` group or a
        raw ``xarray.Dataset``) with ``forecast_pt`` over ``time_future``.
    observed, dates, last_price, ticker
        Observed targets, historical as-of dates, the spot price reference, and a title
        label, all aligned to the fitted ``time`` axis.
    """
    post = idata_fit.posterior[state_var]
    n_time = post.sizes['time']
    use_dates = dates is not None and len(dates) == n_time
    hx = mdates.date2num(np.asarray(dates)) if use_dates else np.arange(n_time, dtype=float)
    hist_med = post.median(('chain', 'draw')).values
    _hdi = post.azstats.hdi(prob=0.94)
    hlo = _hdi.sel(ci_bound='lower').values
    hhi = _hdi.sel(ci_bound='upper').values

    pg = pred.predictions if hasattr(pred, 'predictions') else pred
    fpt = pg['forecast_pt']
    tf = np.asarray(pg['time_future'].values)
    if np.issubdtype(tf.dtype, np.datetime64):
        fx = mdates.date2num(tf)
    else:
        fx = hx[-1] + np.asarray(tf, dtype=float)
    f_med = fpt.median(('chain', 'draw')).values
    f_lo = fpt.quantile(0.03, dim=('chain', 'draw')).values
    f_hi = fpt.quantile(0.97, dim=('chain', 'draw')).values

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(hx, hlo, hhi, color=hist_color, alpha=0.25, label='94% HDI (fit)')
    ax.plot(hx, hist_med, color=hist_color, lw=2.2, label='Kalman state (fit)')
    if observed is not None:
        ax.scatter(hx, np.asarray(observed, dtype=float), color=observed_color, s=34,
                   zorder=6, edgecolor='#1e1e1e', linewidth=0.6, label='observed target')
    ax.fill_between(fx, f_lo, f_hi, color=fc_color, alpha=0.22, label='94% PI (forecast)')
    ax.plot(np.r_[hx[-1], fx], np.r_[hist_med[-1], f_med], color=fc_color, lw=2.2,
            ls='--', marker='o', label='forecast pt')
    ax.axvline(hx[-1], color='#bbbbbb', ls=':', lw=1.2)
    if last_price is not None and np.isfinite(last_price):
        ax.axhline(float(last_price), ls='--', color='#bbbbbb', lw=1.0, label='last price')
    if 'label' in pg.coords:
        for xv, lb, yv in zip(fx, np.asarray(pg['label'].values), f_med):
            ax.annotate(str(lb), (xv, yv), fontsize=7, rotation=25,
                        color=fc_color, ha='left', va='bottom')
    if use_dates:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.set_xlabel('as-of date' if use_dates else 'time step')
    ax.set_ylabel('price target')
    ax.set_title('Kalman state + structural forecast'
                 + (f' — {ticker}' if ticker else ''))
    ax.legend(fontsize=8, framealpha=0.25, loc='best')
    return fig, ax


def build_noise_wideners(df: pd.DataFrame, *, fillna: bool = True) -> dict[str, np.ndarray]:
    """Model-facing observation-noise wideners for one snapshot frame.

    SINGLE SOURCE OF TRUTH for the observation-noise wideners shared by the §2.4c EDA
    panel and the §4.2 model containers, so the picture and the likelihood agree by
    construction. Each quantity is returned in the units the measurement model consumes
    (``sigma_obs = sigma_obs_base * (1 + range + cv + 0.5*vol) / sqrt(n_analysts)``).

    Parameters
    ----------
    df
        A ``kalman_df``-shaped frame (universe EDA or the modelling ``model_df``).
    fillna
        ``True`` (the §4.2 model contract): missing inputs -> 0 and the non-negative
        wideners are clipped at 0, so ``sigma_obs`` is always finite. ``False`` (the EDA
        contract): keep NaNs and the raw signed values, so the marginals show the true
        distribution before the model's 0-fill / clip.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``range``, ``cv``, ``vol``, ``sqrt_n`` plus the realised per-row
        ``multiplier`` = ``(1 + range + cv + 0.5*vol) / sqrt_n``.
    """
    range_col = 'feat_pt_range_norm' if 'feat_pt_range_norm' in df.columns else None
    sigma_col = 'feat_pt_noise_sigma' if 'feat_pt_noise_sigma' in df.columns else None
    vol_cols = [c for c in ('feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y')
                if c in df.columns]

    def _col(name):
        s = df[name].astype('float64') if name and name in df.columns \
            else pd.Series(np.nan, index=df.index)
        return s.fillna(0.0) if fillna else s

    range_s = _col(range_col)
    if sigma_col and sigma_col in df.columns:
        lp = df['last_price'].astype('float64')
        lp = lp.clip(lower=1e-9) if fillna else lp.where(lp > 0)
        cv_s = _col(sigma_col) / lp
    else:
        cv_s = pd.Series(0.0 if fillna else np.nan, index=df.index)
    vol_s = (df[vol_cols].astype('float64').mean(axis=1) if vol_cols
             else pd.Series(0.0 if fillna else np.nan, index=df.index))
    if 'n_analysts' in df.columns:
        n_an = df['n_analysts'].astype('float64').clip(lower=1.0)
    else:
        n_an = pd.Series(1.0, index=df.index)
    sqrt_n = np.sqrt(n_an.to_numpy())

    if fillna:
        # Model contract: non-negative, finite.
        range_s = range_s.clip(lower=0.0)
        cv_s = cv_s.clip(lower=0.0)
        vol_s = vol_s.fillna(0.0).clip(lower=0.0)
        mult = (1.0 + range_s.to_numpy() + cv_s.to_numpy()
                + 0.5 * vol_s.to_numpy()) / sqrt_n
    else:
        # EDA contract: clip only the genuinely-non-negative terms for the multiplier
        # so it stays comparable to the model, but leave per-feature marginals raw.
        mult = (1.0 + range_s.clip(lower=0).fillna(0).to_numpy()
                + cv_s.clip(lower=0).fillna(0).to_numpy()
                + 0.5 * vol_s.clip(lower=0).fillna(0).to_numpy()) / sqrt_n

    return {'range': range_s.to_numpy(), 'cv': cv_s.to_numpy(),
            'vol': vol_s.to_numpy(), 'sqrt_n': sqrt_n, 'multiplier': mult}


def build_realized_vol_path(vol_term_structure, dates) -> Optional[np.ndarray]:
    """Map a 1m/3m/6m/1y volatility term-structure onto an irregular asof_date grid.

    Returns a strictly-positive per-time realized-volatility anchor aligned to
    ``dates`` (recent dates -> short-horizon vol, older dates -> long-horizon vol, via
    linear interpolation in lookback-days). Only the *shape* informs the SV log-vol
    prior, so the units of ``vol_term_structure`` (percent vs decimal) are irrelevant.
    Returns ``None`` when no finite, positive term-structure point is available.
    """
    horizons = np.array([30.0, 90.0, 180.0, 365.0])  # 1m, 3m, 6m, 1y in days
    vols = np.asarray(vol_term_structure, dtype='float64')
    ok = np.isfinite(vols) & (vols > 0)
    if not ok.any():
        return None
    d = pd.DatetimeIndex(dates)
    lookback = (d.max() - d).days.to_numpy().astype('float64')  # 0 at the latest date
    order = np.argsort(horizons[ok])
    rv = np.interp(lookback, horizons[ok][order], vols[ok][order])  # clamps at the ends
    return np.clip(rv, 1e-6, None)


def resolve_db_url(env_file: str = 'environment_variables.txt') -> str:
    """Return ``DB_URL`` from the environment, falling back to environment_variables.txt.

    The process may have been started without sourcing ``set_env.ps1``, in which case
    ``os.environ`` has no ``DB_URL``; we then parse the ``KEY=VALUE`` lines of the
    project's ``environment_variables.txt`` as a fallback.
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
        "Run `. .\\set_env.ps1` before launching, or add a DB_URL line."
    )


def fetch_history_columns(engine, keep: Sequence[str],
                          pattern: str = HIST_COL_PATTERN) -> tuple[list[str], str]:
    """Discover the ``*_ago`` history columns + reference columns present in pml.pml_df.

    Returns the ordered column list and a pre-quoted ``SELECT`` column expression.
    """
    with engine.connect() as conn:
        hist_cols = pd.read_sql(
            text("""
                 SELECT column_name
                 FROM information_schema.columns
                 WHERE table_schema = 'pml'
                   AND table_name = 'pml_df'
                   AND (column_name ~ :pat OR column_name = ANY (:keep))
                 ORDER BY column_name
                 """),
            conn, params={'pat': pattern, 'keep': list(keep)},
        )['column_name'].tolist()
    col_sql = ', '.join(f'"{c}"' for c in hist_cols)
    return hist_cols, col_sql


# =============================================================================
# Data loading + feature-role resolution
# =============================================================================
def load_kalman_df(engine) -> pd.DataFrame:
    """Load the cross-sectional ``pml.mv_pymc_kalman_pt`` snapshot (one row per ISIN)."""
    with engine.connect() as conn:
        df = pd.read_sql(text(KALMAN_DF_QUERY), conn)
    logger.info("Loaded kalman_df: %s", df.shape)
    return df


def load_feature_catalogue(engine) -> pd.DataFrame:
    """Load the ``kalman_pt`` rows of ``pml.vw_pymc_feature_catalogue``."""
    with engine.connect() as conn:
        cat = pd.read_sql(text(FEATURE_CATALOGUE_QUERY), conn)
    logger.info("Loaded feature_catalogue: %s", cat.shape)
    return cat


@dataclass
class FeatureRoles:
    """Column groups resolved by ``pymc_role`` from the feature catalogue + MV schema."""

    predictor_cols: list[str] = field(default_factory=list)
    coord_cols: list[str] = field(default_factory=list)
    response_cols: list[str] = field(default_factory=list)
    classification_coords: list[str] = field(default_factory=list)
    fiscal_calendar_cols: list[str] = field(default_factory=list)
    day_count_cols: list[str] = field(default_factory=list)


def resolve_feature_roles(kalman_df: pd.DataFrame,
                          feature_catalogue: pd.DataFrame) -> FeatureRoles:
    """Resolve column groups by ``pymc_role`` (catalogue SSOT, MV-schema fallback).

    ``predictor_cols`` mirrors ``KalmanFilterPriceTarget._resolve_kalman_feature_aliases``
    (mutable_predictor aliases for ``model_target='kalman_pt'``); ``KNOWN_FEATURES`` is a
    resilience fallback for MV columns absent from the catalogue snapshot.
    """
    if 'model_target' in feature_catalogue.columns:
        if not (feature_catalogue['model_target'] == 'kalman_pt').all():
            warnings.warn("feature_catalogue is not fully filtered to model_target='kalman_pt'.")

    catalogue = feature_catalogue.copy()
    catalogue['present'] = catalogue['feature_alias'].isin(kalman_df.columns)

    role_summary = (
        catalogue.groupby(['pymc_role', 'feature_role'])['present']
        .agg(n_columns='size', n_present='sum')
        .reset_index()
    )
    display(role_summary)

    present = catalogue.loc[catalogue['present']]
    predictor_cols = present.loc[present['pymc_role'] == 'mutable_predictor', 'feature_alias'].tolist()
    coord_cols = present.loc[present['pymc_role'] == 'coord', 'feature_alias'].tolist()
    response_cols = present.loc[present['pymc_role'].isin(['response', 'observed']), 'feature_alias'].tolist()

    for col in KNOWN_FEATURES:
        if col in kalman_df.columns and col not in predictor_cols:
            predictor_cols.append(col)
    if 'observed_pt' in kalman_df.columns and 'observed_pt' not in response_cols:
        response_cols.append('observed_pt')

    classification_coords = [c for c in CLASSIFICATION_COORDS_ALL if c in kalman_df.columns]
    fiscal_calendar_cols = [c for c in FISCAL_CALENDAR_COLS_ALL if c in kalman_df.columns]
    day_count_cols = [c for c in DAY_COUNT_COLS_ALL if c in kalman_df.columns]

    # Keep only non-date coords as categorical-effect candidates; fall back to the
    # curated classification list when the catalogue exposes no plain coords.
    coord_cols = [c for c in coord_cols if c not in fiscal_calendar_cols] or classification_coords.copy()

    roles = FeatureRoles(
        predictor_cols=predictor_cols, coord_cols=coord_cols, response_cols=response_cols,
        classification_coords=classification_coords, fiscal_calendar_cols=fiscal_calendar_cols,
        day_count_cols=day_count_cols,
    )
    print(f'#predictors     : {len(predictor_cols)} -> {predictor_cols}')
    print(f'#response       : {len(response_cols)} -> {response_cols}')
    print(f'#coords         : {len(coord_cols)} -> {coord_cols}')
    print(f'#classification : {len(classification_coords)} -> {classification_coords}')
    print(f'#fiscal-calendar: {len(fiscal_calendar_cols)} -> {fiscal_calendar_cols}')
    print(f'#day-count      : {len(day_count_cols)} -> {day_count_cols}')
    return roles


# =============================================================================
# 2. Exploratory Data Analysis (EDA)
# =============================================================================
def run_eda(kalman_df: pd.DataFrame, roles: FeatureRoles) -> None:
    """Exploratory views over ``kalman_df`` (source: ``pml.mv_pymc_kalman_pt``).

    Treats the ``feat_*`` columns through the lens of their state-space role: drift
    features -> the state-transition mean (``beta`` slopes), noise wideners -> the
    measurement-noise scale ``sigma_obs``. All plots are side-effecting only.
    """
    # 2.1 Shape, dtype, missingness overview.
    eda_overview = pd.DataFrame({
        'dtype': kalman_df.dtypes.astype(str),
        'n_missing': kalman_df.isna().sum(),
        'pct_missing': (kalman_df.isna().mean() * 100).round(1),
        'n_unique': kalman_df.nunique(),
    })
    print(f'kalman_df shape: {kalman_df.shape}')
    display(eda_overview.sort_values('pct_missing', ascending=False).head(30))

    # 2.2 Expected upside by industry — arviz_plots ridge.
    _d = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    _d['upside_pct'] = (_d['observed_pt'] / _d['last_price'] - 1.0) * 100.0
    _d['upside_pct'] = _d['upside_pct'].clip(-100, 500)
    _d['industry'] = (_d['industry'] if 'industry' in _d.columns
                      else pd.Series('Unknown', index=_d.index))
    _d['industry'] = _d['industry'].fillna('Unknown').astype(str)
    _counts = _d['industry'].value_counts()
    _keep = _counts[_counts >= 5].index.tolist()
    _d = _d[_d['industry'].isin(_keep)]
    _industries = sorted(_d['industry'].unique())
    if _industries:
        _max_n = int(_d['industry'].value_counts().max())
        _arr = np.full((len(_industries), _max_n), np.nan)
        for i, ind in enumerate(_industries):
            vals = _d.loc[_d['industry'] == ind, 'upside_pct'].to_numpy()
            _arr[i, :len(vals)] = vals
        _ds_ridge = xr.Dataset(
            {'implied_upside_pct': (('industry', 'sample'), _arr)},
            coords={'industry': _industries},
        )
        pc = azp.plot_ridge(_ds_ridge, var_names=['implied_upside_pct'],
                            sample_dims=['sample'], combined=True)
        pc.add_title('Implied upside (%) by industry — consensus observed_pt vs last_price')
        pc.show()
        display(_d['upside_pct'].describe())

    # 2.3 Classification-coord cardinality.
    card = {c: kalman_df[c].nunique() for c in roles.classification_coords}
    display(pd.Series(card).sort_values(ascending=False))

    # 2.4a Distributional summary of the feat_* columns via arviz-stats.
    eda_drift = [c for c in ('feat_pt_drift', 'feat_price_drift', 'feat_pt_high_drift',
                             'feat_pt_low_drift', 'feat_pt_median_drift',
                             'feat_coverage_drift', 'feat_pt_noise_drift',
                             'feat_total_return_ytd')
                 if c in kalman_df.columns]
    eda_noise = [c for c in ('feat_pt_range_norm', 'feat_pt_noise_sigma',
                             'feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y')
                 if c in kalman_df.columns]
    eda_features = eda_drift + eda_noise

    _eda = kalman_df[eda_features].astype('float64')
    _lo = _eda.quantile(0.01)
    _hi = _eda.quantile(0.99)
    _eda_w = _eda.clip(lower=_lo, upper=_hi, axis=1)
    _eda_ds = xr.Dataset({col: ('sample', _eda_w[col].to_numpy()) for col in eda_features})
    feat_summary = azs.summary(
        _eda_ds, var_names=eda_features, kind='stats', round_to=4,
        sample_dims='sample', skipna=True,
    )
    print(f'Distributional summary (winsorised 1/99 pct) for {len(eda_features)} feat_* columns:')
    display(feat_summary)

    # 2.4b Drift-feature marginals as an arviz_plots ridge (z-scored).
    if eda_drift:
        _drift_std = ((_eda_w[eda_drift] - _eda_w[eda_drift].mean())
                      / _eda_w[eda_drift].std(ddof=0).replace(0, 1.0)).dropna()
        _n = len(_drift_std)
        _drift_arr = np.full((len(eda_drift), _n), np.nan)
        for i, col in enumerate(eda_drift):
            _drift_arr[i] = _drift_std[col].to_numpy()
        _ds_drift_ridge = xr.Dataset(
            {'drift_feature_z': (('feature', 'sample'), _drift_arr)},
            coords={'feature': eda_drift},
        )
        pc = azp.plot_ridge(_ds_drift_ridge, var_names=['drift_feature_z'],
                            sample_dims=['sample'], combined=True)
        pc.add_title('Standardised drift-feature marginals (state-transition mean inputs)')
        pc.show()

    # 2.4c Observation-noise wideners — RAW (un-winsorised), on the model-facing scale.
    _w = build_noise_wideners(kalman_df, fillna=False)
    _widener_specs = [
        ('range  (feat_pt_range_norm)', _w['range'], False),
        ('cv  (feat_pt_noise_sigma / last_price)', _w['cv'], False),
        ('vol mean  (feat_vol_*, mean)', _w['vol'], False),
    ]
    if 'feat_pt_noise_drift' in kalman_df.columns:
        _widener_specs.insert(
            2, ('noise drift  (feat_pt_noise_drift, signed)',
                kalman_df['feat_pt_noise_drift'].astype('float64').to_numpy(), True))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    _pal = sns.color_palette('flare', len(_widener_specs))
    _rows = []
    for (label, arr, is_signed), c in zip(_widener_specs, _pal):
        v = arr[np.isfinite(arr)]
        if v.size <= 5:
            continue
        med, p99 = float(np.nanmedian(v)), float(np.nanpercentile(v, 99))
        sns.kdeplot(v, ax=axes[0], color=c, fill=True, alpha=0.12, lw=1.7,
                    clip=(None if is_signed else 0.0, None), bw_adjust=0.9)
        axes[0].axvline(med, color=c, ls=':', lw=1.1, alpha=0.8)
        _rows.append((label, c, med, p99, float(v.min())))
    axes[0].set_xscale('symlog', linthresh=0.1)
    axes[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[0].set_title('Observation-noise wideners (raw, model-scaled → sigma_obs)')
    axes[0].set_xlabel('model-facing value  (symlog; signed where applicable)')
    axes[0].set_ylabel('density')
    axes[0].legend(handles=[Line2D([0], [0], color=c, lw=2.2,
                                   label=f'{lab}\n  med={m:.2g}, p99={p:.2g}, min={mn:.2g}')
                            for lab, c, m, p, mn in _rows],
                   fontsize=6.5, framealpha=0.25, loc='upper right')

    mult = _w['multiplier']
    mult = mult[np.isfinite(mult)]
    sns.histplot(mult, ax=axes[1], bins=80, stat='density', color='#56b4e9',
                 alpha=0.35, edgecolor='none')
    sns.kdeplot(mult, ax=axes[1], color='#56b4e9', lw=2.0, clip=(0.0, None))
    _mult_med, _mult_p99 = float(np.nanmedian(mult)), float(np.nanpercentile(mult, 99))
    axes[1].axvline(1.0, color='#bbbbbb', ls='--', lw=1.2,
                    label='multiplier = 1  (sigma_obs == sigma_obs_base)')
    axes[1].axvline(_mult_med, color='#ffb000', lw=1.6, label=f'median = {_mult_med:.2f}')
    axes[1].axvline(_mult_p99, color='#cc79a7', ls=':', lw=1.6, label=f'p99 = {_mult_p99:.2f}')
    axes[1].set_xscale('symlog', linthresh=1.0)
    axes[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[1].set_title('Realised sigma_obs multiplier  (1 + range + cv + ½·vol) / √n')
    axes[1].set_xlabel('sigma_obs / sigma_obs_base  (symlog)')
    axes[1].set_ylabel('density')
    axes[1].legend(fontsize=7, framealpha=0.25)
    plt.show()

    print('Observation-noise wideners (raw, un-winsorised, model-facing):')
    for lab, _c, m, p, mn in _rows:
        print(f'  - {lab:<42s} median={m:>9.3g}  p99={p:>10.3g}  min={mn:>8.3g}  '
              f'(p99/median tail ratio={p / m if m else float("nan"):.1f})')
    print(f'  sigma_obs multiplier: median={_mult_med:.2f}, p99={_mult_p99:.2f}; '
          f'{(mult > 1.0).mean() * 100:.0f}% of names widen sigma_obs above base, '
          f'{(mult < 1.0).mean() * 100:.0f}% (high-coverage) tighten it below base.')

    # 2.4d Feature collinearity heatmap (Spearman, robust to heavy feat_* tails).
    _corr_cols = [c for c in eda_features if kalman_df[c].notna().sum() > 5]
    _corr = kalman_df[_corr_cols].astype('float64').corr(method='spearman')
    fig, ax = plt.subplots(figsize=(9, 7.5), layout='constrained')
    sns.heatmap(_corr, ax=ax, cmap='vlag', center=0.0, vmin=-1, vmax=1,
                square=True, linewidths=0.4, linecolor='#1e1e1e',
                cbar_kws={'shrink': 0.7, 'label': 'Spearman ρ'},
                annot=True, fmt='.2f', annot_kws={'size': 6})
    ax.set_title('feat_* correlation — drift vs noise-widener blocks', pad=10)
    plt.show()

    # 2.4e Implied upside vs drift/noise signals, grouped by sector.
    _g = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    _g['upside_pct'] = ((_g['observed_pt'] / _g['last_price'] - 1.0) * 100.0).clip(-100, 500)
    _g['sector'] = _g.get('sector', pd.Series('Unknown', index=_g.index)).fillna('Unknown')
    if 'feat_pt_noise_sigma' in _g.columns:
        _g['noise_cv'] = (_g['feat_pt_noise_sigma'].astype('float64')
                          / _g['last_price'].clip(lower=1e-9)).clip(0, 1)
    _g['vol_6m'] = _g.get('feat_vol_6m', pd.Series(np.nan, index=_g.index)).astype('float64')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _top_sectors = _g['sector'].value_counts().head(9).index.tolist()
    _gs = _g[_g['sector'].isin(_top_sectors)]
    _pal = dict(zip(_top_sectors, sns.color_palette('Set2', len(_top_sectors))))
    if 'noise_cv' in _gs.columns:
        for sec, sub in _gs.groupby('sector'):
            axes[0].scatter(sub['noise_cv'], sub['upside_pct'], s=8, alpha=0.4,
                            color=_pal[sec], label=sec)
        axes[0].set_xlabel('consensus noise CV  (feat_pt_noise_sigma / last_price)')
        axes[0].set_ylabel('implied upside (%)')
        axes[0].set_title('Upside vs consensus dispersion')
        axes[0].set_xlim(0, 0.5)
    for sec, sub in _gs.groupby('sector'):
        axes[1].scatter(sub['vol_6m'], sub['upside_pct'], s=8, alpha=0.4,
                        color=_pal[sec], label=sec)
    axes[1].set_xlabel('6m volatility  (feat_vol_6m)')
    axes[1].set_title('Upside vs 6m volatility')
    axes[1].set_xlim(0, float(np.nanquantile(_gs['vol_6m'], 0.97))
                     if _gs['vol_6m'].notna().any() else 1)
    axes[1].legend(fontsize=7, framealpha=0.25, title='sector', title_fontsize=8,
                   loc='upper right')
    plt.tight_layout()
    plt.show()

    # 2.4f Empirical per-group implied-upside forest (EDA preview of §5 group effects).
    _fe = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    _fe['upside_pct'] = ((_fe['observed_pt'] / _fe['last_price'] - 1.0) * 100.0).clip(-100, 200)
    _group_preview = [c for c in ('region', 'sector', 'industry', 'size_class',
                                  'style_class', 'unit', 'exchange') if c in _fe.columns]
    _min_per_level = 5
    _boot_draws = 4000
    for _coord in _group_preview:
        lab = _fe[_coord].fillna('Unknown').astype(str)
        _counts = lab.value_counts()
        levels = sorted(lv for lv in _counts.index if _counts[lv] >= _min_per_level)
        if len(levels) < 2:
            continue
        # Bootstrap each level's per-name upside to a common draw count and stack as a
        # single posterior chain -> (chain=1, draw, <coord>) with no NaN padding.
        _boot = np.empty((1, _boot_draws, len(levels)), dtype='float64')
        for i, lv in enumerate(levels):
            vals = _fe.loc[lab == lv, 'upside_pct'].to_numpy()
            _boot[0, :, i] = vals[rng.integers(0, vals.size, _boot_draws)]
        _level_labels = [f'{lv}  (n={_counts[lv]})' for lv in levels]
        _ds = xr.Dataset(
            {'implied_upside_pct': (('chain', 'draw', _coord), _boot)},
            coords={_coord: _level_labels},
        )
        _figsize = (9.0, max(2.2, 0.45 * len(levels) + 1.3))
        pc = azp.plot_forest(
            _ds, var_names=['implied_upside_pct'], combined=True,
            labels=[_coord], backend='matplotlib',
            figure_kwargs={'figsize': _figsize},
        )
        pc.add_title(f'Empirical implied upside (%) by {_coord} — EDA group-effect preview')
        _ax = pc.viz['plot'].sel(column='forest').item()
        _ax.axvline(0, color='#bbbbbb', ls='--', lw=1.0, zorder=0)
        _ax.set_xlabel('implied upside vs last_price  (observed_pt / last_price − 1, %)')
        pc.show()


# =============================================================================
# 3. State-space feature mapping (Kalman semantics)
# =============================================================================
def map_state_space_features(kalman_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Map ``mv_pymc_kalman_pt`` ``feat_*`` columns onto Kalman state-space roles.

    Returns the drift-feature list (state-transition mean / ``beta`` slopes) and a tidy
    role-mapping frame.

    Notes
    -----
    LEAKAGE GUARDRAIL: ``feat_implied_upside = (observed_pt - last_price)/last_price`` is
    a deterministic function of the RESPONSE; ``log1p(feat_implied_upside)`` IS the
    log-uplift the model targets, so it must never enter the drift-PREDICTOR matrix.
    """
    drift_features = [c for c in ('feat_pt_drift', 'feat_price_drift',
                                  'feat_pt_high_drift', 'feat_pt_low_drift',
                                  'feat_pt_median_drift', 'feat_coverage_drift',
                                  'feat_total_return_ytd')
                      if c in kalman_df.columns]
    assert 'feat_implied_upside' not in drift_features, (
        'feat_implied_upside must not be a drift predictor (target leakage).'
    )
    range_col = 'feat_pt_range_norm' if 'feat_pt_range_norm' in kalman_df.columns else None
    sigma_col = 'feat_pt_noise_sigma' if 'feat_pt_noise_sigma' in kalman_df.columns else None
    vol_cols = [c for c in ('feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y')
                if c in kalman_df.columns]

    mapping_rows: list[tuple[str, str]] = [
        (c, 'drift / state-transition mean (beta)') for c in drift_features
    ]
    if range_col:
        mapping_rows.append((range_col, 'observation-noise widener (range)'))
    if sigma_col:
        mapping_rows.append((sigma_col, 'observation-noise widener (consensus sigma)'))
    mapping_rows += [(c, 'observation-noise widener (volatility)') for c in vol_cols]
    mapping = pd.DataFrame(mapping_rows, columns=['mv_column', 'state_space_role'])

    print(f'Drift features : {drift_features}')
    print(f'Noise drivers  : range={range_col}, sigma={sigma_col}, vol={vol_cols}')
    display(mapping)
    return drift_features, mapping


# =============================================================================
# 4. PyMC-aligned data containers
# =============================================================================
@dataclass
class ModelData:
    """Numeric arrays / coords feeding the cross-sectional state-space model (§5)."""

    model_df: pd.DataFrame
    isin_labels: np.ndarray
    drift_features: list[str]
    group_effects: list[str]
    categorical_coords: list[str]
    coord_uniques: dict[str, np.ndarray]
    coord_idx: dict[str, np.ndarray]
    log_last: np.ndarray
    log_obs: np.ndarray
    log_uplift_obs: np.ndarray
    x_drift: np.ndarray
    range_norm_xs: np.ndarray
    noise_cv_xs: np.ndarray
    vol_xs: np.ndarray
    sqrt_n_xs: np.ndarray


def build_model_data(kalman_df: pd.DataFrame, roles: FeatureRoles,
                     drift_features: list[str]) -> ModelData:
    """Filter to log-space-usable rows and build the PyMC data containers.

    Keeps rows with strictly-positive ``observed_pt`` / ``last_price`` and >=1 analyst,
    builds categorical coords (classification coords only — never the fiscal-calendar
    date coords), the standardised drift matrix, the log-uplift target, and the
    non-negative observation-noise drivers (shared :func:`build_noise_wideners`).
    """
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

    # Categorical group effects use ONLY the hierarchical classification coords.
    categorical_coords = [c for c in roles.classification_coords
                          if c in model_df.columns and c not in ('isin', 'ticker')]
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col in categorical_coords:
        labels = model_df[col].fillna('Unknown').astype(str).values
        uniques, idx = np.unique(labels, return_inverse=True)
        coord_uniques[col] = uniques
        coord_idx[col] = idx.astype('int64')
    print(f'Categorical coords ({len(categorical_coords)}): {categorical_coords}')

    # Log-space transition anchor + observation (strictly-positive prices/targets).
    log_last = np.log(model_df['last_price'].astype('float64').to_numpy())
    log_obs = np.log(model_df['observed_pt'].astype('float64').to_numpy())

    # SSOT log-uplift TARGET: log1p(feat_implied_upside) == log_obs - log_last, computed
    # once robustly in SQL. Per-row fallback keeps the script runnable before the MV is
    # refreshed with the column.
    if 'feat_implied_upside' in model_df.columns:
        _iu = model_df['feat_implied_upside'].astype('float64').to_numpy()
        log_uplift_obs = np.where(
            np.isfinite(_iu), np.log1p(np.clip(_iu, -0.999, None)), log_obs - log_last
        )
        _src = 'feat_implied_upside (SSOT)'
    else:
        log_uplift_obs = log_obs - log_last
        _src = 'log_obs - log_last (fallback; feat_implied_upside absent)'
    print(f'log_last: {log_last.shape}, log_obs: {log_obs.shape}, '
          f'log_uplift_obs: {log_uplift_obs.shape} [{_src}]')

    # 4.1 Standardised drift-feature matrix (state-transition mean inputs).
    x_drift_raw = model_df[drift_features].astype(float)
    x_drift_std = (x_drift_raw - x_drift_raw.mean()) / x_drift_raw.std(ddof=0).replace(0, 1.0)
    x_drift = x_drift_std.fillna(0.0).to_numpy()
    print(f'Drift matrix X_drift: {x_drift.shape}  ({drift_features})')

    # 4.2 Non-negative observation-noise drivers (shared helper, model contract).
    _nw = build_noise_wideners(model_df, fillna=True)
    range_norm_xs = _nw['range']
    noise_cv_xs = _nw['cv']
    vol_xs = _nw['vol']
    sqrt_n_xs = _nw['sqrt_n']
    print(f'noise drivers (mean) — range:{range_norm_xs.mean():.3f}  '
          f'cv:{noise_cv_xs.mean():.3f}  vol:{vol_xs.mean():.3f}')

    group_effects = [c for c in _CANDIDATE_GROUPS if c in coord_idx]

    return ModelData(
        model_df=model_df, isin_labels=isin_labels, drift_features=drift_features,
        group_effects=group_effects, categorical_coords=categorical_coords,
        coord_uniques=coord_uniques, coord_idx=coord_idx,
        log_last=log_last, log_obs=log_obs, log_uplift_obs=log_uplift_obs,
        x_drift=x_drift, range_norm_xs=range_norm_xs, noise_cv_xs=noise_cv_xs,
        vol_xs=vol_xs, sqrt_n_xs=sqrt_n_xs,
    )


# =============================================================================
# 5. Cross-sectional state-space model (log-space Kalman update)
# =============================================================================
def build_kalman_pt_model(data: ModelData, *, robust: bool = True) -> "pm.Model":
    """Cross-sectional log-space state-space price-target model.

    One observation per ISIN: ``log(observed_pt)`` is a noisy measurement of a latent
    log fair-value state anchored at ``log(last_price)`` and shifted by a hierarchical,
    drift-feature-driven log-uplift. The posterior ``expected_pt`` is the Kalman-smoothed
    price target — a shrinkage between the drift-implied prior and the noisy consensus.

    Parameters
    ----------
    data
        Containers from :func:`build_model_data`.
    robust
        ``True`` -> Student-t measurement likelihood (default; absorbs analyst
        outliers). ``False`` -> Normal-likelihood twin.
    """
    coords = {'isin': data.isin_labels, 'drift_feature': data.drift_features}
    for col in data.group_effects:
        coords[col] = data.coord_uniques[col]

    with pm.Model(coords=coords) as model:
        log_last_d = pm.Data('log_last_price', data.log_last, dims='isin')
        Xd = pm.Data('drift_features', data.x_drift, dims=('isin', 'drift_feature'))
        rng_d = pm.Data('feat_pt_range_norm', data.range_norm_xs, dims='isin')
        cv_d = pm.Data('feat_pt_noise_cv', data.noise_cv_xs, dims='isin')
        vol_d = pm.Data('feat_vol_mean', data.vol_xs, dims='isin')
        sqn = pm.Data('sqrt_n_analysts', data.sqrt_n_xs, dims='isin')
        log_uplift_obs_d = pm.Data('log_uplift_observed', data.log_uplift_obs, dims='isin')
        idx_data = {col: pm.Data(f'{col}_idx', data.coord_idx[col], dims='isin')
                    for col in data.group_effects}

        # --- State-transition mean: hierarchical drift regression on log-uplift.
        # Data-informed anchor: centre the global log-uplift on the cross-sectional
        # MEDIAN observed uplift (an aggregate over ISINs -> no leakage).
        mu_anchor = float(np.median(data.log_uplift_obs))
        mu_global = pm.Normal('mu_global', mu_anchor, 0.25)
        beta = pm.Normal('beta', 0.0, 0.25, dims='drift_feature')
        eta = mu_global + pt.dot(Xd, beta)
        for col in data.group_effects:
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

        # --- Measurement likelihood in log space. We observe the log-UPLIFT directly
        #     (SSOT: log1p(feat_implied_upside)), so the likelihood mean is `log_uplift`.
        if robust:
            nu = pm.Gamma('nu', alpha=2.0, beta=0.1)
            pm.StudentT('log_uplift_obs', nu=nu, mu=log_uplift, sigma=sigma_obs,
                        observed=log_uplift_obs_d, dims='isin')
        else:
            pm.Normal('log_uplift_obs', mu=log_uplift, sigma=sigma_obs,
                      observed=log_uplift_obs_d, dims='isin')

        # --- Screening outputs on the price scale.
        pm.Deterministic('expected_pt', pt.exp(log_state), dims='isin')
        pm.Deterministic('expected_upside', pt.exp(log_uplift) - 1.0, dims='isin')
    return model


# =============================================================================
# 5b. Fused MvGRW panel model (Model A + Model B)
# =============================================================================
# D-dimensional joint response series broadcast across the T fiscal anchors.
# Mirrors PANEL_RESPONSE_COLS in _price_target_mc.py, re-homed onto the kalman MV.
KALMAN_PANEL_RESPONSE_COLS: tuple[str, ...] = (
    'feat_implied_upside',
    'observed_pt',
    'price_target_median',
    'price_target_high',
    'price_target_low',
    'feat_pt_drift',
)


def prepare_kalman_panel_inputs(
        kalman_df: pd.DataFrame,
        roles: Optional[FeatureRoles] = None,
        drift_features: Optional[list[str]] = None,
        *,
        classification_coords: Optional[Sequence[str]] = None,
        fiscal_anchor_cols: Sequence[str] = FISCAL_CALENDAR_COLS_ALL,
        response_cols: Sequence[str] = KALMAN_PANEL_RESPONSE_COLS,
) -> KalmanPanelInputs:
    """Assemble the 3-D MvGRW panel tensors for :func:`build_fused_kalman_pt_model`.

    The Kalman analogue of
    ``probabilistic_ml_model.pymc_models._price_target_mc.prepare_price_target_panel_inputs``.
    Pure-NumPy/pandas (no PyMC import) so it is unit-testable in isolation. Keeps
    rows with strictly-positive ``observed_pt`` / ``last_price`` and reuses
    :func:`build_noise_wideners` so the per-ISIN expected volatility / dispersion
    CV / precision drivers match the §4.2 cross-sectional model by construction.

    Parameters
    ----------
    kalman_df : pandas.DataFrame
        Frame backed by ``pml.mv_pymc_kalman_pt``.
    roles : FeatureRoles, optional
        Resolved column groups. When supplied, ``roles.classification_coords``
        seeds the categorical group-effect coords. May be ``None`` so the helper
        is usable directly from the inline notebook (which never builds a
        ``FeatureRoles``); the coords then fall back to ``classification_coords``
        or :data:`CLASSIFICATION_COORDS_ALL` intersected with the frame columns.
    drift_features : list[str], optional
        Standardised state-transition-mean inputs (from
        :func:`map_state_space_features`). Required — derived by the caller.
    classification_coords : Sequence[str], optional
        Explicit categorical group-effect coords. Overrides ``roles`` when given.
    fiscal_anchor_cols : Sequence[str]
        Ordered DATE columns defining the ``T`` random-walk anchors.
    response_cols : Sequence[str]
        ``D`` response series columns (broadcast across the ``T`` anchors).

    Returns
    -------
    KalmanPanelInputs

    Raises
    ------
    ValueError
        If ``drift_features`` is not supplied or no rows survive filtering.
    """
    if drift_features is None:
        raise ValueError(
            "prepare_kalman_panel_inputs requires drift_features "
            "(call map_state_space_features(kalman_df) first)."
        )
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

    isin_labels = model_df['isin'].astype(str).to_numpy()
    n_isin = len(model_df)
    if n_isin == 0:
        raise ValueError('No modelling rows after filtering observed_pt / last_price.')

    # --- Fiscal-anchor time matrix (n_isin, T) --------------------------------
    anchors = [c for c in fiscal_anchor_cols if c in model_df.columns]
    if not anchors:
        raise KeyError(f'None of the fiscal-anchor columns {list(fiscal_anchor_cols)} present.')
    t0 = pd.to_datetime(model_df[anchors[0]], errors='coerce')
    t_days = np.stack(
        [(pd.to_datetime(model_df[c], errors='coerce') - t0).dt.days.to_numpy(dtype='float64')
         for c in anchors],
        axis=1,
    )
    t_mean = np.nanmean(t_days) if np.isfinite(t_days).any() else 0.0
    t_std = np.nanstd(t_days)
    t_std = t_std if t_std and np.isfinite(t_std) else 1.0
    t_scaled = np.nan_to_num((t_days - t_mean) / t_std, nan=0.0)

    # --- Response tensor (n_isin, T, D) standardised --------------------------
    resp = [c for c in response_cols if c in model_df.columns]
    if not resp:
        raise KeyError(f'None of the response columns {list(response_cols)} present.')
    T = t_scaled.shape[1]
    D = len(resp)
    Y = np.stack(
        [np.tile(model_df[c].astype('float64').to_numpy()[:, None], (1, T)) for c in resp],
        axis=-1,
    )
    y_mean = Y.reshape(-1, D).mean(axis=0)
    y_std = Y.reshape(-1, D).std(axis=0)
    y_std = np.where(y_std > 1e-6, y_std, 1.0)
    Y_std = np.nan_to_num((Y - y_mean) / y_std, nan=0.0)

    # --- Standardised drift design matrix (state-transition mean inputs) ------
    x_raw = model_df[drift_features].astype(float)
    x_std = ((x_raw - x_raw.mean()) / x_raw.std(ddof=0).replace(0, 1.0)).fillna(0.0).to_numpy()

    # --- Model-A / σ drivers (shared SSOT helper) -----------------------------
    # expected_vol == feat_vol_* term-structure mean (the volatility the
    # risk_adj_return is conditioned on); cv == consensus noise CV widening σ.
    _nw = build_noise_wideners(model_df, fillna=True)
    expected_vol = _nw['vol']
    dispersion_cv = _nw['cv']
    sqrt_n = _nw['sqrt_n']
    n_analysts = model_df['n_analysts'].astype('float64').clip(lower=1).to_numpy()

    # --- Categorical group-effect coords (classification coords only) ---------
    # Resolve the coord source: explicit arg > roles object > module default.
    if classification_coords is not None:
        _coord_src: Sequence[str] = classification_coords
    elif roles is not None:
        _coord_src = roles.classification_coords
    else:
        _coord_src = CLASSIFICATION_COORDS_ALL
    categorical_coords = [c for c in _coord_src
                          if c in model_df.columns and c not in ('isin', 'ticker')]
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col in categorical_coords:
        labels = model_df[col].fillna('Unknown').astype(str).to_numpy()
        uniques, idx = np.unique(labels, return_inverse=True)
        coord_uniques[col] = uniques
        coord_idx[col] = idx.astype('int64')

    print(f'Fused panel — isins:{n_isin}  T:{T}  D:{D} ({resp})')
    print(f'  drift_features:{len(drift_features)}  expected_vol(mean):{np.nanmean(expected_vol):.3f}'
          f'  cv(mean):{np.nanmean(dispersion_cv):.3f}')

    return KalmanPanelInputs(
        frame=model_df, isins=isin_labels, Y=Y_std, t_scaled=t_scaled,
        X_drift=x_std, n_analysts=n_analysts, sqrt_n_analysts=sqrt_n,
        expected_vol=expected_vol, dispersion_cv=dispersion_cv,
        drift_names=list(drift_features), response_names=list(resp),
        coord_uniques=coord_uniques, coord_idx=coord_idx,
    )


# --- Fused-model posterior variable groupings (single source of truth) --------
# Scalars (global hyper-parameters); group-effect scales ``sigma_<coord>`` are
# appended at runtime from the panel coords actually present.
FUSED_SCALAR_VARS: tuple[str, ...] = (
    'mu_global', 'mu_logit', 'sigma_logit', 'sigma_base', 'nu',
)
# Vector hyper-parameters (have a non-sample dim): drift slopes + GRW innovation scales.
FUSED_VECTOR_VARS: tuple[str, ...] = (
    'beta', 'sigma_alpha_innov', 'sigma_beta_innov',
)


def build_panel_model(panel: KalmanPanelInputs, *, robust: bool = True) -> "pm.Model":
    """Build the fused MvGRW + volatility-conditioned model and render its graph."""
    model = build_fused_kalman_pt_model(panel, robust=robust)
    try:
        pm.model_to_graphviz(model)
    except Exception:  # pragma: no cover - graphviz optional
        pass
    return model


def present_group_effects(idata) -> list[str]:
    """Return the coord names that received a hierarchical ``<coord>_effect`` in the fit.

    Derived from the posterior rather than a hard-coded list so it tracks whichever
    of :data:`_FUSED_KALMAN_GROUP_EFFECTS` survived the panel-coord intersection.
    """
    post = getattr(idata, 'posterior', idata)
    return [v[:-len('_effect')] for v in post.data_vars if str(v).endswith('_effect')]


def _panel_response_stats(panel: KalmanPanelInputs) -> dict[str, tuple[float, float]]:
    """Per-response-series ``(mean, std)`` reproducing ``prepare_kalman_panel_inputs``.

    The fused model standardises each ``y_series`` column before fitting, so the
    per-ISIN baseline ``mu_isin``/``risk_adj_return`` lives on the dimensionless
    standardised scale. These stats invert that standardisation back onto a chosen
    response series (the primary ``feat_implied_upside`` target) for human-facing
    screening readouts. Mirrors the exact ``(Y - mean) / std`` used at fit time
    (population std; tiling across ``T`` leaves the moments unchanged), with a
    NaN-aware fallback so a partially-missing column still yields a finite mapping.
    """
    T = panel.Y.shape[1]
    stats: dict[str, tuple[float, float]] = {}
    for col in panel.response_names:
        v = panel.frame[col].astype('float64').to_numpy()
        tiled = np.tile(v[:, None], (1, T)).reshape(-1)
        mean = float(np.mean(tiled))
        std = float(np.std(tiled))
        if not np.isfinite(mean):
            mean = float(np.nanmean(tiled)) if np.isfinite(np.nanmean(tiled)) else 0.0
        if not np.isfinite(std) or std <= 1e-6:
            std = float(np.nanstd(tiled))
        if not np.isfinite(std) or std <= 1e-6:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def panel_posterior_upside(
        idata, panel: KalmanPanelInputs,
        *, source: str = 'posterior',
) -> tuple[xr.DataArray, xr.DataArray]:
    """De-standardise ``risk_adj_return`` into ``(expected_upside, expected_pt)`` draws.

    The fused MvGRW baseline ``mu_isin = risk_adj_return`` is a per-ISIN shift on the
    standardised response scale shared across the ``D`` series. Mapping it through the
    primary ``feat_implied_upside`` series' standardisation recovers an interpretable
    implied-upside (decimal) per posterior draw, and ``expected_pt = last_price *
    (1 + expected_upside)`` lifts it to price units.

    Parameters
    ----------
    idata
        Fitted inference object (or one carrying a ``prior`` group when
        ``source='prior'``).
    panel
        The :class:`KalmanPanelInputs` the model was fit on.
    source
        ``'posterior'`` (default) or ``'prior'`` — which group to read
        ``risk_adj_return`` from.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        ``(expected_upside, expected_pt)`` over ``(chain, draw, isin)``.
    """
    group = getattr(idata, source)
    rar = group['risk_adj_return']
    stats = _panel_response_stats(panel)
    key = 'feat_implied_upside' if 'feat_implied_upside' in stats else panel.response_names[0]
    mean, std = stats[key]
    eu = (mean + rar * std).rename('expected_upside')
    last = xr.DataArray(
        panel.frame['last_price'].astype('float64').to_numpy(),
        dims='isin', coords={'isin': rar.coords['isin']},
    )
    ept = (last * (1.0 + eu)).rename('expected_pt')
    return eu, ept


@dataclass
class ScreenContext:
    """Screening artifacts derived from the fused-panel posterior.

    Attributes
    ----------
    eu, ept : xarray.DataArray
        Posterior ``expected_upside`` (decimal) / ``expected_pt`` (price) draws over
        ``(chain, draw, isin)``, de-standardised via :func:`panel_posterior_upside`.
    results : pandas.DataFrame
        Per-ISIN screening table (sorted by expected upside), consumed by §13/§14.
    mc_summary : pandas.DataFrame
        Structural-TS Monte-Carlo per-ISIN risk-adjusted-return summary
        (:func:`summarize_mc_returns`).
    """

    eu: xr.DataArray
    ept: xr.DataArray
    results: pd.DataFrame
    mc_summary: pd.DataFrame


# =============================================================================
# 6. Prior predictive checks
# =============================================================================
def run_prior_predictive(model: "pm.Model", panel: KalmanPanelInputs):
    """Sample the fused-model prior and sanity-check the implied-upside / risk scale.

    The fused MvGRW baseline ``risk_adj_return`` is de-standardised onto the primary
    ``feat_implied_upside`` series (:func:`panel_posterior_upside`) so the prior over
    implied upside can be eyeballed against the empirical consensus upside. The
    logit-normal ``achieve_prob`` and heteroscedastic ``sigma_isin`` priors are shown
    alongside so the §5b refinements are visible before any data is seen.
    """
    var_names = ['expected_return', 'risk_adj_return', 'achieve_prob', 'sigma_isin']
    with model:
        prior_idata = pm.sample_prior_predictive(
            draws=1000, var_names=var_names,
            random_seed=RANDOM_SEED, return_inferencedata=True,
        )

    eu_prior, _ = panel_posterior_upside(prior_idata, panel, source='prior')
    prior_up = eu_prior.values.reshape(-1)
    emp_up = (panel.frame['observed_pt'] / panel.frame['last_price'] - 1.0).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(np.clip(prior_up, -1, 2), bins=80, density=True, alpha=0.6,
                 color='#56b4e9', label='prior expected_upside')
    axes[0].hist(np.clip(emp_up[np.isfinite(emp_up)], -1, 2), bins=80, density=True,
                 histtype='step', linewidth=1.5, color='#ffb000',
                 label='empirical observed_pt/last_price − 1')
    axes[0].axvline(0, color='#bbbbbb', ls='--', lw=1.0)
    axes[0].set_title('Prior implied upside vs empirical')
    axes[0].set_xlabel('implied upside (decimal)')
    axes[0].legend(fontsize=8, framealpha=0.25)

    # Model-A achievement probability prior (logit-normal).
    ap = prior_idata.prior['achieve_prob'].values.reshape(-1)
    axes[1].hist(ap, bins=60, density=True, color='#cc79a7', alpha=0.7)
    axes[1].set_title('Prior achieve_prob (logit-normal)')
    axes[1].set_xlabel('P(achieve)')
    axes[1].set_xlim(0, 1)

    # Heteroscedastic measurement scale sigma_isin = sigma_base * (1 + cv) / sqrt(n).
    si = prior_idata.prior['sigma_isin'].values.reshape(-1)
    si = si[np.isfinite(si)]
    axes[2].hist(np.clip(si, 0, np.nanpercentile(si, 99)), bins=60, density=True,
                 color='#2ca02c', alpha=0.7)
    axes[2].set_title('Prior sigma_isin  (heteroscedastic scale)')
    axes[2].set_xlabel('sigma_isin')
    plt.tight_layout()
    plt.show()

    print(f'Prior expected_upside: median={np.nanmedian(prior_up):.3f}, '
          f'p01/p99=({np.nanpercentile(prior_up, 1):.2f}, {np.nanpercentile(prior_up, 99):.2f}); '
          f'empirical median={np.nanmedian(emp_up):.3f}.')
    return prior_idata


# =============================================================================
# 7. Posterior inference (NUTS)
# =============================================================================
def sample_posterior(model: "pm.Model", prior_idata):
    """Sample the posterior, trying nutpie -> numpyro -> pymc in priority order.

    Merges the prior groups into the posterior idata for one-object downstream access.
    """
    sample_kwargs = dict(
        draws=4500, tune=1500, chains=4, cores=2,
        target_accept=0.95, random_seed=RANDOM_SEED,
        progressbar=True, return_inferencedata=True,
        idata_kwargs={"log_likelihood": False},
    )

    candidate_samplers = []
    if _ilu.find_spec("nutpie") is not None:
        candidate_samplers.append("nutpie")
    if _ilu.find_spec("numpyro") is not None:
        candidate_samplers.append("numpyro")
    candidate_samplers.append("pymc")  # always available — pure-Python NUTS
    print(f"Available NUTS samplers (in priority order): {candidate_samplers}")

    sampling_errors = []
    idata = None
    for sampler in candidate_samplers:
        try:
            with model:
                idata = pm.sample(nuts_sampler=sampler, **sample_kwargs)
            print(f"Sampled successfully with nuts_sampler={sampler!r}.")
            break
        except Exception as e:  # pragma: no cover - environment-dependent fallback
            sampling_errors.append((sampler, repr(e)))
            print(f"nuts_sampler={sampler!r} failed: {e!r}")

    if idata is None:
        raise RuntimeError(
            "All NUTS samplers failed:\n"
            + "\n".join(f"  - {s}: {err}" for s, err in sampling_errors)
        )

    return extend_datatree(idata, prior_idata)


# =============================================================================
# 8. Posterior predictive checks
# =============================================================================
def run_posterior_predictive(model: "pm.Model", idata, panel: KalmanPanelInputs) -> None:
    """Sample the fused-panel posterior predictive and draw calibration diagnostics.

    The fused likelihood ``target_pct_obs`` is the standardised ``(isin, time,
    y_series)`` response tensor, so the checks pool the replicated draws against the
    observed standardised responses: an ECDF overlay, a per-``y_series`` 94% coverage
    table, and (best-effort) the arviz PIT calibration plot.
    """
    with model:
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True,
            random_seed=RANDOM_SEED, progressbar=True,
        )

    # (a) arviz distributional overlay — replicated draws vs the observed ECDF.
    try:
        pc_ppc = azp.plot_ppc_dist(
            idata, group="posterior_predictive", var_names=["target_pct_obs"],
            kind="ecdf", num_samples=400, backend="matplotlib",
        )
        pc_ppc.show()
    except Exception as e:  # pragma: no cover - arviz multidim PPC is best-effort
        print(f"arviz plot_ppc_dist skipped: {e!r}")

    pp = idata.posterior_predictive['target_pct_obs']
    obs = idata.observed_data['target_pct_obs']

    # (b) Robust pooled ECDF overlay (observed standardised response vs predictive draws).
    pp_stack = pp.stack(sample=('chain', 'draw'))
    n_samp = pp_stack.sizes['sample']
    pick = np.linspace(0, n_samp - 1, min(60, n_samp)).astype(int)
    obs_flat = np.asarray(obs.values).reshape(-1)
    obs_flat = obs_flat[np.isfinite(obs_flat)]
    obs_sorted = np.sort(obs_flat)
    ecdf_y = np.linspace(0, 1, len(obs_sorted))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for s in pick:
        rep = np.asarray(pp_stack.isel(sample=s).values).reshape(-1)
        rep = np.sort(rep[np.isfinite(rep)])
        ax.plot(rep, np.linspace(0, 1, len(rep)), color='#56b4e9', alpha=0.12, lw=0.8)
    ax.plot(obs_sorted, ecdf_y, color='#ffb000', lw=2.2, label='observed')
    ax.plot([], [], color='#56b4e9', lw=1.2, label='posterior-predictive draws')
    ax.set_xlim(np.nanpercentile(obs_flat, 0.5), np.nanpercentile(obs_flat, 99.5))
    ax.set_xlabel('standardised response  (target_pct_obs)')
    ax.set_ylabel('ECDF')
    ax.set_title('Posterior-predictive ECDF overlay — fused MvGRW panel')
    ax.legend(fontsize=8, framealpha=0.25)
    plt.tight_layout()
    plt.show()

    # (c) Per-y_series 94% predictive-interval coverage.
    lo = pp.quantile(0.03, dim=('chain', 'draw'))
    hi = pp.quantile(0.97, dim=('chain', 'draw'))
    inside = ((obs >= lo) & (obs <= hi))
    cover = inside.mean(('isin', 'time'))
    print('Per-y_series 94% posterior-predictive coverage (target ≈ 0.94):')
    for name in panel.response_names:
        try:
            c = float(cover.sel(y_series=name).values)
            print(f'  - {name:<24s}: {c:.2%}')
        except Exception:
            continue

    # (d) arviz PIT calibration (best-effort; should track the diagonal).
    try:
        pc_pit = azp.plot_ppc_pit(idata, var_names=["target_pct_obs"], backend="matplotlib")
        pc_pit.show()
    except Exception as e:  # pragma: no cover - diagnostic is best-effort
        print(f"PPC PIT calibration plot skipped: {e!r}")


# =============================================================================
# 9. MCMC diagnostics
# =============================================================================
def run_diagnostics(idata, panel: KalmanPanelInputs) -> None:
    """R-hat / ESS summary, divergence count, trace / rank-dist and forest plots.

    Targets the fused-model hyper-parameters: the global scalars
    (:data:`FUSED_SCALAR_VARS`), the per-coord hierarchical scales ``sigma_<coord>``,
    the drift slopes ``beta`` and the GRW innovation scales
    ``sigma_alpha_innov`` / ``sigma_beta_innov``.
    """
    group_effects = present_group_effects(idata)
    posterior = idata.posterior
    requested = [*FUSED_SCALAR_VARS, *FUSED_VECTOR_VARS]
    for grp in group_effects:
        requested.extend([f'sigma_{grp}', f'{grp}_effect'])

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
    display(summary.sort_values('r_hat', ascending=False).head(50))

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
    posterior = (posterior_tree.dataset if hasattr(posterior_tree, "dataset")
                 else posterior_tree.to_dataset())
    keep_vars = _non_empty_vars(posterior)
    rhat_ds = azs.rhat(posterior[keep_vars])
    ess_ds = azs.ess(posterior[keep_vars], method='bulk')

    max_rhat = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
    min_ess = float(min(float(ess_ds[v].min()) for v in ess_ds.data_vars))
    grp_keys = [f'sigma_{g}' for g in group_effects if f'sigma_{g}' in rhat_ds.data_vars]
    grp_report = {v: (float(rhat_ds[v].max()), float(ess_ds[v].min())) for v in grp_keys}

    print(f'Divergences: {n_div}')
    print(f'Max R-hat:   {max_rhat:.4f}')
    print(f'Min ESS:     {min_ess:.1f}')
    if grp_report:
        print('Group-effect scale diagnostics (max R-hat, min ESS):')
        for v, (r, e) in grp_report.items():
            print(f'  - {v:>20s}: r_hat={r:.3f}, ess_bulk={e:.1f}')

    # 9.3 Trace + marginal densities. plot_trace can crash when a single call mixes
    # variables whose non-sample dims differ, so we split scalar vs vector vars and
    # keep a per-variable fallback.
    post_trace = idata.posterior
    requested = [*FUSED_SCALAR_VARS, *FUSED_VECTOR_VARS,
                 *(f'sigma_{g}' for g in group_effects)]
    trace_vars = [v for v in requested if v in post_trace.data_vars]

    def _extra_dims(_v):
        return [d for d in post_trace[_v].dims if d not in ('chain', 'draw')]

    scalar_vars = [v for v in trace_vars if not _extra_dims(v)]
    vector_vars = [v for v in trace_vars if _extra_dims(v)]

    def _show_trace(_vars):
        azp.plot_trace(idata, var_names=_vars, backend='matplotlib').show()

    if scalar_vars:
        try:
            _show_trace(scalar_vars)
        except ValueError as exc:
            print(f'Combined scalar trace failed ({exc}); plotting per variable.')
            for sv in scalar_vars:
                _show_trace([sv])
    for vv in vector_vars:
        _show_trace([vv])
    if not scalar_vars and not vector_vars:
        print('No trace-eligible variables available.')

    # 9.3b Fractional-rank Delta-ECDF plots (same scalar/vector split).
    def _show_rank_dist(_vars):
        azp.plot_rank_dist(idata, var_names=_vars, backend='matplotlib').show()

    if scalar_vars:
        try:
            _show_rank_dist(scalar_vars)
        except ValueError as exc:
            print(f'Combined scalar rank-dist failed ({exc}); plotting per variable.')
            for sv in scalar_vars:
                _show_rank_dist([sv])
    for vv in vector_vars:
        _show_rank_dist([vv])

    # 9.4 Forest of hierarchical group-effect scales, drift slopes and GRW innovations.
    forest_vars = [f'sigma_{g}' for g in group_effects
                   if f'sigma_{g}' in idata.posterior.data_vars]
    forest_vars += [v for v in ('beta', 'sigma_alpha_innov', 'sigma_beta_innov')
                    if v in idata.posterior.data_vars]
    if forest_vars:
        pc = azp.plot_forest(idata, var_names=forest_vars, combined=True)
        pc.add_title('Group-effect scales (sigma_<coord>), drift slopes (beta) '
                     'and GRW innovation scales')
        pc.show()
    else:
        print('No group-effect / beta variables in posterior - skipped.')


# =============================================================================
# 10. Expected price targets — posterior summary
# =============================================================================
def summarize_panel_screen(idata, panel: KalmanPanelInputs,
                           *, horizon: int = 4, rho: float = 0.85) -> ScreenContext:
    """Build the per-ISIN screening table from the fused-panel posterior.

    The screen has two complementary readouts:

    * **De-standardised upside** — the fused MvGRW baseline ``risk_adj_return`` mapped
      back onto the primary ``feat_implied_upside`` series (:func:`panel_posterior_upside`)
      gives ``expected_upside`` / ``expected_pt`` and their 94% HDI bands.
    * **Structural-TS Monte-Carlo** — the per-ISIN ``risk_adj_return`` / ``sigma_isin``
      / ``nu`` draws feed :func:`simulate_lagged_risk_adjusted_returns` /
      :func:`summarize_mc_returns` for the canonical risk-adjusted forward-return
      distribution (``er_mean``, percentiles, ``prob_pos``).

    Returns a :class:`ScreenContext` carrying the upside draws and the sorted
    ``results`` frame consumed by §13/§14.
    """
    post = idata.posterior
    frame = panel.frame

    # De-standardised expected upside / price-target draws (chain, draw, isin).
    eu, ept = panel_posterior_upside(idata, panel)
    exp_up = eu.mean(('chain', 'draw')).values
    exp_pt = ept.mean(('chain', 'draw')).values
    _ept_s = ept.stack(s=('chain', 'draw'))
    pt_lo = _ept_s.quantile(0.03, dim='s').values
    pt_hi = _ept_s.quantile(0.97, dim='s').values
    prob_pos = (eu > 0).mean(('chain', 'draw')).values
    risk_adj = post['risk_adj_return'].mean(('chain', 'draw')).values

    # Structural-TS Monte-Carlo over the per-ISIN risk-adjusted-return latent.
    mu_draws = (post['risk_adj_return'].stack(sample=('chain', 'draw'))
                .transpose('isin', 'sample').values)
    sigma_draws = (post['sigma_isin'].stack(sample=('chain', 'draw'))
                   .transpose('isin', 'sample').values)
    nu_draws = post['nu'].stack(sample=('chain', 'draw')).values
    mc = simulate_lagged_risk_adjusted_returns(
        mu_draws, sigma_draws, nu_draws, horizon=horizon, rho=rho,
        random_seed=RANDOM_SEED,
    )
    mc_summary = summarize_mc_returns(mc, np.asarray(panel.isins))

    results = pd.DataFrame({
        'isin': np.asarray(panel.isins),
        'ticker': frame.get('ticker'),
        'name': frame.get('name'),
        'region': frame.get('region'),
        'country': frame.get('country'),
        'unit': frame.get('unit'),
        'exchange': frame.get('exchange'),
        'sector': frame.get('sector'),
        'industry': frame.get('industry'),
        'size_class': frame.get('size_class'),
        'style_class': frame.get('style_class'),
        'last_price': frame['last_price'].to_numpy(),
        'observed_pt': frame['observed_pt'].to_numpy(),
        'expected_pt': exp_pt,
        'expected_pt_hdi_lo': pt_lo,
        'expected_pt_hdi_hi': pt_hi,
        'expected_upside_pct': exp_up * 100,
        'risk_adj_return': risk_adj,
        'prob_pos': prob_pos,
        'implied_upside_pct': (
            frame['feat_implied_upside'].to_numpy() * 100
            if 'feat_implied_upside' in frame.columns
            else (frame['observed_pt'] / frame['last_price'] - 1.0).to_numpy() * 100
        ),
        'total_return_ytd_pct': (
            frame['feat_total_return_ytd'].to_numpy() * 100
            if 'feat_total_return_ytd' in frame.columns else np.nan
        ),
        'n_analysts': frame['n_analysts'].to_numpy(),
    })
    # Merge the MC risk-adjusted-return summary (er_mean / percentiles / prob_pos_mc).
    results = results.merge(
        mc_summary.rename(columns={'prob_pos': 'mc_prob_pos'}), on='isin', how='left',
    )
    results = results.sort_values('expected_upside_pct', ascending=False).reset_index(drop=True)
    print(f'Fused-panel screen for {len(results)} ISINs '
          f'(MC horizon={horizon}, rho={rho}).')
    display(results.head(50))

    model_df = frame

    # Shrinkage view: fused-panel expected_pt vs raw consensus observed_pt.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(results['observed_pt'], results['expected_pt'], s=8, alpha=0.4)
    hi = float(np.nanquantile(results['observed_pt'], 0.99))
    ax.plot([0, hi], [0, hi], '--', color='#888888', linewidth=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel('consensus observed_pt')
    ax.set_ylabel('fused-panel expected_pt')
    ax.set_title('Fused-panel expected target vs raw consensus')
    plt.tight_layout()
    plt.show()

    # Per-industry expected_upside posterior — arviz_plots forest with HDIs.
    eu_pct = eu * 100.0
    _industry_per_isin = model_df['industry'].fillna('Unknown').astype(str).to_numpy()
    _industry_da = xr.DataArray(
        _industry_per_isin, dims='isin', coords={'isin': eu.coords['isin']},
    )
    expected_upside_by_industry = eu_pct.groupby(_industry_da.rename('industry')).mean('isin')
    _ds_forest = xr.Dataset({'expected_upside_pct': expected_upside_by_industry})
    pc = azp.plot_forest(_ds_forest, var_names=['expected_upside_pct'], combined=True)
    pc.add_title('Per-industry expected upside (%) — posterior mean and 94% HDI')
    pc.show()

    # §5b model internals (Model A risk discount + Model B GRW components).
    plot_fused_model_effects(idata, panel)

    _plot_comparative_returns(eu, results, model_df)
    return ScreenContext(eu=eu, ept=ept, results=results, mc_summary=mc_summary)


def plot_fused_model_effects(idata, panel: KalmanPanelInputs) -> None:
    """Visualise the fused MvGRW internals — Model-A risk discount + Model-B GRW.

    Four panels make the §5b structure legible:

    (a) ``expected_return`` → ``risk_adj_return`` coloured by the per-ISIN expected
        volatility — the ``exp(-risk_penalty * expected_vol)`` discount that is the
        Kalman-specific refinement.
    (b) the logit-normal ``achieve_prob`` against the risk-adjusted return.
    (c) the heteroscedastic ``sigma_isin`` against analyst count, coloured by the
        consensus dispersion CV (``sigma_isin = sigma_base * (1 + cv) / sqrt(n)``).
    (d) the Model-B Gaussian-random-walk slope ``beta_t`` per ``y_series`` over time.
    """
    post = idata.posterior
    er = post['expected_return'].mean(('chain', 'draw')).values
    rar = post['risk_adj_return'].mean(('chain', 'draw')).values
    ap = post['achieve_prob'].mean(('chain', 'draw')).values
    si = post['sigma_isin'].mean(('chain', 'draw')).values
    exp_vol = np.asarray(panel.expected_vol, dtype='float64')
    n_an = np.asarray(panel.n_analysts, dtype='float64')
    cv = np.asarray(panel.dispersion_cv, dtype='float64')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) Volatility risk discount.
    sc = axes[0, 0].scatter(er, rar, c=exp_vol, cmap='viridis', s=14, alpha=0.75)
    _lim = [float(np.nanmin([er.min(), rar.min()])), float(np.nanmax([er.max(), rar.max()]))]
    axes[0, 0].plot(_lim, _lim, '--', color='#bbbbbb', lw=1.1)
    axes[0, 0].set_xlabel('expected_return (latent)')
    axes[0, 0].set_ylabel('risk_adj_return = expected_return · e^(−λ·vol)')
    axes[0, 0].set_title('Model A — volatility risk discount')
    fig.colorbar(sc, ax=axes[0, 0], shrink=0.8, label='expected volatility')

    # (b) Achievement probability vs risk-adjusted return.
    axes[0, 1].scatter(rar, ap, s=14, alpha=0.6, color='#cc79a7')
    axes[0, 1].axhline(0.5, color='#bbbbbb', ls='--', lw=1.0)
    axes[0, 1].set_xlabel('risk_adj_return')
    axes[0, 1].set_ylabel('achieve_prob (logit-normal)')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('Model A — achievement probability')

    # (c) Heteroscedastic measurement scale.
    sc2 = axes[1, 0].scatter(n_an, si, c=np.clip(cv, 0, np.nanpercentile(cv, 99)),
                             cmap='magma', s=14, alpha=0.75)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('n_analysts (log)')
    axes[1, 0].set_ylabel('sigma_isin')
    axes[1, 0].set_title('Model A — heteroscedastic scale  σ·(1+cv)/√n')
    fig.colorbar(sc2, ax=axes[1, 0], shrink=0.8, label='dispersion CV')

    # (d) Model-B GRW trend slope per y_series over time.
    if 'beta_t' in post.data_vars:
        beta_t = post['beta_t'].mean(('chain', 'draw'))  # (time, y_series)
        times = np.asarray(beta_t['time'].values)
        for name in panel.response_names:
            try:
                axes[1, 1].plot(times, beta_t.sel(y_series=name).values, marker='o',
                                ms=3, lw=1.4, label=str(name))
            except Exception:
                continue
        axes[1, 1].axhline(0, color='#bbbbbb', ls='--', lw=1.0)
        axes[1, 1].set_xlabel('time index (fiscal anchor)')
        axes[1, 1].set_ylabel('beta_t (GRW slope)')
        axes[1, 1].set_title('Model B — MvGRW slope per y_series')
        axes[1, 1].legend(fontsize=6.5, framealpha=0.25, ncol=2)
    else:
        axes[1, 1].set_visible(False)
    plt.tight_layout()
    plt.show()


def _plot_comparative_returns(eu, results: pd.DataFrame, model_df: pd.DataFrame) -> None:
    """Section 10b: feat_implied_upside vs expected_upside vs total_return_ytd views.

    ``eu`` is the de-standardised ``expected_upside`` posterior (chain, draw, isin).
    """
    _comp = results.dropna(subset=['expected_upside_pct']).copy()

    # (1) Shrinkage scatter: raw analyst-implied upside vs Kalman-smoothed expected upside.
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(_comp['implied_upside_pct'], _comp['expected_upside_pct'],
               s=10, alpha=0.45, color='#56b4e9', label='ISIN')
    _both = np.r_[_comp['implied_upside_pct'].to_numpy(), _comp['expected_upside_pct'].to_numpy()]
    _both = _both[np.isfinite(_both)]
    _lo, _hi = float(np.nanpercentile(_both, 1)), float(np.nanpercentile(_both, 99))
    ax.plot([_lo, _hi], [_lo, _hi], '--', color='#bbbbbb', lw=1.1, label='y = x (no shrinkage)')
    ax.axhline(0, color='#555555', lw=0.8)
    ax.axvline(0, color='#555555', lw=0.8)
    ax.set_xlim(_lo, _hi)
    ax.set_ylim(_lo, _hi)
    ax.set_xlabel('raw implied upside  feat_implied_upside (%)')
    ax.set_ylabel('Kalman-smoothed expected upside (%)')
    ax.set_title('Posterior shrinkage of analyst-implied upside')
    ax.legend(fontsize=8, framealpha=0.25)
    plt.tight_layout()
    plt.show()

    # (2) arviz_plots KDE of the posterior cross-sectional-average expected upside.
    eu_pct = eu * 100.0
    try:
        _dist = xr.Dataset({'expected_upside_pct': eu_pct.mean('isin')})
        pc_d = azp.plot_dist(_dist, kind='kde', var_names=['expected_upside_pct'],
                             sample_dims=['chain', 'draw'], backend='matplotlib')
        pc_d.add_title('Cross-sectional avg expected upside (%) - posterior')
        pc_d.show()
    except Exception as _e:
        print(f'plot_dist KDE skipped: {_e!r}')

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for _col, _lab, _c in [
        ('implied_upside_pct', 'raw implied upside (consensus)', '#ffb000'),
        ('expected_upside_pct', 'Kalman-smoothed expected upside', '#56b4e9'),
        ('total_return_ytd_pct', 'realised total return YTD', '#cc79a7'),
    ]:
        _v = pd.to_numeric(_comp.get(_col), errors='coerce').to_numpy()
        _v = _v[np.isfinite(_v)]
        if _v.size > 5:
            _v = _v[(_v >= np.nanpercentile(_v, 1)) & (_v <= np.nanpercentile(_v, 99))]
        if _v.size:
            sns.kdeplot(_v, ax=ax, label=_lab, color=_c, fill=True, alpha=0.18, lw=1.8)
    ax.axvline(0, color='#bbbbbb', ls='--', lw=1.0)
    ax.set_xlabel('return / upside (%)')
    ax.set_title('Expected vs implied vs realised returns - distributional comparison')
    ax.legend(fontsize=8, framealpha=0.25)
    plt.tight_layout()
    plt.show()

    # (3) Per-sector forest-style comparison.
    _sector_da = xr.DataArray(
        model_df['sector'].fillna('Unknown').astype(str).to_numpy(),
        dims='isin', coords={'isin': eu_pct.coords['isin']},
    )
    eu_by_sector = eu_pct.groupby(_sector_da.rename('sector')).mean('isin')
    _stack = eu_by_sector.stack(s=('chain', 'draw'))
    _sec = [str(s) for s in eu_by_sector['sector'].values]
    _mean = _stack.mean('s').values
    _q_lo = _stack.quantile(0.03, 's').values
    _q_hi = _stack.quantile(0.97, 's').values
    _ref = (_comp.assign(sector=_comp['sector'].fillna('Unknown').astype(str))
            .groupby('sector')[['implied_upside_pct', 'total_return_ytd_pct']].mean()
            .reindex(_sec))
    _order = np.argsort(_mean)
    _y = np.arange(len(_sec))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.42 * len(_sec))))
    ax.errorbar(_mean[_order], _y,
                xerr=[(_mean - _q_lo)[_order], (_q_hi - _mean)[_order]],
                fmt='o', color='#56b4e9', ecolor='#56b4e9', elinewidth=1.4, capsize=3,
                label='expected upside (posterior mean, 94% HDI)')
    ax.scatter(_ref['implied_upside_pct'].to_numpy()[_order], _y, marker='s',
               color='#ffb000', s=34, zorder=6, label='raw implied upside (mean)')
    ax.scatter(_ref['total_return_ytd_pct'].to_numpy()[_order], _y, marker='x',
               color='#cc79a7', s=44, zorder=6, label='realised total return YTD (mean)')
    ax.axvline(0, color='#bbbbbb', ls='--', lw=1.0)
    ax.set_yticks(_y)
    ax.set_yticklabels(np.array(_sec)[_order])
    ax.set_xlabel('return / upside (%)')
    ax.set_title('Per-sector: expected vs implied vs realised returns')
    ax.legend(fontsize=8, framealpha=0.25, loc='best')
    plt.show()


# =============================================================================
# 10c. Export — analytics.kalman_filtered_price_targets
# =============================================================================
def export_analytics(idata, panel: KalmanPanelInputs, screen: ScreenContext,
                     *, write: bool = False) -> pd.DataFrame:
    """Build the ``analytics.kalman_filtered_price_targets`` row-set from the fused posterior.

    Maps the fused MvGRW + volatility-conditioned posterior onto the analytics table's
    Kalman columns:

    * ``price_target_kalman`` / ``kalman_estimate`` — de-standardised ``expected_pt``.
    * ``implied_return_kalman`` / ``expected_upside_kalman`` — de-standardised
      ``expected_upside`` (decimal).
    * ``kalman_gain`` — the logit-normal ``achieve_prob`` (the smoother's confidence
      analogue: probability the implied target is achieved).
    * ``signal_strength`` — ``|E[risk_adj_return]| / sd(risk_adj_return)``.

    Set ``write=True`` to append the rows to the DB sink via ``export_to_analytics_db``
    (``DB_ANALYTICS_SCHEMA``, default ``analytics``).
    """
    model_df = panel.frame
    _post = idata.posterior
    _est = screen.ept             # (chain, draw, isin) — de-standardised price target
    _eu = screen.eu               # (chain, draw, isin) — de-standardised implied upside
    _rar = _post['risk_adj_return']  # (chain, draw, isin) — risk-adjusted return latent

    kalman_estimate = _est.mean(('chain', 'draw')).values
    kalman_variance = _est.var(('chain', 'draw')).values
    expected_upside_kalman = _eu.mean(('chain', 'draw')).values
    implied_return_kalman = expected_upside_kalman
    kalman_gain = _post['achieve_prob'].mean(('chain', 'draw')).values
    _rar_mean = _rar.mean(('chain', 'draw')).values
    _rar_sd = _rar.std(('chain', 'draw')).values
    signal_strength = np.where(_rar_sd > 0, np.abs(_rar_mean) / _rar_sd, np.nan)

    def _idcol(name):
        if name in model_df.columns:
            return model_df[name].to_numpy()
        return np.full(len(model_df), None, dtype=object)

    kalman_results = pd.DataFrame({
        'isin': np.asarray(panel.isins),
        'ticker': _idcol('ticker'),
        'name': _idcol('name'),
        'country': _idcol('country'),
        'exchange': _idcol('exchange'),
        'sector': _idcol('sector'),
        'industry': _idcol('industry'),
        'implied_return_kalman': implied_return_kalman,
        'expected_upside_kalman': expected_upside_kalman,
        'price_target_kalman': kalman_estimate,
        'kalman_estimate': kalman_estimate,
        'kalman_variance': kalman_variance,
        'kalman_gain': kalman_gain,
        'signal_strength': signal_strength,
        'original_price': model_df['last_price'].to_numpy(),
        'original_target': model_df['observed_pt'].to_numpy(),
    })
    print(f'Built kalman_filtered_price_targets row-set: {kalman_results.shape}')
    display(kalman_results.sort_values('expected_upside_kalman', ascending=False)
            .head(25).round(4))

    if write:
        from probabilistic_ml_model.data_utils.data_utils import export_to_analytics_db
        _n = export_to_analytics_db(kalman_results, 'kalman_filtered_price_targets',
                                    if_exists='append')
        print(f'Appended {_n} rows to analytics.kalman_filtered_price_targets.')
    else:
        print('write=False -> not persisted. Pass write=True to append to the DB sink.')
    return kalman_results


# =============================================================================
# 11. Single-ISIN time-series Kalman filter (+ 11b stochastic volatility)
# =============================================================================
def run_single_isin_filter(frame: pd.DataFrame, engine) -> Optional[dict]:
    """Fit the literal single-security GRW filter on the richest ``*_ago`` history.

    The time axis is reconstructed from the embedded ``*_ago`` price-target cohort,
    anchored on ``income_statement_report_date`` and projected forward to the next
    fiscal events via ``KalmanFilterPriceTarget.forecast()``. Uses the funnel-free
    *marginalized* GRW with a structural trend. ``frame`` is the fused panel's
    modelling frame (``panel.frame``). Returns context for the §11b SV variant
    (or ``None`` when no ISIN has >=2 ``*_ago`` observations / DB is unavailable).
    """
    model_df = frame
    try:
        keep = ('isin', 'ticker', 'last_price', 'price_target', 'market_cap',
                'income_statement_report_date', 'next_earnings', 'expected_report_date')
        hist_cols, col_sql = fetch_history_columns(engine, keep)
        cohort = model_df['isin'].astype(str).tolist()
        with engine.connect() as conn:
            # Rank the cohort by the most recent earnings (next_earnings closest to
            # today), breaking ties by the largest market cap, and pull the top 20
            # candidates.  cf. SELECT GREATEST(market_cap) WHERE next_earnings = current_date.
            # The first of these that has >=2 *_ago observations is fitted below.
            snap = pd.read_sql(
                text(
                    f'SELECT {col_sql} FROM pml.pml_df '
                    'WHERE isin = ANY(:isins) '
                    '  AND next_earnings IS NOT NULL '
                    '  AND market_cap IS NOT NULL '
                    'ORDER BY ABS(next_earnings - CURRENT_DATE) ASC, '
                    '         market_cap DESC '
                    'LIMIT 20'
                ),
                conn, params={'isins': cohort},
            )
        n_ago = sum(c.endswith('_ago') for c in hist_cols)
        print(f'Pulled pml.pml_df history frame: {snap.shape}  ({n_ago} *_ago columns).')

        long_df, eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
            snap, now_cols=('price_target',),
            fiscal_anchor_col='income_statement_report_date',
        )
        # Honour the SQL ranking: take the first candidate (most recent earnings /
        # largest market cap) that is eligible, falling back to select_target_isin.
        eligible_isins = set(eligible.index.astype(str)) if eligible is not None else set()
        ranked = snap['isin'].astype(str).tolist()
        chosen = next((i for i in ranked if i in eligible_isins), None)
        if chosen is None:
            chosen = KalmanFilterPriceTarget.select_target_isin(eligible, cohort=cohort)

        if chosen is None or date_col is None:
            print('No ISIN has >= 2 *_ago price-target observations; section skipped.')
            return None

        ts = (long_df.loc[long_df['isin'] == chosen, ['asof_date', 'price_target']]
              .dropna().sort_values('asof_date').reset_index(drop=True))
        dates = pd.DatetimeIndex(ts['asof_date'])
        observed = ts['price_target'].to_numpy()

        _row = model_df.loc[model_df['isin'] == chosen]
        ticker = str(_row['ticker'].iloc[0]) if len(_row) else str(chosen)
        last_price = float(_row['last_price'].iloc[0]) if len(_row) else None

        print(f'Fitting single-ISIN Kalman filter for {chosen} ({ticker}) '
              f'on {len(observed)} observations spanning '
              f'{dates.min():%Y-%m-%d} – {dates.max():%Y-%m-%d}.')

        kf = KalmanFilterPriceTarget()
        kf_idata, kf_model = kf.fit(
            price_targets=observed, isin=str(chosen), dates=dates,
            samples=2500, tune=1000, chains=8,
            random_seed=RANDOM_SEED, parameterization='marginalized', trend=True,
            target_accept=0.95, nuts_sampler='nutpie',
        )

        n_div = int(kf_idata.sample_stats['diverging'].sum())
        print(f'{_kf_fit_kind(kf_idata)} fit: {len(observed)} obs, '
              f'{dates.max().year - dates.min().year}y span, divergences={n_div}.')
        display(azs.summary(
            kf_idata,
            var_names=_present_vars(kf_idata, ['sigma_state', 'sigma_obs', 'beta_trend',
                                               'log_state_init', 'vol_step_size', 'nu_obs',
                                               'vol_anchor_offset']),
            round_to=4))

        plot_price_target_path(
            kf_idata, observed=observed, dates=dates,
            last_price=last_price, ticker=ticker,
        ).show()

        # Structural forecast to the next fiscal events.
        last_obs = dates.max()
        fc_specs = []
        for _col, _lbl in (('next_earnings', 'next_earnings'),
                           ('expected_report_date', 'expected_report')):
            if len(_row) and _col in _row.columns:
                _d = pd.to_datetime(_row[_col].iloc[0], errors='coerce')
                if pd.notna(_d) and _d > last_obs:
                    fc_specs.append((_d, _lbl))
        if fc_specs:
            _fdates = [d for d, _ in fc_specs]
            _labels = [lbl for _, lbl in fc_specs]
            _horizons = [int((d - last_obs).days) for d in _fdates]
            pred = kf.forecast(_horizons, fiscal_dates=_fdates, labels=_labels,
                               last_price=last_price)
            fig_fc, _ = plot_kalman_forecast(
                kf_idata, pred, observed=observed, dates=dates,
                last_price=last_price, ticker=ticker)
            fig_fc.show()
            _pg = pred.predictions
            fc_tbl = pd.DataFrame({
                'fiscal_event': _labels,
                'date': [d.strftime('%Y-%m-%d') for d in _fdates],
                'horizon_days': _horizons,
                'forecast_pt': _pg['forecast_pt'].mean(('chain', 'draw')).values,
                'forecast_pt_lo': _pg['forecast_pt'].quantile(0.03, dim=('chain', 'draw')).values,
                'forecast_pt_hi': _pg['forecast_pt'].quantile(0.97, dim=('chain', 'draw')).values,
            })
            if last_price:
                fc_tbl['implied_upside_pct'] = (fc_tbl['forecast_pt'] / last_price - 1.0) * 100
            print(f'Structural forecast to {len(fc_specs)} fiscal event(s) for {ticker}:')
            display(fc_tbl.round(3))
        else:
            print(f'No future fiscal event beyond the last observation for {chosen}; '
                  f'forecast skipped.')

        return {'kf': kf, 'chosen': chosen, 'dates': dates, 'observed': observed,
                'ticker': ticker, 'last_price': last_price, 'row': _row}
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 11 (single-ISIN time-series Kalman) skipped: {e!r}')
        return None


def run_single_isin_stochastic_vol(ctx: Optional[dict]) -> None:
    """Section 11b: re-fit the single-ISIN series with stochastic volatility.

    The scalar ``sigma_obs`` is replaced by a latent log-volatility random walk
    (``sigma_obs_t = exp(log_vol_t)``) under a robust Student-t likelihood; the realized
    volatility term-structure (``feat_vol_*``) anchors the *shape* of the log-vol prior.
    """
    try:
        if not ctx:
            print('Section 11 produced no fitted ISIN; stochastic-volatility variant skipped.')
            return
        chosen, dates, observed = ctx['chosen'], ctx['dates'], ctx['observed']
        ticker, _row = ctx['ticker'], ctx['row']
        _vts = [
            _row[c].iloc[0] if (len(_row) and c in _row.columns) else np.nan
            for c in ('feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y')
        ]
        rv_path = build_realized_vol_path(_vts, dates)
        print('SV realized-vol anchor (per-time):',
              np.round(rv_path, 2) if rv_path is not None else 'unavailable -> log(scale)')

        kf_sv = KalmanFilterPriceTarget()
        kf_sv_idata, _ = kf_sv.fit(
            price_targets=observed, isin=str(chosen), dates=dates,
            samples=2500, tune=2000, chains=8, random_seed=RANDOM_SEED,
            stochastic_volatility=True, realized_vol=rv_path,
            parameterization='non_centered', trend=True, nuts_sampler='nutpie',
        )

        n_div = int(kf_sv_idata.sample_stats['diverging'].sum())
        print(f'Stochastic-volatility fit: {len(observed)} obs, divergences={n_div}.')
        _sv_vars = [v for v in ('sigma_state', 'vol_step_size', 'nu_obs',
                                 'vol_anchor_offset', 'beta_trend')
                    if v in kf_sv_idata.posterior]
        display(azs.summary(kf_sv_idata, var_names=_sv_vars, round_to=4))

        _so = kf_sv_idata.posterior['sigma_obs']
        _mean = _so.mean(('chain', 'draw')).values
        _lo = _so.quantile(0.03, dim=('chain', 'draw')).values
        _hi = _so.quantile(0.97, dim=('chain', 'draw')).values
        fig_sv, ax_sv = plt.subplots(figsize=(11, 4))
        ax_sv.plot(dates, _mean, color='tab:orange', lw=2,
                   label='posterior mean $\\sigma_{obs}(t)$')
        ax_sv.fill_between(dates, _lo, _hi, color='tab:orange', alpha=0.25, label='94% HDI')
        ax_sv.set_title(f'Stochastic volatility - time-varying observation noise ({ticker})')
        ax_sv.set_ylabel('$\\sigma_{obs}$ (log-price scale)')
        ax_sv.set_xlabel('asof_date')
        ax_sv.legend()
        fig_sv.autofmt_xdate()
        plt.show()

        _trace_vars = [v for v in _sv_vars
                       if v in ('vol_step_size', 'nu_obs', 'vol_anchor_offset')]
        if _trace_vars:
            azp.plot_trace(kf_sv_idata, var_names=_trace_vars)
            plt.show()
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 11b (stochastic volatility) skipped: {e!r}')


# =============================================================================
# 12. Mingle-ISIN earnings-window Kalman filter (+ 12b stochastic volatility)
# =============================================================================
def run_mingled_cohort_filter(frame: pd.DataFrame, engine) -> Optional[dict]:
    """Mingle the recent-earnings cohort into one consensus series and refit the filter.

    For every ISIN whose ``next_earnings`` lands in the ±10-day window the embedded
    ``*_ago`` cohort is unpivoted and the cross-sectional **median** price target taken at
    each shared ``asof_date`` — a single earnings-cohort consensus over time. Fit with the
    marginalized GRW (+trend). ``frame`` is the fused panel's modelling frame
    (``panel.frame``). Returns context (``comparison``, ``snap``, ``mingled`` ...)
    for §12b and §14, or ``None`` when the window yields fewer than 2 observations.
    """
    model_df = frame
    try:
        keep = ('isin', 'ticker', 'last_price', 'price_target', 'next_earnings',
                'income_statement_report_date', 'expected_report_date')
        hist_cols, col_sql = fetch_history_columns(engine, keep)
        with engine.connect() as conn:
            snap = pd.read_sql(
                text(f"""
                    SELECT {col_sql}
                    FROM pml.pml_df
                    WHERE next_earnings >= '2026-01-01'
                      AND next_earnings >= current_date - INTERVAL '10 days'
                      AND next_earnings <= current_date + INTERVAL '10 days'
                """),
                conn,
            )
        n_ago = sum(c.endswith('_ago') for c in hist_cols)
        n_cohort = snap['isin'].nunique() if 'isin' in snap.columns else 0
        print(f'Recent-earnings window cohort: {snap.shape[0]} rows / {n_cohort} ISINs '
              f'({n_ago} *_ago columns).')

        long_df, _eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
            snap, now_cols=('price_target',),
            fiscal_anchor_col='income_statement_report_date',
        )
        if date_col is None or long_df.empty:
            print('No *_ago price-target history in the earnings window; section skipped.')
            return {'snap': snap, 'col_sql': col_sql, 'comparison': None, 'mingled': None}

        mingled = (
            long_df.groupby('asof_date', as_index=False)
            .agg(price_target=('price_target', 'median'), n_isin=('isin', 'nunique'))
            .sort_values('asof_date').reset_index(drop=True)
        )
        if len(mingled) < 2:
            print(f'Mingled cohort has only {len(mingled)} distinct as-of date(s); '
                  'need >= 2 for a Kalman fit. Section skipped.')
            return {'snap': snap, 'col_sql': col_sql, 'comparison': None, 'mingled': mingled}

        dates = pd.DatetimeIndex(mingled['asof_date'])
        observed = mingled['price_target'].to_numpy()
        last_price = (float(np.nanmedian(snap['last_price']))
                      if 'last_price' in snap.columns
                         and np.isfinite(np.nanmedian(snap['last_price'])) else None)
        label = f'EARNINGS-COHORT (n={n_cohort})'

        print(f'Fitting mingled-cohort Kalman filter on {len(observed)} consensus '
              f'observations spanning {dates.min():%Y-%m-%d} ... {dates.max():%Y-%m-%d}.')

        kf = KalmanFilterPriceTarget()
        kf_idata, kf_model = kf.fit(
            price_targets=observed, isin=label, dates=dates,
            samples=2500, tune=2000, chains=8,
            random_seed=RANDOM_SEED, parameterization='marginalized', trend=True,
            target_accept=0.95, nuts_sampler='nutpie',
        )

        n_div = int(kf_idata.sample_stats['diverging'].sum())
        print(f'{_kf_fit_kind(kf_idata)} fit: {len(observed)} obs, '
              f'{(dates.max() - dates.min()).days}d span, divergences={n_div}.')
        display(azs.summary(
            kf_idata,
            var_names=_present_vars(kf_idata, ['sigma_state', 'sigma_obs', 'beta_trend',
                                               'log_state_init', 'vol_step_size', 'nu_obs',
                                               'vol_anchor_offset']),
            round_to=4))

        # (a) Headline composition.
        plot_price_target_path(
            kf_idata, observed=observed, dates=dates,
            last_price=last_price, ticker=label,
        ).show()

        # (a2) Structural forecast to the cohort's next fiscal events.
        last_obs = dates.max()
        fc_specs = []
        for _col, _lbl in (('next_earnings', 'next_earnings'),
                           ('expected_report_date', 'expected_report')):
            if _col in snap.columns:
                _d = pd.to_datetime(snap[_col], errors='coerce').dropna()
                if not _d.empty:
                    _dm = _d.median()
                    if pd.notna(_dm) and _dm > last_obs:
                        fc_specs.append((_dm, _lbl))
        if fc_specs:
            _fdates = [d for d, _ in fc_specs]
            _labels = [lbl for _, lbl in fc_specs]
            _horizons = [int((d - last_obs).days) for d in _fdates]
            pred = kf.forecast(_horizons, fiscal_dates=_fdates, labels=_labels,
                               last_price=last_price)
            fig_fc, _ = plot_kalman_forecast(
                kf_idata, pred, observed=observed, dates=dates,
                last_price=last_price, ticker=label)
            fig_fc.show()
            _pg = pred.predictions
            fc_tbl = pd.DataFrame({
                'fiscal_event': _labels,
                'date': [d.strftime('%Y-%m-%d') for d in _fdates],
                'horizon_days': _horizons,
                'forecast_pt': _pg['forecast_pt'].mean(('chain', 'draw')).values,
                'forecast_pt_lo': _pg['forecast_pt'].quantile(0.03, dim=('chain', 'draw')).values,
                'forecast_pt_hi': _pg['forecast_pt'].quantile(0.97, dim=('chain', 'draw')).values,
            })
            print(f'Cohort structural forecast to {len(fc_specs)} fiscal event(s):')
            display(fc_tbl.round(3))

        # (b) ArviZ forest of the per-as-of-date expected_pt posterior HDIs.
        _state = kf_idata.posterior['state']
        _state = _state.assign_coords(time=[d.strftime('%Y-%m-%d') for d in dates])
        pc_state = azp.plot_forest(_state.to_dataset(), var_names=['state'],
                                   combined=True, backend='matplotlib')
        _ax_state = pc_state.viz['plot'].sel(column='forest').item()
        if last_price is not None:
            _ax_state.axvline(last_price, ls='--', color='#bbbbbb', lw=1.2,
                              label='cohort last_price')
            _ax_state.legend(fontsize=8, framealpha=0.25)
        _ax_state.set_xlabel('expected_pt (price)')
        pc_state.add_title(f'Expected price target (Kalman state) per as-of date - {label}')
        pc_state.show()

        # (c) Tidy comparison table.
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

        return {'snap': snap, 'col_sql': col_sql, 'comparison': comparison,
                'mingled': mingled, 'label': label, 'last_price': last_price}
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 12 (mingle-ISIN earnings-window Kalman) skipped: {e!r}')
        return None


def run_mingled_cohort_stochastic_vol(frame: pd.DataFrame, ctx: Optional[dict]) -> None:
    """Section 12b: stochastic-volatility variant of the mingled earnings cohort.

    The anchor is the cohort-median ``feat_vol_*`` term-structure (from the fused panel's
    modelling frame ``panel.frame``), mapped onto the mingled ``asof_date`` grid.
    """
    try:
        mingled = ctx.get('mingled') if ctx else None
        if mingled is None or len(mingled) < 2:
            print('Section 12 produced no mingled cohort; stochastic-volatility variant skipped.')
            return
        model_df = frame
        snap = ctx.get('snap')
        label = ctx.get('label', 'EARNINGS-COHORT')
        last_price = ctx.get('last_price')

        _dates_sv = pd.DatetimeIndex(mingled['asof_date'])
        _observed_sv = mingled['price_target'].to_numpy()

        _cohort = (snap['isin'].astype(str).unique()
                   if snap is not None and 'isin' in snap.columns else [])
        _mv = model_df[model_df['isin'].astype(str).isin(_cohort)]
        _vts = [
            float(np.nanmedian(_mv[c]))
            if (c in _mv.columns and _mv[c].notna().any()) else np.nan
            for c in ('feat_vol_1m', 'feat_vol_3m', 'feat_vol_6m', 'feat_vol_1y')
        ]
        rv_path = build_realized_vol_path(_vts, _dates_sv)
        print('Cohort SV realized-vol anchor (per-time):',
              np.round(rv_path, 2) if rv_path is not None else 'unavailable -> log(scale)')

        kf_sv = KalmanFilterPriceTarget()
        kf_sv_idata, _ = kf_sv.fit(
            price_targets=_observed_sv, isin=label, dates=_dates_sv,
            samples=2500, tune=2000, chains=8, random_seed=RANDOM_SEED,
            stochastic_volatility=True, realized_vol=rv_path,
            parameterization='non_centered', trend=True, nuts_sampler='nutpie',
        )

        n_div = int(kf_sv_idata.sample_stats['diverging'].sum())
        print(f'Mingled-cohort stochastic-volatility fit: {len(_observed_sv)} obs, '
              f'divergences={n_div}.')
        _sv_vars = [v for v in ('sigma_state', 'vol_step_size', 'nu_obs',
                                 'vol_anchor_offset', 'beta_trend')
                    if v in kf_sv_idata.posterior]
        display(azs.summary(kf_sv_idata, var_names=_sv_vars, round_to=4))

        _so = kf_sv_idata.posterior['sigma_obs']
        _mean = _so.mean(('chain', 'draw')).values
        _lo = _so.quantile(0.03, dim=('chain', 'draw')).values
        _hi = _so.quantile(0.97, dim=('chain', 'draw')).values
        fig_sv, ax_sv = plt.subplots(figsize=(11, 4))
        ax_sv.plot(_dates_sv, _mean, color='tab:purple', lw=2,
                   label='posterior mean $\\sigma_{obs}(t)$')
        ax_sv.fill_between(_dates_sv, _lo, _hi, color='tab:purple', alpha=0.25, label='94% HDI')
        ax_sv.set_title(f'Stochastic volatility - mingled cohort observation noise ({label})')
        ax_sv.set_ylabel('$\\sigma_{obs}$ (log-price scale)')
        ax_sv.set_xlabel('asof_date')
        ax_sv.legend()
        fig_sv.autofmt_xdate()
        plt.show()

        _trace_vars = [v for v in _sv_vars
                       if v in ('vol_step_size', 'nu_obs', 'vol_anchor_offset')]
        if _trace_vars:
            azp.plot_trace(kf_sv_idata, var_names=_trace_vars)
            plt.show()
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 12b (stochastic volatility) skipped: {e!r}')


# =============================================================================
# 13. Granular earnings-cohort posterior-predictive forest (+ 13.1 further views)
# =============================================================================
def run_granular_forest(idata, results: pd.DataFrame, panel: KalmanPanelInputs,
                        screen: ScreenContext, engine) -> Optional[dict]:
    """Per-ISIN forest of the fused-panel expected-price posterior for the cohort.

    Keeps the same earnings-cohort definition as §12 but stays per-ISIN granular and
    reuses the fitted fused posterior: the de-standardised ``expected_pt`` draws
    (:attr:`ScreenContext.ept`) per cohort name are plotted as a forest, with the raw
    analyst ``observed_pt`` overlaid as points. Returns context (``ppc_tree``, ``keep``,
    ``cohort_meta`` ...) for §13.1 / §14.
    """
    try:
        keep_cols_tuple = ('isin', 'ticker', 'last_price', 'price_target', 'next_earnings',
                           'income_statement_report_date', 'expected_report_date')
        _hist_cols, col_sql = fetch_history_columns(engine, keep_cols_tuple)
        with engine.connect() as conn:
            cohort_meta = pd.read_sql(
                text(f"""
                    SELECT {col_sql}
                    FROM pml.pml_df
                    WHERE next_earnings >= '2026-01-01'
                      AND next_earnings >= current_date - INTERVAL '10 days'
                      AND next_earnings <= current_date + INTERVAL '10 days'
                """),
                conn,
            )
        cohort_isins_all = cohort_meta['isin'].astype(str).unique().tolist()
        print(f'Recent-earnings cohort (next_earnings +/-10d): {len(cohort_isins_all)} ISINs.')

        ept = screen.ept
        modelled = set(np.asarray(ept.coords['isin'].values).astype(str).tolist())
        cohort_isins = [i for i in cohort_isins_all if i in modelled]
        if not cohort_isins:
            raise RuntimeError(
                'No earnings-window ISIN overlaps the fitted fused panel posterior '
                f'({len(cohort_isins_all)} cohort ISINs, {len(modelled)} modelled).'
            )
        print(f'Cohort ISINs overlapping the fitted posterior: {len(cohort_isins)}.')

        MAX_FOREST = 50
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

        # Fused-panel expected-price posterior draws + observed analyst target per name.
        pp_price = ept.sel(isin=forest_isins).rename('expected_price')
        _obs = (panel.frame.assign(isin=panel.frame['isin'].astype(str))
                .drop_duplicates('isin').set_index('isin')['observed_pt']
                .reindex(forest_isins).astype('float64'))
        obs_price = xr.DataArray(
            _obs.to_numpy(), dims='isin', coords={'isin': forest_isins},
        ).rename('expected_price')
        ppc_tree = xr.DataTree.from_dict({
            'posterior': pp_price.to_dataset(),
            'observed_data': obs_price.to_dataset(),
        })

        # Reference bands from the POSTERIOR HDIs of pooled expected_pt.
        exp_pt_pool = ept.sel(isin=forest_isins).stack(s=('chain', 'draw', 'isin'))
        _q = lambda p: float(exp_pt_pool.quantile(p).values)
        band94 = (_q(0.03), _q(0.97))
        band50 = (_q(0.25), _q(0.75))
        cohort_last_price = float(np.nanmedian(
            cohort_meta.loc[cohort_meta['isin'].isin(forest_isins), 'last_price']
        ))

        pc = azp.plot_forest(
            ppc_tree, group='posterior', combined=True,
            labels=['isin'], backend='matplotlib',
        )
        pc.map(azv.scatter_x, 'observations', data=ppc_tree.observed_data.ds,
               coords={'column': 'forest'}, color='#ffb000')
        pc.map(azv.labelled_x, 'xlabel', coords={'column': 'forest'},
               text='expected price (simulated)  -  points = observed analyst target',
               ignore_aes='y')
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

        _cols = ['isin', 'ticker', 'name', 'sector', 'last_price', 'observed_pt',
                 'expected_pt', 'expected_pt_hdi_lo', 'expected_pt_hdi_hi',
                 'expected_upside_pct']
        display(keep[[c for c in _cols if c in keep.columns]]
                .round(3).reset_index(drop=True))

        return {'ppc_tree': ppc_tree, 'keep': keep, 'forest_isins': forest_isins,
                'cohort_last_price': cohort_last_price, 'cohort_meta': cohort_meta}
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 13 (granular earnings-cohort posterior-predictive forest) skipped: {e!r}')
        return None


def run_granular_further_views(prior_idata, panel: KalmanPanelInputs,
                               screen: ScreenContext,
                               forest_ctx: Optional[dict]) -> None:
    """Section 13.1: results-keyed reference-band forest + cohort distribution KDE."""
    if not forest_ctx:
        print('Run the Section 13 forest first (ppc_tree / keep / cohort_last_price '
              'not in scope).')
        return
    ppc_tree = forest_ctx['ppc_tree']
    keep = forest_ctx['keep']
    forest_isins = forest_ctx['forest_isins']
    cohort_last_price = forest_ctx['cohort_last_price']

    # 13a. Forest with reference bands keyed off the stored results dataframe.
    band_lo = float(np.nanmedian(keep['expected_pt_hdi_lo']))
    band_hi = float(np.nanmedian(keep['expected_pt_hdi_hi']))
    band_med = float(np.nanmedian(keep['expected_pt']))

    pc2 = azp.plot_forest(ppc_tree, group='posterior', combined=True,
                          labels=['isin'], backend='matplotlib')
    pc2.map(azv.scatter_x, 'observations', data=ppc_tree.observed_data.ds,
            coords={'column': 'forest'}, color='#ffb000')
    pc2.map(azv.labelled_x, 'xlabel', coords={'column': 'forest'},
            text='expected price (simulated)  -  band = cohort-median results HDI '
                 '[expected_pt_hdi_lo, expected_pt_hdi_hi]',
            ignore_aes='y')
    pc2.coords = {'column': 'forest'}
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

    # 13b. Cohort distribution KDE: implied upside vs expected return.
    model_df = panel.frame
    VAR = 'cohort_expected_upside_pct'
    _fi = [str(s) for s in forest_isins]

    def _cohort_mean_pct(da):
        return (da.sel(isin=forest_isins).mean('isin') * 100.0).rename(VAR)

    def _consensus_upside_pct():
        m = model_df.set_index(model_df['isin'].astype(str))
        if 'feat_implied_upside' in m.columns:
            s, src = m['feat_implied_upside'].astype('float64'), 'feat_implied_upside (SSOT)'
        else:
            s = m['observed_pt'].astype('float64') / m['last_price'].astype('float64') - 1.0
            src = 'observed_pt/last_price - 1 (fallback)'
        return (s.reindex(_fi).dropna() * 100.0), src

    eu_prior, _ = panel_posterior_upside(prior_idata, panel, source='prior')
    cohort_upside = _cohort_mean_pct(screen.eu)
    prior_cohort = _cohort_mean_pct(eu_prior)
    _cons, _cons_src = _consensus_upside_pct()
    cons_da = xr.DataArray(_cons.to_numpy(), dims='isin',
                           coords={'isin': _cons.index.to_numpy()}).rename(VAR)

    _C_POST, _C_PRIOR, _C_CONS, _C_REF = '#1f77b4', '#ff7f0e', '#2ca02c', '#bbbbbb'
    series = [
        (cohort_upside, ['chain', 'draw'], dict(color=_C_POST, linewidth=2.2),
         'posterior E[upside] (cohort mean)'),
        (prior_cohort, ['chain', 'draw'], dict(color=_C_PRIOR, linewidth=2.0, linestyle='--'),
         'prior E[upside] (cohort mean)'),
    ]
    if len(_cons) >= 2:
        series.append((cons_da, ['isin'], dict(color=_C_CONS, linewidth=2.2),
                       'consensus implied upside (across names)'))

    pc3 = None
    for da, sample_dims, style, _ in series:
        pc3 = azp.plot_dist(
            da.to_dataset(), kind='kde', var_names=[VAR], sample_dims=sample_dims,
            backend='matplotlib', plot_collection=pc3, visuals={'dist': style},
            **({'figure_kwargs': {'figsize': (13, 5.5), 'layout': 'constrained'}}
               if pc3 is None else {}),
        )
    if pc3 is None:
        raise RuntimeError('No KDE series to plot (expected posterior + prior).')

    ax = pc3.get_target(VAR, {})
    fig = ax.get_figure()
    cons_mean = float(_cons.mean()) if len(_cons) else float('nan')
    ax.axvline(0.0, color=_C_REF, linestyle='--', linewidth=1.3, zorder=1)
    if np.isfinite(cons_mean):
        ax.axvline(cons_mean, color=_C_CONS, linestyle=':', linewidth=1.6, zorder=1)

    _all = np.concatenate([cohort_upside.values.ravel(), prior_cohort.values.ravel(),
                           _cons.to_numpy()])
    _lo, _hi = np.nanpercentile(_all, [1, 99])
    _pad = 0.05 * (_hi - _lo)
    ax.set_xlim(_lo - _pad, _hi + _pad)
    ax.set_xlabel('upside vs last_price (%)', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.set_title('Cohort upside (%): consensus implied vs expected prior/posterior '
                 '(earnings window +/-10d)', fontsize=12, pad=10)
    handles = [Line2D([0], [0], label=label, **style) for _, _, style, label in series]
    handles += [
        Line2D([0], [0], color=_C_CONS, lw=1.6, ls=':',
               label=f'consensus cohort mean ({cons_mean:.1f}%)'),
        Line2D([0], [0], color=_C_REF, lw=1.3, ls='--', label='0% break-even'),
    ]
    fig.legend(handles=handles, fontsize=9, loc='upper left',
               bbox_to_anchor=(0.78, 0.97), borderaxespad=0.0, framealpha=0.9)
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.0, 0.76, 1.0))
    except (AttributeError, TypeError):
        pass
    pc3.show()

    _p_pos = float((cohort_upside > 0).mean().values) * 100.0
    _p_vs_cons = (float((cohort_upside > cons_mean).mean().values) * 100.0
                  if np.isfinite(cons_mean) else float('nan'))
    print(f'Consensus implied upside [{_cons_src}]: cohort mean = {cons_mean:.2f}% '
          f'across {len(_cons)} names.')
    print(f'Expected upside (cohort mean): prior = {float(prior_cohort.mean()):.2f}%, '
          f'posterior = {float(cohort_upside.mean()):.2f}%;  '
          f'P(posterior cohort upside > 0) = {_p_pos:.1f}%;  '
          f'P(posterior cohort upside > consensus mean) = {_p_vs_cons:.1f}%.')


# =============================================================================
# 14. Comprehensive summary + actionable recommendations
# =============================================================================
def run_summary(results: pd.DataFrame, screen: ScreenContext,
                cohort_ctx: Optional[dict], mingled_ctx: Optional[dict]) -> None:
    """Consolidate the screen into a decision-oriented earnings-cohort vs baseline read."""
    cohort_meta = cohort_ctx.get('cohort_meta') if cohort_ctx else None
    comparison = mingled_ctx.get('comparison') if mingled_ctx else None

    def _fmt(x, nd=1, suf=''):
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

    universe = results.copy()

    # ---- A. Cross-sectional: earnings cohort vs the rest of the universe ----
    _cohort_ids = set(cohort_meta['isin'].astype(str)) if cohort_meta is not None else None
    if _cohort_ids:
        _in = universe['isin'].astype(str).isin(_cohort_ids)
        cohort, rest = universe[_in].copy(), universe[~_in].copy()
        groups = [('Earnings cohort (+/-10d)', cohort),
                  ('Historical baseline (not reporting)', rest),
                  ('Full universe', universe)]
    else:
        cohort = rest = None
        groups = [('Full universe', universe)]
        print('Section 13 cohort_meta not available - showing the universe only.')

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
            'median_n_analysts': (df['n_analysts'].median() if 'n_analysts' in df else np.nan),
        })
    summary_tbl = pd.DataFrame(_rows).set_index('group').round(2)
    print('Cross-sectional summary - expected price targets by group:')
    display(summary_tbl)

    if cohort is not None and len(cohort) and 'sector' in cohort.columns:
        sector_mix = (cohort.assign(sector=cohort['sector'].fillna('Unknown'))
                      .groupby('sector')
                      .agg(n=('isin', 'size'),
                           median_upside_pct=('expected_upside_pct', 'median'))
                      .sort_values('n', ascending=False).round(2))
        print('\nEarnings-cohort sector tilt:')
        display(sector_mix.head(10))

    # ---- B. Time-series: recent vs historical mingled cohort trail ----
    hist_drift = implied_now = first = last = None
    if comparison is not None and len(comparison) >= 2:
        first, last = comparison.iloc[0], comparison.iloc[-1]
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
        print(f'- {len(cohort)} names report within +/-10d. Median expected upside '
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
        print('- Earnings cohort not available (Section 13 was skipped).')
    if first is not None and last is not None:
        print(f'- Historical target trail ({first["asof_date"]} -> {last["asof_date"]}): mingled '
              f'cohort consensus {"rose" if (hist_drift or 0) >= 0 else "fell"} '
              f'{_fmt(abs(hist_drift), 1, "%")}; latest Kalman-smoothed target implies '
              f'{_fmt(implied_now, 1, "%")} upside vs cohort last price.')
    else:
        print('- Historical mingled trail not available (Section 12 was skipped).')

    # ---- D. Distributional view: cohort vs universe expected upside ----
    try:
        if _cohort_ids:
            _eu = screen.eu * 100.0
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


def run_recommendations(idata, panel: KalmanPanelInputs, results: pd.DataFrame,
                        screen: ScreenContext, cohort_ctx: Optional[dict]) -> None:
    """Section 14b: distil the fused-panel screen into OVERWEIGHT/NEUTRAL/UNDERWEIGHT signals.

    Read-only over ``idata``, the panel frame, the de-standardised ``screen.eu``
    expected-upside draws and the §10 ``results`` table.
    """
    model_df = panel.frame
    cohort_meta = cohort_ctx.get('cohort_meta') if cohort_ctx else None

    post = idata.posterior
    eu = screen.eu * 100.0
    isin_dim = eu.coords['isin']
    univ_mean = float(eu.mean(('chain', 'draw', 'isin')))
    OW_PP, UW_PP = 2.0, -2.0
    P_HI, P_LO = 0.60, 0.40

    def _verdict(mean_pp, p_pos):
        d = mean_pp - univ_mean
        if d >= OW_PP and p_pos >= P_HI:
            return 'OVERWEIGHT'
        if d <= UW_PP or p_pos <= P_LO:
            return 'UNDERWEIGHT'
        return 'NEUTRAL'

    def _na(x, nd=0, suf=''):
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return 'n/a'
            return f'{x:.{nd}f}{suf}'
        except Exception:
            return 'n/a'

    print('=' * 88)
    print('KALMAN PRICE-TARGET SCREEN - ACTIONABLE INVESTMENT RECOMMENDATIONS')
    print('=' * 88)

    # 1. Posterior reliability.
    _keys = [v for v in (*FUSED_SCALAR_VARS, 'beta') if v in post]
    try:
        _rh = azs.rhat(post[_keys])
        max_rhat = max(float(_rh[v].max()) for v in _rh.data_vars)
    except Exception:
        max_rhat = float('nan')
    try:
        n_div = int(idata.sample_stats['diverging'].sum())
    except Exception:
        n_div = -1
    print(f'\n1. POSTERIOR RELIABILITY    max R-hat={max_rhat:.4f}    '
          f'divergences={n_div if n_div >= 0 else "n/a"}')
    if np.isfinite(max_rhat) and max_rhat <= 1.01 and n_div == 0:
        print('   [OK]  High-quality posterior - signals are safe to act on.')
    elif np.isfinite(max_rhat) and max_rhat <= 1.05:
        print('   [~]   Acceptable convergence - size positions conservatively.')
    else:
        print('   [!!]  Convergence concerns - treat signals as indicative only.')

    # 2. Tail risk (Student-t df).
    if 'nu' in post:
        nu_mean = float(post['nu'].mean())
        tail = ('heavy - analyst-target outliers are material; prefer CVaR-aware sizing'
                if nu_mean <= 7 else
                'moderate - Gaussian risk approximation broadly acceptable')
        print(f'\n2. TAIL RISK    Student-t df (nu)={nu_mean:.1f}  ->  {tail}.')

    # 3. Universe posture.
    print(f'\n3. UNIVERSE POSTURE    posterior-mean upside={univ_mean:.2f}%    '
          f'P(upside>0)={float((eu > 0).mean()):.0%}    names={eu.sizes["isin"]}')
    print(f'   Rule: OVERWEIGHT if group upside > universe+{OW_PP:.0f}pp and P(>0)>={P_HI:.0%}; '
          f'UNDERWEIGHT if < universe{UW_PP:.0f}pp or P(>0)<={P_LO:.0%}.')

    # 4. Group allocation signals (hierarchical coords).
    _coords = [c for c in ('region', 'sector', 'industry', 'size_class', 'style_class', 'unit')
               if c in model_df.columns]
    for col in _coords:
        lab = model_df[col].fillna('Unknown').astype(str).to_numpy()
        da = xr.DataArray(lab, dims='isin', coords={'isin': isin_dim})
        grp = eu.groupby(da.rename(col)).mean('isin')
        stk = grp.stack(s=('chain', 'draw'))
        gmean = stk.mean('s'); glo = stk.quantile(0.03, 's'); ghi = stk.quantile(0.97, 's')
        gpos = (grp > 0).mean(('chain', 'draw'))
        counts = pd.Series(lab).value_counts()
        rows = [(str(g), float(gmean.sel({col: g})), float(glo.sel({col: g})),
                 float(ghi.sel({col: g})), float(gpos.sel({col: g})),
                 int(counts.get(str(g), 0)))
                for g in grp[col].values]
        rows.sort(key=lambda r: r[1], reverse=True)
        print(f'\n4.{col.upper()} SIGNALS  ({len(rows)} groups)')

        def _emit(r):
            gs, m, lo, hi, pp, n = r
            print(f'   {_verdict(m, pp):>11s}  {gs:<26.26s}  upside={m:6.2f}%  '
                  f'CI=[{lo:6.2f},{hi:6.2f}]  P(>0)={pp:4.0%}  n={n}')

        if len(rows) > 14:
            for r in rows[:7]:
                _emit(r)
            print(f'   ... {len(rows) - 14} mid-ranked {col} groups omitted ...')
            for r in rows[-7:]:
                _emit(r)
        else:
            for r in rows:
                _emit(r)

    # 5. Name-level action list.
    p_pos_name = (eu > 0).mean(('chain', 'draw')).to_series()
    nm = results.copy()
    nm['p_upside_pos'] = nm['isin'].map(p_pos_name).astype('float64')
    _den = nm['expected_pt'].replace(0, np.nan)
    nm['band_width_pct'] = (nm['expected_pt_hdi_hi'] - nm['expected_pt_hdi_lo']) / _den * 100.0
    _wide = float(np.nanpercentile(nm['band_width_pct'], 80)) if len(nm) else float('inf')

    def _nm_label(r):
        t = r.get('ticker')
        return t if isinstance(t, str) and t.strip() else str(r['isin'])

    longs = nm[(nm['expected_upside_pct'] > 0) & (nm['p_upside_pos'] >= 0.80)] \
        .sort_values(['p_upside_pos', 'expected_upside_pct'], ascending=False)
    shorts = nm[(nm['expected_upside_pct'] < 0) & (nm['p_upside_pos'] <= 0.20)] \
        .sort_values(['p_upside_pos', 'expected_upside_pct'])
    print('\n5. NAME-LEVEL ACTIONS')
    print(f'   --- High-conviction LONGS (upside>0, P(>0)>=80%): {len(longs)} names ---')
    for _, r in longs.head(10).iterrows():
        print(f'   BUY    {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
              f'P(>0)={r["p_upside_pos"]:4.0%}  band={_na(r["band_width_pct"], 1, "%")}  '
              f'n_analysts={_na(r.get("n_analysts"))}')
    print(f'   --- AVOID / SHORT candidates (upside<0, P(>0)<=20%): {len(shorts)} names ---')
    for _, r in shorts.head(10).iterrows():
        print(f'   AVOID  {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
              f'P(>0)={r["p_upside_pos"]:4.0%}  band={_na(r["band_width_pct"], 1, "%")}  '
              f'n_analysts={_na(r.get("n_analysts"))}')

    # 6. Position-sizing caution.
    risky = nm[(nm['band_width_pct'] >= _wide) | (nm['n_analysts'].fillna(0) <= 2)]
    print(f'\n6. SIZE-DOWN WATCH (band in top quintile >= {_wide:.1f}% or n_analysts<=2): '
          f'{len(risky)} names')
    for _, r in risky.sort_values('band_width_pct', ascending=False).head(8).iterrows():
        print(f'   CAUTION {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
              f'band={_na(r["band_width_pct"], 1, "%")}  n_analysts={_na(r.get("n_analysts"))}'
              f'  -> wide posterior: size to the downside of the credible band.')

    # 7. Earnings-cohort tilt.
    _cids = set(cohort_meta['isin'].astype(str)) if cohort_meta is not None else None
    if _cids:
        _coh = nm[nm['isin'].astype(str).isin(_cids)]
        if len(_coh):
            stance = ('lean INTO' if _coh['expected_upside_pct'].median() >= univ_mean
                      else 'lean AWAY from')
            print(f'\n7. EARNINGS-WEEK COHORT ({len(_coh)} names)  median upside='
                  f'{_coh["expected_upside_pct"].median():.2f}%  '
                  f'high-conviction longs={int((_coh["p_upside_pos"] >= 0.8).sum())}')
            print(f'   -> Pre-earnings stance: {stance} the reporting cohort vs the '
                  f'broader universe; gate entries on the SIZE-DOWN WATCH list above.')

    print('\n' + '=' * 88)
    print('Signals are model-implied screens from analyst-target dynamics, NOT investment '
          'advice; combine with fundamentals, liquidity and risk limits.')
    print('=' * 88)


# =============================================================================
# Entry point
# =============================================================================
def main(*, run_eda_section: bool = True, write_analytics: bool = False,
         robust: bool = True) -> dict:
    """Run the full Kalman price-target workflow end-to-end on the fused panel model.

    The cross-sectional spine is the §5b fused MvGRW + volatility-conditioned model
    (:func:`build_fused_kalman_pt_model`). Per-ISIN screening signals are de-standardised
    from the posterior ``risk_adj_return`` baseline and cross-checked with the canonical
    structural-TS Monte-Carlo of risk-adjusted forward returns.

    Parameters
    ----------
    run_eda_section
        When ``True`` (default), render the section-2 EDA panels.
    write_analytics
        When ``True``, append the §10c screen to ``analytics.kalman_filtered_price_targets``.
    robust
        When ``True`` (default), use the Student-t panel likelihood (absorbs analyst
        outliers); ``False`` selects the Normal-likelihood twin.

    Returns
    -------
    dict
        Key artifacts (``idata``, ``results``, ``kalman_results``, ``panel``, ``screen``)
        for programmatic reuse.
    """
    logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
    setup_plotting()

    engine = create_engine(resolve_db_url())

    # Data load + role resolution.
    kalman_df = load_kalman_df(engine)
    feature_catalogue = load_feature_catalogue(engine)
    roles = resolve_feature_roles(kalman_df, feature_catalogue)

    # §2 EDA (optional).
    if run_eda_section:
        run_eda(kalman_df, roles)

    # §3 state-space feature mapping + §4 fused-panel data containers.
    drift_features, _mapping = map_state_space_features(kalman_df)
    panel = prepare_kalman_panel_inputs(kalman_df, roles, drift_features)

    # §5b fused model -> §6 prior -> §7 posterior -> §8 PPC.
    model = build_panel_model(panel, robust=robust)
    prior_idata = run_prior_predictive(model, panel)
    idata = sample_posterior(model, prior_idata)
    run_posterior_predictive(model, idata, panel)

    # §9 diagnostics -> §10 screening table -> §10c export.
    run_diagnostics(idata, panel)
    screen = summarize_panel_screen(idata, panel)
    results = screen.results
    kalman_results = export_analytics(idata, panel, screen, write=write_analytics)

    # §11 single-ISIN filter (+11b SV).
    single_ctx = run_single_isin_filter(panel.frame, engine)
    run_single_isin_stochastic_vol(single_ctx)

    # §12 mingled cohort (+12b SV).
    mingled_ctx = run_mingled_cohort_filter(panel.frame, engine)
    run_mingled_cohort_stochastic_vol(panel.frame, mingled_ctx)

    # §13 granular forest (+13.1 further views).
    forest_ctx = run_granular_forest(idata, results, panel, screen, engine)
    run_granular_further_views(prior_idata, panel, screen, forest_ctx)

    # §14 summary + 14b recommendations.
    run_summary(results, screen, forest_ctx, mingled_ctx)
    run_recommendations(idata, panel, results, screen, forest_ctx)

    return {'idata': idata, 'prior_idata': prior_idata, 'results': results,
            'kalman_results': kalman_results, 'panel': panel, 'screen': screen}


if __name__ == '__main__':
    main()