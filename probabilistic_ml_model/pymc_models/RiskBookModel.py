"""
CVaR-aware risk book — the *decision analysis* stage of the Bayesian workflow.

Turns the fused Kalman panel posterior into an actionable long book: per-name
risk analytics (posterior P(upside>0), credible-band width, expected volatility,
CVaR, reward-to-risk ratios) and a STARR-ranked, cap-and-spill sized portfolio.

This is the package home for logic that previously lived inline in the
``pymc_kalman_filter_pt.py`` workflow script (§10b). The script keeps a thin
wrapper that resolves the sizing knobs from ``KalmanRunConfig`` and delegates
here, so workflow-level configuration stays in the script while the model logic
is importable, testable and reusable on its own.

**Unit convention.** Every return/risk column is a raw decimal (0.25 = +25%),
matching ``analytics.kalman_filtered_price_targets``; ratios
(``ret_vol_ratio``, ``starr``, ``expected_sharpe``) are dimensionless. Percent
scaling happens only at visualization / print boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from probabilistic_ml_model.pymc_models._workflow import posterior_dataset

logger = logging.getLogger(__name__)

#: Defaults mirroring the ``KalmanRunConfig`` fields of the same name, so a
#: direct call to :func:`compute_cvar_aware_book` reproduces the workflow.
DEFAULT_CVAR_ALPHA = 0.05
DEFAULT_WEIGHT_CAP = 0.10
DEFAULT_K_BOOK = 25
DEFAULT_P_LONG = 0.50
DEFAULT_MCAP_R_MAX = 0.01

#: Smallest dispersion accepted as a reward-to-risk DENOMINATOR, in decimal
#: return units. A ``> 0`` test is not enough: when every Monte-Carlo draw for a
#: name lands on the same uplift-clip bound its ``er_sd`` collapses to a
#: denormal (~4e-16) that is still strictly positive, and the ratio comes back
#: at 1e15 rather than as a missing value. The 2026-08-15 export shipped
#: ``expected_sharpe_ratio = -4.28e15`` for three names that way, which destroys
#: any AVG / ORDER BY / axis scaling a consumer applies to the column. 1e-4 is
#: 1bp of return dispersion — below that the ratio is an artefact of the clip,
#: not a measurement, so it is published as NULL.
MIN_RATIO_DENOMINATOR = 1e-4

#: Absolute floor on ``tail_risk``, in decimal return units. Last resort only —
#: it applies to a name with no usable return dispersion at all.
MIN_TAIL_RISK = 0.01

#: Default floor on ``tail_risk`` as a fraction of the name's own Monte-Carlo
#: return sd. Without it the absolute :data:`MIN_TAIL_RISK` binds for every name
#: whose simulated 5% quantile happens to be POSITIVE, and STARR becomes
#: ``100 x expected_upside`` for exactly those names — a ranking discontinuity
#: dressed as conviction. 0.25 charges a quarter of the name's own dispersion,
#: so a name with no simulated loss is still charged for the spread it has.
DEFAULT_TAIL_RISK_VOL_FLOOR_K = 0.25

__all__ = [
    "DEFAULT_TAIL_RISK_VOL_FLOOR_K",
    "MIN_TAIL_RISK",
    "DEFAULT_CVAR_ALPHA",
    "DEFAULT_K_BOOK",
    "DEFAULT_MCAP_R_MAX",
    "DEFAULT_P_LONG",
    "DEFAULT_WEIGHT_CAP",
    "MIN_RATIO_DENOMINATOR",
    "RiskBook",
    "compute_cvar_aware_book",
]


@dataclass(frozen=True, eq=False)
class RiskBook:
    """CVaR-aware sizing artifacts shared by the analytics export and the screen.

    Attributes
    ----------
    analytics : pandas.DataFrame
        Per-ISIN copy of the screen ``results`` table (including the
        ``mcap_global_r`` size-rank ratio) augmented with the risk columns
        ``p_upside_pos``, ``kalman_gain``, ``p_upside_pos_cond``, ``band_width``,
        ``exp_vol``, ``cvar05``, ``ret_vol_ratio``, ``tail_risk``, ``starr``,
        ``expected_sharpe`` and the normalised ``book_weight`` (0 for names
        outside the sized book). All return/risk columns are raw decimals
        (0.25 = +25%); ratios are dimensionless. ``p_upside_pos_cond`` is the
        PRIMARY probability column and the one rankings use: since 2026-08-20 the
        v2 workflow passes it in as P(risk-adjusted forward return > 0), computed
        from the Monte-Carlo draws. When the caller does not supply it this falls
        back to ``mc_prob_pos * kalman_gain``, which orders names but whose level
        is not a probability of anything.
    book : pandas.DataFrame
        The sized top-``k_book`` long subset (``starr``-ranked) carrying a
        ``weight`` column that sums to 1 (100% gross) after the per-name cap.
    summary : dict[str, float]
        Portfolio-level metrics (``port_up``, ``port_cvar``, ``wavg_cvar``,
        ``port_vol`` — decimal returns; ``starr_book``, ``div``, ``n_book``
        dimensionless) and the sizing parameters (``alpha``, ``cap``, ``k_book``,
        ``p_long``, ``mcap_r_max``, plus the derived ``univ_gain``, the
        conditional-scale gate ``p_long_cond = p_long * univ_gain`` and
        ``n_mcap_eligible`` — the count of names passing the market-cap gate).
    """

    analytics: pd.DataFrame
    book: pd.DataFrame
    summary: dict[str, float]


def _cap_normalize_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Normalise non-negative scores to sum 1 under an iterative per-name cap.

    Cap-and-spill: clip names above ``cap`` and redistribute the excess pro-rata
    to the uncapped names, repeating until none breach the cap (or the cap is
    infeasible for the given breadth).

    Parameters
    ----------
    w
        Non-negative score per name (e.g. the STARR ratio).
    cap
        Maximum single-name weight.

    Returns
    -------
    numpy.ndarray
        Weights summing to 1 (when feasible), each ``<= cap``.
    """
    w = np.clip(np.asarray(w, dtype='float64'), 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(len(w), 1.0 / len(w)) if len(w) else w
    w = w / s
    for _ in range(64):  # cap-and-spill until no name breaches the cap
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def compute_cvar_aware_book(
        idata: Any,
        eu: Any,
        results: pd.DataFrame,
        *,
        alpha: float = DEFAULT_CVAR_ALPHA,
        cap: float = DEFAULT_WEIGHT_CAP,
        k_book: int = DEFAULT_K_BOOK,
        p_long: float = DEFAULT_P_LONG,
        mcap_r_max: float = DEFAULT_MCAP_R_MAX,
        return_draws: Optional[np.ndarray] = None,
        tail_risk_vol_floor_k: float = DEFAULT_TAIL_RISK_VOL_FLOOR_K,
) -> RiskBook:
    """Build the CVaR-aware long book and per-name risk analytics (SSOT).

    Re-ranks the screen on risk-adjusted terms — reward per unit *expected
    volatility* and reward per unit *expected shortfall* — then sizes a long book
    on the STARR (reward-to-CVaR) ratio with a per-name cap.

    .. versionchanged:: 2026-08-20
       ``cvar05`` and ``exp_vol`` are computed from ``return_draws`` — the
       Monte-Carlo forward-return distribution — when it is supplied. They were
       previously derived from ``eu``, the posterior draws of the expected
       upside, which is *estimation* uncertainty about a point rather than
       outcome uncertainty. The consequences were not subtle: on run
       ``49e84d7e9d59`` the exported ``cvar05`` was POSITIVE for 88.4 % of names
       and correlated 0.9998 with ``expected_upside``, the 25-name book reported
       a weighted 5 % expected shortfall of +42.7 %, and ``exp_vol`` had a median
       of 0.47pp against a Monte-Carlo return sd of 19.03pp — a factor of 40.
       The estimation-uncertainty view is not lost: the screen's
       ``expected_upside_sd`` column is exactly the old ``exp_vol``, under a name
       that says what it is.

    Parameters
    ----------
    idata
        Fused-panel inference data. Retained for provenance and for the
        ``achieve_prob`` fallback; the ``kalman_gain`` term is preferentially
        read from ``results``.
    eu
        Posterior ``expected_upside`` draws over ``(chain, draw, isin)`` as
        decimals (``ScreenContext.eu``). Still the source of ``p_upside_pos`` and
        the ``expected_upside -> CVaR`` dispersion leg of ``tail_risk``, and the
        fallback for ``cvar05`` / ``exp_vol`` when ``return_draws`` is absent.
    return_draws
        Monte-Carlo forward returns as decimals, shape
        ``(n_isin, n_samples)``, row-aligned to ``results['isin']``. Supply
        ``ScreenDraws.pooled_returns``, whose pooling matches the exported
        ``er_*`` summary so a CVaR taken here and the ``er_p05`` in the same row
        describe one distribution.
    tail_risk_vol_floor_k
        Floor on ``tail_risk`` as a fraction of the name's own return sd. The
        absolute :data:`MIN_TAIL_RISK` floor binds for names whose simulated 5 %
        quantile is positive — 13.4 % of the universe and 14 of 25 book names on
        run ``49e84d7e9d59`` — and turns STARR into ``100 x expected_upside`` for
        exactly those names, which is what made the ranking bimodal (median 2.35,
        p75 25.6). ``0.0`` restores the pre-2026-08-20 behaviour.
    results
        Per-ISIN screen table; must carry ``isin``, ``expected_pt``,
        ``expected_pt_hdi_{lo,hi}`` and ``expected_upside``. ``mcap_global_r``,
        ``mc_prob_pos`` and ``er_mean`` / ``er_sd`` / ``er_p05`` are used when
        present.
    alpha
        CVaR tail probability (default 0.05 = 5% expected shortfall).
    cap
        Maximum single-name weight after cap-and-spill normalisation.
    k_book
        Number of names in the sized long book.
    p_long
        Minimum *conditional* positive-upside probability for book eligibility.
        Compared against ``p_long * univ_gain`` so the gate scales with the
        universe-average state confidence.
    mcap_r_max
        Market-cap pre-selection gate: candidates need
        ``mcap_global_r < mcap_r_max``, where
        ``feat_mcap_country_r = (100 - market_cap_country_r) / 100`` (smaller =
        larger cap). Names with a missing rank are excluded (strict, matching the
        NULL semantics of the SQL candidate filters).

    Returns
    -------
    RiskBook
        Per-name ``analytics``, the sized ``book`` and the portfolio ``summary``.

    Notes
    -----
    Pure pandas/xarray — no sampling — so it is cheap to re-run with different
    sizing parameters against one fitted posterior.
    """
    nm = results.copy()

    # Posterior P(upside>0) and credible-band width (mirror the §5 name screen).
    p_pos_name = (eu > 0).mean(('chain', 'draw')).to_series()
    nm['p_upside_pos'] = nm['isin'].map(p_pos_name).astype('float64')
    _den = nm['expected_pt'].replace(0, np.nan)
    nm['band_width'] = (nm['expected_pt_hdi_hi'] - nm['expected_pt_hdi_lo']) / _den

    # ---- the probability columns -------------------------------------------
    # ``kalman_gain`` is taken from the screen frame when the caller supplies it.
    # It used to be read straight off the posterior as ``achieve_prob``, i.e.
    # ``sigmoid(risk_adj_return)`` — a sigmoid of a standardised log-uplift,
    # which is not the probability of any defined event and centres on 0.5 by
    # construction. The workflow now computes P(risk_adj_return > 0) as a
    # reduction over draws and passes it down; the posterior read stays as the
    # fallback for an idata fitted before that change.
    if 'kalman_gain' in nm.columns and pd.to_numeric(
            nm['kalman_gain'], errors='coerce').notna().any():
        nm['kalman_gain'] = pd.to_numeric(nm['kalman_gain'], errors='coerce')
    else:
        try:
            _gain_ser = posterior_dataset(idata)['achieve_prob'].mean(
                ('chain', 'draw')).to_series()
            _gain_ser.index = _gain_ser.index.astype(str)
            nm['kalman_gain'] = nm['isin'].astype(str).map(_gain_ser).astype('float64')
            logger.warning(
                'kalman_gain absent from the screen frame; falling back to the '
                'legacy achieve_prob sigmoid, whose LEVEL is not interpretable.'
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('achieve_prob (kalman_gain) unavailable: %s — conditional '
                           'P(upside>0) falls back to the unconditional probability.', exc)
            nm['kalman_gain'] = 1.0
    # Likewise ``p_upside_pos_cond``: the workflow computes it directly as
    # P(risk-adjusted forward return > 0) from the MC draws. The product below
    # is the documented fallback — it still ORDERS names, but multiplying a
    # probability by a state-confidence term does not give the probability of
    # anything, so its level must not be read as one.
    if 'p_upside_pos_cond' not in nm.columns or not pd.to_numeric(
            nm['p_upside_pos_cond'], errors='coerce').notna().any():
        _p_base = (pd.to_numeric(nm['mc_prob_pos'], errors='coerce')
                   if 'mc_prob_pos' in nm.columns else nm['p_upside_pos'])
        nm['p_upside_pos_cond'] = (_p_base.fillna(nm['p_upside_pos'])
                                   * nm['kalman_gain'].fillna(1.0)).astype('float64')
    else:
        nm['p_upside_pos_cond'] = pd.to_numeric(
            nm['p_upside_pos_cond'], errors='coerce').astype('float64')
    _gain_vals = nm['kalman_gain'].to_numpy(dtype='float64')
    univ_gain = (float(np.nanmean(_gain_vals))
                 if np.isfinite(_gain_vals).any() else 1.0)
    if not np.isfinite(univ_gain) or univ_gain <= 0:
        univ_gain = 1.0
    p_long_cond = p_long * univ_gain

    # Per-name CVaR and expected volatility from the posterior upside draws
    # (decimal return units). exp_vol is the per-name std of the draws — a
    # genuinely forward-looking dispersion (the absolute feat_vol_* MV levels
    # were replaced by the feat_vol_drift widener, which is a signed drift,
    # not a level).
    _row_of: dict[str, int] = {}
    _eu_vals = None
    _isin_order = None
    try:
        _eu_s = eu.stack(s=('chain', 'draw')).transpose('isin', 's')
        _eu_vals = np.ascontiguousarray(_eu_s.values, dtype='float64')  # (n_isin, n_samples)
        _isin_order = _eu_s.coords['isin'].values.astype(str)
        _row_of = {is_: i for i, is_ in enumerate(_isin_order)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning('posterior upside draws unusable: %s', exc)
        _eu_vals = None

    def _tail_stats(vals: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cvar_alpha, sd)`` per row of a ``(n_isin, n_samples)`` array.

        Chunked over the name axis. The Monte-Carlo array reaches
        ``6500 x chains*draws*horizon`` — about 1.7 GB at the production budget —
        and ``np.quantile`` plus a boolean mask plus a ``np.where`` on the whole
        thing at once peaks at several times that. Row-blocking is exact, not an
        approximation: every statistic here is per-row.
        """
        n = vals.shape[0]
        cvar = np.empty(n, dtype='float64')
        sd = np.empty(n, dtype='float64')
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(i0 + chunk, n))
            block = vals[sl]
            _var = np.quantile(block, alpha, axis=1, keepdims=True)
            _mask = block <= _var
            _denom = np.maximum(_mask.sum(axis=1), 1)
            cvar[sl] = np.where(_mask, block, 0.0).sum(axis=1) / _denom
            sd[sl] = block.std(axis=1)
        return cvar, sd

    # ``cvar05`` / ``exp_vol`` come from the FORWARD-RETURN draws when they are
    # supplied. Deriving them from ``eu`` measures how precisely the mean is
    # estimated, not how badly the position can do — see the versionchanged note.
    _ret = None
    if return_draws is not None:
        _ret = np.asarray(return_draws, dtype='float64')
        if _ret.ndim != 2 or _ret.shape[0] != len(nm):
            logger.warning(
                'return_draws has shape %s but the screen carries %d names; '
                'falling back to the posterior upside draws, which makes cvar05 '
                'an estimation-uncertainty quantity rather than a loss quantile.',
                _ret.shape, len(nm),
            )
            _ret = None
    if _ret is not None:
        _cvar, _sd = _tail_stats(_ret)
        nm['cvar05'] = _cvar
        nm['exp_vol'] = _sd
    elif _eu_vals is not None and _isin_order is not None:
        logger.warning(
            'no return_draws supplied: cvar05 and exp_vol fall back to the '
            'posterior upside draws. They are then ESTIMATION uncertainty, not '
            'return risk -- cvar05 will be positive for most names.'
        )
        _cvar, _sd = _tail_stats(_eu_vals)
        nm['cvar05'] = pd.Series(_cvar, index=_isin_order).reindex(
            nm['isin'].astype(str)).to_numpy(dtype='float64')
        nm['exp_vol'] = pd.Series(_sd, index=_isin_order).reindex(
            nm['isin'].astype(str)).to_numpy(dtype='float64')
    else:  # pragma: no cover - defensive
        nm['cvar05'] = np.nan
        nm['exp_vol'] = np.nan

    # Reward-to-risk ratios (all dimensionless — decimal / decimal).
    #   * ret_vol_ratio    - reward per unit return dispersion when
    #     ``return_draws`` is supplied; per unit POSTERIOR dispersion (parameter
    #     uncertainty, not a Sharpe ratio) on the fallback path.
    #   * expected_sharpe  - er_mean / er_sd over Monte-Carlo draws of the LOG
    #     PRICE-TARGET UPLIFT (the distance from price to the smoothed target),
    #     not of a realised equity return. It reads as a t-statistic on the
    #     uplift estimate, so book values of 5-7 are normal and are NOT
    #     investment Sharpe ratios. Exported as ``expected_sharpe_ratio``.
    #   * tail_risk        - binding downside: the largest of the mean->CVaR
    #     dispersion, the Student-t MC 5% loss magnitude, and a RELATIVE floor at
    #     ``tail_risk_vol_floor_k`` of the name's own return sd, with the
    #     absolute ``MIN_TAIL_RISK`` as a last resort.
    #
    #     The relative floor was added 2026-08-20, and it is worth being precise
    #     about what it does and does not fix. The failure it was written for:
    #     with ``cvar05`` taken from the posterior of the MEAN, the mean-to-CVaR
    #     dispersion is only ~1pp, so any name whose simulated 5% quantile was
    #     positive had no binding downside term at all and fell to the 1pp
    #     absolute floor — making ``starr`` ``100 x expected_upside`` for it.
    #     That was 29.6% of the universe on run ``49e84d7e9d59`` and it selected
    #     the book: 14 of 25 names sat on the floor and 24 of 25 had
    #     ``er_p05 > 0``, while only 16 of the top-100 STARR names were top-100
    #     by upside.
    #
    #     Supplying ``return_draws`` fixes that on its own: ``expected_upside -
    #     cvar05`` then becomes roughly ``2 * er_sd``, which dominates any
    #     sensible fraction of ``er_sd``, and the absolute floor is unreachable.
    #     So on the primary path this term is inert BY CONSTRUCTION — it is
    #     defence for the ``return_draws=None`` fallback (where the old
    #     behaviour returns exactly) and for a degenerate MC distribution. Kept
    #     because both of those are reachable and neither announces itself in
    #     the output.
    #   * starr            - reward per unit expected-shortfall (STARR ratio).
    #
    # Every denominator is floored at ``MIN_RATIO_DENOMINATOR`` and every result
    # re-checked for finiteness: a clip-pinned name yields a denormal-but-
    # positive dispersion that a bare ``> 0`` guard lets through as ~1e15. See
    # the constant's docstring. ``starr`` needs no floor — ``tail_risk`` already
    # carries one — but is finiteness-checked for symmetry.
    def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        """Return ``num / den`` with a degenerate denominator mapped to NaN."""
        _num = pd.to_numeric(num, errors='coerce')
        _den = pd.to_numeric(den, errors='coerce')
        out = _num / _den.where(_den >= MIN_RATIO_DENOMINATOR)
        return out.where(np.isfinite(out))

    nm['ret_vol_ratio'] = _safe_ratio(nm['expected_upside'], nm['exp_vol'])
    if {'er_mean', 'er_sd'} <= set(nm.columns):
        nm['expected_sharpe'] = _safe_ratio(nm['er_mean'], nm['er_sd'])
    else:  # pragma: no cover - older screen frame without the MC summary
        nm['expected_sharpe'] = np.nan
    _md_disp = (nm['expected_upside'] - nm['cvar05']).fillna(0.0).clip(lower=0.0)
    _mc_loss = (-nm['er_p05'].astype('float64')
                if 'er_p05' in nm.columns else pd.Series(0.0, index=nm.index)).fillna(0.0)
    _vol_floor = (
        float(tail_risk_vol_floor_k)
        * pd.to_numeric(nm['er_sd'], errors='coerce').fillna(0.0).clip(lower=0.0)
        if 'er_sd' in nm.columns
        else pd.Series(0.0, index=nm.index)
    )
    nm['tail_risk'] = np.maximum.reduce([
        _md_disp.to_numpy(),
        _mc_loss.to_numpy(),
        np.asarray(_vol_floor, dtype='float64'),
        np.full(len(nm), MIN_TAIL_RISK),
    ])
    nm['starr'] = _safe_ratio(nm['expected_upside'], nm['tail_risk'])

    # Market-cap pre-selection: only top-of-country names (mcap_global_r <
    # mcap_r_max, i.e. raw market_cap_country_r > 98 at the 0.02 default) are
    # long-book eligible. Strict on missing ranks — NaN < x is False — matching
    # the NULL semantics of the §11–§13 SQL candidate filters.
    if 'mcap_global_r' in nm.columns:
        _mcap_r = pd.to_numeric(nm['mcap_global_r'], errors='coerce')
        _mcap_ok = (_mcap_r < mcap_r_max).fillna(False).to_numpy(dtype=bool)
    else:  # pre-0.9.9.12 results frame
        logger.warning('results frame lacks mcap_global_r — the market-cap '
                       'pre-selection gate (mcap_r_max=%.4f) is skipped.', mcap_r_max)
        _mcap_ok = np.ones(len(nm), dtype=bool)

    # Sized long book: STARR-ranked, cap-and-spill normalised to 100% gross.
    nm['book_weight'] = 0.0
    summary: dict[str, float] = {
        'alpha': alpha, 'cap': cap, 'k_book': float(k_book), 'p_long': p_long,
        'p_long_cond': p_long_cond, 'univ_gain': univ_gain,
        'mcap_r_max': mcap_r_max, 'n_mcap_eligible': float(_mcap_ok.sum()),
        'n_book': 0.0, 'port_up': float('nan'), 'port_cvar': float('nan'),
        'wavg_cvar': float('nan'), 'port_vol': float('nan'),
        'starr_book': float('nan'), 'div': float('nan'),
    }
    _book = nm[(nm['expected_upside'] > 0) & (nm['p_upside_pos_cond'] >= p_long_cond)
               & np.isfinite(nm['starr']) & _mcap_ok].copy()
    if len(_book):
        _book = _book.sort_values('starr', ascending=False).head(k_book)
        _w = _cap_normalize_weights(_book['starr'].to_numpy(), cap)
        _book = _book.assign(weight=_w)
        nm.loc[_book.index, 'book_weight'] = _w  # stamp weights back (0 elsewhere)
        _book = _book.reset_index(drop=True)

        # Portfolio analytics from the joint posterior upside draws of held names.
        port_cvar = port_up = wavg_cvar = float('nan')
        if _eu_vals is not None:
            _rows = [_row_of.get(str(i)) for i in _book['isin']]
            _keep = [j for j, idx in enumerate(_rows) if idx is not None]
            if _keep:
                _wk = _w[_keep] / _w[_keep].sum()
                _held = _eu_vals[[_rows[j] for j in _keep]]  # (k, n_samples)
                _port_draws = _wk @ _held
                _pv = np.quantile(_port_draws, alpha)
                port_cvar = float(_port_draws[_port_draws <= _pv].mean())
                port_up = float(_port_draws.mean())
                wavg_cvar = float(np.nansum(_wk * _book['cvar05'].to_numpy()[_keep]))
        port_vol = float(np.nansum(
            _book['weight'].to_numpy() * _book['exp_vol'].to_numpy()))
        summary.update(
            n_book=float(len(_book)), port_up=port_up, port_cvar=port_cvar,
            wavg_cvar=wavg_cvar, port_vol=port_vol,
            starr_book=(port_up / abs(port_cvar)
                        if np.isfinite(port_cvar) and port_cvar != 0 else float('nan')),
            div=(wavg_cvar / port_cvar
                 if np.isfinite(port_cvar) and port_cvar != 0 else float('nan')),
        )
    else:
        _book = _book.assign(weight=pd.Series(dtype='float64')).reset_index(drop=True)

    return RiskBook(analytics=nm, book=_book, summary=summary)