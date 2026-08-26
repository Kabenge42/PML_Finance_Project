"""Probabilistic decision layer: objective functions, generative risk, capital allocation.

This module turns a posterior *predictive* distribution of forward returns into
decisions. It is the second half of the pair described in
:mod:`~probabilistic_ml_model.pymc_models.KalmanForecast`, and it operates on draws
— never on summary columns — because every quantity here is a functional of the
whole distribution rather than of its first two moments.

The framework, in order
-----------------------
**Decide by minimising expected loss.** Specify an objective ``L(R, D'')`` over
actions ``R`` and predicted outcomes ``D''``, average it over the posterior
predictive, and take the minimiser: ``argmin_R E[L(R, D'')]``. That is
:func:`expected_loss` / :func:`minimize_expected_loss`. The point of routing a
decision through an explicit loss is that subjective consequences — a risk limit, a
career, a mandate — can be priced consistently alongside the objective probabilities
instead of being applied as an unrecorded override afterwards.

**Measure risk from the tail, not from the variance.** Volatility is a symmetric
ensemble average and says nothing about the shape of what it is averaging.
:func:`generative_var`, :func:`generative_expected_shortfall` and
:func:`generative_tail_risk` read the 1-in-N quantile, the mean beyond it, and the
worst simulated outcome directly off the predictive draws. Each is strictly more
conservative than the last, and the third is the only one that is not itself an
average over the region where averages behave worst.

**Size positions for a trajectory, not for an ensemble.** Investing is
non-ergodic: an investor experiences one path, not the average over all paths, and
the time average of a multiplicative process is below its ensemble average by the
volatility drag. :func:`ergodicity_report` measures that gap on the actual draws,
and :func:`kelly_fraction_from_draws` maximises the expected log growth rate, which
is the criterion that maximises terminal wealth on a single trajectory without
risking ruin. :func:`fractional_kelly` exists because the full Kelly fraction is
acutely sensitive to an edge estimated from a non-stationary process, and
overbetting is the one error the criterion does not forgive.

**Keep mean-variance as a labelled contrast.** :func:`mean_variance_frontier` is
here so the comparison can be *measured* on this universe rather than asserted. It
is not the recommendation.

What this module inherits, and what it must not duplicate
---------------------------------------------------------
The sizing primitive is
:func:`~probabilistic_ml_model.pymc_models.RiskBookModel._cap_normalize_weights` —
the package's only weight-construction routine — and the ratio floors come from that
module too. There are already three copies of the ``max(-cvar05, k*er_sd,
MIN_TAIL_RISK)`` tail-risk denominator in the tree (``RiskBookModel``,
``dashboards/geib/charts/kelly.py``, and a legacy ``.pyi``); this module deliberately
adds no fourth, and does not restate ``DEFAULT_K_BOOK`` or ``DEFAULT_MCAP_R_MAX``,
whose committed values already disagree with the ones the pipeline passes.

:func:`downside_deviation` is offered as the replacement that recommendation 03 of
the post-run analysis asks for. The shipped ``tail_risk`` collapses to its relative
volatility floor for every name the book selects — ``0.25 * er_sd``, so ``starr``
reduces to ``4 * expected_upside / er_sd``, a reward-to-variability ratio under the
name of a tail ratio, correlating 0.9936 with ``expected_sharpe_ratio``. A downside
deviation stays sensitive across the favourable region instead of going flat there.

Units
-----
Every return is a **raw decimal** (0.25 = +25%), matching the project-wide
convention since 0.9.9.7. Risk measures are returned in the same signed convention:
a loss is negative, so :func:`generative_var` normally returns a negative number and
``GVaR >= GES >= GTR``. :func:`downside_deviation` is the exception and is a
positive magnitude, because it is a dispersion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from probabilistic_ml_model.pymc_models.RiskBookModel import (
    MIN_RATIO_DENOMINATOR,
    _cap_normalize_weights,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_KELLY_MULTIPLIER",
    "DEFAULT_VAR_PROB",
    "MAX_KELLY_FRACTION",
    "LossFunction",
    "LinearPositionLoss",
    "Portfolio",
    "downside_deviation",
    "ergodicity_report",
    "expected_loss",
    "fractional_kelly",
    "generative_expected_shortfall",
    "generative_tail_risk",
    "generative_var",
    "kelly_fraction_from_draws",
    "mean_variance_frontier",
    "minimize_expected_loss",
    "optimize_portfolio",
    "terminal_wealth_curve",
]

#: Default confidence for :func:`generative_var` / :func:`generative_expected_shortfall`.
#: 0.99 is the regulatory convention; the risk book uses ``alpha = 0.05`` on the other
#: side of the same statistic, i.e. ``prob = 0.95``.
DEFAULT_VAR_PROB: float = 0.95

#: Default Kelly scaling. Half-Kelly is the practitioner's standard guard: it retains
#: roughly three quarters of the growth rate for about half the volatility, and — the
#: reason it exists — it keeps the position below the optimum when the edge has been
#: overestimated, which is the error the criterion punishes hardest.
DEFAULT_KELLY_MULTIPLIER: float = 0.25

#: Hard cap on any single-name Kelly fraction. Binds only when the draws contain no
#: losing scenario at all, where the log-optimal fraction is unbounded and the answer
#: is a statement about the simulation rather than about the opportunity.
MAX_KELLY_FRACTION: float = 1.0

_EPS = 1e-12


# ---------------------------------------------------------------------------
# A. The decision framework
# ---------------------------------------------------------------------------


class LossFunction(Protocol):
    """Cost of taking ``action`` when the outcome is ``outcomes``.

    Implementations return an array broadcastable against ``outcomes``, in whatever
    units the decision is denominated — currency, utility, or an arbitrary but
    *consistently calibrated* point scale. The absolute numbers do not matter; what
    matters is that the relative magnitudes reflect the relative consequences.
    """

    def __call__(self, action: Any, outcomes: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


@dataclass(frozen=True)
class LinearPositionLoss:
    """Loss of a linear position: ``L = -(position_value * return)``.

    The reference implementation of :class:`LossFunction`, and the one that makes
    the expected loss of holding a position equal minus its expected profit.

    Parameters
    ----------
    position_value
        Market value of the position. The action passed to :func:`expected_loss` is
        the *fraction* of that value held, so ``action=0`` is flat and ``action=1``
        is fully invested.

    Examples
    --------
    >>> import numpy as np
    >>> loss = LinearPositionLoss(position_value=100_000.0)
    >>> draws = np.array([0.10, -0.05, 0.02])
    >>> float(np.mean(loss(1.0, draws)))
    -2333.3333333333335
    """

    position_value: float = 1.0

    def __call__(self, action: Any, outcomes: np.ndarray) -> np.ndarray:
        return -(float(action) * self.position_value) * np.asarray(outcomes, dtype="float64")


def expected_loss(
        loss: LossFunction,
        outcomes: np.ndarray,
        action: Any,
) -> float:
    """Average a loss function over the posterior predictive draws.

    ``E[L(R, D'')] = mean_s L(R, D''_s)``. The draws are equally weighted because
    they are already samples *from* the predictive distribution — the probability
    weighting the textbook formula applies to enumerated outcomes is carried by the
    sampling itself.

    Parameters
    ----------
    loss
        Any :class:`LossFunction`.
    outcomes
        Predictive draws, any shape; averaged over every axis.
    action
        The action to price.

    Returns
    -------
    float
        Expected loss. Non-finite draws are excluded, and an all-non-finite input
        returns ``nan`` rather than raising — a name with no usable draws should
        drop out of a ranking, not abort it.
    """
    vals = np.asarray(loss(action, np.asarray(outcomes, dtype="float64")), dtype="float64")
    finite = np.isfinite(vals)
    if not finite.any():
        return float("nan")
    return float(vals[finite].mean())


def minimize_expected_loss(
        loss: LossFunction,
        outcomes: np.ndarray,
        actions: Sequence[Any],
) -> tuple[Any, pd.DataFrame]:
    """Choose the action minimising expected loss, and show the whole table.

    The table is returned alongside the winner on purpose. A decision that reports
    only its argmin cannot be audited: the reader cannot see whether the choice was
    decisive or whether two actions were separated by less than Monte-Carlo error.

    Parameters
    ----------
    loss
        Any :class:`LossFunction`.
    outcomes
        Predictive draws.
    actions
        Candidate actions. Must be non-empty.

    Returns
    -------
    tuple[Any, pandas.DataFrame]
        The minimising action, and a frame of ``action`` / ``expected_loss`` sorted
        ascending. Actions whose expected loss is ``nan`` are ranked last and never
        chosen.

    Raises
    ------
    ValueError
        If ``actions`` is empty, or if every action prices to ``nan``.
    """
    acts = list(actions)
    if not acts:
        raise ValueError("actions must be non-empty")
    losses = [expected_loss(loss, outcomes, a) for a in acts]
    table = pd.DataFrame({"action": acts, "expected_loss": losses})
    table = table.sort_values("expected_loss", na_position="last").reset_index(drop=True)
    if not np.isfinite(table["expected_loss"]).any():
        raise ValueError(
            "every candidate action priced to nan; the outcome draws carry no finite values"
        )
    return table.loc[0, "action"], table


# ---------------------------------------------------------------------------
# B. Generative risk measures
# ---------------------------------------------------------------------------


def _finite_1d(returns: np.ndarray) -> np.ndarray:
    """Flatten to 1-D and drop non-finite entries."""
    vals = np.asarray(returns, dtype="float64").ravel()
    return vals[np.isfinite(vals)]


def generative_var(returns: np.ndarray, *, prob: float = DEFAULT_VAR_PROB) -> float:
    """Generative Value at Risk: the ``1 - prob`` quantile of the predictive draws.

    "There is a ``prob`` probability that the loss will not exceed this." Read
    directly off the posterior predictive rather than assumed Gaussian and scaled by
    a z-score, which is what makes it *generative*: it inherits whatever skew and
    kurtosis the model actually produces.

    Parameters
    ----------
    returns
        Predictive draws as decimals, any shape.
    prob
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        The quantile, in return units — normally negative. ``nan`` if no finite draw.
    """
    if not (0.0 < prob < 1.0):
        raise ValueError(f"prob must be in (0, 1), got {prob!r}")
    vals = _finite_1d(returns)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 1.0 - prob))


def generative_expected_shortfall(
        returns: np.ndarray,
        *,
        prob: float = DEFAULT_VAR_PROB,
) -> float:
    """Generative Expected Shortfall: the mean of the draws at or below the GVaR.

    Answers what VaR cannot — *how bad* the breach is when it happens. Still an
    average, and averaging over the least ergodic region of the distribution is
    exactly the criticism :func:`generative_tail_risk` responds to.

    Returns
    -------
    float
        Mean of the tail, ``<=`` :func:`generative_var` by construction. ``nan`` if
        no finite draw.
    """
    vals = _finite_1d(returns)
    if vals.size == 0:
        return float("nan")
    var = float(np.quantile(vals, 1.0 - prob))
    tail = vals[vals <= var]
    if tail.size == 0:  # pragma: no cover - only when every draw is identical
        return var
    return float(tail.mean())


def generative_tail_risk(returns: np.ndarray) -> float:
    """Generative Tail Risk: the worst outcome the model simulated.

    Neither a quantile nor an average, which is the point. If an extreme loss
    impairs the capital base there is no later period in which to collect the
    expected shortfall, so for a position that is effectively short volatility — any
    long equity holding is — the quantity to hedge against is the worst case the
    model can produce, not the average of the cases beyond a threshold.

    It is by construction the most sensitive statistic here to the number of draws:
    it can only fall as ``n_scenarios`` rises. Quote it with the draw count.

    Returns
    -------
    float
        Minimum finite draw, or ``nan`` if there is none.
    """
    vals = _finite_1d(returns)
    if vals.size == 0:
        return float("nan")
    return float(vals.min())


def downside_deviation(returns: np.ndarray, *, threshold: float = 0.0) -> float:
    """Root-mean-square shortfall below ``threshold``; zero above it.

    ``sqrt(mean(min(r - threshold, 0)**2))`` over ALL draws, not only the ones below
    the threshold — dividing by the full count is what keeps it a property of the
    whole distribution and stops a name with two bad draws out of ten thousand from
    reporting the same downside as one with two thousand.

    Offered as the ranking denominator recommendation 03 of the post-run analysis
    asks for. The shipped ``tail_risk`` is ``max(-cvar05, k * er_sd, MIN_TAIL_RISK)``,
    and for every name the book actually selects the first leg is negative — a
    favourable modelled tail — so the relative volatility floor takes the whole
    denominator and the ratio stops being a tail ratio at all. Three candidate
    reshapings of that maximum were scored and found numerically identical over
    exactly those names, which is why a different *quantity* is needed rather than a
    different maximum. This one is strictly monotone in the favourable region: as
    the left tail improves it keeps falling instead of going flat.

    Returns
    -------
    float
        A non-negative magnitude in return units. ``0.0`` when no draw falls below
        the threshold; ``nan`` when there are no finite draws.
    """
    vals = _finite_1d(returns)
    if vals.size == 0:
        return float("nan")
    short = np.minimum(vals - float(threshold), 0.0)
    return float(np.sqrt(np.mean(short * short)))


# ---------------------------------------------------------------------------
# C. Ergodicity
# ---------------------------------------------------------------------------


def ergodicity_report(returns: np.ndarray, *, periods: Optional[int] = None) -> dict[str, float]:
    """Contrast the ensemble average with the growth rate a single path realises.

    The ensemble average — the ordinary expected return — is what every investor
    would collectively receive. The time-average growth rate is what *one* investor
    receives by compounding a sequence of them, and it is lower. The gap is the
    volatility drag, approximately half the variance, and it is the reason a
    position sized on expected value alone can have a positive expectation and a
    negative growth rate at the same time.

    Parameters
    ----------
    returns
        Predictive draws as decimals.
    periods
        Number of compounding periods used for the growth-rate estimate. Defaults to
        the number of finite draws, i.e. compounding the whole sample once.

    Returns
    -------
    dict[str, float]
        ``ensemble_average`` (arithmetic mean), ``time_average`` (exact geometric
        growth rate ``exp(mean(log(1+r))) - 1``), ``volatility_drag`` (the
        difference), ``half_variance`` (the classical approximation to it),
        ``prob_ruin`` (share of draws at or below -100%), and ``n_draws``.

        When any draw is at or below -100% the geometric mean is undefined; those
        draws are excluded from ``time_average`` and counted in ``prob_ruin``, which
        is the honest reading — a path through total loss has no growth rate.
    """
    vals = _finite_1d(returns)
    n = int(vals.size)
    if n == 0:
        return {
            "ensemble_average": float("nan"),
            "time_average": float("nan"),
            "volatility_drag": float("nan"),
            "half_variance": float("nan"),
            "prob_ruin": float("nan"),
            "n_draws": 0.0,
        }
    ruined = vals <= -1.0
    survivors = vals[~ruined]
    ensemble = float(vals.mean())
    if survivors.size:
        time_avg = float(np.expm1(np.mean(np.log1p(survivors))))
    else:  # pragma: no cover - every path ruined
        time_avg = -1.0
    _ = periods  # accepted for call-site symmetry; the estimate is per-period already
    return {
        "ensemble_average": ensemble,
        "time_average": time_avg,
        "volatility_drag": ensemble - time_avg,
        "half_variance": float(0.5 * vals.var()),
        "prob_ruin": float(ruined.mean()),
        "n_draws": float(n),
    }


def terminal_wealth_curve(
        returns: np.ndarray,
        *,
        fractions: Optional[np.ndarray] = None,
        start_capital: float = 100_000.0,
        n_bets: Optional[int] = None,
) -> pd.DataFrame:
    """Terminal wealth as a function of position size, compounded over the draws.

    Reproduces the simulation that shows why maximising expected value ruins an
    investor even when the edge is real: wealth rises with position size, peaks at
    the log-optimal fraction, and falls away to ruin beyond roughly twice it. The
    curve is the argument; the peak is :func:`kelly_fraction_from_draws`.

    Parameters
    ----------
    returns
        Predictive draws as decimals, treated as an i.i.d. sequence of bets.
    fractions
        Position sizes to evaluate. Defaults to ``0.00 .. 0.99`` in steps of 0.01.
    start_capital
        Initial capital.
    n_bets
        Truncate the sequence to this many draws. Defaults to all of them.

    Returns
    -------
    pandas.DataFrame
        ``fraction``, ``terminal_wealth``, ``log_growth`` (mean log growth per bet).
        Computed in log space, so a thousand-bet sequence does not overflow.
    """
    vals = _finite_1d(returns)
    if n_bets is not None:
        vals = vals[: int(n_bets)]
    if fractions is None:
        fractions = np.arange(0.0, 1.0, 0.01)
    fracs = np.asarray(fractions, dtype="float64")

    rows = []
    for f in fracs:
        growth = 1.0 + f * vals
        if np.any(growth <= 0.0):
            # Ruin: one bet takes the capital to zero or below, and nothing after it
            # can recover. Recorded as such rather than as a very small number.
            rows.append((float(f), 0.0, float("-inf")))
            continue
        mean_log = float(np.mean(np.log(growth)))
        rows.append(
            (float(f), float(start_capital * np.exp(mean_log * vals.size)), mean_log)
        )
    return pd.DataFrame(rows, columns=["fraction", "terminal_wealth", "log_growth"])


# ---------------------------------------------------------------------------
# D. Capital allocation
# ---------------------------------------------------------------------------


def _max_feasible_fraction(returns: np.ndarray) -> float:
    """Largest ``f`` with ``1 + f*r > 0`` for every draw.

    Beyond this the position is wiped out by at least one simulated scenario, so the
    expected log growth is ``-inf`` and the criterion is undefined rather than merely
    unattractive.
    """
    worst = float(returns.min())
    if worst >= 0.0:
        # No losing scenario at all: log growth rises without bound in f, which says
        # more about the simulation than about the opportunity.
        return float("inf")
    return -1.0 / worst


def kelly_fraction_from_draws(
        returns: np.ndarray,
        *,
        multiplier: float = 1.0,
        max_fraction: float = MAX_KELLY_FRACTION,
        tol: float = 1e-10,
        max_iter: int = 200,
) -> float:
    """Log-optimal position size, solved on the draws themselves.

    Maximises ``E[log(1 + f*r)]`` over the predictive draws. The derivative
    ``g(f) = E[r / (1 + f*r)]`` is strictly decreasing in ``f`` on the feasible
    interval, so the optimum is found by bisection on ``g`` — exact, dependency-free,
    and with no gradient step to tune.

    This is the **general** form. The familiar ``(p*b - q) / b`` is its two-outcome
    special case, and it is what
    ``dashboards/geib/charts/kelly.py::_calculate_kelly_fraction`` computes: to use
    it that card must first reduce a whole distribution to a win probability ``p``
    and an odds ratio ``b``, and manufacturing ``b`` is what forces the mirrored
    ``max(-cvar05, k*er_sd, MIN_TAIL_RISK)`` denominator and its two hand-copied
    constants. Solving on the draws needs no ``b``, so it needs no floor, so there is
    nothing to keep in sync.

    ``g(0) = E[r]``, so a non-positive edge returns ``0.0`` immediately: this
    function never recommends a short. Sizing the other side of a negative-edge bet
    is a different decision with different constraints and belongs to its own action
    space, not to a sign flip here.

    Parameters
    ----------
    returns
        Predictive draws for one name, as decimals.
    multiplier
        Scales the result. ``1.0`` is full Kelly; see :func:`fractional_kelly` and
        :data:`DEFAULT_KELLY_MULTIPLIER` for why that is rarely what you want.
    max_fraction
        Cap applied after solving. Binds when the draws contain no loss.
    tol, max_iter
        Bisection tolerance and iteration ceiling.

    Returns
    -------
    float
        A fraction in ``[0, max_fraction]``. ``nan`` if there are no finite draws.
    """
    vals = _finite_1d(returns)
    if vals.size == 0:
        return float("nan")

    edge = float(vals.mean())
    if edge <= 0.0:
        return 0.0

    hi_feasible = _max_feasible_fraction(vals)
    if not np.isfinite(hi_feasible):
        return float(min(max_fraction, max_fraction * multiplier))

    # Search strictly inside the feasible interval; g -> -inf at its right edge.
    lo, hi = 0.0, hi_feasible * (1.0 - 1e-9)

    def g(f: float) -> float:
        return float(np.mean(vals / (1.0 + f * vals)))

    if g(hi) > 0.0:
        f_star = hi
    else:
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        f_star = 0.5 * (lo + hi)

    return float(np.clip(f_star * multiplier, 0.0, max_fraction))


def fractional_kelly(
        fraction: float,
        *,
        multiplier: float = DEFAULT_KELLY_MULTIPLIER,
) -> float:
    """Scale a Kelly fraction down, and never up.

    Overbetting is the asymmetric error: past the optimum the growth rate falls, and
    it keeps falling to ruin, while underbetting only forgoes growth. The edge here
    is estimated from a non-stationary process on a prior-driven shrinkage, so the
    fraction should be treated as an upper bound.

    Raises
    ------
    ValueError
        If ``multiplier`` is outside ``(0, 1]`` — scaling a Kelly fraction *up* is
        never a defensible use of this function, and silently allowing it would make
        the guard decorative.
    """
    if not (0.0 < multiplier <= 1.0):
        raise ValueError(
            f"multiplier must be in (0, 1]; got {multiplier!r}. Scaling above full "
            "Kelly is overbetting, which this guard exists to prevent."
        )
    if not np.isfinite(fraction):
        return float("nan")
    return float(max(0.0, fraction) * multiplier)


@dataclass(frozen=True, eq=False)
class Portfolio:
    """A sized book plus the risk it carries, all measured on the joint draws.

    Attributes
    ----------
    weights
        ``isin -> weight``, summing to 1 over the held names, every weight ``<= cap``.
    analytics
        Per-name frame: ``isin``, ``expected_return``, ``kelly_fraction``,
        ``gvar``, ``ges``, ``gtr``, ``downside_dev``, ``reward_to_downside``,
        ``weight``. Names outside the book carry weight 0 and keep their statistics.
    summary
        Portfolio-level quantities computed on the **portfolio return vector**
        ``w @ draws``, not aggregated from per-name figures: ``port_expected``,
        ``port_gvar``, ``port_ges``, ``port_gtr``, ``port_vol``, ``port_kelly``,
        ``log_growth``, ``diversification_ratio``, ``effective_n``, ``n_book``,
        plus the sizing parameters.

        ``port_vol`` here is the standard deviation of that vector, so it prices the
        cross-name correlation the draws contain. That is a different quantity from
        ``RiskBook.summary['port_vol']``, which is a weighted *sum* of per-name
        volatilities and therefore assumes perfect correlation; the two agree only
        when every name moves together.
    """

    weights: pd.Series
    analytics: pd.DataFrame
    summary: dict[str, float]


def _log_growth(weights: np.ndarray, draws: np.ndarray) -> float:
    """Expected log growth of a portfolio, ``E[log(1 + w @ r)]``.

    ``-inf`` when any scenario wipes the portfolio out, which is the correct value
    and is what keeps the optimiser away from those weights.
    """
    port = weights @ draws
    growth = 1.0 + port
    if np.any(growth <= 0.0):
        return float("-inf")
    return float(np.mean(np.log(growth)))


def optimize_portfolio(
        returns: np.ndarray,
        isins: Sequence[str],
        *,
        k_book: int = 50,
        cap: float = 0.10,
        kelly_multiplier: float = DEFAULT_KELLY_MULTIPLIER,
        var_prob: float = DEFAULT_VAR_PROB,
        eligible: Optional[np.ndarray] = None,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        random_seed: int = 42,
) -> Portfolio:
    """Size a long book by maximising joint expected log growth.

    Selection ranks candidates on reward-to-downside-deviation — the ratio
    recommendation 03 asks for — takes the top ``k_book``, then optimises weights
    over the **joint** draws by exponentiated gradient ascent on
    ``E[log(1 + w @ r)]``, projecting onto the capped simplex each iteration with
    ``RiskBookModel._cap_normalize_weights``.

    Optimising on the joint draws rather than on per-name summaries is the whole
    point: correlation enters through the scenarios, so two names that fall together
    cannot both be sized as though they diversified each other. Whether they do fall
    together is a property of the draws — if the forecast layer generated them with
    independent shocks, this routine will faithfully report the free diversification
    that assumption creates. Check ``ForecastDraws.factor_share`` before quoting
    ``diversification_ratio``.

    Parameters
    ----------
    returns
        ``(n_isin, n_scenarios)`` joint draws as decimals. Use
        ``ForecastDraws.terminal`` — the cumulative horizon return — rather than
        pooled per-step marginals, which mix horizons.
    isins
        ``(n_isin,)`` labels aligned to axis 0. **Required**, not optional: aligning
        risk columns by position rather than by key is the failure this project has
        already shipped once.
    k_book
        Number of names in the book. No default is asserted here — ``DEFAULT_K_BOOK``
        in ``RiskBookModel`` and the value the pipeline actually passes disagree, so
        this module takes it from the caller.
    cap
        Maximum single-name weight.
    kelly_multiplier
        Applied to the reported ``port_kelly``; weights themselves are relative and
        already sum to 1.
    var_prob
        Confidence for the GVaR / GES columns.
    eligible
        Optional boolean mask over names. Anything already excluded upstream (market
        cap, support, out-of-support rows) belongs here rather than being re-derived.
    max_iter, learning_rate
        Exponentiated-gradient budget and step size.
    random_seed
        Unused by the optimiser, which is deterministic; accepted so the call site
        reads like the rest of the pipeline.

    Returns
    -------
    Portfolio

    Raises
    ------
    ValueError
        If ``isins`` does not match ``returns``, or ``returns`` is not 2-D.
    """
    draws = np.asarray(returns, dtype="float64")
    if draws.ndim != 2:
        raise ValueError(f"returns must be 2-D (n_isin, n_scenarios), got {draws.shape}")
    labels = np.asarray(isins)
    if labels.shape[0] != draws.shape[0]:
        raise ValueError(
            f"isins has {labels.shape[0]} entries for {draws.shape[0]} rows of returns"
        )
    if not (0.0 < cap <= 1.0):
        raise ValueError(f"cap must be in (0, 1], got {cap!r}")
    _ = random_seed

    n_isin = draws.shape[0]
    per_name = {
        "isin": labels,
        "expected_return": draws.mean(axis=1),
        "kelly_fraction": np.array(
            [kelly_fraction_from_draws(draws[i]) for i in range(n_isin)]
        ),
        "gvar": np.array([generative_var(draws[i], prob=var_prob) for i in range(n_isin)]),
        "ges": np.array(
            [generative_expected_shortfall(draws[i], prob=var_prob) for i in range(n_isin)]
        ),
        "gtr": np.array([generative_tail_risk(draws[i]) for i in range(n_isin)]),
        "downside_dev": np.array([downside_deviation(draws[i]) for i in range(n_isin)]),
    }
    analytics = pd.DataFrame(per_name)

    # Reward per unit downside. Floored by MIN_RATIO_DENOMINATOR for the same reason
    # RiskBookModel floors its ratios: a denormal denominator passes a bare `> 0`
    # guard and publishes a ratio of 1e15.
    den = analytics["downside_dev"].where(
        analytics["downside_dev"] >= MIN_RATIO_DENOMINATOR
    )
    analytics["reward_to_downside"] = (analytics["expected_return"] / den).where(
        lambda s: np.isfinite(s)
    )

    mask = np.ones(n_isin, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.shape[0] != n_isin:
        raise ValueError(f"eligible has {mask.shape[0]} entries for {n_isin} names")
    selectable = (
        mask
        & (analytics["expected_return"].to_numpy() > 0.0)
        & np.isfinite(analytics["reward_to_downside"].to_numpy())
    )

    analytics["weight"] = 0.0
    chosen = (
        analytics.loc[selectable]
        .sort_values("reward_to_downside", ascending=False)
        .head(int(k_book))
    )
    summary: dict[str, float] = {
        "k_book": float(k_book),
        "cap": float(cap),
        "var_prob": float(var_prob),
        "kelly_multiplier": float(kelly_multiplier),
        "n_eligible": float(int(selectable.sum())),
        "n_book": float(len(chosen)),
    }

    if chosen.empty:
        logger.warning(
            "no name passed selection (%d eligible of %d); returning an empty book",
            int(selectable.sum()),
            n_isin,
        )
        return Portfolio(
            weights=pd.Series(dtype="float64"),
            analytics=analytics,
            summary={
                **summary,
                "port_expected": float("nan"),
                "port_gvar": float("nan"),
                "port_ges": float("nan"),
                "port_gtr": float("nan"),
                "port_vol": float("nan"),
                "port_kelly": float("nan"),
                "log_growth": float("nan"),
                "diversification_ratio": float("nan"),
                "effective_n": 0.0,
            },
        )

    rows = chosen.index.to_numpy()
    held = draws[rows]
    k = len(rows)

    # Exponentiated gradient ascent on E[log(1 + w @ r)]. Multiplicative updates keep
    # the weights non-negative without a projection, and the cap-and-spill step is
    # what puts them back on the capped simplex.
    w = _cap_normalize_weights(np.ones(k), cap)
    best_w, best_obj = w.copy(), _log_growth(w, held)
    for _ in range(int(max_iter)):
        port = w @ held
        growth = 1.0 + port
        if np.any(growth <= 0.0):
            break
        grad = (held / growth).mean(axis=1)
        scale = np.max(np.abs(grad))
        if not np.isfinite(scale) or scale <= _EPS:
            break
        w_new = w * np.exp(learning_rate * grad / scale)
        w_new = _cap_normalize_weights(w_new, cap)
        obj = _log_growth(w_new, held)
        if not np.isfinite(obj):
            break
        if obj > best_obj:
            best_w, best_obj = w_new.copy(), obj
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    w = best_w

    analytics.loc[rows, "weight"] = w
    weights = pd.Series(w, index=analytics.loc[rows, "isin"].to_numpy(), name="weight")

    port_draws = w @ held
    wavg_downside = float(np.nansum(w * chosen["downside_dev"].to_numpy()))
    port_downside = downside_deviation(port_draws)

    if np.isfinite(port_downside) and port_downside < MIN_RATIO_DENOMINATOR:
        # Not a missing value — a finding. A long book with no modelled downside at
        # all is what cross-sectionally INDEPENDENT forward draws produce: pooling 25
        # names averages their idiosyncratic shocks to nothing, so the portfolio tail
        # vanishes and every reward-to-risk ratio built on it divides by ~0. If the
        # draws came from KalmanForecast, check `ForecastDraws.factor_share`; at 0.0
        # this outcome is an artefact of the generator, not a property of the book.
        logger.warning(
            "the sized book has effectively no modelled downside (deviation %.2e < "
            "%.0e): its 1-in-%d outcome is %+.2f%%. Diversification is being counted "
            "as free, which is what independent forward shocks produce. "
            "diversification_ratio is reported as nan rather than as a large number.",
            port_downside,
            MIN_RATIO_DENOMINATOR,
            int(round(1.0 / max(1.0 - var_prob, _EPS))),
            100.0 * generative_var(port_draws, prob=var_prob),
        )
    summary.update(
        port_expected=float(port_draws.mean()),
        port_gvar=generative_var(port_draws, prob=var_prob),
        port_ges=generative_expected_shortfall(port_draws, prob=var_prob),
        port_gtr=generative_tail_risk(port_draws),
        # The standard deviation of the PORTFOLIO vector, so cross-name correlation
        # is priced. Not comparable with RiskBook.summary['port_vol'].
        port_vol=float(port_draws.std()),
        port_kelly=kelly_fraction_from_draws(port_draws, multiplier=kelly_multiplier),
        log_growth=float(best_obj),
        # > 1 means the book's downside is smaller than the weighted average of its
        # names', i.e. diversification was earned. == 1 means the draws move together.
        diversification_ratio=(
            wavg_downside / port_downside
            if np.isfinite(port_downside) and port_downside >= MIN_RATIO_DENOMINATOR
            else float("nan")
        ),
        effective_n=float(1.0 / np.sum(w * w)) if np.sum(w * w) > 0 else 0.0,
    )
    logger.info(
        "book: %d names, effective N %.1f, E[r] %.2f%%, GVaR %.2f%%, growth %.5f",
        len(rows),
        summary["effective_n"],
        100.0 * summary["port_expected"],
        100.0 * summary["port_gvar"],
        summary["log_growth"],
    )
    return Portfolio(weights=weights, analytics=analytics, summary=summary)


def mean_variance_frontier(
        returns: np.ndarray,
        isins: Sequence[str],
        *,
        n_portfolios: int = 5000,
        risk_free_rate: float = 0.0,
        random_seed: int = 42,
) -> dict[str, Any]:
    """Markowitz mean-variance frontier — the labelled contrast, not the recommendation.

    Present so the comparison with :func:`optimize_portfolio` can be *measured* on
    this universe. Mean-variance treats volatility as total risk, which is symmetric
    and therefore charges a name for its upside; it optimises a single period, so it
    has nothing to say about the multiplicative dynamics an investor actually
    experiences; and its weights are famously unstable in the return estimates,
    which here are posterior means of a latent that moves between refreshes.

    The covariance is estimated **from the joint draws**, not from a separate
    simulation, so at least the two books are being compared on one distribution.

    Parameters
    ----------
    returns
        ``(n_isin, n_scenarios)`` joint draws.
    isins
        Labels aligned to axis 0.
    n_portfolios
        Dirichlet(1) random portfolios sampled.
    risk_free_rate
        Subtracted in the Sharpe numerator.
    random_seed
        Seed for the Dirichlet draws.

    Returns
    -------
    dict[str, Any]
        ``max_sharpe_weights`` (a ``pandas.Series``), ``max_sharpe`` /
        ``max_sharpe_return`` / ``max_sharpe_vol``, and the raw ``returns`` /
        ``vols`` / ``sharpe`` arrays for plotting the cloud.
    """
    draws = np.asarray(returns, dtype="float64")
    labels = np.asarray(isins)
    if draws.ndim != 2:
        raise ValueError(f"returns must be 2-D, got {draws.shape}")
    if labels.shape[0] != draws.shape[0]:
        raise ValueError(
            f"isins has {labels.shape[0]} entries for {draws.shape[0]} rows"
        )

    mu = draws.mean(axis=1)
    cov = np.cov(draws)
    if cov.ndim == 0:  # pragma: no cover - single name
        cov = cov.reshape(1, 1)

    rng = np.random.default_rng(random_seed)
    w = rng.dirichlet(np.ones(draws.shape[0]), size=int(n_portfolios))
    rets = w @ mu
    vols = np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", w, cov, w), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(vols > MIN_RATIO_DENOMINATOR, (rets - risk_free_rate) / vols, np.nan)

    if not np.isfinite(sharpe).any():  # pragma: no cover - degenerate universe
        raise ValueError("no portfolio produced a finite Sharpe ratio")
    best = int(np.nanargmax(sharpe))
    return {
        "max_sharpe_weights": pd.Series(w[best], index=labels, name="weight"),
        "max_sharpe": float(sharpe[best]),
        "max_sharpe_return": float(rets[best]),
        "max_sharpe_vol": float(vols[best]),
        "returns": rets,
        "vols": vols,
        "sharpe": sharpe,
    }