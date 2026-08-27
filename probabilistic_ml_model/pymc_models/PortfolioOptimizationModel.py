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
    MIN_TAIL_RISK,
    _cap_normalize_weights,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_KELLY_MULTIPLIER",
    "DEFAULT_RANKING_RULE",
    "DEFAULT_VAR_PROB",
    "MAX_KELLY_FRACTION",
    "RANKING_RULES",
    "RANKING_RULES_EXTERNAL",
    "RANK_TIEBREAK",
    "RELATIVE_DENOMINATOR_Q",
    "kelly_report",
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

#: Relative floor on a ranking denominator, as a fraction of the eligible universe's
#: MEDIAN denominator. ``0.0`` disables it and reproduces the absolute-floor-only
#: behaviour exactly.
#:
#: Why a second floor. ``MIN_RATIO_DENOMINATOR`` is an absolute guard against a
#: denormal, and it does its job: it excludes the names whose modelled downside is
#: exactly zero. The names it *admits* are the problem. Measured on run
#: ``448e7f055ef3``, every one of the twenty-five names ``reward_to_downside``
#: selected had a downside deviation between 0.000101 and 0.000538 against a universe
#: median of 0.0150 -- two orders of magnitude below it -- and the resulting ratio
#: spanned 75 to 4,997 inside a single book. That is not a reward-to-risk ranking; it
#: is a ranking on the ABSENCE of modelled downside, and the model's left tail is the
#: one thing nothing has validated.
#:
#: The default is 0.0 on purpose. Turning this on changes which names a book holds,
#: and that decision belongs to the caller and to a realised-return vintage, not to a
#: constant. What it is for is measurement: set it, read the log line saying how many
#: book names the relative floor binds on, and the answer is the finding.
RELATIVE_DENOMINATOR_Q: float = 0.0

#: Tie-break for a SATURATED ranking column, applied left to right after the rule.
#:
#: ``p_upside_pos_cond`` is bounded -- its virtue, and the reason it is the one
#: untried ranking candidate -- but it saturates: 59.4% of the universe sits at
#: exactly 1.0. A top-25 cut therefore lands entirely inside the tie, and without an
#: explicit rule ``argsort``'s ordering silently becomes the selection. Probability
#: first, then how much the name's own trail actually moved its estimate, then
#: magnitude.
RANK_TIEBREAK: tuple[str, ...] = ("shrink_gain", "expected_return")

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
        ``fraction``, ``log_growth`` (mean log growth per bet),
        ``log_terminal_wealth`` (natural log of the terminal capital),
        ``terminal_wealth`` and ``terminal_wealth_overflow``.

        ``log_growth`` and ``log_terminal_wealth`` are computed in log space and
        are always finite. ``terminal_wealth`` is the exponential of the second,
        and is **NaN where that has no float64 representation** -- with the flag
        beside it saying which rows those are.

    Notes
    -----
    The overflow is not an arithmetic slip, which is why the answer is not to
    clamp it. Compounding ``n`` bets at a mean log growth of ``g`` gives terminal
    capital ``exp(g*n)``, and on run ``486df52e7014`` the sized book's own draws
    put ``g*n`` near 845: the number is genuinely about ``10**367``, and float64
    stops at ``~1.8e308``. Every fraction past the point where the curve crosses
    that ceiling used to return ``+inf`` -- 21 of the 100 default fractions --
    which is not a wealth, and the ranking of two infinities is undefined.

    This function's own docstring used to claim that computing in log space meant
    a long sequence "does not overflow". That was true of the first column and
    false of the one anybody reads. ``log_terminal_wealth`` makes it true of a
    column that carries the whole answer: it is monotone in the same argument,
    it is finite everywhere the curve is defined, and the peak -- which is the
    point of the curve, and what :func:`kelly_fraction_from_draws` returns --
    sits at the same fraction in both.

    NaN rather than ``inf`` follows the project's export rule: an unrepresentable
    or undefined quantity is NULL with a BOOLEAN beside it saying why, because an
    infinity is neither exportable nor aggregatable. See ``kelly_unbounded`` in
    :func:`kelly_report` for the first instance of the same move.
    """
    vals = _finite_1d(returns)
    if n_bets is not None:
        vals = vals[: int(n_bets)]
    if fractions is None:
        fractions = np.arange(0.0, 1.0, 0.01)
    fracs = np.asarray(fractions, dtype="float64")

    # log(largest finite float64). Past this the exponential is not representable,
    # and the honest report is that the number is too large to write down -- not a
    # sentinel that sorts above every real value.
    log_max = float(np.log(np.finfo("float64").max))
    log_start = float(np.log(start_capital)) if start_capital > 0.0 else float("-inf")

    rows = []
    n_overflow = 0
    for f in fracs:
        growth = 1.0 + f * vals
        if np.any(growth <= 0.0):
            # Ruin: one bet takes the capital to zero or below, and nothing after it
            # can recover. Recorded as such rather than as a very small number.
            rows.append((float(f), float("-inf"), float("-inf"), 0.0, False))
            continue
        mean_log = float(np.mean(np.log(growth)))
        log_wealth = log_start + mean_log * vals.size
        over = log_wealth > log_max
        n_overflow += int(over)
        rows.append((
            float(f),
            mean_log,
            log_wealth,
            float("nan") if over else float(np.exp(log_wealth)),
            bool(over),
        ))

    if n_overflow:
        logger.info(
            "terminal wealth exceeds float64 for %d of %d fractions (log wealth up "
            "to %.0f against a %.0f ceiling): compounding %d bets at this growth "
            "rate is not a representable number. terminal_wealth is NaN there; read "
            "log_terminal_wealth, which is finite and has its peak at the same "
            "fraction.",
            n_overflow, len(fracs),
            max((r[2] for r in rows if np.isfinite(r[2])), default=float("nan")),
            log_max, int(vals.size),
        )
    return pd.DataFrame(
        rows,
        columns=["fraction", "log_growth", "log_terminal_wealth",
                 "terminal_wealth", "terminal_wealth_overflow"],
    )


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


def kelly_report(
        returns: np.ndarray,
        *,
        multiplier: float = 1.0,
        max_fraction: float = MAX_KELLY_FRACTION,
) -> dict[str, float]:
    """The Kelly fraction plus enough context to tell a solution from a pin.

    :func:`kelly_fraction_from_draws` returns ``max_fraction`` in two very different
    situations: when the log-optimal fraction genuinely sits at the cap, and when
    ``E[log(1 + f*r)]`` never turned over anywhere inside the feasible interval
    because no draw loses money. On run ``448e7f055ef3`` the second case covered
    **89.3% of the universe** and all 25 names of the decision book. A column reading
    ``1.000`` for nine names in ten is not a sizing recommendation; it is the
    bisection reporting that it had nothing to solve, and it must not be readable as
    the former.

    Returns
    -------
    dict[str, float]
        ``kelly_fraction`` -- as :func:`kelly_fraction_from_draws`.
        ``kelly_interior`` -- 1.0 when the solution lies strictly inside
        ``(0, max_fraction)``, i.e. the criterion actually chose it.
        ``kelly_endpoint_score`` -- ``E[r / (1 + f*r)]`` evaluated at ``max_fraction``.
        Positive means log growth was still RISING at the cap, which is the signature
        of a pin. Zero would be a genuine optimum sitting exactly there.
        ``kelly_max_feasible`` -- the largest ``f`` with ``1 + f*r > 0`` on every
        draw, or ``nan`` when no scenario loses money and no finite bound exists.
        ``kelly_unbounded`` -- 1.0 in exactly that case, which is the real binding
        constraint there and the thing to report instead of a fraction.

    Notes
    -----
    ``kelly_max_feasible`` USED to carry ``inf`` for the unbounded case, on the
    argument that reporting the constraint honestly beat reporting a fraction that
    did not exist. The argument stands; the encoding did not. An infinity cannot be
    exported -- ``export_finite`` blocks it, and a PostgreSQL ``float8 Infinity``
    poisons every downstream ``AVG`` and ``ORDER BY`` -- and on run ``6efb530d5881``
    this single column, at ``+inf`` for 626 of 6,513 names, was the whole of a
    blocking export failure on a fit that had passed every model gate.

    NULL plus a boolean says the same thing and survives the boundary: "no finite
    bound exists" is precisely what a NULL means, and ``kelly_unbounded`` says why.
    It is the same move ``kelly_interior`` already makes for ``kelly_fraction`` --
    a companion flag rather than an in-band value that has to be decoded.

    ``_max_feasible_fraction`` still returns ``inf`` internally; it is a
    computation, and :func:`kelly_fraction_from_draws` branches on
    ``np.isfinite`` of it.
    """
    vals = _finite_1d(returns)
    if vals.size == 0:
        nan = float("nan")
        return {"kelly_fraction": nan, "kelly_interior": nan,
                "kelly_endpoint_score": nan, "kelly_max_feasible": nan,
                "kelly_unbounded": nan}

    f = kelly_fraction_from_draws(vals, multiplier=multiplier, max_fraction=max_fraction)
    feasible = _max_feasible_fraction(vals)
    cap_f = min(max_fraction, feasible * (1.0 - 1e-9)) if np.isfinite(feasible) else max_fraction
    with np.errstate(divide="ignore", invalid="ignore"):
        endpoint = float(np.mean(vals / (1.0 + cap_f * vals)))
    interior = bool(_EPS < f < max_fraction * (1.0 - 1e-9))
    unbounded = not np.isfinite(feasible)
    return {
        "kelly_fraction": float(f),
        "kelly_interior": float(interior),
        "kelly_endpoint_score": endpoint,
        # NULL, not inf -- see Notes. The flag beside it is what carries the case.
        "kelly_max_feasible": float("nan") if unbounded else float(feasible),
        "kelly_unbounded": float(unbounded),
    }


# ---------------------------------------------------------------------------
# E. Ranking rules — three arms, one SSOT, no default moved
# ---------------------------------------------------------------------------
#
# Two candidate ranking denominators have now been shipped and measured, and both
# failed the same way. `tail_risk = max(-cvar05, 0.25*er_sd, MIN_TAIL_RISK)` collapses
# to its volatility floor for every name the book selects, so STARR is a
# reward-to-VARIABILITY ratio wearing a tail ratio's name. `downside_dev` was proposed
# as the cure and correlates 0.9948 with the STARR it replaced and 0.9883 with
# expected Sharpe -- one near-Sharpe ratio traded for another -- while selecting
# denominators two orders of magnitude below the universe median.
#
# `p_upside_pos_cond` is the one untried candidate, and its virtue is structural: a
# probability is BOUNDED, so it cannot be inflated by a vanishing denominator, which
# is the failure mode both ratio candidates share. Whether it is BETTER is a question
# about realised returns and nothing else, which is why all three ship as labelled
# arms and the default does not move.

#: Ranking rule name -> the analytics column it sorts on, descending.
RANKING_RULES: dict[str, str] = {
    "reward_to_downside": "reward_to_downside",
    "reward_to_cvar": "reward_to_cvar",
    "p_upside_pos_cond": "p_upside_pos_cond",
}

#: Rules whose column this module cannot compute from the draws and must be handed.
RANKING_RULES_EXTERNAL: frozenset[str] = frozenset({"p_upside_pos_cond"})

#: The arm that ships. Changing this changes the book, so it changes on evidence
#: about realised returns, not on a correlation.
DEFAULT_RANKING_RULE: str = "reward_to_downside"


def _floor_denominator(
        values: np.ndarray,
        eligible: np.ndarray,
        relative_q: float,
        label: str,
) -> tuple[np.ndarray, float, int]:
    """Mask denominators below the absolute and relative floors, and count the effect.

    The floor **excludes**, it does not clamp. Clamping would hand every name below
    the floor the same capped-but-still-enormous ratio and leave it in the running --
    which admits precisely the names this floor exists to keep out. It is also what
    the absolute floor already did before the relative one joined it, so excluding
    keeps the shipped arm bit-identical when ``relative_q`` is 0.

    Returns
    -------
    tuple[numpy.ndarray, float, int]
        The masked denominator (``nan`` below the floor), the floor, and the number
        of ELIGIBLE names it removed. That count is the point: a book whose every
        name sits below a floor derived from the universe median is a book selected
        on the absence of modelled risk, and the only way anyone finds that out is if
        it is counted.
    """
    den = np.asarray(values, dtype="float64").copy()
    rel = 0.0
    if relative_q > 0.0:
        pool = den[eligible & np.isfinite(den) & (den > 0.0)]
        if pool.size:
            rel = float(relative_q * np.median(pool))
    floor = max(MIN_RATIO_DENOMINATOR, rel)
    below = np.isfinite(den) & (den < floor)
    binds = int(np.sum(eligible & below))
    if binds:
        logger.info(
            "%s: the denominator floor %.3e excludes %d of %d eligible names "
            "(absolute %.0e, relative %.3e = %.3g x universe median)",
            label, floor, binds, int(eligible.sum()), MIN_RATIO_DENOMINATOR, rel, relative_q,
        )
    den[below] = np.nan
    return den, floor, binds


def _attach_ranking_columns(
        analytics: "pd.DataFrame",
        *,
        rank_by: str,
        rank_values: Optional[np.ndarray],
        rank_isins: Optional[Sequence[str]],
        eligible: np.ndarray,
        relative_denominator_q: float,
) -> "pd.DataFrame":
    """Build every ranking column, plus the diagnostics that make one readable.

    ``rank_denominator_pctile`` is the column that would have made the measured
    failure visible without recomputing anything: it is where each name's ranking
    denominator sits in the eligible universe's distribution. Every name of the run
    ``448e7f055ef3`` book sat in its bottom ~2%.
    """
    out = analytics.copy()
    ranks_needing_values = rank_by in RANKING_RULES_EXTERNAL

    for name, denom_col in (("reward_to_downside", "downside_dev"),
                            ("reward_to_cvar", "tail_risk")):
        den, floor, _binds = _floor_denominator(
            out[denom_col].to_numpy(), eligible, relative_denominator_q, name
        )
        # A BOOLEAN, not the masked denominator. `{denom}_admitted` used to carry
        # `den` itself, which is `{denom}` wherever the floor did not bind and NaN
        # where it did -- so with the default `relative_denominator_q = 0.0` it was
        # a byte-identical copy of its own source for most or all names, and
        # `export_duplicate_content` flagged `tail_risk == tail_risk_admitted` on
        # every run (0 of 6,513 cells masked on run `6efb530d5881`). The only
        # information the float column ever added over `{denom}` was WHICH names the
        # floor removed, which is exactly what this boolean is.
        out[f"{denom_col}_floored"] = ~np.isfinite(den) & np.isfinite(
            pd.to_numeric(out[denom_col], errors="coerce").to_numpy()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = out["expected_return"].to_numpy() / den
        out[name] = np.where(np.isfinite(ratio), ratio, np.nan)

    if ranks_needing_values:
        col = RANKING_RULES[rank_by]
        if rank_values is None:
            raise ValueError(
                f"rank_by={rank_by!r} needs rank_values: {col!r} is a screen column, "
                "not something the forward draws can produce. Pass it (and "
                "rank_isins) rather than letting this module invent one."
            )
        values = np.asarray(rank_values, dtype="float64")
        if rank_isins is None:
            if values.shape[0] != len(out):
                raise ValueError(
                    f"rank_values has {values.shape[0]} entries for {len(out)} names "
                    "and no rank_isins to align them by"
                )
            logger.warning(
                "rank_values aligned BY POSITION because rank_isins was not given. "
                "Pass rank_isins: a screen sorted by expected_upside against draws in "
                "universe order is a permutation no length check can see."
            )
            out[col] = values
        else:
            keyed = pd.Series(values, index=pd.Index(np.asarray(rank_isins), name="isin"))
            out[col] = keyed.reindex(out["isin"].to_numpy()).to_numpy()
            missing = int(np.sum(~np.isfinite(out[col].to_numpy())))
            if missing:
                logger.warning(
                    "%d of %d names have no %s value after the ISIN join; they cannot "
                    "be ranked on this arm", missing, len(out), col,
                )

    # Where each name's ranking denominator sits in the eligible universe. The column
    # that turns "selected on the absence of downside" from a diagnosis into a read.
    #
    # The percentile only. There used to be a `rank_denominator` column beside it,
    # holding a verbatim copy of `downside_dev` so a reader could see which column
    # ranked -- but WHICH column ranked is one fact about the run, not 6,513
    # identical rows, and as data it was simply a duplicate the export gate flagged.
    # It lives in `Portfolio.summary['rank_denominator_col']` instead, where one
    # fact belongs.
    denom_col = {"reward_to_downside": "downside_dev",
                 "reward_to_cvar": "tail_risk"}.get(rank_by)
    if denom_col is not None:
        pool = out.loc[eligible, denom_col].to_numpy()
        pool = pool[np.isfinite(pool)]
        if pool.size:
            out["rank_denominator_pctile"] = [
                float(np.mean(pool <= v)) if np.isfinite(v) else np.nan
                for v in out[denom_col].to_numpy()
            ]
    else:
        # A bounded probability has no denominator to report, which is exactly why it
        # is the one candidate immune to this failure mode.
        out["rank_denominator_pctile"] = np.nan
    return out


def _cap_normalize_with_groups(
        w: np.ndarray,
        cap: float,
        *,
        groups: Optional[np.ndarray] = None,
        group_cap: Optional[float] = None,
        max_passes: int = 64,
) -> np.ndarray:
    """Project onto the capped simplex, then onto the group-capped one, and repeat.

    The per-name cap is :func:`RiskBookModel._cap_normalize_weights` unchanged --
    there is no second copy of it here. The group pass scales any group over
    ``group_cap`` down to it and spills the remainder onto the groups below their cap,
    proportionally. The two constraints interact, so the passes alternate until both
    hold or ``max_passes`` is spent.

    A sector cap is a decision to take deliberately. Its absence is also a decision,
    and on run ``448e7f055ef3`` that decision produced a book **60.9% in Information
    Technology** -- taken by omission, which is the only way it should never be taken.
    """
    out = _cap_normalize_weights(w, cap)
    if groups is None or group_cap is None or not (0.0 < group_cap < 1.0):
        return out

    codes = np.asarray(groups)
    for _ in range(max_passes):
        totals = {g: out[codes == g].sum() for g in np.unique(codes)}
        over = {g: t for g, t in totals.items() if t > group_cap + 1e-12}
        if not over:
            break
        excess = 0.0
        for g, total in over.items():
            sel = codes == g
            scale = group_cap / total
            excess += total - group_cap
            out[sel] *= scale
        room = np.array([
            max(group_cap - totals[g], 0.0) if g not in over else 0.0 for g in codes
        ])
        headroom = np.array([
            (max(group_cap - totals[g], 0.0) if g not in over else 0.0) for g in np.unique(codes)
        ]).sum()
        if headroom <= _EPS:
            # Nowhere to spill: the cap cannot be met at this book size. Normalise and
            # let the caller see the breach in the exported concentration column
            # rather than looping forever on an infeasible constraint.
            logger.warning(
                "sector cap %.0f%% is infeasible for this book (%d groups); the "
                "weights are normalised without it",
                100.0 * group_cap, len(totals),
            )
            break
        # Spill proportionally to each under-cap name's share of its group's headroom.
        weight_room = np.where(room > 0, out + _EPS, 0.0)
        if weight_room.sum() <= _EPS:
            break
        out = out + excess * weight_room / weight_room.sum()
        out = _cap_normalize_weights(out, cap)
    total = out.sum()
    return out / total if total > 0 else out


@dataclass(frozen=True, eq=False)
class Portfolio:
    """A sized book plus the risk it carries, all measured on the joint draws.

    Attributes
    ----------
    weights
        ``isin -> weight``, summing to 1 over the held names, every weight ``<= cap``.
    analytics
        Per-name frame: ``isin``, ``expected_return``, ``er_sd``, ``er_p05``,
        ``er_p95``, ``kelly_fraction``, ``gvar``, ``ges``, ``gtr``,
        ``downside_dev``, ``reward_to_downside``, ``weight``. Names outside the
        book carry weight 0 and keep their statistics.

        Four companion **booleans** travel with the quantities they qualify —
        ``kelly_interior``, ``kelly_unbounded``, ``tail_risk_on_floor`` and
        ``{denominator}_floored``. Each says something about a number that the
        number itself cannot: whether the criterion chose it, whether a bound
        exists at all, whether a floor took over. They are booleans on purpose —
        that keeps them out of ``select_dtypes([np.number])``, so a flag can never
        trip the finiteness or duplicate-content export gates, and it keeps them
        SQL ``boolean`` rather than a 0.0/1.0 double a reader has to decode.

        The ``er_*`` block is named as the rest of the pipeline names it, so a
        consumer that reads a forward-return distribution off any frame --
        ``apply_out_of_support`` is the one that matters -- finds the same
        quantities here. ``er_mean`` is absent on purpose: ``expected_return``
        already holds it, and two names for one quantity is what
        ``export_duplicate_content`` flags.
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
        rank_by: str = DEFAULT_RANKING_RULE,
        rank_values: Optional[np.ndarray] = None,
        rank_isins: Optional[Sequence[str]] = None,
        tail_risk_vol_floor_k: float = 0.25,
        relative_denominator_q: float = RELATIVE_DENOMINATOR_Q,
        groups: Optional[Sequence[Any]] = None,
        sector_cap: Optional[float] = None,
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
        Number of names in the book. Taken from the caller rather than defaulted to
        ``RiskBookModel.DEFAULT_K_BOOK``. The two had disagreed; as of 2026-08-26 the
        constant (50) and what ``KalmanRunConfigV2`` passes (50) agree, and the same
        holds for ``DEFAULT_MCAP_R_MAX`` / ``mcap_global_r_max`` at 0.03 — verified,
        not assumed. Deferring to the caller anyway is what stops the two drifting
        apart again silently.
    cap
        Maximum single-name weight.
    kelly_multiplier
        Applied to the reported ``port_kelly``; weights themselves are relative and
        already sum to 1.
    var_prob
        Confidence for the GVaR / GES columns.
    eligible
        Optional boolean mask over names. Anything already excluded upstream (market
        cap, support, out-of-support rows, the size-down watch) belongs here rather
        than being re-derived.
    rank_by
        Which arm of :data:`RANKING_RULES` selects the book. The default does not
        move; the other arms are contrasts, and every frame records which one ran.
    rank_values, rank_isins
        The column for a rule this module cannot compute -- currently only
        ``p_upside_pos_cond``, which comes from the screen. Aligned **by ISIN** when
        ``rank_isins`` is given, which it should be: passing values in the screen's
        ``expected_upside`` order against draws in universe order is the positional
        join this project has already shipped once.
    tail_risk_vol_floor_k
        Relative volatility floor in the ``reward_to_cvar`` arm's denominator,
        mirroring ``RiskBookModel``. Load-bearing rather than inert: it is what the
        denominator collapses to for every name the book selects.
    relative_denominator_q
        See :data:`RELATIVE_DENOMINATOR_Q`. ``0.0`` (the default) reproduces the
        absolute-floor-only behaviour exactly.
    groups, sector_cap
        Optional per-name group labels and a maximum weight for any one group.
        ``None`` keeps today's behaviour -- see :func:`_cap_normalize_with_groups`
        for why the absence of a sector cap is itself a decision.
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

    if rank_by not in RANKING_RULES:
        raise ValueError(
            f"Unknown rank_by {rank_by!r}. Valid arms: {sorted(RANKING_RULES)}"
        )

    n_isin = draws.shape[0]
    kelly = [kelly_report(draws[i], max_fraction=MAX_KELLY_FRACTION) for i in range(n_isin)]
    per_name = {
        "isin": labels,
        "expected_return": draws.mean(axis=1),
        "er_sd": draws.std(axis=1),
        "kelly_fraction": np.array([k["kelly_fraction"] for k in kelly]),
        # Not decoration: without these a `kelly_fraction` of 1.000 for nine names in
        # ten is unreadable as the pin it is. See `kelly_report`.
        "kelly_interior": np.array([k["kelly_interior"] for k in kelly]).astype(bool),
        "kelly_endpoint_score": np.array([k["kelly_endpoint_score"] for k in kelly]),
        # NaN where no finite bound exists; `kelly_unbounded` says which those are.
        # A bool, deliberately: booleans sit outside `select_dtypes([np.number])`,
        # so a flag can never itself trip the finiteness or duplicate-content gates,
        # and it lands as a SQL `boolean` rather than as a 0.0/1.0 double.
        "kelly_max_feasible": np.array([k["kelly_max_feasible"] for k in kelly]),
        "kelly_unbounded": np.array([k["kelly_unbounded"] for k in kelly]).astype(bool),
        "gvar": np.array([generative_var(draws[i], prob=var_prob) for i in range(n_isin)]),
        "ges": np.array(
            [generative_expected_shortfall(draws[i], prob=var_prob) for i in range(n_isin)]
        ),
        "gtr": np.array([generative_tail_risk(draws[i]) for i in range(n_isin)]),
        "downside_dev": np.array([downside_deviation(draws[i]) for i in range(n_isin)]),
        # Both tails, under the project's `er_*` names, from the same terminal
        # draws `expected_return` and `er_sd` summarise. `er_p05` alone is not
        # enough: `apply_out_of_support` tests the UPPER clip bound on the 5th
        # percentile and the LOWER one on the 95th, so a frame carrying only the
        # former gets the lower bound tested on a mean -- which matched zero
        # affected names when it was last measured -- or, before 2026-08-27, on a
        # column that did not exist at all (`KeyError: 'er_mean'` on the 15b
        # decision frame). `er_mean` is deliberately NOT emitted: it would be
        # byte-identical to `expected_return`, which is precisely what the
        # `export_duplicate_content` gate exists to flag.
        "er_p05": np.quantile(draws, 0.05, axis=1),
        "er_p95": np.quantile(draws, 0.95, axis=1),
    }
    analytics = pd.DataFrame(per_name)

    # The `reward_to_cvar` arm: RiskBookModel's shipped STARR denominator, computed
    # here on the terminal draws so the two rankings contrast like with like. The
    # `expected_upside - cvar05` leg is deliberately absent, as it is on the
    # return-draw path there -- it fell as the tail improved, so the ratio rose on
    # numerator and denominator together.
    analytics["tail_risk"] = np.maximum.reduce([
        -analytics["er_p05"].to_numpy(),
        float(tail_risk_vol_floor_k) * analytics["er_sd"].to_numpy(),
        np.full(n_isin, MIN_TAIL_RISK),
    ])
    analytics["tail_risk_on_floor"] = (
        analytics["tail_risk"].to_numpy()
        <= float(tail_risk_vol_floor_k) * analytics["er_sd"].to_numpy() + 1e-12
    )

    mask = np.ones(n_isin, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.shape[0] != n_isin:
        raise ValueError(f"eligible has {mask.shape[0]} entries for {n_isin} names")

    analytics = _attach_ranking_columns(
        analytics,
        rank_by=rank_by,
        rank_values=rank_values,
        rank_isins=rank_isins,
        eligible=mask,
        relative_denominator_q=float(relative_denominator_q),
    )

    rank_col = RANKING_RULES[rank_by]
    selectable = (
        mask
        & (analytics["expected_return"].to_numpy() > 0.0)
        & np.isfinite(analytics[rank_col].to_numpy())
    )

    analytics["weight"] = 0.0
    # Explicit tie-break. `p_upside_pos_cond` saturates at 1.0 for the majority of the
    # universe, so a top-k cut lands inside the tie and argsort's order would silently
    # become the selection rule. Descending on every key; missing keys are skipped.
    sort_cols = [rank_col] + [c for c in RANK_TIEBREAK
                              if c in analytics.columns and c != rank_col]
    chosen = (
        analytics.loc[selectable]
        .sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
        .head(int(k_book))
    )
    tie_span = int(
        (analytics.loc[selectable, rank_col] == chosen[rank_col].min()).sum()
    ) if len(chosen) else 0
    if tie_span > 1 and len(chosen) >= int(k_book):
        logger.info(
            "the %s cut at rank %d fell among %d tied names; broken on %s",
            rank_col, int(k_book), tie_span, list(sort_cols[1:]) or "nothing",
        )

    summary: dict[str, Any] = {
        "k_book": float(k_book),
        "cap": float(cap),
        "var_prob": float(var_prob),
        "kelly_multiplier": float(kelly_multiplier),
        "n_eligible": float(int(selectable.sum())),
        "n_book": float(len(chosen)),
        "rank_by": rank_by,
        # WHICH column the ranking divided by, as one fact rather than as a
        # per-name copy of that column. `p_upside_pos_cond` is a bounded
        # probability and has no denominator, which is the whole reason it is the
        # one arm immune to a vanishing one.
        "rank_denominator_col": {"reward_to_downside": "downside_dev",
                                 "reward_to_cvar": "tail_risk"}.get(rank_by, ""),
        "rank_tiebreak": ",".join(sort_cols[1:]),
        "rank_tie_span": float(tie_span),
        "sector_cap": float(sector_cap) if sector_cap is not None else float("nan"),
        "relative_denominator_q": float(relative_denominator_q),
        # The 89.3% finding, as a tracked number rather than an anecdote.
        "kelly_interior_share": float(analytics.loc[mask, "kelly_interior"].mean())
        if mask.any() else float("nan"),
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

    held_groups: Optional[np.ndarray] = None
    if groups is not None and sector_cap is not None:
        all_groups = np.asarray(groups)
        if all_groups.shape[0] != n_isin:
            raise ValueError(
                f"groups has {all_groups.shape[0]} entries for {n_isin} names"
            )
        held_groups = all_groups[rows]

    def _project(vec: np.ndarray) -> np.ndarray:
        return _cap_normalize_with_groups(
            vec, cap, groups=held_groups, group_cap=sector_cap
        )

    # Exponentiated gradient ascent on E[log(1 + w @ r)]. Multiplicative updates keep
    # the weights non-negative without a projection, and the cap-and-spill step is
    # what puts them back on the capped simplex.
    w = _project(np.ones(k))
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
        w_new = _project(w_new)
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
        # Reported at the FRACTIONAL multiplier, not full Kelly. Overbetting is the
        # asymmetric error and the edge here rests on a prior-driven shrinkage, so
        # the full fraction is an upper bound rather than a recommendation.
        port_kelly_full=kelly_fraction_from_draws(port_draws),
        # Where the book's names sit in the eligible universe's denominator
        # distribution. On the shipped arm this is the measurement that says whether
        # the ranking chose low risk or merely unmodelled risk.
        book_denominator_pctile_max=(
            float(chosen["rank_denominator_pctile"].max())
            if "rank_denominator_pctile" in chosen else float("nan")
        ),
        book_kelly_interior_share=float(chosen["kelly_interior"].mean()),
    )
    if held_groups is not None:
        totals = pd.Series(w).groupby(pd.Series(held_groups).to_numpy()).sum()
        summary["top_group_weight"] = float(totals.max())
        summary["top_group"] = str(totals.idxmax())
        summary["n_groups"] = float(totals.size)
    elif groups is not None:
        all_groups = np.asarray(groups)
        totals = pd.Series(w).groupby(all_groups[rows]).sum()
        summary["top_group_weight"] = float(totals.max())
        summary["top_group"] = str(totals.idxmax())
        summary["n_groups"] = float(totals.size)
    logger.info(
        "book [%s]: %d names, effective N %.1f, E[r] %.2f%%, GVaR %.2f%%, growth "
        "%.5f, interior Kelly %.0f%% of book, top group %s",
        rank_by,
        len(rows),
        summary["effective_n"],
        100.0 * summary["port_expected"],
        100.0 * summary["port_gvar"],
        summary["log_growth"],
        100.0 * summary["book_kelly_interior_share"],
        f"{summary['top_group_weight']:.1%}" if "top_group_weight" in summary else "n/a",
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