"""Decision communication — postures and action lists, as frames rather than prints.

What this module is for
-----------------------
The v1 workflow ends at §14b ``run_recommendations``, three hundred lines that turn a
posterior into an *actionable posture*: group over/underweights, a name-level action
list, a size-down watch, a demotion list, and a reliability gate over all of it. The v2
workflow has **no equivalent at all** — it is numerically richer (three ranking rules,
Kelly, the generative risk triple) and rhetorically empty, emitting ranked frames that
never say what to do with them.

The logic is worth keeping. The medium is not: a console block cannot be exported,
diffed across runs, gated, plotted or read by a dashboard, which is why a pipeline that
tracks twenty-five gate verdicts tracks zero postures. Every function here returns a
DataFrame; :func:`render_recommendations` formats what they return and computes
nothing.

Why the group signals matter more than they look
------------------------------------------------
:func:`group_allocation_signals` is the only **shrunk** decision quantity in the
project, and it is the structural opposite of the failure the post-run analysis keeps
measuring downstream. A reward-to-risk ratio rewards a name for having little modelled
risk, so thin evidence *inflates* it — that is how a book came to hold twenty-five
names whose downside deviation sat two orders of magnitude below the universe median.
Here, thin or noisy evidence pulls a group's signal *toward zero*: ``lambda_g``
multiplies the raw excess by ``tau^2 / (tau^2 + s_g^2)``, so a group the data cannot
resolve gets no verdict rather than an extreme one. It also answers a question neither
sized book answers — which sector or region to overweight — which is exactly what a
book 60.9% concentrated in one sector has no counterweight to.

The confidence argument, and why it is an argument
--------------------------------------------------
v1 conditions every probability on the posterior ``achieve_prob``. **v2 removed that
variable** — a sigmoid of a standardised log-uplift is the probability of no defined
event. In v2 the quantity carrying the same meaning is ``shrink_gain``; ``kalman_gain``
keeps its name for one release but is now ``P(risk_adj_return > 0)``. So ``confidence``
is a parameter of these functions, never a posterior read: v1 passes ``achieve_prob``,
v2 passes ``shrink_gain``, and neither caller has to know about the other's variable.
A default of 1.0 makes the conditioning inert, which is a legitimate state to be in and
is reported as such rather than hidden.

Units
-----
Frames keep the project's **raw-decimal** convention (0.25 = +25%). Percent scaling
happens in :func:`render_recommendations` and nowhere else.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_GROUP_N",
    "GROUP_SIGNAL_COORDS",
    "HDI_LO",
    "HDI_HI",
    "VERDICTS",
    "group_allocation_signals",
    "name_action_list",
    "size_down_watch",
    "demotion_list",
    "reliability_posture",
    "render_recommendations",
    "fmt_or_na",
    "display_label",
]

#: Minimum names behind a group before it receives a posture verdict.
#:
#: A one-to-three name "exchange" is a stock pick, not an allocation signal. The gate
#: is applied **before** the shrinkage statistics so thin groups neither receive
#: verdicts nor distort ``tau^2`` and the +/-1 sd band that every other group is judged
#: against. Ported from v1, where it was a function-local literal.
MIN_GROUP_N: int = 15

#: Coordinate columns that carry group postures, in reporting order.
#:
#: v1 kept two overlapping lists — one for the prints, a shorter one for the forest —
#: which is how a coord could be reported in text and absent from the picture. One list.
GROUP_SIGNAL_COORDS: tuple[str, ...] = (
    "region",
    "trading_region",
    "country_name",
    "trading_country_name",
    "exchange_name",
    "unit_name",
    "sector",
    "industry",
    "size_class",
    "style_class",
)

#: Credible-interval bounds, matching v1's ``_HDI_LO`` / ``_HDI_HI`` (an 92% interval).
HDI_LO: float = 0.04
HDI_HI: float = 0.96

#: The three postures. Ordered strong-to-weak so a sort is meaningful.
VERDICTS: tuple[str, ...] = ("OVERWEIGHT", "NEUTRAL", "UNDERWEIGHT")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Formatting helpers, lifted from v1 so both renderers share one copy
# ---------------------------------------------------------------------------


def fmt_or_na(x: Any, nd: int = 1, suf: str = "") -> str:
    """Nan-safe fixed-point formatter with an ``'n/a'`` fallback."""
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{x:.{nd}f}{suf}"
    except Exception:
        return "n/a"


def display_label(row: Any) -> str:
    """Company display name, falling back to the ISIN."""
    name = row.get("name")
    return name if isinstance(name, str) and name.strip() else str(row["isin"])


# ---------------------------------------------------------------------------
# 1. Group allocation signals — the shrunk posture
# ---------------------------------------------------------------------------


def _draws_2d(latent: Any) -> np.ndarray:
    """Coerce ``(chain, draw, isin)`` / ``(sample, isin)`` / ``(isin, sample)``.

    Returns ``(n_isin, n_samples)``. An xarray object is recognised by its ``dims``,
    so the ISIN axis is found by NAME rather than guessed from a shape — two axes of
    equal length would otherwise transpose a whole universe in silence.
    """
    dims = getattr(latent, "dims", None)
    if dims is not None and "isin" in dims:
        arr = np.asarray(latent)
        axis = list(dims).index("isin")
        arr = np.moveaxis(arr, axis, -1)
        return arr.reshape(-1, arr.shape[-1]).T
    arr = np.asarray(latent)
    if arr.ndim == 3:                      # (chain, draw, isin)
        return arr.reshape(-1, arr.shape[-1]).T
    if arr.ndim != 2:
        raise ValueError(f"latent must be 2-D or 3-D, got shape {arr.shape}")
    return arr


def group_allocation_signals(
        latent: Any,
        coords: pd.DataFrame,
        *,
        confidence: Optional[np.ndarray] = None,
        levels: Sequence[str] = GROUP_SIGNAL_COORDS,
        min_group_n: int = MIN_GROUP_N,
        p_hi: float = 0.80,
        p_lo: float = 0.20,
        positive_threshold: float = 0.0,
) -> pd.DataFrame:
    """Hierarchically shrunk over/underweight postures, one row per group.

    For each level, each group's raw excess over the universe posture is shrunk by

    ``lambda_g = tau^2 / (tau^2 + s_g^2)``

    where ``tau^2`` is the between-group variance of the posterior group means (the
    signal) and ``s_g`` is group *g*'s own posterior sd (its noise). A group the data
    resolves well keeps its excess; one it does not is pulled toward zero. The
    over/underweight band is +/-1 cross-group sd of the **shrunk** excess, computed per
    level, which is what makes it posture-relative rather than a static threshold that
    means different things in different universes.

    Parameters
    ----------
    latent
        The decision quantity's draws — ``(chain, draw, isin)``, ``(sample, isin)`` or
        ``(n_isin, n_samples)``. In whatever units the caller reports; raw decimals by
        this project's convention. Pass the **forward terminal returns** to get a
        posture on the quantity a book is sized from, or the upside posterior to get
        the one v1 reports; both are legitimate and they are a contrast, not a
        substitute for each other.
    coords
        Per-name frame aligned row-for-row with the ISIN axis of ``latent``, carrying
        an ``isin`` column and any of ``levels``.
    confidence
        ``(n_isin,)`` per-name state confidence in ``[0, 1]``. v1 passes the posterior
        ``achieve_prob``; v2 passes ``shrink_gain``. The group's positive-probability
        is multiplied by its group-mean confidence, and the ``p_hi`` / ``p_lo`` gates
        are rescaled by the **universe-mean** confidence so both sides of the
        comparison sit on the same conditional scale. ``None`` leaves the conditioning
        inert at 1.0 and records that in ``univ_confidence``.
    levels
        Coordinate columns to report. Missing ones are skipped, not an error.
    min_group_n
        See :data:`MIN_GROUP_N`.
    p_hi, p_lo
        Nominal probability gates, before the confidence rescale.
    positive_threshold
        The bar ``P(> threshold)`` is measured against. ``0.0`` asks whether the group
        beats zero; a positive value asks whether it beats a hurdle.

    Returns
    -------
    pandas.DataFrame
        ``level``, ``group``, ``n``, ``mean``, ``hdi_lo``, ``hdi_hi``, ``sd``,
        ``excess_raw``, ``lambda_g``, ``excess_shrunk``, ``p_pos``, ``confidence``,
        ``p_pos_cond``, ``band``, ``verdict``, ``univ_mean``, ``univ_confidence``,
        ``n_dropped``. Empty when no level has an eligible group.
    """
    draws = _draws_2d(latent)
    n_isin = draws.shape[0]
    if len(coords) != n_isin:
        raise ValueError(
            f"coords has {len(coords)} rows for {n_isin} names in latent. These must "
            "be aligned row-for-row; join them by ISIN before calling."
        )

    name_mean = draws.mean(axis=1)
    univ_mean = float(np.nanmean(name_mean))

    if confidence is None:
        conf = np.ones(n_isin)
        logger.info(
            "group_allocation_signals called without `confidence`; the conditioning "
            "is inert. v1 passes achieve_prob, v2 passes shrink_gain -- note that "
            "achieve_prob does NOT exist in a v2 posterior."
        )
    else:
        conf = np.asarray(confidence, dtype="float64")
        if conf.shape[0] != n_isin:
            raise ValueError(
                f"confidence has {conf.shape[0]} entries for {n_isin} names"
            )
    univ_conf = float(np.nanmean(conf))
    if not np.isfinite(univ_conf) or univ_conf <= 0:
        univ_conf = 1.0
    hi_gate, lo_gate = p_hi * univ_conf, p_lo * univ_conf

    rows: list[dict[str, Any]] = []
    for level in levels:
        if level not in coords.columns:
            continue
        labels = coords[level].fillna("Unknown").astype(str).to_numpy()
        counts = pd.Series(labels).value_counts()
        eligible = [g for g in counts.index if int(counts[g]) >= min_group_n]
        n_dropped = int(counts.size - len(eligible))
        if not eligible:
            logger.info(
                "%s: 0 of %d groups reach n>=%d; no posture reported",
                level, int(counts.size), min_group_n,
            )
            continue

        stats: list[dict[str, Any]] = []
        for group in eligible:
            sel = labels == group
            gdraws = draws[sel].mean(axis=0)          # group-mean per posterior sample
            stats.append({
                "level": level,
                "group": str(group),
                "n": int(sel.sum()),
                "mean": float(gdraws.mean()),
                "sd": float(gdraws.std()),
                "hdi_lo": float(np.quantile(gdraws, HDI_LO)),
                "hdi_hi": float(np.quantile(gdraws, HDI_HI)),
                "p_pos": float((gdraws > positive_threshold).mean()),
                "confidence": float(np.nanmean(conf[sel])),
            })

        frame = pd.DataFrame(stats)
        frame["excess_raw"] = frame["mean"] - univ_mean
        # tau^2 is the between-group variance of the group MEANS: the signal the
        # shrinkage weighs each group's own noise against.
        tau2 = float(frame["excess_raw"].var(ddof=0))
        frame["lambda_g"] = (
            tau2 / (tau2 + frame["sd"] ** 2) if tau2 > 0 else 0.0
        )
        frame["excess_shrunk"] = frame["lambda_g"] * frame["excess_raw"]
        band = float(frame["excess_shrunk"].std(ddof=0))
        frame["band"] = band
        frame["p_pos_cond"] = frame["p_pos"] * frame["confidence"]
        frame["univ_mean"] = univ_mean
        frame["univ_confidence"] = univ_conf
        frame["n_dropped"] = n_dropped

        ow = band if band > 0 else float("inf")
        frame["verdict"] = [
            _verdict(ex, pc, ow, -ow, hi_gate, lo_gate)
            for ex, pc in zip(frame["excess_shrunk"], frame["p_pos_cond"])
        ]
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=[
            "level", "group", "n", "mean", "sd", "hdi_lo", "hdi_hi", "p_pos",
            "confidence", "excess_raw", "lambda_g", "excess_shrunk", "band",
            "p_pos_cond", "univ_mean", "univ_confidence", "n_dropped", "verdict",
        ])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["level", "excess_shrunk"], ascending=[True, False])
    logger.info(
        "group signals: %d groups over %d levels; universe posture %.4f, mean "
        "confidence %.3f, gates %.2f/%.2f",
        len(out), out["level"].nunique(), univ_mean, univ_conf, hi_gate, lo_gate,
    )
    return out.reset_index(drop=True)


def _verdict(excess_shrunk: float, p_cond: float,
             ow_thr: float, uw_thr: float,
             p_hi: float, p_lo: float) -> str:
    """Posture from the shrunk excess and the conditional positive probability."""
    if not np.isfinite(excess_shrunk):
        return "NEUTRAL"
    if excess_shrunk >= ow_thr and p_cond >= p_hi:
        return "OVERWEIGHT"
    if excess_shrunk <= uw_thr or p_cond <= p_lo:
        return "UNDERWEIGHT"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# 2. Name-level actions
# ---------------------------------------------------------------------------


def name_action_list(
        analytics: pd.DataFrame,
        *,
        confidence_scale: float = 1.0,
        hi: float = 0.75,
        lo: float = 0.25,
        prob_col: str = "p_upside_pos_cond",
        return_col: str = "expected_upside",
) -> pd.DataFrame:
    """Per-name BUY / AVOID / HOLD, on the conditional probability scale.

    The gates are the nominal ``hi`` / ``lo`` rescaled by ``confidence_scale``, which
    must be the same universe-mean confidence ``prob_col`` was itself conditioned on.
    Comparing an unconditional gate against a conditional column is how a screen ends
    up with no high-conviction names and no explanation.

    Returns
    -------
    pandas.DataFrame
        The input plus ``action`` and the two gate values, sorted with the strongest
        longs first and the strongest avoids last.
    """
    out = analytics.copy()
    for col in (prob_col, return_col):
        if col not in out.columns:
            raise KeyError(
                f"name_action_list needs {col!r}; got {sorted(out.columns)[:12]}..."
            )
    hi_gate, lo_gate = hi * confidence_scale, lo * confidence_scale
    prob = pd.to_numeric(out[prob_col], errors="coerce")
    ret = pd.to_numeric(out[return_col], errors="coerce")

    action = np.where(
        (ret > 0) & (prob >= hi_gate), "BUY",
        np.where((ret < 0) & (prob <= lo_gate), "AVOID", "HOLD"),
    )
    out["action"] = action
    out["gate_hi"] = hi_gate
    out["gate_lo"] = lo_gate
    order = pd.Categorical(out["action"], categories=["BUY", "HOLD", "AVOID"],
                           ordered=True)
    out = out.assign(_o=order).sort_values(
        ["_o", prob_col, return_col], ascending=[True, False, False]
    ).drop(columns="_o")
    logger.info(
        "name actions: %d BUY, %d AVOID, %d HOLD (gates %.2f/%.2f at confidence "
        "scale %.3f)",
        int((out.action == "BUY").sum()), int((out.action == "AVOID").sum()),
        int((out.action == "HOLD").sum()), hi_gate, lo_gate, confidence_scale,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. The size-down watch — a veto orthogonal to the ranking
# ---------------------------------------------------------------------------


def size_down_watch(
        analytics: pd.DataFrame,
        *,
        band_quantile: float = 0.95,
        min_analysts: int = 2,
        band_col: str = "band_width",
        analyst_col: str = "n_analysts",
) -> pd.DataFrame:
    """Names whose posterior is too wide, or whose analyst panel too thin, to size up.

    This is the piece of v1's §14b that most directly addresses what the post-run
    analysis measures downstream. Both sized books select names whose modelled loss
    distributions barely reach below zero, and a **wide posterior** and a **thin
    analyst panel** are precisely the conditions that produce a forward simulation
    with no credible left tail. The watch names them without touching the rank — it is
    a veto, not a re-ranking, which is what lets it be read as an independent opinion.

    Parameters
    ----------
    analytics
        Per-name frame. Missing columns are tolerated; the corresponding leg is simply
        not applied, and which legs ran is recorded on the result.
    band_quantile
        Posterior band width at or above this quantile of the universe is flagged.
    min_analysts
        At or below this coverage is flagged.

    Returns
    -------
    pandas.DataFrame
        One row per FLAGGED name, with ``flag_wide_band``, ``flag_thin_coverage``,
        ``size_down_flag`` and ``band_threshold``. ``attrs['legs']`` lists the legs
        that were applied.
    """
    frame = analytics.copy()
    legs: list[str] = []

    wide = pd.Series(False, index=frame.index)
    threshold = float("nan")
    if band_col in frame.columns:
        band = pd.to_numeric(frame[band_col], errors="coerce")
        if band.notna().any():
            threshold = float(np.nanquantile(band, band_quantile))
            wide = band >= threshold
            legs.append(f"{band_col}>=q{band_quantile:.2f}")

    thin = pd.Series(False, index=frame.index)
    if analyst_col in frame.columns:
        thin = pd.to_numeric(frame[analyst_col], errors="coerce").fillna(0) <= min_analysts
        legs.append(f"{analyst_col}<={min_analysts}")

    if not legs:
        logger.warning(
            "size_down_watch found neither %r nor %r; no veto applied. An empty watch "
            "because the columns are absent is not the same as an empty watch because "
            "nothing was flagged.",
            band_col, analyst_col,
        )

    frame["flag_wide_band"] = wide
    frame["flag_thin_coverage"] = thin
    frame["size_down_flag"] = wide | thin
    frame["band_threshold"] = threshold
    out = frame.loc[frame["size_down_flag"]].copy()
    if band_col in out.columns:
        out = out.sort_values(band_col, ascending=False)
    out.attrs["legs"] = legs
    logger.info(
        "size-down watch: %d of %d names flagged (%d wide band, %d thin coverage); "
        "legs applied: %s",
        len(out), len(frame), int(wide.sum()), int(thin.sum()), legs or "none",
    )
    return out.reset_index(drop=True)


def size_down_mask(analytics: pd.DataFrame, isins: Sequence[str], **kwargs: Any) -> np.ndarray:
    """The watch as an ``eligible`` mask for ``optimize_portfolio``, aligned by ISIN.

    ``True`` means *not* flagged, i.e. sizeable. Keep this **opt-in**: applying it
    changes which names a book holds, and the value of doing so is a question about
    realised returns. What is not optional is measuring it — pass the mask, compare
    the books, and report how many names it removed.
    """
    watch = size_down_watch(analytics, **kwargs)
    flagged = set(watch["isin"].astype(str)) if "isin" in watch.columns else set()
    return np.array([str(i) not in flagged for i in isins], dtype=bool)


# ---------------------------------------------------------------------------
# 4. Demotion — what the risk screen rejected, and why
# ---------------------------------------------------------------------------


def demotion_list(
        analytics: pd.DataFrame,
        *,
        return_col: str = "expected_upside",
        ratio_col: str = "ret_vol_ratio",
        top_n: int = 20,
        max_rows: int = 10,
) -> pd.DataFrame:
    """High raw return, bottom-half risk-adjusted: the names the vol screen demotes.

    Naming what a ranking **rejects** is what makes it auditable. A screen that only
    ever publishes its winners cannot be argued with, because the cases where it
    disagrees with the raw signal are exactly the ones a reader would want to check.

    Returns
    -------
    pandas.DataFrame
        Up to ``max_rows`` names drawn from the top ``top_n`` by ``return_col`` whose
        ``ratio_col`` sits below the eligible median, with ``universe_median_ratio``.
    """
    frame = analytics.copy()
    for col in (return_col, ratio_col):
        if col not in frame.columns:
            logger.info("demotion_list: %r absent; nothing to report", col)
            return frame.iloc[0:0]
    ratio = pd.to_numeric(frame[ratio_col], errors="coerce")
    ret = pd.to_numeric(frame[return_col], errors="coerce")
    pool = frame[ratio.notna() & (ret > 0)]
    if pool.empty:
        return frame.iloc[0:0]
    median_ratio = float(pd.to_numeric(pool[ratio_col], errors="coerce").median())
    high = pool.sort_values(return_col, ascending=False).head(max(top_n, max_rows))
    out = high[pd.to_numeric(high[ratio_col], errors="coerce") < median_ratio].copy()
    out["universe_median_ratio"] = median_ratio
    out = out.sort_values(return_col, ascending=False).head(max_rows)
    logger.info(
        "demotion: %d of the top %d by %s fall below the universe median %s of %.3f",
        len(out), len(high), return_col, ratio_col, median_ratio,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Reliability — the posture that conditions every posture above
# ---------------------------------------------------------------------------


def reliability_posture(
        *,
        diagnostics: Optional[pd.DataFrame] = None,
        idata: Any = None,
        n_divergences: Optional[int] = None,
        r_hat_ok: float = 1.01,
        r_hat_warn: float = 1.05,
) -> dict[str, Any]:
    """Whether the fit is good enough to act on, as a dict rather than a print.

    Reads a diagnostics frame by preference — that is what v2 already exports, and it
    works for a replay that has no ``InferenceData`` at all — falling back to the
    posterior when one is supplied.

    Returns
    -------
    dict[str, Any]
        ``max_r_hat``, ``min_ess_bulk``, ``n_divergences``, ``nu``, ``posture``
        (``OK`` / ``ACCEPTABLE`` / ``CONCERN``) and ``advice``.
    """
    max_rhat = float("nan")
    min_ess = float("nan")
    nu = float("nan")

    if diagnostics is not None and len(diagnostics):
        frame = diagnostics
        if "r_hat" in frame.columns:
            max_rhat = float(pd.to_numeric(frame["r_hat"], errors="coerce").max())
        if "ess_bulk" in frame.columns:
            min_ess = float(pd.to_numeric(frame["ess_bulk"], errors="coerce").min())
        name_col = next((c for c in ("index", "parameter", "var_name")
                         if c in frame.columns), None)
        if name_col is not None and "mean" in frame.columns:
            hit = frame[frame[name_col].astype(str).str.fullmatch("nu")]
            if len(hit):
                nu = float(pd.to_numeric(hit["mean"], errors="coerce").iloc[0])

    if idata is not None:
        try:
            stats = idata.sample_stats if hasattr(idata, "sample_stats") else idata["sample_stats"]
            if n_divergences is None and "diverging" in stats:
                n_divergences = int(np.asarray(stats["diverging"]).sum())
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("could not read sample_stats: %s", exc)
        if not np.isfinite(nu):
            try:
                post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
                if "nu" in post:
                    nu = float(np.asarray(post["nu"]).mean())
            except Exception:  # pragma: no cover - defensive
                pass

    if np.isfinite(max_rhat) and max_rhat <= r_hat_ok and (n_divergences or 0) == 0:
        posture = "OK"
        advice = "High-quality posterior; the signals below are safe to act on."
    elif np.isfinite(max_rhat) and max_rhat <= r_hat_warn:
        posture = "ACCEPTABLE"
        advice = "Acceptable convergence; size positions conservatively."
    else:
        posture = "CONCERN"
        advice = "Convergence concerns; treat every signal below as indicative only."

    # Not a tail-weight reading. A nu falling toward its 2.5 floor is the signal to
    # inspect the observation scale, because that is what it absorbs when the scale is
    # mis-specified -- which is why it is reported here rather than acted on.
    if np.isfinite(nu) and nu <= 7.0:
        advice += (" Student-t df is low: analyst-target outliers are material, so "
                   "prefer tail-aware sizing over mean-based ranking.")

    out = {
        "max_r_hat": max_rhat,
        "min_ess_bulk": min_ess,
        "n_divergences": int(n_divergences) if n_divergences is not None else None,
        "nu": nu,
        "posture": posture,
        "advice": advice,
    }
    logger.info(
        "reliability: %s (max R-hat %s, min ESS %s, divergences %s, nu %s)",
        posture, fmt_or_na(max_rhat, 4), fmt_or_na(min_ess, 0),
        out["n_divergences"], fmt_or_na(nu, 2),
    )
    return out


# ---------------------------------------------------------------------------
# 6. The console view — formats, computes nothing
# ---------------------------------------------------------------------------


def render_recommendations(
        *,
        reliability: Optional[dict[str, Any]] = None,
        group_signals: Optional[pd.DataFrame] = None,
        actions: Optional[pd.DataFrame] = None,
        watch: Optional[pd.DataFrame] = None,
        demoted: Optional[pd.DataFrame] = None,
        book: Optional[pd.DataFrame] = None,
        book_summary: Optional[dict[str, Any]] = None,
        title: str = "KALMAN SCREEN - ACTIONABLE RECOMMENDATIONS",
        max_rows: int = 10,
        printer: Any = print,
) -> None:
    """Print the frames above. Every argument optional; absent sections are skipped.

    Percent scaling happens **here and only here** — the frames keep raw decimals, so
    a consumer that is not a terminal never has to undo a display decision.
    """
    rule = "=" * 88
    printer(rule)
    printer(title)
    printer(rule)

    if reliability:
        printer(
            f"\n1. POSTERIOR RELIABILITY   max R-hat="
            f"{fmt_or_na(reliability.get('max_r_hat'), 4)}   "
            f"min ESS={fmt_or_na(reliability.get('min_ess_bulk'), 0)}   "
            f"divergences={reliability.get('n_divergences')}   "
            f"nu={fmt_or_na(reliability.get('nu'), 2)}"
        )
        marker = {"OK": "[OK]", "ACCEPTABLE": "[~] ", "CONCERN": "[!!]"}.get(
            reliability.get("posture", ""), "[?] "
        )
        printer(f"   {marker}  {reliability.get('advice', '')}")

    if group_signals is not None and len(group_signals):
        printer(
            f"\n2. GROUP POSTURE   {len(group_signals)} groups over "
            f"{group_signals['level'].nunique()} levels   "
            f"universe={group_signals['univ_mean'].iloc[0] * 100:.2f}%   "
            f"mean confidence={group_signals['univ_confidence'].iloc[0]:.2f}"
        )
        printer(
            f"   Rule: excess vs universe, shrunk by lambda=tau^2/(tau^2+s_g^2); "
            f"OVERWEIGHT at >= +1 cross-group sd of the shrunk excess and a high "
            f"conditional P(>0). Groups with n<{MIN_GROUP_N} are excluded before the "
            f"shrinkage statistics."
        )
        for level, block in group_signals.groupby("level", sort=False):
            band = float(block["band"].iloc[0]) * 100
            dropped = int(block["n_dropped"].iloc[0])
            printer(f"\n   -- {level.upper()}  ({len(block)} groups, {dropped} dropped "
                    f"n<{MIN_GROUP_N})  band=+/-{band:.2f}pp shrunk excess")
            shown = block if len(block) <= max_rows else pd.concat(
                [block.head(max_rows // 2), block.tail(max_rows // 2)]
            )
            for _, r in shown.iterrows():
                printer(
                    f"   {r['verdict']:>11s}  {r['group']:<26.26s}  "
                    f"mean={r['mean'] * 100:6.2f}%  "
                    f"CI=[{r['hdi_lo'] * 100:6.2f},{r['hdi_hi'] * 100:6.2f}]  "
                    f"xs={r['excess_shrunk'] * 100:5.2f}pp  "
                    f"(raw {r['excess_raw'] * 100:5.2f}, lambda={r['lambda_g']:.2f})  "
                    f"P={r['p_pos_cond']:4.0%}  n={int(r['n'])}"
                )
            if len(block) > max_rows:
                printer(f"   ... {len(block) - max_rows} mid-ranked groups omitted ...")

    if actions is not None and len(actions):
        for label, key in (("High-conviction LONGS", "BUY"), ("AVOID candidates", "AVOID")):
            block = actions[actions["action"] == key]
            printer(f"\n3. {label}: {len(block)} names")
            for _, r in block.head(max_rows).iterrows():
                printer(
                    f"   {key:<6s} {display_label(r):<20.20s}  "
                    f"return={r.get('expected_upside', float('nan')) * 100:6.2f}%  "
                    f"P={r.get('p_upside_pos_cond', float('nan')):4.0%}  "
                    f"n_analysts={fmt_or_na(r.get('n_analysts'), 0)}"
                )

    if watch is not None and len(watch):
        printer(f"\n4. SIZE-DOWN WATCH: {len(watch)} names "
                f"({', '.join(watch.attrs.get('legs', [])) or 'no legs applied'})")
        printer("   A veto orthogonal to the ranking: a wide posterior or a thin "
                "analyst panel is what produces a forward simulation with no credible "
                "left tail, which is what both sized books select on.")
        for _, r in watch.head(max_rows).iterrows():
            printer(
                f"   CAUTION {display_label(r):<20.20s}  "
                f"band={fmt_or_na(r.get('band_width', float('nan')) * 100, 1, '%')}  "
                f"n_analysts={fmt_or_na(r.get('n_analysts'), 0)}"
            )

    if demoted is not None and len(demoted):
        printer(f"\n5. DEMOTED BY RISK ({len(demoted)} names: high return, "
                f"bottom-half risk-adjusted)")
        for _, r in demoted.iterrows():
            printer(
                f"   TRIM   {display_label(r):<20.20s}  "
                f"return={r.get('expected_upside', float('nan')) * 100:6.2f}%  "
                f"ratio={fmt_or_na(r.get('ret_vol_ratio'), 2)} vs universe median "
                f"{fmt_or_na(r.get('universe_median_ratio'), 2)}"
            )

    if book is not None and len(book):
        summary = book_summary or {}
        printer(
            f"\n6. SIZED BOOK   rank_by={summary.get('rank_by', '?')}   "
            f"{len(book)} names   effective N="
            f"{fmt_or_na(summary.get('effective_n'), 1)}   "
            f"top group={fmt_or_na((summary.get('top_group_weight') or float('nan')) * 100, 1, '%')}"
        )
        wcol = next((c for c in ("weight", "book_weight") if c in book.columns), None)
        for _, r in book.head(max_rows).iterrows():
            printer(
                f"   {display_label(r):<20.20s}  "
                f"wt={(r[wcol] * 100 if wcol else float('nan')):5.1f}%  "
                f"E[r]={r.get('expected_return', float('nan')) * 100:7.2f}%  "
                f"GVaR={fmt_or_na(r.get('gvar', float('nan')) * 100, 1, '%'):>7}"
            )

    printer("\n" + rule)
    printer(
        "Model-implied screens from analyst-target dynamics, NOT investment advice. "
        "Every gate above scores the model against the trail it was fitted to; none "
        "of them can distinguish a favourable modelled tail from an optimistic one. "
        "Size on the risk budget and combine with fundamentals, liquidity and limits."
    )
    printer(rule)
