"""The ELPD comparison must work on a panel that partitions.

Why this file exists
--------------------
``run_model_comparison`` was built, documented, gated and exported, and could not
produce an ELPD contrast on **any** real panel. The v2 likelihood is one
``MvStudentT`` per covariance group, so ``log_likelihood`` carries
``target_pct_obs_g0..gN`` and ``az.compare`` raises ``TypeError: Encountered
error trying to compute ELPD from model <arm>`` because it cannot choose among
them.

It went unnoticed for four editions of the post-run analysis because the
**single-group case works**. ``_simulate_panel`` produces a fully-observed panel,
which partitions into exactly one group and emits a single unsuffixed
``target_pct_obs`` -- so every self-test passed while the production path could
not run. Measured 2026-08-25: a ``baseline`` vs ``level_off`` contrast fitted
both arms cleanly at zero divergences in 9.7 minutes, on a panel that split into
3 groups of [776, 20, 4], and produced nothing.

The lesson these tests encode is narrow and worth stating: a fixture that cannot
reach the branch under test is not coverage. Every case here **forces** a
partition.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("arviz")

import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402

import pymc_kalman_filter_pt_v2 as v2  # noqa: E402
from probabilistic_ml_model.pymc_models._workflow import (  # noqa: E402
    attach_log_likelihood,
)
from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (  # noqa: E402
    KalmanModelConfig,
    _simulate_panel,
    build_kalman_pt_model_v2,
    covariance_groups_for,
)

LOOKBACKS = ("1y", "3m", "1w")


def _partitioned_panel(n_isin: int = 90):
    """A panel that splits into more than one covariance group.

    ``partition_covariance_groups`` splits on the OBSERVED-COLUMN PATTERN, so
    blanking different columns for different slices of names is what creates
    groups. A fully-observed panel -- what ``_simulate_panel`` returns -- is a
    single group and cannot exercise the stitching at all.
    """
    panel, _ = _simulate_panel(n_isin=n_isin, lookbacks=LOOKBACKS)
    cfg = KalmanModelConfig(lookbacks=LOOKBACKS)
    Y = np.asarray(panel.Y, dtype="float64").copy()
    Y[: n_isin // 6, 0] = np.nan          # oldest column missing
    Y[n_isin // 6: n_isin // 4, 1] = np.nan  # middle column missing
    panel = dc_replace(panel, Y=Y)
    groups, _ = covariance_groups_for(panel, cfg)
    assert len(groups) > 1, "fixture failed to create a multi-group panel"
    return panel, cfg, groups


@pytest.fixture
def fitted_multigroup():
    """A short fit on a partitioned panel, rebuilt PER TEST.

    Deliberately not module-scoped despite the cost. `collapse_group_loglik`
    rewrites `log_likelihood` in place, so a shared fixture makes the suite
    order-dependent: the two tests that read the raw grouped variables only pass
    while they happen to run before the first test that stitches them. That is
    the kind of green suite that goes red under `-p no:randomly` removal, `-k`
    selection, or a reordering, and it would be hiding exactly the mutation this
    file exists to characterise.
    """
    panel, cfg, groups = _partitioned_panel()
    model = build_kalman_pt_model_v2(panel, config=cfg)
    with model:
        idata = pm.sample(
            draws=60, tune=60, chains=2, cores=1, progressbar=False,
            random_seed=3, compute_convergence_checks=False,
        )
    attach_log_likelihood(idata, model)
    return panel, cfg, groups, idata


def test_the_panel_actually_partitions(fitted_multigroup):
    """Guard the fixture itself: no partition means no coverage."""
    panel, cfg, groups, idata = fitted_multigroup
    assert len(groups) > 1
    names = sorted(str(k) for k in idata.log_likelihood.data_vars)
    assert names == sorted(f"target_pct_obs_g{i}" for i in range(len(groups))), names


def test_raw_grouped_loglik_cannot_be_compared(fitted_multigroup):
    """Pin the defect, so a regression is loud rather than silent.

    If a future arviz makes this work on its own, this test fails and the
    stitching can be revisited deliberately -- which is the point of pinning a
    known-bad behaviour rather than only the fix.
    """
    _panel, _cfg, _groups, idata = fitted_multigroup
    with pytest.raises(Exception):
        az.compare({"a": idata, "b": idata})


def test_collapse_covers_every_name_exactly_once(fitted_multigroup):
    """The stitched variable spans the panel with no gap and no NaN."""
    panel, cfg, _groups, idata = fitted_multigroup
    fixed = v2.collapse_group_loglik(idata, panel, cfg)
    ll = fixed.log_likelihood

    assert list(ll.data_vars) == ["target_pct_obs"]
    arr = ll["target_pct_obs"]
    assert arr.dims == ("chain", "draw", "isin")
    # The pointwise unit is the NAME: a name's T cells are correlated by
    # construction, so leaving out one cell would not be a leave-one-out.
    assert arr.shape[2] == panel.n_isin
    assert not np.isnan(np.asarray(arr)).any()


def test_collapsed_loglik_scores(fitted_multigroup):
    """az.compare and az.loo both run once the groups are stitched."""
    panel, cfg, _groups, idata = fitted_multigroup
    fixed = v2.collapse_group_loglik(idata, panel, cfg)

    tab = az.compare({"a": fixed, "b": fixed})
    assert len(tab) == 2
    # Identical arms must tie: a non-zero elpd_diff here would mean the stitch
    # is not deterministic in the row order.
    assert float(np.abs(tab["elpd_diff"]).max()) == pytest.approx(0.0, abs=1e-9)

    loo = az.loo(fixed)
    # ArviZ 1.x exposes `.elpd`; `.elpd_loo` was REMOVED and a getattr fallback
    # on the old name yields a silent nan.
    assert np.isfinite(loo.elpd)
    assert np.isfinite(loo.se)


def test_single_group_panel_is_passed_through(fitted_multigroup):
    """A one-group panel already has one variable and must not be rewritten."""
    panel, _ = _simulate_panel(n_isin=60, lookbacks=LOOKBACKS)
    cfg = KalmanModelConfig(lookbacks=LOOKBACKS)
    groups, _ = covariance_groups_for(panel, cfg)
    assert len(groups) == 1, "fully-observed panel should be one group"

    model = build_kalman_pt_model_v2(panel, config=cfg)
    with model:
        idata = pm.sample(
            draws=40, tune=40, chains=2, cores=1, progressbar=False,
            random_seed=5, compute_convergence_checks=False,
        )
    attach_log_likelihood(idata, model)
    before = list(idata.log_likelihood.data_vars)
    out = v2.collapse_group_loglik(idata, panel, cfg)
    assert list(out.log_likelihood.data_vars) == before == ["target_pct_obs"]


def test_collapse_is_idempotent(fitted_multigroup):
    """A second call must return the stitched arm, not raise about consumed vars.

    `collapse_group_loglik` rewrites `log_likelihood` IN PLACE (the convention
    `attach_log_likelihood` sets). Without idempotence the second call looks for
    `target_pct_obs_g*` -- which the first call replaced -- and raises a KeyError
    about variables the caller never touched.
    """
    panel, cfg, _groups, idata = fitted_multigroup
    once = v2.collapse_group_loglik(idata, panel, cfg)
    twice = v2.collapse_group_loglik(once, panel, cfg)
    assert list(twice.log_likelihood.data_vars) == ["target_pct_obs"]
    np.testing.assert_allclose(
        np.asarray(once.log_likelihood["target_pct_obs"]),
        np.asarray(twice.log_likelihood["target_pct_obs"]),
    )


def test_missing_group_variable_raises_rather_than_scoring_a_subset():
    """Dropping a group silently would score the arms on different names."""
    panel, cfg, groups = _partitioned_panel(n_isin=60)
    model = build_kalman_pt_model_v2(panel, config=cfg)
    with model:
        idata = pm.sample(
            draws=30, tune=30, chains=2, cores=1, progressbar=False,
            random_seed=7, compute_convergence_checks=False,
        )
    attach_log_likelihood(idata, model)

    # arviz 1.x hands back an xarray DataTree, whose `log_likelihood` node is a
    # Dataset -- drop on the DATASET and assign the node back.
    ll = idata.log_likelihood.to_dataset() if hasattr(
        idata.log_likelihood, "to_dataset") else idata.log_likelihood
    victim = sorted(str(k) for k in ll.data_vars)[-1]
    idata.log_likelihood = ll.drop_vars(victim)

    with pytest.raises(KeyError, match="log_likelihood lacks"):
        v2.collapse_group_loglik(idata, panel, cfg)
