"""Smoke tests for the InferenceData shim in _pymc_arviz_compat."""
import numpy as np
import xarray as xr

from probabilistic_ml_model._pymc_arviz_compat import InferenceData


def test_tuple_unpacking():
    """Tuples (data, warmup) are unpacked so groups hold plain Datasets."""
    posterior_ds = xr.Dataset({"alpha": ("chain", np.array([1.0, 2.0]))})
    warmup_posterior_ds = xr.Dataset({"alpha": ("chain", np.array([0.5, 0.7]))})
    stats_ds = xr.Dataset(
        {"diverging": (("chain", "draw"), np.array([[False, True], [False, False]]))}
    )
    warmup_stats_ds = xr.Dataset(
        {"diverging": (("chain", "draw"), np.array([[False, False], [False, False]]))}
    )

    idata = InferenceData(
        posterior=(posterior_ds, warmup_posterior_ds),
        sample_stats=(stats_ds, warmup_stats_ds),
    )

    # Groups should only contain unpacked Datasets, not tuples
    assert isinstance(idata.get("posterior"), xr.Dataset)
    assert isinstance(idata.get("sample_stats"), xr.Dataset)

    # __getitem__ access (used by convergence.py)
    assert isinstance(idata["posterior"], xr.Dataset)
    assert "chain" in idata["posterior"].sizes

    # .get() on sample_stats Dataset (convergence.py line 139)
    ss = idata.get("sample_stats")
    assert ss is not None
    diverging = ss.get("diverging", None)
    assert diverging is not None

    # warmup should NOT be stored when save_warmup=False (default)
    assert "warmup_posterior" not in idata.groups()
    assert "warmup_sample_stats" not in idata.groups()


def test_tuple_unpacking_with_warmup():
    """When save_warmup=True, warmup groups should be stored."""
    posterior_ds = xr.Dataset({"alpha": ("chain", np.array([1.0]))})
    warmup_posterior_ds = xr.Dataset({"alpha": ("chain", np.array([0.5]))})

    idata = InferenceData(
        save_warmup=True,
        posterior=(posterior_ds, warmup_posterior_ds),
    )

    assert "posterior" in idata.groups()
    assert "warmup_posterior" in idata.groups()
    assert isinstance(idata["warmup_posterior"], xr.Dataset)


def test_plain_dataset_still_works():
    """Non-tuple Dataset values should still work as before."""
    ds = xr.Dataset({"x": ("d", np.array([1, 2, 3]))})
    idata = InferenceData(posterior=ds)
    assert isinstance(idata["posterior"], xr.Dataset)
    assert idata.get("posterior") is ds


def test_get_missing_key():
    """Missing keys return default via .get()."""
    idata = InferenceData()
    assert idata.get("nonexistent", None) is None


def test_getitem_missing_key():
    """Missing keys raise KeyError via []."""
    idata = InferenceData()
    try:
        _ = idata["nonexistent"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_contains():
    ds = xr.Dataset({"x": ("d", np.array([1]))})
    idata = InferenceData(posterior=ds)
    assert "posterior" in idata
    assert "missing" not in idata


def test_none_groups_skipped():
    """None values should be silently skipped."""
    ds = xr.Dataset({"x": ("d", np.array([1]))})
    idata = InferenceData(posterior=ds, sample_stats=None)
    assert "sample_stats" not in idata.groups()
    assert idata.get("sample_stats") is None


def test_tuple_with_none_warmup():
    """Tuple where warmup element is None."""
    ds = xr.Dataset({"diverging": ("draw", np.array([False, True]))})
    idata = InferenceData(sample_stats=(ds, None))
    assert isinstance(idata.get("sample_stats"), xr.Dataset)


def test_tuple_with_none_main():
    """Tuple where main element is None should skip the group."""
    warmup_ds = xr.Dataset({"x": ("d", np.array([1]))})
    idata = InferenceData(posterior=(None, warmup_ds))
    assert "posterior" not in idata.groups()
