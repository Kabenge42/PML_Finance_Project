"""
Tests for ProbabilisticLinearRegression idata_kwargs workaround.

Validates that the PLR model passes ``idata_kwargs={"posterior_predictive": {}}``
to ``pm.sample`` so that the arviz-base 1.0 / PyMC incompatibility
(TypeError: 'NoneType' object is not iterable in posterior_predictive_to_xarray)
does not surface.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPLRIdataKwargs:
    """Verify the posterior_predictive workaround in ProbabilisticLinearRegression.fit."""

    def test_fit_passes_idata_kwargs_to_pm_sample(self):
        """pm.sample must receive idata_kwargs with posterior_predictive={}."""
        from probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel import (
            ProbabilisticLinearRegression,
        )

        fake_idata = MagicMock(name="idata")

        with patch(
            "probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel.pm"
        ) as mock_pm:
            mock_pm.sample.return_value = fake_idata
            mock_pm.Model.return_value.__enter__ = MagicMock(return_value=None)
            mock_pm.Model.return_value.__exit__ = MagicMock(return_value=False)
            mock_pm.Normal = MagicMock()
            mock_pm.HalfNormal = MagicMock()
            mock_pm.Data = MagicMock()

            with patch(
                "probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel.pt"
            ) as mock_pt:
                mock_pt.add.return_value = MagicMock()
                mock_pt.dot.return_value = MagicMock()

                plr = ProbabilisticLinearRegression()
                X = np.random.default_rng(42).standard_normal((20, 3))
                y = np.random.default_rng(42).standard_normal(20)

                plr.fit(X, y, samples=10, tune=5, chains=1, cores=1)

            # Verify pm.sample was called with idata_kwargs
            call_kwargs = mock_pm.sample.call_args
            assert "idata_kwargs" in call_kwargs.kwargs, (
                "pm.sample must be called with idata_kwargs"
            )
            idata_kw = call_kwargs.kwargs["idata_kwargs"]
            assert "posterior_predictive" in idata_kw, (
                "idata_kwargs must contain 'posterior_predictive' key"
            )
            assert idata_kw["posterior_predictive"] == {}, (
                "posterior_predictive must be an empty dict to avoid NoneType iteration"
            )

    def test_fit_source_contains_idata_kwargs(self):
        """Quick source-level check that idata_kwargs is present in fit()."""
        from probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel import (
            ProbabilisticLinearRegression,
        )

        src = inspect.getsource(ProbabilisticLinearRegression.fit)
        assert "idata_kwargs" in src
        assert "posterior_predictive" in src

    def test_fit_does_not_pass_posterior_predictive_none(self):
        """Ensure we never pass posterior_predictive=None to pm.sample."""
        from probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel import (
            ProbabilisticLinearRegression,
        )

        fake_idata = MagicMock(name="idata")

        with patch(
            "probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel.pm"
        ) as mock_pm:
            mock_pm.sample.return_value = fake_idata
            mock_pm.Model.return_value.__enter__ = MagicMock(return_value=None)
            mock_pm.Model.return_value.__exit__ = MagicMock(return_value=False)
            mock_pm.Normal = MagicMock()
            mock_pm.HalfNormal = MagicMock()
            mock_pm.Data = MagicMock()

            with patch(
                "probabilistic_ml_model.pml_models.ProbabilisticLinearRegressionModel.pt"
            ) as mock_pt:
                mock_pt.add.return_value = MagicMock()
                mock_pt.dot.return_value = MagicMock()

                plr = ProbabilisticLinearRegression()
                X = np.random.default_rng(42).standard_normal((20, 3))
                y = np.random.default_rng(42).standard_normal(20)

                plr.fit(X, y, samples=10, tune=5, chains=1, cores=1)

            idata_kw = mock_pm.sample.call_args.kwargs.get("idata_kwargs", {})
            pp = idata_kw.get("posterior_predictive")
            assert pp is not None, (
                "posterior_predictive must not be None — that causes the TypeError"
            )
