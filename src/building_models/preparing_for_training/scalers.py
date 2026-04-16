from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)


ScalerMethod = Literal[
    "standard",
    "minmax",
    "maxabs",
    "quantile",
    "power",
    "l1",
    "l2",
    "max",
]


class Scaler:
    """
    Utility class for feature scaling and normalization.

    This class provides a unified interface for several scikit-learn
    preprocessing transformers, including scaling-based methods and
    vector normalization methods.

    Supported methods
    -----------------
    - "standard": StandardScaler
    - "minmax": MinMaxScaler
    - "maxabs": MaxAbsScaler
    - "quantile": QuantileTransformer
    - "power": PowerTransformer
    - "l1": Normalizer(norm="l1")
    - "l2": Normalizer(norm="l2")
    - "max": Normalizer(norm="max")
    """

    @classmethod
    def get_scaler(cls, method: ScalerMethod, **kwargs: Any):
        """
        Return an instantiated scikit-learn scaler/normalizer.

        Parameters
        ----------
        method : ScalerMethod
            Scaling or normalization method to use.
        **kwargs : Any
            Additional keyword arguments passed to the underlying
            scikit-learn transformer.

        Returns
        -------
        object
            Instantiated scikit-learn transformer.

        Raises
        ------
        ValueError
            If the provided method is not supported.
        """
        method = method.lower()

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "maxabs": MaxAbsScaler,
            "quantile": QuantileTransformer,
            "power": PowerTransformer,
            "l1": lambda **kw: Normalizer(norm="l1", **kw),
            "l2": lambda **kw: Normalizer(norm="l2", **kw),
            "max": lambda **kw: Normalizer(norm="max", **kw),
        }

        if method not in scalers:
            available_methods = ", ".join(scalers.keys())
            raise ValueError(
                f"Unsupported scaling method '{method}'. "
                f"Available methods are: {available_methods}."
            )

        scaler_cls = scalers[method]
        return scaler_cls(**kwargs)

    @classmethod
    def fit(cls, X: np.ndarray, method: ScalerMethod = "standard", **kwargs: Any):
        """
        Fit a scaler/normalizer to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        method : ScalerMethod, default="standard"
            Scaling or normalization method.
        **kwargs : Any
            Additional parameters for the transformer.

        Returns
        -------
        object
            Fitted transformer.
        """
        scaler = cls.get_scaler(method=method, **kwargs)
        scaler.fit(X)
        return scaler

    @classmethod
    def transform(cls, X: np.ndarray, fitted_scaler: Any) -> np.ndarray:
        """
        Transform data using a previously fitted scaler/normalizer.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        fitted_scaler : Any
            Previously fitted scikit-learn transformer.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return fitted_scaler.transform(X)

    @classmethod
    def fit_transform(
        cls,
        X: np.ndarray,
        method: ScalerMethod = "standard",
        return_scaler: bool = False,
        **kwargs: Any,
    ):
        """
        Fit and transform data in one step.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        method : ScalerMethod, default="standard"
            Scaling or normalization method.
        return_scaler : bool, default=False
            Whether to also return the fitted transformer.
        **kwargs : Any
            Additional parameters for the transformer.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, object]
            Transformed data, or transformed data together with the fitted scaler
            if `return_scaler=True`.
        """
        scaler = cls.get_scaler(method=method, **kwargs)
        X_scaled = scaler.fit_transform(X)

        if return_scaler:
            return X_scaled, scaler
        return X_scaled