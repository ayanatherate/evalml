"""Component that applies a log transformation to the target data."""
import numpy as np
import pandas as pd
from skopt.space import Categorical

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class LogTransformer(Transformer):
    """Applies a log transformation to the target data."""

    name = "Log Transformer"

    hyperparameter_ranges = {"do_transform": Categorical([True, False])}
    """{"do_transform": Categorical([True, False])}"""
    modifies_features = False
    modifies_target = True

    def __init__(self, do_transform=True, random_seed=0):
        self.do_transform = do_transform
        super().__init__(parameters={'do_transform': do_transform},
                         component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the LogTransformer.

        Args:
            X (pd.DataFrame or np.ndarray): Ignored.
            y (pd.Series, optional): Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Log transforms the target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target data to log transform.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is log transformed.
        """
        if y is None:
            return X, y
        y_ww = infer_feature_types(y)
        if self.do_transform:
            self.min = y_ww.min()
            if self.min <= 0:
                y_ww = y_ww + abs(self.min) + 1
            y_t = infer_feature_types(np.log(y_ww))
        else:
            y_t = y_ww
        return X, y_t

    def fit_transform(self, X, y=None):
        """Log transforms the target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to log transform.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is log transformed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y):
        """Apply exponential to target data.

        Args:
            y (pd.Series): Target variable.

        Returns:
            pd.Series: Target with exponential applied.

        """
        y_ww = infer_feature_types(y)
        if self.do_transform:
            y_inv = np.exp(y_ww)
            if self.min <= 0:
                y_inv = y_inv - abs(self.min) - 1

            y_inv = infer_feature_types(pd.Series(y_inv, index=y_ww.index))
        else:
            y_inv = y_ww
        return y_inv

    @property
    def include_in_graph(self):
        return self.do_transform
