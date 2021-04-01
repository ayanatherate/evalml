
import numpy as np
import pandas as pd
import woodwork as ww

from evalml.utils.gen_utils import is_all_numeric
from evalml import Integer, Boolean

numeric_and_boolean_ww = [Integer, ww.logical_types.Double, Boolean]


def _numpy_to_pandas(array):
    if len(array.shape) == 1:
        data = pd.Series(array)
    else:
        data = pd.DataFrame(array)
    return data


def _list_to_pandas(list):
    return _numpy_to_pandas(np.array(list))


def infer_feature_types(data, feature_types=None):
    """Create a Woodwork structure from the given list, pandas, or numpy input, with specified types for columns.
        If a column's type is not specified, it will be inferred by Woodwork.

    Arguments:
        data (pd.DataFrame, pd.Series): Input data to convert to a Woodwork data structure.
        feature_types (string, ww.logical_type obj, dict, optional): If data is a 2D structure, feature_types must be a dictionary
            mapping column names to the type of data represented in the column. If data is a 1D structure, then feature_types must be
            a Woodwork logical type or a string representing a Woodwork logical type ("Double", "Integer", "Boolean", "Categorical", "Datetime", "NaturalLanguage")

    Returns:
        A Woodwork data structure where the data type of each column was either specified or inferred.
    """
    if isinstance(data, list):
        data = _list_to_pandas(data)
    elif isinstance(data, np.ndarray):
        data = _numpy_to_pandas(data)

    ww_data = data.copy()

    if isinstance(data, pd.Series):
        if data.ww._schema is not None:
            ww_data = ww.init_series(ww_data, **data.ww._schema)
        else:
            ww_data = ww.init_series(ww_data, logical_type=feature_types)
    else:
        if data.ww.schema is not None:
            ww_data.ww.init(logical_types=data.ww.logical_types,
                            semantic_tags=data.ww.semantic_tags)
        else:
            ww_data.ww.init(logical_types=feature_types)

    return ww_data


def _retain_custom_types_and_initalize_woodwork(old_logical_types, new_dataframe, ltypes_to_ignore=None):
    """
    Helper method which will take an old Woodwork DataTable and a new pandas DataFrame and return a
    new DataTable that will try to retain as many logical types from the old DataTable that exist in the new
    pandas DataFrame as possible.

    Arguments:
        old_datatable (ww.DataTable): Woodwork DataTable to use
        new_dataframe (pd.DataFrame): Pandas data structure
        ltypes_to_ignore (list): List of Woodwork logical types to ignore. Columns from the old DataTable that have a logical type
        specified in this list will not have their logical types carried over to the new DataTable returned

    Returns:
        A new DataTable where any of the columns that exist in the old input DataTable and the new DataFrame try to retain
        the original logical type, if possible and not specified to be ignored.
    """
    if ltypes_to_ignore is None:
        ltypes_to_ignore = []
    col_intersection = set(old_logical_types.keys()).intersection(set(new_dataframe.columns))
    retained_logical_types = {col: ltype for col, ltype in old_logical_types.items() if col in col_intersection and ltype not in ltypes_to_ignore}
    # retained_logical_types = {}
    # for col in col_intersection:
    #     if logical_types[col] in ltypes_to_ignore:
    #         continue
    #     if str(new_dataframe[col].dtype) != logical_types[col].primary_dtype:
    #         try:
    #             new_dataframe[col].astype(logical_types[col].primary_dtype)
    #             retained_logical_types[col] = logical_types[col]
    #         except (ValueError, TypeError):
    #             pass
    new_dataframe.ww.init(logical_types=retained_logical_types)
    return new_dataframe


def _convert_numeric_dataset_pandas(X, y):
    """Convert numeric and non-null data to pandas datatype. Raises ValueError if there is null or non-numeric data.
    Used with data sampler strategies.

    Arguments:
        X (pd.DataFrame, np.ndarray, ww.DataTable): Data to transform
        y (pd.Series, np.ndarray, ww.DataColumn): Target data

    Returns:
        Tuple(pd.DataFrame, pd.Series): Transformed X and y"""
    X_ww = infer_feature_types(X)
    if not is_all_numeric(X_ww):
        raise ValueError('Values not all numeric or there are null values provided in the dataset')
    y_ww = infer_feature_types(y)
    X_ww = _convert_woodwork_types_wrapper(X_ww)
    y_ww = _convert_woodwork_types_wrapper(y_ww)
    return X_ww, y_ww
