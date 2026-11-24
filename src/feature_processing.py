"""Shared raw-record cleaning and feature preparation for train and predict."""

import numpy as np
import pandas as pd

from src.config import get_config


def clean_records(df):
    """Apply the configured duplicate and missing-record policy."""
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if get_config("preprocessing", "drop_duplicates", default=True):
        result = result.drop_duplicates()
    if get_config("preprocessing", "handle_missing_values", default=True):
        result = result.dropna()
    if result.empty:
        raise ValueError("No customer records remain after cleaning.")
    return result


def prepare_feature_frame(df, feature_columns, gender_mapping):
    """Select features, encode Gender and validate finite numeric values."""
    if not feature_columns:
        raise ValueError("Feature columns are not configured.")

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    features = df.loc[:, feature_columns].copy()
    if "Gender" in features.columns:
        if not gender_mapping:
            raise ValueError("Gender mapping is missing in config.yaml.")

        normalized_mapping = {
            str(label).strip().casefold(): int(value)
            for label, value in gender_mapping.items()
        }
        valid_encoded = set(normalized_mapping.values())
        gender = features["Gender"]

        if pd.api.types.is_numeric_dtype(gender):
            encoded = pd.to_numeric(gender, errors="raise")
            if not set(encoded.dropna().unique()).issubset(valid_encoded):
                raise ValueError("Gender contains values outside the configured encoding.")
        else:
            labels = gender.astype("string").str.strip().str.casefold()
            encoded = labels.map(normalized_mapping)
            if encoded.isna().any():
                unknown = labels[encoded.isna()].dropna().unique().tolist()
                raise ValueError(f"Unknown or missing Gender values: {unknown}")

        features["Gender"] = encoded.astype(int)

    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="raise")

    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature data contains missing or infinite values.")
    return features
