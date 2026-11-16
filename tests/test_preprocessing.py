import numpy as np
import pandas as pd

from src.preprocess import (
    select_features,
    encode_gender,
    scale_features
)


def sample_data():
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Age": [20, 30, 40, 50],
        "Annual Income (k$)": [20, 40, 60, 80],
        "Spending Score (1-100)": [30, 50, 70, 90]
    })


def test_select_features():
    df = sample_data()

    result = select_features(df)

    expected_columns = [
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    assert list(result.columns) == expected_columns
    assert result.shape == (4, 4)


def test_encode_gender():
    df = sample_data()

    result = encode_gender(df)

    assert result["Gender"].tolist() == [0, 1, 0, 1]
    assert pd.api.types.is_numeric_dtype(result["Gender"])


def test_scale_features():
    df = sample_data()

    encoded = encode_gender(select_features(df))

    standard_df, minmax_df = scale_features(encoded)

    assert standard_df.shape == (4, 4)
    assert minmax_df.shape == (4, 4)

    assert np.isfinite(standard_df.to_numpy()).all()
    assert np.isfinite(minmax_df.to_numpy()).all()

    assert np.allclose(
        minmax_df.min().to_numpy(),
        0
    )

    assert np.allclose(
        minmax_df.max().to_numpy(),
        1
    )


def test_preprocessing_pipeline():
    df = sample_data()

    features = select_features(df)
    encoded = encode_gender(features)
    standard_df, minmax_df = scale_features(encoded)

    assert len(standard_df) == 4
    assert len(minmax_df) == 4

    assert standard_df.isnull().sum().sum() == 0
    assert minmax_df.isnull().sum().sum() == 0