import numpy as np
import pytest

from src.models.kmeans import KMeans


@pytest.fixture
def sample_data():
    """Tạo dữ liệu mẫu gồm 2 cụm khách hàng."""
    return np.array([
        [1.0, 1.0],
        [1.2, 0.8],
        [0.8, 1.2],
        [8.0, 8.0],
        [8.2, 7.8],
        [7.8, 8.2]
    ])


def test_default_parameters():
    """Kiểm tra tham số mặc định của KMeans."""
    model = KMeans()

    assert model.n_clusters == 5
    assert model.max_iter == 300
    assert model.tol == 0.0001
    assert model.random_state == 42
    assert model.init == "k-means++"

    assert model.cluster_centers_ is None


def test_fit(sample_data):
    """Kiểm tra quá trình huấn luyện mô hình."""
    model = KMeans(
        n_clusters=2,
        random_state=42
    )

    result = model.fit(sample_data)

    assert result is model
    assert model.cluster_centers_.shape == (2, 2)
    assert model.labels_.shape == (6,)
    assert len(np.unique(model.labels_)) == 2

    assert np.isfinite(model.inertia_)
    assert model.inertia_ >= 0

    assert 1 <= model.n_iter_ <= model.max_iter


def test_predict_before_fit():
    """Kiểm tra dự đoán khi mô hình chưa huấn luyện."""
    model = KMeans(n_clusters=2)

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0, 1.0]]))


def test_predict(sample_data):
    """Kiểm tra khả năng dự đoán khách hàng."""
    model = KMeans(
        n_clusters=2,
        random_state=42
    )

    model.fit(sample_data)

    new_customers = np.array([
        [1.0, 1.0],
        [1.1, 1.1],
        [8.0, 8.0]
    ])

    predictions = model.predict(new_customers)

    assert predictions.shape == (3,)
    assert predictions[0] == predictions[1]
    assert predictions[0] != predictions[2]


def test_fit_predict(sample_data):
    """Kiểm tra hàm fit_predict."""
    model = KMeans(
        n_clusters=2,
        random_state=42
    )

    labels = model.fit_predict(sample_data)

    assert labels.shape == (6,)
    assert len(np.unique(labels)) == 2


def test_insufficient_samples():
    """Kiểm tra dữ liệu ít hơn số cụm yêu cầu."""
    data = np.array([
        [1.0, 1.0],
        [2.0, 2.0]
    ])

    model = KMeans(n_clusters=3)

    with pytest.raises(ValueError):
        model.fit(data)


def test_predict_wrong_features(sample_data):
    """Kiểm tra dữ liệu đầu vào sai số lượng đặc trưng."""
    model = KMeans(
        n_clusters=2,
        random_state=42
    )

    model.fit(sample_data)

    invalid_data = np.array([
        [1.0, 2.0, 3.0]
    ])

    with pytest.raises(ValueError):
        model.predict(invalid_data)